use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::*;

// GPU step — kinematics, drag, thermal-modulated energy drains, and world
// bounce, per cell. Mirror of `Cell::step_with_climate`.

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
pub struct StepParamsGpu {
    pub num_cells: u32,
    pub _pad_a0: u32,
    pub _pad_a1: u32,
    pub _pad_a2: u32,
    pub dt: f32,
    pub world_half_x: f32,
    pub world_half_y: f32,
    pub world_half_z: f32,
    pub gravity: f32,
    pub drag: f32,
    pub angular_drag: f32,
    pub energy_cost_per_v_sq: f32,
    pub angular_energy_cost: f32,
    pub vision_cost_per_radius: f32,
    pub body_cost_factor: f32,
    pub age_decay_per_sec: f32,
    pub fixed_timestep_hz: f32,
    pub spike_cost_per_sec: f32,
    pub shell_cost_per_sec: f32,
    pub attack_cost_per_sec: f32,
    pub pitch_clamp: f32,
    /// Thermal stratification — mirrors the `THERMAL_*` constants in `lib`.
    /// Drives the z-gradient temperature, Q10 metabolism multiplier, and
    /// scales all energy drains in the shader.
    pub thermal_top: f32,
    pub thermal_bottom: f32,
    pub thermal_q10: f32,
    pub thermal_ref_temp: f32,
    /// Time-varying thermal: diurnal (per-tick, surface-weighted) and
    /// seasonal (per-generation, uniform shift).
    pub thermal_diurnal_amp: f32,
    pub thermal_seasonal_amp: f32,
    /// Phase fraction `(tick mod period) / period` in `[0, 1)` — the caller
    /// precomputes this so the shader doesn't lose precision casting `u64`
    /// directly to `f32` on long runs.
    pub thermal_diurnal_phase: f32,
    pub thermal_seasonal_phase: f32,
    /// `log2(thermal_q10)` precomputed on CPU. Lets the shader replace
    /// `pow(q10, x)` with `exp2(x * thermal_log2_q10)` (one log2 saved per cell
    /// per tick). Callers must keep this in sync with `thermal_q10`.
    pub thermal_log2_q10: f32,
    pub _pad_b0: u32,
    pub _pad_b1: u32,
    pub _pad_b2: u32,
}

pub struct StepGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    capacity: usize,
    params_buf: wgpu::Buffer,
    pos_buf: wgpu::Buffer,
    vel_buf: wgpu::Buffer,
    heading_buf: wgpu::Buffer,
    pitch_buf: wgpu::Buffer,
    ang_vel_buf: wgpu::Buffer,
    pitch_vel_buf: wgpu::Buffer,
    age_buf: wgpu::Buffer,
    cooldown_buf: wgpu::Buffer,
    energy_buf: wgpu::Buffer,
    body_dims_buf: wgpu::Buffer,
    aux_buf: wgpu::Buffer,
    pos_rb: wgpu::Buffer,
    vel_rb: wgpu::Buffer,
    heading_rb: wgpu::Buffer,
    pitch_rb: wgpu::Buffer,
    ang_vel_rb: wgpu::Buffer,
    pitch_vel_rb: wgpu::Buffer,
    age_rb: wgpu::Buffer,
    cooldown_rb: wgpu::Buffer,
    energy_rb: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    cached_cells_bg: Option<wgpu::BindGroup>,
    cached_cells_epoch: u64,
}

#[derive(Debug, Clone)]
pub struct StepResult {
    pub positions: Vec<[f32; 3]>,
    pub velocities: Vec<[f32; 3]>,
    pub headings: Vec<f32>,
    pub pitches: Vec<f32>,
    pub angular_velocities: Vec<f32>,
    pub pitch_velocities: Vec<f32>,
    pub ages: Vec<u32>,
    pub cooldowns: Vec<u32>,
    pub energies: Vec<f32>,
}

impl StepGpu {
    pub fn with_context(ctx: &GpuContext, capacity: usize) -> Result<Self, String> {
        Self::with_device_inner(Arc::clone(&ctx.device), Arc::clone(&ctx.queue), capacity)
    }

    pub fn new(capacity: usize) -> Result<Self, String> {
        let ctx = GpuContext::new()?;
        Self::with_device_inner(ctx.device, ctx.queue, capacity)
    }

    fn with_device_inner(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        capacity: usize,
    ) -> Result<Self, String> {
        assert!(capacity > 0);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("step"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/step.wgsl").into()),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..12)
            .map(|i| {
                let ty = if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else if i >= 10 {
                    wgpu::BufferBindingType::Storage { read_only: true }
                } else {
                    wgpu::BufferBindingType::Storage { read_only: false }
                };
                wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            })
            .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("step-bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("step-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("step-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("step"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("step-params"),
            contents: bytemuck::bytes_of(&StepParamsGpu::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let f = std::mem::size_of::<f32>() as u64;
        let n = capacity as u64;
        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let stor_dst_src =
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let stor_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;

        let pos_buf = mk("step-pos", n * 3 * f, stor_dst_src);
        let vel_buf = mk("step-vel", n * 3 * f, stor_dst_src);
        let heading_buf = mk("step-heading", n * f, stor_dst_src);
        let pitch_buf = mk("step-pitch", n * f, stor_dst_src);
        let ang_vel_buf = mk("step-ang-vel", n * f, stor_dst_src);
        let pitch_vel_buf = mk("step-pitch-vel", n * f, stor_dst_src);
        let age_buf = mk("step-age", n * f, stor_dst_src);
        let cooldown_buf = mk("step-cooldown", n * f, stor_dst_src);
        let energy_buf = mk("step-energy", n * f, stor_dst_src);
        let body_dims_buf = mk("step-body-dims", n * 3 * f, stor_dst);
        let aux_buf = mk("step-aux", n * 4 * f, stor_dst);

        let pos_rb = mk("step-pos-rb", n * 3 * f, read);
        let vel_rb = mk("step-vel-rb", n * 3 * f, read);
        let heading_rb = mk("step-heading-rb", n * f, read);
        let pitch_rb = mk("step-pitch-rb", n * f, read);
        let ang_vel_rb = mk("step-ang-vel-rb", n * f, read);
        let pitch_vel_rb = mk("step-pitch-vel-rb", n * f, read);
        let age_rb = mk("step-age-rb", n * f, read);
        let cooldown_rb = mk("step-cooldown-rb", n * f, read);
        let energy_rb = mk("step-energy-rb", n * f, read);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("step-bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: vel_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: heading_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pitch_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: ang_vel_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: pitch_vel_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: age_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: cooldown_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: energy_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: body_dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: aux_buf.as_entire_binding() },
            ],
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            capacity,
            params_buf,
            pos_buf,
            vel_buf,
            heading_buf,
            pitch_buf,
            ang_vel_buf,
            pitch_vel_buf,
            age_buf,
            cooldown_buf,
            energy_buf,
            body_dims_buf,
            aux_buf,
            pos_rb,
            vel_rb,
            heading_rb,
            pitch_rb,
            ang_vel_rb,
            pitch_vel_rb,
            age_rb,
            cooldown_rb,
            energy_rb,
            bind_group,
            cached_cells_bg: None,
            cached_cells_epoch: 0,
        })
    }

    /// Apply step pass. Inputs upload + dispatch + readback.
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &mut self,
        positions: &[[f32; 3]],
        velocities: &[[f32; 3]],
        headings: &[f32],
        pitches: &[f32],
        angular_velocities: &[f32],
        pitch_velocities: &[f32],
        ages: &[u32],
        cooldowns: &[u32],
        energies: &[f32],
        body_dims: &[[f32; 3]],
        aux: &[[f32; 4]],
        params: StepParamsGpu,
    ) -> StepResult {
        let n = positions.len();
        assert!(n <= self.capacity, "step capacity overflow");
        if n == 0 {
            return StepResult {
                positions: Vec::new(),
                velocities: Vec::new(),
                headings: Vec::new(),
                pitches: Vec::new(),
                angular_velocities: Vec::new(),
                pitch_velocities: Vec::new(),
                ages: Vec::new(),
                cooldowns: Vec::new(),
                energies: Vec::new(),
            };
        }
        let mut params = params;
        params.num_cells = n as u32;

        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(&self.pos_buf, 0, bytemuck::cast_slice(positions));
        self.queue.write_buffer(&self.vel_buf, 0, bytemuck::cast_slice(velocities));
        self.queue.write_buffer(&self.heading_buf, 0, bytemuck::cast_slice(headings));
        self.queue.write_buffer(&self.pitch_buf, 0, bytemuck::cast_slice(pitches));
        self.queue.write_buffer(&self.ang_vel_buf, 0, bytemuck::cast_slice(angular_velocities));
        self.queue.write_buffer(&self.pitch_vel_buf, 0, bytemuck::cast_slice(pitch_velocities));
        self.queue.write_buffer(&self.age_buf, 0, bytemuck::cast_slice(ages));
        self.queue.write_buffer(&self.cooldown_buf, 0, bytemuck::cast_slice(cooldowns));
        self.queue.write_buffer(&self.energy_buf, 0, bytemuck::cast_slice(energies));
        self.queue.write_buffer(&self.body_dims_buf, 0, bytemuck::cast_slice(body_dims));
        self.queue.write_buffer(&self.aux_buf, 0, bytemuck::cast_slice(aux));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("step-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("step-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
        }
        let f = std::mem::size_of::<f32>() as u64;
        let v3 = (n as u64) * 3 * f;
        let v1 = (n as u64) * f;
        encoder.copy_buffer_to_buffer(&self.pos_buf, 0, &self.pos_rb, 0, v3);
        encoder.copy_buffer_to_buffer(&self.vel_buf, 0, &self.vel_rb, 0, v3);
        encoder.copy_buffer_to_buffer(&self.heading_buf, 0, &self.heading_rb, 0, v1);
        encoder.copy_buffer_to_buffer(&self.pitch_buf, 0, &self.pitch_rb, 0, v1);
        encoder.copy_buffer_to_buffer(&self.ang_vel_buf, 0, &self.ang_vel_rb, 0, v1);
        encoder.copy_buffer_to_buffer(&self.pitch_vel_buf, 0, &self.pitch_vel_rb, 0, v1);
        encoder.copy_buffer_to_buffer(&self.age_buf, 0, &self.age_rb, 0, v1);
        encoder.copy_buffer_to_buffer(&self.cooldown_buf, 0, &self.cooldown_rb, 0, v1);
        encoder.copy_buffer_to_buffer(&self.energy_buf, 0, &self.energy_rb, 0, v1);
        self.queue.submit(Some(encoder.finish()));

        let pos_s = self.pos_rb.slice(0..v3);
        let vel_s = self.vel_rb.slice(0..v3);
        let h_s = self.heading_rb.slice(0..v1);
        let p_s = self.pitch_rb.slice(0..v1);
        let av_s = self.ang_vel_rb.slice(0..v1);
        let pv_s = self.pitch_vel_rb.slice(0..v1);
        let age_s = self.age_rb.slice(0..v1);
        let cd_s = self.cooldown_rb.slice(0..v1);
        let en_s = self.energy_rb.slice(0..v1);
        pos_s.map_async(wgpu::MapMode::Read, |_| {});
        vel_s.map_async(wgpu::MapMode::Read, |_| {});
        h_s.map_async(wgpu::MapMode::Read, |_| {});
        p_s.map_async(wgpu::MapMode::Read, |_| {});
        av_s.map_async(wgpu::MapMode::Read, |_| {});
        pv_s.map_async(wgpu::MapMode::Read, |_| {});
        age_s.map_async(wgpu::MapMode::Read, |_| {});
        cd_s.map_async(wgpu::MapMode::Read, |_| {});
        en_s.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let pos_data = pos_s.get_mapped_range();
        let vel_data = vel_s.get_mapped_range();
        let pos_f: &[f32] = bytemuck::cast_slice(&pos_data);
        let vel_f: &[f32] = bytemuck::cast_slice(&vel_data);
        let positions: Vec<[f32; 3]> = (0..n)
            .map(|i| [pos_f[i * 3], pos_f[i * 3 + 1], pos_f[i * 3 + 2]])
            .collect();
        let velocities: Vec<[f32; 3]> = (0..n)
            .map(|i| [vel_f[i * 3], vel_f[i * 3 + 1], vel_f[i * 3 + 2]])
            .collect();
        drop(pos_data);
        drop(vel_data);
        let result = StepResult {
            positions,
            velocities,
            headings: bytemuck::cast_slice::<u8, f32>(&h_s.get_mapped_range()).to_vec(),
            pitches: bytemuck::cast_slice::<u8, f32>(&p_s.get_mapped_range()).to_vec(),
            angular_velocities: bytemuck::cast_slice::<u8, f32>(&av_s.get_mapped_range()).to_vec(),
            pitch_velocities: bytemuck::cast_slice::<u8, f32>(&pv_s.get_mapped_range()).to_vec(),
            ages: bytemuck::cast_slice::<u8, u32>(&age_s.get_mapped_range()).to_vec(),
            cooldowns: bytemuck::cast_slice::<u8, u32>(&cd_s.get_mapped_range()).to_vec(),
            energies: bytemuck::cast_slice::<u8, f32>(&en_s.get_mapped_range()).to_vec(),
        };
        self.pos_rb.unmap();
        self.vel_rb.unmap();
        self.heading_rb.unmap();
        self.pitch_rb.unmap();
        self.ang_vel_rb.unmap();
        self.pitch_vel_rb.unmap();
        self.age_rb.unmap();
        self.cooldown_rb.unmap();
        self.energy_rb.unmap();
        result
    }

    /// Persistent variant: binds the shared `CellsGpu` buffers (position,
    /// velocity, heading, pitch, ang_vel, pitch_vel, age, cooldown, energy,
    /// body_dims, aux) instead of internal duplicates. The step shader
    /// mutates all of them in place; the hot loop reads back later via
    /// `download_full_batch`.
    pub fn dispatch_with_cells(
        &mut self,
        cells: &CellsGpu,
        num_cells: usize,
        params: StepParamsGpu,
    ) {
        if num_cells == 0 {
            return;
        }
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("step-dispatch-cells-encoder"),
        });
        self.dispatch_with_cells_into(&mut encoder, cells, num_cells, params);
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn dispatch_with_cells_into(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        cells: &CellsGpu,
        num_cells: usize,
        params: StepParamsGpu,
    ) {
        if num_cells == 0 {
            return;
        }
        let mut params = params;
        params.num_cells = num_cells as u32;
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        let cells_epoch = cells.epoch();
        if self.cached_cells_bg.is_none() || self.cached_cells_epoch != cells_epoch {
            self.cached_cells_bg = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("step-bg-cells"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: cells.position_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: cells.velocities_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: cells.heading_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: cells.pitch_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: cells.angular_velocity_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: cells.pitch_velocity_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: cells.age_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: cells.cooldown_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: cells.energy_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 10, resource: cells.body_dims_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 11, resource: cells.aux_buffer().as_entire_binding() },
                ],
            }));
            self.cached_cells_epoch = cells_epoch;
        }
        let bind_group = self.cached_cells_bg.as_ref().unwrap();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("step-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        let workgroups = (num_cells as u32 + 63) / 64;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
}

