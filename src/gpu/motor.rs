use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::*;
use super::*;

// ============================================================================
// Sprint 50: GPU motor — applies brain outputs to velocity / angular / pitch
// ============================================================================

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
pub struct MotorParams {
    pub num_cells: u32,
    pub dt: f32,
    pub drag_coefficient: f32,
    pub _pad0: u32,
}

pub struct MotorGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    capacity: usize,
    params_buf: wgpu::Buffer,
    outputs_buf: wgpu::Buffer,
    headings_buf: wgpu::Buffer,
    pitches_buf: wgpu::Buffer,
    max_speeds_buf: wgpu::Buffer,
    turn_rates_buf: wgpu::Buffer,
    eff_radii_buf: wgpu::Buffer,
    velocities_buf: wgpu::Buffer,
    angular_buf: wgpu::Buffer,
    pitch_vel_buf: wgpu::Buffer,
    velocities_readback: wgpu::Buffer,
    angular_readback: wgpu::Buffer,
    pitch_vel_readback: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pos_packed: Vec<f32>,
}

impl MotorGpu {
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
            label: Some("motor"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/motor.wgsl").into()),
        });

        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..10)
            .map(|i| {
                let ty = if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else if i < 7 {
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
            label: Some("motor-bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("motor-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("motor-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("motor"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("motor-params"),
            contents: bytemuck::bytes_of(&MotorParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let (
            outputs_buf,
            headings_buf,
            pitches_buf,
            max_speeds_buf,
            turn_rates_buf,
            eff_radii_buf,
            velocities_buf,
            angular_buf,
            pitch_vel_buf,
            velocities_readback,
            angular_readback,
            pitch_vel_readback,
        ) = Self::alloc_buffers(&device, capacity);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("motor-bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: outputs_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: headings_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pitches_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: max_speeds_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: turn_rates_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: eff_radii_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: velocities_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: angular_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: pitch_vel_buf.as_entire_binding() },
            ],
        });
        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            capacity,
            params_buf,
            outputs_buf,
            headings_buf,
            pitches_buf,
            max_speeds_buf,
            turn_rates_buf,
            eff_radii_buf,
            velocities_buf,
            angular_buf,
            pitch_vel_buf,
            velocities_readback,
            angular_readback,
            pitch_vel_readback,
            bind_group,
            pos_packed: Vec::new(),
        })
    }

    fn alloc_buffers(
        device: &wgpu::Device,
        capacity: usize,
    ) -> (
        wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer,
        wgpu::Buffer, wgpu::Buffer, wgpu::Buffer,
        wgpu::Buffer, wgpu::Buffer, wgpu::Buffer,
    ) {
        let f = std::mem::size_of::<f32>();
        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let stor_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let stor_dst_src =
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let n = capacity as u64;
        (
            mk("motor-outputs", n * BRAIN_OUTPUTS as u64 * f as u64, stor_dst),
            mk("motor-headings", n * f as u64, stor_dst),
            mk("motor-pitches", n * f as u64, stor_dst),
            mk("motor-max-speeds", n * f as u64, stor_dst),
            mk("motor-turn-rates", n * f as u64, stor_dst),
            mk("motor-eff-radii", n * f as u64, stor_dst),
            mk("motor-velocities", n * 3 * f as u64, stor_dst_src),
            mk("motor-angular", n * f as u64, stor_dst_src),
            mk("motor-pitch-vel", n * f as u64, stor_dst_src),
            mk("motor-velocities-rb", n * 3 * f as u64, read),
            mk("motor-angular-rb", n * f as u64, read),
            mk("motor-pitch-vel-rb", n * f as u64, read),
        )
    }

    /// Aplikuje motor pass na inputs. Vrací (new_velocities, new_angular,
    /// new_pitch_vel). dt + drag_coefficient z volajícího kontextu.
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &mut self,
        outputs: &[[f32; BRAIN_OUTPUTS]],
        headings: &[f32],
        pitches: &[f32],
        max_speeds: &[f32],
        turn_rates: &[f32],
        eff_radii: &[f32],
        velocities_in: &[[f32; 3]],
        angular_in: &[f32],
        pitch_vel_in: &[f32],
        dt: f32,
        drag_coefficient: f32,
    ) -> (Vec<[f32; 3]>, Vec<f32>, Vec<f32>) {
        let n = outputs.len();
        assert!(n <= self.capacity, "motor capacity overflow");
        if n == 0 {
            return (Vec::new(), Vec::new(), Vec::new());
        }

        let mut outputs_flat: Vec<f32> = Vec::with_capacity(n * BRAIN_OUTPUTS);
        for o in outputs { outputs_flat.extend_from_slice(o); }
        self.pos_packed.clear();
        self.pos_packed.reserve(n * 3);
        for v in velocities_in { self.pos_packed.extend_from_slice(v); }

        let params = MotorParams {
            num_cells: n as u32,
            dt,
            drag_coefficient,
            ..MotorParams::default()
        };
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(&self.outputs_buf, 0, bytemuck::cast_slice(&outputs_flat));
        self.queue.write_buffer(&self.headings_buf, 0, bytemuck::cast_slice(headings));
        self.queue.write_buffer(&self.pitches_buf, 0, bytemuck::cast_slice(pitches));
        self.queue.write_buffer(&self.max_speeds_buf, 0, bytemuck::cast_slice(max_speeds));
        self.queue.write_buffer(&self.turn_rates_buf, 0, bytemuck::cast_slice(turn_rates));
        self.queue.write_buffer(&self.eff_radii_buf, 0, bytemuck::cast_slice(eff_radii));
        self.queue.write_buffer(&self.velocities_buf, 0, bytemuck::cast_slice(&self.pos_packed));
        self.queue.write_buffer(&self.angular_buf, 0, bytemuck::cast_slice(angular_in));
        self.queue.write_buffer(&self.pitch_vel_buf, 0, bytemuck::cast_slice(pitch_vel_in));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("motor-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("motor-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
        }
        let f = std::mem::size_of::<f32>() as u64;
        let v_bytes = (n as u64) * 3 * f;
        let s_bytes = (n as u64) * f;
        encoder.copy_buffer_to_buffer(&self.velocities_buf, 0, &self.velocities_readback, 0, v_bytes);
        encoder.copy_buffer_to_buffer(&self.angular_buf, 0, &self.angular_readback, 0, s_bytes);
        encoder.copy_buffer_to_buffer(&self.pitch_vel_buf, 0, &self.pitch_vel_readback, 0, s_bytes);
        self.queue.submit(Some(encoder.finish()));

        let v_slice = self.velocities_readback.slice(0..v_bytes);
        let a_slice = self.angular_readback.slice(0..s_bytes);
        let p_slice = self.pitch_vel_readback.slice(0..s_bytes);
        v_slice.map_async(wgpu::MapMode::Read, |_| {});
        a_slice.map_async(wgpu::MapMode::Read, |_| {});
        p_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let velocities: Vec<[f32; 3]> = {
            let data = v_slice.get_mapped_range();
            let f: &[f32] = bytemuck::cast_slice(&data);
            (0..n).map(|i| [f[i * 3], f[i * 3 + 1], f[i * 3 + 2]]).collect()
        };
        let angular: Vec<f32> = {
            let data = a_slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&data).to_vec()
        };
        let pitch_vel: Vec<f32> = {
            let data = p_slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&data).to_vec()
        };
        self.velocities_readback.unmap();
        self.angular_readback.unmap();
        self.pitch_vel_readback.unmap();
        (velocities, angular, pitch_vel)
    }

    /// Sprint 62: persistent variant — bind CellsGpu shared buffery (last_outputs,
    /// heading, pitch, max_speed, turn_rate, eff_radius, velocity, angular_velocity,
    /// pitch_velocity) místo own duplicates. Brain forward už zapsal last_outputs,
    /// motor čte direct + mutuje velocity buffers in-place. No readback — output
    /// stays GPU-side pro chained brownian / batch readback v hot loop.
    pub fn dispatch_with_cells(
        &mut self,
        cells: &CellsGpu,
        num_cells: usize,
        dt: f32,
        drag_coefficient: f32,
    ) {
        if num_cells == 0 {
            return;
        }
        let params = MotorParams {
            num_cells: num_cells as u32,
            dt,
            drag_coefficient,
            _pad0: 0,
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        // Per-call bind group s cells accessory (vlastní bind_group field je
        // pre-Sprint-62 standalone, drží MotorGpu duplicate buffery).
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("motor-bg-cells"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cells.last_outputs_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cells.heading_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cells.pitch_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cells.max_speed_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cells.turn_rate_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: cells.eff_radius_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: cells.velocities_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: cells.angular_velocity_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: cells.pitch_velocity_buffer().as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("motor-dispatch-cells-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("motor-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (num_cells as u32 + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}

