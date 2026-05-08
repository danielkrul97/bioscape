use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::*;

// ============================================================================
// Sprint 51: GPU brownian (xoshiro128++ per-cell RNG) + Hebbian update
// ============================================================================

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct BrownianParams {
    num_cells: u32,
    has_z: u32,
    thermal_noise: f32,
    sqrt_dt: f32,
}

pub struct BrownianGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    capacity: usize,
    params_buf: wgpu::Buffer,
    velocities_buf: wgpu::Buffer,
    state_buf: wgpu::Buffer,
    velocities_rb: wgpu::Buffer,
    state_rb: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    cached_persistent_bg: Option<wgpu::BindGroup>,
    cached_persistent_epoch: u64,
}

impl BrownianGpu {
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
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brownian"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/brownian.wgsl").into()),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..3)
            .map(|i| {
                let ty = if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage { read_only: false }
                };
                wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                }
            })
            .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brownian-bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brownian-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("brownian-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("brownian"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brownian-params"),
            contents: bytemuck::bytes_of(&BrownianParams::default()),
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
        let stor_dst_src = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let velocities_buf = mk("brownian-vel", n * 3 * f, stor_dst_src);
        let state_buf = mk("brownian-state", n * 4 * 4, stor_dst_src);
        let velocities_rb = mk("brownian-vel-rb", n * 3 * f, read);
        let state_rb = mk("brownian-state-rb", n * 4 * 4, read);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brownian-bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: velocities_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: state_buf.as_entire_binding() },
            ],
        });
        Ok(Self {
            device, queue, pipeline, bind_group_layout, capacity,
            params_buf, velocities_buf, state_buf, velocities_rb, state_rb, bind_group,
            cached_persistent_bg: None,
            cached_persistent_epoch: 0,
        })
    }

    pub fn compute(
        &mut self,
        velocities_in: &[[f32; 3]],
        state_in: &[[u32; 4]],
        thermal_noise: f32,
        dt: f32,
        has_z: bool,
    ) -> (Vec<[f32; 3]>, Vec<[u32; 4]>) {
        let n = velocities_in.len();
        assert_eq!(state_in.len(), n);
        assert!(n <= self.capacity);
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let vel_flat: Vec<f32> = velocities_in.iter().flatten().copied().collect();
        let state_flat: Vec<u32> = state_in.iter().flatten().copied().collect();
        let params = BrownianParams {
            num_cells: n as u32,
            has_z: if has_z { 1 } else { 0 },
            thermal_noise,
            sqrt_dt: dt.sqrt(),
        };
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(&self.velocities_buf, 0, bytemuck::cast_slice(&vel_flat));
        self.queue.write_buffer(&self.state_buf, 0, bytemuck::cast_slice(&state_flat));
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("brownian-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("brownian-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
        }
        let v_bytes = (n as u64) * 3 * 4;
        let s_bytes = (n as u64) * 4 * 4;
        encoder.copy_buffer_to_buffer(&self.velocities_buf, 0, &self.velocities_rb, 0, v_bytes);
        encoder.copy_buffer_to_buffer(&self.state_buf, 0, &self.state_rb, 0, s_bytes);
        self.queue.submit(Some(encoder.finish()));
        let v_s = self.velocities_rb.slice(0..v_bytes);
        let s_s = self.state_rb.slice(0..s_bytes);
        v_s.map_async(wgpu::MapMode::Read, |_| {});
        s_s.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let v_data = v_s.get_mapped_range();
        let s_data = s_s.get_mapped_range();
        let v_f: &[f32] = bytemuck::cast_slice(&v_data);
        let s_u: &[u32] = bytemuck::cast_slice(&s_data);
        let velocities: Vec<[f32; 3]> = (0..n).map(|i| [v_f[i*3], v_f[i*3+1], v_f[i*3+2]]).collect();
        let state: Vec<[u32; 4]> = (0..n).map(|i| [s_u[i*4], s_u[i*4+1], s_u[i*4+2], s_u[i*4+3]]).collect();
        drop(v_data);
        drop(s_data);
        self.velocities_rb.unmap();
        self.state_rb.unmap();
        (velocities, state)
    }

    pub fn velocities_buffer(&self) -> &wgpu::Buffer {
        &self.velocities_buf
    }
    pub fn state_buffer(&self) -> &wgpu::Buffer {
        &self.state_buf
    }

    /// Sprint 51: persistent-mode dispatch — bindujeme CellsGpu buffery místo
    /// internal. Žádný upload/download, žádný realloc. Volá se v hot loopu
    /// po `cells_gpu.upload_velocities(...)`.
    pub fn compute_persistent(
        &mut self,
        cells_gpu: &CellsGpu,
        n: usize,
        thermal_noise: f32,
        dt: f32,
        has_z: bool,
    ) {
        if n == 0 {
            return;
        }
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("brownian-encoder-persistent"),
        });
        self.compute_persistent_into(&mut encoder, cells_gpu, n, thermal_noise, dt, has_z);
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn compute_persistent_into(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        cells_gpu: &CellsGpu,
        n: usize,
        thermal_noise: f32,
        dt: f32,
        has_z: bool,
    ) {
        if n == 0 {
            return;
        }
        let params = BrownianParams {
            num_cells: n as u32,
            has_z: if has_z { 1 } else { 0 },
            thermal_noise,
            sqrt_dt: dt.sqrt(),
        };
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        let cells_epoch = cells_gpu.epoch();
        if self.cached_persistent_bg.is_none() || self.cached_persistent_epoch != cells_epoch {
            self.cached_persistent_bg = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("brownian-bg-persistent"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: cells_gpu.velocities_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: cells_gpu.xoshiro_state_buffer().as_entire_binding() },
                ],
            }));
            self.cached_persistent_epoch = cells_epoch;
        }
        let bind_group = self.cached_persistent_bg.as_ref().unwrap();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("brownian-pass-persistent"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
    }
}

