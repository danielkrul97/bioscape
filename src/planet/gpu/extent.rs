//! Bounding reduction feeding the adaptive spatial-hash resize: the max over
//! all particles of `max(|x|,|y|,|z|)` (containment) and of `h_i` (resolution).
//! See `shaders/planet_extent.wgsl`; the reduction is bit-deterministic
//! (atomicMax on non-negative float bits).

use crate::gpu::GpuContext;
use crate::planet::gpu::PlanetGpu;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct ExtentParams {
    num_particles: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    rho_surface: f32,
    pad_a0: f32,
    pad_a1: f32,
    pad_a2: f32,
}

pub struct ExtentGpu {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    extent_buf: wgpu::Buffer,
    extent_rb: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl ExtentGpu {
    pub fn new(ctx: Arc<GpuContext>, gpu: &PlanetGpu) -> Result<Self, String> {
        let device = &ctx.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planet-extent"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/planet_extent.wgsl").into(),
            ),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..5)
            .map(|i| {
                let ty = match i {
                    0 => wgpu::BufferBindingType::Uniform,
                    4 => wgpu::BufferBindingType::Storage { read_only: false },
                    _ => wgpu::BufferBindingType::Storage { read_only: true },
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
        let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("planet-extent-bgl"),
            entries: &entries,
        });
        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("planet-extent-pl"),
            bind_group_layouts: &[&bg_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("planet-extent-pipeline"),
            layout: Some(&pl_layout),
            module: &shader,
            entry_point: Some("extent_max"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet-extent-params"),
            contents: bytemuck::bytes_of(&ExtentParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let extent_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet-extent-buf"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let extent_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet-extent-rb"),
            size: 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet-extent-bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gpu.positions_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gpu.smoothing_lengths_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: gpu.densities_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: extent_buf.as_entire_binding() },
            ],
        });

        Ok(Self {
            ctx,
            pipeline,
            params_buf,
            extent_buf,
            extent_rb,
            bind_group,
        })
    }

    /// Reduce to `(max |coord|, max h)` over the first `n` particles. Clears
    /// the accumulators, runs the atomicMax reduction, reads the two u32 back
    /// (bitcast to f32). One blocking `poll(Wait)` — call only every K ticks.
    pub fn compute(&self, n: usize, rho_surface: f32) -> (f32, f32) {
        if n == 0 {
            return (0.0, 0.0);
        }
        let params = ExtentParams {
            num_particles: n as u32,
            rho_surface,
            ..ExtentParams::default()
        };
        self.ctx
            .queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-extent-encoder"),
            });
        // Reset both accumulators to +0.0 (all-zero bits) before the atomicMax.
        encoder.clear_buffer(&self.extent_buf, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("planet-extent-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let wg = ((n as u32) + 63) / 64;
            pass.dispatch_workgroups(wg, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.extent_buf, 0, &self.extent_rb, 0, 8);
        self.ctx.queue.submit(Some(encoder.finish()));

        let slice = self.extent_rb.slice(0..8);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.ctx.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let bits: &[u32] = bytemuck::cast_slice(&data);
        let max_coord = f32::from_bits(bits[0]);
        let max_h = f32::from_bits(bits[1]);
        drop(data);
        self.extent_rb.unmap();
        (max_coord, max_h)
    }
}
