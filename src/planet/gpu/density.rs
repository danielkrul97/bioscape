//! GPU SPH density estimator + adaptive smoothing-length update.
//! See `shaders/planet_density.wgsl`.

use crate::gpu::GpuContext;
use crate::planet::gpu::{PlanetGpu, SpatialHashGpu};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct DensityParams {
    num_particles: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    world_half: f32,
    cell_size: f32,
    h_min: f32,
    h_max: f32,
}

pub struct DensityGpu {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pub world_half: f32,
    pub cell_size: f32,
    pub h_min: f32,
    pub h_max: f32,
}

impl DensityGpu {
    pub fn new(
        ctx: Arc<GpuContext>,
        gpu: &PlanetGpu,
        hash: &SpatialHashGpu,
    ) -> Result<Self, String> {
        let device = &ctx.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planet-density"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/planet_density.wgsl").into(),
            ),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..7)
            .map(|i| {
                let ty = match i {
                    0 => wgpu::BufferBindingType::Uniform,
                    3 | 4 => wgpu::BufferBindingType::Storage { read_only: false },
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
            label: Some("planet-density-bgl"),
            entries: &entries,
        });
        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("planet-density-pl"),
            bind_group_layouts: &[&bg_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("planet-density-pipeline"),
            layout: Some(&pl_layout),
            module: &shader,
            entry_point: Some("density"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let world_half = hash.world_half;
        let cell_size = hash.cell_size();
        let h_max = hash.max_supported_h();
        let h_min = 0.01 * cell_size;

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet-density-params"),
            contents: bytemuck::bytes_of(&DensityParams {
                world_half,
                cell_size,
                h_min,
                h_max,
                ..DensityParams::default()
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet-density-bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gpu.positions_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gpu.masses_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: gpu.smoothing_lengths_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: gpu.densities_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: hash.offsets_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: hash.sorted_buffer().as_entire_binding() },
            ],
        });

        Ok(Self {
            ctx,
            pipeline,
            params_buf,
            bind_group,
            world_half,
            cell_size,
            h_min,
            h_max,
        })
    }

    pub fn dispatch(&self, n: usize) {
        if n == 0 {
            return;
        }
        let params = DensityParams {
            num_particles: n as u32,
            world_half: self.world_half,
            cell_size: self.cell_size,
            h_min: self.h_min,
            h_max: self.h_max,
            ..DensityParams::default()
        };
        self.ctx
            .queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-density-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("planet-density-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let wg = ((n as u32) + 63) / 64;
            pass.dispatch_workgroups(wg, 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }
}

/// CPU reference for Wendland C2 — used by tests.
pub fn wendland_c2_cpu(r: f32, h: f32) -> f32 {
    let q = r / h;
    if q >= 2.0 {
        return 0.0;
    }
    let h3 = h * h * h;
    let one_minus_half_q = 1.0 - 0.5 * q;
    let factor = one_minus_half_q.powi(4);
    (21.0 / (16.0 * std::f32::consts::PI * h3)) * factor * (1.0 + 2.0 * q)
}
