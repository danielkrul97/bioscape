//! Merged SPH non-gravity force pipeline — pressure + Monaghan
//! artificial viscosity in a single neighbour scan. Supersedes the
//! standalone `PressureGpu` + `ViscosityGpu` pair (removed in S220).
//!
//! See `shaders/planet_sph_force.wgsl` for the per-neighbour algorithm.

use crate::gpu::GpuContext;
use crate::planet::gpu::{PlanetGpu, SpatialHashGpu};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct SphForceParams {
    num_particles: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    world_half: f32,
    cell_size: f32,
    eos_k: f32,
    eos_gamma: f32,
    alpha: f32,
    beta: f32,
    pad_a0: f32,
    pad_a1: f32,
}

pub struct SphForceGpu {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pub world_half: f32,
    pub cell_size: f32,
}

impl SphForceGpu {
    pub fn new(
        ctx: Arc<GpuContext>,
        gpu: &PlanetGpu,
        hash: &SpatialHashGpu,
    ) -> Result<Self, String> {
        let device = &ctx.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planet-sph-force"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/planet_sph_force.wgsl").into(),
            ),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..11)
            .map(|i| {
                let ty = match i {
                    0 => wgpu::BufferBindingType::Uniform,
                    6 | 10 => wgpu::BufferBindingType::Storage { read_only: false },
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
            label: Some("planet-sph-force-bgl"),
            entries: &entries,
        });
        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("planet-sph-force-pl"),
            bind_group_layouts: &[&bg_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("planet-sph-force-pipeline"),
            layout: Some(&pl_layout),
            module: &shader,
            entry_point: Some("sph_force"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let world_half = hash.world_half;
        let cell_size = hash.cell_size();

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet-sph-force-params"),
            contents: bytemuck::bytes_of(&SphForceParams {
                world_half,
                cell_size,
                ..SphForceParams::default()
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet-sph-force-bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gpu.positions_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gpu.velocities_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: gpu.masses_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: gpu.smoothing_lengths_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: gpu.densities_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: gpu.accelerations_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: hash.offsets_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: hash.sorted_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: gpu.internal_energies_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: gpu.du_dt_buffer().as_entire_binding() },
            ],
        });

        Ok(Self {
            ctx,
            pipeline,
            params_buf,
            bind_group,
            world_half,
            cell_size,
        })
    }

    pub fn dispatch(&self, n: usize, eos_k: f32, eos_gamma: f32, alpha: f32, beta: f32) {
        if n == 0 {
            return;
        }
        let params = SphForceParams {
            num_particles: n as u32,
            world_half: self.world_half,
            cell_size: self.cell_size,
            eos_k,
            eos_gamma,
            alpha,
            beta,
            ..SphForceParams::default()
        };
        self.ctx
            .queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-sph-force-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("planet-sph-force-pass"),
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
