//! Thermal integrator pipeline — explicit Euler step on the per-particle
//! internal energy buffer with safety clamps. See
//! `shaders/planet_thermal_integrate.wgsl` for the per-particle formula.

use crate::gpu::GpuContext;
use crate::planet::gpu::PlanetGpu;
use crate::planet::thermal;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct ThermalParams {
    num_particles: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    dt: f32,
    u_min: f32,
    u_max: f32,
    pad_a0: f32,
}

pub struct ThermalIntegrateGpu {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl ThermalIntegrateGpu {
    pub fn new(ctx: Arc<GpuContext>, gpu: &PlanetGpu) -> Result<Self, String> {
        let device = &ctx.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planet-thermal-integrate"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/planet_thermal_integrate.wgsl").into(),
            ),
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
            label: Some("planet-thermal-integrate-bgl"),
            entries: &entries,
        });
        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("planet-thermal-integrate-pl"),
            bind_group_layouts: &[&bg_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("planet-thermal-integrate-pipeline"),
            layout: Some(&pl_layout),
            module: &shader,
            entry_point: Some("thermal_integrate"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet-thermal-integrate-params"),
            contents: bytemuck::bytes_of(&ThermalParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet-thermal-integrate-bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gpu.internal_energies_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gpu.du_dt_buffer().as_entire_binding() },
            ],
        });

        Ok(Self {
            ctx,
            pipeline,
            params_buf,
            bind_group,
        })
    }

    pub fn dispatch(&self, n: usize, dt: f32) {
        if n == 0 {
            return;
        }
        let params = ThermalParams {
            num_particles: n as u32,
            dt,
            u_min: thermal::U_MIN,
            u_max: thermal::U_MAX,
            ..ThermalParams::default()
        };
        self.ctx
            .queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-thermal-integrate-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("planet-thermal-integrate-pass"),
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
