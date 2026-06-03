//! Phase pass — maps per-particle internal energy to solid fraction
//! `phi` via the enthalpy map and stores it in the `phase_frac` buffer.
//! See `shaders/planet_phase.wgsl`. The WGSL `phase_of` is concatenated
//! from `shaders/planet_phase_common.wgsl` so the map is single-source
//! with `crate::planet::thermal::phase_of`.

use crate::gpu::GpuContext;
use crate::planet::gpu::PlanetGpu;
use crate::planet::thermal;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct PhaseParams {
    num_particles: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    t_m: f32,
    l: f32,
    pad_a0: f32,
    pad_a1: f32,
}

pub struct PhaseGpu {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl PhaseGpu {
    pub fn new(ctx: Arc<GpuContext>, gpu: &PlanetGpu) -> Result<Self, String> {
        let device = &ctx.device;
        let src = format!(
            "{}\n{}",
            include_str!("../../../shaders/planet_phase_common.wgsl"),
            include_str!("../../../shaders/planet_phase.wgsl"),
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planet-phase"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..4)
            .map(|i| {
                let ty = match i {
                    0 => wgpu::BufferBindingType::Uniform,
                    2 => wgpu::BufferBindingType::Storage { read_only: false },
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
            label: Some("planet-phase-bgl"),
            entries: &entries,
        });
        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("planet-phase-pl"),
            bind_group_layouts: &[&bg_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("planet-phase-pipeline"),
            layout: Some(&pl_layout),
            module: &shader,
            entry_point: Some("phase"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet-phase-params"),
            contents: bytemuck::bytes_of(&PhaseParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet-phase-bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gpu.internal_energies_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gpu.phase_frac_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: gpu.mat_t_m_buffer().as_entire_binding() },
            ],
        });

        Ok(Self {
            ctx,
            pipeline,
            params_buf,
            bind_group,
        })
    }

    pub fn dispatch(&self, n: usize) {
        if n == 0 {
            return;
        }
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-phase-encoder"),
            });
        self.encode(&mut encoder, n);
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, n: usize) {
        if n == 0 {
            return;
        }
        let params = PhaseParams {
            num_particles: n as u32,
            t_m: thermal::MELT_TEMPERATURE_T_M,
            l: thermal::LATENT_HEAT_FUSION_L,
            ..PhaseParams::default()
        };
        self.ctx
            .queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("planet-phase-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        let wg = ((n as u32) + 63) / 64;
        pass.dispatch_workgroups(wg, 1, 1);
    }
}
