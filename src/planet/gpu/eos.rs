//! Per-particle EoS precompute pass — fills `pressure` / `sound_speed`
//! buffers so the SPH force loop reads them instead of recomputing the
//! equation of state for every neighbour pair. See `shaders/planet_eos.wgsl`.
//! The `eos` formula mirrors the one formerly inline in `planet_sph_force`;
//! `planet_phase_common.wgsl` is concatenated for the cohesion clamp.

use crate::gpu::GpuContext;
use crate::planet::gpu::PlanetGpu;
use crate::planet::thermal;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct EosParams {
    num_particles: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    u_vap: f32,
    eos_gamma: f32,
    c0: f32,
    tait_n: f32,
    p_tens: f32,
    l: f32,
    melt_coh_frac: f32,
    pad_a1: f32,
}

pub struct EosGpu {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    // S231 configurable condensed-EoS stiffness; default = thermal consts.
    // Must match the values handed to `SphForceGpu::set_stiffness`.
    c0: f32,
    tait_n: f32,
}

impl EosGpu {
    pub fn new(ctx: Arc<GpuContext>, gpu: &PlanetGpu) -> Result<Self, String> {
        let device = &ctx.device;
        let src = format!(
            "{}\n{}",
            include_str!("../../../shaders/planet_phase_common.wgsl"),
            include_str!("../../../shaders/planet_eos.wgsl"),
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planet-eos"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..7)
            .map(|i| {
                let ty = match i {
                    0 => wgpu::BufferBindingType::Uniform,
                    5 | 6 => wgpu::BufferBindingType::Storage { read_only: false },
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
            label: Some("planet-eos-bgl"),
            entries: &entries,
        });
        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("planet-eos-pl"),
            bind_group_layouts: &[&bg_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("planet-eos-pipeline"),
            layout: Some(&pl_layout),
            module: &shader,
            entry_point: Some("eos"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet-eos-params"),
            contents: bytemuck::bytes_of(&EosParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet-eos-bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gpu.densities_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gpu.internal_energies_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: gpu.mat_rho0_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: gpu.mat_t_m_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: gpu.pressure_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: gpu.sound_speed_buffer().as_entire_binding() },
            ],
        });

        Ok(Self {
            ctx,
            pipeline,
            params_buf,
            bind_group,
            c0: thermal::TAIT_REF_SOUND_SPEED_C0,
            tait_n: thermal::TAIT_EXPONENT_N,
        })
    }

    /// Override the condensed-EoS stiffness (S231). Must mirror the value
    /// passed to `SphForceGpu::set_stiffness` so both agree on the EoS.
    pub fn set_stiffness(&mut self, c0: f32, tait_n: f32) {
        self.c0 = c0;
        self.tait_n = tait_n;
    }

    pub fn dispatch(&self, n: usize, eos_gamma: f32) {
        if n == 0 {
            return;
        }
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-eos-encoder"),
            });
        self.encode(&mut encoder, n, eos_gamma);
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    /// Record the per-particle EoS pass on a caller-owned encoder. Runs
    /// after density (needs final ρ) and before the force passes that read
    /// `pressure` / `sound_speed`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, n: usize, eos_gamma: f32) {
        if n == 0 {
            return;
        }
        let params = EosParams {
            num_particles: n as u32,
            u_vap: thermal::VAPORIZATION_ENERGY_U_VAP,
            eos_gamma,
            c0: self.c0,
            tait_n: self.tait_n,
            p_tens: thermal::TENSILE_STRENGTH_P_TENS,
            l: thermal::LATENT_HEAT_FUSION_L,
            melt_coh_frac: thermal::MELT_COHESION_FRAC,
            ..EosParams::default()
        };
        self.ctx
            .queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("planet-eos-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        let wg = ((n as u32) + 63) / 64;
        pass.dispatch_workgroups(wg, 1, 1);
    }
}
