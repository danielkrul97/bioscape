//! Elastic-solid stress pipelines (S225): the Bonet–Lok gradient
//! correction, the Jaumann/Hooke deviatoric stress rate, and the explicit
//! stress integrator. See `shaders/planet_grad_correction.wgsl`,
//! `planet_stress_rate.wgsl`, `planet_stress_integrate.wgsl`.
//!
//! Per-tick order: grad_correction → stress_rate → stress_integrate, all
//! after density (they need final ρ/h) and, from S226, before sph_force
//! reads the stress. Each is a per-i accumulation over the deterministic
//! neighbour walk; neighbour data is read-only.

use crate::gpu::GpuContext;
use crate::planet::gpu::{PlanetGpu, SpatialHashGpu};
use crate::planet::thermal;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

fn build(
    ctx: &Arc<GpuContext>,
    label: &str,
    wgsl: &str,
    entry: &str,
    rw_bindings: &[u32],
    n_bindings: u32,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let device = &ctx.device;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..n_bindings)
        .map(|i| {
            let ty = if i == 0 {
                wgpu::BufferBindingType::Uniform
            } else {
                wgpu::BufferBindingType::Storage {
                    read_only: !rw_bindings.contains(&i),
                }
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
        label: Some(label),
        entries: &entries,
    });
    let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[&bg_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl_layout),
        module: &shader,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    (pipeline, bg_layout)
}

// --- Bonet–Lok gradient correction --------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct GradCorrParams {
    num_particles: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    world_half: f32,
    cell_size: f32,
    lambda: f32,
    pad_a0: f32,
}

pub struct GradCorrectionGpu {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    world_half: f32,
    cell_size: f32,
}

impl GradCorrectionGpu {
    pub fn new(ctx: Arc<GpuContext>, gpu: &PlanetGpu, hash: &SpatialHashGpu) -> Result<Self, String> {
        let (pipeline, bg_layout) = build(
            &ctx,
            "planet-grad-correction",
            include_str!("../../../shaders/planet_grad_correction.wgsl"),
            "grad_correction",
            &[5],
            8,
        );
        let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet-grad-correction-params"),
            contents: bytemuck::bytes_of(&GradCorrParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet-grad-correction-bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gpu.positions_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gpu.masses_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: gpu.smoothing_lengths_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: gpu.densities_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: gpu.grad_correction_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: hash.offsets_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: hash.sorted_buffer().as_entire_binding() },
            ],
        });
        Ok(Self {
            ctx,
            pipeline,
            params_buf,
            bind_group,
            world_half: hash.world_half,
            cell_size: hash.cell_size(),
        })
    }

    /// Adaptive resize: adopt the hash's new grid so `bucket_xyz` agrees.
    pub fn set_grid(&mut self, world_half: f32, cell_size: f32) {
        self.world_half = world_half;
        self.cell_size = cell_size;
    }

    pub fn dispatch(&self, n: usize) {
        if n == 0 {
            return;
        }
        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("planet-grad-correction-encoder"),
        });
        self.encode(&mut encoder, n);
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, n: usize) {
        if n == 0 {
            return;
        }
        let params = GradCorrParams {
            num_particles: n as u32,
            world_half: self.world_half,
            cell_size: self.cell_size,
            lambda: thermal::GRAD_CORRECTION_LAMBDA,
            ..GradCorrParams::default()
        };
        self.ctx.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("planet-grad-correction-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
    }
}

// --- Jaumann/Hooke stress rate ------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct StressRateParams {
    num_particles: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    world_half: f32,
    cell_size: f32,
    g0: f32,
    t_m: f32,
    l: f32,
    pad_b0: f32,
    pad_b1: f32,
    pad_b2: f32,
}

pub struct StressRateGpu {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    world_half: f32,
    cell_size: f32,
    g0: f32, // S231 configurable shear modulus (default = thermal const)
}

impl StressRateGpu {
    pub fn new(ctx: Arc<GpuContext>, gpu: &PlanetGpu, hash: &SpatialHashGpu) -> Result<Self, String> {
        let src = format!(
            "{}\n{}",
            include_str!("../../../shaders/planet_phase_common.wgsl"),
            include_str!("../../../shaders/planet_stress_rate.wgsl"),
        );
        let (pipeline, bg_layout) = build(
            &ctx,
            "planet-stress-rate",
            &src,
            "stress_rate",
            &[8],
            13,
        );
        let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet-stress-rate-params"),
            contents: bytemuck::bytes_of(&StressRateParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet-stress-rate-bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gpu.positions_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gpu.velocities_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: gpu.masses_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: gpu.smoothing_lengths_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: gpu.densities_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: gpu.dev_stress_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: gpu.grad_correction_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: gpu.ds_dt_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: hash.offsets_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: hash.sorted_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: gpu.internal_energies_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: gpu.mat_t_m_buffer().as_entire_binding() },
            ],
        });
        Ok(Self {
            ctx,
            pipeline,
            params_buf,
            bind_group,
            world_half: hash.world_half,
            cell_size: hash.cell_size(),
            g0: thermal::SHEAR_MODULUS_G0,
        })
    }

    pub fn set_g0(&mut self, g0: f32) {
        self.g0 = g0;
    }

    /// Adaptive resize: adopt the hash's new grid so `bucket_xyz` agrees.
    pub fn set_grid(&mut self, world_half: f32, cell_size: f32) {
        self.world_half = world_half;
        self.cell_size = cell_size;
    }

    pub fn dispatch(&self, n: usize) {
        if n == 0 {
            return;
        }
        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("planet-stress-rate-encoder"),
        });
        self.encode(&mut encoder, n);
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, n: usize) {
        if n == 0 {
            return;
        }
        let params = StressRateParams {
            num_particles: n as u32,
            world_half: self.world_half,
            cell_size: self.cell_size,
            g0: self.g0,
            t_m: thermal::MELT_TEMPERATURE_T_M,
            l: thermal::LATENT_HEAT_FUSION_L,
            ..StressRateParams::default()
        };
        self.ctx.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("planet-stress-rate-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
    }
}

// --- Monaghan-2000 artificial stress ------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct ArtStressParams {
    num_particles: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    rho0: f32,
    eos_gamma: f32,
    u_vap: f32,
    c0: f32,
    tait_n: f32,
    t_m: f32,
    l: f32,
    p_tens: f32,
    eps_art: f32,
    melt_coh_frac: f32,
    pad_c1: f32,
    pad_c2: f32,
}

pub struct ArtificialStressGpu {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    c0: f32,
    tait_n: f32,
}

impl ArtificialStressGpu {
    pub fn new(ctx: Arc<GpuContext>, gpu: &PlanetGpu) -> Result<Self, String> {
        let src = format!(
            "{}\n{}",
            include_str!("../../../shaders/planet_phase_common.wgsl"),
            include_str!("../../../shaders/planet_artificial_stress.wgsl"),
        );
        let (pipeline, bg_layout) = build(&ctx, "planet-artificial-stress", &src, "artificial_stress", &[4], 7);
        let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet-artificial-stress-params"),
            contents: bytemuck::bytes_of(&ArtStressParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet-artificial-stress-bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gpu.densities_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gpu.internal_energies_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: gpu.dev_stress_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: gpu.art_stress_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: gpu.mat_rho0_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: gpu.mat_t_m_buffer().as_entire_binding() },
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

    pub fn set_stiffness(&mut self, c0: f32, tait_n: f32) {
        self.c0 = c0;
        self.tait_n = tait_n;
    }

    pub fn dispatch(&self, n: usize, rho0: f32, eos_gamma: f32) {
        if n == 0 {
            return;
        }
        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("planet-artificial-stress-encoder"),
        });
        self.encode(&mut encoder, n, rho0, eos_gamma);
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, n: usize, rho0: f32, eos_gamma: f32) {
        if n == 0 {
            return;
        }
        let params = ArtStressParams {
            num_particles: n as u32,
            rho0,
            eos_gamma,
            u_vap: thermal::VAPORIZATION_ENERGY_U_VAP,
            c0: self.c0,
            tait_n: self.tait_n,
            t_m: thermal::MELT_TEMPERATURE_T_M,
            l: thermal::LATENT_HEAT_FUSION_L,
            p_tens: thermal::TENSILE_STRENGTH_P_TENS,
            eps_art: thermal::ARTIFICIAL_STRESS_EPSILON,
            melt_coh_frac: thermal::MELT_COHESION_FRAC,
            ..ArtStressParams::default()
        };
        self.ctx.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("planet-artificial-stress-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
    }
}

// --- Explicit stress integrator -----------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct StressIntegrateParams {
    num_particles: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    dt: f32,
    y0: f32,
    t_m: f32,
    l: f32,
    g0: f32,
    plastic_cap: f32,
    pad_b0: f32,
    pad_b1: f32,
}

pub struct StressIntegrateGpu {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    y0: f32,
    g0: f32,
}

impl StressIntegrateGpu {
    pub fn new(ctx: Arc<GpuContext>, gpu: &PlanetGpu) -> Result<Self, String> {
        let src = format!(
            "{}\n{}",
            include_str!("../../../shaders/planet_phase_common.wgsl"),
            include_str!("../../../shaders/planet_stress_integrate.wgsl"),
        );
        let (pipeline, bg_layout) = build(
            &ctx,
            "planet-stress-integrate",
            &src,
            "stress_integrate",
            &[1, 5],
            7,
        );
        let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet-stress-integrate-params"),
            contents: bytemuck::bytes_of(&StressIntegrateParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet-stress-integrate-bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gpu.dev_stress_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gpu.ds_dt_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: gpu.internal_energies_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: gpu.densities_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: gpu.du_plastic_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: gpu.mat_t_m_buffer().as_entire_binding() },
            ],
        });
        Ok(Self {
            ctx,
            pipeline,
            params_buf,
            bind_group,
            y0: thermal::YIELD_STRENGTH_Y0,
            g0: thermal::SHEAR_MODULUS_G0,
        })
    }

    pub fn set_stiffness(&mut self, y0: f32, g0: f32) {
        self.y0 = y0;
        self.g0 = g0;
    }

    pub fn dispatch(&self, n: usize, dt: f32) {
        if n == 0 {
            return;
        }
        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("planet-stress-integrate-encoder"),
        });
        self.encode(&mut encoder, n, dt);
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, n: usize, dt: f32) {
        if n == 0 {
            return;
        }
        let params = StressIntegrateParams {
            num_particles: n as u32,
            dt,
            y0: self.y0,
            t_m: thermal::MELT_TEMPERATURE_T_M,
            l: thermal::LATENT_HEAT_FUSION_L,
            g0: self.g0,
            plastic_cap: thermal::PLASTIC_HEAT_MAX_FRAC,
            ..StressIntegrateParams::default()
        };
        self.ctx.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("planet-stress-integrate-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
    }
}
