//! `PlanetGpu` — unified GPU state for the leapfrog hot loop.
//!
//! Owns the shared particle data buffers (positions, velocities,
//! accelerations, masses) and the three pipelines that operate on
//! them (nbody, kick, drift). Each pipeline has its own uniform
//! params buffer + bind group, but all bind groups reference the
//! same data buffers.
//!
//! Hot loop per tick (KDK leapfrog):
//!   1. kick (dt/2, a_old)
//!   2. drift (dt)
//!   3. nbody → new accelerations
//!   4. kick (dt/2, a_new)
//!
//! All four dispatches are recorded into one command encoder and
//! submitted with a single `queue.submit`. Readback only happens
//! when the caller asks for it (`download_state`).

use crate::gpu::GpuContext;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct NBodyParams {
    num_particles: u32,
    pad_a0: u32,
    pad_a1: u32,
    pad_a2: u32,
    g: f32,
    eps2: f32,
    pad_b0: f32,
    pad_b1: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct StepParams {
    num_particles: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    dt: f32,
    pad_a0: f32,
    pad_a1: f32,
    pad_a2: f32,
}

pub struct PlanetGpu {
    pub ctx: Arc<GpuContext>,
    pub capacity: usize,

    positions_buf: wgpu::Buffer,
    velocities_buf: wgpu::Buffer,
    accelerations_buf: wgpu::Buffer,
    grav_accel_buf: wgpu::Buffer,
    masses_buf: wgpu::Buffer,
    smoothing_lengths_buf: wgpu::Buffer,
    densities_buf: wgpu::Buffer,
    /// Sprint 202: per-particle internal energy per unit mass.
    internal_energies_buf: wgpu::Buffer,
    /// Sprint 202: per-particle scratch buffer for `du/dt` accumulation
    /// across one tick (viscous, adiabatic, conduction). Cleared by the
    /// thermal integrator at the end of each tick.
    du_dt_buf: wgpu::Buffer,
    /// Sprint 223: per-particle solid fraction `phi ∈ [0, 1]`, written
    /// end-of-tick by the phase pass. CPU-readable cache for diagnostics /
    /// rendering / block labelling; GPU mechanics recompute `phase_of`
    /// inline from `u`, so this is not on the force path.
    phase_fracs_buf: wgpu::Buffer,
    /// Sprint 225: persistent symmetric deviatoric stress tensor, packed
    /// `[Sxx, Syy, Szz, Sxy, Sxz, Syz]` per particle (6N). Carries elastic
    /// memory ACROSS ticks — not scratch.
    dev_stress_buf: wgpu::Buffer,
    /// Sprint 225: per-tick deviatoric stress rate (6N scratch), cleared by
    /// the stress integrator. Mirrors the `du_dt` scratch pattern.
    ds_dt_buf: wgpu::Buffer,
    /// Sprint 225: per-particle Bonet–Lok kernel-gradient correction matrix
    /// (3×3 row-major, 9N scratch), recomputed each tick.
    grad_corr_buf: wgpu::Buffer,
    /// Sprint 228: per-particle Monaghan-2000 artificial-stress tensor
    /// (symmetric 6N, scratch), recomputed each tick from the total stress.
    art_stress_buf: wgpu::Buffer,
    /// Sprint 229: per-step plastic-work heating (N, scratch) from the von
    /// Mises return, applied to `u` by the thermal integrator.
    du_plastic_buf: wgpu::Buffer,
    /// Sprint 232: per-particle material reference density `ρ0` (N) and melt
    /// temperature `T_m` (N) — read by the EoS / phase map per particle.
    mat_rho0_buf: wgpu::Buffer,
    mat_t_m_buf: wgpu::Buffer,
    /// Per-particle EoS outputs (N each), filled by the EoS pass after
    /// density so the force loop reads them instead of recomputing the EoS
    /// for every neighbour pair.
    pressure_buf: wgpu::Buffer,
    sound_speed_buf: wgpu::Buffer,

    positions_rb: wgpu::Buffer,
    velocities_rb: wgpu::Buffer,
    accelerations_rb: wgpu::Buffer,
    smoothing_lengths_rb: wgpu::Buffer,
    densities_rb: wgpu::Buffer,
    internal_energies_rb: wgpu::Buffer,
    phase_fracs_rb: wgpu::Buffer,
    dev_stress_rb: wgpu::Buffer,
    grad_corr_rb: wgpu::Buffer,

    nbody_pipeline: wgpu::ComputePipeline,
    nbody_params_buf: wgpu::Buffer,
    nbody_bg: wgpu::BindGroup,
    nbody_grav_bg: wgpu::BindGroup,

    kick_pipeline: wgpu::ComputePipeline,
    kick_params_buf: wgpu::Buffer,
    kick_bg: wgpu::BindGroup,

    drift_pipeline: wgpu::ComputePipeline,
    drift_params_buf: wgpu::Buffer,
    drift_bg: wgpu::BindGroup,
}

impl PlanetGpu {
    pub fn new(ctx: Arc<GpuContext>, capacity: usize) -> Result<Self, String> {
        let device = &ctx.device;
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
        let stor = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;

        // Positions are vec4-packed (16-byte stride, .w padding) so the
        // per-particle gather in every neighbour pass hits one aligned load
        // instead of three strided f32 reads. velocities/accelerations stay
        // tightly packed (3·f) — they are not gathered in the hot loops.
        let positions_buf = mk("planet-positions", n * 4 * f, stor);
        let velocities_buf = mk("planet-velocities", n * 4 * f, stor);
        let accelerations_buf = mk("planet-accelerations", n * 3 * f, stor);
        // S231: holds gravity-only acceleration across sub-steps (operator
        // split — gravity is computed once per outer step and held fixed).
        let grav_accel_buf = mk("planet-grav-accel", n * 3 * f, stor);
        let masses_buf = mk("planet-masses", n * f, stor);
        let smoothing_lengths_buf = mk("planet-h", n * f, stor);
        let densities_buf = mk("planet-rho", n * f, stor);
        let internal_energies_buf = mk("planet-u", n * f, stor);
        let du_dt_buf = mk("planet-du-dt", n * f, stor);
        let phase_fracs_buf = mk("planet-phase-frac", n * f, stor);
        let dev_stress_buf = mk("planet-dev-stress", n * 6 * f, stor);
        let ds_dt_buf = mk("planet-ds-dt", n * 6 * f, stor);
        let grad_corr_buf = mk("planet-grad-corr", n * 9 * f, stor);
        let art_stress_buf = mk("planet-art-stress", n * 6 * f, stor);
        let du_plastic_buf = mk("planet-du-plastic", n * f, stor);
        let mat_rho0_buf = mk("planet-mat-rho0", n * f, stor);
        let mat_t_m_buf = mk("planet-mat-t-m", n * f, stor);
        let pressure_buf = mk("planet-pressure", n * f, stor);
        let sound_speed_buf = mk("planet-sound-speed", n * f, stor);
        // Seed single-material defaults (ρ0 = 1, T_m = melt const) so a
        // PlanetGpu used without `upload_materials` (unit tests) is valid;
        // `PlanetWorld::init_gpu_full` overwrites with per-particle values.
        ctx.queue.write_buffer(
            &mat_rho0_buf,
            0,
            bytemuck::cast_slice(&vec![1.0_f32; capacity]),
        );
        ctx.queue.write_buffer(
            &mat_t_m_buf,
            0,
            bytemuck::cast_slice(&vec![crate::planet::thermal::MELT_TEMPERATURE_T_M; capacity]),
        );
        let positions_rb = mk("planet-positions-rb", n * 4 * f, read);
        let velocities_rb = mk("planet-velocities-rb", n * 4 * f, read);
        let accelerations_rb = mk("planet-accelerations-rb", n * 3 * f, read);
        let smoothing_lengths_rb = mk("planet-h-rb", n * f, read);
        let densities_rb = mk("planet-rho-rb", n * f, read);
        let internal_energies_rb = mk("planet-u-rb", n * f, read);
        let phase_fracs_rb = mk("planet-phase-frac-rb", n * f, read);
        let dev_stress_rb = mk("planet-dev-stress-rb", n * 6 * f, read);
        let grad_corr_rb = mk("planet-grad-corr-rb", n * 9 * f, read);

        // NBody pipeline
        let nbody_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planet-nbody"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/planet_nbody.wgsl").into(),
            ),
        });
        let nbody_layout = make_layout(device, "planet-nbody-bgl", &[false, true, true, false]);
        let (nbody_pipeline, _) = build_pipeline(device, "planet-nbody", &nbody_shader, "nbody", &nbody_layout);
        let nbody_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet-nbody-params"),
            contents: bytemuck::bytes_of(&NBodyParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let nbody_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet-nbody-bg"),
            layout: &nbody_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: nbody_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: positions_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: masses_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: accelerations_buf.as_entire_binding() },
            ],
        });
        // S231: same nbody pipeline writing into the gravity-only buffer.
        let nbody_grav_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet-nbody-grav-bg"),
            layout: &nbody_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: nbody_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: positions_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: masses_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: grav_accel_buf.as_entire_binding() },
            ],
        });

        // Kick pipeline
        let kick_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planet-kick"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/planet_kick.wgsl").into(),
            ),
        });
        let kick_layout = make_layout(device, "planet-kick-bgl", &[false, false, true]);
        let (kick_pipeline, _) = build_pipeline(device, "planet-kick", &kick_shader, "kick", &kick_layout);
        let kick_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet-kick-params"),
            contents: bytemuck::bytes_of(&StepParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let kick_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet-kick-bg"),
            layout: &kick_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: kick_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: velocities_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: accelerations_buf.as_entire_binding() },
            ],
        });

        // Drift pipeline
        let drift_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planet-drift"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/planet_drift.wgsl").into(),
            ),
        });
        let drift_layout = make_layout(device, "planet-drift-bgl", &[false, false, true]);
        let (drift_pipeline, _) = build_pipeline(device, "planet-drift", &drift_shader, "drift", &drift_layout);
        let drift_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet-drift-params"),
            contents: bytemuck::bytes_of(&StepParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let drift_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet-drift-bg"),
            layout: &drift_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: drift_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: positions_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: velocities_buf.as_entire_binding() },
            ],
        });

        Ok(Self {
            ctx,
            capacity,
            positions_buf,
            velocities_buf,
            accelerations_buf,
            grav_accel_buf,
            masses_buf,
            smoothing_lengths_buf,
            densities_buf,
            internal_energies_buf,
            du_dt_buf,
            phase_fracs_buf,
            dev_stress_buf,
            ds_dt_buf,
            grad_corr_buf,
            art_stress_buf,
            du_plastic_buf,
            mat_rho0_buf,
            mat_t_m_buf,
            pressure_buf,
            sound_speed_buf,
            positions_rb,
            velocities_rb,
            accelerations_rb,
            smoothing_lengths_rb,
            densities_rb,
            internal_energies_rb,
            phase_fracs_rb,
            dev_stress_rb,
            grad_corr_rb,
            nbody_pipeline,
            nbody_params_buf,
            nbody_bg,
            nbody_grav_bg,
            kick_pipeline,
            kick_params_buf,
            kick_bg,
            drift_pipeline,
            drift_params_buf,
            drift_bg,
        })
    }

    /// Upload initial particle state. Caller is responsible for the
    /// initial acceleration (call `compute_accelerations` once after
    /// this to seed `a_0`).
    pub fn upload_state(
        &self,
        positions: &[[f32; 3]],
        velocities: &[[f32; 3]],
        masses: &[f32],
    ) {
        let n = positions.len();
        assert!(n <= self.capacity);
        assert_eq!(velocities.len(), n);
        assert_eq!(masses.len(), n);
        if n == 0 {
            return;
        }
        // Positions and velocities are vec4-packed on the GPU; pad to [x,y,z,0].
        let pos: Vec<f32> = positions.iter().flat_map(|p| [p[0], p[1], p[2], 0.0]).collect();
        let vel: Vec<f32> = velocities.iter().flat_map(|p| [p[0], p[1], p[2], 0.0]).collect();
        self.ctx
            .queue
            .write_buffer(&self.positions_buf, 0, bytemuck::cast_slice(&pos));
        self.ctx
            .queue
            .write_buffer(&self.velocities_buf, 0, bytemuck::cast_slice(&vel));
        self.ctx
            .queue
            .write_buffer(&self.masses_buf, 0, bytemuck::cast_slice(masses));
    }

    /// One-shot nbody dispatch — computes accelerations from current
    /// positions/masses. Used for `a_0` seeding and the S206-style
    /// correctness test.
    pub fn compute_accelerations(&self, n: usize, g: f32, softening: f32) {
        if n == 0 {
            return;
        }
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-accel-encoder"),
            });
        self.encode_compute_accelerations(&mut encoder, n, g, softening);
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    /// Record an nbody dispatch on a caller-owned encoder. Lets the
    /// caller batch many compute passes into one `queue.submit` to
    /// amortise driver overhead.
    pub fn encode_compute_accelerations(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        n: usize,
        g: f32,
        softening: f32,
    ) {
        if n == 0 {
            return;
        }
        let params = NBodyParams {
            num_particles: n as u32,
            g,
            eps2: softening * softening,
            ..NBodyParams::default()
        };
        self.ctx
            .queue
            .write_buffer(&self.nbody_params_buf, 0, bytemuck::bytes_of(&params));
        self.encode_nbody(encoder, n);
    }

    /// S231: compute gravity into the gravity-only buffer (`grav_accel`),
    /// leaving `accelerations` untouched. Used once per outer step so the
    /// sub-cycle can re-seed `accelerations = grav_accel` before each
    /// (cheap) SPH/stress force eval without re-running the O(N²) nbody.
    pub fn encode_gravity_into_grav(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        n: usize,
        g: f32,
        softening: f32,
    ) {
        if n == 0 {
            return;
        }
        let params = NBodyParams {
            num_particles: n as u32,
            g,
            eps2: softening * softening,
            ..NBodyParams::default()
        };
        self.ctx
            .queue
            .write_buffer(&self.nbody_params_buf, 0, bytemuck::bytes_of(&params));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("planet-nbody-grav-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.nbody_pipeline);
        pass.set_bind_group(0, &self.nbody_grav_bg, &[]);
        let wg = ((n as u32) + 127) / 128;
        pass.dispatch_workgroups(wg, 1, 1);
    }

    /// S231: seed `accelerations = grav_accel` (gravity-only) so the SPH /
    /// stress passes can add onto it within a sub-step.
    pub fn encode_copy_grav_to_accel(&self, encoder: &mut wgpu::CommandEncoder, n: usize) {
        if n == 0 {
            return;
        }
        let bytes = (n as u64) * 3 * std::mem::size_of::<f32>() as u64;
        encoder.copy_buffer_to_buffer(&self.grav_accel_buf, 0, &self.accelerations_buf, 0, bytes);
    }

    /// Half-kick dispatch — one of the two `v += dt/2 · a` updates in
    /// a KDK leapfrog. Owns its own encoder + submit; for SPH the
    /// caller invokes `kick → drift → density → nbody → pressure →
    /// viscosity → kick` directly so each force pass can refresh the
    /// shared accelerations buffer.
    pub fn kick(&self, n: usize, dt_half: f32) {
        if n == 0 {
            return;
        }
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-kick-encoder"),
            });
        self.encode_kick_step(&mut encoder, n, dt_half);
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    /// Record a half-kick on a caller-owned encoder.
    pub fn encode_kick_step(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        n: usize,
        dt_half: f32,
    ) {
        if n == 0 {
            return;
        }
        let params = StepParams {
            num_particles: n as u32,
            dt: dt_half,
            ..StepParams::default()
        };
        self.ctx
            .queue
            .write_buffer(&self.kick_params_buf, 0, bytemuck::bytes_of(&params));
        self.encode_kick(encoder, n);
    }

    /// Drift dispatch — `x += dt · v`.
    pub fn drift(&self, n: usize, dt: f32) {
        if n == 0 {
            return;
        }
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-drift-encoder"),
            });
        self.encode_drift_step(&mut encoder, n, dt);
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    /// Record a drift on a caller-owned encoder.
    pub fn encode_drift_step(&self, encoder: &mut wgpu::CommandEncoder, n: usize, dt: f32) {
        if n == 0 {
            return;
        }
        let params = StepParams {
            num_particles: n as u32,
            dt,
            ..StepParams::default()
        };
        self.ctx
            .queue
            .write_buffer(&self.drift_params_buf, 0, bytemuck::bytes_of(&params));
        self.encode_drift(encoder, n);
    }

    /// Full KDK leapfrog step on the GPU. Records all four dispatches
    /// into a single command encoder and submits.
    pub fn step_leapfrog(&self, n: usize, dt: f32, g: f32, softening: f32) {
        if n == 0 {
            return;
        }
        let kick_params = StepParams {
            num_particles: n as u32,
            dt: 0.5 * dt,
            ..StepParams::default()
        };
        let drift_params = StepParams {
            num_particles: n as u32,
            dt,
            ..StepParams::default()
        };
        let nbody_params = NBodyParams {
            num_particles: n as u32,
            g,
            eps2: softening * softening,
            ..NBodyParams::default()
        };
        let q = &self.ctx.queue;
        q.write_buffer(&self.kick_params_buf, 0, bytemuck::bytes_of(&kick_params));
        q.write_buffer(&self.drift_params_buf, 0, bytemuck::bytes_of(&drift_params));
        q.write_buffer(&self.nbody_params_buf, 0, bytemuck::bytes_of(&nbody_params));

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-step-encoder"),
            });
        self.encode_kick(&mut encoder, n);
        self.encode_drift(&mut encoder, n);
        self.encode_nbody(&mut encoder, n);
        self.encode_kick(&mut encoder, n);
        q.submit(Some(encoder.finish()));
    }

    pub fn positions_buffer(&self) -> &wgpu::Buffer {
        &self.positions_buf
    }
    pub fn velocities_buffer(&self) -> &wgpu::Buffer {
        &self.velocities_buf
    }
    pub fn accelerations_buffer(&self) -> &wgpu::Buffer {
        &self.accelerations_buf
    }
    pub fn masses_buffer(&self) -> &wgpu::Buffer {
        &self.masses_buf
    }
    pub fn smoothing_lengths_buffer(&self) -> &wgpu::Buffer {
        &self.smoothing_lengths_buf
    }
    pub fn densities_buffer(&self) -> &wgpu::Buffer {
        &self.densities_buf
    }
    pub fn internal_energies_buffer(&self) -> &wgpu::Buffer {
        &self.internal_energies_buf
    }
    pub fn du_dt_buffer(&self) -> &wgpu::Buffer {
        &self.du_dt_buf
    }
    pub fn phase_frac_buffer(&self) -> &wgpu::Buffer {
        &self.phase_fracs_buf
    }
    pub fn dev_stress_buffer(&self) -> &wgpu::Buffer {
        &self.dev_stress_buf
    }
    pub fn ds_dt_buffer(&self) -> &wgpu::Buffer {
        &self.ds_dt_buf
    }
    pub fn grad_correction_buffer(&self) -> &wgpu::Buffer {
        &self.grad_corr_buf
    }
    pub fn art_stress_buffer(&self) -> &wgpu::Buffer {
        &self.art_stress_buf
    }
    pub fn du_plastic_buffer(&self) -> &wgpu::Buffer {
        &self.du_plastic_buf
    }
    pub fn mat_rho0_buffer(&self) -> &wgpu::Buffer {
        &self.mat_rho0_buf
    }
    pub fn mat_t_m_buffer(&self) -> &wgpu::Buffer {
        &self.mat_t_m_buf
    }
    pub fn pressure_buffer(&self) -> &wgpu::Buffer {
        &self.pressure_buf
    }
    pub fn sound_speed_buffer(&self) -> &wgpu::Buffer {
        &self.sound_speed_buf
    }

    pub fn upload_materials(&self, mat_rho0: &[f32], mat_t_m: &[f32]) {
        if mat_rho0.is_empty() {
            return;
        }
        self.ctx
            .queue
            .write_buffer(&self.mat_rho0_buf, 0, bytemuck::cast_slice(mat_rho0));
        self.ctx
            .queue
            .write_buffer(&self.mat_t_m_buf, 0, bytemuck::cast_slice(mat_t_m));
    }

    /// Zero the persistent deviatoric stress buffer. Called once at init so
    /// a fresh body starts unstressed (`S = 0` ⇒ stress passes are a strict
    /// no-op until strain accumulates).
    pub fn clear_dev_stress(&self, n: usize) {
        if n == 0 {
            return;
        }
        let bytes = (n as u64) * 6 * std::mem::size_of::<f32>() as u64;
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-dev-stress-clear"),
            });
        encoder.clear_buffer(&self.dev_stress_buf, 0, Some(bytes));
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    /// Upload a packed `[Sxx,Syy,Szz,Sxy,Sxz,Syz]·N` deviatoric stress
    /// field. Test-only seam for imposing a known stress state.
    pub fn upload_dev_stress(&self, s: &[f32]) {
        if s.is_empty() {
            return;
        }
        self.ctx
            .queue
            .write_buffer(&self.dev_stress_buf, 0, bytemuck::cast_slice(s));
    }

    pub fn download_dev_stress(&self, n: usize) -> Vec<f32> {
        self.download_f32(&self.dev_stress_buf, &self.dev_stress_rb, n * 6)
    }
    pub fn download_ds_dt(&self, n: usize) -> Vec<f32> {
        self.download_f32(&self.ds_dt_buf, &self.dev_stress_rb, n * 6)
    }
    pub fn download_grad_correction(&self, n: usize) -> Vec<f32> {
        self.download_f32(&self.grad_corr_buf, &self.grad_corr_rb, n * 9)
    }
    pub fn download_art_stress(&self, n: usize) -> Vec<f32> {
        self.download_f32(&self.art_stress_buf, &self.dev_stress_rb, n * 6)
    }

    pub fn upload_smoothing_lengths(&self, h: &[f32]) {
        if h.is_empty() {
            return;
        }
        assert!(h.len() <= self.capacity);
        self.ctx
            .queue
            .write_buffer(&self.smoothing_lengths_buf, 0, bytemuck::cast_slice(h));
    }

    pub fn upload_densities(&self, rho: &[f32]) {
        if rho.is_empty() {
            return;
        }
        assert!(rho.len() <= self.capacity);
        self.ctx
            .queue
            .write_buffer(&self.densities_buf, 0, bytemuck::cast_slice(rho));
    }

    pub fn upload_internal_energies(&self, u: &[f32]) {
        if u.is_empty() {
            return;
        }
        assert!(u.len() <= self.capacity);
        self.ctx
            .queue
            .write_buffer(&self.internal_energies_buf, 0, bytemuck::cast_slice(u));
    }

    /// Zero the `du/dt` scratch buffer. Called once after initial upload
    /// so the first SPH dispatch sees a clean slate before overwriting.
    pub fn clear_du_dt(&self, n: usize) {
        if n == 0 {
            return;
        }
        let bytes = (n as u64) * std::mem::size_of::<f32>() as u64;
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-du-dt-clear"),
            });
        encoder.clear_buffer(&self.du_dt_buf, 0, Some(bytes));
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn download_smoothing_lengths(&self, n: usize) -> Vec<f32> {
        self.download_f32(&self.smoothing_lengths_buf, &self.smoothing_lengths_rb, n)
    }

    pub fn download_densities(&self, n: usize) -> Vec<f32> {
        self.download_f32(&self.densities_buf, &self.densities_rb, n)
    }

    pub fn download_internal_energies(&self, n: usize) -> Vec<f32> {
        self.download_f32(&self.internal_energies_buf, &self.internal_energies_rb, n)
    }

    pub fn download_phase_fracs(&self, n: usize) -> Vec<f32> {
        self.download_f32(&self.phase_fracs_buf, &self.phase_fracs_rb, n)
    }

    /// Readback of the `du/dt` scratch buffer. Used by thermal-pass unit
    /// tests (e.g. the conduction plateau oracle); not on the hot path.
    pub fn download_du_dt(&self, n: usize) -> Vec<f32> {
        self.download_f32(&self.du_dt_buf, &self.internal_energies_rb, n)
    }

    fn download_f32(&self, src: &wgpu::Buffer, rb: &wgpu::Buffer, n: usize) -> Vec<f32> {
        if n == 0 {
            return Vec::new();
        }
        let bytes = (n as u64) * 4;
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-f32-readback"),
            });
        encoder.copy_buffer_to_buffer(src, 0, rb, 0, bytes);
        self.ctx.queue.submit(Some(encoder.finish()));
        let slice = rb.slice(0..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.ctx.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        rb.unmap();
        out
    }

    fn encode_nbody(&self, encoder: &mut wgpu::CommandEncoder, n: usize) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("planet-nbody-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.nbody_pipeline);
        pass.set_bind_group(0, &self.nbody_bg, &[]);
        // nbody shader uses workgroup_size(128) — match the divisor.
        let wg = ((n as u32) + 127) / 128;
        pass.dispatch_workgroups(wg, 1, 1);
    }

    fn encode_kick(&self, encoder: &mut wgpu::CommandEncoder, n: usize) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("planet-kick-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.kick_pipeline);
        pass.set_bind_group(0, &self.kick_bg, &[]);
        let wg = ((n as u32) + 63) / 64;
        pass.dispatch_workgroups(wg, 1, 1);
    }

    fn encode_drift(&self, encoder: &mut wgpu::CommandEncoder, n: usize) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("planet-drift-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.drift_pipeline);
        pass.set_bind_group(0, &self.drift_bg, &[]);
        let wg = ((n as u32) + 63) / 64;
        pass.dispatch_workgroups(wg, 1, 1);
    }

    /// Render-path readback: positions + (optionally) internal energies
    /// in one encoder, one `queue.submit`, one `device.poll(Wait)`.
    /// Replaces 2× separate `download_positions` / `download_internal_energies`
    /// calls that each blocked the CPU on its own GPU sync.
    pub fn download_render(
        &self,
        n: usize,
        with_temperature: bool,
    ) -> (Vec<[f32; 3]>, Vec<f32>) {
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let pos_bytes = (n as u64) * 4 * 4; // vec4-packed positions
        let u_bytes = (n as u64) * 4;
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-render-readback"),
            });
        encoder.copy_buffer_to_buffer(
            &self.positions_buf,
            0,
            &self.positions_rb,
            0,
            pos_bytes,
        );
        if with_temperature {
            encoder.copy_buffer_to_buffer(
                &self.internal_energies_buf,
                0,
                &self.internal_energies_rb,
                0,
                u_bytes,
            );
        }
        self.ctx.queue.submit(Some(encoder.finish()));

        let pos_slice = self.positions_rb.slice(0..pos_bytes);
        pos_slice.map_async(wgpu::MapMode::Read, |_| {});
        let u_slice = if with_temperature {
            let s = self.internal_energies_rb.slice(0..u_bytes);
            s.map_async(wgpu::MapMode::Read, |_| {});
            Some(s)
        } else {
            None
        };
        self.ctx.device.poll(wgpu::Maintain::Wait);

        let pos_data = pos_slice.get_mapped_range();
        let pf: &[f32] = bytemuck::cast_slice(&pos_data);
        let positions: Vec<[f32; 3]> = (0..n)
            .map(|i| [pf[i * 4], pf[i * 4 + 1], pf[i * 4 + 2]])
            .collect();
        drop(pos_data);
        self.positions_rb.unmap();

        let internal_energies = if let Some(s) = u_slice {
            let d = s.get_mapped_range();
            let v: Vec<f32> = bytemuck::cast_slice(&d).to_vec();
            drop(d);
            self.internal_energies_rb.unmap();
            v
        } else {
            Vec::new()
        };
        (positions, internal_energies)
    }

    /// Full state readback (positions, velocities, accelerations, h, ρ, u,
    /// φ) using one encoder + one `device.poll(Wait)` — same shape as the
    /// separate `download_*` methods but far fewer GPU sync points.
    /// Returned in fixed order: (positions, velocities, accelerations,
    /// smoothing_lengths, densities, internal_energies, phase_fracs).
    #[allow(clippy::type_complexity)]
    pub fn download_full(
        &self,
        n: usize,
    ) -> (
        Vec<[f32; 3]>,
        Vec<[f32; 3]>,
        Vec<[f32; 3]>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    ) {
        if n == 0 {
            return (
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            );
        }
        let v3 = (n as u64) * 3 * 4;
        let v4 = (n as u64) * 4 * 4; // vec4-packed positions
        let v1 = (n as u64) * 4;
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-full-readback"),
            });
        encoder.copy_buffer_to_buffer(&self.positions_buf, 0, &self.positions_rb, 0, v4);
        encoder.copy_buffer_to_buffer(&self.velocities_buf, 0, &self.velocities_rb, 0, v4);
        encoder.copy_buffer_to_buffer(
            &self.accelerations_buf,
            0,
            &self.accelerations_rb,
            0,
            v3,
        );
        encoder.copy_buffer_to_buffer(
            &self.smoothing_lengths_buf,
            0,
            &self.smoothing_lengths_rb,
            0,
            v1,
        );
        encoder.copy_buffer_to_buffer(&self.densities_buf, 0, &self.densities_rb, 0, v1);
        encoder.copy_buffer_to_buffer(
            &self.internal_energies_buf,
            0,
            &self.internal_energies_rb,
            0,
            v1,
        );
        encoder.copy_buffer_to_buffer(&self.phase_fracs_buf, 0, &self.phase_fracs_rb, 0, v1);
        self.ctx.queue.submit(Some(encoder.finish()));

        let pos_s = self.positions_rb.slice(0..v4);
        let vel_s = self.velocities_rb.slice(0..v4);
        let acc_s = self.accelerations_rb.slice(0..v3);
        let h_s = self.smoothing_lengths_rb.slice(0..v1);
        let rho_s = self.densities_rb.slice(0..v1);
        let u_s = self.internal_energies_rb.slice(0..v1);
        let phi_s = self.phase_fracs_rb.slice(0..v1);
        for s in [&pos_s, &vel_s, &acc_s, &h_s, &rho_s, &u_s, &phi_s] {
            s.map_async(wgpu::MapMode::Read, |_| {});
        }
        self.ctx.device.poll(wgpu::Maintain::Wait);

        let to_vec3 = |slice: &wgpu::BufferSlice, n: usize| -> Vec<[f32; 3]> {
            let d = slice.get_mapped_range();
            let f: &[f32] = bytemuck::cast_slice(&d);
            (0..n)
                .map(|i| [f[i * 3], f[i * 3 + 1], f[i * 3 + 2]])
                .collect()
        };
        // vec4-packed (stride 4) — drop the .w padding.
        let to_vec3_s4 = |slice: &wgpu::BufferSlice, n: usize| -> Vec<[f32; 3]> {
            let d = slice.get_mapped_range();
            let f: &[f32] = bytemuck::cast_slice(&d);
            (0..n)
                .map(|i| [f[i * 4], f[i * 4 + 1], f[i * 4 + 2]])
                .collect()
        };
        let to_vec1 = |slice: &wgpu::BufferSlice| -> Vec<f32> {
            let d = slice.get_mapped_range();
            bytemuck::cast_slice(&d).to_vec()
        };

        let positions = to_vec3_s4(&pos_s, n);
        let velocities = to_vec3_s4(&vel_s, n);
        let accelerations = to_vec3(&acc_s, n);
        let smoothing_lengths = to_vec1(&h_s);
        let densities = to_vec1(&rho_s);
        let internal_energies = to_vec1(&u_s);
        let phase_fracs = to_vec1(&phi_s);

        self.positions_rb.unmap();
        self.velocities_rb.unmap();
        self.accelerations_rb.unmap();
        self.smoothing_lengths_rb.unmap();
        self.densities_rb.unmap();
        self.internal_energies_rb.unmap();
        self.phase_fracs_rb.unmap();

        (
            positions,
            velocities,
            accelerations,
            smoothing_lengths,
            densities,
            internal_energies,
            phase_fracs,
        )
    }

    pub fn download_positions(&self, n: usize) -> Vec<[f32; 3]> {
        self.download_vec4_xyz(&self.positions_buf, &self.positions_rb, n)
    }

    pub fn download_velocities(&self, n: usize) -> Vec<[f32; 3]> {
        self.download_vec4_xyz(&self.velocities_buf, &self.velocities_rb, n)
    }

    pub fn download_accelerations(&self, n: usize) -> Vec<[f32; 3]> {
        self.download_vec3(&self.accelerations_buf, &self.accelerations_rb, n)
    }

    fn download_vec3(
        &self,
        src: &wgpu::Buffer,
        rb: &wgpu::Buffer,
        n: usize,
    ) -> Vec<[f32; 3]> {
        if n == 0 {
            return Vec::new();
        }
        let bytes = (n as u64) * 3 * std::mem::size_of::<f32>() as u64;
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-readback"),
            });
        encoder.copy_buffer_to_buffer(src, 0, rb, 0, bytes);
        self.ctx.queue.submit(Some(encoder.finish()));
        let slice = rb.slice(0..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.ctx.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let f: &[f32] = bytemuck::cast_slice(&data);
        let out: Vec<[f32; 3]> = (0..n)
            .map(|i| [f[i * 3], f[i * 3 + 1], f[i * 3 + 2]])
            .collect();
        drop(data);
        rb.unmap();
        out
    }

    /// Readback for a vec4-packed (16-byte stride) buffer, dropping `.w`.
    fn download_vec4_xyz(
        &self,
        src: &wgpu::Buffer,
        rb: &wgpu::Buffer,
        n: usize,
    ) -> Vec<[f32; 3]> {
        if n == 0 {
            return Vec::new();
        }
        let bytes = (n as u64) * 4 * std::mem::size_of::<f32>() as u64;
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-readback-vec4"),
            });
        encoder.copy_buffer_to_buffer(src, 0, rb, 0, bytes);
        self.ctx.queue.submit(Some(encoder.finish()));
        let slice = rb.slice(0..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.ctx.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let f: &[f32] = bytemuck::cast_slice(&data);
        let out: Vec<[f32; 3]> = (0..n)
            .map(|i| [f[i * 4], f[i * 4 + 1], f[i * 4 + 2]])
            .collect();
        drop(data);
        rb.unmap();
        out
    }
}

fn make_layout(
    device: &wgpu::Device,
    label: &str,
    storage_read_only: &[bool],
) -> wgpu::BindGroupLayout {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = storage_read_only
        .iter()
        .enumerate()
        .map(|(i, &read_only)| {
            let ty = if i == 0 {
                wgpu::BufferBindingType::Uniform
            } else {
                wgpu::BufferBindingType::Storage { read_only }
            };
            wgpu::BindGroupLayoutEntry {
                binding: i as u32,
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
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn build_pipeline(
    device: &wgpu::Device,
    label: &str,
    shader: &wgpu::ShaderModule,
    entry: &str,
    layout: &wgpu::BindGroupLayout,
) -> (wgpu::ComputePipeline, wgpu::PipelineLayout) {
    let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}-pl")),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl_layout),
        module: shader,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    (pipeline, pl_layout)
}
