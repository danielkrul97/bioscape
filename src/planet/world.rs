//! `PlanetWorld` — central state + tick driver for the torus planet
//! experiment. Pre-S207 owns particle state only; GPU pipelines attach
//! incrementally as later sprints land.

use crate::gpu::GpuContext;
use crate::planet::gpu::{
    ArtificialStressGpu, DensityGpu, EosGpu, ExtentGpu, GradCorrectionGpu, PhaseGpu, PlanetGpu,
    SpatialHashGpu, SphForceGpu, StressIntegrateGpu, StressRateGpu, ThermalConductionGpu,
    ThermalIntegrateGpu,
};
use crate::planet::particle::Particles;
use clap::ValueEnum;
use std::sync::Arc;
use wgpu;

/// Initial particle distribution shape. Sets which generator
/// `init::generate` calls and which size fields in `PlanetConfig`
/// are read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum PlanetShape {
    /// Solid torus, parametrised by `r_major` + `r_minor`. The
    /// historical default that the rest of the documentation talks
    /// about.
    #[default]
    Torus,
    /// Axis-aligned cube, parametrised by `cube_side` (full edge
    /// length, centred on origin).
    Cube,
    /// Flat disc (oblate cylinder), parametrised by `pancake_radius`
    /// and `pancake_height`.
    Pancake,
}

/// Tunables for one experiment run. All units are normalised
/// (`G = M_total = R_major = 1`); see `docs/sprints/203-212-torus-planet-sph.md`
/// for the SI conversion table.
#[derive(Debug, Clone)]
pub struct PlanetConfig {
    pub shape: PlanetShape,
    pub g_const: f32,
    pub softening: f32,
    pub dt: f32,
    pub total_mass: f32,
    /// Torus major radius (distance from z-axis to tube centre).
    pub r_major: f32,
    /// Torus minor radius (tube cross-section radius).
    pub r_minor: f32,
    /// Cube edge length (centred on origin, extent `±cube_side/2`).
    pub cube_side: f32,
    /// Pancake disc radius.
    pub pancake_radius: f32,
    /// Pancake disc thickness along z (extent `±pancake_height/2`).
    pub pancake_height: f32,
    pub omega: f32,
    pub seed: u64,
    pub n_particles: usize,
    /// SPH polytropic exponent. Defaults to 5/3 (monatomic ideal gas).
    pub eos_gamma: f32,
    /// SPH polytropic coefficient K in `P = K · rho^gamma`.
    pub eos_k: f32,
    /// Monaghan artificial-viscosity linear term (α). 1.0 is the
    /// standard choice (Monaghan 1992).
    pub visc_alpha: f32,
    /// Monaghan artificial-viscosity quadratic term (β). 2.0 is the
    /// standard choice (Monaghan 1992).
    pub visc_beta: f32,
    /// Sprint 231: number of elastic sub-steps per outer (gravity) step.
    /// 1 = no sub-cycling (default path, unchanged). >1 enables the
    /// operator-split inner loop so stiff solids stay inside the elastic
    /// CFL at the same outer `dt` while gravity (O(N²)) runs once.
    pub n_substeps: u32,
    /// Sprint 231: solid shear modulus G0 (default = `SHEAR_MODULUS_G0`).
    /// Crank up (with `n_substeps > 1`) for rigid "rock" blocks.
    pub shear_modulus: f32,
    /// Sprint 231: condensed reference sound speed c0 (default = `TAIT_REF_SOUND_SPEED_C0`).
    pub tait_c0: f32,
    /// Sprint 231: Tait exponent n (default = `TAIT_EXPONENT_N`).
    pub tait_exponent: f32,
    /// Sprint 231: von Mises yield strength Y0 (default = `YIELD_STRENGTH_Y0`).
    pub yield_strength: f32,
}

impl Default for PlanetConfig {
    fn default() -> Self {
        // Size defaults are chosen so all three shapes have the same
        // initial volume `V ≈ 0.790` (the torus value with R=1, r=0.2):
        //   V_torus   = 2π² · 1 · 0.04           ≈ 0.7896
        //   V_cube    = 0.924³                   ≈ 0.7895
        //   V_pancake = π · 1.0² · 0.251         ≈ 0.7886
        // → comparable mean density, t_ff, and CFL across shapes so
        // batch sweeps remain like-for-like.
        Self {
            shape: PlanetShape::Torus,
            g_const: 1.0,
            softening: 0.01,
            dt: 1e-3,
            total_mass: 1.0,
            r_major: 1.0,
            r_minor: 0.2,
            cube_side: 0.924,
            pancake_radius: 1.0,
            pancake_height: 0.251,
            omega: 0.0,
            seed: 0,
            n_particles: 10_000,
            eos_gamma: 5.0 / 3.0,
            eos_k: 0.1,
            visc_alpha: 1.0,
            visc_beta: 2.0,
            n_substeps: 1,
            shear_modulus: crate::planet::thermal::SHEAR_MODULUS_G0,
            tait_c0: crate::planet::thermal::TAIT_REF_SOUND_SPEED_C0,
            tait_exponent: crate::planet::thermal::TAIT_EXPONENT_N,
            yield_strength: crate::planet::thermal::YIELD_STRENGTH_Y0,
        }
    }
}

/// "Primary radius" of the configured shape — the length scale used
/// for `t_ff`, `Ω_circ`, and any other characteristic-time formula.
/// Torus uses the major radius; cube uses half the edge length;
/// pancake uses the disc radius.
pub fn primary_radius(config: &PlanetConfig) -> f32 {
    match config.shape {
        PlanetShape::Torus => config.r_major,
        PlanetShape::Cube => 0.5 * config.cube_side,
        PlanetShape::Pancake => config.pancake_radius,
    }
}

/// Largest bounding-box extent of the configured shape (from origin).
/// `init_gpu_full` multiplies this by a slack factor to size the
/// spatial-hash grid.
pub fn shape_max_extent(config: &PlanetConfig) -> f32 {
    match config.shape {
        PlanetShape::Torus => config.r_major + config.r_minor,
        PlanetShape::Cube => 0.5 * config.cube_side,
        PlanetShape::Pancake => config.pancake_radius.max(0.5 * config.pancake_height),
    }
}

/// Initial mean density `M / V` for the configured shape — analytic
/// volume, no SPH kernel involved. Used by the radiation pass to pick
/// a surface threshold (`ρ < frac · ρ_mean ⇒ surface`).
pub fn shape_volume(config: &PlanetConfig) -> f32 {
    let pi = std::f32::consts::PI;
    match config.shape {
        PlanetShape::Torus => 2.0 * pi * pi * config.r_major * config.r_minor * config.r_minor,
        PlanetShape::Cube => config.cube_side.powi(3),
        PlanetShape::Pancake => pi * config.pancake_radius * config.pancake_radius * config.pancake_height,
    }
}

pub fn rho_mean_init(config: &PlanetConfig) -> f32 {
    let v = shape_volume(config).max(1e-30);
    config.total_mass / v
}

pub struct PlanetWorld {
    pub particles: Particles,
    pub config: PlanetConfig,
    pub tick: u64,
    pub time: f32,
    pub gpu_ctx: Option<Arc<GpuContext>>,
    pub gpu_state: Option<PlanetGpu>,
    pub hash: Option<SpatialHashGpu>,
    pub density: Option<DensityGpu>,
    /// Per-particle EoS precompute (pressure + sound speed). Runs after
    /// density so the force loop reads `P`/`c` instead of recomputing the
    /// EoS for every neighbour pair.
    pub eos: Option<EosGpu>,
    /// Merged pressure + Monaghan-viscosity pass. Replaced the two
    /// standalone pipelines in S220 (~30 % step speedup at N=25k).
    pub sph_force: Option<SphForceGpu>,
    /// Sprint 202: explicit-Euler integrator for the per-particle
    /// internal energy buffer + safety clamps + scratch clear.
    pub thermal_integrate: Option<ThermalIntegrateGpu>,
    /// Sprint 204: Cleary–Monaghan SPH thermal conduction pass.
    pub thermal_conduction: Option<ThermalConductionGpu>,
    /// Sprint 223: enthalpy phase pass — maps `u` to solid fraction `phi`
    /// into the GPU `phase_frac` buffer end-of-tick (CPU-readable cache).
    pub phase: Option<PhaseGpu>,
    /// Sprint 225: elastic-solid stress passes — Bonet–Lok gradient
    /// correction, Jaumann/Hooke deviatoric stress rate, explicit integrate.
    pub grad_correction: Option<GradCorrectionGpu>,
    pub stress_rate: Option<StressRateGpu>,
    pub stress_integrate: Option<StressIntegrateGpu>,
    /// Sprint 228: Monaghan-2000 artificial-stress pass (tensile-instability
    /// cure + cohesion). Runs after stress_integrate, before sph_force.
    pub artificial_stress: Option<ArtificialStressGpu>,
    /// Bounding-extent reduction driving the adaptive grid resize (H1).
    pub extent: Option<ExtentGpu>,
    /// Bounding-half of the world used by the spatial hash. Set in
    /// `init_gpu_full` and then adapted each `RESIZE_EVERY` ticks to track
    /// the collapsing body (H1) so cell_size follows the occupied region.
    pub world_half: f32,
    /// Initial (and maximum) `world_half` — the adaptive resize never grows
    /// past this conservatively-sized starting grid.
    pub initial_world_half: f32,
    /// Sprint 205: initial mean density (analytic `M/V`). Constant for
    /// the run; reused every tick as the radiation pass's surface
    /// threshold via `ρ < frac · ρ_mean_init`.
    pub rho_mean_init: f32,
}

impl PlanetWorld {
    pub fn new(config: PlanetConfig) -> Self {
        let n = config.n_particles;
        let rho_mean = rho_mean_init(&config);
        Self {
            particles: Particles::with_capacity(n),
            config,
            tick: 0,
            time: 0.0,
            gpu_ctx: None,
            gpu_state: None,
            hash: None,
            density: None,
            eos: None,
            sph_force: None,
            thermal_integrate: None,
            thermal_conduction: None,
            phase: None,
            grad_correction: None,
            stress_rate: None,
            stress_integrate: None,
            artificial_stress: None,
            extent: None,
            world_half: 2.5,
            initial_world_half: 2.5,
            rho_mean_init: rho_mean,
        }
    }

    /// Initialise the shared GPU context. Skipped pre-S206 — the CPU
    /// integrator runs without one.
    pub fn init_gpu(&mut self) -> Result<(), String> {
        if self.gpu_ctx.is_none() {
            self.gpu_ctx = Some(Arc::new(GpuContext::new()?));
        }
        Ok(())
    }

    /// Allocate the full SPH+gravity GPU pipeline, upload the current
    /// particle state, and compute the initial acceleration field
    /// (gravity + pressure + viscosity from current ρ, h).
    pub fn init_gpu_full(&mut self) -> Result<(), String> {
        if self.particles.is_empty() {
            return Err("no particles to upload — call init.torus_uniform first".into());
        }
        let ctx = Arc::new(GpuContext::new()?);
        let n = self.particles.len();
        // 50 % slack over the shape's largest bounding extent. Earlier
        // 100 % gave coarse buckets (cell_size ~ 0.16 vs h_init ~ 0.05)
        // so each particle scanned 3-4× more neighbours than necessary.
        // Tight grid → SPH passes 2-3× faster at the price of less
        // headroom for runaway expansion (a particle past `world_half`
        // clamps to the edge bucket and pollutes neighbour lookups, but
        // the integration is still stable).
        let world_half = shape_max_extent(&self.config) * 1.5;
        self.world_half = world_half;
        self.initial_world_half = world_half;

        let gpu = PlanetGpu::new(Arc::clone(&ctx), n)?;
        let hash = SpatialHashGpu::new(Arc::clone(&ctx), n, world_half, gpu.positions_buffer())?;
        let density = DensityGpu::new(Arc::clone(&ctx), &gpu, &hash)?;
        let mut eos = EosGpu::new(Arc::clone(&ctx), &gpu)?;
        let mut sph_force = SphForceGpu::new(Arc::clone(&ctx), &gpu, &hash)?;
        let thermal_conduction = ThermalConductionGpu::new(Arc::clone(&ctx), &gpu, &hash)?;
        let thermal_integrate = ThermalIntegrateGpu::new(Arc::clone(&ctx), &gpu)?;
        let phase = PhaseGpu::new(Arc::clone(&ctx), &gpu)?;
        let grad_correction = GradCorrectionGpu::new(Arc::clone(&ctx), &gpu, &hash)?;
        let mut stress_rate = StressRateGpu::new(Arc::clone(&ctx), &gpu, &hash)?;
        let mut stress_integrate = StressIntegrateGpu::new(Arc::clone(&ctx), &gpu)?;
        let mut artificial_stress = ArtificialStressGpu::new(Arc::clone(&ctx), &gpu)?;
        let extent = ExtentGpu::new(Arc::clone(&ctx), &gpu)?;
        // S231: apply configurable stiffness (defaults = thermal consts).
        sph_force.set_stiffness(self.config.tait_c0, self.config.tait_exponent);
        eos.set_stiffness(self.config.tait_c0, self.config.tait_exponent);
        stress_rate.set_g0(self.config.shear_modulus);
        stress_integrate.set_stiffness(self.config.yield_strength, self.config.shear_modulus);
        artificial_stress.set_stiffness(self.config.tait_c0, self.config.tait_exponent);

        gpu.upload_state(
            &self.particles.positions,
            &self.particles.velocities,
            &self.particles.masses,
        );
        gpu.upload_smoothing_lengths(&self.particles.smoothing_lengths);
        gpu.upload_densities(&self.particles.densities);
        gpu.upload_internal_energies(&self.particles.internal_energies);
        gpu.upload_materials(&self.particles.mat_rho0, &self.particles.mat_t_m);
        gpu.clear_du_dt(n);
        gpu.clear_dev_stress(n);

        // Seed `a_0`: density → gravity (OVERWRITES the acc buffer for
        // all valid i < n) → merged pressure+viscosity (adds; viscosity
        // is a no-op at t=0 unless any pair is already approaching).
        // No prior zero-fill needed — nbody.wgsl writes every valid slot.
        // sph_force also writes the initial `du/dt` (zero at t=0 since
        // viscosity is zero) which the first integrator call will apply.
        hash.rebuild(n);
        density.dispatch(n);
        eos.dispatch(n, self.config.eos_gamma);
        gpu.compute_accelerations(n, self.config.g_const, self.config.softening);
        artificial_stress.dispatch(n, self.rho_mean_init, self.config.eos_gamma);
        sph_force.dispatch(
            n,
            self.rho_mean_init,
            self.config.eos_gamma,
            self.config.visc_alpha,
            self.config.visc_beta,
        );
        // Seed the phase_frac buffer from the uploaded internal energies so
        // CPU diagnostics see a valid solid fraction before the first tick.
        phase.dispatch(n);

        self.gpu_ctx = Some(ctx);
        self.gpu_state = Some(gpu);
        self.hash = Some(hash);
        self.density = Some(density);
        self.eos = Some(eos);
        self.sph_force = Some(sph_force);
        self.thermal_conduction = Some(thermal_conduction);
        self.thermal_integrate = Some(thermal_integrate);
        self.phase = Some(phase);
        self.grad_correction = Some(grad_correction);
        self.stress_rate = Some(stress_rate);
        self.stress_integrate = Some(stress_integrate);
        self.artificial_stress = Some(artificial_stress);
        self.extent = Some(extent);
        Ok(())
    }

    /// Adapt the spatial-hash grid to the current body so the fixed 32³ grid
    /// tracks gravitational collapse. Every `RESIZE_EVERY` ticks reduce the
    /// bounding extent + max smoothing length on the GPU and pick
    ///   world_half = clamp(max(1.05·extent, K·max_h), floor, initial)
    /// — the `extent` term keeps every particle inside the grid, the `max_h`
    /// term keeps `cell_size ≥ (4/3)·max_h` so `h` is never clamped harder
    /// than before (the neighbour SET is unchanged; only cell bucketing, and
    /// thus float summation order, shifts). Deterministic: the reduction is an
    /// exact atomicMax and the gate is a deterministic threshold.
    fn maybe_resize_grid(&mut self, n: usize) {
        const RESIZE_EVERY: u64 = 64;
        // h_max = 0.75·cell_size and cell_size = 2·world_half/GRID_N, so
        // world_half = (GRID_N/2)·(4/3)·max_h·margin keeps h_max ≥ margin·max_h.
        const H_HEADROOM: f32 = 1.15;
        if self.extent.is_none() || self.tick % RESIZE_EVERY != 0 {
            return;
        }
        // Escape hatch: pin the grid to its initial size (pre-H1 behaviour).
        if std::env::var_os("BIOSCAPE_STATIC_GRID").is_some() {
            return;
        }
        let rho_surface = crate::planet::thermal::SURFACE_DENSITY_FRAC * self.rho_mean_init;
        let (max_coord, max_h) = self.extent.as_ref().unwrap().compute(n, rho_surface);
        if !(max_coord.is_finite() && max_h.is_finite()) || max_coord <= 0.0 {
            return;
        }
        let grid_n = crate::planet::gpu::GRID_N as f32;
        let from_h = 0.5 * grid_n * (4.0 / 3.0) * max_h * H_HEADROOM;
        let floor = 0.05 * self.initial_world_half;
        let target = (1.05 * max_coord).max(from_h).clamp(floor, self.initial_world_half);
        // Only re-grid on a meaningful change to avoid churn when settled.
        if (target - self.world_half).abs() <= 0.02 * self.world_half {
            return;
        }
        self.apply_grid(target);
    }

    /// Push a new `world_half` (and derived `cell_size`) to every pass that
    /// buckets by position, so they all agree on the grid. The hash derives
    /// `cell_size` from `world_half`; the rest store it explicitly.
    fn apply_grid(&mut self, world_half: f32) {
        self.world_half = world_half;
        let cell_size = 2.0 * world_half / crate::planet::gpu::GRID_N as f32;
        self.hash.as_mut().unwrap().set_grid(world_half, cell_size);
        self.density.as_mut().unwrap().set_grid(world_half, cell_size);
        self.sph_force.as_mut().unwrap().set_grid(world_half, cell_size);
        self.thermal_conduction.as_mut().unwrap().set_grid(world_half, cell_size);
        self.grad_correction.as_mut().unwrap().set_grid(world_half, cell_size);
        self.stress_rate.as_mut().unwrap().set_grid(world_half, cell_size);
    }

    /// One KDK leapfrog step on the GPU with full SPH (density,
    /// pressure, viscosity) + self-gravity + thermal integration.
    /// Requires `init_gpu_full` to have been called first.
    ///
    /// All 9 compute passes share a single command encoder and one
    /// `queue.submit`. Earlier code created a fresh encoder + submit
    /// per pass, which compounded driver overhead (~9× per tick →
    /// 36× per 4-step frame) and starved the GPU between dispatches.
    pub fn tick_sph(&mut self) {
        let n = self.particles.len();
        if n == 0 {
            return;
        }
        if self.config.n_substeps > 1 {
            self.tick_sph_substepped(n);
            return;
        }
        self.maybe_resize_grid(n);
        let ctx = self.gpu_ctx.as_ref().expect("call init_gpu_full first");
        let gpu = self.gpu_state.as_ref().unwrap();
        let hash = self.hash.as_ref().unwrap();
        let density = self.density.as_ref().unwrap();
        let eos = self.eos.as_ref().unwrap();
        let sph_force = self.sph_force.as_ref().unwrap();
        let thermal_conduction = self.thermal_conduction.as_ref().unwrap();
        let thermal_integrate = self.thermal_integrate.as_ref().unwrap();
        let phase = self.phase.as_ref().unwrap();
        let grad_correction = self.grad_correction.as_ref().unwrap();
        let stress_rate = self.stress_rate.as_ref().unwrap();
        let stress_integrate = self.stress_integrate.as_ref().unwrap();
        let artificial_stress = self.artificial_stress.as_ref().unwrap();
        let dt = self.config.dt;

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-tick-sph"),
            });
        gpu.encode_kick_step(&mut encoder, n, 0.5 * dt);
        gpu.encode_drift_step(&mut encoder, n, dt);
        hash.encode_rebuild(&mut encoder, n);
        density.encode(&mut encoder, n);
        // Pressure + sound speed once per particle from the final ρ/u, read
        // by the force pass instead of recomputing the EoS per neighbour.
        eos.encode(&mut encoder, n, self.config.eos_gamma);
        // Elastic-solid stress (S225): corrected gradient → Jaumann rate →
        // explicit integrate. Runs after density (final ρ/h); from S226 it
        // is read by sph_force, so it must precede the force pass.
        grad_correction.encode(&mut encoder, n);
        stress_rate.encode(&mut encoder, n);
        stress_integrate.encode(&mut encoder, n, dt);
        // Artificial stress from the just-updated total stress, read by
        // sph_force this tick (tensile-instability cure + cohesion, S228).
        artificial_stress.encode(&mut encoder, n, self.rho_mean_init, self.config.eos_gamma);
        gpu.encode_compute_accelerations(
            &mut encoder,
            n,
            self.config.g_const,
            self.config.softening,
        );
        sph_force.encode(
            &mut encoder,
            n,
            self.rho_mean_init,
            self.config.eos_gamma,
            self.config.visc_alpha,
            self.config.visc_beta,
        );
        thermal_conduction.encode(&mut encoder, n);
        thermal_integrate.encode(&mut encoder, n, dt, self.rho_mean_init);
        // Refresh the CPU-readable solid fraction from the just-updated u.
        phase.encode(&mut encoder, n);
        gpu.encode_kick_step(&mut encoder, n, 0.5 * dt);
        ctx.queue.submit(Some(encoder.finish()));

        self.tick += 1;
        self.time += dt;
    }

    /// S231 — operator-split sub-cycled tick for stiff solids. Gravity
    /// (O(N²)) is computed ONCE per outer step into `grav_accel` and held
    /// fixed (the gravitational field changes negligibly over `dt ≪ t_ff`);
    /// the stiff elastic/pressure physics is sub-cycled `n_substeps` times
    /// on `dt_sub = dt/n_sub`, keeping it inside the elastic CFL at the same
    /// outer `dt`. Each sub-step re-seeds `accelerations = grav_accel` then
    /// adds the SPH + deviatoric + cohesion forces. Used only when
    /// `config.n_substeps > 1`; the single-step path is unchanged.
    fn tick_sph_substepped(&mut self, n: usize) {
        self.maybe_resize_grid(n);
        let ctx = self.gpu_ctx.as_ref().expect("call init_gpu_full first");
        let gpu = self.gpu_state.as_ref().unwrap();
        let hash = self.hash.as_ref().unwrap();
        let density = self.density.as_ref().unwrap();
        let eos = self.eos.as_ref().unwrap();
        let sph_force = self.sph_force.as_ref().unwrap();
        let thermal_conduction = self.thermal_conduction.as_ref().unwrap();
        let thermal_integrate = self.thermal_integrate.as_ref().unwrap();
        let phase = self.phase.as_ref().unwrap();
        let grad_correction = self.grad_correction.as_ref().unwrap();
        let stress_rate = self.stress_rate.as_ref().unwrap();
        let stress_integrate = self.stress_integrate.as_ref().unwrap();
        let artificial_stress = self.artificial_stress.as_ref().unwrap();
        let dt = self.config.dt;
        let n_sub = self.config.n_substeps.max(1);
        let dt_sub = dt / n_sub as f32;
        let g = self.config.g_const;
        let eps = self.config.softening;
        let rho0 = self.rho_mean_init;
        let gamma = self.config.eos_gamma;
        let alpha = self.config.visc_alpha;
        let beta = self.config.visc_beta;

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-tick-sph-substepped"),
            });
        // Slow physics, once per outer step: gravity → grav_accel (held fixed).
        gpu.encode_gravity_into_grav(&mut encoder, n, g, eps);
        for _ in 0..n_sub {
            gpu.encode_kick_step(&mut encoder, n, 0.5 * dt_sub);
            gpu.encode_drift_step(&mut encoder, n, dt_sub);
            hash.encode_rebuild(&mut encoder, n);
            density.encode(&mut encoder, n);
            eos.encode(&mut encoder, n, gamma);
            grad_correction.encode(&mut encoder, n);
            stress_rate.encode(&mut encoder, n);
            stress_integrate.encode(&mut encoder, n, dt_sub);
            artificial_stress.encode(&mut encoder, n, rho0, gamma);
            // Re-seed gravity, then add the (sub-cycled) SPH + stress forces.
            gpu.encode_copy_grav_to_accel(&mut encoder, n);
            sph_force.encode(&mut encoder, n, rho0, gamma, alpha, beta);
            thermal_conduction.encode(&mut encoder, n);
            thermal_integrate.encode(&mut encoder, n, dt_sub, rho0);
            gpu.encode_kick_step(&mut encoder, n, 0.5 * dt_sub);
        }
        phase.encode(&mut encoder, n);
        ctx.queue.submit(Some(encoder.finish()));

        self.tick += 1;
        self.time += dt;
    }

    /// Regenerate particles from `self.config` and re-seed the GPU
    /// pipeline. Deterministic: same `seed` ⇒ identical post-reset
    /// state. Reuses the existing buffers and bind groups, so the
    /// only cost is one upload + the per-tick force passes (density,
    /// gravity, pressure, viscosity). Tick counter and time are
    /// rolled back to zero.
    pub fn reset(&mut self) {
        self.particles = crate::planet::init::generate(&self.config);
        self.tick = 0;
        self.time = 0.0;
        let n = self.particles.len();
        if n == 0 || self.gpu_state.is_none() {
            return;
        }
        let g = self.config.g_const;
        let softening = self.config.softening;
        let rho0 = self.rho_mean_init;
        let gamma = self.config.eos_gamma;
        let alpha = self.config.visc_alpha;
        let beta = self.config.visc_beta;
        // The regenerated body is back at its initial extent; restore the full
        // grid (a collapse-shrunk one wouldn't contain it).
        self.apply_grid(self.initial_world_half);
        {
            let gpu = self.gpu_state.as_ref().unwrap();
            let hash = self.hash.as_ref().unwrap();
            let density = self.density.as_ref().unwrap();
            let eos = self.eos.as_ref().unwrap();
            let sph_force = self.sph_force.as_ref().unwrap();
            let phase = self.phase.as_ref().unwrap();
            let artificial_stress = self.artificial_stress.as_ref().unwrap();
            gpu.upload_state(
                &self.particles.positions,
                &self.particles.velocities,
                &self.particles.masses,
            );
            gpu.upload_smoothing_lengths(&self.particles.smoothing_lengths);
            gpu.upload_densities(&self.particles.densities);
            gpu.upload_internal_energies(&self.particles.internal_energies);
            gpu.upload_materials(&self.particles.mat_rho0, &self.particles.mat_t_m);
            gpu.clear_du_dt(n);
            gpu.clear_dev_stress(n);
            hash.rebuild(n);
            density.dispatch(n);
            eos.dispatch(n, gamma);
            gpu.compute_accelerations(n, g, softening);
            artificial_stress.dispatch(n, rho0, gamma);
            sph_force.dispatch(n, rho0, gamma, alpha, beta);
            phase.dispatch(n);
        }
        // Pull h / ρ back so CPU-side diagnostics see the post-density
        // values rather than the analytic init estimates.
        self.download_state();
    }

    /// Pull GPU state back into `self.particles`. Used between ticks
    /// to drive CPU-side diagnostics or persistence. All six buffers
    /// stream back through one encoder + one `poll(Wait)` — earlier
    /// per-buffer downloads compounded six CPU stalls per call.
    pub fn download_state(&mut self) {
        let n = self.particles.len();
        if n == 0 {
            return;
        }
        let gpu = self.gpu_state.as_ref().expect("call init_gpu_full first");
        let (pos, vel, acc, h, rho, u, phi) = gpu.download_full(n);
        self.particles.positions = pos;
        self.particles.velocities = vel;
        self.particles.accelerations = acc;
        self.particles.smoothing_lengths = h;
        self.particles.densities = rho;
        self.particles.internal_energies = u;
        self.particles.phase_fracs = phi;
    }

    /// Minimal render-path readback: positions, plus internal energies
    /// when the caller renders thermal colouring. Single `poll(Wait)`.
    /// Velocities / densities / etc. stay stale until the next full
    /// `download_state` (HUD throttled to a few Hz is fine).
    pub fn download_for_render(&mut self, with_temperature: bool) {
        let n = self.particles.len();
        if n == 0 {
            return;
        }
        let gpu = self.gpu_state.as_ref().expect("call init_gpu_full first");
        let (pos, u) = gpu.download_render(n, with_temperature);
        self.particles.positions = pos;
        if with_temperature {
            self.particles.internal_energies = u;
        }
    }

    /// Free-fall time scale `t_ff = sqrt(R³ / (G·M))` using the
    /// configured shape's primary radius (see `primary_radius`).
    pub fn t_ff(&self) -> f32 {
        let r = primary_radius(&self.config);
        let r3 = r.powi(3);
        let gm = self.config.g_const * self.config.total_mass;
        (r3 / gm.max(1e-30)).sqrt()
    }

    /// Critical Keplerian rotation rate at the primary radius:
    /// `Ω_circ = sqrt(GM / R³)`.
    pub fn omega_circ(&self) -> f32 {
        1.0 / self.t_ff()
    }

    /// Advance one step. S205: CPU leapfrog + CPU N² gravity. S207
    /// swaps the body for GPU dispatches; the CPU path stays available
    /// for small-N validation and as the GPU reference oracle.
    pub fn tick(&mut self) {
        let g = self.config.g_const;
        let eps = self.config.softening;
        let dt = self.config.dt;
        crate::planet::integrator::leapfrog_step(&mut self.particles, dt, |p| {
            crate::planet::gravity_cpu::compute_acceleration(p, g, eps);
        });
        self.tick += 1;
        self.time += dt;
    }

    /// Initialise accelerations (call once before the first `tick`)
    /// so the first leapfrog half-kick uses a valid `a_0`.
    pub fn seed_accelerations(&mut self) {
        crate::planet::gravity_cpu::compute_acceleration(
            &mut self.particles,
            self.config.g_const,
            self.config.softening,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_t_ff_unity() {
        let w = PlanetWorld::new(PlanetConfig::default());
        let t = w.t_ff();
        assert!((t - 1.0).abs() < 1e-6, "t_ff = {t}");
        assert!((w.omega_circ() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn tick_advances_clock() {
        let mut w = PlanetWorld::new(PlanetConfig::default());
        let dt = w.config.dt;
        w.tick();
        assert_eq!(w.tick, 1);
        assert!((w.time - dt).abs() < 1e-9);
    }
}
