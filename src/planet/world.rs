//! `PlanetWorld` — central state + tick driver for the torus planet
//! experiment. Pre-S207 owns particle state only; GPU pipelines attach
//! incrementally as later sprints land.

use crate::gpu::GpuContext;
use crate::planet::gpu::{
    DensityGpu, PlanetGpu, SpatialHashGpu, SphForceGpu, ThermalIntegrateGpu,
};
use crate::planet::particle::Particles;
use clap::ValueEnum;
use std::sync::Arc;

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

pub struct PlanetWorld {
    pub particles: Particles,
    pub config: PlanetConfig,
    pub tick: u64,
    pub time: f32,
    pub gpu_ctx: Option<Arc<GpuContext>>,
    pub gpu_state: Option<PlanetGpu>,
    pub hash: Option<SpatialHashGpu>,
    pub density: Option<DensityGpu>,
    /// Merged pressure + Monaghan-viscosity pass. Replaced the two
    /// standalone pipelines in S220 (~30 % step speedup at N=25k).
    pub sph_force: Option<SphForceGpu>,
    /// Sprint 202: explicit-Euler integrator for the per-particle
    /// internal energy buffer + safety clamps + scratch clear.
    pub thermal_integrate: Option<ThermalIntegrateGpu>,
    /// Bounding-half of the world used by the spatial hash. Set in
    /// `init_gpu_full` from torus extent + 100 % slack so post-collapse
    /// particles still land in the hash grid.
    pub world_half: f32,
}

impl PlanetWorld {
    pub fn new(config: PlanetConfig) -> Self {
        let n = config.n_particles;
        Self {
            particles: Particles::with_capacity(n),
            config,
            tick: 0,
            time: 0.0,
            gpu_ctx: None,
            gpu_state: None,
            hash: None,
            density: None,
            sph_force: None,
            thermal_integrate: None,
            world_half: 2.5,
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

        let gpu = PlanetGpu::new(Arc::clone(&ctx), n)?;
        let hash = SpatialHashGpu::new(Arc::clone(&ctx), n, world_half, gpu.positions_buffer())?;
        let density = DensityGpu::new(Arc::clone(&ctx), &gpu, &hash)?;
        let sph_force = SphForceGpu::new(Arc::clone(&ctx), &gpu, &hash)?;
        let thermal_integrate = ThermalIntegrateGpu::new(Arc::clone(&ctx), &gpu)?;

        gpu.upload_state(
            &self.particles.positions,
            &self.particles.velocities,
            &self.particles.masses,
        );
        gpu.upload_smoothing_lengths(&self.particles.smoothing_lengths);
        gpu.upload_densities(&self.particles.densities);
        gpu.upload_internal_energies(&self.particles.internal_energies);
        gpu.clear_du_dt(n);

        // Seed `a_0`: density → gravity (OVERWRITES the acc buffer for
        // all valid i < n) → merged pressure+viscosity (adds; viscosity
        // is a no-op at t=0 unless any pair is already approaching).
        // No prior zero-fill needed — nbody.wgsl writes every valid slot.
        // sph_force also writes the initial `du/dt` (zero at t=0 since
        // viscosity is zero) which the first integrator call will apply.
        hash.rebuild(n);
        density.dispatch(n);
        gpu.compute_accelerations(n, self.config.g_const, self.config.softening);
        sph_force.dispatch(
            n,
            self.config.eos_k,
            self.config.eos_gamma,
            self.config.visc_alpha,
            self.config.visc_beta,
        );

        self.gpu_ctx = Some(ctx);
        self.gpu_state = Some(gpu);
        self.hash = Some(hash);
        self.density = Some(density);
        self.sph_force = Some(sph_force);
        self.thermal_integrate = Some(thermal_integrate);
        Ok(())
    }

    /// One KDK leapfrog step on the GPU with full SPH (density,
    /// pressure, viscosity) + self-gravity + thermal integration.
    /// Requires `init_gpu_full` to have been called first.
    pub fn tick_sph(&mut self) {
        let n = self.particles.len();
        if n == 0 {
            return;
        }
        let gpu = self.gpu_state.as_ref().expect("call init_gpu_full first");
        let hash = self.hash.as_ref().unwrap();
        let density = self.density.as_ref().unwrap();
        let sph_force = self.sph_force.as_ref().unwrap();
        let thermal_integrate = self.thermal_integrate.as_ref().unwrap();
        let dt = self.config.dt;

        gpu.kick(n, 0.5 * dt);
        gpu.drift(n, dt);
        hash.rebuild(n);
        density.dispatch(n);
        gpu.compute_accelerations(n, self.config.g_const, self.config.softening);
        sph_force.dispatch(
            n,
            self.config.eos_k,
            self.config.eos_gamma,
            self.config.visc_alpha,
            self.config.visc_beta,
        );
        thermal_integrate.dispatch(n, dt);
        gpu.kick(n, 0.5 * dt);

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
        let k = self.config.eos_k;
        let gamma = self.config.eos_gamma;
        let alpha = self.config.visc_alpha;
        let beta = self.config.visc_beta;
        {
            let gpu = self.gpu_state.as_ref().unwrap();
            let hash = self.hash.as_ref().unwrap();
            let density = self.density.as_ref().unwrap();
            let sph_force = self.sph_force.as_ref().unwrap();
            gpu.upload_state(
                &self.particles.positions,
                &self.particles.velocities,
                &self.particles.masses,
            );
            gpu.upload_smoothing_lengths(&self.particles.smoothing_lengths);
            gpu.upload_densities(&self.particles.densities);
            gpu.upload_internal_energies(&self.particles.internal_energies);
            gpu.clear_du_dt(n);
            hash.rebuild(n);
            density.dispatch(n);
            gpu.compute_accelerations(n, g, softening);
            sph_force.dispatch(n, k, gamma, alpha, beta);
        }
        // Pull h / ρ back so CPU-side diagnostics see the post-density
        // values rather than the analytic init estimates.
        self.download_state();
    }

    /// Pull GPU state back into `self.particles`. Used between ticks
    /// to drive CPU-side diagnostics or persistence.
    pub fn download_state(&mut self) {
        let n = self.particles.len();
        if n == 0 {
            return;
        }
        let gpu = self.gpu_state.as_ref().expect("call init_gpu_full first");
        self.particles.positions = gpu.download_positions(n);
        self.particles.velocities = gpu.download_velocities(n);
        self.particles.accelerations = gpu.download_accelerations(n);
        self.particles.smoothing_lengths = gpu.download_smoothing_lengths(n);
        self.particles.densities = gpu.download_densities(n);
        self.particles.internal_energies = gpu.download_internal_energies(n);
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
