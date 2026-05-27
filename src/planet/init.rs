//! Initial state generators for the planet shape experiments. One
//! generator per `PlanetShape` variant + a `generate` dispatcher that
//! routes by `config.shape`.
//!
//! All generators write a `Particles` with positions, velocities
//! (rigid rotation about z if `config.omega ≠ 0`), zero accelerations,
//! per-particle mass `m = M_total / N`, and a uniform initial
//! smoothing length derived from the analytic mean density
//! `ρ̄ = M_total / V_shape`. Volumes:
//!
//! - **Torus**: `V = 2π² R r²`, rejection-sampled inside the implicit
//!   surface `(√(x²+y²) − R)² + z² ≤ r²` from a bounding box.
//!   Acceptance ≈ 34 % at `R=1, r=0.2`.
//! - **Cube**: `V = side³`, uniform in `[-side/2, side/2]³`, 100 %
//!   acceptance.
//! - **Pancake**: `V = π R² h`, uniform-area disc (`ρ = R√u`,
//!   θ ~ U(0,2π)) × uniform z in `[-h/2, h/2]`. 100 % acceptance.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::planet::particle::Particles;
use crate::planet::thermal::internal_energy_of;
use crate::planet::world::{primary_radius, PlanetConfig, PlanetShape};

/// Initial temperature distribution. Picked at run start; the per-particle
/// `internal_energies` buffer is filled accordingly and then evolves under
/// the thermal model (viscous + adiabatic + conduction + radiation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum TemperatureProfile {
    /// All particles start at `core` (interpreted as the uniform value).
    #[default]
    Uniform,
    /// Quadratic radial falloff `T(r) = core + (surface − core)·(r/R)²`.
    /// Smooth — equilibrates gently under conduction.
    HotCore,
    /// Step profile: core temperature inside `R/2`, surface outside.
    /// Sharp — useful for stress-testing conduction.
    Differentiated,
}

/// Overwrite `internal_energies` according to a temperature profile.
/// Takes the *primary radius* of the shape so r/R is well-defined for
/// each generator; non-spherical shapes still work because all profiles
/// only need a monotonic radial coordinate.
pub fn apply_temperature_profile(
    particles: &mut Particles,
    profile: TemperatureProfile,
    core_temp: f32,
    surface_temp: f32,
    r_primary: f32,
) {
    let r_max = r_primary.max(1e-30);
    for (i, pos) in particles.positions.iter().enumerate() {
        let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
        let t = match profile {
            TemperatureProfile::Uniform => core_temp,
            TemperatureProfile::HotCore => {
                let f = (r / r_max).min(1.0);
                core_temp + (surface_temp - core_temp) * f * f
            }
            TemperatureProfile::Differentiated => {
                if r < 0.5 * r_max {
                    core_temp
                } else {
                    surface_temp
                }
            }
        };
        particles.internal_energies[i] = internal_energy_of(t).max(0.0);
    }
}

/// Smoothing-length coefficient `η` in `h = η · (m / ρ_mean)^(1/3)`.
/// Wendland C2 is well-behaved at ~50 neighbours, which for a 3D
/// uniform distribution lands close to `η = 1.3`.
pub const SPH_SMOOTHING_ETA: f32 = 1.3;

/// Generate a uniform-density torus particle distribution.
///
/// Returns a fully populated `Particles` with positions, velocities
/// (rigid rotation about z), accelerations (zero), masses
/// (`M_total / N`), and a uniform initial smoothing length derived
/// from the analytic mean density `ρ_mean = M_total / V_torus`.
pub fn torus_uniform(config: &PlanetConfig) -> Particles {
    let n = config.n_particles;
    let r_major = config.r_major;
    let r_minor = config.r_minor;
    assert!(n > 0, "n_particles must be > 0");
    assert!(r_minor > 0.0 && r_major > r_minor, "r_minor must be < r_major");
    assert!(config.total_mass > 0.0, "total_mass must be > 0");

    let v_torus = 2.0 * std::f32::consts::PI * std::f32::consts::PI * r_major * r_minor * r_minor;
    let rho_mean = config.total_mass / v_torus;
    let mass_per = config.total_mass / n as f32;
    let h_init = SPH_SMOOTHING_ETA * (mass_per / rho_mean).cbrt();

    let omega = config.omega;
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut particles = Particles::with_capacity(n);

    let box_xy = r_major + r_minor;
    let r_min_sq = r_minor * r_minor;
    while particles.len() < n {
        let x: f32 = rng.random_range(-box_xy..box_xy);
        let y: f32 = rng.random_range(-box_xy..box_xy);
        let z: f32 = rng.random_range(-r_minor..r_minor);
        let rxy = (x * x + y * y).sqrt();
        let dxy = rxy - r_major;
        if dxy * dxy + z * z > r_min_sq {
            continue;
        }
        let vx = -omega * y;
        let vy = omega * x;
        particles.push([x, y, z], [vx, vy, 0.0], mass_per);
        let last = particles.len() - 1;
        particles.smoothing_lengths[last] = h_init;
        particles.densities[last] = rho_mean;
    }
    particles
}

/// Set `config.omega` from a fraction of the critical Keplerian rate
/// `Ω_circ = √(G·M / R³)`, where `R` is the shape's primary radius
/// (see `world::primary_radius`). Convenience helper for CLI parsing.
pub fn omega_from_frac(config: &PlanetConfig, frac: f32) -> f32 {
    let r = primary_radius(config);
    let omega_circ = (config.g_const * config.total_mass / r.powi(3)).sqrt();
    frac * omega_circ
}

/// Dispatch to the per-shape generator based on `config.shape`. This
/// is the entry point both binaries use; direct calls to
/// `torus_uniform` / `cube_uniform` / `pancake_uniform` remain for
/// targeted unit tests.
pub fn generate(config: &PlanetConfig) -> Particles {
    match config.shape {
        PlanetShape::Torus => torus_uniform(config),
        PlanetShape::Cube => cube_uniform(config),
        PlanetShape::Pancake => pancake_uniform(config),
    }
}

/// Uniform-density cube centred at the origin with edge length
/// `config.cube_side`. No rejection sampling — every draw is
/// accepted, so this is the fastest of the three generators.
pub fn cube_uniform(config: &PlanetConfig) -> Particles {
    let n = config.n_particles;
    let side = config.cube_side;
    assert!(n > 0, "n_particles must be > 0");
    assert!(side > 0.0, "cube_side must be > 0");
    assert!(config.total_mass > 0.0, "total_mass must be > 0");

    let v_cube = side * side * side;
    let rho_mean = config.total_mass / v_cube;
    let mass_per = config.total_mass / n as f32;
    let h_init = SPH_SMOOTHING_ETA * (mass_per / rho_mean).cbrt();

    let omega = config.omega;
    let half = 0.5 * side;
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut particles = Particles::with_capacity(n);
    for _ in 0..n {
        let x: f32 = rng.random_range(-half..half);
        let y: f32 = rng.random_range(-half..half);
        let z: f32 = rng.random_range(-half..half);
        let vx = -omega * y;
        let vy = omega * x;
        particles.push([x, y, z], [vx, vy, 0.0], mass_per);
        let last = particles.len() - 1;
        particles.smoothing_lengths[last] = h_init;
        particles.densities[last] = rho_mean;
    }
    particles
}

/// Uniform-density flat disc (oblate cylinder) centred at the origin.
/// Radial samples use the standard CDF inverse `ρ = R · √u` so the
/// areal density on the disc is uniform; z is uniform in
/// `[-h/2, h/2]`.
pub fn pancake_uniform(config: &PlanetConfig) -> Particles {
    let n = config.n_particles;
    let r_disc = config.pancake_radius;
    let h_disc = config.pancake_height;
    assert!(n > 0, "n_particles must be > 0");
    assert!(r_disc > 0.0, "pancake_radius must be > 0");
    assert!(h_disc > 0.0, "pancake_height must be > 0");
    assert!(config.total_mass > 0.0, "total_mass must be > 0");

    let v_disc = std::f32::consts::PI * r_disc * r_disc * h_disc;
    let rho_mean = config.total_mass / v_disc;
    let mass_per = config.total_mass / n as f32;
    let h_init = SPH_SMOOTHING_ETA * (mass_per / rho_mean).cbrt();

    let omega = config.omega;
    let half_h = 0.5 * h_disc;
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut particles = Particles::with_capacity(n);
    for _ in 0..n {
        let u1: f32 = rng.random_range(0.0..1.0);
        let u2: f32 = rng.random_range(0.0..1.0);
        let radius_sample = r_disc * u1.sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        let x = radius_sample * theta.cos();
        let y = radius_sample * theta.sin();
        let z: f32 = rng.random_range(-half_h..half_h);
        let vx = -omega * y;
        let vy = omega * x;
        particles.push([x, y, z], [vx, vy, 0.0], mass_per);
        let last = particles.len() - 1;
        particles.smoothing_lengths[last] = h_init;
        particles.densities[last] = rho_mean;
    }
    particles
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(n: usize) -> PlanetConfig {
        PlanetConfig {
            n_particles: n,
            r_major: 1.0,
            r_minor: 0.2,
            total_mass: 1.0,
            seed: 42,
            omega: 0.0,
            ..PlanetConfig::default()
        }
    }

    #[test]
    fn count_and_mass() {
        let cfg = test_config(2_000);
        let p = torus_uniform(&cfg);
        assert_eq!(p.len(), 2_000);
        let m = p.total_mass();
        assert!((m - 1.0).abs() < 1e-5, "total mass = {m}");
    }

    #[test]
    fn particles_inside_torus() {
        let cfg = test_config(1_000);
        let p = torus_uniform(&cfg);
        let r_major = cfg.r_major;
        let r_minor_sq = cfg.r_minor * cfg.r_minor;
        for pos in &p.positions {
            let rxy = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
            let dxy = rxy - r_major;
            let d2 = dxy * dxy + pos[2] * pos[2];
            assert!(d2 <= r_minor_sq + 1e-5, "point outside torus: d2={d2}");
        }
    }

    #[test]
    fn center_of_mass_near_origin() {
        let cfg = test_config(20_000);
        let p = torus_uniform(&cfg);
        let com = p.center_of_mass();
        for c in com {
            assert!(c.abs() < 0.02, "COM = {:?}", com);
        }
    }

    #[test]
    fn smoothing_lengths_uniform() {
        let cfg = test_config(500);
        let p = torus_uniform(&cfg);
        let h0 = p.smoothing_lengths[0];
        assert!(h0 > 0.0);
        for &h in &p.smoothing_lengths {
            assert!((h - h0).abs() < 1e-6);
        }
    }

    #[test]
    fn rigid_rotation_velocity() {
        let mut cfg = test_config(1_000);
        cfg.omega = 0.5;
        let p = torus_uniform(&cfg);
        for (pos, vel) in p.positions.iter().zip(&p.velocities) {
            let vx_expected = -0.5 * pos[1];
            let vy_expected = 0.5 * pos[0];
            assert!((vel[0] - vx_expected).abs() < 1e-6);
            assert!((vel[1] - vy_expected).abs() < 1e-6);
            assert!(vel[2].abs() < 1e-9);
        }
    }

    #[test]
    fn principal_axes_torus_shape() {
        // For an idealised cold torus of major R, minor r, uniform density:
        //   I_zz = M (R² + 3r²/4)
        //   I_xx = I_yy = M (R²/2 + 5r²/8)
        // (standard solid-torus moments). Discrete sample should approach
        // these within statistical fluctuations.
        let cfg = test_config(20_000);
        let p = torus_uniform(&cfg);
        let m = cfg.total_mass as f64;
        let r = cfg.r_major as f64;
        let a = cfg.r_minor as f64;
        let i_zz_expected = m * (r * r + 0.75 * a * a);
        let i_xx_expected = m * (0.5 * r * r + 0.625 * a * a);

        let mut i_xx = 0.0_f64;
        let mut i_yy = 0.0_f64;
        let mut i_zz = 0.0_f64;
        for (pos, &mm) in p.positions.iter().zip(&p.masses) {
            let x = pos[0] as f64;
            let y = pos[1] as f64;
            let z = pos[2] as f64;
            let m64 = mm as f64;
            i_xx += m64 * (y * y + z * z);
            i_yy += m64 * (x * x + z * z);
            i_zz += m64 * (x * x + y * y);
        }
        let tol = 0.02; // 2 % statistical tolerance at N=20k
        assert!(
            (i_zz - i_zz_expected).abs() / i_zz_expected < tol,
            "i_zz = {i_zz}, expected {i_zz_expected}"
        );
        assert!(
            (i_xx - i_xx_expected).abs() / i_xx_expected < tol,
            "i_xx = {i_xx}, expected {i_xx_expected}"
        );
        assert!(
            (i_yy - i_xx_expected).abs() / i_xx_expected < tol,
            "i_yy = {i_yy}, expected {i_xx_expected}"
        );
    }
}
