//! Integration tests for the torus planet experiment. Run with
//! `cargo test --test planet_integration`. Kept out of the lib's
//! `#[cfg(test)]` modules so the (pre-existing, unrelated)
//! `src/gpu/tests.rs` compile failure doesn't block validation.

use bioscape::gpu::GpuContext;
use bioscape::planet::{
    gpu::NBodyGpu,
    gravity_cpu::{compute_acceleration, potential_energy},
    init::{cube_uniform, generate, omega_from_frac, pancake_uniform, torus_uniform},
    integrator::leapfrog_step,
    Particles, PlanetConfig, PlanetShape, PlanetWorld,
};
use core::f32::consts::PI;

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

fn cube_config(n: usize, side: f32) -> PlanetConfig {
    PlanetConfig {
        shape: PlanetShape::Cube,
        n_particles: n,
        cube_side: side,
        total_mass: 1.0,
        seed: 7,
        omega: 0.0,
        ..PlanetConfig::default()
    }
}

fn pancake_config(n: usize, radius: f32, height: f32) -> PlanetConfig {
    PlanetConfig {
        shape: PlanetShape::Pancake,
        n_particles: n,
        pancake_radius: radius,
        pancake_height: height,
        total_mass: 1.0,
        seed: 11,
        omega: 0.0,
        ..PlanetConfig::default()
    }
}

#[test]
fn torus_count_and_mass() {
    let cfg = test_config(2_000);
    let p = torus_uniform(&cfg);
    assert_eq!(p.len(), 2_000);
    let m = p.total_mass();
    assert!((m - 1.0).abs() < 1e-5, "total mass = {m}");
}

#[test]
fn torus_particles_inside_volume() {
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
fn torus_principal_moments_match_analytic() {
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
        let (x, y, z, m64) = (pos[0] as f64, pos[1] as f64, pos[2] as f64, mm as f64);
        i_xx += m64 * (y * y + z * z);
        i_yy += m64 * (x * x + z * z);
        i_zz += m64 * (x * x + y * y);
    }
    let tol = 0.02;
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

#[test]
fn torus_rigid_rotation_velocity() {
    let mut cfg = test_config(1_000);
    cfg.omega = 0.5;
    let p = torus_uniform(&cfg);
    for (pos, vel) in p.positions.iter().zip(&p.velocities) {
        assert!((vel[0] + 0.5 * pos[1]).abs() < 1e-6);
        assert!((vel[1] - 0.5 * pos[0]).abs() < 1e-6);
        assert!(vel[2].abs() < 1e-9);
    }
}

#[test]
fn omega_from_frac_helper() {
    let cfg = test_config(1);
    let o = omega_from_frac(&cfg, 0.5);
    let omega_circ = (cfg.g_const * cfg.total_mass / cfg.r_major.powi(3)).sqrt();
    assert!((o - 0.5 * omega_circ).abs() < 1e-6);
}

#[test]
fn cube_count_and_mass() {
    let cfg = cube_config(2_000, 0.924);
    let p = cube_uniform(&cfg);
    assert_eq!(p.len(), 2_000);
    let m = p.total_mass();
    assert!((m - 1.0).abs() < 1e-5, "cube total mass = {m}");
}

#[test]
fn cube_particles_inside_volume() {
    let cfg = cube_config(2_000, 1.0);
    let p = cube_uniform(&cfg);
    let half = 0.5;
    for pos in &p.positions {
        for d in 0..3 {
            assert!(
                pos[d].abs() <= half + 1e-5,
                "cube particle outside [-{half}, {half}]: pos = {:?}",
                pos
            );
        }
    }
}

#[test]
fn cube_principal_moments() {
    // Solid cube of side `s`, uniform density: all three principal
    // moments are equal at `M · s² / 6`. Discrete sample reproduces
    // this within ~2 % statistical scatter at N=20k.
    let side = 1.0_f32;
    let cfg = cube_config(20_000, side);
    let p = cube_uniform(&cfg);
    let m = cfg.total_mass as f64;
    let s2 = (side as f64).powi(2);
    let expected = m * s2 / 6.0;

    let mut i_xx = 0.0_f64;
    let mut i_yy = 0.0_f64;
    let mut i_zz = 0.0_f64;
    for (pos, &mm) in p.positions.iter().zip(&p.masses) {
        let (x, y, z, m64) = (pos[0] as f64, pos[1] as f64, pos[2] as f64, mm as f64);
        i_xx += m64 * (y * y + z * z);
        i_yy += m64 * (x * x + z * z);
        i_zz += m64 * (x * x + y * y);
    }
    let tol = 0.03;
    for (label, val) in [("i_xx", i_xx), ("i_yy", i_yy), ("i_zz", i_zz)] {
        assert!(
            (val - expected).abs() / expected < tol,
            "{label} = {val}, expected {expected}"
        );
    }
}

#[test]
fn pancake_count_and_mass() {
    let cfg = pancake_config(2_000, 1.0, 0.251);
    let p = pancake_uniform(&cfg);
    assert_eq!(p.len(), 2_000);
    let m = p.total_mass();
    assert!((m - 1.0).abs() < 1e-5, "pancake total mass = {m}");
}

#[test]
fn pancake_particles_inside_volume() {
    let radius = 1.0_f32;
    let height = 0.25_f32;
    let cfg = pancake_config(2_000, radius, height);
    let p = pancake_uniform(&cfg);
    let r2_max = radius * radius;
    let half_h = 0.5 * height;
    for pos in &p.positions {
        let r2 = pos[0] * pos[0] + pos[1] * pos[1];
        assert!(
            r2 <= r2_max + 1e-5,
            "pancake particle r² = {r2}, max {r2_max}"
        );
        assert!(
            pos[2].abs() <= half_h + 1e-5,
            "pancake particle |z| = {} > {}",
            pos[2].abs(),
            half_h
        );
    }
}

#[test]
fn pancake_principal_moments() {
    // Solid disc radius R, thickness h, uniform density:
    //   I_zz = M · R² / 2                (spin axis)
    //   I_xx = I_yy = M · (R²/4 + h²/12) (in-plane axes)
    // 20k samples is enough to land within ~3 % of analytic for both
    // moments; the I_zz / I_xx ratio approaches 2 in the thin limit.
    let radius = 1.0_f32;
    let height = 0.2_f32;
    let cfg = pancake_config(20_000, radius, height);
    let p = pancake_uniform(&cfg);
    let m = cfg.total_mass as f64;
    let r2 = (radius as f64).powi(2);
    let h2 = (height as f64).powi(2);
    let i_zz_expected = m * r2 / 2.0;
    let i_xx_expected = m * (r2 / 4.0 + h2 / 12.0);

    let mut i_xx = 0.0_f64;
    let mut i_yy = 0.0_f64;
    let mut i_zz = 0.0_f64;
    for (pos, &mm) in p.positions.iter().zip(&p.masses) {
        let (x, y, z, m64) = (pos[0] as f64, pos[1] as f64, pos[2] as f64, mm as f64);
        i_xx += m64 * (y * y + z * z);
        i_yy += m64 * (x * x + z * z);
        i_zz += m64 * (x * x + y * y);
    }
    let tol = 0.03;
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

#[test]
fn generate_dispatcher_routes_correctly() {
    // For each shape, `generate` must produce the same particle set
    // as the explicit constructor (within determinism — same RNG
    // seed pulled identically).
    for shape in [PlanetShape::Torus, PlanetShape::Cube, PlanetShape::Pancake] {
        let mut cfg = PlanetConfig::default();
        cfg.shape = shape;
        cfg.n_particles = 500;
        cfg.seed = 99;
        let via_dispatch = generate(&cfg);
        let via_direct = match shape {
            PlanetShape::Torus => torus_uniform(&cfg),
            PlanetShape::Cube => cube_uniform(&cfg),
            PlanetShape::Pancake => pancake_uniform(&cfg),
        };
        assert_eq!(via_dispatch.len(), via_direct.len(), "{:?}", shape);
        assert_eq!(via_dispatch.len(), 500);
        for (a, b) in via_dispatch.positions.iter().zip(&via_direct.positions) {
            assert_eq!(a, b, "shape {:?} dispatch mismatch", shape);
        }
    }
}

fn kepler_acc(pos: [f32; 3]) -> [f32; 3] {
    let r2 = pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2];
    let r = r2.sqrt();
    let inv_r3 = 1.0 / (r2 * r);
    [-pos[0] * inv_r3, -pos[1] * inv_r3, -pos[2] * inv_r3]
}

fn kepler_energy(pos: [f32; 3], vel: [f32; 3]) -> f32 {
    let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
    let v2 = vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2];
    0.5 * v2 - 1.0 / r
}

#[test]
fn leapfrog_circular_kepler_energy_conserved() {
    let mut p = Particles::with_capacity(1);
    p.push([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0);
    p.accelerations[0] = kepler_acc(p.positions[0]);
    let e0 = kepler_energy(p.positions[0], p.velocities[0]);
    let period = 2.0 * PI;
    let steps_per_period = 200;
    let n_periods = 10;
    let dt = period / steps_per_period as f32;
    for _ in 0..(steps_per_period * n_periods) {
        leapfrog_step(&mut p, dt, |pp| {
            pp.accelerations[0] = kepler_acc(pp.positions[0]);
        });
    }
    let ef = kepler_energy(p.positions[0], p.velocities[0]);
    let rel = ((ef - e0) / e0).abs();
    assert!(rel < 1e-3, "energy drift {rel}; e0={e0} ef={ef}");
}

#[test]
fn leapfrog_eccentric_kepler_energy_bounded() {
    let mut p = Particles::with_capacity(1);
    let v_apo = (1.0_f32 / 3.0).sqrt();
    p.push([1.5, 0.0, 0.0], [0.0, v_apo, 0.0], 0.0);
    p.accelerations[0] = kepler_acc(p.positions[0]);
    let e0 = kepler_energy(p.positions[0], p.velocities[0]);
    let period = 2.0 * PI;
    let dt = period / 1000.0;
    let n_steps = (5.0 * period / dt) as usize;
    let mut max_drift = 0.0_f32;
    for _ in 0..n_steps {
        leapfrog_step(&mut p, dt, |pp| {
            pp.accelerations[0] = kepler_acc(pp.positions[0]);
        });
        let e = kepler_energy(p.positions[0], p.velocities[0]);
        max_drift = max_drift.max(((e - e0) / e0).abs());
    }
    assert!(max_drift < 5e-3, "eccentric orbit max drift {max_drift}");
}

#[test]
fn cpu_gravity_two_body_symmetric() {
    let mut p = Particles::with_capacity(2);
    p.push([-1.0, 0.0, 0.0], [0.0; 3], 1.0);
    p.push([1.0, 0.0, 0.0], [0.0; 3], 1.0);
    compute_acceleration(&mut p, 1.0, 0.0);
    assert!(p.accelerations[0][0] > 0.0);
    assert!(p.accelerations[1][0] < 0.0);
    assert!((p.accelerations[0][0] + p.accelerations[1][0]).abs() < 1e-6);
    assert!((p.accelerations[0][0] - 0.25).abs() < 1e-6);
}

#[test]
fn cpu_potential_energy_two_body() {
    let mut p = Particles::with_capacity(2);
    p.push([-1.0, 0.0, 0.0], [0.0; 3], 1.0);
    p.push([1.0, 0.0, 0.0], [0.0; 3], 1.0);
    let u = potential_energy(&p, 1.0, 0.0);
    assert!((u + 0.5).abs() < 1e-9, "u = {u}");
}

#[test]
fn gpu_nbody_matches_cpu_reference() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip gpu_nbody_matches_cpu_reference: no GPU adapter ({e})");
            return;
        }
    };
    let cfg = test_config(1024);
    let mut p = torus_uniform(&cfg);
    let g = 1.0_f32;
    let softening = 0.02_f32;

    compute_acceleration(&mut p, g, softening);
    let cpu_acc = p.accelerations.clone();

    let mut nbody = NBodyGpu::with_context(&ctx, 1024).expect("nbody init");
    let gpu_acc = nbody.compute(&p.positions, &p.masses, g, softening);

    assert_eq!(gpu_acc.len(), cpu_acc.len());
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (c, g_) in cpu_acc.iter().zip(&gpu_acc) {
        for d in 0..3 {
            let diff = (c[d] - g_[d]).abs();
            max_abs = max_abs.max(diff);
            let mag = c[d].abs().max(1e-12);
            max_rel = max_rel.max(diff / mag);
        }
    }
    eprintln!("gpu vs cpu: max_abs={max_abs} max_rel={max_rel}");
    assert!(max_abs < 5e-4, "max abs diff {max_abs}");
}

fn energy_two_body(
    positions: &[[f32; 3]],
    velocities: &[[f32; 3]],
    masses: &[f32],
    g: f32,
) -> f64 {
    let mut t = 0.0_f64;
    for (v, &m) in velocities.iter().zip(masses) {
        let v2 = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) as f64;
        t += 0.5 * (m as f64) * v2;
    }
    let dx = (positions[1][0] - positions[0][0]) as f64;
    let dy = (positions[1][1] - positions[0][1]) as f64;
    let dz = (positions[1][2] - positions[0][2]) as f64;
    let r = (dx * dx + dy * dy + dz * dz).sqrt();
    let u = -(g as f64) * (masses[0] as f64) * (masses[1] as f64) / r;
    t + u
}

/// Set up the merged-SPH pipeline for a 2-particle pair test,
/// dispatch once, and return per-particle acceleration. Shared by
/// the three pair tests that follow.
fn run_sph_force_pair(
    ctx: &std::sync::Arc<GpuContext>,
    velocities: [[f32; 3]; 2],
) -> [[f32; 3]; 2] {
    let positions = vec![[-0.1_f32, 0.0, 0.0], [0.1, 0.0, 0.0]];
    let velocities = velocities.to_vec();
    let masses = vec![1.0_f32, 1.0];
    let smoothing = vec![0.3_f32, 0.3];
    let densities_init = vec![1.0_f32, 1.0];
    let world_half = 2.5_f32;
    let n = positions.len();

    let gpu = bioscape::planet::gpu::PlanetGpu::new(std::sync::Arc::clone(ctx), 64).expect("gpu");
    gpu.upload_state(&positions, &velocities, &masses);
    gpu.upload_smoothing_lengths(&smoothing);
    gpu.upload_densities(&densities_init);
    let zero_acc: Vec<f32> = vec![0.0; n * 3];
    ctx.queue
        .write_buffer(gpu.accelerations_buffer(), 0, bytemuck::cast_slice(&zero_acc));

    let hash = bioscape::planet::gpu::SpatialHashGpu::new(
        std::sync::Arc::clone(ctx),
        64,
        world_half,
        gpu.positions_buffer(),
    )
    .expect("hash");
    hash.rebuild(n);

    let sph_force = bioscape::planet::gpu::SphForceGpu::new(
        std::sync::Arc::clone(ctx),
        &gpu,
        &hash,
    )
    .expect("sph_force");
    sph_force.dispatch(n, 1.0, 5.0 / 3.0, 1.0, 2.0);

    let acc = gpu.download_accelerations(n);
    [acc[0], acc[1]]
}

#[test]
fn gpu_sph_force_static_pair_newton_third_law() {
    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip gpu_sph_force_static_pair: no GPU adapter ({e})");
            return;
        }
    };
    // Static pair (v = 0): viscosity branch gated off, only pressure
    // contributes. Newton's 3rd law holds exactly for equal h.
    let acc = run_sph_force_pair(&ctx, [[0.0; 3], [0.0; 3]]);
    eprintln!("static  a_1 = {:?}, a_2 = {:?}", acc[0], acc[1]);
    assert!(acc[0][0] < 0.0, "particle 1 should be pushed to -x, got {}", acc[0][0]);
    assert!(acc[1][0] > 0.0, "particle 2 should be pushed to +x, got {}", acc[1][0]);
    let sum_x = acc[0][0] + acc[1][0];
    assert!(sum_x.abs() < 1e-4, "Newton 3rd law violated: sum_x = {sum_x}");
    assert!(acc[0][1].abs() < 1e-4 && acc[0][2].abs() < 1e-4);
    assert!(acc[1][1].abs() < 1e-4 && acc[1][2].abs() < 1e-4);
}

#[test]
fn gpu_sph_force_approaching_pair_adds_viscosity() {
    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip gpu_sph_force_approaching: no GPU adapter ({e})");
            return;
        }
    };
    // Approaching pair (v_1.x = +0.5, v_2.x = -0.5): viscosity branch
    // active. Pressure pushes apart (a_1.x < 0, a_2.x > 0) and viscosity
    // ADDS to that deceleration → |a.x| larger than the static-pair case.
    let static_acc = run_sph_force_pair(&ctx, [[0.0; 3], [0.0; 3]]);
    let approaching_acc =
        run_sph_force_pair(&ctx, [[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]]);
    eprintln!(
        "static a_1.x = {:.6} vs approaching a_1.x = {:.6}",
        static_acc[0][0], approaching_acc[0][0]
    );
    assert!(approaching_acc[0][0] < 0.0);
    assert!(approaching_acc[1][0] > 0.0);
    // Viscosity strictly increases the magnitude of the apart-push.
    assert!(
        approaching_acc[0][0] < static_acc[0][0],
        "viscosity should make a_1.x more negative: static={}, approach={}",
        static_acc[0][0],
        approaching_acc[0][0]
    );
    assert!(
        approaching_acc[1][0] > static_acc[1][0],
        "viscosity should make a_2.x more positive: static={}, approach={}",
        static_acc[1][0],
        approaching_acc[1][0]
    );
    // Newton's 3rd law still holds for the symmetric setup.
    let sum_x = approaching_acc[0][0] + approaching_acc[1][0];
    assert!(sum_x.abs() < 1e-4, "Newton 3rd law violated: sum_x = {sum_x}");
}

#[test]
fn gpu_sph_force_separating_pair_matches_static() {
    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip gpu_sph_force_separating: no GPU adapter ({e})");
            return;
        }
    };
    // Separating pair: viscosity branch gated off (v_ij·r_ij > 0),
    // so the result must equal the static (v = 0) case bit-for-bit
    // within FP tolerance. Pressure still pushes them apart.
    let static_acc = run_sph_force_pair(&ctx, [[0.0; 3], [0.0; 3]]);
    let separating_acc =
        run_sph_force_pair(&ctx, [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]);
    eprintln!(
        "static a_1.x = {:.6} vs separating a_1.x = {:.6}",
        static_acc[0][0], separating_acc[0][0]
    );
    for d in 0..3 {
        for i in 0..2 {
            let diff = (static_acc[i][d] - separating_acc[i][d]).abs();
            assert!(
                diff < 1e-5,
                "separating pair must give same accel as static (viscosity off): \
                 a_{}[{}] diff = {}",
                i,
                d,
                diff
            );
        }
    }
    // Sanity: pressure still pushes them apart.
    assert!(separating_acc[0][0] < 0.0);
    assert!(separating_acc[1][0] > 0.0);
}

#[test]
fn gpu_sph_density_uniform_grid() {
    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip gpu_sph_density_uniform_grid: no GPU adapter ({e})");
            return;
        }
    };
    let n_per_side = 16usize;
    let dx = 1.0_f32 / n_per_side as f32;
    let mut positions: Vec<[f32; 3]> = Vec::new();
    for iz in 0..n_per_side {
        for iy in 0..n_per_side {
            for ix in 0..n_per_side {
                let x = -0.5 + (ix as f32 + 0.5) * dx;
                let y = -0.5 + (iy as f32 + 0.5) * dx;
                let z = -0.5 + (iz as f32 + 0.5) * dx;
                positions.push([x, y, z]);
            }
        }
    }
    let n = positions.len();
    let m_per = 1.0 / n as f32;
    let masses = vec![m_per; n];
    let velocities = vec![[0.0; 3]; n];
    let rho_true: f32 = 1.0;
    let h_init = 1.3 * (m_per / rho_true).cbrt();
    let smoothing = vec![h_init; n];
    let densities_init = vec![rho_true; n];

    let world_half = 2.5_f32;
    let gpu = bioscape::planet::gpu::PlanetGpu::new(std::sync::Arc::clone(&ctx), n)
        .expect("planet gpu");
    gpu.upload_state(&positions, &velocities, &masses);
    gpu.upload_smoothing_lengths(&smoothing);
    gpu.upload_densities(&densities_init);

    let hash = bioscape::planet::gpu::SpatialHashGpu::new(
        std::sync::Arc::clone(&ctx),
        n,
        world_half,
        gpu.positions_buffer(),
    )
    .expect("hash");
    let density = bioscape::planet::gpu::DensityGpu::new(
        std::sync::Arc::clone(&ctx),
        &gpu,
        &hash,
    )
    .expect("density");

    hash.rebuild(n);
    density.dispatch(n);
    let rhos = gpu.download_densities(n);
    let hs = gpu.download_smoothing_lengths(n);

    let mut max_rel_err = 0.0_f32;
    let mut count_interior = 0;
    let interior_radius = 0.3_f32;
    for (pos, &rho) in positions.iter().zip(&rhos) {
        let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
        if r < interior_radius {
            let rel = (rho - rho_true).abs() / rho_true;
            max_rel_err = max_rel_err.max(rel);
            count_interior += 1;
        }
    }
    eprintln!(
        "interior particles = {count_interior}, max_rel_err = {max_rel_err}, h_after = {:.4}",
        hs[0]
    );
    assert!(count_interior > 50, "too few interior particles");
    assert!(max_rel_err < 0.10, "density estimate too noisy: {max_rel_err}");
    for &h in &hs {
        assert!(h.is_finite() && h > 0.0);
    }
    for &rho in &rhos {
        assert!(rho.is_finite() && rho > 0.0, "rho not positive finite: {rho}");
    }
}

#[test]
fn gpu_spatial_hash_bucket_assignment() {
    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip gpu_spatial_hash_bucket_assignment: no GPU adapter ({e})");
            return;
        }
    };
    let world_half = 2.5_f32;
    let gpu = bioscape::planet::gpu::PlanetGpu::new(std::sync::Arc::clone(&ctx), 1024).expect("planet gpu");
    let positions: Vec<[f32; 3]> = (0..200)
        .map(|i| {
            let t = i as f32 * 0.1;
            [t.cos() * 0.8, t.sin() * 0.8, (t * 0.3).sin() * 0.2]
        })
        .collect();
    let velocities = vec![[0.0; 3]; positions.len()];
    let masses = vec![1.0; positions.len()];
    gpu.upload_state(&positions, &velocities, &masses);
    let hash = bioscape::planet::gpu::SpatialHashGpu::new(
        std::sync::Arc::clone(&ctx),
        1024,
        world_half,
        gpu.positions_buffer(),
    )
    .expect("hash init");
    hash.rebuild(positions.len());
    let offsets = hash.download_offsets();
    let sorted = hash.download_sorted(positions.len());
    let total = offsets[bioscape::planet::gpu::NUM_BUCKETS as usize];
    assert_eq!(
        total as usize,
        positions.len(),
        "total scattered = {total}, expected {}",
        positions.len()
    );

    // Verify each particle is in the bucket its position maps to.
    let mut found_in_bucket = vec![false; positions.len()];
    for b in 0..bioscape::planet::gpu::NUM_BUCKETS {
        let start = offsets[b as usize] as usize;
        let end = offsets[b as usize + 1] as usize;
        for &p_idx in &sorted[start..end] {
            let expected = bioscape::planet::gpu::bucket_id_cpu(positions[p_idx as usize], world_half);
            assert_eq!(b, expected, "particle {p_idx} placed in {b}, expected {expected}");
            found_in_bucket[p_idx as usize] = true;
        }
    }
    assert!(found_in_bucket.iter().all(|&b| b), "some particles missing from hash");

    // Sorted within each bucket (deterministic neighbour order).
    for b in 0..bioscape::planet::gpu::NUM_BUCKETS {
        let start = offsets[b as usize] as usize;
        let end = offsets[b as usize + 1] as usize;
        for w in start + 1..end {
            assert!(sorted[w] > sorted[w - 1], "bucket {b} not sorted");
        }
    }
}

#[test]
fn gpu_two_body_circular_orbit_energy_conserved() {
    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip gpu_two_body: no GPU adapter ({e})");
            return;
        }
    };
    let g = 1.0_f32;
    let softening = 0.0_f32;
    let m = 0.5_f32;
    let a = 1.0_f32;
    // Circular orbit: each particle at distance `a` from COM (separation 2a).
    // v² = G m / (4 a) for two equal masses.
    let v = (g * m / (4.0 * a)).sqrt();
    let dt = 0.01_f32;
    // 2-body period T = 2π · (2a)^(3/2) / sqrt(G · 2m) = 4π√2 ≈ 17.77 at our scale.
    // Run for ~5 periods → ~9000 steps.
    let n_steps = 9_000;

    let positions = vec![[-a, 0.0, 0.0], [a, 0.0, 0.0]];
    let velocities = vec![[0.0, -v, 0.0], [0.0, v, 0.0]];
    let masses = vec![m, m];

    let gpu = bioscape::planet::gpu::PlanetGpu::new(ctx, 64).expect("planet gpu");
    gpu.upload_state(&positions, &velocities, &masses);
    gpu.compute_accelerations(2, g, softening);

    let p0 = gpu.download_positions(2);
    let v0 = gpu.download_velocities(2);
    let e0 = energy_two_body(&p0, &v0, &masses, g);

    let mut max_drift = 0.0_f64;
    let sample_every = 100;
    for step in 0..n_steps {
        gpu.step_leapfrog(2, dt, g, softening);
        if step % sample_every == 0 {
            let p = gpu.download_positions(2);
            let v = gpu.download_velocities(2);
            let e = energy_two_body(&p, &v, &masses, g);
            let drift = ((e - e0) / e0.abs()).abs();
            max_drift = max_drift.max(drift);
        }
    }
    eprintln!("2-body energy drift max = {max_drift} (e0 = {e0})");
    assert!(max_drift < 5e-3, "energy drift {max_drift}");
}

#[test]
fn gpu_nbody_zero_particles_safe() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip gpu_nbody_zero_particles_safe: no GPU adapter ({e})");
            return;
        }
    };
    let mut nbody = NBodyGpu::with_context(&ctx, 64).expect("nbody init");
    let out = nbody.compute(&[], &[], 1.0, 0.01);
    assert!(out.is_empty());
}

#[test]
fn full_sph_gravity_tick_smoke() {
    let ctx = match GpuContext::new() {
        Ok(_) => (),
        Err(e) => {
            eprintln!("skip full_sph_gravity_tick_smoke: no GPU adapter ({e})");
            return;
        }
    };
    let _ = ctx;

    let cfg = PlanetConfig {
        n_particles: 2_000,
        r_major: 1.0,
        r_minor: 0.2,
        total_mass: 1.0,
        omega: 0.5,
        seed: 17,
        dt: 1e-3,
        ..PlanetConfig::default()
    };
    let mut world = PlanetWorld::new(cfg.clone());
    world.particles = torus_uniform(&cfg);
    world.init_gpu_full().expect("gpu init");

    world.download_state();
    let p0 = world.particles.positions[0];
    let diag0 = bioscape::planet::diagnostics::ScalarDiagnostics::compute(&world.particles);

    for _ in 0..50 {
        world.tick_sph();
    }

    world.download_state();
    let p1 = world.particles.positions[0];
    let diag1 = bioscape::planet::diagnostics::ScalarDiagnostics::compute(&world.particles);

    let moved = (p1[0] - p0[0]).abs() + (p1[1] - p0[1]).abs() + (p1[2] - p0[2]).abs();
    assert!(moved > 0.0, "particle 0 did not move after 50 ticks");

    // Finiteness: no NaN/inf anywhere.
    for v in &world.particles.positions {
        for c in v {
            assert!(c.is_finite(), "position not finite: {c}");
        }
    }
    for v in &world.particles.velocities {
        for c in v {
            assert!(c.is_finite(), "velocity not finite: {c}");
        }
    }
    for &rho in &world.particles.densities {
        assert!(rho.is_finite() && rho > 0.0, "rho not positive finite: {rho}");
    }
    for &h in &world.particles.smoothing_lengths {
        assert!(h.is_finite() && h > 0.0, "h not positive finite: {h}");
    }

    // Angular momentum about z should not drift wildly over 50 ticks with
    // good leapfrog. dt × n_steps × Ω = 0.05 × 0.5 = 0.025 t_ff fraction.
    let lz_drift = (diag1.angular_momentum_z - diag0.angular_momentum_z).abs()
        / diag0.angular_momentum_z.abs().max(1e-12);
    assert!(lz_drift < 0.05, "Lz drift {lz_drift} too large");

    // Principal moments computable and sensibly ordered.
    let mom = bioscape::planet::diagnostics::principal_moments(&world.particles);
    assert!(mom[0] >= mom[1] && mom[1] >= mom[2]);
    for m in mom {
        assert!(m.is_finite() && m > 0.0);
    }
}

#[test]
fn cfl_dt_finite_for_torus_init() {
    let cfg = PlanetConfig {
        n_particles: 1_000,
        r_major: 1.0,
        r_minor: 0.2,
        total_mass: 1.0,
        omega: 0.5,
        seed: 3,
        ..PlanetConfig::default()
    };
    let p = torus_uniform(&cfg);
    let dt = bioscape::planet::diagnostics::cfl_dt(&p, cfg.eos_k, cfg.eos_gamma, 0.3);
    assert!(dt.is_finite() && dt > 0.0, "cfl dt = {dt}");
    eprintln!("torus init CFL dt = {dt:.5}");
}

#[test]
fn world_tick_advances_clock_and_state() {
    let mut cfg = test_config(50);
    cfg.dt = 1e-3;
    let mut w = PlanetWorld::new(cfg.clone());
    w.particles = torus_uniform(&cfg);
    w.seed_accelerations();
    let p0 = w.particles.positions[0];
    w.tick();
    assert_eq!(w.tick, 1);
    let p1 = w.particles.positions[0];
    let moved = (p1[0] - p0[0]).abs() + (p1[1] - p0[1]).abs() + (p1[2] - p0[2]).abs();
    assert!(moved > 0.0, "particles should move under self-gravity");
}
