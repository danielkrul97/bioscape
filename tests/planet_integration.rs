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

/// Build a cubic lattice, impose a velocity field `vel_fn(pos)`, run the
/// Bonet–Lok gradient correction + Jaumann stress rate once (with S = 0),
/// and return `(positions, ds_dt[6N])`. Shared by the S225 stress oracles.
fn run_stress_rate_on_lattice(
    ctx: &std::sync::Arc<GpuContext>,
    vel_fn: impl Fn([f32; 3]) -> [f32; 3],
) -> (Vec<[f32; 3]>, Vec<f32>) {
    let n_side = 16usize;
    let dx = 1.0_f32 / n_side as f32;
    let mut positions: Vec<[f32; 3]> = Vec::new();
    for iz in 0..n_side {
        for iy in 0..n_side {
            for ix in 0..n_side {
                positions.push([
                    -0.5 + (ix as f32 + 0.5) * dx,
                    -0.5 + (iy as f32 + 0.5) * dx,
                    -0.5 + (iz as f32 + 0.5) * dx,
                ]);
            }
        }
    }
    let n = positions.len();
    let velocities: Vec<[f32; 3]> = positions.iter().map(|&p| vel_fn(p)).collect();
    let masses = vec![1.0_f32; n];
    let densities = vec![1.0_f32; n]; // Bonet–Lok correction is V_j-scale invariant
    let smoothing = vec![dx; n]; // 2h = 2·dx < cell_size ⇒ full 3×3×3 coverage

    let gpu = bioscape::planet::gpu::PlanetGpu::new(std::sync::Arc::clone(ctx), n).unwrap();
    gpu.upload_state(&positions, &velocities, &masses);
    gpu.upload_smoothing_lengths(&smoothing);
    gpu.upload_densities(&densities);
    gpu.upload_internal_energies(&vec![0.01_f32; n]); // cold ⇒ φ = 1 ⇒ G = G0
    gpu.clear_dev_stress(n);
    let hash = bioscape::planet::gpu::SpatialHashGpu::new(
        std::sync::Arc::clone(ctx),
        n,
        2.5,
        gpu.positions_buffer(),
    )
    .unwrap();
    hash.rebuild(n);
    let gc = bioscape::planet::gpu::GradCorrectionGpu::new(std::sync::Arc::clone(ctx), &gpu, &hash).unwrap();
    let sr = bioscape::planet::gpu::StressRateGpu::new(std::sync::Arc::clone(ctx), &gpu, &hash).unwrap();
    gc.dispatch(n);
    sr.dispatch(n);
    (positions, gpu.download_ds_dt(n))
}

/// Bonet–Lok-corrected velocity gradient reproduces a linear shear field
/// exactly: for `v = (γy, 0, 0)` the stress rate is `dS/dt = 2G·dev(ε̇)`,
/// i.e. `dSxy = G·γ` and every other component ≈ 0.
#[test]
fn gpu_stress_rate_linear_shear_matches_analytic() {
    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip gpu_stress_rate_linear_shear: no GPU adapter ({e})");
            return;
        }
    };
    let gamma = 0.1_f32;
    let g0 = bioscape::planet::thermal::SHEAR_MODULUS_G0;
    let (positions, ds) = run_stress_rate_on_lattice(&ctx, |p| [gamma * p[1], 0.0, 0.0]);
    let expect_sxy = g0 * gamma;
    let mut max_sxy_err = 0.0_f32;
    let mut max_other = 0.0_f32;
    let mut count = 0;
    for (i, p) in positions.iter().enumerate() {
        if p[0] * p[0] + p[1] * p[1] + p[2] * p[2] > 0.25 * 0.25 {
            continue; // interior only (boundary kernels are truncated)
        }
        count += 1;
        let b = i * 6;
        max_sxy_err = max_sxy_err.max((ds[b + 3] - expect_sxy).abs());
        for &k in &[0usize, 1, 2, 4, 5] {
            max_other = max_other.max(ds[b + k].abs());
        }
    }
    eprintln!("interior={count} dSxy err={max_sxy_err:.5} (expect {expect_sxy}), max other={max_other:.5}");
    assert!(count > 50, "too few interior particles");
    assert!(max_sxy_err < 0.1 * expect_sxy, "dSxy off by {max_sxy_err}");
    assert!(max_other < 0.1 * expect_sxy, "spurious off-shear rate {max_other}");
}

/// Rigid-rotation objectivity: for `v = ω × x` the strain rate is zero, so
/// the corrected stress rate must vanish (no spurious stress in a rotating
/// solid). This is the property the Bonet–Lok correction exists to ensure.
#[test]
fn gpu_stress_rate_rigid_rotation_objective() {
    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip gpu_stress_rate_rigid_rotation: no GPU adapter ({e})");
            return;
        }
    };
    let omega = 0.5_f32;
    let (positions, ds) = run_stress_rate_on_lattice(&ctx, |p| [-omega * p[1], omega * p[0], 0.0]);
    let mut max_rate = 0.0_f32;
    let mut count = 0;
    for (i, p) in positions.iter().enumerate() {
        if p[0] * p[0] + p[1] * p[1] + p[2] * p[2] > 0.25 * 0.25 {
            continue;
        }
        count += 1;
        for k in 0..6 {
            max_rate = max_rate.max(ds[i * 6 + k].abs());
        }
    }
    eprintln!("rigid-rotation max |dS/dt| (interior) = {max_rate:.6} (G·ω = {})", omega);
    assert!(count > 50);
    // Without Bonet–Lok this would be O(G·ω) ≈ 0.5; with it, ~machine noise.
    assert!(max_rate < 0.02, "spurious stress under rigid rotation: {max_rate}");
}

/// Artificial-stress eigensolve: for σ = S with S_xy = 1 (P = 0), the only
/// tensile principal stress is +1 along (1,1,0)/√2, so the principal-frame
/// R̂ = −ε·(that projector)/ρ² = −ε/2·[[1,1,0],[1,1,0],[0,0,0]]. This checks
/// the Jacobi eigensolve + back-rotation, not just a diagonal shortcut.
#[test]
fn gpu_artificial_stress_principal_frame() {
    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip gpu_artificial_stress_principal_frame: no GPU adapter ({e})");
            return;
        }
    };
    let eps = bioscape::planet::thermal::ARTIFICIAL_STRESS_EPSILON;
    let n = 4usize;
    let gpu = bioscape::planet::gpu::PlanetGpu::new(std::sync::Arc::clone(&ctx), 64).unwrap();
    gpu.upload_densities(&vec![1.0_f32; n]); // ρ = ρ0 ⇒ P = 0 ⇒ σ = S
    gpu.upload_internal_energies(&vec![0.01_f32; n]); // cold ⇒ φ = 1
    let mut s = vec![0.0_f32; n * 6];
    for i in 0..n {
        s[i * 6 + 3] = 1.0; // S_xy
    }
    gpu.upload_dev_stress(&s);
    let asg = bioscape::planet::gpu::ArtificialStressGpu::new(std::sync::Arc::clone(&ctx), &gpu).unwrap();
    asg.dispatch(n, 1.0, 5.0 / 3.0);
    let r = gpu.download_art_stress(n);
    let expect = -eps * 0.5;
    for i in 0..n {
        let b = i * 6;
        assert!((r[b + 0] - expect).abs() < 1e-3, "R̂xx {} != {expect}", r[b + 0]);
        assert!((r[b + 1] - expect).abs() < 1e-3, "R̂yy {} != {expect}", r[b + 1]);
        assert!((r[b + 3] - expect).abs() < 1e-3, "R̂xy {} != {expect}", r[b + 3]);
        assert!(r[b + 2].abs() < 1e-3 && r[b + 4].abs() < 1e-3 && r[b + 5].abs() < 1e-3);
    }
}

/// Cohesion signature across phases at low density (ρ < ρ0): cold solid and
/// molten condensed matter are both under tension (P < 0) and pull together
/// (the melt fuses), while a hot gas has positive pressure and pushes apart.
/// (art_stress is zero here, isolating the EoS tension that supplies cohesion.)
#[test]
fn gpu_cohesion_cold_pair_attracts() {
    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip gpu_cohesion_cold_pair_attracts: no GPU adapter ({e})");
            return;
        }
    };
    let n = 2usize;
    let setup = |u: f32, rho: f32| -> [[f32; 3]; 2] {
        let gpu =
            bioscape::planet::gpu::PlanetGpu::new(std::sync::Arc::clone(&ctx), 64).unwrap();
        gpu.upload_state(&[[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]], &[[0.0; 3]; 2], &[1.0; 2]);
        gpu.upload_smoothing_lengths(&[0.3; 2]);
        gpu.upload_densities(&[rho; 2]);
        gpu.upload_internal_energies(&[u; 2]);
        gpu.clear_dev_stress(n);
        ctx.queue
            .write_buffer(gpu.accelerations_buffer(), 0, bytemuck::cast_slice(&vec![0.0_f32; n * 3]));
        let hash = bioscape::planet::gpu::SpatialHashGpu::new(
            std::sync::Arc::clone(&ctx),
            64,
            2.5,
            gpu.positions_buffer(),
        )
        .unwrap();
        hash.rebuild(n);
        let eos = bioscape::planet::gpu::EosGpu::new(std::sync::Arc::clone(&ctx), &gpu).unwrap();
        eos.dispatch(n, 5.0 / 3.0);
        let f = bioscape::planet::gpu::SphForceGpu::new(std::sync::Arc::clone(&ctx), &gpu, &hash).unwrap();
        f.dispatch(n, 1.0, 5.0 / 3.0, 1.0, 2.0); // rho0 = 1.0
        let a = gpu.download_accelerations(n);
        [a[0], a[1]]
    };
    let cold = setup(0.05, 0.3); // cold solid, stretched (ρ < ρ0) ⇒ tension
    assert!(
        cold[0][0] > 1e-4 && cold[1][0] < -1e-4,
        "cold solid pair should cohere (attract): {cold:?}"
    );
    let molten = setup(1.0, 0.3); // molten condensed (φ=0) ⇒ melt cohesion ⇒ fuses
    assert!(
        molten[0][0] > 1e-4 && molten[1][0] < -1e-4,
        "molten pair should cohere/fuse (melt cohesion): {molten:?}"
    );
    let hot = setup(10.0, 0.3); // hot gas (u ≥ u_vap) ⇒ positive pressure
    assert!(
        hot[0][0] < -1e-4 && hot[1][0] > 1e-4,
        "hot gas pair should repel: {hot:?}"
    );
}

/// von Mises return mapping: an over-yield stress is projected back onto
/// the yield surface (σ_vm = Y0) in a solid, and forced to zero in a
/// liquid (φ = 0 ⇒ Y = 0) — the remelt relaxation mechanism.
#[test]
fn gpu_stress_integrate_von_mises_yield_and_remelt() {
    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip gpu_stress_integrate_von_mises: no GPU adapter ({e})");
            return;
        }
    };
    let y0 = bioscape::planet::thermal::YIELD_STRENGTH_Y0;
    let n = 4usize;
    let gpu = bioscape::planet::gpu::PlanetGpu::new(std::sync::Arc::clone(&ctx), 64).unwrap();
    let si = bioscape::planet::gpu::StressIntegrateGpu::new(std::sync::Arc::clone(&ctx), &gpu).unwrap();
    // Over-yield S_xy = 1.0 (σ_vm = √3 > Y0). ds_dt is zero-initialised, so
    // the integrator applies only the von Mises projection.
    let mut s = vec![0.0_f32; n * 6];
    for i in 0..n {
        s[i * 6 + 3] = 1.0;
    }

    // Solid (cold u ⇒ φ = 1 ⇒ Y = Y0): clamp to the yield surface.
    gpu.upload_dev_stress(&s);
    gpu.upload_internal_energies(&vec![0.01_f32; n]);
    si.dispatch(n, 1e-3);
    let solid = gpu.download_dev_stress(n);
    for i in 0..n {
        let vm = (3.0 * solid[i * 6 + 3] * solid[i * 6 + 3]).sqrt();
        assert!((vm - y0).abs() < 1e-3, "solid von Mises not clamped to Y0: {vm}");
    }

    // Liquid (hot u ⇒ φ = 0 ⇒ Y = 0): stress forced to zero (remelt).
    gpu.upload_dev_stress(&s);
    gpu.upload_internal_energies(&vec![10.0_f32; n]);
    si.dispatch(n, 1e-3);
    let liquid = gpu.download_dev_stress(n);
    for i in 0..n {
        assert!(
            liquid[i * 6 + 3].abs() < 1e-5,
            "molten stress not relaxed: {}",
            liquid[i * 6 + 3]
        );
    }
}

/// The deviatoric stress contraction conserves momentum: an imposed
/// uniform `S_xy` on a pair produces equal-and-opposite transverse forces.
#[test]
fn gpu_sph_force_deviatoric_newton_third_law() {
    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip gpu_sph_force_deviatoric: no GPU adapter ({e})");
            return;
        }
    };
    let n = 2usize;
    let gpu = bioscape::planet::gpu::PlanetGpu::new(std::sync::Arc::clone(&ctx), 64).unwrap();
    gpu.upload_state(&[[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]], &[[0.0; 3]; 2], &[1.0; 2]);
    gpu.upload_smoothing_lengths(&[0.3; 2]);
    gpu.upload_densities(&[1.0; 2]);
    gpu.upload_internal_energies(&[10.0; 2]); // gas branch ⇒ isotropic P symmetric
    // Uniform deviatoric stress S_xy = 0.5 on both, packed [xx,yy,zz,xy,xz,yz].
    gpu.upload_dev_stress(&[0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]);
    ctx.queue
        .write_buffer(gpu.accelerations_buffer(), 0, bytemuck::cast_slice(&vec![0.0_f32; n * 3]));
    let hash = bioscape::planet::gpu::SpatialHashGpu::new(
        std::sync::Arc::clone(&ctx),
        64,
        2.5,
        gpu.positions_buffer(),
    )
    .unwrap();
    hash.rebuild(n);
    let f = bioscape::planet::gpu::SphForceGpu::new(std::sync::Arc::clone(&ctx), &gpu, &hash).unwrap();
    f.dispatch(n, 1.0, 5.0 / 3.0, 1.0, 2.0);
    let a = gpu.download_accelerations(n);
    eprintln!("deviatoric pair: a0={:?} a1={:?}", a[0], a[1]);
    for d in 0..3 {
        assert!(
            (a[0][d] + a[1][d]).abs() < 1e-3,
            "Newton 3rd violated in axis {d}: {} + {}",
            a[0][d],
            a[1][d]
        );
    }
    // S_xy with the pair on the x-axis produces a transverse (y) force.
    assert!(
        a[0][1].abs() > 1e-3 && a[1][1].abs() > 1e-3,
        "deviatoric S_xy should produce a transverse force: {:?} {:?}",
        a[0],
        a[1]
    );
}

/// A cold solid torus run with stress coupling stays finite, develops
/// deviatoric stress (the solid resists deformation), and does not collapse
/// to a point — the elastic CFL holds in the soft (path A) regime.
#[test]
fn gpu_elastic_solid_resists_and_stays_finite() {
    if GpuContext::new().is_err() {
        eprintln!("skip gpu_elastic_solid_resists: no GPU adapter");
        return;
    }
    let cfg = PlanetConfig {
        n_particles: 2_000,
        r_major: 1.0,
        r_minor: 0.2,
        total_mass: 1.0,
        omega: 0.3,
        seed: 9,
        dt: 1e-3,
        ..PlanetConfig::default()
    };
    let mut w = PlanetWorld::new(cfg.clone());
    w.particles = torus_uniform(&cfg); // cold u = 0.01 ⇒ solid (φ = 1)
    w.init_gpu_full().expect("gpu init");
    for _ in 0..100 {
        w.tick_sph();
    }
    w.download_state();
    let n = w.particles.len();
    for v in &w.particles.positions {
        for c in v {
            assert!(c.is_finite(), "position not finite: {c}");
        }
    }
    let s = w.gpu_state.as_ref().unwrap().download_dev_stress(n);
    let max_s = s.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
    assert!(max_s.is_finite() && max_s > 1e-4, "solid developed no stress: max|S|={max_s}");
    let max_r = w
        .particles
        .positions
        .iter()
        .map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt())
        .fold(0.0_f32, f32::max);
    eprintln!("elastic solid: max|S|={max_s:.4}, max_r={max_r:.3}");
    assert!(max_r > 0.4, "solid collapsed to a point: max_r={max_r}");
}

/// Path C (S231): a stiff solid violates the elastic CFL at the fixed
/// outer `dt` with a single step (blows up), but the operator-split inner
/// sub-cycling keeps it stable AND it carries large deviatoric stress
/// (a genuinely rigid block, not slush).
#[test]
fn gpu_stiff_solid_needs_substepping() {
    if GpuContext::new().is_err() {
        eprintln!("skip gpu_stiff_solid_needs_substepping: no GPU adapter");
        return;
    }
    let run = |n_sub: u32, dt: f32, stiff: bool| -> (bool, f32, f32, f32) {
        let (g0, c0, y0) = if stiff { (200.0, 8.0, 100.0) } else {
            (
                bioscape::planet::thermal::SHEAR_MODULUS_G0,
                bioscape::planet::thermal::TAIT_REF_SOUND_SPEED_C0,
                bioscape::planet::thermal::YIELD_STRENGTH_Y0,
            )
        };
        let cfg = PlanetConfig {
            n_particles: 1_500,
            r_major: 1.0,
            r_minor: 0.2,
            total_mass: 1.0,
            omega: 0.3,
            seed: 31,
            dt,
            shear_modulus: g0,
            tait_c0: c0,
            tait_exponent: if stiff { 4.0 } else { bioscape::planet::thermal::TAIT_EXPONENT_N },
            yield_strength: y0,
            n_substeps: n_sub,
            ..PlanetConfig::default()
        };
        let mut w = PlanetWorld::new(cfg.clone());
        w.particles = torus_uniform(&cfg);
        w.init_gpu_full().expect("gpu init");
        for _ in 0..60 {
            w.tick_sph();
        }
        w.download_state();
        let n = w.particles.len();
        let mut finite = true;
        let mut max_r = 0.0_f32;
        let mut max_v = 0.0_f32;
        for (p, v) in w.particles.positions.iter().zip(&w.particles.velocities) {
            for d in 0..3 {
                if !p[d].is_finite() || !v[d].is_finite() {
                    finite = false;
                }
            }
            max_r = max_r.max((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt());
            max_v = max_v.max((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
        }
        let max_s = w
            .gpu_state
            .as_ref()
            .unwrap()
            .download_dev_stress(n)
            .iter()
            .fold(0.0_f32, |m, &x| m.max(x.abs()));
        (finite, max_r, max_v, max_s)
    };

    // Rigidity (at dt where both are stable): the sub-cycled stiff solid
    // carries far more deviatoric stress than the soft baseline.
    let (fin_stiff, r_stiff, _, s_stiff) = run(10, 1e-3, true);
    let (_, _, _, s_soft) = run(1, 1e-3, false);
    assert!(fin_stiff && r_stiff < 5.0, "stiff sub-cycled run unstable: finite={fin_stiff} r={r_stiff}");
    assert!(
        s_stiff > 5.0 && s_stiff > 5.0 * s_soft.max(1e-6),
        "stiff solid should be far more rigid than soft: max|S| stiff={s_stiff} soft={s_soft}"
    );

    // Necessity + sufficiency: at an outer dt that violates the bulk acoustic
    // CFL (c0·dt/h ≈ 1.6 > 1), a single step blows up but sub-cycling — which
    // advances the stiff physics on dt/n_sub while gravity stays outer —
    // stays stable and bounded.
    let (fin_sub, r_sub, v_sub, _) = run(20, 1e-2, true);
    let (fin1, r1, v1, _) = run(1, 1e-2, true);
    eprintln!(
        "rigidity: max|S| stiff={s_stiff:.2} vs soft={s_soft:.3}; \
         dt=1e-2 n_sub=20 finite={fin_sub} r={r_sub:.2} v={v_sub:.2}; n_sub=1 finite={fin1} r={r1:.2} v={v1:.2}"
    );
    assert!(fin_sub && r_sub < 20.0, "sub-cycling should stabilise the CFL-violating dt: finite={fin_sub} r={r_sub}");
    assert!(
        !fin1 || r1 > 100.0 || v1 > 500.0,
        "single step at the CFL-violating dt should be unstable: finite={fin1} r={r1} v={v1}"
    );
}

/// The sub-stepped integration path is deterministic (byte-identical reruns).
#[test]
fn gpu_substepped_tick_deterministic() {
    if GpuContext::new().is_err() {
        eprintln!("skip gpu_substepped_tick_deterministic: no GPU adapter");
        return;
    }
    let cfg = PlanetConfig {
        n_particles: 1_200,
        r_major: 1.0,
        r_minor: 0.2,
        total_mass: 1.0,
        omega: 0.3,
        seed: 32,
        dt: 1e-3,
        shear_modulus: 50.0,
        tait_c0: 3.0,
        tait_exponent: 4.0,
        yield_strength: 25.0,
        n_substeps: 4,
        ..PlanetConfig::default()
    };
    let run = || {
        let mut w = PlanetWorld::new(cfg.clone());
        w.particles = torus_uniform(&cfg);
        w.init_gpu_full().expect("gpu init");
        for _ in 0..20 {
            w.tick_sph();
        }
        w.download_state();
        (w.particles.positions.clone(), w.particles.velocities.clone())
    };
    let (p1, v1) = run();
    let (p2, v2) = run();
    assert_eq!(p1, p2, "sub-stepped positions not byte-identical");
    assert_eq!(v1, v2, "sub-stepped velocities not byte-identical");
}

/// Block detection: a cold solid body is one large connected block; once
/// fully molten there are no solid blocks. The two-sided melt+block signature.
#[test]
fn gpu_block_detection_solid_vs_molten() {
    if GpuContext::new().is_err() {
        eprintln!("skip gpu_block_detection: no GPU adapter");
        return;
    }
    use bioscape::planet::diagnostics::count_solid_blocks;
    use bioscape::planet::thermal::{LATENT_HEAT_FUSION_L as L, MELT_TEMPERATURE_T_M as TM};
    let cfg = PlanetConfig {
        n_particles: 1_500,
        r_major: 1.0,
        r_minor: 0.2,
        total_mass: 1.0,
        omega: 0.3,
        seed: 21,
        dt: 1e-3,
        ..PlanetConfig::default()
    };
    let mut w = PlanetWorld::new(cfg.clone());
    w.particles = torus_uniform(&cfg);
    w.init_gpu_full().expect("gpu init");
    let n = w.particles.len();
    for _ in 0..30 {
        w.tick_sph();
    }
    w.download_state();
    let (n_blocks, largest) = count_solid_blocks(&w.particles, 1.5);
    eprintln!("cold solid: {n_blocks} block(s), largest = {largest}/{n}");
    assert!(n_blocks >= 1, "cold solid should form at least one block");
    assert!(
        largest as f32 > 0.5 * n as f32,
        "most of the solid should be one connected block: {largest}/{n}"
    );

    // Melt everything ⇒ no solid blocks.
    w.gpu_state
        .as_ref()
        .unwrap()
        .upload_internal_energies(&vec![TM + L + 0.5; n]);
    for _ in 0..5 {
        w.tick_sph();
    }
    w.download_state();
    let (_, largest_molten) = count_solid_blocks(&w.particles, 1.5);
    eprintln!("molten: largest block = {largest_molten}/{n}");
    assert!(
        (largest_molten as f32) < 0.05 * n as f32,
        "molten body should have no solid block: {largest_molten}/{n}"
    );
}

/// Full remelt cycle: a cold solid develops deviatoric stress; once heated
/// past the liquidus (φ → 0) the von Mises yield (Y = Y0·φ² → 0) dissolves
/// that stress within a few ticks and the body reverts to a fluid.
#[test]
fn gpu_remelt_dissolves_stress() {
    if GpuContext::new().is_err() {
        eprintln!("skip gpu_remelt_dissolves_stress: no GPU adapter");
        return;
    }
    use bioscape::planet::thermal::{LATENT_HEAT_FUSION_L as L, MELT_TEMPERATURE_T_M as TM};
    let cfg = PlanetConfig {
        n_particles: 2_000,
        r_major: 1.0,
        r_minor: 0.2,
        total_mass: 1.0,
        omega: 0.3,
        seed: 13,
        dt: 1e-3,
        ..PlanetConfig::default()
    };
    let mut w = PlanetWorld::new(cfg.clone());
    w.particles = torus_uniform(&cfg); // cold ⇒ solid
    w.init_gpu_full().expect("gpu init");
    let n = w.particles.len();
    for _ in 0..50 {
        w.tick_sph();
    }
    let max_s_solid = w
        .gpu_state
        .as_ref()
        .unwrap()
        .download_dev_stress(n)
        .iter()
        .fold(0.0_f32, |m, &x| m.max(x.abs()));
    assert!(max_s_solid > 1e-3, "cold solid developed no stress: {max_s_solid}");

    // Melt everything: u well above the liquidus ⇒ φ = 0.
    let hot = vec![TM + L + 0.5; n];
    w.gpu_state.as_ref().unwrap().upload_internal_energies(&hot);
    for _ in 0..5 {
        w.tick_sph();
    }
    w.download_state();
    let max_s_liquid = w
        .gpu_state
        .as_ref()
        .unwrap()
        .download_dev_stress(n)
        .iter()
        .fold(0.0_f32, |m, &x| m.max(x.abs()));
    let mean_phi =
        w.particles.phase_fracs.iter().sum::<f32>() / n as f32;
    eprintln!("remelt: max|S| solid={max_s_solid:.4} → liquid={max_s_liquid:.6}, mean φ={mean_phi:.4}");
    assert!(mean_phi < 0.05, "particles did not melt: mean φ={mean_phi}");
    assert!(
        max_s_liquid < 0.1 * max_s_solid,
        "remelt did not dissolve the stress: {max_s_liquid} vs {max_s_solid}"
    );
}

/// Multi-material (S232): at a single uniform energy, a refractory core
/// (high T_m) stays solid while the surrounding volatile crust (default,
/// lower T_m) melts — heterogeneous melting driven by per-particle T_m.
#[test]
fn gpu_multimaterial_heterogeneous_melting() {
    if GpuContext::new().is_err() {
        eprintln!("skip gpu_multimaterial_heterogeneous_melting: no GPU adapter");
        return;
    }
    let cfg = PlanetConfig {
        shape: PlanetShape::Cube,
        n_particles: 3_000,
        cube_side: 1.0,
        total_mass: 1.0,
        omega: 0.0,
        seed: 41,
        dt: 1e-3,
        ..PlanetConfig::default()
    };
    let mut w = PlanetWorld::new(cfg.clone());
    w.particles = cube_uniform(&cfg);
    let n = w.particles.len();
    // Energy between the crust liquidus (0.45) and the core solidus (0.50):
    // crust melts, refractory core stays solid.
    for e in &mut w.particles.internal_energies {
        *e = 0.48;
    }
    // Refractory, denser core (T_m = 0.5, ρ0 = 2) within r < 0.3.
    bioscape::planet::init::assign_core_material(&mut w.particles, 0.3, 2.0, 0.5);
    w.init_gpu_full().expect("gpu init");
    w.download_state();

    let (mut core_phi, mut core_n, mut crust_phi, mut crust_n) = (0.0_f32, 0, 0.0_f32, 0);
    for i in 0..n {
        let p = w.particles.positions[i];
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if r < 0.25 {
            core_phi += w.particles.phase_fracs[i];
            core_n += 1;
        } else if r > 0.4 {
            crust_phi += w.particles.phase_fracs[i];
            crust_n += 1;
        }
    }
    let core_mean = core_phi / core_n.max(1) as f32;
    let crust_mean = crust_phi / crust_n.max(1) as f32;
    eprintln!("heterogeneous melt: refractory core φ={core_mean:.3}, volatile crust φ={crust_mean:.3}");
    assert!(core_n > 20 && crust_n > 20, "too few particles per region");
    assert!(core_mean > 0.9, "refractory core should stay solid: φ={core_mean}");
    assert!(crust_mean < 0.1, "volatile crust should melt: φ={crust_mean}");
}

/// Per-particle ρ0 controls the EoS: at the SAME actual density, a low-ρ0
/// material is compressed (P > 0, repels) while a high-ρ0 material is below
/// its reference (P < 0, attracts) — the driver of gravitational differentiation.
#[test]
fn gpu_multimaterial_rho0_controls_eos() {
    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip gpu_multimaterial_rho0_controls_eos: no GPU adapter ({e})");
            return;
        }
    };
    let pair = |rho0_mat: f32| -> [[f32; 3]; 2] {
        let n = 2usize;
        let gpu = bioscape::planet::gpu::PlanetGpu::new(std::sync::Arc::clone(&ctx), 64).unwrap();
        gpu.upload_state(&[[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]], &[[0.0; 3]; 2], &[1.0; 2]);
        gpu.upload_smoothing_lengths(&[0.3; 2]);
        gpu.upload_densities(&[1.5; 2]); // same actual density for both materials
        gpu.upload_internal_energies(&[0.05; 2]); // cold ⇒ condensed branch
        gpu.upload_materials(&[rho0_mat; 2], &[bioscape::planet::thermal::MELT_TEMPERATURE_T_M; 2]);
        gpu.clear_dev_stress(n);
        ctx.queue
            .write_buffer(gpu.accelerations_buffer(), 0, bytemuck::cast_slice(&vec![0.0_f32; n * 3]));
        let hash = bioscape::planet::gpu::SpatialHashGpu::new(
            std::sync::Arc::clone(&ctx),
            64,
            2.5,
            gpu.positions_buffer(),
        )
        .unwrap();
        hash.rebuild(n);
        let eos = bioscape::planet::gpu::EosGpu::new(std::sync::Arc::clone(&ctx), &gpu).unwrap();
        eos.dispatch(n, 5.0 / 3.0);
        let f = bioscape::planet::gpu::SphForceGpu::new(std::sync::Arc::clone(&ctx), &gpu, &hash).unwrap();
        f.dispatch(n, 1.0, 5.0 / 3.0, 1.0, 2.0);
        let a = gpu.download_accelerations(n);
        [a[0], a[1]]
    };
    let low = pair(1.0); // ρ=1.5 > ρ0 ⇒ compressed ⇒ repel
    assert!(
        low[0][0] < -1e-4 && low[1][0] > 1e-4,
        "low-ρ0 material at ρ>ρ0 should repel: {low:?}"
    );
    let high = pair(3.0); // ρ=1.5 < ρ0 ⇒ under reference ⇒ tension ⇒ attract
    assert!(
        high[0][0] > 1e-4 && high[1][0] < -1e-4,
        "high-ρ0 material at ρ<ρ0 should cohere/attract (drives sinking): {high:?}"
    );
}

/// A two-material body integrates deterministically (byte-identical reruns).
#[test]
fn gpu_multimaterial_deterministic() {
    if GpuContext::new().is_err() {
        eprintln!("skip gpu_multimaterial_deterministic: no GPU adapter");
        return;
    }
    let cfg = PlanetConfig {
        shape: PlanetShape::Cube,
        n_particles: 1_500,
        cube_side: 1.0,
        total_mass: 1.0,
        omega: 0.0,
        seed: 42,
        dt: 1e-3,
        ..PlanetConfig::default()
    };
    let run = || {
        let mut w = PlanetWorld::new(cfg.clone());
        w.particles = cube_uniform(&cfg);
        bioscape::planet::init::assign_core_material(&mut w.particles, 0.3, 2.0, 0.5);
        w.init_gpu_full().expect("gpu init");
        for _ in 0..25 {
            w.tick_sph();
        }
        w.download_state();
        (w.particles.positions.clone(), w.particles.velocities.clone())
    };
    let (p1, v1) = run();
    let (p2, v2) = run();
    assert_eq!(p1, p2, "multi-material positions not byte-identical");
    assert_eq!(v1, v2, "multi-material velocities not byte-identical");
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
    // u above the vaporisation energy ⇒ ideal-gas branch, so these pair
    // tests exercise the EoS reduction (gas branch == pre-S224 ideal gas).
    gpu.upload_internal_energies(&vec![10.0_f32; n]);
    gpu.upload_materials(&vec![1.0_f32; n], &vec![bioscape::planet::thermal::MELT_TEMPERATURE_T_M; n]);
    gpu.clear_dev_stress(n); // S = 0 ⇒ deviatoric term off (reduction)
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

    // EoS precompute (pressure/sound speed) must run before sph_force, which
    // now reads those buffers instead of recomputing the EoS per pair.
    let eos = bioscape::planet::gpu::EosGpu::new(std::sync::Arc::clone(ctx), &gpu).expect("eos");
    eos.dispatch(n, 5.0 / 3.0);

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

/// Condensed Tait branch: compression (`ρ > ρ0`) pushes apart, pressure
/// vanishes at `ρ = ρ0`, a stretched solid (`ρ < ρ0`, φ=1) is under tension
/// and coheres (attracts), while a stretched liquid (φ=0) is clamped to
/// `P ≥ 0` (no tension → cavitates).
#[test]
fn gpu_condensed_eos_compression_and_tension() {
    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip gpu_condensed_eos: no GPU adapter ({e})");
            return;
        }
    };
    let n = 2usize;
    let setup = |u: f32, rho: f32| -> [[f32; 3]; 2] {
        let gpu =
            bioscape::planet::gpu::PlanetGpu::new(std::sync::Arc::clone(&ctx), 64).unwrap();
        gpu.upload_state(&[[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]], &[[0.0; 3]; 2], &[1.0; 2]);
        gpu.upload_smoothing_lengths(&[0.3; 2]);
        gpu.upload_densities(&[rho; 2]);
        gpu.upload_internal_energies(&[u; 2]);
        gpu.clear_dev_stress(n);
        ctx.queue
            .write_buffer(gpu.accelerations_buffer(), 0, bytemuck::cast_slice(&vec![0.0_f32; n * 3]));
        let hash = bioscape::planet::gpu::SpatialHashGpu::new(
            std::sync::Arc::clone(&ctx),
            64,
            2.5,
            gpu.positions_buffer(),
        )
        .unwrap();
        hash.rebuild(n);
        let eos = bioscape::planet::gpu::EosGpu::new(std::sync::Arc::clone(&ctx), &gpu).unwrap();
        eos.dispatch(n, 5.0 / 3.0);
        let f = bioscape::planet::gpu::SphForceGpu::new(
            std::sync::Arc::clone(&ctx),
            &gpu,
            &hash,
        )
        .unwrap();
        f.dispatch(n, 1.0, 5.0 / 3.0, 1.0, 2.0); // rho0 = 1.0
        let a = gpu.download_accelerations(n);
        [a[0], a[1]]
    };
    let compressed = setup(0.05, 2.0);
    assert!(
        compressed[0][0] < -1e-4 && compressed[1][0] > 1e-4,
        "compression (ρ>ρ0) should repel: {compressed:?}"
    );
    let at_ref = setup(0.05, 1.0);
    assert!(at_ref[0][0].abs() < 1e-4, "P should vanish at ρ0: {at_ref:?}");
    // Cold solid stretched ⇒ tension ⇒ cohesion (attracts).
    let solid_stretched = setup(0.05, 0.5);
    assert!(
        solid_stretched[0][0] > 1e-4 && solid_stretched[1][0] < -1e-4,
        "stretched solid should cohere under tension: {solid_stretched:?}"
    );
    // Melt cohesion: a stretched liquid (φ=0) now coheres (attracts) instead
    // of cavitating at P=0 — molten matter fuses — but more weakly than the
    // solid at the same density (floor drops to MELT_COHESION_FRAC·P_tens).
    let liquid_stretched = setup(1.0, 0.5);
    assert!(
        liquid_stretched[0][0] > 1e-4 && liquid_stretched[1][0] < -1e-4,
        "molten matter should cohere/fuse under tension: {liquid_stretched:?}"
    );
    assert!(
        liquid_stretched[0][0] < solid_stretched[0][0],
        "melt cohesion must be weaker than solid: melt={} solid={}",
        liquid_stretched[0][0],
        solid_stretched[0][0]
    );
}

/// The generalised CFL must see the condensed sound speed (~c0√n) for a
/// cold solid, not the near-zero ideal-gas `√(γ(γ−1)u)` that the old
/// estimate reported.
#[test]
fn cfl_dt_uses_condensed_sound_speed() {
    let cfg = PlanetConfig {
        n_particles: 1_000,
        r_major: 1.0,
        r_minor: 0.2,
        total_mass: 1.0,
        omega: 0.0,
        seed: 3,
        ..PlanetConfig::default()
    };
    let mut p = torus_uniform(&cfg);
    for u in &mut p.internal_energies {
        *u = 0.05; // cold solid, well below the melt point
    }
    let rho0 = bioscape::planet::world::rho_mean_init(&cfg);
    let c_courant = 0.3_f32;
    let dt = bioscape::planet::diagnostics::cfl_dt(&p, cfg.eos_gamma, rho0, c_courant);
    assert!(dt.is_finite() && dt > 0.0, "cfl dt = {dt}");
    // Particles are at rest, ρ ≈ ρ0, so c_eff = c_courant·h/dt should land
    // near c0·√n ≈ 1.73 — proving the condensed branch drives the CFL.
    let h = p.smoothing_lengths[0];
    let c_eff = c_courant * h / dt;
    eprintln!("cold-solid CFL dt = {dt:.5}, implied c_eff = {c_eff:.3}");
    assert!(
        c_eff > 1.0 && c_eff < 3.0,
        "implied sound speed {c_eff} not in the condensed range (~1.73)"
    );
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
    let rho0 = bioscape::planet::world::rho_mean_init(&cfg);
    let dt = bioscape::planet::diagnostics::cfl_dt(&p, cfg.eos_gamma, rho0, 0.3);
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

// --- S223a: determinism harness + phase map ------------------------------

/// Hard determinism gate. Two independent runs of the full SPH+gravity
/// tick loop from the same seed must produce bit-identical state. This
/// guards the byte-identity invariant every melting/fusion sprint relies
/// on (S223–S230 reduce to a no-op when their physics is disabled, so any
/// non-determinism they introduce shows up here first).
#[test]
fn gpu_planet_tick_deterministic_rerun() {
    if GpuContext::new().is_err() {
        eprintln!("skip gpu_planet_tick_deterministic_rerun: no GPU adapter");
        return;
    }
    let cfg = PlanetConfig {
        n_particles: 1_500,
        r_major: 1.0,
        r_minor: 0.2,
        total_mass: 1.0,
        omega: 0.5,
        seed: 23,
        dt: 1e-3,
        ..PlanetConfig::default()
    };
    let run = || {
        let mut w = PlanetWorld::new(cfg.clone());
        w.particles = torus_uniform(&cfg);
        w.init_gpu_full().expect("gpu init");
        for _ in 0..30 {
            w.tick_sph();
        }
        w.download_state();
        (
            w.particles.positions.clone(),
            w.particles.velocities.clone(),
            w.particles.internal_energies.clone(),
        )
    };
    let (p1, v1, u1) = run();
    let (p2, v2, u2) = run();
    assert_eq!(p1, p2, "positions not byte-identical across reruns");
    assert_eq!(v1, v2, "velocities not byte-identical across reruns");
    assert_eq!(u1, u2, "internal energies not byte-identical across reruns");
}

/// Enthalpy phase map: temperature plateaus at `T_m` across the melt band
/// while the solid fraction sweeps 1 → 0, and the map is continuous.
#[test]
fn phase_map_plateau_and_monotone() {
    use bioscape::planet::thermal::{
        phase_of, LATENT_HEAT_FUSION_L as L, MELT_TEMPERATURE_T_M as TM,
    };
    // Cold solid: T == u, fully solid.
    let cold = phase_of(0.5 * TM);
    assert!((cold.t - 0.5 * TM).abs() < 1e-6 && (cold.phi - 1.0).abs() < 1e-6);

    // Across the band: T pinned at T_m, phi strictly decreasing 1 → 0.
    let u_sol = TM;
    let u_liq = TM + L;
    let mut last_phi = 1.0_f32;
    for k in 0..=10 {
        let u = u_sol + (k as f32 / 10.0) * L;
        let p = phase_of(u);
        assert!((p.t - TM).abs() < 1e-4, "T not on plateau at u={u}: {}", p.t);
        assert!(p.phi <= last_phi + 1e-6, "phi not monotone");
        last_phi = p.phi;
    }
    assert!((phase_of(u_sol).phi - 1.0).abs() < 1e-4);
    assert!(phase_of(u_liq - 1e-4).phi < 1e-3);

    // Hot liquid: phi == 0, T rises again above the plateau.
    let hot = phase_of(u_liq + 0.2);
    assert!(hot.phi.abs() < 1e-6 && (hot.t - (TM + 0.2)).abs() < 1e-4);

    // Continuity of T at both knots.
    let eps = 1e-4;
    assert!((phase_of(u_sol - eps).t - phase_of(u_sol + eps).t).abs() < 1e-3);
    assert!((phase_of(u_liq - eps).t - phase_of(u_liq + eps).t).abs() < 1e-3);
}

/// Single-source guarantee: the WGSL `phase_of` (concatenated from
/// planet_phase_common.wgsl) must agree with the Rust mirror across a
/// grid spanning solid → mush → liquid.
#[test]
fn gpu_phase_map_matches_cpu() {
    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip gpu_phase_map_matches_cpu: no GPU adapter ({e})");
            return;
        }
    };
    use bioscape::planet::thermal::{
        phase_of, LATENT_HEAT_FUSION_L as L, MELT_TEMPERATURE_T_M as TM,
    };
    let n = 64usize;
    let u_max = TM + L + 0.3;
    let us: Vec<f32> = (0..n)
        .map(|i| u_max * (i as f32) / ((n - 1) as f32))
        .collect();
    let gpu = bioscape::planet::gpu::PlanetGpu::new(std::sync::Arc::clone(&ctx), n).unwrap();
    gpu.upload_internal_energies(&us);
    let phase = bioscape::planet::gpu::PhaseGpu::new(std::sync::Arc::clone(&ctx), &gpu).unwrap();
    phase.dispatch(n);
    let phi_gpu = gpu.download_phase_fracs(n);
    for (i, &u) in us.iter().enumerate() {
        let expect = phase_of(u).phi;
        assert!(
            (phi_gpu[i] - expect).abs() < 1e-5,
            "phase_of mismatch at u={u}: gpu={} cpu={}",
            phi_gpu[i],
            expect
        );
    }
}

/// The latent plateau must be respected by conduction: two particles both
/// inside the melt band sit at T_m, so the heat flux between them is zero.
/// A control pair straddling the solidus conducts.
#[test]
fn gpu_conduction_plateau_no_intra_band_flux() {
    let ctx = match GpuContext::new() {
        Ok(c) => std::sync::Arc::new(c),
        Err(e) => {
            eprintln!("skip gpu_conduction_plateau: no GPU adapter ({e})");
            return;
        }
    };
    use bioscape::planet::thermal::{LATENT_HEAT_FUSION_L as L, MELT_TEMPERATURE_T_M as TM};
    let n = 2usize;
    let setup = |u: [f32; 2]| -> Vec<f32> {
        let gpu =
            bioscape::planet::gpu::PlanetGpu::new(std::sync::Arc::clone(&ctx), 64).unwrap();
        gpu.upload_state(&[[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]], &[[0.0; 3]; 2], &[1.0; 2]);
        gpu.upload_smoothing_lengths(&[0.3; 2]);
        gpu.upload_densities(&[1.0; 2]);
        gpu.upload_internal_energies(&u);
        gpu.clear_du_dt(n);
        let hash = bioscape::planet::gpu::SpatialHashGpu::new(
            std::sync::Arc::clone(&ctx),
            64,
            2.5,
            gpu.positions_buffer(),
        )
        .unwrap();
        hash.rebuild(n);
        let cond = bioscape::planet::gpu::ThermalConductionGpu::new(
            std::sync::Arc::clone(&ctx),
            &gpu,
            &hash,
        )
        .unwrap();
        cond.dispatch(n);
        gpu.download_du_dt(n)
    };
    // Both mushy (different u, same T = T_m): no flux.
    let band = setup([TM + 0.25 * L, TM + 0.75 * L]);
    assert!(
        band[0].abs() < 1e-5 && band[1].abs() < 1e-5,
        "intra-band conduction should vanish, got {band:?}"
    );
    // Across the solidus (cold solid ↔ mush at T_m): real flux.
    let grad = setup([0.5 * TM, TM + 0.5 * L]);
    assert!(
        grad[0].abs() > 1e-6 || grad[1].abs() > 1e-6,
        "expected conduction across the gradient, got {grad:?}"
    );
}

/// End-to-end: the GPU `phase_frac` buffer downloaded into `Particles`
/// matches `phase_of(u)` elementwise and spans the full [0,1] range when
/// the initial energy field crosses the melt band.
#[test]
fn gpu_phase_frac_spans_and_matches_u() {
    let ctx = match GpuContext::new() {
        Ok(_) => (),
        Err(e) => {
            eprintln!("skip gpu_phase_frac_spans_and_matches_u: no GPU adapter ({e})");
            return;
        }
    };
    let _ = ctx;
    use bioscape::planet::thermal::{
        phase_of, LATENT_HEAT_FUSION_L as L, MELT_TEMPERATURE_T_M as TM,
    };
    let cfg = PlanetConfig {
        n_particles: 1_000,
        r_major: 1.0,
        r_minor: 0.2,
        total_mass: 1.0,
        omega: 0.0,
        seed: 5,
        dt: 1e-3,
        ..PlanetConfig::default()
    };
    let mut w = PlanetWorld::new(cfg.clone());
    w.particles = torus_uniform(&cfg);
    let n = w.particles.len();
    let u_hi = TM + L + 0.3;
    for i in 0..n {
        w.particles.internal_energies[i] = u_hi * (i as f32) / ((n - 1) as f32);
    }
    w.init_gpu_full().expect("gpu init");
    w.download_state();

    let mut min_phi = 1.0_f32;
    let mut max_phi = 0.0_f32;
    for i in 0..n {
        let u = w.particles.internal_energies[i];
        let expect = phase_of(u).phi;
        assert!(
            (w.particles.phase_fracs[i] - expect).abs() < 1e-4,
            "phi vs phase_of(u) mismatch at {i}: {} vs {}",
            w.particles.phase_fracs[i],
            expect
        );
        assert!(w.particles.phase_fracs[i] >= -1e-6 && w.particles.phase_fracs[i] <= 1.0 + 1e-6);
        min_phi = min_phi.min(w.particles.phase_fracs[i]);
        max_phi = max_phi.max(w.particles.phase_fracs[i]);
    }
    assert!(min_phi < 0.01, "expected fully-molten particles, min_phi={min_phi}");
    assert!(max_phi > 0.99, "expected fully-solid particles, max_phi={max_phi}");
}
