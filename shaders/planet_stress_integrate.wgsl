// Deviatoric stress integrator (S225/S227). Explicit Euler on the
// persistent stress tensor, then a von Mises radial-return projection:
//
//   S ← S + dt · dS/dt
//   J2 = ½ S:S,   σ_vm = √(3 J2),   Y_i = Y0 · φ_i²
//   if σ_vm > Y_i:  S *= Y_i / σ_vm        (return to the yield surface)
//
// Because Y_i = Y0·φ² → 0 as the particle melts (φ → 0), the projection
// forces S → 0 the instant it crosses the liquidus — this IS the remelt
// mechanism (no separate relaxation constant). `dS/dt` is overwritten by
// the stress-rate pass each tick, so no clear is needed.
//
// `phase_of` is prepended from shaders/planet_phase_common.wgsl.

struct StressIntegrateParams {
    num_particles: u32,
    pad0: u32, pad1: u32, pad2: u32,
    dt: f32,
    y0: f32,
    t_m: f32,
    l: f32,
    g0: f32,
    plastic_cap: f32,
    pad_b0: f32, pad_b1: f32,
}

@group(0) @binding(0) var<uniform> params: StressIntegrateParams;
@group(0) @binding(1) var<storage, read_write> dev_stress: array<f32>;
@group(0) @binding(2) var<storage, read> ds_dt: array<f32>;
@group(0) @binding(3) var<storage, read> internal_energies: array<f32>;
@group(0) @binding(4) var<storage, read> densities: array<f32>;
@group(0) @binding(5) var<storage, read_write> du_plastic: array<f32>;
@group(0) @binding(6) var<storage, read> mat_t_m: array<f32>;

@compute @workgroup_size(64)
fn stress_integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }
    let base = i * 6u;

    let sxx = dev_stress[base + 0u] + params.dt * ds_dt[base + 0u];
    let syy = dev_stress[base + 1u] + params.dt * ds_dt[base + 1u];
    let szz = dev_stress[base + 2u] + params.dt * ds_dt[base + 2u];
    let sxy = dev_stress[base + 3u] + params.dt * ds_dt[base + 3u];
    let sxz = dev_stress[base + 4u] + params.dt * ds_dt[base + 4u];
    let syz = dev_stress[base + 5u] + params.dt * ds_dt[base + 5u];

    // von Mises radial return. J2 = ½ S:S (S symmetric, traceless-ish).
    let j2 = 0.5 * (sxx * sxx + syy * syy + szz * szz) + sxy * sxy + sxz * sxz + syz * syz;
    let sigma_vm = sqrt(3.0 * j2);
    let u = max(internal_energies[i], 1e-6);
    let phi = phase_of(u, mat_t_m[i], params.l).phi;
    let y_i = params.y0 * phi * phi;
    var f: f32 = 1.0;
    if (sigma_vm > y_i) {
        f = y_i / max(sigma_vm, 1e-20);
    }

    dev_stress[base + 0u] = sxx * f;
    dev_stress[base + 1u] = syy * f;
    dev_stress[base + 2u] = szz * f;
    dev_stress[base + 3u] = sxy * f;
    dev_stress[base + 4u] = sxz * f;
    dev_stress[base + 5u] = syz * f;

    // Plastic-work heating (S229): the elastic energy shed by the return,
    // per unit mass, = J2_trial·(1−f²)/(2Gρ). Released as heat. Capped so a
    // hard one-step yield can't blow up the explicit thermal step.
    let g = max(params.g0 * phi * phi, 1e-6);
    let rho = max(densities[i], 1e-30);
    let dq = j2 * (1.0 - f * f) / (2.0 * g * rho);
    du_plastic[i] = min(dq, params.plastic_cap * u);
}
