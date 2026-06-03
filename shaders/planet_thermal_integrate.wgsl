// Thermal integrator — applies `dt · du/dt` to the per-particle internal
// energy buffer, adds Stefan–Boltzmann radiation on surface particles,
// clamps to safety bounds, and clears `du_dt` for the next tick.
//
//   du_rad_i  = −ε σ (T_i⁴ − T_space⁴) · surface_flag_i
//   du_rad_i  = max(du_rad_i, −u_i · max_rad_frac / dt)   (safety clamp)
//   u_i      ← clamp(u_i + dt · (du_dt[i] + du_rad_i), u_min, u_max)
//   du_dt[i] ← 0
//
// Surface flag is computed inline: a particle counts as "surface" when
// its density falls below `rho_surface = surface_frac · rho_mean_init`.

struct ThermalParams {
    num_particles: u32,
    pad0: u32, pad1: u32, pad2: u32,
    dt: f32,
    u_min: f32,
    u_max: f32,
    inv_cv: f32,
    sigma: f32,
    emissivity: f32,
    t_space: f32,
    rho_surface: f32,
    max_rad_frac: f32,
    t_m: f32,
    l: f32,
    pad_a2: f32,
}

@group(0) @binding(0) var<uniform> params: ThermalParams;
@group(0) @binding(1) var<storage, read_write> internal_energies: array<f32>;
@group(0) @binding(2) var<storage, read_write> du_dt: array<f32>;
@group(0) @binding(3) var<storage, read> densities: array<f32>;
@group(0) @binding(4) var<storage, read> du_plastic: array<f32>;
@group(0) @binding(5) var<storage, read> mat_t_m: array<f32>;

@compute @workgroup_size(64)
fn thermal_integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let u_old = max(internal_energies[i], params.u_min);
    let du_sph = du_dt[i];

    // Surface flag — sparse particles radiate to space.
    let rho_i = densities[i];
    let is_surface = select(0.0, 1.0, rho_i < params.rho_surface);

    // Stefan–Boltzmann radiation. T⁴ scaling means hot anomalies could
    // drain `u` in one substep; clamp the loss to `max_rad_frac` of u_i
    // per tick so the explicit Euler stays stable. Sensible temperature
    // via the enthalpy map so a melting particle radiates at T_m (was
    // `u_old * inv_cv` pre-S223).
    let t_i = phase_of(u_old, mat_t_m[i], params.l).t;
    let t_i4 = t_i * t_i * t_i * t_i;
    let t_sp = params.t_space;
    let t_sp4 = t_sp * t_sp * t_sp * t_sp;
    var du_rad = -params.emissivity * params.sigma * (t_i4 - t_sp4) * is_surface;
    let rad_floor = -u_old * params.max_rad_frac / max(params.dt, 1e-30);
    du_rad = max(du_rad, rad_floor);

    // Plastic-work heating (S229) is a per-step energy increment, not a
    // rate, so it is added directly (not × dt).
    let du_total = du_sph + du_rad;
    let u_new = clamp(u_old + params.dt * du_total + du_plastic[i], params.u_min, params.u_max);
    internal_energies[i] = u_new;
    du_dt[i] = 0.0;
}
