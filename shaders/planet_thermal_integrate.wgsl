// Thermal integrator — applies `dt · du/dt` to the per-particle internal
// energy buffer, clamps to safety bounds, and clears `du_dt` so the next
// tick can overwrite fresh source terms.
//
// S202 form: explicit Euler with no radiation. S205 will add the
// Stefan–Boltzmann surface cooling term inside the same shader, gated
// on the `surface_flag` buffer.
//
//   u_i  ← clamp(u_i + dt · du_dt[i], u_min, u_max)
//   du_dt[i] ← 0

struct ThermalParams {
    num_particles: u32,
    pad0: u32, pad1: u32, pad2: u32,
    dt: f32,
    u_min: f32,
    u_max: f32,
    pad_a0: f32,
}

@group(0) @binding(0) var<uniform> params: ThermalParams;
@group(0) @binding(1) var<storage, read_write> internal_energies: array<f32>;
@group(0) @binding(2) var<storage, read_write> du_dt: array<f32>;

@compute @workgroup_size(64)
fn thermal_integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let u_old = internal_energies[i];
    let du = du_dt[i];
    let u_new = clamp(u_old + params.dt * du, params.u_min, params.u_max);
    internal_energies[i] = u_new;
    du_dt[i] = 0.0;
}
