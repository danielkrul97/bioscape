// Per-particle equation-of-state precompute. Writes `pressure[i]` and
// `sound_speed[i]` once per particle so the O(neighbours) force loop in
// planet_sph_force.wgsl reads them instead of recomputing the EoS (two
// `pow` + a phase lookup) for every (i, j) pair. Each particle is visited
// as a neighbour ~(neighbour count) times, so this turns an O(N·k) EoS
// cost into O(N).
//
// The formula is the exact `eos_pc` that used to live inline in
// planet_sph_force.wgsl — same branches, same clamps, same float ops — so
// `pressure`/`sound_speed` are bit-identical to the old per-pair values.
//
// `phase_of` is prepended from planet_phase_common.wgsl (the condensed
// cohesion clamp needs the solid fraction).

const U_MIN: f32 = 1.0e-6;

struct EosParams {
    num_particles: u32,
    pad0: u32, pad1: u32, pad2: u32,
    u_vap: f32,
    eos_gamma: f32,
    c0: f32,
    tait_n: f32,
    p_tens: f32,
    l: f32,
    melt_coh_frac: f32, pad_a1: f32,
}

@group(0) @binding(0) var<uniform> params: EosParams;
@group(0) @binding(1) var<storage, read> densities: array<f32>;
@group(0) @binding(2) var<storage, read> internal_energies: array<f32>;
@group(0) @binding(3) var<storage, read> mat_rho0: array<f32>;
@group(0) @binding(4) var<storage, read> mat_t_m: array<f32>;
@group(0) @binding(5) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(6) var<storage, read_write> sound_speed: array<f32>;

@compute @workgroup_size(64)
fn eos(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let rho = max(densities[i], 1e-30);
    let u = max(internal_energies[i], U_MIN);
    var p_out: f32;
    var c_out: f32;
    if (u >= params.u_vap) {
        let gm1 = params.eos_gamma - 1.0;
        p_out = rho * u * gm1;
        c_out = sqrt(max(params.eos_gamma * gm1 * u, 0.0));
    } else {
        let r0 = max(mat_rho0[i], 1e-30);
        let ratio = max(rho / r0, 1e-6);
        let k0 = r0 * params.c0 * params.c0;
        let p_raw = (k0 / params.tait_n) * (pow(ratio, params.tait_n) - 1.0);
        // Condensed-matter cohesion: the tension floor interpolates from
        // −P_tens (solid, φ=1) down to −P_tens·melt_coh_frac (melt, φ=0), so
        // molten matter stays cohesive and fuses; only gas is cohesionless.
        let phi = phase_of(u, mat_t_m[i], params.l).phi;
        let coh = params.melt_coh_frac + (1.0 - params.melt_coh_frac) * phi;
        p_out = max(p_raw, -params.p_tens * coh);
        c_out = params.c0 * sqrt(params.tait_n) * pow(ratio, 0.5 * (params.tait_n - 1.0));
    }
    pressure[i] = p_out;
    sound_speed[i] = c_out;
}
