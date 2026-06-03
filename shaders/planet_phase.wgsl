// Phase pass — map per-particle internal energy `u` to the solid
// fraction `phi` and store it in the `phase_frac` buffer. Pure local
// map, no neighbours; embarrassingly parallel and deterministic.
//
// Runs at the end of a tick (after the internal energy is updated) so the
// `phase_frac` buffer is the canonical CPU-readable solid fraction for
// diagnostics / rendering / connected-component block labelling. GPU
// mechanics passes recompute `phase_of` inline from this-tick `u` (no
// one-tick lag at the melt front), so this buffer is a cache, not a
// dependency of the force path.
//
// `phase_of` is prepended from shaders/planet_phase_common.wgsl.

struct PhaseParams {
    num_particles: u32,
    pad0: u32, pad1: u32, pad2: u32,
    t_m: f32,
    l: f32,
    pad_a0: f32, pad_a1: f32,
}

@group(0) @binding(0) var<uniform> params: PhaseParams;
@group(0) @binding(1) var<storage, read> internal_energies: array<f32>;
@group(0) @binding(2) var<storage, read_write> phase_frac: array<f32>;
@group(0) @binding(3) var<storage, read> mat_t_m: array<f32>;

@compute @workgroup_size(64)
fn phase(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }
    let ps = phase_of(internal_energies[i], mat_t_m[i], params.l);
    phase_frac[i] = ps.phi;
}
