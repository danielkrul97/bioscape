// Deterministic reduction feeding the adaptive spatial-hash resize. Computes
// two maxima over all particles:
//   slot 0: max(|x|, |y|, |z|)                  — bounding half-extent
//   slot 1: max(h_i) over BULK particles only   — resolution bound
//
// "Bulk" = ρ_i > rho_surface (the same surface threshold the radiation pass
// uses). The grid is resized to max(1.05·extent, k·max_h_bulk): the first
// term keeps every particle inside the grid; the second keeps cell_size big
// enough that the BULK never has its h clamped harder than its natural value
// (so the bulk neighbour set — and hence the mechanics — is unchanged, only
// the summation order shifts). Low-density surface/ejecta particles (excluded
// here) may have their h clamped as the grid tightens, which is acceptable:
// they are already treated as a free surface, and clamping keeps the 3×3×3
// stencil covering their 2h, so no neighbours are ever missed.
//
// `atomicMax` on the bit pattern of a non-negative IEEE float is exact and
// order-independent (non-negative floats sort identically as u32), so the
// result is bit-deterministic regardless of thread scheduling.

struct ExtentParams {
    num_particles: u32,
    pad0: u32, pad1: u32, pad2: u32,
    rho_surface: f32,
    pad_a0: f32, pad_a1: f32, pad_a2: f32,
}

@group(0) @binding(0) var<uniform> params: ExtentParams;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> smoothing_lengths: array<f32>;
@group(0) @binding(3) var<storage, read> densities: array<f32>;
@group(0) @binding(4) var<storage, read_write> extent: array<atomic<u32>, 2>;

@compute @workgroup_size(64)
fn extent_max(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }
    let x = abs(positions[i].x);
    let y = abs(positions[i].y);
    let z = abs(positions[i].z);
    let coord = max(x, max(y, z));
    atomicMax(&extent[0], bitcast<u32>(coord));
    if (densities[i] > params.rho_surface) {
        atomicMax(&extent[1], bitcast<u32>(smoothing_lengths[i]));
    }
}
