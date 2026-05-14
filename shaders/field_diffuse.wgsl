// FieldGpu diffuse kernel: 7-point Jacobi stencil + multiplicative decay.
// Split from deposit so `grid_in` / `grid_out` are plain `array<u32>` —
// diffuse has no cross-thread contention (single writer per voxel), so the
// pre-split `atomicLoad` / `atomicStore` calls were pure overhead.
//
// xy toroidal (cylinder), z bounded (Neumann zero-flux). Wave 4 masked path
// treats `obstacle_mask[idx] != 0` voxels as zero-flux walls.
//
// Deposit lives in `field_deposit.wgsl` and shares the same bind group
// layout — grid_in must stay `atomic` there for the CAS-based atomic float
// add. Memory ordering between deposit and diffuse is enforced by the
// dispatch boundary (separate compute pass on the host).

struct Params {
    res_x: u32,
    res_y: u32,
    res_z: u32,
    num_sources: u32,
    diffusion: f32,
    decay: f32,
    cell_size_x: f32,
    cell_size_y: f32,
    cell_size_z: f32,
    world_half_x: f32,
    world_half_y: f32,
    world_half_z: f32,
    // Wave 4: when non-zero, masked voxels become Neumann zero-flux walls.
    mask_active: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Sources binding is required by the shared bind group layout but unused
// by diffuse (deposit owns it).
@group(0) @binding(1) var<storage, read> sources_unused: array<vec4<f32>>;
// grid_in is read-only by diffuse but the shared layout (with deposit's CAS)
// makes it writable storage — declare `read_write` to match.
@group(0) @binding(2) var<storage, read_write> grid_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> grid_out: array<u32>;
@group(0) @binding(4) var<storage, read> obstacle_mask: array<u32>;
// Fixed-point deposit accumulator written by field_deposit.wgsl. Decoded and
// folded into the f32 grid here — keeps the deposit add associative (host
// zeroes it each step) while diffuse stays the single writer per voxel.
@group(0) @binding(5) var<storage, read_write> deposit_accum: array<u32>;

const DEPOSIT_FIXED_INV: f32 = 1.0 / 1048576.0;

// Effective field value = persistent f32 grid + this step's decoded deposit.
fn grid_val(idx: u32) -> f32 {
    return bitcast<f32>(grid_in[idx]) + f32(deposit_accum[idx]) * DEPOSIT_FIXED_INV;
}

@compute @workgroup_size(4, 4, 4)
fn diffuse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nx = params.res_x;
    let ny = params.res_y;
    let nz = params.res_z;
    let i = gid.x;
    let j = gid.y;
    let k = gid.z;
    if (i >= nx || j >= ny || k >= nz) {
        return;
    }
    let plane = nx * ny;
    let plane_k = k * plane;
    let row_jk = plane_k + j * nx;
    let idx = row_jk + i;
    let masked = params.mask_active != 0u;
    if (masked && obstacle_mask[idx] != 0u) {
        grid_out[idx] = bitcast<u32>(0.0);
        return;
    }
    let center = grid_val(idx);

    // Branchless toroidal wrap on xy.
    let i_left  = select(i - 1u, nx - 1u, i == 0u);
    let i_right = select(i + 1u, 0u,      i + 1u == nx);
    let j_up    = select(j - 1u, ny - 1u, j == 0u);
    let j_down  = select(j + 1u, 0u,      j + 1u == ny);
    // Branchless Neumann zero-flux on z.
    let k_back  = select(k - 1u, 0u,      k == 0u);
    let k_front = select(k + 1u, nz - 1u, k + 1u == nz);

    let idx_left  = row_jk + i_left;
    let idx_right = row_jk + i_right;
    let idx_up    = plane_k + j_up * nx + i;
    let idx_down  = plane_k + j_down * nx + i;
    let idx_back  = k_back * plane + j * nx + i;
    let idx_front = k_front * plane + j * nx + i;
    var left  = grid_val(idx_left);
    var right = grid_val(idx_right);
    var up    = grid_val(idx_up);
    var down  = grid_val(idx_down);
    var back  = grid_val(idx_back);
    var front = grid_val(idx_front);
    if (masked) {
        if (obstacle_mask[idx_left]  != 0u) { left  = center; }
        if (obstacle_mask[idx_right] != 0u) { right = center; }
        if (obstacle_mask[idx_up]    != 0u) { up    = center; }
        if (obstacle_mask[idx_down]  != 0u) { down  = center; }
        if (obstacle_mask[idx_back]  != 0u) { back  = center; }
        if (obstacle_mask[idx_front] != 0u) { front = center; }
    }

    let new_val = center + params.diffusion * (left + right + up + down + back + front - 6.0 * center);
    grid_out[idx] = bitcast<u32>(new_val * params.decay);
}
