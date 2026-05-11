// Sprint 46 + 56: GPU equivalent of lib::SmellField (3D). Ping-pong storage
// (grid_a, grid_b); per-tick bind group swap selects "in" vs "out".
//
// Two entry points:
//   deposit — per source: atomic float add to grid_in[idx] via CAS loop
//             (WGSL has no native atomic<f32>). xy modulo wrap (toroidal),
//             z out-of-range → no-op.
//   diffuse — per cell (i,j,k): 7-point Jacobi stencil + multiplicative decay.
//             xy toroidal (cylinder), z bounded (Neumann zero-flux).
//             Matches CPU SmellField::step (Sprint 53/54).
//
// Bind layout note: grids are declared atomic because deposit needs CAS.
// Diffuse has no contention (single writer per cell) and would benefit from
// non-atomic loads/stores, but that requires splitting into two modules with
// distinct bind layouts. Single-file form here pays the atomic overhead in
// diffuse as the trade-off.

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
    // Wave 4: when non-zero, the diffuse stencil treats voxels with
    // `mask[idx] != 0u` as Neumann zero-flux walls — neighbour reads of
    // masked cells substitute the center value (no flux through wall) and
    // masked cells themselves write 0. Mirror of CPU `SmellField::step_masked`.
    mask_active: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sources: array<vec4<f32>>;          // xyz + amount
@group(0) @binding(2) var<storage, read_write> grid_in: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> grid_out: array<atomic<u32>>;
// Wave 4: per-voxel obstacle mask (binding 4). Same layout as `grid_in`.
// Read-only; populated by `FieldGpu::upload_obstacle_mask`. Zeroed when no
// maze is active so even with `mask_active = 0` the binding is well-defined.
@group(0) @binding(4) var<storage, read> obstacle_mask: array<u32>;

@compute @workgroup_size(64)
fn deposit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_sources) {
        return;
    }
    let s = sources[i];
    let pos = s.xyz;
    let amount = s.w;

    let nx = i32(params.res_x);
    let ny = i32(params.res_y);
    let nz = i32(params.res_z);
    let zi = i32(floor((pos.z + params.world_half_z) / params.cell_size_z));
    if (zi < 0 || zi >= nz) {
        return;
    }
    let xi_raw = i32(floor((pos.x + params.world_half_x) / params.cell_size_x));
    let yi_raw = i32(floor((pos.y + params.world_half_y) / params.cell_size_y));
    // Normalize for negative modulo (WGSL i32 % follows truncated division).
    let xi = ((xi_raw % nx) + nx) % nx;
    let yi = ((yi_raw % ny) + ny) % ny;
    let idx = u32(zi * nx * ny + yi * nx + xi);

    // Atomic float add via CAS — no native atomic<f32> in WGSL.
    var old_bits: u32 = atomicLoad(&grid_in[idx]);
    loop {
        let new_bits: u32 = bitcast<u32>(bitcast<f32>(old_bits) + amount);
        let res = atomicCompareExchangeWeak(&grid_in[idx], old_bits, new_bits);
        if (res.exchanged) {
            break;
        }
        old_bits = res.old_value;
    }
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
    let idx = k * plane + j * nx + i;
    let masked = params.mask_active != 0u;
    // Wave 4: masked center = wall — write 0 and skip the stencil entirely.
    if (masked && obstacle_mask[idx] != 0u) {
        atomicStore(&grid_out[idx], bitcast<u32>(0.0));
        return;
    }
    let center = bitcast<f32>(atomicLoad(&grid_in[idx]));

    // Branchless toroidal wrap on xy.
    let i_left  = select(i - 1u, nx - 1u, i == 0u);
    let i_right = select(i + 1u, 0u,      i + 1u == nx);
    let j_up    = select(j - 1u, ny - 1u, j == 0u);
    let j_down  = select(j + 1u, 0u,      j + 1u == ny);

    // Branchless Neumann zero-flux on z: at the boundary the ghost-cell index
    // collapses to the boundary plane itself, so the loaded value equals
    // `center`, matching the original `back = center` / `front = center`
    // fallback. (E.g. k == 0 → k_back == 0 → load is grid_in[idx].)
    let k_back  = select(k - 1u, 0u,      k == 0u);
    let k_front = select(k + 1u, nz - 1u, k + 1u == nz);

    // Wave 4: when masked, neighbours that are walls fall back to `center`
    // (Neumann zero-flux through the wall). `read_neighbor` keeps the
    // hot-path branchless when no maze is active.
    let idx_left  = k * plane + j * nx + i_left;
    let idx_right = k * plane + j * nx + i_right;
    let idx_up    = k * plane + j_up * nx + i;
    let idx_down  = k * plane + j_down * nx + i;
    let idx_back  = k_back * plane + j * nx + i;
    let idx_front = k_front * plane + j * nx + i;
    let raw_left  = bitcast<f32>(atomicLoad(&grid_in[idx_left]));
    let raw_right = bitcast<f32>(atomicLoad(&grid_in[idx_right]));
    let raw_up    = bitcast<f32>(atomicLoad(&grid_in[idx_up]));
    let raw_down  = bitcast<f32>(atomicLoad(&grid_in[idx_down]));
    let raw_back  = bitcast<f32>(atomicLoad(&grid_in[idx_back]));
    let raw_front = bitcast<f32>(atomicLoad(&grid_in[idx_front]));
    var left  = raw_left;
    var right = raw_right;
    var up    = raw_up;
    var down  = raw_down;
    var back  = raw_back;
    var front = raw_front;
    if (masked) {
        if (obstacle_mask[idx_left]  != 0u) { left  = center; }
        if (obstacle_mask[idx_right] != 0u) { right = center; }
        if (obstacle_mask[idx_up]    != 0u) { up    = center; }
        if (obstacle_mask[idx_down]  != 0u) { down  = center; }
        if (obstacle_mask[idx_back]  != 0u) { back  = center; }
        if (obstacle_mask[idx_front] != 0u) { front = center; }
    }

    let new_val = center + params.diffusion * (left + right + up + down + back + front - 6.0 * center);
    atomicStore(&grid_out[idx], bitcast<u32>(new_val * params.decay));
}