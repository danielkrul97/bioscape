// Sprint 46 + 56: GPU equivalent `lib::SmellField` 3D. Ping-pong storage
// (grid_a, grid_b), per-tick rounded bind group swap určuje "in" vs "out".
//
// Sprint 56: 3D 7-point Jacobi stencil + xy toroidal (cylinder topology),
// z bounded (Neumann zero-flux). Match CPU `SmellField::step` ze Sprintu
// 53/54.
//
// 2 entry points:
//   deposit — per source (pos_xyz, amount): atomic float add do grid_in[idx]
//             přes CAS loop (WGSL nemá nativní atomic<f32>). xy wrap modulo,
//             z out-of-range → no-op.
//   diffuse — per cell (i, j, k): 7-point stencil + multiplicative decay,
//             xy wrap kolem indexů, z fallback na center na hranicích.

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
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sources: array<f32>;
@group(0) @binding(2) var<storage, read_write> grid_in: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> grid_out: array<atomic<u32>>;

@compute @workgroup_size(64)
fn deposit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_sources) {
        return;
    }
    let px = sources[i * 4u + 0u];
    let py = sources[i * 4u + 1u];
    let pz = sources[i * 4u + 2u];
    let amount = sources[i * 4u + 3u];
    // Sprint 56: xy modulo wrap (toroidal); z bounds-check (skip mimo z-volume).
    let nx = i32(params.res_x);
    let ny = i32(params.res_y);
    let nz = i32(params.res_z);
    let zi = i32(floor((pz + params.world_half_z) / params.cell_size_z));
    if (zi < 0 || zi >= nz) {
        return;
    }
    let xi_raw = i32(floor((px + params.world_half_x) / params.cell_size_x));
    let yi_raw = i32(floor((py + params.world_half_y) / params.cell_size_y));
    let xi = ((xi_raw % nx) + nx) % nx;
    let yi = ((yi_raw % ny) + ny) % ny;
    let idx = u32(zi * nx * ny + yi * nx + xi);
    var old_bits: u32 = atomicLoad(&grid_in[idx]);
    loop {
        let old_f = bitcast<f32>(old_bits);
        let new_bits: u32 = bitcast<u32>(old_f + amount);
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
    let center = bitcast<f32>(atomicLoad(&grid_in[idx]));
    // Sprint 56: xy wrap toroidal — left at i=0 čte i=nx-1.
    var i_left: u32; var i_right: u32; var j_up: u32; var j_down: u32;
    if (i == 0u) { i_left = nx - 1u; } else { i_left = i - 1u; }
    if (i + 1u == nx) { i_right = 0u; } else { i_right = i + 1u; }
    if (j == 0u) { j_up = ny - 1u; } else { j_up = j - 1u; }
    if (j + 1u == ny) { j_down = 0u; } else { j_down = j + 1u; }
    let left = bitcast<f32>(atomicLoad(&grid_in[k * plane + j * nx + i_left]));
    let right = bitcast<f32>(atomicLoad(&grid_in[k * plane + j * nx + i_right]));
    let up = bitcast<f32>(atomicLoad(&grid_in[k * plane + j_up * nx + i]));
    let down = bitcast<f32>(atomicLoad(&grid_in[k * plane + j_down * nx + i]));
    // z bounded (Neumann zero-flux): fallback na center.
    var back = center;
    var front = center;
    if (k > 0u) {
        back = bitcast<f32>(atomicLoad(&grid_in[idx - plane]));
    }
    if (k + 1u < nz) {
        front = bitcast<f32>(atomicLoad(&grid_in[idx + plane]));
    }
    let new_val = center + params.diffusion * (left + right + up + down + back + front - 6.0 * center);
    let decayed = new_val * params.decay;
    atomicStore(&grid_out[idx], bitcast<u32>(decayed));
}
