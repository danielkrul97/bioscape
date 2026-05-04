// Sprint 46: GPU equivalent `lib::SmellField`. Ping-pong storage buffery
// (grid_a, grid_b) drží stav mezi ticky; bind group rounded swap určuje
// per-tick which is "in" vs "out".
//
// 2 entry points:
//   deposit — per source (pos_xy, amount): atomic float add do grid_in[idx]
//             přes CAS loop (WGSL nemá nativní atomic<f32>).
//   diffuse — per cell (i, j): 5-point Jacobi stencil + multiplicative decay,
//             read grid_in, write grid_out.

struct Params {
    resolution: u32,
    num_sources: u32,
    diffusion: f32,
    decay: f32,
    cell_size_x: f32,
    cell_size_y: f32,
    world_half_x: f32,
    world_half_y: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sources: array<f32>;
@group(0) @binding(2) var<storage, read_write> grid_in: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> grid_out: array<atomic<u32>>;

// CAS loop pro f32 atomic add. WGSL nepovoluje storage pointer jako
// function param (Naga validation), takže CAS je inlined v deposit.

@compute @workgroup_size(64)
fn deposit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_sources) {
        return;
    }
    let px = sources[i * 3u + 0u];
    let py = sources[i * 3u + 1u];
    let amount = sources[i * 3u + 2u];
    let xi = i32(floor((px + params.world_half_x) / params.cell_size_x));
    let yi = i32(floor((py + params.world_half_y) / params.cell_size_y));
    let n = i32(params.resolution);
    if (xi < 0 || xi >= n || yi < 0 || yi >= n) {
        return;
    }
    let idx = u32(yi * n + xi);
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

@compute @workgroup_size(8, 8)
fn diffuse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.resolution;
    let i = gid.x;
    let j = gid.y;
    if (i >= n || j >= n) {
        return;
    }
    let idx = j * n + i;
    let center = bitcast<f32>(atomicLoad(&grid_in[idx]));
    var left = center;
    var right = center;
    var up = center;
    var down = center;
    if (i > 0u) {
        left = bitcast<f32>(atomicLoad(&grid_in[idx - 1u]));
    }
    if (i + 1u < n) {
        right = bitcast<f32>(atomicLoad(&grid_in[idx + 1u]));
    }
    if (j > 0u) {
        up = bitcast<f32>(atomicLoad(&grid_in[idx - n]));
    }
    if (j + 1u < n) {
        down = bitcast<f32>(atomicLoad(&grid_in[idx + n]));
    }
    let new_val = center + params.diffusion * (left + right + up + down - 4.0 * center);
    let decayed = new_val * params.decay;
    atomicStore(&grid_out[idx], bitcast<u32>(decayed));
}
