// Sprint 45: GPU counting sort pro spatial hash. 3 passes:
//   count       — atomicAdd na counts[bucket] per cell
//   prefix_sum  — exclusive prefix sum counts → offsets, current resetuje
//                 counts na 0 aby scatter mohl použít stejný buffer jako
//                 per-bucket write counter
//   scatter     — atomicAdd(&counts[bucket]) → write_pos relativní k
//                 offsets[bucket]; sorted_cells[write_pos] = cell_idx
//
// Bucket grid je fixed 64×32×4 = 8192 buckets, krytí ±2048 / ±512 / ±128
// world units při GRID_CELL_SIZE = 64. Cells mimo bounds jsou clampované do
// boundary bucketů (Cell::step world_half clamp guarantuje ±960 / ±540 / ±2,
// takže boundary clampu nikdy nevyužijeme — jen safety net).
//
// Lookup (Sprint 46+ shadery): pro pos compute bucket(pos), iteruj 3³
// neighborů, pro každý čti offsets[b]..offsets[b+1] range v sorted_cells.

const GRID_NX: i32 = 64;
const GRID_NY: i32 = 32;
const GRID_NZ: i32 = 4;
const HALF_NX: i32 = 32;
const HALF_NY: i32 = 16;
const HALF_NZ: i32 = 2;
const NUM_BUCKETS: u32 = 8192u; // = GRID_NX * GRID_NY * GRID_NZ

struct Params {
    num_cells: u32,
    cell_size: f32,
    // Sprint 55: world_half xy pro toroidal wrap. z bounded (cylinder topology).
    world_half_x: f32,
    world_half_y: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> sorted_cells: array<u32>;

fn bucket_id_of(pos: vec3<f32>) -> u32 {
    // Sprint 55: wrap xy do [-half, half) než spočítáme bucket — toroidal
    // cylinder topology. z stále bounded (clamp).
    let wx = 2.0 * params.world_half_x;
    let wy = 2.0 * params.world_half_y;
    let pos_wx = pos.x - floor((pos.x + params.world_half_x) / wx) * wx;
    let pos_wy = pos.y - floor((pos.y + params.world_half_y) / wy) * wy;
    let bx = i32(floor(pos_wx / params.cell_size)) + HALF_NX;
    let by = i32(floor(pos_wy / params.cell_size)) + HALF_NY;
    let bz = i32(floor(pos.z / params.cell_size)) + HALF_NZ;
    let bx_c = clamp(bx, 0, GRID_NX - 1);
    let by_c = clamp(by, 0, GRID_NY - 1);
    let bz_c = clamp(bz, 0, GRID_NZ - 1);
    return u32(bx_c + by_c * GRID_NX + bz_c * GRID_NX * GRID_NY);
}

@compute @workgroup_size(64)
fn count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }
    let pos = vec3<f32>(
        positions[i * 3u + 0u],
        positions[i * 3u + 1u],
        positions[i * 3u + 2u],
    );
    let b = bucket_id_of(pos);
    atomicAdd(&counts[b], 1u);
}

// Single-thread serial exclusive scan. NUM_BUCKETS = 8192 → ~8 µs na
// dnešním dGPU (memory-bound). Pro Sprint 45 stačí; Sprint 47+ může
// nahradit hierarchical Blelloch scan, pokud benchmark ukáže potřebu.
@compute @workgroup_size(1)
fn prefix_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }
    var sum: u32 = 0u;
    for (var b: u32 = 0u; b < NUM_BUCKETS; b = b + 1u) {
        offsets[b] = sum;
        let c = atomicLoad(&counts[b]);
        sum = sum + c;
        atomicStore(&counts[b], 0u);
    }
    offsets[NUM_BUCKETS] = sum;
}

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }
    let pos = vec3<f32>(
        positions[i * 3u + 0u],
        positions[i * 3u + 1u],
        positions[i * 3u + 2u],
    );
    let b = bucket_id_of(pos);
    // counts byl resetnut na 0 v prefix_sum; teď ho používáme jako per-bucket
    // running write index. Ne-atomic offset[b] read je safe — prefix_sum už
    // doběhl (separate dispatch barrier).
    let local = atomicAdd(&counts[b], 1u);
    let write_pos = offsets[b] + local;
    sorted_cells[write_pos] = i;
}
