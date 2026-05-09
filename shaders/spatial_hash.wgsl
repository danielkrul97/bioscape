// GPU counting sort over a fixed bucket grid:
//   count       — per cell, atomicAdd(counts[bucket]).
//   prefix_sum  — exclusive scan of counts → offsets, and reset counts to 0
//                 so scatter can reuse the buffer as a per-bucket write
//                 cursor. Two-phase workgroup-parallel scan; see kernel.
//   scatter     — atomicAdd(counts[bucket]) yields write_pos relative to
//                 offsets[bucket]; sorted_cells[write_pos] = cell_idx.
//
// Bucket grid: 64×32×4 = 8192 buckets covering ±2048 / ±512 / ±128 world
// units at GRID_CELL_SIZE = 64. World half is ±960 / ±540 / ±2, so the
// boundary clamps in `bucket_id_of` are a safety net rather than a hot path.

const GRID_NX: i32 = 64;
const GRID_NY: i32 = 32;
const GRID_NZ: i32 = 4;
const HALF_NX: i32 = 32;
const HALF_NY: i32 = 16;
const HALF_NZ: i32 = 2;
const NUM_BUCKETS: u32 = 8192u; // GRID_NX * GRID_NY * GRID_NZ

// Workgroup-parallel scan tuning: SCAN_WG threads, ELEMS_PER_THREAD per thread.
// Product must equal NUM_BUCKETS so the dispatch covers the whole grid.
const SCAN_WG: u32 = 256u;
const ELEMS_PER_THREAD: u32 = 32u; // SCAN_WG * ELEMS_PER_THREAD = NUM_BUCKETS

struct Params {
    num_cells: u32,
    cell_size: f32,
    world_half_x: f32,
    world_half_y: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> sorted_cells: array<u32>;

// XY uses toroidal wrap (cylinder topology); Z is bounded by the floor/ceiling
// clamp below.
fn bucket_id_of(pos: vec3<f32>) -> u32 {
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

// Two-phase workgroup-parallel exclusive scan over NUM_BUCKETS = 8192.
// Phase 1: each thread serially scans its ELEMS_PER_THREAD slice, resetting
//   counts to 0 along the way, and writes its slice total to `partial[tid]`.
// Phase 2: thread 0 serially scans `partial[0..SCAN_WG]` (cheap at 256
//   elements; replaces the dominant 8192-element serial loop the older
//   single-thread kernel did, leaving the heavy lifting in Phases 1/3).
// Phase 3: each thread reads its block_offset = partial[tid] and writes
//   block_offset + local[k] to offsets[base + k] for k in 0..ELEMS_PER_THREAD.
// Host already dispatches (1, 1, 1) — workgroup_size(SCAN_WG) makes the
// single workgroup multi-threaded; no host change required.
var<workgroup> partial: array<u32, 256>;

@compute @workgroup_size(256)
fn prefix_sum(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let base = tid * ELEMS_PER_THREAD;

    var local: array<u32, 32>;
    var sum: u32 = 0u;
    for (var k: u32 = 0u; k < ELEMS_PER_THREAD; k = k + 1u) {
        let c = atomicLoad(&counts[base + k]);
        local[k] = sum;
        sum = sum + c;
        atomicStore(&counts[base + k], 0u);
    }
    partial[tid] = sum;
    workgroupBarrier();

    if (tid == 0u) {
        var s: u32 = 0u;
        for (var k: u32 = 0u; k < SCAN_WG; k = k + 1u) {
            let p = partial[k];
            partial[k] = s;
            s = s + p;
        }
        offsets[NUM_BUCKETS] = s;
    }
    workgroupBarrier();

    let block_offset = partial[tid];
    for (var k: u32 = 0u; k < ELEMS_PER_THREAD; k = k + 1u) {
        offsets[base + k] = block_offset + local[k];
    }
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
    // counts was reset to 0 by prefix_sum; here it acts as a per-bucket
    // running write cursor. The non-atomic offsets[b] read is safe because
    // the separate dispatch supplies a memory barrier between the two passes.
    let local = atomicAdd(&counts[b], 1u);
    let write_pos = offsets[b] + local;
    sorted_cells[write_pos] = i;
}
