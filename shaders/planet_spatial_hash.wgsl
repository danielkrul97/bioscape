// GPU counting sort over a 32³ = 32 768 bucket grid covering
// `[-world_half, +world_half]³`. Used by SPH (S209+) to enumerate
// neighbours within 2h.
//
// Phases (host dispatches each as a separate compute pass):
//   1. `count`        — atomicAdd(counts[bucket]).
//   2. `prefix_sum`   — exclusive scan over counts → offsets, resets
//                       counts to 0 so scatter can reuse the slot as a
//                       per-bucket write cursor.
//   3. `scatter`      — counts now serves as a write-cursor; writes
//                       sorted_particles[offsets[b] + counts[b]++] = i.
//   4. `sort_buckets` — ascending insertion sort within each bucket
//                       for deterministic neighbour enumeration order.

const GRID_N: i32 = 32;
const HALF_N: i32 = 16;
const NUM_BUCKETS: u32 = 32768u; // 32³

const SCAN_WG: u32 = 256u;
const ELEMS_PER_THREAD: u32 = 128u; // SCAN_WG * ELEMS_PER_THREAD = NUM_BUCKETS

struct Params {
    num_particles: u32,
    pad0: u32, pad1: u32, pad2: u32,
    world_half: f32,
    cell_size: f32,
    pad_a0: f32, pad_a1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> sorted_particles: array<u32>;

fn bucket_id_of(pos: vec3<f32>) -> u32 {
    let bx = i32(floor(pos.x / params.cell_size)) + HALF_N;
    let by = i32(floor(pos.y / params.cell_size)) + HALF_N;
    let bz = i32(floor(pos.z / params.cell_size)) + HALF_N;
    let bx_c = clamp(bx, 0, GRID_N - 1);
    let by_c = clamp(by, 0, GRID_N - 1);
    let bz_c = clamp(bz, 0, GRID_N - 1);
    return u32(bx_c + by_c * GRID_N + bz_c * GRID_N * GRID_N);
}

@compute @workgroup_size(64)
fn count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }
    let pos = vec3<f32>(
        positions[i * 3u + 0u],
        positions[i * 3u + 1u],
        positions[i * 3u + 2u],
    );
    let b = bucket_id_of(pos);
    atomicAdd(&counts[b], 1u);
}

var<workgroup> partial: array<u32, 256>;

@compute @workgroup_size(256)
fn prefix_sum(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let base = tid * ELEMS_PER_THREAD;

    var local: array<u32, 128>;
    var sum: u32 = 0u;
    for (var k: u32 = 0u; k < ELEMS_PER_THREAD; k = k + 1u) {
        let c = atomicLoad(&counts[base + k]);
        local[k] = sum;
        sum = sum + c;
        atomicStore(&counts[base + k], 0u);
    }
    partial[tid] = sum;
    workgroupBarrier();

    var stride: u32 = 1u;
    loop {
        if (stride >= SCAN_WG) { break; }
        let val = select(0u, partial[tid - stride], tid >= stride);
        workgroupBarrier();
        partial[tid] = partial[tid] + val;
        workgroupBarrier();
        stride = stride * 2u;
    }

    let total = partial[SCAN_WG - 1u];
    let inclusive = partial[tid];
    workgroupBarrier();
    partial[tid] = inclusive - sum;
    if (tid == 0u) {
        offsets[NUM_BUCKETS] = total;
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
    if (i >= params.num_particles) { return; }
    let pos = vec3<f32>(
        positions[i * 3u + 0u],
        positions[i * 3u + 1u],
        positions[i * 3u + 2u],
    );
    let b = bucket_id_of(pos);
    let local = atomicAdd(&counts[b], 1u);
    let write_pos = offsets[b] + local;
    sorted_particles[write_pos] = i;
}

@compute @workgroup_size(64)
fn sort_buckets(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= NUM_BUCKETS) { return; }
    let start = offsets[b];
    let end = offsets[b + 1u];
    for (var k = start + 1u; k < end; k = k + 1u) {
        let v = sorted_particles[k];
        var j = k;
        loop {
            if (j <= start) { break; }
            let prev = sorted_particles[j - 1u];
            if (prev <= v) { break; }
            sorted_particles[j] = prev;
            j = j - 1u;
        }
        sorted_particles[j] = v;
    }
}
