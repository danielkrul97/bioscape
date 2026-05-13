// Sprint 47: GPU reduction over cell SoA → mean position, mean speed^2, mean
// energy. Post-S190 two-pass design:
//   reduce_partial: dispatch N/256 workgroups in parallel. Each workgroup
//     sums its 256-cell slice via the same tree-reduce as before and writes
//     5 partial sums to `partials[wg_id * 5 + 0..5]`. This is the SM-parallel
//     pass — previously the whole reduction ran in one workgroup, leaving
//     95 % of the GPU idle on large N.
//   reduce_final: single workgroup folds `partials[]` (up to several hundred
//     entries) into the final 5 sums in `output[0..5]`. Strided loop keeps
//     it correct for any number of partials.

struct StatsParams {
    num_cells: u32,
    num_partials: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: StatsParams;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> velocities: array<f32>;
@group(0) @binding(3) var<storage, read> energies: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read_write> partials: array<f32>;

const WG_SIZE: u32 = 256u;

var<workgroup> sh_x: array<f32, 256>;
var<workgroup> sh_y: array<f32, 256>;
var<workgroup> sh_z: array<f32, 256>;
var<workgroup> sh_speed_sq: array<f32, 256>;
var<workgroup> sh_energy: array<f32, 256>;

// Tree-reduce the per-thread sh_* slots in place: 256 → 128 → … → 1.
fn workgroup_tree_reduce(tid: u32) {
    var s: u32 = WG_SIZE / 2u;
    loop {
        if (s == 0u) {
            break;
        }
        if (tid < s) {
            sh_x[tid] = sh_x[tid] + sh_x[tid + s];
            sh_y[tid] = sh_y[tid] + sh_y[tid + s];
            sh_z[tid] = sh_z[tid] + sh_z[tid + s];
            sh_speed_sq[tid] = sh_speed_sq[tid] + sh_speed_sq[tid + s];
            sh_energy[tid] = sh_energy[tid] + sh_energy[tid + s];
        }
        workgroupBarrier();
        s = s / 2u;
    }
}

@compute @workgroup_size(256)
fn reduce_partial(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let i = wg.x * WG_SIZE + tid;
    let n = params.num_cells;

    var sum_x: f32 = 0.0;
    var sum_y: f32 = 0.0;
    var sum_z: f32 = 0.0;
    var sum_speed_sq: f32 = 0.0;
    var sum_energy: f32 = 0.0;

    if (i < n) {
        sum_x = positions[i * 3u + 0u];
        sum_y = positions[i * 3u + 1u];
        sum_z = positions[i * 3u + 2u];
        let vx = velocities[i * 3u + 0u];
        let vy = velocities[i * 3u + 1u];
        let vz = velocities[i * 3u + 2u];
        sum_speed_sq = vx * vx + vy * vy + vz * vz;
        sum_energy = energies[i];
    }

    sh_x[tid] = sum_x;
    sh_y[tid] = sum_y;
    sh_z[tid] = sum_z;
    sh_speed_sq[tid] = sum_speed_sq;
    sh_energy[tid] = sum_energy;
    workgroupBarrier();

    workgroup_tree_reduce(tid);

    if (tid == 0u) {
        let base = wg.x * 5u;
        partials[base + 0u] = sh_x[0];
        partials[base + 1u] = sh_y[0];
        partials[base + 2u] = sh_z[0];
        partials[base + 3u] = sh_speed_sq[0];
        partials[base + 4u] = sh_energy[0];
    }
}

@compute @workgroup_size(256)
fn reduce_final(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let np = params.num_partials;

    // Strided gather over the partials array — handles any np ≥ 0.
    var sum_x: f32 = 0.0;
    var sum_y: f32 = 0.0;
    var sum_z: f32 = 0.0;
    var sum_speed_sq: f32 = 0.0;
    var sum_energy: f32 = 0.0;

    var k: u32 = tid;
    loop {
        if (k >= np) {
            break;
        }
        let base = k * 5u;
        sum_x = sum_x + partials[base + 0u];
        sum_y = sum_y + partials[base + 1u];
        sum_z = sum_z + partials[base + 2u];
        sum_speed_sq = sum_speed_sq + partials[base + 3u];
        sum_energy = sum_energy + partials[base + 4u];
        k = k + WG_SIZE;
    }

    sh_x[tid] = sum_x;
    sh_y[tid] = sum_y;
    sh_z[tid] = sum_z;
    sh_speed_sq[tid] = sum_speed_sq;
    sh_energy[tid] = sum_energy;
    workgroupBarrier();

    workgroup_tree_reduce(tid);

    if (tid == 0u) {
        output[0] = sh_x[0];
        output[1] = sh_y[0];
        output[2] = sh_z[0];
        output[3] = sh_speed_sq[0];
        output[4] = sh_energy[0];
    }
}
