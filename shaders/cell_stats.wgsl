// Sprint 47: GPU reduction over cell SoA → mean position, mean speed^2, mean
// energy. Single workgroup of 256 threads, tree reduction přes workgroup-shared
// memory. Pro N≤256·40 = 10240 dostatek; větší N by potřeboval multi-workgroup
// + final reduce, ale 10k je strop tohoto sprintu.

struct StatsParams {
    num_cells: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: StatsParams;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> velocities: array<f32>;
@group(0) @binding(3) var<storage, read> energies: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const WG_SIZE: u32 = 256u;

var<workgroup> sh_x: array<f32, 256>;
var<workgroup> sh_y: array<f32, 256>;
var<workgroup> sh_z: array<f32, 256>;
var<workgroup> sh_speed_sq: array<f32, 256>;
var<workgroup> sh_energy: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var sum_x: f32 = 0.0;
    var sum_y: f32 = 0.0;
    var sum_z: f32 = 0.0;
    var sum_speed_sq: f32 = 0.0;
    var sum_energy: f32 = 0.0;

    let n = params.num_cells;
    var i: u32 = tid;
    loop {
        if (i >= n) {
            break;
        }
        sum_x = sum_x + positions[i * 3u + 0u];
        sum_y = sum_y + positions[i * 3u + 1u];
        sum_z = sum_z + positions[i * 3u + 2u];
        let vx = velocities[i * 3u + 0u];
        let vy = velocities[i * 3u + 1u];
        let vz = velocities[i * 3u + 2u];
        sum_speed_sq = sum_speed_sq + vx * vx + vy * vy + vz * vz;
        sum_energy = sum_energy + energies[i];
        i = i + WG_SIZE;
    }

    sh_x[tid] = sum_x;
    sh_y[tid] = sum_y;
    sh_z[tid] = sum_z;
    sh_speed_sq[tid] = sum_speed_sq;
    sh_energy[tid] = sum_energy;
    workgroupBarrier();

    // Tree reduction: 256 → 128 → ... → 1.
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

    if (tid == 0u) {
        output[0] = sh_x[0];
        output[1] = sh_y[0];
        output[2] = sh_z[0];
        output[3] = sh_speed_sq[0];
        output[4] = sh_energy[0];
    }
}
