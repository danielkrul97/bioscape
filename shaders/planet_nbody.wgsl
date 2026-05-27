// Direct N² gravitational acceleration with Plummer softening:
//
//   a_i = G · Σ_j  m_j (x_j − x_i) / (|x_j − x_i|² + ε²)^(3/2)
//
// Workgroup tiling: each WG of 64 threads cooperatively loads a tile
// of 64 source particles into workgroup-shared memory, then each
// thread accumulates force contributions from that tile against its
// own target particle. Self-interaction is masked by setting the
// effective source mass to 0 when k_global == i (branch-free via
// `select`). With Plummer ε > 0 the geometric factor at zero
// separation is finite anyway, but the mask keeps results
// bit-identical to the CPU reference which uses an explicit skip.

struct NBodyParams {
    num_particles: u32,
    pad_a0: u32, pad_a1: u32, pad_a2: u32,
    g: f32,
    eps2: f32,
    pad_b0: f32, pad_b1: f32,
}

@group(0) @binding(0) var<uniform> params: NBodyParams;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> masses: array<f32>;
@group(0) @binding(3) var<storage, read_write> accelerations: array<f32>;

const TILE: u32 = 128u;
var<workgroup> shared_pm: array<vec4<f32>, TILE>;

@compute @workgroup_size(128)
fn nbody(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = gid.x;
    let l = lid.x;
    let n = params.num_particles;
    let g = params.g;
    let eps2 = params.eps2;
    let i_valid = i < n;

    var xi = vec3<f32>(0.0);
    if (i_valid) {
        xi.x = positions[i * 3u + 0u];
        xi.y = positions[i * 3u + 1u];
        xi.z = positions[i * 3u + 2u];
    }

    var ax = 0.0;
    var ay = 0.0;
    var az = 0.0;

    let n_tiles = (n + TILE - 1u) / TILE;
    var tile: u32 = 0u;
    loop {
        if (tile >= n_tiles) { break; }
        let j_global = tile * TILE + l;
        if (j_global < n) {
            shared_pm[l] = vec4<f32>(
                positions[j_global * 3u + 0u],
                positions[j_global * 3u + 1u],
                positions[j_global * 3u + 2u],
                masses[j_global],
            );
        } else {
            shared_pm[l] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }
        workgroupBarrier();

        // Always loop the full TILE — out-of-range entries have mass=0
        // and contribute nothing. Self-pair (k_global == i) masked by
        // forcing inv_r3 to 0 (not mj to 0) so eps²=0 + self-distance=0
        // doesn't materialise an `inf` that survives the later multiply.
        for (var k: u32 = 0u; k < TILE; k = k + 1u) {
            let k_global = tile * TILE + k;
            let pm = shared_pm[k];
            let dx = pm.x - xi.x;
            let dy = pm.y - xi.y;
            let dz = pm.z - xi.z;
            let r2 = dx * dx + dy * dy + dz * dz + eps2;
            let inv_r3 = select(1.0 / (r2 * sqrt(r2)), 0.0, k_global == i);
            let f = g * pm.w * inv_r3;
            ax = ax + f * dx;
            ay = ay + f * dy;
            az = az + f * dz;
        }
        workgroupBarrier();
        tile = tile + 1u;
    }

    if (i_valid) {
        accelerations[i * 3u + 0u] = ax;
        accelerations[i * 3u + 1u] = ay;
        accelerations[i * 3u + 2u] = az;
    }
}
