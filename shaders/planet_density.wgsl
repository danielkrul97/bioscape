// SPH density estimator with Wendland C2 kernel (3D).
//
//   W(r, h) = (21 / (16π h³)) · (1 − q/2)⁴ · (1 + 2q),    q = r/h ≤ 2
//             0,                                          q > 2
//
// Per particle i: scan the 27 spatial-hash buckets around `pos[i]`,
// sum `m_j · W(|x_i − x_j|, h_i)` over all neighbours (including
// self), then update `h_i ← η · (m_i / ρ_i)^(1/3)` for the next
// step.

const PI: f32 = 3.14159265358979;
const GRID_N: i32 = 32;
const HALF_N: i32 = 16;
const ETA: f32 = 1.3;

struct DensityParams {
    num_particles: u32,
    pad0: u32, pad1: u32, pad2: u32,
    world_half: f32,
    cell_size: f32,
    h_min: f32,
    h_max: f32,
}

@group(0) @binding(0) var<uniform> params: DensityParams;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> masses: array<f32>;
@group(0) @binding(3) var<storage, read_write> smoothing_lengths: array<f32>;
@group(0) @binding(4) var<storage, read_write> densities: array<f32>;
@group(0) @binding(5) var<storage, read> hash_offsets: array<u32>;
@group(0) @binding(6) var<storage, read> sorted_particles: array<u32>;

fn bucket_xyz(pos: vec3<f32>) -> vec3<i32> {
    let bx = i32(floor(pos.x / params.cell_size)) + HALF_N;
    let by = i32(floor(pos.y / params.cell_size)) + HALF_N;
    let bz = i32(floor(pos.z / params.cell_size)) + HALF_N;
    return vec3<i32>(
        clamp(bx, 0, GRID_N - 1),
        clamp(by, 0, GRID_N - 1),
        clamp(bz, 0, GRID_N - 1),
    );
}

fn wendland_c2(r: f32, h: f32) -> f32 {
    let q = r / h;
    if (q >= 2.0) { return 0.0; }
    let h3 = h * h * h;
    let one_minus_half_q = 1.0 - 0.5 * q;
    let factor = one_minus_half_q * one_minus_half_q * one_minus_half_q * one_minus_half_q;
    return (21.0 / (16.0 * PI * h3)) * factor * (1.0 + 2.0 * q);
}

@compute @workgroup_size(64)
fn density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let xi = vec3<f32>(
        positions[i].x,
        positions[i].y,
        positions[i].z,
    );
    let mi = masses[i];
    let h = smoothing_lengths[i];

    var rho: f32 = 0.0;
    let b = bucket_xyz(xi);

    for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                let nbx = clamp(b.x + dx, 0, GRID_N - 1);
                let nby = clamp(b.y + dy, 0, GRID_N - 1);
                let nbz = clamp(b.z + dz, 0, GRID_N - 1);
                let bucket = u32(nbx + nby * GRID_N + nbz * GRID_N * GRID_N);
                let start = hash_offsets[bucket];
                let end = hash_offsets[bucket + 1u];
                for (var slot: u32 = start; slot < end; slot = slot + 1u) {
                    let j = sorted_particles[slot];
                    let xj = vec3<f32>(
                        positions[j].x,
                        positions[j].y,
                        positions[j].z,
                    );
                    let d = xj - xi;
                    let r = sqrt(dot(d, d));
                    rho = rho + masses[j] * wendland_c2(r, h);
                }
            }
        }
    }

    densities[i] = rho;
    // Update smoothing length via the standard `h = η · (m/ρ)^(1/3)`
    // relation, clamped to the grid's supported range so the 3×3×3
    // stencil keeps covering 2h.
    let h_new = ETA * pow(mi / max(rho, 1e-30), 1.0 / 3.0);
    smoothing_lengths[i] = clamp(h_new, params.h_min, params.h_max);
}
