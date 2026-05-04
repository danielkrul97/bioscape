// Sprint 50: GPU mirror headless::resolve_collisions. Per cell:
//   - query SpatialHashGpu pro neighbors v search_radius
//   - pro každého souseda j ≠ i s d² < pair_r² && d² > 0:
//     delta[i] += (d/|d|) × overlap × 0.5
//   - output delta[i] (write-only per i — žádný atomic)
//
// search_radius = CELL_RADIUS × (eff_r_i + max_axis_i × 2). Konzervativní bound
// match pre-Sprint-50 CPU helper.

const GRID_NX: i32 = 64;
const GRID_NY: i32 = 32;
const GRID_NZ: i32 = 4;
const HALF_NX: i32 = 32;
const HALF_NY: i32 = 16;
const HALF_NZ: i32 = 2;

struct CollisionParams {
    num_cells: u32,
    cell_size: f32,
    cell_radius_const: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: CollisionParams;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> eff_radii: array<f32>;
@group(0) @binding(3) var<storage, read> max_axes: array<f32>;
@group(0) @binding(4) var<storage, read> hash_offsets: array<u32>;
@group(0) @binding(5) var<storage, read> hash_sorted: array<u32>;
@group(0) @binding(6) var<storage, read_write> deltas: array<f32>;

@compute @workgroup_size(64)
fn collision(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }
    let pos_i = vec3<f32>(
        positions[i * 3u + 0u],
        positions[i * 3u + 1u],
        positions[i * 3u + 2u],
    );
    let r_i = eff_radii[i];
    let search_r = params.cell_radius_const * (r_i + max_axes[i] * 2.0);
    let r_cells = i32(ceil(search_r / params.cell_size));

    let bx_base = i32(floor(pos_i.x / params.cell_size)) + HALF_NX;
    let by_base = i32(floor(pos_i.y / params.cell_size)) + HALF_NY;
    let bz_base = i32(floor(pos_i.z / params.cell_size)) + HALF_NZ;

    var dx_acc: f32 = 0.0;
    var dy_acc: f32 = 0.0;
    var dz_acc: f32 = 0.0;

    for (var dx = -r_cells; dx <= r_cells; dx = dx + 1) {
        for (var dy = -r_cells; dy <= r_cells; dy = dy + 1) {
            for (var dz = -r_cells; dz <= r_cells; dz = dz + 1) {
                let bx = clamp(bx_base + dx, 0, GRID_NX - 1);
                let by = clamp(by_base + dy, 0, GRID_NY - 1);
                let bz = clamp(bz_base + dz, 0, GRID_NZ - 1);
                let b = u32(bx + by * GRID_NX + bz * GRID_NX * GRID_NY);
                let start = hash_offsets[b];
                let end = hash_offsets[b + 1u];
                for (var k = start; k < end; k = k + 1u) {
                    let j = hash_sorted[k];
                    if (j == i) {
                        continue;
                    }
                    let r_j = eff_radii[j];
                    let pair_r = params.cell_radius_const * (r_i + r_j);
                    let pair_r2 = pair_r * pair_r;
                    let pj = vec3<f32>(
                        positions[j * 3u + 0u],
                        positions[j * 3u + 1u],
                        positions[j * 3u + 2u],
                    );
                    let d = pos_i - pj;
                    let d2 = dot(d, d);
                    if (d2 < pair_r2 && d2 > 0.0) {
                        let dist = sqrt(d2);
                        let overlap = pair_r - dist;
                        let scale = overlap * 0.5 / dist;
                        dx_acc = dx_acc + d.x * scale;
                        dy_acc = dy_acc + d.y * scale;
                        dz_acc = dz_acc + d.z * scale;
                    }
                }
            }
        }
    }

    deltas[i * 3u + 0u] = dx_acc;
    deltas[i * 3u + 1u] = dy_acc;
    deltas[i * 3u + 2u] = dz_acc;
}
