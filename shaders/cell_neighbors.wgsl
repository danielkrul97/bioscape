// Sprint 49: GPU broad-phase neighbor query. Konzumuje (offsets, sorted_cells)
// produkované SpatialHashGpu (Sprint 45) — ne-rebuilduje hash, ten musí být
// ready před dispatch.
//
// Per cell i:
//   - bucket = bucket_id(positions[i])
//   - iteruj 3³ buckets kolem
//   - pro každého souseda j != i:
//     - dx,dy,dz = j - i
//     - d2 = dx² + dy² + dz²
//     - if d2 ≤ vision_radius²[i]:
//        - count++
//        - if d2 < best_d2: best = (j, dx, dy, dz, radii[j])
//
// Output (per cell, 5 floats):
//   [0] nearest_cell_dx (0 if no neighbor)
//   [1] nearest_cell_dy
//   [2] nearest_cell_dz
//   [3] nearest_cell_radius (-1 if no neighbor — sentinel)
//   [4] neighbors_in_vision count (as f32; bitcast<u32> caller side)
//
// Layout musí matchnout SpatialHashGpu (Sprint 45) bucket grid.

const GRID_NX: i32 = 64;
const GRID_NY: i32 = 32;
const GRID_NZ: i32 = 4;
const HALF_NX: i32 = 32;
const HALF_NY: i32 = 16;
const HALF_NZ: i32 = 2;

struct Params {
    num_cells: u32,
    cell_size: f32,
    /// Sprint 55: world bounds pro toroidal bucket wrap + min-image distance.
    world_half_x: f32,
    world_half_y: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<f32>;        // N*3
@group(0) @binding(2) var<storage, read> radii: array<f32>;            // N
@group(0) @binding(3) var<storage, read> vision_radii: array<f32>;     // N
@group(0) @binding(4) var<storage, read> hash_offsets: array<u32>;     // NUM_BUCKETS+1
@group(0) @binding(5) var<storage, read> hash_sorted: array<u32>;      // N
@group(0) @binding(6) var<storage, read_write> output: array<f32>;     // N*5

// Sprint 55: minimum-image displacement na xy axes. Returns dx tak, že
// |dx| ≤ world_half. Pre-Sprint-55 byl raw subtraction.
fn min_image_xy(d: f32, half: f32) -> f32 {
    let w = 2.0 * half;
    if (d > half) { return d - w; }
    if (d < -half) { return d + w; }
    return d;
}

fn bucket_id_of(pos: vec3<f32>) -> u32 {
    // Sprint 55: wrap xy do [-half, half) než spočítáme bucket.
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
fn neighbors(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }
    let pos_i = vec3<f32>(
        positions[i * 3u + 0u],
        positions[i * 3u + 1u],
        positions[i * 3u + 2u],
    );
    let vr = vision_radii[i];
    let vr2 = vr * vr;

    var best_d2 = vr2 + 1.0;
    var best_dx: f32 = 0.0;
    var best_dy: f32 = 0.0;
    var best_dz: f32 = 0.0;
    var best_radius: f32 = -1.0;
    var count: u32 = 0u;

    // Sprint 55: iterate 3³ buckets s ghost positions (pos_i + offset). bucket_id_of
    // wraps pos automaticky → toroidal lookup. Narrow-phase používá min_image_xy
    // pro delta (dx, dy přes wrap → |dx|, |dy| ≤ world_half).
    let r_cells = i32(ceil(vr / params.cell_size));
    let cs = params.cell_size;
    for (var dx = -r_cells; dx <= r_cells; dx = dx + 1) {
        for (var dy = -r_cells; dy <= r_cells; dy = dy + 1) {
            for (var dz = -r_cells; dz <= r_cells; dz = dz + 1) {
                let nbr_pos = vec3<f32>(
                    pos_i.x + f32(dx) * cs,
                    pos_i.y + f32(dy) * cs,
                    pos_i.z + f32(dz) * cs,
                );
                let b = bucket_id_of(nbr_pos);
                let start = hash_offsets[b];
                let end = hash_offsets[b + 1u];
                for (var k = start; k < end; k = k + 1u) {
                    let j = hash_sorted[k];
                    if (j == i) {
                        continue;
                    }
                    let pos_j = vec3<f32>(
                        positions[j * 3u + 0u],
                        positions[j * 3u + 1u],
                        positions[j * 3u + 2u],
                    );
                    // Sprint 55: min-image delta — toroidal-aware distance.
                    let dxf = min_image_xy(pos_j.x - pos_i.x, params.world_half_x);
                    let dyf = min_image_xy(pos_j.y - pos_i.y, params.world_half_y);
                    let dzf = pos_j.z - pos_i.z;
                    let d2 = dxf * dxf + dyf * dyf + dzf * dzf;
                    if (d2 <= vr2) {
                        count = count + 1u;
                        if (d2 < best_d2) {
                            best_d2 = d2;
                            best_dx = dxf;
                            best_dy = dyf;
                            best_dz = dzf;
                            best_radius = radii[j];
                        }
                    }
                }
            }
        }
    }

    let off = i * 5u;
    output[off + 0u] = best_dx;
    output[off + 1u] = best_dy;
    output[off + 2u] = best_dz;
    output[off + 3u] = best_radius;
    output[off + 4u] = bitcast<f32>(count);
}
