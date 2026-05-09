// Per-cell collision resolution against spatial-hash neighbors. For each
// pair (i, j) with d² < (CELL_RADIUS × (eff_r_i + eff_r_j))² and d² > 0,
// `deltas[i]` accumulates (d/|d|) × overlap × 0.5. Output is write-only
// per i — no atomics needed. The XY world is toroidal, so the search
// neighborhood walks 3D ghost positions around `pos_i` to cover wrap, and
// pair distances use the min-image convention. Search radius bound matches
// the CPU helper: CELL_RADIUS × (eff_r_i + max_axis_i × 2).

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
    world_half_x: f32,
    world_half_y: f32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: CollisionParams;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> eff_radii: array<f32>;
@group(0) @binding(3) var<storage, read> max_axes: array<f32>;
@group(0) @binding(4) var<storage, read> hash_offsets: array<u32>;
@group(0) @binding(5) var<storage, read> hash_sorted: array<u32>;
@group(0) @binding(6) var<storage, read_write> deltas: array<f32>;

fn min_image_xy(d: f32, half: f32) -> f32 {
    let w = 2.0 * half;
    if (d > half) { return d - w; }
    if (d < -half) { return d + w; }
    return d;
}

fn bucket_coords_of(pos: vec3<f32>) -> vec3<i32> {
    let wx = 2.0 * params.world_half_x;
    let wy = 2.0 * params.world_half_y;
    let pos_wx = pos.x - floor((pos.x + params.world_half_x) / wx) * wx;
    let pos_wy = pos.y - floor((pos.y + params.world_half_y) / wy) * wy;
    let bx = i32(floor(pos_wx / params.cell_size)) + HALF_NX;
    let by = i32(floor(pos_wy / params.cell_size)) + HALF_NY;
    let bz = i32(floor(pos.z / params.cell_size)) + HALF_NZ;
    return vec3<i32>(
        clamp(bx, 0, GRID_NX - 1),
        clamp(by, 0, GRID_NY - 1),
        clamp(bz, 0, GRID_NZ - 1),
    );
}

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
    let crc = params.cell_radius_const;
    let crc_r_i = crc * r_i;
    let search_r = crc * (r_i + max_axes[i] * 2.0);
    let cs = params.cell_size;
    let r_cells = i32(ceil(search_r / cs));

    var dx_acc: f32 = 0.0;
    var dy_acc: f32 = 0.0;
    var dz_acc: f32 = 0.0;

    // Resolve the center bucket once, walk neighbors via integer ±wrap on xy
    // (z clamped). Replaces the per-iteration `bucket_id_wrapped(ghost_pos)`
    // chain — same pattern as `sensor_gather.wgsl`.
    let center = bucket_coords_of(pos_i);
    for (var dz = -r_cells; dz <= r_cells; dz = dz + 1) {
        let bz = clamp(center.z + dz, 0, GRID_NZ - 1);
        for (var dy = -r_cells; dy <= r_cells; dy = dy + 1) {
            var by = center.y + dy;
            if (by < 0) { by = by + GRID_NY; }
            else if (by >= GRID_NY) { by = by - GRID_NY; }
            for (var dx = -r_cells; dx <= r_cells; dx = dx + 1) {
                var bx = center.x + dx;
                if (bx < 0) { bx = bx + GRID_NX; }
                else if (bx >= GRID_NX) { bx = bx - GRID_NX; }
                let b = u32(bx + by * GRID_NX + bz * GRID_NX * GRID_NY);
                let start = hash_offsets[b];
                let end = hash_offsets[b + 1u];
                for (var k = start; k < end; k = k + 1u) {
                    let j = hash_sorted[k];
                    // No `j == i` guard: same-cell pairs give d² = 0, which the
                    // `d2 > 0.0` filter below rejects. Skipping the explicit
                    // check removes one branch per neighbor.
                    let r_j = eff_radii[j];
                    let pair_r = crc_r_i + crc * r_j;
                    let pair_r2 = pair_r * pair_r;
                    let pj = vec3<f32>(
                        positions[j * 3u + 0u],
                        positions[j * 3u + 1u],
                        positions[j * 3u + 2u],
                    );
                    let d = vec3<f32>(
                        min_image_xy(pos_i.x - pj.x, params.world_half_x),
                        min_image_xy(pos_i.y - pj.y, params.world_half_y),
                        pos_i.z - pj.z,
                    );
                    let d2 = dot(d, d);
                    if (d2 < pair_r2 && d2 > 0.0) {
                        // Algebraically: overlap*0.5/dist = pair_r*0.5/dist - 0.5.
                        // Replaces sqrt + divide with a single rsqrt + fma.
                        let scale = pair_r * 0.5 * inverseSqrt(d2) - 0.5;
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
