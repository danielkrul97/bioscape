// Sprint 49: GPU broad-phase neighbor query. Consumes (offsets, sorted_cells)
// produced by SpatialHashGpu (Sprint 45) — does not rebuild the hash; that
// must be ready before dispatch.
//
// Per cell i:
//   - locate its center bucket
//   - iterate (2*r+1)³ buckets around it (toroidal on xy, clamped on z)
//   - for each candidate j != i, compute min-image distance and track nearest
//
// Output (per cell, 5 floats):
//   [0..2] nearest cell delta (xyz; 0 if no neighbor)
//   [3]    nearest cell radius (-1 sentinel if no neighbor)
//   [4]    neighbors_in_vision count (bitcast<u32> on caller side)
//
// Bucket grid layout must match SpatialHashGpu (Sprint 45).

const GRID_NX: i32 = 64;
const GRID_NY: i32 = 32;
const GRID_NZ: i32 = 4;
const HALF_NX: i32 = 32;
const HALF_NY: i32 = 16;
const HALF_NZ: i32 = 2;

struct Params {
    num_cells: u32,
    cell_size: f32,
    // World half-extents for toroidal xy wrap and min-image distance (Sprint 55).
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

// Toroidal minimum-image displacement on xy. Returns dx with |dx| ≤ half.
fn min_image_xy(d: f32, half: f32) -> f32 {
    let w = 2.0 * half;
    if (d > half) { return d - w; }
    if (d < -half) { return d + w; }
    return d;
}

// Bucket coordinates for a position. xy is wrapped to [-half, half) before
// bucketing (toroidal); z is not wrapped, only clamped to grid extents.
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

    // Sentinel ensures the first valid candidate always wins the tracking branch.
    var best_d2 = vr2 + 1.0;
    var best_dx: f32 = 0.0;
    var best_dy: f32 = 0.0;
    var best_dz: f32 = 0.0;
    var best_radius: f32 = -1.0;
    var count: u32 = 0u;

    // Resolve the center bucket once and walk neighbors via integer
    // increment + wrap, instead of recomputing bucket_id_of() per ghost
    // position (each call carries a floor/div on the xy wrap).
    //
    // Assumes r_cells < GRID_N/2, i.e. vision_radius < world half-extent —
    // otherwise the simple ±wrap below would alias. Larger ranges would need
    // proper modulo, but spatial hashing itself breaks down before that.
    let center = bucket_coords_of(pos_i);
    let r_cells = i32(ceil(vr / params.cell_size));

    for (var dz = -r_cells; dz <= r_cells; dz = dz + 1) {
        // z is not toroidal; out-of-range z maps to the boundary bucket
        // (matches pre-refactor semantics — may revisit it for r_cells > 1).
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
                    if (j == i) {
                        continue;
                    }
                    let pos_j = vec3<f32>(
                        positions[j * 3u + 0u],
                        positions[j * 3u + 1u],
                        positions[j * 3u + 2u],
                    );
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