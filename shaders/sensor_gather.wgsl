// Sprint 50: GPU mirror of World::brain_act sensor gather (headless). Per cell:
//   - cell_hash query → nearest cell + neighbors_in_vision count
//   - food_hash query → nearest food
//   - sample 3D smell + pheromone + vibration gradients via finite difference
//
// Consumes:
//   SpatialHashGpu (cells)        — Sprint 45
//   SpatialHashGpu (foods)        — separate instance
//   FieldGpu × 3 (smell, pheromone, vibration) — grids read as array<u32>
//                                                and bitcast to f32.
//
// Output layout per cell (19 f32, stride 19 — V7 added vibration grad+amp):
//   [0..3]   nearest_food.dx,dy,dz   (0 if has_food == 0)
//   [3]      has_food (0.0 / 1.0)
//   [4..7]   nearest_cell.dx,dy,dz
//   [7]      nearest_cell radius (-1.0 = no cell — sentinel)
//   [8..11]  smell_grad.x, .y, .z
//   [11..14] pheromone_grad.x, .y, .z
//   [14]     neighbors_in_vision count, bitcast<f32>(u32)
//   [15..18] vibration_grad.x, .y, .z
//   [18]     vibration_amp (scalar sample at cell pos)

const GRID_NX: i32 = 64;
const GRID_NY: i32 = 32;
const GRID_NZ: i32 = 4;
const HALF_NX: i32 = 32;
const HALF_NY: i32 = 16;
const HALF_NZ: i32 = 2;

struct SensorParams {
    num_cells: u32,
    num_foods: u32,
    hash_cell_size: f32,
    world_half_x: f32,
    world_half_y: f32,
    world_half_z: f32,
    field_res_x: u32,
    field_res_y: u32,
    field_res_z: u32,
    field_eps: f32,
    field_world_half_x: f32,
    field_world_half_y: f32,
    field_world_half_z: f32,
    // Wave 5 maze fields. `maze_active != 0` enables LOS raycast that
    // filters food/cell candidates blocked by walls. Whisker raycast
    // deferred (needs heading + pitch bindings, see binding 13 comment).
    maze_active: u32,
    maze_res_x: u32,
    maze_res_y: u32,
}

@group(0) @binding(0) var<uniform> params: SensorParams;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> eff_radii: array<f32>;
@group(0) @binding(3) var<storage, read> vision_radii: array<f32>;
@group(0) @binding(4) var<storage, read> food_positions: array<f32>;
@group(0) @binding(5) var<storage, read> cell_hash_offsets: array<u32>;
@group(0) @binding(6) var<storage, read> cell_hash_sorted: array<u32>;
@group(0) @binding(7) var<storage, read> food_hash_offsets: array<u32>;
@group(0) @binding(8) var<storage, read> food_hash_sorted: array<u32>;
@group(0) @binding(9) var<storage, read> smell_grid: array<u32>;
@group(0) @binding(10) var<storage, read> pheromone_grid: array<u32>;
@group(0) @binding(11) var<storage, read_write> output: array<f32>;
@group(0) @binding(12) var<storage, read> vibration_grid: array<u32>;
// Wave 5: maze occupancy mask (binding 13). Same format as
// step.wgsl::maze_mask — one u32 per voxel, row-major
// `vy * maze_res_x + vx`, value 0 = passable. Read by `raycast_blocked`
// for LOS filtering. Whisker raycast deferred — would need per-cell
// heading + pitch as additional bindings, repacking to fit the storage
// limit; postponed to Wave 6.
@group(0) @binding(13) var<storage, read> maze_mask: array<u32>;

// Toroidal minimum-image displacement (used for both x and y).
fn min_image_xy(d: f32, half: f32) -> f32 {
    let w = 2.0 * half;
    if (d > half) { return d - w; }
    if (d < -half) { return d + w; }
    return d;
}

// Wave 5: xy raycast against the maze obstacle mask. Mirror of
// `ObstacleField::raycast_blocked` — uniform-step sampling at half-voxel
// pitch. Returns true if any voxel between `origin` and `target` is
// occupied. Caller MUST check `params.maze_active != 0u` first; this
// helper assumes it.
fn raycast_blocked(origin: vec3<f32>, tgt: vec3<f32>) -> bool {
    let dx = tgt.x - origin.x;
    let dy = tgt.y - origin.y;
    let dist = sqrt(dx * dx + dy * dy);
    if (dist < 1e-3) {
        return false;
    }
    let nx = i32(params.maze_res_x);
    let ny = i32(params.maze_res_y);
    let cs_x = (2.0 * params.world_half_x) / f32(params.maze_res_x);
    let cs_y = (2.0 * params.world_half_y) / f32(params.maze_res_y);
    let step = min(cs_x, cs_y) * 0.5;
    let n_steps = u32(ceil(dist / step));
    if (n_steps <= 1u) {
        return false;
    }
    for (var k: u32 = 1u; k < n_steps; k = k + 1u) {
        let t = f32(k) * step / dist;
        let px = origin.x + dx * t;
        let py = origin.y + dy * t;
        let xi = i32(floor((px + params.world_half_x) / cs_x));
        let yi = i32(floor((py + params.world_half_y) / cs_y));
        if (xi < 0 || xi >= nx || yi < 0 || yi >= ny) {
            return true; // outside bounds = solid in maze mode
        }
        let idx = u32(yi) * params.maze_res_x + u32(xi);
        if (maze_mask[idx] != 0u) {
            return true;
        }
    }
    return false;
}

// Bucket coordinates for a position. xy is wrapped to [-half, half) before
// bucketing (toroidal); z is clamped (not toroidal).
fn bucket_coords_of(pos: vec3<f32>) -> vec3<i32> {
    let wx = 2.0 * params.world_half_x;
    let wy = 2.0 * params.world_half_y;
    let pos_wx = pos.x - floor((pos.x + params.world_half_x) / wx) * wx;
    let pos_wy = pos.y - floor((pos.y + params.world_half_y) / wy) * wy;
    let bx = i32(floor(pos_wx / params.hash_cell_size)) + HALF_NX;
    let by = i32(floor(pos_wy / params.hash_cell_size)) + HALF_NY;
    let bz = i32(floor(pos.z / params.hash_cell_size)) + HALF_NZ;
    return vec3<i32>(
        clamp(bx, 0, GRID_NX - 1),
        clamp(by, 0, GRID_NY - 1),
        clamp(bz, 0, GRID_NZ - 1),
    );
}

// Field grid constants — computed once and reused across all gradient samples.
// Storing inverse cell sizes turns 18 per-sample divisions into 18 multiplies.
struct FieldConsts {
    nx: i32,
    ny: i32,
    nz: i32,
    inv_cell_x: f32,
    inv_cell_y: f32,
    inv_cell_z: f32,
};

struct FieldSample {
    smell: f32,
    pheromone: f32,
    vibration: f32,
};

// Single-position sample of all three grids. xy is toroidal (modulo wrap); z
// out of range yields a zero triple (matches CPU SmellField sample-on-bounds).
// Vibration shares the same resolution + world_half as smell/pheromone (the
// renderer/headless both allocate `VIBRATION_GRID_RES = SMELL_GRID_RES`), so a
// single addressing pass serves all three reads.
fn sample_all_at(pos: vec3<f32>, fc: FieldConsts) -> FieldSample {
    let zi = i32(floor((pos.z + params.field_world_half_z) * fc.inv_cell_z));
    if (zi < 0 || zi >= fc.nz) {
        return FieldSample(0.0, 0.0, 0.0);
    }
    let xi_raw = i32(floor((pos.x + params.field_world_half_x) * fc.inv_cell_x));
    let yi_raw = i32(floor((pos.y + params.field_world_half_y) * fc.inv_cell_y));
    let xi = ((xi_raw % fc.nx) + fc.nx) % fc.nx;
    let yi = ((yi_raw % fc.ny) + fc.ny) % fc.ny;
    let idx = u32(zi * fc.nx * fc.ny + yi * fc.nx + xi);
    return FieldSample(
        bitcast<f32>(smell_grid[idx]),
        bitcast<f32>(pheromone_grid[idx]),
        bitcast<f32>(vibration_grid[idx]),
    );
}

@compute @workgroup_size(64)
fn sensor_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    // Resolve center bucket once and walk neighbors via integer ±wrap on xy
    // (z clamped). Replaces the per-iteration bucket_id_wrapped() chain.
    // Assumes r_cells < GRID_N/2 (vision radius < world half-extent).
    let center = bucket_coords_of(pos_i);
    let r_cells = i32(ceil(vr / params.hash_cell_size));

    // Cell scan accumulators.
    var best_cell_d2 = vr2 + 1.0;
    var best_cell_dx: f32 = 0.0;
    var best_cell_dy: f32 = 0.0;
    var best_cell_dz: f32 = 0.0;
    var best_cell_radius: f32 = -1.0;
    var neighbors_count: u32 = 0u;

    // Food scan accumulators.
    var best_food_d2 = vr2 + 1.0;
    var best_food_dx: f32 = 0.0;
    var best_food_dy: f32 = 0.0;
    var best_food_dz: f32 = 0.0;
    var has_food: f32 = 0.0;

    // Uniform across the whole dispatch — no warp divergence on this branch.
    let scan_foods = params.num_foods > 0u;

    // Fused outer loop: each (dx,dy,dz) resolves one bucket id, then both
    // hashes are queried at that bucket. Halves bucket-coord overhead.
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

                // Cell scan: nearest neighbor + count.
                let c_start = cell_hash_offsets[b];
                let c_end = cell_hash_offsets[b + 1u];
                for (var k = c_start; k < c_end; k = k + 1u) {
                    let j = cell_hash_sorted[k];
                    if (j == i) {
                        continue;
                    }
                    let pj = vec3<f32>(
                        positions[j * 3u + 0u],
                        positions[j * 3u + 1u],
                        positions[j * 3u + 2u],
                    );
                    let dxf = min_image_xy(pj.x - pos_i.x, params.world_half_x);
                    let dyf = min_image_xy(pj.y - pos_i.y, params.world_half_y);
                    let dzf = pj.z - pos_i.z;
                    let d2 = dxf * dxf + dyf * dyf + dzf * dzf;
                    if (d2 <= vr2) {
                        // Wave 5: LOS check — skip neighbors hidden by walls.
                        // Mirror of CPU `bioscape::los_clear`. Only fires
                        // when the maze is active to keep the no-maze path
                        // identical.
                        if (params.maze_active != 0u && raycast_blocked(pos_i, pj)) {
                            continue;
                        }
                        neighbors_count = neighbors_count + 1u;
                        if (d2 < best_cell_d2) {
                            best_cell_d2 = d2;
                            best_cell_dx = dxf;
                            best_cell_dy = dyf;
                            best_cell_dz = dzf;
                            best_cell_radius = eff_radii[j];
                        }
                    }
                }

                // Food scan: nearest only.
                if (scan_foods) {
                    let f_start = food_hash_offsets[b];
                    let f_end = food_hash_offsets[b + 1u];
                    for (var k = f_start; k < f_end; k = k + 1u) {
                        let f = food_hash_sorted[k];
                        let pf = vec3<f32>(
                            food_positions[f * 3u + 0u],
                            food_positions[f * 3u + 1u],
                            food_positions[f * 3u + 2u],
                        );
                        let dxf = min_image_xy(pf.x - pos_i.x, params.world_half_x);
                        let dyf = min_image_xy(pf.y - pos_i.y, params.world_half_y);
                        let dzf = pf.z - pos_i.z;
                        let d2 = dxf * dxf + dyf * dyf + dzf * dzf;
                        if (d2 <= vr2 && d2 < best_food_d2) {
                            // Wave 5: LOS check — skip food blocked by walls.
                            if (params.maze_active != 0u && raycast_blocked(pos_i, pf)) {
                                continue;
                            }
                            best_food_d2 = d2;
                            best_food_dx = dxf;
                            best_food_dy = dyf;
                            best_food_dz = dzf;
                            has_food = 1.0;
                        }
                    }
                }
            }
        }
    }

    // Field gradients: 6 dual-grid samples instead of 12 single-grid ones.
    let fc = FieldConsts(
        i32(params.field_res_x),
        i32(params.field_res_y),
        i32(params.field_res_z),
        f32(params.field_res_x) / (2.0 * params.field_world_half_x),
        f32(params.field_res_y) / (2.0 * params.field_world_half_y),
        f32(params.field_res_z) / (2.0 * params.field_world_half_z),
    );
    let eps = params.field_eps;
    let s_xp = sample_all_at(vec3<f32>(pos_i.x + eps, pos_i.y, pos_i.z), fc);
    let s_xm = sample_all_at(vec3<f32>(pos_i.x - eps, pos_i.y, pos_i.z), fc);
    let s_yp = sample_all_at(vec3<f32>(pos_i.x, pos_i.y + eps, pos_i.z), fc);
    let s_ym = sample_all_at(vec3<f32>(pos_i.x, pos_i.y - eps, pos_i.z), fc);
    let s_zp = sample_all_at(vec3<f32>(pos_i.x, pos_i.y, pos_i.z + eps), fc);
    let s_zm = sample_all_at(vec3<f32>(pos_i.x, pos_i.y, pos_i.z - eps), fc);
    // 7th sample at cell position for vibration amplitude (smell/pheromone use
    // gradient only). Reuses the same addressing path — no extra divergence
    // hit, just one extra grid read.
    let s_center = sample_all_at(pos_i, fc);
    let inv_2eps = 0.5 / eps;

    let smell_grad = vec3<f32>(
        (s_xp.smell - s_xm.smell) * inv_2eps,
        (s_yp.smell - s_ym.smell) * inv_2eps,
        (s_zp.smell - s_zm.smell) * inv_2eps,
    );
    let pheromone_grad = vec3<f32>(
        (s_xp.pheromone - s_xm.pheromone) * inv_2eps,
        (s_yp.pheromone - s_ym.pheromone) * inv_2eps,
        (s_zp.pheromone - s_zm.pheromone) * inv_2eps,
    );
    let vibration_grad = vec3<f32>(
        (s_xp.vibration - s_xm.vibration) * inv_2eps,
        (s_yp.vibration - s_ym.vibration) * inv_2eps,
        (s_zp.vibration - s_zm.vibration) * inv_2eps,
    );
    let vibration_amp = s_center.vibration;

    let off = i * 19u;
    output[off + 0u] = best_food_dx;
    output[off + 1u] = best_food_dy;
    output[off + 2u] = best_food_dz;
    output[off + 3u] = has_food;
    output[off + 4u] = best_cell_dx;
    output[off + 5u] = best_cell_dy;
    output[off + 6u] = best_cell_dz;
    output[off + 7u] = best_cell_radius;
    output[off + 8u] = smell_grad.x;
    output[off + 9u] = smell_grad.y;
    output[off + 10u] = smell_grad.z;
    output[off + 11u] = pheromone_grad.x;
    output[off + 12u] = pheromone_grad.y;
    output[off + 13u] = pheromone_grad.z;
    output[off + 14u] = bitcast<f32>(neighbors_count);
    output[off + 15u] = vibration_grad.x;
    output[off + 16u] = vibration_grad.y;
    output[off + 17u] = vibration_grad.z;
    output[off + 18u] = vibration_amp;
}