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
    // Sprint 195: current sim tick — seeds the whisker transduction noise.
    // _pad rounds the uniform struct to 80 bytes (mirrors SensorParamsGpu).
    tick: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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
// for LOS filtering and whisker raycast.
@group(0) @binding(13) var<storage, read> maze_mask: array<u32>;
// Wave 6: per-cell heading + pitch for whisker raycast direction.
// Allocated by SensorGatherGpu, uploaded each dispatch_no_readback.
@group(0) @binding(14) var<storage, read> headings: array<f32>;
@group(0) @binding(15) var<storage, read> pitches: array<f32>;
// Wave L: per-channel pheromone grids (ch1, ch2). Same resolution +
// world_half as `pheromone_grid` (ch0). Gradients fed to brain inputs
// 21..26 via populate_inputs.wgsl.
@group(0) @binding(16) var<storage, read> pheromone_grid_ch1: array<u32>;
@group(0) @binding(17) var<storage, read> pheromone_grid_ch2: array<u32>;
// Sprint 195: per-cell whisker spring-damper state, 12 f32 per cell,
// layout [deflection0..6, velocity0..6]. Persistent across ticks (owned
// by CellsGpu::whisker_state_buf), mutated in place below.
@group(0) @binding(18) var<storage, read_write> whisker_state: array<f32>;

// Toroidal minimum-image displacement (used for both x and y).
fn min_image_xy(d: f32, half: f32) -> f32 {
    let w = 2.0 * half;
    if (d > half) { return d - w; }
    if (d < -half) { return d + w; }
    return d;
}

// Sprint 195: deterministic stateless transduction noise for the whisker
// spring-damper model. Mirrors `obstacles::whisker_noise` with identical
// integer arithmetic (WGSL u32 ops wrap by default). Returns [-1, 1).
fn whisker_noise(cell_index: u32, tick: u32, whisker_k: u32) -> f32 {
    var h: u32 = cell_index * 0x9E3779B9u;
    h = h ^ (tick * 0x85EBCA6Bu);
    h = h ^ (whisker_k * 0xC2B2AE35u);
    h = h ^ (h >> 16u);
    h = h * 0x7FEB352Du;
    h = h ^ (h >> 15u);
    h = h * 0x846CA68Bu;
    h = h ^ (h >> 16u);
    return f32(h >> 8u) / 8388608.0 - 1.0;
}

// Wave 6: single-direction whisker raycast in xy. Mirrors one ray of
// `ObstacleField::whisker_distances`. `dir` is a unit vector in world
// frame (caller derives from heading/pitch). Returns normalized
// free-distance in [0, 1] — 1.0 = clear within WHISKER_RANGE, 0.0 =
// wall touching origin. ±z directions short-circuit to 1.0 since walls
// span full z height; caller skips those rays.
fn whisker_distance(origin: vec3<f32>, dir: vec3<f32>) -> f32 {
    if (abs(dir.x) < 1e-6 && abs(dir.y) < 1e-6) {
        return 1.0;
    }
    let cs_x = (2.0 * params.world_half_x) / f32(params.maze_res_x);
    let cs_y = (2.0 * params.world_half_y) / f32(params.maze_res_y);
    let step = min(cs_x, cs_y) * 0.5;
    let whisker_range: f32 = 90.0; // mirrors lib::WHISKER_RANGE
    let n_steps = u32(ceil(whisker_range / step));
    let nx = i32(params.maze_res_x);
    let ny = i32(params.maze_res_y);
    for (var k: u32 = 1u; k <= n_steps; k = k + 1u) {
        let t = f32(k) * step;
        let px = origin.x + dir.x * t;
        let py = origin.y + dir.y * t;
        let xi = i32(floor((px + params.world_half_x) / cs_x));
        let yi = i32(floor((py + params.world_half_y) / cs_y));
        if (xi < 0 || xi >= nx || yi < 0 || yi >= ny) {
            return clamp(t / whisker_range, 0.0, 1.0);
        }
        let idx = u32(yi) * params.maze_res_x + u32(xi);
        if (maze_mask[idx] != 0u) {
            return clamp(t / whisker_range, 0.0, 1.0);
        }
    }
    return 1.0;
}

// Wave 5: xy raycast against the maze obstacle mask. Mirror of
// `ObstacleField::raycast_blocked`. Caller MUST check `params.maze_active
// != 0u` first; this helper assumes it.
//
// Amanatides–Woo voxel traversal (DDA): each voxel along the ray is
// visited exactly once. Pre-S188 uniform-step sampling at half-voxel
// pitch oversampled by ~2× and used `sqrt(d²)` per call as an early-out
// — the squared-length check below makes that sqrt unnecessary.
fn raycast_blocked(origin: vec3<f32>, tgt: vec3<f32>) -> bool {
    let dx = tgt.x - origin.x;
    let dy = tgt.y - origin.y;
    let d2 = dx * dx + dy * dy;
    if (d2 < 1e-6) {
        return false;
    }
    let nx = i32(params.maze_res_x);
    let ny = i32(params.maze_res_y);
    let cs_x = (2.0 * params.world_half_x) / f32(params.maze_res_x);
    let cs_y = (2.0 * params.world_half_y) / f32(params.maze_res_y);
    let inv_cs_x = 1.0 / cs_x;
    let inv_cs_y = 1.0 / cs_y;

    let ox = origin.x + params.world_half_x;
    let oy = origin.y + params.world_half_y;
    let tx = tgt.x + params.world_half_x;
    let ty = tgt.y + params.world_half_y;
    var xi = i32(floor(ox * inv_cs_x));
    var yi = i32(floor(oy * inv_cs_y));
    let xi_end = i32(floor(tx * inv_cs_x));
    let yi_end = i32(floor(ty * inv_cs_y));

    let step_x = select(-1, 1, dx >= 0.0);
    let step_y = select(-1, 1, dy >= 0.0);

    // t_max_{x,y} = parametric ray distance to next voxel boundary on each
    // axis (t ∈ [0, 1] where 0 = origin, 1 = target). t_delta_{x,y} = the
    // increment after stepping one voxel along that axis.
    let big: f32 = 1.0e30;
    var t_max_x: f32 = big;
    var t_delta_x: f32 = big;
    if (abs(dx) > 1e-20) {
        let inv_dx = 1.0 / dx;
        let next_bx = (f32(xi) + select(0.0, 1.0, dx > 0.0)) * cs_x;
        t_max_x = (next_bx - ox) * inv_dx;
        t_delta_x = abs(cs_x * inv_dx);
    }
    var t_max_y: f32 = big;
    var t_delta_y: f32 = big;
    if (abs(dy) > 1e-20) {
        let inv_dy = 1.0 / dy;
        let next_by = (f32(yi) + select(0.0, 1.0, dy > 0.0)) * cs_y;
        t_max_y = (next_by - oy) * inv_dy;
        t_delta_y = abs(cs_y * inv_dy);
    }

    let max_steps = nx + ny + 2;
    for (var k: i32 = 0; k < max_steps; k = k + 1) {
        if (xi == xi_end && yi == yi_end) {
            return false;
        }
        if (t_max_x < t_max_y) {
            xi = xi + step_x;
            t_max_x = t_max_x + t_delta_x;
        } else {
            yi = yi + step_y;
            t_max_y = t_max_y + t_delta_y;
        }
        if (xi < 0 || xi >= nx || yi < 0 || yi >= ny) {
            return true;
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
    pheromone_ch1: f32,
    pheromone_ch2: f32,
    vibration: f32,
};

// Single-position sample of all five grids. xy is toroidal (modulo wrap); z
// out of range yields a zero triple (matches CPU SmellField sample-on-bounds).
// All fields share the same resolution + world_half (renderer/headless both
// allocate them on the same FieldGpu shape), so one addressing pass serves
// every read.
fn sample_all_at(pos: vec3<f32>, fc: FieldConsts) -> FieldSample {
    let zi = i32(floor((pos.z + params.field_world_half_z) * fc.inv_cell_z));
    if (zi < 0 || zi >= fc.nz) {
        return FieldSample(0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let xi_raw = i32(floor((pos.x + params.field_world_half_x) * fc.inv_cell_x));
    let yi_raw = i32(floor((pos.y + params.field_world_half_y) * fc.inv_cell_y));
    let xi = ((xi_raw % fc.nx) + fc.nx) % fc.nx;
    let yi = ((yi_raw % fc.ny) + fc.ny) % fc.ny;
    let idx = u32(zi * fc.nx * fc.ny + yi * fc.nx + xi);
    return FieldSample(
        bitcast<f32>(smell_grid[idx]),
        bitcast<f32>(pheromone_grid[idx]),
        bitcast<f32>(pheromone_grid_ch1[idx]),
        bitcast<f32>(pheromone_grid_ch2[idx]),
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
    // z is clamped (not toroidal) and GRID_NZ is tiny (4): iterate the
    // distinct bz planes directly. The old `dz in -r_cells..=r_cells` +
    // `clamp` rescanned boundary planes whenever r_cells reached a z edge
    // — wasted bucket walks plus a neighbors_count inflated by the rescan.
    let bz_lo = max(center.z - r_cells, 0);
    let bz_hi = min(center.z + r_cells, GRID_NZ - 1);
    // Corner-bucket cull: the (2·r_cells+1)³ cube circumscribes the vision
    // sphere, so ~⅓ of its buckets lie entirely beyond `vr`. A bucket at
    // offset (dx,dy,dz) is at least `max(|·|-1,0)` buckets away on each
    // axis from any point in the centre bucket — if that lower bound
    // exceeds `vr` the bucket cannot hold an in-vision cell or food, so it
    // is skipped. Conservative: never culls a reachable bucket → no
    // behaviour change.
    let vr_cells = vr / params.hash_cell_size;
    let vr_cells_2 = vr_cells * vr_cells;
    for (var bz = bz_lo; bz <= bz_hi; bz = bz + 1) {
        let z_term = f32(max(abs(bz - center.z) - 1, 0));
        let z_contrib = z_term * z_term;
        for (var dy = -r_cells; dy <= r_cells; dy = dy + 1) {
            let y_term = f32(max(abs(dy) - 1, 0));
            let zy_contrib = z_contrib + y_term * y_term;
            if (zy_contrib > vr_cells_2) {
                continue;
            }
            var by = center.y + dy;
            if (by < 0) { by = by + GRID_NY; }
            else if (by >= GRID_NY) { by = by - GRID_NY; }
            for (var dx = -r_cells; dx <= r_cells; dx = dx + 1) {
                let x_term = f32(max(abs(dx) - 1, 0));
                if (zy_contrib + x_term * x_term > vr_cells_2) {
                    continue;
                }
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
    let pheromone_grad_ch1 = vec3<f32>(
        (s_xp.pheromone_ch1 - s_xm.pheromone_ch1) * inv_2eps,
        (s_yp.pheromone_ch1 - s_ym.pheromone_ch1) * inv_2eps,
        (s_zp.pheromone_ch1 - s_zm.pheromone_ch1) * inv_2eps,
    );
    let pheromone_grad_ch2 = vec3<f32>(
        (s_xp.pheromone_ch2 - s_xm.pheromone_ch2) * inv_2eps,
        (s_yp.pheromone_ch2 - s_ym.pheromone_ch2) * inv_2eps,
        (s_zp.pheromone_ch2 - s_zm.pheromone_ch2) * inv_2eps,
    );
    let vibration_grad = vec3<f32>(
        (s_xp.vibration - s_xm.vibration) * inv_2eps,
        (s_yp.vibration - s_ym.vibration) * inv_2eps,
        (s_zp.vibration - s_zm.vibration) * inv_2eps,
    );
    let vibration_amp = s_center.vibration;

    // Wave 6: 6-direction whisker raycast in body frame. Mirror of
    // `ObstacleField::whisker_distances` — uniform-step samples from cell
    // position along [+forward, -forward, +right, -right, +up, -down],
    // returns normalized free-distance in [0, 1] (1 = clear, 0 = wall at
    // origin). Always writes (zeros when maze inactive) so the populate
    // shader can copy unconditionally.
    var whisker0: f32 = 1.0;
    var whisker1: f32 = 1.0;
    var whisker2: f32 = 1.0;
    var whisker3: f32 = 1.0;
    var whisker4: f32 = 1.0;
    var whisker5: f32 = 1.0;
    if (params.maze_active != 0u) {
        let heading = headings[i];
        let pitch = pitches[i];
        let cp = cos(pitch);
        let fwd = vec3<f32>(cos(heading) * cp, sin(heading) * cp, sin(pitch));
        let right = vec3<f32>(-sin(heading), cos(heading), 0.0);
        whisker0 = whisker_distance(pos_i, fwd);
        whisker1 = whisker_distance(pos_i, vec3<f32>(-fwd.x, -fwd.y, -fwd.z));
        whisker2 = whisker_distance(pos_i, right);
        whisker3 = whisker_distance(pos_i, vec3<f32>(-right.x, -right.y, -right.z));
        // ±z directions always clear (xy-only walls); skip raycast.
    }

    // Sprint 195: whisker spring-damper. The raycast value is the wall-
    // imposed target deflection (target = 1 - raw); each whisker is a driven
    // damped oscillator that overshoots / rings / settles. (defl, vel) state
    // persists per cell in `whisker_state` (12 f32/cell). Gated on
    // maze_active so non-maze runs keep the neutral 1.0 signal exactly.
    // Mirrors the CPU integrator in `World::update_whiskers`.
    var raw = array<f32, 6>(whisker0, whisker1, whisker2, whisker3, whisker4, whisker5);
    var sensed = raw;
    if (params.maze_active != 0u) {
        let whisker_stiffness: f32 = 360.0; // mirrors lib::WHISKER_STIFFNESS
        let whisker_damping: f32 = 11.0;    // mirrors lib::WHISKER_DAMPING
        let whisker_noise_amp: f32 = 0.03;  // mirrors lib::WHISKER_NOISE_AMPLITUDE
        let whisker_dt: f32 = 1.0 / 60.0;
        let wbase = i * 12u;
        for (var k: u32 = 0u; k < 6u; k = k + 1u) {
            let tgt = 1.0 - raw[k];
            var defl = whisker_state[wbase + k];
            var vel = whisker_state[wbase + 6u + k];
            let accel = whisker_stiffness * (tgt - defl) - whisker_damping * vel;
            vel = vel + accel * whisker_dt;
            defl = defl + vel * whisker_dt;
            whisker_state[wbase + k] = defl;
            whisker_state[wbase + 6u + k] = vel;
            let noise = whisker_noise(i, params.tick, k) * whisker_noise_amp;
            sensed[k] = clamp(clamp(1.0 - defl, 0.0, 1.0) + noise, 0.0, 1.0);
        }
    }

    let off = i * 31u;
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
    output[off + 19u] = sensed[0];
    output[off + 20u] = sensed[1];
    output[off + 21u] = sensed[2];
    output[off + 22u] = sensed[3];
    output[off + 23u] = sensed[4];
    output[off + 24u] = sensed[5];
    // Wave L: ch1/ch2 pheromone gradients. populate_inputs maps these to
    // brain inputs 21..26.
    output[off + 25u] = pheromone_grad_ch1.x;
    output[off + 26u] = pheromone_grad_ch1.y;
    output[off + 27u] = pheromone_grad_ch1.z;
    output[off + 28u] = pheromone_grad_ch2.x;
    output[off + 29u] = pheromone_grad_ch2.y;
    output[off + 30u] = pheromone_grad_ch2.z;
}