// GPU per-cell eat_food candidate selection — mirror of headless
// `World::eat_food` Pass 1 (parallel CPU bucket-walk + ellipsoid eat_test).
// Output is the per-cell `(food_idx, value)` candidate; CPU resolves the
// race (first-cell-wins per food + cluster food share + Hebbian reward)
// in a sequential post-pass.
//
// The shader iterates the food hash buckets in (-r_cells..=r_cells)³
// around the cell's position and returns the FIRST food whose center
// lies inside the cell's pose-rotated ellipsoid (eat_factor =
// EAT_RADIUS). Bucket walk order is deterministic across runs (same as
// `food_spawn.wgsl`'s cell-exclusion walk), so the per-cell first match
// is reproducible without any cross-cell coordination.
//
// Headless `eat_food` uses plain `Cell::eat_test` (ellipsoid only) — the
// spike-cone path (`eat_test_with_spikes`) isn't wired into either binary
// today, so the shader skips it to keep the per-thread cost down.

const GRID_NX: i32 = 64;
const GRID_NY: i32 = 32;
const GRID_NZ: i32 = 4;
const HALF_NX: i32 = 32;
const HALF_NY: i32 = 16;
const HALF_NZ: i32 = 2;

// Mirrors `bioscape::food::FoodKind`: Plant = 0, Carrion = 1.
const FOOD_KIND_PLANT: u32 = 0u;
const FOOD_KIND_CARRION: u32 = 1u;

struct EatFoodParams {
    num_cells: u32,
    num_foods: u32,
    cell_size: f32,
    eat_radius: f32,
    world_half_x: f32,
    world_half_y: f32,
    world_half_z: f32,
    world_map_nx: u32,
    world_map_ny: u32,
    world_map_nz: u32,
    fixed_timestep_hz: f32,
    plant_food_value: f32,
    carrion_food_value: f32,
    carrion_decay_per_sec: f32,
    world_map_food_floor: f32,
    world_map_food_amp: f32,
}

@group(0) @binding(0) var<uniform> params: EatFoodParams;
@group(0) @binding(1) var<storage, read> cell_positions: array<f32>;
@group(0) @binding(2) var<storage, read> cell_headings: array<f32>;
@group(0) @binding(3) var<storage, read> cell_pitches: array<f32>;
@group(0) @binding(4) var<storage, read> cell_body_dims: array<f32>;
@group(0) @binding(5) var<storage, read> cell_carnivore: array<f32>;
@group(0) @binding(6) var<storage, read> cell_max_axes: array<f32>;
@group(0) @binding(7) var<storage, read> food_positions: array<f32>;
@group(0) @binding(8) var<storage, read> food_kinds: array<u32>;
@group(0) @binding(9) var<storage, read> food_age_ticks: array<u32>;
@group(0) @binding(10) var<storage, read> hash_offsets: array<u32>;
@group(0) @binding(11) var<storage, read> hash_sorted: array<u32>;
@group(0) @binding(12) var<storage, read> world_map_field: array<f32>;
@group(0) @binding(13) var<storage, read_write> out_food_idx: array<u32>;
@group(0) @binding(14) var<storage, read_write> out_value: array<f32>;

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

// Nearest-bucket sample of WorldMap richness — mirrors `WorldMap::sample`.
fn sample_world_map(pos: vec3<f32>) -> f32 {
    let nx = i32(params.world_map_nx);
    let ny = i32(params.world_map_ny);
    let nz = i32(params.world_map_nz);
    let cs_x = 2.0 * params.world_half_x / f32(nx);
    let cs_y = 2.0 * params.world_half_y / f32(ny);
    let cs_z = 2.0 * params.world_half_z / f32(nz);
    let xi_raw = i32(floor((pos.x + params.world_half_x) / cs_x));
    let yi_raw = i32(floor((pos.y + params.world_half_y) / cs_y));
    let zi_raw = i32(floor((pos.z + params.world_half_z) / cs_z));
    let xi = ((xi_raw % nx) + nx) % nx;
    let yi = ((yi_raw % ny) + ny) % ny;
    let zi = clamp(zi_raw, 0, nz - 1);
    let idx = u32(zi) * u32(nx) * u32(ny) + u32(yi) * u32(nx) + u32(xi);
    return world_map_field[idx];
}

fn food_base_value(kind: u32) -> f32 {
    if (kind == FOOD_KIND_CARRION) {
        return params.carrion_food_value;
    }
    return params.plant_food_value;
}

fn eat_efficiency(kind: u32, carnivore_score: f32) -> f32 {
    let s = clamp(carnivore_score, 0.0, 1.0);
    if (kind == FOOD_KIND_CARRION) {
        return 0.5;
    }
    return 1.0 - s;
}

@compute @workgroup_size(64)
fn eat_food(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }
    // Default: no match. Sentinel `num_foods` (one past last valid idx)
    // is treated by the CPU race-resolution as "this cell ate nothing".
    out_food_idx[i] = params.num_foods;
    out_value[i] = 0.0;
    if (params.num_foods == 0u) {
        return;
    }

    let pos = vec3<f32>(
        cell_positions[i * 3u + 0u],
        cell_positions[i * 3u + 1u],
        cell_positions[i * 3u + 2u],
    );
    let heading = cell_headings[i];
    let pitch = cell_pitches[i];
    let body = vec3<f32>(
        cell_body_dims[i * 3u + 0u],
        cell_body_dims[i * 3u + 1u],
        cell_body_dims[i * 3u + 2u],
    );
    let carnivore = cell_carnivore[i];
    let max_axis = cell_max_axes[i];

    // Body frame (forward + right + up). Mirror of `bioscape::body_basis`.
    let cp = cos(pitch);
    let fwd = vec3<f32>(cos(heading) * cp, sin(heading) * cp, sin(pitch));
    let right = vec3<f32>(-sin(heading), cos(heading), 0.0);
    let up = vec3<f32>(
        fwd.y * right.z - fwd.z * right.y,
        fwd.z * right.x - fwd.x * right.z,
        fwd.x * right.y - fwd.y * right.x,
    );

    // Ellipsoid semi-axes (positive, EPSILON guarded). `eat_factor`
    // multiplies each body dim — matches `Cell::eat_test(food, EAT_RADIUS)`.
    let eat_f = params.eat_radius;
    let l = max(body.x * eat_f, 1e-9);
    let w = max(body.y * eat_f, 1e-9);
    let h = max(body.z * eat_f, 1e-9);

    let search_r = eat_f * max_axis;
    let r_cells = i32(ceil(search_r / params.cell_size));
    let center = bucket_coords_of(pos);
    var found: bool = false;

    for (var dz: i32 = -r_cells; dz <= r_cells; dz = dz + 1) {
        if (found) { break; }
        let bz_raw = center.z + dz;
        if (bz_raw < 0 || bz_raw >= GRID_NZ) {
            continue;
        }
        let bz = bz_raw;
        for (var dy: i32 = -r_cells; dy <= r_cells; dy = dy + 1) {
            if (found) { break; }
            var by = center.y + dy;
            if (by < 0) { by = by + GRID_NY; }
            else if (by >= GRID_NY) { by = by - GRID_NY; }
            for (var dx: i32 = -r_cells; dx <= r_cells; dx = dx + 1) {
                if (found) { break; }
                var bx = center.x + dx;
                if (bx < 0) { bx = bx + GRID_NX; }
                else if (bx >= GRID_NX) { bx = bx - GRID_NX; }
                let b = u32(bx + by * GRID_NX + bz * GRID_NX * GRID_NY);
                let start = hash_offsets[b];
                let end = hash_offsets[b + 1u];
                for (var k = start; k < end; k = k + 1u) {
                    if (found) { break; }
                    let f = hash_sorted[k];
                    if (f >= params.num_foods) {
                        continue;
                    }
                    let fp = vec3<f32>(
                        food_positions[f * 3u + 0u],
                        food_positions[f * 3u + 1u],
                        food_positions[f * 3u + 2u],
                    );
                    // Toroidal min-image: ghost food position in the same
                    // wraparound neighbourhood as `pos`.
                    let mdx = min_image_xy(fp.x - pos.x, params.world_half_x);
                    let mdy = min_image_xy(fp.y - pos.y, params.world_half_y);
                    let mdz = fp.z - pos.z;
                    // Body-frame projection. CPU `eat_test_pose` builds
                    // `d_par, d_right, d_up` and runs the ellipsoid form.
                    let dp = mdx * fwd.x + mdy * fwd.y + mdz * fwd.z;
                    let dr = mdx * right.x + mdy * right.y + mdz * right.z;
                    let du = mdx * up.x + mdy * up.y + mdz * up.z;
                    let inside =
                        (dp / l) * (dp / l)
                            + (dr / w) * (dr / w)
                            + (du / h) * (du / h)
                            <= 1.0;
                    if (!inside) {
                        continue;
                    }
                    // First match wins. Compute value with world-map
                    // richness multiplier + age decay + eat_efficiency.
                    let kind = food_kinds[f];
                    let age_sec = f32(food_age_ticks[f]) / params.fixed_timestep_hz;
                    let value_factor = max(
                        1.0 - params.carrion_decay_per_sec * age_sec,
                        0.0,
                    );
                    let richness = sample_world_map(
                        vec3<f32>(fp.x, fp.y, 0.0),
                    );
                    let multiplier = params.world_map_food_floor
                        + params.world_map_food_amp * richness;
                    let eff = eat_efficiency(kind, carnivore);
                    let value = food_base_value(kind) * multiplier * value_factor * eff;
                    out_food_idx[i] = f;
                    out_value[i] = value;
                    found = true;
                }
            }
        }
    }
}
