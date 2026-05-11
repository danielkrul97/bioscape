// Wave J: GPU food rejection sampling. Mirrors `World::spawn_food` per
// attempt: uniform random pos in world bounds → reject by richness-weighted
// probability → optionally reject by obstacle mask → reject if too close
// to an existing cell. K parallel attempts; CPU picks valid ones up to the
// per-tick budget. Per-thread xoshiro128++ state mutates in-place so
// successive ticks see fresh streams.

const GRID_NX: i32 = 64;
const GRID_NY: i32 = 32;
const GRID_NZ: i32 = 4;
const HALF_NX: i32 = 32;
const HALF_NY: i32 = 16;
const HALF_NZ: i32 = 2;

struct FoodSpawnParams {
    num_attempts: u32,
    rejection_strength: f32,
    eat_radius: f32,
    cell_size: f32,
    world_half_x: f32,
    world_half_y: f32,
    world_half_z: f32,
    num_cells: u32,
    world_map_nx: u32,
    world_map_ny: u32,
    world_map_nz: u32,
    obstacle_active: u32,
    obstacle_nx: u32,
    obstacle_ny: u32,
    obstacle_nz: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: FoodSpawnParams;
@group(0) @binding(1) var<storage, read_write> xoshiro_state: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> world_map_field: array<f32>;
@group(0) @binding(3) var<storage, read> obstacle_mask: array<u32>;
@group(0) @binding(4) var<storage, read> cell_positions: array<f32>;
@group(0) @binding(5) var<storage, read> cell_max_axes: array<f32>;
@group(0) @binding(6) var<storage, read> hash_offsets: array<u32>;
@group(0) @binding(7) var<storage, read> hash_sorted: array<u32>;
@group(0) @binding(8) var<storage, read_write> candidate_positions: array<f32>;
@group(0) @binding(9) var<storage, read_write> valid_mask: array<u32>;

fn rotl_u32(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

fn xoshiro_next(state: ptr<function, vec4<u32>>) -> u32 {
    let s = *state;
    let result = rotl_u32(s.x + s.w, 7u) + s.x;
    let t = s.y << 9u;
    var s2 = s.z ^ s.x;
    var s3 = s.w ^ s.y;
    let s1 = s.y ^ s2;
    let s0 = s.x ^ s3;
    s2 = s2 ^ t;
    s3 = rotl_u32(s3, 11u);
    *state = vec4<u32>(s0, s1, s2, s3);
    return result;
}

fn uniform01(state: ptr<function, vec4<u32>>) -> f32 {
    let bits = xoshiro_next(state);
    return bitcast<f32>((bits >> 9u) | 0x3F800000u) - 1.0;
}

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

// Nearest-bucket sample matching `WorldMap::sample` on CPU.
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
    // xy toroidal wrap, z clamp.
    let xi = ((xi_raw % nx) + nx) % nx;
    let yi = ((yi_raw % ny) + ny) % ny;
    let zi = clamp(zi_raw, 0, nz - 1);
    let idx = u32(zi) * u32(nx) * u32(ny) + u32(yi) * u32(nx) + u32(xi);
    return world_map_field[idx];
}

// Same bucket geometry as `ObstacleField::sample` (xy toroidal, z clamp).
// Returns true when the bucket is marked occupied (1u in the mask).
fn sample_obstacle(pos: vec3<f32>) -> bool {
    let nx = i32(params.obstacle_nx);
    let ny = i32(params.obstacle_ny);
    let nz = i32(params.obstacle_nz);
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
    return obstacle_mask[idx] != 0u;
}

@compute @workgroup_size(64)
fn food_spawn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_attempts) {
        return;
    }
    var s = xoshiro_state[i];

    // Uniform pos in [-half, half)^3.
    let rx = uniform01(&s) * 2.0 * params.world_half_x - params.world_half_x;
    let ry = uniform01(&s) * 2.0 * params.world_half_y - params.world_half_y;
    let rz = uniform01(&s) * 2.0 * params.world_half_z - params.world_half_z;
    let pos = vec3<f32>(rx, ry, rz);

    candidate_positions[i * 3u + 0u] = rx;
    candidate_positions[i * 3u + 1u] = ry;
    candidate_positions[i * 3u + 2u] = rz;
    valid_mask[i] = 0u;

    // Richness-weighted rejection (probability ∝ FOOD_REJECTION_STRENGTH × (1 - richness)).
    let richness = clamp(sample_world_map(pos), 0.0, 1.0);
    let rej_threshold = params.rejection_strength * (1.0 - richness);
    let rej_draw = uniform01(&s);
    if (rej_draw < rej_threshold) {
        xoshiro_state[i] = s;
        return;
    }

    // Obstacle mask rejection (only when maze active).
    if (params.obstacle_active != 0u && sample_obstacle(pos)) {
        xoshiro_state[i] = s;
        return;
    }

    // Cell exclusion: any existing cell within EAT_RADIUS × max_axis disqualifies.
    // CPU helper uses `EAT_RADIUS × MAX_BODY_LENGTH` as the conservative search
    // radius, then narrow-phases against per-cell `max_axis`.
    let search_r = params.eat_radius * 4.0; // MAX_BODY_LENGTH = 4.0 in params/morphology.rs
    let cs = params.cell_size;
    let r_cells = i32(ceil(search_r / cs));
    let center = bucket_coords_of(pos);
    var blocked: bool = false;
    for (var dz = -r_cells; dz <= r_cells; dz = dz + 1) {
        if (blocked) { break; }
        let bz = clamp(center.z + dz, 0, GRID_NZ - 1);
        for (var dy = -r_cells; dy <= r_cells; dy = dy + 1) {
            if (blocked) { break; }
            var by = center.y + dy;
            if (by < 0) { by = by + GRID_NY; }
            else if (by >= GRID_NY) { by = by - GRID_NY; }
            for (var dx = -r_cells; dx <= r_cells; dx = dx + 1) {
                if (blocked) { break; }
                var bx = center.x + dx;
                if (bx < 0) { bx = bx + GRID_NX; }
                else if (bx >= GRID_NX) { bx = bx - GRID_NX; }
                let b = u32(bx + by * GRID_NX + bz * GRID_NX * GRID_NY);
                let start = hash_offsets[b];
                let end = hash_offsets[b + 1u];
                for (var k = start; k < end; k = k + 1u) {
                    let cell_idx = hash_sorted[k];
                    if (cell_idx >= params.num_cells) {
                        continue;
                    }
                    let exclusion = params.eat_radius * cell_max_axes[cell_idx];
                    let exclusion2 = exclusion * exclusion;
                    let cp = vec3<f32>(
                        cell_positions[cell_idx * 3u + 0u],
                        cell_positions[cell_idx * 3u + 1u],
                        cell_positions[cell_idx * 3u + 2u],
                    );
                    let d = vec3<f32>(
                        min_image_xy(pos.x - cp.x, params.world_half_x),
                        min_image_xy(pos.y - cp.y, params.world_half_y),
                        pos.z - cp.z,
                    );
                    if (dot(d, d) < exclusion2) {
                        blocked = true;
                        break;
                    }
                }
            }
        }
    }
    xoshiro_state[i] = s;
    if (!blocked) {
        valid_mask[i] = 1u;
    }
}
