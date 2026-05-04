// Sprint 50: GPU mirror celého sensor gather kroku z `World::brain_act`
// (headless). Per cell:
//   - cell_hash query → nearest cell + neighbors_in_vision
//   - food_hash query → nearest food
//   - sample smell + pheromone fields (finite-diff gradient at pos.xy)
//
// Konzumuje výstupy:
//   SpatialHashGpu (cells) — Sprint 45
//   SpatialHashGpu (foods) — separate instance
//   FieldGpu × 2 (smell, pheromone) — Sprint 46, čteme grid jako array<u32>
//                                     a bitcastujeme na f32.
//
// Output layout per cell (15 f32, stride 15 — Sprint 56 přidal z-složky gradientů):
//   [0..3]   nearest_food.dx,dy,dz   (0 pokud has_food == 0)
//   [3]      has_food (0.0 / 1.0)
//   [4..7]   nearest_cell.dx,dy,dz
//   [7]      nearest_cell radius (-1.0 = no cell — sentinel)
//   [8..11]  smell_grad.x, .y, .z
//   [11..14] pheromone_grad.x, .y, .z
//   [14]     neighbors_in_vision count, bitcast<f32>(u32)

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
    // Sprint 56: 3D field params (per-axis resolution + bounds).
    field_res_x: u32,
    field_res_y: u32,
    field_res_z: u32,
    field_eps: f32,
    field_world_half_x: f32,
    field_world_half_y: f32,
    field_world_half_z: f32,
    _pad0: u32,
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

// Sprint 55: hash bucket wrap (toroidal).
fn bucket_id_wrapped(pos: vec3<f32>) -> u32 {
    let wx = 2.0 * params.world_half_x;
    let wy = 2.0 * params.world_half_y;
    let pos_wx = pos.x - floor((pos.x + params.world_half_x) / wx) * wx;
    let pos_wy = pos.y - floor((pos.y + params.world_half_y) / wy) * wy;
    let bx = i32(floor(pos_wx / params.hash_cell_size)) + HALF_NX;
    let by = i32(floor(pos_wy / params.hash_cell_size)) + HALF_NY;
    let bz = i32(floor(pos.z / params.hash_cell_size)) + HALF_NZ;
    let bx_c = clamp(bx, 0, GRID_NX - 1);
    let by_c = clamp(by, 0, GRID_NY - 1);
    let bz_c = clamp(bz, 0, GRID_NZ - 1);
    return u32(bx_c + by_c * GRID_NX + bz_c * GRID_NX * GRID_NY);
}

fn min_image_xy(d: f32, half: f32) -> f32 {
    let w = 2.0 * half;
    if (d > half) { return d - w; }
    if (d < -half) { return d + w; }
    return d;
}

// Sprint 56: 3D field sample. xy modulo wrap (toroidal), z out-of-range → 0.
fn sample_field_3d(grid_kind: u32, pos: vec3<f32>) -> f32 {
    let nx = i32(params.field_res_x);
    let ny = i32(params.field_res_y);
    let nz = i32(params.field_res_z);
    let cell_x = (2.0 * params.field_world_half_x) / f32(nx);
    let cell_y = (2.0 * params.field_world_half_y) / f32(ny);
    let cell_z = (2.0 * params.field_world_half_z) / f32(nz);
    let zi = i32(floor((pos.z + params.field_world_half_z) / cell_z));
    if (zi < 0 || zi >= nz) {
        return 0.0;
    }
    let xi_raw = i32(floor((pos.x + params.field_world_half_x) / cell_x));
    let yi_raw = i32(floor((pos.y + params.field_world_half_y) / cell_y));
    let xi = ((xi_raw % nx) + nx) % nx;
    let yi = ((yi_raw % ny) + ny) % ny;
    let idx = u32(zi * nx * ny + yi * nx + xi);
    if (grid_kind == 0u) {
        return bitcast<f32>(smell_grid[idx]);
    }
    return bitcast<f32>(pheromone_grid[idx]);
}

fn gradient_at_3d(grid_kind: u32, pos: vec3<f32>) -> vec3<f32> {
    let eps = params.field_eps;
    let f_xp = sample_field_3d(grid_kind, vec3<f32>(pos.x + eps, pos.y, pos.z));
    let f_xm = sample_field_3d(grid_kind, vec3<f32>(pos.x - eps, pos.y, pos.z));
    let f_yp = sample_field_3d(grid_kind, vec3<f32>(pos.x, pos.y + eps, pos.z));
    let f_ym = sample_field_3d(grid_kind, vec3<f32>(pos.x, pos.y - eps, pos.z));
    let f_zp = sample_field_3d(grid_kind, vec3<f32>(pos.x, pos.y, pos.z + eps));
    let f_zm = sample_field_3d(grid_kind, vec3<f32>(pos.x, pos.y, pos.z - eps));
    let inv = 1.0 / (2.0 * eps);
    return vec3<f32>(
        (f_xp - f_xm) * inv,
        (f_yp - f_ym) * inv,
        (f_zp - f_zm) * inv,
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
    let r_cells = i32(ceil(vr / params.hash_cell_size));
    let cs = params.hash_cell_size;

    // Sprint 56: cell broad-phase přes ghost positions + bucket_id_wrapped
    // (toroidal). Narrow-phase min-image distance.
    var best_cell_d2 = vr2 + 1.0;
    var best_cell_dx: f32 = 0.0;
    var best_cell_dy: f32 = 0.0;
    var best_cell_dz: f32 = 0.0;
    var best_cell_radius: f32 = -1.0;
    var neighbors_count: u32 = 0u;
    for (var dx = -r_cells; dx <= r_cells; dx = dx + 1) {
        for (var dy = -r_cells; dy <= r_cells; dy = dy + 1) {
            for (var dz = -r_cells; dz <= r_cells; dz = dz + 1) {
                let nbr_pos = vec3<f32>(
                    pos_i.x + f32(dx) * cs,
                    pos_i.y + f32(dy) * cs,
                    pos_i.z + f32(dz) * cs,
                );
                let b = bucket_id_wrapped(nbr_pos);
                let start = cell_hash_offsets[b];
                let end = cell_hash_offsets[b + 1u];
                for (var k = start; k < end; k = k + 1u) {
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
            }
        }
    }

    // Food broad-phase: nearest food (also toroidal).
    var best_food_d2 = vr2 + 1.0;
    var best_food_dx: f32 = 0.0;
    var best_food_dy: f32 = 0.0;
    var best_food_dz: f32 = 0.0;
    var has_food: f32 = 0.0;
    if (params.num_foods > 0u) {
        for (var dx = -r_cells; dx <= r_cells; dx = dx + 1) {
            for (var dy = -r_cells; dy <= r_cells; dy = dy + 1) {
                for (var dz = -r_cells; dz <= r_cells; dz = dz + 1) {
                    let nbr_pos = vec3<f32>(
                        pos_i.x + f32(dx) * cs,
                        pos_i.y + f32(dy) * cs,
                        pos_i.z + f32(dz) * cs,
                    );
                    let b = bucket_id_wrapped(nbr_pos);
                    let start = food_hash_offsets[b];
                    let end = food_hash_offsets[b + 1u];
                    for (var k = start; k < end; k = k + 1u) {
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

    // Sprint 56: 3D field gradient samples (xy wrapped via sample_field_3d).
    let smell_grad = gradient_at_3d(0u, pos_i);
    let pheromone_grad = gradient_at_3d(1u, pos_i);

    // Pack output (stride 15: + smell_grad.z + pheromone_grad.z).
    let off = i * 15u;
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
}
