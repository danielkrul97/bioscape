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
// Output layout per cell (13 f32, stride 13):
//   [0..3]  nearest_food.dx,dy,dz   (0 pokud has_food == 0)
//   [3]     has_food (0.0 / 1.0)
//   [4..7]  nearest_cell.dx,dy,dz
//   [7]     nearest_cell radius (-1.0 = no cell — sentinel)
//   [8..10] smell_grad.x, smell_grad.y
//   [10..12] pheromone_grad.x, pheromone_grad.y
//   [12]    neighbors_in_vision count, bitcast<f32>(u32)

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
    field_resolution: u32,
    field_eps: f32,
    field_world_half_x: f32,
    field_world_half_y: f32,
    _pad0: u32,
    _pad1: u32,
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

fn bucket_xyz(pos: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(
        i32(floor(pos.x / params.hash_cell_size)) + HALF_NX,
        i32(floor(pos.y / params.hash_cell_size)) + HALF_NY,
        i32(floor(pos.z / params.hash_cell_size)) + HALF_NZ,
    );
}

fn sample_field(grid_kind: u32, pos_x: f32, pos_y: f32) -> f32 {
    let res = i32(params.field_resolution);
    let cell_x = (2.0 * params.field_world_half_x) / f32(res);
    let cell_y = (2.0 * params.field_world_half_y) / f32(res);
    let xi = i32(floor((pos_x + params.field_world_half_x) / cell_x));
    let yi = i32(floor((pos_y + params.field_world_half_y) / cell_y));
    if (xi < 0 || xi >= res || yi < 0 || yi >= res) {
        return 0.0;
    }
    let idx = u32(yi * res + xi);
    if (grid_kind == 0u) {
        return bitcast<f32>(smell_grid[idx]);
    }
    return bitcast<f32>(pheromone_grid[idx]);
}

fn gradient_at(grid_kind: u32, pos_x: f32, pos_y: f32) -> vec2<f32> {
    let eps = params.field_eps;
    let f_xp = sample_field(grid_kind, pos_x + eps, pos_y);
    let f_xm = sample_field(grid_kind, pos_x - eps, pos_y);
    let f_yp = sample_field(grid_kind, pos_x, pos_y + eps);
    let f_ym = sample_field(grid_kind, pos_x, pos_y - eps);
    let inv = 1.0 / (2.0 * eps);
    return vec2<f32>((f_xp - f_xm) * inv, (f_yp - f_ym) * inv);
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
    let base = bucket_xyz(pos_i);

    // Cell broad-phase: nearest cell + count.
    var best_cell_d2 = vr2 + 1.0;
    var best_cell_dx: f32 = 0.0;
    var best_cell_dy: f32 = 0.0;
    var best_cell_dz: f32 = 0.0;
    var best_cell_radius: f32 = -1.0;
    var neighbors_count: u32 = 0u;
    for (var dx = -r_cells; dx <= r_cells; dx = dx + 1) {
        for (var dy = -r_cells; dy <= r_cells; dy = dy + 1) {
            for (var dz = -r_cells; dz <= r_cells; dz = dz + 1) {
                let bx = clamp(base.x + dx, 0, GRID_NX - 1);
                let by = clamp(base.y + dy, 0, GRID_NY - 1);
                let bz = clamp(base.z + dz, 0, GRID_NZ - 1);
                let b = u32(bx + by * GRID_NX + bz * GRID_NX * GRID_NY);
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
                    let dxf = pj.x - pos_i.x;
                    let dyf = pj.y - pos_i.y;
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

    // Food broad-phase: nearest food.
    var best_food_d2 = vr2 + 1.0;
    var best_food_dx: f32 = 0.0;
    var best_food_dy: f32 = 0.0;
    var best_food_dz: f32 = 0.0;
    var has_food: f32 = 0.0;
    if (params.num_foods > 0u) {
        for (var dx = -r_cells; dx <= r_cells; dx = dx + 1) {
            for (var dy = -r_cells; dy <= r_cells; dy = dy + 1) {
                for (var dz = -r_cells; dz <= r_cells; dz = dz + 1) {
                    let bx = clamp(base.x + dx, 0, GRID_NX - 1);
                    let by = clamp(base.y + dy, 0, GRID_NY - 1);
                    let bz = clamp(base.z + dz, 0, GRID_NZ - 1);
                    let b = u32(bx + by * GRID_NX + bz * GRID_NX * GRID_NY);
                    let start = food_hash_offsets[b];
                    let end = food_hash_offsets[b + 1u];
                    for (var k = start; k < end; k = k + 1u) {
                        let f = food_hash_sorted[k];
                        let pf = vec3<f32>(
                            food_positions[f * 3u + 0u],
                            food_positions[f * 3u + 1u],
                            food_positions[f * 3u + 2u],
                        );
                        let dxf = pf.x - pos_i.x;
                        let dyf = pf.y - pos_i.y;
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

    // Field gradient samples.
    let smell_grad = gradient_at(0u, pos_i.x, pos_i.y);
    let pheromone_grad = gradient_at(1u, pos_i.x, pos_i.y);

    // Pack output (stride 13).
    let off = i * 13u;
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
    output[off + 10u] = pheromone_grad.x;
    output[off + 11u] = pheromone_grad.y;
    output[off + 12u] = bitcast<f32>(neighbors_count);
}
