// Sprint 50: GPU mirror headless::predate. 2 passes:
//   herd_count — per cell, count neighbors v HERD_RADIUS (write-only per i).
//   attack     — per attacker s `attack > threshold`, find smaller cells v
//                pair_r, compute gain (size + spike + dilution), atomic-add
//                drain do victim energy_delta + damage_delta. Self-gain
//                accumulated lokálně, atomic-add na konci.
//
// Atomic float add: WGSL CAS loop přes `atomicCompareExchangeWeak` +
// `bitcast<u32>` ↔ `bitcast<f32>`. Storage pointer ne-passable do funkce
// (Naga validation), takže CAS je inlined per call site (3× v `attack`).

const GRID_NX: i32 = 64;
const GRID_NY: i32 = 32;
const GRID_NZ: i32 = 4;
const HALF_NX: i32 = 32;
const HALF_NY: i32 = 16;
const HALF_NZ: i32 = 2;

struct PredateParams {
    num_cells: u32,
    cell_size: f32,
    cell_radius_const: f32,
    size_ratio_threshold: f32,
    herd_radius_sq: f32,
    attack_threshold: f32,
    predation_gain: f32,
    predation_drain: f32,
    spike_dot_threshold: f32,
    spike_bonus: f32,
    dilution_k: f32,
    _pad0: u32,
    /// Sprint 55: toroidal bounds.
    world_half_x: f32,
    world_half_y: f32,
    _pad1: u32,
    _pad2: u32,
}

fn min_image_xy(d: f32, half: f32) -> f32 {
    let w = 2.0 * half;
    if (d > half) { return d - w; }
    if (d < -half) { return d + w; }
    return d;
}

fn bucket_id_wrapped(pos: vec3<f32>) -> u32 {
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

@group(0) @binding(0) var<uniform> params: PredateParams;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> eff_radii: array<f32>;
@group(0) @binding(3) var<storage, read> headings: array<f32>;
@group(0) @binding(4) var<storage, read> spike_lengths: array<f32>;
@group(0) @binding(5) var<storage, read> attack_signals: array<f32>;
@group(0) @binding(6) var<storage, read> hash_offsets: array<u32>;
@group(0) @binding(7) var<storage, read> hash_sorted: array<u32>;
@group(0) @binding(8) var<storage, read_write> herd_counts: array<u32>;
@group(0) @binding(9) var<storage, read_write> energy_delta: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> damage_delta: array<atomic<u32>>;

@compute @workgroup_size(64)
fn herd_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }
    let pos_i = vec3<f32>(
        positions[i * 3u + 0u],
        positions[i * 3u + 1u],
        positions[i * 3u + 2u],
    );
    let herd_r = sqrt(params.herd_radius_sq);
    let r_cells = i32(ceil(herd_r / params.cell_size));
    let cs = params.cell_size;
    var count: u32 = 0u;
    // Sprint 55: ghost positions + bucket_id_wrapped + min-image distance.
    for (var dx = -r_cells; dx <= r_cells; dx = dx + 1) {
        for (var dy = -r_cells; dy <= r_cells; dy = dy + 1) {
            for (var dz = -r_cells; dz <= r_cells; dz = dz + 1) {
                let nbr_pos = vec3<f32>(
                    pos_i.x + f32(dx) * cs,
                    pos_i.y + f32(dy) * cs,
                    pos_i.z + f32(dz) * cs,
                );
                let b = bucket_id_wrapped(nbr_pos);
                let start = hash_offsets[b];
                let end = hash_offsets[b + 1u];
                for (var k = start; k < end; k = k + 1u) {
                    let j = hash_sorted[k];
                    if (j == i) {
                        continue;
                    }
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
                    if (dot(d, d) < params.herd_radius_sq) {
                        count = count + 1u;
                    }
                }
            }
        }
    }
    herd_counts[i] = count;
}

@compute @workgroup_size(64)
fn attack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }
    let attack_strength = max(attack_signals[i], 0.0);
    if (attack_strength <= params.attack_threshold) {
        return;
    }
    let pos_i = vec3<f32>(
        positions[i * 3u + 0u],
        positions[i * 3u + 1u],
        positions[i * 3u + 2u],
    );
    let r_i = eff_radii[i];
    let spike = spike_lengths[i];
    let heading = headings[i];

    // Search radius: pair_r = CELL_RADIUS × (r_i + r_j). Předpoklad r_j ≤ r_i
    // (size ratio threshold předtím), tedy max pair_r = 2 × CELL_RADIUS × r_i.
    let max_pair_r = 2.0 * params.cell_radius_const * r_i;
    let r_cells = i32(ceil(max_pair_r / params.cell_size));
    let cs = params.cell_size;

    var self_gain: f32 = 0.0;

    // Sprint 55: ghost positions + bucket_id_wrapped + min-image distance.
    for (var dx = -r_cells; dx <= r_cells; dx = dx + 1) {
        for (var dy = -r_cells; dy <= r_cells; dy = dy + 1) {
            for (var dz = -r_cells; dz <= r_cells; dz = dz + 1) {
                let nbr_pos = vec3<f32>(
                    pos_i.x + f32(dx) * cs,
                    pos_i.y + f32(dy) * cs,
                    pos_i.z + f32(dz) * cs,
                );
                let b = bucket_id_wrapped(nbr_pos);
                let start = hash_offsets[b];
                let end = hash_offsets[b + 1u];
                for (var k = start; k < end; k = k + 1u) {
                    let j = hash_sorted[k];
                    if (j == i) {
                        continue;
                    }
                    let r_j = eff_radii[j];
                    if (r_i < params.size_ratio_threshold * r_j) {
                        continue;
                    }
                    let pair_r = params.cell_radius_const * (r_i + r_j);
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
                    if (d2 < pair_r2) {
                        var gain = params.predation_gain;
                        if (spike > 0.0 && d2 > 0.0) {
                            let inv_d = 1.0 / sqrt(d2);
                            let to_j_x = -d.x * inv_d;
                            let to_j_y = -d.y * inv_d;
                            let cos_angle = cos(heading) * to_j_x + sin(heading) * to_j_y;
                            if (cos_angle >= params.spike_dot_threshold) {
                                gain = gain + params.predation_gain * spike * params.spike_bonus;
                            }
                        }
                        let n_neigh = f32(herd_counts[j]);
                        let dilution = 1.0 / (1.0 + params.dilution_k * n_neigh);
                        gain = gain * dilution;
                        self_gain = self_gain + gain;

                        // Atomic add -drain to energy_delta[j].
                        var old_e: u32 = atomicLoad(&energy_delta[j]);
                        loop {
                            let new_e: u32 =
                                bitcast<u32>(bitcast<f32>(old_e) - params.predation_drain);
                            let r = atomicCompareExchangeWeak(&energy_delta[j], old_e, new_e);
                            if (r.exchanged) {
                                break;
                            }
                            old_e = r.old_value;
                        }
                        // Atomic add +drain to damage_delta[j].
                        var old_d: u32 = atomicLoad(&damage_delta[j]);
                        loop {
                            let new_d: u32 =
                                bitcast<u32>(bitcast<f32>(old_d) + params.predation_drain);
                            let r = atomicCompareExchangeWeak(&damage_delta[j], old_d, new_d);
                            if (r.exchanged) {
                                break;
                            }
                            old_d = r.old_value;
                        }
                    }
                }
            }
        }
    }

    // Self-gain atomic add (cell i mohl být zároveň target jiného attacker).
    if (self_gain != 0.0) {
        var old_s: u32 = atomicLoad(&energy_delta[i]);
        loop {
            let new_s: u32 = bitcast<u32>(bitcast<f32>(old_s) + self_gain);
            let r = atomicCompareExchangeWeak(&energy_delta[i], old_s, new_s);
            if (r.exchanged) {
                break;
            }
            old_s = r.old_value;
        }
    }
}
