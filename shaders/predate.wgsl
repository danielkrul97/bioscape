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
    let bx_base = i32(floor(pos_i.x / params.cell_size)) + HALF_NX;
    let by_base = i32(floor(pos_i.y / params.cell_size)) + HALF_NY;
    let bz_base = i32(floor(pos_i.z / params.cell_size)) + HALF_NZ;
    var count: u32 = 0u;
    for (var dx = -r_cells; dx <= r_cells; dx = dx + 1) {
        for (var dy = -r_cells; dy <= r_cells; dy = dy + 1) {
            for (var dz = -r_cells; dz <= r_cells; dz = dz + 1) {
                let bx = clamp(bx_base + dx, 0, GRID_NX - 1);
                let by = clamp(by_base + dy, 0, GRID_NY - 1);
                let bz = clamp(bz_base + dz, 0, GRID_NZ - 1);
                let b = u32(bx + by * GRID_NX + bz * GRID_NX * GRID_NY);
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
                    let d = pos_i - pj;
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

    let bx_base = i32(floor(pos_i.x / params.cell_size)) + HALF_NX;
    let by_base = i32(floor(pos_i.y / params.cell_size)) + HALF_NY;
    let bz_base = i32(floor(pos_i.z / params.cell_size)) + HALF_NZ;

    var self_gain: f32 = 0.0;

    for (var dx = -r_cells; dx <= r_cells; dx = dx + 1) {
        for (var dy = -r_cells; dy <= r_cells; dy = dy + 1) {
            for (var dz = -r_cells; dz <= r_cells; dz = dz + 1) {
                let bx = clamp(bx_base + dx, 0, GRID_NX - 1);
                let by = clamp(by_base + dy, 0, GRID_NY - 1);
                let bz = clamp(bz_base + dz, 0, GRID_NZ - 1);
                let b = u32(bx + by * GRID_NX + bz * GRID_NX * GRID_NY);
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
                    let d = pos_i - pj;
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
