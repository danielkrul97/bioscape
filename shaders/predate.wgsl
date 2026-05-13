// GPU mirror of `headless::predate` — two compute kernels:
//   herd_count — per cell, count neighbors within HERD_RADIUS (write-only).
//   attack     — per attacker (with attack > threshold), find smaller cells
//                inside pair_r, compute gain (size + multi-spike cone +
//                herd-density dilution), atomic-add drain to victim's
//                energy_delta and damage_delta. Self-gain accumulates
//                locally, single atomic-add at the end.
//
// `spikes_packed` per cell = 5 × vec4 (length, azim, elev, complexity);
// `spike_counts[i]` u32 marks how many slots are active.
//
// Atomic float-add is implemented via a CAS loop over `atomicCompareExchangeWeak`
// with `bitcast` between u32 / f32 storage. Naga rejects passing storage
// pointers into functions, so the CAS is inlined at each call site rather
// than factored into a helper.

const GRID_NX: i32 = 64;
const GRID_NY: i32 = 32;
const GRID_NZ: i32 = 4;
const HALF_NX: i32 = 32;
const HALF_NY: i32 = 16;
const HALF_NZ: i32 = 2;

const SPIKE_SLOTS: u32 = 5u;
// Must match `lib.rs::COMPLEXITY_ATTACK_GAIN` — keep in sync manually.
const COMPLEXITY_ATTACK_GAIN: f32 = 0.5;

// Per-victim attacker ID slots. K = 8 is enough headroom to detect a
// bonded pair when a victim is mobbed: pack metric needs ≥1 bonded
// pair among the attackers, so as long as one such pair lands inside
// the first K slots we count it as a pack. Higher K costs n * 4 B more
// storage and proportionally longer pair scan on the CPU. K must match
// `predate.rs::MAX_ATTACKERS_PER_VICTIM`.
const MAX_ATTACKERS_PER_VICTIM: u32 = 8u;

struct PredateParams {
    num_cells: u32,
    cell_size: f32,
    cell_radius_const: f32,
    // Sprint 187: per-cell `attack_gates` + `predation_size_ratios` replace
    // the pre-S187 uniform thresholds. `defense_floor` clamps the per-victim
    // defense multiplier so a saturated pool can't invert the sign.
    defense_floor: f32,
    herd_radius_sq: f32,
    _pad_attack_threshold: f32,
    predation_gain: f32,
    predation_drain: f32,
    spike_dot_threshold: f32,
    spike_bonus: f32,
    dilution_k: f32,
    _pad0: u32,
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

@group(0) @binding(0) var<uniform> params: PredateParams;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> eff_radii: array<f32>;
@group(0) @binding(3) var<storage, read> headings: array<f32>;
@group(0) @binding(4) var<storage, read> spikes_packed: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> attack_signals: array<f32>;
@group(0) @binding(6) var<storage, read> hash_offsets: array<u32>;
@group(0) @binding(7) var<storage, read> hash_sorted: array<u32>;
@group(0) @binding(8) var<storage, read_write> herd_counts: array<u32>;
// Per-attacker accumulated self_gain (single writer per `i` — only thread
// `i` writes to `energy_gain[i]`, so `atomicStore` is enough). CPU derives
// `energy_delta[i] = energy_gain[i] - damage_delta[i]` after readback.
@group(0) @binding(9) var<storage, read_write> energy_gain: array<atomic<u32>>;
// Per-victim accumulated drain (multi-writer — every attacker that hits
// `j` adds `predation_drain`). Still needs CAS for the float-add.
@group(0) @binding(10) var<storage, read_write> damage_delta: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read> spike_counts: array<u32>;
@group(0) @binding(12) var<storage, read> pitches: array<f32>;
// Pack-hunting diagnostics. The shader doesn't know about bonds — it
// writes per-attacker hit/gain totals and the first K attacker IDs per
// victim, then the CPU does the bond-pair lookup to classify swarm vs
// pack. Each buffer is reset to 0 per dispatch.
//
// `total_events` (CPU `attack_events.len()` / `predation_events_gen`) is
// derived host-side as `sum(attacker_event_count)` — the dedicated single-slot
// atomic counter pre-S188 serialized every (i,j) hit through one cache line.
@group(0) @binding(13) var<storage, read_write> attacker_event_count: array<atomic<u32>>;
@group(0) @binding(14) var<storage, read_write> attacker_gain_sum: array<atomic<u32>>;
@group(0) @binding(15) var<storage, read_write> victim_attacker_count: array<atomic<u32>>;
// Layout: K slots per victim, IDs stored as `attacker_idx + 1` so 0 = empty.
@group(0) @binding(16) var<storage, read_write> victim_attackers: array<atomic<u32>>;
// Per-bucket minimum eff_radius. Populated by `bucket_min_radius`, consumed
// by `attack` for the size-ratio whole-bucket prefilter. Empty buckets get
// a large sentinel so they always fail the guard (and so get skipped).
@group(0) @binding(17) var<storage, read_write> min_radius_per_bucket: array<f32>;
// Sprint 187: per-cell aggression genome traits.
@group(0) @binding(18) var<storage, read> attack_gates: array<f32>;
@group(0) @binding(19) var<storage, read> predation_size_ratios: array<f32>;
// Per-victim summed defense from bonded partners' `defense_contribution`,
// CPU-aggregated each tick (similar pattern to bonded_inbox).
@group(0) @binding(20) var<storage, read> defense_pool: array<f32>;

// Per-bucket min(eff_radii) reduction. One thread per bucket walks
// `hash_sorted[hash_offsets[b]..hash_offsets[b+1]]` and writes the smallest
// `eff_radii[j]` it sees. `attack` uses this for a fast pre-filter:
// `r_i < size_ratio_threshold * bucket_min` → no cell in the bucket can be
// preyed on, skip the whole bucket walk. Empty buckets emit `1.0e30` so the
// guard rejects them.
const NUM_BUCKETS: u32 = 8192u; // GRID_NX * GRID_NY * GRID_NZ

@compute @workgroup_size(64)
fn bucket_min_radius(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= NUM_BUCKETS) {
        return;
    }
    let start = hash_offsets[b];
    let end = hash_offsets[b + 1u];
    if (start == end) {
        min_radius_per_bucket[b] = 1.0e30;
        return;
    }
    var m: f32 = 1.0e30;
    for (var k = start; k < end; k = k + 1u) {
        let j = hash_sorted[k];
        m = min(m, eff_radii[j]);
    }
    min_radius_per_bucket[b] = m;
}

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
    let cs = params.cell_size;
    let r_cells = i32(ceil(sqrt(params.herd_radius_sq) / cs));
    var count: u32 = 0u;
    // Resolve the center bucket once and walk neighbors via integer ±wrap on
    // xy (z clamped) — replaces the per-iteration `bucket_id_wrapped` chain.
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
    if (attack_strength <= attack_gates[i]) {
        return;
    }
    let size_ratio_i = predation_size_ratios[i];
    let pos_i = vec3<f32>(
        positions[i * 3u + 0u],
        positions[i * 3u + 1u],
        positions[i * 3u + 2u],
    );
    let r_i = eff_radii[i];
    let crc = params.cell_radius_const;
    let crc_r_i = crc * r_i;
    let yaw_i = headings[i];
    let pitch_i = pitches[i];
    let n_spikes = min(spike_counts[i], SPIKE_SLOTS);
    let spikes_base = i * SPIKE_SLOTS;

    // Precompute spike world-space directions and per-spike attack factors.
    // These depend only on the attacker's pose and spike geometry, not on
    // the victim — without the hoist, every (i, j) pair recomputes 4 trig
    // calls per spike inside the bucket walk.
    var spike_dir: array<vec3<f32>, SPIKE_SLOTS>;
    var spike_factor: array<f32, SPIKE_SLOTS>;
    var spike_active: u32 = 0u;
    for (var s: u32 = 0u; s < n_spikes; s = s + 1u) {
        let spk = spikes_packed[spikes_base + s];
        if (spk.x <= 0.0) {
            continue;
        }
        let yaw_s = yaw_i + spk.y;
        let pit_s = pitch_i + spk.z;
        let cos_p = cos(pit_s);
        spike_dir[spike_active] =
            vec3<f32>(cos(yaw_s) * cos_p, sin(yaw_s) * cos_p, sin(pit_s));
        let cmplx = clamp(spk.w, 0.0, 1.0);
        spike_factor[spike_active] =
            spk.x * (1.0 + COMPLEXITY_ATTACK_GAIN * cmplx) * params.spike_bonus;
        spike_active = spike_active + 1u;
    }

    // Size-ratio threshold guarantees r_j ≤ r_i, so 2·CELL_RADIUS·r_i is
    // the largest pair_r this attacker can encounter.
    let cs = params.cell_size;
    let r_cells = i32(ceil(2.0 * crc_r_i / cs));

    var self_gain: f32 = 0.0;

    // Resolve the center bucket once, walk neighbors via integer ±wrap on xy
    // (z clamped). Halves bucket-coord overhead vs the ghost-position pattern.
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
                // Whole-bucket size-ratio prefilter: if this attacker can't
                // even prey on the smallest cell in the bucket, the entire
                // bucket walk is wasted. Saves both the hash_offsets fetch
                // chain and every per-pair load for large attackers.
                if (r_i < size_ratio_i * min_radius_per_bucket[b]) {
                    continue;
                }
                let start = hash_offsets[b];
                let end = hash_offsets[b + 1u];
                for (var k = start; k < end; k = k + 1u) {
                    let j = hash_sorted[k];
                    if (j == i) {
                        continue;
                    }
                    let r_j = eff_radii[j];
                    if (r_i < size_ratio_i * r_j) {
                        continue;
                    }
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
                    if (d2 < pair_r2) {
                        atomicAdd(&attacker_event_count[i], 1u);
                        let slot = atomicAdd(&victim_attacker_count[j], 1u);
                        if (slot < MAX_ATTACKERS_PER_VICTIM) {
                            atomicStore(
                                &victim_attackers[j * MAX_ATTACKERS_PER_VICTIM + slot],
                                i + 1u,
                            );
                        }
                        var gain = params.predation_gain;
                        if (spike_active > 0u && d2 > 0.0) {
                            let inv_d = inverseSqrt(d2);
                            // d = pos_i - pj, so the unit vector toward the
                            // victim is -d * inv_d.
                            let to_target =
                                vec3<f32>(-d.x * inv_d, -d.y * inv_d, -d.z * inv_d);
                            var spike_bonus: f32 = 0.0;
                            for (var s: u32 = 0u; s < spike_active; s = s + 1u) {
                                let cos_a = dot(spike_dir[s], to_target);
                                if (cos_a >= params.spike_dot_threshold) {
                                    spike_bonus = spike_bonus + spike_factor[s];
                                }
                            }
                            gain = gain + params.predation_gain * spike_bonus;
                        }
                        let n_neigh = f32(herd_counts[j]);
                        let dilution = 1.0 / (1.0 + params.dilution_k * n_neigh);
                        // Sprint 187: bonded victims get a defense multiplier
                        // reducing both attacker gain and victim drain. The
                        // floor (`defense_floor`) bounds maximum protection
                        // so kin-defense can't make any cluster immune.
                        let def_factor = max(1.0 - defense_pool[j], params.defense_floor);
                        gain = gain * dilution * def_factor;
                        let applied_drain = params.predation_drain * def_factor;
                        self_gain = self_gain + gain;

                        // Single CAS float-add: only `damage_delta[j]` is
                        // touched per pair-hit. `energy_delta` on the CPU is
                        // derived as `energy_gain[i] - damage_delta[i]`, so
                        // we don't need a parallel `energy_delta[j] -= drain`
                        // CAS — that pre-S188 path doubled the atomic
                        // contention on mobbed victims for no semantic gain.
                        var old_d: u32 = atomicLoad(&damage_delta[j]);
                        loop {
                            let new_d: u32 =
                                bitcast<u32>(bitcast<f32>(old_d) + applied_drain);
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

    // Self-gain: thread `i` is the only writer to `energy_gain[i]`, so a
    // plain atomicStore is enough — no CAS needed. (Drain from other
    // attackers lands in `damage_delta[i]`, kept on its own buffer.)
    if (self_gain != 0.0) {
        atomicStore(&energy_gain[i], bitcast<u32>(self_gain));
        atomicStore(&attacker_gain_sum[i], bitcast<u32>(self_gain));
    }
}
