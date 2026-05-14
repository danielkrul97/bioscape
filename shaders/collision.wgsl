// Per-cell collision resolution against spatial-hash neighbors. For each
// pair (i, j) with d² < (CELL_RADIUS × (eff_r_i + eff_r_j))² and d² > 0,
// `deltas[i]` accumulates (d/|d|) × overlap × 0.5 (position depenetration),
// and `vel_deltas[i]` accumulates an inelastic damping impulse along the
// contact normal when the pair is closing (v_rel · n < 0). Outputs are
// write-only per i — no atomics needed. The XY world is toroidal, so the
// search neighborhood walks 3D ghost positions around `pos_i` to cover
// wrap, and pair distances use the min-image convention. Search radius
// bound matches the CPU helper: CELL_RADIUS × (eff_r_i + max_axis_i × 2).

const GRID_NX: i32 = 64;
const GRID_NY: i32 = 32;
const GRID_NZ: i32 = 4;
const HALF_NX: i32 = 32;
const HALF_NY: i32 = 16;
const HALF_NZ: i32 = 2;

struct CollisionParams {
    num_cells: u32,
    cell_size: f32,
    cell_radius_const: f32,
    collision_restitution: f32,
    world_half_x: f32,
    world_half_y: f32,
    adhesion_strength: f32,
    adhesion_cross_type: f32,
    adhesion_range_factor: f32,
    bond_break_factor: f32,
    bonds_per_cell: u32,
    max_contacts_per_cell: u32,
    // Sprint 192: dt = 1 / FIXED_TIMESTEP_HZ. Multiplies Hookean bond force
    // to convert it into a per-tick velocity delta (`Δv = F · dt`). Pre-S192
    // omitted dt → bond impulses were 60× too strong.
    dt: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: CollisionParams;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> eff_radii: array<f32>;
@group(0) @binding(3) var<storage, read> max_axes: array<f32>;
@group(0) @binding(4) var<storage, read> hash_offsets: array<u32>;
@group(0) @binding(5) var<storage, read> hash_sorted: array<u32>;
@group(0) @binding(6) var<storage, read_write> deltas: array<f32>;
@group(0) @binding(7) var<storage, read> velocities: array<f32>;
@group(0) @binding(8) var<storage, read_write> vel_deltas: array<f32>;
@group(0) @binding(9) var<storage, read> adhesion_types: array<u32>;
@group(0) @binding(10) var<storage, read> bond_partner_idx: array<i32>;
@group(0) @binding(11) var<storage, read> bond_rest: array<f32>;
@group(0) @binding(12) var<storage, read> bond_stiffness: array<f32>;
@group(0) @binding(13) var<storage, read> bond_damping: array<f32>;
@group(0) @binding(14) var<storage, read_write> contact_count: array<atomic<u32>>;
@group(0) @binding(15) var<storage, read_write> contact_partners: array<u32>;

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

@compute @workgroup_size(64)
fn collision(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }
    let pos_i = vec3<f32>(
        positions[i * 3u + 0u],
        positions[i * 3u + 1u],
        positions[i * 3u + 2u],
    );
    let vel_i = vec3<f32>(
        velocities[i * 3u + 0u],
        velocities[i * 3u + 1u],
        velocities[i * 3u + 2u],
    );
    let r_i = eff_radii[i];
    let crc = params.cell_radius_const;
    let crc_r_i = crc * r_i;
    // Sprint 66: search radius covers both collision contact range AND
    // adhesion falloff range. ADHESION_RANGE_FACTOR typically expands
    // the search ~3×, so adhesion neighbors must be reachable.
    let collision_r = crc * (r_i + max_axes[i] * 2.0);
    let search_r = collision_r * max(1.0, params.adhesion_range_factor);
    let cs = params.cell_size;
    let r_cells = i32(ceil(search_r / cs));
    let damp_coeff = 0.5 * (1.0 - params.collision_restitution);
    let type_i = adhesion_types[i];

    var dx_acc: f32 = 0.0;
    var dy_acc: f32 = 0.0;
    var dz_acc: f32 = 0.0;
    var vdx_acc: f32 = 0.0;
    var vdy_acc: f32 = 0.0;
    var vdz_acc: f32 = 0.0;

    // Resolve the center bucket once, walk neighbors via integer ±wrap on xy
    // (z clamped). Replaces the per-iteration `bucket_id_wrapped(ghost_pos)`
    // chain — same pattern as `sensor_gather.wgsl`.
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
                    // No `j == i` guard: same-cell pairs give d² = 0, which the
                    // `d2 > 0.0` filter below rejects. Skipping the explicit
                    // check removes one branch per neighbor.
                    let r_j = eff_radii[j];
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
                    if (d2 < pair_r2 && d2 > 0.0) {
                        // Algebraically: overlap*0.5/dist = pair_r*0.5/dist - 0.5.
                        // Replaces sqrt + divide with a single rsqrt + fma.
                        let inv_d = inverseSqrt(d2);
                        let scale = pair_r * 0.5 * inv_d - 0.5;
                        dx_acc = dx_acc + d.x * scale;
                        dy_acc = dy_acc + d.y * scale;
                        dz_acc = dz_acc + d.z * scale;
                        let n = d * inv_d;
                        let vel_j = vec3<f32>(
                            velocities[j * 3u + 0u],
                            velocities[j * 3u + 1u],
                            velocities[j * 3u + 2u],
                        );
                        let v_rel = vel_i - vel_j;
                        let v_rel_n = dot(v_rel, n);
                        if (v_rel_n < 0.0) {
                            let damp = -v_rel_n * damp_coeff;
                            vdx_acc = vdx_acc + damp * n.x;
                            vdy_acc = vdy_acc + damp * n.y;
                            vdz_acc = vdz_acc + damp * n.z;
                        }
                        // Sprint 66: record per-pair contact events for bond
                        // formation. Dedupe symmetric pair by keeping only
                        // i < j; CPU resolves partner cell_ids via the
                        // tick-stable id_to_idx map after readback.
                        if (i < j) {
                            let slot = atomicAdd(&contact_count[i], 1u);
                            if (slot < params.max_contacts_per_cell) {
                                let base = i * params.max_contacts_per_cell + slot;
                                contact_partners[base] = j;
                            }
                        }
                    } else if (d2 > 0.0) {
                        // Sprint 66 differential adhesion: out-of-contact pairs
                        // get a linear-falloff velocity nudge along ±n. Same-type
                        // attracts (positive coefficient), cross-type repels
                        // (negative coefficient via ADHESION_CROSS_TYPE).
                        let adhesion_range = pair_r * params.adhesion_range_factor;
                        let adhesion_range2 = adhesion_range * adhesion_range;
                        if (d2 < adhesion_range2) {
                            let inv_d = inverseSqrt(d2);
                            let dist = d2 * inv_d;
                            let falloff = (adhesion_range - dist) / (adhesion_range - pair_r);
                            var coeff: f32 = params.adhesion_strength;
                            if (adhesion_types[j] != type_i) {
                                coeff = coeff * params.adhesion_cross_type;
                            }
                            let mag = -coeff * falloff;
                            vdx_acc = vdx_acc + mag * d.x * inv_d;
                            vdy_acc = vdy_acc + mag * d.y * inv_d;
                            vdz_acc = vdz_acc + mag * d.z * inv_d;
                        }
                    }
                }
            }
        }
    }

    // Sprint 66 spring bonds: each cell carries up to `bonds_per_cell` slots
    // pre-resolved by the caller to partner indices (-1 = empty). The bond
    // force is Hookean spring × (dist − rest) plus per-bond linear damping
    // along the spring axis. Overstretched bonds (dist > rest × break_factor)
    // contribute zero force — the CPU side handles the actual break decision
    // in a follow-up pass.
    let bond_base = i * params.bonds_per_cell;
    for (var slot = 0u; slot < params.bonds_per_cell; slot = slot + 1u) {
        let bond_idx = bond_base + slot;
        let j_signed = bond_partner_idx[bond_idx];
        if (j_signed < 0) {
            continue;
        }
        let j = u32(j_signed);
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
        if (d2 <= 1e-20) {
            continue;
        }
        let inv_d = inverseSqrt(d2);
        let dist = d2 * inv_d;
        let rest = bond_rest[bond_idx];
        let break_len = rest * params.bond_break_factor;
        if (dist > break_len) {
            continue;
        }
        let n = d * inv_d;
        let extension = dist - rest;
        let stiffness = bond_stiffness[bond_idx];
        let damping = bond_damping[bond_idx];
        let spring = -stiffness * extension;
        let vel_j = vec3<f32>(
            velocities[j * 3u + 0u],
            velocities[j * 3u + 1u],
            velocities[j * 3u + 2u],
        );
        let v_rel = vel_i - vel_j;
        let v_rel_n = dot(v_rel, n);
        let damp = -damping * v_rel_n;
        // Sprint 192: integrate Hookean force over the tick. Pre-S192 wrote
        // `mag` directly to `vel_deltas` (effectively 60× too strong).
        let mag = (spring + damp) * params.dt;
        vdx_acc = vdx_acc + mag * n.x;
        vdy_acc = vdy_acc + mag * n.y;
        vdz_acc = vdz_acc + mag * n.z;
    }

    deltas[i * 3u + 0u] = dx_acc;
    deltas[i * 3u + 1u] = dy_acc;
    deltas[i * 3u + 2u] = dz_acc;
    vel_deltas[i * 3u + 0u] = vdx_acc;
    vel_deltas[i * 3u + 1u] = vdy_acc;
    vel_deltas[i * 3u + 2u] = vdz_acc;
}
