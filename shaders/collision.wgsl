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
    // to convert it into a per-tick velocity delta (`Δv = F · dt / m`).
    // Pre-S192 omitted dt → bond impulses were 60× too strong; Sprint 202
    // adds the `/m` term.
    dt: f32,
    // Sprint 202: bond bending stiffness/damping. Restoring force pulls each
    // bond-pair on cell i back toward the rest cosine recorded at the moment
    // the pair first appeared. Zero disables bending; conservative defaults
    // in `params::physics`.
    bond_bend_stiffness: f32,
    bond_bend_damping: f32,
    // Sprint 202 hotfix: stability ceiling on dt/mass for the bond spring +
    // damper + bending impulse (`crate::DT_OVER_M_BOND_MAX`). Dividing by the
    // 0.01 mass floor pushed small/stiff cells past the explicit-Euler limit.
    dt_over_m_bond_max: f32,
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
// Sprint 202: per-cell inertial mass for bond spring impulse conversion
// (`Δv = F · dt / mass`). Adhesion and depenetration stay mass-free —
// adhesion is a per-tick velocity nudge (not a force), and depenetration
// is position-only.
@group(0) @binding(16) var<storage, read> masses: array<f32>;
// Sprint 202: per-cell heading/pitch used by bond-bending to apply the
// restoring force in the bond-pair plane. Bend storage:
// `bond_rest_cos[i * BPC² + a * BPC + b]` = cosine of the rest angle
// between bond slot `a` and slot `b` on cell `i`, captured at the moment
// the second bond formed. `b > a` is canonical; `b <= a` slots are unused.
@group(0) @binding(17) var<storage, read> bond_rest_cos: array<f32>;
// Sprint 203: per-cell heading + body dims for the oriented-ellipsoid contact
// radius. Pitch is ignored (clamped ±15°, negligible for the collision
// footprint). `headings[j]` / `body_dims[j*3..]` give the neighbor's ellipsoid.
@group(0) @binding(18) var<storage, read> headings: array<f32>;
@group(0) @binding(19) var<storage, read> body_dims: array<f32>;

// Fixed upper bound for the per-thread bond stack arrays in the bending pass.
// WGSL array sizes must be compile-time; this must stay ≥ params.bonds_per_cell
// (= Rust MAX_BONDS_PER_CELL). Runtime loops are bounded by params.bonds_per_cell.
const BOND_SLOTS: u32 = 10u;

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

// Ellipsoid support radius along unit direction `n` (world frame). Body axes:
// x = body_length (heading-forward), y = body_width (right), z = body_height
// (up); pitch ignored. For an isotropic body (l=w=h=s) returns s, so pair_r
// reduces to the legacy CELL_RADIUS·(eff_r_i + eff_r_j) sphere test.
fn ellipsoid_support(n: vec3<f32>, heading: f32, dims: vec3<f32>) -> f32 {
    let ch = cos(heading);
    let sh = sin(heading);
    let n_fwd = n.x * ch + n.y * sh;
    let n_right = -n.x * sh + n.y * ch;
    let a = dims.x * n_fwd;
    let b = dims.y * n_right;
    let c = dims.z * n.z;
    return sqrt(a * a + b * b + c * c);
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
    let head_i = headings[i];
    let dims_i = vec3<f32>(
        body_dims[i * 3u + 0u],
        body_dims[i * 3u + 1u],
        body_dims[i * 3u + 2u],
    );
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
    // z is clamped (not toroidal) and GRID_NZ is tiny (4): iterate the distinct
    // bz planes directly. A `dz in -r_cells..=r_cells` + per-iteration `clamp`
    // rescans boundary planes whenever r_cells reaches a z edge, double-counting
    // depenetration/adhesion contributions for those neighbors.
    let center = bucket_coords_of(pos_i);
    let bz_lo = max(center.z - r_cells, 0);
    let bz_hi = min(center.z + r_cells, GRID_NZ - 1);
    for (var bz = bz_lo; bz <= bz_hi; bz = bz + 1) {
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
                    // Same-cell (d²=0) / coincident: skip. Directional pair_r
                    // needs the separation normal, so compute d before pair_r.
                    if (d2 <= 0.0) {
                        continue;
                    }
                    let n_dir = d * inverseSqrt(d2);
                    // Sprint 203: oriented-ellipsoid contact radius — each cell's
                    // extent along the contact normal is its body-ellipsoid
                    // support. Isotropic bodies reduce to the legacy sphere test.
                    let dims_j = vec3<f32>(
                        body_dims[j * 3u + 0u],
                        body_dims[j * 3u + 1u],
                        body_dims[j * 3u + 2u],
                    );
                    let pair_r = crc * (
                        ellipsoid_support(n_dir, head_i, dims_i)
                        + ellipsoid_support(n_dir, headings[j], dims_j)
                    );
                    let pair_r2 = pair_r * pair_r;
                    if (d2 < pair_r2) {
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
                    } else {
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
    // along the spring axis. Sprint 202: divided by cell mass so the same
    // genome-encoded stiffness produces consistent kinematics regardless of
    // cell size. Overstretched bonds (dist > rest × break_factor) contribute
    // zero force — the CPU side handles the actual break decision in a
    // follow-up pass.
    let bond_base = i * params.bonds_per_cell;
    let inv_mass = 1.0 / max(masses[i], 0.01);
    // Sprint 202 hotfix: clamp dt/mass to the explicit-Euler stability ceiling.
    // Dividing the spring+damper impulse by the 0.01 mass floor let small/stiff
    // cells blow up (k·dt²/m, c·dt/m ≫ stability limit) → overstretch → break.
    let dt_over_m = min(params.dt * inv_mass, params.dt_over_m_bond_max);
    // First pass: cache per-bond unit direction (partner → i) and distance for
    // every geometrically present bond. `bond_present` excludes only missing
    // (-1) or coincident (d² ≈ 0) partners; overstretched bonds stay present
    // for the bending pass (so the angle force stays momentum-consistent with
    // the neighbours), but the spring force below still skips them. BPC ≤ BOND_SLOTS.
    var bond_dir: array<vec3<f32>, BOND_SLOTS>;
    var bond_dist: array<f32, BOND_SLOTS>;
    var bond_present: array<u32, BOND_SLOTS>;
    for (var slot = 0u; slot < BOND_SLOTS; slot = slot + 1u) {
        bond_dir[slot] = vec3<f32>(0.0, 0.0, 0.0);
        bond_dist[slot] = 0.0;
        bond_present[slot] = 0u;
    }
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
        let n = d * inv_d;
        bond_dir[slot] = n;
        bond_dist[slot] = dist;
        bond_present[slot] = 1u;
        let rest = bond_rest[bond_idx];
        if (dist > rest * params.bond_break_factor) {
            continue;  // overstretched: no spring force, CPU prunes it
        }
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
        // Sprint 192/202: integrate Hookean force over the tick AND divide
        // by mass (`Δv = F · dt / m`, dt/m clamped above).
        let mag = (spring + damp) * dt_over_m;
        vdx_acc = vdx_acc + mag * n.x;
        vdy_acc = vdy_acc + mag * n.y;
        vdz_acc = vdz_acc + mag * n.z;
    }

    // Sprint 202 (rewritten): bond bending as a CONSERVATIVE cosine angle
    // spring V = ½·K·(cosθ − cos_rest)² between two bonds meeting at a shared
    // cell, plus a momentum-conserving Rayleigh damping D·d(cosθ)/dt. Forces
    // come from the exact gradient ∂cosθ/∂p, so the vertex and both arms sum
    // to zero net force (Newton's third law) — no spurious energy injection,
    // unlike the pre-rewrite un-normalised-bisector heuristic. Atomics-free:
    // each cell only writes its OWN impulse, in two roles. `u_*` = unit vector
    // from the shared vertex toward a cell; `bond_dir` points partner → self,
    // so the vertex role negates its own dirs while the arm role's own
    // `bond_dir` already equals u (vertex → self). Vertex + both arms read the
    // same positions/velocities → identical cosθ, ∂cos/∂p and ċos → the per-
    // angle forces cancel to fp tolerance.
    if (params.bond_bend_stiffness > 0.0) {
        let bpc = params.bonds_per_cell;
        let kk = params.bond_bend_stiffness;
        let dd = params.bond_bend_damping;

        // Role VERTEX: i is the shared cell; iterate pairs of its own bonds.
        let cos_base_i = i * bpc * bpc;
        for (var a = 0u; a < bpc; a = a + 1u) {
            if (bond_present[a] == 0u) { continue; }
            let u_a = -bond_dir[a];               // i → arm a
            let r_a = bond_dist[a];
            let pa = u32(bond_partner_idx[bond_base + a]);
            let vel_a = vec3<f32>(velocities[pa*3u+0u], velocities[pa*3u+1u], velocities[pa*3u+2u]);
            for (var b = a + 1u; b < bpc; b = b + 1u) {
                if (bond_present[b] == 0u) { continue; }
                let u_b = -bond_dir[b];           // i → arm b
                let r_b = bond_dist[b];
                let cos_t = dot(u_a, u_b);
                let ga = (u_b - cos_t * u_a) / r_a;   // ∂cos/∂p_a
                let gb = (u_a - cos_t * u_b) / r_b;   // ∂cos/∂p_b
                let pb = u32(bond_partner_idx[bond_base + b]);
                let vel_b = vec3<f32>(velocities[pb*3u+0u], velocities[pb*3u+1u], velocities[pb*3u+2u]);
                let cos_dot = dot(ga, vel_a - vel_i) + dot(gb, vel_b - vel_i);
                let cos_rest = bond_rest_cos[cos_base_i + a * bpc + b];
                let lam = kk * (cos_t - cos_rest) + dd * cos_dot;
                // ∂cos/∂p_i = -(ga+gb); F_i = -lam·∂cos/∂p_i = lam·(ga+gb).
                let f_i = lam * (ga + gb);
                vdx_acc = vdx_acc + f_i.x * dt_over_m;
                vdy_acc = vdy_acc + f_i.y * dt_over_m;
                vdz_acc = vdz_acc + f_i.z * dt_over_m;
            }
        }

        // Role ARM: i is an arm of a neighbour v's angle (v; i, b). For each
        // present bond i→v, scan v's other bonds for siblings b.
        for (var slot = 0u; slot < bpc; slot = slot + 1u) {
            if (bond_present[slot] == 0u) { continue; }
            let v_signed = bond_partner_idx[bond_base + slot];
            if (v_signed < 0) { continue; }
            let v = u32(v_signed);
            let u_vc = bond_dir[slot];            // v → i (this cell is the arm)
            let r_vc = bond_dist[slot];
            let v_base = v * bpc;
            // v's slot pointing back to i — needed for the rest-cos lookup.
            var slot_vc: u32 = bpc;
            for (var s = 0u; s < bpc; s = s + 1u) {
                if (bond_partner_idx[v_base + s] == i32(i)) { slot_vc = s; break; }
            }
            if (slot_vc == bpc) { continue; }
            let pv = vec3<f32>(positions[v*3u+0u], positions[v*3u+1u], positions[v*3u+2u]);
            let vel_v = vec3<f32>(velocities[v*3u+0u], velocities[v*3u+1u], velocities[v*3u+2u]);
            let cos_base_v = v * bpc * bpc;
            for (var s = 0u; s < bpc; s = s + 1u) {
                if (s == slot_vc) { continue; }
                let b_signed = bond_partner_idx[v_base + s];
                if (b_signed < 0) { continue; }
                let b = u32(b_signed);
                let pb = vec3<f32>(positions[b*3u+0u], positions[b*3u+1u], positions[b*3u+2u]);
                let dvb = vec3<f32>(
                    min_image_xy(pb.x - pv.x, params.world_half_x),
                    min_image_xy(pb.y - pv.y, params.world_half_y),
                    pb.z - pv.z,
                );
                let r_vb2 = dot(dvb, dvb);
                if (r_vb2 <= 1e-20) { continue; }
                let inv_vb = inverseSqrt(r_vb2);
                let r_vb = r_vb2 * inv_vb;
                let u_vb = dvb * inv_vb;          // v → b
                let cos_t = dot(u_vc, u_vb);
                let ga = (u_vb - cos_t * u_vc) / r_vc;   // ∂cos/∂p_i (this arm)
                let gb = (u_vc - cos_t * u_vb) / r_vb;   // ∂cos/∂p_b
                let vel_b = vec3<f32>(velocities[b*3u+0u], velocities[b*3u+1u], velocities[b*3u+2u]);
                let cos_dot = dot(ga, vel_i - vel_v) + dot(gb, vel_b - vel_v);
                let cos_rest = bond_rest_cos[cos_base_v + slot_vc * bpc + s];
                let lam = kk * (cos_t - cos_rest) + dd * cos_dot;
                // F_i = -lam·∂cos/∂p_i = -lam·ga.
                let f_i = -lam * ga;
                vdx_acc = vdx_acc + f_i.x * dt_over_m;
                vdy_acc = vdy_acc + f_i.y * dt_over_m;
                vdz_acc = vdz_acc + f_i.z * dt_over_m;
            }
        }
    }

    deltas[i * 3u + 0u] = dx_acc;
    deltas[i * 3u + 1u] = dy_acc;
    deltas[i * 3u + 2u] = dz_acc;
    vel_deltas[i * 3u + 0u] = vdx_acc;
    vel_deltas[i * 3u + 1u] = vdy_acc;
    vel_deltas[i * 3u + 2u] = vdz_acc;
}
