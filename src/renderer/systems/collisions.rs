use bevy::diagnostic::Diagnostics;
use bevy::prelude::*;
use bioscape::{
    ADHESION_RANGE_FACTOR, ATTACK_THRESHOLD, BOND_BREAK_THRESHOLD, BOND_FORM_THRESHOLD,
    BOND_FORM_TICKS, BOND_FORMATION_COST, BOND_MAINTENANCE_PER_SEC, BOND_REST_LENGTH_SLACK, Bond,
    CELL_RADIUS, CONTACT_DECAY_TICKS, DILUTION_K, FIXED_TIMESTEP_HZ, HERD_RADIUS, MAX_BONDS_PER_CELL,
    PREDATION_DRAIN_PER_TICK, PREDATION_GAIN_PER_TICK, SIZE_RATIO_THRESHOLD, WORLD_HALF,
    adhesion_velocity_delta, bond_velocity_delta,
};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Instant;

use super::super::components::{CellEntity, Dying};
use super::super::config::{DIAG_COLLISIONS, DIAG_PREDATION};
use super::super::resources::{CellGrid, ContactProgress};
#[cfg(feature = "gpu")]
use super::super::resources_gpu::GpuFullPipeline;

// Generous broad-phase upper bound on "other" effective_radius — captures
// candidates even when neighbors are oversized. Narrow-phase uses pair sum.
pub(crate) const BROAD_PHASE_SIZE_BUDGET: f32 = 3.0;

/// Sprint 66: snapshot row pro renderer collision/adhesion/bond pass.
pub(crate) struct SnapEntry {
    pub(crate) entity: Entity,
    pub(crate) cell_id: u64,
    pub(crate) position: [f32; 3],
    pub(crate) velocity: [f32; 3],
    pub(crate) radius: f32,
    pub(crate) adhesion_type: u8,
    pub(crate) bonds: [Option<Bond>; MAX_BONDS_PER_CELL],
}

pub(crate) fn cell_predates_on_neighbor(
    grid: Res<CellGrid>,
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
    mut diag: Diagnostics,
    mut energy_changes_scratch: Local<FxHashMap<Entity, f32>>,
    mut damage_changes_scratch: Local<FxHashMap<Entity, f32>>,
    mut snapshot_scratch: Local<Vec<(Entity, [f32; 3])>>,
    mut bond_counts_scratch: Local<FxHashMap<Entity, u32>>,
    mut herd_counts_vec_scratch: Local<Vec<u32>>,
    mut herd_counts_scratch: Local<FxHashMap<Entity, u32>>,
) {
    let t_total = Instant::now();
    // R-#5: persistent scratchy. Pre-fix: 6 fresh allocs per tick. Sprint 58
    // používal HashMap → FxHashMap (fixed seed); reuse navíc eliminuje alloc.
    energy_changes_scratch.clear();
    damage_changes_scratch.clear();
    let energy_changes = &mut *energy_changes_scratch;
    let damage_changes = &mut *damage_changes_scratch;

    // Sprint 29 selfish-herd: pre-compute herd count per cell (počet sousedů
    // ve `HERD_RADIUS`). Snapshot + bond_counts plněny single-pass přes
    // cells.iter() místo dvou nezávislých walků.
    let herd_r2 = HERD_RADIUS * HERD_RADIUS;
    snapshot_scratch.clear();
    bond_counts_scratch.clear();
    for (e, c) in cells.iter() {
        snapshot_scratch.push((e, c.0.position));
        bond_counts_scratch.insert(e, c.0.n_bonds());
    }
    let snapshot = snapshot_scratch.as_slice();
    let bond_counts = &*bond_counts_scratch;
    let grid_ref = &grid.0;
    herd_counts_vec_scratch.clear();
    snapshot
        .par_iter()
        .map(|(entity, pos)| {
            let mut count: u32 = 0;
            grid_ref.for_each_in_radius_toroidal(
                *pos,
                HERD_RADIUS,
                WORLD_HALF,
                |other, other_pos, _| {
                    if other == *entity {
                        return;
                    }
                    let d = bioscape::min_image_delta(*pos, other_pos, WORLD_HALF);
                    if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < herd_r2 {
                        count += 1;
                    }
                },
            );
            count
        })
        .collect_into_vec(&mut herd_counts_vec_scratch);
    let herd_counts_vec = herd_counts_vec_scratch.as_slice();
    herd_counts_scratch.clear();
    for ((e, _), c) in snapshot.iter().zip(herd_counts_vec.iter()) {
        herd_counts_scratch.insert(*e, *c);
    }
    let herd_counts = &*herd_counts_scratch;

    for (entity_a, cell_a) in &cells {
        // Sprint 27: attack je opt-in přes brain output[6]. Bez aktivního
        // signálu kontakty s menšími cells jen kolize (řešené v
        // resolve_cell_collisions), ne predace.
        if cell_a.0.last_outputs[6].max(0.0) <= ATTACK_THRESHOLD {
            continue;
        }
        let pos_a = cell_a.0.position;
        let radius_a = cell_a.0.phenotype.effective_radius();
        let broad_r = CELL_RADIUS * (radius_a + BROAD_PHASE_SIZE_BUDGET);

        grid.0
            .for_each_in_radius_toroidal(pos_a, broad_r, WORLD_HALF, |entity_b, pos_b, radius_b| {
                if entity_b == entity_a {
                    return;
                }
                if radius_a < SIZE_RATIO_THRESHOLD * radius_b {
                    return;
                }
                let pair_r = CELL_RADIUS * (radius_a + radius_b);
                let pair_r2 = pair_r * pair_r;
                // Sprint 54: min-image delta a→b. Spike bonus volá `spike_bonus_against`
                // s pos_b — pro toroidal upravíme target pos do min-image frame.
                let d = bioscape::min_image_delta(pos_a, pos_b, WORLD_HALF);
                let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if d2 >= pair_r2 {
                    return;
                }
                let ghost_b = [pos_a[0] + d[0], pos_a[1] + d[1], pos_a[2] + d[2]];
                let bonus = cell_a.0.spike_bonus_against(ghost_b);
                let gain_raw = PREDATION_GAIN_PER_TICK + bonus;
                let prey_neighbors = *herd_counts.get(&entity_b).unwrap_or(&0);
                let dilution = 1.0 / (1.0 + DILUTION_K * prey_neighbors as f32);
                // Sprint 69: bonded prey takes less damage + yields less energy.
                // bond_count_b je 0 pokud entity_b není v map (mrtvá / not yet
                // v snapshot) — graceful fallback na "no defense".
                let bond_count_b = *bond_counts.get(&entity_b).unwrap_or(&0);
                let defense = bioscape::bond_defense_factor(bond_count_b);
                let gain = gain_raw * dilution * defense;
                let drain = PREDATION_DRAIN_PER_TICK * defense;
                *energy_changes.entry(entity_a).or_insert(0.0) += gain;
                *energy_changes.entry(entity_b).or_insert(0.0) -= drain;
                *damage_changes.entry(entity_b).or_insert(0.0) += drain;
            });
    }

    for (entity, delta) in energy_changes.iter() {
        if let Ok((_, mut cell)) = cells.get_mut(*entity) {
            cell.0.energy += *delta;
        }
    }
    for (entity, delta) in damage_changes.iter() {
        if let Ok((_, mut cell)) = cells.get_mut(*entity) {
            cell.0.damage_accum += *delta;
        }
    }
    diag.add_measurement(&DIAG_PREDATION, || t_total.elapsed().as_secs_f64() * 1000.0);
}

pub(crate) fn resolve_cell_collisions(
    grid: Res<CellGrid>,
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
    mut contact_progress: ResMut<ContactProgress>,
    mut diag: Diagnostics,
    #[cfg(feature = "gpu")] gpu_full: Option<ResMut<GpuFullPipeline>>,
    mut snapshot_scratch: Local<Vec<SnapEntry>>,
    mut entity_to_idx_scratch: Local<FxHashMap<Entity, usize>>,
    mut id_to_idx_scratch: Local<FxHashMap<u64, usize>>,
    mut results_scratch: Local<Vec<(Entity, [f32; 3], [f32; 3], Vec<u64>)>>,
    mut seen_pairs_scratch: Local<FxHashSet<(u64, u64)>>,
    mut candidates_scratch: Local<Vec<(u64, u64)>>,
) {
    let t_total = Instant::now();
    // R-#6: persistent Local scratchy. Pre-fix: 7 fresh allocs (snapshot,
    // entity_to_idx, id_to_idx, results+per-cell Vec<u64>, seen_pairs,
    // positions, candidates) per tick. Reuse zachová capacity.
    snapshot_scratch.clear();
    entity_to_idx_scratch.clear();
    id_to_idx_scratch.clear();
    for (e, c) in cells.iter() {
        let idx = snapshot_scratch.len();
        snapshot_scratch.push(SnapEntry {
            entity: e,
            cell_id: c.0.cell_id,
            position: c.0.position,
            velocity: c.0.velocity,
            radius: c.0.phenotype.effective_radius(),
            adhesion_type: c.0.genome.adhesion_type,
            bonds: c.0.bonds,
        });
        entity_to_idx_scratch.insert(e, idx);
        id_to_idx_scratch.insert(c.0.cell_id, idx);
    }
    let entity_to_idx = &*entity_to_idx_scratch;
    let id_to_idx = &*id_to_idx_scratch;
    let grid_ref = &grid.0;
    // Phase 1 (parallel): per-cell delta + vel_delta + collected contacts.
    // Inner Vec<u64> per cell — drop persisted approach kvůli rayon
    // collect_into_vec: existing inner Vecs se znovu použijí jen pokud délka
    // results == n a indexy se zachovají. Reuse outer scratch je primary win;
    // inner Vec capacity přežije přes ticky.
    results_scratch.clear();
    #[cfg(feature = "gpu")]
    let used_gpu = gpu_full.is_some();
    #[cfg(not(feature = "gpu"))]
    let used_gpu = false;
    #[cfg(feature = "gpu")]
    if let Some(mut gpu_res) = gpu_full {
        // Bevy `ResMut::deref_mut` doesn't allow split-field borrows, so the
        // sibling-field `&gpu.cell_hash` arg passed to `gpu.collision.compute`
        // would clash. Re-borrow through `&mut *res` to get a raw
        // `&mut GpuFullPipeline` — the borrow checker handles split borrows
        // through that.
        let gpu = &mut *gpu_res;
        let snapshot = snapshot_scratch.as_slice();
        let n = snapshot.len();
        if n > 0 {
            let positions: Vec<[f32; 3]> = snapshot.iter().map(|s| s.position).collect();
            let velocities: Vec<[f32; 3]> = snapshot.iter().map(|s| s.velocity).collect();
            let eff_radii: Vec<f32> = snapshot.iter().map(|s| s.radius).collect();
            // SnapEntry doesn't carry phenotype.max_axis(); approximate with
            // the renderer's BROAD_PHASE_SIZE_BUDGET / 2 (the CPU broad-phase
            // also uses `radius + BUDGET` rather than a per-cell max_axis).
            let max_axes: Vec<f32> = snapshot
                .iter()
                .map(|_| BROAD_PHASE_SIZE_BUDGET * 0.5)
                .collect();
            let adhesion_types: Vec<u32> =
                snapshot.iter().map(|s| s.adhesion_type as u32).collect();
            let slots = MAX_BONDS_PER_CELL;
            let total = n * slots;
            let mut partner_idx = vec![-1_i32; total];
            let mut rest = vec![0.0_f32; total];
            let mut stiff = vec![0.0_f32; total];
            let mut damp = vec![0.0_f32; total];
            for (i, s) in snapshot.iter().enumerate() {
                for (slot_idx, slot) in s.bonds.iter().enumerate() {
                    if let Some(b) = slot {
                        if let Some(&j) = id_to_idx_scratch.get(&b.other_cell_id) {
                            let idx = i * slots + slot_idx;
                            partner_idx[idx] = j as i32;
                            rest[idx] = b.rest_length;
                            stiff[idx] = b.stiffness;
                            damp[idx] = b.damping;
                        }
                    }
                }
            }
            gpu.cell_hash.dispatch(&positions);
            let result = gpu.collision.compute(
                &positions,
                &velocities,
                &eff_radii,
                &max_axes,
                &adhesion_types,
                &partner_idx,
                &rest,
                &stiff,
                &damp,
                &gpu.cell_hash,
            );
            let max_contacts = bioscape::MAX_COLLISION_CONTACTS_PER_CELL as usize;
            // Allocate one contact Vec per cell up-front so the canonicalized
            // dedup pass below can drop a partner into either i's or j's list
            // without coupling write order to iteration order.
            let mut contact_vecs: Vec<Vec<u64>> = (0..n).map(|_| Vec::new()).collect();
            for i in 0..n {
                let count = (result.contact_count[i] as usize).min(max_contacts);
                let base = i * max_contacts;
                let cell_id_i = snapshot[i].cell_id;
                for s in 0..count {
                    let j = result.contact_partners[base + s] as usize;
                    if j >= n {
                        continue;
                    }
                    let cell_id_j = snapshot[j].cell_id;
                    // CPU dedupe uses `cell_id_low < cell_id_high`; GPU uses
                    // `idx_low < idx_high`. Re-canonicalize so the lower-id
                    // cell always owns the partner reference.
                    if cell_id_i < cell_id_j {
                        contact_vecs[i].push(cell_id_j);
                    } else if cell_id_j < cell_id_i {
                        contact_vecs[j].push(cell_id_i);
                    }
                }
            }
            for (i, s) in snapshot.iter().enumerate() {
                results_scratch.push((
                    s.entity,
                    result.position_deltas[i],
                    result.velocity_deltas[i],
                    std::mem::take(&mut contact_vecs[i]),
                ));
            }
        }
    }
    if !used_gpu {
    {
        // Scoped `snapshot` borrow ends before Phase 2 mutates snapshot_scratch
        // (folds the post-Phase-2 `positions_scratch` rebuild into the existing
        // SoA snapshot).
        let snapshot = snapshot_scratch.as_slice();
    snapshot
        .par_iter()
        .map(|s_a| {
            let entity_a = s_a.entity;
            let pos_a = s_a.position;
            let vel_a = s_a.velocity;
            let radius_a = s_a.radius;
            let cell_id_a = s_a.cell_id;
            let collision_r = CELL_RADIUS * (radius_a + BROAD_PHASE_SIZE_BUDGET);
            let adhesion_r = collision_r * ADHESION_RANGE_FACTOR;
            let broad_r = collision_r.max(adhesion_r);
            let mut delta = [0.0_f32, 0.0_f32, 0.0_f32];
            let mut vel_delta = [0.0_f32, 0.0_f32, 0.0_f32];
            // Typical contact count per cell is 0–4 (kissing number bound for
            // overlapping spheres in 3D ≈ 12); pre-allocating avoids the
            // small-grow allocations on the rayon hot path.
            let mut local_contacts: Vec<u64> = Vec::with_capacity(8);
            grid_ref.for_each_in_radius_toroidal(
                pos_a,
                broad_r,
                WORLD_HALF,
                |entity_b, pos_b, radius_b| {
                    if entity_b == entity_a {
                        return;
                    }
                    let Some(&j_idx) = entity_to_idx.get(&entity_b) else {
                        return;
                    };
                    let pair_r = CELL_RADIUS * (radius_a + radius_b);
                    let pair_r2 = pair_r * pair_r;
                    let d_vec = bioscape::min_image_delta(pos_b, pos_a, WORLD_HALF);
                    let d2 = d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2];
                    let d = d2.sqrt();
                    let in_contact = d2 < pair_r2 && d2 > 0.0;
                    if in_contact {
                        let overlap = pair_r - d;
                        let inv_d = 1.0 / d;
                        let nx = d_vec[0] * inv_d;
                        let ny = d_vec[1] * inv_d;
                        let nz = d_vec[2] * inv_d;
                        let half_overlap = overlap * 0.5;
                        delta[0] += nx * half_overlap;
                        delta[1] += ny * half_overlap;
                        delta[2] += nz * half_overlap;
                        let vel_b = snapshot[j_idx].velocity;
                        let v_rel = [
                            vel_a[0] - vel_b[0],
                            vel_a[1] - vel_b[1],
                            vel_a[2] - vel_b[2],
                        ];
                        let v_rel_n = v_rel[0] * nx + v_rel[1] * ny + v_rel[2] * nz;
                        if v_rel_n < 0.0 {
                            let damp =
                                -v_rel_n * 0.5 * (1.0 - bioscape::COLLISION_RESTITUTION);
                            vel_delta[0] += damp * nx;
                            vel_delta[1] += damp * ny;
                            vel_delta[2] += damp * nz;
                        }
                        let cell_id_b = snapshot[j_idx].cell_id;
                        if cell_id_a < cell_id_b {
                            local_contacts.push(cell_id_b);
                        }
                    } else if d > 0.0 {
                        let same_type =
                            s_a.adhesion_type == snapshot[j_idx].adhesion_type;
                        let dv = adhesion_velocity_delta(d_vec, d, pair_r, same_type);
                        vel_delta[0] += dv[0];
                        vel_delta[1] += dv[1];
                        vel_delta[2] += dv[2];
                    }
                },
            );
            // Sprint 66: spring bond force pro každý živý bond.
            for bond_opt in s_a.bonds.iter() {
                if let Some(bond) = bond_opt {
                    if let Some(&j_idx) = id_to_idx.get(&bond.other_cell_id) {
                        let pos_j = snapshot[j_idx].position;
                        let vel_j = snapshot[j_idx].velocity;
                        let d_vec =
                            bioscape::min_image_delta(pos_j, pos_a, WORLD_HALF);
                        let dist =
                            (d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2])
                                .sqrt();
                        let (dv, _broken) =
                            bond_velocity_delta(bond, d_vec, dist, vel_a, vel_j);
                        vel_delta[0] += dv[0];
                        vel_delta[1] += dv[1];
                        vel_delta[2] += dv[2];
                    }
                }
            }
            (entity_a, delta, vel_delta, local_contacts)
        })
        .collect_into_vec(&mut results_scratch);
    } // end snapshot scope; Phase 2 below mutates snapshot_scratch
    } // end `if !used_gpu` (Wave H)
    let results = results_scratch.as_slice();

    // Phase 2 (sequential): apply deltas + bond age/prune + contact tracker
    // + bond formation.
    let dt = 1.0 / FIXED_TIMESTEP_HZ;
    seen_pairs_scratch.clear();
    let seen_pairs = &mut *seen_pairs_scratch;
    for (entity, delta, vel_delta, contacts) in results {
        let Ok((_, mut cell)) = cells.get_mut(*entity) else {
            continue;
        };
        cell.0.position[0] += delta[0];
        cell.0.position[1] += delta[1];
        cell.0.position[2] += delta[2];
        cell.0.velocity[0] += vel_delta[0];
        cell.0.velocity[1] += vel_delta[1];
        cell.0.velocity[2] += vel_delta[2];
        // Mirror the position update into the snapshot so the bond-pruning
        // phase below can read post-Phase-2 partner positions through
        // `id_to_idx` instead of rebuilding a separate cell_id→pos map.
        if let Some(&idx) = entity_to_idx.get(entity) {
            let snap_pos = &mut snapshot_scratch[idx].position;
            snap_pos[0] += delta[0];
            snap_pos[1] += delta[1];
            snap_pos[2] += delta[2];
        }
        let cell_id_a = cell.0.cell_id;
        for &other_id in contacts {
            let key = (cell_id_a, other_id);
            seen_pairs.insert(key);
            let entry = contact_progress.0.entry(key).or_insert(0);
            *entry = entry.saturating_add(1);
        }
    }
    // Bond pruning + maintenance — partner positions read from the synced
    // snapshot (no separate HashMap rebuild).
    for (_, mut cell) in cells.iter_mut() {
        let outputs_9 = cell.0.last_outputs[9];
        let explicit_break = outputs_9 < BOND_BREAK_THRESHOLD;
        let pos_i = cell.0.position;
        let mut bond_count = 0_usize;
        for slot in 0..MAX_BONDS_PER_CELL {
            let Some(bond) = cell.0.bonds[slot] else { continue };
            if explicit_break {
                cell.0.bonds[slot] = None;
                continue;
            }
            let pos_j = match id_to_idx.get(&bond.other_cell_id) {
                Some(&idx) => snapshot_scratch[idx].position,
                None => {
                    cell.0.bonds[slot] = None;
                    continue;
                }
            };
            let d_vec = bioscape::min_image_delta(pos_j, pos_i, WORLD_HALF);
            let d = (d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2]).sqrt();
            if d > bond.rest_length * bioscape::BOND_BREAK_FACTOR || d <= f32::EPSILON {
                cell.0.bonds[slot] = None;
                continue;
            }
            if let Some(b) = cell.0.bonds[slot].as_mut() {
                b.age_ticks = b.age_ticks.saturating_add(1);
            }
            bond_count += 1;
        }
        if bond_count > 0 {
            cell.0.energy -= bond_count as f32 * BOND_MAINTENANCE_PER_SEC * dt;
        }
    }
    // Contact tracker decay.
    contact_progress.0.retain(|key, ticks| {
        if seen_pairs.contains(key) {
            true
        } else if *ticks > CONTACT_DECAY_TICKS {
            *ticks -= CONTACT_DECAY_TICKS;
            true
        } else {
            false
        }
    });
    // Bond formation — kandidáti, kteří dosáhli BOND_FORM_TICKS thresholdu.
    candidates_scratch.clear();
    for (&pair, &ticks) in contact_progress.0.iter() {
        if ticks >= BOND_FORM_TICKS {
            candidates_scratch.push(pair);
        }
    }
    let snapshot = snapshot_scratch.as_slice();
    for &(id_a, id_b) in candidates_scratch.iter() {
        let Some(&i_a) = id_to_idx.get(&id_a) else { continue };
        let Some(&i_b) = id_to_idx.get(&id_b) else { continue };
        let sa = &snapshot[i_a];
        let sb = &snapshot[i_b];
        if sa.adhesion_type != sb.adhesion_type {
            continue;
        }
        let Ok([(_, mut ca), (_, mut cb)]) = cells.get_many_mut([sa.entity, sb.entity]) else {
            continue;
        };
        if ca.0.last_outputs[9] <= BOND_FORM_THRESHOLD
            || cb.0.last_outputs[9] <= BOND_FORM_THRESHOLD
        {
            continue;
        }
        let already = ca
            .0
            .bonds
            .iter()
            .any(|b| b.map(|bb| bb.other_cell_id == id_b).unwrap_or(false));
        if already {
            continue;
        }
        let slot_a = ca.0.bonds.iter().position(|b| b.is_none());
        let slot_b = cb.0.bonds.iter().position(|b| b.is_none());
        let (Some(sa_slot), Some(sb_slot)) = (slot_a, slot_b) else { continue };
        let pos_a = ca.0.position;
        let pos_b = cb.0.position;
        let d_vec = bioscape::min_image_delta(pos_b, pos_a, WORLD_HALF);
        let dist =
            (d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2]).sqrt();
        let rest = dist * BOND_REST_LENGTH_SLACK;
        // Sprint 68: per-bond stiffness/damping = mean obou cells' genes.
        let stiffness =
            (ca.0.genome.bond_stiffness + cb.0.genome.bond_stiffness) * 0.5;
        let damping = (ca.0.genome.bond_damping + cb.0.genome.bond_damping) * 0.5;
        ca.0.bonds[sa_slot] = Some(Bond {
            other_cell_id: id_b,
            rest_length: rest,
            stiffness,
            damping,
            age_ticks: 0,
        });
        cb.0.bonds[sb_slot] = Some(Bond {
            other_cell_id: id_a,
            rest_length: rest,
            stiffness,
            damping,
            age_ticks: 0,
        });
        ca.0.energy -= BOND_FORMATION_COST;
        cb.0.energy -= BOND_FORMATION_COST;
        contact_progress.0.remove(&(id_a, id_b));
    }
    diag.add_measurement(&DIAG_COLLISIONS, || t_total.elapsed().as_secs_f64() * 1000.0);
}
