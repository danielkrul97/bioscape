use bevy::diagnostic::Diagnostics;
use bevy::prelude::*;
use bioscape::{
    ATTACK_THRESHOLD, BOND_BREAK_THRESHOLD, BOND_FORMATION_COST, BOND_FORM_THRESHOLD,
    BOND_FORM_TICKS, BOND_MAINTENANCE_PER_SEC, BOND_REST_LENGTH_SLACK, Bond, CELL_RADIUS,
    CONTACT_DECAY_TICKS, DILUTION_K, FIXED_TIMESTEP_HZ, HERD_RADIUS, MAX_BONDS_PER_CELL,
    PREDATION_DRAIN_PER_TICK, PREDATION_GAIN_PER_TICK, SIZE_RATIO_THRESHOLD, WORLD_HALF,
};
use rustc_hash::FxHashSet;
use std::time::Instant;

use super::super::components::{CellEntity, Dying};
use super::super::config::{DIAG_COLLISIONS, DIAG_PREDATION};
use super::super::resources::ContactProgress;
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
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
    mut diag: Diagnostics,
    #[cfg(feature = "gpu")] gpu_full: Option<ResMut<GpuFullPipeline>>,
) {
    let t_total = Instant::now();

    // Wave H: GPU predation path. Computes herd + per-pair attack with
    // atomic energy/damage accumulation. Pack-hunting CSV diagnostics
    // (bonded/solo/swarm/pack) are NOT computed here — the shader doesn't
    // emit per-event tuples. CPU fallback still does the full metric set.
    #[cfg(feature = "gpu")]
    if let Some(mut gpu_res) = gpu_full {
        let gpu = &mut *gpu_res;
        let mut entities: Vec<Entity> = Vec::new();
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut eff_radii: Vec<f32> = Vec::new();
        let mut headings: Vec<f32> = Vec::new();
        let mut pitches: Vec<f32> = Vec::new();
        let mut attack_signals: Vec<f32> = Vec::new();
        let mut spike_counts: Vec<u32> = Vec::new();
        let mut spikes_packed: Vec<[f32; 4]> = Vec::new();
        for (e, c) in cells.iter() {
            entities.push(e);
            positions.push(c.0.position);
            eff_radii.push(c.0.phenotype.effective_radius());
            headings.push(c.0.heading);
            pitches.push(c.0.pitch);
            attack_signals.push(c.0.last_outputs[6].max(0.0));
            let mut active = 0u32;
            for s in 0..bioscape::SPIKE_SLOTS {
                let spike = c.0.phenotype.spikes[s];
                if spike.length > 0.0 {
                    active += 1;
                }
                spikes_packed.push([
                    spike.length,
                    spike.azimuth_offset,
                    spike.elevation_offset,
                    spike.complexity,
                ]);
            }
            spike_counts.push(active);
        }
        let n = entities.len();
        if n == 0 {
            diag.add_measurement(&DIAG_PREDATION, || {
                t_total.elapsed().as_secs_f64() * 1000.0
            });
            return;
        }
        let params = bioscape::gpu::PredateParamsGpu {
            num_cells: 0, // populated by compute()
            cell_size: bioscape::GRID_CELL_SIZE,
            cell_radius_const: CELL_RADIUS,
            size_ratio_threshold: SIZE_RATIO_THRESHOLD,
            herd_radius_sq: HERD_RADIUS * HERD_RADIUS,
            attack_threshold: ATTACK_THRESHOLD,
            predation_gain: PREDATION_GAIN_PER_TICK,
            predation_drain: PREDATION_DRAIN_PER_TICK,
            spike_dot_threshold: bioscape::SPIKE_DOT_THRESHOLD,
            spike_bonus: bioscape::SPIKE_PREDATION_BONUS,
            dilution_k: DILUTION_K,
            world_half_x: WORLD_HALF[0],
            world_half_y: WORLD_HALF[1],
            ..bioscape::gpu::PredateParamsGpu::default()
        };
        gpu.cell_hash.dispatch(&positions);
        let result = gpu.predate.compute(
            &positions,
            &eff_radii,
            &headings,
            &pitches,
            &spikes_packed,
            &spike_counts,
            &attack_signals,
            &gpu.cell_hash,
            params,
        );
        for (i, entity) in entities.iter().enumerate() {
            if let Ok((_, mut cell)) = cells.get_mut(*entity) {
                cell.0.energy += result.energy_delta[i];
                cell.0.damage_accum += result.damage_delta[i];
            }
        }
        diag.add_measurement(&DIAG_PREDATION, || t_total.elapsed().as_secs_f64() * 1000.0);
        return;
    }

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
