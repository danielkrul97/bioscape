use bevy::prelude::*;
use bioscape::{
    Bond, CELL_RADIUS, Cell, Hunter, WORLD_HALF, nearest_attackable_cell,
};
use rand::Rng;
use rustc_hash::{FxHashMap, FxHashSet};

use super::super::components::{CellEntity, Dying, FoodEntity, HunterEntity};
use super::super::resources::{
    Clock, FoodMaterial, FoodMesh, HunterCellGrid, HunterContactProgress, HunterMaterial,
    HunterMesh, NextHunterId, SmellResource, WorldExtent,
};

pub(crate) fn step_hunters(
    mut hunters: Query<(Entity, &mut HunterEntity)>,
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
    smell: Res<SmellResource>,
    fixed_time: Res<Time<Fixed>>,
    mut hunter_cell_grid: ResMut<HunterCellGrid>,
    mut entities_scratch: Local<Vec<Entity>>,
    mut cells_scratch: Local<Vec<Cell>>,
    mut hunters_snapshot_scratch: Local<Vec<bioscape::HunterSnapshotMin>>,
    mut attacks_scratch: Local<Vec<(Entity, f32)>>,
    mut pack_shares_scratch: Local<Vec<(u64, f32)>>,
) {
    let dt = fixed_time.delta_secs();
    // R-#8: single iter() do dvou paralelních scratchů (entity + cell). Pre-fix
    // double collect (`cell_snapshot` + odvozený `cells_only`) byl redundantní.
    entities_scratch.clear();
    cells_scratch.clear();
    for (e, c) in cells.iter() {
        entities_scratch.push(e);
        cells_scratch.push(c.0);
    }
    let cells_only = cells_scratch.as_slice();
    let cell_entities = entities_scratch.as_slice();
    // R-#3: persistent HunterCellGrid Resource — pre-fix `SpatialGrid::new()`
    // per tick. `rebuild` zachová bucket Vec capacity přes ticky.
    hunter_cell_grid.0.rebuild(
        cells_only
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.position, ())),
    );
    let hunter_cell_grid_ref = &hunter_cell_grid.0;
    // R-#9: persistent hunter snapshot scratch.
    hunters_snapshot_scratch.clear();
    hunters_snapshot_scratch.extend(
        hunters
            .iter()
            .map(|(_, h)| bioscape::HunterSnapshotMin::from_hunter(&h.0)),
    );
    let hunters_snapshot = hunters_snapshot_scratch.as_slice();
    // Persistent attack/pack_share scratchy.
    attacks_scratch.clear();
    pack_shares_scratch.clear();
    let attacks = &mut *attacks_scratch;
    let pack_shares = &mut *pack_shares_scratch;
    for (_, mut h) in &mut hunters {
        // Sprint 90: sensor gather + brain forward + hybrid motor (seek+brain).
        let sensors = bioscape::gather_hunter_sensors(
            &h.0,
            cells_only,
            hunter_cell_grid_ref,
            hunters_snapshot,
            &smell.0,
            WORLD_HALF,
        );
        let target_idx_pre =
            nearest_attackable_cell(&h.0, cells_only, hunter_cell_grid_ref, WORLD_HALF);
        let seek_target = target_idx_pre.map(|i| cells_only[i].position);
        let inputs = bioscape::populate_hunter_brain_inputs(&mut h.0, &sensors);
        let (hidden, outputs) = h.0.genome.brain.forward_with_state(&inputs);
        h.0.last_inputs = inputs;
        h.0.last_hidden = hidden;
        h.0.last_outputs = outputs;
        h.0.apply_brain_motor(&outputs, seek_target, dt, WORLD_HALF);
        h.0.step(dt, WORLD_HALF);
        // Attack check (post-step pozice).
        let target_idx =
            nearest_attackable_cell(&h.0, cells_only, hunter_cell_grid_ref, WORLD_HALF);
        let attack_r = h.0.genome.attack_radius;
        let attack_r2 = attack_r * attack_r;
        let damage = h.0.genome.damage_per_tick;
        let mut gain = 0.0_f32;
        if let Some(i) = target_idx {
            let d = bioscape::min_image_delta(
                h.0.position,
                cells_only[i].position,
                WORLD_HALF,
            );
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if d2 < attack_r2 {
                // Sprint 92: edge-vulnerability — damage scales s exposure
                // (= 1 - n_bonds × EXPOSURE_PER_BOND). Solo cells take full
                // damage, deeply interior cells take 0.
                let exposure = bioscape::cell_exposure(cells_only[i].n_bonds());
                let damage_dealt = damage * exposure * dt;
                attacks.push((cell_entities[i], damage_dealt));
                gain = damage_dealt * bioscape::HUNTER_ENERGY_PER_DAMAGE;
                // Sprint 101: pack share queue.
                for bond_opt in h.0.bonds.iter() {
                    if let Some(bond) = bond_opt {
                        pack_shares.push((
                            bond.other_cell_id,
                            gain * bioscape::HUNTER_BOND_KILL_SHARE_FRAC,
                        ));
                    }
                }
            }
        }
        h.0.apply_energy_costs(dt);
        h.0.energy += gain;
    }
    // Sprint 101: distribute pack shares.
    if !pack_shares.is_empty() {
        let id_to_entity: FxHashMap<u64, Entity> = hunters
            .iter()
            .map(|(e, h)| (h.0.hunter_id, e))
            .collect();
        for &(id, energy) in pack_shares.iter() {
            if let Some(&entity) = id_to_entity.get(&id) {
                if let Ok((_, mut h)) = hunters.get_mut(entity) {
                    h.0.energy += energy;
                }
            }
        }
    }
    for &(entity, damage) in attacks.iter() {
        if let Ok((_, mut cell)) = cells.get_mut(entity) {
            cell.0.energy -= damage;
            cell.0.damage_accum += damage;
        }
    }
}

/// Sprint 89: hunter lifecycle — death (energy ≤ 0 → drop carrion + despawn),
/// reproduce (energy ≥ THRESHOLD + cooldown 0 → split + clone+mutate child),
/// floor respawn (n_hunters == 0 → 1 fresh genome). MAX_POP cap brání runaway.
/// Asexual v1 — Sprint 91+ může přidat sexual pairing.
pub(crate) fn hunters_lifecycle(
    hunters: Query<(Entity, &HunterEntity)>,
    extent: Res<WorldExtent>,
    hunter_mesh: Res<HunterMesh>,
    hunter_material: Res<HunterMaterial>,
    food_mesh: Res<FoodMesh>,
    food_material: Res<FoodMaterial>,
    clock: Res<Clock>,
    mut next_hunter_id: ResMut<NextHunterId>,
    mut commands: Commands,
    mut alive_scratch: Local<Vec<(Entity, Hunter)>>,
    mut fertile_scratch: Local<Vec<(Entity, [f32; 3])>>,
    mut lookup_scratch: Local<FxHashMap<Entity, Hunter>>,
) {
    let mut rng = rand::rng();
    let half = extent.as_array();
    let current_gen = clock.0.generation;
    alive_scratch.clear();
    alive_scratch.extend(hunters.iter().map(|(e, h)| (e, h.0)));
    let alive = alive_scratch.as_slice();

    // Floor respawn: pokud all extinct, spawn 1 fresh genome (předchází total
    // predator collapse blokující arms race).
    if alive.is_empty() {
        let id = next_hunter_id.0;
        next_hunter_id.0 += 1;
        let h = Hunter::random(&mut rng, half, id, id, current_gen);
        commands.spawn((
            HunterEntity(h),
            Mesh3d(hunter_mesh.0.clone()),
            MeshMaterial3d(hunter_material.0.clone()),
            Transform::from_xyz(h.position[0], h.position[1], h.position[2]),
        ));
        return;
    }

    // Death pass.
    for (entity, h) in alive {
        if h.energy <= 0.0 {
            commands.entity(*entity).despawn();
            for _ in 0..bioscape::HUNTER_CARRION_DROP {
                let pos = [
                    (h.position[0] + rng.random_range(-CELL_RADIUS..CELL_RADIUS))
                        .clamp(-half[0], half[0]),
                    (h.position[1] + rng.random_range(-CELL_RADIUS..CELL_RADIUS))
                        .clamp(-half[1], half[1]),
                    h.position[2].clamp(-half[2], half[2]),
                ];
                commands.spawn((
                    FoodEntity(bioscape::Food {
                        position: pos,
                        age_ticks: 0,
                        kind: bioscape::FoodKind::HunterCarrion,
                    }),
                    Mesh3d(food_mesh.0.clone()),
                    MeshMaterial3d(food_material.0.clone()),
                    Transform::from_xyz(pos[0], pos[1], pos[2]),
                    Visibility::Hidden,
                ));
            }
        }
    }

    // Sprint 98: sexual reproduction. Mirror headless logic — pair fertile
    // entities přes prostorovou blízkost, každý pár → 1 mating child, oba
    // rodiče zaplatí (halve + cooldown). Floor respawn nahoře pokrývá total
    // extinction.
    let alive_count = alive.iter().filter(|(_, h)| h.energy > 0.0).count();
    let budget = bioscape::HUNTER_MAX_POP.saturating_sub(alive_count);
    if budget == 0 {
        return;
    }
    fertile_scratch.clear();
    fertile_scratch.extend(
        alive
            .iter()
            .filter(|(_, h)| {
                h.energy >= bioscape::HUNTER_REPRODUCE_THRESHOLD
                    && h.reproduce_cooldown_ticks == 0
            })
            .map(|(e, h)| (*e, h.position)),
    );
    if fertile_scratch.len() < 2 {
        return;
    }
    let mating_r2 = bioscape::HUNTER_MATING_RADIUS * bioscape::HUNTER_MATING_RADIUS;
    let matings = bioscape::pair_fertile(fertile_scratch.as_slice(), mating_r2, budget, WORLD_HALF);
    lookup_scratch.clear();
    lookup_scratch.extend(alive.iter().map(|(e, h)| (*e, *h)));
    let lookup = &*lookup_scratch;
    for &(ea, eb) in &matings {
        let parent_a = match lookup.get(&ea) {
            Some(p) => *p,
            None => continue,
        };
        let parent_b = match lookup.get(&eb) {
            Some(p) => *p,
            None => continue,
        };
        // Halve both parents PŘED make_*_mating_child (energy semantics z cell
        // mating: child.energy = a + b součet už halved values, takže celkem
        // konzervovaná energy a + b post-mating).
        let mut a_halved = parent_a;
        let mut b_halved = parent_b;
        a_halved.energy *= 0.5;
        b_halved.energy *= 0.5;
        let id = next_hunter_id.0;
        next_hunter_id.0 += 1;
        let child = bioscape::make_hunter_mating_child(
            &a_halved,
            &b_halved,
            &mut rng,
            half,
            id,
            current_gen,
        );
        commands.spawn((
            HunterEntity(child),
            Mesh3d(hunter_mesh.0.clone()),
            MeshMaterial3d(hunter_material.0.clone()),
            Transform::from_xyz(child.position[0], child.position[1], child.position[2]),
        ));
        // Update parent ECS components: halved energy + cooldown.
        a_halved.reproduce_cooldown_ticks = bioscape::HUNTER_REPRODUCE_COOLDOWN_TICKS;
        b_halved.reproduce_cooldown_ticks = bioscape::HUNTER_REPRODUCE_COOLDOWN_TICKS;
        commands.entity(ea).insert(HunterEntity(a_halved));
        commands.entity(eb).insert(HunterEntity(b_halved));
    }
}

/// Sprint 71: sync Hunter.position → Transform pro renderer. Hunter mesh
/// se nerotuje (sphere — žádná orientace), takže Transform se updatuje jen
/// translation. Mesh scale je fixed v setup, neměnné per-tick.
pub(crate) fn sync_hunter_transforms(mut hunters: Query<(&HunterEntity, &mut Transform)>) {
    for (h, mut transform) in &mut hunters {
        transform.translation = Vec3::new(h.0.position[0], h.0.position[1], h.0.position[2]);
    }
}

/// Sprint 100: pool last_hidden napříč hunter packem (mirror headless
/// `pool_bonded_hunter_hidden`). Snapshot all → call lib helper → write
/// back via commands.entity().insert. ECS-flavored wrapper.
pub(crate) fn pool_bonded_hunter_hidden_system(
    hunters: Query<(Entity, &HunterEntity)>,
    mut commands: Commands,
    mut entities_scratch: Local<Vec<Entity>>,
    mut hunters_only_scratch: Local<Vec<Hunter>>,
) {
    // R-#10: collapse double snapshot na single iter() do dvou paralelních
    // scratchů. Pre-fix `state` + odvozený `hunters_only` byly redundantní.
    entities_scratch.clear();
    hunters_only_scratch.clear();
    for (e, h) in hunters.iter() {
        entities_scratch.push(e);
        hunters_only_scratch.push(h.0);
    }
    if hunters_only_scratch.is_empty() {
        return;
    }
    bioscape::pool_bonded_hunter_hidden(&mut hunters_only_scratch);
    for (entity, updated) in entities_scratch.iter().zip(hunters_only_scratch.iter()) {
        commands.entity(*entity).insert(HunterEntity(*updated));
    }
}

/// Sprint 99: hunter-hunter collision + adhesion + bond physics. Mirror
/// headless `resolve_hunter_collisions` — O(N²) sequential pro N ≤ 50.
/// Snapshot all hunters → compute deltas + bond formation/pruning →
/// write back via `commands.entity().insert()`.
pub(crate) fn resolve_hunter_collisions(
    hunters: Query<(Entity, &HunterEntity)>,
    mut contact: ResMut<HunterContactProgress>,
    mut commands: Commands,
    mut alive_scratch: Local<Vec<(Entity, Hunter)>>,
    mut id_to_pos_scratch: Local<FxHashMap<u64, usize>>,
    mut pos_deltas_scratch: Local<Vec<[f32; 3]>>,
    mut vel_deltas_scratch: Local<Vec<[f32; 3]>>,
    mut in_contact_pairs_scratch: Local<FxHashSet<(u64, u64)>>,
    mut new_progress_scratch: Local<FxHashMap<(u64, u64), u32>>,
    mut bond_candidates_scratch: Local<Vec<(u64, u64)>>,
) {
    // R-#9: persistent Local scratchy. Pre-fix: 6 fresh allocs per tick.
    alive_scratch.clear();
    alive_scratch.extend(hunters.iter().map(|(e, h)| (e, h.0)));
    let alive = alive_scratch.as_slice();
    let n = alive.len();
    if n < 2 {
        return;
    }
    let hunter_radius = |h: &Hunter| h.genome.body_size * CELL_RADIUS;
    id_to_pos_scratch.clear();
    for (i, (_, h)) in alive.iter().enumerate() {
        id_to_pos_scratch.insert(h.hunter_id, i);
    }
    let id_to_pos = &*id_to_pos_scratch;

    pos_deltas_scratch.clear();
    pos_deltas_scratch.resize(n, [0.0; 3]);
    vel_deltas_scratch.clear();
    vel_deltas_scratch.resize(n, [0.0; 3]);
    in_contact_pairs_scratch.clear();
    let pos_deltas = &mut *pos_deltas_scratch;
    let vel_deltas = &mut *vel_deltas_scratch;
    let in_contact_pairs = &mut *in_contact_pairs_scratch;

    for i in 0..n {
        let (_, hunter_i) = &alive[i];
        let pos_i = hunter_i.position;
        let vel_i = hunter_i.velocity;
        let radius_i = hunter_radius(hunter_i);
        let type_i = hunter_i.genome.adhesion_type;
        let id_i = hunter_i.hunter_id;
        for j in 0..n {
            if i == j {
                continue;
            }
            let (_, hunter_j) = &alive[j];
            let pos_j = hunter_j.position;
            let radius_j = hunter_radius(hunter_j);
            let pair_r = radius_i + radius_j;
            let d_vec = bioscape::min_image_delta(pos_j, pos_i, WORLD_HALF);
            let d2 = d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2];
            let d = d2.sqrt();
            let in_contact = d2 < pair_r * pair_r && d2 > 0.0;
            if in_contact {
                let overlap = pair_r - d;
                let nx = d_vec[0] / d;
                let ny = d_vec[1] / d;
                let nz = d_vec[2] / d;
                pos_deltas[i][0] -= nx * overlap * 0.5;
                pos_deltas[i][1] -= ny * overlap * 0.5;
                pos_deltas[i][2] -= nz * overlap * 0.5;
                let id_j = hunter_j.hunter_id;
                let pair = if id_i < id_j { (id_i, id_j) } else { (id_j, id_i) };
                in_contact_pairs.insert(pair);
            } else if d > 0.0 {
                let type_j = hunter_j.genome.adhesion_type;
                let same_type = type_i == type_j;
                let dv = bioscape::adhesion_velocity_delta(d_vec, d, pair_r, same_type);
                vel_deltas[i][0] += dv[0];
                vel_deltas[i][1] += dv[1];
                vel_deltas[i][2] += dv[2];
            }
        }
        // Apply bond spring forces.
        for bond_opt in hunter_i.bonds.iter() {
            if let Some(bond) = bond_opt {
                if let Some(&j_idx) = id_to_pos.get(&bond.other_cell_id) {
                    let (_, hunter_j) = &alive[j_idx];
                    let pos_j = hunter_j.position;
                    let vel_j = hunter_j.velocity;
                    let d_vec = bioscape::min_image_delta(pos_j, pos_i, WORLD_HALF);
                    let dist = (d_vec[0] * d_vec[0]
                        + d_vec[1] * d_vec[1]
                        + d_vec[2] * d_vec[2])
                        .sqrt();
                    let (dv, _broken) =
                        bioscape::bond_velocity_delta(bond, d_vec, dist, vel_i, vel_j);
                    vel_deltas[i][0] += dv[0];
                    vel_deltas[i][1] += dv[1];
                    vel_deltas[i][2] += dv[2];
                }
            }
        }
    }

    // Contact tracker update (mirror headless). Reuse new_progress_scratch
    // přes std::mem::swap — pre-fix byla `FxHashMap::default()` per tick.
    new_progress_scratch.clear();
    for &pair in in_contact_pairs.iter() {
        let prev = contact.0.get(&pair).copied().unwrap_or(0);
        new_progress_scratch.insert(pair, prev.saturating_add(1));
    }
    for (&pair, &val) in contact.0.iter() {
        if !in_contact_pairs.contains(&pair) && val > 1 {
            new_progress_scratch.insert(pair, val - 1);
        }
    }
    std::mem::swap(&mut contact.0, &mut *new_progress_scratch);

    // R-#9: dropped `alive.clone()` — mutuj alive_scratch in-place.
    for ((entity_pair, pd), vd) in alive_scratch
        .iter_mut()
        .zip(pos_deltas_scratch.iter())
        .zip(vel_deltas_scratch.iter())
    {
        let h = &mut entity_pair.1;
        h.position[0] += pd[0];
        h.position[1] += pd[1];
        h.position[2] += pd[2];
        h.velocity[0] += vd[0];
        h.velocity[1] += vd[1];
        h.velocity[2] += vd[2];
    }

    // Bond formation — collect candidate pair IDs into a persistent scratch
    // so the immutable borrow on `contact` ends before we mutate `alive_scratch`.
    bond_candidates_scratch.clear();
    bond_candidates_scratch.extend(
        contact
            .0
            .iter()
            .filter(|(_, &t)| t >= bioscape::BOND_FORM_TICKS)
            .map(|(&pair, _)| pair),
    );
    for &(id_a, id_b) in bond_candidates_scratch.iter() {
        let (Some(&a_idx), Some(&b_idx)) =
            (id_to_pos.get(&id_a), id_to_pos.get(&id_b))
        else {
            continue;
        };
        if alive_scratch[a_idx].1.genome.adhesion_type
            != alive_scratch[b_idx].1.genome.adhesion_type
        {
            continue;
        }
        // Sprint 100: brain output[9] gate.
        if alive_scratch[a_idx].1.last_outputs[9] < bioscape::BOND_FORM_THRESHOLD
            || alive_scratch[b_idx].1.last_outputs[9] < bioscape::BOND_FORM_THRESHOLD
        {
            continue;
        }
        let already = alive_scratch[a_idx]
            .1
            .bonds
            .iter()
            .any(|b| b.as_ref().map_or(false, |bb| bb.other_cell_id == id_b));
        if already {
            continue;
        }
        let slot_a = alive_scratch[a_idx]
            .1
            .bonds
            .iter()
            .position(|b| b.is_none());
        let slot_b = alive_scratch[b_idx]
            .1
            .bonds
            .iter()
            .position(|b| b.is_none());
        if let (Some(sa), Some(sb)) = (slot_a, slot_b) {
            let pos_a = alive_scratch[a_idx].1.position;
            let pos_b = alive_scratch[b_idx].1.position;
            let d_vec = bioscape::min_image_delta(pos_b, pos_a, WORLD_HALF);
            let dist =
                (d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2]).sqrt();
            let rest = dist * bioscape::BOND_REST_LENGTH_SLACK;
            alive_scratch[a_idx].1.bonds[sa] = Some(Bond {
                other_cell_id: id_b,
                rest_length: rest,
                stiffness: bioscape::BOND_STIFFNESS,
                damping: bioscape::BOND_DAMPING,
                age_ticks: 0,
            });
            alive_scratch[b_idx].1.bonds[sb] = Some(Bond {
                other_cell_id: id_a,
                rest_length: rest,
                stiffness: bioscape::BOND_STIFFNESS,
                damping: bioscape::BOND_DAMPING,
                age_ticks: 0,
            });
        }
    }

    // Pruning + age increment.
    for (_, hunter) in alive_scratch.iter_mut() {
        for bond_opt in hunter.bonds.iter_mut() {
            if let Some(bond) = bond_opt {
                if !id_to_pos.contains_key(&bond.other_cell_id) {
                    *bond_opt = None;
                } else {
                    bond.age_ticks = bond.age_ticks.saturating_add(1);
                }
            }
        }
    }

    // Writeback ECS state.
    for (entity, h) in alive_scratch.iter() {
        commands.entity(*entity).insert(HunterEntity(*h));
    }
}
