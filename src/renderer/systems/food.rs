use bevy::diagnostic::Diagnostics;
use bevy::prelude::*;
use bioscape::{
    EAT_RADIUS, FOOD_SPAWN_RATE, Food, LEARNING_RATE, MAX_BODY_LENGTH, MAX_SPAWN_ATTEMPTS,
    WORLD_HALF, reject_food_for_richness,
};
use rand::Rng;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Instant;

use super::super::components::{CellEntity, Dying, FoodEntity};
use super::super::config::DIAG_EAT_FOOD;
use super::super::resources::{
    CellEntityLookups, CellGrid, CellSlotMap, Clock, CoopFoodResource, FoodDensityFactor, FoodGrid,
    FoodMaterial, FoodMesh, WorldExtent, WorldMapResource,
};
use super::super::resources_gpu::GpuFullPipeline;
use super::super::world_map::{food_multiplier, food_target};

pub(crate) fn spawn_food(
    foods: Query<(), With<FoodEntity>>,
    cells: Query<&CellEntity, Without<Dying>>,
    extent: Res<WorldExtent>,
    factor: Res<FoodDensityFactor>,
    cell_grid: Res<CellGrid>,
    world_map: Res<WorldMapResource>,
    food_mesh: Res<FoodMesh>,
    food_material: Res<FoodMaterial>,
    mut commands: Commands,
) {
    let target = food_target(&extent, factor.0);
    let count = foods.iter().count();
    if count >= target {
        return;
    }
    let to_spawn = (target - count).min(FOOD_SPAWN_RATE);
    let mut rng = rand::rng();
    let half = extent.as_array();
    // Sprint 41: bump broad-phase budget na MAX_BODY_LENGTH — worst-case max_axis
    // ellipsoid může extending podél long axis až o tuto velikost. BROAD_PHASE_SIZE_BUDGET
    // = 3.0 (effective_radius default) by missnul cells s max_axis blízko 4.0.
    let broad_r = EAT_RADIUS * MAX_BODY_LENGTH;

    'spawn: for _ in 0..to_spawn {
        for _ in 0..MAX_SPAWN_ATTEMPTS {
            let candidate = Food::random(&mut rng, half);
            // Sprint 31: rejection sampling proti uniform — bias k rich zonám.
            // Spotřebovává retry budget jako cell-exclusion check níž.
            let richness = world_map
                .0
                .sample([candidate.position[0], candidate.position[1], 0.0]);
            if reject_food_for_richness(&mut rng, richness) {
                continue;
            }
            let mut blocked = false;
            cell_grid.0.for_each_in_radius_toroidal(
                candidate.position,
                broad_r,
                WORLD_HALF,
                |entity, cell_pos, _radius| {
                    if blocked {
                        return;
                    }
                    // Match headless: exclusion uses ellipsoid's max_axis, not
                    // effective_radius. Elongated cells extend past their sphere
                    // approximation along the long axis, so a sphere-radius
                    // exclusion would let food spawn inside the ellipsoid.
                    let max_axis = cells
                        .get(entity)
                        .map(|c| c.0.phenotype.max_axis())
                        .unwrap_or(MAX_BODY_LENGTH);
                    let exclusion = EAT_RADIUS * max_axis;
                    let d = bioscape::min_image_delta(candidate.position, cell_pos, WORLD_HALF);
                    if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < exclusion * exclusion {
                        blocked = true;
                    }
                },
            );
            if !blocked {
                commands.spawn((
                    FoodEntity(candidate),
                    Mesh3d(food_mesh.0.clone()),
                    MeshMaterial3d(food_material.0.clone()),
                    Transform::from_xyz(
                        candidate.position[0],
                        candidate.position[1],
                        candidate.position[2],
                    ),
                    Visibility::Hidden,
                ));
                continue 'spawn;
            }
        }
        // All MAX_SPAWN_ATTEMPTS rolls fell inside someone's eat radius — skip
        // this slot. Population is dense enough that the food world is at
        // (ecological) saturation; we'll catch up later when cells move.
    }
}

/// Sprint 128: per-tick spawn coop food node pokud pod cap. Single Bernoulli
/// draw; pozice uniform world bounds (žádný richness check — coop nodes nejsou
/// vázané na food density mapu).
pub(crate) fn spawn_coop_food(
    extent: Res<WorldExtent>,
    mut coop: ResMut<CoopFoodResource>,
    clock: Res<Clock>,
) {
    if coop.0.len() >= bioscape::COOP_FOOD_MAX_CONCURRENT {
        return;
    }
    let mut rng = rand::rng();
    if rng.random::<f32>() >= bioscape::COOP_FOOD_SPAWN_RATE_PER_TICK {
        return;
    }
    let pos = bioscape::random_coop_position(&mut rng, extent.as_array());
    coop.0
        .push(bioscape::CoopFood::new(pos, clock.0.tick));
}

/// Sprint 128: per-tick arrival registration → trigger pokus → cleanup.
/// Cells in arrival radius dostávají reward při dosažení threshold; expirace
/// odstraní node bez reward. Counter logika (per-gen solved/failed) tu není —
/// renderer pouze visualizuje, headless drží authoritative metrics.
pub(crate) fn update_coop_food(
    mut coop: ResMut<CoopFoodResource>,
    cells: Query<&CellEntity, Without<Dying>>,
    clock: Res<Clock>,
    mut cell_snapshot_scratch: Local<Vec<(u64, [f32; 3])>>,
) {
    if coop.0.is_empty() {
        return;
    }
    let r2 = bioscape::COOP_FOOD_ARRIVAL_RADIUS * bioscape::COOP_FOOD_ARRIVAL_RADIUS;
    // R-#12: persistent Local snapshot — pre-fix fresh `Vec` collect per tick.
    cell_snapshot_scratch.clear();
    cell_snapshot_scratch.extend(cells.iter().map(|c| (c.0.cell_id, c.0.position)));
    let cell_snapshot = cell_snapshot_scratch.as_slice();
    for c in coop.0.iter_mut() {
        if c.triggered {
            continue;
        }
        for (id, pos) in cell_snapshot.iter() {
            let d = bioscape::min_image_delta(c.position, *pos, WORLD_HALF);
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if d2 <= r2 {
                let _ = bioscape::register_coop_arrival(c, *id);
            }
        }
    }
    let current_tick = clock.0.tick;
    // Reward distribution + cleanup. Two-pass: detect indices to mutate, pak
    // apply na cells v separate query (Cells query je `Without<Dying>` immutable
    // tady, takže reward write přes side query není možný v 1 systému).
    // V této první iteraci coop reward v rendereru čistě cleanup-only —
    // headless drží authoritative reward distribuci, renderer slouží jako
    // visualization. Pro plný coupling v rendereru je nutný separátní system
    // s `Query<&mut CellEntity>` — odloženo do pozdější iterace.
    let mut i = 0;
    while i < coop.0.len() {
        let arrivals = coop.0[i].arrivals.len();
        if arrivals >= bioscape::COOP_FOOD_REQUIRED_ARRIVALS && !coop.0[i].triggered {
            coop.0[i].triggered = true;
        }
        if coop.0[i].triggered || coop.0[i].is_expired(current_tick) {
            coop.0.swap_remove(i);
            continue;
        }
        i += 1;
    }
}

pub(crate) fn cell_eats_food(
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
    food_grid: Res<FoodGrid>,
    world_map: Res<WorldMapResource>,
    slot_map: Res<CellSlotMap>,
    lookups: Res<CellEntityLookups>,
    gpu_full: Option<ResMut<GpuFullPipeline>>,
    mut commands: Commands,
    mut diag: Diagnostics,
    mut snapshot_scratch: Local<Vec<(Entity, [f32; 3], f32, [f32; 3], f32, f32, f32, f32, bool)>>,
    mut cell_carnivore_scratch: Local<FxHashMap<Entity, f32>>,
    mut candidates_scratch: Local<Vec<Option<(Entity, f32)>>>,
    mut eaten_scratch: Local<FxHashSet<Entity>>,
    mut share_deltas_scratch: Local<Vec<(Entity, f32)>>,
    mut rewards_scratch: Local<Vec<f32>>,
) {
    let t_total = Instant::now();

    // Sprint 52: GPU Hebbian dispatch consumes rewards Vec[N]. The
    // GpuFullPipeline carries its own Hebbian instance.
    let use_gpu_hebbian = gpu_full.is_some();

    rewards_scratch.clear();
    if use_gpu_hebbian {
        rewards_scratch.resize(slot_map.len(), 0.0);
    }
    let rewards = &mut *rewards_scratch;

    // Sprint 58: 3-pass refactor (mirror Sprint 57 headless eat_food).
    // Pass 1 (par): per-cell candidate selection. Snapshot s pose data pro
    // toroidal-aware eat_test_pose (lib helper bez `&Cell`).
    // R-#13: Local scratch pro snapshot/cell_carnivore/candidates — pre-fix
    // 3 fresh allocs per tick.
    snapshot_scratch.clear();
    cell_carnivore_scratch.clear();
    for (e, c) in cells.iter() {
        let has_bonds = c.0.bonds.iter().any(|b| b.is_some());
        snapshot_scratch.push((
            e,
            c.0.position,
            c.0.phenotype.max_axis(),
            [
                c.0.phenotype.body_length,
                c.0.phenotype.body_width,
                c.0.phenotype.body_height,
            ],
            c.0.heading,
            c.0.pitch,
            c.0.last_best_food_d2,
            c.0.genome.vision_radius,
            has_bonds,
        ));
        cell_carnivore_scratch.insert(e, c.0.genome.carnivore_score);
    }
    let snapshot = snapshot_scratch.as_slice();
    let food_grid_ref = &food_grid.0;
    let cell_carnivore = &*cell_carnivore_scratch;
    candidates_scratch.clear();
    // eat_food skip optim: gate `!has_bonds` chrání determinismus — bonded cells
    // v dense clusterech mohou mít spring-impulse pohyb > 30 jednotek/tick, což
    // by změnilo first-cell-wins ordering v Pass 2. Solo cells mají
    // predictable kinetiku (velocity + drag + brownian, ~5 jednotek/tick).
    const EAT_FOOD_MOVE_SLACK: f32 = 10.0;
    snapshot
        .par_iter()
        .map(|(entity, pos, max_axis, dims, heading, pitch, last_best_food_d2, vision_r, has_bonds)| {
            let eat_r = EAT_RADIUS * *max_axis;
            let skip_threshold = eat_r + EAT_FOOD_MOVE_SLACK;
            let skip_threshold_sq = skip_threshold * skip_threshold;
            let sensor_covers = vision_r * vision_r >= skip_threshold_sq;
            if !has_bonds && sensor_covers && *last_best_food_d2 > skip_threshold_sq {
                return None;
            }
            let carnivore_score = cell_carnivore.get(entity).copied().unwrap_or(0.0);
            let mut ate: Option<(Entity, f32)> = None;
            food_grid_ref.for_each_in_radius_toroidal(
                *pos,
                eat_r,
                WORLD_HALF,
                |food_e, food_pos, food_kind| {
                    if ate.is_some() {
                        return;
                    }
                    // Sprint 54: ghost food s min-imaged position pro toroidal eat_test.
                    let md = bioscape::min_image_delta(*pos, food_pos, WORLD_HALF);
                    let ghost_pos = [pos[0] + md[0], pos[1] + md[1], food_pos[2]];
                    if bioscape::eat_test_pose(*pos, *heading, *pitch, *dims, ghost_pos, EAT_RADIUS) {
                        // Sprint 92: food value = base_value(kind) × multiplier × value_factor
                        // × eat_efficiency(kind, carnivore_score). Carrion vyžaduje
                        // carnivore digestion; plant je herbivore-friendly.
                        let efficiency = bioscape::eat_efficiency(food_kind, carnivore_score);
                        let value = bioscape::food_base_value(food_kind)
                            * food_multiplier(
                                world_map.0.sample([food_pos[0], food_pos[1], 0.0]),
                            )
                            * Food {
                                position: food_pos,
                                age_ticks: 0,
                                kind: food_kind,
                            }
                            .value_factor()
                            * efficiency;
                        ate = Some((food_e, value));
                    }
                },
            );
            ate
        })
        .collect_into_vec(&mut candidates_scratch);
    let candidates = candidates_scratch.as_slice();

    // R-#2: id_to_entity z persistent CellEntityLookups (built v
    // `rebuild_cell_entity_lookups` na začátku ticku). Cells layout je stable
    // od tady přes celé eat_food.
    let id_to_entity = &lookups.id_to_entity;

    // Pass 2 (sequential): resolve race + apply energy + Hebbian. First-cell-wins
    // per food entity (matches pre-Sprint-58 ordering).
    // Sprint 78: cluster food share. Sebráno do share_deltas Vec během iterace,
    // aplikováno post-loop kvůli simultaneous mutable borrow.
    eaten_scratch.clear();
    share_deltas_scratch.clear();
    let eaten = &mut *eaten_scratch;
    let share_deltas = &mut *share_deltas_scratch;
    for ((entity, _, _, _, _, _, _, _, _), opt) in snapshot.iter().zip(candidates.iter()) {
        if let Some((food_e, value)) = opt {
            if eaten.contains(food_e) {
                continue;
            }
            eaten.insert(*food_e);
            let (bonds_copy, donor_state) = if let Ok((_, mut cell)) = cells.get_mut(*entity) {
                cell.0.energy += *value;
                let copy = cell.0.bonds;
                let state = cell.0.cell_state;
                if use_gpu_hebbian {
                    if let Some(slot) = slot_map.slot_of(*entity) {
                        if slot < rewards.len() {
                            rewards[slot] = 1.0;
                        }
                    }
                } else {
                    let last_hidden = cell.0.last_hidden;
                    let last_outputs = cell.0.last_outputs;
                    // Wave 3: trace-based reward — credits motor outputs from
                    // up to ~120 ticks back (1/HEBBIAN_TRACE_DECAY_PER_SEC at
                    // 60 Hz), not just this tick's pre·post.
                    cell.0.genome.brain.hebbian_apply_reward(
                        &last_hidden,
                        &last_outputs,
                        1.0,
                        LEARNING_RATE,
                    );
                }
                (copy, state)
            } else {
                continue;
            };
            // Sprint 78: share s bonded partnery (free reward, no
            // conservation — modeluje tissue cooperation).
            // Sprint 80: donor's cell_state moduluje fraction. State≈0
            // (selfish) → ~0% share; state≈1 (altruist) → plný 30% share.
            // Sprint 87: cluster-size bonus — cells hluboko v tkáni sdílí
            // víc, posiluje selekci proti tissue-regime collapse.
            let n_bonds = bonds_copy.iter().filter(|b| b.is_some()).count() as f32;
            let cluster_mult = 1.0 + (n_bonds - 1.0).max(0.0)
                * bioscape::BOND_FOOD_SHARE_CLUSTER_BONUS;
            let share_value =
                *value * bioscape::BOND_FOOD_SHARE_FRAC * donor_state * cluster_mult;
            if share_value > 0.0 {
                for bond_opt in bonds_copy.iter() {
                    if let Some(bond) = bond_opt {
                        if let Some(&partner_e) =
                            id_to_entity.get(&bond.other_cell_id)
                        {
                            if partner_e != *entity {
                                share_deltas.push((partner_e, share_value));
                            }
                        }
                    }
                }
            }
        }
    }

    // Sprint 78: aplikuj food share delty (po Pass 2 main loop).
    for &(e, delta) in share_deltas.iter() {
        if let Ok((_, mut cell)) = cells.get_mut(e) {
            cell.0.energy += delta;
        }
    }

    // Pass 3: main-thread Commands flush (despawn nelze v par_iter).
    for food_e in eaten.iter() {
        commands.entity(*food_e).despawn();
    }

    // Wave 7: trace-based reward apply replaces the legacy instantaneous
    // `compute_persistent`. The GPU hebbian_step pass (run per-tick from
    // `apply_eligibility_step`) has been decaying + accumulating
    // `cells.brain_traces`; this dispatch credits the recent motor
    // pattern with `Δw = lr · reward · trace`.
    if let Some(mut gpu_full) = gpu_full {
        let n = slot_map.len();
        if n > 0 && rewards.iter().any(|&r| r > 0.0) {
            let pipeline = &mut *gpu_full;
            pipeline.cells.upload_rewards(rewards);
            pipeline
                .hebbian
                .dispatch_apply_reward_persistent(&pipeline.cells, n, LEARNING_RATE);
        }
    }
    let _ = rewards;
    diag.add_measurement(&DIAG_EAT_FOOD, || t_total.elapsed().as_secs_f64() * 1000.0);
}
