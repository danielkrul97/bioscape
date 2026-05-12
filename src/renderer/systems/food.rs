use bevy::diagnostic::Diagnostics;
use bevy::prelude::*;
use bioscape::{
    EAT_RADIUS, FOOD_SPAWN_RATE, Food, LEARNING_RATE, MAX_SPAWN_ATTEMPTS,
    WORLD_HALF,
};
use bioscape::gpu::{EatFoodParamsGpu, FoodSpawnParamsGpu};
use rand::Rng;
use rustc_hash::FxHashSet;
use std::time::Instant;

use super::super::components::{CellEntity, Dying, FoodEntity};
use super::super::config::DIAG_EAT_FOOD;
use super::super::resources::{
    CellSlotMap, Clock, CoopFoodResource, FoodDensityFactor, FoodMaterial, FoodMesh, MazeWorld,
    WorldExtent,
};
use super::super::resources_gpu::GpuFullPipeline;
use super::super::world_map::food_target;

/// GPU rejection-sampling food spawn — mirror of headless
/// `World::spawn_food_dispatch`. Generates `to_spawn × MAX_SPAWN_ATTEMPTS`
/// candidates in parallel on the GPU (world-map richness + obstacle mask +
/// cell-radius exclusion), then CPU consumes the first `to_spawn` valid hits
/// as Bevy `FoodEntity` spawn commands.
///
/// Replaces the previous CPU `cell_grid.for_each_in_radius_toroidal` loop
/// which sat at the top of perf flamegraphs (~27 % of CPU). The CPU
/// `CellGrid` / `WorldMapResource` snapshots are no longer needed by this
/// system — the rebuild only stayed alive because of this single consumer.
pub(crate) fn spawn_food(
    foods: Query<(), With<FoodEntity>>,
    cells: Query<&CellEntity, Without<Dying>>,
    extent: Res<WorldExtent>,
    factor: Res<FoodDensityFactor>,
    maze: Res<MazeWorld>,
    food_mesh: Res<FoodMesh>,
    food_material: Res<FoodMaterial>,
    gpu_full: Option<ResMut<GpuFullPipeline>>,
    mut commands: Commands,
) {
    let target = food_target(&extent, factor.0);
    let count = foods.iter().count();
    if count >= target {
        return;
    }
    let to_spawn = (target - count).min(FOOD_SPAWN_RATE);
    if to_spawn == 0 {
        return;
    }
    let Some(mut gpu_res) = gpu_full else { return };

    let positions: Vec<[f32; 3]> = cells.iter().map(|c| c.0.position).collect();
    let max_axes: Vec<f32> = cells.iter().map(|c| c.0.phenotype.max_axis()).collect();

    let mut rng = rand::rng();
    let k = to_spawn * MAX_SPAWN_ATTEMPTS;
    let seeds: Vec<[u32; 4]> = (0..k)
        .map(|_| {
            [
                rng.random::<u32>(),
                rng.random::<u32>(),
                rng.random::<u32>(),
                rng.random::<u32>(),
            ]
        })
        .collect();

    let obstacle_active = maze.field.is_some();
    let (obs_nx, obs_ny, obs_nz) = maze
        .field
        .as_ref()
        .map(|o| {
            (
                o.resolution[0] as u32,
                o.resolution[1] as u32,
                o.resolution[2] as u32,
            )
        })
        .unwrap_or((1, 1, 1));
    let half = extent.as_array();
    let params = FoodSpawnParamsGpu {
        num_attempts: 0, // populated by compute()
        rejection_strength: bioscape::FOOD_REJECTION_STRENGTH,
        eat_radius: EAT_RADIUS,
        cell_size: bioscape::GRID_CELL_SIZE,
        world_half_x: half[0],
        world_half_y: half[1],
        world_half_z: half[2],
        num_cells: 0, // populated by compute()
        world_map_nx: bioscape::WORLD_MAP_RES as u32,
        world_map_ny: bioscape::WORLD_MAP_RES as u32,
        world_map_nz: bioscape::WORLD_MAP_RES_Z as u32,
        obstacle_active: if obstacle_active { 1 } else { 0 },
        obstacle_nx: obs_nx,
        obstacle_ny: obs_ny,
        obstacle_nz: obs_nz,
        _pad0: 0,
    };

    let gpu = &mut *gpu_res;
    gpu.cell_hash.dispatch(&positions);
    gpu.food_spawn.seed_attempts(&seeds);
    let result = gpu
        .food_spawn
        .compute(k, &positions, &max_axes, &gpu.cell_hash, params);

    let mut pushed = 0usize;
    for i in 0..k {
        if pushed >= to_spawn {
            break;
        }
        if result.valid_mask[i] != 0 {
            let food = Food {
                position: result.candidate_positions[i],
                age_ticks: 0,
                kind: bioscape::FoodKind::Plant,
            };
            commands.spawn((
                FoodEntity(food),
                Mesh3d(food_mesh.0.clone()),
                MeshMaterial3d(food_material.0.clone()),
                Transform::from_xyz(food.position[0], food.position[1], food.position[2]),
                Visibility::Hidden,
            ));
            pushed += 1;
        }
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

/// Persistent per-tick scratch for `cell_eats_food`. Bundled into one
/// `Local` to keep the system param count under Bevy's 16 cap (Locals,
/// Queries, Resources all count against the same budget).
#[derive(Default)]
pub(crate) struct EatFoodScratch {
    cell_entities: Vec<Entity>,
    cell_positions: Vec<[f32; 3]>,
    cell_headings: Vec<f32>,
    cell_pitches: Vec<f32>,
    cell_body_dims: Vec<[f32; 3]>,
    cell_carnivore: Vec<f32>,
    cell_max_axes: Vec<f32>,
    food_entities: Vec<Entity>,
    food_positions: Vec<[f32; 3]>,
    food_kinds: Vec<u32>,
    food_age_ticks: Vec<u32>,
    eaten: FxHashSet<Entity>,
    share_deltas: Vec<(Entity, f32)>,
    rewards: Vec<f32>,
}

pub(crate) fn cell_eats_food(
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
    foods: Query<(Entity, &FoodEntity)>,
    slot_map: Res<CellSlotMap>,
    lookups: Res<super::super::resources::CellEntityLookups>,
    gpu_full: Option<ResMut<GpuFullPipeline>>,
    mut commands: Commands,
    mut diag: Diagnostics,
    mut scratch: Local<EatFoodScratch>,
) {
    let t_total = Instant::now();

    let Some(mut gpu_res) = gpu_full else { return };

    let s = &mut *scratch;
    s.rewards.clear();
    s.rewards.resize(slot_map.len(), 0.0);

    // Snapshot cells (dense, slot-aligned for the GPU dispatch).
    s.cell_entities.clear();
    s.cell_positions.clear();
    s.cell_headings.clear();
    s.cell_pitches.clear();
    s.cell_body_dims.clear();
    s.cell_carnivore.clear();
    s.cell_max_axes.clear();
    for (e, c) in cells.iter() {
        s.cell_entities.push(e);
        s.cell_positions.push(c.0.position);
        s.cell_headings.push(c.0.heading);
        s.cell_pitches.push(c.0.pitch);
        s.cell_body_dims.push([
            c.0.phenotype.body_length,
            c.0.phenotype.body_width,
            c.0.phenotype.body_height,
        ]);
        s.cell_carnivore.push(c.0.genome.carnivore_score);
        s.cell_max_axes.push(c.0.phenotype.max_axis());
    }

    // Snapshot foods. Index in this Vec is the dense food index the shader
    // will return via `food_idx`; we map back to Bevy `Entity` afterward.
    s.food_entities.clear();
    s.food_positions.clear();
    s.food_kinds.clear();
    s.food_age_ticks.clear();
    for (e, f) in foods.iter() {
        s.food_entities.push(e);
        s.food_positions.push(f.0.position);
        s.food_kinds.push(f.0.kind as u32);
        s.food_age_ticks.push(f.0.age_ticks);
    }
    let n_cells = s.cell_entities.len();
    let n_foods = s.food_entities.len();

    // GPU Pass 1: per-cell candidate (food_idx, value). Sentinel `n_foods`
    // means "this cell ate nothing this tick".
    let candidates: Vec<Option<(usize, f32)>> = if n_cells == 0 || n_foods == 0 {
        vec![None; n_cells]
    } else {
        let params = EatFoodParamsGpu {
            num_cells: 0,
            num_foods: 0,
            cell_size: bioscape::GRID_CELL_SIZE,
            eat_radius: EAT_RADIUS,
            world_half_x: WORLD_HALF[0],
            world_half_y: WORLD_HALF[1],
            world_half_z: WORLD_HALF[2],
            world_map_nx: bioscape::WORLD_MAP_RES as u32,
            world_map_ny: bioscape::WORLD_MAP_RES as u32,
            world_map_nz: bioscape::WORLD_MAP_RES_Z as u32,
            fixed_timestep_hz: bioscape::FIXED_TIMESTEP_HZ,
            plant_food_value: bioscape::PLANT_FOOD_VALUE,
            carrion_food_value: bioscape::CARRION_FOOD_VALUE,
            carrion_decay_per_sec: bioscape::CARRION_DECAY_PER_SEC,
            world_map_food_floor: bioscape::WORLD_MAP_FOOD_FLOOR,
            world_map_food_amp: bioscape::WORLD_MAP_FOOD_AMP,
        };
        let gpu = &mut *gpu_res;
        // food_hash needs current food positions; predate dispatches the
        // cell_hash with up-to-date positions, but the food_hash hasn't been
        // refreshed since spawn_food (which uses cell_hash not food_hash).
        gpu.food_hash.dispatch(&s.food_positions);
        let result = gpu.eat_food.compute(
            &s.cell_positions,
            &s.cell_headings,
            &s.cell_pitches,
            &s.cell_body_dims,
            &s.cell_carnivore,
            &s.cell_max_axes,
            &s.food_positions,
            &s.food_kinds,
            &s.food_age_ticks,
            &gpu.food_hash,
            params,
        );
        let sentinel = n_foods as u32;
        (0..n_cells)
            .map(|i| {
                let f = result.food_idx[i];
                if f >= sentinel {
                    None
                } else {
                    Some((f as usize, result.value[i]))
                }
            })
            .collect()
    };

    let id_to_entity = &lookups.id_to_entity;

    // Pass 2 (sequential): resolve race + apply energy + Hebbian. First-cell-wins
    // per food entity (matches pre-Sprint-58 ordering).
    // Sprint 78: cluster food share. Sebráno do share_deltas Vec během iterace,
    // aplikováno post-loop kvůli simultaneous mutable borrow.
    s.eaten.clear();
    s.share_deltas.clear();
    // Pass 2 (CPU, sequential): map GPU candidate indices back to Bevy
    // entities, resolve first-cell-wins race, apply energy + share + reward.
    for (cell_idx, opt) in candidates.iter().enumerate() {
        let Some((food_idx, value)) = opt else { continue };
        if *food_idx >= s.food_entities.len() {
            continue;
        }
        let food_e = s.food_entities[*food_idx];
        if s.eaten.contains(&food_e) {
            continue;
        }
        s.eaten.insert(food_e);
        let entity = s.cell_entities[cell_idx];
        let (bonds_copy, donor_state) = if let Ok((_, mut cell)) = cells.get_mut(entity) {
            cell.0.energy += *value;
            let copy = cell.0.bonds;
            let state = cell.0.cell_state;
            if let Some(slot) = slot_map.slot_of(entity) {
                if slot < s.rewards.len() {
                    s.rewards[slot] = 1.0;
                }
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
        let cluster_mult = 1.0
            + (n_bonds - 1.0).max(0.0) * bioscape::BOND_FOOD_SHARE_CLUSTER_BONUS;
        let share_value =
            *value * bioscape::BOND_FOOD_SHARE_FRAC * donor_state * cluster_mult;
        if share_value > 0.0 {
            for bond_opt in bonds_copy.iter() {
                if let Some(bond) = bond_opt {
                    if let Some(&partner_e) = id_to_entity.get(&bond.other_cell_id) {
                        if partner_e != entity {
                            s.share_deltas.push((partner_e, share_value));
                        }
                    }
                }
            }
        }
    }

    // Sprint 78: aplikuj food share delty (po Pass 2 main loop).
    // Drain into a temporary to release the &mut borrow on `s` before
    // touching `cells.get_mut`.
    let share_deltas: Vec<(Entity, f32)> = s.share_deltas.drain(..).collect();
    for (e, delta) in share_deltas {
        if let Ok((_, mut cell)) = cells.get_mut(e) {
            cell.0.energy += delta;
        }
    }

    // Pass 3: main-thread Commands flush (despawn nelze v par_iter).
    for food_e in s.eaten.iter() {
        commands.entity(*food_e).despawn();
    }

    // Wave 7: trace-based reward apply against `cells.brain_traces`.
    let n_alive = slot_map.len();
    if n_alive > 0 && s.rewards.iter().any(|&r| r > 0.0) {
        let pipeline = &mut *gpu_res;
        pipeline.cells.upload_rewards(&s.rewards);
        pipeline
            .hebbian
            .dispatch_apply_reward_persistent(&pipeline.cells, n_alive, LEARNING_RATE);
    }
    // CPU `cell.last_hidden` / `last_outputs` Hebbian path was the
    // pre-Wave-N fallback when GPU was absent — removed since gpu_full is
    // mandatory. `LEARNING_RATE` only flows through the GPU dispatch.
    let _ = LEARNING_RATE;
    diag.add_measurement(&DIAG_EAT_FOOD, || t_total.elapsed().as_secs_f64() * 1000.0);
}
