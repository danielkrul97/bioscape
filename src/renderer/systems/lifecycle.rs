use bevy::prelude::*;
use bioscape::{
    CARRION_FOOD_COUNT, CELL_RADIUS, Cell, Food, MATING_COOLDOWN_TICKS, MATING_PHEROMONE_THRESHOLD,
    MATING_RADIUS, MAX_POPULATION, REPRODUCE_THRESHOLD, WORLD_HALF,
};
use rand::Rng;

use super::super::components::{CellEntity, Dying, FoodEntity};
use super::super::config::DEATH_FADE_TICKS;
use super::super::material::{adhesion_material, cell_rotation, cell_scale, BioMaterial};
use super::super::resources::{
    AdhesionMaterials, CellMesh, CellSlotMap, FoodMaterial, FoodMesh, NextCellId, WorldExtent,
};
#[cfg(feature = "gpu")]
use super::super::resources_gpu::{GpuBrainState, GpuFullPipeline};

pub(crate) fn cell_reproduces_on_threshold(
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
    cell_mesh: Res<CellMesh>,
    mut adhesion_materials: ResMut<AdhesionMaterials>,
    mut bio_materials: ResMut<Assets<BioMaterial>>,
    mut slot_map: ResMut<CellSlotMap>,
    mut next_cell_id: ResMut<NextCellId>,
    #[cfg(feature = "gpu")] gpu_state: Option<Res<GpuBrainState>>,
    #[cfg(feature = "gpu")] gpu_full: Option<ResMut<GpuFullPipeline>>,
    mut commands: Commands,
    mut fertile_scratch: Local<Vec<(Entity, [f32; 3])>>,
    mut to_spawn_scratch: Local<Vec<Cell>>,
    #[cfg(feature = "gpu")] mut cppn_dispatch_scratch: Local<Vec<(usize, bioscape::Cppn)>>,
) {
    // `slot_map` tracks every live (non-Dying) cell — equivalent to walking the
    // `Query<_, Without<Dying>>` but in O(1).
    let current_pop = slot_map.len();
    if current_pop >= MAX_POPULATION {
        return;
    }
    let budget = MAX_POPULATION - current_pop;

    // R-#11: persistent Local fertile snapshot.
    fertile_scratch.clear();
    fertile_scratch.extend(
        cells
            .iter()
            .filter(|(_, c)| {
                c.0.energy >= REPRODUCE_THRESHOLD
                    && c.0.last_outputs[2] > MATING_PHEROMONE_THRESHOLD
                    && c.0.reproduce_cooldown_ticks == 0
            })
            .map(|(e, c)| (e, c.0.position)),
    );
    let fertile = fertile_scratch.as_slice();
    let mating_r2 = MATING_RADIUS * MATING_RADIUS;
    let matings = bioscape::pair_fertile(fertile, mating_r2, budget, WORLD_HALF);

    // In `--gpu-full` we materialise child brain weights via GPU CPPN
    // dispatch after spawn; the chained `crossover().mutate()` already
    // skips `Brain::from_cppn` via `make_mating_child_no_brain`.
    #[cfg(feature = "gpu")]
    let use_gpu_cppn = gpu_full.is_some();
    #[cfg(not(feature = "gpu"))]
    let use_gpu_cppn = false;

    // Sprint 52: před crossover sync parent brains z GPU (post-Hebbian je
    // canonical). Pokud GPU available; jinak no-op (CPU brain je canonical).
    #[cfg(feature = "gpu")]
    if let Some(gpu) = gpu_state.as_ref() {
        for &(a, b) in &matings {
            if let (Some(slot_a), Some(slot_b)) = (slot_map.slot_of(a), slot_map.slot_of(b)) {
                let brain_a = gpu.cells.download_brain_at(slot_a);
                let brain_b = gpu.cells.download_brain_at(slot_b);
                if let Ok([(_, mut ca), (_, mut cb)]) = cells.get_many_mut([a, b]) {
                    ca.0.genome.brain = brain_a;
                    cb.0.genome.brain = brain_b;
                }
            }
        }
    } else if let Some(gpu) = gpu_full.as_ref() {
        for &(a, b) in &matings {
            if let (Some(slot_a), Some(slot_b)) = (slot_map.slot_of(a), slot_map.slot_of(b)) {
                let brain_a = gpu.cells.download_brain_at(slot_a);
                let brain_b = gpu.cells.download_brain_at(slot_b);
                if let Ok([(_, mut ca), (_, mut cb)]) = cells.get_many_mut([a, b]) {
                    ca.0.genome.brain = brain_a;
                    cb.0.genome.brain = brain_b;
                }
            }
        }
    }

    let mut rng = rand::rng();
    to_spawn_scratch.clear();
    for (a, b) in matings {
        let Ok([(_, mut cell_a), (_, mut cell_b)]) = cells.get_many_mut([a, b]) else {
            continue;
        };
        cell_a.0.energy *= 0.5;
        cell_b.0.energy *= 0.5;
        cell_a.0.reproduce_cooldown_ticks = MATING_COOLDOWN_TICKS;
        cell_b.0.reproduce_cooldown_ticks = MATING_COOLDOWN_TICKS;
        // Sprint 66: child gets stable cell_id from monotonic counter.
        let child_id = next_cell_id.0;
        next_cell_id.0 += 1;
        let child = if use_gpu_cppn {
            bioscape::make_mating_child_no_brain(&cell_a.0, &cell_b.0, &mut rng, child_id)
        } else {
            bioscape::make_mating_child(&cell_a.0, &cell_b.0, &mut rng, child_id)
        };
        to_spawn_scratch.push(child);
    }

    #[cfg(feature = "gpu")]
    cppn_dispatch_scratch.clear();
    let mesh = cell_mesh.0.clone();
    for cell in to_spawn_scratch.drain(..) {
        let mat = adhesion_material(
            &mut adhesion_materials,
            &mut bio_materials,
            cell.genome.adhesion_type,
        );
        // `Cppn` is `Copy`; clone the value before `spawn()` consumes `cell`
        // so we can dispatch the GPU CPPN materialisation after the loop.
        #[cfg(feature = "gpu")]
        let cppn_copy = cell.genome.cppn;
        let lineage_id = cell.lineage_id;
        let turn_rate = cell.genome.turn_rate;
        let entity = commands
            .spawn((
                CellEntity(cell),
                Mesh3d(mesh.clone()),
                MeshMaterial3d(mat),
                Transform::from_xyz(cell.position[0], cell.position[1], cell.position[2])
                    .with_rotation(cell_rotation(cell.heading, cell.pitch))
                    .with_scale(cell_scale(&cell.phenotype)),
            ))
            .id();
        let slot = slot_map.allocate(entity);
        #[cfg(feature = "gpu")]
        if let Some(gpu) = gpu_state.as_ref() {
            gpu.cells.upload_brain_at(slot, &cell.genome.brain);
            gpu.cells.upload_xoshiro_seed_at(
                slot,
                lineage_id ^ (slot as u64).wrapping_mul(0x9E3779B97F4A7C15),
            );
        }
        #[cfg(feature = "gpu")]
        if let Some(gpu) = gpu_full.as_ref() {
            // Skip `upload_brain_at` — the GPU CPPN dispatch below writes
            // brain weights directly into `cells.brain_weights_buf`.
            gpu.cells.upload_xoshiro_seed_at(
                slot,
                lineage_id ^ (slot as u64).wrapping_mul(0x9E3779B97F4A7C15),
            );
            gpu.cells.upload_turn_rate_at(slot, turn_rate);
            cppn_dispatch_scratch.push((slot, cppn_copy));
        }
        let _ = slot;
    }

    // Single GPU CPPN dispatch covers every child spawned this frame.
    #[cfg(feature = "gpu")]
    if let Some(mut gpu) = gpu_full {
        if !cppn_dispatch_scratch.is_empty() {
            // Build `&Cppn` references into the owned scratch Vec; the dispatch
            // signature takes a slice of `(slot, &Cppn)` pairs.
            let pairs: Vec<(usize, &bioscape::Cppn)> = cppn_dispatch_scratch
                .iter()
                .map(|(s, c)| (*s, c))
                .collect();
            let pipeline = &mut *gpu;
            pipeline.cppn.dispatch(&pairs, &pipeline.cells);
        }
    }
}

pub(crate) fn cell_dies_on_zero_energy(
    cells: Query<(Entity, &CellEntity), Without<Dying>>,
    extent: Res<WorldExtent>,
    food_mesh: Res<FoodMesh>,
    food_material: Res<FoodMaterial>,
    mut slot_map: ResMut<CellSlotMap>,
    #[cfg(feature = "gpu")] gpu_state: Option<Res<GpuBrainState>>,
    #[cfg(feature = "gpu")] gpu_full: Option<Res<GpuFullPipeline>>,
    mut commands: Commands,
) {
    let mut rng = rand::rng();
    let half = extent.as_array();
    for (entity, cell) in &cells {
        if cell.0.energy <= 0.0 {
            commands.entity(entity).insert(Dying {
                ticks_left: DEATH_FADE_TICKS,
            });
            for _ in 0..CARRION_FOOD_COUNT {
                let pos = [
                    (cell.0.position[0] + rng.random_range(-CELL_RADIUS..CELL_RADIUS))
                        .clamp(-half[0], half[0]),
                    (cell.0.position[1] + rng.random_range(-CELL_RADIUS..CELL_RADIUS))
                        .clamp(-half[1], half[1]),
                    cell.0.position[2].clamp(-half[2], half[2]),
                ];
                commands.spawn((
                    FoodEntity(Food {
                        position: pos,
                        age_ticks: 0,
                        kind: bioscape::FoodKind::Carrion,
                    }),
                    Mesh3d(food_mesh.0.clone()),
                    MeshMaterial3d(food_material.0.clone()),
                    Transform::from_xyz(pos[0], pos[1], pos[2]),
                    Visibility::Hidden,
                ));
            }
            // Sprint 52: release slot ihned na Dying. Entity ještě existuje
            // pro fade animaci (Without<Dying> ji vyloučí ze sim systems).
            // GPU swap_to drží sloty dense.
            if let Some((freed_slot, moved)) = slot_map.release(entity) {
                #[cfg(feature = "gpu")]
                if let Some(gpu) = gpu_state.as_ref() {
                    if let Some(_moved_entity) = moved {
                        // moved cell je ve slot_map.slot_of(moved_entity) = freed_slot
                        // teď. Source je old_slot = current cell count (po release).
                        gpu.cells.swap_to(freed_slot, slot_map.len());
                    }
                }
                #[cfg(feature = "gpu")]
                if let Some(gpu) = gpu_full.as_ref() {
                    if moved.is_some() {
                        gpu.cells.swap_to(freed_slot, slot_map.len());
                    }
                }
                let _ = freed_slot;
                let _ = moved;
            }
        }
    }
}

pub(crate) fn tick_death_fade(
    mut dying: Query<(Entity, &mut Dying, &CellEntity, &mut Transform)>,
    mut commands: Commands,
) {
    for (entity, mut d, cell, mut transform) in &mut dying {
        if d.ticks_left == 0 {
            commands.entity(entity).despawn();
            continue;
        }
        d.ticks_left -= 1;
        let progress = d.ticks_left as f32 / DEATH_FADE_TICKS as f32;
        // Sprint 36: fade jen přes scale shrinkout. Alpha fade by chtělo
        // Material handle adjustment per cell (StandardMaterial alpha_mode +
        // base_color.alpha). Sprint 38+ může to vyřešit; teď postačí scaling.
        transform.scale = cell_scale(&cell.0.phenotype) * progress;
    }
}
