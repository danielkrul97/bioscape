//! Sprint 176: shared-driver tick system. Runs `world.tick(&mut rng)`
//! once per Bevy FixedUpdate. Sprint 177 adds `sync_simworld_to_cellentity`
//! to overwrite legacy `CellEntity` state with SimWorld cells, so the
//! visual pipeline (sync_transforms, gizmos, materials) now reflects
//! SimWorld dynamics. Legacy tick systems continue to run in parallel
//! but their writes are immediately clobbered.

use bevy::prelude::*;
use bioscape::{N_PHEROMONE_CHANNELS, SPIKE_SLOTS};

use super::super::components::{CellEntity, SpikeEntity};
use super::super::material::{adhesion_material, cell_rotation, cell_scale, BioMaterial};
use super::super::resources::{
    AdhesionMaterials, CellEntityPool, CellMesh, CellSlotMap, SimRng, SimWorld, SpikeMaterial,
    SpikeMesh,
};

pub(crate) fn sim_tick(
    mut sim_world: ResMut<SimWorld>,
    mut sim_rng: ResMut<SimRng>,
) {
    let rng = &mut sim_rng.0;
    let gen_ended = sim_world.0.tick(rng);
    // The CPU vibration shadow is refreshed lazily by `update_stats_overlay`
    // (its only consumer) — a per-tick GPU grid readback here meant a full
    // `Wait` barrier spent feeding a throttled debug HUD.

    // Parity with headless `main.rs` post-tick cleanup: at the boundary
    // of a finished generation, reset the per-gen counters that the
    // stats overlay surfaces. Without this they accumulate indefinitely
    // and `bonds_formed_gen`, `predation_events_gen`, `ph_burst_score`,
    // … keep climbing instead of reflecting per-gen activity.
    if gen_ended.is_some() {
        let world = &mut sim_world.0;
        world.births_gen = 0;
        world.deaths_gen = 0;
        world.fertile_ticks_gen = 0;
        world.predation_events_gen = 0;
        world.bonds_formed_gen = 0;
        world.bonds_broken_gen = 0;
        world.bonded_attacks_gen = 0;
        world.solo_attacks_gen = 0;
        world.bonded_attack_gain_sum_gen = 0.0;
        world.solo_attack_gain_sum_gen = 0.0;
        world.swarm_attacks_gen = 0;
        world.pack_attacks_gen = 0;
        world.attack_victims_gen = 0;
        world.coop_food_solved_gen = 0;
        world.coop_food_failed_gen = 0;
        world.coop_food_arrivals_sum_gen = 0;
        world.coop_food_events_gen = 0;
        world.goal_zone_ticks_gen = 0;
        world.goal_unique_reachers_gen.clear();
        for cell in &mut world.cells {
            cell.burst_accum = [0.0; N_PHEROMONE_CHANNELS];
        }
    }
}

/// Sprint 185: lifecycle-aware sync. S195 adds entity pooling.
///
/// Lifecycle (per tick):
/// - **Shrink** (`slot_map.len() > world.cells.len()` after swap_remove): the
///   trailing `CellEntity` records are *not despawned* — instead toggled
///   `Visibility::Hidden` and stashed in `CellEntityPool`. This sidesteps
///   Bevy issue #22065 where every `Entity::despawn()` on an
///   `ExtendedMaterial` user leaks an entry in
///   `specialized_material_pipeline_cache`.
/// - **Grow** (`slot_map.len() < world.cells.len()`): try to recycle a
///   hidden entity from the pool first by re-inserting `CellEntity`,
///   swapping the `MeshMaterial3d` (adhesion-type can differ from the
///   previous tenant), updating `Transform`, and flipping back to
///   `Visibility::Visible`. Spike children stay alive — their
///   `owner: Entity` id is unchanged across recycles, and `sync_spikes`
///   reads fresh data from the recycled `CellEntity` next frame. Only when
///   the pool is empty (typically just at startup) do we `spawn` a fresh
///   `CellEntity` + `SPIKE_SLOTS` spike children.
///
/// After lifecycle adjustments, `world.cells[slot]` is copied into every
/// owned `CellEntity`. Brand-new entities don't appear in the `cells` query
/// during this same system run — they're seeded with the correct cell data
/// at spawn time (and overwritten normally next tick).
pub(crate) fn sync_simworld_to_cellentity(
    sim_world: Res<SimWorld>,
    mut slot_map: ResMut<CellSlotMap>,
    mut pool: ResMut<CellEntityPool>,
    cell_mesh: Res<CellMesh>,
    spike_mesh: Res<SpikeMesh>,
    spike_material: Res<SpikeMaterial>,
    mut adhesion_materials: ResMut<AdhesionMaterials>,
    mut bio_materials: ResMut<Assets<BioMaterial>>,
    mut commands: Commands,
    mut cells: Query<&mut CellEntity>,
) {
    let world_cells = &sim_world.0.cells;
    let current_len = slot_map.slot_to_entity.len();
    let target_len = world_cells.len();

    if current_len > target_len {
        // Shrink: hide + pool, don't despawn.
        for slot in (target_len..current_len).rev() {
            let entity = slot_map.slot_to_entity[slot];
            commands
                .entity(entity)
                .insert(Visibility::Hidden);
            slot_map.entity_to_slot.remove(&entity);
            pool.free_cells.push(entity);
        }
        slot_map.slot_to_entity.truncate(target_len);
    } else if current_len < target_len {
        // Grow: pop from pool first, fresh spawn only as fallback.
        for slot in current_len..target_len {
            let cell = &world_cells[slot];
            let material = adhesion_material(
                &mut adhesion_materials,
                &mut bio_materials,
                cell.genome.adhesion_type,
            );
            let transform =
                Transform::from_xyz(cell.position[0], cell.position[1], cell.position[2])
                    .with_rotation(cell_rotation(cell.heading, cell.pitch))
                    .with_scale(cell_scale(&cell.phenotype));
            let entity = if let Some(recycled) = pool.free_cells.pop() {
                // Recycle: refresh CellEntity data, swap material (adhesion
                // type of the new tenant likely differs), refresh transform,
                // unhide. Spike children survive — their `owner` id is
                // stable, only the `CellEntity` payload changed.
                commands
                    .entity(recycled)
                    .insert(CellEntity(*cell))
                    .insert(MeshMaterial3d(material))
                    .insert(transform)
                    .insert(Visibility::Visible);
                recycled
            } else {
                // Pool empty (startup or under cap): spawn fresh entity +
                // its spike children.
                let entity = commands
                    .spawn((
                        CellEntity(*cell),
                        Mesh3d(cell_mesh.0.clone()),
                        MeshMaterial3d(material),
                        transform,
                    ))
                    .id();
                for s in 0..SPIKE_SLOTS as u8 {
                    commands.spawn((
                        SpikeEntity { owner: entity, slot: s },
                        Mesh3d(spike_mesh.0.clone()),
                        MeshMaterial3d(spike_material.0.clone()),
                        Transform::default(),
                        Visibility::Hidden,
                    ));
                }
                entity
            };
            slot_map.allocate(entity);
        }
    }

    let synced = slot_map.slot_to_entity.len().min(target_len);
    for slot in 0..synced {
        let entity = slot_map.slot_to_entity[slot];
        if let Ok(mut cell_entity) = cells.get_mut(entity) {
            cell_entity.0 = world_cells[slot];
        }
    }
}
