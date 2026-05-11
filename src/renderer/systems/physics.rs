use bevy::diagnostic::Diagnostics;
use bevy::prelude::*;
use bioscape::{PHYSICS_CONFIG, WORLD_HALF};
use std::time::Instant;

use super::super::components::{CellEntity, Dying, FoodEntity};
use super::super::config::DIAG_BROWNIAN;
use super::super::resources::{Clock, EventCalendarResource, MazeWorld, WorldExtent, WorldMapResource};
use super::super::world_map::hazard_drain;
#[cfg(feature = "gpu")]
use super::super::resources_gpu::GpuFullPipeline;

pub(crate) fn step_cells(
    time: Res<Time>,
    extent: Res<WorldExtent>,
    clock: Res<Clock>,
    events: Res<EventCalendarResource>,
    maze: Res<MazeWorld>,
    mut cells: Query<&mut CellEntity>,
    #[cfg(feature = "gpu")] gpu_full: Option<Res<GpuFullPipeline>>,
) {
    // Full GPU pipeline: kinematics + drag + energy + bounce už proběhly v
    // `cells_brain_act_gpu_full` (StepGpu shader); tento systém je no-op.
    #[cfg(feature = "gpu")]
    if gpu_full.is_some() {
        return;
    }
    let dt = time.delta_secs();
    let half = extent.as_array();
    let tick = clock.0.tick;
    let gen = clock.0.generation;
    // ClimateShift offset is per-cell; thermal phase terms are uniform across
    // the dispatch, so we precompute the `ThermalCtx` once and let
    // `step_with_thermal` reuse it (saves two sin / modulo pairs per cell).
    let event_slice = events.0.events.as_slice();
    let thermal_ctx = bioscape::ThermalCtx::for_tick(tick, gen);
    let obstacles = maze.field.as_ref();
    cells.par_iter_mut().for_each(|mut cell| {
        let climate_offset = bioscape::climate_shock_offset(
            event_slice,
            gen,
            [cell.0.position[0], cell.0.position[1]],
            half,
        );
        cell.0.step_with_thermal_maze(
            dt,
            half,
            &thermal_ctx,
            &PHYSICS_CONFIG,
            climate_offset,
            obstacles,
        );
    });
}

/// Brownian noise — gaussian velocity perturbation. Fused into
/// `cells_brain_act_gpu_full` (motor → brownian → batch readback) so this
/// system is a no-op when GpuFullPipeline is present (always after wave N).
pub(crate) fn apply_brownian_motion(
    time: Res<Time>,
    extent: Res<WorldExtent>,
    mut cells: Query<&mut CellEntity, Without<Dying>>,
    mut diag: Diagnostics,
    #[cfg(feature = "gpu")] gpu_full: Option<Res<GpuFullPipeline>>,
) {
    // Full GPU pipeline: brownian dispatchnut v `cells_brain_act_gpu_full`
    // (xoshiro128++ shader) — tento systém je no-op.
    #[cfg(feature = "gpu")]
    if gpu_full.is_some() {
        return;
    }
    // Per-cell xoshiro128++ stream now lives on `Cell.xoshiro_state` (seeded
    // from `cell_id` at spawn / reproduce). Identical algorithm to the GPU
    // shader, so CPU and GPU paths produce the same brownian samples — the
    // dominant source of CPU/GPU trajectory drift before this change.
    let t_total = Instant::now();
    let sqrt_dt = time.delta_secs().sqrt();
    let half_z = extent.as_array()[2];
    cells.par_iter_mut().for_each(|mut cell| {
        cell.0.apply_brownian(sqrt_dt, half_z);
    });
    diag.add_measurement(&DIAG_BROWNIAN, || t_total.elapsed().as_secs_f64() * 1000.0);
}

/// Sprint 38: gravity drift na food. Aktualizuje Food.position[2] + sync
/// Transform.translation.z aby viditelně klesalo k dnu.
///
/// Sprint 131: 2-pass — par_iter_mut na gravity + transform sync (31k+ entities
/// per tick = bulk of cost), pak sériový age_step + despawn (Commands buffer
/// není Sync; sequential pass jen čte position[2] do age_step).
pub(crate) fn apply_food_gravity(
    time: Res<Time>,
    extent: Res<WorldExtent>,
    mut foods: Query<(Entity, &mut FoodEntity, &mut Transform)>,
    mut commands: Commands,
) {
    let dt = time.delta_secs();
    let half_z = extent.as_array()[2];
    foods.par_iter_mut().for_each(|(_entity, mut food, mut transform)| {
        food.0.apply_gravity(dt, half_z);
        transform.translation.z = food.0.position[2];
    });
    for (entity, mut food, _transform) in &mut foods {
        if !food.0.age_step() {
            commands.entity(entity).despawn();
        }
    }
}

pub(crate) fn apply_environmental_hazards(
    time: Res<Time>,
    world_map: Res<WorldMapResource>,
    clock: Res<Clock>,
    events: Res<EventCalendarResource>,
    mut cells: Query<&mut CellEntity, Without<Dying>>,
) {
    let dt = time.delta_secs();
    let gen = clock.0.generation;
    let tick = clock.0.tick;
    // Sprint 113: par_iter_mut — sample/shock_multiplier čistě read-only;
    // mutace pouze cell.energy/damage_accum, žádný cross-cell state.
    let event_slice = events.0.events.as_slice();
    let world_map_ref = &world_map.0;
    cells.par_iter_mut().for_each(|mut cell| {
        let noise = world_map_ref
            .sample([cell.0.position[0], cell.0.position[1], cell.0.position[2]]);
        let shock_mult = bioscape::hazard_shock_multiplier(
            cell.0.position,
            event_slice,
            gen,
            tick,
            WORLD_HALF,
        );
        let drain = hazard_drain(noise) * dt * shock_mult;
        cell.0.energy -= drain;
        cell.0.damage_accum += drain;
    });
}
