use bevy::diagnostic::Diagnostics;
use bevy::prelude::*;
use bioscape::{
    AGE_DECAY_PER_SEC, ATTACK_COST_PER_SEC, BRAIN_HIDDEN, BRAIN_INPUTS, BRAIN_INPUTS_SENSORY,
    BRAIN_RECURRENT, CYCLE_GEN_PERIOD, DAMAGE_NORMALIZATION_GAIN, DENSITY_NORM_COUNT,
    DRAG_COEFFICIENT, FIXED_TIMESTEP_HZ, GRAVITY as PHYS_GRAVITY, PHEROMONE_NORMALIZATION_GAIN,
    PHYSICS_CONFIG, REPRODUCE_THRESHOLD, SHELL_COST_PER_SEC, SMELL_GRID_RES, SMELL_GRID_RES_Z,
    SMELL_NORMALIZATION_GAIN, SMELL_SAMPLE_EPSILON, SPIKE_COST_PER_SEC, THERMAL_NOISE,
};
use bioscape::gpu::{PopulateInputsParams, SensorParamsGpu, StepParamsGpu};
use std::time::Instant;

use super::super::components::{CellEntity, Dying, FoodEntity};
use super::super::config::DIAG_BRAIN_ACT;
use super::super::resources::{CellSlotMap, Clock, CoopFoodResource, MazeWorld, WorldExtent};
use super::super::resources_gpu::GpuFullPipeline;

/// Full GPU pipeline brain_act: zrcadlí headless `brain_act_gpu_full`.
/// Single Wait barrier per tick — všechny GPU compute fáze (spatial hash,
/// sensor gather, populate inputs, brain forward, motor, brownian, step)
/// běží mezi sebou bez CPU readback; jediný `device.poll(Maintain::Wait)`
/// přes `download_full_batch_into` na konci. CPU fáze `step_cells`,
/// `apply_brownian_motion`, `cells_brain_act` (CPU/GPU-brain-only) se v
/// gpu_full režimu stávají no-op.
///
/// Schedule: běží před `cells_brain_act` v Phase 1 chain (po
/// `pool_bonded_hidden_cells`); CPU `cells_brain_act` se sám skipuje pokud
/// `GpuFullPipeline` resource existuje.
pub(crate) fn cells_brain_act_gpu_full(
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
    foods: Query<&FoodEntity>,
    coop_foods: Res<CoopFoodResource>,
    slot_map: Res<CellSlotMap>,
    mut pipeline: ResMut<GpuFullPipeline>,
    fixed_time: Res<Time<Fixed>>,
    clock: Res<Clock>,
    extent: Res<WorldExtent>,
    maze: Res<MazeWorld>,
    mut diag: Diagnostics,
) {
    let t_total = Instant::now();
    let dt = fixed_time.delta_secs();
    let n = slot_map.len();
    if n == 0 {
        return;
    }
    let world_half = extent.as_array();
    // Bevy `ResMut<T>::deref_mut` neumožňuje split-field borrows, takže
    // všechny `pipeline.field.method()` volání s args odkazujícími na sibling
    // fields by failovaly. Explicit deref → raw `&mut GpuFullPipeline`
    // unblockne split borrows (Rust borrow checker rozumí přes `&mut *T`).
    let pipeline = &mut *pipeline;

    // Phase 1a: resize SoA scratch + push food positions sequentially (small
    // pool — typically O(food_target)).
    let food_n = foods.iter().count() + coop_foods.0.len();
    pipeline.scratch.food_positions.clear();
    pipeline.scratch.food_positions.reserve(food_n);
    for food in &foods {
        pipeline.scratch.food_positions.push(food.0.position);
    }
    for coop in coop_foods.0.iter() {
        pipeline.scratch.food_positions.push(coop.position);
    }
    pipeline.scratch.resize_snapshot(n);

    // Phase 1b: parallel snapshot. Each closure invocation owns its slot
    // (`slot_map.slot_of(entity)` is one-to-one for live cells), so writing
    // to per-slot raw pointers is race-free even though the pointers
    // themselves are shared. `par_iter_mut` already serialises the mutable
    // `Cell` access per entity.
    //
    // `MutPtr<T>` / `ConstPtr<T>` wrap raw pointers into Send+Sync newtypes
    // so Rust 2021 disjoint capture in the closure can capture each field
    // without re-tripping the auto-trait check on `*mut T` / `*const T`.
    #[derive(Copy, Clone)]
    struct MutPtr<T>(*mut T);
    // SAFETY: parallel tasks index disjoint slots; no aliasing.
    unsafe impl<T> Send for MutPtr<T> {}
    unsafe impl<T> Sync for MutPtr<T> {}
    impl<T> MutPtr<T> {
        #[inline]
        unsafe fn add(self, n: usize) -> *mut T {
            unsafe { self.0.add(n) }
        }
    }
    #[derive(Copy, Clone)]
    struct ConstPtr<T>(*const T);
    unsafe impl<T> Send for ConstPtr<T> {}
    unsafe impl<T> Sync for ConstPtr<T> {}
    impl<T> ConstPtr<T> {
        #[inline]
        unsafe fn add(self, n: usize) -> *const T {
            unsafe { self.0.add(n) }
        }
    }

    let snap_positions = MutPtr(pipeline.scratch.positions.as_mut_ptr());
    let snap_eff_radii = MutPtr(pipeline.scratch.eff_radii.as_mut_ptr());
    let snap_vision_radii = MutPtr(pipeline.scratch.vision_radii.as_mut_ptr());
    let snap_energies = MutPtr(pipeline.scratch.energies.as_mut_ptr());
    let snap_headings = MutPtr(pipeline.scratch.headings.as_mut_ptr());
    let snap_pitches = MutPtr(pipeline.scratch.pitches.as_mut_ptr());
    let snap_damage_accums = MutPtr(pipeline.scratch.damage_accums.as_mut_ptr());
    let snap_max_speeds = MutPtr(pipeline.scratch.max_speeds.as_mut_ptr());
    let snap_velocities = MutPtr(pipeline.scratch.velocities.as_mut_ptr());
    let snap_angular_vels = MutPtr(pipeline.scratch.angular_vels.as_mut_ptr());
    let snap_pitch_vels = MutPtr(pipeline.scratch.pitch_vels.as_mut_ptr());
    let snap_ages = MutPtr(pipeline.scratch.ages.as_mut_ptr());
    let snap_cooldowns = MutPtr(pipeline.scratch.cooldowns.as_mut_ptr());
    let snap_body_dims = MutPtr(pipeline.scratch.body_dims.as_mut_ptr());
    let snap_aux = MutPtr(pipeline.scratch.aux.as_mut_ptr());
    let snap_hidden_ns = MutPtr(pipeline.scratch.hidden_ns.as_mut_ptr());
    let snap_bonded_inboxes = MutPtr(pipeline.scratch.bonded_inboxes.as_mut_ptr());
    let slot_map_ref = &*slot_map;
    cells.par_iter_mut().for_each(|(entity, mut cell_entity)| {
        let Some(slot) = slot_map_ref.slot_of(entity) else { return };
        let cell = &mut cell_entity.0;
        cell.apply_shell_absorb(dt);
        cell.last_best_food_d2 = 0.0;
        // SAFETY: `slot < n` (guaranteed by slot_map invariant) and each slot
        // is touched by exactly one closure invocation per tick.
        unsafe {
            *snap_positions.add(slot) = cell.position;
            *snap_eff_radii.add(slot) = cell.phenotype.effective_radius();
            *snap_vision_radii.add(slot) = cell.genome.vision_radius;
            *snap_energies.add(slot) = cell.energy;
            *snap_headings.add(slot) = cell.heading;
            *snap_pitches.add(slot) = cell.pitch;
            *snap_damage_accums.add(slot) = cell.damage_accum;
            *snap_max_speeds.add(slot) = cell.genome.max_speed;
            *snap_velocities.add(slot) = cell.velocity;
            *snap_angular_vels.add(slot) = cell.angular_velocity;
            *snap_pitch_vels.add(slot) = cell.pitch_velocity;
            *snap_ages.add(slot) = cell.age as u32;
            *snap_cooldowns.add(slot) = cell.reproduce_cooldown_ticks;
            *snap_body_dims.add(slot) = [
                cell.phenotype.body_length,
                cell.phenotype.body_width,
                cell.phenotype.body_height,
            ];
            *snap_aux.add(slot) = [
                cell.phenotype.total_spike_cost_factor(),
                cell.phenotype.shell_thickness,
                cell.genome.vision_radius,
                cell.last_outputs[6].max(0.0),
            ];
            *snap_hidden_ns.add(slot) = cell.genome.brain.hidden_n;
            *snap_bonded_inboxes.add(slot) = cell.bonded_inbox;
        }
    });

    // Aliases pro split borrow — upload_* berou &[T] sliced ze scratch fields.
    let positions = pipeline.scratch.positions.as_slice();
    let eff_radii = pipeline.scratch.eff_radii.as_slice();
    let vision_radii = pipeline.scratch.vision_radii.as_slice();
    let food_positions = pipeline.scratch.food_positions.as_slice();
    let energies = pipeline.scratch.energies.as_slice();
    let headings = pipeline.scratch.headings.as_slice();
    let pitches = pipeline.scratch.pitches.as_slice();
    let damage_accums = pipeline.scratch.damage_accums.as_slice();
    let max_speeds = pipeline.scratch.max_speeds.as_slice();
    let velocities = pipeline.scratch.velocities.as_slice();
    let angular_vels = pipeline.scratch.angular_vels.as_slice();
    let pitch_vels = pipeline.scratch.pitch_vels.as_slice();
    let ages = pipeline.scratch.ages.as_slice();
    let cooldowns = pipeline.scratch.cooldowns.as_slice();
    let body_dims = pipeline.scratch.body_dims.as_slice();
    let aux = pipeline.scratch.aux.as_slice();
    let hidden_ns = pipeline.scratch.hidden_ns.as_slice();

    // Phase 2: uploads.
    pipeline.cells.upload_metadata(
        energies,
        headings,
        pitches,
        damage_accums,
        max_speeds,
        eff_radii,
    );
    pipeline.cells.upload_velocities(velocities);
    pipeline
        .cells
        .upload_angular_pitch(angular_vels, pitch_vels);
    pipeline.cells.upload_positions(positions);
    pipeline.cells.upload_age_cooldown(ages, cooldowns);
    pipeline.cells.upload_body_dims(body_dims);
    pipeline.cells.upload_aux(aux);
    pipeline
        .cells
        .upload_bonded_inboxes(pipeline.scratch.bonded_inboxes.as_slice());

    let sensor_params = SensorParamsGpu {
        num_cells: n as u32,
        num_foods: food_positions.len() as u32,
        hash_cell_size: bioscape::GRID_CELL_SIZE,
        world_half_x: world_half[0],
        world_half_y: world_half[1],
        world_half_z: world_half[2],
        field_res_x: SMELL_GRID_RES as u32,
        field_res_y: SMELL_GRID_RES as u32,
        field_res_z: SMELL_GRID_RES_Z as u32,
        field_eps: SMELL_SAMPLE_EPSILON,
        field_world_half_x: world_half[0],
        field_world_half_y: world_half[1],
        field_world_half_z: world_half[2],
        _pad0: 0,
    };
    let populate_params = PopulateInputsParams {
        num_cells: n as u32,
        brain_inputs: BRAIN_INPUTS as u32,
        brain_inputs_sensory: BRAIN_INPUTS_SENSORY as u32,
        brain_hidden: BRAIN_HIDDEN as u32,
        brain_recurrent: BRAIN_RECURRENT as u32,
        smell_norm_gain: SMELL_NORMALIZATION_GAIN,
        phero_norm_gain: PHEROMONE_NORMALIZATION_GAIN,
        damage_norm_gain: DAMAGE_NORMALIZATION_GAIN,
        density_norm: DENSITY_NORM_COUNT,
        reproduce_threshold: REPRODUCE_THRESHOLD,
        vibration_norm_gain: bioscape::VIBRATION_NORMALIZATION_GAIN,
        _pad0: 0,
    };
    let has_z = world_half[2] > 0.0;
    let step_params = StepParamsGpu {
        num_cells: n as u32,
        _pad_a0: 0,
        _pad_a1: 0,
        _pad_a2: 0,
        dt,
        world_half_x: world_half[0],
        world_half_y: world_half[1],
        world_half_z: world_half[2],
        gravity: PHYS_GRAVITY,
        drag: PHYSICS_CONFIG.drag,
        angular_drag: PHYSICS_CONFIG.angular_drag,
        energy_cost_per_v_sq: PHYSICS_CONFIG.energy_cost_per_v_sq,
        angular_energy_cost: PHYSICS_CONFIG.angular_energy_cost,
        vision_cost_per_radius: PHYSICS_CONFIG.vision_cost_per_radius,
        body_cost_factor: PHYSICS_CONFIG.body_cost_factor,
        age_decay_per_sec: AGE_DECAY_PER_SEC,
        fixed_timestep_hz: FIXED_TIMESTEP_HZ,
        spike_cost_per_sec: SPIKE_COST_PER_SEC,
        shell_cost_per_sec: SHELL_COST_PER_SEC,
        attack_cost_per_sec: ATTACK_COST_PER_SEC,
        pitch_clamp: core::f32::consts::FRAC_PI_6 * 0.5,
        thermal_top: bioscape::THERMAL_TOP,
        thermal_bottom: bioscape::THERMAL_BOTTOM,
        thermal_q10: bioscape::THERMAL_Q10,
        thermal_ref_temp: bioscape::THERMAL_REF_TEMP,
        thermal_diurnal_amp: bioscape::THERMAL_DIURNAL_AMP,
        thermal_seasonal_amp: bioscape::THERMAL_SEASONAL_AMP,
        thermal_diurnal_phase: (clock.0.tick % bioscape::THERMAL_DIURNAL_PERIOD_TICKS)
            as f32
            / bioscape::THERMAL_DIURNAL_PERIOD_TICKS as f32,
        thermal_seasonal_phase: (clock.0.generation % CYCLE_GEN_PERIOD) as f32
            / CYCLE_GEN_PERIOD as f32,
        thermal_log2_q10: bioscape::THERMAL_Q10.log2(),
        // Wave 4: maze fields. Mask uploaded once when MazeWorld toggle
        // flipped on (input.rs::toggle_maze_world).
        maze_active: if maze.is_active() { 1 } else { 0 },
        maze_res_x: maze
            .field
            .as_ref()
            .map(|f| f.resolution[0] as u32)
            .unwrap_or(0),
        maze_res_y: maze
            .field
            .as_ref()
            .map(|f| f.resolution[1] as u32)
            .unwrap_or(0),
    };

    let mut encoder = pipeline
        .cells
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu-full-tick"),
        });

    pipeline.cell_hash.dispatch_into(&mut encoder, positions);
    pipeline.food_hash.dispatch_into(&mut encoder, food_positions);
    pipeline.sensor.dispatch_no_readback_into(
        &mut encoder,
        positions,
        eff_radii,
        vision_radii,
        food_positions,
        &pipeline.cell_hash,
        &pipeline.food_hash,
        &pipeline.smell,
        &pipeline.pheromone,
        &pipeline.vibration,
        sensor_params,
    );
    pipeline
        .populate
        .dispatch_into(&mut encoder, &pipeline.cells, &pipeline.sensor, populate_params);
    pipeline.brain.forward_persistent_into(&mut encoder, &pipeline.cells, n, hidden_ns);
    pipeline.motor.dispatch_with_cells_into(
        &mut encoder,
        &pipeline.cells,
        n,
        dt,
        DRAG_COEFFICIENT,
    );
    pipeline.brownian.compute_persistent_into(
        &mut encoder,
        &pipeline.cells,
        n,
        THERMAL_NOISE,
        dt,
        has_z,
    );
    pipeline
        .step
        .dispatch_with_cells_into(&mut encoder, &pipeline.cells, n, step_params);
    pipeline.cells.download_full_copy_into(&mut encoder, n);

    pipeline
        .cells
        .queue()
        .submit(Some(encoder.finish()));

    pipeline.cells.download_full_read_into(
        n,
        &mut pipeline.scratch.dl_hiddens,
        &mut pipeline.scratch.dl_outputs,
        &mut pipeline.scratch.dl_velocities,
        &mut pipeline.scratch.dl_angular,
        &mut pipeline.scratch.dl_pitch,
        &mut pipeline.scratch.dl_positions,
        &mut pipeline.scratch.dl_ages,
        &mut pipeline.scratch.dl_cooldowns,
        &mut pipeline.scratch.dl_energies,
    );

    // Phase 11: parallel writeback. Each cell's fields come from `slot` in
    // each `dl_*` Vec — same per-slot ownership invariant as the snapshot.
    let wb_hiddens = ConstPtr::<[f32; BRAIN_HIDDEN]>(pipeline.scratch.dl_hiddens.as_ptr());
    let wb_outputs = ConstPtr::<[f32; bioscape::BRAIN_OUTPUTS]>(pipeline.scratch.dl_outputs.as_ptr());
    let wb_velocities = ConstPtr::<[f32; 3]>(pipeline.scratch.dl_velocities.as_ptr());
    let wb_angular = ConstPtr::<f32>(pipeline.scratch.dl_angular.as_ptr());
    let wb_pitch = ConstPtr::<f32>(pipeline.scratch.dl_pitch.as_ptr());
    let wb_positions = ConstPtr::<[f32; 3]>(pipeline.scratch.dl_positions.as_ptr());
    let wb_ages = ConstPtr::<u32>(pipeline.scratch.dl_ages.as_ptr());
    let wb_cooldowns = ConstPtr::<u32>(pipeline.scratch.dl_cooldowns.as_ptr());
    let wb_energies = ConstPtr::<f32>(pipeline.scratch.dl_energies.as_ptr());
    cells.par_iter_mut().for_each(|(entity, mut cell_entity)| {
        let Some(slot) = slot_map_ref.slot_of(entity) else { return };
        let cell = &mut cell_entity.0;
        // SAFETY: `slot < n` and each slot is consumed by one closure.
        unsafe {
            cell.last_hidden = *wb_hiddens.add(slot);
            cell.last_outputs = *wb_outputs.add(slot);
            cell.velocity = *wb_velocities.add(slot);
            cell.angular_velocity = *wb_angular.add(slot);
            cell.pitch_velocity = *wb_pitch.add(slot);
            cell.position = *wb_positions.add(slot);
            cell.age = (*wb_ages.add(slot)) as u64;
            cell.reproduce_cooldown_ticks = *wb_cooldowns.add(slot);
            cell.energy = *wb_energies.add(slot);
        }
        cell.damage_accum = 0.0;
    });

    diag.add_measurement(&DIAG_BRAIN_ACT, || t_total.elapsed().as_secs_f64() * 1000.0);
}
