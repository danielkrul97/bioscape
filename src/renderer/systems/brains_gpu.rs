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
use super::super::resources::{CellSlotMap, Clock, CoopFoodResource, WorldExtent};
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
    mut cells: Query<&mut CellEntity, Without<Dying>>,
    foods: Query<&FoodEntity>,
    coop_foods: Res<CoopFoodResource>,
    slot_map: Res<CellSlotMap>,
    mut pipeline: ResMut<GpuFullPipeline>,
    fixed_time: Res<Time<Fixed>>,
    clock: Res<Clock>,
    extent: Res<WorldExtent>,
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

    // Phase 1: CPU snapshot — iterujeme cells v slot order (slot_map.slot_to_entity)
    // aby buffer indexy odpovídaly GPU slotům. apply_shell_absorb mutuje
    // damage_accum předem; last_best_food_d2 = 0.0 disable eat_food skip
    // (sensor běžel na GPU, CPU nemá přístup k best_food_d2).
    let food_n = foods.iter().count() + coop_foods.0.len();
    pipeline.scratch.clear_and_reserve(n, food_n);
    for slot in 0..n {
        let entity = slot_map.slot_to_entity[slot];
        let Ok(mut cell_entity) = cells.get_mut(entity) else { continue };
        let cell = &mut cell_entity.0;
        cell.apply_shell_absorb(dt);
        cell.last_best_food_d2 = 0.0;
        let s = &mut pipeline.scratch;
        s.positions.push(cell.position);
        s.eff_radii.push(cell.phenotype.effective_radius());
        s.vision_radii.push(cell.genome.vision_radius);
        s.energies.push(cell.energy);
        s.headings.push(cell.heading);
        s.pitches.push(cell.pitch);
        s.damage_accums.push(cell.damage_accum);
        s.max_speeds.push(cell.genome.max_speed);
        s.velocities.push(cell.velocity);
        s.angular_vels.push(cell.angular_velocity);
        s.pitch_vels.push(cell.pitch_velocity);
        s.ages.push(cell.age as u32);
        s.cooldowns.push(cell.reproduce_cooldown_ticks);
        s.body_dims.push([
            cell.phenotype.body_length,
            cell.phenotype.body_width,
            cell.phenotype.body_height,
        ]);
        s.aux.push([
            cell.phenotype.total_spike_cost_factor(),
            cell.phenotype.shell_thickness,
            cell.genome.vision_radius,
            cell.last_outputs[6].max(0.0),
        ]);
        s.hidden_ns.push(cell.genome.brain.hidden_n);
    }
    // Foods + coop_foods do single sensor pool.
    let s = &mut pipeline.scratch;
    for food in &foods {
        s.food_positions.push(food.0.position);
    }
    for coop in coop_foods.0.iter() {
        s.food_positions.push(coop.position);
    }

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
        _pad0: 0,
        _pad1: 0,
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

    // Phase 11: writeback to ECS — iterujeme slot order, get_mut entity, write
    // back fields. damage_accum reset (mirror populate_inputs shader behavior).
    let dl = &pipeline.scratch;
    for slot in 0..n {
        let entity = slot_map.slot_to_entity[slot];
        let Ok(mut cell_entity) = cells.get_mut(entity) else { continue };
        let cell = &mut cell_entity.0;
        cell.last_hidden = dl.dl_hiddens[slot];
        cell.last_outputs = dl.dl_outputs[slot];
        cell.velocity = dl.dl_velocities[slot];
        cell.angular_velocity = dl.dl_angular[slot];
        cell.pitch_velocity = dl.dl_pitch[slot];
        cell.position = dl.dl_positions[slot];
        cell.age = dl.dl_ages[slot] as u64;
        cell.reproduce_cooldown_ticks = dl.dl_cooldowns[slot];
        cell.energy = dl.dl_energies[slot];
        cell.damage_accum = 0.0;
    }

    diag.add_measurement(&DIAG_BRAIN_ACT, || t_total.elapsed().as_secs_f64() * 1000.0);
}
