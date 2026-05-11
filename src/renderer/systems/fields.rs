use bevy::diagnostic::Diagnostics;
use bevy::prelude::*;
use bioscape::{
    N_PHEROMONE_CHANNELS, PHEROMONE_BASELINE_EMIT, PHEROMONE_BRAIN_MOD, PHEROMONE_COST_PER_RATE,
    PHEROMONE_DECAY_PER_CH, PHEROMONE_DIFFUSION_PER_CH, SMELL_DECAY, SMELL_DIFFUSION,
    SMELL_PER_FOOD, VIBRATION_DECAY, VIBRATION_DIFFUSION,
};
use std::time::Instant;

use super::super::components::{CellEntity, Dying, FoodEntity};
use super::super::config::{DIAG_FOOD_COUNT, DIAG_PHEROMONE, DIAG_SMELL, DIAG_VIBRATION};
use super::super::resources::{MazeWorld, PheromoneResource, SmellResource, VibrationResource};
use super::super::resources_gpu::GpuFullPipeline;

pub(crate) fn update_smell_field(
    time: Res<Time>,
    foods: Query<&FoodEntity>,
    smell: ResMut<SmellResource>,
    _maze: Res<MazeWorld>,
    gpu_full: Option<ResMut<GpuFullPipeline>>,
    mut diag: Diagnostics,
) {
    let t = Instant::now();
    let dt = time.delta_secs();
    diag.add_measurement(&DIAG_FOOD_COUNT, || foods.iter().count() as f64);

    // GPU pipeline owns the smell field — sensor shader reads
    // `gpu_full.smell.current_grid_buffer()` direct, no CPU readback.
    // SmellResource CPU shadow stays stale; checkpoint readback (not
    // yet wired) is the only consumer.
    if let Some(mut gpu_full) = gpu_full {
        for food in &foods {
            gpu_full.smell.add_source(
                [food.0.position[0], food.0.position[1], food.0.position[2]],
                SMELL_PER_FOOD * dt,
            );
        }
        gpu_full.smell.step(SMELL_DIFFUSION, SMELL_DECAY, dt);
    }
    let _ = smell;
    diag.add_measurement(&DIAG_SMELL, || t.elapsed().as_secs_f64() * 1000.0);
}

pub(crate) fn update_pheromone_field(
    time: Res<Time>,
    pheromone: ResMut<PheromoneResource>,
    _maze: Res<MazeWorld>,
    gpu_full: Option<ResMut<GpuFullPipeline>>,
    mut diag: Diagnostics,
) {
    // Diffuse + decay BEFORE this tick's emissions (in emit_pheromones, which
    // runs after brain_act) — brain reads gradient from the end of previous
    // tick, no self-feedback.
    let t = Instant::now();
    let dt = time.delta_secs();

    // Wave L: all 3 channels live on GPU; sensor_gather reads gradients
    // direct via storage bindings, no CPU readback.
    if let Some(mut gpu_full) = gpu_full {
        gpu_full
            .pheromone
            .step(PHEROMONE_DIFFUSION_PER_CH[0], PHEROMONE_DECAY_PER_CH[0], dt);
        gpu_full
            .pheromone_ch1
            .step(PHEROMONE_DIFFUSION_PER_CH[1], PHEROMONE_DECAY_PER_CH[1], dt);
        gpu_full
            .pheromone_ch2
            .step(PHEROMONE_DIFFUSION_PER_CH[2], PHEROMONE_DECAY_PER_CH[2], dt);
    }
    let _ = pheromone;
    diag.add_measurement(&DIAG_PHEROMONE, || t.elapsed().as_secs_f64() * 1000.0);
}

pub(crate) fn update_vibration_field(
    time: Res<Time>,
    mut vibration: ResMut<VibrationResource>,
    maze: Res<MazeWorld>,
    gpu_full: Option<ResMut<GpuFullPipeline>>,
    cells: Query<&CellEntity, Without<Dying>>,
    mut diag: Diagnostics,
) {
    // Deposit per cell, then diffuse + decay. Runs after `update_pheromone_field`
    // and before `cells_brain_act` so brains read a freshly stepped field.
    let t = Instant::now();
    let dt = time.delta_secs();

    // Full GPU pipeline: deposit + diffuse on the GPU FieldGpu so the sensor
    // gather shader can read the buffer directly via storage binding — no
    // CPU readback for the sensor stage. We do download the grid into the
    // CPU shadow at the end of the tick so the gizmo overlay (`V` toggle)
    // and any future CSV/diagnostic reader see real values; cost is one
    // ~256 KB readback per tick which is acceptable against the perf
    // headroom gained by skipping the per-tick *sensor* readback.
    if let Some(mut gpu_full) = gpu_full {
        for cell in &cells {
            let emit = bioscape::vibration_emit_for_cell(&cell.0);
            if emit > 0.0 {
                gpu_full.vibration.add_source(
                    [cell.0.position[0], cell.0.position[1], cell.0.position[2]],
                    emit * dt,
                );
            }
        }
        gpu_full
            .vibration
            .step(VIBRATION_DIFFUSION, VIBRATION_DECAY, dt);
        // CPU vibration shadow refresh — gizmo overlay (`V` toggle) reads
        // VibrationResource. Cost: one ~256 KB readback/tick.
        let grid = gpu_full.vibration.download();
        vibration.0.replace_grid_from(&grid);
    }
    let _ = (maze, cells);
    diag.add_measurement(&DIAG_VIBRATION, || t.elapsed().as_secs_f64() * 1000.0);
}

pub(crate) fn emit_pheromones(
    time: Res<Time>,
    pheromone: ResMut<PheromoneResource>,
    gpu_full: Option<ResMut<GpuFullPipeline>>,
    mut cells: Query<&mut CellEntity, Without<Dying>>,
) {
    let dt = time.delta_secs();

    // Per-channel emission. Brain output slots:
    //   [2]  = ch0 emit (slow, mating-friendly)
    //   [10] = ch1 emit (medium decay)
    //   [11] = ch2 emit (fast decay, bursty)
    // Cost = sum of all positive emissions × PHEROMONE_COST_PER_RATE.
    const EMIT_SLOTS: [usize; N_PHEROMONE_CHANNELS] = [2, 10, 11];

    if let Some(mut gpu_full) = gpu_full {
        for mut cell in &mut cells {
            let pos = [cell.0.position[0], cell.0.position[1], cell.0.position[2]];
            let mut total_emit = 0.0_f32;
            let mut emits = [0.0_f32; N_PHEROMONE_CHANNELS];
            for ch in 0..N_PHEROMONE_CHANNELS {
                let mod_strength = cell.0.last_outputs[EMIT_SLOTS[ch]].max(0.0);
                let brain_emit = PHEROMONE_BRAIN_MOD * mod_strength;
                emits[ch] = brain_emit;
                total_emit += brain_emit;
                let rate = PHEROMONE_BASELINE_EMIT + brain_emit;
                match ch {
                    0 => gpu_full.pheromone.add_source(pos, rate * dt),
                    1 => gpu_full.pheromone_ch1.add_source(pos, rate * dt),
                    _ => gpu_full.pheromone_ch2.add_source(pos, rate * dt),
                }
                let prev = cell.0.last_emit[ch];
                let delta = brain_emit - prev;
                cell.0.burst_accum[ch] += delta * delta;
            }
            cell.0.last_emit = emits;
            cell.0.energy -= PHEROMONE_COST_PER_RATE * total_emit * dt;
        }
    }
    let _ = pheromone;
}

