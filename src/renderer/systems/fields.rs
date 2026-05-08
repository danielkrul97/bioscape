use bevy::diagnostic::Diagnostics;
use bevy::prelude::*;
use bioscape::{
    N_PHEROMONE_CHANNELS, PHEROMONE_BASELINE_EMIT, PHEROMONE_BRAIN_MOD, PHEROMONE_COST_PER_RATE,
    PHEROMONE_DECAY_PER_CH, PHEROMONE_DIFFUSION_PER_CH, SMELL_DECAY, SMELL_DIFFUSION,
    SMELL_PER_FOOD,
};
use std::time::Instant;

use super::super::components::{CellEntity, Dying, FoodEntity};
use super::super::config::{DIAG_FOOD_COUNT, DIAG_PHEROMONE, DIAG_SMELL};
use super::super::resources::{PheromoneResource, SmellResource};
#[cfg(feature = "gpu")]
use super::super::resources_gpu::{GpuFieldState, GpuFullPipeline};

pub(crate) fn update_smell_field(
    time: Res<Time>,
    foods: Query<&FoodEntity>,
    mut smell: ResMut<SmellResource>,
    #[cfg(feature = "gpu")] gpu_field: Option<ResMut<GpuFieldState>>,
    #[cfg(feature = "gpu")] gpu_full: Option<ResMut<GpuFullPipeline>>,
    mut diag: Diagnostics,
) {
    let t = Instant::now();
    let dt = time.delta_secs();
    diag.add_measurement(&DIAG_FOOD_COUNT, || foods.iter().count() as f64);

    // Full GPU pipeline: pipeline.smell read přímo sensor shaderem v
    // `cells_brain_act_gpu_full` přes storage binding — žádný CPU readback,
    // SmellResource neaktualizujeme (CPU sensor path stejně skipne).
    #[cfg(feature = "gpu")]
    if let Some(mut gpu_full) = gpu_full {
        for food in &foods {
            gpu_full.smell.add_source(
                [food.0.position[0], food.0.position[1], food.0.position[2]],
                SMELL_PER_FOOD * dt,
            );
        }
        gpu_full.smell.step(SMELL_DIFFUSION, SMELL_DECAY, dt);
        let _ = smell;
        diag.add_measurement(&DIAG_SMELL, || t.elapsed().as_secs_f64() * 1000.0);
        return;
    }

    // Sprint 59: pokud GpuFieldState available, GPU deposit + diffuse, readback
    // do CPU SmellResource pro sensor gather (gradient_at v cells_brain_act).
    #[cfg(feature = "gpu")]
    if let Some(mut gpu) = gpu_field {
        for food in &foods {
            gpu.smell.add_source(
                [food.0.position[0], food.0.position[1], food.0.position[2]],
                SMELL_PER_FOOD * dt,
            );
        }
        gpu.smell.step(SMELL_DIFFUSION, SMELL_DECAY, dt);
        let grid = gpu.smell.download();
        smell.0.replace_grid_from(&grid);
        diag.add_measurement(&DIAG_SMELL, || t.elapsed().as_secs_f64() * 1000.0);
        return;
    }

    for food in &foods {
        smell
            .0
            .add_source([food.0.position[0], food.0.position[1], food.0.position[2]], SMELL_PER_FOOD * dt);
    }
    smell.0.step(SMELL_DIFFUSION, SMELL_DECAY, dt);
    diag.add_measurement(&DIAG_SMELL, || t.elapsed().as_secs_f64() * 1000.0);
}

pub(crate) fn update_pheromone_field(
    time: Res<Time>,
    mut pheromone: ResMut<PheromoneResource>,
    #[cfg(feature = "gpu")] gpu_field: Option<ResMut<GpuFieldState>>,
    #[cfg(feature = "gpu")] gpu_full: Option<ResMut<GpuFullPipeline>>,
    mut diag: Diagnostics,
) {
    // Diffuse + decay BEFORE this tick's emissions (in emit_pheromones, which
    // runs after brain_act). Stejně jako headless — brainy detekují gradient
    // ze stavu pole na konci minulého ticku, žádný self-feedback.
    let t = Instant::now();
    let dt = time.delta_secs();

    // Full GPU pipeline: ch0 step na pipeline.pheromone bez readback (sensor
    // shader čte storage buffer direct). ch1/ch2 vždy CPU.
    #[cfg(feature = "gpu")]
    if let Some(mut gpu_full) = gpu_full {
        gpu_full
            .pheromone
            .step(PHEROMONE_DIFFUSION_PER_CH[0], PHEROMONE_DECAY_PER_CH[0], dt);
        for ch in 1..N_PHEROMONE_CHANNELS {
            pheromone.fields[ch].step(
                PHEROMONE_DIFFUSION_PER_CH[ch],
                PHEROMONE_DECAY_PER_CH[ch],
                dt,
            );
        }
        diag.add_measurement(&DIAG_PHEROMONE, || t.elapsed().as_secs_f64() * 1000.0);
        return;
    }

    // Sprint 126: ch0 GPU path zachovaný (FieldGpu má jen single channel).
    // ch1/ch2 vždy CPU step — nárůst load je marginal (2× další 64×64×16 grid
    // diffusion).
    #[cfg(feature = "gpu")]
    if let Some(mut gpu) = gpu_field {
        gpu.pheromone.step(PHEROMONE_DIFFUSION_PER_CH[0], PHEROMONE_DECAY_PER_CH[0], dt);
        let grid = gpu.pheromone.download();
        pheromone.fields[0].replace_grid_from(&grid);
        for ch in 1..N_PHEROMONE_CHANNELS {
            pheromone.fields[ch].step(PHEROMONE_DIFFUSION_PER_CH[ch], PHEROMONE_DECAY_PER_CH[ch], dt);
        }
        diag.add_measurement(&DIAG_PHEROMONE, || t.elapsed().as_secs_f64() * 1000.0);
        return;
    }

    for ch in 0..N_PHEROMONE_CHANNELS {
        pheromone.fields[ch].step(PHEROMONE_DIFFUSION_PER_CH[ch], PHEROMONE_DECAY_PER_CH[ch], dt);
    }
    diag.add_measurement(&DIAG_PHEROMONE, || t.elapsed().as_secs_f64() * 1000.0);
}

pub(crate) fn emit_pheromones(
    time: Res<Time>,
    mut pheromone: ResMut<PheromoneResource>,
    #[cfg(feature = "gpu")] gpu_field: Option<ResMut<GpuFieldState>>,
    #[cfg(feature = "gpu")] gpu_full: Option<ResMut<GpuFullPipeline>>,
    mut cells: Query<&mut CellEntity, Without<Dying>>,
) {
    let dt = time.delta_secs();

    // Sprint 126: per-channel emission. Brain output sloty:
    //   [2]  = ch0 emit (existing slow channel, mating-friendly)
    //   [10] = ch1 emit (medium decay)
    //   [11] = ch2 emit (fast decay, bursty)
    // Cost = sum of all positive emissions × PHEROMONE_COST_PER_RATE.
    const EMIT_SLOTS: [usize; N_PHEROMONE_CHANNELS] = [2, 10, 11];

    #[cfg(feature = "gpu")]
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
                if ch == 0 {
                    gpu_full.pheromone.add_source(pos, rate * dt);
                } else {
                    pheromone.fields[ch].add_source(pos, rate * dt);
                }
                let prev = cell.0.last_emit[ch];
                let delta = brain_emit - prev;
                cell.0.burst_accum[ch] += delta * delta;
            }
            cell.0.last_emit = emits;
            cell.0.energy -= PHEROMONE_COST_PER_RATE * total_emit * dt;
        }
        return;
    }

    #[cfg(feature = "gpu")]
    if let Some(mut gpu) = gpu_field {
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
                if ch == 0 {
                    gpu.pheromone.add_source(pos, rate * dt);
                } else {
                    pheromone.fields[ch].add_source(pos, rate * dt);
                }
                let prev = cell.0.last_emit[ch];
                let delta = brain_emit - prev;
                cell.0.burst_accum[ch] += delta * delta;
            }
            cell.0.last_emit = emits;
            cell.0.energy -= PHEROMONE_COST_PER_RATE * total_emit * dt;
        }
        return;
    }

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
            pheromone.fields[ch].add_source(pos, rate * dt);
            let prev = cell.0.last_emit[ch];
            let delta = brain_emit - prev;
            cell.0.burst_accum[ch] += delta * delta;
        }
        cell.0.last_emit = emits;
        cell.0.energy -= PHEROMONE_COST_PER_RATE * total_emit * dt;
    }
}
