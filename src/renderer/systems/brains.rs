use bevy::diagnostic::Diagnostics;
use bevy::prelude::*;
use bioscape::{
    BRAIN_INPUTS, N_PHEROMONE_CHANNELS, PHEROMONE_SAMPLE_EPSILON, SMELL_SAMPLE_EPSILON, WORLD_HALF,
};
use rustc_hash::FxHashMap;
use std::time::Instant;

use super::super::components::{CellEntity, Dying};
#[cfg(feature = "gpu")]
use super::super::config::DIAG_BRAIN_GPU_RT;
use super::super::config::{DIAG_BRAIN_ACT, DIAG_CELL_COUNT};
use super::super::resources::{
    CellGrid, CellSlotMap, Clock, CoopFoodResource, FoodGrid, PheromoneResource, SmellResource,
};
#[cfg(feature = "gpu")]
use super::super::resources_gpu::{GpuBrainState, GpuFullPipeline};

/// Sprint 94: pre-brain pass. Compute `pooled_hidden` per cell = mean
/// `last_hidden` over self + bonded partners (1-hop). Cluster cells získají
/// shared recurrent state. Solo cells: pooled == self. Runs before
/// `cells_brain_act` v `FixedUpdate` chain.
pub(crate) fn pool_bonded_hidden_cells(
    mut cells: Query<&mut CellEntity, Without<Dying>>,
    mut id_to_hidden_scratch: Local<FxHashMap<u64, [f32; bioscape::BRAIN_HIDDEN]>>,
) {
    // R-#7: persistent Local scratch — pre-fix `Vec` snapshot + fresh
    // FxHashMap collect per tick. Direct insert eliminuje intermediate Vec.
    id_to_hidden_scratch.clear();
    for c in cells.iter() {
        id_to_hidden_scratch.insert(c.0.cell_id, c.0.last_hidden);
    }
    if id_to_hidden_scratch.is_empty() {
        return;
    }
    let id_to_hidden = &*id_to_hidden_scratch;
    for mut cell in &mut cells {
        let pooled = bioscape::pool_bonded_hidden(&cell.0, |partner_id| {
            if partner_id == cell.0.cell_id {
                return None;
            }
            id_to_hidden.get(&partner_id).copied()
        });
        cell.0.pooled_hidden = pooled;
    }
}

pub(crate) fn cells_brain_act(
    time: Res<Time>,
    cell_grid: Res<CellGrid>,
    food_grid: Res<FoodGrid>,
    smell: Res<SmellResource>,
    pheromone: Res<PheromoneResource>,
    coop_foods: Res<CoopFoodResource>,
    slot_map: Res<CellSlotMap>,
    clock: Res<Clock>,
    #[cfg(feature = "gpu")] gpu_state: Option<ResMut<GpuBrainState>>,
    #[cfg(feature = "gpu")] gpu_full: Option<Res<GpuFullPipeline>>,
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
    mut diag: Diagnostics,
    mut coop_positions_scratch: Local<Vec<[f32; 3]>>,
    mut id_to_inputs_scratch: Local<FxHashMap<u64, [f32; BRAIN_INPUTS]>>,
    #[cfg(feature = "gpu")] mut inputs_by_slot_scratch: Local<Vec<[f32; BRAIN_INPUTS]>>,
) {
    // Full GPU pipeline: separate `cells_brain_act_gpu_full` system handles
    // all of brain_act + motor + step + brownian on GPU; this CPU/GPU-brain-only
    // path is no-op when gpu_full is active.
    #[cfg(feature = "gpu")]
    if gpu_full.is_some() {
        return;
    }
    let _t_total = Instant::now();
    let dt = time.delta_secs();
    diag.add_measurement(&DIAG_CELL_COUNT, || cells.iter().count() as f64);

    // Sprint 87: clock pro thermal_local sensor. Capture u64 mimo closure aby
    // kopie šly do Bevy threadu bez dalšího `Clock` borrow.
    let tick = clock.0.tick;
    let gen = clock.0.generation;

    // Sprint 130: par_iter_mut sensor gather (Bevy QueryParIter::for_each).
    // Pre-S130 sériový for-loop přes cells trval ~6.5 ms / tick (gather hits
    // cell_grid + food_grid radius queries — drahé per-cell). Captures jsou
    // všechny &Sync (food_grid, cell_grid, smell, pheromone). Per-cell scratch
    // pad: zapíšeme post-gain inputs do `cell.last_inputs`, použijeme jako
    // canonical source pro pool_bonded_sensors / GPU upload v následujícím
    // sequential pass — last_inputs se stejně přepisují na konci brain_act,
    // takže scratch use je behavior-neutral.
    coop_positions_scratch.clear();
    coop_positions_scratch.extend(coop_foods.0.iter().map(|c| c.position));
    let coop_positions_ref = coop_positions_scratch.as_slice();
    let food_grid_ref = &food_grid.0;
    let cell_grid_ref = &cell_grid.0;
    let smell_ref = &smell.0;
    let pheromone_ref = &pheromone.fields;
    cells.par_iter_mut().for_each(|(entity, mut cell)| {
        let pos = cell.0.position;
        let vision_r = cell.0.genome.vision_radius;
        let vr2 = vision_r * vision_r;
        let fov = cell.0.genome.vision_fov;
        let skip_cone = fov >= bioscape::MAX_VISION_FOV;
        let cos_fov = fov.cos();
        let fwd = bioscape::forward_vector(cell.0.heading, cell.0.pitch);
        let mut nearest_food: Option<[f32; 3]> = None;
        let mut best_food_d2 = f32::MAX;
        food_grid_ref.for_each_in_radius_toroidal(pos, vision_r, WORLD_HALF, |_, fp, _| {
            let d = bioscape::min_image_delta(pos, fp, WORLD_HALF);
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if d2 > vr2 || d2 >= best_food_d2 {
                return;
            }
            if !skip_cone && !bioscape::fov_cone_accept(d, d2, fwd, cos_fov) {
                return;
            }
            best_food_d2 = d2;
            nearest_food = Some(d);
        });
        for cp in coop_positions_ref.iter() {
            let d = bioscape::min_image_delta(pos, *cp, WORLD_HALF);
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if d2 > vr2 || d2 >= best_food_d2 {
                continue;
            }
            if !skip_cone && !bioscape::fov_cone_accept(d, d2, fwd, cos_fov) {
                continue;
            }
            best_food_d2 = d2;
            nearest_food = Some(d);
        }
        let mut nearest_cell: Option<([f32; 3], f32)> = None;
        let mut best_cell_d2 = f32::MAX;
        let mut neighbors_in_vision: u32 = 0;
        cell_grid_ref.for_each_in_radius_toroidal(
            pos,
            vision_r,
            WORLD_HALF,
            |other, other_pos, other_radius| {
                if other == entity {
                    return;
                }
                let d = bioscape::min_image_delta(pos, other_pos, WORLD_HALF);
                let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if d2 > vr2 {
                    return;
                }
                if !skip_cone && !bioscape::fov_cone_accept(d, d2, fwd, cos_fov) {
                    return;
                }
                neighbors_in_vision += 1;
                if d2 < best_cell_d2 {
                    best_cell_d2 = d2;
                    nearest_cell = Some((d, other_radius));
                }
            },
        );
        let pos_xyz = [pos[0], pos[1], pos[2]];
        let smell_grad = smell_ref.gradient_at(pos_xyz, SMELL_SAMPLE_EPSILON);
        let mut pheromone_grads = [[0.0_f32; 3]; N_PHEROMONE_CHANNELS];
        for ch in 0..N_PHEROMONE_CHANNELS {
            pheromone_grads[ch] =
                pheromone_ref[ch].gradient_at(pos_xyz, PHEROMONE_SAMPLE_EPSILON);
        }
        let temperature_local = bioscape::temperature_at_z(pos[2], WORLD_HALF, tick, gen);
        let sensors = bioscape::BrainSensors {
            nearest_food,
            nearest_cell,
            neighbors_in_vision,
            smell_grad,
            pheromone_grads,
            temperature_local,
        };
        cell.0.apply_shell_absorb(dt);
        // eat_food skip optim: cache d² k nejbližšímu food pro `cell_eats_food`
        // early skip kandidát gather (`f32::MAX` pokud sensor nic nenašel).
        cell.0.last_best_food_d2 = best_food_d2;
        let mut inputs = bioscape::populate_brain_inputs(&mut cell.0, &sensors, vision_r);
        bioscape::apply_sensor_gains(&mut inputs, &cell.0.genome.sensor_gains);
        cell.0.last_inputs = inputs;
    });

    // Phase 2: serial HashMap build z post-gain inputs uložených v cell.last_inputs.
    // Persistent Local scratch — pre-fix `FxHashMap::default()` per tick.
    id_to_inputs_scratch.clear();
    for (_entity, cell) in cells.iter() {
        id_to_inputs_scratch.insert(cell.0.cell_id, cell.0.last_inputs);
    }
    let id_to_inputs = &*id_to_inputs_scratch;

    #[cfg(feature = "gpu")]
    if let Some(mut gpu) = gpu_state {
        let n = slot_map.len();
        if n == 0 {
            diag.add_measurement(&DIAG_BRAIN_ACT, || _t_total.elapsed().as_secs_f64() * 1000.0);
            return;
        }
        // Build inputs vec indexed by slot. Iterate alive query, look up slot,
        // place inputs at slot index. Slots jsou dense 0..n.
        // Persistent Local scratch — pre-fix `vec![[0.0; INPUTS]; n]` (~640 KB
        // pro n=2000) per tick.
        inputs_by_slot_scratch.clear();
        inputs_by_slot_scratch.resize(n, [0.0; BRAIN_INPUTS]);
        let inputs_by_slot = &mut *inputs_by_slot_scratch;
        for (entity, cell) in cells.iter() {
            let Some(slot) = slot_map.slot_of(entity) else {
                continue;
            };
            let own = id_to_inputs
                .get(&cell.0.cell_id)
                .copied()
                .unwrap_or([0.0; BRAIN_INPUTS]);
            inputs_by_slot[slot] = bioscape::pool_bonded_sensors(&cell.0, &own, |partner_id| {
                if partner_id == cell.0.cell_id {
                    return None;
                }
                id_to_inputs.get(&partner_id).copied()
            });
        }
        let t_gpu = Instant::now();
        gpu.cells.upload_inputs(&inputs_by_slot);
        let gpu = &mut *gpu;
        gpu.brain.forward_persistent(&gpu.cells, n);
        let (hiddens, outputs) = gpu.cells.download_hidden_outputs(n);
        diag.add_measurement(&DIAG_BRAIN_GPU_RT, || t_gpu.elapsed().as_secs_f64() * 1000.0);
        for (entity, mut cell) in &mut cells {
            let Some(slot) = slot_map.slot_of(entity) else {
                continue;
            };
            cell.0.last_inputs = inputs_by_slot[slot];
            cell.0.last_hidden = hiddens[slot];
            cell.0.last_outputs = outputs[slot];
            cell.0.apply_brain_motor(&outputs[slot], dt);
        }
        diag.add_measurement(&DIAG_BRAIN_ACT, || _t_total.elapsed().as_secs_f64() * 1000.0);
        return;
    }

    // CPU fallback (no GPU available or feature disabled).
    // Sprint 113: par_iter_mut — pool_bonded_sensors + forward_with_state jen
    // čtou (`&self`); jediná mutace je self-cell (last_inputs/hidden/outputs +
    // apply_brain_motor). id_to_inputs HashMap se sdílí immutable přes Send+Sync.
    let _ = slot_map;
    let id_to_inputs_ref = &id_to_inputs;
    cells.par_iter_mut().for_each(|(_entity, mut cell)| {
        let own = id_to_inputs_ref
            .get(&cell.0.cell_id)
            .copied()
            .unwrap_or([0.0; BRAIN_INPUTS]);
        let inputs = bioscape::pool_bonded_sensors(&cell.0, &own, |partner_id| {
            if partner_id == cell.0.cell_id {
                return None;
            }
            id_to_inputs_ref.get(&partner_id).copied()
        });
        let (hidden, outputs) = cell.0.genome.brain.forward_with_state(&inputs);
        cell.0.last_inputs = inputs;
        cell.0.last_hidden = hidden;
        cell.0.last_outputs = outputs;
        cell.0.apply_brain_motor(&outputs, dt);
    });
    diag.add_measurement(&DIAG_BRAIN_ACT, || _t_total.elapsed().as_secs_f64() * 1000.0);
}

pub(crate) fn apply_cell_morph(time: Res<Time>, mut cells: Query<&mut CellEntity, Without<Dying>>) {
    let dt = time.delta_secs();
    cells.par_iter_mut().for_each(|mut cell| {
        cell.0.apply_morph(dt);
    });
}
