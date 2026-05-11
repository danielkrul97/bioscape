use bevy::prelude::*;
use rustc_hash::FxHashMap;

use super::super::components::{CellEntity, Dying};

/// Wave 3: per-tick eligibility-trace decay+accumulate. Runs after the
/// brain forward pass so traces capture this tick's activations before any
/// reward event (eat / predation / novelty) fires. Always-on, independent
/// of maze toggle. Runs every tick so SIMD pressure matters — the
/// per-cell `Brain::hebbian_step` body is a hot path.
///
/// Wave 7: when `GpuFullPipeline` is active, the equivalent shader
/// (`hebbian_step.wgsl`) runs on-device against `cells.brain_traces`.
/// Skip the CPU pass to avoid double-decay; CPU `genome.brain.trace_w*`
/// stays stale until next-gen sync.
pub(crate) fn apply_eligibility_step(
    time: Res<Time>,
    mut cells: Query<&mut CellEntity, Without<Dying>>,
    gpu_full: Option<ResMut<super::super::resources_gpu::GpuFullPipeline>>,
) {
    let dt = time.delta_secs();
    if let Some(gpu) = gpu_full {
        let gpu = &*gpu;
        let n = cells.iter().count();
        if n > 0 {
            gpu.hebbian.dispatch_step_persistent(
                &gpu.cells,
                n,
                dt,
                bioscape::HEBBIAN_TRACE_DECAY_PER_SEC,
            );
        }
        return;
    }
    cells.par_iter_mut().for_each(|mut cell| {
        let last_inputs = cell.0.last_inputs;
        let last_hidden = cell.0.last_hidden;
        let last_outputs = cell.0.last_outputs;
        cell.0.genome.brain.hebbian_step(
            &last_inputs,
            &last_hidden,
            &last_outputs,
            dt,
            bioscape::HEBBIAN_TRACE_DECAY_PER_SEC,
        );
    });
}

/// Wave 2: episodic novelty reward pass. Runs after motion (positions
/// reflect this tick's outcome). For each cell, bins position to a coarse
/// novelty grid; if not in the cell's recent visit history, fires a small
/// Hebbian reward against last-tick activations. Encourages exploration.
/// Always-on regardless of maze toggle. GPU full pipeline reads brain
/// weights persistent on GPU — we sync them back at the end of generation
/// (`sync_brains_from_gpu`), so on `--gpu-full` the CPU novelty Hebbian
/// patches the persistent CPU shadow only; weights re-sync next gen. Live
/// effect lands during the next reproduce/CPPN regenerate. Acceptable Wave
/// 2 trade-off; Wave 3 brings GPU novelty hook.
pub(crate) fn apply_episodic_novelty(
    extent: Res<super::super::resources::WorldExtent>,
    mut cells: Query<(Entity, &mut CellEntity), Without<Dying>>,
    slot_map: Res<super::super::resources::CellSlotMap>,
    gpu_full: Option<ResMut<super::super::resources_gpu::GpuFullPipeline>>,
    mut novelty_rewards_scratch: Local<Vec<f32>>,
) {
    let half = extent.as_array();
    if let Some(mut gpu) = gpu_full {
        let n = slot_map.len();
        if n == 0 {
            return;
        }
        novelty_rewards_scratch.clear();
        novelty_rewards_scratch.resize(n, 0.0);
        let mut any = false;
        for (entity, mut cell) in &mut cells {
            let Some(slot) = slot_map.slot_of(entity) else {
                continue;
            };
            let v = bioscape::Cell::novelty_voxel_index(cell.0.position, half);
            if cell.0.check_novelty(v) {
                novelty_rewards_scratch[slot] = bioscape::NOVELTY_REWARD_MAGNITUDE;
                any = true;
            }
        }
        if any {
            let pipeline = &mut *gpu;
            pipeline.cells.upload_rewards(&novelty_rewards_scratch);
            pipeline
                .hebbian
                .dispatch_apply_reward_persistent(&pipeline.cells, n, bioscape::LEARNING_RATE);
        }
        return;
    }
    cells.par_iter_mut().for_each(|(_entity, mut cell)| {
        let v = bioscape::Cell::novelty_voxel_index(cell.0.position, half);
        if !cell.0.check_novelty(v) {
            return;
        }
        let last_hidden = cell.0.last_hidden;
        let last_outputs = cell.0.last_outputs;
        // Wave 3: trace-based reward.
        cell.0.genome.brain.hebbian_apply_reward(
            &last_hidden,
            &last_outputs,
            bioscape::NOVELTY_REWARD_MAGNITUDE,
            bioscape::LEARNING_RATE,
        );
    });
}

/// Wave 2: per-cell whisker raycast pass. Fills `cell.last_whisker_distances`
/// from `MazeWorld.field` so the sensor gather closure can read them
/// without `MazeWorld` plumbed through (would push cells_brain_act over
/// Bevy's 16-param system limit). Runs before `cells_brain_act`.
pub(crate) fn update_whisker_distances(
    maze: Res<super::super::resources::MazeWorld>,
    mut cells: Query<&mut CellEntity, Without<Dying>>,
) {
    let Some(field) = maze.field.as_ref() else {
        return;
    };
    cells.par_iter_mut().for_each(|mut cell| {
        cell.0.last_whisker_distances =
            field.whisker_distances(cell.0.position, cell.0.heading, cell.0.pitch);
    });
}

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

/// Pre-brain pass: aggregate bonded peers' `last_outputs` message channels
/// into `cell.bonded_inbox`. Mirrors `pool_bonded_hidden_cells` flow. Runs
/// before `cells_brain_act` so populate_brain_inputs reads fresh inbox.
pub(crate) fn pool_bond_messages_cells(
    mut cells: Query<&mut CellEntity, Without<Dying>>,
    mut id_to_outputs_scratch: Local<FxHashMap<u64, [f32; bioscape::BRAIN_OUTPUTS]>>,
) {
    id_to_outputs_scratch.clear();
    for c in cells.iter() {
        id_to_outputs_scratch.insert(c.0.cell_id, c.0.last_outputs);
    }
    if id_to_outputs_scratch.is_empty() {
        return;
    }
    let id_to_outputs = &*id_to_outputs_scratch;
    for mut cell in &mut cells {
        let inbox = bioscape::pool_bond_messages(&cell.0, |partner_id| {
            if partner_id == cell.0.cell_id {
                return None;
            }
            id_to_outputs.get(&partner_id).copied()
        });
        cell.0.bonded_inbox = inbox;
    }
}

pub(crate) fn apply_cell_morph(time: Res<Time>, mut cells: Query<&mut CellEntity, Without<Dying>>) {
    let dt = time.delta_secs();
    cells.par_iter_mut().for_each(|mut cell| {
        cell.0.apply_morph(dt);
    });
}
