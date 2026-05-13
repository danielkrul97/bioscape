//! Per-tick recorder for the activation-history mini-plot. Push the
//! current selected cell's `last_hidden` and `last_outputs` into a
//! rolling buffer; drop the oldest sample once the window is full.
//!
//! Recording is silent when no cell is selected, resets on identity
//! change, and stops once the cell is flagged deceased so the plot
//! freezes alongside the snapshot.

use bevy::prelude::*;

use super::super::resources::SimWorld;
use super::{ActivationHistory, SelectedCell, HISTORY_LEN};

pub(super) fn record_history(
    selected: Res<SelectedCell>,
    sim_world: Res<SimWorld>,
    mut history: ResMut<ActivationHistory>,
) {
    let target = selected.cell_id;
    if history.cell_id != target {
        history.reset_for(target);
    }
    let Some(id) = target else {
        return;
    };
    if selected.deceased {
        return;
    }
    let Some(idx) = sim_world.0.find_cell_idx_by_id(id) else {
        return;
    };
    let cell = &sim_world.0.cells[idx];
    if history.hidden.len() >= HISTORY_LEN {
        history.hidden.pop_front();
        history.outputs.pop_front();
    }
    history.hidden.push_back(cell.last_hidden);
    history.outputs.push_back(cell.last_outputs);
}
