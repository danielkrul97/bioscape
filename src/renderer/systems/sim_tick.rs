//! Sprint 176: shared-driver tick system. Runs `world.tick(&mut rng)`
//! once per Bevy FixedUpdate. Sprint 177 adds `sync_simworld_to_cellentity`
//! to overwrite legacy `CellEntity` state with SimWorld cells, so the
//! visual pipeline (sync_transforms, gizmos, materials) now reflects
//! SimWorld dynamics. Legacy tick systems continue to run in parallel
//! but their writes are immediately clobbered.

use bevy::prelude::*;

use super::super::components::CellEntity;
use super::super::resources::{CellSlotMap, SimRng, SimWorld};

pub(crate) fn sim_tick(
    mut sim_world: ResMut<SimWorld>,
    mut sim_rng: ResMut<SimRng>,
) {
    let rng = &mut sim_rng.0;
    sim_world.0.tick(rng);
    // Sprint 183+: keep the CPU vibration shadow current so the stats
    // overlay can sample it without per-frame GPU readback. `tick` is
    // 60 Hz so this download adds ~few ms/tick — acceptable for renderer
    // diagnostics. Headless skips this because it samples vibration only
    // at gen-end (`sync_vibration_from_gpu` directly).
    sim_world.0.sync_vibration_from_gpu();
}

/// Sprint 177: copy each `world.cells[slot]` into the `CellEntity` of
/// the entity registered at the same slot in `CellSlotMap`. Index-based
/// pairing — works while the legacy lifecycle (S178) drives slot
/// allocation/release matching SimWorld's swap_remove order. Misalignment
/// after long runs causes cosmetic glitches but never panics: out-of-range
/// slots are silently skipped.
pub(crate) fn sync_simworld_to_cellentity(
    sim_world: Res<SimWorld>,
    slot_map: Res<CellSlotMap>,
    mut cells: Query<&mut CellEntity>,
) {
    let world_cells = &sim_world.0.cells;
    let n = slot_map.slot_to_entity.len().min(world_cells.len());
    for slot in 0..n {
        let entity = slot_map.slot_to_entity[slot];
        if let Ok(mut cell_entity) = cells.get_mut(entity) {
            cell_entity.0 = world_cells[slot];
        }
    }
}
