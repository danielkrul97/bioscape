//! Sprint 176: shared-driver tick system. Runs `world.tick(&mut rng)`
//! once per Bevy FixedUpdate. For now legacy renderer ECS systems
//! continue running in parallel — both populations evolve independently,
//! visual still reads the legacy `CellEntity` components.
//!
//! Sprint 177 will add `sync_simworld_to_cellentity` to overwrite the
//! legacy state with `world.cells` positions, making the SimWorld pop
//! the canonical visible source. S178 removes the legacy systems.

use bevy::prelude::*;

use super::super::resources::{SimRng, SimWorld};

pub(crate) fn sim_tick(
    mut sim_world: ResMut<SimWorld>,
    mut sim_rng: ResMut<SimRng>,
) {
    let rng = &mut sim_rng.0;
    sim_world.0.tick(rng);
}
