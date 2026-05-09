use bevy::prelude::*;
use bioscape::{Cell, Food};

#[derive(Component)]
pub(super) struct CellEntity(pub(super) Cell);

#[derive(Component)]
pub(super) struct FoodEntity(pub(super) Food);

#[derive(Component)]
pub(super) struct Dying {
    pub(super) ticks_left: u32,
}

#[derive(Component)]
pub(super) struct StatsRoot;

#[derive(Component)]
pub(super) struct StatsText;

#[derive(Component)]
pub(super) struct WorldMapOverlay;

#[derive(Message, Debug, Clone, Copy)]
pub(super) struct GenerationEnded {
    pub(super) generation: u64,
}

#[derive(Message, Debug, Clone, Copy)]
pub(super) struct EpochEnded {
    pub(super) epoch: u64,
}

/// Sprint 69: snapshot row pro bond gizmo rendering.
#[derive(Clone, Copy)]
pub(super) struct BondSnapshot {
    pub(super) cell_id: u64,
    pub(super) start: Vec3,
    pub(super) color: Color,
    pub(super) partners: [Option<u64>; bioscape::MAX_BONDS_PER_CELL],
}
