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

/// Decorative spike rendered as a top-level entity tracking its owner cell.
/// Avoids ChildOf scale composition with the parent ellipsoid. `slot` selects
/// which Phenotype.spikes[k] drives the transform.
#[derive(Component)]
pub(super) struct SpikeEntity {
    pub(super) owner: Entity,
    pub(super) slot: u8,
}

#[derive(Component)]
pub(super) struct StatsRoot;

#[derive(Component)]
pub(super) struct StatsText;

#[derive(Component)]
pub(super) struct WorldMapOverlay;

/// Marker for one wall-voxel mesh in the maze. The toggle handler scans for
/// every entity carrying this and despawns it when the maze is turned off,
/// then respawns the set when re-enabled.
#[derive(Component)]
pub(super) struct MazeWallEntity;

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
