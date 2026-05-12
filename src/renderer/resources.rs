use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bioscape::{
    EventCalendar, ObstacleField, SimClock, SmellField, WorldMap,
    INITIAL_CELLS, N_PHEROMONE_CHANNELS,
};
use rustc_hash::FxHashMap;
use std::path::PathBuf;
use std::time::Instant;

use super::config::{CAMERA_PITCH_INITIAL, CAMERA_SCALE_INITIAL};
use super::material::BioMaterial;

#[derive(Resource, Default)]
pub(super) struct TickCounter {
    pub(super) ticks_this_frame: u32,
    pub(super) sim_ms_this_frame: f64,
    pub(super) tick_start: Option<Instant>,
}

/// Sprint 174: newtype wrapping shared `bioscape::sim::World`. Initialised
/// at startup alongside renderer's existing `GpuFullPipeline`. Sprint 176
/// adds the per-frame `sim_tick` system + `sync_simworld_to_cellentity`
/// position copy; legacy renderer tick systems continue running in
/// parallel until S177 removes them.
#[derive(Resource)]
pub(super) struct SimWorld(pub(super) bioscape::sim::World);

/// Sprint 176: deterministic RNG for the renderer's `sim_tick`. Mirrors
/// headless `StdRng::seed_from_u64(seed)` so the same seed reproduces
/// the same trajectory across both binaries. Seeded with `WORLD_MAP_SEED`
/// at startup — overridable via CLI in a future sprint.
#[derive(Resource)]
pub(super) struct SimRng(pub(super) rand::rngs::StdRng);

#[derive(Resource, Debug, Clone, Copy)]
pub(super) struct WorldExtent {
    pub(super) half_x: f32,
    pub(super) half_y: f32,
    pub(super) half_z: f32,
}

impl WorldExtent {
    pub(super) fn as_array(self) -> [f32; 3] {
        [self.half_x, self.half_y, self.half_z]
    }
}

#[derive(Resource, Debug)]
pub(super) struct Clock(pub(super) SimClock);

#[derive(Resource, Debug, Clone, Copy)]
pub(super) struct FoodDensityFactor(pub(super) f32);

impl Default for FoodDensityFactor {
    fn default() -> Self {
        Self(1.0)
    }
}

/// Persistent cell_id ↔ Entity ↔ position lookup. Build raz/tick v
/// `rebuild_cell_entity_lookups` před brain_act fází; consume v
/// resolve_cell_collisions, cell_eats_food, draw_bond_gizmos, pool_bonded_*.
/// Pre-fix: 6+ systémů buildovalo vlastní `FxHashMap` per tick.
#[derive(Resource, Default)]
pub(super) struct CellEntityLookups {
    pub(super) id_to_entity: FxHashMap<u64, Entity>,
    pub(super) id_to_position: FxHashMap<u64, [f32; 3]>,
    pub(super) entity_to_idx: FxHashMap<Entity, usize>,
    /// Vec indexed by `entity_to_idx[Entity]` — drží position pro O(1) lookup
    /// v collision phase 2.
    pub(super) positions_by_idx: Vec<[f32; 3]>,
}

impl CellEntityLookups {
    #[allow(dead_code)]
    pub(super) fn rebuild<'a>(
        &mut self,
        iter: impl Iterator<Item = (Entity, u64, [f32; 3])>,
    ) {
        self.id_to_entity.clear();
        self.id_to_position.clear();
        self.entity_to_idx.clear();
        self.positions_by_idx.clear();
        for (entity, cell_id, pos) in iter {
            let idx = self.positions_by_idx.len();
            self.positions_by_idx.push(pos);
            self.id_to_entity.insert(cell_id, entity);
            self.id_to_position.insert(cell_id, pos);
            self.entity_to_idx.insert(entity, idx);
        }
    }
}

/// Sprint 66: monotonic counter pro Cell.cell_id přidělování. Initial pop
/// uses ids 0..INITIAL_CELLS, takže start = INITIAL_CELLS. Children z
/// reproduce čerpají odsud.
#[derive(Resource)]
pub(super) struct NextCellId(pub(super) u64);

impl Default for NextCellId {
    fn default() -> Self {
        Self(INITIAL_CELLS as u64)
    }
}

/// Sprint 66: per-pair contact tick tracker. Klíč je `(min_id, max_id)`
/// stable Cell.cell_id páru. Resource žije celý běh — generation reset
/// nemažeme (kontakt může běžet napříč generační hranicí).
#[derive(Resource, Default)]
pub(super) struct ContactProgress(pub(super) FxHashMap<(u64, u64), u32>);

/// Sprint 109: deterministicky vygenerovaný kalendář environmentálních shocků
/// pro celý běh rendereru. Default empty (no-op). Sprint 110+ integruje efekty
/// per shock kind. Init z env varu `BIOSCAPE_SHOCKS_MEAN_GENS` (parse u32);
/// ignoruje ho při unset / `0`.
#[derive(Resource, Default)]
pub(super) struct EventCalendarResource(pub(super) EventCalendar);

/// Sprint 52: maps Bevy `Entity` ↔ slot index v `CellsGpu` SoA bufferech.
/// Sloty jsou dense (0..n, žádné holes) přes swap_remove pattern při death.
#[derive(Resource, Default)]
pub(super) struct CellSlotMap {
    pub(super) slot_to_entity: Vec<Entity>,
    pub(super) entity_to_slot: FxHashMap<Entity, usize>,
}

#[allow(dead_code)]
impl CellSlotMap {
    pub(super) fn allocate(&mut self, entity: Entity) -> usize {
        let slot = self.slot_to_entity.len();
        self.slot_to_entity.push(entity);
        self.entity_to_slot.insert(entity, slot);
        slot
    }

    /// Release slot pro entity. Vrací `Some((freed_slot, moved_entity))`
    /// pokud entity byla zaregistrovaná. `moved_entity` je Some pokud
    /// freed_slot byl zaplněn cell ze zadního slotu (swap_remove pattern).
    pub(super) fn release(&mut self, entity: Entity) -> Option<(usize, Option<Entity>)> {
        let slot = self.entity_to_slot.remove(&entity)?;
        let last = self.slot_to_entity.len() - 1;
        let moved = if slot != last {
            let moved_entity = self.slot_to_entity[last];
            self.slot_to_entity[slot] = moved_entity;
            self.entity_to_slot.insert(moved_entity, slot);
            Some(moved_entity)
        } else {
            None
        };
        self.slot_to_entity.pop();
        Some((slot, moved))
    }

    pub(super) fn slot_of(&self, entity: Entity) -> Option<usize> {
        self.entity_to_slot.get(&entity).copied()
    }

    pub(super) fn len(&self) -> usize {
        self.slot_to_entity.len()
    }
}

#[derive(Resource)]
pub(super) struct CellMesh(pub(super) Handle<Mesh>);

#[derive(Resource)]
pub(super) struct FoodMesh(pub(super) Handle<Mesh>);

#[derive(Resource)]
pub(super) struct FoodMaterial(pub(super) Handle<StandardMaterial>);

#[derive(Resource)]
pub(super) struct SpikeMesh(pub(super) Handle<Mesh>);

#[derive(Resource)]
pub(super) struct SpikeMaterial(pub(super) Handle<StandardMaterial>);

/// Sprint 36: per-lineage material cache. Lineage hue → handle do
/// `Assets<StandardMaterial>`. Bevy automaticky deduplikuje stejné materialy
/// na renderer instances draw call.
///
/// Sprint 69: keyovaný podle `adhesion_type` (0..ADHESION_TYPE_COUNT) místo
/// `lineage_id`. 8 distinct hues = vidíš "tribes" na první pohled, jakmile
/// začne Steinberg sorting (same-type cells gravitují k sobě). Pre-Sprint 69
/// se barvilo podle lineage_hue (random hue per linii) — ten signal byl
/// užitečný pro fyzickou separaci, ale přebíjel adhesion clustering. Lineage
/// info zůstává v HUD + CSV `lineages` count.
#[derive(Resource, Default)]
pub(super) struct AdhesionMaterials(pub(super) [Option<Handle<BioMaterial>>; 8]);

/// Sprint 36 orbit camera state. Camera obíhá kolem `target` ve sférických
/// souřadnicích (yaw + pitch). Distance camera→target je fixní
/// `CAMERA_OFFSET_DISTANCE`; "zoom" modifikuje `scale` (orthographic projection
/// scale). Yaw = rotace kolem world Z, pitch = elevace nad xy plochou
/// (0 = horizon, π/2 = top-down).
#[derive(Resource, Debug, Clone, Copy)]
pub(super) struct OrbitCamera {
    pub(super) target: Vec3,
    pub(super) yaw: f32,
    pub(super) pitch: f32,
    /// Orthographic scale (world units per pixel). Menší = zoom in.
    pub(super) scale: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            yaw: 0.0,
            pitch: CAMERA_PITCH_INITIAL,
            scale: CAMERA_SCALE_INITIAL,
        }
    }
}

impl OrbitCamera {
    pub(super) fn transform(&self) -> Transform {
        let cos_p = self.pitch.cos();
        let offset = Vec3::new(
            -self.yaw.sin() * cos_p,
            -self.yaw.cos() * cos_p,
            self.pitch.sin(),
        ) * super::config::CAMERA_OFFSET_DISTANCE;
        let pos = self.target + offset;
        Transform::from_translation(pos).looking_at(self.target, Vec3::Z)
    }
}

#[derive(Resource)]
#[allow(dead_code)]
pub(super) struct SmellResource(pub(super) SmellField);

/// Sprint 126: multi-channel pheromone fields. Pole `[SmellField; N_PHEROMONE_CHANNELS]`
/// — každý kanál má vlastní decay/diffusion (viz `PHEROMONE_DECAY_PER_CH` / `_DIFFUSION_PER_CH`).
/// ch0 = slow (mating-friendly, backward-compat), ch1 medium, ch2 fast (bursty).
#[derive(Resource)]
pub(super) struct PheromoneResource {
    pub(super) fields: [SmellField; bioscape::N_PHEROMONE_CHANNELS],
}

/// Motion-driven mechanosensory field. Reuses `SmellField` (3D scalar with
/// diffusion + decay); deposit per cell happens in `update_vibration_field`.
/// CPU-only — no GPU shader counterpart.
#[derive(Resource)]
pub(super) struct VibrationResource(pub(super) SmellField);

/// Sprint 128: cooperative food packets. Vec uloženo přímo v Resource (žádná
/// per-node Entity — coop food má jen pozici a stav, žádné rendering aspekty
/// v této verzi).
#[derive(Resource, Default)]
pub(super) struct CoopFoodResource(pub(super) Vec<bioscape::CoopFood>);

#[derive(Resource)]
pub(super) struct WorldMapResource(pub(super) WorldMap);

/// Maze world toggle. When `field` is `Some`, the renderer routes wall
/// collision (`step_cells`), masked diffusion (smell/pheromone/vibration
/// fields) and vision LOS (`cells_brain_act`) through the maze-aware code
/// paths, and a set of `MazeWallEntity` boxes is rendered for the occupied
/// voxels. Pressing `KeyL` flips this — toggling fully allocates or
/// deallocates the obstacle field + per-grid masks. Diffusion masks are
/// precomputed once at allocation; the mask resolutions match
/// `SmellField` / pheromone / vibration grid sizes so per-tick lookup is a
/// flat indexing op.
#[derive(Resource, Default)]
pub(super) struct MazeWorld {
    pub(super) field: Option<ObstacleField>,
    pub(super) smell_mask: Option<Vec<bool>>,
    pub(super) pheromone_masks: [Option<Vec<bool>>; N_PHEROMONE_CHANNELS],
    pub(super) vibration_mask: Option<Vec<bool>>,
}

impl MazeWorld {
    pub(super) fn is_active(&self) -> bool {
        self.field.is_some()
    }
}

/// Wave 3: bundle `MazeWorld` + `CoopFoodResource` into one `SystemParam`
/// so `cells_brain_act` can carry both without crossing Bevy's 16-param
/// system cap. The two fields together replace the previous `coop_foods`
/// param; net result is 15 params with maze access in scope.
#[derive(SystemParam)]
#[allow(dead_code)]
pub(super) struct MazeAndCoop<'w> {
    pub(super) maze: Res<'w, MazeWorld>,
    pub(super) coop_foods: Res<'w, CoopFoodResource>,
}

#[derive(Resource, Clone)]
pub(super) struct ScreencastConfig {
    pub(super) dir: PathBuf,
    pub(super) interval_secs: f32,
    pub(super) duration_secs: f32,
    pub(super) started_at: Option<f32>,
    pub(super) last_capture: f32,
    pub(super) frame_idx: u32,
}
