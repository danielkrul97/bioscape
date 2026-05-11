use bevy::prelude::*;
use bioscape::gpu::{
    BrainGpu, BrownianGpu, CellsGpu, CollisionGpu, CppnGpu, FieldGpu, HebbianGpu, MotorGpu,
    PopulateInputsGpu, PredateGpu, SensorGatherGpu, SpatialHashGpu, StepGpu,
};

/// Sprint 52: GPU compute state pro renderer. Drží persistent CellsGpu +
/// BrainGpu/HebbianGpu/BrownianGpu na shared GpuContext. Insert se v `setup`
/// pokud GpuContext::new uspěje; pokud selže, Resource zůstává `None` a
/// systems gracefully fallbacknou na CPU.
#[derive(Resource)]
pub(super) struct GpuBrainState {
    pub(super) cells: CellsGpu,
    pub(super) brain: BrainGpu,
    pub(super) hebbian: HebbianGpu,
    /// Sprint 129: brownian dispatch dropped, GPU resource kept resident
    /// to avoid setup churn. Field unused — CPU path is now canonical.
    #[allow(dead_code)]
    pub(super) brownian: BrownianGpu,
}

/// Sprint 59: separate Resource pro GPU smell + pheromone fields. Oddělen od
/// `GpuBrainState` aby `update_smell_field` (ResMut<GpuFieldState>) nesoutěžil
/// s ostatními systemy o brain access. Insert v setup pokud GpuContext init
/// uspěje; jinak CPU SmellResource path drží.
#[derive(Resource)]
pub(super) struct GpuFieldState {
    pub(super) smell: FieldGpu,
    pub(super) pheromone: FieldGpu,
}

/// Full GPU pipeline (mirror headless `--gpu-full`). Při insert nahradí
/// `cells_brain_act` / `apply_brownian_motion` / `step_cells` GPU pipeline
/// se single-Wait readback. **Default on**; opt-out přes `BIOSCAPE_GPU_FULL=0`.
/// Legacy `BIOSCAPE_GPU_BRAIN=1` má prioritu (mutually exclusive).
///
/// Drží vlastní `cells: CellsGpu` (sdíleno přes `GpuContext` clone s field
/// state); `cell_hash`/`food_hash` `SpatialHashGpu` pro sensor broad-phase;
/// `sensor` + `populate` + `motor` + `step` + `brownian` GPU stages. Vše na
/// jednom `GpuContext`, single readback per tick přes `download_full_batch_into`.
#[derive(Resource)]
pub(super) struct GpuFullPipeline {
    pub(super) cells: CellsGpu,
    pub(super) brain: BrainGpu,
    pub(super) hebbian: HebbianGpu,
    pub(super) brownian: BrownianGpu,
    pub(super) smell: FieldGpu,
    pub(super) pheromone: FieldGpu,
    /// V7: motion-driven mechanosensory field. Deposit per cell mirrors
    /// `emit_pheromones` pattern but the emission formula is hard-coded
    /// (no brain output channel). Brain reads grad + amp via the chained
    /// `sensor_gather → populate_inputs` pipeline.
    pub(super) vibration: FieldGpu,
    pub(super) cell_hash: SpatialHashGpu,
    pub(super) food_hash: SpatialHashGpu,
    pub(super) sensor: SensorGatherGpu,
    pub(super) populate: PopulateInputsGpu,
    pub(super) motor: MotorGpu,
    pub(super) step: StepGpu,
    /// Wave H: GPU collision broad-phase. Mirrors headless GpuFullState.
    pub(super) collision: CollisionGpu,
    /// Wave H: GPU predate (herd + atomic attack accumulation).
    pub(super) predate: PredateGpu,
    /// GPU CPPN — materialises child brain weights direct → cells.brain_weights_buf
    /// at child slots, replacing per-child `upload_brain_at`. Dispatched once
    /// per `cell_reproduces_on_threshold` invocation.
    pub(super) cppn: CppnGpu,
    pub(super) scratch: bioscape::gpu::GpuFullScratch,
}
