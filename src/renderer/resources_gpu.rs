use bevy::prelude::*;
use bioscape::gpu::{
    BrainGpu, BrownianGpu, CellsGpu, CollisionGpu, CppnGpu, EatFoodGpu, FieldGpu, FoodSpawnGpu,
    HebbianGpu, MotorGpu, PopulateInputsGpu, PredateGpu, SensorGatherGpu, SpatialHashGpu, StepGpu,
};

/// Full GPU pipeline (mirror headless GPU mandatory path). Single-Wait
/// readback covers brain + Hebbian + Brownian + Field + sensor + populate +
/// motor + step + collision + predate + food spawn per tick. Init failure
/// in `setup` panics — no CPU compute fallback.
#[derive(Resource)]
pub(super) struct GpuFullPipeline {
    pub(super) cells: CellsGpu,
    pub(super) brain: BrainGpu,
    pub(super) hebbian: HebbianGpu,
    pub(super) brownian: BrownianGpu,
    pub(super) smell: FieldGpu,
    pub(super) pheromone: FieldGpu,
    /// Wave L: per-channel pheromone fields (ch1, ch2). ch0 = `pheromone`
    /// above. sensor_gather binding 16/17 reads from these for brain
    /// inputs 21..26.
    pub(super) pheromone_ch1: FieldGpu,
    pub(super) pheromone_ch2: FieldGpu,
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
    /// Wave J port: GPU rejection-sampling food spawn. K-attempts per
    /// dispatch with world-map richness + obstacle mask + cell exclusion
    /// against `cell_hash`. CPU keeps the variable-allocation control
    /// plane (spawn entities for valid candidates).
    pub(super) food_spawn: FoodSpawnGpu,
    /// GPU per-cell eat_food candidate selection. Mirrors headless port;
    /// CPU resolves the first-cell-wins race against the returned
    /// `(food_idx, value)` arrays. Cuts the renderer's `cell_eats_food`
    /// CPU `food_grid.for_each_in_radius_toroidal` hotspot.
    pub(super) eat_food: EatFoodGpu,
    /// GPU CPPN — materialises child brain weights direct → cells.brain_weights_buf
    /// at child slots, replacing per-child `upload_brain_at`. Dispatched once
    /// per `cell_reproduces_on_threshold` invocation.
    pub(super) cppn: CppnGpu,
    pub(super) scratch: bioscape::gpu::GpuFullScratch,
}
