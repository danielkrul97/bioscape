//! GPU compute scaffolding for batch brain forward + the wider per-tick
//! pipeline (sensor gather, motor, brownian, predate, step, hebbian, …).
//!
//! Gated behind the `gpu` feature flag so machines without the wgpu stack
//! get a slim build.
//!
//! Each subsystem here owns a wgpu pipeline, persistent storage buffers
//! sized to `capacity` cells, and a per-call `compute` / `dispatch_*`
//! entrypoint. The full-GPU path (`CellsGpu`) keeps state resident on the
//! device across ticks; the legacy upload/readback path (`forward_batch`,
//! etc.) is kept for tests and partial-GPU configurations.

mod brain;
mod brownian;
mod cells;
mod collision;
mod context;
mod cppn;
mod eat_food;
mod field;
mod food_spawn;
mod hebbian;
mod motor;
mod populate_inputs;
mod predate;
mod profiler;
mod scratch;
mod sensor_gather;
mod spatial_hash;
mod excitability;
mod izhikevich;
mod stats;
mod stdp;
mod step;
mod synaptic_scale;

#[cfg(test)]
mod tests;

pub use brain::*;
pub use brownian::*;
pub use cells::*;
pub use collision::*;
pub use context::*;
pub use cppn::*;
pub use eat_food::*;
pub use field::*;
pub use food_spawn::*;
pub use hebbian::*;
pub use motor::*;
pub use populate_inputs::*;
pub use predate::*;
pub use profiler::*;
pub use scratch::*;
pub use sensor_gather::*;
pub use spatial_hash::*;
pub use excitability::*;
pub use izhikevich::*;
pub use stats::*;
pub use stdp::*;
pub use step::*;
pub use synaptic_scale::*;
