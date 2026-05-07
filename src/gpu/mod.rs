//! Sprint 44: GPU compute scaffolding pro batch brain forward pass.
//!
//! Ohraničeno feature gate `gpu` — bez `--features gpu` modul vůbec
//! nezkompiluje wgpu, takže build na strojích bez GPU stacku zůstává štíhlý.
//!
//! Architektonicky drží `BrainGpu` perzistentní wgpu device + storage buffery
//! sized na `capacity` cells. Per `forward_batch`: upload inputs + per-cell
//! weights, dispatch compute, readback hidden + outputs. State on GPU mezi
//! ticky **NE** drží — to je Sprint 47.

mod brain;
mod brownian;
mod cells;
mod collision;
mod context;
mod field;
mod hebbian;
mod motor;
mod neighbors;
mod populate_inputs;
mod predate;
mod sensor_gather;
mod spatial_hash;
mod stats;
mod step;

#[cfg(test)]
mod tests;

pub use brain::*;
pub use brownian::*;
pub use cells::*;
pub use collision::*;
pub use context::*;
pub use field::*;
pub use hebbian::*;
pub use motor::*;
pub use neighbors::*;
pub use populate_inputs::*;
pub use predate::*;
pub use sensor_gather::*;
pub use spatial_hash::*;
pub use stats::*;
pub use step::*;
