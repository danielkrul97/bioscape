//! Sprint 178+ post-refactor renderer systems. Simulation logic lives
//! in `bioscape::sim::World::tick`; renderer keeps only the visual
//! pipeline (transform sync, spike entity sync) and the shared-driver
//! adapter (`sim_tick`). Pre-S178 files (`brains.rs`, `brains_gpu.rs`,
//! `collisions.rs`, `fields.rs`, `food.rs`, `lifecycle.rs`, `physics.rs`)
//! were deleted in cleanup — git history retains them under the
//! `gpu-only` branch pre-`refactor(renderer): unschedule legacy tick
//! systems (Sprint 178)`.

pub(super) mod sim_tick;
pub(super) mod spikes;
pub(super) mod transforms;

pub(super) use sim_tick::*;
pub(super) use spikes::*;
pub(super) use transforms::*;
