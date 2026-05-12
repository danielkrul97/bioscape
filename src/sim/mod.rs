//! Shared simulation driver — tick orchestration, World state, GPU
//! pipeline init. Pre-S173 lived in `src/bin/headless/world.rs`; moved
//! to a library module so renderer (`src/main.rs`) can run the same
//! simulation logic instead of its own parallel ECS implementation.
//!
//! Headless + renderer both import `bioscape::sim::World` and call
//! `world.tick(rng)` per simulation step. Binary-specific concerns
//! (CSV logging, checkpoint serialization, ECS rendering, UI) stay
//! in the respective binary crates.

pub mod world;

pub use world::*;
