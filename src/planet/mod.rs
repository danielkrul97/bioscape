//! Torus planet stability experiment — self-gravitating SPH fluid in
//! initial torus configuration. Independent of the biology simulation:
//! shares only the GPU context fabric (`crate::gpu::GpuContext`).
//!
//! Entry points:
//! - `bioscape::planet::PlanetWorld` — owns particle state, GPU pipelines, tick driver.
//! - `bioscape::planet::init::torus_uniform` — initial state generator.
//!
//! Binaries: `planet_view` (Bevy 3D viewer), `planet_headless` (batch sweep).
//!
//! Sprint plan: see `docs/sprints/203-212-torus-planet-sph.md`.

pub mod particle;
pub mod world;
pub mod init;
pub mod integrator;
pub mod gravity_cpu;
pub mod diagnostics;
pub mod gpu;

pub use particle::Particles;
pub use world::{primary_radius, shape_max_extent, PlanetConfig, PlanetShape, PlanetWorld};
