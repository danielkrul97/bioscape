//! Sprint 183 (post-S182): JSON config file for renderer runtime
//! overrides. Headless has a clap CLI surface (`--seed`, `--maze`,
//! `--initial-izhikevich-frac`, …); renderer instead reads a single
//! `bioscape.json` from CWD at startup. Missing or unparseable file
//! falls back to library defaults — exactly what the renderer did
//! pre-S183.
//!
//! All fields are `Option<...>` (or have explicit numeric defaults) so
//! a partial config only overrides what it names. Example:
//!
//! ```json
//! {
//!   "seed": 100,
//!   "initial_izhikevich_frac": 0.5,
//!   "maze": "medium"
//! }
//! ```
//!
//! Anything not listed inherits from `bioscape::*` const defaults.

use bevy::log;
use bevy::prelude::Resource;
use bioscape::{MazeDifficulty, INITIAL_CELLS, MATING_RADIUS, MAX_POPULATION, WORLD_MAP_SEED};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Default filename loaded from the renderer's working directory.
pub(super) const CONFIG_FILENAME: &str = "bioscape.json";

#[derive(Debug, Clone, Resource, Serialize, Deserialize)]
pub(super) struct SimConfig {
    /// Seed for the renderer's `SimRng` (drives reproduce mutations etc).
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// `WORLD_MAP_SEED` override. `None` → lib default.
    #[serde(default)]
    pub map_seed: Option<u64>,
    #[serde(default)]
    pub mating_radius: Option<f32>,
    #[serde(default)]
    pub initial_cells: Option<usize>,
    #[serde(default)]
    pub max_population: Option<usize>,
    /// Fraction of initial population pre-seeded as
    /// `NeuronModel::Izhikevich` (Sprint 159). 0.0 = pure Perceptron
    /// start; 0.5 = balanced; 1.0 = pure Izhikevich.
    #[serde(default)]
    pub initial_izhikevich_frac: f32,
    /// `"easy"`, `"medium"`, `"hard"`, or `None`/omitted for no maze.
    #[serde(default)]
    pub maze: Option<String>,
}

fn default_seed() -> u64 {
    WORLD_MAP_SEED
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            seed: WORLD_MAP_SEED,
            map_seed: None,
            mating_radius: None,
            initial_cells: None,
            max_population: None,
            initial_izhikevich_frac: 0.0,
            maze: None,
        }
    }
}

impl SimConfig {
    /// Try to read `path` as JSON; on any error log a warning and fall
    /// back to defaults. Missing file is silent (treated as "user opted
    /// for defaults").
    pub(super) fn load_or_default(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(s) => match serde_json::from_str::<SimConfig>(&s) {
                Ok(cfg) => {
                    log::info!(
                        "sim config loaded from {} (seed={}, izh_frac={:.2}, maze={:?})",
                        path.display(),
                        cfg.seed,
                        cfg.initial_izhikevich_frac,
                        cfg.maze
                    );
                    cfg
                }
                Err(e) => {
                    log::warn!(
                        "sim config parse failed at {} ({}); using defaults",
                        path.display(),
                        e
                    );
                    Self::default()
                }
            },
            Err(_) => {
                log::info!(
                    "no sim config at {} — using library defaults",
                    path.display()
                );
                Self::default()
            }
        }
    }

    pub(super) fn resolved_map_seed(&self) -> u64 {
        self.map_seed.unwrap_or(WORLD_MAP_SEED)
    }

    pub(super) fn resolved_mating_radius(&self) -> f32 {
        self.mating_radius.unwrap_or(MATING_RADIUS)
    }

    pub(super) fn resolved_initial_cells(&self) -> usize {
        self.initial_cells.unwrap_or(INITIAL_CELLS)
    }

    pub(super) fn resolved_max_population(&self) -> usize {
        self.max_population.unwrap_or(MAX_POPULATION)
    }

    pub(super) fn resolved_maze(&self) -> Option<MazeDifficulty> {
        self.maze.as_deref().and_then(MazeDifficulty::parse)
    }
}
