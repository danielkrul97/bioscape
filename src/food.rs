use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::*;

/// Tagged food kind. `Plant` is the ambient spawn (always available,
/// baseline value); `Carrion` drops on cell death; `HunterCarrion` drops on
/// hunter death (richest reward). Per-kind digestion efficiency is gated by
/// `genome.carnivore_score` — see `eat_efficiency`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum FoodKind {
    Plant = 0,
    Carrion = 1,
    HunterCarrion = 2,
}

impl Default for FoodKind {
    fn default() -> Self {
        FoodKind::Plant
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Food {
    pub position: [f32; 3],
    /// Ticks since spawn. Drives `value_factor` decay; starts at 0 for both
    /// fresh ambient food and carrion (no carrion-specific staleness offset).
    pub age_ticks: u32,
    #[serde(default)]
    pub kind: FoodKind,
}

/// Base food value per kind. Carrion is denser than plant (concentrated
/// biomass) and hunter carrion is densest (apex predator drop).
pub const PLANT_FOOD_VALUE: f32 = 20.0;
pub const CARRION_FOOD_VALUE: f32 = 30.0;
pub const HUNTER_CARRION_FOOD_VALUE: f32 = 50.0;

// Cooperative food node — a high-value spawn that yields nothing until N
// cells arrive within a time window. Creates fitness coupling for
// recruitment signaling: solo cells get nothing, a coordinated trio gets
// the full reward, so selection rewards "I see food, signal peers."

pub const COOP_FOOD_REQUIRED_ARRIVALS: usize = 3;
/// Time window (ticks) since spawn. After expiry, the node despawns with
/// no reward distributed.
pub const COOP_FOOD_TIME_WINDOW_TICKS: u32 = 120;
/// Per-participant reward on a successful trigger. 4× the plant baseline —
/// large enough to justify the loitering cost of waiting for peers.
pub const COOP_FOOD_REWARD_PER_CELL: f32 = 80.0;
/// Acceptance radius for "arrived". Larger than the regular eat radius
/// because the coop node represents a gathering point, not a literal
/// food pickup — cells signal nearby, they don't need to overlap.
pub const COOP_FOOD_ARRIVAL_RADIUS: f32 = 30.0;
/// Per-tick Poisson-like spawn rate, calibrated for ~10–15 coop nodes per
/// 600-tick generation.
pub const COOP_FOOD_SPAWN_RATE_PER_TICK: f32 = 0.02;
/// Cap on simultaneously live coop nodes. Bounds memory and per-tick
/// arrival-scan cost.
pub const COOP_FOOD_MAX_CONCURRENT: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoopFood {
    pub position: [f32; 3],
    pub spawn_tick: u64,
    /// Set of unique `cell_id`s that have been inside `ARRIVAL_RADIUS` for
    /// at least one tick. Stored as `Vec<u64>` to preserve insertion order
    /// for serde — lookup is O(N), acceptable because N stays small
    /// (typically < 10).
    pub arrivals: Vec<u64>,
    /// True once the arrivals threshold was reached and the reward was
    /// distributed; the node despawns at the end of that tick.
    pub triggered: bool,
}

impl CoopFood {
    pub fn new(position: [f32; 3], spawn_tick: u64) -> Self {
        Self {
            position,
            spawn_tick,
            arrivals: Vec::new(),
            triggered: false,
        }
    }

    /// True once `age >= TIME_WINDOW`. Callers test this *after* attempting
    /// trigger so a node that triggers on its expiry tick isn't mis-classified
    /// as "expired without reward".
    #[inline]
    pub fn is_expired(&self, current_tick: u64) -> bool {
        current_tick.saturating_sub(self.spawn_tick) >= COOP_FOOD_TIME_WINDOW_TICKS as u64
    }
}

/// Register `cell_id` as having arrived. Insertion order is preserved;
/// duplicates are ignored (a cell can be inside the radius for multiple
/// ticks). Returns `true` only on the first registration.
pub fn register_coop_arrival(coop: &mut CoopFood, cell_id: u64) -> bool {
    if coop.arrivals.iter().any(|id| *id == cell_id) {
        return false;
    }
    coop.arrivals.push(cell_id);
    true
}

/// Attempt to trigger the threshold and distribute rewards. Returns `true`
/// only when the arrivals threshold was met and the node had not yet
/// triggered. Callers propagate the return value into per-gen counters
/// (`coop_food_solved`).
pub fn try_trigger_coop(coop: &mut CoopFood, cells: &mut [Cell]) -> bool {
    if coop.triggered || coop.arrivals.len() < COOP_FOOD_REQUIRED_ARRIVALS {
        return false;
    }
    // Build a `cell_id → idx` map once, then index in O(1). Avoids the previous
    // O(arrivals × cells) `iter_mut().find` per arrival, which became visible
    // at large populations.
    let mut id_to_idx: rustc_hash::FxHashMap<u64, usize> =
        rustc_hash::FxHashMap::with_capacity_and_hasher(cells.len(), Default::default());
    for (i, c) in cells.iter().enumerate() {
        id_to_idx.insert(c.cell_id, i);
    }
    for cell_id in &coop.arrivals {
        if let Some(&idx) = id_to_idx.get(cell_id) {
            cells[idx].energy += COOP_FOOD_REWARD_PER_CELL;
        }
    }
    coop.triggered = true;
    true
}

/// Random position inside the world bounds (toroidal XY, optionally
/// non-zero Z). Mirrors `Food::random`.
pub fn random_coop_position(rng: &mut impl Rng, world_half: [f32; 3]) -> [f32; 3] {
    let z = if world_half[2] > 0.0 {
        rng.random_range(-world_half[2]..world_half[2])
    } else {
        0.0
    };
    [
        rng.random_range(-world_half[0]..world_half[0]),
        rng.random_range(-world_half[1]..world_half[1]),
        z,
    ]
}

/// Per-tick arrival scan: every cell within `ARRIVAL_RADIUS` (toroidal-aware)
/// of each non-triggered coop node is registered as an arrival.
pub fn register_coop_arrivals_for_all(coops: &mut [CoopFood], cells: &[Cell], world_half: [f32; 3]) {
    let r2 = COOP_FOOD_ARRIVAL_RADIUS * COOP_FOOD_ARRIVAL_RADIUS;
    for coop in coops.iter_mut() {
        if coop.triggered {
            continue;
        }
        for cell in cells.iter() {
            let d = min_image_delta(coop.position, cell.position, world_half);
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if d2 <= r2 {
                let _ = register_coop_arrival(coop, cell.cell_id);
            }
        }
    }
}

#[inline]
pub fn food_base_value(kind: FoodKind) -> f32 {
    match kind {
        FoodKind::Plant => PLANT_FOOD_VALUE,
        FoodKind::Carrion => CARRION_FOOD_VALUE,
        FoodKind::HunterCarrion => HUNTER_CARRION_FOOD_VALUE,
    }
}

/// Digestion efficiency for `(kind, carnivore_score)`. Continuous trade-off:
/// `score = 0` is a pure herbivore (Plant only), `score = 1` is a pure
/// carnivore (HunterCarrion only). `Carrion` (cell remains) sits at 0.5 for
/// every score so it doesn't drive specialization in either direction.
#[inline]
pub fn eat_efficiency(kind: FoodKind, carnivore_score: f32) -> f32 {
    let s = carnivore_score.clamp(0.0, 1.0);
    match kind {
        FoodKind::Plant => 1.0 - s,
        FoodKind::Carrion => 0.5,
        FoodKind::HunterCarrion => s,
    }
}

impl Food {
    pub fn random(rng: &mut impl Rng, world_half: [f32; 3]) -> Self {
        // Skip the z-axis RNG draw when the world is 2D so the RNG stream
        // matches the pre-3D baseline byte-for-byte.
        let z = if world_half[2] > 0.0 {
            rng.random_range(-world_half[2]..world_half[2])
        } else {
            0.0
        };
        Self {
            position: [
                rng.random_range(-world_half[0]..world_half[0]),
                rng.random_range(-world_half[1]..world_half[1]),
                z,
            ],
            age_ticks: 0,
            kind: FoodKind::Plant,
        }
    }

    /// Sink the food downward at `FOOD_SINK_RATE`, clamped to the floor.
    /// No-op when the z-volume is inactive.
    pub fn apply_gravity(&mut self, dt: f32, world_half_z: f32) {
        if world_half_z <= 0.0 {
            return;
        }
        self.position[2] = (self.position[2] - FOOD_SINK_RATE * dt).max(-world_half_z);
    }

    /// Linear value decay by age, clamped at 0.
    pub fn value_factor(&self) -> f32 {
        let age_sec = self.age_ticks as f32 / FIXED_TIMESTEP_HZ;
        (1.0 - CARRION_DECAY_PER_SEC * age_sec).max(0.0)
    }

    /// Increment age by one tick. Returns `false` once the food has
    /// expired (value_factor ≤ 0) so the caller can despawn it.
    pub fn age_step(&mut self) -> bool {
        self.age_ticks = self.age_ticks.saturating_add(1);
        self.value_factor() > 0.0
    }
}
