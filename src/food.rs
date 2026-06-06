use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::*;

/// Tagged food kind. `Plant` is the ambient spawn (always available,
/// baseline value); `Carrion` drops on cell death. Per-kind digestion
/// efficiency is gated by `genome.carnivore_score` — see `eat_efficiency`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum FoodKind {
    Plant = 0,
    Carrion = 1,
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
/// biomass).
pub const PLANT_FOOD_VALUE: f32 = 20.0;
pub const CARRION_FOOD_VALUE: f32 = 30.0;

// Cooperative food node — a high-value spawn that yields nothing until a
// complementary-signal *handshake* happens between co-present cells. The node
// reads each present cell's two bond-message emissions (out[12] = signal A,
// out[13] = signal B); it unlocks only when at least one cell signals role A and
// a *distinct* cell signals role B at the same time, and rewards only the
// role-players (co-present non-signalers — free-riders — get nothing). This is
// the social-coevolution loop: a cell's payoff is contingent on another cell
// perceiving the situation and emitting the complementary signal, so the
// otherwise-dormant communication channels finally come under selection and a
// division-of-signalling-labour protocol can evolve.

/// Legacy anonymous-arrival threshold. Retained for the `arrivals_avg` metric
/// and old tests; the live trigger is the signal handshake, not this count.
pub const COOP_FOOD_REQUIRED_ARRIVALS: usize = 3;
/// Minimum rectified emission (`max(0, out[12 or 13])`) for a present cell to
/// count as a handshake role-player (provider) on that channel.
pub const HANDSHAKE_SIGNAL_THRESHOLD: f32 = 0.3;
/// Energy/sec a cell pays per unit of handshake signal it emits (A + B summed,
/// rectified), charged every tick regardless of location. This is what turns
/// signalling from constitutive into *conditional*: emitting both channels
/// everywhere is a standing tax, so selection favours cells that signal only
/// when it pays — i.e. when they perceive a coop node and a prospective partner.
/// That perceive-then-signal contingency is the social-intelligence hook (it
/// makes the recurrent brain earn its keep), and the cost also keeps signalling
/// honest so the handshake can't collapse into everyone-emits-everything. Small
/// vs the `COOP_FOOD_REWARD_PER_CELL` payoff so a real handshake stays worth it.
pub const HANDSHAKE_SIGNAL_COST: f32 = 0.25;
/// Range over which a cell perceives an unanswered handshake call at a coop node
/// (brain input slot 39). Several × `COOP_FOOD_ARRIVAL_RADIUS` so a cell can sense
/// the opportunity from a distance, navigate in (via the node's nearest-food
/// direction slot) and emit the missing complementary role — the call→response
/// loop the blind handshake lacked.
pub const COOP_CALL_PERCEPTION_RADIUS: f32 = 150.0;
/// Time window (ticks) since spawn. After expiry, the node despawns with
/// no reward distributed.
pub const COOP_FOOD_TIME_WINDOW_TICKS: u32 = 120;
/// Per-participant reward on a successful trigger. 4× the plant baseline —
/// large enough to justify the loitering cost of waiting for peers.
pub const COOP_FOOD_REWARD_PER_CELL: f32 = 160.0;
/// Acceptance radius for "arrived". Larger than the regular eat radius
/// because the coop node represents a gathering point, not a literal
/// food pickup — cells signal nearby, they don't need to overlap.
pub const COOP_FOOD_ARRIVAL_RADIUS: f32 = 30.0;
/// Per-tick Poisson-like spawn rate, calibrated for ~10–15 coop nodes per
/// 600-tick generation.
pub const COOP_FOOD_SPAWN_RATE_PER_TICK: f32 = 0.05;
/// Cap on simultaneously live coop nodes. Bounds memory and per-tick
/// arrival-scan cost.
pub const COOP_FOOD_MAX_CONCURRENT: usize = 8;

/// Sprint 209: single-cell multi-step "ripening" food. Unlike the grab-and-go
/// plant food, it yields nothing until one cell *processes* it for several
/// consecutive ticks (progress builds while a cell stays in range, decays
/// otherwise). Rewards a persistent "commit and stay" policy that has to hold a
/// goal in recurrent memory — the single-cell counterpart to `CoopFood`'s
/// multi-cell coordination. A separate world list (like `CoopFood`), so the
/// GPU eat-food path and the `Food` struct are untouched.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RipeningFood {
    pub position: [f32; 3],
    pub spawn_tick: u64,
    /// Ticks of sustained processing accumulated so far.
    pub progress: u32,
}

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

/// Evaluate the complementary-signal handshake among the cells currently present
/// at a coop node. `present` is `(idx, a, b)` per cell, where `a`/`b` are the raw
/// signal-A (`out[12]`) and signal-B (`out[13]`) emissions (rectified here). A
/// cell *provides* role A when `a ≥ T` and role B when `b ≥ T` (non-exclusive —
/// a cell may provide both). The handshake unlocks only when both roles are
/// covered by at least two *distinct* present cells; it then returns every
/// provider's index (the reward set). A node where everyone calls (A) but nobody
/// responds (B) stays locked — that asymmetry is the pressure that pulls the
/// responder role into existence. Non-signalling bystanders are never rewarded
/// (no free-riding), and the per-tick `HANDSHAKE_SIGNAL_COST` keeps providers
/// from emitting both channels for free.
pub fn coop_handshake_winners(present: &[(usize, f32, f32)]) -> Option<Vec<usize>> {
    let t = HANDSHAKE_SIGNAL_THRESHOLD;
    let mut winners: Vec<usize> = Vec::new();
    let mut has_a = false;
    let mut has_b = false;
    for &(idx, a, b) in present {
        let a = a.max(0.0);
        let b = b.max(0.0);
        if a >= t || b >= t {
            winners.push(idx);
        }
        has_a |= a >= t;
        has_b |= b >= t;
    }
    // Both roles present AND carried by ≥2 distinct cells (a lone cell emitting
    // both can't handshake with itself).
    if has_a && has_b && winners.len() >= 2 {
        Some(winners)
    } else {
        None
    }
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
pub fn register_coop_arrivals_for_all(
    coops: &mut [CoopFood],
    cells: &[Cell],
    world_half: [f32; 3],
) {
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
    }
}

/// Digestion efficiency for `(kind, carnivore_score)`. Continuous trade-off:
/// `score = 0` is a pure herbivore (Plant only). `Carrion` (cell remains)
/// sits at 0.5 for every score so it doesn't drive specialization in either
/// direction.
#[inline]
pub fn eat_efficiency(kind: FoodKind, carnivore_score: f32) -> f32 {
    let s = carnivore_score.clamp(0.0, 1.0);
    match kind {
        FoodKind::Plant => 1.0 - s,
        FoodKind::Carrion => 0.5,
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
        (1.0 - FOOD_DECAY_PER_SEC * age_sec).max(0.0)
    }

    /// Increment age by one tick. Returns `false` once the food has
    /// expired (value_factor ≤ 0) so the caller can despawn it.
    pub fn age_step(&mut self) -> bool {
        self.age_ticks = self.age_ticks.saturating_add(1);
        self.value_factor() > 0.0
    }
}
