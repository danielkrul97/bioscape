use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::*;

/// Cap on simultaneously alive non-evolving hunters in the legacy spawn
/// pool. With 12 hunters and current damage rates, solo cells absorb only
/// ~0.17 energy/gen — 18× less than bond maintenance, so solo strategies
/// still dominate without further pressure.
pub const HUNTER_TARGET_COUNT: usize = 12;
/// Hunter detection range. Hunter actively pursues cells inside; outside
/// range it random-walks. Set above `MATING_RADIUS` so hunters detect
/// cells before broad-phase mating windows close.
pub const HUNTER_VISION_RADIUS: f32 = 200.0;
/// Attack range — kept smaller than vision so hunters must close in and
/// cells get a window to flee.
pub const HUNTER_ATTACK_RADIUS: f32 = 18.0;
/// Per-tick damage applied to cells in the attack range, both as energy
/// loss and as an addition to `damage_accum` (the brain's damage signal).
/// Linear, no spike bonus — non-evolving hunters have no evolved phenotype.
pub const HUNTER_DAMAGE_PER_TICK: f32 = 8.0;
/// Hunter top speed, intentionally above the cell `MAX_SPEED` cap so cells
/// can't simply outrun pursuit; clustering (≥ 3 bonds = immunity) must be
/// the dominant escape strategy.
pub const HUNTER_MAX_SPEED: f32 = 300.0;
/// Acceleration coefficient for the simple seek-toward-target motion model
/// used by non-evolving hunters: `velocity += dt × ACC × (target_dir −
/// current_dir)`, then clamped to `MAX_SPEED`.
pub const HUNTER_ACC: f32 = 80.0;
/// Random-walk noise applied to idle hunters (no cell in vision). Slow
/// drift so the hunter eventually wanders into a target.
pub const HUNTER_IDLE_DRIFT: f32 = 30.0;
/// Bond-count threshold for hunter discoverability. Cells with ≥ this
/// many bonds drop out of `nearest_attackable_cell` — a cluster is
/// considered "too deeply interior to bother".
pub const HUNTER_BOND_IMMUNITY_THRESHOLD: u32 = 4;
/// Per-bond exposure reduction: `exposure = max(0, 1 − n_bonds × this)²`.
/// At 0/1/2/3 bonds the linear factor lands on 1.0/0.7/0.4/0.1 (squared
/// 1.0/0.49/0.16/0.01) — a 3-bond cluster takes only ~1 % damage. Ramps
/// to near-immunity one bond earlier than `HUNTER_BOND_IMMUNITY_THRESHOLD`,
/// which still gates the binary "is this cell visible" filter.
pub const EXPOSURE_PER_BOND: f32 = 0.30;

/// Per-tick drain coefficient for sensor specialization:
/// `drain = sum(sensor_gains) × this × dt`. Default neutral sum is 3 × 1.0,
/// so the baseline drain is ~0.9/s — comparable with body maintenance.
/// Cells that turn off duplicate sensors in a cluster save proportionally,
/// making specialization net-positive.
pub const SENSOR_GAIN_COST: f32 = 0.3;
/// Per-category gain range. 0 = sensor effectively off, 1 = neutral, 2 =
/// boosted (better detection, higher cost).
pub const MIN_SENSOR_GAIN: f32 = 0.0;
pub const MAX_SENSOR_GAIN: f32 = 2.0;
/// Three sensor categories indexed into `Genome.sensor_gains`:
/// 0 = Vision (food delta, cell delta, rel_size, density),
/// 1 = Chemistry (smell + pheromone gradients),
/// 2 = Defensive (damage signal, thermal_local). Proprio sensors (energy,
/// speed, heading) are always-on and not gained — a cell still needs its
/// own state even in a deep-specialist mode.
pub const SENSOR_CATEGORY_VISION: usize = 0;
pub const SENSOR_CATEGORY_CHEMISTRY: usize = 1;
pub const SENSOR_CATEGORY_DEFENSIVE: usize = 2;
pub const N_SENSOR_CATEGORIES: usize = 3;

/// Map a brain input slot index to its sensor category, or `None` for
/// proprio slots (not gained, not pooled). Used by `apply_sensor_gains`
/// (per-cell multiply) and `pool_bonded_sensors` (max-pool environmental
/// slots over the bond network).
#[inline]
pub fn sensor_slot_category(slot: usize) -> Option<usize> {
    match slot {
        // Food delta (slot 0,1,15) + cell delta (2,3,16) + rel_size (6) +
        // density (13) → Vision
        0 | 1 | 2 | 3 | 6 | 13 | 15 | 16 => Some(SENSOR_CATEGORY_VISION),
        // Smell (7,8,17) + pheromone (11,12,19) → Chemistry
        7 | 8 | 11 | 12 | 17 | 19 => Some(SENSOR_CATEGORY_CHEMISTRY),
        // Damage (14) + thermal (20) → Defensive
        14 | 20 => Some(SENSOR_CATEGORY_DEFENSIVE),
        // Energy (4), speed (5), heading (9,10,18) → proprio, no gain
        _ => None,
    }
}
/// Half-angle (rad) of the directional FOV cone for non-evolving hunters.
/// π/3 = 60° → 120° forward cone. Predators classically have frontal eyes,
/// so cells can flank into the blind spot. Fixed constant — the legacy
/// hunter has no genome.
pub const HUNTER_VISION_FOV: f32 = core::f32::consts::PI / 3.0;
/// Minimum |velocity|² for the directional vision cone to engage. Below
/// the threshold the forward direction is undefined and the hunter falls
/// back to omnidirectional vision so an idle hunter can still find a
/// target. Threshold is well below `HUNTER_IDLE_DRIFT²` so the cone is
/// effectively always active during pursuit or drift.
pub const HUNTER_FORWARD_SPEED_THRESHOLD_SQ: f32 = 1.0;

// ─── Heritable hunter genome (replaces non-evolving hunter) ──────────────────
// The non-evolving hunter (fixed `HUNTER_*` constants) only ever applied
// asymmetric selection: cells evolved to evade but predators didn't. The
// genome below introduces a hunter lifecycle (energy, reproduction, death),
// turning the dynamic into a biological arms race — predator parameters
// drift with predation success while cells keep evolving evasion.

/// Gene ranges — the legacy `HUNTER_VISION_RADIUS`, `HUNTER_MAX_SPEED` etc.
/// values sit roughly in the middle of these ranges and double as defaults
/// for `HunterGenome::random`.
pub const MIN_HUNTER_VISION_RADIUS: f32 = 50.0;
pub const MAX_HUNTER_VISION_RADIUS: f32 = 400.0;
pub const MIN_HUNTER_VISION_FOV: f32 = core::f32::consts::PI / 12.0;
pub const MAX_HUNTER_VISION_FOV: f32 = core::f32::consts::PI;
pub const MIN_HUNTER_MAX_SPEED: f32 = 100.0;
pub const MAX_HUNTER_MAX_SPEED: f32 = 500.0;
pub const MIN_HUNTER_ACC: f32 = 40.0;
pub const MAX_HUNTER_ACC: f32 = 160.0;
pub const MIN_HUNTER_ATTACK_RADIUS: f32 = 10.0;
pub const MAX_HUNTER_ATTACK_RADIUS: f32 = 40.0;
pub const MIN_HUNTER_DAMAGE: f32 = 2.0;
pub const MAX_HUNTER_DAMAGE: f32 = 16.0;
pub const MIN_HUNTER_BODY_SIZE: f32 = 0.5;
pub const MAX_HUNTER_BODY_SIZE: f32 = 2.5;

/// Initial energy at hunter spawn / floor respawn. Higher than cell
/// `INITIAL_ENERGY=100` — a chase cycle can span many ticks without a
/// kill, so hunters need a longer survival runway.
pub const HUNTER_INITIAL_ENERGY: f32 = 500.0;
/// Energy threshold for reproduction. Set equal to `HUNTER_INITIAL_ENERGY`
/// so a fresh hunter is fertile from spawn (cooldown 0); the only gate
/// that remains is spatial proximity to a mate plus the post-coupling
/// cooldown. Higher thresholds emptied the population in smoke runs
/// because the rare proximity events couldn't keep up with mortality.
pub const HUNTER_REPRODUCE_THRESHOLD: f32 = 500.0;
/// Hard cap on the hunter population. Without it a predator boom from a
/// successful generation grows exponentially and drives prey extinction.
pub const HUNTER_MAX_POP: usize = 50;
/// Per-tick vision drain: `vision_radius × fov_factor × VISION_COST × dt`.
pub const HUNTER_VISION_COST: f32 = 0.01;
/// Per-tick kinetic drain: `|v|² × MOTION_COST × dt`.
pub const HUNTER_MOTION_COST: f32 = 0.0001;
/// Per-tick body maintenance: `body_size³ × BODY_COST × dt`.
pub const HUNTER_BODY_COST: f32 = 0.5;
/// Always-on attack upkeep: `damage_per_tick × ATTACK_UPKEEP × dt`. Hunters
/// can trade-off lower damage for lower upkeep — useful for low-energy
/// survivors when the prey field is sparse.
pub const HUNTER_ATTACK_UPKEEP: f32 = 0.02;
/// Energy gained per damage dealt. Calibrated to compensate for the
/// `EXPOSURE_PER_BOND` defense scaling — at average exposure ~0.85 the
/// effective gain (~10.2) sits just above the legacy non-defended value.
pub const HUNTER_ENERGY_PER_DAMAGE: f32 = 12.0;
/// Carrion drops on hunter death. 2× the cell-death drop because hunters
/// carry more biomass.
pub const HUNTER_CARRION_DROP: usize = 2;
/// Reproduce cooldown (ticks) after a split. Prevents instant re-reproduce
/// before cells can catch up; ~half a generation.
pub const HUNTER_REPRODUCE_COOLDOWN_TICKS: u32 = 300;
/// Pack-hunting kill share: a bonded hunter scoring a kill gives each
/// partner `gain × FRAC` extra energy — not conserved, models "pack feed
/// dynamic". A pack of 6 collects ~3.5× the solo payoff, enough for
/// selection to favor packs.
pub const HUNTER_BOND_KILL_SHARE_FRAC: f32 = 0.5;

/// Maximum distance for two fertile hunters to pair. Hunter density is
/// much lower than cell density (max 50 vs max 2500 in the same world
/// volume); a smaller mating radius would mean parents simply never meet.
/// Set to parity with `HUNTER_VISION_RADIUS` — biologically "they can see
/// the partner".
pub const HUNTER_MATING_RADIUS: f32 = 200.0;
/// Brain `output[0]` turn-yaw rate (rad/s). Fixed, not gene-encoded —
/// cells have `turn_rate ∈ [1, 5]`; this 3.0 sits mid-range.
pub const HUNTER_TURN_RATE: f32 = 3.0;
/// Brain `output[7]` turn-pitch rate (rad/s). Lower than `HUNTER_TURN_RATE`
/// because hunter pitch is unclamped (cells have a ±π/12 cap), so a
/// gentler rate avoids overshoot.
pub const HUNTER_PITCH_RATE: f32 = 1.0;

/// Per-hunter heritable parameters. Drift each generation under selection
/// pressure from predation success and survival cost. The brain field
/// reuses the cell `Brain` struct; slot semantics are re-mapped for the
/// hunter in `populate_hunter_brain_inputs` and `apply_brain_motor`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HunterGenome {
    pub vision_radius: f32,
    pub vision_fov: f32,
    pub max_speed: f32,
    pub acceleration: f32,
    pub attack_radius: f32,
    pub damage_per_tick: f32,
    pub body_size: f32,
    pub color_hue: f32,
    /// Cadherin-like recognition for hunter-hunter bonds (8 types, same as
    /// cells). Same-type hunters attract via adhesion and form persistent
    /// bonds on contact; cross-type pairs repel. The hunter and cell
    /// adhesion pools are independent — no hunter-cell bonds.
    pub adhesion_type: u8,
    /// Behavioral controller. Cell-only outputs (morph, attack signal,
    /// bond) are ignored by the hunter motor; the hunter only consumes
    /// `output[0]` (turn), `[1]` (thrust), `[7]` (pitch).
    pub brain: Brain,
}

impl HunterGenome {
    /// Initial random draw — middle of each gene range with ~30 % spread,
    /// enough population diversity to produce a selection signal in
    /// 30–100 gen smoke runs.
    pub fn random(rng: &mut impl Rng) -> Self {
        Self {
            vision_radius: rng.random_range(100.0..300.0),
            vision_fov: rng.random_range(
                core::f32::consts::PI / 6.0..core::f32::consts::PI * 0.75,
            ),
            max_speed: rng.random_range(200.0..400.0),
            acceleration: rng.random_range(60.0..120.0),
            attack_radius: rng.random_range(12.0..28.0),
            damage_per_tick: rng.random_range(4.0..12.0),
            body_size: rng.random_range(0.8..1.6),
            color_hue: rng.random_range(0.0..HUE_RANGE),
            adhesion_type: rng.random_range(0..ADHESION_TYPE_COUNT),
            // `Brain::random` applies `INNATE_THRUST_BIAS`, so fresh hunters
            // start with positive thrust → forward motion; selection then
            // tunes the turn/pitch outputs into coordinated chase behavior.
            brain: Brain::random(rng),
        }
    }

    pub fn mutate(&self, rng: &mut impl Rng, cfg: &HunterMutationConfig) -> Self {
        Self {
            vision_radius: (self.vision_radius + gaussian(rng) * cfg.sigma_vision_radius)
                .clamp(MIN_HUNTER_VISION_RADIUS, MAX_HUNTER_VISION_RADIUS),
            vision_fov: (self.vision_fov + gaussian(rng) * cfg.sigma_vision_fov)
                .clamp(MIN_HUNTER_VISION_FOV, MAX_HUNTER_VISION_FOV),
            max_speed: (self.max_speed + gaussian(rng) * cfg.sigma_max_speed)
                .clamp(MIN_HUNTER_MAX_SPEED, MAX_HUNTER_MAX_SPEED),
            acceleration: (self.acceleration + gaussian(rng) * cfg.sigma_acceleration)
                .clamp(MIN_HUNTER_ACC, MAX_HUNTER_ACC),
            attack_radius: (self.attack_radius + gaussian(rng) * cfg.sigma_attack_radius)
                .clamp(MIN_HUNTER_ATTACK_RADIUS, MAX_HUNTER_ATTACK_RADIUS),
            damage_per_tick: (self.damage_per_tick + gaussian(rng) * cfg.sigma_damage)
                .clamp(MIN_HUNTER_DAMAGE, MAX_HUNTER_DAMAGE),
            body_size: (self.body_size + gaussian(rng) * cfg.sigma_body_size)
                .clamp(MIN_HUNTER_BODY_SIZE, MAX_HUNTER_BODY_SIZE),
            color_hue: (self.color_hue + gaussian(rng) * cfg.sigma_color_hue)
                .rem_euclid(HUE_RANGE),
            adhesion_type: if cfg.adhesion_flip_rate > 0.0
                && ADHESION_TYPE_COUNT > 1
                && rng.random::<f32>() < cfg.adhesion_flip_rate
            {
                let mut t = rng.random_range(0..ADHESION_TYPE_COUNT - 1);
                if t >= self.adhesion_type {
                    t += 1;
                }
                t
            } else {
                self.adhesion_type
            },
            brain: self.brain.mutate(rng, cfg.sigma_brain),
        }
    }

    pub fn crossover(a: &HunterGenome, b: &HunterGenome, rng: &mut impl Rng) -> Self {
        Self {
            vision_radius: if rng.random::<bool>() { a.vision_radius } else { b.vision_radius },
            vision_fov: if rng.random::<bool>() { a.vision_fov } else { b.vision_fov },
            max_speed: if rng.random::<bool>() { a.max_speed } else { b.max_speed },
            acceleration: if rng.random::<bool>() { a.acceleration } else { b.acceleration },
            attack_radius: if rng.random::<bool>() { a.attack_radius } else { b.attack_radius },
            damage_per_tick: if rng.random::<bool>() {
                a.damage_per_tick
            } else {
                b.damage_per_tick
            },
            body_size: if rng.random::<bool>() { a.body_size } else { b.body_size },
            color_hue: if rng.random::<bool>() { a.color_hue } else { b.color_hue },
            adhesion_type: if rng.random::<bool>() { a.adhesion_type } else { b.adhesion_type },
            brain: Brain::crossover(&a.brain, &b.brain, rng),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HunterMutationConfig {
    pub sigma_vision_radius: f32,
    pub sigma_vision_fov: f32,
    pub sigma_max_speed: f32,
    pub sigma_acceleration: f32,
    pub sigma_attack_radius: f32,
    pub sigma_damage: f32,
    pub sigma_body_size: f32,
    pub sigma_color_hue: f32,
    /// Brain-weights gaussian sigma. Matches the cell `sigma_brain` — brain
    /// landscape is large, drift is naturally slow.
    pub sigma_brain: f32,
    /// Per-child probability of flipping `adhesion_type` (uniform pick of a
    /// different type). Mirrors the cell `adhesion_flip_rate`.
    pub adhesion_flip_rate: f32,
}

/// Hunter mutation rates — slightly higher than the cell config because the
/// hunter population is small (12–50) so the per-generation evolution
/// signal needs more drift to stay visible.
pub const HUNTER_MUTATION_CONFIG: HunterMutationConfig = HunterMutationConfig {
    sigma_vision_radius: 10.0,    // 2.9 % of [50, 400]
    sigma_vision_fov: 0.08,       // 2.8 % of FOV range
    sigma_max_speed: 12.0,        // 3.0 % of [100, 500]
    sigma_acceleration: 4.0,      // 3.3 % of [40, 160]
    sigma_attack_radius: 1.0,     // 3.3 % of [10, 40]
    sigma_damage: 0.4,            // 2.9 % of [2, 16]
    sigma_body_size: 0.06,        // 3.0 % of [0.5, 2.5]
    sigma_color_hue: 5.0,         // 1.4 % of HUE_RANGE — slow lineage drift
    sigma_brain: 0.2,
    adhesion_flip_rate: ADHESION_MUTATION_RATE,
};

/// Evolving environmental predator. Per-tick behavior is brain-driven (with
/// a hybrid seek-bootstrap, see `apply_brain_motor`); attacks land on cells
/// with `n_bonds() < HUNTER_BOND_IMMUNITY_THRESHOLD` inside the attack
/// radius.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Hunter {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    /// Stable monotonic identifier. Not used for cell bond resolution —
    /// hunter and cell ID spaces are independent.
    pub hunter_id: u64,
    pub genome: HunterGenome,
    pub energy: f32,
    pub age: u64,
    pub reproduce_cooldown_ticks: u32,
    pub lineage_id: u64,
    pub lineage_birth_gen: u64,
    /// Yaw heading (rad). `output[0]` drives `angular_velocity`, integrated
    /// in `step` like the cell.
    pub heading: f32,
    /// Pitch (rad), unclamped — hunters can move freely in z, while cells
    /// are clamped to ±π/12.
    pub pitch: f32,
    pub angular_velocity: f32,
    pub pitch_velocity: f32,
    /// Brain I/O state for the recurrent channel and diagnostics.
    #[serde(with = "serde_arr_inputs")]
    pub last_inputs: [f32; BRAIN_INPUTS],
    #[serde(with = "serde_arr_hidden")]
    pub last_hidden: [f32; BRAIN_HIDDEN],
    pub last_outputs: [f32; BRAIN_OUTPUTS],
    /// Persistent spring bonds between hunters (no hunter-cell bonds).
    /// `Bond.other_cell_id` stores the partner's `hunter_id` here — the
    /// field name is reused to keep a single `Bond` struct shared across
    /// both bond pools.
    pub bonds: [Option<Bond>; MAX_BONDS_PER_CELL],
    /// Pack-level pooled hidden state (mirror of `Cell.pooled_hidden`).
    /// Mean of `self.last_hidden` and all bonded partners' `last_hidden`,
    /// recomputed per tick. Solo hunters get a copy of their own
    /// `last_hidden` — no behavioral change. Bonded hunters share a
    /// recurrent context for coordinated hunts.
    #[serde(default = "default_pooled_hidden", with = "serde_arr_hidden")]
    pub pooled_hidden: [f32; BRAIN_HIDDEN],
}

impl Hunter {
    /// Random init: random position + zero velocity + random genome.
    pub fn random(
        rng: &mut impl Rng,
        world_half: [f32; 3],
        hunter_id: u64,
        lineage_id: u64,
        lineage_birth_gen: u64,
    ) -> Self {
        let genome = HunterGenome::random(rng);
        Self::from_genome(rng, genome, world_half, hunter_id, lineage_id, lineage_birth_gen)
    }

    /// Spawn with an explicit genome (used by clone-with-mutate during
    /// reproduction and by the floor respawn). Position is random; velocity
    /// starts as a 30 %-speed forward kick; brain state is zero.
    pub fn from_genome(
        rng: &mut impl Rng,
        genome: HunterGenome,
        world_half: [f32; 3],
        hunter_id: u64,
        lineage_id: u64,
        lineage_birth_gen: u64,
    ) -> Self {
        let pos_z = if world_half[2] > 0.0 {
            rng.random_range(-world_half[2]..world_half[2])
        } else {
            0.0
        };
        let direction = rng.random_range(0.0..TAU);
        Self {
            position: [
                rng.random_range(-world_half[0]..world_half[0]),
                rng.random_range(-world_half[1]..world_half[1]),
                pos_z,
            ],
            velocity: [
                direction.cos() * genome.max_speed * 0.3,
                direction.sin() * genome.max_speed * 0.3,
                0.0,
            ],
            hunter_id,
            genome,
            energy: HUNTER_INITIAL_ENERGY,
            age: 0,
            reproduce_cooldown_ticks: 0,
            lineage_id,
            lineage_birth_gen,
            heading: direction,
            pitch: 0.0,
            angular_velocity: 0.0,
            pitch_velocity: 0.0,
            last_inputs: [0.0; BRAIN_INPUTS],
            last_hidden: [0.0; BRAIN_HIDDEN],
            last_outputs: [0.0; BRAIN_OUTPUTS],
            bonds: [None; MAX_BONDS_PER_CELL],
            pooled_hidden: [0.0; BRAIN_HIDDEN],
        }
    }

    /// Per-tick energy drains: vision (∝ `radius × fov_factor`), motion
    /// (∝ `|v|²`), body maintenance (∝ `body_size³`), attack upkeep
    /// (∝ `damage_per_tick`). No aging ramp — hunter lifecycles are short.
    pub fn apply_energy_costs(&mut self, dt: f32) {
        let fov_factor = vision_fov_factor(self.genome.vision_fov);
        self.energy -= self.genome.vision_radius * HUNTER_VISION_COST * fov_factor * dt;
        let v_mag_sq = self.velocity[0] * self.velocity[0]
            + self.velocity[1] * self.velocity[1]
            + self.velocity[2] * self.velocity[2];
        self.energy -= v_mag_sq * HUNTER_MOTION_COST * dt;
        let s = self.genome.body_size;
        self.energy -= s * s * s * HUNTER_BODY_COST * dt;
        self.energy -= self.genome.damage_per_tick * HUNTER_ATTACK_UPKEEP * dt;
    }

    /// Brain-driven motion with a hybrid seek bootstrap. Brain outputs
    /// `[0]`=turn-yaw, `[1]`=thrust, `[7]`=turn-pitch; the rest are ignored.
    ///
    /// The hybrid mixes a deterministic seek-toward-prey direction (60 %)
    /// with the brain output (40 %). Without it, random initial brains
    /// can't chase — random turn outputs produce spinning and the
    /// population collapses into the floor-respawn loop. As selection
    /// shapes the brain to match seek, the mix becomes redundant; if the
    /// brain learns a different strategy (cluster around hot zones, ambush,
    /// retreat at low energy), it dominates.
    pub fn apply_brain_motor(
        &mut self,
        outputs: &[f32; BRAIN_OUTPUTS],
        seek_target: Option<[f32; 3]>,
        dt: f32,
        world_half: [f32; 3],
    ) {
        let brain_turn = outputs[0].clamp(-1.0, 1.0);
        let brain_pitch = outputs[7].clamp(-1.0, 1.0);
        let thrust = outputs[1].clamp(-1.0, 1.0);
        // Compute seek-based turn modulator.
        let (seek_turn, seek_pitch) = match seek_target {
            Some(t) => {
                let d = min_image_delta(self.position, t, world_half);
                let desired_yaw = d[1].atan2(d[0]);
                // Shortest angular distance → [-π, π].
                let mut yaw_diff = desired_yaw - self.heading;
                while yaw_diff > core::f32::consts::PI {
                    yaw_diff -= TAU;
                }
                while yaw_diff < -core::f32::consts::PI {
                    yaw_diff += TAU;
                }
                let dist_xy = (d[0] * d[0] + d[1] * d[1]).sqrt();
                let desired_pitch = if dist_xy > 1e-3 {
                    d[2].atan2(dist_xy)
                } else {
                    0.0
                };
                let pitch_diff = desired_pitch - self.pitch;
                // Normalize na [-1, 1] motor space.
                (
                    (yaw_diff / core::f32::consts::PI).clamp(-1.0, 1.0),
                    (pitch_diff / core::f32::consts::PI).clamp(-1.0, 1.0),
                )
            }
            None => (0.0, 0.0),
        };
        let seek_mix = 0.6;
        let turn = (brain_turn * (1.0 - seek_mix) + seek_turn * seek_mix).clamp(-1.0, 1.0);
        let pitch_t =
            (brain_pitch * (1.0 - seek_mix) + seek_pitch * seek_mix).clamp(-1.0, 1.0);
        self.angular_velocity = turn * HUNTER_TURN_RATE;
        self.pitch_velocity = pitch_t * HUNTER_PITCH_RATE;
        let fwd = forward_vector(self.heading, self.pitch);
        let acc = thrust * self.genome.acceleration;
        self.velocity[0] += fwd[0] * acc * dt;
        self.velocity[1] += fwd[1] * acc * dt;
        self.velocity[2] += fwd[2] * acc * dt;
        let speed_sq = self.velocity[0] * self.velocity[0]
            + self.velocity[1] * self.velocity[1]
            + self.velocity[2] * self.velocity[2];
        let max_sq = self.genome.max_speed * self.genome.max_speed;
        if speed_sq > max_sq {
            let scale = self.genome.max_speed / speed_sq.sqrt();
            self.velocity[0] *= scale;
            self.velocity[1] *= scale;
            self.velocity[2] *= scale;
        }
    }

    /// Pure passive integration: position + heading + pitch update plus
    /// toroidal wrap / z bounce. Active forces (`apply_brain_motor`) must
    /// run before `step`.
    pub fn step(&mut self, dt: f32, world_half: [f32; 3]) {
        self.age = self.age.saturating_add(1);
        if self.reproduce_cooldown_ticks > 0 {
            self.reproduce_cooldown_ticks -= 1;
        }
        // Integrate.
        self.position[0] += self.velocity[0] * dt;
        self.position[1] += self.velocity[1] * dt;
        self.position[2] += self.velocity[2] * dt;
        self.heading += self.angular_velocity * dt;
        self.pitch += self.pitch_velocity * dt;
        // Toroidal wrap xy, z bounce.
        let wx = 2.0 * world_half[0];
        let wy = 2.0 * world_half[1];
        if self.position[0] >= world_half[0] || self.position[0] < -world_half[0] {
            let p = self.position[0] + world_half[0];
            self.position[0] = p - (p / wx).floor() * wx - world_half[0];
        }
        if self.position[1] >= world_half[1] || self.position[1] < -world_half[1] {
            let p = self.position[1] + world_half[1];
            self.position[1] = p - (p / wy).floor() * wy - world_half[1];
        }
        if world_half[2] > 0.0 && self.position[2].abs() > world_half[2] {
            self.velocity[2] = -self.velocity[2];
            self.position[2] = self.position[2].clamp(-world_half[2], world_half[2]);
        }
    }
}

/// Hunter sensor context — predator-flavored subset of the cell
/// `BrainSensors`. "Prey" means the nearest attackable cell (within
/// `vision_radius`, inside the FOV cone, and with
/// `n_bonds() < HUNTER_BOND_IMMUNITY_THRESHOLD`).
#[derive(Debug, Clone, Copy)]
pub struct HunterBrainSensors {
    /// Min-image delta from hunter to the nearest prey
    /// (`cell.position − hunter.position`).
    pub nearest_prey: Option<[f32; 3]>,
    /// `phenotype.effective_radius` of the nearest prey — gives the brain
    /// a "smaller or larger than me" signal for chase-tactic trade-offs.
    pub nearest_prey_size: f32,
    /// Count of attackable cells inside the vision range / cone.
    pub neighbors_in_vision: u32,
    /// Smell field gradient at the hunter's position. Cells emit smell when
    /// they eat, so this acts as a chemical clue toward nearby cell activity.
    pub smell_grad: [f32; 3],
    /// Delta to the nearest same-type hunter in vision — a pack-member /
    /// bond-contact candidate. Lets the brain seek or avoid the pack.
    pub nearest_pack_member: Option<[f32; 3]>,
    /// Same-type hunters in vision (pack density signal).
    pub same_type_in_vision: u32,
}

/// Minimal hunter snapshot for the pack-sense scan in
/// `gather_hunter_sensors`. Holds only the fields the same-type vision
/// check needs, so the caller can avoid a per-tick deep clone of the full
/// `Hunter` struct (genome + brain weights + bonds + recurrent state).
#[derive(Debug, Clone, Copy)]
pub struct HunterSnapshotMin {
    pub hunter_id: u64,
    pub position: [f32; 3],
    pub adhesion_type: u8,
}

impl HunterSnapshotMin {
    pub fn from_hunter(h: &Hunter) -> Self {
        Self {
            hunter_id: h.hunter_id,
            position: h.position,
            adhesion_type: h.genome.adhesion_type,
        }
    }
}

/// Gather hunter sensors using the cell spatial grid. Caller rebuilds
/// `cell_grid` from `cells.iter().enumerate().map(|(i, c)| (i, c.position, ()))`;
/// this function only walks the 3³ buckets around the hunter and does the
/// narrow-phase distance + cone test. `other_hunters` stays as brute force
/// — `HUNTER_MAX_POP ≤ 50` so `H²` is trivial — but it's typed as
/// `&[HunterSnapshotMin]` so the caller doesn't need a per-tick deep clone.
pub fn gather_hunter_sensors(
    hunter: &Hunter,
    cells: &[Cell],
    cell_grid: &SpatialGrid<usize, ()>,
    other_hunters: &[HunterSnapshotMin],
    smell: &SmellField,
    world_half: [f32; 3],
) -> HunterBrainSensors {
    let vision_r = hunter.genome.vision_radius;
    let vision_r2 = vision_r * vision_r;
    let cos_fov = hunter.genome.vision_fov.cos();
    let speed_sq = hunter.velocity[0] * hunter.velocity[0]
        + hunter.velocity[1] * hunter.velocity[1]
        + hunter.velocity[2] * hunter.velocity[2];
    let cone_active = speed_sq > HUNTER_FORWARD_SPEED_THRESHOLD_SQ;
    let forward = if cone_active {
        let inv = 1.0 / speed_sq.sqrt();
        [
            hunter.velocity[0] * inv,
            hunter.velocity[1] * inv,
            hunter.velocity[2] * inv,
        ]
    } else {
        [0.0; 3]
    };
    let mut best: Option<([f32; 3], f32, f32)> = None; // (delta, d2, prey_size)
    let mut count: u32 = 0;
    cell_grid.for_each_in_radius_toroidal(
        hunter.position,
        vision_r,
        world_half,
        |idx, ghost_pos, ()| {
            let c = &cells[idx];
            if c.n_bonds() >= HUNTER_BOND_IMMUNITY_THRESHOLD {
                return;
            }
            let d = [
                ghost_pos[0] - hunter.position[0],
                ghost_pos[1] - hunter.position[1],
                ghost_pos[2] - hunter.position[2],
            ];
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if d2 >= vision_r2 {
                return;
            }
            if cone_active && !fov_cone_accept(d, d2, forward, cos_fov) {
                return;
            }
            count += 1;
            let prey_size = c.phenotype.effective_radius();
            match best {
                None => best = Some((d, d2, prey_size)),
                Some((_, bd2, _)) if d2 < bd2 => best = Some((d, d2, prey_size)),
                _ => {}
            }
        },
    );
    let smell_grad = smell.gradient_at(hunter.position, SMELL_SAMPLE_EPSILON);
    // Pack scan — same-type hunters in vision. `HUNTER_MAX_POP ≤ 50`, so
    // brute-force `H²` is fine; a spatial grid wouldn't help.
    let mut nearest_pack: Option<([f32; 3], f32)> = None;
    let mut same_type_count: u32 = 0;
    let own_type = hunter.genome.adhesion_type;
    let own_id = hunter.hunter_id;
    for o in other_hunters {
        if o.hunter_id == own_id {
            continue;
        }
        if o.adhesion_type != own_type {
            continue;
        }
        let d = min_image_delta(hunter.position, o.position, world_half);
        let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if d2 >= vision_r2 {
            continue;
        }
        same_type_count += 1;
        match nearest_pack {
            None => nearest_pack = Some((d, d2)),
            Some((_, bd2)) if d2 < bd2 => nearest_pack = Some((d, d2)),
            _ => {}
        }
    }
    HunterBrainSensors {
        nearest_prey: best.map(|(d, _, _)| d),
        nearest_prey_size: best.map(|(_, _, s)| s).unwrap_or(0.0),
        neighbors_in_vision: count,
        smell_grad,
        nearest_pack_member: nearest_pack.map(|(d, _)| d),
        same_type_in_vision: same_type_count,
    }
}

/// Populate the brain input vector for a hunter. Reuses the cell input
/// layout with predator-specific slot semantics:
///   0,1,15  nearest-prey delta (cell food slots)
///   2,3,16  nearest-pack-member delta (same-type hunter pull)
///   4       energy / `HUNTER_REPRODUCE_THRESHOLD`
///   5       speed / max_speed
///   6       prey-size relative (prey / own_size − 1)
///   7,8,17  smell gradient x/y/z
///   9,10,18 heading x/y/z (forward vector)
///   11      pack size normalized (`n_bonds / MAX_BONDS_PER_CELL`)
///   12      pack density (same-type cells / `DENSITY_NORM_COUNT`)
///   13      cell density in vision (count / `DENSITY_NORM_COUNT`)
///   14, 19, 20  filler (unused for hunters)
///   21..52  recurrent slots — `pooled_hidden`
pub fn populate_hunter_brain_inputs(
    hunter: &mut Hunter,
    sensors: &HunterBrainSensors,
) -> [f32; BRAIN_INPUTS] {
    let vision_r = hunter.genome.vision_radius.max(0.01);
    let max_speed = hunter.genome.max_speed.max(1e-3);
    let speed_xy = hunter.velocity[0].hypot(hunter.velocity[1]);
    let speed_norm = (speed_xy / max_speed).clamp(0.0, 1.0);
    let energy_norm = (hunter.energy / HUNTER_REPRODUCE_THRESHOLD).clamp(0.0, 1.5);
    let mut inputs = [0.0_f32; BRAIN_INPUTS];
    if let Some(d) = sensors.nearest_prey {
        inputs[0] = d[0] / vision_r;
        inputs[1] = d[1] / vision_r;
        inputs[15] = d[2] / vision_r;
        let own_size = hunter.genome.body_size.max(0.01);
        inputs[6] = (sensors.nearest_prey_size - own_size) / own_size;
    }
    if let Some(d) = sensors.nearest_pack_member {
        inputs[2] = d[0] / vision_r;
        inputs[3] = d[1] / vision_r;
        inputs[16] = d[2] / vision_r;
    }
    inputs[4] = energy_norm;
    inputs[5] = speed_norm;
    inputs[7] = (sensors.smell_grad[0] * SMELL_NORMALIZATION_GAIN).tanh();
    inputs[8] = (sensors.smell_grad[1] * SMELL_NORMALIZATION_GAIN).tanh();
    inputs[17] = (sensors.smell_grad[2] * SMELL_NORMALIZATION_GAIN).tanh();
    let fwd = forward_vector(hunter.heading, hunter.pitch);
    inputs[9] = fwd[0];
    inputs[10] = fwd[1];
    inputs[18] = fwd[2];
    let n_bonds = hunter.bonds.iter().filter(|b| b.is_some()).count() as f32;
    inputs[11] = (n_bonds / MAX_BONDS_PER_CELL as f32).min(1.0);
    inputs[12] = (sensors.same_type_in_vision as f32 / DENSITY_NORM_COUNT).tanh();
    inputs[13] = (sensors.neighbors_in_vision as f32 / DENSITY_NORM_COUNT).tanh();
    // Recurrent slots feed `pooled_hidden`, not `last_hidden`, so bonded
    // hunters share recurrent state. Solo hunters end up with
    // `pooled_hidden == last_hidden` from `pool_bonded_hunter_hidden`.
    inputs[BRAIN_INPUTS_SENSORY..BRAIN_INPUTS_SENSORY + BRAIN_RECURRENT]
        .copy_from_slice(&hunter.pooled_hidden[..BRAIN_RECURRENT]);
    inputs
}

/// Pool `last_hidden` across each bonded hunter pack (hunter mirror of
/// `pool_bonded_hidden` for cells). Runs before the brain forward pass.
/// Solo hunters end up with a copy of their own `last_hidden`.
pub fn pool_bonded_hunter_hidden(hunters: &mut [Hunter]) {
    let n = hunters.len();
    if n == 0 {
        return;
    }
    let id_to_idx: rustc_hash::FxHashMap<u64, usize> = hunters
        .iter()
        .enumerate()
        .map(|(i, h)| (h.hunter_id, i))
        .collect();
    let snapshot: Vec<[f32; BRAIN_HIDDEN]> = hunters.iter().map(|h| h.last_hidden).collect();
    let bonds_snapshot: Vec<[Option<Bond>; MAX_BONDS_PER_CELL]> =
        hunters.iter().map(|h| h.bonds).collect();
    for i in 0..n {
        let mut sum = snapshot[i];
        let mut count = 1usize;
        for bond_opt in bonds_snapshot[i].iter() {
            if let Some(bond) = bond_opt {
                if let Some(&j) = id_to_idx.get(&bond.other_cell_id) {
                    for k in 0..BRAIN_HIDDEN {
                        sum[k] += snapshot[j][k];
                    }
                    count += 1;
                }
            }
        }
        if count > 1 {
            let inv = 1.0 / count as f32;
            for k in 0..BRAIN_HIDDEN {
                sum[k] *= inv;
            }
        }
        hunters[i].pooled_hidden = sum;
    }
}

/// Asexual clone-with-mutate, kept for the floor-respawn path. Parent
/// splits energy 50/50 with the child; the child inherits a mutated genome
/// and the parent's `lineage_id` (lineage continuity). The primary
/// reproduction path is sexual (`make_hunter_mating_child`).
pub fn make_hunter_child(
    parent: &Hunter,
    rng: &mut impl Rng,
    world_half: [f32; 3],
    hunter_id: u64,
    current_gen: u64,
) -> Hunter {
    let child_genome = parent.genome.mutate(rng, &HUNTER_MUTATION_CONFIG);
    let mut child = Hunter::from_genome(
        rng,
        child_genome,
        world_half,
        hunter_id,
        parent.lineage_id,
        current_gen,
    );
    // Spawn at parent position; the next step + idle drift will scatter.
    child.position = parent.position;
    child.energy = parent.energy * 0.5;
    child.reproduce_cooldown_ticks = HUNTER_REPRODUCE_COOLDOWN_TICKS;
    child
}

/// Sexual reproduction for hunters, symmetric with cells'
/// `make_mating_child`. Per-field 50/50 crossover plus a brain crossover,
/// then mutation. Lineage follows `parent_a` (single-parent inheritance);
/// spawn position is the midpoint and energy is `a + b` (callers halve
/// both parents before calling).
///
/// RNG draw order is fixed at: crossover → mutate → `from_genome`
/// (3 position draws + 1 direction draw, immediately overridden by the
/// midpoint write below). Reordering here shifts the RNG stream and
/// breaks CSV reproducibility across seeds.
pub fn make_hunter_mating_child(
    parent_a: &Hunter,
    parent_b: &Hunter,
    rng: &mut impl Rng,
    world_half: [f32; 3],
    hunter_id: u64,
    current_gen: u64,
) -> Hunter {
    let child_genome = HunterGenome::crossover(&parent_a.genome, &parent_b.genome, rng)
        .mutate(rng, &HUNTER_MUTATION_CONFIG);
    let mut child = Hunter::from_genome(
        rng,
        child_genome,
        world_half,
        hunter_id,
        parent_a.lineage_id,
        current_gen,
    );
    child.position = [
        (parent_a.position[0] + parent_b.position[0]) * 0.5,
        (parent_a.position[1] + parent_b.position[1]) * 0.5,
        (parent_a.position[2] + parent_b.position[2]) * 0.5,
    ];
    child.energy = parent_a.energy + parent_b.energy;
    child.reproduce_cooldown_ticks = HUNTER_REPRODUCE_COOLDOWN_TICKS;
    child
}

/// Index of the nearest attackable cell in vision range, or `None`. A cell
/// is attackable when `n_bonds() < HUNTER_BOND_IMMUNITY_THRESHOLD` — the
/// hunter ignores immune clusters entirely. Vision uses the heritable
/// `vision_radius` and `vision_fov` from the hunter's genome; below the
/// `HUNTER_FORWARD_SPEED_THRESHOLD_SQ` speed floor the cone falls back to
/// omnidirectional so a stalled hunter can still acquire a target.
pub fn nearest_attackable_cell(
    hunter: &Hunter,
    cells: &[Cell],
    cell_grid: &SpatialGrid<usize, ()>,
    world_half: [f32; 3],
) -> Option<usize> {
    let vision_r = hunter.genome.vision_radius;
    let vision_r2 = vision_r * vision_r;
    let cos_fov = hunter.genome.vision_fov.cos();
    let speed_sq = hunter.velocity[0] * hunter.velocity[0]
        + hunter.velocity[1] * hunter.velocity[1]
        + hunter.velocity[2] * hunter.velocity[2];
    let cone_active = speed_sq > HUNTER_FORWARD_SPEED_THRESHOLD_SQ;
    let forward = if cone_active {
        let inv = 1.0 / speed_sq.sqrt();
        [
            hunter.velocity[0] * inv,
            hunter.velocity[1] * inv,
            hunter.velocity[2] * inv,
        ]
    } else {
        [0.0; 3]
    };
    let mut best: Option<(usize, f32)> = None;
    cell_grid.for_each_in_radius_toroidal(
        hunter.position,
        vision_r,
        world_half,
        |idx, ghost_pos, ()| {
            let c = &cells[idx];
            if c.n_bonds() >= HUNTER_BOND_IMMUNITY_THRESHOLD {
                return;
            }
            // Vector from hunter to cell — sign matters for the cone test.
            // `ghost_pos` is already toroidally min-image (grid-wrapped).
            let d = [
                ghost_pos[0] - hunter.position[0],
                ghost_pos[1] - hunter.position[1],
                ghost_pos[2] - hunter.position[2],
            ];
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if d2 >= vision_r2 {
                return;
            }
            if cone_active && !fov_cone_accept(d, d2, forward, cos_fov) {
                return;
            }
            match best {
                None => best = Some((idx, d2)),
                Some((_, bd2)) if d2 < bd2 => best = Some((idx, d2)),
                _ => {}
            }
        },
    );
    best.map(|(i, _)| i)
}
