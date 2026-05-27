use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::*;

/// Persistent spring bond between two cells. Stateful — survives across
/// ticks until it tears (overstretch) or the partner dies. Each `Bond` is
/// stored in only one of the two cells' slots; Newton's 3rd law applies
/// because each side independently runs its own spring-force update.
/// Stiffness and damping freeze at formation (the pair-mean of both genomes)
/// — later mutations in those genes affect only future bonds, making
/// existing bonds stable contracts rather than per-tick re-evaluations.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Bond {
    /// Stable identifier of the partner (`Cell.cell_id`). Resolved to an
    /// index each tick via `FxHashMap<u64, usize>`; if the partner died,
    /// the bond is dangling and the pruning pass drops it.
    pub other_cell_id: u64,
    /// Spring rest length in world units. Set at formation to
    /// `current_distance × BOND_REST_LENGTH_SLACK`.
    pub rest_length: f32,
    pub stiffness: f32,
    pub damping: f32,
    /// Ticks since formation. Diagnostic only — no physics impact.
    pub age_ticks: u32,
}

/// Sprint 196: per-host endosymbiont passenger. Carries its own full `Genome`
/// so it can co-evolve independently of the host (mitochondria-style: separate
/// genetic lineage living inside the host). `lineage_id` uses a counter
/// independent from host lineages and stays stable across predation transfer;
/// `age` is reset at reproduction (offspring symbiont = fresh copy) and
/// persists across transfer events. Sprint 196 only plumbs data and
/// inheritance — no shader reads the symbiont yet.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Symbiont {
    pub genome: Genome,
    pub lineage_id: u64,
    pub age: u64,
    #[serde(default)]
    pub deficit_streak: u32,
}

impl Symbiont {
    pub fn new(genome: Genome, lineage_id: u64) -> Self {
        Self { genome, lineage_id, age: 0, deficit_streak: 0 }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Cell {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    /// Yaw rate — rotation around the vertical axis. Brain output[0].
    pub angular_velocity: f32,
    /// Pitch rate — rotation around the horizontal axis perpendicular to
    /// the xy projection of forward. Brain output[7]; same drag as yaw.
    pub pitch_velocity: f32,
    pub energy: f32,
    /// Persists even when velocity hits zero — `atan2(0, 0)` would otherwise
    /// collapse to 0 and bias evolution toward east-facing motion.
    pub heading: f32,
    /// Pitch angle, clamped to `[-π/12, +π/12]` in `integrate_kinematics`.
    /// Together with `heading` it defines the forward unit vector
    /// `(cos(y)·cos(p), sin(y)·cos(p), sin(p))`. No roll — cells are
    /// axially symmetric.
    pub pitch: f32,
    /// Lineage tracking — inherited from parent at reproduction (no
    /// mutation). `birth_gen` records the generation when the lineage was
    /// created (initial population: 0; speciation events would bump it).
    pub lineage_id: u64,
    pub lineage_birth_gen: u64,
    /// Recent activations from the last brain forward pass. Hebbian updates
    /// read these to credit-assign reward events with a 1-tick myopic window.
    #[serde(with = "serde_arr_inputs")]
    pub last_inputs: [f32; BRAIN_INPUTS],
    #[serde(with = "serde_arr_hidden")]
    pub last_hidden: [f32; BRAIN_HIDDEN],
    pub last_outputs: [f32; BRAIN_OUTPUTS],
    /// Per-channel emission from the previous tick. `emit_pheromones`
    /// updates this after computing the new emit value.
    #[serde(default)]
    pub last_emit: [f32; N_PHEROMONE_CHANNELS],
    /// Squared tick-to-tick emit deltas, accumulated across the generation:
    /// `burst_accum[ch] += (current − last_emit[ch])²`. Reset in
    /// `write_stats`. Higher value = burstier (continuous emit has small
    /// frame-to-frame deltas → low burst score).
    #[serde(default)]
    pub burst_accum: [f32; N_PHEROMONE_CHANNELS],
    /// Cluster-shared brain memory: pre-tick mean of `last_hidden` over the
    /// bond network (self + bonded partners). The recurrent input slots feed
    /// `pooled_hidden` instead of `last_hidden`, giving clustered cells a
    /// proto-distributed-cognition context window. Solo cells fall back to
    /// `pooled_hidden == last_hidden`.
    #[serde(default = "default_pooled_hidden", with = "serde_arr_hidden")]
    pub pooled_hidden: [f32; BRAIN_HIDDEN],
    /// Bond-mediated communication inbox. Mean of bonded peers' previous-tick
    /// `last_outputs[12..14]` (= partneři brain output[BRAIN_OUTPUTS-2..]).
    /// Computed pre-brain_act, read by populate_brain_inputs do slots [27..29].
    /// Solo cells: zero (no bonds → no inbox).
    #[serde(default = "default_bonded_inbox")]
    pub bonded_inbox: [f32; N_BOND_MSG_CHANNELS],
    /// Involuntary energy drain accumulated this tick (predation + hazard).
    /// The brain reads it next tick as a damage input, then it resets to 0.
    /// Voluntary costs (driven by the cell's own outputs) are NOT written
    /// here — this slot is reserved for external injuries.
    pub damage_accum: f32,
    /// Ticks since spawn or mating birth. Drives the aging body-cost ramp.
    pub age: u64,
    /// Refractory period after mating, decremented per tick. Mating gating
    /// reads `== 0`.
    pub reproduce_cooldown_ticks: u32,
    /// Stable per-cell identifier for bond resolution; assigned by a
    /// World-level monotonic counter at spawn / mating birth. `lineage_id`
    /// is shared across a lineage, so we need a separate per-cell id.
    pub cell_id: u64,
    /// Persistent spring bonds. Fixed-size array of `Option<Bond>` so `Cell`
    /// stays `Copy` and there are no per-cell heap allocations.
    pub bonds: [Option<Bond>; MAX_BONDS_PER_CELL],
    /// Sprint 202: rest cosine for every pair of bond slots, captured at the
    /// moment the second bond of the pair forms. `bond_rest_cos[a][b] = cos`
    /// where `cos` is the angle between bond `a` and bond `b` as seen from
    /// this cell at formation time. Symmetric matrix (`[a][b] == [b][a]`);
    /// diagonal and rows/cols of empty slots are unused (the shader skips
    /// inactive bonds). Cleared per slot when a bond breaks.
    #[serde(default = "default_bond_rest_cos")]
    pub bond_rest_cos: [[f32; MAX_BONDS_PER_CELL]; MAX_BONDS_PER_CELL],
    /// Bistable cell state in `[0, 1]`. Not in the genome — inherited from
    /// the parent with noise, acting as phenotypic memory across generations.
    /// Positive feedback around 0.5 plus a `n_bonds()` bias produces two
    /// stable attractors (~0 selfish, ~1 altruist) and regulates food-share
    /// fraction within a bonded cluster.
    pub cell_state: f32,
    /// Squared distance to the closest food seen by the sensor stage; set in
    /// `brain_act` from the food-grid scan. `eat_food` uses it as an early
    /// skip: if `d² > (EAT_RADIUS × max_axis + slack)²`, no food is reachable
    /// even with maximum per-tick movement, so the food-grid query can be
    /// skipped entirely.
    ///
    /// `f32::MAX` = no food in sensor range. `0.0` (set on the `--gpu-full`
    /// path) disables the skip when the CPU sensor stage didn't run.
    #[serde(default = "f32_max_default")]
    pub last_best_food_d2: f32,
    /// Per-cell xoshiro128++ stream for Brownian motion. The state is
    /// derived from `cell_id` at spawn / reproduce so CPU and GPU paths
    /// share the same RNG sequence — that's the single biggest lever for
    /// CPU/GPU trajectory parity (per-tick velocity perturbation was the
    /// fastest source of divergence). `serde(default)` keeps older
    /// checkpoints loadable; the default state is a sentinel non-zero that
    /// older saves overwrite on first `apply_brownian` re-seed.
    #[serde(default)]
    pub xoshiro_state: Xoshiro128PlusPlus,
    /// Sensed whisker signal from the previous tick — the spring-damper
    /// model's `sensed_k = clamp(1 − deflection, 0, 1) + noise`, NOT the raw
    /// raycast distance (Sprint 195). Populated by `update_whiskers`; read by
    /// `populate_brain_inputs` into slots [33..39] and by the renderer
    /// whisker overlay. Defaults to all-ones (no walls) so when the maze is
    /// off the brain reads neutral whisker signal.
    #[serde(default = "default_whiskers")]
    pub last_whisker_distances: [f32; WHISKER_COUNT],
    /// Sprint 195: per-whisker spring-damper state — `deflection` (the bent
    /// angle, 0 = straight) and its time derivative. Driven by wall proximity
    /// (`target = 1 − raw_raycast`), integrated semi-implicit Euler each tick
    /// in `update_whiskers`. May overshoot below 0 / above 1 (ringing). The
    /// GPU full pipeline keeps the equivalent state in `whisker_state_buf`.
    #[serde(default = "default_whiskers_zero")]
    pub whisker_deflection: [f32; WHISKER_COUNT],
    #[serde(default = "default_whiskers_zero")]
    pub whisker_deflection_vel: [f32; WHISKER_COUNT],
    /// Wave 2 episodic novelty: ring buffer of recently visited
    /// `NOVELTY_GRID_CELL_SIZE`-bucketed voxel indices. New entries push out
    /// the oldest. `novelty_head` is the next-write slot index. When the
    /// cell enters a voxel not in the buffer, it earns a Hebbian reward
    /// boost via `NOVELTY_REWARD_MAGNITUDE` and the index is recorded.
    #[serde(default = "default_novelty_history")]
    pub novelty_history: [u32; NOVELTY_HISTORY_LEN],
    #[serde(default)]
    pub novelty_head: u8,
    /// Sprint 134: consecutive ticks the cell took non-zero damage from the
    /// predate dispatch. Reset to 0 on the first damage-free tick. Drives the
    /// `EscapedAttack` reward — once the streak crosses
    /// `ESCAPE_STREAK_THRESHOLD` and the cell makes it through a tick without
    /// new damage, it earns a Hebbian credit for whatever motor pattern
    /// pulled it out.
    #[serde(default)]
    pub under_attack_streak: u16,
    /// Sprint 134: throttle so the escape reward fires at most once per
    /// `ESCAPE_COOLDOWN_TICKS` window. Without it, a cell oscillating between
    /// `damage > 0` and `damage = 0` ticks would harvest the bonus every
    /// other tick.
    #[serde(default)]
    pub escape_cooldown_ticks: u16,
    /// Tracks "was in a hazard zone (taking drain) on the previous tick" so
    /// `apply_hazards` can detect the transition-out event and push a
    /// `RewardKind::HazardAvoided` reward — credits the sensing+motor
    /// decision that produced the escape, not just demographic survival.
    #[serde(default)]
    pub was_in_hazard_last_tick: bool,
    pub phenotype: Phenotype,
    pub genome: Genome,
    /// Sprint 196: endosymbiont passenger. `None` for the bulk of cells; the
    /// bearer fraction at init is `SYMBIONT_INIT_FRACTION` and changes only
    /// at reproduction (P_inherit) and predation transfer (P_capture).
    #[serde(default)]
    pub symbiont: Option<Symbiont>,
}

fn default_whiskers() -> [f32; WHISKER_COUNT] {
    [1.0; WHISKER_COUNT]
}

fn default_whiskers_zero() -> [f32; WHISKER_COUNT] {
    [0.0; WHISKER_COUNT]
}

fn default_novelty_history() -> [u32; NOVELTY_HISTORY_LEN] {
    [u32::MAX; NOVELTY_HISTORY_LEN]
}

fn f32_max_default() -> f32 {
    f32::MAX
}

#[derive(Debug, Clone, Copy)]
pub struct SpikeAttackEntry {
    pub dir: [f32; 3],
    pub gain: f32,
}

impl SpikeAttackEntry {
    pub const ZERO: Self = Self {
        dir: [0.0, 0.0, 0.0],
        gain: 0.0,
    };
}

#[derive(Debug, Clone, Copy)]
pub struct SpikeAttackCache {
    pub entries: [SpikeAttackEntry; SPIKE_SLOTS],
    pub count: usize,
}

impl Default for SpikeAttackCache {
    fn default() -> Self {
        Self {
            entries: [SpikeAttackEntry::ZERO; SPIKE_SLOTS],
            count: 0,
        }
    }
}

impl Cell {
    pub fn random(
        rng: &mut impl Rng,
        world_half: [f32; 3],
        lineage_id: u64,
        lineage_birth_gen: u64,
        cell_id: u64,
    ) -> Self {
        let genome = Genome::random(rng);
        Self::from_genome(rng, genome, world_half, lineage_id, lineage_birth_gen, cell_id)
    }

    /// Number of populated bond slots. Predation defense scales with this.
    #[inline]
    pub fn n_bonds(&self) -> u32 {
        self.bonds.iter().filter(|b| b.is_some()).count() as u32
    }

    /// Sprint 201: incoming-damage multiplier. Bearer cells take only
    /// `(1 - SYMBIONT_DAMAGE_RESIST_FRAC)` of each damage event — applied at
    /// hazards, predation hits, and maze wall bumps. Non-bearers see no
    /// reduction (multiplier = 1.0).
    #[inline]
    pub fn damage_resist_factor(&self) -> f32 {
        if self.symbiont.is_some() {
            1.0 - SYMBIONT_DAMAGE_RESIST_FRAC
        } else {
            1.0
        }
    }

    pub fn from_genome(
        rng: &mut impl Rng,
        genome: Genome,
        world_half: [f32; 3],
        lineage_id: u64,
        lineage_birth_gen: u64,
        cell_id: u64,
    ) -> Self {
        let direction = rng.random_range(0.0..TAU);
        let phenotype = Phenotype::from_genome(&genome);
        // Skip the z-axis RNG draw when the world is 2D so the RNG stream
        // matches the pre-3D baseline byte-for-byte.
        let pos_z = if world_half[2] > 0.0 {
            rng.random_range(-world_half[2]..world_half[2])
        } else {
            0.0
        };
        Self {
            position: [
                rng.random_range(-world_half[0]..world_half[0]),
                rng.random_range(-world_half[1]..world_half[1]),
                pos_z,
            ],
            velocity: [
                direction.cos() * genome.max_speed,
                direction.sin() * genome.max_speed,
                0.0,
            ],
            angular_velocity: 0.0,
            pitch_velocity: 0.0,
            energy: INITIAL_ENERGY,
            heading: direction,
            // Cells start horizontal; z-motion only develops through brain
            // output[7]. Drawing a random initial pitch would consume an
            // RNG slot and break reproducibility against the 2D baseline.
            pitch: 0.0,
            lineage_id,
            lineage_birth_gen,
            last_inputs: [0.0; BRAIN_INPUTS],
            last_hidden: [0.0; BRAIN_HIDDEN],
            last_outputs: [0.0; BRAIN_OUTPUTS],
            last_emit: [0.0; N_PHEROMONE_CHANNELS],
            burst_accum: [0.0; N_PHEROMONE_CHANNELS],
            pooled_hidden: [0.0; BRAIN_HIDDEN],
            bonded_inbox: [0.0; N_BOND_MSG_CHANNELS],
            damage_accum: 0.0,
            age: 0,
            reproduce_cooldown_ticks: 0,
            cell_id,
            bonds: [None; MAX_BONDS_PER_CELL],
            bond_rest_cos: [[0.0; MAX_BONDS_PER_CELL]; MAX_BONDS_PER_CELL],
            // Kick around 0.5 (the unstable fixed point of the feedback)
            // so cells don't get stuck in the neutral zone at spawn. The
            // RNG draw is appended at the end so earlier draws retain
            // their original order.
            cell_state: 0.5 + rng.random_range(-CELL_STATE_INIT_KICK..CELL_STATE_INIT_KICK),
            last_best_food_d2: f32::MAX,
            // Seed mirrors the GPU upload path (`CellsGpu::upload_xoshiro_seeds`
            // called with `c.cell_id`). CPU and GPU brownian streams stay in
            // lockstep as long as both initialize from the same cell_id.
            xoshiro_state: Xoshiro128PlusPlus::from_cell_id(cell_id),
            last_whisker_distances: [1.0; WHISKER_COUNT],
            whisker_deflection: [0.0; WHISKER_COUNT],
            whisker_deflection_vel: [0.0; WHISKER_COUNT],
            novelty_history: [u32::MAX; NOVELTY_HISTORY_LEN],
            novelty_head: 0,
            under_attack_streak: 0,
            escape_cooldown_ticks: 0,
            was_in_hazard_last_tick: false,
            phenotype,
            genome,
            symbiont: None,
        }
    }

    /// Per-tick physics: kinematic update from velocity / angular_velocity,
    /// anisotropic + angular drag, energy drains, world bounce. Brain-applied
    /// forces must be integrated into the velocity / angular_velocity fields
    /// *before* `step` runs — `step` is purely passive integration and
    /// dissipation.
    ///
    /// Anisotropic drag splits velocity into heading-aligned (par) vs.
    /// heading-perpendicular (perp) components. The cross-section in body
    /// frame differs: forward motion sees `width`, sideways motion sees
    /// `length`. For an isotropic body (length = width = height) the result
    /// reduces to the original radially symmetric drag.
    pub fn step(
        &mut self,
        dt: f32,
        world_half: [f32; 3],
        tick: u64,
        generation: u64,
        physics: &PhysicsConfig,
    ) {
        self.step_with_climate(dt, world_half, tick, generation, physics, 0.0);
    }

    /// Step variant that accepts a precomputed shock-aware climate offset.
    /// `climate_offset = 0.0` reproduces `step` exactly.
    pub fn step_with_climate(
        &mut self,
        dt: f32,
        world_half: [f32; 3],
        tick: u64,
        generation: u64,
        physics: &PhysicsConfig,
        climate_offset: f32,
    ) {
        let ctx = ThermalCtx::for_tick(tick, generation);
        self.step_with_thermal(dt, world_half, &ctx, physics, climate_offset);
    }

    /// Hot-loop variant: caller builds `ThermalCtx::for_tick` once per
    /// dispatch and passes it in, saving two sin / modulo pairs per cell.
    /// Bit-identical with `step_with_climate(.., tick, generation, ..,
    /// climate_offset)` when `ctx == ThermalCtx::for_tick(tick, generation)`.
    pub fn step_with_thermal(
        &mut self,
        dt: f32,
        world_half: [f32; 3],
        ctx: &ThermalCtx,
        physics: &PhysicsConfig,
        climate_offset: f32,
    ) {
        // Aging + cooldown decrement run before energy costs so the age ramp
        // sees the post-increment age.
        self.age = self.age.saturating_add(1);
        if let Some(s) = self.symbiont.as_mut() {
            s.age = s.age.saturating_add(1);
        }
        if self.reproduce_cooldown_ticks > 0 {
            self.reproduce_cooldown_ticks -= 1;
        }
        self.integrate_kinematics(dt, world_half);
        self.apply_anisotropic_drag(dt, physics);
        self.apply_angular_drag(dt, physics);
        self.apply_energy_costs(dt, world_half, ctx, physics, climate_offset);
        self.apply_world_bounce(world_half);
        self.update_cell_state(dt);
    }

    /// Maze-aware variant of `step_with_thermal`. When `obstacles` is None,
    /// reduces to `step_with_thermal` exactly. When Some, the XY world
    /// boundary becomes hard walls (no toroidal wrap) and a post-bounce
    /// obstacle collision pass pushes the cell out of any wall it overlaps.
    pub fn step_with_thermal_maze(
        &mut self,
        dt: f32,
        world_half: [f32; 3],
        ctx: &ThermalCtx,
        physics: &PhysicsConfig,
        climate_offset: f32,
        obstacles: Option<&ObstacleField>,
    ) {
        self.age = self.age.saturating_add(1);
        if let Some(s) = self.symbiont.as_mut() {
            s.age = s.age.saturating_add(1);
        }
        if self.reproduce_cooldown_ticks > 0 {
            self.reproduce_cooldown_ticks -= 1;
        }
        self.integrate_kinematics(dt, world_half);
        self.apply_anisotropic_drag(dt, physics);
        self.apply_angular_drag(dt, physics);
        self.apply_energy_costs(dt, world_half, ctx, physics, climate_offset);
        match obstacles {
            Some(field) => {
                self.apply_world_bounce_bounded(world_half);
                self.apply_obstacle_collision(field);
            }
            None => self.apply_world_bounce(world_half),
        }
        self.update_cell_state(dt);
    }

    /// Bistable cell-state dynamics. Deterministic on purpose — adding RNG
    /// here would break the seed reproducibility of `step`; noise only
    /// enters at inheritance time in `make_mating_child`.
    ///
    /// Update rule: `s' = s + K · (s − 0.5) · dt + bias · n_bonds · dt`.
    /// The `(s − 0.5)` feedback pushes the state away from the unstable
    /// fixed point at 0.5 toward either 0 or 1; the `n_bonds` term
    /// consistently biases highly bonded cells toward the altruist
    /// attractor while solo cells drift toward selfish.
    fn update_cell_state(&mut self, dt: f32) {
        let n_bonds = self.n_bonds() as f32;
        let feedback = CELL_STATE_FEEDBACK_K * (self.cell_state - 0.5) * dt;
        let env_drive = CELL_STATE_BOND_BIAS * n_bonds * dt;
        self.cell_state = (self.cell_state + feedback + env_drive).clamp(0.0, 1.0);
    }

    fn integrate_kinematics(&mut self, dt: f32, world_half: [f32; 3]) {
        self.position[0] += self.velocity[0] * dt;
        self.position[1] += self.velocity[1] * dt;
        self.position[2] += self.velocity[2] * dt;
        self.heading += self.angular_velocity * dt;
        // Gravity only fires when there's a z-volume, and runs before drag
        // so drag can balance it into a terminal velocity.
        if world_half[2] > 0.0 {
            self.velocity[2] -= GRAVITY * dt;
        }
        // Pitch capped at ±π/12 — tight enough that random brain noise only
        // drifts z slightly, while still allowing deliberate dives.
        self.pitch = (self.pitch + self.pitch_velocity * dt).clamp(
            -core::f32::consts::FRAC_PI_6 * 0.5,
            core::f32::consts::FRAC_PI_6 * 0.5,
        );
    }

    fn apply_anisotropic_drag(&mut self, dt: f32, physics: &PhysicsConfig) {
        let [fx, fy, fz] = forward_vector(self.heading, self.pitch);
        let v_par = self.velocity[0] * fx + self.velocity[1] * fy + self.velocity[2] * fz;
        let v_perp_x = self.velocity[0] - v_par * fx;
        let v_perp_y = self.velocity[1] - v_par * fy;
        let v_perp_z = self.velocity[2] - v_par * fz;
        let v_perp_mag =
            (v_perp_x * v_perp_x + v_perp_y * v_perp_y + v_perp_z * v_perp_z).sqrt();
        let length = self.phenotype.body_length;
        let width = self.phenotype.body_width;
        let drag_par_factor = physics.drag * v_par.abs() * width * dt;
        let drag_perp_factor = physics.drag * v_perp_mag * length * dt;
        let new_v_par = v_par - drag_par_factor * v_par;
        let new_v_perp_x = v_perp_x - drag_perp_factor * v_perp_x;
        let new_v_perp_y = v_perp_y - drag_perp_factor * v_perp_y;
        let new_v_perp_z = v_perp_z - drag_perp_factor * v_perp_z;
        self.velocity[0] = new_v_par * fx + new_v_perp_x;
        self.velocity[1] = new_v_par * fy + new_v_perp_y;
        self.velocity[2] = new_v_par * fz + new_v_perp_z;
    }

    /// Yaw and pitch share one drag multiplier so they decay at the same rate.
    fn apply_angular_drag(&mut self, dt: f32, physics: &PhysicsConfig) {
        let drag_factor = (1.0 - physics.angular_drag * dt).max(0.0);
        self.angular_velocity *= drag_factor;
        self.pitch_velocity *= drag_factor;
    }

    /// Per-tick energy drains: v², rotational (yaw only — pitch is free,
    /// otherwise random brains would lose 2× to rotation), vision, body
    /// volume maintenance, spike, attack upkeep, plus a thermal-stress
    /// penalty. Most drains are `metabolism_factor(T)`-scaled so warm cells
    /// burn ~2.46× faster than the cold floor (~0.41×) — niche separation
    /// by depth.
    fn apply_energy_costs(
        &mut self,
        dt: f32,
        world_half: [f32; 3],
        ctx: &ThermalCtx,
        physics: &PhysicsConfig,
        climate_offset: f32,
    ) {
        let temp =
            temperature_at_z_with_ctx(self.position[2], world_half, ctx) + climate_offset;
        let metabolism = metabolism_factor(temp);
        let dt_eff = dt * metabolism;
        let v = self.velocity;
        let v_mag_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        self.energy -= v_mag_sq * physics.energy_cost_per_v_sq * dt_eff;
        let av = self.angular_velocity;
        let eff_r = self.phenotype.effective_radius();
        self.energy -= eff_r * eff_r * av * av * physics.angular_energy_cost * dt_eff;
        // Vision cost scales with fov_factor — narrower cones pay less but
        // lose information once the cone filter activates in sensor gather.
        let fov_factor = vision_fov_factor(self.genome.vision_fov);
        self.energy -=
            self.genome.vision_radius * physics.vision_cost_per_radius * fov_factor * dt_eff;
        // Body maintenance: per-volume, with an aging ramp.
        let age_sec = self.age as f32 / FIXED_TIMESTEP_HZ;
        let aging_factor = 1.0 + AGE_DECAY_PER_SEC * age_sec;
        self.energy -=
            self.phenotype.volume() * physics.body_cost_factor * aging_factor * dt_eff;
        let mut spike_cost = 0.0;
        for spike in self.phenotype.active_spikes() {
            spike_cost += spike.length * spike_complexity_cost_factor(spike.complexity);
        }
        self.energy -= spike_cost * SPIKE_COST_PER_SEC * dt_eff;
        self.energy -= self.phenotype.shell_thickness * SHELL_COST_PER_SEC * dt_eff;
        let attack_strength = self.last_outputs[6].max(0.0);
        self.energy -= attack_strength * ATTACK_COST_PER_SEC * dt_eff;
        // Thermal-stress penalty: quadratic in deviation from the per-cell
        // optimum, intentionally NOT Q10-modulated (uses `dt`, not `dt_eff`)
        // — this models extra stress damage, not a metabolic rate change.
        let dev = (temp - self.genome.thermal_optimum) / 13.0;
        self.energy -= dev * dev * physics.thermal_optimum_penalty * dt;
        let total_sensor_gain: f32 = self.genome.sensor_gains.iter().sum();
        self.energy -= total_sensor_gain * SENSOR_GAIN_COST * dt_eff;
        let active_vib_emit = self.last_outputs[VIBRATION_EMIT_OUTPUT].max(0.0) * MAX_ACTIVE_EMIT;
        self.energy -= active_vib_emit * VIBRATION_EMIT_COST * dt_eff;
    }

    /// XY uses toroidal wrap (cylinder topology), Z reflects. Reflective XY
    /// walls created edge bias (cells accumulating at the boundary, evolution
    /// finding wall-hugging exploits, and smell/pheromone gradients
    /// degenerating at the Neumann boundary); the toroidal wrap removes that.
    /// Z stays bounded — gravity + food sink + carrion drop need a hard
    /// floor/ceiling.
    fn apply_world_bounce(&mut self, world_half: [f32; 3]) {
        let wx = 2.0 * world_half[0];
        let wy = 2.0 * world_half[1];
        // f32 emulation of `rem_euclid`: pos - floor((pos + half) / w) * w.
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

    /// Maze-mode XY boundary: hard walls (clamp + zero outward velocity),
    /// no toroidal wrap. Z behaves as in `apply_world_bounce`. Used by the
    /// world tick when an `ObstacleField` is active so cells cannot wrap
    /// around the maze edge.
    pub fn apply_world_bounce_bounded(&mut self, world_half: [f32; 3]) {
        if self.position[0] > world_half[0] {
            self.position[0] = world_half[0];
            if self.velocity[0] > 0.0 {
                self.velocity[0] = 0.0;
            }
        } else if self.position[0] < -world_half[0] {
            self.position[0] = -world_half[0];
            if self.velocity[0] < 0.0 {
                self.velocity[0] = 0.0;
            }
        }
        if self.position[1] > world_half[1] {
            self.position[1] = world_half[1];
            if self.velocity[1] > 0.0 {
                self.velocity[1] = 0.0;
            }
        } else if self.position[1] < -world_half[1] {
            self.position[1] = -world_half[1];
            if self.velocity[1] < 0.0 {
                self.velocity[1] = 0.0;
            }
        }
        if world_half[2] > 0.0 && self.position[2].abs() > world_half[2] {
            self.velocity[2] = -self.velocity[2];
            self.position[2] = self.position[2].clamp(-world_half[2], world_half[2]);
        }
    }

    /// Resolve overlap with maze obstacles. Pushes the cell back along the
    /// shortest separating axis, kills outward-facing velocity components,
    /// and accrues a small `damage_accum` so the brain perceives the bump
    /// next tick (slot 14). Returns true if a collision was resolved.
    pub fn apply_obstacle_collision(&mut self, field: &ObstacleField) -> bool {
        let radius = self.phenotype.effective_radius().max(crate::CELL_RADIUS * 0.5);
        let push = field.collision_push(self.position, radius);
        if push[0].abs() < 1e-4 && push[1].abs() < 1e-4 {
            return false;
        }
        self.position[0] += push[0];
        self.position[1] += push[1];
        let pmag2 = push[0] * push[0] + push[1] * push[1];
        if pmag2 > 1e-8 {
            let inv = 1.0 / pmag2.sqrt();
            let nx = push[0] * inv;
            let ny = push[1] * inv;
            let v_dot_n = self.velocity[0] * nx + self.velocity[1] * ny;
            if v_dot_n < 0.0 {
                self.velocity[0] -= v_dot_n * nx;
                self.velocity[1] -= v_dot_n * ny;
            }
        }
        self.damage_accum += MAZE_WALL_BUMP_DAMAGE * self.damage_resist_factor();
        true
    }

    /// Apply runtime morph from brain outputs to the phenotype and charge
    /// energy proportional to the realized delta. Called between `brain_act`
    /// and `step`. Output mapping: `[3]=length`, `[4]=width`, `[8]=height`,
    /// `[5]=spike`. The height slot uses `[8]` instead of a lower index so
    /// the original output indices stay stable.
    pub fn apply_morph(&mut self, dt: f32) {
        let morph = [
            self.last_outputs[3],
            self.last_outputs[4],
            self.last_outputs[8],
            self.last_outputs[5],
        ];
        let total_delta = self.phenotype.apply_morph(morph, MORPH_RATE, dt);
        self.energy -= MORPH_COST_PER_DELTA * total_delta;
    }

    /// Apply brain motor outputs to velocity / angular_velocity /
    /// pitch_velocity. Reads `outputs[0] = turn`, `[1] = thrust`,
    /// `[7] = pitch`.
    pub fn apply_brain_motor(&mut self, outputs: &[f32; BRAIN_OUTPUTS], dt: f32) {
        // Sprint 202: F=ma with `mass = volume × MASS_DENSITY`. Pre-S202 used
        // `effective_radius` (linear scaling with size) as a mass proxy after
        // a full-volume attempt killed untrained populations; the new
        // `MASS_DENSITY = 0.1` keeps a typical 3³ cell at mass ≈ 2.7,
        // matching the pre-S202 motor scale at typical size while restoring
        // proper cubic dependence at the extremes.
        let mass = self.phenotype.mass();
        let turn_rate = self.genome.turn_rate;
        let max_speed = self.genome.max_speed;
        let turn_signal = outputs[0];
        let thrust_norm = (outputs[1] + 1.0) * 0.5;
        let pitch_signal = outputs[7];
        let ang_acc = turn_signal * turn_rate / mass;
        self.angular_velocity += ang_acc * dt;
        let pitch_acc = pitch_signal * turn_rate / mass;
        self.pitch_velocity += pitch_acc * dt;
        let a_max = DRAG_COEFFICIENT * max_speed * max_speed / mass;
        let a = thrust_norm * a_max;
        let fwd = forward_vector(self.heading, self.pitch);
        self.velocity[0] += a * fwd[0] * dt;
        self.velocity[1] += a * fwd[1] * dt;
        self.velocity[2] += a * fwd[2] * dt;
    }

    /// Predation bonus for an attacker pointing at least one spike toward
    /// the target (cosine ≥ `SPIKE_DOT_THRESHOLD`). All active spikes
    /// contribute their own cone test.
    pub fn spike_bonus_against(&self, target_pos: [f32; 3]) -> f32 {
        let cache = self.build_spike_attack_cache();
        self.spike_bonus_against_cached(target_pos, &cache)
    }

    /// Pre-rotated spike directions + per-spike gain contributions. Pure
    /// cache derived from `heading`, `pitch`, `phenotype.spikes`, and global
    /// constants — same inputs as `spike_bonus_against`, just hoisted out
    /// of the per-pair loop in the predate hot path.
    pub fn build_spike_attack_cache(&self) -> SpikeAttackCache {
        let mut entries = [SpikeAttackEntry::ZERO; SPIKE_SLOTS];
        let mut count = 0usize;
        for spike in self.phenotype.active_spikes() {
            if spike.length <= 0.0 {
                continue;
            }
            let dir = spike_direction(self.heading, self.pitch, spike);
            let gain = PREDATION_GAIN_PER_TICK
                * spike.length
                * spike_complexity_attack_factor(spike.complexity)
                * SPIKE_PREDATION_BONUS;
            entries[count] = SpikeAttackEntry { dir, gain };
            count += 1;
        }
        SpikeAttackCache { entries, count }
    }

    /// Cached variant — `cache` must come from `build_spike_attack_cache`
    /// on this same cell (heading/pitch must match).
    pub fn spike_bonus_against_cached(
        &self,
        target_pos: [f32; 3],
        cache: &SpikeAttackCache,
    ) -> f32 {
        if cache.count == 0 {
            return 0.0;
        }
        let dx = target_pos[0] - self.position[0];
        let dy = target_pos[1] - self.position[1];
        let dz = target_pos[2] - self.position[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        if dist_sq < f32::EPSILON {
            return 0.0;
        }
        let dist = dist_sq.sqrt();
        let to_target = [dx / dist, dy / dist, dz / dist];
        let mut total = 0.0;
        for entry in &cache.entries[..cache.count] {
            let cos_angle = entry.dir[0] * to_target[0]
                + entry.dir[1] * to_target[1]
                + entry.dir[2] * to_target[2];
            if cos_angle < SPIKE_DOT_THRESHOLD {
                continue;
            }
            total += entry.gain;
        }
        total
    }

    /// Pure ellipsoid acceptance test — no mutation. Semi-axes are
    /// `phenotype × eat_factor`. Binaries pair this with a broad-phase
    /// `max_axis` cull before calling.
    pub fn eat_test(&self, food: &Food, eat_factor: f32) -> bool {
        eat_test_pose(
            self.position,
            self.heading,
            self.pitch,
            [
                self.phenotype.body_length,
                self.phenotype.body_width,
                self.phenotype.body_height,
            ],
            food.position,
            eat_factor,
        )
    }

    /// Extended eat test combining the body ellipsoid with one cone per
    /// active spike. Each cone has its apex at `cell_pos + dir × (eff_r +
    /// length)`, axis along the spike direction, half-angle scaled by
    /// `spike_complexity_grab_factor`, and reach `length ×
    /// SPIKE_GRAB_REACH_BONUS` from the tip. The cell eats the food if the
    /// ellipsoid OR any active spike cone accepts it.
    pub fn eat_test_with_spikes(&self, food: &Food, eat_factor: f32) -> bool {
        if self.eat_test(food, eat_factor) {
            return true;
        }
        let eff_r = self.phenotype.effective_radius();
        for spike in self.phenotype.active_spikes() {
            if spike.length <= 0.0 {
                continue;
            }
            let dir = spike_direction(self.heading, self.pitch, spike);
            let tip_dist = eff_r + spike.length;
            let tip = [
                self.position[0] + dir[0] * tip_dist,
                self.position[1] + dir[1] * tip_dist,
                self.position[2] + dir[2] * tip_dist,
            ];
            let dx = food.position[0] - tip[0];
            let dy = food.position[1] - tip[1];
            let dz = food.position[2] - tip[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let max_range = spike.length * SPIKE_GRAB_REACH_BONUS;
            if dist_sq > max_range * max_range {
                continue;
            }
            if dist_sq < f32::EPSILON {
                return true;
            }
            let dist = dist_sq.sqrt();
            let cos_food = (dx * dir[0] + dy * dir[1] + dz * dir[2]) / dist;
            let half_angle =
                SPIKE_GRAB_HALF_ANGLE * spike_complexity_grab_factor(spike.complexity);
            if cos_food >= half_angle.cos() {
                return true;
            }
        }
        false
    }

    /// Eat the food (charge `food_value`) if the ellipsoid + spike cones
    /// accept it.
    pub fn try_eat(&mut self, food: &Food, eat_factor: f32, food_value: f32) -> bool {
        if self.eat_test_with_spikes(food, eat_factor) {
            self.energy += food_value;
            true
        } else {
            false
        }
    }

    /// Absorb part of `damage_accum` with the shell before the brain reads
    /// the damage signal. Caller invokes this *before* `populate_brain_inputs`
    /// (which reads and resets the accumulator). `shell × ABSORB × dt`,
    /// floored at 0.
    pub fn apply_shell_absorb(&mut self, dt: f32) {
        let absorb = self.phenotype.shell_thickness * SHELL_ABSORB_PER_TICK * dt;
        self.damage_accum = (self.damage_accum - absorb).max(0.0);
    }

    /// Wave 2 episodic novelty: bin `pos` to a `NOVELTY_GRID_CELL_SIZE`-sized
    /// voxel and return its packed index. Out-of-bounds positions clamp into
    /// the grid (rare — cells stay inside world bounds).
    pub fn novelty_voxel_index(pos: [f32; 3], world_half: [f32; 3]) -> u32 {
        let nx = ((2.0 * world_half[0]) / NOVELTY_GRID_CELL_SIZE).ceil() as i32;
        let ny = ((2.0 * world_half[1]) / NOVELTY_GRID_CELL_SIZE).ceil() as i32;
        let xi = (((pos[0] + world_half[0]) / NOVELTY_GRID_CELL_SIZE).floor() as i32)
            .clamp(0, nx - 1);
        let yi = (((pos[1] + world_half[1]) / NOVELTY_GRID_CELL_SIZE).floor() as i32)
            .clamp(0, ny - 1);
        (yi as u32) * (nx as u32) + (xi as u32)
    }

    /// Wave 2: returns `true` and records the voxel if this cell hasn't been
    /// in `voxel` recently (i.e. not in `novelty_history`); returns `false`
    /// (no recording) if the voxel is already in the buffer. The ring
    /// buffer evicts the oldest entry when a novel voxel is recorded.
    pub fn check_novelty(&mut self, voxel: u32) -> bool {
        if voxel == u32::MAX {
            return false;
        }
        for &v in self.novelty_history.iter() {
            if v == voxel {
                return false;
            }
        }
        let head = self.novelty_head as usize;
        self.novelty_history[head] = voxel;
        self.novelty_head = ((head + 1) % NOVELTY_HISTORY_LEN) as u8;
        true
    }

    /// Brownian-motion gaussian noise on velocity. The `√dt` scaling is the
    /// correct stochastic integration (Wiener process), not linear in dt.
    /// The z-axis is perturbed only when the z-volume is active.
    ///
    /// Uses the per-cell `xoshiro_state` so the RNG stream is deterministic
    /// per cell *and identical across CPU and GPU compute paths* (the GPU
    /// shader in `brownian.wgsl` runs the same xoshiro128++ + Box-Muller on
    /// each per-cell state). Consumes two gaussian pairs per call to mirror
    /// the shader: pair 1 → x, y; pair 2 → z (second component discarded
    /// in 2D mode so the stream state stays in lockstep regardless of
    /// `world_half_z`).
    pub fn apply_brownian(&mut self, sqrt_dt: f32, world_half_z: f32) {
        // Sprint 202: brownian kicks scale as 1/√mass (equipartition).
        let inv_sqrt_mass = self.phenotype.mass().sqrt().recip();
        let scale = THERMAL_NOISE * sqrt_dt * inv_sqrt_mass;
        let (gx, gy) = self.xoshiro_state.next_gaussian_pair();
        self.velocity[0] += gx * scale;
        self.velocity[1] += gy * scale;
        if world_half_z > 0.0 {
            let (gz, _drop) = self.xoshiro_state.next_gaussian_pair();
            self.velocity[2] += gz * scale;
        }
    }
}
