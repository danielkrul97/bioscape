//! Headless harness — pure simulation loop, no Bevy renderer.
//!
//! Usage: `cargo run --release --bin headless -- [seed] [max_gens] [out_path]`
//! Defaults: seed=0, max_gens=500, out_path=run_seed{seed}.csv
//!
//! Logs per-generation stats (cell_count, mean/dev for max_speed/vision/body_size,
//! food count, density factor) to CSV. Reproducible: same seed → identical run.

use crate::{
    reject_food_for_richness, Bond, Cell, CoopFood, EventCalendar, Food, Genome, MazeDifficulty, ShockEvent, ShockKind,
    ObstacleField, SimClock, SmellField, SpatialGrid, Symbiont, WorldMap, BOND_BREAK_THRESHOLD,
    BOND_FORMATION_COST, BOND_FORM_THRESHOLD, BOND_FORM_TICKS, BOND_MAINTENANCE_PER_SEC,
    BOND_REST_LENGTH_SLACK, BRAIN_RECURRENT, CARRION_FOOD_COUNT, CELL_RADIUS,
    CONTACT_DECAY_TICKS, COOP_FOOD_MAX_CONCURRENT, COOP_FOOD_SPAWN_RATE_PER_TICK,
    CYCLE_AMPLITUDE, CYCLE_GEN_PERIOD, EAT_RADIUS, FIXED_TIMESTEP_HZ, FOOD_SPAWN_RATE,
    GENERATIONS_PER_EPOCH, GRID_CELL_SIZE, HAZARD_AMP, HAZARD_DRAIN_PER_SEC, HAZARD_FLOOR,
    LEARNING_RATE, MATING_COOLDOWN_TICKS, MATING_PHEROMONE_THRESHOLD, MAX_BONDS_PER_CELL,
    MAX_SPAWN_ATTEMPTS, N_PHEROMONE_CHANNELS, PHEROMONE_BASELINE_EMIT, PHEROMONE_BRAIN_MOD,
    PHEROMONE_COST_PER_RATE, PHEROMONE_DECAY_PER_CH, PHEROMONE_DIFFUSION_PER_CH,
    PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z, PHEROMONE_SAMPLE_EPSILON, PHYSICS_CONFIG,
    SMELL_DECAY, SMELL_DIFFUSION, SMELL_GRID_RES, SMELL_GRID_RES_Z,
    SMELL_PER_FOOD, SMELL_SAMPLE_EPSILON, SYMBIONT_CAPTURE_P, SYMBIONT_INIT_FRACTION,
    SYMBIONT_PHOTO_RATE, SYMBIONT_PHOTO_Z_THRESHOLD, SYMBIONT_UPKEEP_DEFICIT_TICKS,
    SYMBIONT_UPKEEP_PER_SEC, TICKS_PER_GENERATION, VIBRATION_DECAY,
    VIBRATION_DIFFUSION, VIBRATION_GRID_RES, VIBRATION_GRID_RES_Z, VIBRATION_SAMPLE_EPSILON,
    WORLD_HALF, WORLD_MAP_BASE_RES, WORLD_MAP_BASE_RES_Z, WORLD_MAP_FOOD_AMP,
    WORLD_MAP_FOOD_FLOOR, WORLD_MAP_RES, WORLD_MAP_RES_Z, WORLD_UNITS_PER_FOOD,
};
use crate::{BRAIN_HIDDEN, BRAIN_INPUTS};
use crate::{
    gpu::{
        BrainGpu, BrownianGpu, CellsGpu, CollisionGpu, CppnGpu, EatFoodGpu, EatFoodParamsGpu,
        FieldGpu, FoodSpawnGpu, FoodSpawnParamsGpu, GpuContext, GpuFullScratch, GpuProfiler,
        HebbianGpu,
        MotorGpu, PopulateInputsGpu, PopulateInputsParams, PredateGpu, PredateParamsGpu,
        ExcitabilityGpu, IzhikevichGpu, SensorGatherGpu, SensorParamsGpu, SpatialHashGpu,
        StdpApplyGpu, StdpEncodePreGpu, StdpStepGpu, StepGpu, StepParamsGpu, SynapticScaleGpu,
    },
    AGE_DECAY_PER_SEC, ATTACK_COST_PER_SEC, BRAIN_INPUTS_SENSORY,
    BRAIN_OUTPUTS, CARRION_DECAY_PER_SEC, CARRION_FOOD_VALUE, DAMAGE_NORMALIZATION_GAIN,
    DENSITY_NORM_COUNT, DRAG_COEFFICIENT, ESCAPE_COOLDOWN_TICKS,
    ESCAPE_STREAK_THRESHOLD, FlushMode, GRAVITY as PHYS_GRAVITY,
    PHEROMONE_NORMALIZATION_GAIN, PLANT_FOOD_VALUE,
    PREDATION_REWARD_MAX, RewardAccumulator, RewardKind,
    ACTIVITY_EMA_ALPHA, DEFAULT_STDP_A_MINUS, DEFAULT_STDP_A_PLUS,
    EXCITABILITY_DRIFT_PER_TICK, GRAM_SCHMIDT_STRENGTH, GS_PERIOD_TICKS, SCALING_PERIOD_TICKS,
    SCARCITY_FLOOR, SCARCITY_RAMP_END_GEN, SHELL_COST_PER_SEC,
    LATERAL_INHIBITION_ALPHA, SMELL_NORMALIZATION_GAIN, SPIKE_COST_PER_SEC, THERMAL_NOISE,
    VELOCITY_CAP_FACTOR, WEIGHT_DECAY_PER_TICK, W_NORM_CAP, W_NORM_FLOOR,
};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

/// Sprint 48: versioned binary header pro checkpoint files.
/// Sprint 66 bump: Cell rozšířen o `cell_id` + `bonds`, BRAIN_OUTPUTS 9→10.
/// Sprint 68 bump: Genome rozšířen o `bond_stiffness` + `bond_damping`,
/// Bond rozšířen o `stiffness` + `damping`. Starší V1/V2 už nelze
/// deserializovat — load vrací error.
/// Sprint 103 bump: BRAIN_HIDDEN 32→50, BRAIN_INPUTS 53→71. Brain weight
/// arrays resize → V3 savefiles incompatible.
/// Sprint 126 bump: multi-channel pheromones (3 fields), BRAIN_INPUTS_SENSORY
/// 21→27, BRAIN_INPUTS 71→77, BRAIN_OUTPUTS 10→12. V4 savefiles incompatible.
const CHECKPOINT_MAGIC: &[u8; 8] = b"BIOSCP01";
/// V6: Cell.last_best_food_d2 added. `serde(default)` zajistí backward-compat
/// při deserializaci V5 dat (default = f32::MAX → eat_food skip vždy v 1. ticku
/// po loadu, neutral, brain_act nastaví v 1. ticku).
/// V7: vibration sensing — new `vibration` SmellField, BRAIN_INPUTS_SENSORY
/// 29→33, BRAIN_INPUTS 74→78, `Genome.sensor_gains` 3→4 (Mechano added). V6
/// savefiles incompatible (brain weight matrices resize).
/// V8: per-cell `xoshiro_state` added to `Cell` for unified CPU/GPU brownian
/// RNG stream. V7 saves would deserialize via `serde(default)` to identical
/// sentinel state across cells — that would make every cell move in noise
/// lockstep, so we force-fail the version mismatch instead.
const CHECKPOINT_VERSION: u32 = 8;

/// Sprint 48: serializovatelný snapshot sim state. Skip fields:
/// - SpatialGrid (rebuild from cells/foods on load)
/// - bench_timings (per-tick diagnostic, ne state)
/// - GPU subsystémy (re-init on load)
/// - scratch Vecs (re-alloc lazily)
/// RNG state se NEUKLÁDÁ — load resetuje RNG ze --seed argument. Pro full
/// reproducibility add chacha state serializace v pozdějším sprintu.
#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    pub version: u32,
    pub cells: Vec<Cell>,
    pub foods: Vec<Food>,
    pub clock: SimClock,
    pub density_factor: f32,
    pub smell: SmellField,
    /// Sprint 126: multi-channel pheromone fields. Backward-compat: starší
    /// V4 checkpointy s `pheromone: SmellField` (ch0 only) už nelze
    /// deserializovat — version bump.
    pub pheromone_fields: Vec<SmellField>,
    /// V7: motion-driven mechanosensory field. Same 3D scalar SmellField
    /// type, different decay/diffusion constants. Persisted because the
    /// field carries state across the generation boundary (it does not
    /// reset).
    pub vibration: SmellField,
    pub map: WorldMap,
    pub births_gen: u64,
    pub deaths_gen: u64,
    pub fertile_ticks_gen: u64,
    pub predation_events_gen: u64,
    pub mating_radius: f32,
    pub max_population: usize,
}

/// Sprint 43: per-fáze accumulator (mikrosekundy). World::tick zvyšuje každou
/// dobu a main je čte/resetuje per generation. Default je all-zero.
#[derive(Debug, Default, Clone, Copy)]
pub struct PhaseTimings {
    pub update_smell: f64,
    pub update_pheromone: f64,
    pub update_vibration: f64,
    pub brain_act: f64,
    pub emit_pheromones: f64,
    pub apply_morph: f64,
    pub apply_brownian: f64,
    pub step: f64,
    pub apply_food_gravity: f64,
    pub apply_hazards: f64,
    pub resolve_collisions: f64,
    pub predate: f64,
    pub eat_food: f64,
    pub spawn_food: f64,
    pub reproduce: f64,
    pub die_and_drop_carrion: f64,
}

pub struct World {
    pub cells: Vec<Cell>,
    pub foods: Vec<Food>,
    /// Sprint 128: cooperative food nodes. Lifecycle waiting → triggered/expired
    /// (separate od regular `Food` — odlišný spawn rate, žádný eat-by-cell,
    /// reward distribuovaný up-front při dosažení threshold).
    pub coop_foods: Vec<CoopFood>,
    /// Sprint 128: per-gen counters pro CSV diagnostiku. Reset v end-of-gen
    /// stejně jako `bonds_formed_gen`.
    pub coop_food_solved_gen: u64,
    pub coop_food_failed_gen: u64,
    /// Sprint 128: suma arrivals.len() přes všechny coop nodes ke konci jejich
    /// lifecyklu (trigger nebo expire) — dělená total events daň mean per gen.
    pub coop_food_arrivals_sum_gen: u64,
    pub coop_food_events_gen: u64,
    pub clock: SimClock,
    pub density_factor: f32,
    pub smell: SmellField,
    /// Sprint 126: multi-channel pheromone fields (3 nezávislých polí).
    /// ch0 = slow (mating-friendly, decay 0.3), ch1 medium (1.5),
    /// ch2 fast (5.0, bursty). GPU path používá pouze ch0 ve `gpu.pheromone`,
    /// ch1/ch2 vždy CPU step.
    pub pheromone_fields: [SmellField; N_PHEROMONE_CHANNELS],
    /// V7: motion-driven mechanosensory field. Each cell deposits an
    /// amplitude proportional to its kinetic + rotational activity (see
    /// `crate::vibration_emit_for_cell`); the field then diffuses + decays
    /// every tick. Brain reads gradient + amplitude as inputs [29..32]. CPU
    /// only — no GPU shader counterpart in this cut.
    pub vibration: SmellField,
    pub map: WorldMap,
    // Sprint 43: spatial hashes pro broad-phase. Rebuild před fází, která
    // neighbors používá — `cell_grid` před brain_act/resolve_collisions/predate,
    // `food_grid` před brain_act/eat_food.
    pub cell_grid: SpatialGrid<usize, f32>,
    pub food_grid: SpatialGrid<usize, crate::FoodKind>,
    // Persistent scratch — reused per tick to avoid hot-loop allocations.
    pub deltas_scratch: Vec<[f32; 3]>,
    /// Sprint 65: collision velocity damping (inelastic) — per pair, closing
    /// velocity podél separation normal je halved. Cell i sees pair (i, j),
    /// computes own delta; symmetric (Newton 3rd law) když j visits i.
    pub velocity_deltas_scratch: Vec<[f32; 3]>,
    pub eaten_scratch: Vec<bool>,
    /// Persistent cell_id → idx scratch. Built once at the start of each tick
    /// (`rebuild_id_to_idx`) and consumed by pool_bonded_hidden, brain_act,
    /// resolve_collisions, eat_food. Avoids fresh FxHashMap allocations per tick.
    pub id_to_idx_scratch: rustc_hash::FxHashMap<u64, usize>,
    /// Persistent contact-list scratch — outer index = cell idx, inner Vec
    /// drží `cell_id_j` (>i, dedupe). Pre-fix: `Vec<Vec<u64>>` collected per
    /// tick → 1+N alloc/tick. Persistent reuse zachová capacity inner Vecs.
    pub contact_lists_scratch: Vec<Vec<u64>>,
    /// Phase 2 scratch — pozice po Phase-1 apply. Pre-fix: fresh
    /// `Vec<[f32;3]>` per tick.
    pub positions_snapshot_scratch: Vec<[f32; 3]>,
    /// Phase 2 scratch — set párů viděných v aktuálním ticku. Pre-fix: fresh
    /// FxHashSet per tick.
    pub seen_pairs_scratch: rustc_hash::FxHashSet<(u64, u64)>,
    /// Phase 2 scratch — bond-formation candidate pairs. Pre-fix: fresh Vec
    /// collected per tick.
    pub bond_candidates_scratch: Vec<(u64, u64)>,
    /// Phase 2 scratch — set of canonicalized (min_id, max_id) párů, které
    /// jsou už bonded. O(1) lookup namísto lineárního scanu Cell.bonds per
    /// kandidáta.
    pub bonded_pairs_scratch: rustc_hash::FxHashSet<(u64, u64)>,
    /// Scratch for `die_and_drop_carrion` Phase 3 — surviving cell_ids,
    /// used to prune bonds whose partner was just swap-removed. Rebuilt
    /// only on ticks where a death occurred.
    pub survivor_ids_scratch: rustc_hash::FxHashSet<u64>,
    pub hidden_snapshot_scratch: Vec<[f32; BRAIN_HIDDEN]>,
    pub inputs_scratch: Vec<[f32; BRAIN_INPUTS]>,
    pub births_gen: u64,
    pub deaths_gen: u64,
    pub fertile_ticks_gen: u64,
    pub predation_events_gen: u64,
    pub sym_sheds_gen: u64,
    /// Sprint 66: monotonic counter pro stable Cell.cell_id přidělování.
    /// Initial population uses 0..INITIAL_CELLS, takže start = INITIAL_CELLS.
    pub next_cell_id: u64,
    /// Sprint 196: independent monotonic counter for `Symbiont.lineage_id`.
    /// Separate from `next_cell_id` so host and symbiont lineages live in
    /// disjoint ID spaces — a symbiont's identity is decoupled from any
    /// specific host (predation transfer preserves the lineage_id).
    pub next_symbiont_lineage_id: u64,
    /// Sprint 66: per-pair (min_id, max_id) → consecutive contact ticks.
    /// Vstupy se přidávají v `resolve_collisions` Phase 2 (sequential merge),
    /// odebírají při decay timeout. Sparse — pouze dvojice s aktuálním kontaktem.
    pub contact_progress: rustc_hash::FxHashMap<(u64, u64), u32>,
    /// Sprint 66 diagnostic counter — počet bondů vytvořených v aktuální generaci.
    /// Zatím log-only, future může být CSV column.
    pub bonds_formed_gen: u64,
    /// Sprint 66 diagnostic — počet bondů přervaných v aktuální generaci.
    pub bonds_broken_gen: u64,
    // Pack-hunting diagnostics. Per-gen totals — main.rs resetuje na gen end.
    // `bonded_*` / `solo_*` rozlišuje útoky podle attacker.n_bonds() ≥ 1
    // (selekční signál: bonded vs solo per-attack gain efficiency).
    // `swarm_attacks_gen` = oběti zasažené ≥2 distinct attackery v 1 ticku;
    // `pack_attacks_gen` = z toho subset, kde aspoň jedna dvojice attackerů
    // je vzájemně bonded (behavioral signal: bondovaná koordinace, ne
    // náhodný cluster overlap). `attack_victims_gen` = celkový počet
    // distinct (victim, tick) párů — denominator pro swarm/pack fraction.
    pub bonded_attacks_gen: u64,
    pub solo_attacks_gen: u64,
    pub bonded_attack_gain_sum_gen: f64,
    pub solo_attack_gain_sum_gen: f64,
    pub swarm_attacks_gen: u64,
    pub pack_attacks_gen: u64,
    pub attack_victims_gen: u64,
    pub mating_radius: f32,
    // Sprint 43: runtime override `MAX_POPULATION` consts. Default = const, CLI
    // může nastavit výš (potřeba pro bench při N > 1000).
    pub max_population: usize,
    /// Sprint 109: deterministicky vygenerovaný kalendář environmentálních
    /// shocků. Default empty (no-op) — sim loop ho zatím nečte; integrace
    /// efektů přijde v Sprintu 110+.
    pub events: EventCalendar,
    /// Active `HazardPulse` events at the current generation. Refreshed
    /// once per tick by `refresh_active_events`. Without this scratch,
    /// `hazard_shock_multiplier` would scan all ~50k calendar events per
    /// cell per tick (200 cells × 60 Hz × 50k = 600M iter/s).
    pub active_hazard_events: Vec<ShockEvent>,
    /// Active `ClimateShift` events at the current generation. Same
    /// rationale as `active_hazard_events`.
    pub active_climate_events: Vec<ShockEvent>,
    /// Cursor into `events.events` — the first index that hasn't yet
    /// fully ended. Monotonically non-decreasing across ticks; advanced
    /// in `refresh_active_events` past events whose
    /// `start_gen + duration_gen <= generation`. Reset to 0 only at
    /// `World` construction (fresh ctor or checkpoint load — both rebuild
    /// the whole struct).
    pub first_live_event_idx: usize,
    // Sprint 87 Hamilton-rule sweep: runtime overrides pro food share. Default
    // = pre-sweep behavior (BOND_FOOD_SHARE_FRAC, no kin filter).
    pub share_frac: f32,
    pub kin_filter: bool,
    // Runtime multiplikátory pro intra-species predation experiment sweep.
    // Default 1.0 = baseline behavior. Násobí se na site of use.
    pub predation_gain_mult: f32,
    pub predation_drain_mult: f32,
    pub food_factor_mult: f32,
    pub bench_timings: PhaseTimings,
    /// Static voxel maze. `Some` when started with `--maze[=easy|medium|hard]`.
    /// When set, the world bounds become hard XY walls (no toroidal wrap),
    /// cells push out of occupied voxels, smell/pheromone/vibration diffuse
    /// with Neumann boundaries on walls, vision raycast checks LOS, and
    /// per-gen navigation metrics (goal-zone visits, time-to-goal) are
    /// recorded.
    pub obstacles: Option<ObstacleField>,
    /// Per-grid Neumann masks derived from `obstacles` at allocation time.
    /// All `None` when `obstacles` is `None` — fast path stays byte-identical
    /// to pre-maze behavior.
    pub smell_mask: Option<Vec<bool>>,
    pub pheromone_masks: [Option<Vec<bool>>; N_PHEROMONE_CHANNELS],
    pub vibration_mask: Option<Vec<bool>>,
    /// Wave 3 curriculum: ramp schedule. Each entry `(MazeDifficulty,
    /// end_gen)` means "use this difficulty up through `end_gen` (exclusive
    /// upper bound)". The last entry's `end_gen` is `u64::MAX` (run forever
    /// at the final difficulty). Empty when `--maze-stages` not passed —
    /// the world stays on the single difficulty given by `--maze`.
    pub maze_curriculum: Vec<(MazeDifficulty, u64)>,
    /// Wave 3: monotonic seed offset bumped each curriculum rebuild so
    /// successive maze topologies for the same `map_seed` differ.
    pub maze_seed_step: u64,
    /// Per-gen sum of (cells in goal zone) accumulated each tick. Divide by
    /// (`TICKS_PER_GENERATION × cells.len()`) for mean fraction-time-at-goal.
    pub goal_zone_ticks_gen: u64,
    /// `cell_id`s of cells that touched the goal zone in the current
    /// generation. Cleared at gen end. `len() / cells.len()` = unique
    /// reachers fraction this gen.
    pub goal_unique_reachers_gen: rustc_hash::FxHashSet<u64>,
    /// `cell_id` → tick of first goal entry. Lifelong record (never reset),
    /// used for the time-to-goal histogram. New entries appear only when a
    /// cell first touches the goal.
    pub goal_first_reach_tick: rustc_hash::FxHashMap<u64, u64>,
    // Sprint 44: pokud `Some`, brain_act offloaduje forward pass na GPU.
    // Sensor gather + populate_brain_inputs + apply_brain_motor zůstává CPU.
    pub gpu: Option<BrainGpu>,
    // Sprint 51: full-GPU brain pipeline. Když Some, drží brain weights
    // persistent na GPU mezi ticky (eliminuje 30 MB/tick upload Sprintu 44),
    // GPU Hebbian replace CPU brain.hebbian_update, GPU Brownian replace
    // CPU apply_brownian. Sensor/motor/step/collision/predate zůstávají CPU
    // rayon (Sprint 50 standalone shadery jsou ready, integrace je Sprint 52+).
    pub gpu_full: Option<GpuFullState>,
}

// `GpuFullScratch` přesunut do `crate::gpu::scratch` (lib) — sdílen mezi
// headless `--gpu-full` pathem a renderer `BIOSCAPE_GPU_FULL=1` pathem.

pub struct GpuFullState {
    pub cells: CellsGpu,
    pub brain: BrainGpu,
    pub hebbian: HebbianGpu,
    pub brownian: BrownianGpu,
    /// Wave H: GPU collision broad-phase (position depenetration + velocity
    /// damping + soft adhesion + spring-bond forces + contact events).
    /// CPU side keeps Phase 2 (apply deltas), Phase 3 (bond pruning) and
    /// Phase 4 (contact_progress / bond formation) — sparse / variable-
    /// allocation work that does not fit the GPU pair loop.
    pub collision: CollisionGpu,
    /// Wave H: GPU predation (herd count + per-pair attack with atomic
    /// energy/damage accumulation). Pack-hunting CSV diagnostics
    /// (`bonded_attacks_gen` etc.) stay zero on this path — extending the
    /// shader to emit per-event tuples is follow-up work.
    pub predate: PredateGpu,
    /// Wave J: GPU food spawn rejection sampling. K-attempts per dispatch;
    /// CPU consumes valid candidates up to the per-tick budget. CPU still
    /// owns `World::foods: Vec<Food>` (variable allocation = control
    /// plane); GPU just does the rejection work.
    pub food_spawn: FoodSpawnGpu,
    /// GPU eat_food candidate selection. Mirrors `World::eat_food` Pass 1:
    /// per cell, walk `food_hash` buckets within `EAT_RADIUS × max_axis`
    /// and return the first food whose center lies inside the pose-rotated
    /// ellipsoid. CPU does Pass 2 (first-cell-wins race + cluster share +
    /// Hebbian reward) on the returned `(food_idx, value)` arrays. Cuts
    /// the largest CPU hotspot (`SpatialGrid::for_each_in_radius`, ~27 %
    /// of total CPU pre-fix).
    pub eat_food: EatFoodGpu,
    /// Sprint 59: GPU smell + pheromone field (3D 7-point Jacobi).
    /// Sprint 60: po wire SensorGatherGpu už NEČTE CPU SmellField shadow —
    /// sensor shader bere field grid storage buffer direct. Per-tick readback
    /// eliminován; CPU `World.smell` / `pheromone` zůstávají jen pro
    /// checkpoint serialization (po `--gpu-full` jsou CPU shadows out-of-date).
    pub smell: FieldGpu,
    pub pheromone: FieldGpu,
    /// Wave L: per-channel pheromone fields. ch0 = `pheromone` (above);
    /// ch1 and ch2 here. All three step independently with the per-channel
    /// `PHEROMONE_DIFFUSION_PER_CH` / `PHEROMONE_DECAY_PER_CH` constants
    /// and feed sensor_gather binding 16 / 17 for brain-input gradients.
    pub pheromone_ch1: FieldGpu,
    pub pheromone_ch2: FieldGpu,
    /// V7: motion-driven mechanosensory field on GPU. Deposit + step inline
    /// each tick; sensor_gather shader binds the current grid buffer at
    /// binding 12. CPU `World.vibration` shadow is not synced on the
    /// `--gpu-full` path (kept only for checkpoint round-trip).
    pub vibration: FieldGpu,
    /// Sprint 60: GPU spatial hashes pro sensor broad-phase. Per-tick
    /// `dispatch()` (no readback) + sensor shader čte `offsets_buffer()` /
    /// `sorted_buffer()` přes binding group.
    pub cell_hash: SpatialHashGpu,
    pub food_hash: SpatialHashGpu,
    pub sensor: SensorGatherGpu,
    /// Sprint 61: GPU populate_brain_inputs shader fuze sensor output + cell
    /// metadata → brain inputs buffer. Eliminuje sensor 60 KB readback round-trip.
    pub populate: PopulateInputsGpu,
    /// Sprint 62: motor on GPU (apply brain outputs → velocity/ang_vel/pitch_vel).
    /// Fused s brownian dispatch v brain_act → single batch readback eliminuje
    /// round-trip #2 (hidden+outputs) i round-trip #3 (velocities).
    pub motor: MotorGpu,
    /// Sprint 63: step on GPU (kinematics + drag + energy + bounce).
    /// Fused do brain_act batch readback. Skip CPU `step` fáze v `--gpu-full`.
    pub step: StepGpu,
    /// CPPN substrate query on GPU — dispatched per reproduce phase to
    /// materialise child brain weights directly into `cells.brain_weights_buf`,
    /// skipping per-child CPU `Brain::from_cppn`.
    pub cppn: CppnGpu,
    /// Sprint 147: per-tick Izhikevich forward (overlay on top of the
    /// shared `BrainGpu` output — Perceptron cells are filtered out inside
    /// the shader). Runs immediately after `BrainGpu::forward_persistent`.
    pub izhikevich: IzhikevichGpu,
    /// Sprint 165-S167: STDP go-live pipeline. Pre-spike encoder marks
    /// `pre_spike_times` from sensory inputs; `stdp_step` decays + bumps
    /// per-neuron traces; `stdp_apply` mutates `w1` on reward events.
    pub stdp_encode_pre: StdpEncodePreGpu,
    pub stdp_step: StdpStepGpu,
    pub stdp_apply: StdpApplyGpu,
    /// Sprint 138: homeostatic synaptic scaling. Dispatched every
    /// `SCALING_PERIOD_TICKS` against `cells.brain_weights_buf` so runaway
    /// Hebbian growth doesn't saturate tanh and lock the brain into a
    /// single action over a long lineage.
    pub synaptic_scale: SynapticScaleGpu,
    /// Sprint 139: intrinsic excitability regulator. Per-tick linear
    /// drift of `b1[h]` toward `-DRIFT × activity_avg[h]`, where
    /// `activity_avg` is a per-cell, per-neuron signed EMA of `last_hidden`.
    /// Pulls saturated neurons back into the responsive tanh zone without
    /// disturbing the Hebbian weight matrix.
    pub excitability: ExcitabilityGpu,
    /// Persistent CPU snapshots — reused per tick, zachovává kapacitu.
    pub scratch: GpuFullScratch,
    /// Serialize-and-time per-shader GPU profiler. Disabled unless
    /// `init_gpu_full_profiled(true)` — then every method is a no-op.
    pub profiler: GpuProfiler,
}

impl World {
    #[allow(dead_code)]
    pub fn new(
        rng: &mut impl Rng,
        map_seed: u64,
        mating_radius: f32,
        initial_cells: usize,
        max_population: usize,
        events: EventCalendar,
    ) -> Self {
        Self::new_with_maze(
            rng,
            map_seed,
            mating_radius,
            initial_cells,
            max_population,
            events,
            None,
        )
    }

    pub fn new_with_maze(
        rng: &mut impl Rng,
        map_seed: u64,
        mating_radius: f32,
        initial_cells: usize,
        max_population: usize,
        events: EventCalendar,
        maze_difficulty: Option<MazeDifficulty>,
    ) -> Self {
        // Sprint 53: WorldMap a SmellField/Pheromone jsou plně 3D volumetric.
        // Food richness sampling používá z=0 (canonical surface depth) aby
        // food clustery zůstaly xy-stratifikované (consistent s pre-Sprint-53
        // biome semantikou); hazards samplují full 3D pozici.
        let map = WorldMap::new(
            [WORLD_MAP_RES, WORLD_MAP_RES, WORLD_MAP_RES_Z],
            [WORLD_MAP_BASE_RES, WORLD_MAP_BASE_RES, WORLD_MAP_BASE_RES_Z],
            WORLD_HALF,
            map_seed,
        );
        // Maze obstacles seeded from `map_seed` so the same `--seed N`
        // reproduces the same maze across runs.
        let obstacles =
            maze_difficulty.map(|d| ObstacleField::new_maze(WORLD_HALF, map_seed, d));
        let smell_mask = obstacles
            .as_ref()
            .map(|o| o.mask_for_grid([SMELL_GRID_RES, SMELL_GRID_RES, SMELL_GRID_RES_Z]));
        let pheromone_masks: [Option<Vec<bool>>; N_PHEROMONE_CHANNELS] =
            std::array::from_fn(|_| {
                obstacles.as_ref().map(|o| {
                    o.mask_for_grid([
                        PHEROMONE_GRID_RES,
                        PHEROMONE_GRID_RES,
                        PHEROMONE_GRID_RES_Z,
                    ])
                })
            });
        let vibration_mask = obstacles.as_ref().map(|o| {
            o.mask_for_grid([
                VIBRATION_GRID_RES,
                VIBRATION_GRID_RES,
                VIBRATION_GRID_RES_Z,
            ])
        });
        // Initial cells: in maze mode reject spawn positions inside walls
        // (rejection sampling against the obstacle field). Non-maze path
        // unchanged.
        let mut cells: Vec<Cell> = (0..initial_cells)
            .map(|i| {
                if let Some(field) = obstacles.as_ref() {
                    for _ in 0..MAX_SPAWN_ATTEMPTS {
                        let c = Cell::random(rng, WORLD_HALF, i as u64, 0, i as u64);
                        if !field.sample(c.position) {
                            return c;
                        }
                    }
                    Cell::random(rng, WORLD_HALF, i as u64, 0, i as u64)
                } else {
                    Cell::random(rng, WORLD_HALF, i as u64, 0, i as u64)
                }
            })
            .collect();
        // Sprint 196: seed the bearer pool. With probability SYMBIONT_INIT_FRACTION
        // per cell, attach a freshly random Symbiont with its own independent
        // lineage_id. Without this seed pool the predation-derived origin
        // pathway has nothing to copy from. Runs after the cells collect so the
        // host RNG sequence stays untouched until the symbiont draws begin.
        let mut next_symbiont_lineage_id: u64 = 0;
        for cell in cells.iter_mut() {
            if rng.random::<f32>() < SYMBIONT_INIT_FRACTION {
                let sym_genome = Genome::random(rng);
                cell.symbiont = Some(Symbiont::new(sym_genome, next_symbiont_lineage_id));
                next_symbiont_lineage_id += 1;
            }
        }
        let target = food_target(1.0);
        let foods = (0..target)
            .map(|_| {
                for _ in 0..MAX_SPAWN_ATTEMPTS {
                    let candidate = Food::random(rng, WORLD_HALF);
                    let richness = map.sample([candidate.position[0], candidate.position[1], 0.0]);
                    if reject_food_for_richness(rng, richness) {
                        continue;
                    }
                    if let Some(field) = obstacles.as_ref() {
                        if field.sample(candidate.position) {
                            continue;
                        }
                    }
                    return candidate;
                }
                Food::random(rng, WORLD_HALF)
            })
            .collect();
        Self {
            cells,
            foods,
            coop_foods: Vec::new(),
            coop_food_solved_gen: 0,
            coop_food_failed_gen: 0,
            coop_food_arrivals_sum_gen: 0,
            coop_food_events_gen: 0,
            clock: SimClock::new(TICKS_PER_GENERATION, GENERATIONS_PER_EPOCH),
            density_factor: 1.0,
            smell: SmellField::new([SMELL_GRID_RES, SMELL_GRID_RES, SMELL_GRID_RES_Z], WORLD_HALF),
            pheromone_fields: std::array::from_fn(|_| {
                SmellField::new(
                    [PHEROMONE_GRID_RES, PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z],
                    WORLD_HALF,
                )
            }),
            vibration: SmellField::new(
                [VIBRATION_GRID_RES, VIBRATION_GRID_RES, VIBRATION_GRID_RES_Z],
                WORLD_HALF,
            ),
            map,
            cell_grid: SpatialGrid::new(GRID_CELL_SIZE, WORLD_HALF),
            food_grid: SpatialGrid::new(GRID_CELL_SIZE, WORLD_HALF),
            deltas_scratch: Vec::new(),
            velocity_deltas_scratch: Vec::new(),
            eaten_scratch: Vec::new(),
            id_to_idx_scratch: rustc_hash::FxHashMap::default(),
            contact_lists_scratch: Vec::new(),
            positions_snapshot_scratch: Vec::new(),
            seen_pairs_scratch: rustc_hash::FxHashSet::default(),
            bond_candidates_scratch: Vec::new(),
            bonded_pairs_scratch: rustc_hash::FxHashSet::default(),
            survivor_ids_scratch: rustc_hash::FxHashSet::default(),
            hidden_snapshot_scratch: Vec::new(),
            inputs_scratch: Vec::new(),
            births_gen: 0,
            deaths_gen: 0,
            fertile_ticks_gen: 0,
            predation_events_gen: 0,
            sym_sheds_gen: 0,
            next_cell_id: initial_cells as u64,
            next_symbiont_lineage_id,
            contact_progress: rustc_hash::FxHashMap::default(),
            bonds_formed_gen: 0,
            bonds_broken_gen: 0,
            bonded_attacks_gen: 0,
            solo_attacks_gen: 0,
            bonded_attack_gain_sum_gen: 0.0,
            solo_attack_gain_sum_gen: 0.0,
            swarm_attacks_gen: 0,
            pack_attacks_gen: 0,
            attack_victims_gen: 0,
            mating_radius,
            max_population,
            events,
            active_hazard_events: Vec::new(),
            active_climate_events: Vec::new(),
            first_live_event_idx: 0,
            share_frac: crate::BOND_FOOD_SHARE_FRAC,
            kin_filter: false,
            predation_gain_mult: 1.0,
            predation_drain_mult: 1.0,
            food_factor_mult: 1.0,
            bench_timings: PhaseTimings::default(),
            obstacles,
            smell_mask,
            pheromone_masks,
            vibration_mask,
            maze_curriculum: Vec::new(),
            maze_seed_step: 0,
            goal_zone_ticks_gen: 0,
            goal_unique_reachers_gen: rustc_hash::FxHashSet::default(),
            goal_first_reach_tick: rustc_hash::FxHashMap::default(),
            gpu: None,
            gpu_full: None,
        }
    }

    /// Allocate and wire the full GPU pipeline that the post-Wave-N tick
    /// hard-depends on. Mirrors `main.rs`'s GPU init: builds a shared
    /// `GpuContext`, allocates every per-subsystem pipeline at a capacity
    /// covering the initial population, max population, and a 64-cell
    /// floor, uploads the starting brain weights / xoshiro seeds / turn
    /// rates, and finally uploads maze occupancy masks if obstacles are
    /// already present.
    ///
    /// Returns the wgpu adapter error verbatim on failure so the test
    /// helper that calls this can panic with a meaningful message. Tests
    /// expect a GPU to be present; if you want a CPU-only path you'd have
    /// to revive the legacy code that Wave N deleted.
    pub fn init_gpu_full(&mut self) -> Result<(), String> {
        self.init_gpu_full_profiled(false)
    }

    /// Same as `init_gpu_full`, but `gpu_profile = true` arms the
    /// serialize-and-time per-shader `GpuProfiler` on the resulting
    /// `GpuFullState`. Off, the profiler is inert.
    pub fn init_gpu_full_profiled(&mut self, gpu_profile: bool) -> Result<(), String> {
        let cap = self.cells.len().max(self.max_population).max(64);
        let field_sources_cap =
            (food_target(1.0 + CYCLE_AMPLITUDE) + self.max_population) * 2;
        let ctx = GpuContext::new()?;
        let cells_gpu = CellsGpu::with_context(&ctx, cap);
        cells_gpu.upload_brains(self.cells.iter().map(|c| &c.genome.brain));
        cells_gpu.upload_xoshiro_seeds(self.cells.iter().map(|c| c.cell_id));
        let brain = BrainGpu::with_context(&ctx, cap)?;
        let hebbian = HebbianGpu::with_context(&ctx, cap)?;
        let brownian = BrownianGpu::with_context(&ctx, cap)?;
        let smell = FieldGpu::with_context(
            &ctx,
            [SMELL_GRID_RES, SMELL_GRID_RES, SMELL_GRID_RES_Z],
            WORLD_HALF,
            field_sources_cap,
        )?;
        let pheromone = FieldGpu::with_context(
            &ctx,
            [PHEROMONE_GRID_RES, PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z],
            WORLD_HALF,
            field_sources_cap,
        )?;
        let pheromone_ch1 = FieldGpu::with_context(
            &ctx,
            [PHEROMONE_GRID_RES, PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z],
            WORLD_HALF,
            field_sources_cap,
        )?;
        let pheromone_ch2 = FieldGpu::with_context(
            &ctx,
            [PHEROMONE_GRID_RES, PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z],
            WORLD_HALF,
            field_sources_cap,
        )?;
        let vibration = FieldGpu::with_context(
            &ctx,
            [VIBRATION_GRID_RES, VIBRATION_GRID_RES, VIBRATION_GRID_RES_Z],
            WORLD_HALF,
            cap,
        )?;
        let cell_hash = SpatialHashGpu::with_context(
            &ctx,
            cap,
            GRID_CELL_SIZE,
            [WORLD_HALF[0], WORLD_HALF[1]],
        )?;
        let food_capacity = field_sources_cap;
        let food_hash = SpatialHashGpu::with_context(
            &ctx,
            food_capacity,
            GRID_CELL_SIZE,
            [WORLD_HALF[0], WORLD_HALF[1]],
        )?;
        let sensor = SensorGatherGpu::with_context(&ctx, cap, food_capacity)?;
        let populate = PopulateInputsGpu::with_context(&ctx)?;
        let motor = MotorGpu::with_context(&ctx, cap)?;
        let step = StepGpu::with_context(&ctx, cap)?;
        let predate = PredateGpu::with_context(&ctx, cap)?;
        let food_spawn_cap = FOOD_SPAWN_RATE * MAX_SPAWN_ATTEMPTS;
        let world_map_size =
            (crate::WORLD_MAP_RES * crate::WORLD_MAP_RES * crate::WORLD_MAP_RES_Z) as u64;
        let obstacle_mask_cap: u64 = 256 * 256 * 4;
        let food_spawn =
            FoodSpawnGpu::with_context(&ctx, food_spawn_cap, world_map_size, obstacle_mask_cap)?;
        food_spawn.upload_world_map(self.map.field());
        if let Some(obs) = self.obstacles.as_ref() {
            food_spawn.upload_obstacle(&obs.packed_for_gpu());
        }
        let collision = CollisionGpu::with_context(
            &ctx,
            cap,
            GRID_CELL_SIZE,
            CELL_RADIUS,
            crate::COLLISION_RESTITUTION,
            crate::gpu::AdhesionParams {
                strength: crate::ADHESION_STRENGTH,
                cross_type: crate::ADHESION_CROSS_TYPE,
                range_factor: crate::ADHESION_RANGE_FACTOR,
            },
            crate::gpu::BondParams {
                bonds_per_cell: MAX_BONDS_PER_CELL as u32,
                break_factor: crate::BOND_BREAK_FACTOR,
            },
            crate::MAX_COLLISION_CONTACTS_PER_CELL,
            [WORLD_HALF[0], WORLD_HALF[1]],
        )?;
        let cppn = CppnGpu::with_context(&ctx, cap);
        let synaptic_scale = SynapticScaleGpu::with_context(&ctx)?;
        let excitability = ExcitabilityGpu::with_context(&ctx, cap)?;
        let izhikevich = IzhikevichGpu::with_context(&ctx)?;
        let stdp_encode_pre = StdpEncodePreGpu::with_context(&ctx)?;
        let stdp_step = StdpStepGpu::with_context(&ctx)?;
        let stdp_apply = StdpApplyGpu::with_context(&ctx)?;
        // GPU eat_food candidate selection. `food_capacity` matches the
        // worst-case live food count = `food_target(1 + CYCLE_AMPLITUDE)`
        // padded by max_population (carrion drops), already computed as
        // `field_sources_cap` above and stored as the food_hash capacity.
        let eat_food =
            EatFoodGpu::with_context(&ctx, cap, field_sources_cap, world_map_size)?;
        eat_food.upload_world_map(self.map.field());
        let turn_rates: Vec<f32> =
            self.cells.iter().map(|c| c.genome.turn_rate).collect();
        cells_gpu.upload_turn_rates(&turn_rates);
        // Sprint 137: initial population may carry loaded-checkpoint or
        // sigma-drifted rates that differ from the buffer's default fill.
        let learning_rates: Vec<f32> =
            self.cells.iter().map(|c| c.genome.learning_rate).collect();
        let trace_decays: Vec<f32> = self
            .cells
            .iter()
            .map(|c| c.genome.trace_decay_per_sec)
            .collect();
        cells_gpu.upload_learning_rates(&learning_rates);
        cells_gpu.upload_trace_decays(&trace_decays);
        let reproduce_at_energies: Vec<f32> = self
            .cells
            .iter()
            .map(|c| c.genome.reproduce_at_energy)
            .collect();
        cells_gpu.upload_reproduce_at_energies(&reproduce_at_energies);
        // Sprint 147: initial neuron-model assignment. Default population
        // is all-Perceptron, but a loaded checkpoint may carry Izhikevich
        // lineages.
        let neuron_models: Vec<u32> = self
            .cells
            .iter()
            .map(|c| c.genome.neuron_model.as_u32())
            .collect();
        cells_gpu.upload_neuron_models(&neuron_models);

        self.gpu_full = Some(GpuFullState {
            cells: cells_gpu,
            brain,
            hebbian,
            brownian,
            collision,
            predate,
            food_spawn,
            eat_food,
            smell,
            pheromone,
            pheromone_ch1,
            pheromone_ch2,
            vibration,
            cell_hash,
            food_hash,
            sensor,
            populate,
            motor,
            step,
            cppn,
            izhikevich,
            stdp_encode_pre,
            stdp_step,
            stdp_apply,
            synaptic_scale,
            excitability,
            scratch: GpuFullScratch::default(),
            profiler: GpuProfiler::new(ctx.device.clone(), gpu_profile),
        });

        if let Some(field) = self.obstacles.as_ref() {
            let packed = field.packed_for_gpu();
            let smell_mask =
                field.mask_for_grid([SMELL_GRID_RES, SMELL_GRID_RES, SMELL_GRID_RES_Z]);
            let phero_mask = field.mask_for_grid([
                PHEROMONE_GRID_RES,
                PHEROMONE_GRID_RES,
                PHEROMONE_GRID_RES_Z,
            ]);
            let vib_mask = field.mask_for_grid([
                VIBRATION_GRID_RES,
                VIBRATION_GRID_RES,
                VIBRATION_GRID_RES_Z,
            ]);
            if let Some(gpu) = self.gpu_full.as_mut() {
                gpu.step.upload_maze(&packed);
                gpu.sensor.upload_maze(&packed);
                gpu.smell.upload_obstacle_mask(&smell_mask);
                gpu.pheromone.upload_obstacle_mask(&phero_mask);
                gpu.vibration.upload_obstacle_mask(&vib_mask);
            }
        }
        Ok(())
    }

    /// Sprint 48: snapshot sim state do versioned binary blob. Format:
    /// `MAGIC[8] | bincode(Checkpoint)`. Idempotent — caller může volat
    /// průběžně i na konci.
    pub fn save_checkpoint(&self, path: &Path) -> std::io::Result<()> {
        let chk = Checkpoint {
            version: CHECKPOINT_VERSION,
            cells: self.cells.clone(),
            foods: self.foods.clone(),
            clock: self.clock,
            density_factor: self.density_factor,
            smell: self.smell.clone(),
            pheromone_fields: self.pheromone_fields.iter().cloned().collect(),
            vibration: self.vibration.clone(),
            map: self.map.clone(),
            births_gen: self.births_gen,
            deaths_gen: self.deaths_gen,
            fertile_ticks_gen: self.fertile_ticks_gen,
            predation_events_gen: self.predation_events_gen,
            mating_radius: self.mating_radius,
            max_population: self.max_population,
        };
        let mut f = std::fs::File::create(path)?;
        f.write_all(CHECKPOINT_MAGIC)?;
        bincode::serialize_into(&mut f, &chk)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(())
    }

    /// Sprint 48: rekonstrukce World z binary checkpointu. Magic + version
    /// validace. RNG, grids, scratch a GPU se re-inicializují (NE z checkpointu).
    pub fn load_checkpoint(path: &Path) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        if data.len() < 8 || &data[..8] != CHECKPOINT_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad checkpoint magic",
            ));
        }
        let chk: Checkpoint = bincode::deserialize(&data[8..])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        if chk.version != CHECKPOINT_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "checkpoint version {} expected {}",
                    chk.version, CHECKPOINT_VERSION
                ),
            ));
        }
        // Sprint 66: re-derive next_cell_id z max(cell.cell_id) + 1. Contact
        // progress se neukládá (per-tick state); restartuje prázdný.
        let next_cell_id = chk
            .cells
            .iter()
            .map(|c| c.cell_id)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        // Sprint 196: re-derive next_symbiont_lineage_id from loaded cells.
        // Symbiont presence is sparse — `filter_map` skips hosts without one.
        let next_symbiont_lineage_id = chk
            .cells
            .iter()
            .filter_map(|c| c.symbiont.as_ref().map(|s| s.lineage_id))
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        // Sprint 126: deserialize multi-channel pheromone_fields. Délka
        // checkpoint pole se musí shodovat s build-time `N_PHEROMONE_CHANNELS`.
        if chk.pheromone_fields.len() != N_PHEROMONE_CHANNELS {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "checkpoint pheromone_fields len {} expected {}",
                    chk.pheromone_fields.len(),
                    N_PHEROMONE_CHANNELS
                ),
            ));
        }
        let mut pheromone_iter = chk.pheromone_fields.into_iter();
        let pheromone_fields: [SmellField; N_PHEROMONE_CHANNELS] =
            std::array::from_fn(|_| pheromone_iter.next().expect("checked length above"));
        Ok(Self {
            cells: chk.cells,
            foods: chk.foods,
            coop_foods: Vec::new(),
            coop_food_solved_gen: 0,
            coop_food_failed_gen: 0,
            coop_food_arrivals_sum_gen: 0,
            coop_food_events_gen: 0,
            clock: chk.clock,
            density_factor: chk.density_factor,
            smell: chk.smell,
            pheromone_fields,
            vibration: chk.vibration,
            map: chk.map,
            cell_grid: SpatialGrid::new(GRID_CELL_SIZE, WORLD_HALF),
            food_grid: SpatialGrid::new(GRID_CELL_SIZE, WORLD_HALF),
            deltas_scratch: Vec::new(),
            velocity_deltas_scratch: Vec::new(),
            eaten_scratch: Vec::new(),
            id_to_idx_scratch: rustc_hash::FxHashMap::default(),
            contact_lists_scratch: Vec::new(),
            positions_snapshot_scratch: Vec::new(),
            seen_pairs_scratch: rustc_hash::FxHashSet::default(),
            bond_candidates_scratch: Vec::new(),
            bonded_pairs_scratch: rustc_hash::FxHashSet::default(),
            survivor_ids_scratch: rustc_hash::FxHashSet::default(),
            hidden_snapshot_scratch: Vec::new(),
            inputs_scratch: Vec::new(),
            births_gen: chk.births_gen,
            deaths_gen: chk.deaths_gen,
            fertile_ticks_gen: chk.fertile_ticks_gen,
            predation_events_gen: chk.predation_events_gen,
            sym_sheds_gen: 0,
            next_cell_id,
            next_symbiont_lineage_id,
            contact_progress: rustc_hash::FxHashMap::default(),
            bonds_formed_gen: 0,
            bonds_broken_gen: 0,
            bonded_attacks_gen: 0,
            solo_attacks_gen: 0,
            bonded_attack_gain_sum_gen: 0.0,
            solo_attack_gain_sum_gen: 0.0,
            swarm_attacks_gen: 0,
            pack_attacks_gen: 0,
            attack_victims_gen: 0,
            mating_radius: chk.mating_radius,
            max_population: chk.max_population,
            // Sprint 109: kalendář není v checkpointu (per-tick state je
            // deterministicky odvozen z World seed). Resume CLI musí znovu
            // předat `--shocks-mean-gens`; jinak fresh empty.
            events: EventCalendar::default(),
            active_hazard_events: Vec::new(),
            active_climate_events: Vec::new(),
            first_live_event_idx: 0,
            share_frac: crate::BOND_FOOD_SHARE_FRAC,
            kin_filter: false,
            predation_gain_mult: 1.0,
            predation_drain_mult: 1.0,
            food_factor_mult: 1.0,
            bench_timings: PhaseTimings::default(),
            // Maze state is not serialized into the checkpoint — `--maze` on
            // resume reapplies it from CLI. A checkpoint saved with `--maze`
            // but loaded without it will run the homogeneous world; the
            // converse is also fine. Cells in either direction may need a
            // few ticks to drift out of newly-walled-up positions.
            obstacles: None,
            smell_mask: None,
            pheromone_masks: std::array::from_fn(|_| None),
            vibration_mask: None,
            maze_curriculum: Vec::new(),
            maze_seed_step: 0,
            goal_zone_ticks_gen: 0,
            goal_unique_reachers_gen: rustc_hash::FxHashSet::default(),
            goal_first_reach_tick: rustc_hash::FxHashMap::default(),
            gpu: None,
            gpu_full: None,
        })
    }

    /// Rebuild `active_hazard_events` / `active_climate_events` from the
    /// full calendar. Called at the top of each `tick` so the per-cell
    /// shock callers iterate only the small active window instead of the
    /// full ~50k-event calendar.
    ///
    /// Cost is amortised by `first_live_event_idx`, a monotone cursor past
    /// events whose `[start_gen, start_gen + duration_gen)` window is
    /// fully behind us, and by `partition_point` for the not-yet-started
    /// upper bound (events are sorted by `start_gen` ascending). With
    /// 50k events sparsely distributed across 1M generations, each tick
    /// touches only the handful of events overlapping the current gen.
    ///
    /// `FoodCrash` events are skipped — they're a global per-gen scalar
    /// consumed once per generation boundary, not per cell.
    ///
    /// Tests that mutate `events.events` directly (instead of going
    /// through `tick`) must call this manually to keep the scratch in
    /// sync with the calendar.
    pub fn refresh_active_events(&mut self) {
        let gen = self.clock.generation;
        let events = &self.events.events;
        self.active_hazard_events.clear();
        self.active_climate_events.clear();
        if events.is_empty() {
            return;
        }
        while self.first_live_event_idx < events.len() {
            let e = &events[self.first_live_event_idx];
            let end = e.start_gen.saturating_add(e.duration_gen as u64);
            if end <= gen {
                self.first_live_event_idx += 1;
            } else {
                break;
            }
        }
        let upper = events.partition_point(|e| e.start_gen <= gen);
        for e in &events[self.first_live_event_idx..upper] {
            let end = e.start_gen.saturating_add(e.duration_gen as u64);
            if gen >= end {
                // Already ended, but an earlier event with longer duration
                // is still live so we can't advance the cursor past it.
                continue;
            }
            match e.kind {
                ShockKind::HazardPulse => self.active_hazard_events.push(*e),
                ShockKind::ClimateShift => self.active_climate_events.push(*e),
                ShockKind::FoodCrash => {}
            }
        }
    }

    /// Rebuild persistent `id_to_idx_scratch` from current `cells` order.
    /// Cell layout je stable v rámci fází 3–17 (před `reproduce` / `die_*`),
    /// takže build raz per tick stačí. `clear()` zachovává hashmap kapacity.
    fn rebuild_id_to_idx(&mut self) {
        self.id_to_idx_scratch.clear();
        self.id_to_idx_scratch.reserve(self.cells.len());
        for (i, c) in self.cells.iter().enumerate() {
            self.id_to_idx_scratch.insert(c.cell_id, i);
        }
    }

    pub fn tick(&mut self, rng: &mut impl Rng) -> Option<u64> {
        let dt = 1.0 / FIXED_TIMESTEP_HZ;
        let transitions = self.clock.advance();
        // Filter the calendar down to events overlapping the current
        // generation so per-cell shock callers (`apply_hazards`,
        // `climate_offset_at`, the CPU `step` body) iterate O(active) not
        // O(events). At `mean_gens_between=20` over 1M gens the full
        // calendar is ~50k events but typically <10 overlap any given gen.
        self.refresh_active_events();
        // Wave 3 curriculum: check ramp on generation roll-over. Rebuild
        // obstacles + masks when the active stage's difficulty differs from
        // what's currently allocated (and seed walks via maze_seed_step so
        // each stage gets a fresh maze layout).
        if transitions.generation_ended.is_some() && !self.maze_curriculum.is_empty() {
            let target = self.difficulty_for_generation(self.clock.generation);
            let current = self.obstacles.as_ref().map(|o| o.difficulty);
            if target != current {
                match target {
                    Some(diff) => {
                        let base_seed = self
                            .obstacles
                            .as_ref()
                            .map(|o| o.seed)
                            .unwrap_or(0);
                        self.rebuild_maze(diff, base_seed);
                        eprintln!(
                            "curriculum: gen {} → maze {}",
                            self.clock.generation,
                            diff.label()
                        );
                    }
                    None => {
                        self.obstacles = None;
                        self.smell_mask = None;
                        self.pheromone_masks = std::array::from_fn(|_| None);
                        self.vibration_mask = None;
                        eprintln!(
                            "curriculum: gen {} → maze off",
                            self.clock.generation
                        );
                    }
                }
            }
        }
        if transitions.generation_ended.is_some() {
            let phase =
                (self.clock.generation as f32 / CYCLE_GEN_PERIOD as f32) * std::f32::consts::TAU;
            let seasonal = 1.0 + CYCLE_AMPLITUDE * phase.sin();
            // Sprint 113: FoodCrash multiplikátor (1.0 default). Compound
            // přes všechny aktivní FoodCrash, clamp na FOOD_CRASH_MIN_FACTOR.
            let shock_mult = crate::food_density_shock_multiplier(
                &self.events.events,
                self.clock.generation,
            );
            let scarcity = scarcity_factor(self.clock.generation);
            self.density_factor = seasonal * shock_mult * scarcity;
        }

        macro_rules! timed {
            ($field:ident, $call:expr) => {{
                let _t = Instant::now();
                $call;
                self.bench_timings.$field += _t.elapsed().as_secs_f64() * 1e6;
            }};
        }

        timed!(update_smell, self.update_smell(dt));
        timed!(update_pheromone, self.update_pheromone(dt));
        timed!(update_vibration, self.update_vibration(dt));
        // Persistent cell_id → idx — built once per tick, consumed v pool_bonded_hidden,
        // brain_act, resolve_collisions, eat_food. Cell layout je stable
        // od tady přes eat_food; reproduce/die_and_drop_carrion na konci ticku
        // mapu invalidují, ale ta se rebuilduje další tick.
        self.rebuild_id_to_idx();
        // Sprint 94: pool last_hidden across bond network → cluster cells share
        // recurrent state. Must run before brain_act (which reads pooled_hidden).
        self.pool_bonded_hidden();
        self.pool_bond_messages();
        // Wave 2: whisker raycast against the maze. Stores per-cell results
        // on `cell.last_whisker_distances` so the sensor gather closure can
        // read them without an extra `&ObstacleField` capture.
        self.update_whiskers();
        timed!(brain_act, self.run_brain_act(dt));
        // Wave 3: per-tick eligibility-trace decay+accumulate — runs once
        // brain_act has populated this tick's last_inputs/last_hidden/last_outputs.
        // Wave 7: GPU full pipeline runs the equivalent on-device, mutating
        // `cells.brain_traces` in place. Skip CPU pass to avoid double-decay
        // and to keep per-cell `genome.brain.trace_w*` matching the GPU
        // until next-gen sync (`sync_brains_from_gpu`).
        // GPU pipeline runs the equivalent on-device; skip CPU Hebbian
        // to avoid double-decay (CPU `genome.brain.trace_w*` stays stale
        // until next-gen `sync_brains_from_gpu`).
        if let Some(gpu) = self.gpu_full.as_ref() {
            let n = self.cells.len();
            gpu.hebbian.dispatch_step_persistent(
                &gpu.cells,
                n,
                dt,
                crate::HEBBIAN_TRACE_DECAY_PER_SEC,
            );
            // Sprint 139: per-tick excitability regulator. Reads
            // `last_hidden` (already populated by brain_act above), updates
            // its private activity EMA buffer, and slides `b1` slots of
            // `cells.brain_weights_buf` toward the responsive tanh zone.
            gpu.excitability.dispatch(
                &gpu.cells,
                n,
                ACTIVITY_EMA_ALPHA,
                EXCITABILITY_DRIFT_PER_TICK,
            );
        }
        timed!(emit_pheromones, self.emit_pheromones(dt));
        timed!(apply_morph, self.apply_morph(dt));
        timed!(apply_brownian, self.apply_brownian(rng, dt));
        timed!(step, self.step(dt));
        self.apply_symbiont_energy(dt);
        timed!(apply_food_gravity, self.apply_food_gravity(dt));
        timed!(apply_hazards, self.apply_hazards(dt));
        timed!(resolve_collisions, self.resolve_collisions());
        timed!(predate, self.predate(rng));
        timed!(eat_food, self.eat_food());
        timed!(spawn_food, self.spawn_food(rng));
        self.spawn_coop_food(rng);
        self.update_coop_food();
        timed!(reproduce, self.reproduce(rng));
        timed!(die_and_drop_carrion, self.die_and_drop_carrion(rng));
        // Wave 2: episodic novelty reward — runs after position updates so
        // the cell's voxel reflects this tick's motor outcome. Always-on
        // (independent of maze toggle); for a homogeneous world this rewards
        // any exploration. Helps cells break out of local minima.
        self.apply_episodic_novelty();
        // Sprint 138: periodic homeostatic synaptic scaling. Runs after the
        // tick's Hebbian dispatches so this period's weight growth gets
        // capped before the next decay window opens.
        if self.clock.tick % SCALING_PERIOD_TICKS == 0 {
            if let Some(gpu) = self.gpu_full.as_ref() {
                let n = self.cells.len();
                // Cap/decay/floor run every tick; the O(hidden_n²) GS pass is
                // throttled to every GS_PERIOD_TICKS with its per-application
                // strength scaled up by the same factor, so the time-averaged
                // decorrelation rate stays at GRAM_SCHMIDT_STRENGTH. gs = 0 on
                // off-ticks makes the shader branch past the GS block.
                let gs_strength = if self.clock.tick % GS_PERIOD_TICKS == 0 {
                    GRAM_SCHMIDT_STRENGTH * GS_PERIOD_TICKS as f32
                } else {
                    0.0
                };
                gpu.synaptic_scale.dispatch(
                    &gpu.cells,
                    n,
                    W_NORM_CAP,
                    WEIGHT_DECAY_PER_TICK,
                    gs_strength,
                    W_NORM_FLOOR,
                    gpu.brain.hidden_n_buffer(),
                );
            }
        }
        // Maze navigation tracker — no-op when obstacles is None, so the
        // homogeneous-world tick stays byte-identical with pre-maze runs.
        self.track_goal_metrics();

        transitions.generation_ended
    }

    /// Sprint 128: per-tick spawn pokud pod cap. Poisson-like Bernoulli draw
    /// — drobná pravděpodobnost na každý tick než hard quota per gen, aby
    /// distribuce events byla rozprostřena rovnoměrně časem.
    fn spawn_coop_food(&mut self, rng: &mut impl Rng) {
        if self.coop_foods.len() >= COOP_FOOD_MAX_CONCURRENT {
            return;
        }
        if rng.random::<f32>() >= COOP_FOOD_SPAWN_RATE_PER_TICK {
            return;
        }
        let pos = crate::random_coop_position(rng, WORLD_HALF);
        self.coop_foods
            .push(CoopFood::new(pos, self.clock.tick));
    }

    /// Sprint 128: per-tick arrival registration → trigger pokus → cleanup.
    /// Triggered + expired nodes se odstraňují stejným průchodem; counters
    /// se nakrmí pro CSV diagnostiku (reset per gen).
    fn update_coop_food(&mut self) {
        if self.coop_foods.is_empty() {
            return;
        }
        crate::register_coop_arrivals_for_all(
            &mut self.coop_foods,
            &self.cells,
            WORLD_HALF,
        );
        let current_tick = self.clock.tick;
        let cells = &mut self.cells;
        let mut i = 0;
        while i < self.coop_foods.len() {
            let triggered_now = crate::try_trigger_coop(&mut self.coop_foods[i], cells);
            if triggered_now {
                self.coop_food_solved_gen += 1;
                self.coop_food_arrivals_sum_gen +=
                    self.coop_foods[i].arrivals.len() as u64;
                self.coop_food_events_gen += 1;
                self.coop_foods.swap_remove(i);
                continue;
            }
            if self.coop_foods[i].is_expired(current_tick) {
                self.coop_food_failed_gen += 1;
                self.coop_food_arrivals_sum_gen +=
                    self.coop_foods[i].arrivals.len() as u64;
                self.coop_food_events_gen += 1;
                self.coop_foods.swap_remove(i);
                continue;
            }
            i += 1;
        }
    }

    fn apply_morph(&mut self, dt: f32) {
        // Sprint 57: zkoušel jsem rayon par_iter_mut, ale ~2 us sekvenčně vs
        // ~26 us paralelně — rayon spawn overhead převáží práci. Sekvenční win.
        for cell in &mut self.cells {
            cell.apply_morph(dt);
        }
    }

    fn apply_brownian(&mut self, _rng: &mut impl Rng, _dt: f32) {
        // Brownian noise is fused into `brain_act_gpu_full` (motor →
        // brownian → batch readback). This phase is a no-op on the
        // mandatory GPU path; kept as a tick-loop call site so per-phase
        // bench_timings stays comparable across versions.
    }

    /// Sprint 51: GPU brownian s xoshiro128++ per-cell RNG. Upload velocities,
    /// dispatch, download. Ne-deterministic vs CPU (different PRNG), ale
    /// deterministic across GPU runs (xoshiro state seedovaný z cell.lineage_id).
    /// Sprint 62: nyní fused do brain_act_gpu_full pipeline; tato standalone
    /// metoda je dead code preserved pro Sprint 63+ test path.
    #[allow(dead_code)]
    fn apply_brownian_gpu(&mut self, dt: f32) {
        let n = self.cells.len();
        if n == 0 {
            return;
        }
        let velocities: Vec<[f32; 3]> = self.cells.iter().map(|c| c.velocity).collect();
        let gpu = self.gpu_full.as_mut().unwrap();
        gpu.cells.upload_velocities(&velocities);
        gpu.brownian.compute_persistent(
            &gpu.cells,
            n,
            crate::THERMAL_NOISE,
            dt,
            WORLD_HALF[2] > 0.0,
        );
        let new_vels = gpu.cells.download_velocities(n);
        for (cell, v) in self.cells.iter_mut().zip(new_vels.iter()) {
            cell.velocity = *v;
        }
    }

    fn update_smell(&mut self, dt: f32) {
        // Deposit + diffuse on GPU without readback. Sensor shader reads
        // FieldGpu.current_grid_buffer directly — no CPU SmellField sync.
        let gpu = self.gpu_full.as_mut().expect("gpu_full mandatory");
        for food in &self.foods {
            gpu.smell.add_source(
                [food.position[0], food.position[1], food.position[2]],
                SMELL_PER_FOOD * dt,
            );
        }
        gpu.smell.step(SMELL_DIFFUSION, SMELL_DECAY, dt);
    }

    fn update_pheromone(&mut self, dt: f32) {
        // Diffuse + decay BEFORE this tick's emissions (added in
        // emit_pheromones, called after brain_act) so cells read the
        // previous-tick gradient — prevents instant self-feedback.
        // All 3 channels step independently on GPU (Wave L); CPU mirror
        // fields stay out-of-date until checkpoint readback.
        let gpu = self.gpu_full.as_mut().expect("gpu_full mandatory");
        gpu.pheromone.step(PHEROMONE_DIFFUSION_PER_CH[0], PHEROMONE_DECAY_PER_CH[0], dt);
        gpu.pheromone_ch1.step(
            PHEROMONE_DIFFUSION_PER_CH[1],
            PHEROMONE_DECAY_PER_CH[1],
            dt,
        );
        gpu.pheromone_ch2.step(
            PHEROMONE_DIFFUSION_PER_CH[2],
            PHEROMONE_DECAY_PER_CH[2],
            dt,
        );
    }

    fn update_vibration(&mut self, dt: f32) {
        // Deposit each cell's motion-driven emission, then diffuse + decay
        // BEFORE this tick's brain_act samples the gradient. Brain reads
        // a propagated field that already reflects this tick's motion.
        let gpu = self.gpu_full.as_mut().expect("gpu_full mandatory");
        for cell in &self.cells {
            let emit = crate::vibration_emit_for_cell(cell);
            if emit > 0.0 {
                gpu.vibration.add_source(cell.position, emit * dt);
            }
        }
        gpu.vibration.step(VIBRATION_DIFFUSION, VIBRATION_DECAY, dt);
    }

    fn emit_pheromones(&mut self, dt: f32) {
        // Per-channel emission. Brain output slot map:
        //   [2]  = ch0 (slow, mating-friendly)
        //   [10] = ch1 (medium decay)
        //   [11] = ch2 (fast decay, bursty / temporal patterning)
        // Cost = sum of positive emissions × PHEROMONE_COST_PER_RATE.
        const EMIT_SLOTS: [usize; N_PHEROMONE_CHANNELS] = [2, 10, 11];
        let gpu = self.gpu_full.as_mut().expect("gpu_full mandatory");
        for cell in &mut self.cells {
            let pos = [cell.position[0], cell.position[1], cell.position[2]];
            let mut total_emit = 0.0_f32;
            let mut emits = [0.0_f32; N_PHEROMONE_CHANNELS];
            for ch in 0..N_PHEROMONE_CHANNELS {
                let mod_strength = cell.last_outputs[EMIT_SLOTS[ch]].max(0.0);
                let brain_emit = PHEROMONE_BRAIN_MOD * mod_strength;
                emits[ch] = brain_emit;
                total_emit += brain_emit;
                let rate = PHEROMONE_BASELINE_EMIT + brain_emit;
                match ch {
                    0 => gpu.pheromone.add_source(pos, rate * dt),
                    1 => gpu.pheromone_ch1.add_source(pos, rate * dt),
                    _ => gpu.pheromone_ch2.add_source(pos, rate * dt),
                }
                let prev = cell.last_emit[ch];
                let delta = brain_emit - prev;
                cell.burst_accum[ch] += delta * delta;
            }
            cell.last_emit = emits;
            cell.energy -= PHEROMONE_COST_PER_RATE * total_emit * dt;
        }
    }

    /// Sprint 44 + 51: dispatch GPU full / GPU brain / CPU.
    fn run_brain_act(&mut self, dt: f32) {
        {
            if self.gpu_full.is_some() {
                self.brain_act_gpu_full(dt);
                return;
            }
            if self.gpu.is_some() {
                self.brain_act_gpu(dt);
                return;
            }
        }
        self.brain_act(dt);
    }

    /// Sprint 51/60/61/62: --gpu-full brain_act.
    /// Sprint 62 pipeline (single Wait barrier):
    ///   1. CPU `apply_shell_absorb` + snapshot.
    ///   2. CellsGpu uploads (metadata + velocity + ang_vel + pitch_vel).
    ///   3. GPU spatial_hash dispatch cell + food (no readback).
    ///   4. GPU SensorGather dispatch_no_readback (output stays GPU).
    ///   5. GPU PopulateInputs (čte sensor + cells, píše last_inputs).
    ///   6. GPU brain.forward_persistent (čte last_inputs, píše last_hidden + last_outputs).
    ///   7. **GPU motor.dispatch_with_cells** (Sprint 62: čte last_outputs +
    ///      heading/pitch/turn_rate/eff_radius/max_speed, mutuje velocity +
    ///      angular_velocity + pitch_velocity).
    ///   8. **GPU brownian.compute_persistent** (Sprint 51 + 62 fuze: mutuje
    ///      velocity v stejném pipeline chain — apply_brownian fáze v
    ///      `--gpu-full` se stává no-op).
    ///   9. **`download_brain_motor_batch`** (Sprint 62 NEW): single Wait pro
    ///      hidden + outputs + velocity + angular_vel + pitch_vel. 1 RT/tick.
    ///   10. CPU writeback all 5 buffers + apply_brain_motor je SKIPPED (motor
    ///       byl GPU-side).
    /// Round-trip status: 1× `device.poll(Wait)` per tick (vs Sprint 61 2×).
    fn brain_act_gpu_full(&mut self, dt: f32) {
        let n = self.cells.len();
        if n == 0 {
            return;
        }

        // Phase 1: CPU snapshots + apply_shell_absorb (lib helper) na CPU side
        // pre-upload. Damage absorb se aplikuje na CPU damage_accum, pak
        // upload do GPU; populate_inputs shader read+reset.
        for cell in &mut self.cells {
            cell.apply_shell_absorb(dt);
            // eat_food skip optim: GPU sensor pipeline nesdílí best_food_d2 zpět
            // do CPU. Nastavíme 0.0 aby `cell_eats_food` skip nikdy netrigeroval
            // (food_grid query běží jako pre-skip baseline).
            cell.last_best_food_d2 = 0.0;
        }

        // Persistent scratch fill — single pass přes cells, zachovaná kapacita
        // mezi ticky. Pre-fix: 17× `iter().map().collect()` → 17 alloc/tick.
        let food_n = self.foods.len() + self.coop_foods.len();
        let gpu = self.gpu_full.as_mut().expect("gpu_full Some");
        let s = &mut gpu.scratch;
        s.clear_and_reserve(n, food_n);
        for cell in self.cells.iter() {
            s.positions.push(cell.position);
            s.eff_radii.push(cell.phenotype.effective_radius());
            s.vision_radii.push(cell.genome.vision_radius);
            s.energies.push(cell.energy);
            s.headings.push(cell.heading);
            s.pitches.push(cell.pitch);
            s.damage_accums.push(cell.damage_accum);
            s.max_speeds.push(cell.genome.max_speed);
            s.velocities.push(cell.velocity);
            s.angular_vels.push(cell.angular_velocity);
            s.pitch_vels.push(cell.pitch_velocity);
            s.ages.push(cell.age as u32);
            s.cooldowns.push(cell.reproduce_cooldown_ticks);
            s.body_dims.push([
                cell.phenotype.body_length,
                cell.phenotype.body_width,
                cell.phenotype.body_height,
            ]);
            // aux = [spike_length, shell_thickness, vision_radius, attack_strength].
            // Sprint 63: attack je předchozí tick last_outputs[6] (1-tick delay).
            s.aux.push([
                cell.phenotype.total_spike_cost_factor(),
                cell.phenotype.shell_thickness,
                cell.genome.vision_radius,
                cell.last_outputs[6].max(0.0),
            ]);
            s.hidden_ns.push(cell.genome.brain.hidden_n);
            s.bonded_inboxes.push(cell.bonded_inbox);
        }
        // Sprint 128: foods + coop_foods do single sensor pool.
        for food in self.foods.iter() {
            s.food_positions.push(food.position);
        }
        for coop in self.coop_foods.iter() {
            s.food_positions.push(coop.position);
        }

        // Aliases pro zbytek funkce (immutable views — zachovávají původní
        // názvy proměnných, takže downstream kód zůstane beze změny).
        let positions = s.positions.as_slice();
        let eff_radii = s.eff_radii.as_slice();
        let vision_radii = s.vision_radii.as_slice();
        let food_positions = s.food_positions.as_slice();
        let energies = s.energies.as_slice();
        let headings = s.headings.as_slice();
        let pitches = s.pitches.as_slice();
        let damage_accums = s.damage_accums.as_slice();
        let max_speeds = s.max_speeds.as_slice();
        let velocities = s.velocities.as_slice();
        let angular_vels = s.angular_vels.as_slice();
        let pitch_vels = s.pitch_vels.as_slice();
        let positions_for_step = positions;
        let ages = s.ages.as_slice();
        let cooldowns = s.cooldowns.as_slice();
        let body_dims = s.body_dims.as_slice();
        let aux = s.aux.as_slice();

        gpu.profiler.frame_start();
        // Phase 2: upload cell metadata + velocities + angular/pitch velocities.
        gpu.cells.upload_metadata(
            &energies,
            &headings,
            &pitches,
            &damage_accums,
            &max_speeds,
            &eff_radii,
        );
        gpu.cells.upload_velocities(&velocities);
        gpu.cells.upload_angular_pitch(&angular_vels, &pitch_vels);
        // Sprint 63: step shader uploads.
        gpu.cells.upload_positions(&positions_for_step);
        gpu.cells.upload_age_cooldown(&ages, &cooldowns);
        gpu.cells.upload_body_dims(&body_dims);
        gpu.cells.upload_aux(&aux);
        gpu.cells.upload_bonded_inboxes(s.bonded_inboxes.as_slice());
        gpu.profiler.mark("uploads");

        // Phase 3: GPU spatial hash dispatch (no readback).
        gpu.cell_hash.dispatch(&positions);
        gpu.food_hash.dispatch(&food_positions);
        gpu.profiler.mark("spatial_hash");

        // Phase 4: GPU SensorGather dispatch_no_readback. Output v output_buf
        // storage; populate_inputs shader bind direct.
        let sensor_params = SensorParamsGpu {
            num_cells: n as u32,
            num_foods: food_positions.len() as u32,
            hash_cell_size: GRID_CELL_SIZE,
            world_half_x: WORLD_HALF[0],
            world_half_y: WORLD_HALF[1],
            world_half_z: WORLD_HALF[2],
            field_res_x: SMELL_GRID_RES as u32,
            field_res_y: SMELL_GRID_RES as u32,
            field_res_z: SMELL_GRID_RES_Z as u32,
            field_eps: SMELL_SAMPLE_EPSILON,
            field_world_half_x: WORLD_HALF[0],
            field_world_half_y: WORLD_HALF[1],
            field_world_half_z: WORLD_HALF[2],
            // Wave 5: maze fields enable LOS raycast in the shader.
            maze_active: if self.obstacles.is_some() { 1 } else { 0 },
            maze_res_x: self
                .obstacles
                .as_ref()
                .map(|f| f.resolution[0] as u32)
                .unwrap_or(0),
            maze_res_y: self
                .obstacles
                .as_ref()
                .map(|f| f.resolution[1] as u32)
                .unwrap_or(0),
            // Sprint 195: seeds the whisker spring-damper transduction noise.
            tick: self.clock.tick as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        gpu.sensor.dispatch_no_readback(
            &positions,
            &eff_radii,
            &vision_radii,
            &food_positions,
            &headings,
            &pitches,
            &gpu.cell_hash,
            &gpu.food_hash,
            &gpu.smell,
            &gpu.pheromone,
            &gpu.pheromone_ch1,
            &gpu.pheromone_ch2,
            &gpu.vibration,
            gpu.cells.whisker_state_buffer(),
            sensor_params,
        );
        gpu.profiler.mark("sensor_gather");

        // Phase 5: GPU populate_inputs. Čte sensor.output_buf + cells metadata
        // → píše cells.last_inputs_buf. damage_accum_buf reset v shaderu.
        let populate_params = PopulateInputsParams {
            num_cells: n as u32,
            brain_inputs: BRAIN_INPUTS as u32,
            brain_inputs_sensory: BRAIN_INPUTS_SENSORY as u32,
            brain_hidden: BRAIN_HIDDEN as u32,
            brain_recurrent: BRAIN_RECURRENT as u32,
            smell_norm_gain: SMELL_NORMALIZATION_GAIN,
            phero_norm_gain: PHEROMONE_NORMALIZATION_GAIN,
            damage_norm_gain: DAMAGE_NORMALIZATION_GAIN,
            density_norm: DENSITY_NORM_COUNT,
            vibration_amp_gain: crate::VIBRATION_AMP_GAIN,
            vibration_grad_gain: crate::VIBRATION_GRAD_GAIN,
            _pad1: 0,
        };
        gpu.populate
            .dispatch(&gpu.cells, &gpu.sensor, populate_params);
        gpu.profiler.mark("populate_inputs");

        // Phase 6: GPU brain forward_persistent. Čte `last_inputs_buf` direct,
        // píše last_hidden + last_outputs storage buffers.
        gpu.brain.forward_persistent(
            &gpu.cells,
            n,
            &gpu.scratch.hidden_ns,
            LATERAL_INHIBITION_ALPHA,
        );
        gpu.profiler.mark("brain_forward");
        // Sprint 147: Izhikevich forward runs AFTER perceptron — cells whose
        // `neuron_models[i] == Izhikevich` overwrite the perceptron's
        // last_hidden / last_outputs with spike-rate outputs. Perceptron
        // cells are early-exited inside the shader (zero overhead per
        // skipped cell beyond a single buffer read).
        // Sprint 168: STDP pre-spike encoder runs before Izhikevich forward
        // so the forward shader's post-spike write + the encoder's pre-spike
        // write are both visible to the trace step that follows.
        let tick_u32 = self.clock.tick as u32;
        gpu.stdp_encode_pre.dispatch(&gpu.cells, n, tick_u32);
        gpu.profiler.mark("stdp_encode_pre");
        gpu.izhikevich.dispatch(
            &gpu.cells,
            n,
            tick_u32,
            LATERAL_INHIBITION_ALPHA,
            &gpu.scratch.hidden_ns,
        );
        gpu.profiler.mark("izhikevich");
        // Trace decay+accumulate based on both pre and post spike-times.
        gpu.stdp_step
            .dispatch(&gpu.cells, n, tick_u32, crate::DEFAULT_STDP_TAU_TICKS);
        gpu.profiler.mark("stdp_step");

        // Phase 7: GPU motor.dispatch_with_cells. Čte last_outputs + heading/
        // pitch/turn_rate/eff_radius/max_speed, mutuje velocity/angular_vel/
        // pitch_vel buffery in-place. Mirror lib::Cell::apply_brain_motor.
        gpu.motor
            .dispatch_with_cells(&gpu.cells, n, dt, DRAG_COEFFICIENT);
        gpu.profiler.mark("motor");

        // Phase 8: GPU brownian dispatch (Sprint 51 path). Mutuje velocity_buf
        // s xoshiro128++ per-cell noise. Fused do brain_act → apply_brownian
        // fáze v `--gpu-full` skipne (compute už proběhl).
        let has_z = WORLD_HALF[2] > 0.0;
        gpu.brownian
            .compute_persistent(&gpu.cells, n, THERMAL_NOISE, dt, has_z);
        gpu.profiler.mark("brownian");

        // Phase 9: GPU step.dispatch_with_cells. Mirror lib Cell::step
        // (kinematics + drag + energy costs + world bounce). Mutuje position/
        // velocity/heading/pitch/ang_vel/pitch_vel/age/cooldown/energy.
        // Sprint 63: skip CPU `step` fáze v `--gpu-full`.
        let step_params = StepParamsGpu {
            num_cells: n as u32,
            _pad_a0: 0,
            _pad_a1: 0,
            _pad_a2: 0,
            dt,
            world_half_x: WORLD_HALF[0],
            world_half_y: WORLD_HALF[1],
            world_half_z: WORLD_HALF[2],
            gravity: PHYS_GRAVITY,
            drag: PHYSICS_CONFIG.drag,
            angular_drag: PHYSICS_CONFIG.angular_drag,
            energy_cost_per_v_sq: PHYSICS_CONFIG.energy_cost_per_v_sq,
            angular_energy_cost: PHYSICS_CONFIG.angular_energy_cost,
            vision_cost_per_radius: PHYSICS_CONFIG.vision_cost_per_radius,
            body_cost_factor: PHYSICS_CONFIG.body_cost_factor,
            age_decay_per_sec: AGE_DECAY_PER_SEC,
            fixed_timestep_hz: FIXED_TIMESTEP_HZ,
            spike_cost_per_sec: SPIKE_COST_PER_SEC,
            shell_cost_per_sec: SHELL_COST_PER_SEC,
            attack_cost_per_sec: ATTACK_COST_PER_SEC,
            pitch_clamp: core::f32::consts::FRAC_PI_6 * 0.5,
            thermal_top: crate::THERMAL_TOP,
            thermal_bottom: crate::THERMAL_BOTTOM,
            thermal_q10: crate::THERMAL_Q10,
            thermal_ref_temp: crate::THERMAL_REF_TEMP,
            // Sprint 86: per-tick phase fractions, pre-computed na CPU aby
            // shader nemusel řešit u64 modulo + f32 cast.
            thermal_diurnal_amp: crate::THERMAL_DIURNAL_AMP,
            thermal_seasonal_amp: crate::THERMAL_SEASONAL_AMP,
            thermal_diurnal_sin: {
                let phase = (self.clock.tick % crate::THERMAL_DIURNAL_PERIOD_TICKS) as f32
                    / crate::THERMAL_DIURNAL_PERIOD_TICKS as f32;
                (core::f32::consts::TAU * phase).sin()
            },
            thermal_seasonal_sin: {
                let phase = (self.clock.generation % CYCLE_GEN_PERIOD) as f32
                    / CYCLE_GEN_PERIOD as f32;
                (core::f32::consts::TAU * phase).sin()
            },
            thermal_log2_q10: crate::THERMAL_Q10.log2(),
            // Wave 4: maze fields. When obstacles present, mask was uploaded
            // at allocation time (`World::sync_maze_to_gpu`); shader uses it.
            maze_active: if self.obstacles.is_some() { 1 } else { 0 },
            maze_res_x: self
                .obstacles
                .as_ref()
                .map(|f| f.resolution[0] as u32)
                .unwrap_or(0),
            maze_res_y: self
                .obstacles
                .as_ref()
                .map(|f| f.resolution[1] as u32)
                .unwrap_or(0),
        };
        gpu.step.dispatch_with_cells(&gpu.cells, n, step_params);
        gpu.profiler.mark("step");

        // Phase 10: single batch readback (Sprint 63: 9 buffers fused do
        // jednoho Wait barrier). Pre-fix path 9 fresh Vec collect/to_vec —
        // teď přepíše persistent scratch sloty, 0 alloc/free při stable cap.
        gpu.cells.download_full_batch_into(
            n,
            &mut gpu.scratch.dl_inputs,
            &mut gpu.scratch.dl_hiddens,
            &mut gpu.scratch.dl_outputs,
            &mut gpu.scratch.dl_velocities,
            &mut gpu.scratch.dl_angular,
            &mut gpu.scratch.dl_pitch,
            &mut gpu.scratch.dl_positions,
            &mut gpu.scratch.dl_ages,
            &mut gpu.scratch.dl_cooldowns,
            &mut gpu.scratch.dl_energies,
        );
        gpu.profiler.mark("download");

        // Phase 11: CPU writeback. NO apply_brain_motor + NO Cell::step CPU
        // (oba byly GPU-side). damage_accum reset (mirror populate_inputs).
        let dl = &gpu.scratch;
        let inputs = &dl.dl_inputs;
        let hiddens = &dl.dl_hiddens;
        let outputs = &dl.dl_outputs;
        let new_vels = &dl.dl_velocities;
        let new_ang = &dl.dl_angular;
        let new_pitch = &dl.dl_pitch;
        let new_pos = &dl.dl_positions;
        let new_ages = &dl.dl_ages;
        let new_cooldowns = &dl.dl_cooldowns;
        let new_energies = &dl.dl_energies;
        self.cells
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, cell)| {
                cell.last_inputs = inputs[i];
                cell.last_hidden = hiddens[i];
                cell.last_outputs = outputs[i];
                cell.velocity = new_vels[i];
                cell.angular_velocity = new_ang[i];
                cell.pitch_velocity = new_pitch[i];
                cell.position = new_pos[i];
                cell.age = new_ages[i] as u64;
                cell.reproduce_cooldown_ticks = new_cooldowns[i];
                cell.energy = new_energies[i];
                cell.damage_accum = 0.0;
            });
    }

    fn brain_act_gpu(&mut self, dt: f32) {
        let n = self.cells.len();
        if n == 0 {
            return;
        }
        self.cell_grid.rebuild(
            self.cells
                .iter()
                .enumerate()
                .map(|(i, c)| (i, c.position, c.phenotype.effective_radius())),
        );
        self.food_grid.rebuild(
            self.foods
                .iter()
                .enumerate()
                .map(|(i, f)| (i, f.position, f.kind)),
        );

        let cell_grid = &self.cell_grid;
        let food_grid = &self.food_grid;
        let smell = &self.smell;
        let pheromone_fields = &self.pheromone_fields;
        let vibration = &self.vibration;
        let coop_foods = &self.coop_foods;
        let tick = self.clock.tick;
        let gen = self.clock.generation;

        // Phase 1: par_iter_mut nad cells — sensor gather, populate inputs,
        // apply_shell_absorb. Vrací inputs[N] pro GPU dispatch.
        let inputs_vec: Vec<[f32; BRAIN_INPUTS]> = self
            .cells
            .par_iter_mut()
            .enumerate()
            .map(|(i, cell)| {
                let pos = cell.position;
                let vision_r = cell.genome.vision_radius;
                let vr2 = vision_r * vision_r;
                // Sprint 83: cone filter — viz `gather` v main.rs.
                let fov = cell.genome.vision_fov;
                let skip_cone = fov >= crate::MAX_VISION_FOV;
                let cos_fov = fov.cos();
                let fwd = crate::forward_vector(cell.heading, cell.pitch);

                let mut best_food: Option<[f32; 3]> = None;
                let mut best_food_d2 = f32::MAX;
                food_grid.for_each_in_radius_toroidal(pos, vision_r, WORLD_HALF, |_id, fp, _| {
                    let d = crate::min_image_delta(pos, fp, WORLD_HALF);
                    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if d2 > vr2 || d2 >= best_food_d2 {
                        return;
                    }
                    if !skip_cone && !crate::fov_cone_accept(d, d2, fwd, cos_fov) {
                        return;
                    }
                    best_food_d2 = d2;
                    best_food = Some(d);
                });
                // Sprint 128: scan coop foods proti same FOV/range — sdílené
                // input slot (`nearest_food`), aby cells reagovaly stejným
                // approach behavior. Trade-off vůči regular food je čistě
                // distance — coop food má vyšší expected value, ale solo
                // arrival nedostane reward.
                for coop in coop_foods.iter() {
                    let d = crate::min_image_delta(pos, coop.position, WORLD_HALF);
                    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if d2 > vr2 || d2 >= best_food_d2 {
                        continue;
                    }
                    if !skip_cone && !crate::fov_cone_accept(d, d2, fwd, cos_fov) {
                        continue;
                    }
                    best_food_d2 = d2;
                    best_food = Some(d);
                }

                let mut best_cell: Option<([f32; 3], f32)> = None;
                let mut best_cell_d2 = f32::MAX;
                let mut neighbors_in_vision: u32 = 0;
                cell_grid.for_each_in_radius_toroidal(pos, vision_r, WORLD_HALF, |id, op, oradius| {
                    if id == i {
                        return;
                    }
                    let d = crate::min_image_delta(pos, op, WORLD_HALF);
                    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if d2 > vr2 {
                        return;
                    }
                    if !skip_cone && !crate::fov_cone_accept(d, d2, fwd, cos_fov) {
                        return;
                    }
                    neighbors_in_vision += 1;
                    if d2 < best_cell_d2 {
                        best_cell_d2 = d2;
                        best_cell = Some((d, oradius));
                    }
                });

                let pos_xyz = [pos[0], pos[1], pos[2]];
                let smell_grad = smell.gradient_at(pos_xyz, SMELL_SAMPLE_EPSILON);
                let mut pheromone_grads = [[0.0_f32; 3]; N_PHEROMONE_CHANNELS];
                for ch in 0..N_PHEROMONE_CHANNELS {
                    pheromone_grads[ch] =
                        pheromone_fields[ch].gradient_at(pos_xyz, PHEROMONE_SAMPLE_EPSILON);
                }
                let temperature_local =
                    crate::temperature_at_z(pos[2], WORLD_HALF, tick, gen);
                let vibration_grad =
                    vibration.gradient_at(pos_xyz, VIBRATION_SAMPLE_EPSILON);
                let vibration_amp = vibration.sample(pos_xyz);
                let sensors = crate::BrainSensors {
                    nearest_food: best_food,
                    nearest_cell: best_cell,
                    neighbors_in_vision,
                    smell_grad,
                    pheromone_grads,
                    temperature_local,
                    vibration_grad,
                    vibration_amp,
                    whisker_distances: cell.last_whisker_distances,
                };
                cell.apply_shell_absorb(dt);
                // eat_food skip optim: cache nejbližší food d² (viz CPU path).
                cell.last_best_food_d2 = best_food_d2;
                let mut inputs = crate::populate_brain_inputs(cell, &sensors, vision_r);
                crate::apply_sensor_gains(&mut inputs, &cell.genome.sensor_gains);
                inputs
            })
            .collect();

        // Sprint 97: Phase 1b: pool max-magnitude přes bond network. Provede se
        // PŘED GPU uploadem aby brain forward už dostal pooled inputs.
        let id_to_idx = &self.id_to_idx_scratch;
        let pooled_inputs: Vec<[f32; BRAIN_INPUTS]> = self
            .cells
            .par_iter()
            .enumerate()
            .map(|(i, cell)| {
                let own = inputs_vec[i];
                crate::pool_bonded_sensors(cell, &own, |partner_id| {
                    let idx = id_to_idx.get(&partner_id).copied()?;
                    if idx == i {
                        return None;
                    }
                    Some(inputs_vec[idx])
                })
            })
            .collect();

        // Phase 2: GPU forward batch (using pooled inputs).
        let mut hiddens = vec![[0.0_f32; BRAIN_HIDDEN]; n];
        let mut outputs = vec![[0.0_f32; BRAIN_OUTPUTS]; n];
        {
            let gpu = self
                .gpu
                .as_mut()
                .expect("brain_act_gpu called without gpu");
            gpu.forward_batch(
                &pooled_inputs,
                self.cells.iter().map(|c| &c.genome.brain),
                &mut hiddens,
                &mut outputs,
                LATERAL_INHIBITION_ALPHA,
            );
        }

        // Phase 3: par_iter_mut writes back + apply_brain_motor.
        self.cells
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, cell)| {
                cell.last_inputs = pooled_inputs[i];
                cell.last_hidden = hiddens[i];
                cell.last_outputs = outputs[i];
                cell.apply_brain_motor(&outputs[i], dt);
            });
    }

    /// Sprint 94: pre-brain pass. Compute `pooled_hidden` per cell = mean
    /// `last_hidden` over self + bonded partners (1-hop). Cluster cells
    /// získají shared memory přes bond network. Solo cells: pooled == self.
    /// O(n × avg_bonds) — pro pop ~500 a avg_bonds < 1 je negligible cost.
    fn pool_bonded_hidden(&mut self) {
        if self.cells.is_empty() {
            return;
        }
        self.hidden_snapshot_scratch.clear();
        self.hidden_snapshot_scratch.reserve(self.cells.len());
        self.hidden_snapshot_scratch
            .extend(self.cells.iter().map(|c| c.last_hidden));
        let id_to_idx = &self.id_to_idx_scratch;
        let snapshot = &self.hidden_snapshot_scratch;
        for (i, cell) in self.cells.iter_mut().enumerate() {
            let pooled = crate::pool_bonded_hidden(cell, |partner_id| {
                let idx = id_to_idx.get(&partner_id).copied()?;
                if idx == i {
                    return None;
                }
                Some(snapshot[idx])
            });
            cell.pooled_hidden = pooled;
        }
    }

    /// Pre-brain pass: aggregate bonded peers' last_outputs message channels
    /// into cell.bonded_inbox. Mirrors pool_bonded_hidden flow (snapshot →
    /// per-cell mean over partners). Sensors then read inbox into brain inputs.
    fn pool_bond_messages(&mut self) {
        if self.cells.is_empty() {
            return;
        }
        let snapshot: Vec<[f32; crate::BRAIN_OUTPUTS]> =
            self.cells.iter().map(|c| c.last_outputs).collect();
        let id_to_idx = &self.id_to_idx_scratch;
        for (i, cell) in self.cells.iter_mut().enumerate() {
            let inbox = crate::pool_bond_messages(cell, |partner_id| {
                let idx = id_to_idx.get(&partner_id).copied()?;
                if idx == i {
                    return None;
                }
                Some(snapshot[idx])
            });
            cell.bonded_inbox = inbox;
        }
    }

    fn brain_act(&mut self, dt: f32) {
        // Sprint 43: grid build O(N), neighbor query O(k) per cell, par_iter
        // přes cells. Per-cell práce (sensor gather + brain forward + motor) je
        // write-only do vlastní cell — par-safe bez reduction.
        self.cell_grid.rebuild(
            self.cells
                .iter()
                .enumerate()
                .map(|(i, c)| (i, c.position, c.phenotype.effective_radius())),
        );
        self.food_grid.rebuild(
            self.foods
                .iter()
                .enumerate()
                .map(|(i, f)| (i, f.position, f.kind)),
        );

        let cell_grid = &self.cell_grid;
        let food_grid = &self.food_grid;
        let smell = &self.smell;
        let pheromone_fields = &self.pheromone_fields;
        let vibration = &self.vibration;
        let coop_foods = &self.coop_foods;
        let tick = self.clock.tick;
        let gen = self.clock.generation;
        let obstacles = self.obstacles.as_ref();

        // Sprint 97: dvojfáze pro cluster sensor pooling. Phase 1: gather + apply
        // own gains. Phase 2: pool max-magnitude přes bond network + brain forward.
        self.inputs_scratch.clear();
        self.inputs_scratch
            .resize(self.cells.len(), [0.0; BRAIN_INPUTS]);
        self.cells
            .par_iter_mut()
            .zip(self.inputs_scratch.par_iter_mut())
            .enumerate()
            .for_each(|(i, (cell, inputs_slot))| {
                let pos = cell.position;
                let vision_r = cell.genome.vision_radius;
                let vr2 = vision_r * vision_r;
                let fov = cell.genome.vision_fov;
                let skip_cone = fov >= crate::MAX_VISION_FOV;
                let cos_fov = fov.cos();
                let fwd = crate::forward_vector(cell.heading, cell.pitch);

                let mut best_food: Option<[f32; 3]> = None;
                let mut best_food_d2 = f32::MAX;
                food_grid.for_each_in_radius_toroidal(pos, vision_r, WORLD_HALF, |_id, fp, _| {
                    let d = crate::min_image_delta(pos, fp, WORLD_HALF);
                    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if d2 > vr2 || d2 >= best_food_d2 {
                        return;
                    }
                    if !skip_cone && !crate::fov_cone_accept(d, d2, fwd, cos_fov) {
                        return;
                    }
                    if !crate::los_clear(obstacles, pos, fp) {
                        return;
                    }
                    best_food_d2 = d2;
                    best_food = Some(d);
                });
                // Sprint 128: coop food candidates injected do same nearest_food
                // selection. Linear scan — typický coop_foods.len() ≤ 8.
                for coop in coop_foods.iter() {
                    let d = crate::min_image_delta(pos, coop.position, WORLD_HALF);
                    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if d2 > vr2 || d2 >= best_food_d2 {
                        continue;
                    }
                    if !skip_cone && !crate::fov_cone_accept(d, d2, fwd, cos_fov) {
                        continue;
                    }
                    if !crate::los_clear(obstacles, pos, coop.position) {
                        continue;
                    }
                    best_food_d2 = d2;
                    best_food = Some(d);
                }

                let mut best_cell: Option<([f32; 3], f32)> = None;
                let mut best_cell_d2 = f32::MAX;
                let mut neighbors_in_vision: u32 = 0;
                cell_grid.for_each_in_radius_toroidal(pos, vision_r, WORLD_HALF, |id, op, oradius| {
                    if id == i {
                        return;
                    }
                    let d = crate::min_image_delta(pos, op, WORLD_HALF);
                    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if d2 > vr2 {
                        return;
                    }
                    if !skip_cone && !crate::fov_cone_accept(d, d2, fwd, cos_fov) {
                        return;
                    }
                    if !crate::los_clear(obstacles, pos, op) {
                        return;
                    }
                    neighbors_in_vision += 1;
                    if d2 < best_cell_d2 {
                        best_cell_d2 = d2;
                        best_cell = Some((d, oradius));
                    }
                });

                let pos_xyz = [pos[0], pos[1], pos[2]];
                let smell_grad = smell.gradient_at(pos_xyz, SMELL_SAMPLE_EPSILON);
                let mut pheromone_grads = [[0.0_f32; 3]; N_PHEROMONE_CHANNELS];
                for ch in 0..N_PHEROMONE_CHANNELS {
                    pheromone_grads[ch] =
                        pheromone_fields[ch].gradient_at(pos_xyz, PHEROMONE_SAMPLE_EPSILON);
                }
                let temperature_local =
                    crate::temperature_at_z(pos[2], WORLD_HALF, tick, gen);
                let vibration_grad =
                    vibration.gradient_at(pos_xyz, VIBRATION_SAMPLE_EPSILON);
                let vibration_amp = vibration.sample(pos_xyz);
                let sensors = crate::BrainSensors {
                    nearest_food: best_food,
                    nearest_cell: best_cell,
                    neighbors_in_vision,
                    smell_grad,
                    pheromone_grads,
                    temperature_local,
                    vibration_grad,
                    vibration_amp,
                    whisker_distances: cell.last_whisker_distances,
                };

                cell.apply_shell_absorb(dt);
                // eat_food skip optim: cache squared distance k nejbližšímu food
                // (vision-radius scope) pro pozdější `eat_food` early skip.
                cell.last_best_food_d2 = best_food_d2;
                let mut inputs = crate::populate_brain_inputs(cell, &sensors, vision_r);
                crate::apply_sensor_gains(&mut inputs, &cell.genome.sensor_gains);
                *inputs_slot = inputs;
            });

        let id_to_idx = &self.id_to_idx_scratch;
        let inputs_scratch = &self.inputs_scratch;

        self.cells
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, cell)| {
                let own = inputs_scratch[i];
                let pooled = crate::pool_bonded_sensors(cell, &own, |partner_id| {
                    let idx = id_to_idx.get(&partner_id).copied()?;
                    if idx == i {
                        return None;
                    }
                    Some(inputs_scratch[idx])
                });
                let (hidden, outputs) = cell
                    .genome
                    .brain
                    .forward_with_state(&pooled, LATERAL_INHIBITION_ALPHA);
                cell.last_inputs = pooled;
                cell.last_hidden = hidden;
                cell.last_outputs = outputs;
                cell.apply_brain_motor(&outputs, dt);
            });
    }

    fn step(&mut self, dt: f32) {
        // Sprint 63: GPU step je fused do `brain_act_gpu_full` (Phase 9).
        // Tato fáze je v `--gpu-full` no-op; kinematics + drag + energy +
        // bounce už proběhly přes StepGpu shader, position/velocity/age/
        // cooldown/energy jsou writebackd v batch readback Phase 10-11.
        {
            if self.gpu_full.is_some() {
                let _ = dt;
                return;
            }
        }
        // Sprint 57: stejně jako apply_morph, ~16 us sekvenčně vs ~30 us
        // paralelně — work per cell je příliš malý pro rayon. Sekvenční win.
        // Sprint 112: per-cell climate_offset z aktivních ClimateShift shocků
        // se počítá tady (jednou před apply_energy_costs). Default = 0.0 když
        // events prázdné → step_with_climate je byte-identical s step().
        let tick = self.clock.tick;
        let gen = self.clock.generation;
        let events = self.active_climate_events.as_slice();
        let ctx = crate::ThermalCtx::for_tick(tick, gen);
        let obstacles = self.obstacles.as_ref();
        for cell in &mut self.cells {
            let climate_offset = crate::climate_shock_offset(
                events,
                gen,
                [cell.position[0], cell.position[1]],
                WORLD_HALF,
            );
            cell.step_with_thermal_maze(
                dt,
                WORLD_HALF,
                &ctx,
                &PHYSICS_CONFIG,
                climate_offset,
                obstacles,
            );
        }
    }

    /// Wave 3 curriculum: returns the maze difficulty active at `gen` per
    /// `maze_curriculum`, or `None` if curriculum is empty / `gen` is past
    /// the last stage. The current `obstacles.difficulty` is what's already
    /// running; comparing against this value tells us whether to rebuild.
    pub fn difficulty_for_generation(&self, gen: u64) -> Option<MazeDifficulty> {
        for &(diff, end_gen) in &self.maze_curriculum {
            if gen < end_gen {
                return Some(diff);
            }
        }
        None
    }

    /// Wave 3: rebuild `obstacles` + masks at a new difficulty (called from
    /// the per-gen ramp check or whenever the curriculum advances). Cells
    /// keep their positions; the next collision tick pushes any caught in a
    /// freshly-spawned wall back out. Goal-tracking state (`goal_first_reach_tick`,
    /// `goal_unique_reachers_gen`) is preserved across rebuilds — comparing
    /// time-to-goal across difficulty stages is part of the curriculum signal.
    pub fn rebuild_maze(&mut self, difficulty: MazeDifficulty, base_seed: u64) {
        self.maze_seed_step = self.maze_seed_step.wrapping_add(1);
        let seed = base_seed.wrapping_add(self.maze_seed_step);
        let field = ObstacleField::new_maze(WORLD_HALF, seed, difficulty);
        self.smell_mask = Some(
            field.mask_for_grid([SMELL_GRID_RES, SMELL_GRID_RES, SMELL_GRID_RES_Z]),
        );
        self.pheromone_masks = std::array::from_fn(|_| {
            Some(field.mask_for_grid([
                PHEROMONE_GRID_RES,
                PHEROMONE_GRID_RES,
                PHEROMONE_GRID_RES_Z,
            ]))
        });
        self.vibration_mask = Some(field.mask_for_grid([
            VIBRATION_GRID_RES,
            VIBRATION_GRID_RES,
            VIBRATION_GRID_RES_Z,
        ]));
        // Wave 4: re-upload mask + per-grid masks to gpu_full so the
        // in-shader collision and masked diffusion see the new layout from
        // this tick onward.
        if let Some(gpu) = self.gpu_full.as_mut() {
            let packed = field.packed_for_gpu();
            let smell_mask =
                field.mask_for_grid([SMELL_GRID_RES, SMELL_GRID_RES, SMELL_GRID_RES_Z]);
            let phero_mask = field.mask_for_grid([
                PHEROMONE_GRID_RES,
                PHEROMONE_GRID_RES,
                PHEROMONE_GRID_RES_Z,
            ]);
            let vib_mask = field.mask_for_grid([
                VIBRATION_GRID_RES,
                VIBRATION_GRID_RES,
                VIBRATION_GRID_RES_Z,
            ]);
            gpu.step.upload_maze(&packed);
            gpu.sensor.upload_maze(&packed);
            gpu.food_spawn.upload_obstacle(&packed);
            gpu.smell.upload_obstacle_mask(&smell_mask);
            gpu.pheromone.upload_obstacle_mask(&phero_mask);
            gpu.vibration.upload_obstacle_mask(&vib_mask);
        }
        self.obstacles = Some(field);
    }


    /// Wave 2 episodic novelty pass. For each cell, bin its current position
    /// into a coarse novelty grid; if not in the cell's recent visit history,
    /// fire a small Hebbian reward against the accumulated eligibility trace
    /// (Wave 3 — was instantaneous pre·post in Wave 2). Encourages
    /// exploration; trace-based reward credits the action that placed the
    /// cell here even if it was several ticks back.
    ///
    /// Wave 7: in `--gpu-full` mode the CPU brain mutation is a no-op
    /// (CPU brain is stale shadow), so we instead pack novelty rewards
    /// into a per-cell vector and route them through the GPU trace-based
    /// reward dispatch. CPU and GPU paths now both credit novelty against
    /// the same eligibility trace.
    pub fn apply_episodic_novelty(&mut self) {
        use crate::LEARNING_RATE;
        let half = WORLD_HALF;
        if self.gpu_full.is_some() {
            // Decide novelty per cell first (read-only on cell state); then
            // apply rewards via the GPU dispatch in one pass.
            let mut acc = RewardAccumulator::with_capacity(self.cells.len() / 8);
            for (i, cell) in self.cells.iter_mut().enumerate() {
                let v = Cell::novelty_voxel_index(cell.position, half);
                if cell.check_novelty(v) {
                    // Sprint 187: per-cell reward weight replaces the global
                    // NOVELTY_REWARD_MAGNITUDE — exploration drive becomes
                    // an evolvable trait.
                    let w = cell.genome.reward_weights[RewardKind::Novelty as usize];
                    acc.push(i, RewardKind::Novelty, w);
                }
            }
            if !acc.is_empty() {
                let n = self.cells.len();
                let mut rewards: Vec<f32> = vec![0.0; n];
                acc.flush_to(&mut rewards, FlushMode::SumAndClamp);
                let gpu = self.gpu_full.as_ref().unwrap();
                gpu.cells.upload_rewards(&rewards);
                gpu.hebbian
                    .dispatch_apply_reward_persistent(&gpu.cells, n, LEARNING_RATE);
            gpu.stdp_apply.dispatch(
                &gpu.cells,
                n,
                self.clock.tick as u32,
                DEFAULT_STDP_A_PLUS,
                DEFAULT_STDP_A_MINUS,
            );
            }
            return;
        }
        self.cells.par_iter_mut().for_each(|cell| {
            let v = Cell::novelty_voxel_index(cell.position, half);
            if !cell.check_novelty(v) {
                return;
            }
            let last_hidden = cell.last_hidden;
            let last_outputs = cell.last_outputs;
            // Sprint 136: per-cell learning rate replaces the global const.
            let lr = cell.genome.learning_rate;
            // Sprint 187: per-cell novelty weight.
            let reward = cell.genome.reward_weights[RewardKind::Novelty as usize];
            cell.genome.brain.hebbian_apply_reward(
                &last_hidden,
                &last_outputs,
                reward,
                lr,
            );
        });
    }

    /// Per-tick whisker pass (Sprint 195): raycast the maze, then integrate
    /// the per-whisker spring-damper so `cell.last_whisker_distances` holds
    /// the *sensed* value the brain + renderer consume — not the raw raycast.
    /// No-op (leaves defaults at 1.0 = "clear") when `obstacles` is `None`,
    /// matching the maze-inactive GPU gate in `sensor_gather.wgsl`.
    ///
    /// The per-cell loop index `i` seeds the transduction noise and must
    /// equal the GPU dispatch index for the same cell — both iterate
    /// `self.cells` in order and this runs immediately before the GPU upload.
    pub fn update_whiskers(&mut self) {
        let Some(field) = self.obstacles.as_ref() else {
            return;
        };
        let tick = self.clock.tick as u32;
        self.cells.par_iter_mut().enumerate().for_each(|(i, cell)| {
            let raw =
                field.whisker_distances(cell.position, cell.heading, cell.pitch);
            for k in 0..crate::WHISKER_COUNT {
                let noise = crate::whisker_noise(i as u32, tick, k as u32)
                    * crate::WHISKER_NOISE_AMPLITUDE;
                cell.last_whisker_distances[k] = crate::whisker_step(
                    &mut cell.whisker_deflection[k],
                    &mut cell.whisker_deflection_vel[k],
                    raw[k],
                    noise,
                );
            }
        });
    }

    /// Per-tick navigation metric tracker. No-op when `obstacles` is None.
    /// Counts cells currently in the goal zone (per-tick), records first-
    /// reach tick per `cell_id` (lifelong), and tracks unique reachers in
    /// the current generation (cleared at gen end by `csv::write_stats`).
    pub fn track_goal_metrics(&mut self) {
        let Some(field) = self.obstacles.as_ref() else {
            return;
        };
        let mut now_count: u64 = 0;
        let tick = self.clock.tick;
        for cell in &self.cells {
            if !field.at_goal(cell.position) {
                continue;
            }
            now_count += 1;
            self.goal_unique_reachers_gen.insert(cell.cell_id);
            self.goal_first_reach_tick
                .entry(cell.cell_id)
                .or_insert(tick);
        }
        self.goal_zone_ticks_gen += now_count;
    }

    /// Sprint 112: per-cell climate offset helper, sdílený mezi tick hot path
    /// a `write_stats` (CSV column `shock_climate_offset`). Reads the full
    /// calendar (not the per-tick `active_climate_events` scratch) so
    /// direct test access doesn't require manually refreshing the scratch.
    /// CSV is per-gen-end so the O(events) scan is paid once per cell per
    /// gen, not per tick.
    pub fn climate_offset_at(&self, pos_xy: [f32; 2]) -> f32 {
        crate::climate_shock_offset(
            &self.events.events,
            self.clock.generation,
            pos_xy,
            WORLD_HALF,
        )
    }

    fn apply_food_gravity(&mut self, dt: f32) {
        for food in &mut self.foods {
            food.apply_gravity(dt, WORLD_HALF[2]);
        }
        // Sprint 42: aging + despawn expired food (value_factor ≤ 0).
        self.foods.retain_mut(|f| f.age_step());
    }

    fn apply_hazards(&mut self, dt: f32) {
        // Sprint 57: ~14 us sekvenčně vs ~27 us paralelně — work per cell je
        // jen map.sample + 2× scalar update, rayon overhead převáží.
        // Sprint 110: HazardPulse shocks násobí drain (default 1.0 = no-op).
        let gen = self.clock.generation;
        let tick = self.clock.tick;
        let events = self.active_hazard_events.as_slice();
        let mut acc = RewardAccumulator::with_capacity(self.cells.len() / 16);
        for (i, cell) in self.cells.iter_mut().enumerate() {
            let noise = self
                .map
                .sample([cell.position[0], cell.position[1], cell.position[2]]);
            let shock_mult = crate::hazard_shock_multiplier(
                cell.position,
                events,
                gen,
                tick,
                WORLD_HALF,
            );
            let drain = hazard_drain(noise) * dt * shock_mult;
            cell.energy -= drain;
            cell.damage_accum += drain;
            let in_hazard = drain > 0.0;
            if in_hazard {
                // Sprint 187: per-cell damage sensitivity replaces global.
                let w = cell.genome.reward_weights[RewardKind::Damage as usize];
                acc.push(i, RewardKind::Damage, -(drain * w));
            } else if cell.was_in_hazard_last_tick {
                // Transition-out: cell was taking drain last tick, isn't this
                // tick. Credit the brain's sensing+motor decision. Edge case:
                // a HazardPulse shock ending naturally fires this for every
                // cell still inside that tick — small noise vs the dominant
                // signal of cells actively moving out of static map hazards.
                let w = cell.genome.reward_weights[RewardKind::HazardAvoided as usize];
                acc.push(i, RewardKind::HazardAvoided, w);
            }
            cell.was_in_hazard_last_tick = in_hazard;
        }
        if !acc.is_empty() && self.gpu_full.is_some() {
            let n = self.cells.len();
            let mut rewards: Vec<f32> = vec![0.0; n];
            acc.flush_to(&mut rewards, FlushMode::SumAndClamp);
            let gpu = self.gpu_full.as_ref().unwrap();
            gpu.cells.upload_rewards(&rewards);
            gpu.hebbian
                .dispatch_apply_reward_persistent(&gpu.cells, n, LEARNING_RATE);
            gpu.stdp_apply.dispatch(
                &gpu.cells,
                n,
                self.clock.tick as u32,
                DEFAULT_STDP_A_PLUS,
                DEFAULT_STDP_A_MINUS,
            );
        }
    }

    /// Wave H: GPU replacement for the CPU Phase 1 broad-phase pair loop.
    /// Fills `deltas_scratch`, `velocity_deltas_scratch` and
    /// `contact_lists_scratch` from a single GPU dispatch. Phase 2 / 3 / 4
    /// of `resolve_collisions` consume these scratch buffers identically
    /// regardless of which path produced them.
    /// Synchronous dispatch + await — single function that owns the
    /// shared encoder, submits hash + collision in one pass, polls for
    /// completion, and populates `deltas_scratch` / `velocity_deltas_scratch`
    /// / `contact_lists_scratch` for the caller's Phase 2/3/4.
    fn resolve_collisions_gpu_dispatch_and_drain(&mut self) {
        let n = self.cells.len();
        if n == 0 {
            return;
        }
        let slots = MAX_BONDS_PER_CELL;
        let total = n * slots;
        let id_to_idx = &self.id_to_idx_scratch;
        let cells = &self.cells;

        let gpu = self.gpu_full.as_mut().expect("gpu_full Some");
        let s = &mut gpu.scratch;
        s.lt_positions.clear();
        s.lt_eff_radii.clear();
        s.lt_max_axes.clear();
        s.lt_adhesion_types.clear();
        for c in cells.iter() {
            s.lt_positions.push(c.position);
            s.lt_eff_radii.push(c.phenotype.effective_radius());
            s.lt_max_axes.push(c.phenotype.max_axis());
            s.lt_adhesion_types.push(c.genome.adhesion_type as u32);
        }
        s.lt_partner_idx.clear();
        s.lt_partner_idx.resize(total, -1);
        s.lt_bond_rest.clear();
        s.lt_bond_rest.resize(total, 0.0);
        s.lt_bond_stiff.clear();
        s.lt_bond_stiff.resize(total, 0.0);
        s.lt_bond_damp.clear();
        s.lt_bond_damp.resize(total, 0.0);
        for (i, c) in cells.iter().enumerate() {
            for (k, slot) in c.bonds.iter().enumerate() {
                if let Some(b) = slot {
                    if let Some(&j) = id_to_idx.get(&b.other_cell_id) {
                        let idx = i * slots + k;
                        s.lt_partner_idx[idx] = j as i32;
                        s.lt_bond_rest[idx] = b.rest_length;
                        s.lt_bond_stiff[idx] = b.stiffness;
                        s.lt_bond_damp[idx] = b.damping;
                    }
                }
            }
        }

        gpu.profiler.resume();
        let mut encoder = gpu
            .cells
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resolve-collisions-encoder"),
            });
        gpu.cell_hash
            .dispatch_into(&mut encoder, &s.lt_positions);
        if gpu.profiler.enabled() {
            // Flush the hash on its own command buffer so it lands in a
            // distinct bucket. Queue submissions execute in order, so the
            // collision pass still reads a fresh hash.
            gpu.cells.queue().submit(Some(encoder.finish()));
            gpu.profiler.mark("collision_hash");
            encoder = gpu
                .cells
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("resolve-collisions-encoder"),
                });
        }
        gpu.collision.dispatch_into_cells(
            &mut encoder,
            &gpu.cells,
            n,
            &s.lt_eff_radii,
            &s.lt_max_axes,
            &s.lt_adhesion_types,
            &s.lt_partner_idx,
            &s.lt_bond_rest,
            &s.lt_bond_stiff,
            &s.lt_bond_damp,
            &gpu.cell_hash,
        );
        gpu.cells.queue().submit(Some(encoder.finish()));
        gpu.profiler.mark("collision_shader");
        let result = gpu.collision.await_result(n);
        gpu.profiler.mark("collision_readback");

        let max_contacts = crate::MAX_COLLISION_CONTACTS_PER_CELL as usize;
        for i in 0..n {
            self.deltas_scratch[i] = result.position_deltas[i];
            self.velocity_deltas_scratch[i] = result.velocity_deltas[i];
        }
        for i in 0..n {
            let count = (result.contact_count[i] as usize).min(max_contacts);
            let base = i * max_contacts;
            let cell_id_i = self.cells[i].cell_id;
            for s in 0..count {
                let j = result.contact_partners[base + s] as usize;
                if j >= n {
                    continue;
                }
                let cell_id_j = self.cells[j].cell_id;
                if cell_id_i < cell_id_j {
                    self.contact_lists_scratch[i].push(cell_id_j);
                } else if cell_id_j < cell_id_i {
                    self.contact_lists_scratch[j].push(cell_id_i);
                }
            }
        }
    }

    fn resolve_collisions(&mut self) {
        // Sprint 43: grid + rayon. Δ pro každé i je write-only do vlastního
        // slotu. Max search radius = CELL_RADIUS × (radius_i + max_neighbor_r);
        // vyhledáme přes effective_radius_i + GRID_CELL_SIZE konzervativně.
        // Sprint 65: rozšířeno o velocity damping (inelastic, restitution=0).
        // Sprint 66: rozšířeno o (1) differential adhesion (soft attractive
        // force same-type, mírná repulze cross-type, mimo kontakt) a
        // (2) persistent spring bonds (per-cell list, hookean spring + damping).
        // Plus per-pair contact tick tracker pro hybrid bond formation.
        let n = self.cells.len();
        // CPU `cell_grid` is consumed only by the legacy CPU `brain_act` /
        // `brain_act_gpu` paths — both unreachable when `gpu_full` is Some
        // (run_brain_act dispatches to `brain_act_gpu_full` instead). In the
        // gpu-full hot path this rebuild was pure overhead: 1000-cell
        // SpatialGrid rebuild every tick that nothing read. Guard it.
        if self.gpu_full.is_none() {
            self.cell_grid.rebuild(
                self.cells
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, c.position, c.phenotype.effective_radius())),
            );
        }
        self.deltas_scratch.clear();
        self.deltas_scratch.resize(n, [0.0, 0.0, 0.0]);
        self.velocity_deltas_scratch.clear();
        self.velocity_deltas_scratch.resize(n, [0.0, 0.0, 0.0]);
        // P1-#6: persistent contact_lists. Resize a clear inner Vecs (zachová
        // capacity), pak par_iter_mut zip — žádné fresh `Vec<Vec<u64>>` per tick.
        if self.contact_lists_scratch.len() < n {
            self.contact_lists_scratch.resize_with(n, Vec::new);
        }
        for inner in self.contact_lists_scratch.iter_mut().take(n) {
            inner.clear();
        }

        // GPU collision shader covers the broad-phase pair loop
        // (depenetration + velocity damping + adhesion + spring bond forces
        // + contact event detection); CPU Phase 2/3/4 below consume the
        // resulting deltas / contact lists.
        // Synchronous dispatch+drain: 1-tick-lag pipelining was tried and
        // reverted — applying tick-N deltas after another tick-N+1 step
        // integration leaves persistent overlap that predation feeds on.
        self.resolve_collisions_gpu_dispatch_and_drain();

        // Phase 2: sequential apply position/velocity deltas + contact tracker
        // update + bond pruning + bond formation. Vše drží borrow checker happy
        // tím, že jednotlivé fields self.cells iterujeme v jednom průchodu.
        let dt = 1.0 / FIXED_TIMESTEP_HZ;
        for ((cell, delta), vel_delta) in self
            .cells
            .iter_mut()
            .zip(self.deltas_scratch.iter())
            .zip(self.velocity_deltas_scratch.iter())
        {
            cell.position[0] += delta[0];
            cell.position[1] += delta[1];
            cell.position[2] += delta[2];
            cell.velocity[0] += vel_delta[0];
            cell.velocity[1] += vel_delta[1];
            cell.velocity[2] += vel_delta[2];
            // Sprint 188: cap velocity to `max_speed × VELOCITY_CAP_FACTOR`.
            // Bond impulses (`vel_delta` above) are applied without a `dt`
            // scale, so an overstretched bond can drive `|v|` well past
            // what drag can balance — observed 3.6× max_speed in a real
            // run. Hard cap preserves direction, scales magnitude back to
            // the envelope. See `VELOCITY_CAP_FACTOR` for rationale.
            let v_mag_sq = cell.velocity[0] * cell.velocity[0]
                + cell.velocity[1] * cell.velocity[1]
                + cell.velocity[2] * cell.velocity[2];
            let v_cap = cell.genome.max_speed * VELOCITY_CAP_FACTOR;
            let v_cap_sq = v_cap * v_cap;
            if v_mag_sq > v_cap_sq && v_mag_sq > 1e-8 {
                let scale = v_cap / v_mag_sq.sqrt();
                cell.velocity[0] *= scale;
                cell.velocity[1] *= scale;
                cell.velocity[2] *= scale;
            }
        }

        // Sprint 66: bond pruning + age + maintenance + explicit-break.
        // Snapshot positions po Phase-2 apply; pak per-cell projdi bonds:
        //  1. brain output[9] < BREAK_THRESHOLD → drop all bonds této cells.
        //  2. cíl bondu zemřel (id chybí v map) → drop.
        //  3. distance > rest × BREAK_FACTOR → drop (overstretch).
        //  4. jinak inkrement age + accumulate per-cell bond count pro maintenance.
        // Phase 3 / 4 bond lookups need this map (Phase 1 binds its own
        // shorter-lived copy inside the CPU branch above).
        let id_to_idx = &self.id_to_idx_scratch;
        let mut bonds_broken_this_tick: u64 = 0;
        // P1-#7: persistent positions snapshot scratch.
        self.positions_snapshot_scratch.clear();
        self.positions_snapshot_scratch
            .extend(self.cells.iter().map(|c| c.position));
        let positions_snapshot = self.positions_snapshot_scratch.as_slice();
        for i in 0..self.cells.len() {
            let outputs_9 = self.cells[i].last_outputs[9];
            let explicit_break = outputs_9 < BOND_BREAK_THRESHOLD;
            let pos_i = positions_snapshot[i];
            let mut bond_count = 0_usize;
            for slot in 0..MAX_BONDS_PER_CELL {
                let Some(bond) = self.cells[i].bonds[slot] else { continue };
                if explicit_break {
                    self.cells[i].bonds[slot] = None;
                    bonds_broken_this_tick += 1;
                    continue;
                }
                let Some(&j_idx) = id_to_idx.get(&bond.other_cell_id) else {
                    self.cells[i].bonds[slot] = None;
                    bonds_broken_this_tick += 1;
                    continue;
                };
                let pos_j = positions_snapshot[j_idx];
                let d_vec = crate::min_image_delta(pos_j, pos_i, WORLD_HALF);
                let d = (d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2]).sqrt();
                if d > bond.rest_length * crate::BOND_BREAK_FACTOR || d <= f32::EPSILON {
                    self.cells[i].bonds[slot] = None;
                    bonds_broken_this_tick += 1;
                    continue;
                }
                if let Some(b) = self.cells[i].bonds[slot].as_mut() {
                    b.age_ticks = b.age_ticks.saturating_add(1);
                }
                bond_count += 1;
            }
            if bond_count > 0 {
                let cost = bond_count as f32 * BOND_MAINTENANCE_PER_SEC * dt;
                self.cells[i].energy -= cost;
            }
        }
        // Sprint 66: contact_progress update from this-tick's collected
        // contact pairs. Increment for new contacts; decrement-or-prune for
        // pairs not seen this tick. Then attempt bond formation pro kandidáty
        // ≥ BOND_FORM_TICKS, kteří mají match adhesion_type + oba bond_active.
        // P1-#7: reuse persistent seen_pairs / bond_candidates scratch.
        self.seen_pairs_scratch.clear();
        // Accumulate per-pair "i side bond_active" flag — both sides must be true.
        // Cell i shipped (other_id, bond_active_i); for cell j we look it up
        // separately by checking cells[id_to_idx[j]].last_outputs[9].
        for (i, contacts) in self.contact_lists_scratch.iter().take(n).enumerate() {
            let cell_id_i = self.cells[i].cell_id;
            for &other_id in contacts.iter() {
                let key = (cell_id_i, other_id);
                self.seen_pairs_scratch.insert(key);
                let entry = self.contact_progress.entry(key).or_insert(0);
                *entry = entry.saturating_add(1);
            }
        }
        // Decay / prune unseen pairs. Použijeme retain s decrementing.
        let seen_pairs = &self.seen_pairs_scratch;
        self.contact_progress.retain(|key, ticks| {
            if seen_pairs.contains(key) {
                true
            } else {
                if *ticks > CONTACT_DECAY_TICKS {
                    *ticks -= CONTACT_DECAY_TICKS;
                    true
                } else {
                    false
                }
            }
        });
        // Sprint 66: attempt bond formation pro pairs co dosáhly thresholdu.
        // Kontrola: same adhesion_type, oba bond_active, oba mají volný slot.
        // Při úspěchu: zápis bondu na obou stranách + cost na obě cells.
        // Sprint 135: collect BondFormed rewards along the way; dispatch
        // once after the formation loop.
        let mut bond_acc = RewardAccumulator::new();
        let mut bonds_formed_this_tick: u64 = 0;
        self.bond_candidates_scratch.clear();
        for (&(a, b), &ticks) in self.contact_progress.iter() {
            if ticks >= BOND_FORM_TICKS {
                self.bond_candidates_scratch.push((a, b));
            }
        }
        // Take to drop borrow on self before mutating cells inside the loop.
        let candidates = std::mem::take(&mut self.bond_candidates_scratch);
        let mut bonded_pairs = std::mem::take(&mut self.bonded_pairs_scratch);
        bonded_pairs.clear();
        for cell in self.cells.iter() {
            for bond_opt in cell.bonds.iter() {
                if let Some(b) = bond_opt {
                    let pair = if cell.cell_id < b.other_cell_id {
                        (cell.cell_id, b.other_cell_id)
                    } else {
                        (b.other_cell_id, cell.cell_id)
                    };
                    bonded_pairs.insert(pair);
                }
            }
        }
        for (id_a, id_b) in candidates.iter().copied() {
            let Some(&i_a) = id_to_idx.get(&id_a) else { continue };
            let Some(&i_b) = id_to_idx.get(&id_b) else { continue };
            if i_a == i_b {
                continue;
            }
            // Cells must agree on adhesion_type + signal.
            if self.cells[i_a].genome.adhesion_type
                != self.cells[i_b].genome.adhesion_type
            {
                continue;
            }
            if self.cells[i_a].last_outputs[9] <= BOND_FORM_THRESHOLD
                || self.cells[i_b].last_outputs[9] <= BOND_FORM_THRESHOLD
            {
                continue;
            }
            // Volné sloty?
            let slot_a = self.cells[i_a]
                .bonds
                .iter()
                .position(|b| b.is_none());
            let slot_b = self.cells[i_b]
                .bonds
                .iter()
                .position(|b| b.is_none());
            let (Some(sa), Some(sb)) = (slot_a, slot_b) else {
                continue;
            };
            // Skip if už bonded (např. po prior tick — defensive, contact_progress
            // by se měl reseta při formaci, ale i bez toho ne-duplikujeme).
            let pair = if id_a < id_b { (id_a, id_b) } else { (id_b, id_a) };
            if bonded_pairs.contains(&pair) {
                continue;
            }
            let pos_a = positions_snapshot[i_a];
            let pos_b = positions_snapshot[i_b];
            let d_vec = crate::min_image_delta(pos_b, pos_a, WORLD_HALF);
            let dist =
                (d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2]).sqrt();
            let rest = dist * BOND_REST_LENGTH_SLACK;
            // Sprint 68: per-bond stiffness/damping = mean obou cells' genes.
            let stiffness = (self.cells[i_a].genome.bond_stiffness
                + self.cells[i_b].genome.bond_stiffness)
                * 0.5;
            let damping = (self.cells[i_a].genome.bond_damping
                + self.cells[i_b].genome.bond_damping)
                * 0.5;
            self.cells[i_a].bonds[sa] = Some(Bond {
                other_cell_id: id_b,
                rest_length: rest,
                stiffness,
                damping,
                age_ticks: 0,
            });
            self.cells[i_b].bonds[sb] = Some(Bond {
                other_cell_id: id_a,
                rest_length: rest,
                stiffness,
                damping,
                age_ticks: 0,
            });
            // One-shot cost rozdělen na obě cells.
            self.cells[i_a].energy -= BOND_FORMATION_COST;
            self.cells[i_b].energy -= BOND_FORMATION_COST;
            bonds_formed_this_tick += 1;
            bonded_pairs.insert(pair);
            // Sprint 135: credit both new partners with a one-shot positive
            // reward — bond formation is the prerequisite for any
            // multicellular niche the cluster might evolve into.
            // Sprint 187: each cell's own bond-formed weight scales its credit.
            let w_a = self.cells[i_a].genome.reward_weights[RewardKind::BondFormed as usize];
            let w_b = self.cells[i_b].genome.reward_weights[RewardKind::BondFormed as usize];
            bond_acc.push(i_a, RewardKind::BondFormed, w_a);
            bond_acc.push(i_b, RewardKind::BondFormed, w_b);
            // Reset progress entry — nepokouší se znova ihned formovat.
            self.contact_progress.remove(&(id_a, id_b));
        }
        // Vrať buffer zpět (put-back) aby kapacita persistovala přes ticky.
        let mut candidates = candidates;
        candidates.clear();
        self.bond_candidates_scratch = candidates;
        self.bonded_pairs_scratch = bonded_pairs;
        self.bonds_formed_gen += bonds_formed_this_tick;
        self.bonds_broken_gen += bonds_broken_this_tick;

        if !bond_acc.is_empty() && self.gpu_full.is_some() {
            let n = self.cells.len();
            let mut rewards: Vec<f32> = vec![0.0; n];
            bond_acc.flush_to(&mut rewards, FlushMode::SumAndClamp);
            let gpu = self.gpu_full.as_ref().unwrap();
            gpu.cells.upload_rewards(&rewards);
            gpu.hebbian
                .dispatch_apply_reward_persistent(&gpu.cells, n, LEARNING_RATE);
            gpu.stdp_apply.dispatch(
                &gpu.cells,
                n,
                self.clock.tick as u32,
                DEFAULT_STDP_A_PLUS,
                DEFAULT_STDP_A_MINUS,
            );
        }

    }

    /// Predation dispatch — computes herd_counts + per-pair energy/damage
    /// deltas in a single GPU pass and applies them to cells. Sprint 186
    /// extends the shader with per-attacker and per-victim diagnostics
    /// (`attacker_event_count`, `attacker_gain_sum`, `victim_attacker_count`,
    /// first-K `victim_attackers`) so `bonded_attacks_gen`,
    /// `solo_attacks_gen`, `swarm_attacks_gen`, `pack_attacks_gen` and
    /// `attack_victims_gen` track real activity instead of staying at zero.
    fn predate(&mut self, rng: &mut impl Rng) {
        let n = self.cells.len();
        if n == 0 {
            return;
        }
        // Sprint 187: `size_ratio_threshold` and `attack_threshold` migrated
        // to per-cell genome (uploaded as `predation_size_ratios` /
        // `attack_gates` buffers); the uniform slot for attack_threshold is
        // kept as a pad to preserve UBO layout. `defense_floor` matches the
        // pre-S187 `bond_defense_factor` floor (0.4) so the max protection
        // remains capped.
        let params = PredateParamsGpu {
            num_cells: 0, // filled by compute()
            cell_size: crate::GRID_CELL_SIZE,
            cell_radius_const: crate::CELL_RADIUS,
            defense_floor: 0.4,
            herd_radius_sq: crate::HERD_RADIUS * crate::HERD_RADIUS,
            _pad_attack_threshold: 0.0,
            predation_gain: crate::PREDATION_GAIN_PER_TICK * self.predation_gain_mult,
            predation_drain: crate::PREDATION_DRAIN_PER_TICK * self.predation_drain_mult,
            spike_dot_threshold: crate::SPIKE_DOT_THRESHOLD,
            spike_bonus: crate::SPIKE_PREDATION_BONUS,
            dilution_k: crate::DILUTION_K,
            world_half_x: WORLD_HALF[0],
            world_half_y: WORLD_HALF[1],
            ..PredateParamsGpu::default()
        };
        let id_to_idx = &self.id_to_idx_scratch;
        let bond_cap = crate::BOND_DEFENSE_CAP as usize;
        let cells = &self.cells;
        let gpu = self.gpu_full.as_mut().expect("gpu_full Some");
        let s = &mut gpu.scratch;
        s.lt_positions.clear();
        s.lt_eff_radii.clear();
        s.lt_headings.clear();
        s.lt_pitches.clear();
        s.lt_attack_signals.clear();
        s.lt_spike_counts.clear();
        s.lt_spikes_packed.clear();
        s.lt_attack_gates.clear();
        s.lt_predation_size_ratios.clear();
        s.lt_defense_pool.clear();
        for cell in cells.iter() {
            s.lt_positions.push(cell.position);
            s.lt_eff_radii.push(cell.phenotype.effective_radius());
            s.lt_headings.push(cell.heading);
            s.lt_pitches.push(cell.pitch);
            s.lt_attack_signals.push(cell.last_outputs[6].max(0.0));
            s.lt_attack_gates.push(cell.genome.attack_gate);
            s.lt_predation_size_ratios
                .push(cell.genome.predation_size_ratio);
            let mut defense_sum = 0.0_f32;
            let mut bonded = 0usize;
            for bond_opt in cell.bonds.iter() {
                if bonded >= bond_cap {
                    break;
                }
                if let Some(bond) = bond_opt {
                    if let Some(&partner_idx) = id_to_idx.get(&bond.other_cell_id) {
                        if let Some(partner) = cells.get(partner_idx) {
                            defense_sum += partner.genome.defense_contribution;
                            bonded += 1;
                        }
                    }
                }
            }
            s.lt_defense_pool.push(defense_sum);
            let mut active = 0u32;
            for k in 0..crate::SPIKE_SLOTS {
                let spike = cell.phenotype.spikes[k];
                if spike.length > 0.0 {
                    active += 1;
                }
                s.lt_spikes_packed.push([
                    spike.length,
                    spike.azimuth_offset,
                    spike.elevation_offset,
                    spike.complexity,
                ]);
            }
            s.lt_spike_counts.push(active);
        }
        // Refresh GPU spatial hash with current (post-resolve_collisions)
        // positions before the predate dispatch.
        gpu.profiler.resume();
        gpu.cell_hash.dispatch(&s.lt_positions);
        let result = gpu.predate.compute(
            &s.lt_positions,
            &s.lt_eff_radii,
            &s.lt_headings,
            &s.lt_pitches,
            &s.lt_spikes_packed,
            &s.lt_spike_counts,
            &s.lt_attack_signals,
            &s.lt_attack_gates,
            &s.lt_predation_size_ratios,
            &s.lt_defense_pool,
            &gpu.cell_hash,
            params,
        );
        gpu.profiler.mark("predate");
        // Sprint 186: pack-hunting diagnostics. `attacker_event_count[i]`
        // and `attacker_gain_sum[i]` come from the GPU; bonded vs solo
        // partition uses `cell.n_bonds() >= 1`. Per-victim swarm + pack
        // detection scans the first `MAX_ATTACKERS_PER_VICTIM` recorded
        // attackers and looks for a mutually-bonded pair.
        let max_atk = crate::gpu::MAX_ATTACKERS_PER_VICTIM;
        for (i, cell) in self.cells.iter().enumerate() {
            let hits = result.attacker_event_count[i];
            if hits == 0 {
                continue;
            }
            let gain_sum = result.attacker_gain_sum[i] as f64;
            if cell.n_bonds() >= 1 {
                self.bonded_attacks_gen += hits as u64;
                self.bonded_attack_gain_sum_gen += gain_sum;
            } else {
                self.solo_attacks_gen += hits as u64;
                self.solo_attack_gain_sum_gen += gain_sum;
            }
        }
        let n_cells = self.cells.len();
        for j in 0..n_cells {
            let cnt = result.victim_attacker_count[j];
            if cnt == 0 {
                continue;
            }
            self.attack_victims_gen += 1;
            if cnt < 2 {
                continue;
            }
            self.swarm_attacks_gen += 1;
            let base = j * max_atk;
            let mut attackers: [usize; crate::gpu::MAX_ATTACKERS_PER_VICTIM] =
                [usize::MAX; crate::gpu::MAX_ATTACKERS_PER_VICTIM];
            let mut n_attackers = 0usize;
            for k in 0..max_atk {
                let raw = result.victim_attackers[base + k];
                if raw == 0 {
                    continue;
                }
                let idx = (raw - 1) as usize;
                if idx < n_cells {
                    attackers[n_attackers] = idx;
                    n_attackers += 1;
                }
            }
            let mut is_pack = false;
            'pair_search: for a in 0..n_attackers {
                for b in (a + 1)..n_attackers {
                    let ai = attackers[a];
                    let bi = attackers[b];
                    let bid = self.cells[bi].cell_id;
                    if self.cells[ai].bonds.iter().any(|slot| {
                        slot.map(|bond| bond.other_cell_id == bid).unwrap_or(false)
                    }) {
                        is_pack = true;
                        break 'pair_search;
                    }
                }
            }
            if is_pack {
                self.pack_attacks_gen += 1;
            }
        }

        // Sprint 196: predation-derived symbiont acquisition. For each
        // recorded (attacker, victim) hit, when victim is a bearer and
        // attacker is not, with P_capture the attacker absorbs a copy
        // (failed-digestion → endosymbiosis pathway). Victim keeps its
        // symbiont (Symbiont is Copy) so multiple attackers can each
        // independently roll without ordering dependencies.
        for victim_idx in 0..n_cells {
            let count = result.victim_attacker_count[victim_idx];
            if count == 0 {
                continue;
            }
            let Some(victim_sym) = self.cells[victim_idx].symbiont else {
                continue;
            };
            let base = victim_idx * max_atk;
            let slots = (count as usize).min(max_atk);
            for k in 0..slots {
                let raw = result.victim_attackers[base + k];
                if raw == 0 {
                    continue;
                }
                let attacker_idx = (raw - 1) as usize;
                if attacker_idx >= n_cells || attacker_idx == victim_idx {
                    continue;
                }
                if self.cells[attacker_idx].symbiont.is_some() {
                    continue;
                }
                if rng.random::<f32>() < SYMBIONT_CAPTURE_P {
                    let mut transferred = victim_sym;
                    transferred.deficit_streak = 0;
                    self.cells[attacker_idx].symbiont = Some(transferred);
                }
            }
        }

        // Sprint 134: attacker `Predation(+gain × scale)` + victim
        // `EscapedAttack(+magnitude)` once `under_attack_streak` clears
        // a damage-free tick. Push events through the per-tick accumulator
        // and dispatch a separate Hebbian apply pass on the GPU.
        let mut acc = RewardAccumulator::with_capacity(self.cells.len() / 8);
        for (i, cell) in self.cells.iter_mut().enumerate() {
            let gained = result.energy_delta[i];
            let damage_this_tick = result.damage_delta[i];

            cell.energy += gained;
            cell.damage_accum += damage_this_tick;

            if cell.escape_cooldown_ticks > 0 {
                cell.escape_cooldown_ticks -= 1;
            }

            // Sprint 187: per-cell reward sensitivities. PREDATION_REWARD_MAX
            // stays as the single-tick spike clamp (engineering safety, not
            // evolvable) — only the scale moves to the genome.
            let w_pred = cell.genome.reward_weights[RewardKind::Predation as usize];
            let w_dmg = cell.genome.reward_weights[RewardKind::Damage as usize];
            let w_esc = cell.genome.reward_weights[RewardKind::EscapedAttack as usize];

            if gained > 0.0 {
                let mag = (gained * w_pred).min(PREDATION_REWARD_MAX);
                acc.push(i, RewardKind::Predation, mag);
            }

            if damage_this_tick > 0.0 {
                cell.under_attack_streak = cell.under_attack_streak.saturating_add(1);
                // Sprint 135: damage = negative reinforcer. Magnitude is
                // negative; SumAndClamp at flush time keeps net per-cell
                // reward inside `[REWARD_CLAMP_MIN, REWARD_CLAMP_MAX]`.
                acc.push(
                    i,
                    RewardKind::Damage,
                    -(damage_this_tick * w_dmg),
                );
            } else {
                if cell.under_attack_streak >= ESCAPE_STREAK_THRESHOLD
                    && cell.escape_cooldown_ticks == 0
                {
                    acc.push(i, RewardKind::EscapedAttack, w_esc);
                    cell.escape_cooldown_ticks = ESCAPE_COOLDOWN_TICKS;
                } else if cell.under_attack_streak >= 1
                    && cell.escape_cooldown_ticks == 0
                {
                    // ThreatEscaped: brief-encounter survival — took at least
                    // one tick of damage, then escaped before the persistent
                    // EscapedAttack threshold. Shares the escape cooldown so
                    // only one of {EscapedAttack, ThreatEscaped} fires per
                    // window.
                    let w_threat = cell.genome.reward_weights
                        [RewardKind::ThreatEscaped as usize];
                    acc.push(i, RewardKind::ThreatEscaped, w_threat);
                    cell.escape_cooldown_ticks = ESCAPE_COOLDOWN_TICKS;
                }
                cell.under_attack_streak = 0;
            }
        }
        self.predation_events_gen += result.total_events as u64;

        if !acc.is_empty() {
            let n = self.cells.len();
            let mut rewards: Vec<f32> = vec![0.0; n];
            acc.flush_to(&mut rewards, FlushMode::SumAndClamp);
            let gpu = self.gpu_full.as_ref().unwrap();
            gpu.cells.upload_rewards(&rewards);
            gpu.hebbian
                .dispatch_apply_reward_persistent(&gpu.cells, n, LEARNING_RATE);
            gpu.stdp_apply.dispatch(
                &gpu.cells,
                n,
                self.clock.tick as u32,
                DEFAULT_STDP_A_PLUS,
                DEFAULT_STDP_A_MINUS,
            );
        }
    }

    /// 2D worlds skip the mechanic so pre-S197 2D smokes stay byte-identical.
    pub(crate) fn apply_symbiont_energy(&mut self, dt: f32) {
        let half_z = WORLD_HALF[2];
        if half_z <= 0.0 {
            return;
        }
        let z_range = 2.0 * half_z;
        let threshold = SYMBIONT_PHOTO_Z_THRESHOLD;
        let denom = (1.0 - threshold).max(f32::EPSILON);
        let upkeep = SYMBIONT_UPKEEP_PER_SEC * dt;
        let ctx = crate::ThermalCtx::for_tick(self.clock.tick, self.clock.generation);
        let mut sheds_this_pass: u64 = 0;
        for cell in self.cells.iter_mut() {
            if cell.symbiont.is_none() {
                continue;
            }
            let z_norm = ((cell.position[2] + half_z) / z_range).clamp(0.0, 1.0);
            let light = (z_norm - threshold).max(0.0) / denom;
            let net = if light > 0.0 {
                let temp = crate::temperature_at_z_with_ctx(cell.position[2], WORLD_HALF, &ctx);
                let metabolism = crate::metabolism_factor(temp);
                SYMBIONT_PHOTO_RATE * light * metabolism * dt - upkeep
            } else {
                -upkeep
            };
            cell.energy += net;
            let sym = cell.symbiont.as_mut().unwrap();
            if net < 0.0 {
                sym.deficit_streak += 1;
                if sym.deficit_streak > SYMBIONT_UPKEEP_DEFICIT_TICKS {
                    cell.symbiont = None;
                    sheds_this_pass += 1;
                }
            } else {
                sym.deficit_streak = 0;
            }
        }
        self.sym_sheds_gen += sheds_this_pass;
    }


    fn eat_food(&mut self) {
        // Sprint 43: food_grid lookup místo full sweep.
        // Sprint 57: 3-pass paralelizace. Pass 1 paralelně vybere candidate
        // food per cell (read-only test), Pass 2 sekvenčně resolvne race
        // (first-cell-wins per food + Hebbian na CPU), Pass 3 swap_remove.
        // GPU Hebbian zůstává na konci jako pre-Sprint-57.
        //
        // Wave N+: Pass 1 moved to GPU (`EatFoodGpu`). The CPU
        // `SpatialGrid::for_each_in_radius` was the single largest hot
        // spot (~27 % of CPU per `perf report`). The new path uploads
        // food positions/kinds/ages + cell pose snapshots and reads back
        // per-cell `(food_idx, value)` candidates. CPU still does Pass 2
        // (race resolution) and Pass 3 (swap_remove) sequentially.
        self.eaten_scratch.clear();
        self.eaten_scratch.resize(self.foods.len(), false);

        // Sprint 78: cell_id → idx map pro food share lookup. Reuse persistent
        // scratch (built v `tick` start, cells layout v eat_food beze změny).
        let id_to_idx = &self.id_to_idx_scratch;

        // GPU Hebbian is the only path post wave N. Sprint 133 routes eat
        // events through `RewardAccumulator`; the rewards vec is materialised
        // just before dispatch.
        let mut acc = RewardAccumulator::with_capacity(self.cells.len() / 8);
        let use_gpu_hebbian = true;

        // Pass 1 (GPU): per-cell candidate selection. Sentinel `num_foods`
        // is "no match this tick". Empty population → no dispatch, empty
        // candidates.
        let n = self.cells.len();
        let n_foods = self.foods.len();
        let candidates: Vec<Option<(usize, f32)>> = if n == 0 || n_foods == 0 {
            vec![None; n]
        } else {
            let params = EatFoodParamsGpu {
                num_cells: 0,
                num_foods: 0,
                cell_size: GRID_CELL_SIZE,
                eat_radius: EAT_RADIUS,
                world_half_x: WORLD_HALF[0],
                world_half_y: WORLD_HALF[1],
                world_half_z: WORLD_HALF[2],
                world_map_nx: crate::WORLD_MAP_RES as u32,
                world_map_ny: crate::WORLD_MAP_RES as u32,
                world_map_nz: crate::WORLD_MAP_RES_Z as u32,
                fixed_timestep_hz: FIXED_TIMESTEP_HZ,
                plant_food_value: PLANT_FOOD_VALUE,
                carrion_food_value: CARRION_FOOD_VALUE,
                carrion_decay_per_sec: CARRION_DECAY_PER_SEC,
                world_map_food_floor: crate::WORLD_MAP_FOOD_FLOOR,
                world_map_food_amp: crate::WORLD_MAP_FOOD_AMP,
            };
            let cells = &self.cells;
            let foods = &self.foods;
            let gpu = self.gpu_full.as_mut().expect("gpu_full Some");
            let s = &mut gpu.scratch;
            // Persistent scratch — replaces 9 fresh Vec allocations.
            s.lt_positions.clear();
            s.lt_headings.clear();
            s.lt_pitches.clear();
            s.lt_body_dims.clear();
            s.lt_carnivore.clear();
            s.lt_max_axes.clear();
            for c in cells.iter() {
                s.lt_positions.push(c.position);
                s.lt_headings.push(c.heading);
                s.lt_pitches.push(c.pitch);
                s.lt_body_dims.push([
                    c.phenotype.body_length,
                    c.phenotype.body_width,
                    c.phenotype.body_height,
                ]);
                s.lt_carnivore.push(c.genome.carnivore_score);
                s.lt_max_axes.push(c.phenotype.max_axis());
            }
            s.lt_food_positions.clear();
            s.lt_food_kinds.clear();
            s.lt_food_age_ticks.clear();
            for f in foods.iter() {
                s.lt_food_positions.push(f.position);
                s.lt_food_kinds.push(f.kind as u32);
                s.lt_food_age_ticks.push(f.age_ticks);
            }
            // food_hash needs to reflect the current food set — refresh
            // before the eat_food dispatch (predate's hash is keyed on
            // cells, not foods).
            gpu.profiler.resume();
            gpu.food_hash.dispatch(&s.lt_food_positions);
            let result = gpu.eat_food.compute(
                &s.lt_positions,
                &s.lt_headings,
                &s.lt_pitches,
                &s.lt_body_dims,
                &s.lt_carnivore,
                &s.lt_max_axes,
                &s.lt_food_positions,
                &s.lt_food_kinds,
                &s.lt_food_age_ticks,
                &gpu.food_hash,
                params,
            );
            gpu.profiler.mark("eat_food");
            let sentinel = n_foods as u32;
            (0..n)
                .map(|i| {
                    let f = result.food_idx[i];
                    if f >= sentinel {
                        None
                    } else {
                        Some((f as usize, result.value[i]))
                    }
                })
                .collect()
        };

        // Pass 2 (sequential): resolve. Per cell_idx v insertion order — first
        // cell to claim a food wins. Matches pre-Sprint-57 ordering (sekvenční
        // outer loop with eaten_scratch shortcut).
        // Sprint 78: cluster food share. Pokud má eater bondy, partner cells
        // dostávají `value × BOND_FOOD_SHARE_FRAC` extra energy (free reward,
        // no conservation — modeluje tissue cooperation). Sebráno do
        // share_deltas Vec během iterace, aplikováno post-loop kvůli
        // simultaneous mutable borrow.
        let mut ate_cell_indices: Vec<usize> = Vec::new();
        let mut share_deltas: Vec<(usize, f32)> = Vec::new();
        for (cell_idx, opt) in candidates.iter().enumerate() {
            if let Some((food_idx, value)) = opt {
                if self.eaten_scratch[*food_idx] {
                    continue;
                }
                self.eaten_scratch[*food_idx] = true;
                let bonds_copy;
                let donor_state;
                let donor_lineage;
                let donor_altruism;
                {
                    let cell = &mut self.cells[cell_idx];
                    cell.energy += *value;
                    bonds_copy = cell.bonds;
                    donor_state = cell.cell_state;
                    donor_lineage = cell.lineage_id;
                    donor_altruism = cell.genome.altruism_share_frac;
                }
                // Sprint 80: donor's cell_state modulates share fraction.
                // State≈0 (selfish) → ~0%; state≈1 (altruist) → full rate.
                // Sprint 193: share is conservative + sub-linear. The donor
                // distributes a FIXED pool `value × altruism × state` split
                // evenly across its bonded partners, so per-partner share is
                // `1/n` and total energy injected into a cluster no longer
                // scales with its size. Pre-S193 each partner received the
                // full pool times a super-linear `cluster_mult`, making large
                // clusters a runaway free-energy attractor that swamped any
                // foraging-skill selection (S193 5×100-gen sweep:
                // bond_active_frac 0 → 0.93). `self.kin_filter` (Sprint 87
                // Hamilton sweep) still gates sharing to same-lineage
                // partners when set.
                let n_bonds =
                    bonds_copy.iter().filter(|b| b.is_some()).count() as u32;
                let share_value =
                    bonded_food_share(*value, donor_altruism, donor_state, n_bonds);
                if share_value > 0.0 {
                    for bond_opt in bonds_copy.iter() {
                        if let Some(bond) = bond_opt {
                            if let Some(&partner_idx) =
                                id_to_idx.get(&bond.other_cell_id)
                            {
                                if self.kin_filter
                                    && self.cells[partner_idx].lineage_id != donor_lineage
                                {
                                    continue;
                                }
                                if partner_idx != cell_idx {
                                    share_deltas.push((partner_idx, share_value));
                                }
                            }
                        }
                    }
                }
                if use_gpu_hebbian {
                    // Sprint 187: per-cell food-drive weight (pre-S187: 1.0).
                    let w = self.cells[cell_idx].genome.reward_weights
                        [RewardKind::EatFood as usize];
                    acc.push(cell_idx, RewardKind::EatFood, w);
                } else {
                    ate_cell_indices.push(cell_idx);
                }
            }
        }
        // Sprint 78: aplikuj food share delty (po Pass 2 main loop).
        for (idx, delta) in share_deltas {
            self.cells[idx].energy += delta;
        }

        // Pass 2b: CPU Hebbian update sekvenčně. Hebbian je ~700 ops × max
        // ~10-30 cells/tick (kteří snědli), takže paralelizace by stejně byla
        // overhead-bound — sekvenční je rychlejší než thread spawn cost.
        if !use_gpu_hebbian {
            for &cell_idx in &ate_cell_indices {
                let cell = &mut self.cells[cell_idx];
                let last_hidden = cell.last_hidden;
                let last_outputs = cell.last_outputs;
                // Sprint 136: per-cell learning rate.
                let lr = cell.genome.learning_rate;
                // Wave 3: trace-based reward — credits motor outputs from up
                // to ~120 ticks back, not just this tick's pre·post.
                cell.genome.brain.hebbian_apply_reward(
                    &last_hidden,
                    &last_outputs,
                    1.0,
                    lr,
                );
            }
        }

        // Pass 3: swap_remove eaten foods.
        for j in (0..self.eaten_scratch.len()).rev() {
            if self.eaten_scratch[j] {
                self.foods.swap_remove(j);
            }
        }

        // Wave 7: trace-based reward apply replaces the legacy instantaneous
        // `compute_persistent`. Mutates GPU brain_weights using the trace
        // shadow that `dispatch_step_persistent` has been decaying+
        // accumulating since the last reward event. CPU `cell.genome.brain`
        // is NOT updated; sync happens at `reproduce` via `download_brain_at`.
        //
        // Sprint 133: unconditional dispatch preserved (matches pre-S133 even
        // when no cell ate this tick — shader early-exits on `reward == 0`).
        // Sprint 135: flush mode migrated to `SumAndClamp` for consistency
        // with the predate / hazard / bond / mate dispatch sites.
        if use_gpu_hebbian {
            let n = self.cells.len();
            let mut rewards: Vec<f32> = vec![0.0; n];
            acc.flush_to(&mut rewards, FlushMode::SumAndClamp);
            let gpu = self.gpu_full.as_ref().unwrap();
            gpu.cells.upload_rewards(&rewards);
            gpu.hebbian.dispatch_apply_reward_persistent(&gpu.cells, n, LEARNING_RATE);
            gpu.stdp_apply.dispatch(
                &gpu.cells,
                n,
                self.clock.tick as u32,
                DEFAULT_STDP_A_PLUS,
                DEFAULT_STDP_A_MINUS,
            );
        }
    }

    /// GPU rejection sampling — generates K = budget × MAX_SPAWN_ATTEMPTS
    /// candidates in parallel, CPU pushes the first `budget` valid ones into
    /// `self.foods`. Variable allocation stays CPU-side; GPU only does the
    /// rejection work (world_map richness + obstacle mask + cell exclusion).
    fn spawn_food_dispatch(&mut self, rng: &mut impl Rng, budget: usize) {
        if budget == 0 {
            return;
        }
        let k = budget * crate::MAX_SPAWN_ATTEMPTS;
        let positions: Vec<[f32; 3]> = self.cells.iter().map(|c| c.position).collect();
        let max_axes: Vec<f32> = self.cells.iter().map(|c| c.phenotype.max_axis()).collect();
        let seeds: Vec<[u32; 4]> = (0..k)
            .map(|_| {
                [
                    rng.random::<u32>(),
                    rng.random::<u32>(),
                    rng.random::<u32>(),
                    rng.random::<u32>(),
                ]
            })
            .collect();
        let obstacle_active = self.obstacles.is_some();
        let (obs_nx, obs_ny, obs_nz) = self
            .obstacles
            .as_ref()
            .map(|o| (o.resolution[0] as u32, o.resolution[1] as u32, o.resolution[2] as u32))
            .unwrap_or((1, 1, 1));
        let params = FoodSpawnParamsGpu {
            num_attempts: 0, // populated by compute()
            rejection_strength: crate::FOOD_REJECTION_STRENGTH,
            eat_radius: crate::EAT_RADIUS,
            cell_size: crate::GRID_CELL_SIZE,
            world_half_x: WORLD_HALF[0],
            world_half_y: WORLD_HALF[1],
            world_half_z: WORLD_HALF[2],
            num_cells: 0, // populated by compute()
            world_map_nx: self.map.resolution[0] as u32,
            world_map_ny: self.map.resolution[1] as u32,
            world_map_nz: self.map.resolution[2] as u32,
            obstacle_active: if obstacle_active { 1 } else { 0 },
            obstacle_nx: obs_nx,
            obstacle_ny: obs_ny,
            obstacle_nz: obs_nz,
            _pad0: 0,
        };
        let result = {
            let gpu = self.gpu_full.as_mut().expect("gpu_full Some");
            gpu.cell_hash.dispatch(&positions);
            gpu.food_spawn.seed_attempts(&seeds);
            gpu.food_spawn.compute(k, &positions, &max_axes, &gpu.cell_hash, params)
        };
        let mut pushed = 0usize;
        for i in 0..k {
            if pushed >= budget {
                break;
            }
            if result.valid_mask[i] != 0 {
                self.foods.push(Food {
                    position: result.candidate_positions[i],
                    age_ticks: 0,
                    kind: crate::FoodKind::Plant,
                });
                pushed += 1;
            }
        }
    }

    fn spawn_food(&mut self, rng: &mut impl Rng) {
        let target = food_target(self.density_factor * self.food_factor_mult);
        if self.foods.len() >= target {
            return;
        }
        let to_spawn = (target - self.foods.len()).min(FOOD_SPAWN_RATE);
        self.spawn_food_dispatch(rng, to_spawn);
    }


    /// Sync `Genome.brain` for every cell from the persistent GPU buffer
    /// back to CPU. In `--gpu-full + GPU CPPN` mode child brains are produced
    /// directly on the device and the CPU `Genome.brain` field stays at the
    /// `Brain::zeros()` placeholder until something explicitly downloads.
    /// Call this once per generation before serialisation / diagnostics
    /// (`w1_frobenius_std` etc.) — single `Wait` barrier, ~few ms total.
    pub fn sync_brains_from_gpu(&mut self) {
        if let Some(gpu) = self.gpu_full.as_ref() {
            let n = self.cells.len();
            if n == 0 {
                return;
            }
            let brains = gpu.cells.download_brains(n);
            for (cell, brain) in self.cells.iter_mut().zip(brains) {
                // `download_brains` hardcodes `hidden_n = BRAIN_HIDDEN_DEFAULT`
                // (it reads only the weight buffer); the evolved topology size
                // lives in the CPU genome, so preserve it across the replace.
                let hidden_n = cell.genome.brain.hidden_n;
                cell.genome.brain = brain;
                cell.genome.brain.hidden_n = hidden_n;
            }
        }
    }

    /// Linear scan for a cell by its stable `cell_id`. Returns the current
    /// slot index, or `None` if the cell has died / been swap-removed.
    pub fn find_cell_idx_by_id(&self, cell_id: u64) -> Option<usize> {
        self.cells.iter().position(|c| c.cell_id == cell_id)
    }

    /// Download a single cell's brain weights from GPU into the CPU mirror.
    /// Only `w1/b1/w2/b2` are part of `brain_weights_buf`; trace, membrane,
    /// and STDP slots are not synced. Returns `true` if the index existed.
    pub fn sync_cell_brain_from_gpu(&mut self, cell_idx: usize) -> bool {
        let Some(gpu) = self.gpu_full.as_ref() else {
            return false;
        };
        if cell_idx >= self.cells.len() {
            return false;
        }
        let brain = gpu.cells.download_brain_at(cell_idx);
        // `download_brain_at` returns a fresh `Brain` whose w1/b1/w2/b2 are
        // from GPU; the trace / membrane / STDP slots stay at defaults
        // (those buffers aren't part of `brain_weights_buf`). Preserve
        // CPU-side `hidden_n` since `download_brain_at` hardcodes
        // `BRAIN_HIDDEN_DEFAULT` — `hidden_n` lives in the CPU genome and
        // can grow past the default via structural mutation.
        let hidden_n = self.cells[cell_idx].genome.brain.hidden_n;
        self.cells[cell_idx].genome.brain.w1 = brain.w1;
        self.cells[cell_idx].genome.brain.b1 = brain.b1;
        self.cells[cell_idx].genome.brain.w2 = brain.w2;
        self.cells[cell_idx].genome.brain.b2 = brain.b2;
        self.cells[cell_idx].genome.brain.hidden_n = hidden_n;
        true
    }

    /// V7: pull the GPU vibration field back into the CPU shadow so the
    /// per-gen CSV writer can sample it at cell positions. Called from
    /// `main` right before `write_stats`; per-tick CPU shadow stays
    /// out-of-date in `--gpu-full` mode (sensor gather reads the GPU
    /// buffer direct, no readback needed mid-tick).
    pub fn sync_vibration_from_gpu(&mut self) {
        if let Some(gpu) = self.gpu_full.as_mut() {
            let grid = gpu.vibration.download();
            self.vibration.replace_grid_from(&grid);
        }
    }

    fn reproduce(&mut self, rng: &mut impl Rng) {
        let current_pop = self.cells.len();
        if current_pop >= self.max_population {
            return;
        }
        let budget = self.max_population - current_pop;
        let fertile = self.collect_fertile();
        self.fertile_ticks_gen += fertile.len() as u64;
        let mating_r2 = self.mating_radius * self.mating_radius;
        let matings = crate::pair_fertile(&fertile, mating_r2, budget, WORLD_HALF);
        let child_start = self.cells.len();
        let to_spawn = self.spawn_children_from_matings(&matings, rng);
        let n_births = to_spawn.len();
        self.births_gen += n_births as u64;

        // Sprint 135: credit both parents with `MateSignalAccepted` before
        // appending children — parent indices and GPU trace slots are still
        // pre-extend at this point.
        if !matings.is_empty() && self.gpu_full.is_some() {
            let n = self.cells.len();
            let mut mate_acc = RewardAccumulator::with_capacity(matings.len() * 2);
            for &(a, b) in matings.iter() {
                // Sprint 187: each parent's own mating-reward weight.
                let w_a = self.cells[a].genome.reward_weights
                    [RewardKind::MateSignalAccepted as usize];
                let w_b = self.cells[b].genome.reward_weights
                    [RewardKind::MateSignalAccepted as usize];
                mate_acc.push(a, RewardKind::MateSignalAccepted, w_a);
                mate_acc.push(b, RewardKind::MateSignalAccepted, w_b);
            }
            let mut rewards: Vec<f32> = vec![0.0; n];
            mate_acc.flush_to(&mut rewards, FlushMode::SumAndClamp);
            let gpu = self.gpu_full.as_ref().unwrap();
            gpu.cells.upload_rewards(&rewards);
            gpu.hebbian
                .dispatch_apply_reward_persistent(&gpu.cells, n, LEARNING_RATE);
            gpu.stdp_apply.dispatch(
                &gpu.cells,
                n,
                self.clock.tick as u32,
                DEFAULT_STDP_A_PLUS,
                DEFAULT_STDP_A_MINUS,
            );
        }

        self.cells.extend(to_spawn);
        // GPU child upload. In `--gpu-full`, brain weights are produced
        // directly on the device via `CppnGpu::dispatch`, so we skip the
        // per-child `upload_brain_at` round-trip that the CPU path needed
        // (saves ~16 KB × n_children write_buffer per reproduce phase).
        // xoshiro seed + turn_rate uploads stay — they're independent of
        // the brain materialisation path.
        if n_births > 0 && self.gpu_full.is_some() {
            // Borrow split: build `pairs` from `self.cells` (immutable),
            // then take `&mut self.gpu_full` for dispatch. Rust's disjoint
            // field borrows handle this because `cells` and `gpu_full` are
            // distinct fields of `self`.
            let pairs: Vec<(usize, &crate::Cppn)> = (0..n_births)
                .map(|off| {
                    let slot = child_start + off;
                    (slot, &self.cells[slot].genome.cppn)
                })
                .collect();
            let gpu = self.gpu_full.as_mut().unwrap();
            gpu.cppn.dispatch(&pairs, &gpu.cells);
            drop(pairs);
            for off in 0..n_births {
                let slot = child_start + off;
                let child = &self.cells[slot];
                // V7-unification: seed from `cell_id` to keep CPU + GPU
                // xoshiro streams in lockstep across reproduce events.
                gpu.cells.upload_xoshiro_seed_at(slot, child.cell_id);
                gpu.cells.upload_turn_rate_at(slot, child.genome.turn_rate);
                // Sprint 137: child's per-cell Hebbian rates inherited /
                // mutated from parents.
                gpu.cells.upload_rates_at(
                    slot,
                    child.genome.learning_rate,
                    child.genome.trace_decay_per_sec,
                );
                gpu.cells
                    .upload_reproduce_at_energy_at(slot, child.genome.reproduce_at_energy);
                // Sprint 147: child's NeuronModel — slot recycled from a
                // Perceptron parent might need flipping to Izhikevich.
                gpu.cells
                    .upload_neuron_model_at(slot, child.genome.neuron_model.as_u32());
                // Slot may be a recycled one (post-swap_to from a death this
                // tick or earlier); zero per-slot Hebbian state so the child
                // doesn't inherit previous-occupant traces or recurrent input.
                gpu.cells.reset_persistent_brain_state_at(slot);
            }
        }
        let _ = n_births;
    }

    /// Sprint 40: snapshot fertile cells. Sprint 25 mating gating: cells musí
    /// AKTIVNĚ emitovat pheromone (output[2] > threshold) aby reprodukovaly —
    /// selektuje proti free-riders na pheromone field.
    fn collect_fertile(&self) -> Vec<(usize, [f32; 3])> {
        let n = self.cells.len();
        if n == 0 {
            return Vec::new();
        }
        // Frequency-dependent reproduce threshold: common lineages musí mít
        // víc energie aby reprodukovaly. Zachovává diverzitu proti monoculture
        // bez hard cap na lineage size.
        let inv_n = 1.0 / n as f32;
        let mut freq: rustc_hash::FxHashMap<u64, f32> = rustc_hash::FxHashMap::default();
        for c in &self.cells {
            *freq.entry(c.lineage_id).or_insert(0.0) += inv_n;
        }
        self.cells
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                let f = freq.get(&c.lineage_id).copied().unwrap_or(0.0);
                let scaled = c.genome.reproduce_at_energy
                    * (1.0 + crate::LINEAGE_DIVERSITY_ALPHA * f * f);
                c.energy >= scaled
                    && c.last_outputs[2] > MATING_PHEROMONE_THRESHOLD
                    && c.reproduce_cooldown_ticks == 0
            })
            .map(|(i, c)| (i, c.position))
            .collect()
    }

    /// Sprint 40: split-borrow rodiče po indexu, halve energy, vyrobí dítě.
    /// Sprint 51: pokud --gpu-full, downloaduje parent brains z GPU před
    /// crossover (GPU Hebbian je canonical), child brain se uploaduje po
    /// extend.
    fn spawn_children_from_matings(
        &mut self,
        matings: &[(usize, usize)],
        rng: &mut impl Rng,
    ) -> Vec<Cell> {
        // Wave K: skip the per-pair `download_brain_at` round-trip — mating
        // only touches `Genome.cppn` (`make_mating_child_no_brain`), so the
        // parent brain isn't actually read here. CPU `Cell.genome.brain`
        // stays at the post-tick GPU value until the per-generation
        // `sync_brains_from_gpu` batched download refreshes it for
        // diagnostics / checkpoint serialization. Saves O(matings) ×
        // `wgpu::Maintain::Wait` round-trips each tick.
        // Sprint 66: pre-allocate cell_ids for each child before splitting
        // self.cells (split_at_mut would conflict with self.next_cell_id access).
        let child_ids: Vec<u64> = (0..matings.len())
            .map(|_| {
                let id = self.next_cell_id;
                self.next_cell_id += 1;
                id
            })
            .collect();
        // In `--gpu-full` skip CPU `Brain::from_cppn` — child brain weights
        // land on the device through the per-reproduce-phase
        // `CppnGpu::dispatch` after `cells.extend(children)`.
        // GPU CPPN produces child brain weights direct on device; skip
        // per-child CPU `Brain::from_cppn` cost.
        let use_gpu_cppn = true;
        let mut children = Vec::with_capacity(matings.len());
        for (i, &(a, b)) in matings.iter().enumerate() {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            let (left, right) = self.cells.split_at_mut(hi);
            let cell_lo = &mut left[lo];
            let cell_hi = &mut right[0];
            let (cell_a, cell_b) = if a < b {
                (cell_lo, cell_hi)
            } else {
                (cell_hi, cell_lo)
            };
            // Sprint 187: absolute donation replaces halve-and-give. Each
            // parent pays its own gene-encoded `birth_energy`; child energy
            // is the sum, computed inside `make_mating_child` from parent
            // genomes. The fertile-check (`reproduce_at_energy`) plus the
            // `clamp_birth_energy` invariant guarantee the post-subtraction
            // energy stays at or above `REPRODUCE_BIRTH_SAFETY_MARGIN`.
            cell_a.energy -= cell_a.genome.birth_energy;
            cell_b.energy -= cell_b.genome.birth_energy;
            cell_a.reproduce_cooldown_ticks = MATING_COOLDOWN_TICKS;
            cell_b.reproduce_cooldown_ticks = MATING_COOLDOWN_TICKS;
            let child = if use_gpu_cppn {
                crate::make_mating_child_no_brain(cell_a, cell_b, rng, child_ids[i])
            } else {
                crate::make_mating_child(cell_a, cell_b, rng, child_ids[i])
            };
            children.push(child);
        }
        children
    }

    fn die_and_drop_carrion(&mut self, rng: &mut impl Rng) {
        let half = WORLD_HALF;
        let mut new_foods: Vec<Food> = Vec::new();

        // Phase 1: emit carrion food for dead cells (read-only iteration).
        for cell in &self.cells {
            if cell.energy <= 0.0 {
                for _ in 0..CARRION_FOOD_COUNT {
                    let pos = [
                        (cell.position[0] + rng.random_range(-CELL_RADIUS..CELL_RADIUS))
                            .clamp(-half[0], half[0]),
                        (cell.position[1] + rng.random_range(-CELL_RADIUS..CELL_RADIUS))
                            .clamp(-half[1], half[1]),
                        cell.position[2].clamp(-half[2], half[2]),
                    ];
                    new_foods.push(Food {
                        position: pos,
                        age_ticks: 0,
                        kind: crate::FoodKind::Carrion,
                    });
                }
            }
        }

        // Phase 2: remove dead cells. --gpu-full uses swap_remove pattern so
        // GPU brain_weights + xoshiro_state lze udržet in sync přes O(deaths)
        // GPU memcpy operations (žádný full re-upload).
        let deaths = {
            let before = self.cells.len();
            let gpu = self.gpu_full.as_ref().unwrap();
            let mut i = 0;
            while i < self.cells.len() {
                if self.cells[i].energy <= 0.0 {
                    let last = self.cells.len() - 1;
                    if i != last {
                        gpu.cells.swap_to(i, last);
                    }
                    self.cells.swap_remove(i);
                    // i nezvyšuju — moved cell ze slotu last je teď ve slotu i.
                } else {
                    i += 1;
                }
            }
            let deaths = (before - self.cells.len()) as u64;
            self.deaths_gen += deaths;
            deaths
        };

        // Phase 3: prune bonds whose partner was just swap-removed. Without
        // this the dangling bond survives until the next tick's
        // `resolve_collisions` pruning, so any end-of-tick snapshot
        // (inspector, headless dump, checkpoint) would show a bond to a
        // cell that no longer exists.
        if deaths > 0 {
            self.survivor_ids_scratch.clear();
            self.survivor_ids_scratch
                .extend(self.cells.iter().map(|c| c.cell_id));
            let survivors = &self.survivor_ids_scratch;
            let mut pruned: u64 = 0;
            for cell in self.cells.iter_mut() {
                for slot in cell.bonds.iter_mut() {
                    if let Some(bond) = *slot {
                        if !survivors.contains(&bond.other_cell_id) {
                            *slot = None;
                            pruned += 1;
                        }
                    }
                }
            }
            self.bonds_broken_gen += pruned;
        }
        self.foods.extend(new_foods);
    }
}

// World-map richness → food-value multiplier. Used only by tests now (eat_food
// moved to GPU shader which inlines the formula directly).
#[allow(dead_code)]
pub fn food_multiplier(noise: f32) -> f32 {
    WORLD_MAP_FOOD_FLOOR + WORLD_MAP_FOOD_AMP * noise
}

pub fn hazard_drain(noise: f32) -> f32 {
    HAZARD_DRAIN_PER_SEC * (HAZARD_FLOOR + HAZARD_AMP * noise)
}

pub fn food_target(factor: f32) -> usize {
    // Sprint 53: scale s 3D objemem aby food density per volume zůstala
    // konstantní napříč z-expansionem. Pre-Sprint-53 baseline: z=2 → z_extent=4.
    // Volumetric factor = z_extent / 4. Při z=20: 10× food count vs pre-Sprint-53.
    let area = (2.0 * WORLD_HALF[0]) * (2.0 * WORLD_HALF[1]);
    let z_extent = 2.0 * WORLD_HALF[2];
    let z_factor = (z_extent / 4.0).max(1.0);
    ((area / WORLD_UNITS_PER_FOOD) * factor.max(0.0) * z_factor) as usize
}

/// Sprint 193: linearly ramp plant food density from 1.0 at generation 0
/// down to `SCARCITY_FLOOR` at `SCARCITY_RAMP_END_GEN`, then hold the
/// floor. The result is folded into `World::density_factor` once per
/// generation alongside the seasonal cycle and FoodCrash shock — the
/// `spawn_food` call site stays unchanged.
pub fn scarcity_factor(generation: u64) -> f32 {
    if SCARCITY_RAMP_END_GEN == 0 {
        return SCARCITY_FLOOR;
    }
    let t = (generation as f32 / SCARCITY_RAMP_END_GEN as f32).min(1.0);
    1.0 + (SCARCITY_FLOOR - 1.0) * t
}

/// Sprint 193: per-partner bonded food-share amount. When a bonded cell eats
/// food worth `value`, it distributes a fixed pool `value × altruism × state`
/// split evenly across its `n_bonds` partners — so per-partner share is `1/n`
/// and the total energy injected into a cluster is independent of its size.
///
/// Pre-S193 every partner received the full pool scaled by a super-linear
/// `cluster_mult = 1 + (n−1) × cluster_bonus`, making total energy injected
/// per food grow ~quadratically with cluster size. That turned "join the
/// largest cluster" into a runaway attractor that swamped foraging-skill
/// selection; the S193 scarcity ramp amplified it instead of breaking it
/// (see `docs/sprints/193-202-environment-pressure.md`).
pub fn bonded_food_share(value: f32, altruism: f32, state: f32, n_bonds: u32) -> f32 {
    if n_bonds == 0 {
        return 0.0;
    }
    (value * altruism * state) / n_bonds as f32
}

/// Rough static FLOP estimate per simulation tick at the given population.
/// Used by the stats overlay to surface effective compute throughput; the
/// real workload varies ±2× with emergent behavior (predate attack pairs,
/// reward density driving Hebbian/STDP apply, etc.), so treat this as an
/// order-of-magnitude indicator, not a benchmark.
///
/// Major terms (per tick):
/// - brain forward `w1 × x + b1`, tanh, `w2 × h + b2`, tanh — runs for
///   every cell every tick.
/// - hebbian trace decay (every tick) — proportional to weight count.
/// - STDP step (per hidden neuron, every tick).
/// - 5 diffusion fields (smell + 3 pheromones + vibration).
/// - sparse predate / collisions / eat_food / sensor_gather lumped as
///   per-cell constants because the spatial-hash neighbor counts are
///   hard to estimate without an actual measurement.
pub fn estimate_flops_per_tick(n_cells: u64) -> u64 {
    if n_cells == 0 {
        return 0;
    }
    let bi = crate::BRAIN_INPUTS as u64;
    let bh = crate::BRAIN_HIDDEN as u64;
    let bo = crate::BRAIN_OUTPUTS as u64;
    let w1_ops = 2 * bi * bh;
    let w2_ops = 2 * bh * bo;
    let brain_per_cell = w1_ops + w2_ops + bh + bo;
    let weight_count = bi * bh + bh * bo;

    let brain = n_cells * brain_per_cell;
    let hebbian = n_cells * weight_count * 2;
    let stdp = n_cells * bh * 4;
    let excitability = n_cells * bh * 5;
    let synaptic_scale = (n_cells * weight_count * 3) / crate::SCALING_PERIOD_TICKS;
    let field_voxels = (crate::SMELL_GRID_RES
        * crate::SMELL_GRID_RES
        * crate::SMELL_GRID_RES_Z) as u64;
    let fields = 5 * field_voxels * 10;
    let predate = n_cells * 800;
    let collisions = n_cells * 500;
    let sensor_gather = n_cells * 650;
    let other = n_cells * 400;

    brain + hebbian + stdp + excitability + synaptic_scale + fields + predate + collisions
        + sensor_gather + other
}

// Spatial occupancy thresholds for clustering diagnostics. A cell is
// "near edge" if its position falls in the outer 10 % of either axis;
// "in corner" if both axes simultaneously meet that criterion.
pub const EDGE_FRAC_THRESHOLD: f32 = 0.9;

// Sprint 173: integration test fixtures moved to `src/bin/headless/world_tests.rs`
// (declared via `#[cfg(test)] mod world_tests;` in `headless/main.rs`). Keeps
// them under the binary's test runner where they were before the World
// extraction, so `cargo test --lib` doesn't see GPU-adapter-heavy tests that
// fan-out-fail under shared init.

