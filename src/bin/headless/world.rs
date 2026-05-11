//! Headless harness — pure simulation loop, no Bevy renderer.
//!
//! Usage: `cargo run --release --bin headless -- [seed] [max_gens] [out_path]`
//! Defaults: seed=0, max_gens=500, out_path=run_seed{seed}.csv
//!
//! Logs per-generation stats (cell_count, mean/dev for max_speed/vision/body_size,
//! food count, density factor) to CSV. Reproducible: same seed → identical run.

use bioscape::{
    adhesion_velocity_delta, bond_velocity_delta,
    reject_food_for_richness, Bond, Cell, CoopFood, EventCalendar, Food, MazeDifficulty,
    ObstacleField, SimClock, SmellField, SpatialGrid, WorldMap, ADHESION_RANGE_FACTOR,
    ATTACK_THRESHOLD, BOND_BREAK_THRESHOLD,
    BOND_FORMATION_COST, BOND_FORM_THRESHOLD, BOND_FORM_TICKS, BOND_MAINTENANCE_PER_SEC,
    BOND_REST_LENGTH_SLACK, BRAIN_RECURRENT, CARRION_FOOD_COUNT, CELL_RADIUS,
    COLLISION_RESTITUTION, CONTACT_DECAY_TICKS, COOP_FOOD_MAX_CONCURRENT,
    COOP_FOOD_SPAWN_RATE_PER_TICK, CYCLE_AMPLITUDE, CYCLE_GEN_PERIOD,
    DILUTION_K, EAT_RADIUS, FIXED_TIMESTEP_HZ, FOOD_SPAWN_RATE,
    GENERATIONS_PER_EPOCH, GRID_CELL_SIZE, HAZARD_AMP, HAZARD_DRAIN_PER_SEC, HAZARD_FLOOR,
    HERD_RADIUS, LEARNING_RATE, MATING_COOLDOWN_TICKS, MATING_PHEROMONE_THRESHOLD, MAX_BONDS_PER_CELL, MAX_SPAWN_ATTEMPTS,
    N_PHEROMONE_CHANNELS,
    PHEROMONE_BASELINE_EMIT, PHEROMONE_BRAIN_MOD, PHEROMONE_COST_PER_RATE,
    PHEROMONE_DECAY_PER_CH, PHEROMONE_DIFFUSION_PER_CH,
    PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z, PHEROMONE_SAMPLE_EPSILON,
    PHYSICS_CONFIG, PREDATION_DRAIN_PER_TICK, PREDATION_GAIN_PER_TICK, REPRODUCE_THRESHOLD,
    SIZE_RATIO_THRESHOLD, SMELL_DECAY, SMELL_DIFFUSION, SMELL_GRID_RES, SMELL_GRID_RES_Z,
    SMELL_PER_FOOD, SMELL_SAMPLE_EPSILON, TICKS_PER_GENERATION, VIBRATION_DECAY,
    VIBRATION_DIFFUSION, VIBRATION_GRID_RES, VIBRATION_GRID_RES_Z, VIBRATION_SAMPLE_EPSILON,
    WORLD_HALF, WORLD_MAP_BASE_RES, WORLD_MAP_BASE_RES_Z, WORLD_MAP_FOOD_AMP,
    WORLD_MAP_FOOD_FLOOR, WORLD_MAP_RES, WORLD_MAP_RES_Z, WORLD_UNITS_PER_FOOD,
};
use bioscape::{BRAIN_HIDDEN, BRAIN_INPUTS};
#[cfg(feature = "gpu")]
use bioscape::{
    gpu::{
        BrainGpu, BrownianGpu, CellsGpu, CollisionGpu, CppnGpu, FieldGpu, GpuFullScratch,
        HebbianGpu, MotorGpu, PopulateInputsGpu, PopulateInputsParams, PredateGpu,
        PredateParamsGpu, SensorGatherGpu, SensorParamsGpu, SpatialHashGpu, StepGpu, StepParamsGpu,
    },
    AGE_DECAY_PER_SEC, ATTACK_COST_PER_SEC, BRAIN_INPUTS_SENSORY, BRAIN_OUTPUTS,
    DAMAGE_NORMALIZATION_GAIN, DENSITY_NORM_COUNT, DRAG_COEFFICIENT, GRAVITY as PHYS_GRAVITY,
    PHEROMONE_NORMALIZATION_GAIN, SHELL_COST_PER_SEC, SMELL_NORMALIZATION_GAIN,
    SPIKE_COST_PER_SEC, THERMAL_NOISE,
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
    /// `bioscape::vibration_emit_for_cell`); the field then diffuses + decays
    /// every tick. Brain reads gradient + amplitude as inputs [29..32]. CPU
    /// only — no GPU shader counterpart in this cut.
    pub vibration: SmellField,
    pub map: WorldMap,
    // Sprint 43: spatial hashes pro broad-phase. Rebuild před fází, která
    // neighbors používá — `cell_grid` před brain_act/resolve_collisions/predate,
    // `food_grid` před brain_act/eat_food.
    pub cell_grid: SpatialGrid<usize, f32>,
    pub food_grid: SpatialGrid<usize, bioscape::FoodKind>,
    // Persistent scratch — reused per tick to avoid hot-loop allocations.
    pub deltas_scratch: Vec<[f32; 3]>,
    /// Sprint 65: collision velocity damping (inelastic) — per pair, closing
    /// velocity podél separation normal je halved. Cell i sees pair (i, j),
    /// computes own delta; symmetric (Newton 3rd law) když j visits i.
    pub velocity_deltas_scratch: Vec<[f32; 3]>,
    pub energy_deltas_scratch: Vec<f32>,
    pub damage_deltas_scratch: Vec<f32>,
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
    pub hidden_snapshot_scratch: Vec<[f32; BRAIN_HIDDEN]>,
    pub inputs_scratch: Vec<[f32; BRAIN_INPUTS]>,
    pub births_gen: u64,
    pub deaths_gen: u64,
    pub fertile_ticks_gen: u64,
    pub predation_events_gen: u64,
    /// Sprint 66: monotonic counter pro stable Cell.cell_id přidělování.
    /// Initial population uses 0..INITIAL_CELLS, takže start = INITIAL_CELLS.
    pub next_cell_id: u64,
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
    #[cfg(feature = "gpu")]
    pub gpu: Option<BrainGpu>,
    // Sprint 51: full-GPU brain pipeline. Když Some, drží brain weights
    // persistent na GPU mezi ticky (eliminuje 30 MB/tick upload Sprintu 44),
    // GPU Hebbian replace CPU brain.hebbian_update, GPU Brownian replace
    // CPU apply_brownian. Sensor/motor/step/collision/predate zůstávají CPU
    // rayon (Sprint 50 standalone shadery jsou ready, integrace je Sprint 52+).
    #[cfg(feature = "gpu")]
    pub gpu_full: Option<GpuFullState>,
}

// `GpuFullScratch` přesunut do `bioscape::gpu::scratch` (lib) — sdílen mezi
// headless `--gpu-full` pathem a renderer `BIOSCAPE_GPU_FULL=1` pathem.

#[cfg(feature = "gpu")]
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
    /// Sprint 59: GPU smell + pheromone field (3D 7-point Jacobi).
    /// Sprint 60: po wire SensorGatherGpu už NEČTE CPU SmellField shadow —
    /// sensor shader bere field grid storage buffer direct. Per-tick readback
    /// eliminován; CPU `World.smell` / `pheromone` zůstávají jen pro
    /// checkpoint serialization (po `--gpu-full` jsou CPU shadows out-of-date).
    pub smell: FieldGpu,
    pub pheromone: FieldGpu,
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
    /// Persistent CPU snapshots — reused per tick, zachovává kapacitu.
    pub scratch: GpuFullScratch,
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
        let cells: Vec<Cell> = (0..initial_cells)
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
            energy_deltas_scratch: Vec::new(),
            damage_deltas_scratch: Vec::new(),
            eaten_scratch: Vec::new(),
            id_to_idx_scratch: rustc_hash::FxHashMap::default(),
            contact_lists_scratch: Vec::new(),
            positions_snapshot_scratch: Vec::new(),
            seen_pairs_scratch: rustc_hash::FxHashSet::default(),
            bond_candidates_scratch: Vec::new(),
            bonded_pairs_scratch: rustc_hash::FxHashSet::default(),
            hidden_snapshot_scratch: Vec::new(),
            inputs_scratch: Vec::new(),
            births_gen: 0,
            deaths_gen: 0,
            fertile_ticks_gen: 0,
            predation_events_gen: 0,
            next_cell_id: initial_cells as u64,
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
            share_frac: bioscape::BOND_FOOD_SHARE_FRAC,
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
            #[cfg(feature = "gpu")]
            gpu: None,
            #[cfg(feature = "gpu")]
            gpu_full: None,
        }
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
            energy_deltas_scratch: Vec::new(),
            damage_deltas_scratch: Vec::new(),
            eaten_scratch: Vec::new(),
            id_to_idx_scratch: rustc_hash::FxHashMap::default(),
            contact_lists_scratch: Vec::new(),
            positions_snapshot_scratch: Vec::new(),
            seen_pairs_scratch: rustc_hash::FxHashSet::default(),
            bond_candidates_scratch: Vec::new(),
            bonded_pairs_scratch: rustc_hash::FxHashSet::default(),
            hidden_snapshot_scratch: Vec::new(),
            inputs_scratch: Vec::new(),
            births_gen: chk.births_gen,
            deaths_gen: chk.deaths_gen,
            fertile_ticks_gen: chk.fertile_ticks_gen,
            predation_events_gen: chk.predation_events_gen,
            next_cell_id,
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
            share_frac: bioscape::BOND_FOOD_SHARE_FRAC,
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
            #[cfg(feature = "gpu")]
            gpu: None,
            #[cfg(feature = "gpu")]
            gpu_full: None,
        })
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
            let shock_mult = bioscape::food_density_shock_multiplier(
                &self.events.events,
                self.clock.generation,
            );
            self.density_factor = seasonal * shock_mult;
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
        #[cfg(feature = "gpu")]
        let skip_cpu_eligibility = self.gpu_full.is_some();
        #[cfg(not(feature = "gpu"))]
        let skip_cpu_eligibility = false;
        if !skip_cpu_eligibility {
            self.apply_eligibility_step(dt);
        }
        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.gpu_full.as_ref() {
            let n = self.cells.len();
            gpu.hebbian.dispatch_step_persistent(
                &gpu.cells,
                n,
                dt,
                bioscape::HEBBIAN_TRACE_DECAY_PER_SEC,
            );
        }
        timed!(emit_pheromones, self.emit_pheromones(dt));
        timed!(apply_morph, self.apply_morph(dt));
        timed!(apply_brownian, self.apply_brownian(rng, dt));
        timed!(step, self.step(dt));
        timed!(apply_food_gravity, self.apply_food_gravity(dt));
        timed!(apply_hazards, self.apply_hazards(dt));
        timed!(resolve_collisions, self.resolve_collisions());
        timed!(predate, self.predate());
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
        let pos = bioscape::random_coop_position(rng, WORLD_HALF);
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
        bioscape::register_coop_arrivals_for_all(
            &mut self.coop_foods,
            &self.cells,
            WORLD_HALF,
        );
        let current_tick = self.clock.tick;
        let cells = &mut self.cells;
        let mut i = 0;
        while i < self.coop_foods.len() {
            let triggered_now = bioscape::try_trigger_coop(&mut self.coop_foods[i], cells);
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

    fn apply_brownian(&mut self, _rng: &mut impl Rng, dt: f32) {
        #[cfg(feature = "gpu")]
        {
            // Sprint 62: brownian je fused do `brain_act_gpu_full` pipeline
            // (motor → brownian → batch readback). Tato fáze je v `--gpu-full`
            // no-op aby se neaplikoval brownian dvakrát.
            if self.gpu_full.is_some() {
                let _ = dt;
                return;
            }
        }
        self.apply_brownian_cpu(dt);
    }

    fn apply_brownian_cpu(&mut self, dt: f32) {
        // Per-cell xoshiro128++ stream — sjednoceno s GPU shaderem. Pop
        // is small enough that sequential iteration outperforms a rayon
        // par_iter_mut spawn overhead at typical N (≤1500 cells, ~10 µs
        // per loop).
        let sqrt_dt = dt.sqrt();
        for cell in &mut self.cells {
            cell.apply_brownian(sqrt_dt, WORLD_HALF[2]);
        }
    }

    /// Sprint 51: GPU brownian s xoshiro128++ per-cell RNG. Upload velocities,
    /// dispatch, download. Ne-deterministic vs CPU (different PRNG), ale
    /// deterministic across GPU runs (xoshiro state seedovaný z cell.lineage_id).
    /// Sprint 62: nyní fused do brain_act_gpu_full pipeline; tato standalone
    /// metoda je dead code preserved pro Sprint 63+ test path.
    #[cfg(feature = "gpu")]
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
            bioscape::THERMAL_NOISE,
            dt,
            WORLD_HALF[2] > 0.0,
        );
        let new_vels = gpu.cells.download_velocities(n);
        for (cell, v) in self.cells.iter_mut().zip(new_vels.iter()) {
            cell.velocity = *v;
        }
    }

    fn update_smell(&mut self, dt: f32) {
        // Sprint 60: deposit + diffuse na GPU bez readback. Sensor shader
        // (SensorGatherGpu) čte field grid přes storage buffer binding
        // (FieldGpu::current_grid_buffer) — žádná CPU SmellField sync.
        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.gpu_full.as_mut() {
            for food in &self.foods {
                gpu.smell.add_source(
                    [food.position[0], food.position[1], food.position[2]],
                    SMELL_PER_FOOD * dt,
                );
            }
            gpu.smell.step(SMELL_DIFFUSION, SMELL_DECAY, dt);
            return;
        }
        for food in &self.foods {
            self.smell
                .add_source([food.position[0], food.position[1], food.position[2]], SMELL_PER_FOOD * dt);
        }
        match self.smell_mask.as_deref() {
            Some(mask) => self.smell.step_masked(SMELL_DIFFUSION, SMELL_DECAY, dt, mask),
            None => self.smell.step(SMELL_DIFFUSION, SMELL_DECAY, dt),
        }
    }

    fn update_pheromone(&mut self, dt: f32) {
        // Diffuse + decay BEFORE this tick's emissions are added (in
        // emit_pheromones, called after brain_act). Cells thus read the
        // gradient ze stavu pole na předchozí tick — prevents instant
        // self-feedback (cell vidí svůj vlastní právě emitovaný puff).
        // Sprint 126: per-channel decay/diffusion. ch0 GPU step (single
        // FieldGpu instance), ch1/ch2 vždy CPU.
        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.gpu_full.as_mut() {
            gpu.pheromone.step(PHEROMONE_DIFFUSION_PER_CH[0], PHEROMONE_DECAY_PER_CH[0], dt);
            for ch in 1..N_PHEROMONE_CHANNELS {
                self.pheromone_fields[ch].step(
                    PHEROMONE_DIFFUSION_PER_CH[ch],
                    PHEROMONE_DECAY_PER_CH[ch],
                    dt,
                );
            }
            return;
        }
        for ch in 0..N_PHEROMONE_CHANNELS {
            match self.pheromone_masks[ch].as_deref() {
                Some(mask) => self.pheromone_fields[ch].step_masked(
                    PHEROMONE_DIFFUSION_PER_CH[ch],
                    PHEROMONE_DECAY_PER_CH[ch],
                    dt,
                    mask,
                ),
                None => self.pheromone_fields[ch].step(
                    PHEROMONE_DIFFUSION_PER_CH[ch],
                    PHEROMONE_DECAY_PER_CH[ch],
                    dt,
                ),
            }
        }
    }

    fn update_vibration(&mut self, dt: f32) {
        // Deposit each cell's motion-driven emission, then diffuse + decay
        // BEFORE this tick's brain_act samples the gradient. Brain therefore
        // sees a propagated field that already reflects this tick's motion —
        // same one-tick model would also work, but we want vibrations to
        // feel immediate (cells reacting to "right now" stir).
        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.gpu_full.as_mut() {
            for cell in &self.cells {
                let emit = bioscape::vibration_emit_for_cell(cell);
                if emit > 0.0 {
                    gpu.vibration.add_source(cell.position, emit * dt);
                }
            }
            gpu.vibration.step(VIBRATION_DIFFUSION, VIBRATION_DECAY, dt);
            return;
        }
        for cell in &self.cells {
            let emit = bioscape::vibration_emit_for_cell(cell);
            if emit > 0.0 {
                self.vibration.add_source(cell.position, emit * dt);
            }
        }
        match self.vibration_mask.as_deref() {
            Some(mask) => {
                self.vibration
                    .step_masked(VIBRATION_DIFFUSION, VIBRATION_DECAY, dt, mask)
            }
            None => self.vibration.step(VIBRATION_DIFFUSION, VIBRATION_DECAY, dt),
        }
    }

    fn emit_pheromones(&mut self, dt: f32) {
        // Sprint 126: per-channel emission. Brain output sloty:
        //   [2]  = ch0 (slow, mating-friendly)
        //   [10] = ch1 (medium decay)
        //   [11] = ch2 (fast decay, bursty / temporal patterning)
        // Cost = sum všech positive emisí × PHEROMONE_COST_PER_RATE.
        const EMIT_SLOTS: [usize; N_PHEROMONE_CHANNELS] = [2, 10, 11];

        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.gpu_full.as_mut() {
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
                    if ch == 0 {
                        gpu.pheromone.add_source(pos, rate * dt);
                    } else {
                        self.pheromone_fields[ch].add_source(pos, rate * dt);
                    }
                    let prev = cell.last_emit[ch];
                    let delta = brain_emit - prev;
                    cell.burst_accum[ch] += delta * delta;
                }
                cell.last_emit = emits;
                cell.energy -= PHEROMONE_COST_PER_RATE * total_emit * dt;
            }
            return;
        }
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
                self.pheromone_fields[ch].add_source(pos, rate * dt);
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
        #[cfg(feature = "gpu")]
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
    #[cfg(feature = "gpu")]
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

        // Phase 3: GPU spatial hash dispatch (no readback).
        gpu.cell_hash.dispatch(&positions);
        gpu.food_hash.dispatch(&food_positions);

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
            &gpu.vibration,
            sensor_params,
        );

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
            reproduce_threshold: REPRODUCE_THRESHOLD,
            vibration_norm_gain: bioscape::VIBRATION_NORMALIZATION_GAIN,
            _pad0: 0,
        };
        gpu.populate
            .dispatch(&gpu.cells, &gpu.sensor, populate_params);

        // Phase 6: GPU brain forward_persistent. Čte `last_inputs_buf` direct,
        // píše last_hidden + last_outputs storage buffers.
        gpu.brain.forward_persistent(&gpu.cells, n, &gpu.scratch.hidden_ns);

        // Phase 7: GPU motor.dispatch_with_cells. Čte last_outputs + heading/
        // pitch/turn_rate/eff_radius/max_speed, mutuje velocity/angular_vel/
        // pitch_vel buffery in-place. Mirror lib::Cell::apply_brain_motor.
        gpu.motor
            .dispatch_with_cells(&gpu.cells, n, dt, DRAG_COEFFICIENT);

        // Phase 8: GPU brownian dispatch (Sprint 51 path). Mutuje velocity_buf
        // s xoshiro128++ per-cell noise. Fused do brain_act → apply_brownian
        // fáze v `--gpu-full` skipne (compute už proběhl).
        let has_z = WORLD_HALF[2] > 0.0;
        gpu.brownian
            .compute_persistent(&gpu.cells, n, THERMAL_NOISE, dt, has_z);

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
            thermal_top: bioscape::THERMAL_TOP,
            thermal_bottom: bioscape::THERMAL_BOTTOM,
            thermal_q10: bioscape::THERMAL_Q10,
            thermal_ref_temp: bioscape::THERMAL_REF_TEMP,
            // Sprint 86: per-tick phase fractions, pre-computed na CPU aby
            // shader nemusel řešit u64 modulo + f32 cast.
            thermal_diurnal_amp: bioscape::THERMAL_DIURNAL_AMP,
            thermal_seasonal_amp: bioscape::THERMAL_SEASONAL_AMP,
            thermal_diurnal_phase: (self.clock.tick % bioscape::THERMAL_DIURNAL_PERIOD_TICKS)
                as f32
                / bioscape::THERMAL_DIURNAL_PERIOD_TICKS as f32,
            thermal_seasonal_phase: (self.clock.generation % CYCLE_GEN_PERIOD) as f32
                / CYCLE_GEN_PERIOD as f32,
            thermal_log2_q10: bioscape::THERMAL_Q10.log2(),
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

        // Phase 10: single batch readback (Sprint 63: 9 buffers fused do
        // jednoho Wait barrier). Pre-fix path 9 fresh Vec collect/to_vec —
        // teď přepíše persistent scratch sloty, 0 alloc/free při stable cap.
        gpu.cells.download_full_batch_into(
            n,
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

        // Phase 11: CPU writeback. NO apply_brain_motor + NO Cell::step CPU
        // (oba byly GPU-side). damage_accum reset (mirror populate_inputs).
        let dl = &gpu.scratch;
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

    #[cfg(feature = "gpu")]
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
                let skip_cone = fov >= bioscape::MAX_VISION_FOV;
                let cos_fov = fov.cos();
                let fwd = bioscape::forward_vector(cell.heading, cell.pitch);

                let mut best_food: Option<[f32; 3]> = None;
                let mut best_food_d2 = f32::MAX;
                food_grid.for_each_in_radius_toroidal(pos, vision_r, WORLD_HALF, |_id, fp, _| {
                    let d = bioscape::min_image_delta(pos, fp, WORLD_HALF);
                    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if d2 > vr2 || d2 >= best_food_d2 {
                        return;
                    }
                    if !skip_cone && !bioscape::fov_cone_accept(d, d2, fwd, cos_fov) {
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
                    let d = bioscape::min_image_delta(pos, coop.position, WORLD_HALF);
                    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if d2 > vr2 || d2 >= best_food_d2 {
                        continue;
                    }
                    if !skip_cone && !bioscape::fov_cone_accept(d, d2, fwd, cos_fov) {
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
                    let d = bioscape::min_image_delta(pos, op, WORLD_HALF);
                    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if d2 > vr2 {
                        return;
                    }
                    if !skip_cone && !bioscape::fov_cone_accept(d, d2, fwd, cos_fov) {
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
                    bioscape::temperature_at_z(pos[2], WORLD_HALF, tick, gen);
                let vibration_grad =
                    vibration.gradient_at(pos_xyz, VIBRATION_SAMPLE_EPSILON);
                let vibration_amp = vibration.sample(pos_xyz);
                let sensors = bioscape::BrainSensors {
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
                let mut inputs = bioscape::populate_brain_inputs(cell, &sensors, vision_r);
                bioscape::apply_sensor_gains(&mut inputs, &cell.genome.sensor_gains);
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
                bioscape::pool_bonded_sensors(cell, &own, |partner_id| {
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
            let pooled = bioscape::pool_bonded_hidden(cell, |partner_id| {
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
        let snapshot: Vec<[f32; bioscape::BRAIN_OUTPUTS]> =
            self.cells.iter().map(|c| c.last_outputs).collect();
        let id_to_idx = &self.id_to_idx_scratch;
        for (i, cell) in self.cells.iter_mut().enumerate() {
            let inbox = bioscape::pool_bond_messages(cell, |partner_id| {
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
                let skip_cone = fov >= bioscape::MAX_VISION_FOV;
                let cos_fov = fov.cos();
                let fwd = bioscape::forward_vector(cell.heading, cell.pitch);

                let mut best_food: Option<[f32; 3]> = None;
                let mut best_food_d2 = f32::MAX;
                food_grid.for_each_in_radius_toroidal(pos, vision_r, WORLD_HALF, |_id, fp, _| {
                    let d = bioscape::min_image_delta(pos, fp, WORLD_HALF);
                    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if d2 > vr2 || d2 >= best_food_d2 {
                        return;
                    }
                    if !skip_cone && !bioscape::fov_cone_accept(d, d2, fwd, cos_fov) {
                        return;
                    }
                    if !bioscape::los_clear(obstacles, pos, fp) {
                        return;
                    }
                    best_food_d2 = d2;
                    best_food = Some(d);
                });
                // Sprint 128: coop food candidates injected do same nearest_food
                // selection. Linear scan — typický coop_foods.len() ≤ 8.
                for coop in coop_foods.iter() {
                    let d = bioscape::min_image_delta(pos, coop.position, WORLD_HALF);
                    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if d2 > vr2 || d2 >= best_food_d2 {
                        continue;
                    }
                    if !skip_cone && !bioscape::fov_cone_accept(d, d2, fwd, cos_fov) {
                        continue;
                    }
                    if !bioscape::los_clear(obstacles, pos, coop.position) {
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
                    let d = bioscape::min_image_delta(pos, op, WORLD_HALF);
                    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if d2 > vr2 {
                        return;
                    }
                    if !skip_cone && !bioscape::fov_cone_accept(d, d2, fwd, cos_fov) {
                        return;
                    }
                    if !bioscape::los_clear(obstacles, pos, op) {
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
                    bioscape::temperature_at_z(pos[2], WORLD_HALF, tick, gen);
                let vibration_grad =
                    vibration.gradient_at(pos_xyz, VIBRATION_SAMPLE_EPSILON);
                let vibration_amp = vibration.sample(pos_xyz);
                let sensors = bioscape::BrainSensors {
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
                let mut inputs = bioscape::populate_brain_inputs(cell, &sensors, vision_r);
                bioscape::apply_sensor_gains(&mut inputs, &cell.genome.sensor_gains);
                *inputs_slot = inputs;
            });

        let id_to_idx = &self.id_to_idx_scratch;
        let inputs_scratch = &self.inputs_scratch;

        self.cells
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, cell)| {
                let own = inputs_scratch[i];
                let pooled = bioscape::pool_bonded_sensors(cell, &own, |partner_id| {
                    let idx = id_to_idx.get(&partner_id).copied()?;
                    if idx == i {
                        return None;
                    }
                    Some(inputs_scratch[idx])
                });
                let (hidden, outputs) = cell.genome.brain.forward_with_state(&pooled);
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
        #[cfg(feature = "gpu")]
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
        let events = &self.events.events;
        let ctx = bioscape::ThermalCtx::for_tick(tick, gen);
        let obstacles = self.obstacles.as_ref();
        for cell in &mut self.cells {
            let climate_offset = bioscape::climate_shock_offset(
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
        #[cfg(feature = "gpu")]
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
            gpu.smell.upload_obstacle_mask(&smell_mask);
            gpu.pheromone.upload_obstacle_mask(&phero_mask);
            gpu.vibration.upload_obstacle_mask(&vib_mask);
        }
        self.obstacles = Some(field);
    }

    /// Wave 3: per-tick eligibility-trace decay + accumulate. Runs after
    /// brain_act so traces capture the (input, hidden, output) activations
    /// from this tick before any reward event fires. No weight changes —
    /// just trace bookkeeping. Always-on (independent of maze toggle):
    /// decay constant `HEBBIAN_TRACE_DECAY_PER_SEC` works as a soft replay
    /// window for sparse-reward credit assignment.
    pub fn apply_eligibility_step(&mut self, dt: f32) {
        use bioscape::HEBBIAN_TRACE_DECAY_PER_SEC;
        self.cells.par_iter_mut().for_each(|cell| {
            let last_inputs = cell.last_inputs;
            let last_hidden = cell.last_hidden;
            let last_outputs = cell.last_outputs;
            cell.genome.brain.hebbian_step(
                &last_inputs,
                &last_hidden,
                &last_outputs,
                dt,
                HEBBIAN_TRACE_DECAY_PER_SEC,
            );
        });
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
        use bioscape::{LEARNING_RATE, NOVELTY_REWARD_MAGNITUDE};
        let half = WORLD_HALF;
        #[cfg(feature = "gpu")]
        if self.gpu_full.is_some() {
            // Decide novelty per cell first (read-only on cell state); then
            // apply rewards via the GPU dispatch in one pass.
            let mut rewards: Vec<f32> = vec![0.0; self.cells.len()];
            for (i, cell) in self.cells.iter_mut().enumerate() {
                let v = Cell::novelty_voxel_index(cell.position, half);
                if cell.check_novelty(v) {
                    rewards[i] = NOVELTY_REWARD_MAGNITUDE;
                }
            }
            if rewards.iter().any(|&r| r != 0.0) {
                let n = self.cells.len();
                let gpu = self.gpu_full.as_ref().unwrap();
                gpu.cells.upload_rewards(&rewards);
                gpu.hebbian
                    .dispatch_apply_reward_persistent(&gpu.cells, n, LEARNING_RATE);
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
            cell.genome.brain.hebbian_apply_reward(
                &last_hidden,
                &last_outputs,
                NOVELTY_REWARD_MAGNITUDE,
                LEARNING_RATE,
            );
        });
    }

    /// Per-tick whisker raycast pass. Fills `cell.last_whisker_distances`
    /// from `obstacles` so the sensor gather phase reads from the cell
    /// without needing `&ObstacleField` plumbed through. No-op (leaves
    /// defaults at 1.0 = "clear") when `obstacles` is `None`.
    pub fn update_whiskers(&mut self) {
        let Some(field) = self.obstacles.as_ref() else {
            return;
        };
        self.cells.par_iter_mut().for_each(|cell| {
            cell.last_whisker_distances =
                field.whisker_distances(cell.position, cell.heading, cell.pitch);
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
    /// a `write_stats` (CSV column `shock_climate_offset`).
    pub fn climate_offset_at(&self, pos_xy: [f32; 2]) -> f32 {
        bioscape::climate_shock_offset(
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
        let events = &self.events.events;
        for cell in &mut self.cells {
            let noise = self
                .map
                .sample([cell.position[0], cell.position[1], cell.position[2]]);
            let shock_mult = bioscape::hazard_shock_multiplier(
                cell.position,
                events,
                gen,
                tick,
                WORLD_HALF,
            );
            let drain = hazard_drain(noise) * dt * shock_mult;
            cell.energy -= drain;
            cell.damage_accum += drain;
        }
    }

    /// Wave H: GPU replacement for the CPU Phase 1 broad-phase pair loop.
    /// Fills `deltas_scratch`, `velocity_deltas_scratch` and
    /// `contact_lists_scratch` from a single GPU dispatch. Phase 2 / 3 / 4
    /// of `resolve_collisions` consume these scratch buffers identically
    /// regardless of which path produced them.
    #[cfg(feature = "gpu")]
    fn resolve_collisions_gpu_pass1(&mut self) {
        let n = self.cells.len();
        if n == 0 {
            return;
        }
        let positions: Vec<[f32; 3]> = self.cells.iter().map(|c| c.position).collect();
        let velocities: Vec<[f32; 3]> = self.cells.iter().map(|c| c.velocity).collect();
        let eff_radii: Vec<f32> = self
            .cells
            .iter()
            .map(|c| c.phenotype.effective_radius())
            .collect();
        let max_axes: Vec<f32> = self.cells.iter().map(|c| c.phenotype.max_axis()).collect();
        let adhesion_types: Vec<u32> = self
            .cells
            .iter()
            .map(|c| c.genome.adhesion_type as u32)
            .collect();

        let slots = MAX_BONDS_PER_CELL;
        let total = n * slots;
        let mut partner_idx = vec![-1_i32; total];
        let mut rest = vec![0.0_f32; total];
        let mut stiff = vec![0.0_f32; total];
        let mut damp = vec![0.0_f32; total];
        for i in 0..n {
            for (s, slot) in self.cells[i].bonds.iter().enumerate() {
                if let Some(b) = slot {
                    if let Some(&j) = self.id_to_idx_scratch.get(&b.other_cell_id) {
                        let idx = i * slots + s;
                        partner_idx[idx] = j as i32;
                        rest[idx] = b.rest_length;
                        stiff[idx] = b.stiffness;
                        damp[idx] = b.damping;
                    }
                }
            }
        }

        // Refresh GPU spatial hash with current (post-step) positions. The
        // hash from brain_act was keyed on pre-step positions; cells have
        // moved since.
        let result = {
            let gpu = self.gpu_full.as_mut().expect("gpu_full Some");
            gpu.cell_hash.dispatch(&positions);
            gpu.collision.compute(
                &positions,
                &velocities,
                &eff_radii,
                &max_axes,
                &adhesion_types,
                &partner_idx,
                &rest,
                &stiff,
                &damp,
                &gpu.cell_hash,
            )
        };

        let max_contacts = bioscape::MAX_COLLISION_CONTACTS_PER_CELL as usize;
        for i in 0..n {
            self.deltas_scratch[i] = result.position_deltas[i];
            self.velocity_deltas_scratch[i] = result.velocity_deltas[i];
        }
        // Contact events: GPU dedupes by idx (i < j), but CPU `contact_progress`
        // keys on `(cell_id_min, cell_id_max)` — idx and cell_id orderings are
        // independent. Re-canonicalize here so the lower-id cell's contact list
        // always carries the higher-id partner.
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
        self.cell_grid.rebuild(
            self.cells
                .iter()
                .enumerate()
                .map(|(i, c)| (i, c.position, c.phenotype.effective_radius())),
        );
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

        // Wave H: GPU collision shader covers the broad-phase pair loop
        // (depenetration + velocity damping + adhesion + spring bond forces
        // + contact event detection). When `gpu_full` is active we fill the
        // scratch buffers from the GPU result and skip the CPU par_iter.
        #[cfg(feature = "gpu")]
        let used_gpu = if self.gpu_full.is_some() {
            self.resolve_collisions_gpu_pass1();
            true
        } else {
            false
        };
        #[cfg(not(feature = "gpu"))]
        let used_gpu = false;

        // Sprint 66: search radius = max(collision, adhesion). Adhesion má
        // dosah pair_r × ADHESION_RANGE_FACTOR; pair_r = CELL_RADIUS × max_axis × 2.
        // Pro jistotu používáme effective_radius_i × CELL_RADIUS × 2 × FACTOR.
        // Per-i collected contact pairs: (other_cell_id, currently_in_contact).
        // Phase 2 sequentially merges do contact_progress.
        if !used_gpu {
        let id_to_idx = &self.id_to_idx_scratch;
        let cell_grid = &self.cell_grid;
        let cells = &self.cells;
        self.deltas_scratch
            .par_iter_mut()
            .zip(self.velocity_deltas_scratch.par_iter_mut())
            .zip(self.contact_lists_scratch.par_iter_mut().take(n))
            .enumerate()
            .for_each(|(i, ((delta, vel_delta), local_contacts))| {
                let pos_i = cells[i].position;
                let vel_i = cells[i].velocity;
                let radius_i = cells[i].phenotype.effective_radius();
                let type_i = cells[i].genome.adhesion_type;
                let collision_r = CELL_RADIUS * (radius_i + cells[i].phenotype.max_axis() * 2.0);
                let adhesion_r =
                    CELL_RADIUS * (radius_i + cells[i].phenotype.max_axis() * 2.0)
                        * ADHESION_RANGE_FACTOR;
                let search_r = collision_r.max(adhesion_r);
                let cell_id_i = cells[i].cell_id;
                cell_grid.for_each_in_radius_toroidal(
                    pos_i,
                    search_r,
                    WORLD_HALF,
                    |id_j, pos_j, radius_j| {
                        if id_j == i {
                            return;
                        }
                        let pair_r = CELL_RADIUS * (radius_i + radius_j);
                        let pair_r2 = pair_r * pair_r;
                        // d_vec = pos_i - pos_j (j → i direction). Push i along +d_vec.
                        let d_vec = bioscape::min_image_delta(pos_j, pos_i, WORLD_HALF);
                        let d2 = d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2];
                        let d = d2.sqrt();
                        let in_contact = d2 < pair_r2 && d2 > 0.0;
                        if in_contact {
                            let overlap = pair_r - d;
                            let nx = d_vec[0] / d;
                            let ny = d_vec[1] / d;
                            let nz = d_vec[2] / d;
                            // Position depenetration (mass-symmetric, halved).
                            delta[0] += nx * overlap * 0.5;
                            delta[1] += ny * overlap * 0.5;
                            delta[2] += nz * overlap * 0.5;
                            // Sprint 65: velocity damping (inelastic).
                            let vel_j = cells[id_j].velocity;
                            let v_rel = [
                                vel_i[0] - vel_j[0],
                                vel_i[1] - vel_j[1],
                                vel_i[2] - vel_j[2],
                            ];
                            let v_rel_n =
                                v_rel[0] * nx + v_rel[1] * ny + v_rel[2] * nz;
                            if v_rel_n < 0.0 {
                                let damp =
                                    -v_rel_n * 0.5 * (1.0 - COLLISION_RESTITUTION);
                                vel_delta[0] += damp * nx;
                                vel_delta[1] += damp * ny;
                                vel_delta[2] += damp * nz;
                            }
                        } else if d > 0.0 {
                            // Sprint 66: soft adhesion mimo kontakt.
                            let type_j = cells[id_j].genome.adhesion_type;
                            let same_type = type_i == type_j;
                            let dv = adhesion_velocity_delta(d_vec, d, pair_r, same_type);
                            vel_delta[0] += dv[0];
                            vel_delta[1] += dv[1];
                            vel_delta[2] += dv[2];
                        }
                        // Contact tracker — recordujeme jen když cell_id_i je
                        // nižší (deduping symmetric pair). bond_active stranu
                        // ověříme v Phase 2 (potřeba čerstvý last_outputs[9]).
                        let cell_id_j = cells[id_j].cell_id;
                        if in_contact && cell_id_i < cell_id_j {
                            local_contacts.push(cell_id_j);
                        }
                    },
                );
                // Sprint 66: aplikuj spring bond force pro každý living bond
                // této cells. Pokud bond pointer dangling (cíl mrtvý) nebo
                // overstretched, vrátíme zpět informaci pro phase-2 cleanup.
                for bond_opt in cells[i].bonds.iter() {
                    if let Some(bond) = bond_opt {
                        if let Some(&j_idx) = id_to_idx.get(&bond.other_cell_id) {
                            let pos_j = cells[j_idx].position;
                            let vel_j = cells[j_idx].velocity;
                            let d_vec =
                                bioscape::min_image_delta(pos_j, pos_i, WORLD_HALF);
                            let dist =
                                (d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2])
                                    .sqrt();
                            let (dv, _broken) =
                                bond_velocity_delta(bond, d_vec, dist, vel_i, vel_j);
                            // Apply force only — the actual break decision
                            // se rozhodne v Phase 2 (zápis vyžaduje &mut cells).
                            vel_delta[0] += dv[0];
                            vel_delta[1] += dv[1];
                            vel_delta[2] += dv[2];
                        }
                    }
                }
            });
        }

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
                let d_vec = bioscape::min_image_delta(pos_j, pos_i, WORLD_HALF);
                let d = (d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2]).sqrt();
                if d > bond.rest_length * bioscape::BOND_BREAK_FACTOR || d <= f32::EPSILON {
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
            let d_vec = bioscape::min_image_delta(pos_b, pos_a, WORLD_HALF);
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
    }

    /// Wave H: GPU predation dispatch. Computes herd_counts + per-pair
    /// energy/damage deltas in one shader pass and applies them to cells.
    /// Pack-hunting CSV diagnostics (`bonded_attacks_gen` etc.) and
    /// `predation_events_gen` are NOT computed on this path — the shader
    /// doesn't emit per-event tuples. Extending `predate.wgsl` with an
    /// atomic event buffer is follow-up work.
    #[cfg(feature = "gpu")]
    fn predate_gpu(&mut self) {
        let n = self.cells.len();
        if n == 0 {
            return;
        }
        let positions: Vec<[f32; 3]> = self.cells.iter().map(|c| c.position).collect();
        let eff_radii: Vec<f32> = self
            .cells
            .iter()
            .map(|c| c.phenotype.effective_radius())
            .collect();
        let headings: Vec<f32> = self.cells.iter().map(|c| c.heading).collect();
        let pitches: Vec<f32> = self.cells.iter().map(|c| c.pitch).collect();
        let attack_signals: Vec<f32> = self
            .cells
            .iter()
            .map(|c| c.last_outputs[6].max(0.0))
            .collect();
        let mut spike_counts: Vec<u32> = Vec::with_capacity(n);
        let mut spikes_packed: Vec<[f32; 4]> = Vec::with_capacity(n * bioscape::SPIKE_SLOTS);
        for cell in &self.cells {
            let mut active = 0u32;
            for s in 0..bioscape::SPIKE_SLOTS {
                let spike = cell.phenotype.spikes[s];
                if spike.length > 0.0 {
                    active += 1;
                }
                spikes_packed.push([
                    spike.length,
                    spike.azimuth_offset,
                    spike.elevation_offset,
                    spike.complexity,
                ]);
            }
            spike_counts.push(active);
        }
        let params = PredateParamsGpu {
            num_cells: 0, // filled by compute()
            cell_size: bioscape::GRID_CELL_SIZE,
            cell_radius_const: bioscape::CELL_RADIUS,
            size_ratio_threshold: bioscape::SIZE_RATIO_THRESHOLD,
            herd_radius_sq: bioscape::HERD_RADIUS * bioscape::HERD_RADIUS,
            attack_threshold: bioscape::ATTACK_THRESHOLD,
            predation_gain: bioscape::PREDATION_GAIN_PER_TICK * self.predation_gain_mult,
            predation_drain: bioscape::PREDATION_DRAIN_PER_TICK * self.predation_drain_mult,
            spike_dot_threshold: bioscape::SPIKE_DOT_THRESHOLD,
            spike_bonus: bioscape::SPIKE_PREDATION_BONUS,
            dilution_k: bioscape::DILUTION_K,
            world_half_x: WORLD_HALF[0],
            world_half_y: WORLD_HALF[1],
            ..PredateParamsGpu::default()
        };
        let result = {
            let gpu = self.gpu_full.as_mut().expect("gpu_full Some");
            // Refresh GPU spatial hash with current (post-resolve_collisions)
            // positions before the predate dispatch.
            gpu.cell_hash.dispatch(&positions);
            gpu.predate.compute(
                &positions,
                &eff_radii,
                &headings,
                &pitches,
                &spikes_packed,
                &spike_counts,
                &attack_signals,
                &gpu.cell_hash,
                params,
            )
        };
        for (i, cell) in self.cells.iter_mut().enumerate() {
            cell.energy += result.energy_delta[i];
            cell.damage_accum += result.damage_delta[i];
        }
    }

    fn predate(&mut self) {
        #[cfg(feature = "gpu")]
        if self.gpu_full.is_some() {
            self.predate_gpu();
            return;
        }
        // Sprint 43: cell_grid build (sdílený s brain_act ale tam refresh tickem
        // později; rebuildujeme pro jistotu — pozice se mohly hnout v `step()`).
        // Pass 1 (herd_counts): par_iter, write-only per i. Pass 2 (attack
        // events): sekvenční, protože write na victim může kolidovat napříč
        // attackers; grid lookup ale srazí cost na O(N·k).
        let n = self.cells.len();
        self.cell_grid.rebuild(
            self.cells
                .iter()
                .enumerate()
                .map(|(i, c)| (i, c.position, c.phenotype.effective_radius())),
        );
        self.energy_deltas_scratch.clear();
        self.energy_deltas_scratch.resize(n, 0.0);
        self.damage_deltas_scratch.clear();
        self.damage_deltas_scratch.resize(n, 0.0);

        let herd_r2 = HERD_RADIUS * HERD_RADIUS;
        let cells = &self.cells;
        let cell_grid = &self.cell_grid;
        let pred_gain_mult = self.predation_gain_mult;
        let pred_drain_mult = self.predation_drain_mult;
        let herd_counts: Vec<u32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let pos_i = cells[i].position;
                let mut count = 0u32;
                cell_grid.for_each_in_radius_toroidal(pos_i, HERD_RADIUS, WORLD_HALF, |id_j, pos_j, _| {
                    if id_j == i {
                        return;
                    }
                    let d = bioscape::min_image_delta(pos_i, pos_j, WORLD_HALF);
                    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if d2 < herd_r2 {
                        count += 1;
                    }
                });
                count
            })
            .collect();

        // Sprint 57: paralelní attack candidate gathering. Pass 2a sbírá
        // (i, j, gain) eventy bez sdílených writes; Pass 2b aggreguje sekvenčně
        // do energy/damage scratch (řeší race na victim j shared mezi attackery).
        // Spike attack cache built once per attacker — `spike_direction` etc.
        // depend only on attacker pose + spikes, not on per-pair target.
        let attack_events: Vec<(usize, usize, f32, f32)> = (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let attack_signal = cells[i].last_outputs[6].max(0.0);
                if attack_signal <= ATTACK_THRESHOLD {
                    return Vec::new();
                }
                let pos_i = cells[i].position;
                let radius_a = cells[i].phenotype.effective_radius();
                let search_r =
                    CELL_RADIUS * (radius_a + cells[i].phenotype.max_axis() * 2.0);
                let spike_cache = cells[i].build_spike_attack_cache();
                let mut local: Vec<(usize, usize, f32, f32)> = Vec::new();
                cell_grid.for_each_in_radius_toroidal(
                    pos_i,
                    search_r,
                    WORLD_HALF,
                    |j, pos_j, radius_b| {
                        if j == i {
                            return;
                        }
                        if radius_a < SIZE_RATIO_THRESHOLD * radius_b {
                            return;
                        }
                        let pair_r = CELL_RADIUS * (radius_a + radius_b);
                        let pair_r2 = pair_r * pair_r;
                        let d_vec = bioscape::min_image_delta(pos_j, pos_i, WORLD_HALF);
                        let d2 = d_vec[0] * d_vec[0]
                            + d_vec[1] * d_vec[1]
                            + d_vec[2] * d_vec[2];
                        if d2 < pair_r2 {
                            let mut gain = PREDATION_GAIN_PER_TICK;
                            // Sprint 122: multi-spike per-spike cone test +
                            // complexity multiplikátor (S123 zapne complexity).
                            gain += cells[i].spike_bonus_against_cached(pos_j, &spike_cache);
                            let dilution = 1.0 / (1.0 + DILUTION_K * herd_counts[j] as f32);
                            // Sprint 69: bonded prey takes less damage + yields
                            // less energy. Group-defense benefit činí bondování
                            // evolučně positive (Sprint 67.1 ukázal opak bez něj).
                            let defense = bioscape::bond_defense_factor(cells[j].n_bonds());
                            gain *= dilution * defense * pred_gain_mult;
                            local.push((i, j, gain, defense));
                        }
                    },
                );
                local
            })
            .collect();

        // Pack-hunting metrics: per-victim attacker grouping + bonded vs solo
        // per-attack gain. Done before energy mutation tak, aby `cells` shared
        // borrow zůstal platný.
        let mut by_victim: rustc_hash::FxHashMap<usize, Vec<usize>> = rustc_hash::FxHashMap::default();
        let mut tick_bonded_n = 0u64;
        let mut tick_solo_n = 0u64;
        let mut tick_bonded_gain = 0.0_f64;
        let mut tick_solo_gain = 0.0_f64;
        for (i, j, gain, _) in &attack_events {
            by_victim.entry(*j).or_default().push(*i);
            if cells[*i].n_bonds() >= 1 {
                tick_bonded_n += 1;
                tick_bonded_gain += *gain as f64;
            } else {
                tick_solo_n += 1;
                tick_solo_gain += *gain as f64;
            }
        }
        let tick_victims = by_victim.len() as u64;
        let mut tick_swarm = 0u64;
        let mut tick_pack = 0u64;
        for atks in by_victim.values() {
            let mut uniq: Vec<usize> = atks.clone();
            uniq.sort_unstable();
            uniq.dedup();
            if uniq.len() < 2 {
                continue;
            }
            tick_swarm += 1;
            let bonded_pair = (0..uniq.len()).any(|a| {
                let id_a = cells[uniq[a]].cell_id;
                uniq[a + 1..].iter().any(|&b| {
                    cells[b]
                        .bonds
                        .iter()
                        .any(|opt| opt.is_some_and(|bond| bond.other_cell_id == id_a))
                })
            });
            if bonded_pair {
                tick_pack += 1;
            }
        }
        self.bonded_attacks_gen += tick_bonded_n;
        self.solo_attacks_gen += tick_solo_n;
        self.bonded_attack_gain_sum_gen += tick_bonded_gain;
        self.solo_attack_gain_sum_gen += tick_solo_gain;
        self.swarm_attacks_gen += tick_swarm;
        self.pack_attacks_gen += tick_pack;
        self.attack_victims_gen += tick_victims;

        let events: u64 = attack_events.len() as u64;
        for (i, j, gain, defense) in attack_events {
            self.energy_deltas_scratch[i] += gain;
            // Sprint 69: defense škáluje i drain + damage (consistent s gain).
            let drain = PREDATION_DRAIN_PER_TICK * defense * pred_drain_mult;
            self.energy_deltas_scratch[j] -= drain;
            self.damage_deltas_scratch[j] += drain;
        }
        self.predation_events_gen += events;
        for ((cell, energy_delta), dmg_delta) in self
            .cells
            .iter_mut()
            .zip(self.energy_deltas_scratch.iter())
            .zip(self.damage_deltas_scratch.iter())
        {
            cell.energy += energy_delta;
            cell.damage_accum += dmg_delta;
        }
    }

    fn eat_food(&mut self) {
        // Sprint 43: food_grid lookup místo full sweep.
        // Sprint 57: 3-pass paralelizace. Pass 1 paralelně vybere candidate
        // food per cell (read-only test), Pass 2 sekvenčně resolvne race
        // (first-cell-wins per food + Hebbian na CPU), Pass 3 swap_remove.
        // GPU Hebbian zůstává na konci jako pre-Sprint-57.
        self.food_grid.rebuild(
            self.foods
                .iter()
                .enumerate()
                .map(|(i, f)| (i, f.position, f.kind)),
        );
        self.eaten_scratch.clear();
        self.eaten_scratch.resize(self.foods.len(), false);

        // Sprint 78: cell_id → idx map pro food share lookup. Reuse persistent
        // scratch (built v `tick` start, cells layout v eat_food beze změny).
        let id_to_idx = &self.id_to_idx_scratch;

        #[cfg(feature = "gpu")]
        let use_gpu_hebbian = self.gpu_full.is_some();
        #[cfg(not(feature = "gpu"))]
        let use_gpu_hebbian = false;

        let mut rewards: Vec<f32> = if use_gpu_hebbian {
            vec![0.0; self.cells.len()]
        } else {
            Vec::new()
        };

        // Pass 1 (parallel): per-cell candidate selection. Žádná mutace.
        // First match v grid traversal wins (zachovává pre-Sprint-57 sémantiku).
        let cells = &self.cells;
        let foods = &self.foods;
        let map = &self.map;
        let food_grid = &self.food_grid;
        // eat_food skip optim experiment (Sprint 132+ navrh): cache d² nejbližšího
        // food ze sensor scan, skipnout food_grid query pokud `> threshold`.
        // PROBLÉM: cell může být pohlcen kolizí s bonded sousedem (mass-symmetric
        // depenetration + spring impulse), pohyb překračuje konzervativní slack.
        // Pass 2 first-cell-wins ordering pak diverguje → CSV reproducibility
        // breaks. Skip je v headless **disabled** (determinismus je sacred);
        // `cell.last_best_food_d2` field se v headless nečte (renderer ho používá
        // pro interaktivní speed při lower determinism budget).
        let candidates: Vec<Option<(usize, f32)>> = cells
            .par_iter()
            .map(|cell| {
                let pos = cell.position;
                let search_r = EAT_RADIUS * cell.phenotype.max_axis();
                let mut ate: Option<(usize, f32)> = None;
                food_grid.for_each_in_radius_toroidal(
                    pos,
                    search_r,
                    WORLD_HALF,
                    |idx, _fp, _kind| {
                        if ate.is_some() {
                            return;
                        }
                        let food = &foods[idx];
                        let md = bioscape::min_image_delta(pos, food.position, WORLD_HALF);
                        let ghost = Food {
                            position: [pos[0] + md[0], pos[1] + md[1], food.position[2]],
                            age_ticks: food.age_ticks,
                            kind: food.kind,
                        };
                        if cell.eat_test(&ghost, EAT_RADIUS) {
                            // Sprint 92: food value = base_value(kind) ×
                            // multiplier × decay × eat_efficiency(kind, score).
                            let efficiency = bioscape::eat_efficiency(
                                food.kind,
                                cell.genome.carnivore_score,
                            );
                            let value = bioscape::food_base_value(food.kind)
                                * food_multiplier(
                                    map.sample([food.position[0], food.position[1], 0.0]),
                                )
                                * food.value_factor()
                                * efficiency;
                            ate = Some((idx, value));
                        }
                    },
                );
                ate
            })
            .collect();

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
                {
                    let cell = &mut self.cells[cell_idx];
                    cell.energy += *value;
                    bonds_copy = cell.bonds;
                    donor_state = cell.cell_state;
                    donor_lineage = cell.lineage_id;
                }
                // Sprint 80: donor's cell_state modulates share fraction.
                // State≈0 (selfish) → ~0%; state≈1 (altruist) → plný 30%.
                // Sprint 87: cluster-size bonus — víc bondů → vyšší share per
                // partner. Empirie 300-gen: tissue regime kolaboval do gen 200,
                // bonus posiluje selekci pro velké clustery.
                // Sprint 87 Hamilton sweep: `self.share_frac` runtime override
                // místo BOND_FOOD_SHARE_FRAC; `self.kin_filter` skipne sharing
                // do partnerů s jiným lineage_id (= test relatedness coefficientu r).
                let n_bonds = bonds_copy.iter().filter(|b| b.is_some()).count() as f32;
                let cluster_mult = 1.0 + (n_bonds - 1.0).max(0.0)
                    * bioscape::BOND_FOOD_SHARE_CLUSTER_BONUS;
                let share_value =
                    *value * self.share_frac * donor_state * cluster_mult;
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
                    rewards[cell_idx] = 1.0;
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
                // Wave 3: trace-based reward — credits motor outputs from up
                // to ~120 ticks back, not just this tick's pre·post.
                cell.genome.brain.hebbian_apply_reward(
                    &last_hidden,
                    &last_outputs,
                    1.0,
                    LEARNING_RATE,
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
        #[cfg(feature = "gpu")]
        if use_gpu_hebbian {
            let n = self.cells.len();
            let gpu = self.gpu_full.as_ref().unwrap();
            gpu.cells.upload_rewards(&rewards);
            gpu.hebbian.dispatch_apply_reward_persistent(&gpu.cells, n, LEARNING_RATE);
        }
        let _ = rewards;
    }

    fn spawn_food(&mut self, rng: &mut impl Rng) {
        let target = food_target(self.density_factor * self.food_factor_mult);
        if self.foods.len() >= target {
            return;
        }
        // Sprint 43: cell_grid pro exclusion check. Rebuild reuses bucket vec
        // capacities; inner loop místo O(N) full sweep dělá O(k) per candidate.
        // Bound search radius na EAT_RADIUS × MAX_BODY_LENGTH (max_axis nemůže
        // přesáhnout MAX_BODY_LENGTH).
        self.cell_grid.rebuild(
            self.cells
                .iter()
                .enumerate()
                .map(|(i, c)| (i, c.position, c.phenotype.effective_radius())),
        );
        let to_spawn = (target - self.foods.len()).min(FOOD_SPAWN_RATE);
        let max_search_r = EAT_RADIUS * bioscape::MAX_BODY_LENGTH;
        let obstacles = self.obstacles.as_ref();
        'spawn: for _ in 0..to_spawn {
            for _ in 0..MAX_SPAWN_ATTEMPTS {
                let candidate = Food::random(rng, WORLD_HALF);
                let richness = self
                    .map
                    .sample([candidate.position[0], candidate.position[1], 0.0]);
                if reject_food_for_richness(rng, richness) {
                    continue;
                }
                if let Some(field) = obstacles {
                    if field.sample(candidate.position) {
                        continue;
                    }
                }
                let mut blocked = false;
                self.cell_grid.for_each_in_radius_toroidal(
                    candidate.position,
                    max_search_r,
                    WORLD_HALF,
                    |id, cell_pos, _r| {
                        if blocked {
                            return;
                        }
                        let exclusion = EAT_RADIUS * self.cells[id].phenotype.max_axis();
                        let d = bioscape::min_image_delta(candidate.position, cell_pos, WORLD_HALF);
                        if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < exclusion * exclusion {
                            blocked = true;
                        }
                    },
                );
                if !blocked {
                    self.foods.push(candidate);
                    continue 'spawn;
                }
            }
        }
    }

    /// Sync `Genome.brain` for every cell from the persistent GPU buffer
    /// back to CPU. In `--gpu-full + GPU CPPN` mode child brains are produced
    /// directly on the device and the CPU `Genome.brain` field stays at the
    /// `Brain::zeros()` placeholder until something explicitly downloads.
    /// Call this once per generation before serialisation / diagnostics
    /// (`w1_frobenius_std` etc.) — single `Wait` barrier, ~few ms total.
    #[cfg(feature = "gpu")]
    pub fn sync_brains_from_gpu(&mut self) {
        if let Some(gpu) = self.gpu_full.as_ref() {
            let n = self.cells.len();
            if n == 0 {
                return;
            }
            let brains = gpu.cells.download_brains(n);
            for (cell, brain) in self.cells.iter_mut().zip(brains) {
                cell.genome.brain = brain;
            }
        }
    }

    /// V7: pull the GPU vibration field back into the CPU shadow so the
    /// per-gen CSV writer can sample it at cell positions. Called from
    /// `main` right before `write_stats`; per-tick CPU shadow stays
    /// out-of-date in `--gpu-full` mode (sensor gather reads the GPU
    /// buffer direct, no readback needed mid-tick).
    #[cfg(feature = "gpu")]
    pub fn sync_vibration_from_gpu(&mut self) {
        if let Some(gpu) = self.gpu_full.as_mut() {
            let grid = gpu.vibration.download();
            self.vibration.replace_grid_from(&grid);
        }
    }
    #[cfg(not(feature = "gpu"))]
    pub fn sync_vibration_from_gpu(&mut self) {}

    fn reproduce(&mut self, rng: &mut impl Rng) {
        let current_pop = self.cells.len();
        if current_pop >= self.max_population {
            return;
        }
        let budget = self.max_population - current_pop;
        let fertile = self.collect_fertile();
        self.fertile_ticks_gen += fertile.len() as u64;
        let mating_r2 = self.mating_radius * self.mating_radius;
        let matings = bioscape::pair_fertile(&fertile, mating_r2, budget, WORLD_HALF);
        let child_start = self.cells.len();
        let to_spawn = self.spawn_children_from_matings(&matings, rng);
        let n_births = to_spawn.len();
        self.births_gen += n_births as u64;
        self.cells.extend(to_spawn);
        // GPU child upload. In `--gpu-full`, brain weights are produced
        // directly on the device via `CppnGpu::dispatch`, so we skip the
        // per-child `upload_brain_at` round-trip that the CPU path needed
        // (saves ~16 KB × n_children write_buffer per reproduce phase).
        // xoshiro seed + turn_rate uploads stay — they're independent of
        // the brain materialisation path.
        #[cfg(feature = "gpu")]
        if n_births > 0 && self.gpu_full.is_some() {
            // Borrow split: build `pairs` from `self.cells` (immutable),
            // then take `&mut self.gpu_full` for dispatch. Rust's disjoint
            // field borrows handle this because `cells` and `gpu_full` are
            // distinct fields of `self`.
            let pairs: Vec<(usize, &bioscape::Cppn)> = (0..n_births)
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
                let scaled = REPRODUCE_THRESHOLD * (1.0 + bioscape::LINEAGE_DIVERSITY_ALPHA * f);
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
        // Sync parent brains z GPU (sprint 51 GPU Hebbian je canonical).
        // In `--gpu-full + GPU CPPN` mode the parent brain isn't actually read
        // by `make_mating_child_no_brain` (mating only touches `Genome.cppn`),
        // but we keep the download for serialization parity — diagnostic
        // metrics like `w1_frobenius_std` read `Genome.brain` and would be
        // stale otherwise.
        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.gpu_full.as_ref() {
            for &(a, b) in matings {
                let brain_a = gpu.cells.download_brain_at(a);
                let brain_b = gpu.cells.download_brain_at(b);
                self.cells[a].genome.brain = brain_a;
                self.cells[b].genome.brain = brain_b;
            }
        }
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
        #[cfg(feature = "gpu")]
        let use_gpu_cppn = self.gpu_full.is_some();
        #[cfg(not(feature = "gpu"))]
        let use_gpu_cppn = false;
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
            cell_a.energy *= 0.5;
            cell_b.energy *= 0.5;
            cell_a.reproduce_cooldown_ticks = MATING_COOLDOWN_TICKS;
            cell_b.reproduce_cooldown_ticks = MATING_COOLDOWN_TICKS;
            let child = if use_gpu_cppn {
                bioscape::make_mating_child_no_brain(cell_a, cell_b, rng, child_ids[i])
            } else {
                bioscape::make_mating_child(cell_a, cell_b, rng, child_ids[i])
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
                        kind: bioscape::FoodKind::Carrion,
                    });
                }
            }
        }

        // Phase 2: remove dead cells. --gpu-full uses swap_remove pattern so
        // GPU brain_weights + xoshiro_state lze udržet in sync přes O(deaths)
        // GPU memcpy operations (žádný full re-upload).
        #[cfg(feature = "gpu")]
        let gpu_full_active = self.gpu_full.is_some();
        #[cfg(not(feature = "gpu"))]
        let gpu_full_active = false;

        if gpu_full_active {
            let before = self.cells.len();
            #[cfg(feature = "gpu")]
            let gpu = self.gpu_full.as_ref().unwrap();
            let mut i = 0;
            while i < self.cells.len() {
                if self.cells[i].energy <= 0.0 {
                    let last = self.cells.len() - 1;
                    #[cfg(feature = "gpu")]
                    if i != last {
                        gpu.cells.swap_to(i, last);
                    }
                    self.cells.swap_remove(i);
                    // i nezvyšuju — moved cell ze slotu last je teď ve slotu i.
                } else {
                    i += 1;
                }
            }
            self.deaths_gen += (before - self.cells.len()) as u64;
        } else {
            let before = self.cells.len();
            self.cells.retain(|c| c.energy > 0.0);
            self.deaths_gen += (before - self.cells.len()) as u64;
        }
        self.foods.extend(new_foods);
    }
}

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

// Spatial occupancy thresholds for clustering diagnostics. A cell is
// "near edge" if its position falls in the outer 10 % of either axis;
// "in corner" if both axes simultaneously meet that criterion.
pub const EDGE_FRAC_THRESHOLD: f32 = 0.9;

#[cfg(test)]
#[path = "world_tests.rs"]
mod tests;

