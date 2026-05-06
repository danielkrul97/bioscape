//! Headless harness — pure simulation loop, no Bevy renderer.
//!
//! Usage: `cargo run --release --bin headless -- [seed] [max_gens] [out_path]`
//! Defaults: seed=0, max_gens=500, out_path=run_seed{seed}.csv
//!
//! Logs per-generation stats (cell_count, mean/dev for max_speed/vision/body_size,
//! food count, density factor) to CSV. Reproducible: same seed → identical run.

use bioscape::{
    adhesion_velocity_delta, bond_velocity_delta, nearest_attackable_cell,
    reject_food_for_richness, Bond, Cell, EventCalendar, Food, Hunter, ShockScheduleConfig,
    SimClock, SmellField, SpatialGrid, WorldMap, ADHESION_RANGE_FACTOR, ATTACK_THRESHOLD,
    BOND_BREAK_THRESHOLD,
    BOND_FORMATION_COST, BOND_FORM_THRESHOLD, BOND_FORM_TICKS, BOND_MAINTENANCE_PER_SEC,
    BOND_REST_LENGTH_SLACK, BRAIN_RECURRENT, CARRION_FOOD_COUNT, CELL_RADIUS,
    COLLISION_RESTITUTION, CONTACT_DECAY_TICKS, CYCLE_AMPLITUDE, CYCLE_GEN_PERIOD,
    DILUTION_K, EAT_RADIUS, FIXED_TIMESTEP_HZ, FOOD_SPAWN_RATE,
    GENERATIONS_PER_EPOCH, GRID_CELL_SIZE, HAZARD_AMP, HAZARD_DRAIN_PER_SEC, HAZARD_FLOOR,
    HERD_RADIUS, HUNTER_GRID_CELL_SIZE, HUNTER_TARGET_COUNT,
    INITIAL_CELLS, LEARNING_RATE, MATING_COOLDOWN_TICKS, MATING_PHEROMONE_THRESHOLD,
    MATING_RADIUS, MAX_BONDS_PER_CELL, MAX_POPULATION, MAX_SPAWN_ATTEMPTS,
    PHEROMONE_BASELINE_EMIT, PHEROMONE_BRAIN_MOD, PHEROMONE_COST_PER_RATE, PHEROMONE_DECAY,
    PHEROMONE_DIFFUSION, PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z, PHEROMONE_SAMPLE_EPSILON,
    PHYSICS_CONFIG, PREDATION_DRAIN_PER_TICK, PREDATION_GAIN_PER_TICK, REPRODUCE_THRESHOLD,
    SIZE_RATIO_THRESHOLD, SMELL_DECAY, SMELL_DIFFUSION, SMELL_GRID_RES, SMELL_GRID_RES_Z,
    SMELL_PER_FOOD, SMELL_SAMPLE_EPSILON, TICKS_PER_GENERATION, WORLD_HALF,
    WORLD_MAP_BASE_RES, WORLD_MAP_BASE_RES_Z, WORLD_MAP_FOOD_AMP, WORLD_MAP_FOOD_FLOOR,
    WORLD_MAP_RES, WORLD_MAP_RES_Z, WORLD_MAP_SEED, WORLD_UNITS_PER_FOOD,
};
#[cfg(feature = "gpu")]
use bioscape::{
    gpu::{
        BrainGpu, BrownianGpu, CellsGpu, FieldGpu, GpuContext, HebbianGpu, MotorGpu,
        PopulateInputsGpu, PopulateInputsParams, SensorGatherGpu, SensorParamsGpu, SpatialHashGpu,
        StepGpu, StepParamsGpu,
    },
    AGE_DECAY_PER_SEC, ATTACK_COST_PER_SEC, BRAIN_HIDDEN, BRAIN_INPUTS, BRAIN_INPUTS_SENSORY,
    BRAIN_OUTPUTS, DAMAGE_NORMALIZATION_GAIN, DENSITY_NORM_COUNT, DRAG_COEFFICIENT,
    GRAVITY as PHYS_GRAVITY, PHEROMONE_NORMALIZATION_GAIN, SHELL_COST_PER_SEC,
    SMELL_NORMALIZATION_GAIN, SPIKE_COST_PER_SEC, THERMAL_NOISE,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::env;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

/// Sprint 48: versioned binary header pro checkpoint files.
/// Sprint 66 bump: Cell rozšířen o `cell_id` + `bonds`, BRAIN_OUTPUTS 9→10.
/// Sprint 68 bump: Genome rozšířen o `bond_stiffness` + `bond_damping`,
/// Bond rozšířen o `stiffness` + `damping`. Starší V1/V2 už nelze
/// deserializovat — load vrací error.
/// Sprint 103 bump: BRAIN_HIDDEN 32→50, BRAIN_INPUTS 53→71. Brain weight
/// arrays resize → V3 savefiles incompatible.
const CHECKPOINT_MAGIC: &[u8; 8] = b"BIOSCP01";
const CHECKPOINT_VERSION: u32 = 4;

/// Sprint 48: serializovatelný snapshot sim state. Skip fields:
/// - SpatialGrid (rebuild from cells/foods on load)
/// - bench_timings (per-tick diagnostic, ne state)
/// - GPU subsystémy (re-init on load)
/// - scratch Vecs (re-alloc lazily)
/// RNG state se NEUKLÁDÁ — load resetuje RNG ze --seed argument. Pro full
/// reproducibility add chacha state serializace v pozdějším sprintu.
#[derive(Serialize, Deserialize)]
struct Checkpoint {
    version: u32,
    cells: Vec<Cell>,
    foods: Vec<Food>,
    clock: SimClock,
    density_factor: f32,
    smell: SmellField,
    pheromone: SmellField,
    map: WorldMap,
    births_gen: u64,
    deaths_gen: u64,
    fertile_ticks_gen: u64,
    predation_events_gen: u64,
    mating_radius: f32,
    max_population: usize,
}

/// Sprint 43: per-fáze accumulator (mikrosekundy). World::tick zvyšuje každou
/// dobu a main je čte/resetuje per generation. Default je all-zero.
#[derive(Debug, Default, Clone, Copy)]
struct PhaseTimings {
    update_smell: f64,
    update_pheromone: f64,
    brain_act: f64,
    emit_pheromones: f64,
    apply_morph: f64,
    apply_brownian: f64,
    step: f64,
    apply_food_gravity: f64,
    apply_hazards: f64,
    resolve_collisions: f64,
    resolve_hunter_collisions: f64,
    predate: f64,
    hunt: f64,
    eat_food: f64,
    spawn_food: f64,
    reproduce: f64,
    die_and_drop_carrion: f64,
}

struct World {
    cells: Vec<Cell>,
    foods: Vec<Food>,
    clock: SimClock,
    density_factor: f32,
    smell: SmellField,
    pheromone: SmellField,
    map: WorldMap,
    // Sprint 43: spatial hashes pro broad-phase. Rebuild před fází, která
    // neighbors používá — `cell_grid` před brain_act/resolve_collisions/predate,
    // `food_grid` před brain_act/eat_food.
    cell_grid: SpatialGrid<usize, f32>,
    food_grid: SpatialGrid<usize, bioscape::FoodKind>,
    // Persistent scratch — reused per tick to avoid hot-loop allocations.
    deltas_scratch: Vec<[f32; 3]>,
    /// Sprint 65: collision velocity damping (inelastic) — per pair, closing
    /// velocity podél separation normal je halved. Cell i sees pair (i, j),
    /// computes own delta; symmetric (Newton 3rd law) když j visits i.
    velocity_deltas_scratch: Vec<[f32; 3]>,
    energy_deltas_scratch: Vec<f32>,
    damage_deltas_scratch: Vec<f32>,
    eaten_scratch: Vec<bool>,
    births_gen: u64,
    deaths_gen: u64,
    fertile_ticks_gen: u64,
    predation_events_gen: u64,
    /// Sprint 66: monotonic counter pro stable Cell.cell_id přidělování.
    /// Initial population uses 0..INITIAL_CELLS, takže start = INITIAL_CELLS.
    next_cell_id: u64,
    /// Sprint 66: per-pair (min_id, max_id) → consecutive contact ticks.
    /// Vstupy se přidávají v `resolve_collisions` Phase 2 (sequential merge),
    /// odebírají při decay timeout. Sparse — pouze dvojice s aktuálním kontaktem.
    contact_progress: rustc_hash::FxHashMap<(u64, u64), u32>,
    /// Sprint 66 diagnostic counter — počet bondů vytvořených v aktuální generaci.
    /// Zatím log-only, future může být CSV column.
    bonds_formed_gen: u64,
    /// Sprint 66 diagnostic — počet bondů přervaných v aktuální generaci.
    bonds_broken_gen: u64,
    /// Sprint 71: macropredator entities (Hunter). Sprint 89: + heritable
    /// genome + lifecycle (energy, reprodukce, smrt, floor respawn). Populace
    /// dynamic [1, HUNTER_MAX_POP]; initial = HUNTER_TARGET_COUNT.
    hunters: Vec<Hunter>,
    /// Sprint 71 diagnostic — počet hunter útoků v aktuální generaci.
    hunter_attacks_gen: u64,
    /// Sprint 89: monotonic counter pro nové hunter_id při reproduce + floor
    /// respawn. lineage_id = hunter_id pro nové lineage (floor respawn) nebo
    /// parent.lineage_id (reproduce continuation).
    next_hunter_id: u64,
    /// Sprint 89: hunter lifecycle metrics per generation.
    hunter_births_gen: u64,
    hunter_deaths_gen: u64,
    /// Sprint 99: hunter-hunter contact ticks (mirror cells `contact_progress`),
    /// + bond formation/breaking counters. Persistent across ticks; rebuild
    /// per `resolve_hunter_collisions` pass.
    hunter_contact_progress: rustc_hash::FxHashMap<(u64, u64), u32>,
    hunter_bonds_formed_gen: u64,
    hunter_bonds_broken_gen: u64,
    mating_radius: f32,
    // Sprint 43: runtime override `MAX_POPULATION` consts. Default = const, CLI
    // může nastavit výš (potřeba pro bench při N > 1000).
    max_population: usize,
    /// Sprint 109: deterministicky vygenerovaný kalendář environmentálních
    /// shocků. Default empty (no-op) — sim loop ho zatím nečte; integrace
    /// efektů přijde v Sprintu 110+.
    events: EventCalendar,
    // Sprint 87 Hamilton-rule sweep: runtime overrides pro food share. Default
    // = pre-sweep behavior (BOND_FOOD_SHARE_FRAC, no kin filter).
    share_frac: f32,
    kin_filter: bool,
    bench_timings: PhaseTimings,
    // Sprint 44: pokud `Some`, brain_act offloaduje forward pass na GPU.
    // Sensor gather + populate_brain_inputs + apply_brain_motor zůstává CPU.
    #[cfg(feature = "gpu")]
    gpu: Option<BrainGpu>,
    // Sprint 51: full-GPU brain pipeline. Když Some, drží brain weights
    // persistent na GPU mezi ticky (eliminuje 30 MB/tick upload Sprintu 44),
    // GPU Hebbian replace CPU brain.hebbian_update, GPU Brownian replace
    // CPU apply_brownian. Sensor/motor/step/collision/predate zůstávají CPU
    // rayon (Sprint 50 standalone shadery jsou ready, integrace je Sprint 52+).
    #[cfg(feature = "gpu")]
    gpu_full: Option<GpuFullState>,
}

#[cfg(feature = "gpu")]
struct GpuFullState {
    cells: CellsGpu,
    brain: BrainGpu,
    hebbian: HebbianGpu,
    brownian: BrownianGpu,
    /// Sprint 59: GPU smell + pheromone field (3D 7-point Jacobi).
    /// Sprint 60: po wire SensorGatherGpu už NEČTE CPU SmellField shadow —
    /// sensor shader bere field grid storage buffer direct. Per-tick readback
    /// eliminován; CPU `World.smell` / `pheromone` zůstávají jen pro
    /// checkpoint serialization (po `--gpu-full` jsou CPU shadows out-of-date).
    smell: FieldGpu,
    pheromone: FieldGpu,
    /// Sprint 60: GPU spatial hashes pro sensor broad-phase. Per-tick
    /// `dispatch()` (no readback) + sensor shader čte `offsets_buffer()` /
    /// `sorted_buffer()` přes binding group.
    cell_hash: SpatialHashGpu,
    food_hash: SpatialHashGpu,
    sensor: SensorGatherGpu,
    /// Sprint 61: GPU populate_brain_inputs shader fuze sensor output + cell
    /// metadata → brain inputs buffer. Eliminuje sensor 60 KB readback round-trip.
    populate: PopulateInputsGpu,
    /// Sprint 62: motor on GPU (apply brain outputs → velocity/ang_vel/pitch_vel).
    /// Fused s brownian dispatch v brain_act → single batch readback eliminuje
    /// round-trip #2 (hidden+outputs) i round-trip #3 (velocities).
    motor: MotorGpu,
    /// Sprint 63: step on GPU (kinematics + drag + energy + bounce).
    /// Fused do brain_act batch readback. Skip CPU `step` fáze v `--gpu-full`.
    step: StepGpu,
}

impl World {
    fn new(
        rng: &mut impl Rng,
        map_seed: u64,
        mating_radius: f32,
        initial_cells: usize,
        max_population: usize,
        events: EventCalendar,
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
        let cells = (0..initial_cells)
            .map(|i| Cell::random(rng, WORLD_HALF, i as u64, 0, i as u64))
            .collect();
        let target = food_target(1.0);
        let foods = (0..target)
            .map(|_| {
                for _ in 0..MAX_SPAWN_ATTEMPTS {
                    let candidate = Food::random(rng, WORLD_HALF);
                    let richness = map.sample([candidate.position[0], candidate.position[1], 0.0]);
                    if !reject_food_for_richness(rng, richness) {
                        return candidate;
                    }
                }
                Food::random(rng, WORLD_HALF)
            })
            .collect();
        Self {
            cells,
            foods,
            clock: SimClock::new(TICKS_PER_GENERATION, GENERATIONS_PER_EPOCH),
            density_factor: 1.0,
            smell: SmellField::new([SMELL_GRID_RES, SMELL_GRID_RES, SMELL_GRID_RES_Z], WORLD_HALF),
            pheromone: SmellField::new(
                [PHEROMONE_GRID_RES, PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z],
                WORLD_HALF,
            ),
            map,
            cell_grid: SpatialGrid::new(GRID_CELL_SIZE),
            food_grid: SpatialGrid::new(GRID_CELL_SIZE),
            deltas_scratch: Vec::new(),
            velocity_deltas_scratch: Vec::new(),
            energy_deltas_scratch: Vec::new(),
            damage_deltas_scratch: Vec::new(),
            eaten_scratch: Vec::new(),
            births_gen: 0,
            deaths_gen: 0,
            fertile_ticks_gen: 0,
            predation_events_gen: 0,
            next_cell_id: initial_cells as u64,
            contact_progress: rustc_hash::FxHashMap::default(),
            bonds_formed_gen: 0,
            bonds_broken_gen: 0,
            // Sprint 71: spawn HUNTER_TARGET_COUNT hunterů na náhodné pozice.
            // Sprint 89: každý hunter má random genome + lineage. lineage_id
            // = hunter_id (initial population je zakladatelská sada).
            hunters: (0..HUNTER_TARGET_COUNT)
                .map(|i| Hunter::random(rng, WORLD_HALF, i as u64, i as u64, 0))
                .collect(),
            hunter_attacks_gen: 0,
            // Sprint 89: hunter lifecycle counters.
            next_hunter_id: HUNTER_TARGET_COUNT as u64,
            hunter_births_gen: 0,
            hunter_deaths_gen: 0,
            // Sprint 99: hunter bond tracker + counters.
            hunter_contact_progress: rustc_hash::FxHashMap::default(),
            hunter_bonds_formed_gen: 0,
            hunter_bonds_broken_gen: 0,
            mating_radius,
            max_population,
            events,
            share_frac: bioscape::BOND_FOOD_SHARE_FRAC,
            kin_filter: false,
            bench_timings: PhaseTimings::default(),
            #[cfg(feature = "gpu")]
            gpu: None,
            #[cfg(feature = "gpu")]
            gpu_full: None,
        }
    }

    /// Sprint 48: snapshot sim state do versioned binary blob. Format:
    /// `MAGIC[8] | bincode(Checkpoint)`. Idempotent — caller může volat
    /// průběžně i na konci.
    fn save_checkpoint(&self, path: &Path) -> std::io::Result<()> {
        let chk = Checkpoint {
            version: CHECKPOINT_VERSION,
            cells: self.cells.clone(),
            foods: self.foods.clone(),
            clock: self.clock,
            density_factor: self.density_factor,
            smell: self.smell.clone(),
            pheromone: self.pheromone.clone(),
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
    fn load_checkpoint(path: &Path) -> std::io::Result<Self> {
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
        Ok(Self {
            cells: chk.cells,
            foods: chk.foods,
            clock: chk.clock,
            density_factor: chk.density_factor,
            smell: chk.smell,
            pheromone: chk.pheromone,
            map: chk.map,
            cell_grid: SpatialGrid::new(GRID_CELL_SIZE),
            food_grid: SpatialGrid::new(GRID_CELL_SIZE),
            deltas_scratch: Vec::new(),
            velocity_deltas_scratch: Vec::new(),
            energy_deltas_scratch: Vec::new(),
            damage_deltas_scratch: Vec::new(),
            eaten_scratch: Vec::new(),
            births_gen: chk.births_gen,
            deaths_gen: chk.deaths_gen,
            fertile_ticks_gen: chk.fertile_ticks_gen,
            predation_events_gen: chk.predation_events_gen,
            next_cell_id,
            contact_progress: rustc_hash::FxHashMap::default(),
            bonds_formed_gen: 0,
            bonds_broken_gen: 0,
            // Sprint 71: hunters nejsou v checkpointu — re-spawnou se fresh.
            // Sprint 89: po refactor hunters mají genome — checkpoint by je
            // měl serializovat, ale aktuální format nepodporuje. Fresh respawn
            // s random genome (lineage reset).
            hunters: {
                let mut rng = StdRng::seed_from_u64(chk.mating_radius as u64);
                (0..HUNTER_TARGET_COUNT)
                    .map(|i| Hunter::random(&mut rng, WORLD_HALF, i as u64, i as u64, 0))
                    .collect()
            },
            hunter_attacks_gen: 0,
            next_hunter_id: HUNTER_TARGET_COUNT as u64,
            hunter_births_gen: 0,
            hunter_deaths_gen: 0,
            hunter_contact_progress: rustc_hash::FxHashMap::default(),
            hunter_bonds_formed_gen: 0,
            hunter_bonds_broken_gen: 0,
            mating_radius: chk.mating_radius,
            max_population: chk.max_population,
            // Sprint 109: kalendář není v checkpointu (per-tick state je
            // deterministicky odvozen z World seed). Resume CLI musí znovu
            // předat `--shocks-mean-gens`; jinak fresh empty.
            events: EventCalendar::default(),
            share_frac: bioscape::BOND_FOOD_SHARE_FRAC,
            kin_filter: false,
            bench_timings: PhaseTimings::default(),
            #[cfg(feature = "gpu")]
            gpu: None,
            #[cfg(feature = "gpu")]
            gpu_full: None,
        })
    }

    fn tick(&mut self, rng: &mut impl Rng) -> Option<u64> {
        let dt = 1.0 / FIXED_TIMESTEP_HZ;
        let transitions = self.clock.advance();
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
        // Sprint 94: pool last_hidden across bond network → cluster cells share
        // recurrent state. Must run before brain_act (which reads pooled_hidden).
        self.pool_bonded_hidden();
        timed!(brain_act, self.run_brain_act(dt));
        timed!(emit_pheromones, self.emit_pheromones(dt));
        timed!(apply_morph, self.apply_morph(dt));
        timed!(apply_brownian, self.apply_brownian(rng, dt));
        timed!(step, self.step(dt));
        timed!(apply_food_gravity, self.apply_food_gravity(dt));
        timed!(apply_hazards, self.apply_hazards(dt));
        timed!(resolve_collisions, self.resolve_collisions());
        timed!(resolve_hunter_collisions, self.resolve_hunter_collisions());
        // Sprint 100: pool last_hidden napříč hunter packem před hunt fází —
        // tak `populate_hunter_brain_inputs` čte pooled state.
        bioscape::pool_bonded_hunter_hidden(&mut self.hunters);
        timed!(predate, self.predate());
        timed!(hunt, self.hunt(rng, dt));
        // Sprint 89: hunter death + reproduce + floor respawn po hunt phase.
        self.hunter_lifecycle(rng);
        timed!(eat_food, self.eat_food());
        timed!(spawn_food, self.spawn_food(rng));
        timed!(reproduce, self.reproduce(rng));
        timed!(die_and_drop_carrion, self.die_and_drop_carrion(rng));

        transitions.generation_ended
    }

    fn apply_morph(&mut self, dt: f32) {
        // Sprint 57: zkoušel jsem rayon par_iter_mut, ale ~2 us sekvenčně vs
        // ~26 us paralelně — rayon spawn overhead převáží práci. Sekvenční win.
        for cell in &mut self.cells {
            cell.apply_morph(dt);
        }
    }

    fn apply_brownian(&mut self, rng: &mut impl Rng, dt: f32) {
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
        let _ = rng; // touch to satisfy unused warning under gpu cfg path
        let _ = self.apply_brownian_cpu(rng, dt);
    }

    fn apply_brownian_cpu(&mut self, rng: &mut impl Rng, dt: f32) {
        for cell in &mut self.cells {
            cell.apply_brownian(rng, dt, WORLD_HALF[2]);
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
        let gpu = self.gpu_full.as_ref().unwrap();
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
        self.smell.step(SMELL_DIFFUSION, SMELL_DECAY, dt);
    }

    fn update_pheromone(&mut self, dt: f32) {
        // Diffuse + decay BEFORE this tick's emissions are added (in
        // emit_pheromones, called after brain_act). Cells thus read the
        // gradient ze stavu pole na předchozí tick — prevents instant
        // self-feedback (cell vidí svůj vlastní právě emitovaný puff).
        // Sprint 60: GPU step bez readback.
        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.gpu_full.as_mut() {
            gpu.pheromone.step(PHEROMONE_DIFFUSION, PHEROMONE_DECAY, dt);
            return;
        }
        self.pheromone.step(PHEROMONE_DIFFUSION, PHEROMONE_DECAY, dt);
    }

    fn emit_pheromones(&mut self, dt: f32) {
        // Sprint 59: pokud --gpu-full, deposit do GPU pending_sources (flushed
        // at next tick's update_pheromone step()). CPU pheromone.grid není
        // updated — sensor gather už proběhl s pre-emission state (Sprint 38
        // semantika "no self-feedback within tick").
        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.gpu_full.as_mut() {
            for cell in &mut self.cells {
                let mod_strength = cell.last_outputs[2].max(0.0);
                let brain_emit = PHEROMONE_BRAIN_MOD * mod_strength;
                let rate = PHEROMONE_BASELINE_EMIT + brain_emit;
                gpu.pheromone.add_source(
                    [cell.position[0], cell.position[1], cell.position[2]],
                    rate * dt,
                );
                cell.energy -= PHEROMONE_COST_PER_RATE * brain_emit * dt;
            }
            return;
        }
        for cell in &mut self.cells {
            let mod_strength = cell.last_outputs[2].max(0.0);
            let brain_emit = PHEROMONE_BRAIN_MOD * mod_strength;
            let rate = PHEROMONE_BASELINE_EMIT + brain_emit;
            self.pheromone
                .add_source([cell.position[0], cell.position[1], cell.position[2]], rate * dt);
            cell.energy -= PHEROMONE_COST_PER_RATE * brain_emit * dt;
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
        }
        let positions: Vec<[f32; 3]> = self.cells.iter().map(|c| c.position).collect();
        let eff_radii: Vec<f32> = self
            .cells
            .iter()
            .map(|c| c.phenotype.effective_radius())
            .collect();
        let vision_radii: Vec<f32> = self.cells.iter().map(|c| c.genome.vision_radius).collect();
        let food_positions: Vec<[f32; 3]> = self.foods.iter().map(|f| f.position).collect();

        // Per-cell metadata pro populate_inputs + motor + step shaders.
        let energies: Vec<f32> = self.cells.iter().map(|c| c.energy).collect();
        let headings: Vec<f32> = self.cells.iter().map(|c| c.heading).collect();
        let pitches: Vec<f32> = self.cells.iter().map(|c| c.pitch).collect();
        let damage_accums: Vec<f32> = self.cells.iter().map(|c| c.damage_accum).collect();
        let max_speeds: Vec<f32> = self.cells.iter().map(|c| c.genome.max_speed).collect();
        let velocities: Vec<[f32; 3]> = self.cells.iter().map(|c| c.velocity).collect();
        // Sprint 62: motor čte angular_velocity + pitch_velocity. Upload current
        // state — motor shader mutuje a download_brain_motor_batch sync zpět.
        let angular_vels: Vec<f32> = self.cells.iter().map(|c| c.angular_velocity).collect();
        let pitch_vels: Vec<f32> = self.cells.iter().map(|c| c.pitch_velocity).collect();
        // Sprint 62: turn_rate per-tick upload (lazy approach) — reproduce
        // mění turn_rate u nových childů, takže pre-tick refresh je safer
        // než per-event sparse update. ~4 KB/tick negligible.
        let turn_rates: Vec<f32> = self.cells.iter().map(|c| c.genome.turn_rate).collect();
        // Sprint 63: step shader needs position/age/cooldown/body_dims/aux.
        // Apply_morph CPU has run earlier this tick → phenotype bytes current.
        let positions_for_step: Vec<[f32; 3]> = self.cells.iter().map(|c| c.position).collect();
        let ages: Vec<u32> = self.cells.iter().map(|c| c.age as u32).collect();
        let cooldowns: Vec<u32> = self
            .cells
            .iter()
            .map(|c| c.reproduce_cooldown_ticks)
            .collect();
        let body_dims: Vec<[f32; 3]> = self
            .cells
            .iter()
            .map(|c| {
                [
                    c.phenotype.body_length,
                    c.phenotype.body_width,
                    c.phenotype.body_height,
                ]
            })
            .collect();
        // Sprint 63: aux = [spike_length, shell_thickness, vision_radius, attack_strength].
        // Attack je per-tick z `cell.last_outputs[6].max(0.0)` — populated po
        // brain forward (Sprint 62 phase 6). Tady CPU snapshot dává přístup
        // k *předchozí* tick last_outputs (1-tick delay shell_absorb-like).
        // To matchne lib `Cell::apply_energy_costs` semantiku (čte předchozí
        // post-brain attack signal).
        let aux: Vec<[f32; 4]> = self
            .cells
            .iter()
            .map(|c| {
                [
                    c.phenotype.primary_spike_length(),
                    c.phenotype.shell_thickness,
                    c.genome.vision_radius,
                    c.last_outputs[6].max(0.0),
                ]
            })
            .collect();

        let gpu = self.gpu_full.as_mut().expect("gpu_full Some");

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
        gpu.cells.upload_turn_rates(&turn_rates);
        // Sprint 63: step shader uploads.
        gpu.cells.upload_positions(&positions_for_step);
        gpu.cells.upload_age_cooldown(&ages, &cooldowns);
        gpu.cells.upload_body_dims(&body_dims);
        gpu.cells.upload_aux(&aux);

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
            _pad0: 0,
        };
        gpu.sensor.dispatch_no_readback(
            &positions,
            &eff_radii,
            &vision_radii,
            &food_positions,
            &gpu.cell_hash,
            &gpu.food_hash,
            &gpu.smell,
            &gpu.pheromone,
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
            _pad0: 0,
            _pad1: 0,
        };
        gpu.populate
            .dispatch(&gpu.cells, &gpu.sensor, populate_params);

        // Phase 6: GPU brain forward_persistent. Čte `last_inputs_buf` direct,
        // píše last_hidden + last_outputs storage buffers.
        gpu.brain.forward_persistent(&gpu.cells, n);

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
        };
        gpu.step.dispatch_with_cells(&gpu.cells, n, step_params);

        // Phase 10: single batch readback (Sprint 63: 9 buffers fused do
        // jednoho Wait barrier). Hidden + outputs pro Hebbian/predate/emit;
        // velocity + ang_vel + pitch_vel + position + age + cooldown + energy
        // pro CPU phases (eat_food, predate, collisions, reproduce, hazards).
        let (
            hiddens,
            outputs,
            new_vels,
            new_ang,
            new_pitch,
            new_pos,
            new_ages,
            new_cooldowns,
            new_energies,
        ) = gpu.cells.download_full_batch(n);

        // Phase 11: CPU writeback. NO apply_brain_motor + NO Cell::step CPU
        // (oba byly GPU-side). damage_accum reset (mirror populate_inputs).
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
        let pheromone = &self.pheromone;
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
                let pheromone_grad = pheromone.gradient_at(pos_xyz, PHEROMONE_SAMPLE_EPSILON);
                let temperature_local =
                    bioscape::temperature_at_z(pos[2], WORLD_HALF, tick, gen);
                let sensors = bioscape::BrainSensors {
                    nearest_food: best_food,
                    nearest_cell: best_cell,
                    neighbors_in_vision,
                    smell_grad,
                    pheromone_grad,
                    temperature_local,
                };
                cell.apply_shell_absorb(dt);
                let mut inputs = bioscape::populate_brain_inputs(cell, &sensors, vision_r);
                bioscape::apply_sensor_gains(&mut inputs, &cell.genome.sensor_gains);
                inputs
            })
            .collect();

        // Sprint 97: Phase 1b: pool max-magnitude přes bond network. Provede se
        // PŘED GPU uploadem aby brain forward už dostal pooled inputs.
        let id_to_idx: rustc_hash::FxHashMap<u64, usize> = self
            .cells
            .iter()
            .enumerate()
            .map(|(i, c)| (c.cell_id, i))
            .collect();
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
        // Build cell_id → idx map once per tick.
        let id_to_idx: rustc_hash::FxHashMap<u64, usize> = self
            .cells
            .iter()
            .enumerate()
            .map(|(i, c)| (c.cell_id, i))
            .collect();
        // Snapshot last_hidden array — read-only během compute, write-only
        // do pooled_hidden, no aliasing issues.
        let snapshot: Vec<[f32; BRAIN_HIDDEN]> =
            self.cells.iter().map(|c| c.last_hidden).collect();
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
        let pheromone = &self.pheromone;
        let tick = self.clock.tick;
        let gen = self.clock.generation;

        // Sprint 97: dvojfáze pro cluster sensor pooling. Phase 1: gather + apply
        // own gains. Phase 2: pool max-magnitude přes bond network + brain forward.
        let inputs_vec: Vec<[f32; BRAIN_INPUTS]> = self
            .cells
            .par_iter_mut()
            .enumerate()
            .map(|(i, cell)| {
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
                    best_food_d2 = d2;
                    best_food = Some(d);
                });

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
                let pheromone_grad = pheromone.gradient_at(pos_xyz, PHEROMONE_SAMPLE_EPSILON);
                let temperature_local =
                    bioscape::temperature_at_z(pos[2], WORLD_HALF, tick, gen);
                let sensors = bioscape::BrainSensors {
                    nearest_food: best_food,
                    nearest_cell: best_cell,
                    neighbors_in_vision,
                    smell_grad,
                    pheromone_grad,
                    temperature_local,
                };

                cell.apply_shell_absorb(dt);
                let mut inputs = bioscape::populate_brain_inputs(cell, &sensors, vision_r);
                bioscape::apply_sensor_gains(&mut inputs, &cell.genome.sensor_gains);
                inputs
            })
            .collect();

        let id_to_idx: rustc_hash::FxHashMap<u64, usize> = self
            .cells
            .iter()
            .enumerate()
            .map(|(i, c)| (c.cell_id, i))
            .collect();

        self.cells
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, cell)| {
                let own = inputs_vec[i];
                let pooled = bioscape::pool_bonded_sensors(cell, &own, |partner_id| {
                    let idx = id_to_idx.get(&partner_id).copied()?;
                    if idx == i {
                        return None;
                    }
                    Some(inputs_vec[idx])
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
        for cell in &mut self.cells {
            let climate_offset = bioscape::climate_shock_offset(
                events,
                gen,
                [cell.position[0], cell.position[1]],
                WORLD_HALF,
            );
            cell.step_with_climate(dt, WORLD_HALF, tick, gen, &PHYSICS_CONFIG, climate_offset);
        }
    }

    /// Sprint 112: per-cell climate offset helper, sdílený mezi tick hot path
    /// a `write_stats` (CSV column `shock_climate_offset`).
    fn climate_offset_at(&self, pos_xy: [f32; 2]) -> f32 {
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
        // Sprint 66: cell_id → idx map pro O(1) bond lookup.
        let id_to_idx: rustc_hash::FxHashMap<u64, usize> = self
            .cells
            .iter()
            .enumerate()
            .map(|(i, c)| (c.cell_id, i))
            .collect();
        self.deltas_scratch.clear();
        self.deltas_scratch.resize(n, [0.0, 0.0, 0.0]);
        self.velocity_deltas_scratch.clear();
        self.velocity_deltas_scratch.resize(n, [0.0, 0.0, 0.0]);

        let cell_grid = &self.cell_grid;
        let cells = &self.cells;
        // Sprint 66: search radius = max(collision, adhesion). Adhesion má
        // dosah pair_r × ADHESION_RANGE_FACTOR; pair_r = CELL_RADIUS × max_axis × 2.
        // Pro jistotu používáme effective_radius_i × CELL_RADIUS × 2 × FACTOR.
        // Per-i collected contact pairs: (other_cell_id, currently_in_contact).
        // Phase 2 sequentially merges do contact_progress.
        let contact_lists: Vec<Vec<u64>> = (0..n)
            .into_par_iter()
            .zip(self.deltas_scratch.par_iter_mut())
            .zip(self.velocity_deltas_scratch.par_iter_mut())
            .map(|((i, delta), vel_delta)| {
                let pos_i = cells[i].position;
                let vel_i = cells[i].velocity;
                let radius_i = cells[i].phenotype.effective_radius();
                let type_i = cells[i].genome.adhesion_type;
                let collision_r = CELL_RADIUS * (radius_i + cells[i].phenotype.max_axis() * 2.0);
                let adhesion_r =
                    CELL_RADIUS * (radius_i + cells[i].phenotype.max_axis() * 2.0)
                        * ADHESION_RANGE_FACTOR;
                let search_r = collision_r.max(adhesion_r);
                let mut local_contacts: Vec<u64> = Vec::new();
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
                local_contacts
            })
            .collect();

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
        let mut bonds_broken_this_tick: u64 = 0;
        let positions_snapshot: Vec<[f32; 3]> =
            self.cells.iter().map(|c| c.position).collect();
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
        let mut seen_pairs: rustc_hash::FxHashSet<(u64, u64)> =
            rustc_hash::FxHashSet::default();
        // Accumulate per-pair "i side bond_active" flag — both sides must be true.
        // Cell i shipped (other_id, bond_active_i); for cell j we look it up
        // separately by checking cells[id_to_idx[j]].last_outputs[9].
        for (i, contacts) in contact_lists.iter().enumerate() {
            let cell_id_i = self.cells[i].cell_id;
            for &other_id in contacts {
                let key = (cell_id_i, other_id);
                seen_pairs.insert(key);
                let entry = self.contact_progress.entry(key).or_insert(0);
                *entry = entry.saturating_add(1);
            }
        }
        // Decay / prune unseen pairs. Použijeme retain s decrementing.
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
        let candidates: Vec<(u64, u64)> = self
            .contact_progress
            .iter()
            .filter_map(|(&(a, b), &ticks)| {
                if ticks >= BOND_FORM_TICKS {
                    Some((a, b))
                } else {
                    None
                }
            })
            .collect();
        for (id_a, id_b) in candidates {
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
            let already = self.cells[i_a]
                .bonds
                .iter()
                .any(|b| b.map(|bb| bb.other_cell_id == id_b).unwrap_or(false));
            if already {
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
            // Reset progress entry — nepokouší se znova ihned formovat.
            self.contact_progress.remove(&(id_a, id_b));
        }
        self.bonds_formed_gen += bonds_formed_this_tick;
        self.bonds_broken_gen += bonds_broken_this_tick;
    }

    fn predate(&mut self) {
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
                            gain += cells[i].spike_bonus_against(pos_j);
                            let dilution = 1.0 / (1.0 + DILUTION_K * herd_counts[j] as f32);
                            // Sprint 69: bonded prey takes less damage + yields
                            // less energy. Group-defense benefit činí bondování
                            // evolučně positive (Sprint 67.1 ukázal opak bez něj).
                            let defense = bioscape::bond_defense_factor(cells[j].n_bonds());
                            gain *= dilution * defense;
                            local.push((i, j, gain, defense));
                        }
                    },
                );
                local
            })
            .collect();

        let events: u64 = attack_events.len() as u64;
        for (i, j, gain, defense) in attack_events {
            self.energy_deltas_scratch[i] += gain;
            // Sprint 69: defense škáluje i drain + damage (consistent s gain).
            let drain = PREDATION_DRAIN_PER_TICK * defense;
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

    /// Sprint 71: macropredator phase. Per Hunter: najdi nejbližší attackable
    /// cell (vision range, n_bonds < threshold), pohni se k němu, pokud je
    /// v attack range → action damage. Pokud nikdo není attackable (clustery
    /// dominují, nebo cells utekly), random drift.
    ///
    /// Sprint 89: hunters mají genome + lifecycle. Per-tick:
    ///   1. Find target (genome.vision_radius + genome.vision_fov).
    ///   2. step (movement + age tick).
    ///   3. Apply attack pokud target v attack_radius (genome).
    ///   4. apply_energy_costs (vision + motion + body + attack upkeep).
    ///   5. Energy gain ∝ damage dealt (ENERGY_PER_DAMAGE).
    ///
    /// Sprint 99: hunter-hunter physics — collision depenetration + adhesion
    /// (same-type attractive, cross-type weak repulse) + spring bondy. Mirror
    /// cell `resolve_collisions` strukturně, ale O(N²) pro N ≤ 50 hunterů
    /// (žádný spatial grid, sequential, kompaktní). Bond formation gated jen
    /// na contact ≥ BOND_FORM_TICKS + same adhesion_type + free slot —
    /// brain output[9] gate odložen na S100.
    fn resolve_hunter_collisions(&mut self) {
        let n = self.hunters.len();
        if n < 2 {
            return;
        }
        let hunter_radius = |h: &Hunter| h.genome.body_size * CELL_RADIUS;
        let id_to_idx: rustc_hash::FxHashMap<u64, usize> = self
            .hunters
            .iter()
            .enumerate()
            .map(|(i, h)| (h.hunter_id, i))
            .collect();

        let mut pos_deltas: Vec<[f32; 3]> = vec![[0.0; 3]; n];
        let mut vel_deltas: Vec<[f32; 3]> = vec![[0.0; 3]; n];
        let mut in_contact_pairs: rustc_hash::FxHashSet<(u64, u64)> =
            rustc_hash::FxHashSet::default();

        // Phase 1: per-pair forces (pos depenetrace + adhesion + bondy).
        for i in 0..n {
            let pos_i = self.hunters[i].position;
            let vel_i = self.hunters[i].velocity;
            let radius_i = hunter_radius(&self.hunters[i]);
            let type_i = self.hunters[i].genome.adhesion_type;
            let id_i = self.hunters[i].hunter_id;

            for j in 0..n {
                if i == j {
                    continue;
                }
                let pos_j = self.hunters[j].position;
                let radius_j = hunter_radius(&self.hunters[j]);
                let pair_r = radius_i + radius_j;
                let d_vec = bioscape::min_image_delta(pos_j, pos_i, WORLD_HALF);
                let d2 = d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2];
                let d = d2.sqrt();
                let in_contact = d2 < pair_r * pair_r && d2 > 0.0;
                if in_contact {
                    let overlap = pair_r - d;
                    let nx = d_vec[0] / d;
                    let ny = d_vec[1] / d;
                    let nz = d_vec[2] / d;
                    pos_deltas[i][0] -= nx * overlap * 0.5;
                    pos_deltas[i][1] -= ny * overlap * 0.5;
                    pos_deltas[i][2] -= nz * overlap * 0.5;
                    let id_j = self.hunters[j].hunter_id;
                    let pair = if id_i < id_j { (id_i, id_j) } else { (id_j, id_i) };
                    in_contact_pairs.insert(pair);
                } else if d > 0.0 {
                    let type_j = self.hunters[j].genome.adhesion_type;
                    let same_type = type_i == type_j;
                    let dv = bioscape::adhesion_velocity_delta(d_vec, d, pair_r, same_type);
                    vel_deltas[i][0] += dv[0];
                    vel_deltas[i][1] += dv[1];
                    vel_deltas[i][2] += dv[2];
                }
            }

            // Apply own bond spring forces.
            for bond_opt in self.hunters[i].bonds.iter() {
                if let Some(bond) = bond_opt {
                    if let Some(&j_idx) = id_to_idx.get(&bond.other_cell_id) {
                        let pos_j = self.hunters[j_idx].position;
                        let vel_j = self.hunters[j_idx].velocity;
                        let d_vec = bioscape::min_image_delta(pos_j, pos_i, WORLD_HALF);
                        let dist = (d_vec[0] * d_vec[0]
                            + d_vec[1] * d_vec[1]
                            + d_vec[2] * d_vec[2])
                            .sqrt();
                        let (dv, _broken) =
                            bioscape::bond_velocity_delta(bond, d_vec, dist, vel_i, vel_j);
                        vel_deltas[i][0] += dv[0];
                        vel_deltas[i][1] += dv[1];
                        vel_deltas[i][2] += dv[2];
                    }
                }
            }
        }

        // Phase 2: apply position + velocity deltas.
        for ((h, pd), vd) in self
            .hunters
            .iter_mut()
            .zip(pos_deltas.iter())
            .zip(vel_deltas.iter())
        {
            h.position[0] += pd[0];
            h.position[1] += pd[1];
            h.position[2] += pd[2];
            h.velocity[0] += vd[0];
            h.velocity[1] += vd[1];
            h.velocity[2] += vd[2];
        }

        // Phase 3: contact tracker update (increment for active pairs, decay
        // for stale; drop pairs that decayed k 0).
        let mut new_progress: rustc_hash::FxHashMap<(u64, u64), u32> =
            rustc_hash::FxHashMap::default();
        for &pair in &in_contact_pairs {
            let prev = self.hunter_contact_progress.get(&pair).copied().unwrap_or(0);
            new_progress.insert(pair, prev.saturating_add(1));
        }
        for (&pair, &val) in self.hunter_contact_progress.iter() {
            if !in_contact_pairs.contains(&pair) && val > 1 {
                new_progress.insert(pair, val - 1);
            }
        }
        self.hunter_contact_progress = new_progress;

        // Phase 4: bond formation. Gating: contact ≥ BOND_FORM_TICKS, same
        // adhesion_type, neither already bonded to the other, oba mají free slot.
        let candidates: Vec<(u64, u64)> = self
            .hunter_contact_progress
            .iter()
            .filter(|(_, &t)| t >= BOND_FORM_TICKS)
            .map(|(&pair, _)| pair)
            .collect();
        for (id_a, id_b) in candidates {
            let (Some(&a_idx), Some(&b_idx)) = (id_to_idx.get(&id_a), id_to_idx.get(&id_b))
            else {
                continue;
            };
            if self.hunters[a_idx].genome.adhesion_type
                != self.hunters[b_idx].genome.adhesion_type
            {
                continue;
            }
            // Sprint 100: brain output[9] gate — oba hunteři musí mít
            // bond_signal > BOND_FORM_THRESHOLD. Default INNATE_BOND_BIAS=2.5
            // dává tanh(2.5) ≈ 0.99 → většina random brainů gate překročí.
            if self.hunters[a_idx].last_outputs[9] < bioscape::BOND_FORM_THRESHOLD
                || self.hunters[b_idx].last_outputs[9] < bioscape::BOND_FORM_THRESHOLD
            {
                continue;
            }
            let already = self.hunters[a_idx]
                .bonds
                .iter()
                .any(|b| b.as_ref().map_or(false, |bb| bb.other_cell_id == id_b));
            if already {
                continue;
            }
            let slot_a = self.hunters[a_idx].bonds.iter().position(|b| b.is_none());
            let slot_b = self.hunters[b_idx].bonds.iter().position(|b| b.is_none());
            if let (Some(sa), Some(sb)) = (slot_a, slot_b) {
                let pos_a = self.hunters[a_idx].position;
                let pos_b = self.hunters[b_idx].position;
                let d_vec = bioscape::min_image_delta(pos_b, pos_a, WORLD_HALF);
                let dist =
                    (d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2]).sqrt();
                let rest = dist * BOND_REST_LENGTH_SLACK;
                let bond_a = Bond {
                    other_cell_id: id_b,
                    rest_length: rest,
                    stiffness: bioscape::BOND_STIFFNESS,
                    damping: bioscape::BOND_DAMPING,
                    age_ticks: 0,
                };
                let bond_b = Bond {
                    other_cell_id: id_a,
                    rest_length: rest,
                    stiffness: bioscape::BOND_STIFFNESS,
                    damping: bioscape::BOND_DAMPING,
                    age_ticks: 0,
                };
                self.hunters[a_idx].bonds[sa] = Some(bond_a);
                self.hunters[b_idx].bonds[sb] = Some(bond_b);
                self.hunter_bonds_formed_gen += 1;
            }
        }

        // Phase 5: bond pruning — drop dangling (target dead), increment age.
        let mut broken = 0u64;
        for hunter in self.hunters.iter_mut() {
            for bond_opt in hunter.bonds.iter_mut() {
                if let Some(bond) = bond_opt {
                    if !id_to_idx.contains_key(&bond.other_cell_id) {
                        *bond_opt = None;
                        broken += 1;
                    } else {
                        bond.age_ticks = bond.age_ticks.saturating_add(1);
                    }
                }
            }
        }
        self.hunter_bonds_broken_gen += broken;
    }

    /// Two-pass kvůli borrow checkeru: pass 1 sbírá (cell_idx, damage) do
    /// scratch Vec během iterace `&mut self.hunters`, pass 2 apply mutace na
    /// `self.cells` po uvolnění hunter borrow.
    fn hunt(&mut self, rng: &mut impl Rng, dt: f32) {
        let _ = rng; // Sprint 90: brain replaces idle drift, no rng needed in hunt.
        let cells_ref = &self.cells;
        let smell = &self.smell;
        // Sprint 102: cell spatial grid (1× per tick) + minimální hunter
        // snapshot (id, pos, adhesion_type) pro pack sensing. Nahrazuje
        // O(H·N) brute force scan + per-tick deep clone celého `self.hunters`.
        let mut hunter_cell_grid: SpatialGrid<usize, ()> =
            SpatialGrid::new(HUNTER_GRID_CELL_SIZE);
        hunter_cell_grid.rebuild(
            self.cells
                .iter()
                .enumerate()
                .map(|(i, c)| (i, c.position, ())),
        );
        let hunters_snapshot: Vec<bioscape::HunterSnapshotMin> = self
            .hunters
            .iter()
            .map(bioscape::HunterSnapshotMin::from_hunter)
            .collect();
        let mut attacks: Vec<(usize, f32)> = Vec::new();
        // Sprint 101: pack kill share. Při damage_dealt > 0 collected (partner_id,
        // share_amount) — apply na partnery až po mut iter loop.
        let mut pack_shares: Vec<(u64, f32)> = Vec::new();
        for hunter in &mut self.hunters {
            // Sprint 90: sensor gather + brain forward + hybrid motor (seek+brain) +
            // step (kinematic). Replaces Sprint 89 seek-based step.
            let sensors = bioscape::gather_hunter_sensors(
                hunter,
                cells_ref,
                &hunter_cell_grid,
                &hunters_snapshot,
                smell,
                WORLD_HALF,
            );
            let target_idx_pre =
                nearest_attackable_cell(hunter, cells_ref, &hunter_cell_grid, WORLD_HALF);
            let seek_target = target_idx_pre.map(|i| cells_ref[i].position);
            let inputs = bioscape::populate_hunter_brain_inputs(hunter, &sensors);
            let (hidden, outputs) = hunter.genome.brain.forward_with_state(&inputs);
            hunter.last_inputs = inputs;
            hunter.last_hidden = hidden;
            hunter.last_outputs = outputs;
            hunter.apply_brain_motor(&outputs, seek_target, dt, WORLD_HALF);
            hunter.step(dt, WORLD_HALF);
            // Attack check (post-step pozice).
            let target_idx =
                nearest_attackable_cell(hunter, cells_ref, &hunter_cell_grid, WORLD_HALF);
            let attack_r = hunter.genome.attack_radius;
            let attack_r2 = attack_r * attack_r;
            let damage = hunter.genome.damage_per_tick;
            let mut gain = 0.0_f32;
            if let Some(i) = target_idx {
                let d = bioscape::min_image_delta(
                    hunter.position,
                    cells_ref[i].position,
                    WORLD_HALF,
                );
                let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if d2 < attack_r2 {
                    // Sprint 92: edge-vulnerability — damage scales s exposure
                    // (= 1 - n_bonds × EXPOSURE_PER_BOND).
                    let exposure = bioscape::cell_exposure(cells_ref[i].n_bonds());
                    let damage_dealt = damage * exposure * dt;
                    attacks.push((i, damage_dealt));
                    gain = damage_dealt * bioscape::HUNTER_ENERGY_PER_DAMAGE;
                    // Sprint 101: queue share pro každého bonded partnera.
                    for bond_opt in hunter.bonds.iter() {
                        if let Some(bond) = bond_opt {
                            pack_shares.push((
                                bond.other_cell_id,
                                gain * bioscape::HUNTER_BOND_KILL_SHARE_FRAC,
                            ));
                        }
                    }
                }
            }
            hunter.apply_energy_costs(dt);
            hunter.energy += gain;
        }
        // Sprint 101: distribute pack shares post-loop.
        if !pack_shares.is_empty() {
            let id_to_idx: rustc_hash::FxHashMap<u64, usize> = self
                .hunters
                .iter()
                .enumerate()
                .map(|(i, h)| (h.hunter_id, i))
                .collect();
            for (id, energy) in pack_shares {
                if let Some(&i) = id_to_idx.get(&id) {
                    self.hunters[i].energy += energy;
                }
            }
        }
        self.hunter_attacks_gen += attacks.len() as u64;
        for (i, damage) in attacks {
            let cell = &mut self.cells[i];
            cell.energy -= damage;
            cell.damage_accum += damage;
        }
    }

    /// Sprint 89: hunter lifecycle — death + reproduce + floor respawn +
    /// MAX_POP cap. Volá se po `hunt()` v step loop, takže hunters mají
    /// up-to-date energy/cooldown.
    fn hunter_lifecycle(&mut self, rng: &mut impl Rng) {
        let current_gen = self.clock.generation;
        // Death pass — drop carrion + remove hunter.
        let mut i = 0;
        while i < self.hunters.len() {
            if self.hunters[i].energy <= 0.0 {
                let pos = self.hunters[i].position;
                for _ in 0..bioscape::HUNTER_CARRION_DROP {
                    let p = [
                        (pos[0] + rng.random_range(-CELL_RADIUS..CELL_RADIUS))
                            .clamp(-WORLD_HALF[0], WORLD_HALF[0]),
                        (pos[1] + rng.random_range(-CELL_RADIUS..CELL_RADIUS))
                            .clamp(-WORLD_HALF[1], WORLD_HALF[1]),
                        pos[2].clamp(-WORLD_HALF[2], WORLD_HALF[2]),
                    ];
                    self.foods.push(Food {
                        position: p,
                        age_ticks: 0,
                        kind: bioscape::FoodKind::HunterCarrion,
                    });
                }
                self.hunters.swap_remove(i);
                self.hunter_deaths_gen += 1;
            } else {
                i += 1;
            }
        }
        // Floor respawn — pokud all extinct, spawn 1 fresh genome.
        if self.hunters.is_empty() {
            let id = self.next_hunter_id;
            self.next_hunter_id += 1;
            self.hunters
                .push(Hunter::random(rng, WORLD_HALF, id, id, current_gen));
            self.hunter_births_gen += 1;
            return;
        }
        // Sprint 98: sexual reproduction. Pair fertile hunters via spatial
        // proximity (mirror cell mating), each pair → 1 mating child, both
        // parents pay (halve energy + cooldown). Birth rate ~50 % vs old
        // asexual path; floor respawn nahoře pokrývá total extinction.
        let budget = bioscape::HUNTER_MAX_POP.saturating_sub(self.hunters.len());
        if budget == 0 {
            return;
        }
        let fertile: Vec<(usize, [f32; 3])> = self
            .hunters
            .iter()
            .enumerate()
            .filter(|(_, h)| {
                h.energy >= bioscape::HUNTER_REPRODUCE_THRESHOLD
                    && h.reproduce_cooldown_ticks == 0
            })
            .map(|(i, h)| (i, h.position))
            .collect();
        if fertile.len() < 2 {
            return;
        }
        let mating_r2 = bioscape::HUNTER_MATING_RADIUS * bioscape::HUNTER_MATING_RADIUS;
        let matings = bioscape::pair_fertile(&fertile, mating_r2, budget, WORLD_HALF);
        let mut children: Vec<Hunter> = Vec::with_capacity(matings.len());
        for &(a_idx, b_idx) in &matings {
            let id = self.next_hunter_id;
            self.next_hunter_id += 1;
            // Mirror cell mating energy semantics: halve oba rodiče PŘED
            // voláním make_*_mating_child. Function sets `child.energy =
            // parent_a.energy + parent_b.energy`, takže pre-halve dává
            // child = 0.5(a+b) a parents 0.5a, 0.5b → energy konzervovaná.
            let (lo, hi) = if a_idx < b_idx {
                (a_idx, b_idx)
            } else {
                (b_idx, a_idx)
            };
            let (left, right) = self.hunters.split_at_mut(hi);
            let parent_lo = &mut left[lo];
            let parent_hi = &mut right[0];
            let (parent_a, parent_b) = if a_idx < b_idx {
                (parent_lo, parent_hi)
            } else {
                (parent_hi, parent_lo)
            };
            parent_a.energy *= 0.5;
            parent_b.energy *= 0.5;
            parent_a.reproduce_cooldown_ticks = bioscape::HUNTER_REPRODUCE_COOLDOWN_TICKS;
            parent_b.reproduce_cooldown_ticks = bioscape::HUNTER_REPRODUCE_COOLDOWN_TICKS;
            children.push(bioscape::make_hunter_mating_child(
                parent_a, parent_b, rng, WORLD_HALF, id, current_gen,
            ));
        }
        let n_children = children.len();
        self.hunters.extend(children);
        self.hunter_births_gen += n_children as u64;
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

        // Sprint 78: cell_id → idx map pro food share lookup. Cells layout
        // se v eat_food nezmění (no spawn/despawn mid-fáze), takže build-once
        // je bezpečný.
        let id_to_idx: rustc_hash::FxHashMap<u64, usize> = self
            .cells
            .iter()
            .enumerate()
            .map(|(i, c)| (c.cell_id, i))
            .collect();

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
                let last_inputs = cell.last_inputs;
                let last_hidden = cell.last_hidden;
                let last_outputs = cell.last_outputs;
                cell.genome.brain.hebbian_update(
                    &last_inputs,
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

        // Sprint 51: GPU Hebbian dispatch — mutuje brain weights na GPU
        // in-place. CPU `cell.genome.brain` se NE-aktualizuje; sync se dělá
        // až v `reproduce` fázi přes `download_brain_at`.
        #[cfg(feature = "gpu")]
        if use_gpu_hebbian {
            let n = self.cells.len();
            let gpu = self.gpu_full.as_ref().unwrap();
            gpu.cells.upload_rewards(&rewards);
            gpu.hebbian.compute_persistent(&gpu.cells, n, LEARNING_RATE);
        }
        let _ = rewards;
    }

    fn spawn_food(&mut self, rng: &mut impl Rng) {
        let target = food_target(self.density_factor);
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
        'spawn: for _ in 0..to_spawn {
            for _ in 0..MAX_SPAWN_ATTEMPTS {
                let candidate = Food::random(rng, WORLD_HALF);
                let richness = self
                    .map
                    .sample([candidate.position[0], candidate.position[1], 0.0]);
                if reject_food_for_richness(rng, richness) {
                    continue;
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
        // Sprint 51: upload child brains + xoshiro state na GPU.
        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.gpu_full.as_ref() {
            for (off, child) in self.cells[child_start..].iter().enumerate() {
                let slot = child_start + off;
                gpu.cells.upload_brain_at(slot, &child.genome.brain);
                gpu.cells.upload_xoshiro_seed_at(
                    slot,
                    child.lineage_id ^ (slot as u64).wrapping_mul(0x9E3779B97F4A7C15),
                );
            }
        }
        let _ = n_births;
    }

    /// Sprint 40: snapshot fertile cells. Sprint 25 mating gating: cells musí
    /// AKTIVNĚ emitovat pheromone (output[2] > threshold) aby reprodukovaly —
    /// selektuje proti free-riders na pheromone field.
    fn collect_fertile(&self) -> Vec<(usize, [f32; 3])> {
        self.cells
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                c.energy >= REPRODUCE_THRESHOLD
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
            children.push(bioscape::make_mating_child(
                cell_a, cell_b, rng, child_ids[i],
            ));
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

fn food_multiplier(noise: f32) -> f32 {
    WORLD_MAP_FOOD_FLOOR + WORLD_MAP_FOOD_AMP * noise
}

fn hazard_drain(noise: f32) -> f32 {
    HAZARD_DRAIN_PER_SEC * (HAZARD_FLOOR + HAZARD_AMP * noise)
}

fn food_target(factor: f32) -> usize {
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
const EDGE_FRAC_THRESHOLD: f32 = 0.9;

fn write_stats<W: Write>(w: &mut W, world: &World, ticks_per_sec: f64) -> std::io::Result<()> {
    let n = world.cells.len();
    if n == 0 {
        // 64 sloupců: gen + 12 cell metrics 0 + food + density + 10 spatial 0
        // + 3 birth/death + 1 atk_emit 0 + predation + 6 brain/density 0 + 2
        // bonds + 6 bond/adhesion 0 + 2 hunter + 1 immune_frac 0 +
        // 3 cell_state 0 (Sprint 80) + 2 vision_fov 0 (Sprint 83) +
        // 1 thermal 0 (Sprint 85) + 2 thermal_optimum 0 (Sprint 87) +
        // 7 hunter genome 0 (Sprint 89) + 2 diet/exposure 0 (Sprint 93).
        // Sprint 99: + 2 hunter bond means + 2 hunter bond formed/broken counters.
        // Sprint 111: i v empty-pop branch reportujeme aktuální shock stav,
        // protože shocky existují nezávisle na živé populaci.
        let (shock_active, shock_hazard_max) = shock_summary(world);
        // Sprint 113: shock_food_factor reportuje i v empty-pop branch.
        let shock_food_factor =
            bioscape::food_density_shock_multiplier(&world.events.events, world.clock.generation);
        return writeln!(
            w,
            "{},0,0,0,0,0,0,0,0,0,0,0,0,{},{:.3},0,0,0,0,0,0,0,0,0,0,{},{},{},0,{},0,0,0,0,0,0,{},{},0,0,0,0,0,0,{},{},0,0,0,0,0,0,0,0,0,{},{},0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,{},{},0,{},{:.3},0.000,{:.3},0,0.000,0.000,{:.1}",
            world.clock.generation,
            world.foods.len(),
            world.density_factor,
            world.births_gen,
            world.deaths_gen,
            world.fertile_ticks_gen,
            world.predation_events_gen,
            world.bonds_formed_gen,
            world.bonds_broken_gen,
            world.hunter_attacks_gen,
            world.hunters.len(),
            world.hunter_births_gen,
            world.hunter_deaths_gen,
            world.hunter_bonds_formed_gen,
            world.hunter_bonds_broken_gen,
            shock_active,
            shock_hazard_max,
            shock_food_factor,
            ticks_per_sec,
        );
    }
    let mut spd_sum = 0.0_f64;
    let mut spd_sumsq = 0.0_f64;
    let mut vis_sum = 0.0_f64;
    let mut vis_sumsq = 0.0_f64;
    let mut len_sum = 0.0_f64;
    let mut wid_sum = 0.0_f64;
    let mut hgt_sum = 0.0_f64;
    let mut asp_sum = 0.0_f64;
    let mut asp_sumsq = 0.0_f64;
    let mut spk_sum = 0.0_f64;
    let mut spk_max = 0.0_f64;
    let mut ph_emit_sum = 0.0_f64;
    let mut atk_emit_sum = 0.0_f64;
    let mut recurrent_io_sum = 0.0_f64;
    let mut density_sum = 0.0_f64;
    let mut density_sumsq = 0.0_f64;
    let mut dmg_sum = 0.0_f64;
    let mut noise_sum = 0.0_f64;
    let mut energy_sum = 0.0_f64;
    let mut abs_x_sum = 0.0_f64;
    let mut abs_y_sum = 0.0_f64;
    let mut x_sum = 0.0_f64;
    let mut y_sum = 0.0_f64;
    let mut edge_count = 0_u64;
    let mut corner_count = 0_u64;
    let mut lineages: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut oldest_age: u64 = 0;
    // Sprint 67 bond/adhesion diagnostics: per-cell aggregates pro CSV.
    let mut bond_signal_sum = 0.0_f64;
    let mut total_bonds = 0_u64;
    let mut bonded_cells = 0_u64;
    // Sprint 71: cells s ≥ HUNTER_BOND_IMMUNITY_THRESHOLD bondů — immune
    // proti hunteru. Pop-level metric pro „kolik populace dosáhlo proto-tissue".
    let mut immune_cells = 0_u64;
    let mut adhesion_hist = [0_u64; bioscape::ADHESION_TYPE_COUNT as usize];
    // Sprint 68: gene-level diagnostics — mean of bond_stiffness / bond_damping
    // přes celou populaci. Porovnáním s initial draw centerem (BOND_STIFFNESS=4.0,
    // BOND_DAMPING=0.6) se dá detekovat selekční drift.
    let mut bond_stiff_sum = 0.0_f64;
    let mut bond_damp_sum = 0.0_f64;
    // Sprint 80: bistable cell_state diagnostics. Mean + std + altruist_frac
    // (cells s state > 0.6 = committed k altruist attractoru). Bimodální
    // distribuce (mean okolo 0.5, std velký, altruist_frac mezi 0.3-0.7) =
    // očekávaný steady-state. Unimodální drift k 0 = celá populace selfish
    // = food share zcela vypnutý → tissue regime collapsuje.
    let mut state_sum = 0.0_f64;
    let mut state_sumsq = 0.0_f64;
    let mut altruist_count = 0_u64;
    // Sprint 83: per-gen FOV diagnostika. Init populace = π (full sphere);
    // pokles avg + růst dev = aktivní selekce na užší kužel.
    let mut fov_sum = 0.0_f64;
    let mut fov_sumsq = 0.0_f64;
    // Sprint 85: thermal niche metric. Mean temperature na cell pozicích —
    // posun směrem k THERMAL_BOTTOM (~4) = populace migrovala ke dnu (cold,
    // pomalý metabolism), posun k THERMAL_TOP (~30) = surface dwellers.
    let mut temp_sum = 0.0_f64;
    // Sprint 87: thermal_optimum gen statistics. Mean + std → speciation
    // ukazuje, jak se populace rozčlenila mezi cold-prefer / warm-prefer
    // fenotypy. Gen 0 mean ≈ (BOTTOM+TOP)/2 = 17 (uniform init), std ≈ 7.5
    // (uniform sigma).
    let mut topt_sum = 0.0_f64;
    let mut topt_sumsq = 0.0_f64;
    // Sprint 89: hunter genome diagnostics. Mean napříč alive hunters —
    // arms race signal v drift těchto hodnot napříč gens.
    let mut h_spd_sum = 0.0_f64;
    let mut h_vis_sum = 0.0_f64;
    let mut h_fov_sum = 0.0_f64;
    let mut h_dmg_sum = 0.0_f64;
    let mut h_size_sum = 0.0_f64;
    // Sprint 93: per-cell diet + defense diagnostics.
    let mut carnivore_sum = 0.0_f64;
    let mut exposure_sum = 0.0_f64;
    // Sprint 97: per-category sensor_gain means + stddev. Specialization
    // signal — high stddev (bimodální distribuce) = role differentiation
    // mezi cells. Low stddev s drift v meanu = uniform shift, pokles bez
    // specializace. Mean alone nerozliší tyto dva režimy.
    let mut gain_sum = [0.0_f64; bioscape::N_SENSOR_CATEGORIES];
    let mut gain_sumsq = [0.0_f64; bioscape::N_SENSOR_CATEGORIES];
    for c in &world.cells {
        carnivore_sum += c.genome.carnivore_score as f64;
        exposure_sum += bioscape::cell_exposure(c.n_bonds()) as f64;
        for k in 0..bioscape::N_SENSOR_CATEGORIES {
            let g = c.genome.sensor_gains[k] as f64;
            gain_sum[k] += g;
            gain_sumsq[k] += g * g;
        }
    }
    let cell_count = world.cells.len();
    let carnivore_m = if cell_count > 0 { carnivore_sum / cell_count as f64 } else { 0.0 };
    let exposure_m = if cell_count > 0 { exposure_sum / cell_count as f64 } else { 0.0 };
    let gain_mean_dev = |k: usize| -> (f64, f64) {
        if cell_count == 0 { return (0.0, 0.0); }
        let n = cell_count as f64;
        let m = gain_sum[k] / n;
        let var = (gain_sumsq[k] / n - m * m).max(0.0);
        (m, var.sqrt())
    };
    let (gain_vis_m, gain_vis_d) = gain_mean_dev(bioscape::SENSOR_CATEGORY_VISION);
    let (gain_chem_m, gain_chem_d) = gain_mean_dev(bioscape::SENSOR_CATEGORY_CHEMISTRY);
    let (gain_def_m, gain_def_d) = gain_mean_dev(bioscape::SENSOR_CATEGORY_DEFENSIVE);
    // Sprint 107: speciation distance diagnostic. Sample N pairs random,
    // průměr compatibility_distance na CPPN. Indicator of population-wide
    // genetic divergence (vyšší = víc speciation).
    let cppn_compat_m: f64 = {
        let n_cells = world.cells.len();
        if n_cells < 2 {
            0.0
        } else {
            let pairs = (n_cells.min(50)).max(2);
            let mut sum = 0.0_f64;
            let mut count = 0u32;
            for k in 0..pairs {
                let i = k % n_cells;
                let j = (k + n_cells / 2) % n_cells;
                if i != j {
                    sum += bioscape::Cppn::compatibility_distance(
                        &world.cells[i].genome.cppn,
                        &world.cells[j].genome.cppn,
                    ) as f64;
                    count += 1;
                }
            }
            if count > 0 {
                sum / count as f64
            } else {
                0.0
            }
        }
    };
    let n_hunters = world.hunters.len();
    let mut h_bond_count_sum: u64 = 0;
    let mut h_bonded_count: u64 = 0;
    if n_hunters > 0 {
        for h in &world.hunters {
            h_spd_sum += h.genome.max_speed as f64;
            h_vis_sum += h.genome.vision_radius as f64;
            h_fov_sum += h.genome.vision_fov as f64;
            h_dmg_sum += h.genome.damage_per_tick as f64;
            h_size_sum += h.genome.body_size as f64;
            // Sprint 99: hunter bond count
            let nb = h.bonds.iter().filter(|b| b.is_some()).count() as u64;
            h_bond_count_sum += nb;
            if nb > 0 {
                h_bonded_count += 1;
            }
        }
    }
    let nhf = n_hunters as f64;
    let h_spd_m = if n_hunters > 0 { h_spd_sum / nhf } else { 0.0 };
    let h_vis_m = if n_hunters > 0 { h_vis_sum / nhf } else { 0.0 };
    let h_fov_m = if n_hunters > 0 { h_fov_sum / nhf } else { 0.0 };
    let h_dmg_m = if n_hunters > 0 { h_dmg_sum / nhf } else { 0.0 };
    let h_size_m = if n_hunters > 0 { h_size_sum / nhf } else { 0.0 };
    let h_bond_n_m = if n_hunters > 0 {
        h_bond_count_sum as f64 / nhf
    } else {
        0.0
    };
    let h_bond_active_f = if n_hunters > 0 {
        h_bonded_count as f64 / nhf
    } else {
        0.0
    };
    let current_gen = world.clock.generation;
    for c in &world.cells {
        let s = c.genome.max_speed as f64;
        let v = c.genome.vision_radius as f64;
        let l = c.phenotype.body_length as f64;
        let wd = c.phenotype.body_width as f64;
        let hg = c.phenotype.body_height as f64;
        let aspect = if wd > 1e-6 { l / wd } else { 0.0 };
        let spk = c.phenotype.primary_spike_length() as f64;
        spd_sum += s;
        spd_sumsq += s * s;
        vis_sum += v;
        vis_sumsq += v * v;
        len_sum += l;
        wid_sum += wd;
        hgt_sum += hg;
        asp_sum += aspect;
        asp_sumsq += aspect * aspect;
        spk_sum += spk;
        let fov = c.genome.vision_fov as f64;
        fov_sum += fov;
        fov_sumsq += fov * fov;
        // Sprint 86: temperature includes diurnal + seasonal cycles. Snapshot
        // se bere v aktuálním clock state — temp_avg pak odráží environmental
        // teplotu cells **v okamžiku end-of-gen**, ne time-averaged.
        temp_sum += bioscape::temperature_at_z(
            c.position[2],
            WORLD_HALF,
            world.clock.tick,
            world.clock.generation,
        ) as f64;
        let topt = c.genome.thermal_optimum as f64;
        topt_sum += topt;
        topt_sumsq += topt * topt;
        if spk > spk_max {
            spk_max = spk;
        }
        ph_emit_sum += c.last_outputs[2].max(0.0) as f64;
        atk_emit_sum += c.last_outputs[6].max(0.0) as f64;
        // Sprint 28 adoption metric: jak silně se používá recurrent state.
        // Mean |last_hidden| napříč 8 dimenzemi → ∈ [0, 1]. Pokud je ~0,
        // brain ignoruje paměť (recurrent weights se nepřipojily k hidden);
        // pokud roste, paměť je aktivně modulovaná.
        let mut h_abs = 0.0_f64;
        for &h in c.last_hidden.iter() {
            h_abs += h.abs() as f64;
        }
        recurrent_io_sum += h_abs / BRAIN_RECURRENT as f64;
        // Sprint 29 quorum sensing metric: index 13 = local_density input.
        let dens = c.last_inputs[13] as f64;
        density_sum += dens;
        density_sumsq += dens * dens;
        // Sprint 30 damage signal adoption: index 14 = damage input. Když roste
        // napříč generacemi, populace je pod selekčním tlakem (predace/hazard),
        // pokud zůstává ~0, žádná nedobrovolná energy loss neprobíhá.
        dmg_sum += c.last_inputs[14] as f64;
        noise_sum += world.map.sample([c.position[0], c.position[1], c.position[2]]) as f64;
        energy_sum += c.energy as f64;
        let nx = (c.position[0] / WORLD_HALF[0]).clamp(-1.0, 1.0);
        let ny = (c.position[1] / WORLD_HALF[1]).clamp(-1.0, 1.0);
        let ax = nx.abs();
        let ay = ny.abs();
        x_sum += nx as f64;
        y_sum += ny as f64;
        abs_x_sum += ax as f64;
        abs_y_sum += ay as f64;
        let near_x = ax >= EDGE_FRAC_THRESHOLD;
        let near_y = ay >= EDGE_FRAC_THRESHOLD;
        if near_x || near_y {
            edge_count += 1;
        }
        if near_x && near_y {
            corner_count += 1;
        }
        lineages.insert(c.lineage_id);
        let age = current_gen.saturating_sub(c.lineage_birth_gen);
        if age > oldest_age {
            oldest_age = age;
        }
        // Sprint 67 bond/adhesion diagnostics.
        bond_signal_sum += c.last_outputs[9].max(0.0) as f64;
        let cell_bonds = c.bonds.iter().filter(|b| b.is_some()).count() as u64;
        total_bonds += cell_bonds;
        if cell_bonds > 0 {
            bonded_cells += 1;
        }
        // Sprint 71: hunter-immune cells (≥3 bondy → cluster „too big to swallow").
        if cell_bonds >= bioscape::HUNTER_BOND_IMMUNITY_THRESHOLD as u64 {
            immune_cells += 1;
        }
        let t_idx = (c.genome.adhesion_type as usize) % adhesion_hist.len();
        adhesion_hist[t_idx] += 1;
        // Sprint 68 gene means.
        bond_stiff_sum += c.genome.bond_stiffness as f64;
        bond_damp_sum += c.genome.bond_damping as f64;
        // Sprint 80 cell_state.
        let cs = c.cell_state as f64;
        state_sum += cs;
        state_sumsq += cs * cs;
        if cs > 0.6 {
            altruist_count += 1;
        }
    }
    let nf = n as f64;
    let spd_m = spd_sum / nf;
    let vis_m = vis_sum / nf;
    let len_m = len_sum / nf;
    let wid_m = wid_sum / nf;
    let hgt_m = hgt_sum / nf;
    let asp_m = asp_sum / nf;
    let spk_m = spk_sum / nf;
    let spd_d = ((spd_sumsq / nf) - spd_m * spd_m).max(0.0).sqrt();
    let vis_d = ((vis_sumsq / nf) - vis_m * vis_m).max(0.0).sqrt();
    let asp_d = ((asp_sumsq / nf) - asp_m * asp_m).max(0.0).sqrt();
    let ph_emit_m = ph_emit_sum / nf;
    let atk_emit_m = atk_emit_sum / nf;
    let recurrent_io_m = recurrent_io_sum / nf;
    let energy_m = energy_sum / nf;
    let abs_x_m = abs_x_sum / nf;
    let abs_y_m = abs_y_sum / nf;
    let x_m = x_sum / nf;
    let y_m = y_sum / nf;
    let edge_f = edge_count as f64 / nf;
    let corner_f = corner_count as f64 / nf;
    let density_m = density_sum / nf;
    let density_d = ((density_sumsq / nf) - density_m * density_m).max(0.0).sqrt();
    let dmg_m = dmg_sum / nf;
    let noise_m = noise_sum / nf;
    // Sprint 67 bond/adhesion diagnostics.
    let bond_signal_m = bond_signal_sum / nf;
    let mean_bond_count = total_bonds as f64 / nf;
    let bond_active_frac = bonded_cells as f64 / nf;
    // Shannon entropy of adhesion_type distribution, normalized by log2(K)
    // (K = ADHESION_TYPE_COUNT) — uniformní distribuce → 1.0, monokultura → 0.0.
    let adhesion_entropy = {
        let k = adhesion_hist.len() as f64;
        if k <= 1.0 {
            0.0
        } else {
            let mut h = 0.0_f64;
            for &count in adhesion_hist.iter() {
                if count == 0 {
                    continue;
                }
                let p = count as f64 / nf;
                h -= p * p.log2();
            }
            h / k.log2()
        }
    };
    // Sprint 68 gene means.
    let bond_stiff_m = bond_stiff_sum / nf;
    let bond_damp_m = bond_damp_sum / nf;
    // Sprint 80 cell_state stats.
    let state_m = state_sum / nf;
    let state_d = ((state_sumsq / nf) - state_m * state_m).max(0.0).sqrt();
    let altruist_frac = altruist_count as f64 / nf;
    // Sprint 83 vision_fov stats.
    let fov_m = fov_sum / nf;
    let fov_d = ((fov_sumsq / nf) - fov_m * fov_m).max(0.0).sqrt();
    // Sprint 85 thermal niche — mean teplota.
    let temp_m = temp_sum / nf;
    // Sprint 87 thermal_optimum gen.
    let topt_m = topt_sum / nf;
    let topt_d = ((topt_sumsq / nf) - topt_m * topt_m).max(0.0).sqrt();
    // Sprint 29 spatial clustering metric: mean nearest-neighbor distance.
    // Sprint 43: grid lookup s expanding radius. Začni na GRID_CELL_SIZE (=64),
    // pokud nikdo není, double až po WORLD diagonal — typický nn dist je < 50,
    // takže first try téměř vždy najde souseda.
    let nn_dist_m = if n >= 2 {
        let mut grid: SpatialGrid<usize, ()> = SpatialGrid::new(GRID_CELL_SIZE);
        grid.rebuild(
            world
                .cells
                .iter()
                .enumerate()
                .map(|(i, c)| (i, c.position, ())),
        );
        let world_diag = (4.0 * (WORLD_HALF[0] * WORLD_HALF[0] + WORLD_HALF[1] * WORLD_HALF[1]))
            .sqrt();
        let mut sum = 0.0_f64;
        for i in 0..n {
            let pi = world.cells[i].position;
            let mut min_d2 = f32::MAX;
            let mut search_r = GRID_CELL_SIZE;
            while min_d2 == f32::MAX && search_r <= world_diag {
                grid.for_each_in_radius_toroidal(pi, search_r, WORLD_HALF, |j, pj, _| {
                    if i == j {
                        return;
                    }
                    let d = bioscape::min_image_delta(pi, pj, WORLD_HALF);
                    let d2 = d[0] * d[0] + d[1] * d[1];
                    if d2 < min_d2 {
                        min_d2 = d2;
                    }
                });
                search_r *= 2.0;
            }
            if min_d2 < f32::MAX {
                sum += min_d2.sqrt() as f64;
            }
        }
        sum / nf
    } else {
        0.0
    };
    let immune_f = immune_cells as f64 / nf;
    // Sprint 111: shock observability + diversity metrics (Shannon attack
    // entropy, brain w1 frobenius std).
    let (shock_active, shock_hazard_max) = shock_summary(world);
    // Sprint 112: per-population mean climate offset (suma signed offsetů /
    // n). Identical 0.0 když events.empty nebo žádný ClimateShift aktivní.
    let climate_offset_avg: f64 = if n > 0 {
        let sum: f64 = world
            .cells
            .iter()
            .map(|c| world.climate_offset_at([c.position[0], c.position[1]]) as f64)
            .sum();
        sum / nf
    } else {
        0.0
    };
    let lineage_count = lineages.len();
    let behavioral_entropy_attack = attack_entropy(&world.cells);
    let weight_diversity_w1_norm = w1_frobenius_std(&world.cells);
    // Sprint 113: globální FoodCrash multiplikátor (1.0 default, < 1.0 při
    // aktivním FoodCrash). Single per-gen scalar — žádný spatial average.
    let shock_food_factor =
        bioscape::food_density_shock_multiplier(&world.events.events, world.clock.generation);
    writeln!(
        w,
        "{},{},{:.2},{:.3},{:.2},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{:.3},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2},{},{},{},{:.3},{},{:.3},{:.2},{:.3},{:.3},{:.3},{:.3},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2},{:.2},{:.3},{},{},{:.2},{:.2},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{},{:.3},{},{:.3},{:.3},{:.3},{},{:.3},{:.3},{:.1}",
        world.clock.generation,
        n,
        spd_m,
        spd_d,
        vis_m,
        vis_d,
        len_m,
        wid_m,
        hgt_m,
        asp_m,
        asp_d,
        spk_m,
        spk_max,
        world.foods.len(),
        world.density_factor,
        lineages.len(),
        oldest_age,
        ph_emit_m,
        abs_x_m,
        abs_y_m,
        edge_f,
        corner_f,
        x_m,
        y_m,
        energy_m,
        world.births_gen,
        world.deaths_gen,
        world.fertile_ticks_gen,
        atk_emit_m,
        world.predation_events_gen,
        recurrent_io_m,
        nn_dist_m,
        density_m,
        density_d,
        dmg_m,
        noise_m,
        // Sprint 67 bond/adhesion diagnostics.
        world.bonds_formed_gen,
        world.bonds_broken_gen,
        mean_bond_count,
        bond_active_frac,
        bond_signal_m,
        adhesion_entropy,
        // Sprint 68 gene means.
        bond_stiff_m,
        bond_damp_m,
        // Sprint 71 hunter diagnostics.
        world.hunter_attacks_gen,
        world.hunters.len(),
        immune_f,
        // Sprint 80 cell_state diagnostics.
        state_m,
        state_d,
        altruist_frac,
        // Sprint 83 vision_fov diagnostics.
        fov_m,
        fov_d,
        // Sprint 85 thermal niche.
        temp_m,
        // Sprint 87 thermal_optimum gen.
        topt_m,
        topt_d,
        // Sprint 89 hunter genome + lifecycle.
        world.hunter_births_gen,
        world.hunter_deaths_gen,
        h_spd_m,
        h_vis_m,
        h_fov_m,
        h_dmg_m,
        h_size_m,
        // Sprint 93 diet + defense diagnostics.
        carnivore_m,
        exposure_m,
        // Sprint 97 sensor_gain category means + stddev.
        gain_vis_m,
        gain_chem_m,
        gain_def_m,
        gain_vis_d,
        gain_chem_d,
        gain_def_d,
        // Sprint 99 hunter bond stats.
        h_bond_n_m,
        h_bond_active_f,
        world.hunter_bonds_formed_gen,
        world.hunter_bonds_broken_gen,
        // Sprint 107: CPPN speciation distance.
        cppn_compat_m,
        // Sprint 111: shock state + diversity metrics.
        // Sprint 112: shock_climate_offset = mean signed temperature offset
        // přes všechny cells z aktivních ClimateShift shocků (0.0 default).
        // Sprint 113: shock_food_factor = global FoodCrash multiplikátor
        // (1.0 default, clamp na FOOD_CRASH_MIN_FACTOR floor).
        shock_active,
        shock_hazard_max,
        climate_offset_avg,
        shock_food_factor,
        lineage_count,
        behavioral_entropy_attack,
        weight_diversity_w1_norm,
        ticks_per_sec,
    )
}

/// Sprint 111: aktivní shock summary pro CSV. Vrací `(count, hazard_intensity_max)`,
/// kde max je `intensity * shock_ramp_factor(event, gen)` přes všechny aktivní
/// `HazardPulse` eventy. Bez aktivních eventů nebo žádný HazardPulse → `(0, 0.0)`.
fn shock_summary(world: &World) -> (u32, f64) {
    let gen = world.clock.generation;
    let tick = world.clock.tick;
    let mut count: u32 = 0;
    let mut hazard_max: f64 = 0.0;
    for event in world.events.active(gen, tick) {
        count += 1;
        if matches!(event.kind, bioscape::ShockKind::HazardPulse) {
            let factor = bioscape::shock_ramp_factor(event, gen);
            let intensity = (event.intensity * factor) as f64;
            if intensity > hazard_max {
                hazard_max = intensity;
            }
        }
    }
    (count, hazard_max)
}

/// Sprint 111: Shannon entropy histogramu `cell.last_outputs[6]` (attack signal).
/// 8 bins na intervalu [-1, 1], hodnoty mimo se clampují. Empty population → 0.0.
fn attack_entropy(cells: &[Cell]) -> f64 {
    if cells.is_empty() {
        return 0.0;
    }
    const BINS: usize = 8;
    let mut hist = [0_u64; BINS];
    for c in cells {
        let v = c.last_outputs[6].clamp(-1.0, 1.0);
        let mut idx = ((v + 1.0) * 0.5 * BINS as f32) as usize;
        if idx >= BINS {
            idx = BINS - 1;
        }
        hist[idx] += 1;
    }
    let total = cells.len() as f64;
    let mut h = 0.0_f64;
    for &count in hist.iter() {
        if count == 0 {
            continue;
        }
        let p = count as f64 / total;
        h -= p * p.log2();
    }
    h
}

/// Sprint 111: zapíše scheduled shock events do sidecar CSV vedle hlavního
/// out_path (`events_seed{seed}.csv`). Caller už ověřil, že `events` není prázdný.
fn write_events_sidecar(
    out_path: &Path,
    seed: u64,
    events: &EventCalendar,
) -> std::io::Result<()> {
    let dir = out_path.parent().unwrap_or_else(|| Path::new("."));
    let sidecar = dir.join(format!("events_seed{}.csv", seed));
    let file = std::fs::File::create(&sidecar)?;
    let mut w = BufWriter::new(file);
    writeln!(w, "start_gen,duration_gen,kind,intensity,center_x,center_y,radius")?;
    for e in &events.events {
        let kind = match e.kind {
            bioscape::ShockKind::HazardPulse => "hazard_pulse",
            bioscape::ShockKind::ClimateShift => "climate_shift",
            bioscape::ShockKind::FoodCrash => "food_crash",
        };
        let cx = e
            .center_xy
            .map(|c| format!("{:.2}", c[0]))
            .unwrap_or_default();
        let cy = e
            .center_xy
            .map(|c| format!("{:.2}", c[1]))
            .unwrap_or_default();
        let r = e
            .radius
            .map(|r| format!("{:.2}", r))
            .unwrap_or_default();
        writeln!(
            w,
            "{},{},{},{:.3},{},{},{}",
            e.start_gen, e.duration_gen, kind, e.intensity, cx, cy, r
        )?;
    }
    w.flush()?;
    Ok(())
}

/// Sprint 111: per-cell brain w1 Frobenius norm, pak std přes všechny cells.
/// `std = sqrt(mean((x_i - mean)²))`. Empty population → 0.0.
fn w1_frobenius_std(cells: &[Cell]) -> f64 {
    if cells.is_empty() {
        return 0.0;
    }
    let norms: Vec<f64> = cells
        .iter()
        .map(|c| {
            let mut sumsq = 0.0_f64;
            for row in c.genome.brain.w1.iter() {
                for &w in row.iter() {
                    sumsq += (w as f64) * (w as f64);
                }
            }
            sumsq.sqrt()
        })
        .collect();
    let n = norms.len() as f64;
    let mean = norms.iter().sum::<f64>() / n;
    let var = norms.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    var.sqrt()
}

fn main() {
    let raw_args: Vec<String> = env::args().collect();
    // Sprint 44: `--gpu` flag (filtered před positional parsingem). Bez
    // `--features gpu` se flag tiše ignoruje.
    // Sprint 51: `--gpu-full` flag — persistent brain weights + GPU Hebbian +
    // GPU Brownian. Implies --gpu (brain forward na GPU).
    let want_gpu_full = raw_args.iter().any(|a| a == "--gpu-full");
    let want_gpu = want_gpu_full || raw_args.iter().any(|a| a == "--gpu");
    // Sprint 48: `--save=PATH` / `--load=PATH` checkpoint flags. Form
    // `--key=value` aby se PATH ne-leakoval do positional indexingu.
    let save_path: Option<String> = raw_args
        .iter()
        .find_map(|a| a.strip_prefix("--save=").map(|s| s.to_string()));
    let load_path: Option<String> = raw_args
        .iter()
        .find_map(|a| a.strip_prefix("--load=").map(|s| s.to_string()));
    // Sprint 87 Hamilton sweep: `--share-frac=X` runtime override pro
    // BOND_FOOD_SHARE_FRAC, `--kin` zapne kin filter (food share jen na
    // partnery se stejným lineage_id).
    let share_frac_override: Option<f32> = raw_args
        .iter()
        .find_map(|a| a.strip_prefix("--share-frac=").and_then(|s| s.parse().ok()));
    let kin_filter = raw_args.iter().any(|a| a == "--kin");
    // Sprint 109: `--shocks-mean-gens N` (space-separated) nebo
    // `--shocks-mean-gens=N` (= form). Default 0 = no-op (empty kalendář).
    // `consumed_value_idx` drží pozici následujícího raw arg pokud je flag
    // space-separated; ten se musí vyfiltrovat z positional setu.
    let mut shocks_mean_gens: u32 = 0;
    let mut consumed_value_idx: Option<usize> = None;
    for (i, a) in raw_args.iter().enumerate() {
        if let Some(rest) = a.strip_prefix("--shocks-mean-gens") {
            if let Some(eq_val) = rest.strip_prefix('=') {
                if let Ok(v) = eq_val.parse::<u32>() {
                    shocks_mean_gens = v;
                }
            } else if rest.is_empty() {
                if let Some(next) = raw_args.get(i + 1) {
                    if let Ok(v) = next.parse::<u32>() {
                        shocks_mean_gens = v;
                        consumed_value_idx = Some(i + 1);
                    }
                }
            }
            break;
        }
    }
    let args: Vec<String> = raw_args
        .iter()
        .enumerate()
        .filter(|(i, a)| !a.starts_with("--") && Some(*i) != consumed_value_idx)
        .map(|(_, a)| a.clone())
        .collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let max_gens: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(500);
    let out_path = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| format!("run_seed{}.csv", seed));
    let map_seed: u64 = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(WORLD_MAP_SEED);
    let mating_radius: f32 = args
        .get(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(MATING_RADIUS);
    // Sprint 43: positional override pro initial cells / max population /
    // rayon thread count. Default zachovává pre-Sprint-43 chování.
    let initial_cells: usize = args
        .get(6)
        .and_then(|s| s.parse().ok())
        .unwrap_or(INITIAL_CELLS);
    let max_population: usize = args
        .get(7)
        .and_then(|s| s.parse().ok())
        .unwrap_or(MAX_POPULATION);
    let threads: usize = args
        .get(8)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });

    if threads > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global();
    }

    let mut rng = StdRng::seed_from_u64(seed);
    // Sprint 109: kalendář environmentálních shocků. Když mean_gens_between == 0,
    // generate vrátí prázdný kalendář (no-op) — byte-identical s pre-S109 baseline.
    let shock_cfg = if shocks_mean_gens > 0 {
        ShockScheduleConfig {
            mean_gens_between: shocks_mean_gens,
            ..Default::default()
        }
    } else {
        ShockScheduleConfig::default()
    };
    let events = EventCalendar::generate(seed, &shock_cfg, max_gens);
    // Sprint 111: pokud kalendář není prázdný, dump sidecar `events_seed{seed}.csv`
    // vedle hlavního CSV. Žádný file pokud `events` empty (žádný error).
    if !events.events.is_empty() {
        if let Err(e) = write_events_sidecar(Path::new(&out_path), seed, &events) {
            eprintln!("events sidecar: write failed ({e})");
        }
    }
    let mut world = if let Some(path) = load_path.as_ref() {
        match World::load_checkpoint(Path::new(path)) {
            Ok(mut w) => {
                eprintln!(
                    "checkpoint: loaded {} (cells={}, foods={}, gen={}, tick={})",
                    path,
                    w.cells.len(),
                    w.foods.len(),
                    w.clock.generation,
                    w.clock.tick,
                );
                w.events = events.clone();
                w
            }
            Err(e) => {
                eprintln!("checkpoint: load failed ({e}); starting fresh");
                World::new(
                    &mut rng,
                    map_seed,
                    mating_radius,
                    initial_cells,
                    max_population,
                    events.clone(),
                )
            }
        }
    } else {
        World::new(
            &mut rng,
            map_seed,
            mating_radius,
            initial_cells,
            max_population,
            events,
        )
    };
    // Sprint 87 Hamilton sweep: aplikuj CLI overrides AFTER World::new (i po
    // checkpoint load) — nikdy se neserializují, vždy z aktuálního CLI.
    if let Some(sf) = share_frac_override {
        world.share_frac = sf;
    }
    world.kin_filter = kin_filter;

    #[cfg(feature = "gpu")]
    if want_gpu_full {
        let cap = initial_cells.max(max_population).max(64);
        // Sprint 59: FieldGpu sources capacity. Per-tick deposit count =
        // foods (smell) + cells (pheromone). food_target může bumpnout přes
        // density cycles (CYCLE_AMPLITUDE), s safety margin × 2.
        let field_sources_cap = (food_target(1.0 + CYCLE_AMPLITUDE) + max_population) * 2;
        let init = || -> Result<GpuFullState, String> {
            let ctx = GpuContext::new()?;
            let cells_gpu = CellsGpu::with_context(&ctx, cap);
            cells_gpu.upload_brains(world.cells.iter().map(|c| &c.genome.brain));
            cells_gpu.upload_xoshiro_seeds(world.cells.iter().enumerate().map(|(slot, c)| {
                c.lineage_id ^ (slot as u64).wrapping_mul(0x9E3779B97F4A7C15)
            }));
            let brain = BrainGpu::with_context(&ctx, cap)?;
            let hebbian = HebbianGpu::with_context(&ctx, cap)?;
            let brownian = BrownianGpu::with_context(&ctx, cap)?;
            // Sprint 59: smell + pheromone FieldGpu instances, sdílí GpuContext.
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
            // Sprint 60: spatial hashes pro sensor broad-phase. Sdílí
            // GRID_CELL_SIZE konstantu s CPU SpatialGrid; xy world bounds
            // pro toroidal bucket wrap.
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
            // Sprint 62: turn_rate je per-cell genome konstanta. Upload na sim
            // init; reproduce volá `upload_turn_rates` znovu (per-event sparse).
            let turn_rates: Vec<f32> = world.cells.iter().map(|c| c.genome.turn_rate).collect();
            cells_gpu.upload_turn_rates(&turn_rates);
            Ok(GpuFullState {
                cells: cells_gpu,
                brain,
                hebbian,
                brownian,
                smell,
                pheromone,
                cell_hash,
                food_hash,
                sensor,
                populate,
                motor,
                step,
            })
        };
        match init() {
            Ok(state) => {
                eprintln!(
                    "gpu-full: brain + Hebbian + Brownian + Field + SensorGather + PopulateInputs + Motor + Step (cap {} cells, {} field sources)",
                    cap, field_sources_cap
                );
                world.gpu_full = Some(state);
            }
            Err(e) => {
                eprintln!("gpu-full: init failed ({e}); fallback to CPU");
            }
        }
    }
    #[cfg(feature = "gpu")]
    if want_gpu && !want_gpu_full && world.gpu_full.is_none() {
        match BrainGpu::new(initial_cells.max(64)) {
            Ok(g) => {
                eprintln!("gpu: BrainGpu initialized (capacity {})", initial_cells.max(64));
                world.gpu = Some(g);
            }
            Err(e) => {
                eprintln!("gpu: init failed ({e}); falling back to CPU");
            }
        }
    }
    #[cfg(not(feature = "gpu"))]
    if want_gpu {
        eprintln!("gpu: --gpu / --gpu-full requested but binary built without --features gpu");
    }

    let file = std::fs::File::create(&out_path).expect("can't create output file");
    let mut log = BufWriter::new(file);
    writeln!(
        log,
        "gen,cells,spd_avg,spd_dev,vis_avg,vis_dev,len_avg,wid_avg,hgt_avg,asp_avg,asp_dev,spk_avg,spk_max,food,density,lineages,oldest,ph_emit,abs_x,abs_y,edge_frac,corner_frac,mean_x,mean_y,energy_avg,births,deaths,fertile_ticks,atk_emit,predation_events,recurrent_io,nn_dist_avg,density_avg,density_dev,dmg_avg,noise_avg,bonds_formed,bonds_broken,mean_bond_count,bond_active_frac,bond_signal_avg,adhesion_entropy,bond_stiff_avg,bond_damp_avg,hunter_attacks,hunters_alive,immune_frac,state_avg,state_dev,altruist_frac,fov_avg,fov_dev,temp_avg,topt_avg,topt_dev,hunter_births,hunter_deaths,h_spd_avg,h_vis_avg,h_fov_avg,h_dmg_avg,h_size_avg,carnivore_avg,exposure_avg,gain_vis_avg,gain_chem_avg,gain_def_avg,gain_vis_dev,gain_chem_dev,gain_def_dev,h_bond_n,h_bond_active,h_bonds_formed,h_bonds_broken,cppn_compat,shock_active_count,shock_hazard_intensity_max,shock_climate_offset,shock_food_factor,lineage_count,behavioral_entropy_attack,weight_diversity_w1_norm,ticks_per_sec"
    )
    .unwrap();
    write_stats(&mut log, &world, 0.0).unwrap();

    let baseline_samples = 10_000;
    let mut bsum = 0.0_f64;
    let mut brng = StdRng::seed_from_u64(99);
    for _ in 0..baseline_samples {
        let p = [
            brng.random_range(-WORLD_HALF[0]..WORLD_HALF[0]),
            brng.random_range(-WORLD_HALF[1]..WORLD_HALF[1]),
            brng.random_range(-WORLD_HALF[2]..WORLD_HALF[2]),
        ];
        bsum += world.map.sample(p) as f64;
    }
    let noise_baseline = bsum / baseline_samples as f64;
    eprintln!("noise_baseline (uniform-position mean over map): {:.4}", noise_baseline);

    eprintln!(
        "headless: seed={} map_seed={} mating_radius={} max_gens={} out={} initial_cells={} initial_food={} max_pop={} threads={}",
        seed,
        map_seed,
        mating_radius,
        max_gens,
        out_path,
        world.cells.len(),
        world.foods.len(),
        max_population,
        rayon::current_num_threads()
    );
    eprintln!(
        "shocks: mean_gens_between={} scheduled={} (sim loop integration arrives in S110+)",
        shocks_mean_gens,
        world.events.events.len()
    );

    let start = Instant::now();
    let mut gen_start = Instant::now();
    let mut gen_ticks = 0_u64;
    while world.clock.generation < max_gens {
        let gen_ended = world.tick(&mut rng);
        gen_ticks += 1;
        if gen_ended.is_some() {
            let gen_elapsed = gen_start.elapsed().as_secs_f64();
            let tps = if gen_elapsed > 0.0 {
                gen_ticks as f64 / gen_elapsed
            } else {
                0.0
            };
            write_stats(&mut log, &world, tps).unwrap();
            gen_start = Instant::now();
            gen_ticks = 0;
            // Sprint 43: po první dokončené generaci vypiš per-fáze timing
            // (mikrosekundy total + průměr per tick). Reset accumulator.
            if world.clock.generation == 1 {
                let t = world.bench_timings;
                let ticks = TICKS_PER_GENERATION as f64;
                let dump = |name: &str, total_us: f64| {
                    eprintln!(
                        "phase={} n={} ticks={} us_total={:.1} us_avg={:.3}",
                        name,
                        world.cells.len(),
                        TICKS_PER_GENERATION,
                        total_us,
                        total_us / ticks
                    );
                };
                dump("update_smell", t.update_smell);
                dump("update_pheromone", t.update_pheromone);
                dump("brain_act", t.brain_act);
                dump("emit_pheromones", t.emit_pheromones);
                dump("apply_morph", t.apply_morph);
                dump("apply_brownian", t.apply_brownian);
                dump("step", t.step);
                dump("apply_food_gravity", t.apply_food_gravity);
                dump("apply_hazards", t.apply_hazards);
                dump("resolve_collisions", t.resolve_collisions);
                dump("predate", t.predate);
                dump("hunt", t.hunt);
                dump("eat_food", t.eat_food);
                dump("spawn_food", t.spawn_food);
                dump("reproduce", t.reproduce);
                dump("die_and_drop_carrion", t.die_and_drop_carrion);
                world.bench_timings = PhaseTimings::default();
            }
            world.births_gen = 0;
            world.deaths_gen = 0;
            world.fertile_ticks_gen = 0;
            world.predation_events_gen = 0;
            // Sprint 66: bond formation/break per-gen counters.
            world.bonds_formed_gen = 0;
            world.bonds_broken_gen = 0;
            // Sprint 71: hunter attack counter.
            world.hunter_attacks_gen = 0;
            // Sprint 89: hunter lifecycle counters.
            world.hunter_births_gen = 0;
            world.hunter_deaths_gen = 0;
            // Sprint 99: hunter bond counters.
            world.hunter_bonds_formed_gen = 0;
            world.hunter_bonds_broken_gen = 0;
        }
        if world.cells.is_empty() {
            eprintln!("extinction at gen {}", world.clock.generation);
            break;
        }
    }
    log.flush().unwrap();

    if let Some(path) = save_path.as_ref() {
        match world.save_checkpoint(Path::new(path)) {
            Ok(()) => eprintln!(
                "checkpoint: saved to {} (cells={}, gen={}, tick={})",
                path,
                world.cells.len(),
                world.clock.generation,
                world.clock.tick,
            ),
            Err(e) => eprintln!("checkpoint: save failed ({e})"),
        }
    }

    let elapsed = start.elapsed();
    let ticks_per_sec = world.clock.tick as f32 / elapsed.as_secs_f32().max(1e-3);
    eprintln!(
        "done. {} gen, {} ticks in {:.1}s ({:.0} ticks/s). final pop: {}",
        world.clock.generation,
        world.clock.tick,
        elapsed.as_secs_f32(),
        ticks_per_sec,
        world.cells.len()
    );
}
