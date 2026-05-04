//! Headless harness — pure simulation loop, no Bevy renderer.
//!
//! Usage: `cargo run --release --bin headless -- [seed] [max_gens] [out_path]`
//! Defaults: seed=0, max_gens=500, out_path=run_seed{seed}.csv
//!
//! Logs per-generation stats (cell_count, mean/dev for max_speed/vision/body_size,
//! food count, density factor) to CSV. Reproducible: same seed → identical run.

use bioscape::{
    reject_food_for_richness, Cell, Food, SimClock, SmellField, SpatialGrid, WorldMap,
    ATTACK_THRESHOLD, BRAIN_RECURRENT, CARRION_FOOD_COUNT, CELL_RADIUS, CYCLE_AMPLITUDE,
    CYCLE_GEN_PERIOD, DILUTION_K, EAT_RADIUS, FIXED_TIMESTEP_HZ, FOOD_SPAWN_RATE, FOOD_VALUE,
    GENERATIONS_PER_EPOCH, GRID_CELL_SIZE, HAZARD_AMP, HAZARD_DRAIN_PER_SEC, HAZARD_FLOOR,
    HERD_RADIUS, INITIAL_CELLS, LEARNING_RATE, MATING_COOLDOWN_TICKS, MATING_PHEROMONE_THRESHOLD,
    MATING_RADIUS, MAX_POPULATION, MAX_SPAWN_ATTEMPTS, PHEROMONE_BASELINE_EMIT,
    PHEROMONE_BRAIN_MOD, PHEROMONE_COST_PER_RATE, PHEROMONE_DECAY, PHEROMONE_DIFFUSION,
    PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z, PHEROMONE_SAMPLE_EPSILON, PHYSICS_CONFIG,
    PREDATION_DRAIN_PER_TICK, PREDATION_GAIN_PER_TICK, REPRODUCE_THRESHOLD, SIZE_RATIO_THRESHOLD,
    SMELL_DECAY, SMELL_DIFFUSION, SMELL_GRID_RES, SMELL_GRID_RES_Z, SMELL_PER_FOOD,
    SMELL_SAMPLE_EPSILON, TICKS_PER_GENERATION, WORLD_MAP_BASE_RES, WORLD_MAP_BASE_RES_Z,
    WORLD_MAP_FOOD_AMP, WORLD_MAP_FOOD_FLOOR, WORLD_MAP_RES, WORLD_MAP_RES_Z, WORLD_MAP_SEED,
    WORLD_UNITS_PER_FOOD,
};
#[cfg(feature = "gpu")]
use bioscape::{
    gpu::{BrainGpu, BrownianGpu, CellsGpu, GpuContext, HebbianGpu},
    BRAIN_HIDDEN, BRAIN_INPUTS, BRAIN_OUTPUTS,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::env;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

// Headless has no window — fixed extent so seeds reproduce identically across
// machines. Sim parameters live in `bioscape`.
// Sprint 53: WORLD_HALF[2] expanded z=2 → z=20. SmellField + WorldMap +
// Pheromone jsou plně 3D (volumetric grid + 7-point Jacobi diffusion + 3D
// gradient). Cells získávají vertikální environmental sensing (smell_grad_z,
// pheromone_grad_z přes inputs[17,19]). z=20 je conservative bump (z=50 v
// initial smoke způsobil extinkci kolem gen 30 kvůli food sparsity v 25×
// větším objemu).
const WORLD_HALF: [f32; 3] = [960.0, 540.0, 20.0];

/// Sprint 48: versioned binary header pro checkpoint files.
const CHECKPOINT_MAGIC: &[u8; 8] = b"BIOSCP01";
const CHECKPOINT_VERSION: u32 = 1;

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
    predate: f64,
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
    food_grid: SpatialGrid<usize, ()>,
    // Persistent scratch — reused per tick to avoid hot-loop allocations.
    deltas_scratch: Vec<[f32; 3]>,
    energy_deltas_scratch: Vec<f32>,
    damage_deltas_scratch: Vec<f32>,
    eaten_scratch: Vec<bool>,
    births_gen: u64,
    deaths_gen: u64,
    fertile_ticks_gen: u64,
    predation_events_gen: u64,
    mating_radius: f32,
    // Sprint 43: runtime override `MAX_POPULATION` consts. Default = const, CLI
    // může nastavit výš (potřeba pro bench při N > 1000).
    max_population: usize,
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
}

impl World {
    fn new(
        rng: &mut impl Rng,
        map_seed: u64,
        mating_radius: f32,
        initial_cells: usize,
        max_population: usize,
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
            .map(|i| Cell::random(rng, WORLD_HALF, i as u64, 0))
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
            energy_deltas_scratch: Vec::new(),
            damage_deltas_scratch: Vec::new(),
            eaten_scratch: Vec::new(),
            births_gen: 0,
            deaths_gen: 0,
            fertile_ticks_gen: 0,
            predation_events_gen: 0,
            mating_radius,
            max_population,
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
            energy_deltas_scratch: Vec::new(),
            damage_deltas_scratch: Vec::new(),
            eaten_scratch: Vec::new(),
            births_gen: chk.births_gen,
            deaths_gen: chk.deaths_gen,
            fertile_ticks_gen: chk.fertile_ticks_gen,
            predation_events_gen: chk.predation_events_gen,
            mating_radius: chk.mating_radius,
            max_population: chk.max_population,
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
            self.density_factor = 1.0 + CYCLE_AMPLITUDE * phase.sin();
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
        timed!(brain_act, self.run_brain_act(dt));
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
            if self.gpu_full.is_some() {
                self.apply_brownian_gpu(dt);
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
    #[cfg(feature = "gpu")]
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
        self.pheromone.step(PHEROMONE_DIFFUSION, PHEROMONE_DECAY, dt);
    }

    fn emit_pheromones(&mut self, dt: f32) {
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

    /// Sprint 51: --gpu-full brain_act. Sensor gather + populate_brain_inputs
    /// (CPU rayon, jako Sprint 44 path), pak GPU forward s persistent weights
    /// (žádný 30 MB upload), download hidden + outputs, CPU motor.
    #[cfg(feature = "gpu")]
    fn brain_act_gpu_full(&mut self, dt: f32) {
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
                .map(|(i, f)| (i, f.position, ())),
        );
        let cell_grid = &self.cell_grid;
        let food_grid = &self.food_grid;
        let smell = &self.smell;
        let pheromone = &self.pheromone;

        // Phase 1: CPU rayon — sensor gather + populate_brain_inputs.
        // Sprint 54: toroidal sensor gather přes ghost positions + min-image
        // delta, ukládá min-imaged delta do BrainSensors.nearest_*.
        let inputs_vec: Vec<[f32; BRAIN_INPUTS]> = self
            .cells
            .par_iter_mut()
            .enumerate()
            .map(|(i, cell)| {
                let pos = cell.position;
                let vision_r = cell.genome.vision_radius;
                let vr2 = vision_r * vision_r;
                let mut best_food: Option<[f32; 3]> = None;
                let mut best_food_d2 = f32::MAX;
                food_grid.for_each_in_radius_toroidal(pos, vision_r, WORLD_HALF, |_id, fp, ()| {
                    let d = bioscape::min_image_delta(pos, fp, WORLD_HALF);
                    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if d2 <= vr2 && d2 < best_food_d2 {
                        best_food_d2 = d2;
                        best_food = Some(d);
                    }
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
                    if d2 <= vr2 {
                        neighbors_in_vision += 1;
                        if d2 < best_cell_d2 {
                            best_cell_d2 = d2;
                            best_cell = Some((d, oradius));
                        }
                    }
                });
                let pos_xyz = [pos[0], pos[1], pos[2]];
                let smell_grad = smell.gradient_at(pos_xyz, SMELL_SAMPLE_EPSILON);
                let pheromone_grad = pheromone.gradient_at(pos_xyz, PHEROMONE_SAMPLE_EPSILON);
                let sensors = bioscape::BrainSensors {
                    nearest_food: best_food,
                    nearest_cell: best_cell,
                    neighbors_in_vision,
                    smell_grad,
                    pheromone_grad,
                };
                cell.apply_shell_absorb(dt);
                bioscape::populate_brain_inputs(cell, &sensors, vision_r)
            })
            .collect();

        // Phase 2: upload inputs + GPU forward (persistent weights).
        let gpu = self.gpu_full.as_ref().expect("gpu_full Some");
        gpu.cells.upload_inputs(&inputs_vec);
        gpu.brain.forward_persistent(&gpu.cells, n);

        // Phase 3: download hidden + outputs.
        let (hiddens, outputs) = gpu.cells.download_hidden_outputs(n);

        // Phase 4: writeback + motor (CPU).
        self.cells
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, cell)| {
                cell.last_inputs = inputs_vec[i];
                cell.last_hidden = hiddens[i];
                cell.last_outputs = outputs[i];
                cell.apply_brain_motor(&outputs[i], dt);
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
                .map(|(i, f)| (i, f.position, ())),
        );

        let cell_grid = &self.cell_grid;
        let food_grid = &self.food_grid;
        let smell = &self.smell;
        let pheromone = &self.pheromone;

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

                let mut best_food: Option<[f32; 3]> = None;
                let mut best_food_d2 = f32::MAX;
                food_grid.for_each_in_radius_toroidal(pos, vision_r, WORLD_HALF, |_id, fp, ()| {
                    let d = bioscape::min_image_delta(pos, fp, WORLD_HALF);
                    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if d2 <= vr2 && d2 < best_food_d2 {
                        best_food_d2 = d2;
                        best_food = Some(d);
                    }
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
                    if d2 <= vr2 {
                        neighbors_in_vision += 1;
                        if d2 < best_cell_d2 {
                            best_cell_d2 = d2;
                            best_cell = Some((d, oradius));
                        }
                    }
                });

                let pos_xyz = [pos[0], pos[1], pos[2]];
                let smell_grad = smell.gradient_at(pos_xyz, SMELL_SAMPLE_EPSILON);
                let pheromone_grad = pheromone.gradient_at(pos_xyz, PHEROMONE_SAMPLE_EPSILON);
                let sensors = bioscape::BrainSensors {
                    nearest_food: best_food,
                    nearest_cell: best_cell,
                    neighbors_in_vision,
                    smell_grad,
                    pheromone_grad,
                };
                cell.apply_shell_absorb(dt);
                bioscape::populate_brain_inputs(cell, &sensors, vision_r)
            })
            .collect();

        // Phase 2: GPU forward batch.
        let mut hiddens = vec![[0.0_f32; BRAIN_HIDDEN]; n];
        let mut outputs = vec![[0.0_f32; BRAIN_OUTPUTS]; n];
        {
            let gpu = self
                .gpu
                .as_mut()
                .expect("brain_act_gpu called without gpu");
            gpu.forward_batch(
                &inputs_vec,
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
                cell.last_inputs = inputs_vec[i];
                cell.last_hidden = hiddens[i];
                cell.last_outputs = outputs[i];
                cell.apply_brain_motor(&outputs[i], dt);
            });
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
                .map(|(i, f)| (i, f.position, ())),
        );

        let cell_grid = &self.cell_grid;
        let food_grid = &self.food_grid;
        let smell = &self.smell;
        let pheromone = &self.pheromone;

        self.cells
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, cell)| {
                let pos = cell.position;
                let vision_r = cell.genome.vision_radius;
                let vr2 = vision_r * vision_r;

                let mut best_food: Option<[f32; 3]> = None;
                let mut best_food_d2 = f32::MAX;
                food_grid.for_each_in_radius_toroidal(pos, vision_r, WORLD_HALF, |_id, fp, ()| {
                    let d = bioscape::min_image_delta(pos, fp, WORLD_HALF);
                    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if d2 <= vr2 && d2 < best_food_d2 {
                        best_food_d2 = d2;
                        best_food = Some(d);
                    }
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
                    if d2 <= vr2 {
                        neighbors_in_vision += 1;
                        if d2 < best_cell_d2 {
                            best_cell_d2 = d2;
                            best_cell = Some((d, oradius));
                        }
                    }
                });

                let pos_xyz = [pos[0], pos[1], pos[2]];
                let smell_grad = smell.gradient_at(pos_xyz, SMELL_SAMPLE_EPSILON);
                let pheromone_grad = pheromone.gradient_at(pos_xyz, PHEROMONE_SAMPLE_EPSILON);
                let sensors = bioscape::BrainSensors {
                    nearest_food: best_food,
                    nearest_cell: best_cell,
                    neighbors_in_vision,
                    smell_grad,
                    pheromone_grad,
                };

                cell.apply_shell_absorb(dt);
                let inputs = bioscape::populate_brain_inputs(cell, &sensors, vision_r);
                let (hidden, outputs) = cell.genome.brain.forward_with_state(&inputs);
                cell.last_inputs = inputs;
                cell.last_hidden = hidden;
                cell.last_outputs = outputs;
                cell.apply_brain_motor(&outputs, dt);
            });
    }

    fn step(&mut self, dt: f32) {
        // Sprint 57: stejně jako apply_morph, ~16 us sekvenčně vs ~30 us
        // paralelně — work per cell je příliš malý pro rayon. Sekvenční win.
        for cell in &mut self.cells {
            cell.step(dt, WORLD_HALF, &PHYSICS_CONFIG);
        }
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
        for cell in &mut self.cells {
            let noise = self
                .map
                .sample([cell.position[0], cell.position[1], cell.position[2]]);
            let drain = hazard_drain(noise) * dt;
            cell.energy -= drain;
            cell.damage_accum += drain;
        }
    }

    fn resolve_collisions(&mut self) {
        // Sprint 43: grid + rayon. Δ pro každé i je write-only do vlastního
        // slotu. Max search radius = CELL_RADIUS × (radius_i + max_neighbor_r);
        // vyhledáme přes effective_radius_i + GRID_CELL_SIZE konzervativně.
        let n = self.cells.len();
        self.cell_grid.rebuild(
            self.cells
                .iter()
                .enumerate()
                .map(|(i, c)| (i, c.position, c.phenotype.effective_radius())),
        );
        self.deltas_scratch.clear();
        self.deltas_scratch.resize(n, [0.0, 0.0, 0.0]);

        let cell_grid = &self.cell_grid;
        let cells = &self.cells;
        // Bound na search radius — 2× max radius v gridu by stačilo, ale
        // GRID_CELL_SIZE je už ~64; používáme effective_radius_i × CELL_RADIUS × 2
        // jako horní odhad (radius_j ≤ radius_i × ratio threshold). Pro jistotu
        // bumpneme na CELL_RADIUS × max_axis × 2.
        self.deltas_scratch
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, delta)| {
                let pos_i = cells[i].position;
                let radius_i = cells[i].phenotype.effective_radius();
                let search_r = CELL_RADIUS * (radius_i + cells[i].phenotype.max_axis() * 2.0);
                cell_grid.for_each_in_radius_toroidal(pos_i, search_r, WORLD_HALF, |id_j, pos_j, radius_j| {
                    if id_j == i {
                        return;
                    }
                    let pair_r = CELL_RADIUS * (radius_i + radius_j);
                    let pair_r2 = pair_r * pair_r;
                    // Sprint 54: min-image delta — direction i→j přes wrap.
                    let d_vec = bioscape::min_image_delta(pos_j, pos_i, WORLD_HALF);
                    let d2 = d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2];
                    if d2 < pair_r2 && d2 > 0.0 {
                        let d = d2.sqrt();
                        let overlap = pair_r - d;
                        delta[0] += (d_vec[0] / d) * overlap * 0.5;
                        delta[1] += (d_vec[1] / d) * overlap * 0.5;
                        delta[2] += (d_vec[2] / d) * overlap * 0.5;
                    }
                });
            });

        for (cell, delta) in self.cells.iter_mut().zip(self.deltas_scratch.iter()) {
            cell.position[0] += delta[0];
            cell.position[1] += delta[1];
            cell.position[2] += delta[2];
        }
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
        let attack_events: Vec<(usize, usize, f32)> = (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let attack_signal = cells[i].last_outputs[6].max(0.0);
                if attack_signal <= ATTACK_THRESHOLD {
                    return Vec::new();
                }
                let pos_i = cells[i].position;
                let radius_a = cells[i].phenotype.effective_radius();
                let spike = cells[i].phenotype.spike_length;
                let heading = cells[i].heading;
                let search_r =
                    CELL_RADIUS * (radius_a + cells[i].phenotype.max_axis() * 2.0);
                let mut local: Vec<(usize, usize, f32)> = Vec::new();
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
                            if spike > 0.0 && d2 > 0.0 {
                                let inv_d = 1.0 / d2.sqrt();
                                let to_j_x = -d_vec[0] * inv_d;
                                let to_j_y = -d_vec[1] * inv_d;
                                let cos_angle = heading.cos() * to_j_x + heading.sin() * to_j_y;
                                if cos_angle >= bioscape::SPIKE_DOT_THRESHOLD {
                                    gain += PREDATION_GAIN_PER_TICK
                                        * spike
                                        * bioscape::SPIKE_PREDATION_BONUS;
                                }
                            }
                            let dilution = 1.0 / (1.0 + DILUTION_K * herd_counts[j] as f32);
                            gain *= dilution;
                            local.push((i, j, gain));
                        }
                    },
                );
                local
            })
            .collect();

        let events: u64 = attack_events.len() as u64;
        for (i, j, gain) in attack_events {
            self.energy_deltas_scratch[i] += gain;
            self.energy_deltas_scratch[j] -= PREDATION_DRAIN_PER_TICK;
            self.damage_deltas_scratch[j] += PREDATION_DRAIN_PER_TICK;
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
                .map(|(i, f)| (i, f.position, ())),
        );
        self.eaten_scratch.clear();
        self.eaten_scratch.resize(self.foods.len(), false);

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
                    |idx, _fp, ()| {
                        if ate.is_some() {
                            return;
                        }
                        let food = &foods[idx];
                        let md = bioscape::min_image_delta(pos, food.position, WORLD_HALF);
                        let ghost = Food {
                            position: [pos[0] + md[0], pos[1] + md[1], food.position[2]],
                            age_ticks: food.age_ticks,
                        };
                        if cell.eat_test(&ghost, EAT_RADIUS) {
                            let value = FOOD_VALUE
                                * food_multiplier(
                                    map.sample([food.position[0], food.position[1], 0.0]),
                                )
                                * food.value_factor();
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
        let mut ate_cell_indices: Vec<usize> = Vec::new();
        for (cell_idx, opt) in candidates.iter().enumerate() {
            if let Some((food_idx, value)) = opt {
                if self.eaten_scratch[*food_idx] {
                    continue;
                }
                self.eaten_scratch[*food_idx] = true;
                let cell = &mut self.cells[cell_idx];
                cell.energy += *value;
                if use_gpu_hebbian {
                    rewards[cell_idx] = 1.0;
                } else {
                    ate_cell_indices.push(cell_idx);
                }
            }
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
        let mut children = Vec::with_capacity(matings.len());
        for &(a, b) in matings {
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
            children.push(bioscape::make_mating_child(cell_a, cell_b, rng));
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
                    new_foods.push(Food { position: pos, age_ticks: 0 });
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

fn write_stats<W: Write>(w: &mut W, world: &World) -> std::io::Result<()> {
    let n = world.cells.len();
    if n == 0 {
        return writeln!(
            w,
            "{},0,0,0,0,0,0,0,0,0,0,0,{},{:.3},0,0,0,0,0,0,0,0,0,0,{},{},{},0,{},0,0,0,0,0",
            world.clock.generation,
            world.foods.len(),
            world.density_factor,
            world.births_gen,
            world.deaths_gen,
            world.fertile_ticks_gen,
            world.predation_events_gen,
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
    let current_gen = world.clock.generation;
    for c in &world.cells {
        let s = c.genome.max_speed as f64;
        let v = c.genome.vision_radius as f64;
        let l = c.phenotype.body_length as f64;
        let wd = c.phenotype.body_width as f64;
        let hg = c.phenotype.body_height as f64;
        let aspect = if wd > 1e-6 { l / wd } else { 0.0 };
        let spk = c.phenotype.spike_length as f64;
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
    writeln!(
        w,
        "{},{},{:.2},{:.3},{:.2},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{:.3},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2},{},{},{},{:.3},{},{:.3},{:.2},{:.3},{:.3},{:.3},{:.3}",
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
    )
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
    let args: Vec<String> = raw_args
        .iter()
        .filter(|a| !a.starts_with("--"))
        .cloned()
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
    let mut world = if let Some(path) = load_path.as_ref() {
        match World::load_checkpoint(Path::new(path)) {
            Ok(w) => {
                eprintln!(
                    "checkpoint: loaded {} (cells={}, foods={}, gen={}, tick={})",
                    path,
                    w.cells.len(),
                    w.foods.len(),
                    w.clock.generation,
                    w.clock.tick,
                );
                w
            }
            Err(e) => {
                eprintln!("checkpoint: load failed ({e}); starting fresh");
                World::new(&mut rng, map_seed, mating_radius, initial_cells, max_population)
            }
        }
    } else {
        World::new(&mut rng, map_seed, mating_radius, initial_cells, max_population)
    };

    #[cfg(feature = "gpu")]
    if want_gpu_full {
        let cap = initial_cells.max(max_population).max(64);
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
            Ok(GpuFullState {
                cells: cells_gpu,
                brain,
                hebbian,
                brownian,
            })
        };
        match init() {
            Ok(state) => {
                eprintln!(
                    "gpu-full: persistent brain weights + GPU Hebbian + GPU Brownian (capacity {})",
                    cap
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
        "gen,cells,spd_avg,spd_dev,vis_avg,vis_dev,len_avg,wid_avg,hgt_avg,asp_avg,asp_dev,spk_avg,spk_max,food,density,lineages,oldest,ph_emit,abs_x,abs_y,edge_frac,corner_frac,mean_x,mean_y,energy_avg,births,deaths,fertile_ticks,atk_emit,predation_events,recurrent_io,nn_dist_avg,density_avg,density_dev,dmg_avg,noise_avg"
    )
    .unwrap();
    write_stats(&mut log, &world).unwrap();

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

    let start = Instant::now();
    while world.clock.generation < max_gens {
        let gen_ended = world.tick(&mut rng);
        if gen_ended.is_some() {
            write_stats(&mut log, &world).unwrap();
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
