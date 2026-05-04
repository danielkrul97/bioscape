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
    PHEROMONE_GRID_RES, PHEROMONE_SAMPLE_EPSILON, PHYSICS_CONFIG, PREDATION_DRAIN_PER_TICK,
    PREDATION_GAIN_PER_TICK, REPRODUCE_THRESHOLD, SIZE_RATIO_THRESHOLD, SMELL_DECAY,
    SMELL_DIFFUSION, SMELL_GRID_RES, SMELL_PER_FOOD, SMELL_SAMPLE_EPSILON, TICKS_PER_GENERATION,
    WORLD_MAP_BASE_RES, WORLD_MAP_FOOD_AMP, WORLD_MAP_FOOD_FLOOR, WORLD_MAP_RES, WORLD_MAP_SEED,
    WORLD_UNITS_PER_FOOD,
};
#[cfg(feature = "gpu")]
use bioscape::{gpu::BrainGpu, BRAIN_HIDDEN, BRAIN_INPUTS, BRAIN_OUTPUTS};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::env;
use std::io::{BufWriter, Write};
use std::time::Instant;

// Headless has no window — fixed extent so seeds reproduce identically across
// machines. Sim parameters live in `bioscape`.
// Sprint 35: aktivovaná z-osa (2 = velmi mírný 3D layer, ~vision diameter
// pokrývá z plně). Větší z způsobuje extinkci pre-evolved random brainů
// (food density per volume drop + random pitch waste). Sprint 37 ladí z + pitch
// range, jakmile selekce naučí cells deliberátně používat 3D. WorldMap a
// SmellField/Pheromone zůstávají 2D (xy projekce); plné volumetric 3D pole
// je odložené.
const WORLD_HALF: [f32; 3] = [960.0, 540.0, 2.0];

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
}

impl World {
    fn new(
        rng: &mut impl Rng,
        map_seed: u64,
        mating_radius: f32,
        initial_cells: usize,
        max_population: usize,
    ) -> Self {
        // Sprint 32: WorldMap a SmellField stále 2D — projekce xy. Sprint 35
        // promění je na 3D volumetric.
        let world_half_xy = [WORLD_HALF[0], WORLD_HALF[1]];
        let map = WorldMap::new(WORLD_MAP_RES, WORLD_MAP_BASE_RES, world_half_xy, map_seed);
        let cells = (0..initial_cells)
            .map(|i| Cell::random(rng, WORLD_HALF, i as u64, 0))
            .collect();
        let target = food_target(1.0);
        // Sprint 31 spatial clustering: rejection sampling i pro initial food.
        // Bez retry budgetu by clustering nikdy nezastavil; MAX_SPAWN_ATTEMPTS
        // garantuje, že každý slot dostane jídlo, jen distribučně bias.
        let foods = (0..target)
            .map(|_| {
                for _ in 0..MAX_SPAWN_ATTEMPTS {
                    let candidate = Food::random(rng, WORLD_HALF);
                    let richness = map.sample([candidate.position[0], candidate.position[1]]);
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
            smell: SmellField::new(SMELL_GRID_RES, world_half_xy),
            pheromone: SmellField::new(PHEROMONE_GRID_RES, world_half_xy),
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
        }
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
        for cell in &mut self.cells {
            cell.apply_morph(dt);
        }
    }

    fn apply_brownian(&mut self, rng: &mut impl Rng, dt: f32) {
        for cell in &mut self.cells {
            cell.apply_brownian(rng, dt, WORLD_HALF[2]);
        }
    }

    fn update_smell(&mut self, dt: f32) {
        for food in &self.foods {
            self.smell
                .add_source([food.position[0], food.position[1]], SMELL_PER_FOOD * dt);
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
                .add_source([cell.position[0], cell.position[1]], rate * dt);
            cell.energy -= PHEROMONE_COST_PER_RATE * brain_emit * dt;
        }
    }

    /// Sprint 44: dispatch CPU vs GPU forward pass podle `self.gpu`.
    fn run_brain_act(&mut self, dt: f32) {
        #[cfg(feature = "gpu")]
        {
            if self.gpu.is_some() {
                self.brain_act_gpu(dt);
                return;
            }
        }
        self.brain_act(dt);
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
                food_grid.for_each_in_radius(pos, vision_r, |_id, fp, ()| {
                    let dx = fp[0] - pos[0];
                    let dy = fp[1] - pos[1];
                    let dz = fp[2] - pos[2];
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 <= vr2 && d2 < best_food_d2 {
                        best_food_d2 = d2;
                        best_food = Some(fp);
                    }
                });

                let mut best_cell: Option<([f32; 3], f32)> = None;
                let mut best_cell_d2 = f32::MAX;
                let mut neighbors_in_vision: u32 = 0;
                cell_grid.for_each_in_radius(pos, vision_r, |id, op, oradius| {
                    if id == i {
                        return;
                    }
                    let dx = op[0] - pos[0];
                    let dy = op[1] - pos[1];
                    let dz = op[2] - pos[2];
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 <= vr2 {
                        neighbors_in_vision += 1;
                        if d2 < best_cell_d2 {
                            best_cell_d2 = d2;
                            best_cell = Some((op, oradius));
                        }
                    }
                });

                let pos_xy = [pos[0], pos[1]];
                let smell_grad = smell.gradient_at(pos_xy, SMELL_SAMPLE_EPSILON);
                let pheromone_grad = pheromone.gradient_at(pos_xy, PHEROMONE_SAMPLE_EPSILON);
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
                food_grid.for_each_in_radius(pos, vision_r, |_id, fp, ()| {
                    let dx = fp[0] - pos[0];
                    let dy = fp[1] - pos[1];
                    let dz = fp[2] - pos[2];
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 <= vr2 && d2 < best_food_d2 {
                        best_food_d2 = d2;
                        best_food = Some(fp);
                    }
                });

                let mut best_cell: Option<([f32; 3], f32)> = None;
                let mut best_cell_d2 = f32::MAX;
                let mut neighbors_in_vision: u32 = 0;
                cell_grid.for_each_in_radius(pos, vision_r, |id, op, oradius| {
                    if id == i {
                        return;
                    }
                    let dx = op[0] - pos[0];
                    let dy = op[1] - pos[1];
                    let dz = op[2] - pos[2];
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 <= vr2 {
                        neighbors_in_vision += 1;
                        if d2 < best_cell_d2 {
                            best_cell_d2 = d2;
                            best_cell = Some((op, oradius));
                        }
                    }
                });

                let pos_xy = [pos[0], pos[1]];
                let smell_grad = smell.gradient_at(pos_xy, SMELL_SAMPLE_EPSILON);
                let pheromone_grad = pheromone.gradient_at(pos_xy, PHEROMONE_SAMPLE_EPSILON);
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
        for cell in &mut self.cells {
            let noise = self.map.sample([cell.position[0], cell.position[1]]);
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
                cell_grid.for_each_in_radius(pos_i, search_r, |id_j, pos_j, radius_j| {
                    if id_j == i {
                        return;
                    }
                    let pair_r = CELL_RADIUS * (radius_i + radius_j);
                    let pair_r2 = pair_r * pair_r;
                    let dx = pos_i[0] - pos_j[0];
                    let dy = pos_i[1] - pos_j[1];
                    let dz = pos_i[2] - pos_j[2];
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 < pair_r2 && d2 > 0.0 {
                        let d = d2.sqrt();
                        let overlap = pair_r - d;
                        delta[0] += (dx / d) * overlap * 0.5;
                        delta[1] += (dy / d) * overlap * 0.5;
                        delta[2] += (dz / d) * overlap * 0.5;
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
                cell_grid.for_each_in_radius(pos_i, HERD_RADIUS, |id_j, pos_j, _| {
                    if id_j == i {
                        return;
                    }
                    let dx = pos_i[0] - pos_j[0];
                    let dy = pos_i[1] - pos_j[1];
                    let dz = pos_i[2] - pos_j[2];
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 < herd_r2 {
                        count += 1;
                    }
                });
                count
            })
            .collect();

        let mut events: u64 = 0;
        for i in 0..n {
            // Sprint 27: attack je opt-in přes brain. Bez aktivního signálu jsou
            // kontakty s menšími cells jen kolize (řešené v resolve_collisions).
            let attack_signal = self.cells[i].last_outputs[6].max(0.0);
            if attack_signal <= ATTACK_THRESHOLD {
                continue;
            }
            let pos_i = self.cells[i].position;
            let radius_a = self.cells[i].phenotype.effective_radius();
            let spike = self.cells[i].phenotype.spike_length;
            let heading = self.cells[i].heading;
            // Search radius pro pair_r2 = CELL_RADIUS × (r_a + r_b). Bound na
            // r_b ≤ MAX_BODY axis-y → konzervativně CELL_RADIUS × (r_a + max_axis).
            let search_r = CELL_RADIUS * (radius_a + self.cells[i].phenotype.max_axis() * 2.0);
            let mut victims: Vec<(usize, f32, f32, f32, f32)> = Vec::new();
            self.cell_grid
                .for_each_in_radius(pos_i, search_r, |j, pos_j, radius_b| {
                    if j == i {
                        return;
                    }
                    if radius_a < SIZE_RATIO_THRESHOLD * radius_b {
                        return;
                    }
                    let pair_r = CELL_RADIUS * (radius_a + radius_b);
                    let pair_r2 = pair_r * pair_r;
                    let dx = pos_i[0] - pos_j[0];
                    let dy = pos_i[1] - pos_j[1];
                    let dz = pos_i[2] - pos_j[2];
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 < pair_r2 {
                        victims.push((j, dx, dy, dz, d2));
                    }
                });
            for (j, dx, dy, dz, d2) in victims {
                let mut gain = PREDATION_GAIN_PER_TICK;
                if spike > 0.0 && d2 > 0.0 {
                    let inv_d = 1.0 / d2.sqrt();
                    let to_j_x = -dx * inv_d;
                    let to_j_y = -dy * inv_d;
                    let _ = dz;
                    let cos_angle = heading.cos() * to_j_x + heading.sin() * to_j_y;
                    if cos_angle >= bioscape::SPIKE_DOT_THRESHOLD {
                        gain +=
                            PREDATION_GAIN_PER_TICK * spike * bioscape::SPIKE_PREDATION_BONUS;
                    }
                }
                let dilution = 1.0 / (1.0 + DILUTION_K * herd_counts[j] as f32);
                gain *= dilution;
                self.energy_deltas_scratch[i] += gain;
                self.energy_deltas_scratch[j] -= PREDATION_DRAIN_PER_TICK;
                self.damage_deltas_scratch[j] += PREDATION_DRAIN_PER_TICK;
                events += 1;
            }
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
        // Sprint 43: food_grid lookup místo full sweep. Sekvenční, protože
        // despawn `Vec<Food>` je shared mutable + Hebbian update mutuje cell.
        // Per-cell vidi jen kandidáty z 3³ buckets v `EAT_RADIUS × max_axis` —
        // při typickém eat reach ~8 a GRID_CELL_SIZE=64 query overestimate je
        // ~1 bucket → ~10 kandidátů max.
        self.food_grid.rebuild(
            self.foods
                .iter()
                .enumerate()
                .map(|(i, f)| (i, f.position, ())),
        );
        self.eaten_scratch.clear();
        self.eaten_scratch.resize(self.foods.len(), false);

        for cell in self.cells.iter_mut() {
            let pos = cell.position;
            let search_r = EAT_RADIUS * cell.phenotype.max_axis();
            let mut ate_idx: Option<usize> = None;
            self.food_grid
                .for_each_in_radius(pos, search_r, |idx, _fp, ()| {
                    if ate_idx.is_some() || self.eaten_scratch[idx] {
                        return;
                    }
                    let food = &self.foods[idx];
                    let value = FOOD_VALUE
                        * food_multiplier(
                            self.map.sample([food.position[0], food.position[1]]),
                        )
                        * food.value_factor();
                    if cell.try_eat(food, EAT_RADIUS, value) {
                        ate_idx = Some(idx);
                    }
                });
            if let Some(idx) = ate_idx {
                self.eaten_scratch[idx] = true;
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
        for j in (0..self.eaten_scratch.len()).rev() {
            if self.eaten_scratch[j] {
                self.foods.swap_remove(j);
            }
        }
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
                    .sample([candidate.position[0], candidate.position[1]]);
                if reject_food_for_richness(rng, richness) {
                    continue;
                }
                let mut blocked = false;
                self.cell_grid
                    .for_each_in_radius(candidate.position, max_search_r, |id, cell_pos, _r| {
                        if blocked {
                            return;
                        }
                        let exclusion = EAT_RADIUS * self.cells[id].phenotype.max_axis();
                        let dx = candidate.position[0] - cell_pos[0];
                        let dy = candidate.position[1] - cell_pos[1];
                        let dz = candidate.position[2] - cell_pos[2];
                        if dx * dx + dy * dy + dz * dz < exclusion * exclusion {
                            blocked = true;
                        }
                    });
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
        let matings = bioscape::pair_fertile(&fertile, mating_r2, budget);
        let to_spawn = self.spawn_children_from_matings(&matings, rng);
        self.births_gen += to_spawn.len() as u64;
        self.cells.extend(to_spawn);
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
    fn spawn_children_from_matings(
        &mut self,
        matings: &[(usize, usize)],
        rng: &mut impl Rng,
    ) -> Vec<Cell> {
        let mut children = Vec::with_capacity(matings.len());
        for &(a, b) in matings {
            // Split-borrow: pull `hi`-indexed cell from right slice, `lo` z levé.
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
            // Sprint 42: refractory period po mating.
            cell_a.reproduce_cooldown_ticks = MATING_COOLDOWN_TICKS;
            cell_b.reproduce_cooldown_ticks = MATING_COOLDOWN_TICKS;
            children.push(bioscape::make_mating_child(cell_a, cell_b, rng));
        }
        children
    }

    fn die_and_drop_carrion(&mut self, rng: &mut impl Rng) {
        let half = WORLD_HALF;
        let mut new_foods: Vec<Food> = Vec::new();
        let before = self.cells.len();
        for cell in &self.cells {
            if cell.energy <= 0.0 {
                for _ in 0..CARRION_FOOD_COUNT {
                    // Sprint 32: z-osa carrion = z mrtvé buňky (Sprint 32 vždy 0).
                    // Sprint 33+ s aktivním z motion bude carrion v mid-water.
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
        self.cells.retain(|c| c.energy > 0.0);
        self.deaths_gen += (before - self.cells.len()) as u64;
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
    let area = (2.0 * WORLD_HALF[0]) * (2.0 * WORLD_HALF[1]);
    ((area / WORLD_UNITS_PER_FOOD) * factor.max(0.0)) as usize
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
        noise_sum += world.map.sample([c.position[0], c.position[1]]) as f64;
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
                grid.for_each_in_radius(pi, search_r, |j, pj, _| {
                    if i == j {
                        return;
                    }
                    let dx = pi[0] - pj[0];
                    let dy = pi[1] - pj[1];
                    let d2 = dx * dx + dy * dy;
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
    let want_gpu = raw_args.iter().any(|a| a == "--gpu");
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
    let mut world = World::new(&mut rng, map_seed, mating_radius, initial_cells, max_population);

    #[cfg(feature = "gpu")]
    if want_gpu {
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
        eprintln!("gpu: --gpu requested but binary built without --features gpu");
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
