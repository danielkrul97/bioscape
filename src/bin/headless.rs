//! Headless harness — pure simulation loop, no Bevy renderer.
//!
//! Usage: `cargo run --release --bin headless -- [seed] [max_gens] [out_path]`
//! Defaults: seed=0, max_gens=500, out_path=run_seed{seed}.csv
//!
//! Logs per-generation stats (cell_count, mean/dev for max_speed/vision/body_size,
//! food count, density factor) to CSV. Reproducible: same seed → identical run.

use bioscape::{
    Cell, Food, Genome, SimClock, SmellField, WorldMap, BRAIN_HIDDEN, BRAIN_INPUTS, BRAIN_OUTPUTS,
    CARRION_FOOD_COUNT, CELL_RADIUS, CYCLE_AMPLITUDE, CYCLE_GEN_PERIOD, DRAG_COEFFICIENT,
    EAT_RADIUS, FIXED_TIMESTEP_HZ, FOOD_SPAWN_RATE, FOOD_VALUE, GENERATIONS_PER_EPOCH,
    INITIAL_CELLS, LEARNING_RATE, MATING_RADIUS, MAX_POPULATION, MAX_SPAWN_ATTEMPTS,
    MUTATION_CONFIG, PHYSICS_CONFIG, PREDATION_DRAIN_PER_TICK, PREDATION_GAIN_PER_TICK,
    REPRODUCE_THRESHOLD, SIZE_RATIO_THRESHOLD, SMELL_DECAY, SMELL_DIFFUSION, SMELL_GRID_RES,
    SMELL_NORMALIZATION_GAIN, SMELL_PER_FOOD, SMELL_SAMPLE_EPSILON, TICKS_PER_GENERATION,
    WORLD_MAP_BASE_RES, WORLD_MAP_FOOD_AMP, WORLD_MAP_FOOD_FLOOR, WORLD_MAP_RES, WORLD_MAP_SEED,
    WORLD_UNITS_PER_FOOD,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::io::{BufWriter, Write};
use std::time::Instant;

// Headless has no window — fixed extent (Full-HD-equivalent) so seeds
// reproduce identically across machines. Sim parameters live in `bioscape`.
const WORLD_HALF: [f32; 2] = [960.0, 540.0];

struct World {
    cells: Vec<Cell>,
    foods: Vec<Food>,
    clock: SimClock,
    density_factor: f32,
    smell: SmellField,
    map: WorldMap,
    // Persistent scratch — sized like cells/foods, reused per tick to avoid
    // hot-loop allocations.
    positions_scratch: Vec<[f32; 2]>,
    body_sizes_scratch: Vec<f32>,
    food_positions_scratch: Vec<[f32; 2]>,
    deltas_scratch: Vec<[f32; 2]>,
    energy_deltas_scratch: Vec<f32>,
    eaten_scratch: Vec<bool>,
}

impl World {
    fn new(rng: &mut impl Rng) -> Self {
        let map = WorldMap::new(WORLD_MAP_RES, WORLD_MAP_BASE_RES, WORLD_HALF, WORLD_MAP_SEED);
        let cells = (0..INITIAL_CELLS)
            .map(|i| Cell::random(rng, WORLD_HALF, i as u64, 0))
            .collect();
        let target = food_target(1.0);
        let foods = (0..target).map(|_| Food::random(rng, WORLD_HALF)).collect();
        Self {
            cells,
            foods,
            clock: SimClock::new(TICKS_PER_GENERATION, GENERATIONS_PER_EPOCH),
            density_factor: 1.0,
            smell: SmellField::new(SMELL_GRID_RES, WORLD_HALF),
            map,
            positions_scratch: Vec::new(),
            body_sizes_scratch: Vec::new(),
            food_positions_scratch: Vec::new(),
            deltas_scratch: Vec::new(),
            energy_deltas_scratch: Vec::new(),
            eaten_scratch: Vec::new(),
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

        self.update_smell(dt);
        self.brain_act(dt);
        self.step(dt);
        self.resolve_collisions();
        self.predate();
        self.eat_food();
        self.spawn_food(rng);
        self.reproduce(rng);
        self.die_and_drop_carrion(rng);

        transitions.generation_ended
    }

    fn update_smell(&mut self, dt: f32) {
        for food in &self.foods {
            self.smell.add_source(food.position, SMELL_PER_FOOD * dt);
        }
        self.smell.step(SMELL_DIFFUSION, SMELL_DECAY, dt);
    }

    fn brain_act(&mut self, dt: f32) {
        self.positions_scratch.clear();
        self.positions_scratch
            .extend(self.cells.iter().map(|c| c.position));
        self.body_sizes_scratch.clear();
        self.body_sizes_scratch
            .extend(self.cells.iter().map(|c| c.genome.body_size));
        self.food_positions_scratch.clear();
        self.food_positions_scratch
            .extend(self.foods.iter().map(|f| f.position));
        let positions = &self.positions_scratch;
        let body_sizes = &self.body_sizes_scratch;
        let food_positions = &self.food_positions_scratch;

        for i in 0..self.cells.len() {
            let pos = self.cells[i].position;
            let vision_r = self.cells[i].genome.vision_radius;
            let vr2 = vision_r * vision_r;

            let mut best_food: Option<[f32; 2]> = None;
            let mut best_food_d2 = f32::MAX;
            for &fp in food_positions {
                let dx = fp[0] - pos[0];
                let dy = fp[1] - pos[1];
                let d2 = dx * dx + dy * dy;
                if d2 <= vr2 && d2 < best_food_d2 {
                    best_food_d2 = d2;
                    best_food = Some(fp);
                }
            }

            let mut best_cell: Option<([f32; 2], f32)> = None;
            let mut best_cell_d2 = f32::MAX;
            for j in 0..self.cells.len() {
                if j == i {
                    continue;
                }
                let other_pos = positions[j];
                let dx = other_pos[0] - pos[0];
                let dy = other_pos[1] - pos[1];
                let d2 = dx * dx + dy * dy;
                if d2 <= vr2 && d2 < best_cell_d2 {
                    best_cell_d2 = d2;
                    best_cell = Some((other_pos, body_sizes[j]));
                }
            }

            let cell = &mut self.cells[i];
            let max_speed = cell.genome.max_speed;
            let my_size = cell.genome.body_size.max(0.01);
            let speed_norm =
                (cell.velocity[0].hypot(cell.velocity[1]) / max_speed).clamp(0.0, 1.0);
            let energy_norm = (cell.energy / REPRODUCE_THRESHOLD).clamp(0.0, 1.5);

            let mut inputs = [0.0_f32; BRAIN_INPUTS];
            if let Some(target) = best_food {
                inputs[0] = (target[0] - pos[0]) / vision_r;
                inputs[1] = (target[1] - pos[1]) / vision_r;
            }
            if let Some((other, other_size)) = best_cell {
                inputs[2] = (other[0] - pos[0]) / vision_r;
                inputs[3] = (other[1] - pos[1]) / vision_r;
                inputs[6] = (other_size - my_size) / my_size;
            }
            inputs[4] = energy_norm;
            inputs[5] = speed_norm;
            let grad = self.smell.gradient_at(pos, SMELL_SAMPLE_EPSILON);
            inputs[7] = (grad[0] * SMELL_NORMALIZATION_GAIN).tanh();
            inputs[8] = (grad[1] * SMELL_NORMALIZATION_GAIN).tanh();

            let (hidden, outputs) = cell.genome.brain.forward_with_state(&inputs);
            cell.last_inputs = inputs;
            cell.last_hidden = hidden;
            cell.last_outputs = outputs;
            let turn_signal = outputs[0];
            let thrust_norm = (outputs[1] + 1.0) * 0.5;

            let body_size = cell.genome.body_size.max(0.01);
            let turn_rate = cell.genome.turn_rate;
            let ang_acc = turn_signal * turn_rate / body_size;
            cell.angular_velocity += ang_acc * dt;

            let a_max = DRAG_COEFFICIENT * max_speed * max_speed / body_size;
            let a = thrust_norm * a_max;
            let heading = cell.heading;
            cell.velocity[0] += a * heading.cos() * dt;
            cell.velocity[1] += a * heading.sin() * dt;
        }
    }

    fn step(&mut self, dt: f32) {
        for cell in &mut self.cells {
            cell.step(dt, WORLD_HALF, &PHYSICS_CONFIG);
        }
    }

    fn resolve_collisions(&mut self) {
        let n = self.cells.len();
        self.positions_scratch.clear();
        self.positions_scratch
            .extend(self.cells.iter().map(|c| c.position));
        self.body_sizes_scratch.clear();
        self.body_sizes_scratch
            .extend(self.cells.iter().map(|c| c.genome.body_size));
        self.deltas_scratch.clear();
        self.deltas_scratch.resize(n, [0.0, 0.0]);
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let pair_r =
                    CELL_RADIUS * (self.body_sizes_scratch[i] + self.body_sizes_scratch[j]);
                let pair_r2 = pair_r * pair_r;
                let dx = self.positions_scratch[i][0] - self.positions_scratch[j][0];
                let dy = self.positions_scratch[i][1] - self.positions_scratch[j][1];
                let d2 = dx * dx + dy * dy;
                if d2 < pair_r2 && d2 > 0.0 {
                    let d = d2.sqrt();
                    let overlap = pair_r - d;
                    self.deltas_scratch[i][0] += (dx / d) * overlap * 0.5;
                    self.deltas_scratch[i][1] += (dy / d) * overlap * 0.5;
                }
            }
        }
        for (cell, delta) in self.cells.iter_mut().zip(self.deltas_scratch.iter()) {
            cell.position[0] += delta[0];
            cell.position[1] += delta[1];
        }
    }

    fn predate(&mut self) {
        let n = self.cells.len();
        self.positions_scratch.clear();
        self.positions_scratch
            .extend(self.cells.iter().map(|c| c.position));
        self.body_sizes_scratch.clear();
        self.body_sizes_scratch
            .extend(self.cells.iter().map(|c| c.genome.body_size));
        self.energy_deltas_scratch.clear();
        self.energy_deltas_scratch.resize(n, 0.0);
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let size_a = self.body_sizes_scratch[i];
                let size_b = self.body_sizes_scratch[j];
                if size_a < SIZE_RATIO_THRESHOLD * size_b {
                    continue;
                }
                let pair_r = CELL_RADIUS * (size_a + size_b);
                let pair_r2 = pair_r * pair_r;
                let dx = self.positions_scratch[i][0] - self.positions_scratch[j][0];
                let dy = self.positions_scratch[i][1] - self.positions_scratch[j][1];
                let d2 = dx * dx + dy * dy;
                if d2 < pair_r2 {
                    self.energy_deltas_scratch[i] += PREDATION_GAIN_PER_TICK;
                    self.energy_deltas_scratch[j] -= PREDATION_DRAIN_PER_TICK;
                }
            }
        }
        for (cell, delta) in self.cells.iter_mut().zip(self.energy_deltas_scratch.iter()) {
            cell.energy += delta;
        }
    }

    fn eat_food(&mut self) {
        self.eaten_scratch.clear();
        self.eaten_scratch.resize(self.foods.len(), false);
        let eaten = &mut self.eaten_scratch;
        for cell in &mut self.cells {
            let pos = cell.position;
            let eat_r = EAT_RADIUS * cell.genome.body_size;
            let r2 = eat_r * eat_r;
            let mut ate = false;
            for (flag, food) in eaten.iter_mut().zip(self.foods.iter()) {
                if *flag {
                    continue;
                }
                let dx = pos[0] - food.position[0];
                let dy = pos[1] - food.position[1];
                if dx * dx + dy * dy <= r2 {
                    cell.energy += FOOD_VALUE * food_multiplier(self.map.sample(food.position));
                    *flag = true;
                    ate = true;
                    break;
                }
            }
            if ate {
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
        for j in (0..eaten.len()).rev() {
            if eaten[j] {
                self.foods.swap_remove(j);
            }
        }
    }

    fn spawn_food(&mut self, rng: &mut impl Rng) {
        let target = food_target(self.density_factor);
        if self.foods.len() >= target {
            return;
        }
        let to_spawn = (target - self.foods.len()).min(FOOD_SPAWN_RATE);
        'spawn: for _ in 0..to_spawn {
            for _ in 0..MAX_SPAWN_ATTEMPTS {
                let candidate = Food::random(rng, WORLD_HALF);
                let mut blocked = false;
                for cell in &self.cells {
                    let exclusion = EAT_RADIUS * cell.genome.body_size;
                    let dx = candidate.position[0] - cell.position[0];
                    let dy = candidate.position[1] - cell.position[1];
                    if dx * dx + dy * dy < exclusion * exclusion {
                        blocked = true;
                        break;
                    }
                }
                if !blocked {
                    self.foods.push(candidate);
                    continue 'spawn;
                }
            }
        }
    }

    fn reproduce(&mut self, rng: &mut impl Rng) {
        let current_pop = self.cells.len();
        if current_pop >= MAX_POPULATION {
            return;
        }
        let budget = MAX_POPULATION - current_pop;

        // Snapshot fertile (idx, position) — pair indices, not entities.
        let fertile: Vec<(usize, [f32; 2])> = self
            .cells
            .iter()
            .enumerate()
            .filter(|(_, c)| c.energy >= REPRODUCE_THRESHOLD)
            .map(|(i, c)| (i, c.position))
            .collect();

        // Greedy O(N²) pairing on fertile indices.
        let mut paired: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut matings: Vec<(usize, usize)> = Vec::new();
        let mating_r2 = MATING_RADIUS * MATING_RADIUS;
        for i_outer in 0..fertile.len() {
            if matings.len() >= budget {
                break;
            }
            let (a, pos_a) = fertile[i_outer];
            if paired.contains(&a) {
                continue;
            }
            let mut best: Option<(usize, f32)> = None;
            for (j_outer, &(b, pos_b)) in fertile.iter().enumerate() {
                if i_outer == j_outer {
                    continue;
                }
                if paired.contains(&b) {
                    continue;
                }
                let dx = pos_a[0] - pos_b[0];
                let dy = pos_a[1] - pos_b[1];
                let d2 = dx * dx + dy * dy;
                if d2 <= mating_r2 && best.is_none_or(|(_, bd2)| d2 < bd2) {
                    best = Some((b, d2));
                }
            }
            if let Some((b, _)) = best {
                paired.insert(a);
                paired.insert(b);
                matings.push((a, b));
            }
        }

        let mut to_spawn: Vec<Cell> = Vec::new();
        for (a, b) in matings {
            // Borrow both cells mutably via split: one of them gets pulled out
            // first, then the other from the remaining slice.
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            let (left, right) = self.cells.split_at_mut(hi);
            let cell_lo = &mut left[lo];
            let cell_hi = &mut right[0];
            let (cell_a, cell_b) = if a < b {
                (cell_lo, cell_hi)
            } else {
                (cell_hi, cell_lo)
            };

            let energy_a = cell_a.energy * 0.5;
            let energy_b = cell_b.energy * 0.5;
            cell_a.energy *= 0.5;
            cell_b.energy *= 0.5;

            let child_genome = Genome::crossover(&cell_a.genome, &cell_b.genome, rng)
                .mutate(rng, &MUTATION_CONFIG);

            let direction = rng.random_range(0.0..std::f32::consts::TAU);
            let mid_pos = [
                (cell_a.position[0] + cell_b.position[0]) * 0.5,
                (cell_a.position[1] + cell_b.position[1]) * 0.5,
            ];
            to_spawn.push(Cell {
                position: mid_pos,
                velocity: [
                    direction.cos() * child_genome.max_speed,
                    direction.sin() * child_genome.max_speed,
                ],
                angular_velocity: 0.0,
                energy: energy_a + energy_b,
                heading: direction,
                lineage_id: cell_a.lineage_id,
                lineage_birth_gen: cell_a.lineage_birth_gen,
                last_inputs: [0.0; BRAIN_INPUTS],
                last_hidden: [0.0; BRAIN_HIDDEN],
                last_outputs: [0.0; BRAIN_OUTPUTS],
                genome: child_genome,
            });
        }
        self.cells.extend(to_spawn);
    }

    fn die_and_drop_carrion(&mut self, rng: &mut impl Rng) {
        let half = WORLD_HALF;
        let mut new_foods: Vec<Food> = Vec::new();
        for cell in &self.cells {
            if cell.energy <= 0.0 {
                for _ in 0..CARRION_FOOD_COUNT {
                    let pos = [
                        (cell.position[0] + rng.random_range(-CELL_RADIUS..CELL_RADIUS))
                            .clamp(-half[0], half[0]),
                        (cell.position[1] + rng.random_range(-CELL_RADIUS..CELL_RADIUS))
                            .clamp(-half[1], half[1]),
                    ];
                    new_foods.push(Food { position: pos });
                }
            }
        }
        self.cells.retain(|c| c.energy > 0.0);
        self.foods.extend(new_foods);
    }
}

fn food_multiplier(noise: f32) -> f32 {
    WORLD_MAP_FOOD_FLOOR + WORLD_MAP_FOOD_AMP * noise
}

fn food_target(factor: f32) -> usize {
    let area = (2.0 * WORLD_HALF[0]) * (2.0 * WORLD_HALF[1]);
    ((area / WORLD_UNITS_PER_FOOD) * factor.max(0.0)) as usize
}

fn write_stats<W: Write>(w: &mut W, world: &World) -> std::io::Result<()> {
    let n = world.cells.len();
    if n == 0 {
        return writeln!(
            w,
            "{},0,0,0,0,0,0,0,{},{:.3},0,0",
            world.clock.generation,
            world.foods.len(),
            world.density_factor
        );
    }
    let mut spd_sum = 0.0_f64;
    let mut spd_sumsq = 0.0_f64;
    let mut vis_sum = 0.0_f64;
    let mut vis_sumsq = 0.0_f64;
    let mut size_sum = 0.0_f64;
    let mut size_sumsq = 0.0_f64;
    let mut lineages: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut oldest_age: u64 = 0;
    let current_gen = world.clock.generation;
    for c in &world.cells {
        let s = c.genome.max_speed as f64;
        let v = c.genome.vision_radius as f64;
        let bs = c.genome.body_size as f64;
        spd_sum += s;
        spd_sumsq += s * s;
        vis_sum += v;
        vis_sumsq += v * v;
        size_sum += bs;
        size_sumsq += bs * bs;
        lineages.insert(c.lineage_id);
        let age = current_gen.saturating_sub(c.lineage_birth_gen);
        if age > oldest_age {
            oldest_age = age;
        }
    }
    let nf = n as f64;
    let spd_m = spd_sum / nf;
    let vis_m = vis_sum / nf;
    let size_m = size_sum / nf;
    let spd_d = ((spd_sumsq / nf) - spd_m * spd_m).max(0.0).sqrt();
    let vis_d = ((vis_sumsq / nf) - vis_m * vis_m).max(0.0).sqrt();
    let size_d = ((size_sumsq / nf) - size_m * size_m).max(0.0).sqrt();
    writeln!(
        w,
        "{},{},{:.2},{:.3},{:.2},{:.3},{:.3},{:.4},{},{:.3},{},{}",
        world.clock.generation,
        n,
        spd_m,
        spd_d,
        vis_m,
        vis_d,
        size_m,
        size_d,
        world.foods.len(),
        world.density_factor,
        lineages.len(),
        oldest_age,
    )
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let max_gens: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(500);
    let out_path = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| format!("run_seed{}.csv", seed));

    let mut rng = StdRng::seed_from_u64(seed);
    let mut world = World::new(&mut rng);

    let file = std::fs::File::create(&out_path).expect("can't create output file");
    let mut log = BufWriter::new(file);
    writeln!(
        log,
        "gen,cells,spd_avg,spd_dev,vis_avg,vis_dev,size_avg,size_dev,food,density,lineages,oldest"
    )
    .unwrap();
    write_stats(&mut log, &world).unwrap();

    eprintln!(
        "headless: seed={} max_gens={} out={} initial_cells={} initial_food={}",
        seed,
        max_gens,
        out_path,
        world.cells.len(),
        world.foods.len()
    );

    let start = Instant::now();
    while world.clock.generation < max_gens {
        let gen_ended = world.tick(&mut rng);
        if gen_ended.is_some() {
            write_stats(&mut log, &world).unwrap();
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
