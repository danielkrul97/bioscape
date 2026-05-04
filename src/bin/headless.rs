//! Headless harness — pure simulation loop, no Bevy renderer.
//!
//! Usage: `cargo run --release --bin headless -- [seed] [max_gens] [out_path]`
//! Defaults: seed=0, max_gens=500, out_path=run_seed{seed}.csv
//!
//! Logs per-generation stats (cell_count, mean/dev for max_speed/vision/body_size,
//! food count, density factor) to CSV. Reproducible: same seed → identical run.

use bioscape::{
    Cell, Food, Genome, Phenotype, SimClock, SmellField, WorldMap, ATTACK_THRESHOLD,
    BRAIN_HIDDEN, BRAIN_INPUTS, BRAIN_INPUTS_SENSORY, BRAIN_OUTPUTS,
    BRAIN_RECURRENT, CARRION_FOOD_COUNT, CELL_RADIUS,
    CYCLE_AMPLITUDE, CYCLE_GEN_PERIOD, DAMAGE_NORMALIZATION_GAIN,
    DENSITY_NORM_COUNT, DILUTION_K, DRAG_COEFFICIENT,
    reject_food_for_richness,
    EAT_RADIUS, FIXED_TIMESTEP_HZ, FOOD_SPAWN_RATE, FOOD_VALUE, GENERATIONS_PER_EPOCH, HAZARD_AMP,
    HAZARD_DRAIN_PER_SEC, HAZARD_FLOOR, HERD_RADIUS, INITIAL_CELLS, LEARNING_RATE,
    MATING_PHEROMONE_THRESHOLD, MATING_RADIUS,
    MAX_POPULATION, MAX_SPAWN_ATTEMPTS, MUTATION_CONFIG, PHEROMONE_BASELINE_EMIT,
    PHEROMONE_BRAIN_MOD, PHEROMONE_COST_PER_RATE, PHEROMONE_DECAY, PHEROMONE_DIFFUSION,
    PHEROMONE_GRID_RES, PHEROMONE_NORMALIZATION_GAIN, PHEROMONE_SAMPLE_EPSILON, PHYSICS_CONFIG,
    PREDATION_DRAIN_PER_TICK, PREDATION_GAIN_PER_TICK, REPRODUCE_THRESHOLD, SIZE_RATIO_THRESHOLD,
    SMELL_DECAY, SMELL_DIFFUSION, SMELL_GRID_RES, SMELL_NORMALIZATION_GAIN, SMELL_PER_FOOD,
    SMELL_SAMPLE_EPSILON, TICKS_PER_GENERATION, WORLD_MAP_BASE_RES, WORLD_MAP_FOOD_AMP,
    WORLD_MAP_FOOD_FLOOR, WORLD_MAP_RES, WORLD_MAP_SEED, WORLD_UNITS_PER_FOOD,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
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

struct World {
    cells: Vec<Cell>,
    foods: Vec<Food>,
    clock: SimClock,
    density_factor: f32,
    smell: SmellField,
    pheromone: SmellField,
    map: WorldMap,
    // Persistent scratch — sized like cells/foods, reused per tick to avoid
    // hot-loop allocations.
    positions_scratch: Vec<[f32; 3]>,
    radii_scratch: Vec<f32>,
    spike_lengths_scratch: Vec<f32>,
    headings_scratch: Vec<f32>,
    food_positions_scratch: Vec<[f32; 3]>,
    deltas_scratch: Vec<[f32; 3]>,
    energy_deltas_scratch: Vec<f32>,
    damage_deltas_scratch: Vec<f32>,
    eaten_scratch: Vec<bool>,
    births_gen: u64,
    deaths_gen: u64,
    fertile_ticks_gen: u64,
    predation_events_gen: u64,
    mating_radius: f32,
}

impl World {
    fn new(rng: &mut impl Rng, map_seed: u64, mating_radius: f32) -> Self {
        // Sprint 32: WorldMap a SmellField stále 2D — projekce xy. Sprint 35
        // promění je na 3D volumetric.
        let world_half_xy = [WORLD_HALF[0], WORLD_HALF[1]];
        let map = WorldMap::new(WORLD_MAP_RES, WORLD_MAP_BASE_RES, world_half_xy, map_seed);
        let cells = (0..INITIAL_CELLS)
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
            positions_scratch: Vec::new(),
            radii_scratch: Vec::new(),
            spike_lengths_scratch: Vec::new(),
            headings_scratch: Vec::new(),
            food_positions_scratch: Vec::new(),
            deltas_scratch: Vec::new(),
            energy_deltas_scratch: Vec::new(),
            damage_deltas_scratch: Vec::new(),
            eaten_scratch: Vec::new(),
            births_gen: 0,
            deaths_gen: 0,
            fertile_ticks_gen: 0,
            predation_events_gen: 0,
            mating_radius,
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
        self.update_pheromone(dt);
        self.brain_act(dt);
        self.emit_pheromones(dt);
        self.apply_morph(dt);
        self.step(dt);
        self.apply_food_gravity(dt);
        self.apply_hazards(dt);
        self.resolve_collisions();
        self.predate();
        self.eat_food();
        self.spawn_food(rng);
        self.reproduce(rng);
        self.die_and_drop_carrion(rng);

        transitions.generation_ended
    }

    fn apply_morph(&mut self, dt: f32) {
        for cell in &mut self.cells {
            cell.apply_morph(dt);
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

    fn brain_act(&mut self, dt: f32) {
        self.positions_scratch.clear();
        self.positions_scratch
            .extend(self.cells.iter().map(|c| c.position));
        self.radii_scratch.clear();
        self.radii_scratch
            .extend(self.cells.iter().map(|c| c.phenotype.effective_radius()));
        self.food_positions_scratch.clear();
        self.food_positions_scratch
            .extend(self.foods.iter().map(|f| f.position));
        let positions = &self.positions_scratch;
        let radii = &self.radii_scratch;
        let food_positions = &self.food_positions_scratch;

        for i in 0..self.cells.len() {
            let pos = self.cells[i].position;
            let vision_r = self.cells[i].genome.vision_radius;
            let vr2 = vision_r * vision_r;

            let mut best_food: Option<[f32; 3]> = None;
            let mut best_food_d2 = f32::MAX;
            for &fp in food_positions {
                let dx = fp[0] - pos[0];
                let dy = fp[1] - pos[1];
                let dz = fp[2] - pos[2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 <= vr2 && d2 < best_food_d2 {
                    best_food_d2 = d2;
                    best_food = Some(fp);
                }
            }

            let mut best_cell: Option<([f32; 3], f32)> = None;
            let mut best_cell_d2 = f32::MAX;
            let mut neighbors_in_vision: u32 = 0;
            for j in 0..self.cells.len() {
                if j == i {
                    continue;
                }
                let other_pos = positions[j];
                let dx = other_pos[0] - pos[0];
                let dy = other_pos[1] - pos[1];
                let dz = other_pos[2] - pos[2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 <= vr2 {
                    neighbors_in_vision += 1;
                    if d2 < best_cell_d2 {
                        best_cell_d2 = d2;
                        best_cell = Some((other_pos, radii[j]));
                    }
                }
            }

            let cell = &mut self.cells[i];
            let max_speed = cell.genome.max_speed;
            let my_radius = cell.phenotype.effective_radius().max(0.01);
            // Sprint 32: 2D hypot (vz=0). hypot ≠ naive sqrt(a²+b²) bit-by-bit,
            // takže pro CSV identity zůstává hypot. Sprint 33 přejde na 3D.
            let speed_norm =
                (cell.velocity[0].hypot(cell.velocity[1]) / max_speed).clamp(0.0, 1.0);
            let energy_norm = (cell.energy / REPRODUCE_THRESHOLD).clamp(0.0, 1.5);

            let mut inputs = [0.0_f32; BRAIN_INPUTS];
            if let Some(target) = best_food {
                inputs[0] = (target[0] - pos[0]) / vision_r;
                inputs[1] = (target[1] - pos[1]) / vision_r;
                inputs[15] = (target[2] - pos[2]) / vision_r;
            }
            if let Some((other, other_radius)) = best_cell {
                inputs[2] = (other[0] - pos[0]) / vision_r;
                inputs[3] = (other[1] - pos[1]) / vision_r;
                inputs[6] = (other_radius - my_radius) / my_radius;
                inputs[16] = (other[2] - pos[2]) / vision_r;
            }
            inputs[4] = energy_norm;
            inputs[5] = speed_norm;
            // Sprint 32: SmellField/PheromoneField stále 2D; projekce přes xy.
            // Sprint 35 promění je na 3D Jacobi a tahle projekce zmizí.
            let pos_xy = [pos[0], pos[1]];
            let grad = self.smell.gradient_at(pos_xy, SMELL_SAMPLE_EPSILON);
            inputs[7] = (grad[0] * SMELL_NORMALIZATION_GAIN).tanh();
            inputs[8] = (grad[1] * SMELL_NORMALIZATION_GAIN).tanh();
            // inputs[17] = smell_grad_z stays 0 (Sprint 35 unlock).
            // Sprint 33: heading_x, _y nově xy projekce 3D forward (násobeno
            // cos(pitch)). Pro pitch=0 redukuje na pre-Sprint-33 (cos(yaw),
            // sin(yaw)). heading_z = sin(pitch).
            let fwd = bioscape::forward_vector(cell.heading, cell.pitch);
            inputs[9] = fwd[0];
            inputs[10] = fwd[1];
            inputs[18] = fwd[2];
            let pgrad = self.pheromone.gradient_at(pos_xy, PHEROMONE_SAMPLE_EPSILON);
            inputs[11] = (pgrad[0] * PHEROMONE_NORMALIZATION_GAIN).tanh();
            inputs[12] = (pgrad[1] * PHEROMONE_NORMALIZATION_GAIN).tanh();
            // inputs[19] = pheromone_grad_z stays 0 (Sprint 35 unlock).
            // Sprint 29 quorum sensing: počet viditelných sousedů normovaný
            // přes DENSITY_NORM_COUNT, saturován tanhem do [0, 1). Brain dostává
            // skalární info o lokálním zalidnění bez emise (= bez predator
            // exploit Sprintu 24).
            inputs[13] = (neighbors_in_vision as f32 / DENSITY_NORM_COUNT).tanh();
            // Sprint 30: damage signál z minulého ticku. Predation/hazard během
            // minulého ticku přidaly do `damage_accum`; teď to brain čte a
            // resetuje. „Damage" je výhradně nedobrovolná ztráta — voluntární
            // cost (movement, morph, vision, attack) sem nepatří, cell o nich
            // ví přes outputs/energy. Reset až po čtení = klasický 1-tick delay
            // jako u pheromone/recurrent (žádný self-feedback).
            inputs[14] = (cell.damage_accum * DAMAGE_NORMALIZATION_GAIN).tanh();
            cell.damage_accum = 0.0;
            // Sprint 28: recurrent kanál — předchozí hidden activations jsou
            // input pro tento tick. Krátkodobá paměť, brain se přes mutace +
            // Hebbian může naučit ji použít. Při t=0 je `last_hidden` všechno
            // zero (init v Cell::random / reproduce), takže první tick je
            // identický s feed-forward — paměť nabíhá od ticku 1.
            inputs[BRAIN_INPUTS_SENSORY..BRAIN_INPUTS_SENSORY + BRAIN_RECURRENT]
                .copy_from_slice(&cell.last_hidden[..BRAIN_RECURRENT]);

            let (hidden, outputs) = cell.genome.brain.forward_with_state(&inputs);
            cell.last_inputs = inputs;
            cell.last_hidden = hidden;
            cell.last_outputs = outputs;
            let turn_signal = outputs[0];
            let thrust_norm = (outputs[1] + 1.0) * 0.5;
            // Sprint 35: pitch control aktivován — brain output[7] řídí
            // pitch_velocity stejnou mechanikou jako turn_signal (yaw).
            let pitch_signal = outputs[7];

            let body_proxy = my_radius;
            let turn_rate = cell.genome.turn_rate;
            let ang_acc = turn_signal * turn_rate / body_proxy;
            cell.angular_velocity += ang_acc * dt;
            let pitch_acc = pitch_signal * turn_rate / body_proxy;
            cell.pitch_velocity += pitch_acc * dt;

            let a_max = DRAG_COEFFICIENT * max_speed * max_speed / body_proxy;
            let a = thrust_norm * a_max;
            // Sprint 33: pitch=0 stále, takže forward = (cos_y, sin_y, 0) —
            // 3D-ready math přes forward_vector helper, ale fwd[2] = 0.
            let fwd = bioscape::forward_vector(cell.heading, cell.pitch);
            cell.velocity[0] += a * fwd[0] * dt;
            cell.velocity[1] += a * fwd[1] * dt;
            cell.velocity[2] += a * fwd[2] * dt;
        }
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
        let n = self.cells.len();
        self.positions_scratch.clear();
        self.positions_scratch
            .extend(self.cells.iter().map(|c| c.position));
        self.radii_scratch.clear();
        self.radii_scratch
            .extend(self.cells.iter().map(|c| c.phenotype.effective_radius()));
        self.deltas_scratch.clear();
        self.deltas_scratch.resize(n, [0.0, 0.0, 0.0]);
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let pair_r = CELL_RADIUS * (self.radii_scratch[i] + self.radii_scratch[j]);
                let pair_r2 = pair_r * pair_r;
                let dx = self.positions_scratch[i][0] - self.positions_scratch[j][0];
                let dy = self.positions_scratch[i][1] - self.positions_scratch[j][1];
                let dz = self.positions_scratch[i][2] - self.positions_scratch[j][2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < pair_r2 && d2 > 0.0 {
                    let d = d2.sqrt();
                    let overlap = pair_r - d;
                    self.deltas_scratch[i][0] += (dx / d) * overlap * 0.5;
                    self.deltas_scratch[i][1] += (dy / d) * overlap * 0.5;
                    self.deltas_scratch[i][2] += (dz / d) * overlap * 0.5;
                }
            }
        }
        for (cell, delta) in self.cells.iter_mut().zip(self.deltas_scratch.iter()) {
            cell.position[0] += delta[0];
            cell.position[1] += delta[1];
            cell.position[2] += delta[2];
        }
    }

    fn predate(&mut self) {
        let n = self.cells.len();
        self.positions_scratch.clear();
        self.positions_scratch
            .extend(self.cells.iter().map(|c| c.position));
        self.radii_scratch.clear();
        self.radii_scratch
            .extend(self.cells.iter().map(|c| c.phenotype.effective_radius()));
        self.spike_lengths_scratch.clear();
        self.spike_lengths_scratch
            .extend(self.cells.iter().map(|c| c.phenotype.spike_length));
        self.headings_scratch.clear();
        self.headings_scratch
            .extend(self.cells.iter().map(|c| c.heading));
        self.energy_deltas_scratch.clear();
        self.energy_deltas_scratch.resize(n, 0.0);
        self.damage_deltas_scratch.clear();
        self.damage_deltas_scratch.resize(n, 0.0);
        // Sprint 29 selfish-herd: pre-compute count of neighbors within
        // HERD_RADIUS for each cell. Used as dilution multiplier — predátor
        // dostane menší gain z prey, která je obklopena hejnem.
        let herd_r2 = HERD_RADIUS * HERD_RADIUS;
        let mut herd_counts: Vec<u32> = vec![0; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = self.positions_scratch[i][0] - self.positions_scratch[j][0];
                let dy = self.positions_scratch[i][1] - self.positions_scratch[j][1];
                let dz = self.positions_scratch[i][2] - self.positions_scratch[j][2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < herd_r2 {
                    herd_counts[i] += 1;
                    herd_counts[j] += 1;
                }
            }
        }
        let mut events: u64 = 0;
        for i in 0..n {
            // Sprint 27: attack je opt-in přes brain. Bez aktivního signálu jsou
            // kontakty s menšími cells jen kolize (řešené v resolve_collisions).
            let attack_signal = self.cells[i].last_outputs[6].max(0.0);
            if attack_signal <= ATTACK_THRESHOLD {
                continue;
            }
            #[allow(clippy::needless_range_loop)]
            for j in 0..n {
                if i == j {
                    continue;
                }
                let radius_a = self.radii_scratch[i];
                let radius_b = self.radii_scratch[j];
                if radius_a < SIZE_RATIO_THRESHOLD * radius_b {
                    continue;
                }
                let pair_r = CELL_RADIUS * (radius_a + radius_b);
                let pair_r2 = pair_r * pair_r;
                let dx = self.positions_scratch[i][0] - self.positions_scratch[j][0];
                let dy = self.positions_scratch[i][1] - self.positions_scratch[j][1];
                let dz = self.positions_scratch[i][2] - self.positions_scratch[j][2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < pair_r2 {
                    let mut gain = PREDATION_GAIN_PER_TICK;
                    let spike = self.spike_lengths_scratch[i];
                    if spike > 0.0 && d2 > 0.0 {
                        // Sprint 32: heading je 2D yaw, takže forward = (cos, sin, 0).
                        // Cosine s 3D směrem k cíli — z-složka forwardu je 0,
                        // takže příspěvek z dz × 0 zmizí. Sprint 33+ rozšíří
                        // forward na full 3D unit vector přes pitch.
                        let inv_d = 1.0 / d2.sqrt();
                        let to_j_x = -dx * inv_d;
                        let to_j_y = -dy * inv_d;
                        let h = self.headings_scratch[i];
                        let cos_angle = h.cos() * to_j_x + h.sin() * to_j_y;
                        if cos_angle >= bioscape::SPIKE_DOT_THRESHOLD {
                            gain += PREDATION_GAIN_PER_TICK
                                * spike
                                * bioscape::SPIKE_PREDATION_BONUS;
                        }
                    }
                    // Sprint 29 dilution: gain × 1/(1 + K × n_neighbors_prey).
                    // Drain prey beze změny — selfish-herd snižuje payoff lovu,
                    // ne utrpení oběti.
                    let dilution = 1.0 / (1.0 + DILUTION_K * herd_counts[j] as f32);
                    gain *= dilution;
                    self.energy_deltas_scratch[i] += gain;
                    self.energy_deltas_scratch[j] -= PREDATION_DRAIN_PER_TICK;
                    self.damage_deltas_scratch[j] += PREDATION_DRAIN_PER_TICK;
                    events += 1;
                }
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
        self.eaten_scratch.clear();
        self.eaten_scratch.resize(self.foods.len(), false);
        let eaten = &mut self.eaten_scratch;
        for cell in &mut self.cells {
            let pos = cell.position;
            let eat_r = EAT_RADIUS * cell.phenotype.effective_radius();
            let r2 = eat_r * eat_r;
            let mut ate = false;
            for (flag, food) in eaten.iter_mut().zip(self.foods.iter()) {
                if *flag {
                    continue;
                }
                let dx = pos[0] - food.position[0];
                let dy = pos[1] - food.position[1];
                let dz = pos[2] - food.position[2];
                if dx * dx + dy * dy + dz * dz <= r2 {
                    cell.energy += FOOD_VALUE
                        * food_multiplier(self.map.sample([food.position[0], food.position[1]]));
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
                // Sprint 31: rejection sampling proti uniform distribution —
                // bias k rich zonám. Spotřebovává retry budget, takže poor
                // zone občas dostane jídlo (clustering, ne ostré biomy).
                let richness = self
                    .map
                    .sample([candidate.position[0], candidate.position[1]]);
                if reject_food_for_richness(rng, richness) {
                    continue;
                }
                let mut blocked = false;
                for cell in &self.cells {
                    let exclusion = EAT_RADIUS * cell.phenotype.effective_radius();
                    let dx = candidate.position[0] - cell.position[0];
                    let dy = candidate.position[1] - cell.position[1];
                    let dz = candidate.position[2] - cell.position[2];
                    if dx * dx + dy * dy + dz * dz < exclusion * exclusion {
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
        // Sprint 25: cells musí AKTIVNĚ emitovat pheromone (output[2] >
        // threshold) aby byly fertile. Tiché cells nemůžou reprodukovat —
        // selektuje proti free-riders na pheromone field.
        let fertile: Vec<(usize, [f32; 3])> = self
            .cells
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                c.energy >= REPRODUCE_THRESHOLD
                    && c.last_outputs[2] > MATING_PHEROMONE_THRESHOLD
            })
            .map(|(i, c)| (i, c.position))
            .collect();
        self.fertile_ticks_gen += fertile.len() as u64;

        // Greedy O(N²) pairing on fertile indices.
        let mut paired: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut matings: Vec<(usize, usize)> = Vec::new();
        let mating_r2 = self.mating_radius * self.mating_radius;
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
                let dz = pos_a[2] - pos_b[2];
                let d2 = dx * dx + dy * dy + dz * dz;
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
                (cell_a.position[2] + cell_b.position[2]) * 0.5,
            ];
            let child_phenotype = Phenotype::from_genome(&child_genome);
            to_spawn.push(Cell {
                position: mid_pos,
                velocity: [
                    direction.cos() * child_genome.max_speed,
                    direction.sin() * child_genome.max_speed,
                    0.0,
                ],
                angular_velocity: 0.0,
                pitch_velocity: 0.0,
                energy: energy_a + energy_b,
                heading: direction,
                pitch: 0.0,
                lineage_id: cell_a.lineage_id,
                lineage_birth_gen: cell_a.lineage_birth_gen,
                last_inputs: [0.0; BRAIN_INPUTS],
                last_hidden: [0.0; BRAIN_HIDDEN],
                last_outputs: [0.0; BRAIN_OUTPUTS],
                damage_accum: 0.0,
                phenotype: child_phenotype,
                genome: child_genome,
            });
        }
        self.births_gen += to_spawn.len() as u64;
        self.cells.extend(to_spawn);
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
                    new_foods.push(Food { position: pos });
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
    // Uniform reference v Full-HD světě (1920×1080) pro pop=N: 0.5·√(A/N) ≈
    // 720/√N. Hodnoty výrazně pod referencí = clustering. n=1 dá 0 (degenerate);
    // exclude se v interpretaci, zde zapíšeme NaN-safe 0.0 jen pro nepadnutí.
    let nn_dist_m = if n >= 2 {
        let mut sum = 0.0_f64;
        for i in 0..n {
            let mut min_d2 = f32::MAX;
            let pi = world.cells[i].position;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let pj = world.cells[j].position;
                let dx = pi[0] - pj[0];
                let dy = pi[1] - pj[1];
                let d2 = dx * dx + dy * dy;
                if d2 < min_d2 {
                    min_d2 = d2;
                }
            }
            sum += min_d2.sqrt() as f64;
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
    let args: Vec<String> = env::args().collect();
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

    let mut rng = StdRng::seed_from_u64(seed);
    let mut world = World::new(&mut rng, map_seed, mating_radius);

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
        "headless: seed={} map_seed={} mating_radius={} max_gens={} out={} initial_cells={} initial_food={}",
        seed,
        map_seed,
        mating_radius,
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
