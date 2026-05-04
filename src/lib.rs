//! Bioscape simulation core — pure logic, no rendering.
//!
//! Keeping this layer free of Bevy types lets us drive the same world
//! from a windowed renderer (`main.rs`) or a headless batch run later.

use core::f32::consts::TAU;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const HUE_RANGE: f32 = 360.0;
const MIN_SPEED: f32 = 1.0;
const MIN_VISION: f32 = 1.0;
const MIN_TURN_RATE: f32 = 0.1;
const MIN_BODY_SIZE: f32 = 0.3;
pub const INITIAL_ENERGY: f32 = 100.0;
// Brain inputs: 0=food_dx, 1=food_dy, 2=cell_dx, 3=cell_dy, 4=energy,
// 5=speed, 6=rel_size, 7=smell_grad_x, 8=smell_grad_y, 9=heading_x,
// 10=heading_y, 11=pheromone_grad_x, 12=pheromone_grad_y.
pub const BRAIN_INPUTS: usize = 13;
pub const BRAIN_HIDDEN: usize = 8;
// Brain outputs: 0=turn, 1=thrust, 2=pheromone modulation (positive = emit
// more above baseline, costs energy).
pub const BRAIN_OUTPUTS: usize = 3;
/// Inicializační bias na thrust output bin v `Brain::random`. Bez něj má ~½
/// random brainů thrust output blízko nuly (cell se sotva hýbe), což vytvářelo
/// hluboké bottlenecky v ranných generacích. Po prvním selekčním tlaku evoluce
/// hodnotu doladí — bias je jen jumpstart.
pub const INNATE_THRUST_BIAS: f32 = 2.0;
/// Inicializační bias na pheromone output (b2[2]). Sprint 25 vyžaduje aktivní
/// emisi pro reprodukci — bez biasu jen ~25 % párů projde threshold. S bias
/// 1.0 většina random brainů emituje nad threshold v gen 0; selekce pak ladí.
pub const INNATE_PHEROMONE_BIAS: f32 = 1.0;

// Shared sim parameters consumed by both the Bevy renderer (`src/main.rs`)
// and the headless harness (`src/bin/headless.rs`). Single source of truth —
// tune here. Renderer-only and headless-only knobs stay in their binaries.

pub const FIXED_TIMESTEP_HZ: f32 = 60.0;
pub const TICKS_PER_GENERATION: u64 = 600;
pub const GENERATIONS_PER_EPOCH: u64 = 100;

pub const INITIAL_CELLS: usize = 200;
pub const MAX_POPULATION: usize = 1000;

pub const CELL_RADIUS: f32 = 5.0;
pub const EAT_RADIUS: f32 = 8.0;
pub const MATING_RADIUS: f32 = 200.0;

pub const DRAG_COEFFICIENT: f32 = 0.005;
pub const ANGULAR_DRAG: f32 = 1.0;
pub const ENERGY_COST_PER_V_SQ: f32 = 0.0008;
pub const ANGULAR_ENERGY_COST: f32 = 0.05;
pub const VISION_COST_PER_RADIUS: f32 = 0.02;
pub const BODY_COST_FACTOR: f32 = 0.8;

pub const FOOD_VALUE: f32 = 20.0;
pub const FOOD_SPAWN_RATE: usize = 5;
pub const WORLD_UNITS_PER_FOOD: f32 = 2600.0;
// Environmentální hazard layer: passive energy drain v "nebezpečných" zónách.
// Zónová mapa jde z `WorldMap` noise — POSITIVNÍ korelace s food richness:
// bohaté oblasti = nebezpečné (high reward, high risk), chudé = bezpečné.
// Vytváří trade-off niche: efficient cell může těžit rich-dangerous, slabší
// se uchýlí do safe-poor a žije s méně food. Drain za sec při noise=1:
// HAZARD_FLOOR + HAZARD_AMP = celkový max. Ladí se pouze v binárkách.
pub const HAZARD_DRAIN_PER_SEC: f32 = 0.5;
pub const HAZARD_FLOOR: f32 = 0.0;
pub const HAZARD_AMP: f32 = 1.0;

// Pheromone signaling layer. 2D scalar field jako SmellField, ale zdroje jsou
// cells. Sprint 25: BASELINE = 0 (žádné free-rider, žádný predator exploit z
// Sprint 24). Cells musí aktivně emitovat brain output[2] aby vznikl signál,
// **a aby byly způsobilé k reprodukci** — `MATING_PHEROMONE_THRESHOLD` gating.
// Brain detekuje gradient přes `inputs[11..13]`. Cost ∝ emise.
pub const PHEROMONE_GRID_RES: usize = 64;
pub const PHEROMONE_DIFFUSION: f32 = 0.15;
pub const PHEROMONE_DECAY: f32 = 0.3;
pub const PHEROMONE_BASELINE_EMIT: f32 = 0.0;
pub const PHEROMONE_BRAIN_MOD: f32 = 1.0;
pub const PHEROMONE_COST_PER_RATE: f32 = 1.0;
pub const PHEROMONE_SAMPLE_EPSILON: f32 = 10.0;
pub const PHEROMONE_NORMALIZATION_GAIN: f32 = 0.5;
/// Cell musí mít `last_outputs[2] > THRESHOLD` aby byla eligible pro mating.
/// Mating je tak podmíněn aktivní emisí — selektuje proti tichým cells, které
/// by jinak free-ride na public goods of pheromone field.
pub const MATING_PHEROMONE_THRESHOLD: f32 = 0.2;
pub const MAX_SPAWN_ATTEMPTS: usize = 5;
pub const CARRION_FOOD_COUNT: usize = 2;

pub const REPRODUCE_THRESHOLD: f32 = 150.0;
pub const SIZE_RATIO_THRESHOLD: f32 = 1.3;
pub const PREDATION_DRAIN_PER_TICK: f32 = 3.0;
pub const PREDATION_GAIN_PER_TICK: f32 = 1.5;

pub const CYCLE_GEN_PERIOD: u64 = 50;
pub const CYCLE_AMPLITUDE: f32 = 0.15;

pub const SMELL_GRID_RES: usize = 64;
pub const SMELL_DIFFUSION: f32 = 0.15;
pub const SMELL_DECAY: f32 = 0.3;
pub const SMELL_PER_FOOD: f32 = 1.0;
pub const SMELL_SAMPLE_EPSILON: f32 = 10.0;
pub const SMELL_NORMALIZATION_GAIN: f32 = 0.5;

pub const LEARNING_RATE: f32 = 0.005;

pub const WORLD_MAP_RES: usize = 64;
pub const WORLD_MAP_BASE_RES: usize = 8;
pub const WORLD_MAP_SEED: u64 = 1234;
// Food-value multiplier = FLOOR + AMP × noise(pos), noise ∈ [0,1].
// → multiplier ∈ [FLOOR, FLOOR+AMP]. Drives spatial selection on richness.
pub const WORLD_MAP_FOOD_FLOOR: f32 = 0.85;
pub const WORLD_MAP_FOOD_AMP: f32 = 0.3;

pub const MUTATION_CONFIG: MutationConfig = MutationConfig {
    sigma_speed: 3.0,
    sigma_hue: 5.0,
    sigma_vision: 3.0,
    sigma_turn_rate: 0.3,
    sigma_body_size: 0.05,
    sigma_brain: 0.2,
};
pub const PHYSICS_CONFIG: PhysicsConfig = PhysicsConfig {
    drag: DRAG_COEFFICIENT,
    angular_drag: ANGULAR_DRAG,
    energy_cost_per_v_sq: ENERGY_COST_PER_V_SQ,
    angular_energy_cost: ANGULAR_ENERGY_COST,
    vision_cost_per_radius: VISION_COST_PER_RADIUS,
    body_cost_factor: BODY_COST_FACTOR,
};

#[derive(Debug, Clone, Copy)]
pub struct Brain {
    pub w1: [[f32; BRAIN_INPUTS]; BRAIN_HIDDEN],
    pub b1: [f32; BRAIN_HIDDEN],
    pub w2: [[f32; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
    pub b2: [f32; BRAIN_OUTPUTS],
}

impl Brain {
    pub fn random(rng: &mut impl Rng) -> Self {
        let mut w1 = [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN];
        let mut b1 = [0.0; BRAIN_HIDDEN];
        let mut w2 = [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS];
        let mut b2 = [0.0; BRAIN_OUTPUTS];
        for (row, bias) in w1.iter_mut().zip(b1.iter_mut()) {
            for w in row.iter_mut() {
                *w = gaussian(rng);
            }
            *bias = gaussian(rng);
        }
        for (row, bias) in w2.iter_mut().zip(b2.iter_mut()) {
            for w in row.iter_mut() {
                *w = gaussian(rng);
            }
            *bias = gaussian(rng);
        }
        // Innate thrust bias: bumps b2[1] (thrust output) above zero. Posune
        // distribuci `thrust_norm = (tanh(b2 + ...) + 1) / 2` od mean ~0.5
        // (random walk stuck) k mean ~0.7 (consistent forward motion). Hebbian
        // + selekce dál ladí; tohle jen řeší kallové cells co se nehýbou.
        b2[1] += INNATE_THRUST_BIAS;
        // Innate pheromone bias: Sprint 25 vyžaduje active emisi pro mating.
        // Bez biasu by se polovina random cells nemohla reprodukovat.
        b2[2] += INNATE_PHEROMONE_BIAS;
        Self { w1, b1, w2, b2 }
    }

    pub fn forward(&self, inputs: &[f32; BRAIN_INPUTS]) -> [f32; BRAIN_OUTPUTS] {
        self.forward_with_state(inputs).1
    }

    /// Same forward pass as `forward`, but also returns hidden activations
    /// (needed for Hebbian updates).
    pub fn forward_with_state(
        &self,
        inputs: &[f32; BRAIN_INPUTS],
    ) -> ([f32; BRAIN_HIDDEN], [f32; BRAIN_OUTPUTS]) {
        let mut hidden = [0.0_f32; BRAIN_HIDDEN];
        for ((h, row), &bias) in hidden.iter_mut().zip(self.w1.iter()).zip(self.b1.iter()) {
            let mut sum = bias;
            for (&w, &x) in row.iter().zip(inputs.iter()) {
                sum += w * x;
            }
            *h = sum.tanh();
        }
        let mut out = [0.0_f32; BRAIN_OUTPUTS];
        for ((o, row), &bias) in out.iter_mut().zip(self.w2.iter()).zip(self.b2.iter()) {
            let mut sum = bias;
            for (&w, &h) in row.iter().zip(hidden.iter()) {
                sum += w * h;
            }
            *o = sum.tanh();
        }
        (hidden, out)
    }

    /// Reward-modulated Hebbian update. `Δw = lr · reward · pre · post`.
    /// Pre-/post-synaptic activations come from a stored prior forward
    /// pass — this is "myopic" credit assignment (1-tick window). Reward
    /// fires on biologically meaningful events (eating, predation kills).
    pub fn hebbian_update(
        &mut self,
        last_inputs: &[f32; BRAIN_INPUTS],
        last_hidden: &[f32; BRAIN_HIDDEN],
        last_outputs: &[f32; BRAIN_OUTPUTS],
        reward: f32,
        learning_rate: f32,
    ) {
        let lr = learning_rate * reward;
        for (out_h, &h) in self.w1.iter_mut().zip(last_hidden.iter()) {
            for (w, &x) in out_h.iter_mut().zip(last_inputs.iter()) {
                *w += lr * h * x;
            }
        }
        for (b, &h) in self.b1.iter_mut().zip(last_hidden.iter()) {
            *b += lr * h;
        }
        for (out_o, &o) in self.w2.iter_mut().zip(last_outputs.iter()) {
            for (w, &h) in out_o.iter_mut().zip(last_hidden.iter()) {
                *w += lr * o * h;
            }
        }
        for (b, &o) in self.b2.iter_mut().zip(last_outputs.iter()) {
            *b += lr * o;
        }
    }

    pub fn mutate(&self, rng: &mut impl Rng, sigma: f32) -> Self {
        let mut out = *self;
        for (row, bias) in out.w1.iter_mut().zip(out.b1.iter_mut()) {
            for w in row.iter_mut() {
                *w += gaussian(rng) * sigma;
            }
            *bias += gaussian(rng) * sigma;
        }
        for (row, bias) in out.w2.iter_mut().zip(out.b2.iter_mut()) {
            for w in row.iter_mut() {
                *w += gaussian(rng) * sigma;
            }
            *bias += gaussian(rng) * sigma;
        }
        out
    }

    /// Per-row uniform crossover. Each hidden neuron's `w1` row + `b1`
    /// scalar comes from one parent (50/50); same for output neurons. Per-row
    /// rather than per-weight preserves coordinated patterns within a single
    /// neuron's receptive field.
    pub fn crossover(a: &Brain, b: &Brain, rng: &mut impl Rng) -> Brain {
        let mut out = *a;
        for i in 0..BRAIN_HIDDEN {
            if rng.random::<bool>() {
                out.w1[i] = b.w1[i];
                out.b1[i] = b.b1[i];
            }
        }
        for i in 0..BRAIN_OUTPUTS {
            if rng.random::<bool>() {
                out.w2[i] = b.w2[i];
                out.b2[i] = b.b2[i];
            }
        }
        out
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MutationConfig {
    pub sigma_speed: f32,
    pub sigma_hue: f32,
    pub sigma_vision: f32,
    pub sigma_turn_rate: f32,
    pub sigma_body_size: f32,
    pub sigma_brain: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Genome {
    pub max_speed: f32,
    pub color_hue: f32,
    pub vision_radius: f32,
    pub turn_rate: f32,
    pub body_size: f32,
    pub brain: Brain,
}

impl Genome {
    pub fn random(rng: &mut impl Rng) -> Self {
        Self {
            max_speed: rng.random_range(30.0..90.0),
            color_hue: rng.random_range(0.0..HUE_RANGE),
            vision_radius: rng.random_range(20.0..80.0),
            turn_rate: rng.random_range(1.0..5.0),
            body_size: rng.random_range(0.7..1.3),
            brain: Brain::random(rng),
        }
    }

    pub fn mutate(&self, rng: &mut impl Rng, cfg: &MutationConfig) -> Self {
        Self {
            max_speed: (self.max_speed + gaussian(rng) * cfg.sigma_speed).max(MIN_SPEED),
            color_hue: (self.color_hue + gaussian(rng) * cfg.sigma_hue).rem_euclid(HUE_RANGE),
            vision_radius: (self.vision_radius + gaussian(rng) * cfg.sigma_vision).max(MIN_VISION),
            turn_rate: (self.turn_rate + gaussian(rng) * cfg.sigma_turn_rate).max(MIN_TURN_RATE),
            body_size: (self.body_size + gaussian(rng) * cfg.sigma_body_size).max(MIN_BODY_SIZE),
            brain: self.brain.mutate(rng, cfg.sigma_brain),
        }
    }

    /// Per-gene uniform crossover. Each scalar gene picks 50/50 from one
    /// parent; brain uses its own per-row crossover.
    pub fn crossover(a: &Genome, b: &Genome, rng: &mut impl Rng) -> Genome {
        Genome {
            max_speed: if rng.random::<bool>() { a.max_speed } else { b.max_speed },
            color_hue: if rng.random::<bool>() { a.color_hue } else { b.color_hue },
            vision_radius: if rng.random::<bool>() { a.vision_radius } else { b.vision_radius },
            turn_rate: if rng.random::<bool>() { a.turn_rate } else { b.turn_rate },
            body_size: if rng.random::<bool>() { a.body_size } else { b.body_size },
            brain: Brain::crossover(&a.brain, &b.brain, rng),
        }
    }
}

fn gaussian(rng: &mut impl Rng) -> f32 {
    let u1: f32 = rng.random::<f32>().max(f32::EPSILON);
    let u2: f32 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}

#[derive(Debug, Clone, Copy)]
pub struct PhysicsConfig {
    pub drag: f32,
    pub angular_drag: f32,
    pub energy_cost_per_v_sq: f32,
    /// Multiplier on `body_size² × ω² × dt` for rotational kinetic drain.
    /// Decoupled from linear cost so spinning-in-place is properly punished
    /// (otherwise random brains settle into a "spin and starve" local minimum
    /// because rotation is essentially free).
    pub angular_energy_cost: f32,
    pub vision_cost_per_radius: f32,
    pub body_cost_factor: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Cell {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub angular_velocity: f32,
    pub energy: f32,
    // Persists even when velocity hits zero — atan2(0, 0) would otherwise
    // collapse to 0 and bias evolution toward east-facing motion.
    pub heading: f32,
    // Lineage tracking — inherited from parent at reproduction (no mutation).
    // birth_gen records the generation when the lineage was created (initial
    // population: 0; new lineages from speciation events would bump it).
    pub lineage_id: u64,
    pub lineage_birth_gen: u64,
    // Recent activations from the last brain forward pass — Hebbian updates
    // read these to credit-assign on reward events (myopic, 1-tick window).
    pub last_inputs: [f32; BRAIN_INPUTS],
    pub last_hidden: [f32; BRAIN_HIDDEN],
    pub last_outputs: [f32; BRAIN_OUTPUTS],
    pub genome: Genome,
}

impl Cell {
    pub fn random(
        rng: &mut impl Rng,
        world_half: [f32; 2],
        lineage_id: u64,
        lineage_birth_gen: u64,
    ) -> Self {
        let genome = Genome::random(rng);
        Self::from_genome(rng, genome, world_half, lineage_id, lineage_birth_gen)
    }

    pub fn from_genome(
        rng: &mut impl Rng,
        genome: Genome,
        world_half: [f32; 2],
        lineage_id: u64,
        lineage_birth_gen: u64,
    ) -> Self {
        let direction = rng.random_range(0.0..TAU);
        Self {
            position: [
                rng.random_range(-world_half[0]..world_half[0]),
                rng.random_range(-world_half[1]..world_half[1]),
            ],
            velocity: [direction.cos() * genome.max_speed, direction.sin() * genome.max_speed],
            angular_velocity: 0.0,
            energy: INITIAL_ENERGY,
            heading: direction,
            lineage_id,
            lineage_birth_gen,
            last_inputs: [0.0; BRAIN_INPUTS],
            last_hidden: [0.0; BRAIN_HIDDEN],
            last_outputs: [0.0; BRAIN_OUTPUTS],
            genome,
        }
    }

    /// Per-tick physics: kinematic update from velocity / angular_velocity,
    /// quadratic drag on both, energy drains. Brain-applied forces should be
    /// integrated into velocity / angular_velocity *before* step (in
    /// `cells_brain_act`); step is purely passive integration + dissipation.
    pub fn step(&mut self, dt: f32, world_half: [f32; 2], physics: &PhysicsConfig) {
        // Position from current velocity (semi-implicit: drag applied after).
        self.position[0] += self.velocity[0] * dt;
        self.position[1] += self.velocity[1] * dt;
        self.heading += self.angular_velocity * dt;

        // Quadratic linear drag: dv = -DRAG · |v| · v · dt.
        let v_mag = self.velocity[0].hypot(self.velocity[1]);
        let drag_dt = physics.drag * v_mag * dt;
        self.velocity[0] -= drag_dt * self.velocity[0];
        self.velocity[1] -= drag_dt * self.velocity[1];

        // Linear angular drag (multiplicative decay) — cheap, rotation slows
        // exponentially. dω/dt = -ANGULAR_DRAG · ω.
        self.angular_velocity *= (1.0 - physics.angular_drag * dt).max(0.0);

        // Energy: kinetic-motion proxy (v² · cost), rotational kinetic
        // (MoI · ω² · cost ≈ body_size² · ω² · cost), vision, body². Linear
        // distance-based cost is gone — fluid drag makes power ∝ v³ in
        // strict physics, but v² is cheaper to balance and keeps the
        // intuition (faster = more expensive).
        self.energy -= v_mag * v_mag * physics.energy_cost_per_v_sq * dt;
        let bs = self.genome.body_size;
        let av = self.angular_velocity;
        self.energy -= bs * bs * av * av * physics.angular_energy_cost * dt;
        self.energy -= self.genome.vision_radius * physics.vision_cost_per_radius * dt;
        self.energy -= bs * bs * physics.body_cost_factor * dt;

        // Reflect off the world boundary so cells stay visible.
        let mut bounced = false;
        if self.position[0].abs() > world_half[0] {
            self.velocity[0] = -self.velocity[0];
            self.position[0] = self.position[0].clamp(-world_half[0], world_half[0]);
            bounced = true;
        }
        if self.position[1].abs() > world_half[1] {
            self.velocity[1] = -self.velocity[1];
            self.position[1] = self.position[1].clamp(-world_half[1], world_half[1]);
            bounced = true;
        }
        if bounced {
            self.heading = self.velocity[1].atan2(self.velocity[0]);
        }
    }

    pub fn try_eat(&mut self, food: &Food, eat_radius: f32, food_value: f32) -> bool {
        let dx = self.position[0] - food.position[0];
        let dy = self.position[1] - food.position[1];
        if dx * dx + dy * dy <= eat_radius * eat_radius {
            self.energy += food_value;
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Food {
    pub position: [f32; 2],
}

impl Food {
    pub fn random(rng: &mut impl Rng, world_half: [f32; 2]) -> Self {
        Self {
            position: [
                rng.random_range(-world_half[0]..world_half[0]),
                rng.random_range(-world_half[1]..world_half[1]),
            ],
        }
    }
}

/// 2D scalar field with explicit-Jacobi diffusion and exponential decay.
/// Doublet (`grid` + `scratch`) for in-place stepping. Cells tagged at
/// food positions seed the field; cells read its gradient as a smell input.
#[derive(Debug, Clone)]
pub struct SmellField {
    pub resolution: usize,
    pub world_half: [f32; 2],
    grid: Vec<f32>,
    scratch: Vec<f32>,
}

impl SmellField {
    pub fn new(resolution: usize, world_half: [f32; 2]) -> Self {
        let n = resolution * resolution;
        Self {
            resolution,
            world_half,
            grid: vec![0.0; n],
            scratch: vec![0.0; n],
        }
    }

    fn cell_size_x(&self) -> f32 {
        (2.0 * self.world_half[0]) / self.resolution as f32
    }
    fn cell_size_y(&self) -> f32 {
        (2.0 * self.world_half[1]) / self.resolution as f32
    }

    fn idx_of(&self, pos: [f32; 2]) -> Option<usize> {
        let xi = ((pos[0] + self.world_half[0]) / self.cell_size_x()).floor() as i32;
        let yi = ((pos[1] + self.world_half[1]) / self.cell_size_y()).floor() as i32;
        let n = self.resolution as i32;
        if xi < 0 || xi >= n || yi < 0 || yi >= n {
            None
        } else {
            Some((yi as usize) * self.resolution + xi as usize)
        }
    }

    pub fn add_source(&mut self, pos: [f32; 2], amount: f32) {
        if let Some(idx) = self.idx_of(pos) {
            self.grid[idx] += amount;
        }
    }

    /// Single explicit-Jacobi diffusion step + multiplicative decay.
    /// `diffusion` < 0.25 for stability in 2D. `decay_per_sec` is the
    /// continuous-time rate; we discretize as `(1 - decay·dt)`.
    pub fn step(&mut self, diffusion: f32, decay_per_sec: f32, dt: f32) {
        let n = self.resolution;
        let decay = (1.0 - decay_per_sec * dt).max(0.0);
        for j in 0..n {
            for i in 0..n {
                let idx = j * n + i;
                let center = self.grid[idx];
                let left = if i > 0 { self.grid[idx - 1] } else { center };
                let right = if i + 1 < n { self.grid[idx + 1] } else { center };
                let up = if j > 0 { self.grid[idx - n] } else { center };
                let down = if j + 1 < n { self.grid[idx + n] } else { center };
                let new = center + diffusion * (left + right + up + down - 4.0 * center);
                self.scratch[idx] = new * decay;
            }
        }
        std::mem::swap(&mut self.grid, &mut self.scratch);
    }

    pub fn sample(&self, pos: [f32; 2]) -> f32 {
        self.idx_of(pos).map(|i| self.grid[i]).unwrap_or(0.0)
    }

    /// Central differences at `pos ± epsilon` along each axis. Returns
    /// `[d/dx, d/dy]`. Out-of-bounds samples count as 0.
    pub fn gradient_at(&self, pos: [f32; 2], epsilon: f32) -> [f32; 2] {
        let f_xp = self.sample([pos[0] + epsilon, pos[1]]);
        let f_xm = self.sample([pos[0] - epsilon, pos[1]]);
        let f_yp = self.sample([pos[0], pos[1] + epsilon]);
        let f_ym = self.sample([pos[0], pos[1] - epsilon]);
        let inv = 1.0 / (2.0 * epsilon);
        [(f_xp - f_xm) * inv, (f_yp - f_ym) * inv]
    }
}

/// Deterministic 2D scalar field on `[resolution × resolution]` mřížce
/// pokrývající celý svět. Hodnoty v `[0, 1]` z value-noise:
/// `base_resolution × base_resolution` random uniform grid, smoothstep
/// bilinear interp do plné resolution. Generováno jednou při startu, pak
/// jen čtení — žádný update per tick.
///
/// Use case: prostorová modulace mechaniky, která má být nehomogenní —
/// food_richness, hazard, terrain drag, atd. (Sprint 21 = food_richness.)
#[derive(Debug, Clone)]
pub struct WorldMap {
    pub resolution: usize,
    pub world_half: [f32; 2],
    field: Vec<f32>,
}

impl WorldMap {
    pub fn new(
        resolution: usize,
        base_resolution: usize,
        world_half: [f32; 2],
        seed: u64,
    ) -> Self {
        assert!(resolution >= 2 && base_resolution >= 2);
        let mut rng = StdRng::seed_from_u64(seed);
        let base: Vec<f32> = (0..base_resolution * base_resolution)
            .map(|_| rng.random())
            .collect();

        let mut field = vec![0.0_f32; resolution * resolution];
        let scale = (base_resolution as f32 - 1.0) / resolution as f32;
        for j in 0..resolution {
            for i in 0..resolution {
                let u = (i as f32 + 0.5) * scale;
                let v = (j as f32 + 0.5) * scale;
                let x0 = (u.floor() as usize).min(base_resolution - 1);
                let y0 = (v.floor() as usize).min(base_resolution - 1);
                let x1 = (x0 + 1).min(base_resolution - 1);
                let y1 = (y0 + 1).min(base_resolution - 1);
                let fx = (u - x0 as f32).clamp(0.0, 1.0);
                let fy = (v - y0 as f32).clamp(0.0, 1.0);
                let sx = fx * fx * (3.0 - 2.0 * fx);
                let sy = fy * fy * (3.0 - 2.0 * fy);
                let v00 = base[y0 * base_resolution + x0];
                let v10 = base[y0 * base_resolution + x1];
                let v01 = base[y1 * base_resolution + x0];
                let v11 = base[y1 * base_resolution + x1];
                let v0 = v00 * (1.0 - sx) + v10 * sx;
                let v1 = v01 * (1.0 - sx) + v11 * sx;
                field[j * resolution + i] = v0 * (1.0 - sy) + v1 * sy;
            }
        }

        Self {
            resolution,
            world_half,
            field,
        }
    }

    pub fn sample(&self, pos: [f32; 2]) -> f32 {
        let cell_x = (2.0 * self.world_half[0]) / self.resolution as f32;
        let cell_y = (2.0 * self.world_half[1]) / self.resolution as f32;
        let xi = ((pos[0] + self.world_half[0]) / cell_x).floor() as i32;
        let yi = ((pos[1] + self.world_half[1]) / cell_y).floor() as i32;
        let xi = xi.clamp(0, self.resolution as i32 - 1) as usize;
        let yi = yi.clamp(0, self.resolution as i32 - 1) as usize;
        self.field[yi * self.resolution + xi]
    }

    pub fn field(&self) -> &[f32] {
        &self.field
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SimClock {
    pub tick: u64,
    pub generation: u64,
    pub epoch: u64,
    pub ticks_per_generation: u64,
    pub generations_per_epoch: u64,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ClockTransitions {
    pub generation_ended: Option<u64>,
    pub epoch_ended: Option<u64>,
}

impl SimClock {
    pub fn new(ticks_per_generation: u64, generations_per_epoch: u64) -> Self {
        Self {
            tick: 0,
            generation: 0,
            epoch: 0,
            ticks_per_generation,
            generations_per_epoch,
        }
    }

    pub fn advance(&mut self) -> ClockTransitions {
        self.tick += 1;
        let mut transitions = ClockTransitions::default();
        if self.tick.is_multiple_of(self.ticks_per_generation) {
            transitions.generation_ended = Some(self.generation);
            self.generation += 1;
            if self.generation.is_multiple_of(self.generations_per_epoch) {
                transitions.epoch_ended = Some(self.epoch);
                self.epoch += 1;
            }
        }
        transitions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advance_reports_generation_boundary() {
        let mut clock = SimClock::new(3, 2);
        assert_eq!(clock.advance(), ClockTransitions::default());
        assert_eq!(clock.advance(), ClockTransitions::default());
        let t = clock.advance();
        assert_eq!(t.generation_ended, Some(0));
        assert_eq!(t.epoch_ended, None);
        assert_eq!((clock.tick, clock.generation, clock.epoch), (3, 1, 0));
    }

    #[test]
    fn epoch_fires_alongside_generation_boundary() {
        let mut clock = SimClock::new(2, 2);
        clock.advance();
        let t = clock.advance();
        assert_eq!(t.generation_ended, Some(0));
        assert_eq!(t.epoch_ended, None);
        clock.advance();
        let t = clock.advance();
        assert_eq!(t.generation_ended, Some(1));
        assert_eq!(t.epoch_ended, Some(0));
        assert_eq!((clock.tick, clock.generation, clock.epoch), (4, 2, 1));
    }

    fn dummy_brain() -> Brain {
        Brain {
            w1: [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN],
            b1: [0.0; BRAIN_HIDDEN],
            w2: [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
            b2: [0.0; BRAIN_OUTPUTS],
        }
    }

    fn dummy_genome() -> Genome {
        Genome {
            max_speed: 60.0,
            color_hue: 0.0,
            vision_radius: 40.0,
            turn_rate: 2.5,
            body_size: 1.0,
            brain: dummy_brain(),
        }
    }

    fn zero_cfg() -> MutationConfig {
        MutationConfig {
            sigma_speed: 0.0,
            sigma_hue: 0.0,
            sigma_vision: 0.0,
            sigma_turn_rate: 0.0,
            sigma_body_size: 0.0,
            sigma_brain: 0.0,
        }
    }

    #[test]
    fn mutation_with_zero_sigma_is_identity() {
        let mut rng = rand::rng();
        let g = Genome {
            max_speed: 50.0,
            color_hue: 120.0,
            vision_radius: 40.0,
            turn_rate: 2.5,
            body_size: 1.1,
            brain: Brain {
                w1: [[1.0; BRAIN_INPUTS]; BRAIN_HIDDEN],
                b1: [0.3; BRAIN_HIDDEN],
                w2: [[1.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
                b2: [0.5; BRAIN_OUTPUTS],
            },
        };
        let m = g.mutate(&mut rng, &zero_cfg());
        assert_eq!(m.max_speed, 50.0);
        assert_eq!(m.color_hue, 120.0);
        assert_eq!(m.vision_radius, 40.0);
        assert_eq!(m.turn_rate, 2.5);
        assert_eq!(m.body_size, 1.1);
        assert_eq!(m.brain.w1, g.brain.w1);
        assert_eq!(m.brain.b1, g.brain.b1);
        assert_eq!(m.brain.w2, g.brain.w2);
        assert_eq!(m.brain.b2, g.brain.b2);
    }

    #[test]
    fn mutation_keeps_genes_in_valid_ranges() {
        let mut rng = rand::rng();
        let g = dummy_genome();
        let cfg = MutationConfig {
            sigma_speed: 100.0,
            sigma_hue: 1000.0,
            sigma_vision: 100.0,
            sigma_turn_rate: 100.0,
            sigma_body_size: 10.0,
            sigma_brain: 10.0,
        };
        for _ in 0..1000 {
            let m = g.mutate(&mut rng, &cfg);
            assert!(m.max_speed >= MIN_SPEED);
            assert!(m.color_hue >= 0.0 && m.color_hue < HUE_RANGE);
            assert!(m.vision_radius >= MIN_VISION);
            assert!(m.turn_rate >= MIN_TURN_RATE);
            assert!(m.body_size >= MIN_BODY_SIZE);
        }
    }

    fn no_drag_physics(cost_per_v_sq: f32, vision_cost: f32) -> PhysicsConfig {
        PhysicsConfig {
            drag: 0.0,
            angular_drag: 0.0,
            energy_cost_per_v_sq: cost_per_v_sq,
            angular_energy_cost: 0.0,
            vision_cost_per_radius: vision_cost,
            body_cost_factor: 0.0,
        }
    }

    fn base_cell() -> Cell {
        Cell {
            position: [0.0, 0.0],
            velocity: [0.0, 0.0],
            angular_velocity: 0.0,
            energy: 100.0,
            heading: 0.0,
            lineage_id: 0,
            lineage_birth_gen: 0,
            last_inputs: [0.0; BRAIN_INPUTS],
            last_hidden: [0.0; BRAIN_HIDDEN],
            last_outputs: [0.0; BRAIN_OUTPUTS],
            genome: dummy_genome(),
        }
    }

    #[test]
    fn step_drains_energy_from_motion_and_vision() {
        let mut cell = Cell {
            velocity: [60.0, 0.0],
            ..base_cell()
        };
        cell.step(1.0, [1000.0, 1000.0], &no_drag_physics(0.001, 0.05));
        // motion (v² model): 60² × 0.001 × 1.0 = 3.6 energy
        // vision: 40 × 0.05 × 1.0 = 2.0 energy
        // body: 0 (factor = 0)
        // total drained: 5.6 → energy 100 − 5.6 = 94.4
        assert!((cell.energy - 94.4).abs() < 1e-4, "expected ~94.4, got {}", cell.energy);
        assert!((cell.position[0] - 60.0).abs() < 1e-4);
    }

    #[test]
    fn step_bounce_recomputes_heading() {
        let mut cell = Cell {
            position: [99.0, 0.0],
            velocity: [60.0, 0.0],
            ..base_cell()
        };
        cell.step(1.0, [100.0, 100.0], &no_drag_physics(0.0, 0.0));
        // velocity flipped to (-60, 0), heading should now be π.
        assert!((cell.heading - core::f32::consts::PI).abs() < 1e-4);
    }

    #[test]
    fn step_preserves_heading_when_velocity_zero() {
        let mut cell = Cell {
            heading: 1.5,
            ..base_cell()
        };
        cell.step(1.0, [100.0, 100.0], &no_drag_physics(0.0, 0.0));
        // No movement, no bounce, no angular velocity, heading must persist.
        assert_eq!(cell.heading, 1.5);
    }

    #[test]
    fn step_applies_quadratic_drag() {
        let mut cell = Cell {
            velocity: [10.0, 0.0],
            ..base_cell()
        };
        let physics = PhysicsConfig {
            drag: 0.01,
            angular_drag: 0.0,
            energy_cost_per_v_sq: 0.0,
            angular_energy_cost: 0.0,
            vision_cost_per_radius: 0.0,
            body_cost_factor: 0.0,
        };
        cell.step(1.0, [1000.0, 1000.0], &physics);
        // |v| = 10, drag_dt = 0.01 × 10 × 1 = 0.1
        // velocity[0] -= 0.1 × 10 = 1.0 → final velocity[0] = 9.0
        assert!((cell.velocity[0] - 9.0).abs() < 1e-4, "got {}", cell.velocity[0]);
    }

    #[test]
    fn step_drains_energy_from_rotation() {
        let mut cell = Cell {
            angular_velocity: 2.0,
            ..base_cell()
        };
        let physics = PhysicsConfig {
            drag: 0.0,
            angular_drag: 0.0,
            energy_cost_per_v_sq: 0.0,
            angular_energy_cost: 0.05,
            vision_cost_per_radius: 0.0,
            body_cost_factor: 0.0,
        };
        cell.step(1.0, [1000.0, 1000.0], &physics);
        // body_size²(=1) × ω²(=4) × angular_cost(=0.05) × dt(=1) = 0.2 drained
        assert!((cell.energy - 99.8).abs() < 1e-4, "got {}", cell.energy);
    }

    #[test]
    fn step_rotation_cost_independent_of_linear_cost() {
        // Regression: spinning-in-place was a degenerate local minimum because
        // rotational drain piggy-backed on energy_cost_per_v_sq. Now decoupled.
        let mut cell = Cell {
            angular_velocity: 3.0,
            ..base_cell()
        };
        let physics = PhysicsConfig {
            drag: 0.0,
            angular_drag: 0.0,
            energy_cost_per_v_sq: 99.0,
            angular_energy_cost: 0.0,
            vision_cost_per_radius: 0.0,
            body_cost_factor: 0.0,
        };
        cell.step(1.0, [1000.0, 1000.0], &physics);
        assert!((cell.energy - 100.0).abs() < 1e-4, "got {}", cell.energy);
    }

    #[test]
    fn step_applies_angular_drag() {
        let mut cell = Cell {
            angular_velocity: 1.0,
            ..base_cell()
        };
        let physics = PhysicsConfig {
            drag: 0.0,
            angular_drag: 0.5,
            energy_cost_per_v_sq: 0.0,
            angular_energy_cost: 0.0,
            vision_cost_per_radius: 0.0,
            body_cost_factor: 0.0,
        };
        cell.step(1.0, [1000.0, 1000.0], &physics);
        // angular_velocity *= (1 − 0.5 × 1) = 0.5 → 0.5
        assert!((cell.angular_velocity - 0.5).abs() < 1e-4);
    }

    #[test]
    fn try_eat_within_radius_returns_true_and_adds_energy() {
        let mut cell = Cell {
            energy: 50.0,
            ..base_cell()
        };
        let food = Food { position: [5.0, 0.0] };
        assert!(cell.try_eat(&food, 8.0, 20.0));
        assert_eq!(cell.energy, 70.0);
    }

    #[test]
    fn try_eat_outside_radius_returns_false_and_keeps_energy() {
        let mut cell = Cell {
            energy: 50.0,
            ..base_cell()
        };
        let food = Food { position: [20.0, 0.0] };
        assert!(!cell.try_eat(&food, 8.0, 20.0));
        assert_eq!(cell.energy, 50.0);
    }

    #[test]
    fn crossover_picks_genes_from_either_parent() {
        let mut rng = rand::rng();
        let a = Genome {
            max_speed: 30.0,
            color_hue: 10.0,
            vision_radius: 20.0,
            turn_rate: 1.0,
            body_size: 0.5,
            brain: dummy_brain(),
        };
        let b = Genome {
            max_speed: 90.0,
            color_hue: 200.0,
            vision_radius: 80.0,
            turn_rate: 5.0,
            body_size: 1.5,
            brain: dummy_brain(),
        };
        // Run many crossovers, each gene must be one of the two parent values.
        for _ in 0..100 {
            let c = Genome::crossover(&a, &b, &mut rng);
            assert!(c.max_speed == 30.0 || c.max_speed == 90.0);
            assert!(c.color_hue == 10.0 || c.color_hue == 200.0);
            assert!(c.vision_radius == 20.0 || c.vision_radius == 80.0);
            assert!(c.turn_rate == 1.0 || c.turn_rate == 5.0);
            assert!(c.body_size == 0.5 || c.body_size == 1.5);
        }
    }

    #[test]
    fn hebbian_update_with_zero_reward_is_noop() {
        let mut brain = dummy_brain();
        brain.b1[0] = 0.5;
        brain.b2[0] = 0.7;
        let snapshot_b1 = brain.b1;
        let snapshot_b2 = brain.b2;
        brain.hebbian_update(
            &[1.0; BRAIN_INPUTS],
            &[1.0; BRAIN_HIDDEN],
            &[1.0; BRAIN_OUTPUTS],
            0.0,
            0.1,
        );
        assert_eq!(brain.b1, snapshot_b1);
        assert_eq!(brain.b2, snapshot_b2);
    }

    #[test]
    fn hebbian_update_reinforces_when_reward_positive() {
        let mut brain = dummy_brain();
        // hidden = [1.0; 8], output = [1.0; 2], reward = 1.0, lr = 0.1
        // Δb1[i] = 0.1 × 1.0 × hidden[i] = 0.1
        // Δb2[i] = 0.1 × 1.0 × output[i] = 0.1
        brain.hebbian_update(
            &[0.0; BRAIN_INPUTS],
            &[1.0; BRAIN_HIDDEN],
            &[1.0; BRAIN_OUTPUTS],
            1.0,
            0.1,
        );
        for &b in &brain.b1 {
            assert!((b - 0.1).abs() < 1e-5, "b1 got {}", b);
        }
        for &b in &brain.b2 {
            assert!((b - 0.1).abs() < 1e-5, "b2 got {}", b);
        }
    }

    #[test]
    fn world_map_is_deterministic_for_seed() {
        let a = WorldMap::new(32, 8, [500.0, 500.0], 42);
        let b = WorldMap::new(32, 8, [500.0, 500.0], 42);
        assert_eq!(a.field(), b.field());
    }

    #[test]
    fn world_map_seeds_differ() {
        let a = WorldMap::new(32, 8, [500.0, 500.0], 1);
        let b = WorldMap::new(32, 8, [500.0, 500.0], 2);
        assert_ne!(a.field(), b.field());
    }

    #[test]
    fn world_map_values_in_unit_range() {
        let m = WorldMap::new(32, 8, [500.0, 500.0], 7);
        for &v in m.field() {
            assert!((0.0..=1.0).contains(&v), "out of range: {}", v);
        }
    }

    #[test]
    fn world_map_sample_clamps_to_world_bounds() {
        let m = WorldMap::new(8, 4, [100.0, 100.0], 0);
        // Mimo svět musí vracet hodnotu z hraniční buňky, ne panicovat.
        let inside = m.sample([99.0, 99.0]);
        let outside_pos = m.sample([1e6, 1e6]);
        let outside_neg = m.sample([-1e6, -1e6]);
        assert_eq!(outside_pos, m.field()[m.resolution * m.resolution - 1]);
        assert_eq!(outside_neg, m.field()[0]);
        assert!((0.0..=1.0).contains(&inside));
    }

    #[test]
    fn random_brain_average_thrust_is_positive() {
        // Innate thrust bias musí dělat to, k čemu existuje: random buňky
        // mají ze startu thrust output kladný v průměru, takže se hýbou
        // dopředu místo zacyklení v rozporu mezi turn a thrust.
        let mut rng = rand::rng();
        let n = 200;
        let zero_inputs = [0.0_f32; BRAIN_INPUTS];
        let mut sum = 0.0_f64;
        let mut count_positive = 0;
        for _ in 0..n {
            let brain = Brain::random(&mut rng);
            let thrust = brain.forward(&zero_inputs)[1];
            sum += thrust as f64;
            if thrust > 0.0 {
                count_positive += 1;
            }
        }
        let mean = sum / n as f64;
        assert!(mean > 0.3, "expected mean thrust > 0.3, got {}", mean);
        assert!(
            count_positive > n * 3 / 4,
            "expected >75% positive, got {}/{}",
            count_positive,
            n
        );
    }

    #[test]
    fn brain_forward_zero_weights_outputs_tanh_of_output_biases() {
        // Zero weights kill signal flow at both layers — output equals tanh(b2),
        // independent of b1 (the hidden activations get zeroed by w2).
        let mut b2 = [0.0_f32; BRAIN_OUTPUTS];
        b2[0] = 0.5;
        b2[1] = -0.5;
        let brain = Brain {
            w1: [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN],
            b1: [0.7; BRAIN_HIDDEN],
            w2: [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
            b2,
        };
        let outputs = brain.forward(&[0.0; BRAIN_INPUTS]);
        assert!((outputs[0] - 0.5_f32.tanh()).abs() < 1e-6);
        assert!((outputs[1] - (-0.5_f32).tanh()).abs() < 1e-6);
    }
}
