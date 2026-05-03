//! Bioscape simulation core — pure logic, no rendering.
//!
//! Keeping this layer free of Bevy types lets us drive the same world
//! from a windowed renderer (`main.rs`) or a headless batch run later.

use core::f32::consts::TAU;
use rand::Rng;

const HUE_RANGE: f32 = 360.0;
const MIN_SPEED: f32 = 1.0;
const MIN_VISION: f32 = 1.0;
const MIN_TURN_RATE: f32 = 0.1;
const MIN_BODY_SIZE: f32 = 0.3;
pub const INITIAL_ENERGY: f32 = 100.0;
pub const BRAIN_INPUTS: usize = 9;
pub const BRAIN_HIDDEN: usize = 8;
pub const BRAIN_OUTPUTS: usize = 2;

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
        Self { w1, b1, w2, b2 }
    }

    pub fn forward(&self, inputs: &[f32; BRAIN_INPUTS]) -> [f32; BRAIN_OUTPUTS] {
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
        out
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
    pub genome: Genome,
}

impl Cell {
    pub fn random(rng: &mut impl Rng, world_half: [f32; 2]) -> Self {
        let genome = Genome::random(rng);
        Self::from_genome(rng, genome, world_half)
    }

    pub fn from_genome(rng: &mut impl Rng, genome: Genome, world_half: [f32; 2]) -> Self {
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
        self.energy -= bs * bs * av * av * physics.energy_cost_per_v_sq * dt;
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
            vision_cost_per_radius: vision_cost,
            body_cost_factor: 0.0,
        }
    }

    #[test]
    fn step_drains_energy_from_motion_and_vision() {
        let mut cell = Cell {
            position: [0.0, 0.0],
            velocity: [60.0, 0.0],
            angular_velocity: 0.0,
            energy: 100.0,
            heading: 0.0,
            genome: dummy_genome(),
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
            angular_velocity: 0.0,
            energy: 100.0,
            heading: 0.0,
            genome: dummy_genome(),
        };
        cell.step(1.0, [100.0, 100.0], &no_drag_physics(0.0, 0.0));
        // velocity flipped to (-60, 0), heading should now be π.
        assert!((cell.heading - core::f32::consts::PI).abs() < 1e-4);
    }

    #[test]
    fn step_preserves_heading_when_velocity_zero() {
        let mut cell = Cell {
            position: [0.0, 0.0],
            velocity: [0.0, 0.0],
            angular_velocity: 0.0,
            energy: 100.0,
            heading: 1.5,
            genome: dummy_genome(),
        };
        cell.step(1.0, [100.0, 100.0], &no_drag_physics(0.0, 0.0));
        // No movement, no bounce, no angular velocity, heading must persist.
        assert_eq!(cell.heading, 1.5);
    }

    #[test]
    fn step_applies_quadratic_drag() {
        let mut cell = Cell {
            position: [0.0, 0.0],
            velocity: [10.0, 0.0],
            angular_velocity: 0.0,
            energy: 100.0,
            heading: 0.0,
            genome: dummy_genome(),
        };
        let physics = PhysicsConfig {
            drag: 0.01,
            angular_drag: 0.0,
            energy_cost_per_v_sq: 0.0,
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
            position: [0.0, 0.0],
            velocity: [0.0, 0.0],
            angular_velocity: 2.0,
            energy: 100.0,
            heading: 0.0,
            genome: dummy_genome(),
        };
        let physics = PhysicsConfig {
            drag: 0.0,
            angular_drag: 0.0,
            energy_cost_per_v_sq: 0.001,
            vision_cost_per_radius: 0.0,
            body_cost_factor: 0.0,
        };
        cell.step(1.0, [1000.0, 1000.0], &physics);
        // body_size²(=1) × ω²(=4) × cost(=0.001) × dt(=1) = 0.004 drained
        assert!((cell.energy - 99.996).abs() < 1e-4, "got {}", cell.energy);
    }

    #[test]
    fn step_applies_angular_drag() {
        let mut cell = Cell {
            position: [0.0, 0.0],
            velocity: [0.0, 0.0],
            angular_velocity: 1.0,
            energy: 100.0,
            heading: 0.0,
            genome: dummy_genome(),
        };
        let physics = PhysicsConfig {
            drag: 0.0,
            angular_drag: 0.5,
            energy_cost_per_v_sq: 0.0,
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
            position: [0.0, 0.0],
            velocity: [0.0, 0.0],
            angular_velocity: 0.0,
            energy: 50.0,
            heading: 0.0,
            genome: dummy_genome(),
        };
        let food = Food { position: [5.0, 0.0] };
        assert!(cell.try_eat(&food, 8.0, 20.0));
        assert_eq!(cell.energy, 70.0);
    }

    #[test]
    fn try_eat_outside_radius_returns_false_and_keeps_energy() {
        let mut cell = Cell {
            position: [0.0, 0.0],
            velocity: [0.0, 0.0],
            angular_velocity: 0.0,
            energy: 50.0,
            heading: 0.0,
            genome: dummy_genome(),
        };
        let food = Food { position: [20.0, 0.0] };
        assert!(!cell.try_eat(&food, 8.0, 20.0));
        assert_eq!(cell.energy, 50.0);
    }

    #[test]
    fn brain_forward_zero_weights_outputs_tanh_of_output_biases() {
        // Zero weights kill signal flow at both layers — output equals tanh(b2),
        // independent of b1 (the hidden activations get zeroed by w2).
        let brain = Brain {
            w1: [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN],
            b1: [0.7; BRAIN_HIDDEN],
            w2: [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
            b2: [0.5, -0.5],
        };
        let outputs = brain.forward(&[0.0; BRAIN_INPUTS]);
        assert!((outputs[0] - 0.5_f32.tanh()).abs() < 1e-6);
        assert!((outputs[1] - (-0.5_f32).tanh()).abs() < 1e-6);
    }
}
