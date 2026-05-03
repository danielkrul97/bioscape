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
pub const INITIAL_ENERGY: f32 = 100.0;
pub const BRAIN_INPUTS: usize = 6;
pub const BRAIN_OUTPUTS: usize = 2;

#[derive(Debug, Clone, Copy)]
pub struct Brain {
    pub weights: [[f32; BRAIN_INPUTS]; BRAIN_OUTPUTS],
    pub biases: [f32; BRAIN_OUTPUTS],
}

impl Brain {
    pub fn random(rng: &mut impl Rng) -> Self {
        let mut weights = [[0.0; BRAIN_INPUTS]; BRAIN_OUTPUTS];
        let mut biases = [0.0; BRAIN_OUTPUTS];
        for (row, bias) in weights.iter_mut().zip(biases.iter_mut()) {
            for w in row.iter_mut() {
                *w = gaussian(rng);
            }
            *bias = gaussian(rng);
        }
        Self { weights, biases }
    }

    pub fn forward(&self, inputs: &[f32; BRAIN_INPUTS]) -> [f32; BRAIN_OUTPUTS] {
        let mut out = [0.0; BRAIN_OUTPUTS];
        for ((output, row), &bias) in out
            .iter_mut()
            .zip(self.weights.iter())
            .zip(self.biases.iter())
        {
            let mut sum = bias;
            for (&w, &x) in row.iter().zip(inputs.iter()) {
                sum += w * x;
            }
            *output = sum.tanh();
        }
        out
    }

    pub fn mutate(&self, rng: &mut impl Rng, sigma: f32) -> Self {
        let mut weights = self.weights;
        let mut biases = self.biases;
        for (row, bias) in weights.iter_mut().zip(biases.iter_mut()) {
            for w in row.iter_mut() {
                *w += gaussian(rng) * sigma;
            }
            *bias += gaussian(rng) * sigma;
        }
        Self { weights, biases }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MutationConfig {
    pub sigma_speed: f32,
    pub sigma_hue: f32,
    pub sigma_vision: f32,
    pub sigma_turn_rate: f32,
    pub sigma_brain: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Genome {
    pub max_speed: f32,
    pub color_hue: f32,
    pub vision_radius: f32,
    pub turn_rate: f32,
    pub brain: Brain,
}

impl Genome {
    pub fn random(rng: &mut impl Rng) -> Self {
        Self {
            max_speed: rng.random_range(30.0..90.0),
            color_hue: rng.random_range(0.0..HUE_RANGE),
            vision_radius: rng.random_range(20.0..80.0),
            turn_rate: rng.random_range(1.0..5.0),
            brain: Brain::random(rng),
        }
    }

    pub fn mutate(&self, rng: &mut impl Rng, cfg: &MutationConfig) -> Self {
        Self {
            max_speed: (self.max_speed + gaussian(rng) * cfg.sigma_speed).max(MIN_SPEED),
            color_hue: (self.color_hue + gaussian(rng) * cfg.sigma_hue).rem_euclid(HUE_RANGE),
            vision_radius: (self.vision_radius + gaussian(rng) * cfg.sigma_vision).max(MIN_VISION),
            turn_rate: (self.turn_rate + gaussian(rng) * cfg.sigma_turn_rate).max(MIN_TURN_RATE),
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
pub struct Cell {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
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
            energy: INITIAL_ENERGY,
            heading: direction,
            genome,
        }
    }

    pub fn step(
        &mut self,
        dt: f32,
        world_half: [f32; 2],
        energy_cost_per_distance: f32,
        vision_cost_per_radius: f32,
    ) {
        let dx = self.velocity[0] * dt;
        let dy = self.velocity[1] * dt;
        self.energy -= (dx * dx + dy * dy).sqrt() * energy_cost_per_distance;
        self.energy -= self.genome.vision_radius * vision_cost_per_radius * dt;

        self.position[0] += dx;
        self.position[1] += dy;

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
            weights: [[0.0; BRAIN_INPUTS]; BRAIN_OUTPUTS],
            biases: [0.0; BRAIN_OUTPUTS],
        }
    }

    fn dummy_genome() -> Genome {
        Genome {
            max_speed: 60.0,
            color_hue: 0.0,
            vision_radius: 40.0,
            turn_rate: 2.5,
            brain: dummy_brain(),
        }
    }

    fn zero_cfg() -> MutationConfig {
        MutationConfig {
            sigma_speed: 0.0,
            sigma_hue: 0.0,
            sigma_vision: 0.0,
            sigma_turn_rate: 0.0,
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
            brain: Brain {
                weights: [[1.0; BRAIN_INPUTS]; BRAIN_OUTPUTS],
                biases: [0.5; BRAIN_OUTPUTS],
            },
        };
        let m = g.mutate(&mut rng, &zero_cfg());
        assert_eq!(m.max_speed, 50.0);
        assert_eq!(m.color_hue, 120.0);
        assert_eq!(m.vision_radius, 40.0);
        assert_eq!(m.turn_rate, 2.5);
        assert_eq!(m.brain.weights, g.brain.weights);
        assert_eq!(m.brain.biases, g.brain.biases);
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
            sigma_brain: 10.0,
        };
        for _ in 0..1000 {
            let m = g.mutate(&mut rng, &cfg);
            assert!(m.max_speed >= MIN_SPEED);
            assert!(m.color_hue >= 0.0 && m.color_hue < HUE_RANGE);
            assert!(m.vision_radius >= MIN_VISION);
            assert!(m.turn_rate >= MIN_TURN_RATE);
        }
    }

    #[test]
    fn step_drains_energy_from_movement_and_vision() {
        let mut cell = Cell {
            position: [0.0, 0.0],
            velocity: [60.0, 0.0],
            energy: 100.0,
            heading: 0.0,
            genome: dummy_genome(),
        };
        cell.step(1.0, [1000.0, 1000.0], 0.1, 0.05);
        // movement: 60 distance × 0.1 = 6 energy
        // vision: 40 × 0.05 × 1.0 = 2 energy
        // total: 100 - 8 = 92
        assert!((cell.energy - 92.0).abs() < 1e-4, "expected ~92, got {}", cell.energy);
        assert!((cell.position[0] - 60.0).abs() < 1e-4);
    }

    #[test]
    fn step_bounce_recomputes_heading() {
        let mut cell = Cell {
            position: [99.0, 0.0],
            velocity: [60.0, 0.0],
            energy: 100.0,
            heading: 0.0,
            genome: dummy_genome(),
        };
        cell.step(1.0, [100.0, 100.0], 0.0, 0.0);
        // velocity flipped to (-60, 0), heading should now be π.
        assert!((cell.heading - core::f32::consts::PI).abs() < 1e-4);
    }

    #[test]
    fn step_preserves_heading_when_velocity_zero() {
        let mut cell = Cell {
            position: [0.0, 0.0],
            velocity: [0.0, 0.0],
            energy: 100.0,
            heading: 1.5,
            genome: dummy_genome(),
        };
        cell.step(1.0, [100.0, 100.0], 0.0, 0.0);
        // No movement, no bounce, heading must persist.
        assert_eq!(cell.heading, 1.5);
    }

    #[test]
    fn try_eat_within_radius_returns_true_and_adds_energy() {
        let mut cell = Cell {
            position: [0.0, 0.0],
            velocity: [0.0, 0.0],
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
            energy: 50.0,
            heading: 0.0,
            genome: dummy_genome(),
        };
        let food = Food { position: [20.0, 0.0] };
        assert!(!cell.try_eat(&food, 8.0, 20.0));
        assert_eq!(cell.energy, 50.0);
    }

    #[test]
    fn brain_forward_zero_inputs_outputs_tanh_of_biases() {
        let brain = Brain {
            weights: [[0.0; BRAIN_INPUTS]; BRAIN_OUTPUTS],
            biases: [0.5, -0.5],
        };
        let outputs = brain.forward(&[0.0; BRAIN_INPUTS]);
        assert!((outputs[0] - 0.5_f32.tanh()).abs() < 1e-6);
        assert!((outputs[1] - (-0.5_f32).tanh()).abs() < 1e-6);
    }
}
