//! Bioscape simulation core — pure logic, no rendering.
//!
//! Keeping this layer free of Bevy types lets us drive the same world
//! from a windowed renderer (`main.rs`) or a headless batch run later.

use rand::Rng;

#[derive(Debug, Clone, Copy)]
pub struct Cell {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub energy: f32,
}

impl Cell {
    pub fn random(rng: &mut impl Rng, world_half_extent: f32) -> Self {
        Self {
            position: [
                rng.random_range(-world_half_extent..world_half_extent),
                rng.random_range(-world_half_extent..world_half_extent),
            ],
            velocity: [
                rng.random_range(-60.0..60.0),
                rng.random_range(-60.0..60.0),
            ],
            energy: 100.0,
        }
    }

    pub fn step(&mut self, dt: f32, world_half_extent: f32) {
        self.position[0] += self.velocity[0] * dt;
        self.position[1] += self.velocity[1] * dt;

        // Reflect off the world boundary so cells stay visible.
        if self.position[0].abs() > world_half_extent {
            self.velocity[0] = -self.velocity[0];
            self.position[0] = self.position[0].clamp(-world_half_extent, world_half_extent);
        }
        if self.position[1].abs() > world_half_extent {
            self.velocity[1] = -self.velocity[1];
            self.position[1] = self.position[1].clamp(-world_half_extent, world_half_extent);
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
}
