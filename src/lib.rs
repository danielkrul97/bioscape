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
