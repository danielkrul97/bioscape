use serde::{Deserialize, Serialize};

use crate::*;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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
    /// Drain rate for the thermal-optimum penalty: `dev² × penalty × dt`
    /// where `dev = (temp − optimum) / 13.0` (normalized half-range).
    /// Tests override this to 0 when comparing against the GPU step shader,
    /// which does not compute the penalty.
    pub thermal_optimum_penalty: f32,
}

pub const PHYSICS_CONFIG: PhysicsConfig = PhysicsConfig {
    drag: DRAG_COEFFICIENT,
    angular_drag: ANGULAR_DRAG,
    energy_cost_per_v_sq: ENERGY_COST_PER_V_SQ,
    angular_energy_cost: ANGULAR_ENERGY_COST,
    vision_cost_per_radius: VISION_COST_PER_RADIUS,
    body_cost_factor: BODY_COST_FACTOR,
    thermal_optimum_penalty: THERMAL_OPTIMUM_PENALTY,
};

/// Runtime body shape of a cell. Seeded from `Genome` at spawn / reproduction
/// and mutated during life by `apply_morph` (driven by brain output[3..6]).
/// Genotype/phenotype split: runtime morph touches `Phenotype` only, never
/// `Genome` — children receive a fresh phenotype from the parent's genome,
/// no Lamarckism.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Phenotype {
    pub body_length: f32,
    pub body_width: f32,
    pub body_height: f32,
    /// Per-spike runtime state. `length` is morphable via the aggregate
    /// brain output (see `apply_spike_morph`); `azimuth_offset`,
    /// `elevation_offset`, and `complexity` are pure-genetic snapshots.
    /// `spike_count` is also a snapshot — slots only activate / deactivate
    /// at reproduction via mutation.
    #[serde(default = "default_spikes")]
    pub spikes: [Spike; SPIKE_SLOTS],
    #[serde(default = "default_spike_count")]
    pub spike_count: u8,
    /// Genome snapshot — no runtime morph for shell yet.
    pub shell_thickness: f32,
}

impl Phenotype {
    pub fn from_genome(genome: &Genome) -> Self {
        Self {
            body_length: genome.body_length,
            body_width: genome.body_width,
            body_height: genome.body_height,
            spikes: genome.spikes,
            spike_count: genome.spike_count,
            shell_thickness: genome.shell_thickness,
        }
    }

    /// Length of slot 0, or 0 if no spikes are active.
    pub fn primary_spike_length(&self) -> f32 {
        if self.spike_count > 0 {
            self.spikes[0].length
        } else {
            0.0
        }
    }

    pub fn total_spike_length(&self) -> f32 {
        let n = self.spike_count.min(SPIKE_SLOTS as u8) as usize;
        self.spikes[..n].iter().map(|s| s.length).sum()
    }

    pub fn active_spikes(&self) -> &[Spike] {
        let n = self.spike_count.min(SPIKE_SLOTS as u8) as usize;
        &self.spikes[..n]
    }

    /// Aggregate maintenance-cost factor passed to the GPU step shader as
    /// `aux[0]`. Energy drain is `factor × SPIKE_COST_PER_SEC × dt_eff`.
    pub fn total_spike_cost_factor(&self) -> f32 {
        let mut acc = 0.0;
        for spike in self.active_spikes() {
            acc += spike.length * spike_complexity_cost_factor(spike.complexity);
        }
        acc
    }

    /// Slot-0 attack factor (`length × attack_complexity_factor`) consumed
    /// by the GPU predate shader. The GPU only bonuses the primary spike;
    /// the CPU path applies the per-spike bonus across all active slots.
    pub fn primary_spike_attack_factor(&self) -> f32 {
        if self.spike_count == 0 {
            return 0.0;
        }
        let s = self.spikes[0];
        s.length * spike_complexity_attack_factor(s.complexity)
    }

    /// Arithmetic mean of the three axes — proxy used by circular-collision
    /// code paths (eat radius, broad phase). For an isotropic body
    /// (length = width = height = s) it returns s.
    pub fn effective_radius(&self) -> f32 {
        (self.body_length + self.body_width + self.body_height) / 3.0
    }

    /// Largest of the three axes. Broad-phase bucketing for the eat zone
    /// uses this because an elongated ellipsoid can extend past the
    /// `effective_radius` sphere along its long axis.
    pub fn max_axis(&self) -> f32 {
        self.body_length.max(self.body_width).max(self.body_height)
    }

    pub fn volume(&self) -> f32 {
        self.body_length * self.body_width * self.body_height
    }

    /// Sprint 202 — inertial mass (`volume × MASS_DENSITY`). Replaces the
    /// pre-S202 mass proxy `effective_radius` in motor, collision-bond, and
    /// brownian phases. See `params::physics::MASS_DENSITY` for tuning notes.
    pub fn mass(&self) -> f32 {
        (self.volume() * MASS_DENSITY).max(0.01)
    }

    /// Box-proxy surface area `2(lw + lh + wh)` — consistent with `volume()`'s
    /// box proxy `lwh`. Drives the surface-area-to-volume uptake limit.
    pub fn surface_area(&self) -> f32 {
        let l = self.body_length;
        let w = self.body_width;
        let h = self.body_height;
        2.0 * (l * w + l * h + w * h)
    }

    /// Sprint 203 — surface-area-to-volume uptake multiplier on ingested food.
    /// `clamp(sqrt((surface/volume) / SA_V_UPTAKE_REF), MIN, MAX)`. A compact
    /// (low SA:V) body absorbs less per bite; capped at 1.0 so high-SA:V shapes
    /// reach neutral without an energy bonus. See `params::physics` for the
    /// rationale and tuning notes.
    pub fn sa_v_uptake_factor(&self) -> f32 {
        let sa_v = self.surface_area() / self.volume().max(1e-6);
        (sa_v / SA_V_UPTAKE_REF)
            .sqrt()
            .clamp(SA_V_UPTAKE_MIN, SA_V_UPTAKE_MAX)
    }

    /// Sprint 203 — volume-scaled energy storage ceiling. See `ENERGY_STORAGE_*`.
    pub fn max_energy_capacity(&self) -> f32 {
        (self.volume() * ENERGY_STORAGE_DENSITY).max(ENERGY_STORAGE_FLOOR)
    }

    /// Applies the four morph signals from the brain to body dimensions.
    /// Signals below `MORPH_ACTIVATION_THRESHOLD` in absolute value are
    /// dead-zoned to zero so untrained random-brain noise doesn't drift the
    /// phenotype — only deliberate output reshapes the body. Returns the
    /// sum of post-clamp `|Δ|` across all dimensions for the morph cost.
    pub fn apply_morph(&mut self, morph: [f32; 4], rate: f32, dt: f32) -> f32 {
        let gate = |s: f32| -> f32 {
            if s.abs() < MORPH_ACTIVATION_THRESHOLD {
                0.0
            } else {
                s
            }
        };
        let raw_dl = gate(morph[0]) * rate * dt;
        let raw_dw = gate(morph[1]) * rate * dt;
        let raw_dh = gate(morph[2]) * rate * dt;
        let raw_ds = gate(morph[3]) * rate * dt;

        let new_len = (self.body_length + raw_dl).clamp(MIN_BODY_LENGTH, MAX_BODY_LENGTH);
        let new_wid = (self.body_width + raw_dw).clamp(MIN_BODY_WIDTH, MAX_BODY_WIDTH);
        let new_hgt = (self.body_height + raw_dh).clamp(MIN_BODY_HEIGHT, MAX_BODY_HEIGHT);

        let actual_dl = (new_len - self.body_length).abs();
        let actual_dw = (new_wid - self.body_width).abs();
        let actual_dh = (new_hgt - self.body_height).abs();

        self.body_length = new_len;
        self.body_width = new_wid;
        self.body_height = new_hgt;

        // morph[3] is an aggregate signal split across active spikes
        // proportionally to their current length.
        let actual_ds = self.apply_spike_morph(raw_ds);

        actual_dl + actual_dw + actual_dh + actual_ds
    }

    fn apply_spike_morph(&mut self, raw_ds: f32) -> f32 {
        let n = self.spike_count.min(SPIKE_SLOTS as u8) as usize;
        if n == 0 || raw_ds == 0.0 {
            return 0.0;
        }
        let sum_lengths: f32 = self.spikes[..n].iter().map(|s| s.length).sum();
        let mut total_delta = 0.0;
        for i in 0..n {
            let weight = if sum_lengths > f32::EPSILON {
                self.spikes[i].length / sum_lengths
            } else {
                1.0 / n as f32
            };
            // High-complexity spikes morph slower — geometric structure is
            // a commitment, not a behavioral knob. complexity=1 → 50 % rate,
            // complexity=0 → 100 %.
            let rate_factor = 1.0 - 0.5 * self.spikes[i].complexity.clamp(0.0, 1.0);
            let delta = raw_ds * weight * rate_factor;
            let new_len = (self.spikes[i].length + delta).clamp(MIN_SPIKE_LENGTH, MAX_SPIKE_LENGTH);
            total_delta += (new_len - self.spikes[i].length).abs();
            self.spikes[i].length = new_len;
        }
        total_delta
    }
}
