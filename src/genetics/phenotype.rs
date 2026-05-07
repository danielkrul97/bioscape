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
    /// Sprint 87: drain rate koeficient pro thermal_optimum penalty.
    /// `dev² × penalty × dt` kde dev = (temp − optimum) / 13.0 (normalized
    /// half-range). Default 1.0; tests mohou override 0.0 pro disable
    /// (např. `step_gpu_matches_cpu` parita — GPU shader nepočítá penalty).
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

/// Runtime tělesný tvar buňky. Inicializuje se z `Genome` při spawnu /
/// reprodukci (template) a může se měnit za běhu života přes `apply_morph`
/// (řízeno brain output[3..6]). **Genotyp/fenotyp split**: runtime morph
/// modifikuje `Phenotype`, ne `Genome`. Dítě dostane svůj fresh phenotype
/// z rodičovského genomu — žádný Lamarckismus.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Phenotype {
    pub body_length: f32,
    pub body_width: f32,
    /// Sprint 34: vertikální rozměr ellipsoidu.
    pub body_height: f32,
    /// Sprint 121: per-spike runtime stav. `length` je morphable přes brain
    /// output[5] (Sprint 122 aggregate signal); `azimuth_offset`,
    /// `elevation_offset`, `complexity` jsou snapshot z genomu (pure-genetic,
    /// žádný runtime morph). `spike_count` je snapshot — discrete add/remove
    /// se děje jen na reprodukci (mutace).
    #[serde(default = "default_spikes")]
    pub spikes: [Spike; SPIKE_SLOTS],
    #[serde(default = "default_spike_count")]
    pub spike_count: u8,
    /// Sprint 41: snapshot z genomu, runtime morph zatím neexistuje.
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

    /// Sprint 121: primary spike length (slot 0). Pre-S121 callers které
    /// četly `phenotype.spike_length` čtou tohle.
    pub fn primary_spike_length(&self) -> f32 {
        if self.spike_count > 0 {
            self.spikes[0].length
        } else {
            0.0
        }
    }

    /// Sprint 121: sum length přes všechny aktivní spiky. V S121 (spike_count=1)
    /// identické s `primary_spike_length`. Sprint 122+ začne divergovat.
    pub fn total_spike_length(&self) -> f32 {
        let n = self.spike_count.min(SPIKE_SLOTS as u8) as usize;
        self.spikes[..n].iter().map(|s| s.length).sum()
    }

    pub fn active_spikes(&self) -> &[Spike] {
        let n = self.spike_count.min(SPIKE_SLOTS as u8) as usize;
        &self.spikes[..n]
    }

    /// Sprint 124: aggregate spike maintenance cost factor pro GPU step shader
    /// aux[0]. CPU energy drain semantika:
    /// `total_spike_cost_factor × SPIKE_COST_PER_SEC × dt_eff`. S spike_count=1
    /// a complexity=0 redukuje na pre-S121 `spike_length`.
    pub fn total_spike_cost_factor(&self) -> f32 {
        let mut acc = 0.0;
        for spike in self.active_spikes() {
            acc += spike.length * spike_complexity_cost_factor(spike.complexity);
        }
        acc
    }

    /// Sprint 124: primary spike attack factor pro GPU predate shader
    /// `spike_lengths[i]` semantiku. `length × attack_complexity_factor`
    /// pro slot 0 (single-direction predicate). Multi-spike non-primary
    /// sloty na GPU nedostávají bonus — CPU path je multi-spike-faithful.
    pub fn primary_spike_attack_factor(&self) -> f32 {
        if self.spike_count == 0 {
            return 0.0;
        }
        let s = self.spikes[0];
        s.length * spike_complexity_attack_factor(s.complexity)
    }

    /// Proxy pro circular-collision codepaths (eat radius, broad phase).
    /// Sprint 34: aritmetický průměr 3 os; když length=width=height=s, dostane s
    /// — backward compat s pre-Sprint-34 izotropním tělem.
    pub fn effective_radius(&self) -> f32 {
        (self.body_length + self.body_width + self.body_height) / 3.0
    }

    /// Sprint 41: nejvyšší ze tří os — pro broad-phase bucketing eat zóny,
    /// kde ellipsoid může extending podél long axis a sféra `effective_radius`
    /// by ho missnula.
    pub fn max_axis(&self) -> f32 {
        self.body_length.max(self.body_width).max(self.body_height)
    }

    /// Sprint 34: 3D volume = length × width × height. Když length=width=height
    /// =s, dostane s³. Pro pre-Sprint-34 srovnatelnost: tělo s body_height=1
    /// dává area_pre × 1 = area_pre, tj. backward compat při height=1.
    pub fn volume(&self) -> f32 {
        self.body_length * self.body_width * self.body_height
    }

    /// Aplikuje 4 brain morph signály na dimenze tvaru. Signály pod
    /// `MORPH_ACTIVATION_THRESHOLD` v absolutní hodnotě jsou deadzonovány
    /// (no-op) — random brain noise neovlivní phenotype, jen deliberátní
    /// signály z trénovaného brainu. Vrací sumu |Δ| napříč dimenzemi (po
    /// clampu) pro výpočet morph cost.
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

        // Sprint 121: morph[3] aggregate spike length signal — proporčně přes
        // všechny aktivní spiky (per-spike rate ∝ length / sum_lengths). S121
        // s spike_count=1 redukuje na pre-S121 single spike. S122 multi-spike
        // smysluplně rozvrhuje delta.
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
            // Sprint 123: high-complexity spike morphuje pomaleji — geometric
            // structure je commitment, ne behavioral knob. complexity=1 → 50 %
            // rate, complexity=0 → 100 % (pre-S123 sémantika).
            let rate_factor = 1.0 - 0.5 * self.spikes[i].complexity.clamp(0.0, 1.0);
            let delta = raw_ds * weight * rate_factor;
            let new_len = (self.spikes[i].length + delta)
                .clamp(MIN_SPIKE_LENGTH, MAX_SPIKE_LENGTH);
            total_delta += (new_len - self.spikes[i].length).abs();
            self.spikes[i].length = new_len;
        }
        total_delta
    }
}
