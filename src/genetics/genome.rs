use core::f32::consts::TAU;

use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::*;

pub const MUTATION_CONFIG: MutationConfig = MutationConfig {
    sigma_speed: 3.0,
    sigma_hue: 5.0,
    sigma_vision: 3.0,
    sigma_turn_rate: 0.3,
    sigma_body_length: 0.05,
    sigma_body_width: 0.05,
    sigma_body_height: 0.05,
    sigma_spike_length: 0.03,
    sigma_shell: 0.03,
    sigma_brain: 0.2,
    adhesion_flip_rate: ADHESION_MUTATION_RATE,
    // Bond physics drift slower than body params: bond_active_frac runs
    // sub-percent in long smokes, so selection has a weak signal — larger
    // sigma would degenerate into random walk.
    sigma_bond_stiffness: 0.3,
    sigma_bond_damping: 0.05,
    // Structural brain mutations: ~2 % rates land roughly one structural
    // change per cell every 30–50 gens, giving selection time to evaluate.
    add_neuron_rate: 0.02,
    split_link_rate: 0.02,
    remove_neuron_rate: 0.01,
    // ~1.7 % FOV range / gen — slower than body genes so evolution finds an
    // optimum before drifting into MIN_VISION_FOV.
    sigma_vision_fov: 0.05,
    // ~1.9 % range / gen across the 26-unit TOP–BOTTOM band; slow enough
    // for selection to bias depth-coupled niches rather than random walk.
    sigma_thermal_optimum: 0.5,
    sigma_carnivore_score: 0.02,
    sigma_sensor_gain: 0.04,
    // ~5 % of cells per gen flip spike_count by ±1 (clamped to [0, 5]).
    // 0.05 rad ≈ 3° azimuth/elevation drift per gen.
    spike_count_mutation_rate: 0.05,
    sigma_spike_orientation: 0.05,
    // 2 % range / gen on [0, 1] — slow enough for selection to settle
    // around an intermediate sweet-spot instead of running to the extremes.
    sigma_spike_complexity: 0.02,
    // Newly activated non-primary spike slots start at length = 0; this
    // sigma drives them away from zero. Same magnitude as `sigma_spike_length`.
    sigma_spike_length_secondary: 0.03,
    // Sprint 137: drift activated once the GPU pipeline consumes per-cell
    // rates. `sigma_learning_rate` slow (~5 % of range per gen) — Hebbian
    // rate is multiplicative on every reward, so cells over-shooting the
    // sweet spot lose quickly. `sigma_trace_decay` an order of magnitude
    // larger because the [0.1, 5.0] range spans two octaves.
    sigma_learning_rate: 0.001,
    sigma_trace_decay: 0.05,
    // Sprint 144 ships `NeuronModel` plumbing but keeps flip off so the
    // entire population stays Perceptron until S150 turns it on.
    model_flip_rate: 0.0,
    // Sprint 148: STDP drift sigmas default off — S149 activates with
    // reward modulation once the integration is wired through CPU/GPU.
    sigma_stdp_a: 0.0,
    sigma_stdp_tau: 0.0,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MutationConfig {
    pub sigma_speed: f32,
    pub sigma_hue: f32,
    pub sigma_vision: f32,
    pub sigma_turn_rate: f32,
    pub sigma_body_length: f32,
    pub sigma_body_width: f32,
    pub sigma_body_height: f32,
    pub sigma_spike_length: f32,
    pub sigma_shell: f32,
    pub sigma_brain: f32,
    /// Probability of flipping `adhesion_type` per child.
    pub adhesion_flip_rate: f32,
    pub sigma_bond_stiffness: f32,
    pub sigma_bond_damping: f32,
    /// Probability of an `add_neuron` structural mutation per child. Set to
    /// 0 to disable topology evolution; non-zero values diverge the RNG
    /// trajectory from a no-structural-mutation baseline.
    pub add_neuron_rate: f32,
    /// NEAT-style split-link rate: pick a random active link (|w| above
    /// `SPLIT_LINK_THRESHOLD`) and insert a hidden node between its
    /// endpoints. Topology-preserving — forward output is unchanged at the
    /// moment of insertion.
    pub split_link_rate: f32,
    /// Pruning rate that zeros a random hidden neuron — balances the growth
    /// driven by `add_neuron_rate` and `split_link_rate`.
    pub remove_neuron_rate: f32,
    /// Set to 0 to keep all cells at `INITIAL_VISION_FOV`; non-zero values
    /// engage cone filtering in sensor gather, opening the info-loss vs.
    /// energy-cost trade-off.
    pub sigma_vision_fov: f32,
    pub sigma_thermal_optimum: f32,
    /// Slower than most body genes — diet shifts need a food-availability
    /// signal (selection via `eat_efficiency × food_kind`), which itself
    /// changes only gradually.
    pub sigma_carnivore_score: f32,
    /// Per-category gain drift. Specialization needs a cluster-pooling
    /// signal, so selection is conditional on bond presence.
    pub sigma_sensor_gain: f32,
    /// Probability of `spike_count` flipping by ±1 (clamped to
    /// `[0, SPIKE_SLOTS]`). When 0, no RNG draw happens here — keeping the
    /// RNG sequence byte-identical with the pre-multi-spike baseline.
    pub spike_count_mutation_rate: f32,
    /// Gaussian sigma for spike azimuth/elevation offsets in radians/gen.
    pub sigma_spike_orientation: f32,
    /// Gaussian sigma for spike complexity ∈ [0, 1]. When 0, complexity stays
    /// pinned and the `COMPLEXITY_*_GAIN` multipliers degenerate to 1.0.
    pub sigma_spike_complexity: f32,
    /// Length-drift sigma for slots 1..SPIKE_SLOTS. Slot 0 uses
    /// `sigma_spike_length`. When 0, non-primary slots only ever drift via
    /// activation/deactivation through `spike_count_mutation_rate`.
    pub sigma_spike_length_secondary: f32,
    /// Sprint 136: gaussian sigma for per-cell `learning_rate`. When 0, the
    /// rate is inherited verbatim — no RNG draw, byte-identical with the
    /// pre-S136 baseline that read `LEARNING_RATE` const at every call site.
    pub sigma_learning_rate: f32,
    /// Sprint 136: gaussian sigma for per-cell `trace_decay_per_sec`. Same
    /// gating: 0 = no drift, no RNG draw.
    pub sigma_trace_decay: f32,
    /// Sprint 144: per-child probability that `Genome::neuron_model` flips
    /// (Perceptron ↔ Izhikevich). Short-circuited so a zero rate skips the
    /// RNG draw — keeps the pre-S144 RNG sequence intact.
    pub model_flip_rate: f32,
    /// Sprint 148: gaussian sigma for `stdp_a_plus` / `stdp_a_minus`.
    /// Single knob drives both magnitudes (LTP / LTD ratio stays roughly
    /// constant). Default 0 → no drift, no RNG draw.
    pub sigma_stdp_a: f32,
    /// Sprint 148: gaussian sigma for `stdp_tau_ticks`.
    pub sigma_stdp_tau: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Genome {
    pub max_speed: f32,
    pub color_hue: f32,
    pub vision_radius: f32,
    pub turn_rate: f32,
    pub body_length: f32,
    pub body_width: f32,
    pub body_height: f32,
    /// Multi-spike storage. Active range is `spikes[0..spike_count]`; the
    /// rest is zero-initialized and contributes nothing to predation.
    #[serde(default = "default_spikes")]
    pub spikes: [Spike; SPIKE_SLOTS],
    #[serde(default = "default_spike_count")]
    pub spike_count: u8,
    pub shell_thickness: f32,
    /// Differential-adhesion CAM token in `0..ADHESION_TYPE_COUNT`. Same-type
    /// cells attract (Steinberg sorting), cross-type pairs mildly repel; also
    /// gates spring bond formation.
    pub adhesion_type: u8,
    /// Per-cell spring stiffness; the formed `Bond` stores the pair-mean.
    pub bond_stiffness: f32,
    /// Per-cell spring damping; the formed `Bond` stores the pair-mean.
    pub bond_damping: f32,
    /// Half-angle (rad) of the directional FOV cone around `forward_vector`,
    /// in `[MIN_VISION_FOV, MAX_VISION_FOV]`. Init = full sphere; cone
    /// filtering applies in sensor gather and energy cost scales with
    /// `vision_fov_factor(theta)`.
    #[serde(default = "default_vision_fov")]
    pub vision_fov: f32,
    /// Per-cell preferred temperature. The cell pays a quadratic drain
    /// proportional to `|temp − optimum|`, so an initial uniform draw
    /// across the range seeds cold-prefer / warm-prefer speciation.
    #[serde(default = "default_thermal_optimum")]
    pub thermal_optimum: f32,
    /// Digestion specialization in [0, 1]: 0 = pure herbivore, 1 = pure
    /// carnivore. Continuous trade-off via `eat_efficiency(food_kind, score)`.
    #[serde(default)]
    pub carnivore_score: f32,
    /// Per-category sensor gain in `[MIN_SENSOR_GAIN, MAX_SENSOR_GAIN]`.
    /// Index 0 = Vision, 1 = Chemistry, 2 = Defensive, 3 = Mechano. Scales
    /// input strength and adds `sum × SENSOR_GAIN_COST` per-tick drain,
    /// creating room for role differentiation when cluster cells pool sensors.
    #[serde(default = "default_sensor_gains")]
    pub sensor_gains: [f32; N_SENSOR_CATEGORIES],
    pub brain: Brain,
    /// HyperNEAT innate template. The `brain` field is derived from this
    /// CPPN at reproduction time; Hebbian updates during life mutate the
    /// derived brain only, never the CPPN. Non-Lamarckian: only the CPPN
    /// is inherited.
    #[serde(default = "default_cppn")]
    pub cppn: Cppn,
    /// Sprint 136: per-cell Hebbian learning rate. Replaces the global
    /// `LEARNING_RATE` const at every CPU call site (S137 plumbs the GPU
    /// dispatch too). Serde default to the pre-S136 global keeps older
    /// checkpoints loading at the legacy uniform rate.
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f32,
    /// Sprint 136: per-cell eligibility-trace decay (1/s). Replaces
    /// `HEBBIAN_TRACE_DECAY_PER_SEC` at CPU call sites; same checkpoint-
    /// compat story as `learning_rate`.
    #[serde(default = "default_trace_decay_per_sec")]
    pub trace_decay_per_sec: f32,
    /// Sprint 144: which compute model the hidden layer uses. Pre-S144
    /// checkpoints deserialize as `Perceptron`. Mutation flips with rate
    /// `MutationConfig::model_flip_rate` (default 0 → byte-identical
    /// baseline until S150 mixed-population validation flips it on).
    #[serde(default)]
    pub neuron_model: NeuronModel,
    /// Sprint 148: per-cell STDP LTP magnitude (pre-before-post pairing).
    /// Activates only on `NeuronModel::Izhikevich` lineages (the spike
    /// timing path) — `Perceptron` cells ignore. Serde default to the
    /// pre-S148 placeholder so old checkpoints load.
    #[serde(default = "default_stdp_a_plus")]
    pub stdp_a_plus: f32,
    /// Sprint 148: per-cell STDP LTD magnitude (post-before-pre).
    #[serde(default = "default_stdp_a_minus")]
    pub stdp_a_minus: f32,
    /// Sprint 148: per-cell STDP temporal window (ticks).
    #[serde(default = "default_stdp_tau_ticks")]
    pub stdp_tau_ticks: f32,
}

pub fn default_cppn() -> Cppn {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    Cppn::random(&mut rng)
}

pub fn default_sensor_gains() -> [f32; N_SENSOR_CATEGORIES] {
    [1.0; N_SENSOR_CATEGORIES]
}

pub fn default_thermal_optimum() -> f32 {
    THERMAL_REF_TEMP
}

pub fn default_pooled_hidden() -> [f32; BRAIN_HIDDEN] {
    [0.0; BRAIN_HIDDEN]
}

pub fn default_bonded_inbox() -> [f32; N_BOND_MSG_CHANNELS] {
    [0.0; N_BOND_MSG_CHANNELS]
}

pub fn default_vision_fov() -> f32 {
    INITIAL_VISION_FOV
}

pub fn default_spikes() -> [Spike; SPIKE_SLOTS] {
    [Spike::ZERO; SPIKE_SLOTS]
}

pub fn default_spike_count() -> u8 {
    1
}

pub fn default_learning_rate() -> f32 {
    LEARNING_RATE
}

pub fn default_trace_decay_per_sec() -> f32 {
    HEBBIAN_TRACE_DECAY_PER_SEC
}

pub fn default_stdp_a_plus() -> f32 {
    DEFAULT_STDP_A_PLUS
}

pub fn default_stdp_a_minus() -> f32 {
    DEFAULT_STDP_A_MINUS
}

pub fn default_stdp_tau_ticks() -> f32 {
    DEFAULT_STDP_TAU_TICKS
}

impl Genome {
    pub fn random(rng: &mut impl Rng) -> Self {
        // Isotropic body at gen 0 — mutations can create asymmetry only when
        // selection rewards it; no ellipsoid prior is baked in.
        let body_size = rng.random_range(0.7..1.3);
        let cppn = Cppn::random(rng);
        let brain = Brain::from_cppn(&cppn);
        Self {
            max_speed: rng.random_range(30.0..90.0),
            color_hue: rng.random_range(0.0..HUE_RANGE),
            vision_radius: rng.random_range(20.0..80.0),
            turn_rate: rng.random_range(1.0..5.0),
            body_length: body_size,
            body_width: body_size,
            body_height: body_size,
            spikes: {
                let mut spikes = [Spike::ZERO; SPIKE_SLOTS];
                spikes[0] = Spike {
                    length: rng.random_range(0.0..0.1),
                    azimuth_offset: 0.0,
                    elevation_offset: 0.0,
                    complexity: 0.0,
                };
                spikes
            },
            spike_count: 1,
            shell_thickness: rng.random_range(0.0..0.2),
            adhesion_type: rng.random_range(0..ADHESION_TYPE_COUNT),
            bond_stiffness: rng.random_range(BOND_STIFFNESS * 0.5..BOND_STIFFNESS * 1.5),
            bond_damping: rng.random_range(BOND_DAMPING * 0.5..BOND_DAMPING * 1.5),
            vision_fov: INITIAL_VISION_FOV,
            thermal_optimum: rng.random_range(MIN_THERMAL_OPTIMUM..MAX_THERMAL_OPTIMUM),
            // Range [0, 0.5] (not [0, 0.3]) so an immediate carnivore-leaning
            // niche exists on cell-carrion drops; without it the multi-trophic
            // food chain is dormant until sigma drift pushes any cell above
            // 0.5, which can take many generations.
            carnivore_score: rng.random_range(0.0..0.5),
            sensor_gains: [
                rng.random_range(0.7..1.3),
                rng.random_range(0.7..1.3),
                rng.random_range(0.7..1.3),
                rng.random_range(0.7..1.3),
            ],
            brain,
            cppn,
            // Sprint 136: seed at the pre-S136 global so a fresh population
            // matches the constant-rate baseline byte-for-byte until
            // `sigma_learning_rate` / `sigma_trace_decay` activate.
            learning_rate: LEARNING_RATE,
            trace_decay_per_sec: HEBBIAN_TRACE_DECAY_PER_SEC,
            // Sprint 144: all gen-0 cells run the rate-coded perceptron;
            // Izhikevich switches on via mutation once S150 flips
            // `MutationConfig::model_flip_rate` above 0.
            neuron_model: NeuronModel::Perceptron,
            // Sprint 148: STDP parameters seeded at canonical defaults.
            // Inactive while the cell is Perceptron; S149 wires them into
            // the Izhikevich plasticity path.
            stdp_a_plus: DEFAULT_STDP_A_PLUS,
            stdp_a_minus: DEFAULT_STDP_A_MINUS,
            stdp_tau_ticks: DEFAULT_STDP_TAU_TICKS,
        }
    }

    pub fn mutate(&self, rng: &mut impl Rng, cfg: &MutationConfig) -> Self {
        let mut g = self.mutate_no_brain(rng, cfg);
        g.brain = Brain::from_cppn(&g.cppn);
        g
    }

    /// Same gene draws + RNG sequence as `mutate`, but leaves the brain as
    /// `Brain::zeros()`. Callers that materialise the brain via the GPU
    /// CPPN dispatch (or some other path) skip the CPU `Brain::from_cppn`
    /// cost. Production hot path: `--gpu-full` reproduce in headless.
    pub fn mutate_no_brain(&self, rng: &mut impl Rng, cfg: &MutationConfig) -> Self {
        let mutated_cppn = self.cppn.mutate(rng, &CPPN_MUTATION_CONFIG);
        // Discrete adhesion-type flip — pick a uniform replacement that is
        // strictly different from the current type. Falls back to the
        // existing value when ADHESION_TYPE_COUNT == 1 (no other type exists).
        let adhesion_type = if cfg.adhesion_flip_rate > 0.0
            && ADHESION_TYPE_COUNT > 1
            && rng.random::<f32>() < cfg.adhesion_flip_rate
        {
            let mut new_t = rng.random_range(0..ADHESION_TYPE_COUNT - 1);
            if new_t >= self.adhesion_type {
                new_t += 1;
            }
            new_t
        } else {
            self.adhesion_type
        };
        Self {
            max_speed: (self.max_speed + gaussian(rng) * cfg.sigma_speed)
                .clamp(MIN_SPEED, MAX_SPEED),
            color_hue: (self.color_hue + gaussian(rng) * cfg.sigma_hue).rem_euclid(HUE_RANGE),
            vision_radius: (self.vision_radius + gaussian(rng) * cfg.sigma_vision).max(MIN_VISION),
            turn_rate: (self.turn_rate + gaussian(rng) * cfg.sigma_turn_rate).max(MIN_TURN_RATE),
            body_length: (self.body_length + gaussian(rng) * cfg.sigma_body_length)
                .clamp(MIN_BODY_LENGTH, MAX_BODY_LENGTH),
            body_width: (self.body_width + gaussian(rng) * cfg.sigma_body_width)
                .clamp(MIN_BODY_WIDTH, MAX_BODY_WIDTH),
            body_height: (self.body_height + gaussian(rng) * cfg.sigma_body_height)
                .clamp(MIN_BODY_HEIGHT, MAX_BODY_HEIGHT),
            spikes: {
                let mut spikes = self.spikes;
                // Slot 0 keeps the original single-spike semantics (drives
                // the pre-multi-spike RNG sequence when secondary sigmas are 0).
                spikes[0].length = (spikes[0].length + gaussian(rng) * cfg.sigma_spike_length)
                    .clamp(MIN_SPIKE_LENGTH, MAX_SPIKE_LENGTH);
                // Each block is gated by its own sigma so the loop is skipped
                // entirely when the feature is off — that preserves both the
                // RNG sequence and the cycles for unchanged simulations.
                if cfg.sigma_spike_length_secondary > 0.0 {
                    for i in 1..SPIKE_SLOTS {
                        spikes[i].length = (spikes[i].length
                            + gaussian(rng) * cfg.sigma_spike_length_secondary)
                            .clamp(MIN_SPIKE_LENGTH, MAX_SPIKE_LENGTH);
                    }
                }
                if cfg.sigma_spike_orientation > 0.0 {
                    for i in 0..SPIKE_SLOTS {
                        spikes[i].azimuth_offset = (spikes[i].azimuth_offset
                            + gaussian(rng) * cfg.sigma_spike_orientation)
                            .clamp(MIN_SPIKE_AZIMUTH, MAX_SPIKE_AZIMUTH);
                        spikes[i].elevation_offset = (spikes[i].elevation_offset
                            + gaussian(rng) * cfg.sigma_spike_orientation)
                            .clamp(MIN_SPIKE_ELEVATION, MAX_SPIKE_ELEVATION);
                    }
                }
                if cfg.sigma_spike_complexity > 0.0 {
                    for i in 0..SPIKE_SLOTS {
                        spikes[i].complexity = (spikes[i].complexity
                            + gaussian(rng) * cfg.sigma_spike_complexity)
                            .clamp(MIN_SPIKE_COMPLEXITY, MAX_SPIKE_COMPLEXITY);
                    }
                }
                spikes
            },
            spike_count: {
                if cfg.spike_count_mutation_rate > 0.0
                    && rng.random::<f32>() < cfg.spike_count_mutation_rate
                {
                    let dir: bool = rng.random();
                    let cur = self.spike_count as i32;
                    let new = if dir { cur + 1 } else { cur - 1 };
                    new.clamp(0, SPIKE_SLOTS as i32) as u8
                } else {
                    self.spike_count
                }
            },
            shell_thickness: (self.shell_thickness + gaussian(rng) * cfg.sigma_shell)
                .clamp(MIN_SHELL_THICKNESS, MAX_SHELL_THICKNESS),
            adhesion_type,
            bond_stiffness: (self.bond_stiffness + gaussian(rng) * cfg.sigma_bond_stiffness)
                .clamp(MIN_BOND_STIFFNESS, MAX_BOND_STIFFNESS),
            bond_damping: (self.bond_damping + gaussian(rng) * cfg.sigma_bond_damping)
                .clamp(MIN_BOND_DAMPING, MAX_BOND_DAMPING),
            // The remaining genes follow a `sigma > 0` short-circuit so that
            // disabling drift skips the gaussian draw entirely — both for
            // cycle savings and to keep the RNG sequence stable for fixtures
            // captured before the gene was activated.
            vision_fov: if cfg.sigma_vision_fov > 0.0 {
                (self.vision_fov + gaussian(rng) * cfg.sigma_vision_fov)
                    .clamp(MIN_VISION_FOV, MAX_VISION_FOV)
            } else {
                self.vision_fov
            },
            thermal_optimum: if cfg.sigma_thermal_optimum > 0.0 {
                (self.thermal_optimum + gaussian(rng) * cfg.sigma_thermal_optimum)
                    .clamp(MIN_THERMAL_OPTIMUM, MAX_THERMAL_OPTIMUM)
            } else {
                self.thermal_optimum
            },
            carnivore_score: if cfg.sigma_carnivore_score > 0.0 {
                (self.carnivore_score + gaussian(rng) * cfg.sigma_carnivore_score)
                    .clamp(0.0, 1.0)
            } else {
                self.carnivore_score
            },
            sensor_gains: {
                let mut g = self.sensor_gains;
                if cfg.sigma_sensor_gain > 0.0 {
                    for v in g.iter_mut() {
                        *v = (*v + gaussian(rng) * cfg.sigma_sensor_gain)
                            .clamp(MIN_SENSOR_GAIN, MAX_SENSOR_GAIN);
                    }
                }
                g
            },
            cppn: mutated_cppn,
            brain: Brain::zeros(),
            // Sprint 136: per-cell rate drift gated by sigma > 0 short-circuit
            // (same pattern as `vision_fov` in S82) — sigma = 0 default means
            // no RNG draw and no behavior change vs the pre-S136 baseline.
            learning_rate: if cfg.sigma_learning_rate > 0.0 {
                (self.learning_rate + gaussian(rng) * cfg.sigma_learning_rate)
                    .clamp(MIN_LEARNING_RATE, MAX_LEARNING_RATE)
            } else {
                self.learning_rate
            },
            trace_decay_per_sec: if cfg.sigma_trace_decay > 0.0 {
                (self.trace_decay_per_sec + gaussian(rng) * cfg.sigma_trace_decay)
                    .clamp(MIN_TRACE_DECAY_PER_SEC, MAX_TRACE_DECAY_PER_SEC)
            } else {
                self.trace_decay_per_sec
            },
            // Sprint 144: flip neuron model with low rate, short-circuited
            // so a zero rate skips the RNG draw (byte-identical baseline).
            neuron_model: if cfg.model_flip_rate > 0.0
                && rng.random::<f32>() < cfg.model_flip_rate
            {
                match self.neuron_model {
                    NeuronModel::Perceptron => NeuronModel::Izhikevich,
                    NeuronModel::Izhikevich => NeuronModel::Perceptron,
                }
            } else {
                self.neuron_model
            },
            // Sprint 148: STDP parameter drift gated by sigma > 0.
            stdp_a_plus: if cfg.sigma_stdp_a > 0.0 {
                (self.stdp_a_plus + gaussian(rng) * cfg.sigma_stdp_a).clamp(0.0, MAX_STDP_A)
            } else {
                self.stdp_a_plus
            },
            stdp_a_minus: if cfg.sigma_stdp_a > 0.0 {
                (self.stdp_a_minus + gaussian(rng) * cfg.sigma_stdp_a).clamp(0.0, MAX_STDP_A)
            } else {
                self.stdp_a_minus
            },
            stdp_tau_ticks: if cfg.sigma_stdp_tau > 0.0 {
                (self.stdp_tau_ticks + gaussian(rng) * cfg.sigma_stdp_tau)
                    .clamp(MIN_STDP_TAU_TICKS, MAX_STDP_TAU_TICKS)
            } else {
                self.stdp_tau_ticks
            },
        }
    }

    /// Per-gene uniform crossover. Each scalar gene picks 50/50 from one
    /// parent; the brain is **not** materialised here — production callers
    /// always chain `crossover().mutate()` and the mutate step replaces the
    /// brain anyway, so the crossover-side `Brain::from_cppn` would be
    /// thrown away. We return `Brain::zeros()` and trust the chained mutate
    /// (or an explicit `Brain::from_cppn(&genome.cppn)` for non-chained
    /// callers) to fill the brain. Doubles the speed of `make_mating_child`.
    pub fn crossover(a: &Genome, b: &Genome, rng: &mut impl Rng) -> Genome {
        let child_cppn = Cppn::crossover(&a.cppn, &b.cppn, rng);
        Genome {
            max_speed: if rng.random::<bool>() { a.max_speed } else { b.max_speed },
            color_hue: if rng.random::<bool>() { a.color_hue } else { b.color_hue },
            vision_radius: if rng.random::<bool>() { a.vision_radius } else { b.vision_radius },
            turn_rate: if rng.random::<bool>() { a.turn_rate } else { b.turn_rate },
            body_length: if rng.random::<bool>() { a.body_length } else { b.body_length },
            body_width: if rng.random::<bool>() { a.body_width } else { b.body_width },
            body_height: if rng.random::<bool>() { a.body_height } else { b.body_height },
            spikes: {
                let mut spikes = [Spike::ZERO; SPIKE_SLOTS];
                // Slot 0 length keeps the original single-spike crossover
                // (an unconditional bool draw) so RNG fixtures captured before
                // multi-spike still reproduce.
                spikes[0].length = if rng.random::<bool>() {
                    a.spikes[0].length
                } else {
                    b.spikes[0].length
                };
                // The remaining spike attributes use a short-circuit: when
                // the parents already agree, skip the RNG draw. This keeps
                // the RNG sequence byte-identical with a baseline that left
                // those attributes pinned (e.g., complexity = 0 throughout).
                let pick_f32 = |a: f32, b: f32, rng: &mut dyn rand::RngCore| -> f32 {
                    if a == b {
                        a
                    } else if rng.random::<bool>() {
                        a
                    } else {
                        b
                    }
                };
                spikes[0].azimuth_offset =
                    pick_f32(a.spikes[0].azimuth_offset, b.spikes[0].azimuth_offset, rng);
                spikes[0].elevation_offset = pick_f32(
                    a.spikes[0].elevation_offset,
                    b.spikes[0].elevation_offset,
                    rng,
                );
                spikes[0].complexity = pick_f32(a.spikes[0].complexity, b.spikes[0].complexity, rng);
                for i in 1..SPIKE_SLOTS {
                    spikes[i].length = pick_f32(a.spikes[i].length, b.spikes[i].length, rng);
                    spikes[i].azimuth_offset =
                        pick_f32(a.spikes[i].azimuth_offset, b.spikes[i].azimuth_offset, rng);
                    spikes[i].elevation_offset = pick_f32(
                        a.spikes[i].elevation_offset,
                        b.spikes[i].elevation_offset,
                        rng,
                    );
                    spikes[i].complexity =
                        pick_f32(a.spikes[i].complexity, b.spikes[i].complexity, rng);
                }
                spikes
            },
            spike_count: if a.spike_count == b.spike_count {
                a.spike_count
            } else if rng.random::<bool>() {
                a.spike_count
            } else {
                b.spike_count
            },
            shell_thickness: if rng.random::<bool>() { a.shell_thickness } else { b.shell_thickness },
            adhesion_type: if rng.random::<bool>() { a.adhesion_type } else { b.adhesion_type },
            bond_stiffness: if rng.random::<bool>() { a.bond_stiffness } else { b.bond_stiffness },
            bond_damping: if rng.random::<bool>() { a.bond_damping } else { b.bond_damping },
            vision_fov: if a.vision_fov == b.vision_fov {
                a.vision_fov
            } else if rng.random::<bool>() {
                a.vision_fov
            } else {
                b.vision_fov
            },
            thermal_optimum: if rng.random::<bool>() {
                a.thermal_optimum
            } else {
                b.thermal_optimum
            },
            carnivore_score: if rng.random::<bool>() {
                a.carnivore_score
            } else {
                b.carnivore_score
            },
            sensor_gains: {
                let mut g = [0.0_f32; N_SENSOR_CATEGORIES];
                for k in 0..N_SENSOR_CATEGORIES {
                    g[k] = if rng.random::<bool>() {
                        a.sensor_gains[k]
                    } else {
                        b.sensor_gains[k]
                    };
                }
                g
            },
            brain: Brain::zeros(),
            cppn: child_cppn,
            // Sprint 136: same-value short-circuit so populations seeded at
            // the pre-S136 uniform rate (both parents = LEARNING_RATE) keep
            // the RNG sequence stable. Only diverged lineages consume a draw.
            learning_rate: if a.learning_rate == b.learning_rate {
                a.learning_rate
            } else if rng.random::<bool>() {
                a.learning_rate
            } else {
                b.learning_rate
            },
            trace_decay_per_sec: if a.trace_decay_per_sec == b.trace_decay_per_sec {
                a.trace_decay_per_sec
            } else if rng.random::<bool>() {
                a.trace_decay_per_sec
            } else {
                b.trace_decay_per_sec
            },
            // Sprint 144: same-value short-circuit so populations with all-
            // Perceptron parents (S144 default) keep the pre-S144 RNG sequence.
            neuron_model: if a.neuron_model == b.neuron_model {
                a.neuron_model
            } else if rng.random::<bool>() {
                a.neuron_model
            } else {
                b.neuron_model
            },
            // Sprint 148: STDP params follow same-value short-circuit.
            stdp_a_plus: if a.stdp_a_plus == b.stdp_a_plus {
                a.stdp_a_plus
            } else if rng.random::<bool>() {
                a.stdp_a_plus
            } else {
                b.stdp_a_plus
            },
            stdp_a_minus: if a.stdp_a_minus == b.stdp_a_minus {
                a.stdp_a_minus
            } else if rng.random::<bool>() {
                a.stdp_a_minus
            } else {
                b.stdp_a_minus
            },
            stdp_tau_ticks: if a.stdp_tau_ticks == b.stdp_tau_ticks {
                a.stdp_tau_ticks
            } else if rng.random::<bool>() {
                a.stdp_tau_ticks
            } else {
                b.stdp_tau_ticks
            },
        }
    }
}

pub fn gaussian(rng: &mut impl Rng) -> f32 {
    let u1: f32 = rng.random::<f32>().max(f32::EPSILON);
    let u2: f32 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}
