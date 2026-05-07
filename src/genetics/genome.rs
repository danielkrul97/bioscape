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
    // Sprint 68: bond physics genes — pomalejší drift než body params kvůli
    // sub-procentní bond_active_frac v Sprint 67.1 long-run smoke (selekce
    // má slabý signál, větší sigma by způsobil random walk).
    sigma_bond_stiffness: 0.3,
    sigma_bond_damping: 0.05,
    // Sprint 80 Sprint C: structural mutation off by default. Zapnutí přes
    // local `MutationConfig { add_neuron_rate: 0.05, ..MUTATION_CONFIG }`
    // v experimentech.
    //
    // Sprint 104: zapnuto pro NEAT-direct evoluci. 2-3 % rates dají ~1
    // structural mutace per cell per 30-50 gen → pomalá topologická drift,
    // selekce má čas vyhodnotit.
    add_neuron_rate: 0.02,
    split_link_rate: 0.02,
    remove_neuron_rate: 0.01,
    // Sprint 82: FOV gen dormant — Sprint 82 je pure infra (gen + cost factor).
    // Sprint 83: 0.0 → 0.05 — modest drift jako sigma_bond_stiffness (0.3 / 10
    // = 3 % range, sigma_vision_fov 0.05 / 2.88 ≈ 1.7 % FOV range per gen).
    // Pomalejší než tělesné geny aby evoluce stihla najít optimum bez random
    // walku do MIN_VISION_FOV.
    sigma_vision_fov: 0.05,
    // Sprint 87: thermal_optimum drift. 0.5 sim-units / gen ≈ 1.9 % range
    // (range = 26 sim-units = TOP - BOTTOM). Pomalý drift aby selekce stihla
    // tlumit speciaci do depth-coupled niche, ne random walk.
    sigma_thermal_optimum: 0.5,
    // Sprint 92: carnivore_score drift. 0.02 = 2 % range/gen.
    sigma_carnivore_score: 0.02,
    // Sprint 97: sensor_gains drift. 0.04 = 2 % range / gen.
    sigma_sensor_gain: 0.04,
    // Sprint 122: zapnutí discrete spike_count mutace + per-spike orientation drift.
    // ~5 % cells per gen mění spike_count o ±1 (clamp [0, 5]). 0.05 rad ≈ 3°
    // drift v azimuth/elevation per gen — slow per-spike directional evolution.
    spike_count_mutation_rate: 0.05,
    sigma_spike_orientation: 0.05,
    // Sprint 123: complexity drift aktivní. 0.02 = 2 % range/gen ([0, 1] range,
    // 1000-gen drift dosáhne ~MAX). Slow drift aby selekce stihla najít
    // intermediate sweet-spot (~0.4-0.6) místo random walk k extremům.
    sigma_spike_complexity: 0.02,
    // Sprint 122: per-non-primary spike length drift. Když spike_count_mutation_rate
    // aktivuje slot, dítě dostane init length = 0 (nový spike), který
    // postupně mutuje. 0.03 stejně jako sigma_spike_length (slot 0).
    sigma_spike_length_secondary: 0.03,
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
    /// Sprint 66: pravděpodobnost flipu adhesion_type per dítě.
    pub adhesion_flip_rate: f32,
    /// Sprint 68: gaussian sigma pro bond_stiffness gen.
    pub sigma_bond_stiffness: f32,
    /// Sprint 68: gaussian sigma pro bond_damping gen.
    pub sigma_bond_damping: f32,
    /// Sprint 80 Sprint C: pravděpodobnost `add_neuron` structural mutace
    /// per dítě. 0.0 = topology evoluce vypnutá (default; zachovává Sprint
    /// B byte-identical trajectory). Tuning sweep příští sprint zkusí 0.02,
    /// 0.05, 0.1.
    pub add_neuron_rate: f32,
    /// Sprint 104: classic NEAT split_link mutation rate. Vyber random
    /// active link (i,h) s |w|>SPLIT_LINK_THRESHOLD, insert nový hidden
    /// k mezi nimi. Topology-preserving — forward output zachován.
    pub split_link_rate: f32,
    /// Sprint 104: remove_neuron mutation rate. Pick random hidden, zero out.
    /// Pruning balance proti add_neuron + split_link růstu.
    pub remove_neuron_rate: f32,
    /// Sprint 82: gaussian sigma pro `vision_fov` mutaci. `MUTATION_CONFIG`
    /// default = 0 (Sprint 82 pure-infra, FOV gen dormant). Sprint 83+ FOV
    /// aktivuje a tuning experimenty mohou nastavit nenulový drift. Při
    /// drift na MIN_VISION_FOV cells ztrácejí cells/food awareness, ale
    /// platí minimální vision cost — selekční trade-off vstoupí v platnost
    /// až s aktivním cone filterem.
    pub sigma_vision_fov: f32,
    /// Sprint 87: gaussian sigma pro `thermal_optimum` mutaci. ~0.5 sim-units
    /// (~1.9 % range per gen) — pomalý drift relative k init populace
    /// uniform ∈ [BOTTOM, TOP], nechává selekci tlumit speciaci.
    pub sigma_thermal_optimum: f32,
    /// Sprint 92: digestion specialization gen sigma. 0.02 = 2 % range/gen.
    /// Pomalejší než ostatní geny — diet shift vyžaduje food availability
    /// signal (selekce přes eat efficiency × food_kind), který je sám pomalý.
    pub sigma_carnivore_score: f32,
    /// Sprint 97: sensor_gains per-category gaussian sigma. 0.04 = 2 % of
    /// [MIN, MAX] = [0, 2] range/gen. Drift k specializaci vyžaduje cluster
    /// pooling signal — selekce je conditional na bond presence.
    pub sigma_sensor_gain: f32,
    /// Sprint 122: pravděpodobnost diskrétní mutace `spike_count` per dítě.
    /// 0.05 = ~5 % cells per gen flip ±1 (clamp [0, SPIKE_SLOTS]). 0.0 = vypnuté
    /// (Sprint 121 default — gen 0 spike_count=1 propaguje napříč evolucí).
    /// Sprint 123: může enabled.
    pub spike_count_mutation_rate: f32,
    /// Sprint 122: gaussian sigma pro spike azimuth/elevation offsety
    /// (rad/gen). 0.05 ≈ 3°/gen drift — slow per-spike directional evolution.
    pub sigma_spike_orientation: f32,
    /// Sprint 123: gaussian sigma pro spike complexity ∈ [0, 1] mutaci.
    /// 0.0 = vypnuté (Sprint 121/122 default — complexity zaseknutý na 0,
    /// COMPLEXITY_*_GAIN multiplikátory vrací 1.0).
    pub sigma_spike_complexity: f32,
    /// Sprint 122: gaussian sigma pro mutaci `length` na spike sloty 1..SPIKE_SLOTS
    /// (= "non-primary" sloty, které pre-S122 byly drženy na 0). Slot 0 zachovává
    /// existující `sigma_spike_length`. 0.0 = vypnuté → non-primary sloty
    /// drift jen přes activation/deactivation pres `spike_count_mutation_rate`.
    pub sigma_spike_length_secondary: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Genome {
    pub max_speed: f32,
    pub color_hue: f32,
    pub vision_radius: f32,
    pub turn_rate: f32,
    pub body_length: f32,
    pub body_width: f32,
    /// Sprint 34: 3-axis ellipsoid — vertikální rozměr.
    pub body_height: f32,
    /// Sprint 121: multi-spike. `spikes[0..spike_count]` aktivní, zbytek
    /// zero-init. Pre-Sprint-121 single `spike_length` mapuje na
    /// `spikes[0].length` se `spike_count = 1`.
    #[serde(default = "default_spikes")]
    pub spikes: [Spike; SPIKE_SLOTS],
    #[serde(default = "default_spike_count")]
    pub spike_count: u8,
    /// Sprint 41: shell jako passive damage absorber.
    pub shell_thickness: f32,
    /// Sprint 66: differential-adhesion CAM token (∈ 0..ADHESION_TYPE_COUNT).
    /// Same-type cells na sebe atraktivně působí (Steinberg sorting), cross-type
    /// pair má mírnou repulzi. Také gateway pro spring bond formation.
    pub adhesion_type: u8,
    /// Sprint 68: per-cell spring stiffness contribution. Bond mezi dvěma
    /// cells používá průměr obou stiffness při formaci (uložený do Bond struct).
    pub bond_stiffness: f32,
    /// Sprint 68: per-cell spring damping contribution. Stejná semantika jako
    /// bond_stiffness — pair-mean uložený do Bond.
    pub bond_damping: f32,
    /// Sprint 82: půl-úhel kuželu směrového FOV kolem `forward_vector` (rad).
    /// Range `[MIN_VISION_FOV, MAX_VISION_FOV]`. Init = `INITIAL_VISION_FOV`
    /// (= π = full sphere). Sprint 83+ aktivuje cone filter v sensor gather;
    /// energy cost už v Sprint 82 škáluje s `vision_fov_factor(theta)`.
    #[serde(default = "default_vision_fov")]
    pub vision_fov: f32,
    /// Sprint 87: per-cell preferovaná teplota (sim units, range
    /// `[MIN_THERMAL_OPTIMUM, MAX_THERMAL_OPTIMUM]`). Cell platí kvadratický
    /// penalty drain za |temp − optimum|. Init populace random uniform
    /// across range → speciace mezi cold-prefer / warm-prefer fenotypy.
    #[serde(default = "default_thermal_optimum")]
    pub thermal_optimum: f32,
    /// Sprint 92: digestion specialization gen ∈ [0, 1]. 0 = pure herbivore
    /// (plant food only), 1 = pure carnivore (hunter carrion only). Continuous
    /// trade-off — eat_efficiency(food_kind, score). Init populace bias
    /// herbivore-leaning (cells potřebují plant food survive cold start).
    #[serde(default)]
    pub carnivore_score: f32,
    /// Sprint 97: per-category sensor gain ∈ [MIN, MAX]. Index 0 = Vision,
    /// 1 = Chemistry, 2 = Defensive. Modulates input strength + per-tick
    /// energy drain (`sum × SENSOR_GAIN_COST`). Cluster cells s pooled sensors
    /// mohou turn off duplicate sensors → save energy → role differentiation
    /// emergent (scout cells vision-specialist, smell cells chemistry-specialist).
    #[serde(default = "default_sensor_gains")]
    pub sensor_gains: [f32; N_SENSOR_CATEGORIES],
    pub brain: Brain,
    /// Sprint 106: HyperNEAT CPPN — innate template, ze kterého se Brain
    /// derives při make_*_child. Hebbian během života modifies derived
    /// brain, ale na reproduce se brain re-derives z child's CPPN. Non-
    /// Lamarckian: Hebbian gains nepřechází do dalších generací, jen
    /// CPPN topology + weights se dědí (S105 mutations + crossover).
    #[serde(default = "default_cppn")]
    pub cppn: Cppn,
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

pub fn default_vision_fov() -> f32 {
    INITIAL_VISION_FOV
}

pub fn default_spikes() -> [Spike; SPIKE_SLOTS] {
    [Spike::ZERO; SPIKE_SLOTS]
}

pub fn default_spike_count() -> u8 {
    1
}

impl Genome {
    pub fn random(rng: &mut impl Rng) -> Self {
        // Default tělo je izotropní koule (length == width == height). Mutace
        // mohou asymetrii vytvořit, ale jen pokud ji selekce odmění; gen 0
        // nezavádí prior na ellipse fenotyp.
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
            // Sprint 41: mírný počáteční mean, žádný extreme spawn — selekce
            // si shell vytáhne nahoru, pokud má smysl.
            shell_thickness: rng.random_range(0.0..0.2),
            // Sprint 66: uniform draw napříč ADHESION_TYPE_COUNT typů. Initial
            // populace tak má rovnoměrnou type distribution; selekce + drift
            // pak modifikují frekvence.
            adhesion_type: rng.random_range(0..ADHESION_TYPE_COUNT),
            // Sprint 68: initial draw kolem global default ±50 %. Selekce +
            // drift pak rozhrnou rozsah.
            bond_stiffness: rng.random_range(BOND_STIFFNESS * 0.5..BOND_STIFFNESS * 1.5),
            bond_damping: rng.random_range(BOND_DAMPING * 0.5..BOND_DAMPING * 1.5),
            // Sprint 82: full-sphere baseline, žádný RNG draw — initial population
            // kompletně omnidirectional. Cost faktor = 1.0, behavior matches
            // pre-Sprint-82 baseline až do prvního sigma_vision_fov > 0 sprintu.
            vision_fov: INITIAL_VISION_FOV,
            // Sprint 87: random uniform across [BOTTOM, TOP] — initial speciace.
            // RNG draw je breaking change pro Sprint 86 baseline (BRAIN_INPUTS
            // shape change už CSV reproducibility ztratila).
            thermal_optimum: rng.random_range(MIN_THERMAL_OPTIMUM..MAX_THERMAL_OPTIMUM),
            // Sprint 92: bias toward herbivore (range [0, 0.3]) — cold start.
            // Sprint 93: range [0, 0.5] — wider initial spread aby existoval
            // immediate niche pro carnivore-leaning cells na hunter carrion
            // drop sites. Bez tohoto je multi-trophic food chain dormant
            // dokud sigma drift nedotlačí > 0.5 (mnoho gens).
            carnivore_score: rng.random_range(0.0..0.5),
            // Sprint 97: random uniform [0.7, 1.3] per category — small spread
            // around neutral 1.0 aby initial population měla mírně varied
            // sensor profiles bez immediate dramatic specialization.
            sensor_gains: [
                rng.random_range(0.7..1.3),
                rng.random_range(0.7..1.3),
                rng.random_range(0.7..1.3),
            ],
            // Sprint 106: brain je derived z cppn (innate template).
            brain,
            cppn,
        }
    }

    pub fn mutate(&self, rng: &mut impl Rng, cfg: &MutationConfig) -> Self {
        // Sprint 106: cppn mutated first, brain derived po mutaci.
        let mutated_cppn = self.cppn.mutate(rng, &CPPN_MUTATION_CONFIG);
        let derived_brain = Brain::from_cppn(&mutated_cppn);
        // Sprint 66: discrete adhesion_type. Při flipu náhodně vyberu jiný
        // typ ≠ self.adhesion_type. Pokud ADHESION_TYPE_COUNT == 1, fallback
        // ponechá hodnotu (žádný "jiný" typ neexistuje).
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
                // Slot 0 (primary) — pre-S121 single-spike sémantika.
                spikes[0].length = (spikes[0].length + gaussian(rng) * cfg.sigma_spike_length)
                    .clamp(MIN_SPIKE_LENGTH, MAX_SPIKE_LENGTH);
                // Sprint 122: secondary slots length drift (jen pokud sigma > 0).
                if cfg.sigma_spike_length_secondary > 0.0 {
                    for i in 1..SPIKE_SLOTS {
                        spikes[i].length = (spikes[i].length
                            + gaussian(rng) * cfg.sigma_spike_length_secondary)
                            .clamp(MIN_SPIKE_LENGTH, MAX_SPIKE_LENGTH);
                    }
                }
                // Sprint 122: per-slot orientation drift (jen pokud sigma > 0).
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
                // Sprint 123: per-slot complexity drift (jen pokud sigma > 0).
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
                // Sprint 122: discrete ±1 mutace s rate. Pokud rate = 0
                // (Sprint 121 default), žádný RNG draw — byte-identical
                // s pre-S122. Sprint 122 rate = 0.05 → ~5 % cells per gen flip.
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
            // Sprint 82: short-circuit pattern jako Sprint 80 add_neuron_rate —
            // při `sigma_vision_fov = 0` se gaussian draw přeskočí, RNG sekvence
            // zůstává byte-identical s pre-Sprint-82. Při sigma > 0 (Sprint 83+)
            // je drift aktivní a CSV se rozejde od toho sprintu (expected).
            vision_fov: if cfg.sigma_vision_fov > 0.0 {
                (self.vision_fov + gaussian(rng) * cfg.sigma_vision_fov)
                    .clamp(MIN_VISION_FOV, MAX_VISION_FOV)
            } else {
                self.vision_fov
            },
            // Sprint 87: stejný short-circuit pattern. Při sigma=0 mutace
            // přeskočí gaussian draw → cell drží initial uniform draw.
            thermal_optimum: if cfg.sigma_thermal_optimum > 0.0 {
                (self.thermal_optimum + gaussian(rng) * cfg.sigma_thermal_optimum)
                    .clamp(MIN_THERMAL_OPTIMUM, MAX_THERMAL_OPTIMUM)
            } else {
                self.thermal_optimum
            },
            // Sprint 92: short-circuit pattern. sigma_carnivore_score = 0.02
            // default → ~2 % range/gen drift. Clamp na [0, 1].
            carnivore_score: if cfg.sigma_carnivore_score > 0.0 {
                (self.carnivore_score + gaussian(rng) * cfg.sigma_carnivore_score)
                    .clamp(0.0, 1.0)
            } else {
                self.carnivore_score
            },
            // Sprint 97: per-category gaussian drift, clamp [MIN, MAX].
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
            // Sprint 106: HyperNEAT — cppn = innate template, brain derived.
            cppn: mutated_cppn,
            brain: derived_brain,
        }
    }

    /// Per-gene uniform crossover. Each scalar gene picks 50/50 from one
    /// parent; brain uses its own per-row crossover.
    pub fn crossover(a: &Genome, b: &Genome, rng: &mut impl Rng) -> Genome {
        // Sprint 106: cppn crossover (NEAT innovation matching), brain derived
        // z výsledku.
        let child_cppn = Cppn::crossover(&a.cppn, &b.cppn, rng);
        let derived_brain = Brain::from_cppn(&child_cppn);
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
                // Slot 0: pre-S122 length-only crossover (zachovává byte-identity
                // pro S121 testy s spike_count=1, complexity=0, orientation=0).
                spikes[0].length = if rng.random::<bool>() {
                    a.spikes[0].length
                } else {
                    b.spikes[0].length
                };
                // Sprint 122: per-slot multi-attribute crossover pro non-primary
                // sloty + non-length atributy slotu 0. Short-circuit pokud rodiče
                // mají identické hodnoty — žádný RNG draw, byte-identical když
                // všechny S122 sigmy/rates = 0.
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
            // Sprint 121/122: short-circuit pokud parents shodné (pre-S122 vždy
            // 1). Sprint 122 mutace začne flip ±1 → parents se rozejdou →
            // bool draw aktivní.
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
            // Sprint 82: short-circuit při shodě hodnot — pokud `sigma_vision_fov
            // = 0` (S82 default), všechny cells drží INITIAL_VISION_FOV a RNG
            // bool draw se vyhne, takže pre-Sprint-82 CSV zůstává reprodukovatelný.
            // Po Sprint 83+ aktivaci sigmy budou hodnoty divergovat → bool draw
            // se zapne a CSV se rozejde (expected v behavior-change sprintu).
            vision_fov: if a.vision_fov == b.vision_fov {
                a.vision_fov
            } else if rng.random::<bool>() {
                a.vision_fov
            } else {
                b.vision_fov
            },
            // Sprint 87: standard bool crossover. Init populace má unikátní
            // optima (uniform random per cell) → values divergují hned od
            // gen 0, bool draw aktivní vždy.
            thermal_optimum: if rng.random::<bool>() {
                a.thermal_optimum
            } else {
                b.thermal_optimum
            },
            // Sprint 92: standard bool crossover.
            carnivore_score: if rng.random::<bool>() {
                a.carnivore_score
            } else {
                b.carnivore_score
            },
            // Sprint 97: per-category bool crossover.
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
            // Sprint 106: HyperNEAT — brain re-derived z child's cppn,
            // ne crossover parents' brains (Hebbian non-Lamarckian).
            brain: derived_brain,
            cppn: child_cppn,
        }
    }
}

pub fn gaussian(rng: &mut impl Rng) -> f32 {
    let u1: f32 = rng.random::<f32>().max(f32::EPSILON);
    let u2: f32 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}
