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
pub const INITIAL_ENERGY: f32 = 100.0;
// Brain inputs: senzorické 0=food_dx, 1=food_dy, 2=cell_dx, 3=cell_dy,
// 4=energy, 5=speed, 6=rel_size, 7=smell_grad_x, 8=smell_grad_y, 9=heading_x,
// 10=heading_y, 11=pheromone_grad_x, 12=pheromone_grad_y, 13=local_density,
// 14=damage. Sprint 33 přidává 3D rozšíření:
// 15=food_dz, 16=cell_dz, 17=smell_grad_z, 18=heading_z, 19=pheromone_grad_z.
// heading_x/y jsou nově xy projekce 3D forward vektoru (násobeno cos(pitch)),
// heading_z = sin(pitch). Pro pitch=0 jsou identické s pre-Sprint-33 cos/sin
// yaw — mozky natrénované v 2D zachovají chování při horizontálním letu.
// Sprint 28 přidává recurrent kanál: indexy [BRAIN_INPUTS_SENSORY..BRAIN_INPUTS]
// = previous tick `last_hidden` activations (Elman RNN). Genom drží sjednocený
// `w1` matrix 28×8, mutace + Hebbian pracují bez rozlišení sensory vs recurrent.
pub const BRAIN_INPUTS_SENSORY: usize = 20;
pub const BRAIN_HIDDEN: usize = 8;
/// Sprint 28: kolik dimenzí předchozího hidden state se feeduje zpátky jako
/// input. = `BRAIN_HIDDEN` znamená 1:1 mapping (každý neuron má vlastní paměť
/// slot). Menší než HIDDEN by exponoval jen subset hidden state; větší by
/// vyžadoval delay buffer. 1:1 je nejjednodušší a kapacita stačí.
pub const BRAIN_RECURRENT: usize = BRAIN_HIDDEN;
/// Total brain input width = sensory + recurrent. Genom + forward pass +
/// Hebbian + mutace pracují s touto velikostí transparentně.
pub const BRAIN_INPUTS: usize = BRAIN_INPUTS_SENSORY + BRAIN_RECURRENT;
// Brain outputs: 0=turn (yaw rate), 1=thrust, 2=pheromone modulation
// (positive = emit more above baseline, costs energy), 3=morph_length,
// 4=morph_width, 5=morph_spike, 6=attack, 7=turn_pitch (Sprint 33),
// 8=morph_height (Sprint 34, appended kvůli zachování existujících indexů).
// Sprint 26 morph signals: signal × MORPH_RATE × dt přičteno k phenotype dim
// každý tick, energy cost ∝ |delta|. Sprint 27 attack: gating signál pro
// `predate` — bez aktivního output[6] > THRESHOLD se predace nestane.
pub const BRAIN_OUTPUTS: usize = 9;
/// Inicializační bias na thrust output bin v `Brain::random`. Bez něj má ~½
/// random brainů thrust output blízko nuly (cell se sotva hýbe), což vytvářelo
/// hluboké bottlenecky v ranných generacích. Po prvním selekčním tlaku evoluce
/// hodnotu doladí — bias je jen jumpstart.
pub const INNATE_THRUST_BIAS: f32 = 2.0;
/// Inicializační bias na pheromone output (b2[2]). Sprint 25 vyžaduje aktivní
/// emisi pro reprodukci — bez biasu jen ~25 % párů projde threshold. S bias
/// 1.0 většina random brainů emituje nad threshold v gen 0; selekce pak ladí.
pub const INNATE_PHEROMONE_BIAS: f32 = 1.0;
/// Inicializační bias na attack output (b2[6]). Sprint 27: predace je opt-in,
/// ne default. Záměrně 0 — chceme měřit, jestli selekce attack chování objeví
/// sama, nebo zůstane utlumený. Negative bias by ho aktivně potlačoval.
pub const INNATE_ATTACK_BIAS: f32 = 0.0;

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

/// Sprint 38: gravitační zrychlení (sim units / sec²) působící na vše s mass.
/// Hodnota je effective gravity po vztlaku — reálná buňka v vodě má cca 5 %
/// netto force kvůli density ratio 1.05/1.0. Ve volném prostoru by 9.81 m/s²
/// dávalo nereálně rychlý sink; tady malé G + drag dá realistic sedimentation
/// (cells s pitch=0 a žádným thrustem klesnou postupně k dnu).
pub const GRAVITY: f32 = 5.0;
/// Sprint 38: terminal sink rate pro food (food nemá velocity field, pohybuje
/// se konstantní rychlostí dolů). Pomalejší než cells (které mohou plavat),
/// takže food drift k dnu = postupný „benthic deposit". 8 units/sec ~ 4 sec
/// průchod celé z-vrstvy (z=2).
pub const FOOD_SINK_RATE: f32 = 8.0;
/// Sprint 31 spatial clustering: rejection sampling síla. Per uniformně
/// vzorkovaný food candidate je pravděpodobnost zamítnutí
/// `STRENGTH × (1 - richness)`. Při richness=1 (rich zone) nikdy nezamítá,
/// při richness=0 (poor zone) zamítá s pravděpodobností STRENGTH. Sprint 21
/// v1–v5 zkoušel plnou sílu (`1 × (1 - richness)`) → extinkce gen 70–110.
/// 0.3 je kompromis: poor zone má 70 % šanci spawnu (vs. plná = 0 %), takže
/// food není ostře oddělená do biomů, jen mírný gradient. Plus value modulace
/// `WORLD_MAP_FOOD_FLOOR/AMP` jako safety net (i food v poor zone má hodnotu
/// ~85 % baselinu). MAX_SPAWN_ATTEMPTS retry budget zajišťuje, že clustering
/// nikdy úplně nezablokuje spawn.
pub const FOOD_REJECTION_STRENGTH: f32 = 0.3;
// Environmentální hazard layer: passive energy drain v "nebezpečných" zónách.
// Zónová mapa jde z `WorldMap` noise — POSITIVNÍ korelace s food richness:
// bohaté oblasti = nebezpečné (high reward, high risk), chudé = bezpečné.
// Vytváří trade-off niche: efficient cell může těžit rich-dangerous, slabší
// se uchýlí do safe-poor a žije s méně food. Drain za sec při noise=1:
// HAZARD_FLOOR + HAZARD_AMP = celkový max. Ladí se pouze v binárkách.
pub const HAZARD_DRAIN_PER_SEC: f32 = 0.5;
pub const HAZARD_FLOOR: f32 = 0.0;
pub const HAZARD_AMP: f32 = 1.0;

/// Sprint 30 self-preservation: gain pro tanh(damage_accum × GAIN) brain input
/// [14]. Damage_accum = nedobrovolný drain z minulého ticku (predation +
/// hazard). Single predation hit (drain = `PREDATION_DRAIN_PER_TICK` = 3.0)
/// → tanh(1.5) ≈ 0.90 (silný signál). Stabilní hazard ~0.008/tick → tanh ≈
/// 0.004 (neviditelný — chronický hazard nemá být „damage event", spatial
/// avoidance se vyvíjí přes selekční gradient na pozici, ne přes tento input).
/// Multi-attacker pile-on saturuje k 1.0.
pub const DAMAGE_NORMALIZATION_GAIN: f32 = 0.5;

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
/// Sprint 27: attacker.last_outputs[6].max(0) musí být > THRESHOLD aby se
/// predate-on-contact spustila. Bez aktivního brain signálu jsou `cell × cell`
/// kolize jen kolize — energy se nepřevádí. Mirroruje semantiku
/// `MATING_PHEROMONE_THRESHOLD` (input → behaviorální gate).
pub const ATTACK_THRESHOLD: f32 = 0.2;
/// Cena udržování attack módu: COST × max(0, output[6]) za sekundu, paid each
/// tick i když k predaci nedojde. Energie protiváhy "claws out". Bez ceny by
/// selekce favorizovala vždy-zapnutý attack output.
pub const ATTACK_COST_PER_SEC: f32 = 0.5;

/// Sprint 29 quorum sensing: kolik viditelných sousedů saturuje `local_density`
/// brain input (přes `tanh`). Kalibrováno na realistické počty: při typickém
/// vision_radius=50 a pop=200 v 1920×1080 vidí cell náhodně ~0.8 sousedů.
/// `DENSITY_NORM_COUNT=3` dává `tanh(0.8/3)≈0.26` (znatelný signál) a saturuje
/// kolem ~3 visible cells (skutečný cluster). 10 by drželo input pod noise floor.
pub const DENSITY_NORM_COUNT: f32 = 3.0;
/// Sprint 29 selfish-herd: poloměr, ve kterém se počítají sousedé prey pro
/// dilution. Záměrně **těsná** definice „v hejnu" — selfish-herd je o close
/// contact, ne o všech v MATING_RADIUS. Při HERD_RADIUS=50 a pop=200 dává
/// uniform distribuce ~0.76 sousedů, takže dilution při random pop = 1/(1+0.38)
/// ≈ 0.93 (skoro žádný efekt). Skutečný cluster s ~5 close neighbors srazí gain
/// na 1/(1+2.5)=0.29 — silný benefit pro shlukování bez plošného trestu.
pub const HERD_RADIUS: f32 = 50.0;
/// Sprint 29 predator-dilution faktor. `gain *= 1 / (1 + K × n_neighbors_prey)`.
/// K=0.5 → 5 sousedů → gain padá na ~30 %, 10 sousedů → ~17 %. Direct mapping
/// na Hamilton 1971 selfish-herd: bytí v hejnu snižuje atraktivitu cílů pro
/// predátora. Drain z prey zůstává plný — utrpení obětí se nemění, mění se jen
/// payoff útočníka.
pub const DILUTION_K: f32 = 0.5;

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

// Body morphology — Sprint 26. Tělo už není isotropní `body_size: f32`, ale
// 2 osy v body frame: `body_length` (podél heading), `body_width` (kolmo) +
// volitelný `spike_length` (frontální spike). Genom drží genetický template,
// `Phenotype` na cell drží runtime-modifiable hodnotu (genotype/phenotype
// split — runtime morph nemodifikuje gen, dítě dostane fresh phenotype z
// rodičovského genomu). Cena: per-tick maintenance ∝ length×width + spike,
// plus okamžitý cost ∝ rychlost morfingu.
pub const MIN_BODY_LENGTH: f32 = 0.3;
pub const MAX_BODY_LENGTH: f32 = 4.0;
pub const MIN_BODY_WIDTH: f32 = 0.3;
pub const MAX_BODY_WIDTH: f32 = 4.0;
/// Sprint 34: 3-axis ellipsoid — třetí dimenze (vertikálně, ⊥ k length+width).
pub const MIN_BODY_HEIGHT: f32 = 0.3;
pub const MAX_BODY_HEIGHT: f32 = 4.0;
pub const MIN_SPIKE_LENGTH: f32 = 0.0;
pub const MAX_SPIKE_LENGTH: f32 = 2.0;
/// Rychlost runtime morfingu — full brain output dává `MORPH_RATE` jednotek
/// změny tvaru za sekundu. 0.02/s znamená full-range body morph trvá ~50 gen
/// — pomalejší než životnost generace (~10s), takže morph je deliberátní akt
/// přes mnoho generací, ne ad-hoc tick reakce. Initial runs s 0.5/0.1/0.05
/// ukázaly "morph and starve": random brain biases (~0.5 stddev) × rychlý
/// MORPH_RATE → cells rostly o 100 % ve 4 gen → 3× body maintenance kost
/// dřív než selekce stihla optimalizovat brainy → extinkce.
pub const MORPH_RATE: f32 = 0.02;
/// Deadzone — pokud |signal| < threshold, morph se nepoužívá (žádná změna,
/// žádný cost). Filtruje šum z random brain biases (mean 0, ~0.5 stddev),
/// jen vědomě silné morph signály prochází. Threshold 0.7: prob(|tanh(N(0,1))|
/// většího než 0.7) ≈ 0.38, ~62 % random buněk neumí morphovat. Trénovaný
/// brain co chce morphovat dosáhne signálu blízko 1.0 → projde threshold.
pub const MORPH_ACTIVATION_THRESHOLD: f32 = 0.7;
/// Cena morfingu: `MORPH_COST_PER_DELTA × |Δ|` energie za faktická Δ tělesné
/// dimenze (po clampu). Lineární.
pub const MORPH_COST_PER_DELTA: f32 = 2.0;
/// Maintenance cost frontálního spike per sekunda na jednotku spike_length.
/// 0.3/s × 1.0 spike × 60 = 18/gen — srovnatelné s vision cost.
pub const SPIKE_COST_PER_SEC: f32 = 0.3;
/// Multiplier na `PREDATION_GAIN_PER_TICK` při frontálním spike hitu.
/// Bonus = `PREDATION_GAIN × spike_length × SPIKE_PREDATION_BONUS`.
/// 0.5: spike=1.0 dává +50 % predation gain, ne +100 % — méně dramatické,
/// ponechá místo pro non-spike strategie.
pub const SPIKE_PREDATION_BONUS: f32 = 0.5;
/// Cosinus úhlu mezi attacker.heading a vektorem k oběti, nutný pro spike
/// bonus. 0.7 ≈ 45° kužel — jen frontální zásahy.
pub const SPIKE_DOT_THRESHOLD: f32 = 0.7;
/// Pod tímto se spike nerenderuje (smoothness in viz, žádný flicker u
/// minimálních hodnot z gaussian mutací).
pub const SPIKE_RENDER_THRESHOLD: f32 = 0.05;

pub const MUTATION_CONFIG: MutationConfig = MutationConfig {
    sigma_speed: 3.0,
    sigma_hue: 5.0,
    sigma_vision: 3.0,
    sigma_turn_rate: 0.3,
    sigma_body_length: 0.05,
    sigma_body_width: 0.05,
    sigma_body_height: 0.05,
    sigma_spike_length: 0.03,
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
        // Innate attack bias: Sprint 27 — default 0 (opt-in). Mění se přes
        // konstantu, ne ad-hoc tady, ať se chování dá testovat A/B.
        b2[6] += INNATE_ATTACK_BIAS;
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
    pub sigma_body_length: f32,
    pub sigma_body_width: f32,
    pub sigma_body_height: f32,
    pub sigma_spike_length: f32,
    pub sigma_brain: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Genome {
    pub max_speed: f32,
    pub color_hue: f32,
    pub vision_radius: f32,
    pub turn_rate: f32,
    pub body_length: f32,
    pub body_width: f32,
    /// Sprint 34: 3-axis ellipsoid — vertikální rozměr.
    pub body_height: f32,
    pub spike_length: f32,
    pub brain: Brain,
}

impl Genome {
    pub fn random(rng: &mut impl Rng) -> Self {
        // Default tělo je izotropní koule (length == width == height). Mutace
        // mohou asymetrii vytvořit, ale jen pokud ji selekce odmění; gen 0
        // nezavádí prior na ellipse fenotyp.
        let body_size = rng.random_range(0.7..1.3);
        Self {
            max_speed: rng.random_range(30.0..90.0),
            color_hue: rng.random_range(0.0..HUE_RANGE),
            vision_radius: rng.random_range(20.0..80.0),
            turn_rate: rng.random_range(1.0..5.0),
            body_length: body_size,
            body_width: body_size,
            body_height: body_size,
            spike_length: rng.random_range(0.0..0.1),
            brain: Brain::random(rng),
        }
    }

    pub fn mutate(&self, rng: &mut impl Rng, cfg: &MutationConfig) -> Self {
        Self {
            max_speed: (self.max_speed + gaussian(rng) * cfg.sigma_speed).max(MIN_SPEED),
            color_hue: (self.color_hue + gaussian(rng) * cfg.sigma_hue).rem_euclid(HUE_RANGE),
            vision_radius: (self.vision_radius + gaussian(rng) * cfg.sigma_vision).max(MIN_VISION),
            turn_rate: (self.turn_rate + gaussian(rng) * cfg.sigma_turn_rate).max(MIN_TURN_RATE),
            body_length: (self.body_length + gaussian(rng) * cfg.sigma_body_length)
                .clamp(MIN_BODY_LENGTH, MAX_BODY_LENGTH),
            body_width: (self.body_width + gaussian(rng) * cfg.sigma_body_width)
                .clamp(MIN_BODY_WIDTH, MAX_BODY_WIDTH),
            body_height: (self.body_height + gaussian(rng) * cfg.sigma_body_height)
                .clamp(MIN_BODY_HEIGHT, MAX_BODY_HEIGHT),
            spike_length: (self.spike_length + gaussian(rng) * cfg.sigma_spike_length)
                .clamp(MIN_SPIKE_LENGTH, MAX_SPIKE_LENGTH),
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
            body_length: if rng.random::<bool>() { a.body_length } else { b.body_length },
            body_width: if rng.random::<bool>() { a.body_width } else { b.body_width },
            body_height: if rng.random::<bool>() { a.body_height } else { b.body_height },
            spike_length: if rng.random::<bool>() { a.spike_length } else { b.spike_length },
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

/// Runtime tělesný tvar buňky. Inicializuje se z `Genome` při spawnu /
/// reprodukci (template) a může se měnit za běhu života přes `apply_morph`
/// (řízeno brain output[3..6]). **Genotyp/fenotyp split**: runtime morph
/// modifikuje `Phenotype`, ne `Genome`. Dítě dostane svůj fresh phenotype
/// z rodičovského genomu — žádný Lamarckismus.
#[derive(Debug, Clone, Copy)]
pub struct Phenotype {
    pub body_length: f32,
    pub body_width: f32,
    /// Sprint 34: vertikální rozměr ellipsoidu.
    pub body_height: f32,
    pub spike_length: f32,
}

impl Phenotype {
    pub fn from_genome(genome: &Genome) -> Self {
        Self {
            body_length: genome.body_length,
            body_width: genome.body_width,
            body_height: genome.body_height,
            spike_length: genome.spike_length,
        }
    }

    /// Proxy pro circular-collision codepaths (eat radius, broad phase).
    /// Sprint 34: aritmetický průměr 3 os; když length=width=height=s, dostane s
    /// — backward compat s pre-Sprint-34 izotropním tělem.
    pub fn effective_radius(&self) -> f32 {
        (self.body_length + self.body_width + self.body_height) / 3.0
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
        let new_spk = (self.spike_length + raw_ds).clamp(MIN_SPIKE_LENGTH, MAX_SPIKE_LENGTH);

        let actual_dl = (new_len - self.body_length).abs();
        let actual_dw = (new_wid - self.body_width).abs();
        let actual_dh = (new_hgt - self.body_height).abs();
        let actual_ds = (new_spk - self.spike_length).abs();

        self.body_length = new_len;
        self.body_width = new_wid;
        self.body_height = new_hgt;
        self.spike_length = new_spk;

        actual_dl + actual_dw + actual_dh + actual_ds
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Cell {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    /// Yaw rate — rotace okolo vertikální osy. Brain output[0].
    pub angular_velocity: f32,
    /// Sprint 33 pitch rate — rotace okolo horizontální osy kolmé k forwardu
    /// v xy. Brain output[7]. Drag stejný jako u yawu.
    pub pitch_velocity: f32,
    pub energy: f32,
    // Persists even when velocity hits zero — atan2(0, 0) would otherwise
    // collapse to 0 and bias evolution toward east-facing motion.
    pub heading: f32,
    /// Sprint 33: pitch úhel, klampovaný do [-π/2, π/2]. Spolu s heading (yaw)
    /// definuje forward unit vector: (cos(y)·cos(p), sin(y)·cos(p), sin(p)).
    /// Roll vynechán — cells axially symetrické.
    pub pitch: f32,
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
    /// Sprint 30: nedobrovolný energy drain akumulovaný v aktuálním ticku
    /// (predation + hazard). Brain ho čte v dalším ticku jako input[14]
    /// (damage signal), pak resetuje na 0. Voluntární cost se NEZAPISUJE
    /// — cell sama drives ty náklady přes outputs, není to externí útok.
    pub damage_accum: f32,
    pub phenotype: Phenotype,
    pub genome: Genome,
}

impl Cell {
    pub fn random(
        rng: &mut impl Rng,
        world_half: [f32; 3],
        lineage_id: u64,
        lineage_birth_gen: u64,
    ) -> Self {
        let genome = Genome::random(rng);
        Self::from_genome(rng, genome, world_half, lineage_id, lineage_birth_gen)
    }

    pub fn from_genome(
        rng: &mut impl Rng,
        genome: Genome,
        world_half: [f32; 3],
        lineage_id: u64,
        lineage_birth_gen: u64,
    ) -> Self {
        let direction = rng.random_range(0.0..TAU);
        let phenotype = Phenotype::from_genome(&genome);
        // Sprint 32 substrate: z-osa drží determinismus pre-3D semantiky.
        // Když world_half[2] == 0 (Sprint 32 default), z=0 a žádný RNG draw —
        // zachová bit-identical CSV trajektorii. Sprint 33+ unlockne motion +
        // initial draw přes celou krabici.
        let pos_z = if world_half[2] > 0.0 {
            rng.random_range(-world_half[2]..world_half[2])
        } else {
            0.0
        };
        Self {
            position: [
                rng.random_range(-world_half[0]..world_half[0]),
                rng.random_range(-world_half[1]..world_half[1]),
                pos_z,
            ],
            velocity: [
                direction.cos() * genome.max_speed,
                direction.sin() * genome.max_speed,
                0.0,
            ],
            angular_velocity: 0.0,
            pitch_velocity: 0.0,
            energy: INITIAL_ENERGY,
            heading: direction,
            // Sprint 33: cells startují horizontálně (pitch=0). Pohyb v z se
            // vyvine přes brain output[7] = turn_pitch. Random initial pitch
            // by zkonzumoval RNG draw a porušil pre-Sprint-33 reproducibility,
            // pokud bychom chtěli A/B; horizontální start je clean baseline.
            pitch: 0.0,
            lineage_id,
            lineage_birth_gen,
            last_inputs: [0.0; BRAIN_INPUTS],
            last_hidden: [0.0; BRAIN_HIDDEN],
            last_outputs: [0.0; BRAIN_OUTPUTS],
            damage_accum: 0.0,
            phenotype,
            genome,
        }
    }

    /// Per-tick physics: kinematic update from velocity / angular_velocity,
    /// quadratic drag on both, energy drains. Brain-applied forces should be
    /// integrated into velocity / angular_velocity *before* step (in
    /// `cells_brain_act`); step is purely passive integration + dissipation.
    ///
    /// Sprint 26 anisotropic drag: rozložím velocity na heading-aligned (par)
    /// vs heading-perpendicular (perp) komponentu. Cross-section je v body
    /// frame: forward motion „cítí" width (frontální), sideways motion cítí
    /// length. Pro length=width=s drag přesně reprodukuje původní isotropní.
    pub fn step(&mut self, dt: f32, world_half: [f32; 3], physics: &PhysicsConfig) {
        self.position[0] += self.velocity[0] * dt;
        self.position[1] += self.velocity[1] * dt;
        self.position[2] += self.velocity[2] * dt;
        self.heading += self.angular_velocity * dt;
        // Sprint 38: gravity působí pouze pokud je z-volume aktivní. Aplikuje
        // se před drag aby drag mohl balancovat → terminal velocity.
        if world_half[2] > 0.0 {
            self.velocity[2] -= GRAVITY * dt;
        }
        // Sprint 35: pitch range ±π/12 (=15°). Velmi konzervativní — Sprint 37
        // ladí. Random brain pitch noise s tight range = nepatrný drift v z.
        self.pitch = (self.pitch + self.pitch_velocity * dt).clamp(
            -core::f32::consts::FRAC_PI_6 * 0.5,
            core::f32::consts::FRAC_PI_6 * 0.5,
        );

        // Sprint 33: 3D anisotropic drag. Forward = (cos(y)·cos(p), sin(y)·cos(p),
        // sin(p)). Velocity rozdělíme na along-forward a perpendicular-forward.
        // Forward "cítí" width × length (cross-section frontu), perpendicular
        // cítí length × width (boční cross-section). V Sprint 34 se přidá height
        // do anisotropic split — zatím length/width proxy stačí.
        let cos_y = self.heading.cos();
        let sin_y = self.heading.sin();
        let cos_p = self.pitch.cos();
        let sin_p = self.pitch.sin();
        let fx = cos_y * cos_p;
        let fy = sin_y * cos_p;
        let fz = sin_p;
        let v_par = self.velocity[0] * fx + self.velocity[1] * fy + self.velocity[2] * fz;
        let v_perp_x = self.velocity[0] - v_par * fx;
        let v_perp_y = self.velocity[1] - v_par * fy;
        let v_perp_z = self.velocity[2] - v_par * fz;
        let v_perp_mag =
            (v_perp_x * v_perp_x + v_perp_y * v_perp_y + v_perp_z * v_perp_z).sqrt();
        let length = self.phenotype.body_length;
        let width = self.phenotype.body_width;
        let drag_par_factor = physics.drag * v_par.abs() * width * dt;
        let drag_perp_factor = physics.drag * v_perp_mag * length * dt;
        let new_v_par = v_par - drag_par_factor * v_par;
        let new_v_perp_x = v_perp_x - drag_perp_factor * v_perp_x;
        let new_v_perp_y = v_perp_y - drag_perp_factor * v_perp_y;
        let new_v_perp_z = v_perp_z - drag_perp_factor * v_perp_z;
        self.velocity[0] = new_v_par * fx + new_v_perp_x;
        self.velocity[1] = new_v_par * fy + new_v_perp_y;
        self.velocity[2] = new_v_par * fz + new_v_perp_z;

        let drag_factor = (1.0 - physics.angular_drag * dt).max(0.0);
        self.angular_velocity *= drag_factor;
        self.pitch_velocity *= drag_factor;

        // Energy: kinetic v², rotační ω² (yaw + pitch), vision, area-based
        // maintenance, spike maintenance.
        // Sprint 33: v² + ω² teď zahrnují 3D komponenty.
        let v_mag_sq =
            self.velocity[0].powi(2) + self.velocity[1].powi(2) + self.velocity[2].powi(2);
        self.energy -= v_mag_sq * physics.energy_cost_per_v_sq * dt;
        // Sprint 33: angular energy stále jen yaw (matches pre-Sprint-33).
        // Pitch je „free" rotace pro Sprint 33 — random brain biases mají
        // ~equal magnitude yaw + pitch outputs, takže započítání obou by
        // zdvojnásobilo rotační drain a tlačilo random brainy do extinkce.
        // Sprint 37 evaluuje, jestli pitch cost přidat (a v jaké váze).
        let av = self.angular_velocity;
        let eff_r = self.phenotype.effective_radius();
        self.energy -= eff_r * eff_r * av * av * physics.angular_energy_cost * dt;
        self.energy -= self.genome.vision_radius * physics.vision_cost_per_radius * dt;
        // Sprint 34: maintenance ∝ 3D volume = length×width×height. Pro
        // height=1 (pre-Sprint-34 default) se redukuje na 2D area, takže
        // pre-Sprint-34 cells s height=1 platí stejně jako dřív.
        self.energy -= self.phenotype.volume() * physics.body_cost_factor * dt;
        self.energy -= self.phenotype.spike_length * SPIKE_COST_PER_SEC * dt;
        // Sprint 27 attack maintenance: brain v "claws out" módu platí, i když
        // k predaci nedojde. Bez ceny by selekce favorizovala vždy-zapnutý
        // attack output. Cost ∝ max(0, output[6]) — negativní output je
        // "passive" a nestojí nic.
        let attack_strength = self.last_outputs[6].max(0.0);
        self.energy -= attack_strength * ATTACK_COST_PER_SEC * dt;

        let mut bounced_xy = false;
        if self.position[0].abs() > world_half[0] {
            self.velocity[0] = -self.velocity[0];
            self.position[0] = self.position[0].clamp(-world_half[0], world_half[0]);
            bounced_xy = true;
        }
        if self.position[1].abs() > world_half[1] {
            self.velocity[1] = -self.velocity[1];
            self.position[1] = self.position[1].clamp(-world_half[1], world_half[1]);
            bounced_xy = true;
        }
        // Sprint 32: z-bounce active jen když half_z > 0 (Sprint 33+); pro z=0
        // locked režim je position[2]=0, abs() > 0.0 false → no-op, identické
        // s pre-Sprint-32 chováním.
        if world_half[2] > 0.0 && self.position[2].abs() > world_half[2] {
            self.velocity[2] = -self.velocity[2];
            self.position[2] = self.position[2].clamp(-world_half[2], world_half[2]);
        }
        if bounced_xy {
            // Heading recompute z xy velocity. Pitch zůstává — bounce na
            // xy zdi je horizontální event, neměl by smazat vertikální orientaci.
            self.heading = self.velocity[1].atan2(self.velocity[0]);
        }
        // Sprint 33: bounce na strop/podlahu zachová yaw, ale obrátí vz —
        // pitch dopočteme z dot(forward, up). atan2(vz, |v_xy|) by byl ideální
        // ale velocity není zaručeně v body frame. Necháme pitch beze změny;
        // brain musí reagovat na vertikální bounce sám.
    }

    /// Aplikuje runtime morph z brain outputs na phenotype + naúčtuje
    /// energii za realizované Δ. Volá se mezi `brain_act` a `step`.
    /// Sprint 34: morph teď řídí 4 dimenze (length, width, height, spike).
    /// Mapování indexů: [3]=length, [4]=width, [8]=height (Sprint 34, append),
    /// [5]=spike. Index [8] je za attack[6] a turn_pitch[7], které předtím
    /// existovaly — nový height index zachoval stávající indexy beze změny.
    pub fn apply_morph(&mut self, dt: f32) {
        let morph = [
            self.last_outputs[3],
            self.last_outputs[4],
            self.last_outputs[8],
            self.last_outputs[5],
        ];
        let total_delta = self.phenotype.apply_morph(morph, MORPH_RATE, dt);
        self.energy -= MORPH_COST_PER_DELTA * total_delta;
    }

    /// Bonus predation gain pokud má attacker spike a heading je zaměřený
    /// na cíl (cosine > `SPIKE_DOT_THRESHOLD`). Vrací 0 jinak. Volá se z
    /// predate cyklu v rendereru/headlessu nad rámec běžného size-ratio
    /// gainu.
    pub fn spike_bonus_against(&self, target_pos: [f32; 3]) -> f32 {
        if self.phenotype.spike_length <= 0.0 {
            return 0.0;
        }
        let dx = target_pos[0] - self.position[0];
        let dy = target_pos[1] - self.position[1];
        let dz = target_pos[2] - self.position[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        if dist_sq < f32::EPSILON {
            return 0.0;
        }
        let dist = dist_sq.sqrt();
        // Sprint 33: full 3D forward unit vector (yaw + pitch).
        let [fx, fy, fz] = forward_vector(self.heading, self.pitch);
        let cos_angle = (dx * fx + dy * fy + dz * fz) / dist;
        if cos_angle < SPIKE_DOT_THRESHOLD {
            return 0.0;
        }
        PREDATION_GAIN_PER_TICK * self.phenotype.spike_length * SPIKE_PREDATION_BONUS
    }

    pub fn try_eat(&mut self, food: &Food, eat_radius: f32, food_value: f32) -> bool {
        let dx = self.position[0] - food.position[0];
        let dy = self.position[1] - food.position[1];
        let dz = self.position[2] - food.position[2];
        if dx * dx + dy * dy + dz * dz <= eat_radius * eat_radius {
            self.energy += food_value;
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Food {
    pub position: [f32; 3],
}

impl Food {
    pub fn random(rng: &mut impl Rng, world_half: [f32; 3]) -> Self {
        // Sprint 32: z-osa conditional pro deterministický CSV; world_half[2]=0
        // → z=0 bez RNG draw.
        let z = if world_half[2] > 0.0 {
            rng.random_range(-world_half[2]..world_half[2])
        } else {
            0.0
        };
        Self {
            position: [
                rng.random_range(-world_half[0]..world_half[0]),
                rng.random_range(-world_half[1]..world_half[1]),
                z,
            ],
        }
    }

    /// Sprint 38: aplikuje gravitační drift food (sink). Pouze pokud je
    /// z-volume aktivní; jinak no-op (Sprint 32 z=0 setup).
    pub fn apply_gravity(&mut self, dt: f32, world_half_z: f32) {
        if world_half_z <= 0.0 {
            return;
        }
        self.position[2] = (self.position[2] - FOOD_SINK_RATE * dt).max(-world_half_z);
    }
}

/// Sprint 33: 3D forward unit vector z yaw + pitch. Bez roll (cells axially
/// symetrické). Pro pitch=0 redukuje na (cos(yaw), sin(yaw), 0) — backward
/// kompat s pre-Sprint-33 2D heading semantikou.
pub fn forward_vector(yaw: f32, pitch: f32) -> [f32; 3] {
    let cos_p = pitch.cos();
    [yaw.cos() * cos_p, yaw.sin() * cos_p, pitch.sin()]
}

/// Sprint 31: rejection test pro spatial food clustering. Vrací `true` =
/// kandidát zamítnout (zkusit jinou pozici). Probability rejection =
/// `FOOD_REJECTION_STRENGTH × (1 - richness)`. Volá se per uniformně
/// vzorkovaný kandidát; spawn loop drží retry budget (`MAX_SPAWN_ATTEMPTS`),
/// takže clustering jen ladí distribuci, neblokuje úplně.
pub fn reject_food_for_richness(rng: &mut impl Rng, richness: f32) -> bool {
    let r = richness.clamp(0.0, 1.0);
    rng.random::<f32>() < FOOD_REJECTION_STRENGTH * (1.0 - r)
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
            body_length: 1.0,
            body_width: 1.0,
            body_height: 1.0,
            spike_length: 0.0,
            brain: dummy_brain(),
        }
    }

    fn zero_cfg() -> MutationConfig {
        MutationConfig {
            sigma_speed: 0.0,
            sigma_hue: 0.0,
            sigma_vision: 0.0,
            sigma_turn_rate: 0.0,
            sigma_body_length: 0.0,
            sigma_body_width: 0.0,
            sigma_body_height: 0.0,
            sigma_spike_length: 0.0,
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
            body_length: 1.1,
            body_width: 0.9,
            body_height: 1.0,
            spike_length: 0.4,
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
        assert_eq!(m.body_length, 1.1);
        assert_eq!(m.body_width, 0.9);
        assert_eq!(m.spike_length, 0.4);
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
            sigma_body_length: 10.0,
            sigma_body_width: 10.0,
            sigma_body_height: 10.0,
            sigma_spike_length: 10.0,
            sigma_brain: 10.0,
        };
        for _ in 0..1000 {
            let m = g.mutate(&mut rng, &cfg);
            assert!(m.max_speed >= MIN_SPEED);
            assert!(m.color_hue >= 0.0 && m.color_hue < HUE_RANGE);
            assert!(m.vision_radius >= MIN_VISION);
            assert!(m.turn_rate >= MIN_TURN_RATE);
            assert!((MIN_BODY_LENGTH..=MAX_BODY_LENGTH).contains(&m.body_length));
            assert!((MIN_BODY_WIDTH..=MAX_BODY_WIDTH).contains(&m.body_width));
            assert!((MIN_SPIKE_LENGTH..=MAX_SPIKE_LENGTH).contains(&m.spike_length));
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
        let genome = dummy_genome();
        let phenotype = Phenotype::from_genome(&genome);
        Cell {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            angular_velocity: 0.0,
            pitch_velocity: 0.0,
            energy: 100.0,
            heading: 0.0,
            pitch: 0.0,
            lineage_id: 0,
            lineage_birth_gen: 0,
            last_inputs: [0.0; BRAIN_INPUTS],
            last_hidden: [0.0; BRAIN_HIDDEN],
            last_outputs: [0.0; BRAIN_OUTPUTS],
            damage_accum: 0.0,
            phenotype,
            genome,
        }
    }

    #[test]
    fn step_drains_energy_from_motion_and_vision() {
        let mut cell = Cell {
            velocity: [60.0, 0.0, 0.0],
            ..base_cell()
        };
        cell.step(1.0, [1000.0, 1000.0, 0.0], &no_drag_physics(0.001, 0.05));
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
            position: [99.0, 0.0, 0.0],
            velocity: [60.0, 0.0, 0.0],
            ..base_cell()
        };
        cell.step(1.0, [100.0, 100.0, 0.0], &no_drag_physics(0.0, 0.0));
        // velocity flipped to (-60, 0), heading should now be π.
        assert!((cell.heading - core::f32::consts::PI).abs() < 1e-4);
    }

    #[test]
    fn step_preserves_heading_when_velocity_zero() {
        let mut cell = Cell {
            heading: 1.5,
            ..base_cell()
        };
        cell.step(1.0, [100.0, 100.0, 0.0], &no_drag_physics(0.0, 0.0));
        // No movement, no bounce, no angular velocity, heading must persist.
        assert_eq!(cell.heading, 1.5);
    }

    #[test]
    fn step_applies_quadratic_drag() {
        let mut cell = Cell {
            velocity: [10.0, 0.0, 0.0],
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
        cell.step(1.0, [1000.0, 1000.0, 0.0], &physics);
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
        cell.step(1.0, [1000.0, 1000.0, 0.0], &physics);
        // effective_radius²(=1) × ω²(=4) × angular_cost(=0.05) × dt(=1) = 0.2 drained
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
        cell.step(1.0, [1000.0, 1000.0, 0.0], &physics);
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
        cell.step(1.0, [1000.0, 1000.0, 0.0], &physics);
        // angular_velocity *= (1 − 0.5 × 1) = 0.5 → 0.5
        assert!((cell.angular_velocity - 0.5).abs() < 1e-4);
    }

    #[test]
    fn try_eat_within_radius_returns_true_and_adds_energy() {
        let mut cell = Cell {
            energy: 50.0,
            ..base_cell()
        };
        let food = Food { position: [5.0, 0.0, 0.0] };
        assert!(cell.try_eat(&food, 8.0, 20.0));
        assert_eq!(cell.energy, 70.0);
    }

    #[test]
    fn try_eat_outside_radius_returns_false_and_keeps_energy() {
        let mut cell = Cell {
            energy: 50.0,
            ..base_cell()
        };
        let food = Food { position: [20.0, 0.0, 0.0] };
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
            body_length: 0.5,
            body_width: 0.6,
            body_height: 0.7,
            spike_length: 0.0,
            brain: dummy_brain(),
        };
        let b = Genome {
            max_speed: 90.0,
            color_hue: 200.0,
            vision_radius: 80.0,
            turn_rate: 5.0,
            body_length: 1.5,
            body_width: 1.4,
            body_height: 1.3,
            spike_length: 0.8,
            brain: dummy_brain(),
        };
        for _ in 0..100 {
            let c = Genome::crossover(&a, &b, &mut rng);
            assert!(c.max_speed == 30.0 || c.max_speed == 90.0);
            assert!(c.color_hue == 10.0 || c.color_hue == 200.0);
            assert!(c.vision_radius == 20.0 || c.vision_radius == 80.0);
            assert!(c.turn_rate == 1.0 || c.turn_rate == 5.0);
            assert!(c.body_length == 0.5 || c.body_length == 1.5);
            assert!(c.body_width == 0.6 || c.body_width == 1.4);
            assert!(c.spike_length == 0.0 || c.spike_length == 0.8);
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

    #[test]
    fn morph_zero_signal_does_not_change_phenotype() {
        let mut phen = Phenotype {
            body_length: 1.5,
            body_width: 0.8,
            body_height: 1.0,
            spike_length: 0.3,
        };
        let delta = phen.apply_morph([0.0, 0.0, 0.0, 0.0], MORPH_RATE, 0.5);
        assert_eq!(delta, 0.0);
        assert_eq!(phen.body_length, 1.5);
        assert_eq!(phen.body_width, 0.8);
        assert_eq!(phen.body_height, 1.0);
        assert_eq!(phen.spike_length, 0.3);
    }

    #[test]
    fn morph_clamps_to_min_max_bounds() {
        let mut phen = Phenotype {
            body_length: MAX_BODY_LENGTH,
            body_width: MIN_BODY_WIDTH,
            body_height: MAX_BODY_HEIGHT,
            spike_length: MAX_SPIKE_LENGTH,
        };
        // Strong positive signal on length, height & spike (already at max) → no change.
        // Strong negative signal on width (already at min) → no change.
        let delta = phen.apply_morph([1.0, -1.0, 1.0, 1.0], 100.0, 1.0);
        assert_eq!(delta, 0.0);
        assert_eq!(phen.body_length, MAX_BODY_LENGTH);
        assert_eq!(phen.body_width, MIN_BODY_WIDTH);
        assert_eq!(phen.body_height, MAX_BODY_HEIGHT);
        assert_eq!(phen.spike_length, MAX_SPIKE_LENGTH);
    }

    #[test]
    fn morph_returns_total_absolute_delta() {
        let mut phen = Phenotype {
            body_length: 1.0,
            body_width: 1.0,
            body_height: 1.0,
            spike_length: 0.5,
        };
        // signal × rate × dt = 0.8 × 1.0 × 1.0 = 0.8 podél každé osy.
        // Width clampuje na MIN_BODY_WIDTH (0.3), takže |Δ| pro width je
        // 1.0 - 0.3 = 0.7. Total |Δ| = 0.8 (length) + 0.7 (width clamped)
        // + 0.0 (height: signal=0) + 0.8 (spike) = 2.3.
        let delta = phen.apply_morph([0.8, -0.8, 0.0, 0.8], 1.0, 1.0);
        assert!((delta - 2.3).abs() < 1e-5, "got {}", delta);
        assert!((phen.body_length - 1.8).abs() < 1e-5);
        assert!((phen.body_width - MIN_BODY_WIDTH).abs() < 1e-5);
        assert!((phen.spike_length - 1.3).abs() < 1e-5);
    }

    #[test]
    fn morph_signal_below_threshold_is_deadzoned() {
        let mut phen = Phenotype {
            body_length: 1.0,
            body_width: 1.0,
            body_height: 1.0,
            spike_length: 0.0,
        };
        // |signal| < threshold → no change (filters random brain noise).
        let delta = phen.apply_morph(
            [
                MORPH_ACTIVATION_THRESHOLD - 0.01,
                -MORPH_ACTIVATION_THRESHOLD + 0.01,
                0.0,
                0.0,
            ],
            1.0,
            1.0,
        );
        assert_eq!(delta, 0.0);
        assert_eq!(phen.body_length, 1.0);
        assert_eq!(phen.body_width, 1.0);
        assert_eq!(phen.spike_length, 0.0);
    }

    #[test]
    fn cell_apply_morph_updates_phenotype_not_genome() {
        // Genotype/phenotype split: runtime morph nesmí sahat na genome.
        let mut cell = base_cell();
        let original_genome_len = cell.genome.body_length;
        cell.last_outputs[3] = 1.0; // morph_length signal
        cell.apply_morph(1.0);
        assert!(cell.phenotype.body_length > original_genome_len);
        assert_eq!(cell.genome.body_length, original_genome_len);
    }

    #[test]
    fn anisotropic_drag_slower_along_width_when_elongated() {
        // Cell s length=2, width=1, heading=0, motion (10,0) (forward) vs (0,10)
        // (sideways). Forward "cítí" width (=1) jako cross-section, sideways
        // cítí length (=2). Sideways must therefore decay faster.
        let physics = PhysicsConfig {
            drag: 0.01,
            angular_drag: 0.0,
            energy_cost_per_v_sq: 0.0,
            angular_energy_cost: 0.0,
            vision_cost_per_radius: 0.0,
            body_cost_factor: 0.0,
        };
        let make_cell = |vel: [f32; 3]| {
            let mut c = base_cell();
            c.phenotype = Phenotype {
                body_length: 2.0,
                body_width: 1.0,
                body_height: 1.0,
                spike_length: 0.0,
            };
            c.velocity = vel;
            c
        };
        let mut forward = make_cell([10.0, 0.0, 0.0]);
        let mut sideways = make_cell([0.0, 10.0, 0.0]);
        forward.step(1.0, [1000.0, 1000.0, 0.0], &physics);
        sideways.step(1.0, [1000.0, 1000.0, 0.0], &physics);
        // |v| forward after step: 10 - drag·|v|·width·v = 10 - 0.01·10·1·10 = 9.0
        // |v| sideways after step: 10 - 0.01·10·2·10 = 8.0
        let v_forward = forward.velocity[0].hypot(forward.velocity[1]);
        let v_sideways = sideways.velocity[0].hypot(sideways.velocity[1]);
        assert!(v_forward > v_sideways, "forward {} should be > sideways {}", v_forward, v_sideways);
        assert!((v_forward - 9.0).abs() < 1e-3, "forward got {}", v_forward);
        assert!((v_sideways - 8.0).abs() < 1e-3, "sideways got {}", v_sideways);
    }

    #[test]
    fn anisotropic_drag_isotropic_when_axes_equal() {
        // Když length=width=1, anisotropic verze musí dát stejný výsledek jako
        // původní isotropic (regression test pro `step_applies_quadratic_drag`).
        let mut cell = base_cell();
        cell.velocity = [10.0, 0.0, 0.0];
        let physics = PhysicsConfig {
            drag: 0.01,
            angular_drag: 0.0,
            energy_cost_per_v_sq: 0.0,
            angular_energy_cost: 0.0,
            vision_cost_per_radius: 0.0,
            body_cost_factor: 0.0,
        };
        cell.step(1.0, [1000.0, 1000.0, 0.0], &physics);
        assert!((cell.velocity[0] - 9.0).abs() < 1e-4);
    }

    #[test]
    fn spike_bonus_only_when_target_in_front_cone() {
        let mut cell = base_cell();
        cell.position = [0.0, 0.0, 0.0];
        cell.heading = 0.0; // pointing +x
        cell.phenotype.spike_length = 1.0;

        // Target přímo vepředu — bonus se aplikuje.
        let bonus_front = cell.spike_bonus_against([10.0, 0.0, 0.0]);
        assert!(bonus_front > 0.0);

        // Target za zády — bonus = 0.
        let bonus_back = cell.spike_bonus_against([-10.0, 0.0, 0.0]);
        assert_eq!(bonus_back, 0.0);

        // Target přesně na boku — bonus = 0 (cosine = 0 < threshold 0.7).
        let bonus_side = cell.spike_bonus_against([0.0, 10.0, 0.0]);
        assert_eq!(bonus_side, 0.0);
    }

    #[test]
    fn spike_bonus_zero_when_no_spike() {
        let mut cell = base_cell();
        cell.heading = 0.0;
        cell.phenotype.spike_length = 0.0;
        let bonus = cell.spike_bonus_against([10.0, 0.0, 0.0]);
        assert_eq!(bonus, 0.0);
    }

    #[test]
    fn food_rejection_never_rejects_at_max_richness() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..10_000 {
            assert!(!reject_food_for_richness(&mut rng, 1.0));
        }
    }

    #[test]
    fn food_rejection_rate_at_min_richness_matches_strength() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 100_000;
        let rejected = (0..n)
            .filter(|_| reject_food_for_richness(&mut rng, 0.0))
            .count();
        let observed = rejected as f32 / n as f32;
        // Tolerance ±0.01 for sample noise on 100k draws (~3σ for p=0.3).
        assert!(
            (observed - FOOD_REJECTION_STRENGTH).abs() < 0.01,
            "observed reject rate {} vs expected {}",
            observed,
            FOOD_REJECTION_STRENGTH
        );
    }

    #[test]
    fn step_3d_position_advances_with_z_velocity() {
        // Sprint 32 sanity: z-složka pozice musí integrovat z velocity stejně
        // jako x/y, takže Sprint 33+ má pevnou základnu.
        let mut cell = base_cell();
        cell.velocity = [0.0, 0.0, 5.0];
        cell.step(1.0, [1000.0, 1000.0, 1000.0], &no_drag_physics(0.0, 0.0));
        assert!(
            (cell.position[2] - 5.0).abs() < 1e-4,
            "expected z=5.0, got {}",
            cell.position[2]
        );
    }

    #[test]
    fn z_locked_world_keeps_food_planar() {
        // Sprint 32: world_half[2] = 0 znamená Food::random vrací z=0 a
        // nespotřebovává RNG draw na z. Critical pro CSV identity.
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..1_000 {
            let f = Food::random(&mut rng, [100.0, 100.0, 0.0]);
            assert_eq!(f.position[2], 0.0);
        }
    }

    #[test]
    fn step_drains_energy_from_spike_maintenance() {
        let mut cell = base_cell();
        cell.phenotype.spike_length = 0.5;
        let physics = PhysicsConfig {
            drag: 0.0,
            angular_drag: 0.0,
            energy_cost_per_v_sq: 0.0,
            angular_energy_cost: 0.0,
            vision_cost_per_radius: 0.0,
            body_cost_factor: 0.0,
        };
        cell.step(1.0, [1000.0, 1000.0, 0.0], &physics);
        // spike_length(=0.5) × SPIKE_COST_PER_SEC × dt(=1) = 0.15 drained
        let expected_drain = 0.5 * SPIKE_COST_PER_SEC;
        assert!(
            (cell.energy - (100.0 - expected_drain)).abs() < 1e-4,
            "got {}, expected {}",
            cell.energy,
            100.0 - expected_drain
        );
    }
}
