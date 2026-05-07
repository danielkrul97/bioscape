use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::*;

/// = jen 0.17 energy/cell/gen damage, což je 18× méně než bond
/// maintenance 3.0/gen → solo dominuje. 12 hunterů + víc damage zvedá
/// solo cost.
pub const HUNTER_TARGET_COUNT: usize = 12;
/// Vision range Hunteru — vidí cells v této vzdálenosti, aktivně k nim
/// míří. Mimo range = random walk. Sprint 72 zvětšen 120 → 200 (větší
/// než MATING_RADIUS=200, takže hunters detekují cells dřív než stihnou
/// utéct přes broad-phase fail).
pub const HUNTER_VISION_RADIUS: f32 = 200.0;
/// Attack range — Hunter dealuje damage cells uvnitř této vzdálenosti.
/// Menší než vision, takže Hunter musí se sblížit (cells mají šanci utéct).
pub const HUNTER_ATTACK_RADIUS: f32 = 18.0;
/// Damage per tick, který Hunter působí cells v attack range. Aplikuje se
/// jako energy loss + damage_accum (brain damage signal). Lineární per tick,
/// no spike bonus (Hunter je sám bez evolved phenotype). Sprint 74: 4 → 8
/// (2× lethaler). Sprint 73 ukázalo, že solo cells absorbovaly 0.17/gen
/// damage bez problému; 2× zvýší to k ~0.35, což stále nemusí stačit, ale
/// kombinováno s víc hunterů + levnějším bondem se dostávámě k flip pointu.
pub const HUNTER_DAMAGE_PER_TICK: f32 = 8.0;
/// Hunter top speed. Sprint 71 měl 220 — cells dotáhly průměr na 218 a
/// outrun se ukázal jako viable escape route (asp 12.6 pure speedy cells).
/// Sprint 72: 300, nad teoretický cell max_speed cap. Cells nemůžou outrun
/// jen rychlostí — cluster path (≥3 bondy = immunity) musí být dominantní
/// úniková strategie.
pub const HUNTER_MAX_SPEED: f32 = 300.0;
/// Acceleration coefficient. Hunter nemá brain, jen target-seeking velocity
/// adjustment. dt × ACC × (target_dir - current_dir) škáluje na max_speed.
pub const HUNTER_ACC: f32 = 80.0;
/// Hunter random-walk noise při idle (no cell ve vision). Pomaly drift,
/// brzy najde cell + zacílí.
pub const HUNTER_IDLE_DRIFT: f32 = 30.0;
/// Bond count threshold pro **discoverability** Hunteru. Cells s ≥ tomuto
/// počtu bondů Hunter nepronásleduje (`nearest_attackable_cell` vrací None) —
/// cluster je „too deeply interior to bother". Sprint 92: 2 → 4 — replaces
/// binary immunity s gradient damage. Hunters chase low-bond cells normally,
/// vysoký bond count = invisible target (čisté efficiency rozhodnutí).
pub const HUNTER_BOND_IMMUNITY_THRESHOLD: u32 = 4;
/// Sprint 92: per-bond exposure reduction. `exposure = max(0, 1 - n_bonds × this)`.
/// 0.25 → 0/1/2/3+ bonds dávají exposure 1.0/0.75/0.5/0.25. Při 4+ bonds
/// exposure floor = 0 a hunter neuvidí target (HUNTER_BOND_IMMUNITY_THRESHOLD).
pub const EXPOSURE_PER_BOND: f32 = 0.25;

/// Sprint 97: per-tick energy drain coefficient pro sensor specialization.
/// `drain = sum(sensor_gains) × this × dt`. Default sum = 3 × 1.0 = 3 →
/// `3 × 0.3 = 0.9/sec` baseline drain (porovnatelný s body cost 0.5/sec).
/// Cells, které sníží gains v category (turn off duplicate sensors v cluster),
/// šetří proportionally — sensor specialization je net-positive.
pub const SENSOR_GAIN_COST: f32 = 0.3;
/// Sprint 97: range pro `sensor_gains` per category. 0 = sensor effectively
/// off, 1 = neutral, 2 = boosted (lepší detection range, vyšší cost).
pub const MIN_SENSOR_GAIN: f32 = 0.0;
pub const MAX_SENSOR_GAIN: f32 = 2.0;
/// Sprint 97: 3 sensor categories indexované do `Genome.sensor_gains`:
/// 0 = Vision (food delta, cell delta, rel_size, density),
/// 1 = Chemistry (smell, pheromone — chemical gradients ve fields),
/// 2 = Defensive (damage signal, thermal_local).
/// Proprio sensors (energy, speed, heading) jsou always-on, žádný gain
/// — vlastní stav cell musí znát i v deep specialist módě.
pub const SENSOR_CATEGORY_VISION: usize = 0;
pub const SENSOR_CATEGORY_CHEMISTRY: usize = 1;
pub const SENSOR_CATEGORY_DEFENSIVE: usize = 2;
pub const N_SENSOR_CATEGORIES: usize = 3;

/// Sprint 97: maps brain input slot index → sensor category. Returns `None`
/// pro proprio slots (not gained, not pooled). Used by `apply_sensor_gains`
/// (per-cell gain multiply) + `pool_bonded_sensors` (max-pool environmental
/// slots přes bond network).
#[inline]
pub fn sensor_slot_category(slot: usize) -> Option<usize> {
    match slot {
        // Food delta (slot 0,1,15) + cell delta (2,3,16) + rel_size (6) +
        // density (13) → Vision
        0 | 1 | 2 | 3 | 6 | 13 | 15 | 16 => Some(SENSOR_CATEGORY_VISION),
        // Smell (7,8,17) + pheromone (11,12,19) → Chemistry
        7 | 8 | 11 | 12 | 17 | 19 => Some(SENSOR_CATEGORY_CHEMISTRY),
        // Damage (14) + thermal (20) → Defensive
        14 | 20 => Some(SENSOR_CATEGORY_DEFENSIVE),
        // Energy (4), speed (5), heading (9,10,18) → proprio, no gain
        _ => None,
    }
}
/// Sprint 84: hunter směrový FOV half-angle (rad). π/3 = 60° → 120° cone.
/// Predátoři klasicky mají frontal eyes; cells mohou flank-uniknout do blind
/// spotu. Pevná konstanta (Hunter nemá genom), ne pod selekcí.
pub const HUNTER_VISION_FOV: f32 = core::f32::consts::PI / 3.0;
/// Sprint 84: minimum speed² pro aktivní směrový vision. Pod threshold má
/// hunter velocity ~ 0 → není definovaný forward → fallback na omnidirectional
/// (idle hunter „spins around" hledá target). Threshold je sub-tick noise
/// (1 unit/s² ≪ HUNTER_IDLE_DRIFT² = 900); reálně cone aktivní vždy v lovu
/// nebo i drift, jen ne při startovním idle 0-vector.
pub const HUNTER_FORWARD_SPEED_THRESHOLD_SQ: f32 = 1.0;

// ─── Sprint 89: Hunter evolution v1 (parametric genome) ──────────────────────
// Pre-Sprint-89 byl Hunter non-evolving — fixed const (HUNTER_VISION_RADIUS,
// HUNTER_MAX_SPEED, HUNTER_DAMAGE_PER_TICK, …) řídily chování. Cells evolvovaly
// proti hunteru, hunter nikdy zpět → asymmetric selection. Sprint 89 zavádí
// `HunterGenome` + energy/reprodukce/smrt → biological arms race: hunter genes
// drift dle predator success rate, cells continue evolving evasion.
//
// V1 = no brain. Behavior zůstává „seek nearest attackable" (S84), ale
// parametry téhle behavior jsou per-hunter genové. Sprint 90 přidá brain.

/// Gene ranges. Konstanty z S71-S84 (`HUNTER_VISION_RADIUS=200`, …) zůstávají
/// jako defaults v `HunterGenome::random` initial draw range middle.
pub const MIN_HUNTER_VISION_RADIUS: f32 = 50.0;
pub const MAX_HUNTER_VISION_RADIUS: f32 = 400.0;
pub const MIN_HUNTER_VISION_FOV: f32 = core::f32::consts::PI / 12.0;
pub const MAX_HUNTER_VISION_FOV: f32 = core::f32::consts::PI;
pub const MIN_HUNTER_MAX_SPEED: f32 = 100.0;
pub const MAX_HUNTER_MAX_SPEED: f32 = 500.0;
pub const MIN_HUNTER_ACC: f32 = 40.0;
pub const MAX_HUNTER_ACC: f32 = 160.0;
pub const MIN_HUNTER_ATTACK_RADIUS: f32 = 10.0;
pub const MAX_HUNTER_ATTACK_RADIUS: f32 = 40.0;
pub const MIN_HUNTER_DAMAGE: f32 = 2.0;
pub const MAX_HUNTER_DAMAGE: f32 = 16.0;
pub const MIN_HUNTER_BODY_SIZE: f32 = 0.5;
pub const MAX_HUNTER_BODY_SIZE: f32 = 2.5;

/// Initial energy při Hunter spawn / floor respawn. Vyšší než cell
/// `INITIAL_ENERGY=100` — hunter potřebuje delší survival window než single
/// chase cycle, který může trvat víc generation ticks bez kill.
pub const HUNTER_INITIAL_ENERGY: f32 = 500.0;
/// Energy threshold pro reprodukci. Při dosažení parent splituje energy 50/50
/// se child + clone-with-mutate genome.
///
/// Sprint 98 tune 1: 700 → 500 (= HUNTER_INITIAL_ENERGY). Sex vyžaduje
/// dva fertile hunters současně v MATING_RADIUS — vzácná událost při
/// max_pop 50. Při 700 dal smoke 70gen 0 births → pop crash; 600 dal
/// 3 births za 300gen, taky kolaps. 500 = hunter je fertile od spawnu
/// (cooldown 0), gate je pak jen prostorová proximity + cooldown po
/// coupling. Re-fertility čeká jen na restore split-energy 250 → 500
/// (~3-5 gen of hunting), což je rychlejší než dřívější 250 → 700.
pub const HUNTER_REPRODUCE_THRESHOLD: f32 = 500.0;
/// Cap pro hunter populace. Bez něj by predator boom (mnoho cells eaten)
/// → exponenciální růst → prey extinction. 50 = 4× initial S71 count, dostatek
/// pro arms race signal ale prevent runaway.
pub const HUNTER_MAX_POP: usize = 50;
/// Per-tick vision drain coefficient. `vision_radius × fov_factor × VISION_COST × dt`.
/// Mírně vyšší než cell `VISION_COST_PER_RADIUS=0.02` — hunter má větší vision
/// a musí investovat víc energie do detection.
pub const HUNTER_VISION_COST: f32 = 0.01;
/// Per-tick motion drain. `v² × MOTION_COST × dt`. Hunter má rychlejší pohyb
/// + větší masu (body_size 1-2 vs cell ~1) → vyšší kinetic cost než cell
/// `ENERGY_COST_PER_V_SQ=0.0008`.
pub const HUNTER_MOTION_COST: f32 = 0.0001;
/// Body maintenance per tick. `body_size³ × BODY_COST × dt`. Volume-scaled
/// (jako cell). Bigger predator = víc tissue to maintain.
pub const HUNTER_BODY_COST: f32 = 0.5;
/// Attack-mode upkeep, always-on. `damage_per_tick × ATTACK_UPKEEP × dt`.
/// Hunter „claws out" continuously — lze trade-off-it nižší damage = nižší
/// upkeep, vhodné pro low-energy survivors.
pub const HUNTER_ATTACK_UPKEEP: f32 = 0.02;
/// Energy gain per damage dealt (proportional). Sprint 89 v3 = 6.0.
/// Sprint 93: 6.0 → 12.0 — kompenzace S92 exposure scaling (cells s 1-3
/// bondy mají reduced damage = reduced gain). Pre-S93 smoke gen 20 ukázal
/// hunter pop kolaps k 1 (cumulative net negative energy). 12.0 × avg
/// exposure ~0.85 ≈ 10.2 effective gain ≈ 1.7× pre-S92 6.0 × 1.0 = 6.0,
/// kompenzace partial defense + příležitost pro carnivore-niche cells
/// dostat z hunter carrion.
pub const HUNTER_ENERGY_PER_DAMAGE: f32 = 12.0;
/// Carrion drops při hunter death. Mirror cell death (Sprint 27 carrion).
/// 2× value default — hunter větší než cell, víc biomasy.
pub const HUNTER_CARRION_DROP: usize = 2;
/// Reproduce cooldown (ticks) po split. Brání instant re-reproduce před cell
/// catch-up. ~1 generation = 600 ticks.
pub const HUNTER_REPRODUCE_COOLDOWN_TICKS: u32 = 300;
/// Sprint 101: pack hunting payoff. Když bonded hunter zabije cell, každý
/// jeho bonded partner dostane `gain × FRAC` extra energy (free reward, no
/// energy conservation — modeluje "pack feed dynamic"). Mirror cells
/// `BOND_FOOD_SHARE_FRAC` semantiky. 0.5 dává pack-of-6 ~3.5× total payoff
/// vs solo (1 + 5×0.5 = 3.5), dostatek aby selekce favorizovala pack vs solo.
pub const HUNTER_BOND_KILL_SHARE_FRAC: f32 = 0.5;

/// Sprint 98: maximální vzdálenost dvou fertile hunterů, aby se spárovali.
/// Cells mají MATING_RADIUS = 200; hunteři jsou mnohem řidší (max 50 vs
/// max 2500 cells) → density je řádově nižší, takže menší mating radius
/// znamená, že se rodiče nepotkají. Density math: 12 hunterů v 207M unit³
/// dává mean nearest-neighbor ≈ 160 — pod 200 by žádný pár nepřekonal
/// gate, hunter populace by sjela floor respawn loop. 200 = parita s
/// HUNTER_VISION_RADIUS, biologicky „vidí na partnera".
pub const HUNTER_MATING_RADIUS: f32 = 200.0;
/// Sprint 90: brain output[0] turn_yaw multiplier (rad/sec). Hunter má
/// pevnou turn_rate (ne gene); cells mají gene-encoded turn_rate ∈ [1, 5].
/// 3.0 je mid-cell-range. Sprint 91+ může přidat jako gene.
pub const HUNTER_TURN_RATE: f32 = 3.0;
/// Sprint 90: brain output[7] turn_pitch multiplier (rad/sec). Pitch je
/// klampovaný v Cell::integrate_kinematics; pro hunter pitch_velocity je
/// volnější (no clamp), takže nižší rate brání overshoot.
pub const HUNTER_PITCH_RATE: f32 = 1.0;

/// Sprint 89: per-hunter heritable parametry. Pre-Sprint-89 byly fixed
/// const (HUNTER_VISION_RADIUS=200, HUNTER_MAX_SPEED=300, …); now drift
/// per generation. Sprint 90: + `brain` (reuse cell Brain struct, hunter-
/// specific input/output semantic mapping v `populate_hunter_brain_inputs`
/// + `apply_brain_motor`). Adaptive chase behavior emerges from selection
/// — random brains s positive INNATE_THRUST_BIAS startují s forward motion,
/// úspěšní lovci reprodukují svůj brain.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HunterGenome {
    pub vision_radius: f32,
    pub vision_fov: f32,
    pub max_speed: f32,
    pub acceleration: f32,
    pub attack_radius: f32,
    pub damage_per_tick: f32,
    pub body_size: f32,
    pub color_hue: f32,
    /// Sprint 99: cadherin-like recognition pro hunter-hunter bondy. 8 typů
    /// (parita s cells). Same-type hunteři se přitahují adhesion forcou +
    /// formují persistent bondy při kontaktu. Cross-type repulze. Žádný
    /// hunter-cell bond — adhesion pool je per-druhový (cell vs hunter
    /// bondy oddělené).
    pub adhesion_type: u8,
    /// Sprint 90: behavioral controller. Reuse cell `Brain` struct (BRAIN_INPUTS
    /// = 53, BRAIN_HIDDEN = 32, BRAIN_OUTPUTS = 10) — slot semantics
    /// re-mapped pro hunter v `populate_hunter_brain_inputs` (slot 0/1/15
    /// = nearest_prey delta, slot 7-8/17 = smell, slot 9-10/18 = heading,
    /// slot 4 = energy, slot 5 = speed, slot 6 = prey_size_relative,
    /// slot 13 = density). Used outputs: 0 (turn), 1 (thrust), 7 (pitch).
    /// Cell-only outputs (morph, attack, bond) ignored by hunter motor.
    pub brain: Brain,
}

impl HunterGenome {
    /// Initial random draw. Center middle ranges around S71-S84 const defaults
    /// + ~30 % spread → initial population diversity dostatečná pro selekci
    /// signal v 30-100 gen smoke.
    pub fn random(rng: &mut impl Rng) -> Self {
        Self {
            vision_radius: rng.random_range(100.0..300.0),
            vision_fov: rng.random_range(
                core::f32::consts::PI / 6.0..core::f32::consts::PI * 0.75,
            ),
            max_speed: rng.random_range(200.0..400.0),
            acceleration: rng.random_range(60.0..120.0),
            attack_radius: rng.random_range(12.0..28.0),
            damage_per_tick: rng.random_range(4.0..12.0),
            body_size: rng.random_range(0.8..1.6),
            color_hue: rng.random_range(0.0..HUE_RANGE),
            adhesion_type: rng.random_range(0..ADHESION_TYPE_COUNT),
            // Sprint 90: brain init s INNATE_THRUST_BIAS = 2.0 (z Brain::random).
            // Random brains startují s positive thrust → forward motion. Selekce
            // tuneuje turn/pitch outputs k ko-ordinovanému chase behavior.
            brain: Brain::random(rng),
        }
    }

    pub fn mutate(&self, rng: &mut impl Rng, cfg: &HunterMutationConfig) -> Self {
        Self {
            vision_radius: (self.vision_radius + gaussian(rng) * cfg.sigma_vision_radius)
                .clamp(MIN_HUNTER_VISION_RADIUS, MAX_HUNTER_VISION_RADIUS),
            vision_fov: (self.vision_fov + gaussian(rng) * cfg.sigma_vision_fov)
                .clamp(MIN_HUNTER_VISION_FOV, MAX_HUNTER_VISION_FOV),
            max_speed: (self.max_speed + gaussian(rng) * cfg.sigma_max_speed)
                .clamp(MIN_HUNTER_MAX_SPEED, MAX_HUNTER_MAX_SPEED),
            acceleration: (self.acceleration + gaussian(rng) * cfg.sigma_acceleration)
                .clamp(MIN_HUNTER_ACC, MAX_HUNTER_ACC),
            attack_radius: (self.attack_radius + gaussian(rng) * cfg.sigma_attack_radius)
                .clamp(MIN_HUNTER_ATTACK_RADIUS, MAX_HUNTER_ATTACK_RADIUS),
            damage_per_tick: (self.damage_per_tick + gaussian(rng) * cfg.sigma_damage)
                .clamp(MIN_HUNTER_DAMAGE, MAX_HUNTER_DAMAGE),
            body_size: (self.body_size + gaussian(rng) * cfg.sigma_body_size)
                .clamp(MIN_HUNTER_BODY_SIZE, MAX_HUNTER_BODY_SIZE),
            color_hue: (self.color_hue + gaussian(rng) * cfg.sigma_color_hue)
                .rem_euclid(HUE_RANGE),
            // Sprint 99: occasional flip pro selekci na cluster-friendly types.
            adhesion_type: if cfg.adhesion_flip_rate > 0.0
                && ADHESION_TYPE_COUNT > 1
                && rng.random::<f32>() < cfg.adhesion_flip_rate
            {
                let mut t = rng.random_range(0..ADHESION_TYPE_COUNT - 1);
                if t >= self.adhesion_type {
                    t += 1;
                }
                t
            } else {
                self.adhesion_type
            },
            brain: self.brain.mutate(rng, cfg.sigma_brain),
        }
    }

    pub fn crossover(a: &HunterGenome, b: &HunterGenome, rng: &mut impl Rng) -> Self {
        Self {
            vision_radius: if rng.random::<bool>() { a.vision_radius } else { b.vision_radius },
            vision_fov: if rng.random::<bool>() { a.vision_fov } else { b.vision_fov },
            max_speed: if rng.random::<bool>() { a.max_speed } else { b.max_speed },
            acceleration: if rng.random::<bool>() { a.acceleration } else { b.acceleration },
            attack_radius: if rng.random::<bool>() { a.attack_radius } else { b.attack_radius },
            damage_per_tick: if rng.random::<bool>() {
                a.damage_per_tick
            } else {
                b.damage_per_tick
            },
            body_size: if rng.random::<bool>() { a.body_size } else { b.body_size },
            color_hue: if rng.random::<bool>() { a.color_hue } else { b.color_hue },
            adhesion_type: if rng.random::<bool>() { a.adhesion_type } else { b.adhesion_type },
            brain: Brain::crossover(&a.brain, &b.brain, rng),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HunterMutationConfig {
    pub sigma_vision_radius: f32,
    pub sigma_vision_fov: f32,
    pub sigma_max_speed: f32,
    pub sigma_acceleration: f32,
    pub sigma_attack_radius: f32,
    pub sigma_damage: f32,
    pub sigma_body_size: f32,
    pub sigma_color_hue: f32,
    /// Sprint 90: brain weights gaussian sigma. Same magnitude jako cell
    /// `sigma_brain = 0.2` — brain landscape je velký, drift je naturally
    /// slow přes 2058 weights/cell.
    pub sigma_brain: f32,
    /// Sprint 99: per-child probability flipu adhesion_type (na jiný typ
    /// uniformně). Mirror cell `ADHESION_MUTATION_RATE`.
    pub adhesion_flip_rate: f32,
}

/// Sprint 89: hunter mutation rates. Vyšší než cell `MUTATION_CONFIG` (sigma
/// 1-3 % range/gen) — hunter populace menší (12-50), evolution signal slabší
/// per fewer offspring. ~3-4 % range/gen aby drift byl viditelný v 100-gen
/// smoke.
pub const HUNTER_MUTATION_CONFIG: HunterMutationConfig = HunterMutationConfig {
    sigma_vision_radius: 10.0,    // 2.9 % of [50, 400] range
    sigma_vision_fov: 0.08,       // 2.8 % of FOV range
    sigma_max_speed: 12.0,        // 3.0 % of [100, 500]
    sigma_acceleration: 4.0,      // 3.3 % of [40, 160]
    sigma_attack_radius: 1.0,     // 3.3 % of [10, 40]
    sigma_damage: 0.4,            // 2.9 % of [2, 16]
    sigma_body_size: 0.06,        // 3.0 % of [0.5, 2.5]
    sigma_color_hue: 5.0,         // 1.4 % HUE_RANGE — slow drift, lineage tracking
    sigma_brain: 0.2,             // Sprint 90 — match cell sigma_brain
    adhesion_flip_rate: ADHESION_MUTATION_RATE, // Sprint 99 — parita s cells
};

/// Sprint 71: non-evolving environmental predator (Sprint 89 → evolving).
/// Pohybuje se pseudo-AI (seek nejbližší cell ∈ vision range, jinak random
/// drift). Atakuje cells s `n_bonds() < HUNTER_BOND_IMMUNITY_THRESHOLD` v
/// attack range.
///
/// Sprint 89: + `genome` (8 heritable parameters), + `energy` (lifecycle),
/// + lineage tracking.
/// Sprint 90: + brain-driven motion. Heading + pitch (mirror Cell), brain
/// state (last_inputs/hidden/outputs). Random brain s INNATE_THRUST_BIAS
/// startuje s forward motion; turn/pitch outputs evolve k chase tactics.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Hunter {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    /// Stable identifier (procedural, monotonic při init). Nepoužívá se pro
    /// bond resolution — Hunter ≠ Cell.
    pub hunter_id: u64,
    pub genome: HunterGenome,
    pub energy: f32,
    pub age: u64,
    pub reproduce_cooldown_ticks: u32,
    pub lineage_id: u64,
    pub lineage_birth_gen: u64,
    /// Sprint 90: yaw heading (rad). Brain output[0] modifikuje
    /// `angular_velocity`, ten integrate v `step` jako Cell.
    pub heading: f32,
    /// Sprint 90: pitch (rad), unbounded — hunter vertical motion volnější
    /// než cell (Cell má clamp na ±π/12).
    pub pitch: f32,
    pub angular_velocity: f32,
    pub pitch_velocity: f32,
    /// Sprint 90: brain I/O state pro recurrent kanál + diagnostics.
    #[serde(with = "serde_arr_inputs")]
    pub last_inputs: [f32; BRAIN_INPUTS],
    #[serde(with = "serde_arr_hidden")]
    pub last_hidden: [f32; BRAIN_HIDDEN],
    pub last_outputs: [f32; BRAIN_OUTPUTS],
    /// Sprint 99: persistent spring bondy mezi hunters (mirror Cell.bonds).
    /// `Bond.other_cell_id` here stores the other hunter's `hunter_id`
    /// (sémanticky `other_hunter_id`; field reused k zachování single Bond
    /// struct). Same-type adhesion gating + spring physics. Cluster
    /// představuje "wolf pack" — koordinovaná predace přijde v S101.
    pub bonds: [Option<Bond>; MAX_BONDS_PER_CELL],
    /// Sprint 100: pack-level pooled hidden (mirror Cell.pooled_hidden z S94).
    /// Mean(self.last_hidden + bonded partners.last_hidden) per tick.
    /// Solo hunteři: kopie self.last_hidden (nemění chování).
    /// Bonded hunteři: shared recurrent state napříč packem → proto-distributed
    /// cognition pro koordinovaný hon.
    #[serde(default = "default_pooled_hidden", with = "serde_arr_hidden")]
    pub pooled_hidden: [f32; BRAIN_HIDDEN],
}

impl Hunter {
    /// Random init: random position + zero velocity + random genome.
    pub fn random(
        rng: &mut impl Rng,
        world_half: [f32; 3],
        hunter_id: u64,
        lineage_id: u64,
        lineage_birth_gen: u64,
    ) -> Self {
        let genome = HunterGenome::random(rng);
        Self::from_genome(rng, genome, world_half, hunter_id, lineage_id, lineage_birth_gen)
    }

    /// Sprint 89: spawn s explicit genome (used by clone-with-mutate v reprodukci
    /// a floor respawn). Position random, velocity zero, energy = INITIAL.
    /// Sprint 90: + heading random (TAU range), pitch 0, brain state zero.
    pub fn from_genome(
        rng: &mut impl Rng,
        genome: HunterGenome,
        world_half: [f32; 3],
        hunter_id: u64,
        lineage_id: u64,
        lineage_birth_gen: u64,
    ) -> Self {
        let pos_z = if world_half[2] > 0.0 {
            rng.random_range(-world_half[2]..world_half[2])
        } else {
            0.0
        };
        let direction = rng.random_range(0.0..TAU);
        Self {
            position: [
                rng.random_range(-world_half[0]..world_half[0]),
                rng.random_range(-world_half[1]..world_half[1]),
                pos_z,
            ],
            velocity: [
                direction.cos() * genome.max_speed * 0.3,
                direction.sin() * genome.max_speed * 0.3,
                0.0,
            ],
            hunter_id,
            genome,
            energy: HUNTER_INITIAL_ENERGY,
            age: 0,
            reproduce_cooldown_ticks: 0,
            lineage_id,
            lineage_birth_gen,
            heading: direction,
            pitch: 0.0,
            angular_velocity: 0.0,
            pitch_velocity: 0.0,
            last_inputs: [0.0; BRAIN_INPUTS],
            last_hidden: [0.0; BRAIN_HIDDEN],
            last_outputs: [0.0; BRAIN_OUTPUTS],
            // Sprint 99: bondy se formují contact-based v hunter physics phase.
            bonds: [None; MAX_BONDS_PER_CELL],
            // Sprint 100: pooled hidden init = zero, naplní se v
            // `pool_bonded_hunter_hidden` před run_brain_act.
            pooled_hidden: [0.0; BRAIN_HIDDEN],
        }
    }

    /// Sprint 89: per-tick energy drains. Vision (∝ radius × fov_factor),
    /// motion (∝ v²), body maintenance (∝ size³), attack upkeep (∝ damage).
    /// Bez aging ramp (Sprint 42 cells aging) — hunter lifecycle krátký.
    pub fn apply_energy_costs(&mut self, dt: f32) {
        let fov_factor = vision_fov_factor(self.genome.vision_fov);
        self.energy -= self.genome.vision_radius * HUNTER_VISION_COST * fov_factor * dt;
        let v_mag_sq = self.velocity[0] * self.velocity[0]
            + self.velocity[1] * self.velocity[1]
            + self.velocity[2] * self.velocity[2];
        self.energy -= v_mag_sq * HUNTER_MOTION_COST * dt;
        let s = self.genome.body_size;
        self.energy -= s * s * s * HUNTER_BODY_COST * dt;
        self.energy -= self.genome.damage_per_tick * HUNTER_ATTACK_UPKEEP * dt;
    }

    /// Sprint 90: brain-driven motion s **hybrid seek bootstrap**. Brain
    /// outputs[0] = turn_yaw modulator, [1] = thrust, [7] = turn_pitch
    /// modulator. Cell-only outputs (morph, attack signal, bond) ignored.
    ///
    /// Hybrid design pattern: seek-toward-prey direction (deterministic
    /// oracle) je mixed s brain output (`HUNTER_BRAIN_SEEK_MIX = 0.6` seek
    /// + 0.4 brain). Bez tohoto random initial brain neumí chase (random
    /// turn output → spinning), populace kolabuje do floor respawn loop.
    /// S hybridem brain modul dominantní seek direction (např. learned
    /// ambush, prey selection, retreat při low energy). Když brain weights
    /// evolvují k matching seek, mix se stane redundant; když brain learnuje
    /// jiný strategy (cluster around hot zones, etc.), brain dominuje.
    pub fn apply_brain_motor(
        &mut self,
        outputs: &[f32; BRAIN_OUTPUTS],
        seek_target: Option<[f32; 3]>,
        dt: f32,
        world_half: [f32; 3],
    ) {
        let brain_turn = outputs[0].clamp(-1.0, 1.0);
        let brain_pitch = outputs[7].clamp(-1.0, 1.0);
        let thrust = outputs[1].clamp(-1.0, 1.0);
        // Compute seek-based turn modulator.
        let (seek_turn, seek_pitch) = match seek_target {
            Some(t) => {
                let d = min_image_delta(self.position, t, world_half);
                let desired_yaw = d[1].atan2(d[0]);
                // Shortest angular distance → [-π, π].
                let mut yaw_diff = desired_yaw - self.heading;
                while yaw_diff > core::f32::consts::PI {
                    yaw_diff -= TAU;
                }
                while yaw_diff < -core::f32::consts::PI {
                    yaw_diff += TAU;
                }
                let dist_xy = (d[0] * d[0] + d[1] * d[1]).sqrt();
                let desired_pitch = if dist_xy > 1e-3 {
                    d[2].atan2(dist_xy)
                } else {
                    0.0
                };
                let pitch_diff = desired_pitch - self.pitch;
                // Normalize na [-1, 1] motor space.
                (
                    (yaw_diff / core::f32::consts::PI).clamp(-1.0, 1.0),
                    (pitch_diff / core::f32::consts::PI).clamp(-1.0, 1.0),
                )
            }
            None => (0.0, 0.0),
        };
        let seek_mix = 0.6;
        let turn = (brain_turn * (1.0 - seek_mix) + seek_turn * seek_mix).clamp(-1.0, 1.0);
        let pitch_t =
            (brain_pitch * (1.0 - seek_mix) + seek_pitch * seek_mix).clamp(-1.0, 1.0);
        self.angular_velocity = turn * HUNTER_TURN_RATE;
        self.pitch_velocity = pitch_t * HUNTER_PITCH_RATE;
        let fwd = forward_vector(self.heading, self.pitch);
        let acc = thrust * self.genome.acceleration;
        self.velocity[0] += fwd[0] * acc * dt;
        self.velocity[1] += fwd[1] * acc * dt;
        self.velocity[2] += fwd[2] * acc * dt;
        let speed_sq = self.velocity[0] * self.velocity[0]
            + self.velocity[1] * self.velocity[1]
            + self.velocity[2] * self.velocity[2];
        let max_sq = self.genome.max_speed * self.genome.max_speed;
        if speed_sq > max_sq {
            let scale = self.genome.max_speed / speed_sq.sqrt();
            self.velocity[0] *= scale;
            self.velocity[1] *= scale;
            self.velocity[2] *= scale;
        }
    }

    /// Sprint 90: kinematic integration + heading update + toroidal wrap.
    /// Pre-Sprint-90 step měl seek-target logic (Sprint 89); ten se přesunul
    /// do brain (caller volá `apply_brain_motor` před step). Step je teď
    /// čistě passive — integrate position + heading + pitch.
    pub fn step(&mut self, dt: f32, world_half: [f32; 3]) {
        self.age = self.age.saturating_add(1);
        if self.reproduce_cooldown_ticks > 0 {
            self.reproduce_cooldown_ticks -= 1;
        }
        // Integrate.
        self.position[0] += self.velocity[0] * dt;
        self.position[1] += self.velocity[1] * dt;
        self.position[2] += self.velocity[2] * dt;
        self.heading += self.angular_velocity * dt;
        self.pitch += self.pitch_velocity * dt;
        // Toroidal wrap xy, z bounce.
        let wx = 2.0 * world_half[0];
        let wy = 2.0 * world_half[1];
        if self.position[0] >= world_half[0] || self.position[0] < -world_half[0] {
            let p = self.position[0] + world_half[0];
            self.position[0] = p - (p / wx).floor() * wx - world_half[0];
        }
        if self.position[1] >= world_half[1] || self.position[1] < -world_half[1] {
            let p = self.position[1] + world_half[1];
            self.position[1] = p - (p / wy).floor() * wy - world_half[1];
        }
        if world_half[2] > 0.0 && self.position[2].abs() > world_half[2] {
            self.velocity[2] = -self.velocity[2];
            self.position[2] = self.position[2].clamp(-world_half[2], world_half[2]);
        }
    }
}

/// Sprint 90: hunter sensor context (subset of cell `BrainSensors` adapted
/// pro predator semantics). „Prey" = nearest attackable cell (genome
/// vision_radius + fov filter + n_bonds < HUNTER_BOND_IMMUNITY_THRESHOLD).
#[derive(Debug, Clone, Copy)]
pub struct HunterBrainSensors {
    /// Min-image delta od hunter k nearest prey (cell.position − hunter.position).
    pub nearest_prey: Option<[f32; 3]>,
    /// Body size nearest prey (z `cell.phenotype.effective_radius`). Pre brain:
    /// "kořist je menší / větší než já" → trade-off chase tactics.
    pub nearest_prey_size: f32,
    /// Počet attackable cells uvnitř vision range/cone (density signal).
    pub neighbors_in_vision: u32,
    /// Smell field gradient na hunter pozici. Cells emit smell when eating
    /// food → chemical clue pro nearby cell activity.
    pub smell_grad: [f32; 3],
    /// Sprint 100: delta k nearest same-type hunteru ve vision (= pack member
    /// candidate / kontakt k bondu). Brain získá schopnost aktivně hledat
    /// nebo se vyhýbat packu.
    pub nearest_pack_member: Option<[f32; 3]>,
    /// Sprint 100: počet same-type hunters ve vision (pack density signal).
    pub same_type_in_vision: u32,
}

/// Minimální snapshot huntera pro pack-sense scan v `gather_hunter_sensors`.
/// Drží jen pozici, hunter_id a adhesion_type — to jediné, co same-type
/// vision check potřebuje. Nahrazuje per-tick deep clone celého `Hunter`
/// (genome + brain weights + bondy + recurrent state) na ~24-byte Copy.
#[derive(Debug, Clone, Copy)]
pub struct HunterSnapshotMin {
    pub hunter_id: u64,
    pub position: [f32; 3],
    pub adhesion_type: u8,
}

impl HunterSnapshotMin {
    pub fn from_hunter(h: &Hunter) -> Self {
        Self {
            hunter_id: h.hunter_id,
            position: h.position,
            adhesion_type: h.genome.adhesion_type,
        }
    }
}

/// Sprint 102: hunter sensor gather na spatial gridu. `cell_grid` musí být
/// rebuilded přes `cells.iter().enumerate().map(|(i, c)| (i, c.position, ()))`
/// caller-side — funkce iteruje jen 3³ buckets v okolí huntera, narrow-phase
/// distance + cone test. `other_hunters` zůstává brute force (n ≤ ~50 — H²
/// je zanedbatelný), ale typ je `HunterSnapshotMin` místo `&[Hunter]` ⇒
/// caller ušetří deep clone.
pub fn gather_hunter_sensors(
    hunter: &Hunter,
    cells: &[Cell],
    cell_grid: &SpatialGrid<usize, ()>,
    other_hunters: &[HunterSnapshotMin],
    smell: &SmellField,
    world_half: [f32; 3],
) -> HunterBrainSensors {
    let vision_r = hunter.genome.vision_radius;
    let vision_r2 = vision_r * vision_r;
    let cos_fov = hunter.genome.vision_fov.cos();
    let speed_sq = hunter.velocity[0] * hunter.velocity[0]
        + hunter.velocity[1] * hunter.velocity[1]
        + hunter.velocity[2] * hunter.velocity[2];
    let cone_active = speed_sq > HUNTER_FORWARD_SPEED_THRESHOLD_SQ;
    let forward = if cone_active {
        let inv = 1.0 / speed_sq.sqrt();
        [
            hunter.velocity[0] * inv,
            hunter.velocity[1] * inv,
            hunter.velocity[2] * inv,
        ]
    } else {
        [0.0; 3]
    };
    let mut best: Option<([f32; 3], f32, f32)> = None; // (delta, d2, prey_size)
    let mut count: u32 = 0;
    cell_grid.for_each_in_radius_toroidal(
        hunter.position,
        vision_r,
        world_half,
        |idx, ghost_pos, ()| {
            let c = &cells[idx];
            if c.n_bonds() >= HUNTER_BOND_IMMUNITY_THRESHOLD {
                return;
            }
            let d = [
                ghost_pos[0] - hunter.position[0],
                ghost_pos[1] - hunter.position[1],
                ghost_pos[2] - hunter.position[2],
            ];
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if d2 >= vision_r2 {
                return;
            }
            if cone_active && !fov_cone_accept(d, d2, forward, cos_fov) {
                return;
            }
            count += 1;
            let prey_size = c.phenotype.effective_radius();
            match best {
                None => best = Some((d, d2, prey_size)),
                Some((_, bd2, _)) if d2 < bd2 => best = Some((d, d2, prey_size)),
                _ => {}
            }
        },
    );
    let smell_grad = smell.gradient_at(hunter.position, SMELL_SAMPLE_EPSILON);
    // Sprint 100: pack scan — same-type hunteři ve vision range. H² brute
    // force OK: HUNTER_TARGET_COUNT je low (~12), grid by tu nebyl výhra.
    let mut nearest_pack: Option<([f32; 3], f32)> = None;
    let mut same_type_count: u32 = 0;
    let own_type = hunter.genome.adhesion_type;
    let own_id = hunter.hunter_id;
    for o in other_hunters {
        if o.hunter_id == own_id {
            continue;
        }
        if o.adhesion_type != own_type {
            continue;
        }
        let d = min_image_delta(hunter.position, o.position, world_half);
        let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if d2 >= vision_r2 {
            continue;
        }
        same_type_count += 1;
        match nearest_pack {
            None => nearest_pack = Some((d, d2)),
            Some((_, bd2)) if d2 < bd2 => nearest_pack = Some((d, d2)),
            _ => {}
        }
    }
    HunterBrainSensors {
        nearest_prey: best.map(|(d, _, _)| d),
        nearest_prey_size: best.map(|(_, _, s)| s).unwrap_or(0.0),
        neighbors_in_vision: count,
        smell_grad,
        nearest_pack_member: nearest_pack.map(|(d, _)| d),
        same_type_in_vision: same_type_count,
    }
}

/// Sprint 90: brain input vector pro hunter. Reuse cell's 21-slot layout
/// s hunter semantics:
///   0,1,15: nearest_prey delta (= cell food slots)
///   2,3,16: nearest_pack_member delta (Sprint 100: same-type hunter pull)
///   4: own_energy / HUNTER_REPRODUCE_THRESHOLD
///   5: own_speed / max_speed
///   6: prey_size_relative (prey/own — pre brain "small target" awareness)
///   7,8,17: smell_grad x/y/z
///   9,10,18: heading_x/y/z (forward vector)
///   11: pack_size_norm (n_bonds / MAX_BONDS_PER_CELL — Sprint 100)
///   12: pack_density_norm (same_type_in_vision / DENSITY_NORM_COUNT — Sprint 100)
///   13: density_in_vision (count cells / DENSITY_NORM_COUNT)
///   14: filler (cell uses damage)
///   19, 20: filler
///   21..52: recurrent — pooled_hidden (Sprint 100, S94 mirror)
pub fn populate_hunter_brain_inputs(
    hunter: &mut Hunter,
    sensors: &HunterBrainSensors,
) -> [f32; BRAIN_INPUTS] {
    let vision_r = hunter.genome.vision_radius.max(0.01);
    let max_speed = hunter.genome.max_speed.max(1e-3);
    let speed_xy = hunter.velocity[0].hypot(hunter.velocity[1]);
    let speed_norm = (speed_xy / max_speed).clamp(0.0, 1.0);
    let energy_norm = (hunter.energy / HUNTER_REPRODUCE_THRESHOLD).clamp(0.0, 1.5);
    let mut inputs = [0.0_f32; BRAIN_INPUTS];
    if let Some(d) = sensors.nearest_prey {
        inputs[0] = d[0] / vision_r;
        inputs[1] = d[1] / vision_r;
        inputs[15] = d[2] / vision_r;
        let own_size = hunter.genome.body_size.max(0.01);
        inputs[6] = (sensors.nearest_prey_size - own_size) / own_size;
    }
    // Sprint 100: pack member delta — informuje brain o směru k pack mate.
    if let Some(d) = sensors.nearest_pack_member {
        inputs[2] = d[0] / vision_r;
        inputs[3] = d[1] / vision_r;
        inputs[16] = d[2] / vision_r;
    }
    inputs[4] = energy_norm;
    inputs[5] = speed_norm;
    inputs[7] = (sensors.smell_grad[0] * SMELL_NORMALIZATION_GAIN).tanh();
    inputs[8] = (sensors.smell_grad[1] * SMELL_NORMALIZATION_GAIN).tanh();
    inputs[17] = (sensors.smell_grad[2] * SMELL_NORMALIZATION_GAIN).tanh();
    let fwd = forward_vector(hunter.heading, hunter.pitch);
    inputs[9] = fwd[0];
    inputs[10] = fwd[1];
    inputs[18] = fwd[2];
    // Sprint 100: pack size / density signals.
    let n_bonds = hunter.bonds.iter().filter(|b| b.is_some()).count() as f32;
    inputs[11] = (n_bonds / MAX_BONDS_PER_CELL as f32).min(1.0);
    inputs[12] = (sensors.same_type_in_vision as f32 / DENSITY_NORM_COUNT).tanh();
    inputs[13] = (sensors.neighbors_in_vision as f32 / DENSITY_NORM_COUNT).tanh();
    // Sprint 100: pooled_hidden místo last_hidden — bonded hunteři sdílejí
    // recurrent state. Solo hunter má pooled_hidden = self.last_hidden
    // (gathered v `pool_bonded_hunter_hidden` před brain_act).
    inputs[BRAIN_INPUTS_SENSORY..BRAIN_INPUTS_SENSORY + BRAIN_RECURRENT]
        .copy_from_slice(&hunter.pooled_hidden[..BRAIN_RECURRENT]);
    inputs
}

/// Sprint 100: pool last_hidden napříč bonded hunter packem (mirror S94
/// `pool_bonded_hidden` for cells). Pre-brain_act fáze. Solo hunter dostane
/// kopii vlastního last_hidden — žádná change vs pre-S100 chování.
pub fn pool_bonded_hunter_hidden(hunters: &mut [Hunter]) {
    let n = hunters.len();
    if n == 0 {
        return;
    }
    let id_to_idx: rustc_hash::FxHashMap<u64, usize> = hunters
        .iter()
        .enumerate()
        .map(|(i, h)| (h.hunter_id, i))
        .collect();
    let snapshot: Vec<[f32; BRAIN_HIDDEN]> = hunters.iter().map(|h| h.last_hidden).collect();
    let bonds_snapshot: Vec<[Option<Bond>; MAX_BONDS_PER_CELL]> =
        hunters.iter().map(|h| h.bonds).collect();
    for i in 0..n {
        let mut sum = snapshot[i];
        let mut count = 1usize;
        for bond_opt in bonds_snapshot[i].iter() {
            if let Some(bond) = bond_opt {
                if let Some(&j) = id_to_idx.get(&bond.other_cell_id) {
                    for k in 0..BRAIN_HIDDEN {
                        sum[k] += snapshot[j][k];
                    }
                    count += 1;
                }
            }
        }
        if count > 1 {
            let inv = 1.0 / count as f32;
            for k in 0..BRAIN_HIDDEN {
                sum[k] *= inv;
            }
        }
        hunters[i].pooled_hidden = sum;
    }
}

/// Sprint 89: asexual clone-with-mutate. Parent splituje energy 50/50
/// se child, child dostává mutated genome + parent's lineage_id (lineage
/// continuation). Caller sets cooldown + alloc hunter_id.
///
/// Sprint 98: solo cesta zachovaná pro floor respawn, ale primární repro
/// path je teď sexuální (`make_hunter_mating_child`).
pub fn make_hunter_child(
    parent: &Hunter,
    rng: &mut impl Rng,
    world_half: [f32; 3],
    hunter_id: u64,
    current_gen: u64,
) -> Hunter {
    let child_genome = parent.genome.mutate(rng, &HUNTER_MUTATION_CONFIG);
    let mut child = Hunter::from_genome(
        rng,
        child_genome,
        world_half,
        hunter_id,
        parent.lineage_id,
        current_gen,
    );
    // Spawn at parent position (later step + idle drift rozhází).
    child.position = parent.position;
    child.energy = parent.energy * 0.5;
    child.reproduce_cooldown_ticks = HUNTER_REPRODUCE_COOLDOWN_TICKS;
    child
}

/// Sprint 98: sexuální reprodukce hunterů — symetrická k `make_mating_child`
/// pro buňky. Crossover obou rodičovských genomů (per-field 50/50 + brain
/// crossover), pak mutace. Mirror cell semantiky: lineage = parent_a (single-
/// parent inheritance), spawn position = midpoint, energy = a + b (caller
/// halves oba pre-call), cooldown nastaven na child.
///
/// RNG draw order: crossover (gen-by-gen + Brain::crossover) → mutate →
/// from_genome (3 position draws + 1 direction draw, hned overriden).
/// Změna pořadí by porušila CSV reproducibility napříč seedy.
pub fn make_hunter_mating_child(
    parent_a: &Hunter,
    parent_b: &Hunter,
    rng: &mut impl Rng,
    world_half: [f32; 3],
    hunter_id: u64,
    current_gen: u64,
) -> Hunter {
    let child_genome = HunterGenome::crossover(&parent_a.genome, &parent_b.genome, rng)
        .mutate(rng, &HUNTER_MUTATION_CONFIG);
    let mut child = Hunter::from_genome(
        rng,
        child_genome,
        world_half,
        hunter_id,
        parent_a.lineage_id,
        current_gen,
    );
    child.position = [
        (parent_a.position[0] + parent_b.position[0]) * 0.5,
        (parent_a.position[1] + parent_b.position[1]) * 0.5,
        (parent_a.position[2] + parent_b.position[2]) * 0.5,
    ];
    child.energy = parent_a.energy + parent_b.energy;
    child.reproduce_cooldown_ticks = HUNTER_REPRODUCE_COOLDOWN_TICKS;
    child
}

/// Sprint 71: vrací `Some(cell_index)` nejbližší attackable cell ∈ vision
/// range. „Attackable" = `n_bonds() < HUNTER_BOND_IMMUNITY_THRESHOLD`. Vrací
/// nejbližší **z attackable** cells, ne nejbližší absolutně — Hunter nepronásleduje
/// imune clustery (skill: cluster je viditelný, ale ne lovitelný; Hunter musí
/// najít solo cell). Toroidal-aware přes `min_image_delta`.
///
/// Sprint 84: směrový FOV. `hunter.velocity` určuje forward; cells mimo
/// `genome.vision_fov` kuželu nejsou viditelné. Idle hunter (velocity² <
/// `HUNTER_FORWARD_SPEED_THRESHOLD_SQ`) má fallback na omni — bez toho by
/// hunter zaseknutý v 0-velocity stavu nikdy nenašel target.
///
/// Sprint 89: vision_radius + vision_fov teď z `hunter.genome` místo const.
/// Heritable predator detection range/cone — selekce drift per generation.
pub fn nearest_attackable_cell(
    hunter: &Hunter,
    cells: &[Cell],
    cell_grid: &SpatialGrid<usize, ()>,
    world_half: [f32; 3],
) -> Option<usize> {
    let vision_r = hunter.genome.vision_radius;
    let vision_r2 = vision_r * vision_r;
    let cos_fov = hunter.genome.vision_fov.cos();
    let speed_sq = hunter.velocity[0] * hunter.velocity[0]
        + hunter.velocity[1] * hunter.velocity[1]
        + hunter.velocity[2] * hunter.velocity[2];
    let cone_active = speed_sq > HUNTER_FORWARD_SPEED_THRESHOLD_SQ;
    let forward = if cone_active {
        let inv = 1.0 / speed_sq.sqrt();
        [
            hunter.velocity[0] * inv,
            hunter.velocity[1] * inv,
            hunter.velocity[2] * inv,
        ]
    } else {
        [0.0; 3]
    };
    let mut best: Option<(usize, f32)> = None;
    cell_grid.for_each_in_radius_toroidal(
        hunter.position,
        vision_r,
        world_half,
        |idx, ghost_pos, ()| {
            let c = &cells[idx];
            if c.n_bonds() >= HUNTER_BOND_IMMUNITY_THRESHOLD {
                return;
            }
            // Sprint 84: vector from hunter to cell (= c.position − hunter.pos).
            // Pre-Sprint-84 byl `d` drženo jako hunter_pos − c.position (jen pro d²
            // distance, kde znaménko nehraje); cone filter potřebuje směr.
            // Sprint 102: ghost_pos je už toroidálně min-image pozice (grid wrapped).
            let d = [
                ghost_pos[0] - hunter.position[0],
                ghost_pos[1] - hunter.position[1],
                ghost_pos[2] - hunter.position[2],
            ];
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if d2 >= vision_r2 {
                return;
            }
            if cone_active && !fov_cone_accept(d, d2, forward, cos_fov) {
                return;
            }
            match best {
                None => best = Some((idx, d2)),
                Some((_, bd2)) if d2 < bd2 => best = Some((idx, d2)),
                _ => {}
            }
        },
    );
    best.map(|(i, _)| i)
}
