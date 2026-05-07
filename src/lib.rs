//! Bioscape simulation core — pure logic, no rendering.
//!
//! Keeping this layer free of Bevy types lets us drive the same world
//! from a windowed renderer (`main.rs`) or a headless batch run later.

use core::f32::consts::TAU;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::hash::Hash;

#[cfg(feature = "gpu")]
pub mod gpu;

pub mod params;
pub use params::*;

pub mod neural;
pub use neural::*;

pub mod genetics;
pub use genetics::*;

pub mod cell;
pub use cell::*;

/// Sprint 121: jeden spike v multi-spike struktuře. `length` ∈ [MIN, MAX]
/// (= dnešní spike_length), `azimuth_offset`/`elevation_offset` určují směr
/// v body frame relative k forward (0,0 = frontální spike, dnešní default).
/// `complexity` je continuous geometric shape parametr ∈ [0, 1] — Sprint 122
/// drží na 0, Sprint 123 zapne attack/eat/cost vazby.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct Spike {
    pub length: f32,
    pub azimuth_offset: f32,
    pub elevation_offset: f32,
    pub complexity: f32,
}

impl Spike {
    pub const ZERO: Spike = Spike {
        length: 0.0,
        azimuth_offset: 0.0,
        elevation_offset: 0.0,
        complexity: 0.0,
    };
}

// Sprint 41: shell jako passive damage absorber. `shell_thickness` gen
// modifikuje `damage_accum` před zápisem do brain inputu — `apply_shell_absorb`
// odečte `shell × ABSORB_PER_TICK × dt`, floor 0. Maintenance cost lineární
// (defensive armor je drahá). Otevírá defensive niku — silně shellovaná
// buňka neunikne predátorovi, ale unese hazard + occasional spike hits.
pub const MIN_SHELL_THICKNESS: f32 = 0.0;
pub const MAX_SHELL_THICKNESS: f32 = 1.5;
/// Per-second absorb capacity. Single `PREDATION_DRAIN_PER_TICK` = 3.0; shell=1.5
/// plně absorbuje (3.0 ≤ 1.5 × 2.0), shell=1.0 absorbuje 2/3 hit, shell=0.5 půlí.
pub const SHELL_ABSORB_PER_TICK: f32 = 2.0;
/// Maintenance cost ∝ shell. Vyšší než spike (0.3) protože shell pokrývá celý
/// povrch, ne point structure.
pub const SHELL_COST_PER_SEC: f32 = 0.4;

// Sprint 42 life-history: 5 biofyzikálních realismů. Smoke-tuned po A/B isolation.
/// Aging body cost ramp. Při age_sec=100 factor=1.1 (10 % extra). Conservative.
pub const AGE_DECAY_PER_SEC: f32 = 0.001;
/// Brownian thermal noise — robustness signál bez navigation disruption.
/// Per-tick stddev ≈ 0.04 (oproti max_speed ~90, šum < 0.05 %).
pub const THERMAL_NOISE: f32 = 0.3;
/// Refractory period po mating. Mírný — 10 ticks (1/6 sec) nepoznatelný v
/// normálním energy-cycle, ale chrání proti instant remating spam.
pub const MATING_COOLDOWN_TICKS: u32 = 10;
/// Food decay rate — universal aging (carrion + map-spawn). Smoke-tuned z 0.05
/// (extinkce gen 23 — food saturoval na low-pop, vzájemně decay) na 0.0005
/// (gentle aging, food prakticky vždy fresh, decay viditelný jen u dlouho-
/// nesnězeného carrion). Carrion-specific decay je cleaner ale Sprint 42
/// scope-cut na universal gentle.
pub const CARRION_DECAY_PER_SEC: f32 = 0.0005;

// Sprint 66: differential adhesion (Steinberg) + persistent spring bonds.
// Steinbergova differential-adhesion hypothesis: buňky stejného CAM/cadherin
// typu drží spolu silněji než hetero-pair. V simu je to 2-úrovňové:
//   1. Stateless soft attraction (běží každý tick v broad-phase) — výsledek
//      je tekutý agregát, fluid sorting podle adhesion_type.
//   2. Stateful spring bonds (persistent) — vznikají až po prolonged contact
//      + brain output[9] consent na obou stranách. Drží pevný tvar tkáně.
/// Počet diskrétních adhesion typů (cadherin-like CAM tokens). 8 = pohodlných
/// 3 bity, dost pro emergentní niches bez sparse-population per type problému.
pub const ADHESION_TYPE_COUNT: u8 = 8;
/// Pravděpodobnost mutace adhesion_type per dítě. Při flipu se vybere uniformně
/// jiný typ (∈ 0..ADHESION_TYPE_COUNT, ≠ původní). Pomalá rychlost — adhesion
/// niche je strategicky stabilní, příliš časté flipy by rozbily clusters
/// dřív než se prosadí selekce. 5 % dává průměrně 1 změnu za 20 generací.
pub const ADHESION_MUTATION_RATE: f32 = 0.05;
/// Dosah soft-attraction síly v násobcích pair-radius (CELL_RADIUS × (r_i + r_j)).
/// 3× kontaktní vzdálenost = mírná lokální atrakce, neaplikuje se přes celé
/// vision_radius. Zkrátit by zúžilo aglomeraci na overlap-only; rozšířit by
/// vytvářelo dálkový shoaling efekt přes Steinbergův framework.
pub const ADHESION_RANGE_FACTOR: f32 = 3.0;
/// Peak attractive acceleration (pre-mass), aplikuje se jako Δv per tick.
/// Same-type pair → cells se sbližují. Linearly škáluje (1 - d/R). Tuned tak,
/// aby per-tick magnitude ~= drag×v_typ při d=0.5R, takže adhesion balancuje
/// Brownian noise + drift, ne aby cells "schluckly" do single point.
pub const ADHESION_STRENGTH: f32 = 8.0;
/// Cross-type interaction. Steinbergův klasik: hetero-pair je *méně* atraktivní
/// (negative = mírně repulsivní), což emergentně oddělí typy do clusterů
/// podobných k tissue-segregation. -0.3 = 1/4 of same-type magnitude.
pub const ADHESION_CROSS_TYPE: f32 = -0.3;
/// Maximum spring bondů per cell. Sphere packing kissing 12, ale soft cells
/// se realisticky drží ≤ 6 sousedů. Fixní array → no heap alloc.
pub const MAX_BONDS_PER_CELL: usize = 6;
/// Tiků kontaktu (cells in collision range, mutual bond_signal active) než se
/// vytvoří spring bond. Sprint 71 měl 30 (= 0.5 s při 60 Hz). Sprint 76:
/// 30 → 10 (~0.17 s) — Sprint 75 smoke ukázal, že cells se speed 190 mají
/// brief contacts; 30 ticks gating filtroval většinu reálných bond
/// candidates. 10 ticks dovolí formaci i z krátkých but consenting contactů.
/// Risk: nestabilní bondy formující se z náhodného mihem (mitigated tím,
/// že bond_signal threshold zůstává 0.2).
pub const BOND_FORM_TICKS: u32 = 5;
/// Tiků bez kontaktu po kterých contact_progress klesne na 0 (cleanup sparse
/// FxHashMap). Krátký timeout — pár, který se rozejde, ztrácí "track" hned.
pub const CONTACT_DECAY_TICKS: u32 = 5;
/// Spring constant k (acceleration / displacement / mass). 4 dává natural
/// freq ~ √4 / mass; pro mass=1 je perioda ~π s. Damping ji rychle utlumí.
/// Sprint 68: per-bond stiffness se ukládá do Bond struct při formaci jako
/// průměr `genome.bond_stiffness` obou cells. BOND_STIFFNESS zůstává jen
/// jako center pro initial draw v Genome::random.
pub const BOND_STIFFNESS: f32 = 4.0;
/// Sprint 68: per-cell `genome.bond_stiffness` rozsah. Široký rozsah aby
/// selekce mohla zkoušet jak floppy (k≈0.5, slouží spíš jako adhesion bond)
/// tak rigid (k≈16, snapne při menší deformaci).
pub const MIN_BOND_STIFFNESS: f32 = 0.5;
pub const MAX_BOND_STIFFNESS: f32 = 16.0;
/// Damping podél spring axis pro relativní velocity. Bez damping by spring
/// oscilace explodovala (Sprint 65 collision damping je 0.5; bond má jiný
/// regime — drží spojené). 0.6 = critically damped pro typický mass.
/// Sprint 68: per-bond — ukládá se do Bond struct při formaci jako průměr
/// `genome.bond_damping` obou cells. BOND_DAMPING zůstává jako initial
/// draw center.
pub const BOND_DAMPING: f32 = 0.6;
/// Sprint 68: per-cell `genome.bond_damping` rozsah. 0 = under-damped (springs
/// kmitají), 2 = over-damped (rychle ztuhne). 0.6 ≈ critical pro typický mass.
pub const MIN_BOND_DAMPING: f32 = 0.0;
pub const MAX_BOND_DAMPING: f32 = 2.0;
/// Bond se trhá při current_length > rest_length × factor. 2.5 = 150 % strain
/// před break — silný stretch (cell se vlastní motorikou trhá z agregátu)
/// vs naturální oscilace ne-rozbije bond.
pub const BOND_BREAK_FACTOR: f32 = 2.5;
/// Násobitel kontaktní vzdálenosti pro rest_length. 1.05 = mírný "buffer"
/// (kontakt drží trochu volněji než exact touching) → preventivní polštář
/// proti inicial overlap při formaci.
pub const BOND_REST_LENGTH_SLACK: f32 = 1.05;
/// Brain output[9] musí být ≥ tento threshold u OBOU buněk pro vznik bondu.
/// Mirror MATING_PHEROMONE_THRESHOLD / ATTACK_THRESHOLD semantiku.
pub const BOND_FORM_THRESHOLD: f32 = 0.0;
/// Brain output[9] < tento threshold u některé z bonded cells → bond se
/// explicit trhá tento tick. Negative = "pusť mě". Asymmetric: jeden silný
/// negativní signál stačí (escape behavior).
pub const BOND_BREAK_THRESHOLD: f32 = -0.5;
/// Energy cost při formaci bondu (one-shot, paid by initiator). Ne-trivial,
/// aby selekce váhala bonding vs free-roaming.
pub const BOND_FORMATION_COST: f32 = 0.2;
/// Per-second cost udržování každého bondu (paid každý tick). Drobný — bond
/// je výhoda (tissue stability), ale ne free. Sprint 74: 0.1 → 0.05 (2×
/// cheaper). Sprint 73 ukázalo, že bonded 3-cell platí 3.0/gen vs solo
/// 0.17/gen → 18× inverted. Halving maintenance + 2× hunt damage +
/// 1.5× hunters target ~5× shrink toward break-even.
pub const BOND_MAINTENANCE_PER_SEC: f32 = 0.05;
/// Sprint 78: food-share fraction per bond. Když bonded cell eats food
/// (FOOD_VALUE energy), každý bonded partner dostane `FOOD_VALUE × FRAC`
/// extra energy (free reward, no energy conservation — modeluje „tissue
/// metabolic cooperation"). Cluster s 2 bondy: eater +FOOD_VALUE,
/// 2 partneři +0.6 × FOOD_VALUE = +12 each. Total cluster gain je
/// 1 + 2×0.3 = 1.6× větší než solo. Direct positive selection signál
/// pro bonding — fitness payoff přímo, ne přes hunter immunity proxy.
pub const BOND_FOOD_SHARE_FRAC: f32 = 1.0;
/// Sprint 87: cluster-size bonus pro food share fraction. Per-partner share =
/// `FRAC × (1 + (n_bonds − 1) × BONUS) × donor_state`. Cells hluboko v tkáni
/// (víc bondů) sdílí každému partnerovi vyšší podíl — empirie ze 300-gen
/// runs ukázala kolaps tissue regimu (bond_active_frac → 0 do gen 200), takže
/// linear bonus per added bond posiluje selekci pro velké clustery. n=1 →
/// ×1.00, n=2 → ×1.15, n=6 (max) → ×1.75. Žádný cap (max je MAX_BONDS_PER_CELL=6).
pub const BOND_FOOD_SHARE_CLUSTER_BONUS: f32 = 0.15;

// ─── Sprint 80: bistabilní cell-state (epigenetic-like memory) ──────────────
// Per-cell continuous scalar [0,1] s pozitivním feedbackem okolo 0.5 →
// dva stabilní attractory (≈0 = "selfish", ≈1 = "altruist"). State není v
// genomu — dědí se z parenta s šumem, takže lineages můžou rozvinout
// fenotypovou paměť napříč generacemi bez DNA změny. Driver pro switch
// je `n_bonds()` (cell hluboko v clusteru = víc bondů = push k altruist
// attractoru). Coupling: `cell_state` modulates BOND_FOOD_SHARE_FRAC →
// uvnitř clusteru emergují role „donor" vs „free-rider".

/// Pozitivní feedback rate. `s' += K × (s − 0.5) × dt`. Při K=0.5 a dt~0.05
/// (1 tick = 1/20 s) per-tick deflexe je ~0.025 × (s−0.5) — pomalý, ale
/// jistý posun k attractorům, jakmile cell vystoupí z neutrální zóny.
pub const CELL_STATE_FEEDBACK_K: f32 = 0.5;
/// Per-bond enviromentální drive. `s' += BOND_BIAS × n_bonds × dt`. n_bonds=4
/// (typický tissue cell) přidá 0.04 × 0.05 = 0.002 per tick směrem k 1.0.
/// Slabší než feedback, aby pure feedback nevytlačoval cells z 0-attractoru
/// jen proto, že krátce mají bond — bias musí konzistentně tlačit přes víc
/// ticků, než si feedback převezme režii.
pub const CELL_STATE_BOND_BIAS: f32 = 0.04;
/// σ pro Gaussian-like noise při dědění (uniform [-σ, σ] approx). Mating
/// child dostane `(parent_a.state + parent_b.state)/2 + uniform(±σ)`.
/// 0.05 = jeden „skok" za ~10 generací průměrně přes attractor boundary
/// — rare flip, ale ne lock-in.
pub const CELL_STATE_INHERIT_NOISE: f32 = 0.05;
/// Initial population kick okolo 0.5, aby cells nestartovaly přesně na
/// nestabilním fixed pointu (jinak by feedback nikdy nezačal — symetrie).
pub const CELL_STATE_INIT_KICK: f32 = 0.05;

/// Sprint 70: jitter radius pro cluster-aware reproduction. Když má parent
/// bondy a child se má spawn-it blízko něj (uvnitř bond network), použije
/// se random offset v ±tomto rangi. 8.0 = 0.8× pair_radius pro typical
/// post-evolution body (radius ~1.0, pair_r ≈ 10) — tj. uvnitř bond contact
/// distance, takže existing collision-based bond formation chytne v <1 s.
pub const CLUSTER_SPAWN_RADIUS: f32 = 8.0;

/// Sprint 69: predation damage / gain reduction per active bond, kořist-side.
/// Bonded cluster sdílí defense — útok na bonded prey vrací míň energie
/// predátorovi a působí míň damage. Per-bond reduction (capped) převrací
/// Sprint 67.1 závěr, že bonding je individual fitness-cost — teď je to
/// group-defense benefit, který by měl být evolučně positive selekcí pro
/// bondování. 15 % per bond × cap 4 = max 60 % reduction (4-bond clusters
/// jsou v podstatě immune krátce; predátor pořád dostane něco z 1- a 2-bond cells).
pub const BOND_DEFENSE_FRAC: f32 = 0.15;
/// Maximum bondů, které se počítají do defense multiplikátoru. Cap brání
/// stacking abuse (cell s 6 bondy = 100% immune). 4 = sweet spot, kde
/// střední cluster (3-4 bondy) má smysluplnou ochranu, ale solo cell jasně
/// horší (jen 0.85× damage).
pub const BOND_DEFENSE_CAP: u32 = 4;

/// Sprint 69: multiplikátor predation gain + damage podle počtu kořistních
/// bondů. Vrací hodnotu v [0.4, 1.0]. n_bonds=0 → 1.0 (no defense), n_bonds≥4
/// → 1.0 - 0.15×4 = 0.4 (max defense). Linear in n_bonds.
#[inline]
pub fn bond_defense_factor(n_bonds: u32) -> f32 {
    let capped = n_bonds.min(BOND_DEFENSE_CAP) as f32;
    1.0 - BOND_DEFENSE_FRAC * capped
}

/// Sprint 92: exposure factor pro hunter damage. Edge cells fully exposed,
/// interior cells fully shielded — selection pressure favorizuje větší +
/// 3D-spherical clusters.
///
/// Sprint 96: **non-linear quadratic falloff** — `((1 - n × EXPOSURE_PER_BOND))²`.
/// Pre-S96 linear pomalu odměňovala 1-2 bondy (75/50% damage); selection
/// favorizovala solo strategy. Quadratic dramaticky odměňuje first 1-2
/// bondy (56/25% damage) → cost-benefit balance flips ve prospěch
/// bonding. Sprint 95 negative result diagnosed nedostatečný defense
/// reward jako kořen; S96 fixes přes funkční nelinearitu.
///
/// Linear (S92) → Quadratic (S96):
/// - 0 bonds: 1.00 → 1.00 (unchanged)
/// - 1 bond:  0.75 → **0.56** (33% better defense)
/// - 2 bonds: 0.50 → **0.25** (50% better)
/// - 3 bonds: 0.25 → **0.06** (76% better)
/// - ≥4 bonds: 0.00 (still floor)
#[inline]
pub fn cell_exposure(n_bonds: u32) -> f32 {
    let linear = (1.0 - (n_bonds as f32) * EXPOSURE_PER_BOND).max(0.0);
    linear * linear
}

// ─── Sprint 71: macropredator (Hunter) ────────────────────────────────────────
// Sprint 70 long-run odhalil emergent predator-extinction event: cell-vs-cell
// predace zkolapsovala, jakmile byly bonded clustery dostatečně tough. Sprint
// 71 zavádí non-evolving environmental predátora („Hunter") který běží mimo
// Cell selection loop — nikdy nevyhyne, protože není pod evolučním tlakem.
// Hunters atakují solo / lightly-bonded cells; cells s ≥3 bondy jsou immune
// (cluster „too big to swallow" — Volvox/paramecium scenario z reálné biologie).
// Tím dává sim persistent pressure na ≥3-bond clusters = exact tipping point
// pro tissue formation.

/// Cílový počet hunterů ve světě. Sprint 71 měl 3 = příliš sparse
/// (escape-by-speed cells našly free corridors). Sprint 72: 8 hunterů
/// pokrývá víc paths, solo cell má kratší survival window. Sprint 74:
/// 12 — Sprint 73 1000-gen smoke ukázal, že 8 hunterů × 1500 atks/gen
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



/// Sprint 92: food kind tagged enum. Differentiates plant (ambient spawn,
/// always available, baseline value), cell carrion (drops on cell death),
/// hunter carrion (drops on hunter death, richest reward). Eat efficiency
/// per kind je modulated by cell `genome.carnivore_score` ∈ [0, 1] —
/// herbivore digestion vs carnivore digestion trade-off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum FoodKind {
    Plant = 0,
    Carrion = 1,
    HunterCarrion = 2,
}

impl Default for FoodKind {
    fn default() -> Self {
        FoodKind::Plant
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Food {
    pub position: [f32; 3],
    /// Sprint 42: ticks od spawnu. Drives decay of `value_factor`. Init 0
    /// pro fresh food i carrion (univerzální decay, žádný carrion-specific
    /// staleness offset).
    pub age_ticks: u32,
    /// Sprint 92: food kind. `serde(default)` returns Plant pro backward-
    /// compat s pre-S92 checkpointy.
    #[serde(default)]
    pub kind: FoodKind,
}

/// Sprint 92: base food value per kind. Carrion má vyšší value než plant
/// (concentrated biomass), hunter carrion ještě víc (apex predator drop).
pub const PLANT_FOOD_VALUE: f32 = 20.0;
pub const CARRION_FOOD_VALUE: f32 = 30.0;
pub const HUNTER_CARRION_FOOD_VALUE: f32 = 50.0;

/// Sprint 128: cooperative food node — high-value spawn, který nepřináší
/// nic dokud N cells během time window nedorazí. Vytváří fitness coupling
/// pro recruitment signaling: solo cells nedostanou nic, coordinated
/// trio dostane high reward → selekce na "I see food, signal peers".
pub const COOP_FOOD_REQUIRED_ARRIVALS: usize = 3;
/// Time okno (ticks) od spawnu. Po vypršení: despawn bez reward.
pub const COOP_FOOD_TIME_WINDOW_TICKS: u32 = 120;
/// Per-participant reward při úspěšné koordinaci. Asymetricky vysoký vůči
/// regular Plant food (20) — incentive justifying loiter cost.
pub const COOP_FOOD_REWARD_PER_CELL: f32 = 80.0;
/// Radius (sim units), v rámci kterého cell counts as "arrived". Větší než
/// regular eat radius (~20) — coop food má vizuální/aroma signal "here is
/// gathering point", cells nemusí stát přímo na něm.
pub const COOP_FOOD_ARRIVAL_RADIUS: f32 = 30.0;
/// Spawn pravděpodobnost per tick (Poisson-like). Kalibrováno tak, aby vznikalo
/// cca 10-15 coop nodes per generation (600 ticků). 0.02 → ~12 events/gen.
pub const COOP_FOOD_SPAWN_RATE_PER_TICK: f32 = 0.02;
/// Max simultaneous coop nodes ve světě. Cap pro ohraničení complexity
/// (a paměť).
pub const COOP_FOOD_MAX_CONCURRENT: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoopFood {
    pub position: [f32; 3],
    pub spawn_tick: u64,
    /// Set unique cell_ids, které byly v `ARRIVAL_RADIUS` aspoň jeden tick.
    /// Vec<u64> aby zachoval Serde + insertion order; lookup je O(N), ale
    /// N je malé (typicky < 10).
    pub arrivals: Vec<u64>,
    /// True pokud byl threshold dosažen → reward distribuován + bude
    /// despawnut na konci aktuálního ticku.
    pub triggered: bool,
}

impl CoopFood {
    pub fn new(position: [f32; 3], spawn_tick: u64) -> Self {
        Self {
            position,
            spawn_tick,
            arrivals: Vec::new(),
            triggered: false,
        }
    }

    /// True pokud věk > TIME_WINDOW. Caller volá po pokusu o trigger,
    /// aby triggered nodes (které trigger zvládly přesně v expiry frame)
    /// nebyly mylně klasifikovány jako "expired no reward".
    #[inline]
    pub fn is_expired(&self, current_tick: u64) -> bool {
        current_tick.saturating_sub(self.spawn_tick) >= COOP_FOOD_TIME_WINDOW_TICKS as u64
    }
}

/// Sprint 128: zaregistruj cell_id jako arrival. Insertion order zachovaná,
/// duplikáty ignorovány (cell může být v radius víc ticků). Vrací true pokud
/// byl id přidán, false pokud už evidoval.
pub fn register_coop_arrival(coop: &mut CoopFood, cell_id: u64) -> bool {
    if coop.arrivals.iter().any(|id| *id == cell_id) {
        return false;
    }
    coop.arrivals.push(cell_id);
    true
}

/// Sprint 128: pokus o trigger threshold. Vrací true pokud byl reward distribuován
/// (= aspoň REQUIRED arrivals + ne-yet-triggered). Caller propaguje return value
/// do per-gen counterů (coop_food_solved).
pub fn try_trigger_coop(coop: &mut CoopFood, cells: &mut [Cell]) -> bool {
    if coop.triggered || coop.arrivals.len() < COOP_FOOD_REQUIRED_ARRIVALS {
        return false;
    }
    for cell_id in &coop.arrivals {
        if let Some(cell) = cells.iter_mut().find(|c| c.cell_id == *cell_id) {
            cell.energy += COOP_FOOD_REWARD_PER_CELL;
        }
    }
    coop.triggered = true;
    true
}

/// Sprint 128: vyber random pozici uvnitř world bounds (toroidal world,
/// stejná logika jako `Food::random`). Pokud `world_half[2] == 0`, z-osa
/// vrací 0 — backward-compat s pre-S33 baseline.
pub fn random_coop_position(rng: &mut impl Rng, world_half: [f32; 3]) -> [f32; 3] {
    let z = if world_half[2] > 0.0 {
        rng.random_range(-world_half[2]..world_half[2])
    } else {
        0.0
    };
    [
        rng.random_range(-world_half[0]..world_half[0]),
        rng.random_range(-world_half[1]..world_half[1]),
        z,
    ]
}

/// Sprint 128: per-tick scan + arrival registration pro každý coop node.
/// Cell je v radius pokud (toroidal-aware) Euclidean distance ≤ ARRIVAL_RADIUS.
pub fn register_coop_arrivals_for_all(coops: &mut [CoopFood], cells: &[Cell], world_half: [f32; 3]) {
    let r2 = COOP_FOOD_ARRIVAL_RADIUS * COOP_FOOD_ARRIVAL_RADIUS;
    for coop in coops.iter_mut() {
        if coop.triggered {
            continue;
        }
        for cell in cells.iter() {
            let d = min_image_delta(coop.position, cell.position, world_half);
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if d2 <= r2 {
                let _ = register_coop_arrival(coop, cell.cell_id);
            }
        }
    }
}

#[inline]
pub fn food_base_value(kind: FoodKind) -> f32 {
    match kind {
        FoodKind::Plant => PLANT_FOOD_VALUE,
        FoodKind::Carrion => CARRION_FOOD_VALUE,
        FoodKind::HunterCarrion => HUNTER_CARRION_FOOD_VALUE,
    }
}

/// Sprint 92: digestion efficiency per food kind × cell `carnivore_score`.
/// Continuous trade-off: 0 = pure herbivore (plant only), 1 = pure carnivore
/// (hunter carrion only), 0.5 = mixed (everything moderate).
///
/// - Plant + score 0.0 → 1.0 (full)
/// - Plant + score 1.0 → 0.0 (can't digest plants at all)
/// - HunterCarrion + score 0.0 → 0.0 (can't digest)
/// - HunterCarrion + score 1.0 → 1.0 (full)
/// - Carrion (cell) → 0.5 universally — semi-digestible by both diets
///   (compromise food, doesn't drive specialization)
#[inline]
pub fn eat_efficiency(kind: FoodKind, carnivore_score: f32) -> f32 {
    let s = carnivore_score.clamp(0.0, 1.0);
    match kind {
        FoodKind::Plant => 1.0 - s,
        FoodKind::Carrion => 0.5,
        FoodKind::HunterCarrion => s,
    }
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
            age_ticks: 0,
            kind: FoodKind::Plant,
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

    /// Sprint 42: lineární decay value factor podle stáří. Pro age=0 vrací 1.0,
    /// klesá lineárně k nule, pak clampnuto na 0.
    pub fn value_factor(&self) -> f32 {
        let age_sec = self.age_ticks as f32 / FIXED_TIMESTEP_HZ;
        (1.0 - CARRION_DECAY_PER_SEC * age_sec).max(0.0)
    }

    /// Sprint 42: age tick increment. Vrací `false` pokud food expiroval
    /// (value_factor ≤ 0) — caller despawne. Volá se po `apply_gravity`
    /// v hot loopu binárek.
    pub fn age_step(&mut self) -> bool {
        self.age_ticks = self.age_ticks.saturating_add(1);
        self.value_factor() > 0.0
    }
}

/// Sprint 33: 3D forward unit vector z yaw + pitch. Bez roll (cells axially
/// symetrické). Pro pitch=0 redukuje na (cos(yaw), sin(yaw), 0) — backward
/// kompat s pre-Sprint-33 2D heading semantikou.
pub fn forward_vector(yaw: f32, pitch: f32) -> [f32; 3] {
    let cos_p = pitch.cos();
    [yaw.cos() * cos_p, yaw.sin() * cos_p, pitch.sin()]
}

/// Sprint 121: směr spike v world frame. `azimuth_offset` přidá k yaw,
/// `elevation_offset` přidá k pitch. Pre-S121 (azimuth=elevation=0) redukuje
/// na `forward_vector` — single frontal spike.
pub fn spike_direction(yaw: f32, pitch: f32, spike: &Spike) -> [f32; 3] {
    forward_vector(yaw + spike.azimuth_offset, pitch + spike.elevation_offset)
}

/// Sprint 123: complexity multiplikátor na attack predation bonus.
/// `1 + COMPLEXITY_ATTACK_GAIN × complexity`. S complexity=0 vrací 1.0
/// (pre-S123 sémantika), s complexity=1 vrací 1 + COMPLEXITY_ATTACK_GAIN.
#[inline]
pub fn spike_complexity_attack_factor(complexity: f32) -> f32 {
    1.0 + COMPLEXITY_ATTACK_GAIN * complexity.clamp(0.0, 1.0)
}

/// Sprint 123: complexity multiplikátor na maintenance cost. Quadratic —
/// max-complexity (`1 + COMPLEXITY_COST_GAIN`) je výrazně dražší než
/// střední (`1 + COMPLEXITY_COST_GAIN × 0.25`), aby selekce nesložila
/// k degenerate max-complexity single-spike strategii.
#[inline]
pub fn spike_complexity_cost_factor(complexity: f32) -> f32 {
    let c = complexity.clamp(0.0, 1.0);
    1.0 + COMPLEXITY_COST_GAIN * c * c
}

/// Sprint 123: complexity multiplikátor na eat grab cone half-angle.
/// `1 + COMPLEXITY_GRAB_GAIN × complexity` — větvený spike pokrývá širší
/// kužel u tipu.
#[inline]
pub fn spike_complexity_grab_factor(complexity: f32) -> f32 {
    1.0 + COMPLEXITY_GRAB_GAIN * complexity.clamp(0.0, 1.0)
}

/// Sprint 82: cost faktor pro směrový FOV. `theta` = half-angle kuželu kolem
/// `forward_vector`. Solid angle kuželu = 2π(1 − cos θ); normalizováno na
/// [0,1] (full sphere → 1, narrow → 0). Použité jako multiplikátor pro
/// `vision_radius × VISION_COST_PER_RADIUS` v `apply_energy_costs` — užší
/// kužel platí menší vision drain, ale ztrácí informace v slepém úhlu.
#[inline]
pub fn vision_fov_factor(theta: f32) -> f32 {
    let t = theta.clamp(0.0, MAX_VISION_FOV);
    (1.0 - t.cos()) * 0.5
}

/// Sprint 85: lineární z-gradient teploty. Warm at top (`world_half[2]`),
/// cold at bottom (`-world_half[2]`). Pro `world_half[2] == 0` (Sprint 32
/// pre-3D baseline) vrací `THERMAL_REF_TEMP` → `metabolism_factor = 1.0` →
/// drain backward-compat s pre-Sprint-85.
///
/// Sprint 86: time-varying. `tick` parametr aplikuje diurnal oscilaci
/// (surface-weighted, hloubka neoscilluje), `generation` parametr aplikuje
/// uniform seasonal shift (synchronní s food density cyklem). Při
/// `tick = 0, generation = 0` jsou oba sin(0) = 0 → identical s pre-S86.
#[inline]
pub fn temperature_at_z(z: f32, world_half: [f32; 3], tick: u64, generation: u64) -> f32 {
    if world_half[2] <= 0.0 {
        return THERMAL_REF_TEMP;
    }
    let normalized = ((z / world_half[2]) + 1.0) * 0.5;
    let normalized = normalized.clamp(0.0, 1.0);
    let base = THERMAL_BOTTOM + (THERMAL_TOP - THERMAL_BOTTOM) * normalized;
    // Sprint 86: seasonal — uniform shift, period = CYCLE_GEN_PERIOD (50 gen).
    // Modulo gen drží phase v [0, 1) bez f32 precision ztráty pro long runs.
    let seasonal_phase =
        (generation % CYCLE_GEN_PERIOD) as f32 / CYCLE_GEN_PERIOD as f32;
    let seasonal_offset = THERMAL_SEASONAL_AMP * (TAU * seasonal_phase).sin();
    // Sprint 86: diurnal — surface-weighted (× normalized), period 1 day =
    // THERMAL_DIURNAL_PERIOD_TICKS. Bottom (normalized = 0) → no oscillation;
    // surface (normalized = 1) → full AMP.
    let diurnal_phase =
        (tick % THERMAL_DIURNAL_PERIOD_TICKS) as f32 / THERMAL_DIURNAL_PERIOD_TICKS as f32;
    let diurnal_offset =
        THERMAL_DIURNAL_AMP * normalized * (TAU * diurnal_phase).sin();
    base + seasonal_offset + diurnal_offset
}

/// Sprint 85: Q10 metabolism multiplikátor. `Q10^((T − T_REF) / 10)`.
/// `T = THERMAL_REF_TEMP` → 1.0 (no-op, identický s pre-Sprint-85).
/// `T = THERMAL_TOP = 30` (REF + 13) → 2^1.3 ≈ 2.46 (warm cells drain rychleji).
/// `T = THERMAL_BOTTOM = 4` (REF − 13) → 2^−1.3 ≈ 0.41 (cold cells drain pomaleji).
#[inline]
pub fn metabolism_factor(temp: f32) -> f32 {
    THERMAL_Q10.powf((temp - THERMAL_REF_TEMP) / 10.0)
}

/// Sprint 83: per-cell cone test pro sensor gather. Vrací `true` pokud
/// kandidát ve směru `delta` (od cell k targetu) leží uvnitř FOV kuželu
/// kolem `forward`. `cos_fov_threshold` = `cos(vision_fov)` precomputed
/// jednou per cell; volání je hot-path uvnitř `for_each_in_radius_toroidal`.
/// `d2` je `delta · delta` (už spočítáno pro radius test); umožňuje výpočet
/// `|delta|` pomocí jediného sqrt.
///
/// Degenerate case `|delta| ≈ 0` (target přímo na cell pozici) vrací `true`
/// — cell vidí self-overlap region nezávisle na orientaci.
///
/// Pro `cos_fov_threshold = -1.0` (full sphere FOV) by formálně všechny
/// kandidáti procházely; caller však typicky short-circuituje přes
/// `vision_fov >= MAX_VISION_FOV` flag, takže tato funkce se ani nevolá.
#[inline]
pub fn fov_cone_accept(
    delta: [f32; 3],
    d2: f32,
    forward: [f32; 3],
    cos_fov_threshold: f32,
) -> bool {
    if d2 < 1e-12 {
        return true;
    }
    let dot = delta[0] * forward[0] + delta[1] * forward[1] + delta[2] * forward[2];
    // Místo `dot / |delta| >= threshold` porovnáváme bez dělení:
    //   threshold > 0: dot > 0 AND dot² >= threshold² × d²
    //   threshold ≤ 0: dot ≥ threshold × |delta|  → potřebujem sqrt
    // Protože threshold pochází z `cos(vision_fov)` ∈ [cos(π/12), 1] = [0.97, 1] pro
    // typické úzké FOV, plus krátkodobě může jít pod 0 při hemisphere+ FOV během
    // evoluce, používáme jednotnou cestu se sqrt — jednoznačné a numericky stable.
    let mag = d2.sqrt();
    dot >= cos_fov_threshold * mag
}

/// Sprint 41: orthonormální body frame z (yaw, pitch). `fwd` je `forward_vector`,
/// `right` je horizontální (rotace forward_xy o +90°, z=0), `up = fwd × right`.
/// Bez roll. Pro pitch=0 dává up=(0,0,1) a right čistě v xy. Použité pro
/// projekci food vektoru do body frame v ellipsoidní eat-zóně.
pub fn body_basis(yaw: f32, pitch: f32) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let fwd = forward_vector(yaw, pitch);
    let right = [-yaw.sin(), yaw.cos(), 0.0];
    let up = [
        fwd[1] * right[2] - fwd[2] * right[1],
        fwd[2] * right[0] - fwd[0] * right[2],
        fwd[0] * right[1] - fwd[1] * right[0],
    ];
    (fwd, right, up)
}

/// Sprint 58: pure ellipsoid-acceptance test bez `&Cell` reference. Stejná
/// matematika jako `Cell::eat_test` ale parametrizovaná — umožňuje volat z
/// rayon par_iter snapshotu kde nedrží `&Cell` (Bevy Query lifetime).
/// `body_dims = [length, width, height]`.
pub fn eat_test_pose(
    cell_pos: [f32; 3],
    heading: f32,
    pitch: f32,
    body_dims: [f32; 3],
    food_pos: [f32; 3],
    eat_factor: f32,
) -> bool {
    let dx = food_pos[0] - cell_pos[0];
    let dy = food_pos[1] - cell_pos[1];
    let dz = food_pos[2] - cell_pos[2];
    let (fwd, right, up) = body_basis(heading, pitch);
    let d_par = dx * fwd[0] + dy * fwd[1] + dz * fwd[2];
    let d_right = dx * right[0] + dy * right[1] + dz * right[2];
    let d_up = dx * up[0] + dy * up[1] + dz * up[2];
    let l = (body_dims[0] * eat_factor).max(f32::EPSILON);
    let w = (body_dims[1] * eat_factor).max(f32::EPSILON);
    let h = (body_dims[2] * eat_factor).max(f32::EPSILON);
    (d_par / l).powi(2) + (d_right / w).powi(2) + (d_up / h).powi(2) <= 1.0
}

/// Sprint 40: senzorický kontext brainu. Volá se z hot loop binárek po
/// průzkumu okolí (nejbližší food/cell, density, gradient field). Konkrétní
/// gathering algoritmus závisí na binárce (main grid lookup vs headless
/// O(N²) sweep), ale výstup struct je společný — pak `populate_brain_inputs`
/// vyplní sjednocený `[f32; BRAIN_INPUTS]` array.
#[derive(Debug, Clone, Copy)]
pub struct BrainSensors {
    /// Sprint 54: min-imaged signed delta (target − cell), ne absolutní target
    /// pozice. Pro toroidal world je delta správný relativní vektor i přes
    /// world wrap (cell na x=−950 vidí target na x=+950 jako Δx=20, ne 1900).
    /// Pre-Sprint-54 byl absolute target_pos; populate_brain_inputs odečetl
    /// cell.position. Po Sprintu 54 je odečtení už hotové (s wrap).
    pub nearest_food: Option<[f32; 3]>,
    pub nearest_cell: Option<([f32; 3], f32)>,
    pub neighbors_in_vision: u32,
    pub smell_grad: [f32; 3],
    /// Sprint 126: per-channel pheromone gradients. ch0 (= existing slow
    /// channel) backward-compat sloty 11/12/19 v populate_brain_inputs;
    /// ch1, ch2 nové sloty 21-23 / 24-26.
    pub pheromone_grads: [[f32; 3]; N_PHEROMONE_CHANNELS],
    /// Sprint 87: aktuální teplota na cell pozici (sim units, ne normalized).
    /// Caller spočítá `temperature_at_z(pos[2], world_half, tick, gen)`.
    /// `populate_brain_inputs` normalizuje přes `tanh((T − REF) / 10)` →
    /// brain input [-1, 1] (Q10-aware škálování).
    pub temperature_local: f32,
}

/// Sprint 40: jediný source of truth pro brain inputs layout. Pre-refactor byl
/// duplikovaný v `main::cells_brain_act` a `headless::brain_act` — drift mezi
/// nimi by tichost porušil binární CSV identity. `damage_accum` se zde čte +
/// resetuje (1-tick delay konzistentně se Sprint 30 semantikou).
pub fn populate_brain_inputs(
    cell: &mut Cell,
    sensors: &BrainSensors,
    vision_r: f32,
) -> [f32; BRAIN_INPUTS] {
    let pos = cell.position;
    let my_radius = cell.phenotype.effective_radius().max(0.01);
    let max_speed = cell.genome.max_speed;
    // Sprint 32 note: hypot(vx, vy) místo (vx²+vy²+vz²).sqrt() pro ULP
    // identity s pre-Sprint-32 trajektorií. Sprint 33+ vz != 0; hypot
    // ignoruje vz, ale rozdíl je sub-ULP. Sprint 41+ může přejít na 3D mag.
    let speed_norm = (cell.velocity[0].hypot(cell.velocity[1]) / max_speed).clamp(0.0, 1.0);
    let energy_norm = (cell.energy / REPRODUCE_THRESHOLD).clamp(0.0, 1.5);

    let mut inputs = [0.0_f32; BRAIN_INPUTS];
    if let Some(delta) = sensors.nearest_food {
        // Sprint 54: BrainSensors už nese min-imaged delta (toroidal-aware).
        inputs[0] = delta[0] / vision_r;
        inputs[1] = delta[1] / vision_r;
        inputs[15] = delta[2] / vision_r;
    }
    if let Some((delta, other_radius)) = sensors.nearest_cell {
        inputs[2] = delta[0] / vision_r;
        inputs[3] = delta[1] / vision_r;
        inputs[6] = (other_radius - my_radius) / my_radius;
        inputs[16] = delta[2] / vision_r;
    }
    let _ = pos;
    inputs[4] = energy_norm;
    inputs[5] = speed_norm;
    inputs[7] = (sensors.smell_grad[0] * SMELL_NORMALIZATION_GAIN).tanh();
    inputs[8] = (sensors.smell_grad[1] * SMELL_NORMALIZATION_GAIN).tanh();
    inputs[17] = (sensors.smell_grad[2] * SMELL_NORMALIZATION_GAIN).tanh();
    let fwd = forward_vector(cell.heading, cell.pitch);
    inputs[9] = fwd[0];
    inputs[10] = fwd[1];
    inputs[18] = fwd[2];
    inputs[11] = (sensors.pheromone_grads[0][0] * PHEROMONE_NORMALIZATION_GAIN).tanh();
    inputs[12] = (sensors.pheromone_grads[0][1] * PHEROMONE_NORMALIZATION_GAIN).tanh();
    inputs[19] = (sensors.pheromone_grads[0][2] * PHEROMONE_NORMALIZATION_GAIN).tanh();
    inputs[13] = (sensors.neighbors_in_vision as f32 / DENSITY_NORM_COUNT).tanh();
    inputs[14] = (cell.damage_accum * DAMAGE_NORMALIZATION_GAIN).tanh();
    cell.damage_accum = 0.0;
    // Sprint 87: thermal awareness, slot 20. Q10-aware tanh normalizace —
    // (T - REF) / 10 dává tanh(±1.3) ≈ ±0.86 na endpoints [BOTTOM, TOP],
    // tanh(0) = 0 na ref. Diurnal/seasonal posuny mohou krátkodobě saturovat
    // k ±1, což je akceptovatelná oversaturace pro brain signal.
    inputs[20] = ((sensors.temperature_local - THERMAL_REF_TEMP) / 10.0).tanh();
    // Sprint 126: ch1, ch2 pheromone gradients. ch0 zachované na sloty 11/12/19.
    inputs[21] = (sensors.pheromone_grads[1][0] * PHEROMONE_NORMALIZATION_GAIN).tanh();
    inputs[22] = (sensors.pheromone_grads[1][1] * PHEROMONE_NORMALIZATION_GAIN).tanh();
    inputs[23] = (sensors.pheromone_grads[1][2] * PHEROMONE_NORMALIZATION_GAIN).tanh();
    inputs[24] = (sensors.pheromone_grads[2][0] * PHEROMONE_NORMALIZATION_GAIN).tanh();
    inputs[25] = (sensors.pheromone_grads[2][1] * PHEROMONE_NORMALIZATION_GAIN).tanh();
    inputs[26] = (sensors.pheromone_grads[2][2] * PHEROMONE_NORMALIZATION_GAIN).tanh();
    // Sprint 94: cluster-shared brain. Recurrent slots (21..52) čtou
    // `pooled_hidden` (mean self + bonded neighbors z předchozího ticku)
    // místo `last_hidden`. Solo cells: pool == self → behavior identical
    // s pre-Sprint-94. Cluster cells: shared memory → effective větší
    // context window, drives proto-distributed cognition.
    inputs[BRAIN_INPUTS_SENSORY..BRAIN_INPUTS_SENSORY + BRAIN_RECURRENT]
        .copy_from_slice(&cell.pooled_hidden[..BRAIN_RECURRENT]);
    inputs
}

/// Sprint 97: in-place multiplication brain inputs × per-category sensor_gains.
/// Applies gains to environmental + defensive slots, leaves proprio slots
/// (energy, speed, heading) untouched. Solo cells s gain < 1.0 lose info,
/// cluster cells s pooling can compensate via partner signals.
#[inline]
pub fn apply_sensor_gains(inputs: &mut [f32; BRAIN_INPUTS], gains: &[f32; N_SENSOR_CATEGORIES]) {
    for slot in 0..BRAIN_INPUTS_SENSORY {
        if let Some(cat) = sensor_slot_category(slot) {
            inputs[slot] *= gains[cat];
        }
    }
}

/// Sprint 97: pool environmental + defensive sensor slots přes bond network.
/// Max-pooling — for each slot, take maximum over self + bonded partners
/// (post-gain-multiplied values). Allows specialization: cell A s vision gain=0
/// dostane visibility z partnera B s vision gain=2.0 přes max pool.
///
/// Proprio slots (energy, speed, heading) NEpoolováno — vlastní stav cells
/// musí být individuální. Recurrent slots (21..52) mají vlastní mechanismus
/// `pool_bonded_hidden` (S94 mean pooling).
///
/// Solo cell (no bonds): output == self_inputs. Pair / cluster: max nad
/// self + 1-hop partners.
pub fn pool_bonded_sensors<F>(
    cell: &Cell,
    own_inputs: &[f32; BRAIN_INPUTS],
    lookup_partner_inputs: F,
) -> [f32; BRAIN_INPUTS]
where
    F: Fn(u64) -> Option<[f32; BRAIN_INPUTS]>,
{
    let mut pooled = *own_inputs;
    for bond_slot in cell.bonds.iter().flatten() {
        if let Some(partner_inputs) = lookup_partner_inputs(bond_slot.other_cell_id) {
            for slot in 0..BRAIN_INPUTS_SENSORY {
                if sensor_slot_category(slot).is_some() {
                    let p = partner_inputs[slot];
                    if p.abs() > pooled[slot].abs() {
                        // Use max-magnitude (preserves sign for directional
                        // signals like food_dx; partner with stronger signal
                        // wins). Solo equivalent to self.
                        pooled[slot] = p;
                    }
                }
            }
        }
    }
    pooled
}

/// Sprint 94: compute pooled `last_hidden` for a single cell — mean over
/// self + bonded neighbors. Output should be assigned to `cell.pooled_hidden`
/// pre brain_act phase. Bond lookup: `bond.other_cell_id → idx` via caller-
/// supplied lookup (HashMap or array). Missing neighbors (despawned mid-tick)
/// jsou skipnuty.
///
/// Solo cell (n_bonds=0): output == self.last_hidden (no change).
/// Pair (1 bond): output = (self + partner) / 2.
/// Triad / cluster: arithmetic mean over alive bonded subgraph (1-hop only,
/// no transitive — keeps O(n_bonds) per cell, no graph traversal cost).
pub fn pool_bonded_hidden<F>(
    cell: &Cell,
    lookup_partner_hidden: F,
) -> [f32; BRAIN_HIDDEN]
where
    F: Fn(u64) -> Option<[f32; BRAIN_HIDDEN]>,
{
    let mut acc = cell.last_hidden;
    let mut count = 1.0_f32;
    for slot in cell.bonds.iter().flatten() {
        if let Some(partner_hidden) = lookup_partner_hidden(slot.other_cell_id) {
            for k in 0..BRAIN_HIDDEN {
                acc[k] += partner_hidden[k];
            }
            count += 1.0;
        }
    }
    if count > 1.0 {
        let inv = 1.0 / count;
        for k in 0..BRAIN_HIDDEN {
            acc[k] *= inv;
        }
    }
    acc
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

/// Sprint 40: greedy O(N²) párování fertile cells na základě 3D distance.
/// Generic přes Idx (usize v headless, Entity v main) — helper dedupuje
/// pairing logiku, která byla pre-refactor identická v obou binárkách.
pub fn pair_fertile<I>(
    fertile: &[(I, [f32; 3])],
    mating_r2: f32,
    budget: usize,
    world_half: [f32; 3],
) -> Vec<(I, I)>
where
    I: Copy + Eq + std::hash::Hash,
{
    use std::collections::HashSet;
    let mut paired: HashSet<I> = HashSet::new();
    let mut matings: Vec<(I, I)> = Vec::new();
    for i in 0..fertile.len() {
        if matings.len() >= budget {
            break;
        }
        let (a, pos_a) = fertile[i];
        if paired.contains(&a) {
            continue;
        }
        let mut best: Option<(I, f32)> = None;
        for (j, &(b, pos_b)) in fertile.iter().enumerate() {
            if i == j || paired.contains(&b) {
                continue;
            }
            // Sprint 54: min-image distance pro toroidal world.
            let d = min_image_delta(pos_a, pos_b, world_half);
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if d2 <= mating_r2 && best.is_none_or(|(_, bd2)| d2 < bd2) {
                best = Some((b, d2));
            }
        }
        if let Some((b, _)) = best {
            paired.insert(a);
            paired.insert(b);
            matings.push((a, b));
        }
    }
    matings
}

/// Sprint 40: vyrobí dítě z dvou rodičů (immutable refs — parent halving si
/// dělá caller před voláním). Random direction pro startovní heading +
/// crossover + mutate genomu, fresh phenotype z genomu (žádný Lamarckismus),
/// brain stav reset (last_*=0). Energy = a.energy + b.energy (caller už halved).
/// Sprint 66: caller poskytuje `cell_id` (World-level monotonic counter).
/// Sprint 70: vybere parent, do jehož bond clusteru se má spawn dítě. Priorita:
/// 1. Parent s bondy + adhesion_type matchující childovu (= dítě se chytne
///    do existujícího clusteru téhož typu).
/// 2. Pokud ani jeden parent nemá matchující adhesion, ale jeden má bondy,
///    spawn k němu (50 % šance, že se chytne — adhesion mismatch znamená,
///    že to není bond formation kandidát, ale aspoň blízká poloha).
/// 3. Pokud ani jeden parent nemá bondy, vrátí `None` → caller spawne na
///    midpoint (pre-Sprint-70 chování).
pub fn pick_cluster_parent<'a>(
    parent_a: &'a Cell,
    parent_b: &'a Cell,
    child_adhesion_type: u8,
) -> Option<&'a Cell> {
    let a_bonded = parent_a.n_bonds() > 0;
    let b_bonded = parent_b.n_bonds() > 0;
    let a_match = a_bonded && parent_a.genome.adhesion_type == child_adhesion_type;
    let b_match = b_bonded && parent_b.genome.adhesion_type == child_adhesion_type;
    if a_match {
        return Some(parent_a);
    }
    if b_match {
        return Some(parent_b);
    }
    if a_bonded {
        return Some(parent_a);
    }
    if b_bonded {
        return Some(parent_b);
    }
    None
}

pub fn make_mating_child(
    parent_a: &Cell,
    parent_b: &Cell,
    rng: &mut impl Rng,
    cell_id: u64,
) -> Cell {
    // RNG draw order zachovává pre-refactor sekvenci: crossover/mutate FIRST,
    // pak direction. Změna pořadí by porušila CSV identity / reproducibility.
    let child_genome = Genome::crossover(&parent_a.genome, &parent_b.genome, rng)
        .mutate(rng, &MUTATION_CONFIG);
    let direction = rng.random_range(0.0..TAU);
    // Sprint 70: cluster-aware jitter. Draw vždycky (i když ho nepoužijeme)
    // — RNG draw order pak zůstane consistent napříč all children, ne jen
    // bonded-parent větví. Z jitter je 0.3× kvůli užšímu z-rangi (±50 vs xy ±960).
    let jitter_x: f32 = rng.random_range(-CLUSTER_SPAWN_RADIUS..CLUSTER_SPAWN_RADIUS);
    let jitter_y: f32 = rng.random_range(-CLUSTER_SPAWN_RADIUS..CLUSTER_SPAWN_RADIUS);
    let jitter_z: f32 = rng.random_range(
        -CLUSTER_SPAWN_RADIUS * 0.3..CLUSTER_SPAWN_RADIUS * 0.3,
    );
    let mid_pos = [
        (parent_a.position[0] + parent_b.position[0]) * 0.5,
        (parent_a.position[1] + parent_b.position[1]) * 0.5,
        (parent_a.position[2] + parent_b.position[2]) * 0.5,
    ];
    // Sprint 70: pokud má kterýkoliv parent bondy + jeho adhesion_type matchuje
    // childovu, spawn dítě uvnitř jeho bond clusteru. Tím dochází k tipping
    // pointu mezi „cells occasionally bond" a „persistent multi-cell
    // organisms" — children rostou bond network místo aby ho jen redukovaly
    // skrz death (Sprint 67.1 + 69 ukázaly net formed-broken < 0).
    //
    // Sprint 79 audit: jitter může produkovat raw pozici mírně mimo world
    // bounds (max |Δ| = CLUSTER_SPAWN_RADIUS = 8 v xy, 2.4 v z). Následný
    // step() pre-tick aplikuje apply_world_bounce → toroidal xy wrap +
    // z reflective clamp. Jeden tick mezi spawn a step může grid lookup
    // vidět out-of-bounds pozici; for_each_in_radius_toroidal interně
    // používá min_image_delta, takže lookup je correct. No bug, just
    // race-tick edge case — accepted.
    let cluster_parent =
        pick_cluster_parent(parent_a, parent_b, child_genome.adhesion_type);
    let pos = match cluster_parent {
        Some(p) => [
            p.position[0] + jitter_x,
            p.position[1] + jitter_y,
            p.position[2] + jitter_z,
        ],
        None => mid_pos,
    };
    let child_phenotype = Phenotype::from_genome(&child_genome);
    Cell {
        position: pos,
        velocity: [
            direction.cos() * child_genome.max_speed,
            direction.sin() * child_genome.max_speed,
            0.0,
        ],
        angular_velocity: 0.0,
        pitch_velocity: 0.0,
        energy: parent_a.energy + parent_b.energy,
        heading: direction,
        pitch: 0.0,
        lineage_id: parent_a.lineage_id,
        lineage_birth_gen: parent_a.lineage_birth_gen,
        last_inputs: [0.0; BRAIN_INPUTS],
        last_hidden: [0.0; BRAIN_HIDDEN],
        last_outputs: [0.0; BRAIN_OUTPUTS],
        last_emit: [0.0; N_PHEROMONE_CHANNELS],
        burst_accum: [0.0; N_PHEROMONE_CHANNELS],
        pooled_hidden: [0.0; BRAIN_HIDDEN],
        damage_accum: 0.0,
        age: 0,
        // Sprint 42: child startuje s plnou cooldown — rodičovská cooldown
        // se nastaví v binárkách po `make_mating_child`, nezasáhne childa.
        reproduce_cooldown_ticks: 0,
        cell_id,
        // Sprint 66: child startuje bez bondů (čistý slate). Bondy se vytvoří
        // podle vlastního chování dítěte, neinheritují se po rodičích.
        bonds: [None; MAX_BONDS_PER_CELL],
        // Sprint 80: cell_state se DĚDÍ (mid-parent + uniform noise σ ≈
        // CELL_STATE_INHERIT_NOISE), na rozdíl od bondů. Tím vzniká
        // fenotypová paměť přes generace bez genetické změny — lineage
        // může držet altruist nebo selfish režim, dokud noise / drift
        // attractor nepřevrátí. Append na konci struct literálu zachovává
        // pre-Sprint-80 RNG draw order.
        cell_state: ((parent_a.cell_state + parent_b.cell_state) * 0.5
            + rng.random_range(-CELL_STATE_INHERIT_NOISE..CELL_STATE_INHERIT_NOISE))
            .clamp(0.0, 1.0),
        phenotype: child_phenotype,
        genome: child_genome,
    }
}

/// Sprint 66: differential-adhesion kernel pro jeden pár (i, j), aplikuje
/// se ze strany i. Vrací `[Δvx, Δvy, Δvz]` přírůstek na velocity_i (před
/// vynásobením `dt`). Same-type → soft attraction (positive coefficient,
/// pulls i toward j). Cross-type → mírná repulze (negative). Zapojí se až
/// **mimo** kontakt (d > pair_r), takže nekoliduje s collision depenetration.
/// Force shape: linearní falloff `(1 - d/R)`, kde R = `ADHESION_RANGE_FACTOR
/// × pair_r`. Mimo R → 0, takže není potřeba další distance gate v hot loop.
///
/// Vstup `delta_ji` je `pos_i - pos_j` (toroidal min-imaged); `dist`
/// je jeho délka (caller už spočítal). `pair_r` je kontaktní vzdálenost
/// (CELL_RADIUS × (radius_i + radius_j)). `same_type` rozlišuje cadherin
/// kompatibilitu.
pub fn adhesion_velocity_delta(
    delta_ji: [f32; 3],
    dist: f32,
    pair_r: f32,
    same_type: bool,
) -> [f32; 3] {
    if dist <= pair_r || dist <= 0.0 {
        return [0.0; 3];
    }
    let range = pair_r * ADHESION_RANGE_FACTOR;
    if dist >= range {
        return [0.0; 3];
    }
    // Linear falloff: 1 at d=pair_r, 0 at d=range.
    let falloff = (range - dist) / (range - pair_r);
    // Coefficient: positive same-type pulls i toward j (negative along delta_ji
    // = pos_i - pos_j). Cross-type negative coefficient flips sign → push apart.
    let coeff = if same_type {
        ADHESION_STRENGTH
    } else {
        ADHESION_STRENGTH * ADHESION_CROSS_TYPE
    };
    let inv_d = 1.0 / dist;
    let nx = delta_ji[0] * inv_d;
    let ny = delta_ji[1] * inv_d;
    let nz = delta_ji[2] * inv_d;
    let mag = -coeff * falloff;
    [mag * nx, mag * ny, mag * nz]
}

/// Sprint 66: spring-bond force pro jeden bond (drží cell_i, ukazuje na j).
/// Vrací `(velocity_delta_i, broken)` — broken=true pokud se bond v tomto
/// ticku trhá (overstretch). Caller zodpovídá za clear bondu. Damping
/// aplikujeme na rel velocity podél spring osy → utlumí oscilace bez
/// over-damping (kritické pro stabilní tissue).
///
/// `delta_ji` = `pos_i - pos_j` (toroidal min-imaged), `dist` jeho délka,
/// `vel_i`, `vel_j` aktuální velocities (caller předal). Vrací delta NA
/// velocity_i, j strana ji aplikuje sama z vlastního Bond slotu (Newton
/// 3rd law symmetric).
pub fn bond_velocity_delta(
    bond: &Bond,
    delta_ji: [f32; 3],
    dist: f32,
    vel_i: [f32; 3],
    vel_j: [f32; 3],
) -> ([f32; 3], bool) {
    let break_len = bond.rest_length * BOND_BREAK_FACTOR;
    if dist > break_len || dist <= f32::EPSILON {
        return ([0.0; 3], true);
    }
    let inv_d = 1.0 / dist;
    let nx = delta_ji[0] * inv_d;
    let ny = delta_ji[1] * inv_d;
    let nz = delta_ji[2] * inv_d;
    // Spring: extension = dist - rest. Pozitivní = roztažení → pulls i toward j
    // (force along -n_ji, kde n_ji ukazuje od j k i). Negativní = stlačení →
    // pushes i away from j (force along +n_ji).
    let extension = dist - bond.rest_length;
    // Sprint 68: per-bond stiffness/damping (uložené při formaci jako mean
    // obou cells' genome values). BOND_STIFFNESS / BOND_DAMPING konstanty
    // jen pro initial draw v Genome::random.
    let spring = -bond.stiffness * extension;
    // Damping: relativní velocity podél normálu. v_rel = v_i - v_j; closing
    // pair má v_rel·n < 0 (pos_i přibližuje k pos_j). Damping force opacuje
    // relative motion → -bond.damping × v_rel_n × n.
    let v_rel_n = (vel_i[0] - vel_j[0]) * nx
        + (vel_i[1] - vel_j[1]) * ny
        + (vel_i[2] - vel_j[2]) * nz;
    let damp = -bond.damping * v_rel_n;
    let mag = spring + damp;
    ([mag * nx, mag * ny, mag * nz], false)
}

/// Sprint 53: 3D volumetric scalar field s explicit-Jacobi diffusion + decay.
/// Resolution per-axis (`[res_x, res_y, res_z]`) — typicky `[64, 64, 16]` aby
/// matchne aspect rátia tenkého z-sliceu (`world_half_z << world_half_xy`).
/// Grid layout: `idx = z*W*H + y*W + x`. 7-point stencil pro 3D Laplacian.
/// Stabilní při `diffusion < 1/6` (vs `< 1/4` v 2D — pre-Sprint-53 SmellField
/// měl 2D stencil). `SMELL_DIFFUSION = 0.15` zůstává pod oběma limity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmellField {
    pub resolution: [usize; 3],
    pub world_half: [f32; 3],
    grid: Vec<f32>,
    scratch: Vec<f32>,
}

impl SmellField {
    pub fn new(resolution: [usize; 3], world_half: [f32; 3]) -> Self {
        let n = resolution[0] * resolution[1] * resolution[2];
        Self {
            resolution,
            world_half,
            grid: vec![0.0; n],
            scratch: vec![0.0; n],
        }
    }

    fn cell_size(&self, axis: usize) -> f32 {
        (2.0 * self.world_half[axis]) / self.resolution[axis] as f32
    }

    /// Sprint 54: xy wrap (toroidal), z bounded. Mimo z-volume vrací `None`;
    /// xy je vždy modulo zarovnaný do gridu.
    fn idx_of(&self, pos: [f32; 3]) -> Option<usize> {
        let cs_x = self.cell_size(0);
        let cs_y = self.cell_size(1);
        let cs_z = self.cell_size(2);
        let nx = self.resolution[0] as i32;
        let ny = self.resolution[1] as i32;
        let nz = self.resolution[2] as i32;
        let xi = ((pos[0] + self.world_half[0]) / cs_x).floor() as i32;
        let yi = ((pos[1] + self.world_half[1]) / cs_y).floor() as i32;
        let zi = ((pos[2] + self.world_half[2]) / cs_z).floor() as i32;
        if zi < 0 || zi >= nz {
            return None;
        }
        let xi_mod = xi.rem_euclid(nx) as usize;
        let yi_mod = yi.rem_euclid(ny) as usize;
        let nx_us = self.resolution[0];
        let ny_us = self.resolution[1];
        Some((zi as usize) * nx_us * ny_us + yi_mod * nx_us + xi_mod)
    }

    pub fn add_source(&mut self, pos: [f32; 3], amount: f32) {
        if let Some(idx) = self.idx_of(pos) {
            self.grid[idx] += amount;
        }
    }

    /// 7-point Jacobi stencil + multiplicative decay. Sprint 54: toroidal v
    /// xy (left at i=0 čte sloupec i=nx-1, atd.), Neumann zero-flux na z
    /// (z=0 a z=nz-1 fallback na center — odpovídá ground/ceiling, ne wrap).
    /// Stable pro `diffusion < 1/6`.
    ///
    /// Sprint 57: paralelizováno přes z-roviny — každá rovina čte své okolí
    /// (xy stencil + back/front z grid) a zapisuje pouze do své části scratch,
    /// takže žádný write conflict. Pro 12-core CPU + 16 rovin je load balanced.
    ///
    /// Sprint 117: SIMD inner loop přes `wide::f32x8`. Per row (k, j) si
    /// pre-extract row offsets pro center/up/down/back/front (back/front s
    /// Neumann fallback na current plane), pak SIMD chunks po 8 buňkách na
    /// interior `i ∈ [1, nx-9]` (8 lanes × 7 chunks = 56 cells s nx=64).
    /// Boundary cells `i=0` a `i ∈ [nx-7, nx-1]` (8 z 64) scalar fallback —
    /// jediná místa, kde left/right wrap přes x-boundary. Sequential adds
    /// `(((l+r)+u)+d)+b)+f` → bit-identical s pre-S117 scalar verzí (žádný
    /// reduce_add); FP drift jen pokud nx<9.
    pub fn step(&mut self, diffusion: f32, decay_per_sec: f32, dt: f32) {
        use wide::f32x8;
        let nx = self.resolution[0];
        let ny = self.resolution[1];
        let nz = self.resolution[2];
        let decay = (1.0 - decay_per_sec * dt).max(0.0);
        let plane = nx * ny;
        let grid = &self.grid;
        let diffusion_v = f32x8::splat(diffusion);
        let decay_v = f32x8::splat(decay);
        let six_v = f32x8::splat(6.0);
        // SIMD pokrývá `i ∈ [1, simd_end)`, kde simd_end je největší
        // násobek 8 + 1 takový, že i+7 ≤ nx-2 (right read at i+8 ≤ nx-1).
        // Pro nx=64: simd_end = 1 + 7*8 = 57 → chunky i = 1, 9, …, 49.
        let simd_end = if nx >= 9 {
            1 + ((nx - 9) / 8 + 1) * 8
        } else {
            1
        };
        self.scratch
            .par_chunks_mut(plane)
            .enumerate()
            .for_each(|(k, scratch_plane)| {
                let center_plane = k * plane;
                let back_plane = if k > 0 { (k - 1) * plane } else { center_plane };
                let front_plane = if k + 1 < nz {
                    (k + 1) * plane
                } else {
                    center_plane
                };
                for j in 0..ny {
                    let j_up = if j == 0 { ny - 1 } else { j - 1 };
                    let j_down = if j + 1 == ny { 0 } else { j + 1 };
                    let center_row = center_plane + j * nx;
                    let up_row = center_plane + j_up * nx;
                    let down_row = center_plane + j_down * nx;
                    let back_row = back_plane + j * nx;
                    let front_row = front_plane + j * nx;
                    let scalar_cell = |i: usize| -> f32 {
                        let i_left = if i == 0 { nx - 1 } else { i - 1 };
                        let i_right = if i + 1 == nx { 0 } else { i + 1 };
                        let center = grid[center_row + i];
                        let left = grid[center_row + i_left];
                        let right = grid[center_row + i_right];
                        let up = grid[up_row + i];
                        let down = grid[down_row + i];
                        let back = grid[back_row + i];
                        let front = grid[front_row + i];
                        let new = center
                            + diffusion
                                * (left + right + up + down + back + front - 6.0 * center);
                        new * decay
                    };
                    scratch_plane[j * nx] = scalar_cell(0);
                    let mut i = 1;
                    while i < simd_end {
                        let center = f32x8::new(
                            grid[center_row + i..center_row + i + 8].try_into().unwrap(),
                        );
                        let left = f32x8::new(
                            grid[center_row + i - 1..center_row + i + 7]
                                .try_into()
                                .unwrap(),
                        );
                        let right = f32x8::new(
                            grid[center_row + i + 1..center_row + i + 9]
                                .try_into()
                                .unwrap(),
                        );
                        let up = f32x8::new(
                            grid[up_row + i..up_row + i + 8].try_into().unwrap(),
                        );
                        let down = f32x8::new(
                            grid[down_row + i..down_row + i + 8].try_into().unwrap(),
                        );
                        let back = f32x8::new(
                            grid[back_row + i..back_row + i + 8].try_into().unwrap(),
                        );
                        let front = f32x8::new(
                            grid[front_row + i..front_row + i + 8].try_into().unwrap(),
                        );
                        let mut acc = left + right;
                        acc += up;
                        acc += down;
                        acc += back;
                        acc += front;
                        acc -= six_v * center;
                        let new = (center + diffusion_v * acc) * decay_v;
                        let arr: [f32; 8] = new.into();
                        scratch_plane[j * nx + i..j * nx + i + 8].copy_from_slice(&arr);
                        i += 8;
                    }
                    while i < nx {
                        scratch_plane[j * nx + i] = scalar_cell(i);
                        i += 1;
                    }
                }
            });
        std::mem::swap(&mut self.grid, &mut self.scratch);
    }

    pub fn sample(&self, pos: [f32; 3]) -> f32 {
        self.idx_of(pos).map(|i| self.grid[i]).unwrap_or(0.0)
    }

    pub fn grid_ref(&self) -> &[f32] {
        &self.grid
    }

    /// Sprint 59: replace grid contents from external source (GPU readback).
    /// Used for FieldGpu wire-up — GPU computes diffuse+deposit, downloads
    /// snapshot, CPU SmellField holds it pro sensor gather (`gradient_at` + `sample`).
    pub fn replace_grid_from(&mut self, data: &[f32]) {
        debug_assert_eq!(data.len(), self.grid.len());
        self.grid.copy_from_slice(data);
    }

    /// 3D central differences at `pos ± epsilon` along each axis. Returns
    /// `[d/dx, d/dy, d/dz]`. Out-of-bounds samples count as 0.
    pub fn gradient_at(&self, pos: [f32; 3], epsilon: f32) -> [f32; 3] {
        let f_xp = self.sample([pos[0] + epsilon, pos[1], pos[2]]);
        let f_xm = self.sample([pos[0] - epsilon, pos[1], pos[2]]);
        let f_yp = self.sample([pos[0], pos[1] + epsilon, pos[2]]);
        let f_ym = self.sample([pos[0], pos[1] - epsilon, pos[2]]);
        let f_zp = self.sample([pos[0], pos[1], pos[2] + epsilon]);
        let f_zm = self.sample([pos[0], pos[1], pos[2] - epsilon]);
        let inv = 1.0 / (2.0 * epsilon);
        [
            (f_xp - f_xm) * inv,
            (f_yp - f_ym) * inv,
            (f_zp - f_zm) * inv,
        ]
    }
}

/// Sprint 53: deterministic 3D volumetric scalar field. `resolution` per axis,
/// hodnoty v `[0, 1]` z value-noise: `base_resolution³` random uniform grid,
/// smoothstep trilinear interp do plné resolution. Generováno jednou při
/// startu, pak jen čtení — žádný update per tick.
///
/// Use case: prostorová modulace mechaniky, která má být 3D-nehomogenní —
/// food_richness (xy projekce stačí pro food spawn floor), hazard (3D field
/// pro vertikální hazard layers), terrain drag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldMap {
    pub resolution: [usize; 3],
    pub world_half: [f32; 3],
    field: Vec<f32>,
}

impl WorldMap {
    pub fn new(
        resolution: [usize; 3],
        base_resolution: [usize; 3],
        world_half: [f32; 3],
        seed: u64,
    ) -> Self {
        assert!(
            resolution.iter().all(|&r| r >= 2) && base_resolution.iter().all(|&r| r >= 2)
        );
        let mut rng = StdRng::seed_from_u64(seed);
        let base_n = base_resolution[0] * base_resolution[1] * base_resolution[2];
        let base: Vec<f32> = (0..base_n).map(|_| rng.random()).collect();

        let nx = resolution[0];
        let ny = resolution[1];
        let nz = resolution[2];
        let bnx = base_resolution[0];
        let bny = base_resolution[1];
        let bnz = base_resolution[2];
        let scale_x = (bnx as f32 - 1.0) / nx as f32;
        let scale_y = (bny as f32 - 1.0) / ny as f32;
        let scale_z = (bnz as f32 - 1.0) / nz as f32;

        let mut field = vec![0.0_f32; nx * ny * nz];
        for k in 0..nz {
            let w = (k as f32 + 0.5) * scale_z;
            let z0 = (w.floor() as usize).min(bnz - 1);
            let z1 = (z0 + 1).min(bnz - 1);
            let fz = (w - z0 as f32).clamp(0.0, 1.0);
            let sz = fz * fz * (3.0 - 2.0 * fz);
            for j in 0..ny {
                let v = (j as f32 + 0.5) * scale_y;
                let y0 = (v.floor() as usize).min(bny - 1);
                let y1 = (y0 + 1).min(bny - 1);
                let fy = (v - y0 as f32).clamp(0.0, 1.0);
                let sy = fy * fy * (3.0 - 2.0 * fy);
                for i in 0..nx {
                    let u = (i as f32 + 0.5) * scale_x;
                    let x0 = (u.floor() as usize).min(bnx - 1);
                    let x1 = (x0 + 1).min(bnx - 1);
                    let fx = (u - x0 as f32).clamp(0.0, 1.0);
                    let sx = fx * fx * (3.0 - 2.0 * fx);
                    // Trilinear interp s smoothstep blend.
                    let i000 = base[z0 * bnx * bny + y0 * bnx + x0];
                    let i100 = base[z0 * bnx * bny + y0 * bnx + x1];
                    let i010 = base[z0 * bnx * bny + y1 * bnx + x0];
                    let i110 = base[z0 * bnx * bny + y1 * bnx + x1];
                    let i001 = base[z1 * bnx * bny + y0 * bnx + x0];
                    let i101 = base[z1 * bnx * bny + y0 * bnx + x1];
                    let i011 = base[z1 * bnx * bny + y1 * bnx + x0];
                    let i111 = base[z1 * bnx * bny + y1 * bnx + x1];
                    let c00 = i000 * (1.0 - sx) + i100 * sx;
                    let c10 = i010 * (1.0 - sx) + i110 * sx;
                    let c01 = i001 * (1.0 - sx) + i101 * sx;
                    let c11 = i011 * (1.0 - sx) + i111 * sx;
                    let c0 = c00 * (1.0 - sy) + c10 * sy;
                    let c1 = c01 * (1.0 - sy) + c11 * sy;
                    field[k * nx * ny + j * nx + i] = c0 * (1.0 - sz) + c1 * sz;
                }
            }
        }

        Self {
            resolution,
            world_half,
            field,
        }
    }

    /// Sprint 54: xy wrap (toroidal), z clamp (bounded).
    pub fn sample(&self, pos: [f32; 3]) -> f32 {
        let nx = self.resolution[0];
        let ny = self.resolution[1];
        let nz = self.resolution[2];
        let cs_x = (2.0 * self.world_half[0]) / nx as f32;
        let cs_y = (2.0 * self.world_half[1]) / ny as f32;
        let cs_z = (2.0 * self.world_half[2]) / nz as f32;
        let xi = ((pos[0] + self.world_half[0]) / cs_x).floor() as i32;
        let yi = ((pos[1] + self.world_half[1]) / cs_y).floor() as i32;
        let zi = ((pos[2] + self.world_half[2]) / cs_z).floor() as i32;
        let xi = xi.rem_euclid(nx as i32) as usize;
        let yi = yi.rem_euclid(ny as i32) as usize;
        let zi = zi.clamp(0, nz as i32 - 1) as usize;
        self.field[zi * nx * ny + yi * nx + xi]
    }

    pub fn field(&self) -> &[f32] {
        &self.field
    }
}

/// Sprint 54: minimum-image displacement na toroidal xy + bounded z.
/// Vrátí signed delta `b - a` adjustnuté tak, že |dx|, |dy| ≤ `world_half`,
/// dz beze změny (z-osa není wrapped — gravita + food sink + carrion drop
/// vyžadují pevný strop/dno).
///
/// Pro toroidal world: dva body na opačných koncích světa (např. x=−950 a
/// x=+950 při half=960) jsou minimum-image-blízké (Δx=20, ne 1900).
pub fn min_image_delta(a: [f32; 3], b: [f32; 3], world_half: [f32; 3]) -> [f32; 3] {
    let mut dx = b[0] - a[0];
    let mut dy = b[1] - a[1];
    let dz = b[2] - a[2];
    let wx = 2.0 * world_half[0];
    let wy = 2.0 * world_half[1];
    if dx > world_half[0] {
        dx -= wx;
    } else if dx < -world_half[0] {
        dx += wx;
    }
    if dy > world_half[1] {
        dy -= wy;
    } else if dy < -world_half[1] {
        dy += wy;
    }
    [dx, dy, dz]
}

/// Sprint 54: wrap pos.xy do `[-half, half)` (toroidal). z se nepojí.
pub fn wrap_position_xy(pos: [f32; 3], world_half: [f32; 3]) -> [f32; 3] {
    let wx = 2.0 * world_half[0];
    let wy = 2.0 * world_half[1];
    let mut x = pos[0];
    let mut y = pos[1];
    while x >= world_half[0] {
        x -= wx;
    }
    while x < -world_half[0] {
        x += wx;
    }
    while y >= world_half[1] {
        y -= wy;
    }
    while y < -world_half[1] {
        y += wy;
    }
    [x, y, pos[2]]
}

/// Sprint 43: 3D uniform spatial hash. Generic přes `Id` (Bevy `Entity` v
/// rendereru, `usize` v headless) a `P` (per-item payload, např. radius).
///
/// **Determinismus:** rebuild iteruje vstup v pořadí, ve kterém přijde, a Vec
/// v každém bucketu drží push-order. `for_each_in_radius` iteruje 3³ buckets ve
/// fixním (dx, dy, dz) pořadí; `HashMap::get(&key)` je lookup-deterministic.
/// Caller, který předá rebuild items ve stable order (např. `cells.iter().enumerate()`),
/// dostane reprodukovatelný traversal napříč runy. Floats z následných sumací
/// nejsou bit-identical s O(N²) baseline kvůli jinému pořadí akumulace.
pub struct SpatialGrid<Id: Copy + Eq + Hash, P: Copy> {
    cell_size: f32,
    buckets: FxHashMap<(i32, i32, i32), Vec<(Id, [f32; 3], P)>>,
}

impl<Id: Copy + Eq + Hash, P: Copy> SpatialGrid<Id, P> {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            buckets: FxHashMap::default(),
        }
    }

    fn key_of(&self, pos: [f32; 3]) -> (i32, i32, i32) {
        (
            (pos[0] / self.cell_size).floor() as i32,
            (pos[1] / self.cell_size).floor() as i32,
            (pos[2] / self.cell_size).floor() as i32,
        )
    }

    /// Drops stale entries z předchozího rebuildu, ale zachová bucket Vec
    /// kapacity — populace je per-tick relativně stabilní, takže reuse alokace
    /// vyhrává nad clear() celé HashMap.
    pub fn rebuild<I: IntoIterator<Item = (Id, [f32; 3], P)>>(&mut self, items: I) {
        for bucket in self.buckets.values_mut() {
            bucket.clear();
        }
        for (id, pos, payload) in items {
            let key = self.key_of(pos);
            self.buckets.entry(key).or_default().push((id, pos, payload));
        }
    }

    /// Volá `f(id, pos, payload)` pro každý item v 3³ buckets okolo `pos`.
    /// Caller musí narrow-phase distance test dělat sám (grid vrací overestimate).
    pub fn for_each_in_radius<F: FnMut(Id, [f32; 3], P)>(
        &self,
        pos: [f32; 3],
        radius: f32,
        mut f: F,
    ) {
        let r_cells = (radius / self.cell_size).ceil() as i32;
        let (cx, cy, cz) = self.key_of(pos);
        for dx in -r_cells..=r_cells {
            for dy in -r_cells..=r_cells {
                for dz in -r_cells..=r_cells {
                    if let Some(bucket) = self.buckets.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &(id, p, payload) in bucket {
                            f(id, p, payload);
                        }
                    }
                }
            }
        }
    }

    /// Sprint 54: toroidal-aware query přes ghost positions. Pokud je `pos`
    /// blízko xy-boundary (do `radius`), vyšleme dodatečné lookup queries do
    /// "ghost" pozic na opačné straně světa. Z není wrapped (cylinder topology).
    /// Stejný `f` callback se může volat na duplicate items pokud je radius
    /// > world_half — caller musí narrow-phase použít `min_image_delta` aby
    /// duplicates filtroval.
    pub fn for_each_in_radius_toroidal<F: FnMut(Id, [f32; 3], P)>(
        &self,
        pos: [f32; 3],
        radius: f32,
        world_half: [f32; 3],
        mut f: F,
    ) {
        // Center query.
        self.for_each_in_radius(pos, radius, &mut f);
        let wx = 2.0 * world_half[0];
        let wy = 2.0 * world_half[1];
        let near_left = pos[0] < -world_half[0] + radius;
        let near_right = pos[0] > world_half[0] - radius;
        let near_bot = pos[1] < -world_half[1] + radius;
        let near_top = pos[1] > world_half[1] - radius;
        // Edges (4 ghost positions).
        if near_left {
            self.for_each_in_radius([pos[0] + wx, pos[1], pos[2]], radius, &mut f);
        }
        if near_right {
            self.for_each_in_radius([pos[0] - wx, pos[1], pos[2]], radius, &mut f);
        }
        if near_bot {
            self.for_each_in_radius([pos[0], pos[1] + wy, pos[2]], radius, &mut f);
        }
        if near_top {
            self.for_each_in_radius([pos[0], pos[1] - wy, pos[2]], radius, &mut f);
        }
        // Corners (4 ghost positions).
        if near_left && near_bot {
            self.for_each_in_radius([pos[0] + wx, pos[1] + wy, pos[2]], radius, &mut f);
        }
        if near_left && near_top {
            self.for_each_in_radius([pos[0] + wx, pos[1] - wy, pos[2]], radius, &mut f);
        }
        if near_right && near_bot {
            self.for_each_in_radius([pos[0] - wx, pos[1] + wy, pos[2]], radius, &mut f);
        }
        if near_right && near_top {
            self.for_each_in_radius([pos[0] - wx, pos[1] - wy, pos[2]], radius, &mut f);
        }
    }
}

/// Sprint 43: defaultní velikost buňky spatial gridu. ~1.3× max vision_radius
/// (50). Větší = méně buckets, víc kandidátů per query; menší = víc buckets,
/// méně kandidátů. Renderer v `main.rs` má svůj vlastní knob.
pub const GRID_CELL_SIZE: f32 = 64.0;

/// Sprint 102: hunter cell-grid bucket size. Hunter vision_radius je řádově
/// větší než typická cell-cell interakce (100–400 vs ~20), takže `GRID_CELL_SIZE
/// = 64` by dělalo r_cells = 5–7 → 1300+ HashMap lookupů per query a grid
/// by byl pomalejší než brute force při běžné populaci. 200 odpovídá median
/// hunter vision → r_cells = 1–2 → 27–125 lookupů.
pub const HUNTER_GRID_CELL_SIZE: f32 = 200.0;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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

/// Sprint 108: seed-namespace pro shock RNG. Hash s world seedem zajišťuje
/// nezávislý stream — měnit shock plán nezmění RNG cellí logiky.
pub const SHOCK_SCHEDULE_SALT: u64 = 0xCAFE_F00D;

/// Sprint 108: počet ShockKind variant. Drží sync s `ShockKind` enum size.
/// Pokud přidáš variant, bumpni a uprav `ShockScheduleConfig.type_weights`.
pub const SHOCK_KIND_COUNT: usize = 3;

/// Sprint 108: typy environmentálních shocků. Diskretní eventy s rampou
/// (ne smooth cykly) — drží selekční tlak v dlouhých runech.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ShockKind {
    HazardPulse,
    ClimateShift,
    FoodCrash,
}

/// Sprint 108: jeden shock event v kalendáři. Aktivní v generačním okně
/// `[start_gen, start_gen + duration_gen)`; rampa řízená `ramp_gens`.
/// `center_xy`/`radius` `None` znamená globální dosah.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ShockEvent {
    pub kind: ShockKind,
    pub start_gen: u64,
    pub duration_gen: u32,
    pub ramp_gens: u32,
    pub intensity: f32,
    pub center_xy: Option<[f32; 2]>,
    pub radius: Option<f32>,
}

/// Sprint 108: parametry plánovače shocků. `mean_gens_between == 0`
/// znamená no-op (default) — kalendář bude prázdný a integrace v Sprint 109+
/// nemá efekt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShockScheduleConfig {
    pub mean_gens_between: u32,
    pub type_weights: [f32; SHOCK_KIND_COUNT],
    pub intensity_min: f32,
    pub intensity_max: f32,
    pub duration_min_gens: u32,
    pub duration_max_gens: u32,
    pub ramp_gens: u32,
    pub spatial_global_prob: f32,
    pub spatial_radius_min_frac: f32,
    pub spatial_radius_max_frac: f32,
}

impl Default for ShockScheduleConfig {
    fn default() -> Self {
        Self {
            mean_gens_between: 0,
            type_weights: [1.0, 1.0, 1.0],
            intensity_min: 0.3,
            intensity_max: 1.0,
            duration_min_gens: 5,
            duration_max_gens: 15,
            ramp_gens: 2,
            spatial_global_prob: 0.5,
            spatial_radius_min_frac: 0.2,
            spatial_radius_max_frac: 0.6,
        }
    }
}

/// Sprint 108: deterministicky vygenerovaný kalendář shocků pro celý run.
/// Drží i `seed`, ze kterého byl odvozen — pro reproducibility checks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventCalendar {
    pub events: Vec<ShockEvent>,
    pub seed: u64,
}

impl EventCalendar {
    /// Pokud `cfg.mean_gens_between == 0`, vrací prázdný kalendář (no-op).
    /// Jinak deterministicky generuje sekvenci shocků až do `max_gens`
    /// přes Poisson-like inter-arrival times s mean `mean_gens_between`.
    /// Použije `StdRng::seed_from_u64(seed ^ SHOCK_SCHEDULE_SALT)`.
    /// Eventy jsou setříděné vzestupně podle `start_gen`.
    pub fn generate(seed: u64, cfg: &ShockScheduleConfig, max_gens: u64) -> Self {
        let mut calendar = Self {
            events: Vec::new(),
            seed,
        };
        if cfg.mean_gens_between == 0 || max_gens == 0 {
            return calendar;
        }
        let mut rng = StdRng::seed_from_u64(seed ^ SHOCK_SCHEDULE_SALT);
        let mean = cfg.mean_gens_between as f32;
        let intensity_lo = cfg.intensity_min.min(cfg.intensity_max);
        let intensity_hi = cfg.intensity_min.max(cfg.intensity_max);
        let duration_lo = cfg.duration_min_gens.min(cfg.duration_max_gens);
        let duration_hi = cfg.duration_min_gens.max(cfg.duration_max_gens);
        let radius_lo = cfg
            .spatial_radius_min_frac
            .min(cfg.spatial_radius_max_frac)
            .max(0.0);
        let radius_hi = cfg
            .spatial_radius_min_frac
            .max(cfg.spatial_radius_max_frac)
            .max(radius_lo);
        let world_half_xy = WORLD_HALF[0];

        let mut next_start: u64 = 0;
        loop {
            let u: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE);
            let gap_f = (mean * -u.ln()).max(1.0);
            let gap = gap_f as u64;
            let gap = gap.max(1);
            next_start = next_start.saturating_add(gap);
            if next_start >= max_gens {
                break;
            }

            let kind = pick_shock_kind(&mut rng, &cfg.type_weights);
            let intensity = if intensity_hi > intensity_lo {
                rng.random_range(intensity_lo..=intensity_hi)
            } else {
                intensity_lo
            };
            let duration_gen = if duration_hi > duration_lo {
                rng.random_range(duration_lo..=duration_hi)
            } else {
                duration_lo
            };

            let global_roll: f32 = rng.random();
            let (center_xy, radius) = if global_roll < cfg.spatial_global_prob {
                (None, None)
            } else {
                let cx = rng.random_range(-1.0_f32..=1.0) * world_half_xy;
                let cy = rng.random_range(-1.0_f32..=1.0) * world_half_xy;
                let frac = if radius_hi > radius_lo {
                    rng.random_range(radius_lo..=radius_hi)
                } else {
                    radius_lo
                };
                let r = (frac * world_half_xy).max(0.0);
                (Some([cx, cy]), Some(r))
            };

            calendar.events.push(ShockEvent {
                kind,
                start_gen: next_start,
                duration_gen,
                ramp_gens: cfg.ramp_gens,
                intensity,
                center_xy,
                radius,
            });
        }

        calendar.events.sort_by_key(|e| e.start_gen);
        calendar
    }

    /// Sprint 108: shock je aktivní v generačním okně `[start, start + duration)`.
    /// `tick` je ignorován — rampa pracuje v gen units, aby byla nezávislá na
    /// `FIXED_TIMESTEP_HZ`. Signature ho drží pro budoucí tick-level shocks.
    pub fn active(&self, generation: u64, _tick: u64) -> impl Iterator<Item = &ShockEvent> {
        self.events.iter().filter(move |e| {
            let end = e.start_gen.saturating_add(e.duration_gen as u64);
            generation >= e.start_gen && generation < end
        })
    }
}

fn pick_shock_kind(rng: &mut StdRng, weights: &[f32; SHOCK_KIND_COUNT]) -> ShockKind {
    let total: f32 = weights.iter().map(|w| w.max(0.0)).sum();
    if total <= 0.0 {
        return ShockKind::HazardPulse;
    }
    let mut roll = rng.random::<f32>() * total;
    for (i, &w) in weights.iter().enumerate() {
        let w = w.max(0.0);
        if roll < w {
            return match i {
                0 => ShockKind::HazardPulse,
                1 => ShockKind::ClimateShift,
                _ => ShockKind::FoodCrash,
            };
        }
        roll -= w;
    }
    ShockKind::FoodCrash
}

/// Sprint 108: trapezoid (nebo triangle pokud `duration <= 2 * ramp_gens`)
/// envelope shocku. Outside `[start, start + duration)` vrací 0.0; uvnitř
/// 0..=1. Rampa v gen units, ne v sekundách — `FIXED_TIMESTEP_HZ` ji nemění.
pub fn shock_ramp_factor(event: &ShockEvent, generation: u64) -> f32 {
    let duration = event.duration_gen as u64;
    if duration == 0 || generation < event.start_gen {
        return 0.0;
    }
    let end = event.start_gen + duration;
    if generation >= end {
        return 0.0;
    }
    let local = generation - event.start_gen;
    let ramp = event.ramp_gens as u64;

    if duration <= ramp.saturating_mul(2) || ramp == 0 {
        let half = duration as f32 / 2.0;
        if half <= 0.0 {
            return 0.0;
        }
        let dist_from_mid = (local as f32 + 0.5 - half).abs();
        let f = 1.0 - (dist_from_mid / half);
        return f.clamp(0.0, 1.0);
    }

    let plateau_start = ramp;
    let plateau_end = duration - ramp;
    if local < plateau_start {
        let f = (local as f32 + 0.5) / ramp as f32;
        f.clamp(0.0, 1.0)
    } else if local < plateau_end {
        1.0
    } else {
        let into_down = local - plateau_end;
        let f = 1.0 - (into_down as f32 + 0.5) / ramp as f32;
        f.clamp(0.0, 1.0)
    }
}

/// Sprint 110: max bonus k drainu při peak intensity. drain_factor = 1.0 +
/// intensity × ramp × HAZARD_PULSE_MAX_MULTIPLIER_BONUS. Při intensity=1 a
/// peak ramp = 1.0 → drain × 2.0.
pub const HAZARD_PULSE_MAX_MULTIPLIER_BONUS: f32 = 1.0;

/// Sprint 112: max temperature offset (°C) per ClimateShift při peak intensity
/// a full spatial mask. Default direction = warming (signed positive).
/// Peak case: intensity=1, ramp=1, mask=1 → +5°C nad baseline `temperature_at_z`.
pub const CLIMATE_SHIFT_MAX_OFFSET: f32 = 5.0;

/// Sprint 110: multiplikátor hazard drainu na pozici `pos` při dané `(gen, tick)`.
/// Default 1.0 (žádný HazardPulse aktivní). Pro každý active HazardPulse:
/// `1.0 + intensity × ramp_factor × spatial_mask × HAZARD_PULSE_MAX_MULTIPLIER_BONUS`.
/// Multiplicative compound přes všechny aktivní pulsy. Spatial mask je
/// smoothstep falloff od center v xy (z se ignoruje — hazard je vertikálně
/// uniformní), toroidal-aware přes `min_image_delta`. Pure fn, deterministic.
pub fn hazard_shock_multiplier(
    pos: [f32; 3],
    events: &[ShockEvent],
    generation: u64,
    tick: u64,
    world_half: [f32; 3],
) -> f32 {
    let _ = tick;
    let mut multiplier = 1.0_f32;
    for event in events {
        if event.kind != ShockKind::HazardPulse {
            continue;
        }
        let ramp = shock_ramp_factor(event, generation);
        if ramp <= 0.0 {
            continue;
        }
        let mask = match (event.center_xy, event.radius) {
            (Some(center), Some(radius)) if radius > 0.0 => {
                let center3 = [center[0], center[1], pos[2]];
                let d_vec = min_image_delta(center3, pos, world_half);
                let dist_xy = (d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1]).sqrt();
                if dist_xy >= radius {
                    0.0
                } else {
                    let t = (1.0 - dist_xy / radius).clamp(0.0, 1.0);
                    t * t * (3.0 - 2.0 * t)
                }
            }
            _ => 1.0,
        };
        if mask <= 0.0 {
            continue;
        }
        multiplier *= 1.0 + event.intensity * ramp * mask * HAZARD_PULSE_MAX_MULTIPLIER_BONUS;
    }
    multiplier
}

/// Sprint 112: signed temperature offset (°C) z ClimateShift shocků pro pozici
/// `pos_xy`. Default 0.0 (žádný ClimateShift aktivní). Pro každý active event:
/// `intensity × ramp_factor × spatial_mask × CLIMATE_SHIFT_MAX_OFFSET`.
/// Spatial mask je smoothstep falloff přes xy plane (toroidal-aware), 1.0 pro
/// global eventy bez center. Sčítá additivně přes všechny aktivní eventy
/// (warming je positive — cooling by potřeboval per-event signed intensity,
/// budoucí extension). Pure fn, deterministic.
pub fn climate_shock_offset(
    events: &[ShockEvent],
    generation: u64,
    pos_xy: [f32; 2],
    world_half: [f32; 3],
) -> f32 {
    let mut total = 0.0_f32;
    for event in events {
        if event.kind != ShockKind::ClimateShift {
            continue;
        }
        let ramp = shock_ramp_factor(event, generation);
        if ramp <= 0.0 {
            continue;
        }
        let mask = match (event.center_xy, event.radius) {
            (Some(center), Some(radius)) if radius > 0.0 => {
                let a = [pos_xy[0], pos_xy[1], 0.0];
                let b = [center[0], center[1], 0.0];
                let d_vec = min_image_delta(a, b, world_half);
                let dist_xy = (d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1]).sqrt();
                if dist_xy >= radius {
                    0.0
                } else {
                    let t = (1.0 - dist_xy / radius).clamp(0.0, 1.0);
                    t * t * (3.0 - 2.0 * t)
                }
            }
            _ => 1.0,
        };
        if mask <= 0.0 {
            continue;
        }
        total += event.intensity * ramp * mask * CLIMATE_SHIFT_MAX_OFFSET;
    }
    total
}

/// Sprint 113: max drop multiplikátoru při peak intensity. Při intensity=1,
/// peak ramp=1 → density_factor × 0.5 (= half food spawning).
pub const FOOD_CRASH_MAX_DROP: f32 = 0.5;

/// Sprint 113: hard floor pro density factor — i compound shocky nezpůsobí
/// úplný food collapse (extinction). 0.1 = 10% baseline = survival possible
/// pro adapted populace.
pub const FOOD_CRASH_MIN_FACTOR: f32 = 0.1;

/// Sprint 113: globální food density multiplikátor z aktivních FoodCrash shocků.
/// Default 1.0 (žádný FoodCrash aktivní). Pro každý active FoodCrash:
/// `multiplier *= 1.0 - intensity × ramp_factor × FOOD_CRASH_MAX_DROP`.
/// Multiplicative compound přes všechny active FoodCrash. Žádná spatial maska —
/// global per-tick scalar. Min clamp na `FOOD_CRASH_MIN_FACTOR` aby populace
/// měla šanci přežít. Pure fn, deterministic.
pub fn food_density_shock_multiplier(events: &[ShockEvent], generation: u64) -> f32 {
    let mut mult = 1.0_f32;
    for event in events {
        if event.kind != ShockKind::FoodCrash {
            continue;
        }
        let ramp = shock_ramp_factor(event, generation);
        if ramp <= 0.0 {
            continue;
        }
        mult *= 1.0 - event.intensity * ramp * FOOD_CRASH_MAX_DROP;
    }
    mult.max(FOOD_CRASH_MIN_FACTOR)
}

/// Sprint 112: shock-aware varianta `temperature_at_z`. K baseline gradientu
/// přičítá sumu ClimateShift offsetů. Empty events nebo žádný ClimateShift
/// aktivní → byte-identical s `temperature_at_z`. Renderer i headless volají
/// tuto wrapper variantu, pure `temperature_at_z` zůstává nedotčená pro testy
/// a backward-compat.
#[inline]
pub fn temperature_at_z_with_shocks(
    z: f32,
    world_half: [f32; 3],
    tick: u64,
    generation: u64,
    events: &[ShockEvent],
    pos_xy: [f32; 2],
) -> f32 {
    let base = temperature_at_z(z, world_half, tick, generation);
    base + climate_shock_offset(events, generation, pos_xy, world_half)
}

#[cfg(test)]
mod tests;
