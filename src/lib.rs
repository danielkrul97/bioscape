//! Bioscape simulation core — pure logic, no rendering.
//!
//! Keeping this layer free of Bevy types lets us drive the same world
//! from a windowed renderer (`main.rs`) or a headless batch run later.

use core::f32::consts::TAU;
use serde::{Deserialize, Serialize};

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

pub mod food;
pub use food::*;

pub mod reproduction;
pub use reproduction::*;

pub mod chemistry;
pub use chemistry::*;

pub mod world_map;
pub use world_map::*;

pub mod spatial;
pub use spatial::*;

pub mod clock;
pub use clock::*;

pub mod events;
pub use events::*;

pub mod physics_utils;
pub use physics_utils::*;

pub mod sensors;
pub use sensors::*;

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
/// Bond se trhá při current_length > rest_length × factor. 3.0 = 200 % strain
/// před break — víc tolerance kolizním impulse v dense clusterech (Sprint 132+
/// jemný bump z 2.5 pro stabilnější struktury), naturální oscilace stále uvnitř
/// safe zone díky damping.
pub const BOND_BREAK_FACTOR: f32 = 3.0;
/// Násobitel kontaktní vzdálenosti pro rest_length. 1.05 = mírný "buffer"
/// (kontakt drží trochu volněji než exact touching) → preventivní polštář
/// proti inicial overlap při formaci.
pub const BOND_REST_LENGTH_SLACK: f32 = 1.05;
/// Brain output[9] musí být ≥ tento threshold u OBOU buněk pro vznik bondu.
/// Mirror MATING_PHEROMONE_THRESHOLD / ATTACK_THRESHOLD semantiku.
pub const BOND_FORM_THRESHOLD: f32 = 0.0;
/// Brain output[9] < tento threshold u některé z bonded cells → bond se
/// explicit trhá tento tick. Negative = "pusť mě". Asymmetric: jeden silný
/// negativní signál stačí (escape behavior). Sprint 132+ bump z -0.5 na -1.0
/// — predchozí hodnota chytala náhodný brain noise (tanh distribuce přes 0
/// dávala ~17 % per-tick šanci spadnout pod -0.5), takže clustery se rozpadaly
/// dřív, než selekce stihla "naučit" bonding-positive brainy. -1.0 vyžaduje
/// silnější negative drive — break je vědomé, ne stochastické.
pub const BOND_BREAK_THRESHOLD: f32 = -1.0;
/// Energy cost při formaci bondu (one-shot, paid by initiator). Ne-trivial,
/// aby selekce váhala bonding vs free-roaming.
pub const BOND_FORMATION_COST: f32 = 0.2;
/// Per-second cost udržování každého bondu (paid každý tick). Drobný — bond
/// je výhoda (tissue stability), ale ne free. Tuned tak, aby cluster benefit
/// rostl relativně k maintenance cost při low food periods.
pub const BOND_MAINTENANCE_PER_SEC: f32 = 0.04;
/// Sprint 78: food-share fraction per bond. Když bonded cell eats food
/// (FOOD_VALUE energy), každý bonded partner dostane `FOOD_VALUE × FRAC`
/// extra energy (free reward, no energy conservation — modeluje „tissue
/// metabolic cooperation"). Cluster s 2 bondy: eater +FOOD_VALUE,
/// 2 partneři +0.6 × FOOD_VALUE = +12 each. Total cluster gain je
/// 1 + 2×0.3 = 1.6× větší než solo. Direct positive selection signál pro
/// bonding — fitness payoff přímo přes food share.
pub const BOND_FOOD_SHARE_FRAC: f32 = 1.3;
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


// Sdílené testovací fixtures. Není gated #[cfg(test)] aby je mohly importovat
// i binární testy (bin/headless/*_tests.rs) — bin se kompiluje proti lib bez
// vlastního cfg(test) flagu lib.
pub mod test_helpers;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_phase1;

#[cfg(test)]
mod tests_phase2;

#[cfg(test)]
mod tests_phase3;
