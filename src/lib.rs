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

pub mod predator;
pub use predator::*;

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
