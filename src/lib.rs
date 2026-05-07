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

pub mod predator;
pub use predator::*;

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







#[cfg(test)]
mod tests;
