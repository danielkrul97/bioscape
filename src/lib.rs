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
// Sprint 39 patch: 8 → 16 — větší hidden kapacita pro 3D + gravity. 28 inputs
// → 8 hidden bylo příliš stěsnaný "kompresní bottleneck" pro 3D navigaci.
// w1 z 28×8=224 na 36×16=576 weights (2.6×), brain hot-loop ~2× pomalejší.
pub const BRAIN_HIDDEN: usize = 16;
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
// Sprint 66: 9=bond_signal — pozitivní (>BOND_FORM_THRESHOLD) povoluje vznik
// spring bondů, silně negativní (<BOND_BREAK_THRESHOLD) trhá existující bondy.
// Indexy 0–8 zachované, jen append.
// Sprint 26 morph signals: signal × MORPH_RATE × dt přičteno k phenotype dim
// každý tick, energy cost ∝ |delta|. Sprint 27 attack: gating signál pro
// `predate` — bez aktivního output[6] > THRESHOLD se predace nestane.
pub const BRAIN_OUTPUTS: usize = 10;
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
/// Sprint 66 bond signal bias (b2[9]). Default 0 — bond formation se musí
/// objevit selekcí, ne jako prior. Stejná filozofie jako attack bias.
pub const INNATE_BOND_BIAS: f32 = 0.0;

// Shared sim parameters consumed by both the Bevy renderer (`src/main.rs`)
// and the headless harness (`src/bin/headless.rs`). Single source of truth —
// tune here. Renderer-only and headless-only knobs stay in their binaries.

pub const FIXED_TIMESTEP_HZ: f32 = 60.0;
pub const TICKS_PER_GENERATION: u64 = 600;
pub const GENERATIONS_PER_EPOCH: u64 = 100;

pub const INITIAL_CELLS: usize = 200;
/// Sprint 64: 1000 → 2500 (proportional s z=20 → z=50 expansion). Cells
/// density per volume zachovaná: pre-Sprint-64 1.2e-5 cells/unit³, post:
/// stejně. CPU paralelní cesta drží > 60 FPS (Sprint 63 5k = 870 ticks/s).
pub const MAX_POPULATION: usize = 2500;

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

/// Sprint 38: gravitační zrychlení (sim units / sec²) působící na cells.
/// Sprint 65: 5.0 → 0.0 (neutral buoyancy approximation). Pre-Sprint-65
/// vytvářelo selekční tlak směrem k „seď na dně" — cells postupně
/// sedimentovaly, akumulovaly se na floor reflective wall, vertikální
/// motion neměla evoluční benefit (úsilí plavat up = stejně sedneš dolů).
/// Po Sprintu 65 cell density == water density → vertikální motion je
/// 100 % brain-driven. Food sink (`FOOD_SINK_RATE`) zachován — food má
/// vyšší density než cells (benthic deposit semantika), cells musí
/// proaktivně dive za food.
pub const GRAVITY: f32 = 0.0;
/// Sprint 65: collision velocity damping. Restitution 0 = perfectly
/// inelastic — closing velocity podél separation normal je vynulovaná
/// (cells "stick" momentárně, pak se separují přes position depenetration).
/// 1.0 = elastic (perfect bounce). Soft biological cells = 0.0 default.
/// Pre-Sprint-65 cells měly delta_pos depenetration ale velocity neaffected
/// → po push-apart pokračovaly v closing motion → re-overlapped next tick
/// (oscilace + zbytečný compute).
pub const COLLISION_RESTITUTION: f32 = 0.0;
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
/// Sprint 53: z-axis resolution pro pheromone field. Tenčí z-volume + lower
/// res = větší cell_size_z (32 vs 64) → matchne thin world aspect a šetří
/// memory bez ztráty rozlišení v xy.
pub const PHEROMONE_GRID_RES_Z: usize = 16;
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
/// Sprint 53: smell field z-axis resolution. Same reasoning jako PHEROMONE_GRID_RES_Z.
pub const SMELL_GRID_RES_Z: usize = 16;
pub const SMELL_DIFFUSION: f32 = 0.15;
pub const SMELL_DECAY: f32 = 0.3;
pub const SMELL_PER_FOOD: f32 = 1.0;
pub const SMELL_SAMPLE_EPSILON: f32 = 10.0;
pub const SMELL_NORMALIZATION_GAIN: f32 = 0.5;

pub const LEARNING_RATE: f32 = 0.005;

pub const WORLD_MAP_RES: usize = 64;
/// Sprint 53: WorldMap z-axis resolution.
pub const WORLD_MAP_RES_Z: usize = 16;
pub const WORLD_MAP_BASE_RES: usize = 8;
/// Sprint 53: base z-axis resolution pro WorldMap. Lower base → smoother
/// vertical noise (less high-frequency variation v thin z-volume).
pub const WORLD_MAP_BASE_RES_Z: usize = 4;
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
// Sprint 40 cleanup: `SPIKE_RENDER_THRESHOLD` removed — Sprint 36 dropped
// custom spike shader, takže render-threshold const už není referencovaná.

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
/// vytvoří spring bond. 30 ticks = 0.5 s při 60 Hz — protekce proti single-tick
/// pasáži (random sblížení), ale ne tak dlouhá, aby selekce nestihla bonding
/// vyzkoušet v rámci generace (10 s).
pub const BOND_FORM_TICKS: u32 = 30;
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
pub const BOND_FORM_THRESHOLD: f32 = 0.2;
/// Brain output[9] < tento threshold u některé z bonded cells → bond se
/// explicit trhá tento tick. Negative = "pusť mě". Asymmetric: jeden silný
/// negativní signál stačí (escape behavior).
pub const BOND_BREAK_THRESHOLD: f32 = -0.5;
/// Energy cost při formaci bondu (one-shot, paid by initiator). Ne-trivial,
/// aby selekce váhala bonding vs free-roaming.
pub const BOND_FORMATION_COST: f32 = 0.5;
/// Per-second cost udržování každého bondu (paid každý tick). Drobný — bond
/// je výhoda (tissue stability), ale ne free.
pub const BOND_MAINTENANCE_PER_SEC: f32 = 0.1;
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
};
pub const PHYSICS_CONFIG: PhysicsConfig = PhysicsConfig {
    drag: DRAG_COEFFICIENT,
    angular_drag: ANGULAR_DRAG,
    energy_cost_per_v_sq: ENERGY_COST_PER_V_SQ,
    angular_energy_cost: ANGULAR_ENERGY_COST,
    vision_cost_per_radius: VISION_COST_PER_RADIUS,
    body_cost_factor: BODY_COST_FACTOR,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Brain {
    #[serde(with = "serde_arrays_w1")]
    pub w1: [[f32; BRAIN_INPUTS]; BRAIN_HIDDEN],
    pub b1: [f32; BRAIN_HIDDEN],
    #[serde(with = "serde_arrays_w2")]
    pub w2: [[f32; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
    pub b2: [f32; BRAIN_OUTPUTS],
}

// Sprint 48: serde 1 has native const-generic support pro `[T; N]` ale
// nested fixed arrays (`[[f32; 36]; 16]`) potřebují manual workaround.
// Encode jako flat Vec<f32> length × 36, decode reverse.
mod serde_arrays_w1 {
    use super::{BRAIN_HIDDEN, BRAIN_INPUTS};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(
        w: &[[f32; BRAIN_INPUTS]; BRAIN_HIDDEN],
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let flat: Vec<f32> = w.iter().flat_map(|row| row.iter().copied()).collect();
        flat.serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<[[f32; BRAIN_INPUTS]; BRAIN_HIDDEN], D::Error> {
        let flat: Vec<f32> = Vec::deserialize(d)?;
        if flat.len() != BRAIN_HIDDEN * BRAIN_INPUTS {
            return Err(serde::de::Error::custom("w1 length mismatch"));
        }
        let mut out = [[0.0_f32; BRAIN_INPUTS]; BRAIN_HIDDEN];
        for (i, row) in out.iter_mut().enumerate() {
            row.copy_from_slice(&flat[i * BRAIN_INPUTS..(i + 1) * BRAIN_INPUTS]);
        }
        Ok(out)
    }
}

mod serde_arrays_w2 {
    use super::{BRAIN_HIDDEN, BRAIN_OUTPUTS};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(
        w: &[[f32; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let flat: Vec<f32> = w.iter().flat_map(|row| row.iter().copied()).collect();
        flat.serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<[[f32; BRAIN_HIDDEN]; BRAIN_OUTPUTS], D::Error> {
        let flat: Vec<f32> = Vec::deserialize(d)?;
        if flat.len() != BRAIN_OUTPUTS * BRAIN_HIDDEN {
            return Err(serde::de::Error::custom("w2 length mismatch"));
        }
        let mut out = [[0.0_f32; BRAIN_HIDDEN]; BRAIN_OUTPUTS];
        for (i, row) in out.iter_mut().enumerate() {
            row.copy_from_slice(&flat[i * BRAIN_HIDDEN..(i + 1) * BRAIN_HIDDEN]);
        }
        Ok(out)
    }
}

// Sprint 48: serde 1 native podpora `[T; N]` jen pro N ≤ 32. `BRAIN_INPUTS` = 36.
mod serde_arr_inputs {
    use super::BRAIN_INPUTS;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(a: &[f32; BRAIN_INPUTS], s: S) -> Result<S::Ok, S::Error> {
        a.as_slice().serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<[f32; BRAIN_INPUTS], D::Error> {
        let v: Vec<f32> = Vec::deserialize(d)?;
        if v.len() != BRAIN_INPUTS {
            return Err(serde::de::Error::custom("inputs length mismatch"));
        }
        let mut a = [0.0_f32; BRAIN_INPUTS];
        a.copy_from_slice(&v);
        Ok(a)
    }
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
        // Sprint 66 bond signal bias — opt-in jako attack.
        b2[9] += INNATE_BOND_BIAS;
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
    pub spike_length: f32,
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
            brain: Brain::random(rng),
        }
    }

    pub fn mutate(&self, rng: &mut impl Rng, cfg: &MutationConfig) -> Self {
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
            shell_thickness: (self.shell_thickness + gaussian(rng) * cfg.sigma_shell)
                .clamp(MIN_SHELL_THICKNESS, MAX_SHELL_THICKNESS),
            adhesion_type,
            bond_stiffness: (self.bond_stiffness + gaussian(rng) * cfg.sigma_bond_stiffness)
                .clamp(MIN_BOND_STIFFNESS, MAX_BOND_STIFFNESS),
            bond_damping: (self.bond_damping + gaussian(rng) * cfg.sigma_bond_damping)
                .clamp(MIN_BOND_DAMPING, MAX_BOND_DAMPING),
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
            shell_thickness: if rng.random::<bool>() { a.shell_thickness } else { b.shell_thickness },
            adhesion_type: if rng.random::<bool>() { a.adhesion_type } else { b.adhesion_type },
            bond_stiffness: if rng.random::<bool>() { a.bond_stiffness } else { b.bond_stiffness },
            bond_damping: if rng.random::<bool>() { a.bond_damping } else { b.bond_damping },
            brain: Brain::crossover(&a.brain, &b.brain, rng),
        }
    }
}

fn gaussian(rng: &mut impl Rng) -> f32 {
    let u1: f32 = rng.random::<f32>().max(f32::EPSILON);
    let u2: f32 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}

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
}

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
    pub spike_length: f32,
    /// Sprint 41: snapshot z genomu, runtime morph zatím neexistuje.
    pub shell_thickness: f32,
}

impl Phenotype {
    pub fn from_genome(genome: &Genome) -> Self {
        Self {
            body_length: genome.body_length,
            body_width: genome.body_width,
            body_height: genome.body_height,
            spike_length: genome.spike_length,
            shell_thickness: genome.shell_thickness,
        }
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

/// Sprint 66: persistent spring bond mezi dvěma buňkami. Stateful — drží se
/// mezi ticky, dokud se neutrhne (overstretch) nebo cíl nezemře. Bond je
/// **directed** v Cell.bonds slotu (cell_i drží bond → other_cell_id), ale
/// druhá strana má symmetrický slot (Newton 3rd law se realizuje tím, že
/// každý cell aplikuje vlastní spring force).
///
/// Sprint 68: per-bond stiffness + damping (mean obou cells' genome při
/// formaci) — bond je fyzicky jedna pružina, takže k a c jsou symmetric
/// per-pair. Mutace brzdy/tuhosti nezmění existující bondy, ovlivní jen
/// budoucí formace; tato semantika dělá bondy "stabilními kontrakty"
/// místo per-tick re-evaluation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Bond {
    /// Stable identifier druhé buňky (Cell.cell_id). Rezolvuje se na index
    /// per-tick přes `FxHashMap<u64, usize>`. Pokud cíl zemřel, bond je
    /// dangling — pruning pass ho zahodí.
    pub other_cell_id: u64,
    /// Klidová délka spring (světové jednotky). Set při formaci bondu na
    /// `current_distance × BOND_REST_LENGTH_SLACK`.
    pub rest_length: f32,
    /// Sprint 68: per-bond spring constant. Set při formaci jako mean obou
    /// `genome.bond_stiffness`.
    pub stiffness: f32,
    /// Sprint 68: per-bond damping. Set při formaci jako mean obou
    /// `genome.bond_damping`.
    pub damping: f32,
    /// Tiků od formace. Diagnostic/logging — neovlivňuje fyziku.
    pub age_ticks: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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
    #[serde(with = "serde_arr_inputs")]
    pub last_inputs: [f32; BRAIN_INPUTS],
    pub last_hidden: [f32; BRAIN_HIDDEN],
    pub last_outputs: [f32; BRAIN_OUTPUTS],
    /// Sprint 30: nedobrovolný energy drain akumulovaný v aktuálním ticku
    /// (predation + hazard). Brain ho čte v dalším ticku jako input[14]
    /// (damage signal), pak resetuje na 0. Voluntární cost se NEZAPISUJE
    /// — cell sama drives ty náklady přes outputs, není to externí útok.
    pub damage_accum: f32,
    /// Sprint 42: ticks od spawnu / mating-childu. Drives aging body cost ramp.
    pub age: u64,
    /// Sprint 42: refractory period po mating, decremented per tick. Mating
    /// gating čte `== 0`.
    pub reproduce_cooldown_ticks: u32,
    /// Sprint 66: stable identifier pro bond resolution. Monotonic, přidělen
    /// World-level counterem při spawnu / mating childu. Lineage_id je sdílený
    /// per linie (ne unique per cell), takže potřebujeme samostatný cell_id.
    pub cell_id: u64,
    /// Sprint 66: persistent spring bonds. Fixní array `Option<Bond>` aby Cell
    /// zůstal `Copy` a žádný heap alloc per cell. Empty slots = `None`.
    pub bonds: [Option<Bond>; MAX_BONDS_PER_CELL],
    pub phenotype: Phenotype,
    pub genome: Genome,
}

impl Cell {
    pub fn random(
        rng: &mut impl Rng,
        world_half: [f32; 3],
        lineage_id: u64,
        lineage_birth_gen: u64,
        cell_id: u64,
    ) -> Self {
        let genome = Genome::random(rng);
        Self::from_genome(rng, genome, world_half, lineage_id, lineage_birth_gen, cell_id)
    }

    /// Sprint 69: count of populated bond slots. Used in predation defense
    /// (`bond_defense_factor`) — víc bondů = větší ochrana proti útoku.
    #[inline]
    pub fn n_bonds(&self) -> u32 {
        self.bonds.iter().filter(|b| b.is_some()).count() as u32
    }

    pub fn from_genome(
        rng: &mut impl Rng,
        genome: Genome,
        world_half: [f32; 3],
        lineage_id: u64,
        lineage_birth_gen: u64,
        cell_id: u64,
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
            age: 0,
            reproduce_cooldown_ticks: 0,
            cell_id,
            bonds: [None; MAX_BONDS_PER_CELL],
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
        // Sprint 42: aging + cooldown decrement na začátku ticku, aby
        // apply_energy_costs viděl current age v ramp formuli.
        self.age = self.age.saturating_add(1);
        if self.reproduce_cooldown_ticks > 0 {
            self.reproduce_cooldown_ticks -= 1;
        }
        self.integrate_kinematics(dt, world_half);
        self.apply_anisotropic_drag(dt, physics);
        self.apply_angular_drag(dt, physics);
        self.apply_energy_costs(dt, physics);
        self.apply_world_bounce(world_half);
    }

    /// Sprint 40: čistý integrate (position += v · dt, heading + pitch),
    /// gravity, pitch clamp. Žádné drag ani energie — to dělají další fáze.
    fn integrate_kinematics(&mut self, dt: f32, world_half: [f32; 3]) {
        self.position[0] += self.velocity[0] * dt;
        self.position[1] += self.velocity[1] * dt;
        self.position[2] += self.velocity[2] * dt;
        self.heading += self.angular_velocity * dt;
        // Sprint 38: gravity působí pouze pokud je z-volume aktivní. Aplikuje
        // se před drag aby drag mohl balancovat → terminal velocity.
        if world_half[2] > 0.0 {
            self.velocity[2] -= GRAVITY * dt;
        }
        // Sprint 35: pitch range ±π/12. Random brain pitch noise s tight
        // rangem = nepatrný drift v z, ale dovolí vědomé naklonění do z motion.
        self.pitch = (self.pitch + self.pitch_velocity * dt).clamp(
            -core::f32::consts::FRAC_PI_6 * 0.5,
            core::f32::consts::FRAC_PI_6 * 0.5,
        );
    }

    /// Sprint 40: 3D anisotropic drag. Forward = unit vector (yaw + pitch).
    /// Velocity → along-forward + perpendicular-forward; drag_par váhuje
    /// width × length cross-section, drag_perp váhuje length × width.
    /// Pro length=width=s perfectly izotropní (kompat s pre-Sprint-26).
    fn apply_anisotropic_drag(&mut self, dt: f32, physics: &PhysicsConfig) {
        let [fx, fy, fz] = forward_vector(self.heading, self.pitch);
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
    }

    /// Sprint 40: rotační drag pro yaw + pitch. Sdílený multiplier zachovává
    /// pre-refactor chování (oba decay stejnou rychlostí).
    fn apply_angular_drag(&mut self, dt: f32, physics: &PhysicsConfig) {
        let drag_factor = (1.0 - physics.angular_drag * dt).max(0.0);
        self.angular_velocity *= drag_factor;
        self.pitch_velocity *= drag_factor;
    }

    /// Sprint 40: per-tick energy costs. v², rotational (yaw only — Sprint 33
    /// note: pitch je „free" rotace, jinak by random brainy ztrácely 2× rotační
    /// drain), vision, body volume maintenance, spike, attack-mode upkeep.
    fn apply_energy_costs(&mut self, dt: f32, physics: &PhysicsConfig) {
        // Sprint 33: v_mag_sq zahrnuje 3D (vz != 0 v Sprint 35+).
        let v_mag_sq =
            self.velocity[0].powi(2) + self.velocity[1].powi(2) + self.velocity[2].powi(2);
        self.energy -= v_mag_sq * physics.energy_cost_per_v_sq * dt;
        let av = self.angular_velocity;
        let eff_r = self.phenotype.effective_radius();
        self.energy -= eff_r * eff_r * av * av * physics.angular_energy_cost * dt;
        self.energy -= self.genome.vision_radius * physics.vision_cost_per_radius * dt;
        // Sprint 34: maintenance ∝ 3D volume = length×width×height.
        // Sprint 42: aging ramp — starší cells platí postupně víc per volume unit.
        let age_sec = self.age as f32 / FIXED_TIMESTEP_HZ;
        let aging_factor = 1.0 + AGE_DECAY_PER_SEC * age_sec;
        self.energy -= self.phenotype.volume() * physics.body_cost_factor * aging_factor * dt;
        self.energy -= self.phenotype.spike_length * SPIKE_COST_PER_SEC * dt;
        // Sprint 41: shell maintenance — defensive armor stojí víc než spike,
        // protože pokrývá celý povrch.
        self.energy -= self.phenotype.shell_thickness * SHELL_COST_PER_SEC * dt;
        // Sprint 27 attack maintenance: cost ∝ max(0, output[6]).
        let attack_strength = self.last_outputs[6].max(0.0);
        self.energy -= attack_strength * ATTACK_COST_PER_SEC * dt;
    }

    /// Sprint 54: xy modulo wrap (toroidal cylinder topology), z bounce.
    /// Pre-Sprint-54 byly xy walls reflective → cells akumulovaly u krajů
    /// (Sprint 30+ edge_frac/corner_frac metriky), evoluce mohla najít wall
    /// exploit (těsné otáčení před stěnou), smell/pheromone gradients
    /// degenerovaly u Neumann boundary. Toroidal odstraní edge bias —
    /// cell na x=−950 sousedí s cell na x=+950. Z osa zůstává bounded
    /// (gravita + food sink + carrion drop vyžadují pevný strop/dno).
    fn apply_world_bounce(&mut self, world_half: [f32; 3]) {
        let wx = 2.0 * world_half[0];
        let wy = 2.0 * world_half[1];
        // xy wrap (toroidal): pozice → [-half, half).
        if self.position[0] >= world_half[0] || self.position[0] < -world_half[0] {
            // rem_euclid emulace pro f32: pos - floor((pos + half) / w) * w.
            let p = self.position[0] + world_half[0];
            self.position[0] = p - (p / wx).floor() * wx - world_half[0];
        }
        if self.position[1] >= world_half[1] || self.position[1] < -world_half[1] {
            let p = self.position[1] + world_half[1];
            self.position[1] = p - (p / wy).floor() * wy - world_half[1];
        }
        // z stále bounce.
        if world_half[2] > 0.0 && self.position[2].abs() > world_half[2] {
            self.velocity[2] = -self.velocity[2];
            self.position[2] = self.position[2].clamp(-world_half[2], world_half[2]);
        }
        // Heading recompute odstraněn — wrap nezmění směr pohybu.
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

    /// Sprint 40: aplikuje brain motor outputs na velocity / angular_velocity /
    /// pitch_velocity. Použito z hot loop v rendereru i headless po `populate_brain_inputs`
    /// + `Brain::forward_with_state`. Sjednocuje motor mapping logiku, která
    /// byla pre-refactor duplikovaná v `cells_brain_act` (main) a `brain_act`
    /// (headless). Bere `outputs[0] = turn`, `[1] = thrust`, `[7] = pitch`.
    pub fn apply_brain_motor(&mut self, outputs: &[f32; BRAIN_OUTPUTS], dt: f32) {
        // Sprint 42: F=ma — denominator je `mass = effective_radius` (smoke-tuned
        // fallback z `volume()`). Plný volume() byl příliš agresivní inerce penalty
        // pro untrained brainy v Sprint 42 smoke (extinct gen 40). `effective_radius`
        // (= aritmetický průměr 3 os) zachovává inerce-by-size škálování bez
        // kvadratického cost shocku. Cells s objemnějším tělem stále inertia, ale
        // menší magnitude.
        let mass = self.phenotype.effective_radius().max(0.01);
        let turn_rate = self.genome.turn_rate;
        let max_speed = self.genome.max_speed;
        let turn_signal = outputs[0];
        let thrust_norm = (outputs[1] + 1.0) * 0.5;
        let pitch_signal = outputs[7];
        let ang_acc = turn_signal * turn_rate / mass;
        self.angular_velocity += ang_acc * dt;
        let pitch_acc = pitch_signal * turn_rate / mass;
        self.pitch_velocity += pitch_acc * dt;
        let a_max = DRAG_COEFFICIENT * max_speed * max_speed / mass;
        let a = thrust_norm * a_max;
        let fwd = forward_vector(self.heading, self.pitch);
        self.velocity[0] += a * fwd[0] * dt;
        self.velocity[1] += a * fwd[1] * dt;
        self.velocity[2] += a * fwd[2] * dt;
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

    /// Sprint 41: pure ellipsoidní acceptance test bez mutace. Semi-axes ∝
    /// phenotype × `eat_factor`. Pro L=W=H=s shell sféra radius `s × eat_factor`
    /// — backward kompat s pre-Sprint-41 isotropic buňkou. Použito v binárkách
    /// pro broad+narrow phase split (broad uses `max_axis`, narrow tento test).
    pub fn eat_test(&self, food: &Food, eat_factor: f32) -> bool {
        eat_test_pose(
            self.position,
            self.heading,
            self.pitch,
            [
                self.phenotype.body_length,
                self.phenotype.body_width,
                self.phenotype.body_height,
            ],
            food.position,
            eat_factor,
        )
    }

    /// Sprint 41: ellipsoidální acceptance + energy gain při hitu.
    pub fn try_eat(&mut self, food: &Food, eat_factor: f32, food_value: f32) -> bool {
        if self.eat_test(food, eat_factor) {
            self.energy += food_value;
            true
        } else {
            false
        }
    }

    /// Sprint 41: tlumí `damage_accum` shellem před tím, než brain čte damage
    /// signal. Voláno z hot loop binárek **před** `populate_brain_inputs`
    /// (která damage čte + resetuje). `shell × ABSORB × dt` units, floor 0.
    pub fn apply_shell_absorb(&mut self, dt: f32) {
        let absorb = self.phenotype.shell_thickness * SHELL_ABSORB_PER_TICK * dt;
        self.damage_accum = (self.damage_accum - absorb).max(0.0);
    }

    /// Sprint 42: Brownův pohyb — gaussian noise na velocity. `√dt` scaling je
    /// correct stochastic integration (Wiener process), ne lineární dt. z-osa
    /// se rušený jen když je z-volume aktivní (`world_half_z > 0`).
    pub fn apply_brownian(&mut self, rng: &mut impl Rng, dt: f32, world_half_z: f32) {
        let scale = THERMAL_NOISE * dt.sqrt();
        self.velocity[0] += gaussian(rng) * scale;
        self.velocity[1] += gaussian(rng) * scale;
        if world_half_z > 0.0 {
            self.velocity[2] += gaussian(rng) * scale;
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Food {
    pub position: [f32; 3],
    /// Sprint 42: ticks od spawnu. Drives decay of `value_factor`. Init 0
    /// pro fresh food i carrion (univerzální decay, žádný carrion-specific
    /// staleness offset).
    pub age_ticks: u32,
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
    pub pheromone_grad: [f32; 3],
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
    inputs[11] = (sensors.pheromone_grad[0] * PHEROMONE_NORMALIZATION_GAIN).tanh();
    inputs[12] = (sensors.pheromone_grad[1] * PHEROMONE_NORMALIZATION_GAIN).tanh();
    inputs[19] = (sensors.pheromone_grad[2] * PHEROMONE_NORMALIZATION_GAIN).tanh();
    inputs[13] = (sensors.neighbors_in_vision as f32 / DENSITY_NORM_COUNT).tanh();
    inputs[14] = (cell.damage_accum * DAMAGE_NORMALIZATION_GAIN).tanh();
    cell.damage_accum = 0.0;
    inputs[BRAIN_INPUTS_SENSORY..BRAIN_INPUTS_SENSORY + BRAIN_RECURRENT]
        .copy_from_slice(&cell.last_hidden[..BRAIN_RECURRENT]);
    inputs
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
    let mid_pos = [
        (parent_a.position[0] + parent_b.position[0]) * 0.5,
        (parent_a.position[1] + parent_b.position[1]) * 0.5,
        (parent_a.position[2] + parent_b.position[2]) * 0.5,
    ];
    let child_phenotype = Phenotype::from_genome(&child_genome);
    Cell {
        position: mid_pos,
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
        damage_accum: 0.0,
        age: 0,
        // Sprint 42: child startuje s plnou cooldown — rodičovská cooldown
        // se nastaví v binárkách po `make_mating_child`, nezasáhne childa.
        reproduce_cooldown_ticks: 0,
        cell_id,
        // Sprint 66: child startuje bez bondů (čistý slate). Bondy se vytvoří
        // podle vlastního chování dítěte, neinheritují se po rodičích.
        bonds: [None; MAX_BONDS_PER_CELL],
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
    pub fn step(&mut self, diffusion: f32, decay_per_sec: f32, dt: f32) {
        let nx = self.resolution[0];
        let ny = self.resolution[1];
        let nz = self.resolution[2];
        let decay = (1.0 - decay_per_sec * dt).max(0.0);
        let plane = nx * ny;
        let grid = &self.grid;
        self.scratch
            .par_chunks_mut(plane)
            .enumerate()
            .for_each(|(k, scratch_plane)| {
                for j in 0..ny {
                    for i in 0..nx {
                        let idx_in_plane = j * nx + i;
                        let idx = k * plane + idx_in_plane;
                        let center = grid[idx];
                        // Toroidal xy: wrap kolem indexů.
                        let i_left = if i == 0 { nx - 1 } else { i - 1 };
                        let i_right = if i + 1 == nx { 0 } else { i + 1 };
                        let j_up = if j == 0 { ny - 1 } else { j - 1 };
                        let j_down = if j + 1 == ny { 0 } else { j + 1 };
                        let left = grid[k * plane + j * nx + i_left];
                        let right = grid[k * plane + j * nx + i_right];
                        let up = grid[k * plane + j_up * nx + i];
                        let down = grid[k * plane + j_down * nx + i];
                        // z bounded (Neumann): u krajů fallback na center.
                        let back = if k > 0 { grid[idx - plane] } else { center };
                        let front = if k + 1 < nz { grid[idx + plane] } else { center };
                        let new = center
                            + diffusion
                                * (left + right + up + down + back + front - 6.0 * center);
                        scratch_plane[idx_in_plane] = new * decay;
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
            shell_thickness: 0.0,
            adhesion_type: 0,
            bond_stiffness: BOND_STIFFNESS,
            bond_damping: BOND_DAMPING,
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
            sigma_shell: 0.0,
            sigma_brain: 0.0,
            adhesion_flip_rate: 0.0,
            sigma_bond_stiffness: 0.0,
            sigma_bond_damping: 0.0,
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
            shell_thickness: 0.0,
            adhesion_type: 0,
            bond_stiffness: BOND_STIFFNESS,
            bond_damping: BOND_DAMPING,
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
            sigma_shell: 10.0,
            sigma_brain: 10.0,
            adhesion_flip_rate: 0.5,
            sigma_bond_stiffness: 100.0,
            sigma_bond_damping: 10.0,
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
            age: 0,
            reproduce_cooldown_ticks: 0,
            cell_id: 0,
            bonds: [None; MAX_BONDS_PER_CELL],
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
    fn step_xy_wraps_toroidal() {
        // Sprint 54: xy wrap (cylinder topology). Cell s pos x=99, vel +60,
        // dt=1 → integrate kinematic dá pos x=159 → wrap modulo (world half=100,
        // wrap shift 200) → x=−41. Heading se nepojí (žádný bounce).
        let mut cell = Cell {
            position: [99.0, 0.0, 0.0],
            velocity: [60.0, 0.0, 0.0],
            heading: 0.0,
            ..base_cell()
        };
        cell.step(1.0, [100.0, 100.0, 0.0], &no_drag_physics(0.0, 0.0));
        assert!(
            (cell.position[0] - (-41.0)).abs() < 1e-3,
            "expected pos.x ≈ -41 after wrap, got {}",
            cell.position[0]
        );
        // Velocity beze změny po wrapu.
        assert!((cell.velocity[0] - 60.0).abs() < 1e-3);
        // Heading se po wrap nezmění.
        assert!((cell.heading - 0.0).abs() < 1e-3);
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
        let food = Food { position: [5.0, 0.0, 0.0], age_ticks: 0 };
        assert!(cell.try_eat(&food, 8.0, 20.0));
        assert_eq!(cell.energy, 70.0);
    }

    #[test]
    fn try_eat_outside_radius_returns_false_and_keeps_energy() {
        let mut cell = Cell {
            energy: 50.0,
            ..base_cell()
        };
        let food = Food { position: [20.0, 0.0, 0.0], age_ticks: 0 };
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
            shell_thickness: 0.0,
            adhesion_type: 1,
            bond_stiffness: 2.0,
            bond_damping: 0.3,
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
            shell_thickness: 0.5,
            adhesion_type: 5,
            bond_stiffness: 8.0,
            bond_damping: 1.0,
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
        let a = WorldMap::new([32, 32, 8], [8, 8, 4], [500.0, 500.0, 50.0], 42);
        let b = WorldMap::new([32, 32, 8], [8, 8, 4], [500.0, 500.0, 50.0], 42);
        assert_eq!(a.field(), b.field());
    }

    #[test]
    fn world_map_seeds_differ() {
        let a = WorldMap::new([32, 32, 8], [8, 8, 4], [500.0, 500.0, 50.0], 1);
        let b = WorldMap::new([32, 32, 8], [8, 8, 4], [500.0, 500.0, 50.0], 2);
        assert_ne!(a.field(), b.field());
    }

    #[test]
    fn world_map_values_in_unit_range() {
        let m = WorldMap::new([32, 32, 8], [8, 8, 4], [500.0, 500.0, 50.0], 7);
        for &v in m.field() {
            assert!((0.0..=1.0).contains(&v), "out of range: {}", v);
        }
    }

    #[test]
    fn world_map_sample_xy_wraps_z_clamps() {
        // Sprint 54: xy modulo wrap, z clamp. Inside sample ∈ [0,1].
        // Bod přesně na +half_x je přes wrap ekvivalentní -half_x.
        let m = WorldMap::new([8, 8, 4], [4, 4, 2], [100.0, 100.0, 50.0], 0);
        let inside = m.sample([99.0, 99.0, 0.0]);
        // Sample at +half wraps to -half (same grid cell).
        let at_left = m.sample([-100.0, 0.0, 0.0]);
        let at_right_wrap = m.sample([100.0, 0.0, 0.0]);
        assert!((at_left - at_right_wrap).abs() < 1e-6, "xy wrap broken");
        // Z out-of-range clamps (still valid, no panic).
        let above = m.sample([0.0, 0.0, 1e6]);
        let below = m.sample([0.0, 0.0, -1e6]);
        assert!((0.0..=1.0).contains(&above));
        assert!((0.0..=1.0).contains(&below));
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
            shell_thickness: 0.0,
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
            shell_thickness: 0.0,
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
            shell_thickness: 0.0,
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
            shell_thickness: 0.0,
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
                shell_thickness: 0.0,
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

    #[test]
    fn body_basis_orthonormal() {
        let cases = [
            (0.0, 0.0),
            (0.5, 0.0),
            (-1.2, 0.0),
            (1.7, 0.3),
            (-2.4, -0.4),
            (3.1, 0.5),
            (0.7, -0.2),
        ];
        for &(yaw, pitch) in &cases {
            let (fwd, right, up) = body_basis(yaw, pitch);
            let mag = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            assert!((mag(fwd) - 1.0).abs() < 1e-5, "fwd not unit at yaw={yaw} pitch={pitch}");
            assert!((mag(right) - 1.0).abs() < 1e-5, "right not unit");
            assert!((mag(up) - 1.0).abs() < 1e-5, "up not unit");
            assert!(dot(fwd, right).abs() < 1e-5, "fwd·right != 0");
            assert!(dot(fwd, up).abs() < 1e-5, "fwd·up != 0");
            assert!(dot(right, up).abs() < 1e-5, "right·up != 0");
        }
    }

    #[test]
    fn try_eat_isotropic_unchanged_for_unit_sphere() {
        // L=W=H=1, eat_factor=8 → ellipsoid degeneruje na sféru radius 8.
        // Backward-kompat se Sprint 40 sférickou eat-zónou.
        let cell = Cell { energy: 50.0, ..base_cell() };
        let inside = Food { position: [5.0, 0.0, 0.0], age_ticks: 0 };
        let outside = Food { position: [10.0, 0.0, 0.0], age_ticks: 0 };
        let lateral_inside = Food { position: [0.0, 5.0, 0.0], age_ticks: 0 };
        let vertical_inside = Food { position: [0.0, 0.0, 5.0], age_ticks: 0 };
        assert!(cell.eat_test(&inside, 8.0));
        assert!(!cell.eat_test(&outside, 8.0));
        assert!(cell.eat_test(&lateral_inside, 8.0));
        assert!(cell.eat_test(&vertical_inside, 8.0));
    }

    #[test]
    fn try_eat_forward_chip_reaches_further_than_lateral() {
        // Chip: L=2, W=0.5, H=0.5, heading=0 → forward semi-osa = 16, lateral = 4.
        let mut cell = Cell { energy: 50.0, ..base_cell() };
        cell.phenotype = Phenotype {
            body_length: 2.0,
            body_width: 0.5,
            body_height: 0.5,
            spike_length: 0.0,
            shell_thickness: 0.0,
        };
        // Forward at +14: inside ellipsoid (14/16 = 0.875).
        let forward_inside = Food { position: [14.0, 0.0, 0.0], age_ticks: 0 };
        // Lateral at +3.5: inside (3.5/4 = 0.875).
        let lateral_inside = Food { position: [0.0, 3.5, 0.0], age_ticks: 0 };
        // Forward at +17: outside (17/16 > 1).
        let forward_outside = Food { position: [17.0, 0.0, 0.0], age_ticks: 0 };
        // Lateral at +5: outside (5/4 > 1).
        let lateral_outside = Food { position: [0.0, 5.0, 0.0], age_ticks: 0 };
        assert!(cell.eat_test(&forward_inside, 8.0));
        assert!(cell.eat_test(&lateral_inside, 8.0));
        assert!(!cell.eat_test(&forward_outside, 8.0));
        assert!(!cell.eat_test(&lateral_outside, 8.0));
    }

    #[test]
    fn max_axis_returns_largest_dimension() {
        let phen = Phenotype {
            body_length: 2.0,
            body_width: 0.5,
            body_height: 1.5,
            spike_length: 0.0,
            shell_thickness: 0.0,
        };
        assert!((phen.max_axis() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn shell_absorbs_predation_drain() {
        // shell=1.0, ABSORB_PER_TICK=2.0, dt=1 → absorbed 2.0; raw damage 3.0
        // → damage_accum after = 1.0.
        let mut cell = base_cell();
        cell.phenotype.shell_thickness = 1.0;
        cell.damage_accum = PREDATION_DRAIN_PER_TICK; // = 3.0
        cell.apply_shell_absorb(1.0);
        let expected = PREDATION_DRAIN_PER_TICK - 1.0 * SHELL_ABSORB_PER_TICK;
        assert!(
            (cell.damage_accum - expected).abs() < 1e-5,
            "got {}, expected {}",
            cell.damage_accum,
            expected
        );
    }

    #[test]
    fn shell_zero_no_effect() {
        let mut cell = base_cell();
        cell.phenotype.shell_thickness = 0.0;
        cell.damage_accum = 2.5;
        cell.apply_shell_absorb(1.0);
        assert_eq!(cell.damage_accum, 2.5);
    }

    #[test]
    fn shell_does_not_absorb_below_zero() {
        // Big shell, small damage → clamp to 0, ne na negative.
        let mut cell = base_cell();
        cell.phenotype.shell_thickness = MAX_SHELL_THICKNESS;
        cell.damage_accum = 1.0;
        cell.apply_shell_absorb(1.0);
        assert_eq!(cell.damage_accum, 0.0);
    }

    #[test]
    fn shell_cost_scales_linearly() {
        let mut cell = base_cell();
        cell.phenotype.shell_thickness = 1.0;
        let physics = PhysicsConfig {
            drag: 0.0,
            angular_drag: 0.0,
            energy_cost_per_v_sq: 0.0,
            angular_energy_cost: 0.0,
            vision_cost_per_radius: 0.0,
            body_cost_factor: 0.0,
        };
        cell.step(1.0, [1000.0, 1000.0, 0.0], &physics);
        let expected_drain = 1.0 * SHELL_COST_PER_SEC;
        assert!(
            (cell.energy - (100.0 - expected_drain)).abs() < 1e-4,
            "got {}, expected {}",
            cell.energy,
            100.0 - expected_drain
        );
    }

    #[test]
    fn shell_mutation_clamps_to_range() {
        let mut rng = rand::rng();
        let g = dummy_genome();
        let cfg = MutationConfig {
            sigma_speed: 0.0,
            sigma_hue: 0.0,
            sigma_vision: 0.0,
            sigma_turn_rate: 0.0,
            sigma_body_length: 0.0,
            sigma_body_width: 0.0,
            sigma_body_height: 0.0,
            sigma_spike_length: 0.0,
            sigma_shell: 100.0,
            sigma_brain: 0.0,
            adhesion_flip_rate: 0.0,
            sigma_bond_stiffness: 0.0,
            sigma_bond_damping: 0.0,
        };
        for _ in 0..1000 {
            let m = g.mutate(&mut rng, &cfg);
            assert!(
                (MIN_SHELL_THICKNESS..=MAX_SHELL_THICKNESS).contains(&m.shell_thickness),
                "shell out of range: {}",
                m.shell_thickness
            );
        }
    }

    #[test]
    fn step_aging_increases_body_cost() {
        let physics = PhysicsConfig {
            drag: 0.0,
            angular_drag: 0.0,
            energy_cost_per_v_sq: 0.0,
            angular_energy_cost: 0.0,
            vision_cost_per_radius: 0.0,
            body_cost_factor: 1.0,
        };
        // Cell at age 0 → factor 1.0, drain = volume = 1.
        let mut young = base_cell();
        young.age = 0;
        let young_energy_before = young.energy;
        young.step(1.0, [1000.0, 1000.0, 0.0], &physics);
        let young_drain = young_energy_before - young.energy;

        // Cell at age 600 (= 10s) → factor 1 + 0.005×10 = 1.05.
        let mut old = base_cell();
        old.age = 600;
        let old_energy_before = old.energy;
        old.step(1.0, [1000.0, 1000.0, 0.0], &physics);
        let old_drain = old_energy_before - old.energy;

        assert!(
            old_drain > young_drain,
            "old cell should drain more: young={} old={}",
            young_drain,
            old_drain
        );
    }

    #[test]
    fn step_increments_age() {
        let mut cell = base_cell();
        assert_eq!(cell.age, 0);
        cell.step(1.0, [1000.0, 1000.0, 0.0], &no_drag_physics(0.0, 0.0));
        assert_eq!(cell.age, 1);
        cell.step(1.0, [1000.0, 1000.0, 0.0], &no_drag_physics(0.0, 0.0));
        assert_eq!(cell.age, 2);
    }

    #[test]
    fn cooldown_decrements_per_step() {
        let mut cell = base_cell();
        cell.reproduce_cooldown_ticks = 5;
        cell.step(1.0, [1000.0, 1000.0, 0.0], &no_drag_physics(0.0, 0.0));
        assert_eq!(cell.reproduce_cooldown_ticks, 4);
    }

    #[test]
    fn cooldown_does_not_underflow() {
        let mut cell = base_cell();
        cell.reproduce_cooldown_ticks = 0;
        cell.step(1.0, [1000.0, 1000.0, 0.0], &no_drag_physics(0.0, 0.0));
        assert_eq!(cell.reproduce_cooldown_ticks, 0);
    }

    #[test]
    fn motor_scales_inversely_with_mass() {
        // Unit cell eff_r=1 vs tubby cell eff_r=2 (L=W=H=2): tubby pomalejší 2×.
        // Mass scaling používá effective_radius (smoke-tuned fallback z volume).
        let mut unit = base_cell();
        let mut tubby = base_cell();
        tubby.phenotype.body_length = 2.0;
        tubby.phenotype.body_width = 2.0;
        tubby.phenotype.body_height = 2.0;
        let outputs = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        unit.apply_brain_motor(&outputs, 1.0);
        tubby.apply_brain_motor(&outputs, 1.0);
        let unit_v = unit.velocity[0].abs();
        let tubby_v = tubby.velocity[0].abs();
        assert!(
            unit_v > tubby_v,
            "unit cell should accelerate faster: unit={} tubby={}",
            unit_v,
            tubby_v
        );
        let ratio = unit_v / tubby_v.max(1e-6);
        assert!(
            (ratio - 2.0).abs() < 0.2,
            "expected ratio ~2 (eff_r), got {}",
            ratio
        );
    }

    #[test]
    fn brownian_perturbs_zero_velocity() {
        let mut rng = rand::rng();
        let mut cell = base_cell();
        // 100 brownian steps; statisticky téměř jistě některá komponenta != 0.
        for _ in 0..100 {
            cell.apply_brownian(&mut rng, 1.0 / FIXED_TIMESTEP_HZ, 0.0);
        }
        // 2D případ (world_half_z = 0) — z se nesmí měnit.
        assert_eq!(cell.velocity[2], 0.0);
        let v_xy_sq =
            cell.velocity[0] * cell.velocity[0] + cell.velocity[1] * cell.velocity[1];
        assert!(v_xy_sq > 0.0, "expected nonzero velocity from brownian");
    }

    #[test]
    fn brownian_z_only_in_3d_world() {
        let mut rng = rand::rng();
        let mut cell = base_cell();
        // 3D mode: world_half_z > 0 → z se má hýbat.
        for _ in 0..100 {
            cell.apply_brownian(&mut rng, 1.0 / FIXED_TIMESTEP_HZ, 2.0);
        }
        assert!(cell.velocity[2] != 0.0, "expected nonzero z velocity in 3D");
    }

    #[test]
    fn food_value_decays_with_age() {
        let mut food = Food { position: [0.0, 0.0, 0.0], age_ticks: 0 };
        assert!((food.value_factor() - 1.0).abs() < 1e-6);
        // 1 sec = 60 ticks → factor = 1 - CARRION_DECAY_PER_SEC.
        food.age_ticks = 60;
        let expected = 1.0 - CARRION_DECAY_PER_SEC;
        assert!(
            (food.value_factor() - expected).abs() < 1e-4,
            "got {}, expected {}",
            food.value_factor(),
            expected
        );
    }

    #[test]
    fn food_expires_when_zero_value() {
        let mut fresh = Food { position: [0.0, 0.0, 0.0], age_ticks: 0 };
        assert!(fresh.age_step());
        // Past lifetime: age_step bump → value_factor = 0 → returns false.
        // F32 precision: použijeme age daleko za bod expirace, abychom se vyhli
        // ULP edge case (60.0/0.0005 jako u32 rounds k 119999, ne 120000).
        let mut expired = Food {
            position: [0.0, 0.0, 0.0],
            age_ticks: ((FIXED_TIMESTEP_HZ / CARRION_DECAY_PER_SEC) as u32) + 100,
        };
        assert!(!expired.age_step());
    }

    #[test]
    fn child_starts_with_zero_age_and_cooldown() {
        let mut rng = rand::rng();
        let g = dummy_genome();
        let cell_a = Cell::from_genome(&mut rng, g, [100.0, 100.0, 0.0], 0, 0, 1);
        let cell_b = Cell::from_genome(&mut rng, g, [100.0, 100.0, 0.0], 0, 0, 2);
        let child = make_mating_child(&cell_a, &cell_b, &mut rng, 3);
        assert_eq!(child.age, 0);
        assert_eq!(child.reproduce_cooldown_ticks, 0);
    }

    #[test]
    fn spatial_grid_finds_all_neighbors_in_radius() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 1000;
        let half: f32 = 500.0;
        let points: Vec<(usize, [f32; 3], ())> = (0..n)
            .map(|i| {
                (
                    i,
                    [
                        rng.random_range(-half..half),
                        rng.random_range(-half..half),
                        rng.random_range(-1.0..1.0),
                    ],
                    (),
                )
            })
            .collect();

        let mut grid: SpatialGrid<usize, ()> = SpatialGrid::new(GRID_CELL_SIZE);
        grid.rebuild(points.iter().copied());

        let query_pos = [0.0_f32, 0.0, 0.0];
        let radius = 50.0_f32;
        let r2 = radius * radius;

        let mut brute: Vec<usize> = points
            .iter()
            .filter_map(|(i, p, _)| {
                let dx = p[0] - query_pos[0];
                let dy = p[1] - query_pos[1];
                let dz = p[2] - query_pos[2];
                if dx * dx + dy * dy + dz * dz <= r2 {
                    Some(*i)
                } else {
                    None
                }
            })
            .collect();
        brute.sort();

        let mut from_grid: Vec<usize> = Vec::new();
        grid.for_each_in_radius(query_pos, radius, |id, p, _| {
            let dx = p[0] - query_pos[0];
            let dy = p[1] - query_pos[1];
            let dz = p[2] - query_pos[2];
            if dx * dx + dy * dy + dz * dz <= r2 {
                from_grid.push(id);
            }
        });
        from_grid.sort();

        assert_eq!(
            brute, from_grid,
            "grid query missed or extra neighbors vs brute force"
        );
    }

    #[test]
    fn spatial_grid_rebuild_clears_old_buckets() {
        let mut grid: SpatialGrid<usize, ()> = SpatialGrid::new(50.0);
        grid.rebuild(vec![(0_usize, [0.0, 0.0, 0.0], ()), (1, [10.0, 10.0, 0.0], ())]);

        let mut first: Vec<usize> = Vec::new();
        grid.for_each_in_radius([0.0, 0.0, 0.0], 100.0, |id, _, _| first.push(id));
        first.sort();
        assert_eq!(first, vec![0, 1]);

        grid.rebuild(vec![(2_usize, [200.0, 200.0, 0.0], ())]);
        let mut second: Vec<usize> = Vec::new();
        grid.for_each_in_radius([0.0, 0.0, 0.0], 100.0, |id, _, _| second.push(id));
        assert!(
            second.is_empty(),
            "rebuild left stale entries near origin: {:?}",
            second
        );

        let mut third: Vec<usize> = Vec::new();
        grid.for_each_in_radius([200.0, 200.0, 0.0], 100.0, |id, _, _| third.push(id));
        assert_eq!(third, vec![2]);
    }

    #[test]
    fn spatial_grid_query_order_is_stable() {
        let points: Vec<(usize, [f32; 3], ())> = (0..50)
            .map(|i| (i, [i as f32 * 5.0, (i % 7) as f32 * 3.0, 0.0], ()))
            .collect();
        let mut grid: SpatialGrid<usize, ()> = SpatialGrid::new(20.0);
        grid.rebuild(points.iter().copied());

        let mut a: Vec<usize> = Vec::new();
        grid.for_each_in_radius([100.0, 10.0, 0.0], 60.0, |id, _, _| a.push(id));
        let mut b: Vec<usize> = Vec::new();
        grid.for_each_in_radius([100.0, 10.0, 0.0], 60.0, |id, _, _| b.push(id));

        assert_eq!(a, b, "two identical queries returned different order");
    }

    // === Sprint 66: differential adhesion + spring bonds ===

    #[test]
    fn adhesion_is_zero_inside_contact() {
        // d <= pair_r: collision depenetration handles, adhesion no-op.
        let pair_r = 10.0;
        let delta = [pair_r * 0.5, 0.0, 0.0];
        let v = adhesion_velocity_delta(delta, pair_r * 0.5, pair_r, true);
        assert_eq!(v, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn adhesion_is_zero_beyond_range() {
        let pair_r = 10.0;
        let range = pair_r * ADHESION_RANGE_FACTOR;
        let d = range + 1.0;
        let delta = [d, 0.0, 0.0];
        let v = adhesion_velocity_delta(delta, d, pair_r, true);
        assert_eq!(v, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn adhesion_pulls_same_type_inward() {
        // pos_i - pos_j = +x (i je vpravo od j); same-type → attraction
        // znamená velocity i přírůstek směrem -x (k j).
        let pair_r = 10.0;
        let d = pair_r * 1.5;
        let delta = [d, 0.0, 0.0];
        let v = adhesion_velocity_delta(delta, d, pair_r, true);
        assert!(v[0] < 0.0, "expected pull toward j, got Δv = {:?}", v);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 0.0);
    }

    #[test]
    fn adhesion_repels_cross_type_outward() {
        let pair_r = 10.0;
        let d = pair_r * 1.5;
        let delta = [d, 0.0, 0.0];
        let v = adhesion_velocity_delta(delta, d, pair_r, false);
        assert!(v[0] > 0.0, "expected push away, got Δv = {:?}", v);
    }

    #[test]
    fn bond_spring_pulls_when_stretched() {
        // Bond rest 5, current 10 (stretched) → cell i taženo k j.
        let bond = Bond { other_cell_id: 1, rest_length: 5.0, stiffness: BOND_STIFFNESS, damping: BOND_DAMPING, age_ticks: 0 };
        let delta = [10.0, 0.0, 0.0];
        let (v, broken) = bond_velocity_delta(&bond, delta, 10.0, [0.0; 3], [0.0; 3]);
        assert!(!broken);
        assert!(v[0] < 0.0, "stretched bond should pull i toward j, got {:?}", v);
    }

    #[test]
    fn bond_spring_pushes_when_compressed() {
        // Bond rest 10, current 5 (compressed) → cell i tlačeno od j.
        let bond = Bond { other_cell_id: 1, rest_length: 10.0, stiffness: BOND_STIFFNESS, damping: BOND_DAMPING, age_ticks: 0 };
        let delta = [5.0, 0.0, 0.0];
        let (v, broken) = bond_velocity_delta(&bond, delta, 5.0, [0.0; 3], [0.0; 3]);
        assert!(!broken);
        assert!(v[0] > 0.0, "compressed bond should push i away, got {:?}", v);
    }

    #[test]
    fn bond_breaks_past_break_factor() {
        let rest = 5.0;
        let bond = Bond { other_cell_id: 1, rest_length: rest, stiffness: BOND_STIFFNESS, damping: BOND_DAMPING, age_ticks: 0 };
        let stretched = rest * BOND_BREAK_FACTOR + 0.1;
        let (v, broken) = bond_velocity_delta(
            &bond,
            [stretched, 0.0, 0.0],
            stretched,
            [0.0; 3],
            [0.0; 3],
        );
        assert!(broken, "bond should break past BOND_BREAK_FACTOR");
        assert_eq!(v, [0.0; 3]);
    }

    #[test]
    fn bond_damping_opposes_closing_velocity() {
        // Cell i at +x, j at origin, bond at rest. v_i moves toward j (−x).
        // Damping should *resist* closing → push i back (+x).
        let bond = Bond { other_cell_id: 1, rest_length: 5.0, stiffness: BOND_STIFFNESS, damping: BOND_DAMPING, age_ticks: 0 };
        let delta = [5.0, 0.0, 0.0];
        let v_i = [-1.0, 0.0, 0.0];
        let v_j = [0.0, 0.0, 0.0];
        let (dv, _) = bond_velocity_delta(&bond, delta, 5.0, v_i, v_j);
        assert!(dv[0] > 0.0, "damping should oppose closing motion, got {:?}", dv);
    }

    #[test]
    fn bond_defense_factor_solo_is_unity() {
        assert!((bond_defense_factor(0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bond_defense_factor_scales_linearly_until_cap() {
        // 1 bond → 0.85, 2 → 0.70, 3 → 0.55, 4 → 0.40 (cap), 5+ → 0.40.
        assert!((bond_defense_factor(1) - 0.85).abs() < 1e-6);
        assert!((bond_defense_factor(2) - 0.70).abs() < 1e-6);
        assert!((bond_defense_factor(3) - 0.55).abs() < 1e-6);
        assert!((bond_defense_factor(4) - 0.40).abs() < 1e-6);
        assert!((bond_defense_factor(5) - 0.40).abs() < 1e-6);
        assert!((bond_defense_factor(MAX_BONDS_PER_CELL as u32) - 0.40).abs() < 1e-6);
    }

    #[test]
    fn n_bonds_counts_only_populated_slots() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut cell = Cell::random(&mut rng, [960.0, 540.0, 50.0], 0, 0, 0);
        assert_eq!(cell.n_bonds(), 0);
        cell.bonds[0] = Some(Bond {
            other_cell_id: 99,
            rest_length: 5.0,
            age_ticks: 0,
            stiffness: BOND_STIFFNESS,
            damping: 0.6,
        });
        cell.bonds[3] = Some(Bond {
            other_cell_id: 100,
            rest_length: 5.0,
            age_ticks: 0,
            stiffness: BOND_STIFFNESS,
            damping: 0.6,
        });
        assert_eq!(cell.n_bonds(), 2);
    }

    #[test]
    fn adhesion_works_across_toroidal_boundary() {
        // Cell i at x=950, j at x=-950, world half_x=960. Min-image delta
        // by měl být ~20 (přes wrap), ne ~1900.
        let world_half = [960.0, 540.0, 50.0];
        let pos_i = [950.0, 0.0, 0.0];
        let pos_j = [-950.0, 0.0, 0.0];
        let pair_r = 10.0;
        let d_vec = min_image_delta(pos_j, pos_i, world_half);
        let d = (d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2]).sqrt();
        assert!(d < 25.0, "min-image distance should be ~20, got {}", d);
        let v = adhesion_velocity_delta(d_vec, d, pair_r, true);
        // Pull from i toward j přes wrap = +x (i is at +950, j wraps to +970).
        assert!(v[0] > 0.0, "expected wrap-aware pull, got {:?}", v);
    }
}
