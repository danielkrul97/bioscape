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
/// Sprint 73: horní cap pro `genome.max_speed`. Pre-Sprint-73 nebyl žádný cap,
/// jen MIN_SPEED floor — sigma_speed=3.0 mutation drift v Sprint 72 1000-gen
/// smoke produkoval mean spd 344 (12 % nad HUNTER_MAX_SPEED=300). Bez cap
/// je arms race degenerative — outrun je vždy levnější než cluster path,
/// takže multicelularita nikdy nepřijde. 200 = mírně pod Sprint 71 baseline
/// 218 → HUNTER_MAX_SPEED=300 reálně neutekatelný → cluster path se stává
/// jediná viable defense. Nominal cap, ne hard cieling — sigma_speed pořád
/// dovoluje fluktuace v rámci capu.
pub const MAX_SPEED: f32 = 200.0;
const MIN_VISION: f32 = 1.0;
/// Sprint 82: minimum half-angle směrového FOV (radiány). Pod ~17° je vidění
/// degenerované — buňka skoro nic neuvidí, nemá smysl evolvovat dál.
pub const MIN_VISION_FOV: f32 = core::f32::consts::PI / 12.0;
/// Sprint 82: maximum half-angle FOV = π = full sphere (4π str solid angle).
/// `vision_fov_factor(π) = 1.0` → energy cost identický s pre-Sprint-82
/// omnidirectional vision; výchozí hodnota při Genome::random.
pub const MAX_VISION_FOV: f32 = core::f32::consts::PI;
/// Sprint 82: initial vision_fov při Genome::random = full sphere baseline.
/// Cone filter v sensor gather je no-op pro fov ≥ π (cos π = −1, dot vždy ≥ −1).
/// Selekční tlak (cost ∝ fov_factor) může FOV zúžit, pokud info-loss < energy
/// savings; v Sprint 82 je `sigma_vision_fov = 0` → gen drift dormant.
pub const INITIAL_VISION_FOV: f32 = core::f32::consts::PI;
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
// Sprint 39: 8 → 16 — větší hidden kapacita pro 3D + gravity. 28 inputs → 8
// hidden bylo příliš stěsnaný "kompresní bottleneck" pro 3D navigaci.
// w1 z 28×8=224 na 36×16=576 weights (2.6×).
// Sprint 80 (decade NEAT): 16 → 32 jako *storage cap*. Aktivní hidden count je
// per-cell v `Brain.hidden_n`; default spawn = `BRAIN_HIDDEN_DEFAULT` = 16.
// Storage 32 dává prostor pro structural mutace (add_neuron) bez Brain struct
// resize. Dead zone (rows hidden_n..BRAIN_HIDDEN) je zero-initialized a
// nepřispívá do forward passu (zero × x = 0).
// Memory: brain.w1 z 16×36 = 576 floats na 32×52 = 1664 (2.9×).
// Per cell brain footprint ~8 KB; 2500 cells = ~20 MB total. Acceptable.
pub const BRAIN_HIDDEN: usize = 32;
/// Initial active hidden count při spawn. Cell s `hidden_n = BRAIN_HIDDEN_DEFAULT`
/// reprodukuje pre-Sprint-80 behavior byte-identical (dead zone weights = 0,
/// nepřispívá). Structural mutace později rozhojnou na ≤ `BRAIN_HIDDEN`.
pub const BRAIN_HIDDEN_DEFAULT: usize = 16;
/// Floor pro budoucí remove_neuron mutaci. Pod 4 neurony brain nedělá nic
/// užitečného (sensory→hidden kompresor potřebuje minimum kapacity).
pub const BRAIN_HIDDEN_MIN: usize = 4;
/// Sprint 80 Sprint C: gaussian sigma pro inicializaci nových neuronů v
/// `Brain::add_neuron`. Menší než `sigma_brain` (0.2) — NEAT-style minimal
/// disruption: nový neuron startuje s near-zero kontribucí k outputům, takže
/// add_neuron sám o sobě neporouchá existující funkční mozek. Selekce ho
/// pak buď posílí (subsequent weight mutace), nebo nechá v latentní formě.
pub const ADD_NEURON_SIGMA: f32 = 0.1;
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
/// Sprint 66 bond signal bias (b2[9]). Sprint 75: 0 → 1.5. Sprint 74 1000-gen
/// smoke ukázal, že bond density crashed na 0 nehledě na economy rebalance —
/// real bottleneck není maintenance cost, ale **formation gating**: random
/// brainy s bias=0 dávají output[9] > BOND_FORM_THRESHOLD=0.2 jen sporadicky,
/// takže bondy se nikdy nestihnou hromadit do clusteru ≥3 (= immune
/// k hunteru). Bias 1.5 znamená default tanh(b1[9] + 1.5) ≈ 0.9 → většina
/// cells emituje signal nad threshold by default. Bondy se formují přes
/// physics (contact + same adhesion_type), selekce může negativně tunit
/// (cells co nechtějí bondovat se učí brain weights pull b1[9] dolů).
/// Filozoficky stejný posun jako kdyby attack měl positive bias místo 0.
pub const INNATE_BOND_BIAS: f32 = 1.5;

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

// Thermal stratification (Sprint 85). Lineární z-gradient: warm at top, cold
// at bottom (oceán-like). Coupling přes Q10 multiplikátor na all per-tick
// drains v `apply_energy_costs`. Niche separation emergne behaviorálně —
// cells co plavou dolů platí míň energie ale nemají žádnou další výhodu;
// selekce hledá optimum food-density × thermal-cost. Žádný temperature
// field grid — funkce z stačí.
/// Maximum teploty (z = +world_half[2], hladina). Sim units, ne reálné °C.
pub const THERMAL_TOP: f32 = 30.0;
/// Minimum teploty (z = −world_half[2], dno). Sim units.
pub const THERMAL_BOTTOM: f32 = 4.0;
/// Q10 koeficient — biologické rychlosti přibližně 2× per +10 sim-units
/// teploty. Standard biology Q10 ~ 2-3 (enzyme kinetics, metabolic rate).
pub const THERMAL_Q10: f32 = 2.0;
/// Referenční teplota — při T = THERMAL_REF_TEMP je `metabolism_factor = 1.0`,
/// drain identický s pre-Sprint-85. Volena uprostřed [BOTTOM, TOP] aby
/// průměrná cell drain ~ pre-Sprint-85 (ratio top:ref:bottom ≈ 2.46:1:0.41).
pub const THERMAL_REF_TEMP: f32 = 17.0;
/// Sprint 86: peak diurnal amplitude na hladině. Surface temperature osciluje
/// `THERMAL_TOP ± THERMAL_DIURNAL_AMP` v rámci 1 day = `TICKS_PER_GENERATION`
/// ticks (10 s real-time). Hloubka oscilace klesá lineárně k 0 na `THERMAL_BOTTOM`
/// (= mirror reálné termokliny: deep water buffered proti solárnímu cyklu).
pub const THERMAL_DIURNAL_AMP: f32 = 5.0;
/// Sprint 86: full diurnal cycle = 1 generation = 600 ticks = 10 s real-time.
/// Cells with ~1 gen lifespan experience exactly one day. Krátší period =
/// flicker, delší = cells málokdy zažijí přechod.
pub const THERMAL_DIURNAL_PERIOD_TICKS: u64 = TICKS_PER_GENERATION;
/// Sprint 86: peak seasonal amplitude (uniform shift všech depth). Period =
/// `CYCLE_GEN_PERIOD` (50 gen) — sdílený s food density cyklem, takže warm
/// season = abundant food (summer), cold season = scarce food (winter).
/// Coupling vytváří přírodní seasonal niche shift.
pub const THERMAL_SEASONAL_AMP: f32 = 4.0;
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
/// Sprint 77: 0.3 → 0.8 — preventivní hack před Sprintem 78. Sprint 73-76
/// smokes ukázaly konvergenci k asp=12 needles, které fyzicky neclusterují.
/// Sprint 80 attempt-1: 0.8 → 0.3 → tissue collapse (asp 14, mBond 0 do gen 250).
/// Sprint 80 final: revert na 0.8.
pub const MIN_BODY_WIDTH: f32 = 0.8;
pub const MAX_BODY_WIDTH: f32 = 4.0;
/// Sprint 34: 3-axis ellipsoid — třetí dimenze (vertikálně, ⊥ k length+width).
pub const MIN_BODY_HEIGHT: f32 = 0.3;
/// Sprint 80 attempt-2: MAX_BODY_* 4 → 6 (s MIN_BODY_WIDTH=0.8 zpět). Smoke
/// 500 gen ukázal nový pancake attractor — cells konvergovaly k len=5.8 ×
/// wid=5.9 × hgt=0.32 (flat disc), volume ~11 vs S78 ~4.4. mBond collapsed
/// na 0 do gen 50. Wide+flat outcompete clustered cells na food intake area.
/// Sprint 80 final: revert na 4.0.
pub const MAX_BODY_HEIGHT: f32 = 4.0;
pub const MIN_SPIKE_LENGTH: f32 = 0.0;
pub const MAX_SPIKE_LENGTH: f32 = 2.0;
/// Rychlost runtime morfingu — full brain output dává `MORPH_RATE` jednotek
/// změny tvaru za sekundu.
/// Sprint 26: 0.02/s = full-range body morph ~50 gen, pomalejší než životnost
/// generace → morph je víc evoluční než behaviorální parametr. Initial runs
/// s 0.5/0.1/0.05 ukázaly "morph and starve" (random brain biasy × rychlý
/// MORPH_RATE → 3× body maintenance dřív než selekce optimalizuje → extinkce).
/// Sprint 80: 0.02 → 0.05 (2.5× rychlejší). Po S78 stable tissue regime má
/// populace zdravé brainy a "morph and starve" risk je nižší. Cells dostávají
/// in-life shape adaptaci jako reálný behavioral lever — např. roztáhnout
/// tělo pri lovu, smrštit při útěku. MORPH_COST_PER_DELTA=2.0 zůstává jako
/// energy circuit-breaker.
pub const MORPH_RATE: f32 = 0.05;
/// Deadzone — pokud |signal| < threshold, morph se nepoužívá (žádná změna,
/// žádný cost). Filtruje šum z random brain biases (mean 0, ~0.5 stddev),
/// jen vědomě silné morph signály prochází.
/// Sprint 26: threshold 0.7 → prob(|tanh(N(0,1))| > 0.7) ≈ 0.38, ~62 % random
/// buněk neumí morphovat. Sprint 80: 0.7 → 0.3 → ~76 % signálů projde. Brain
/// driven shape control je teď reálný (S78 baseline brainy nejsou random).
/// Trénovaný brain pořád dosahuje silných signálů 1.0; uvolnění gating dává
/// víc behavioral expression bez "extinct"-level rizika (MORPH_RATE × dt ×
/// raw_signal je bound, COST_PER_DELTA absorbuje).
pub const MORPH_ACTIVATION_THRESHOLD: f32 = 0.3;
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
/// vytvoří spring bond. Sprint 71 měl 30 (= 0.5 s při 60 Hz). Sprint 76:
/// 30 → 10 (~0.17 s) — Sprint 75 smoke ukázal, že cells se speed 190 mají
/// brief contacts; 30 ticks gating filtroval většinu reálných bond
/// candidates. 10 ticks dovolí formaci i z krátkých but consenting contactů.
/// Risk: nestabilní bondy formující se z náhodného mihem (mitigated tím,
/// že bond_signal threshold zůstává 0.2).
pub const BOND_FORM_TICKS: u32 = 10;
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
pub const BOND_FOOD_SHARE_FRAC: f32 = 0.3;

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
/// Bond count threshold pro immunity. Cells s ≥ tomuto počtu bondů Hunter
/// nemůže atakovat — cluster je „too big to swallow". Sprint 71 měl 3
/// (proto-tissue minimum). Sprint 76: 3 → 2 — Sprint 75 1000-gen smoke
/// ukázal, že 3-bond threshold byl dosažitelný jen 0.2 % cells krátce;
/// cells s evolved asp~12 (1D needles) fyzicky neclusterují s 3 sousedy.
/// 2-bond cluster je „pair / triad" minimum — pořád proto-multicelular,
/// ale dosažitelný pro elongated body shapes (line of pairs).
pub const HUNTER_BOND_IMMUNITY_THRESHOLD: u32 = 2;
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

/// Sprint 71: non-evolving environmental predator. Pohybuje se pseudo-AI
/// (seek nejbližší cell ∈ vision range, jinak random drift). Atakuje cells
/// s `n_bonds() < HUNTER_BOND_IMMUNITY_THRESHOLD` v attack range. Žádný
/// genome, žádný brain, žádná smrt — Hunter je world feature, ne entity
/// pod selekcí.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Hunter {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    /// Stable identifier (procedural, monotonic při init). Nepoužívá se pro
    /// bond resolution — Hunter ≠ Cell.
    pub hunter_id: u64,
}

impl Hunter {
    /// Random init pozice + zero velocity. World-half určuje rozsah.
    pub fn random(rng: &mut impl Rng, world_half: [f32; 3], hunter_id: u64) -> Self {
        Self {
            position: [
                rng.random_range(-world_half[0]..world_half[0]),
                rng.random_range(-world_half[1]..world_half[1]),
                rng.random_range(-world_half[2]..world_half[2]),
            ],
            velocity: [0.0; 3],
            hunter_id,
        }
    }

    /// Per-tick movement integration: target-seek pokud je cell ve vision
    /// range, jinak random drift. `target_pos` je `Some(pos)` z helperu
    /// `nearest_attackable_cell` (caller). World wrap (toroidal xy) aplikován
    /// stejně jako Cell::apply_world_bounce.
    pub fn step(
        &mut self,
        target_pos: Option<[f32; 3]>,
        rng: &mut impl Rng,
        dt: f32,
        world_half: [f32; 3],
    ) {
        // Seek nebo random drift.
        let desired = match target_pos {
            Some(t) => {
                // Min-image vector self→target (toroidal aware). `min_image_delta(a, b)`
                // vrací `b - a`, takže `(self_pos, t)` dá `t - self_pos`.
                let d = min_image_delta(self.position, t, world_half);
                let mag = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if mag > 1e-6 {
                    let inv = HUNTER_MAX_SPEED / mag;
                    [d[0] * inv, d[1] * inv, d[2] * inv]
                } else {
                    [0.0; 3]
                }
            }
            None => [
                rng.random_range(-HUNTER_IDLE_DRIFT..HUNTER_IDLE_DRIFT),
                rng.random_range(-HUNTER_IDLE_DRIFT..HUNTER_IDLE_DRIFT),
                rng.random_range(-HUNTER_IDLE_DRIFT * 0.3..HUNTER_IDLE_DRIFT * 0.3),
            ],
        };
        // Steer velocity → desired s rate-limited acc. HUNTER_ACC = max
        // change v jednotkách/s; dt-scaled cap brání over-shoot při velkých dt.
        let max_delta = HUNTER_ACC * dt;
        for i in 0..3 {
            let want = desired[i] - self.velocity[i];
            self.velocity[i] += want.clamp(-max_delta, max_delta);
        }
        // Clamp speed.
        let speed_sq = self.velocity[0] * self.velocity[0]
            + self.velocity[1] * self.velocity[1]
            + self.velocity[2] * self.velocity[2];
        let max_sq = HUNTER_MAX_SPEED * HUNTER_MAX_SPEED;
        if speed_sq > max_sq {
            let scale = HUNTER_MAX_SPEED / speed_sq.sqrt();
            self.velocity[0] *= scale;
            self.velocity[1] *= scale;
            self.velocity[2] *= scale;
        }
        // Integrate position.
        self.position[0] += self.velocity[0] * dt;
        self.position[1] += self.velocity[1] * dt;
        self.position[2] += self.velocity[2] * dt;
        // Toroidal wrap xy, z bounce (mirror Cell::apply_world_bounce).
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

/// Sprint 71: vrací `Some(cell_index)` nejbližší attackable cell ∈ vision
/// range. „Attackable" = `n_bonds() < HUNTER_BOND_IMMUNITY_THRESHOLD`. Vrací
/// nejbližší **z attackable** cells, ne nejbližší absolutně — Hunter nepronásleduje
/// imune clustery (skill: cluster je viditelný, ale ne lovitelný; Hunter musí
/// najít solo cell). Toroidal-aware přes `min_image_delta`.
///
/// Sprint 84: směrový FOV. `hunter_velocity` určuje forward; cells mimo
/// `HUNTER_VISION_FOV` kuželu nejsou viditelné. Idle hunter (velocity² <
/// `HUNTER_FORWARD_SPEED_THRESHOLD_SQ`) má fallback na omni — bez toho by
/// hunter zaseknutý v 0-velocity stavu nikdy nenašel target.
pub fn nearest_attackable_cell(
    hunter_pos: [f32; 3],
    hunter_velocity: [f32; 3],
    cells: &[Cell],
    world_half: [f32; 3],
) -> Option<usize> {
    let vision_r2 = HUNTER_VISION_RADIUS * HUNTER_VISION_RADIUS;
    let cos_fov = HUNTER_VISION_FOV.cos();
    let speed_sq = hunter_velocity[0] * hunter_velocity[0]
        + hunter_velocity[1] * hunter_velocity[1]
        + hunter_velocity[2] * hunter_velocity[2];
    let cone_active = speed_sq > HUNTER_FORWARD_SPEED_THRESHOLD_SQ;
    let forward = if cone_active {
        let inv = 1.0 / speed_sq.sqrt();
        [
            hunter_velocity[0] * inv,
            hunter_velocity[1] * inv,
            hunter_velocity[2] * inv,
        ]
    } else {
        [0.0; 3]
    };
    let mut best: Option<(usize, f32)> = None;
    for (i, c) in cells.iter().enumerate() {
        if c.n_bonds() >= HUNTER_BOND_IMMUNITY_THRESHOLD {
            continue;
        }
        // Sprint 84: vector from hunter to cell (= c.position − hunter_pos).
        // Pre-Sprint-84 byl `d` drženo jako hunter_pos − c.position (jen pro d²
        // distance, kde znaménko nehraje); cone filter potřebuje směr.
        let d = min_image_delta(hunter_pos, c.position, world_half);
        let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if d2 >= vision_r2 {
            continue;
        }
        if cone_active && !fov_cone_accept(d, d2, forward, cos_fov) {
            continue;
        }
        match best {
            None => best = Some((i, d2)),
            Some((_, bd2)) if d2 < bd2 => best = Some((i, d2)),
            _ => {}
        }
    }
    best.map(|(i, _)| i)
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
    // Sprint 80 Sprint C: structural mutation off by default. Zapnutí přes
    // local `MutationConfig { add_neuron_rate: 0.05, ..MUTATION_CONFIG }`
    // v experimentech.
    add_neuron_rate: 0.0,
    // Sprint 82: FOV gen dormant — Sprint 82 je pure infra (gen + cost factor).
    // Sprint 83: 0.0 → 0.05 — modest drift jako sigma_bond_stiffness (0.3 / 10
    // = 3 % range, sigma_vision_fov 0.05 / 2.88 ≈ 1.7 % FOV range per gen).
    // Pomalejší než tělesné geny aby evoluce stihla najít optimum bez random
    // walku do MIN_VISION_FOV.
    sigma_vision_fov: 0.05,
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
    /// Sprint 80: active hidden neuron count (≤ BRAIN_HIDDEN storage). Default
    /// = BRAIN_HIDDEN_DEFAULT. forward/mutate/crossover/hebbian iterují jen
    /// [0..hidden_n]; dead zone (hidden_n..BRAIN_HIDDEN) drží 0 a do výpočtu
    /// nepřispívá. Strukturální mutace (add/remove neuron) přijdou v dalších
    /// sprintech a budou tuhle hodnotu měnit.
    #[serde(default = "default_hidden_n")]
    pub hidden_n: u32,
    #[serde(with = "serde_arrays_w1")]
    pub w1: [[f32; BRAIN_INPUTS]; BRAIN_HIDDEN],
    pub b1: [f32; BRAIN_HIDDEN],
    #[serde(with = "serde_arrays_w2")]
    pub w2: [[f32; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
    pub b2: [f32; BRAIN_OUTPUTS],
}

fn default_hidden_n() -> u32 {
    BRAIN_HIDDEN_DEFAULT as u32
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
        Self::random_with_hidden(rng, BRAIN_HIDDEN_DEFAULT as u32)
    }

    /// Sprint 80: variable-size random init. `hidden_n` aktivních neuronů,
    /// zbytek storage (BRAIN_HIDDEN..) zero-initialized a do forward passu
    /// nepřispívá. RNG draws happen only for active region — same seed s
    /// hidden_n=BRAIN_HIDDEN_DEFAULT reprodukuje pre-Sprint-80 sekvenci.
    pub fn random_with_hidden(rng: &mut impl Rng, hidden_n: u32) -> Self {
        debug_assert!(
            (hidden_n as usize) >= BRAIN_HIDDEN_MIN
                && (hidden_n as usize) <= BRAIN_HIDDEN,
            "hidden_n {} out of [{}, {}]",
            hidden_n,
            BRAIN_HIDDEN_MIN,
            BRAIN_HIDDEN
        );
        let h_n = hidden_n as usize;
        // Sprint 80 (storage bump): active input width je sensory + hidden_n,
        // ne celá BRAIN_INPUTS storage. Pro default h_n=16: 20+16 = 36, match
        // Sprint A pre-bump RNG sekvence byte-identical.
        let active_inputs = BRAIN_INPUTS_SENSORY + h_n;
        let mut w1 = [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN];
        let mut b1 = [0.0; BRAIN_HIDDEN];
        let mut w2 = [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS];
        let mut b2 = [0.0; BRAIN_OUTPUTS];
        for i in 0..h_n {
            for j in 0..active_inputs {
                w1[i][j] = gaussian(rng);
            }
            b1[i] = gaussian(rng);
        }
        for (row, bias) in w2.iter_mut().zip(b2.iter_mut()) {
            for j in 0..h_n {
                row[j] = gaussian(rng);
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
        Self { hidden_n, w1, b1, w2, b2 }
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
        let h_n = self.hidden_n as usize;
        let mut hidden = [0.0_f32; BRAIN_HIDDEN];
        for i in 0..h_n {
            let mut sum = self.b1[i];
            for (&w, &x) in self.w1[i].iter().zip(inputs.iter()) {
                sum += w * x;
            }
            hidden[i] = sum.tanh();
        }
        let mut out = [0.0_f32; BRAIN_OUTPUTS];
        for ((o, row), &bias) in out.iter_mut().zip(self.w2.iter()).zip(self.b2.iter()) {
            let mut sum = bias;
            for j in 0..h_n {
                sum += row[j] * hidden[j];
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
        let h_n = self.hidden_n as usize;
        for i in 0..h_n {
            let h = last_hidden[i];
            for (w, &x) in self.w1[i].iter_mut().zip(last_inputs.iter()) {
                *w += lr * h * x;
            }
            self.b1[i] += lr * h;
        }
        for (out_o, &o) in self.w2.iter_mut().zip(last_outputs.iter()) {
            for j in 0..h_n {
                out_o[j] += lr * o * last_hidden[j];
            }
        }
        for (b, &o) in self.b2.iter_mut().zip(last_outputs.iter()) {
            *b += lr * o;
        }
    }

    pub fn mutate(&self, rng: &mut impl Rng, sigma: f32) -> Self {
        let mut out = *self;
        let h_n = self.hidden_n as usize;
        let active_inputs = BRAIN_INPUTS_SENSORY + h_n;
        for i in 0..h_n {
            for j in 0..active_inputs {
                out.w1[i][j] += gaussian(rng) * sigma;
            }
            out.b1[i] += gaussian(rng) * sigma;
        }
        for (row, bias) in out.w2.iter_mut().zip(out.b2.iter_mut()) {
            for j in 0..h_n {
                row[j] += gaussian(rng) * sigma;
            }
            *bias += gaussian(rng) * sigma;
        }
        out
    }

    /// Sprint 80 Sprint C: structural mutation — přidá jeden hidden neuron.
    /// Vrací `true` pokud se neuron přidal, `false` pokud cap (`hidden_n ==
    /// BRAIN_HIDDEN`) nebo `BRAIN_HIDDEN_MIN` violation prevented.
    ///
    /// **Init logic (NEAT-style minimal disruption):**
    /// - `w1[new_idx][0..active_inputs]` = gaussian × sigma (small, drift)
    /// - `b1[new_idx]` = gaussian × sigma
    /// - `w2[*][new_idx]` = gaussian × sigma (output contribution starts small)
    /// - Existing neurons NETKNUTÉ (jejich w1[i][20+new_idx] zůstává 0 = no
    ///   incoming connection from new recurrent slot; selekce + future
    ///   weight mutace mohou prokopnout, pokud má smysl).
    ///
    /// `active_inputs = BRAIN_INPUTS_SENSORY + (hidden_n+1)` zahrnuje vlastní
    /// recurrent slot nového neuronu (= připojení k své vlastní paměti).
    pub fn add_neuron(&mut self, rng: &mut impl Rng, sigma: f32) -> bool {
        let new_idx = self.hidden_n as usize;
        if new_idx >= BRAIN_HIDDEN {
            return false;
        }
        let active_inputs = BRAIN_INPUTS_SENSORY + new_idx + 1;
        for j in 0..active_inputs {
            self.w1[new_idx][j] = gaussian(rng) * sigma;
        }
        self.b1[new_idx] = gaussian(rng) * sigma;
        for o in 0..BRAIN_OUTPUTS {
            self.w2[o][new_idx] = gaussian(rng) * sigma;
        }
        self.hidden_n += 1;
        true
    }

    /// Per-row uniform crossover. Each hidden neuron's `w1` row + `b1`
    /// scalar comes from one parent (50/50); same for output neurons. Per-row
    /// rather than per-weight preserves coordinated patterns within a single
    /// neuron's receptive field. Sprint 80: vyžaduje shodný `hidden_n` u
    /// obou rodičů — topology-aware crossover (mismatched hidden_n) přijde
    /// až s další structural mutací (Sprint D+).
    pub fn crossover(a: &Brain, b: &Brain, rng: &mut impl Rng) -> Brain {
        assert_eq!(
            a.hidden_n, b.hidden_n,
            "Brain::crossover requires matching hidden_n (got {} vs {})",
            a.hidden_n, b.hidden_n
        );
        let mut out = *a;
        let h_n = a.hidden_n as usize;
        for i in 0..h_n {
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
    /// Sprint 80 Sprint C: pravděpodobnost `add_neuron` structural mutace
    /// per dítě. 0.0 = topology evoluce vypnutá (default; zachovává Sprint
    /// B byte-identical trajectory). Tuning sweep příští sprint zkusí 0.02,
    /// 0.05, 0.1.
    pub add_neuron_rate: f32,
    /// Sprint 82: gaussian sigma pro `vision_fov` mutaci. `MUTATION_CONFIG`
    /// default = 0 (Sprint 82 pure-infra, FOV gen dormant). Sprint 83+ FOV
    /// aktivuje a tuning experimenty mohou nastavit nenulový drift. Při
    /// drift na MIN_VISION_FOV cells ztrácejí cells/food awareness, ale
    /// platí minimální vision cost — selekční trade-off vstoupí v platnost
    /// až s aktivním cone filterem.
    pub sigma_vision_fov: f32,
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
    /// Sprint 82: půl-úhel kuželu směrového FOV kolem `forward_vector` (rad).
    /// Range `[MIN_VISION_FOV, MAX_VISION_FOV]`. Init = `INITIAL_VISION_FOV`
    /// (= π = full sphere). Sprint 83+ aktivuje cone filter v sensor gather;
    /// energy cost už v Sprint 82 škáluje s `vision_fov_factor(theta)`.
    #[serde(default = "default_vision_fov")]
    pub vision_fov: f32,
    pub brain: Brain,
}

fn default_vision_fov() -> f32 {
    INITIAL_VISION_FOV
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
            // Sprint 82: full-sphere baseline, žádný RNG draw — initial population
            // kompletně omnidirectional. Cost faktor = 1.0, behavior matches
            // pre-Sprint-82 baseline až do prvního sigma_vision_fov > 0 sprintu.
            vision_fov: INITIAL_VISION_FOV,
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
            spike_length: (self.spike_length + gaussian(rng) * cfg.sigma_spike_length)
                .clamp(MIN_SPIKE_LENGTH, MAX_SPIKE_LENGTH),
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
            brain: {
                let mut b = self.brain.mutate(rng, cfg.sigma_brain);
                // Sprint 80 Sprint C: structural mutace. Default rate=0.0 →
                // bool draw probíhá ale add_neuron se nikdy nevolá → žádný
                // RNG drift vůči Sprint B baseline (jeden extra `rng.random::<f32>()`
                // by trajectory shiftnul). Branchnu na rate>0 abych draw vůbec
                // nepřípravil v default scenáři.
                if cfg.add_neuron_rate > 0.0
                    && rng.random::<f32>() < cfg.add_neuron_rate
                {
                    b.add_neuron(rng, ADD_NEURON_SIGMA);
                }
                b
            },
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
    /// Sprint 80: bistabilní cell-state [0,1]. Není v genomu — dědí se z
    /// parenta s šumem, takže slouží jako fenotypová paměť napříč generacemi.
    /// Pozitivní feedback okolo 0.5 + bias od `n_bonds()` produkuje dva
    /// stabilní attractory: ~0 (selfish) a ~1 (altruist). Reguluje food share
    /// frakci uvnitř bonded clusteru.
    pub cell_state: f32,
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
            // Sprint 80: kick okolo 0.5 (nestabilní fixed point feedback),
            // aby cells nezamrzly přesně v neutrální zóně. Append na konci
            // RNG sekvence — pre-Sprint-80 draws (direction, pos_z, pos_x,
            // pos_y) zůstávají v identickém pořadí, jen za nimi je 1 nový.
            cell_state: 0.5 + rng.random_range(-CELL_STATE_INIT_KICK..CELL_STATE_INIT_KICK),
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
    pub fn step(
        &mut self,
        dt: f32,
        world_half: [f32; 3],
        tick: u64,
        generation: u64,
        physics: &PhysicsConfig,
    ) {
        // Sprint 42: aging + cooldown decrement na začátku ticku, aby
        // apply_energy_costs viděl current age v ramp formuli.
        self.age = self.age.saturating_add(1);
        if self.reproduce_cooldown_ticks > 0 {
            self.reproduce_cooldown_ticks -= 1;
        }
        self.integrate_kinematics(dt, world_half);
        self.apply_anisotropic_drag(dt, physics);
        self.apply_angular_drag(dt, physics);
        // Sprint 86: tick + generation propagace pro time-varying thermal
        // (diurnal + seasonal cykly v `temperature_at_z`).
        self.apply_energy_costs(dt, world_half, tick, generation, physics);
        self.apply_world_bounce(world_half);
        self.update_cell_state(dt);
    }

    /// Sprint 80: bistabilní cell-state dynamika. Pure deterministic (no RNG)
    /// — RNG by zde rozbil seed reproducibility step()u. Šum vstupuje jen
    /// při dědičnosti v `make_mating_child`.
    ///
    /// Update rule:
    ///   s' = s + K · (s − 0.5) · dt + bias · n_bonds · dt
    ///
    /// Pozitivní feedback `(s − 0.5)` táhne stav od nestabilního fixed
    /// pointu k 0 nebo 1 podle aktuální orientace. `n_bonds` bias konzistentně
    /// tlačí cells s víc bondy směrem k altruist attractoru — tissue cells
    /// commitnou k altruismu, solo cells driftují k selfish.
    fn update_cell_state(&mut self, dt: f32) {
        let n_bonds = self.n_bonds() as f32;
        let feedback = CELL_STATE_FEEDBACK_K * (self.cell_state - 0.5) * dt;
        let env_drive = CELL_STATE_BOND_BIAS * n_bonds * dt;
        self.cell_state = (self.cell_state + feedback + env_drive).clamp(0.0, 1.0);
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
    ///
    /// Sprint 85: všechny drains škálují `metabolism_factor(T)` kde T je
    /// teplota na cell pozici. Warm cells (z = +half) drain ~2.46× rychleji
    /// než REF, cold cells (z = -half) drain ~0.41×. Niche separation by
    /// depth — selekce favorizuje cells co najdou energy-optimal z-vrstvu.
    /// Při `world_half[2] = 0` (pre-3D baseline) vrací temperature_at_z
    /// REF_TEMP → metabolism = 1.0 → drain backward-compat.
    fn apply_energy_costs(
        &mut self,
        dt: f32,
        world_half: [f32; 3],
        tick: u64,
        generation: u64,
        physics: &PhysicsConfig,
    ) {
        let metabolism = metabolism_factor(temperature_at_z(
            self.position[2],
            world_half,
            tick,
            generation,
        ));
        let dt_eff = dt * metabolism;
        // Sprint 33: v_mag_sq zahrnuje 3D (vz != 0 v Sprint 35+).
        let v_mag_sq =
            self.velocity[0].powi(2) + self.velocity[1].powi(2) + self.velocity[2].powi(2);
        self.energy -= v_mag_sq * physics.energy_cost_per_v_sq * dt_eff;
        let av = self.angular_velocity;
        let eff_r = self.phenotype.effective_radius();
        self.energy -= eff_r * eff_r * av * av * physics.angular_energy_cost * dt_eff;
        // Sprint 82: cost ∝ radius × fov_factor. Full sphere (fov = π) → factor
        // 1.0 → identický s pre-Sprint-82. Užší kužel platí míň, ale Sprint 83
        // aktivuje cone filter v sensor gather → trade-off info-loss vs energy.
        let fov_factor = vision_fov_factor(self.genome.vision_fov);
        self.energy -=
            self.genome.vision_radius * physics.vision_cost_per_radius * fov_factor * dt_eff;
        // Sprint 34: maintenance ∝ 3D volume = length×width×height.
        // Sprint 42: aging ramp — starší cells platí postupně víc per volume unit.
        let age_sec = self.age as f32 / FIXED_TIMESTEP_HZ;
        let aging_factor = 1.0 + AGE_DECAY_PER_SEC * age_sec;
        self.energy -=
            self.phenotype.volume() * physics.body_cost_factor * aging_factor * dt_eff;
        self.energy -= self.phenotype.spike_length * SPIKE_COST_PER_SEC * dt_eff;
        // Sprint 41: shell maintenance — defensive armor stojí víc než spike,
        // protože pokrývá celý povrch.
        self.energy -= self.phenotype.shell_thickness * SHELL_COST_PER_SEC * dt_eff;
        // Sprint 27 attack maintenance: cost ∝ max(0, output[6]).
        let attack_strength = self.last_outputs[6].max(0.0);
        self.energy -= attack_strength * ATTACK_COST_PER_SEC * dt_eff;
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
            hidden_n: BRAIN_HIDDEN_DEFAULT as u32,
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
            vision_fov: INITIAL_VISION_FOV,
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
            add_neuron_rate: 0.0,
            sigma_vision_fov: 0.0,
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
            vision_fov: INITIAL_VISION_FOV,
            brain: Brain {
                hidden_n: BRAIN_HIDDEN_DEFAULT as u32,
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
            add_neuron_rate: 0.0,
            sigma_vision_fov: 10.0,
        };
        for _ in 0..1000 {
            let m = g.mutate(&mut rng, &cfg);
            assert!(m.max_speed >= MIN_SPEED);
            assert!(m.max_speed <= MAX_SPEED, "Sprint 73: speed cap respected");
            assert!(m.color_hue >= 0.0 && m.color_hue < HUE_RANGE);
            assert!(m.vision_radius >= MIN_VISION);
            assert!(m.turn_rate >= MIN_TURN_RATE);
            assert!((MIN_BODY_LENGTH..=MAX_BODY_LENGTH).contains(&m.body_length));
            assert!((MIN_BODY_WIDTH..=MAX_BODY_WIDTH).contains(&m.body_width));
            assert!((MIN_SPIKE_LENGTH..=MAX_SPIKE_LENGTH).contains(&m.spike_length));
            assert!((MIN_VISION_FOV..=MAX_VISION_FOV).contains(&m.vision_fov));
        }
    }

    #[test]
    fn vision_fov_dormant_preserves_rng_sequence() {
        // Sprint 82 reproducibility guard: při `sigma_vision_fov = 0`
        // (S82 default) musí mutate přeskočit gaussian draw pro FOV gen,
        // jinak Sprint 82 baseline rozejde s pre-Sprint-82 CSV. Verifikuje
        // shodu RNG stavu mezi dormant cestou (krátkou) a aktivní cestou
        // (sigma > 0) po injekci přesně 2 u32 draws (gaussian = 2 u32).
        let mut rng_zero = StdRng::seed_from_u64(0xC0FFEE);
        let mut rng_active = StdRng::seed_from_u64(0xC0FFEE);
        let cfg_zero = MutationConfig {
            sigma_vision_fov: 0.0,
            ..MUTATION_CONFIG
        };
        let cfg_active = MutationConfig {
            sigma_vision_fov: 0.05,
            ..MUTATION_CONFIG
        };
        let g = dummy_genome();
        let _ = g.mutate(&mut rng_zero, &cfg_zero);
        let _ = g.mutate(&mut rng_active, &cfg_active);
        let _: u32 = rng_zero.random();
        let _: u32 = rng_zero.random();
        let next_zero: u32 = rng_zero.random();
        let next_active: u32 = rng_active.random();
        assert_eq!(
            next_zero, next_active,
            "sigma_vision_fov = 0 musí ušetřit přesně 2 u32 RNG draws (gaussian); \
             jinak Sprint 82 nezachová pre-S82 reproducibility"
        );
    }

    #[test]
    fn vision_fov_crossover_skips_rng_when_equal() {
        // Sprint 82 reproducibility guard: pokud oba parents mají identické
        // vision_fov (což je pravda v initial pop kde všichni = INITIAL_VISION_FOV),
        // crossover musí přeskočit bool draw. Verifikuje shodu RNG stavu mezi
        // equal-values cestou (krátkou) a different-values cestou (s draw)
        // po injekci 1 bool draw.
        let mut rng_eq = StdRng::seed_from_u64(0xBEEF);
        let mut rng_diff = StdRng::seed_from_u64(0xBEEF);
        let mut a = dummy_genome();
        let mut b = dummy_genome();
        a.vision_fov = INITIAL_VISION_FOV;
        b.vision_fov = INITIAL_VISION_FOV;
        let _ = Genome::crossover(&a, &b, &mut rng_eq);
        b.vision_fov = MIN_VISION_FOV;
        let _ = Genome::crossover(&a, &b, &mut rng_diff);
        let _ = rng_eq.random::<bool>();
        let next_eq: u32 = rng_eq.random();
        let next_diff: u32 = rng_diff.random();
        assert_eq!(
            next_eq, next_diff,
            "crossover s a.vision_fov == b.vision_fov musí ušetřit přesně 1 bool draw"
        );
    }

    #[test]
    fn temperature_at_z_endpoints() {
        let half = [960.0, 540.0, 50.0];
        // Sprint 86: tick=0, gen=0 → seasonal sin(0)=0, diurnal sin(0)=0,
        // takže static gradient endpoints zůstávají identické s pre-Sprint-86.
        // Top z = +half → THERMAL_TOP.
        assert!((temperature_at_z(50.0, half, 0, 0) - THERMAL_TOP).abs() < 1e-4);
        // Bottom z = -half → THERMAL_BOTTOM.
        assert!((temperature_at_z(-50.0, half, 0, 0) - THERMAL_BOTTOM).abs() < 1e-4);
        // Mid z = 0 → exact midpoint.
        let mid = (THERMAL_TOP + THERMAL_BOTTOM) * 0.5;
        assert!((temperature_at_z(0.0, half, 0, 0) - mid).abs() < 1e-4);
        // Out-of-bounds z → clamp na endpoints.
        assert!((temperature_at_z(1000.0, half, 0, 0) - THERMAL_TOP).abs() < 1e-4);
        assert!((temperature_at_z(-1000.0, half, 0, 0) - THERMAL_BOTTOM).abs() < 1e-4);
        // world_half[2] = 0 (pre-3D baseline) → ref temp fallback (no-op pro
        // metabolism). Důležité pro backward-compat pre-Sprint-33 testů.
        let flat = [960.0, 540.0, 0.0];
        assert!((temperature_at_z(0.0, flat, 0, 0) - THERMAL_REF_TEMP).abs() < 1e-4);
    }

    #[test]
    fn temperature_diurnal_surface_oscillates() {
        // Sprint 86: surface (z = +half) osciluje ±DIURNAL_AMP přes 1 day.
        // Bottom (z = -half) zůstává stabilní (normalized = 0 → diurnal × 0).
        let half = [960.0, 540.0, 50.0];
        let period = THERMAL_DIURNAL_PERIOD_TICKS;
        // Quarter-day → sin(π/2) = +1 → surface = TOP + AMP, bottom = BOTTOM.
        let t_q = period / 4;
        let surf_q = temperature_at_z(50.0, half, t_q, 0);
        let bot_q = temperature_at_z(-50.0, half, t_q, 0);
        assert!((surf_q - (THERMAL_TOP + THERMAL_DIURNAL_AMP)).abs() < 0.05);
        assert!((bot_q - THERMAL_BOTTOM).abs() < 0.05);
        // Three-quarter-day → sin(3π/2) = -1 → surface = TOP − AMP.
        let t_3q = 3 * period / 4;
        let surf_3q = temperature_at_z(50.0, half, t_3q, 0);
        assert!((surf_3q - (THERMAL_TOP - THERMAL_DIURNAL_AMP)).abs() < 0.05);
        // Full day → sin(2π) = 0 → matches initial.
        let surf_full = temperature_at_z(50.0, half, period, 0);
        assert!((surf_full - THERMAL_TOP).abs() < 0.01);
    }

    #[test]
    fn temperature_seasonal_uniform_shift() {
        // Sprint 86: seasonal aplikuje stejný offset napříč all z (uniform shift).
        // Surface i bottom posun stejně. Period = CYCLE_GEN_PERIOD = 50 gen.
        let half = [960.0, 540.0, 50.0];
        let period = CYCLE_GEN_PERIOD;
        // Quarter-cycle → sin(π/2) = 1 → +SEASONAL_AMP shift.
        let surf_q = temperature_at_z(50.0, half, 0, period / 4);
        let bot_q = temperature_at_z(-50.0, half, 0, period / 4);
        assert!((surf_q - (THERMAL_TOP + THERMAL_SEASONAL_AMP)).abs() < 0.05);
        assert!((bot_q - (THERMAL_BOTTOM + THERMAL_SEASONAL_AMP)).abs() < 0.05);
        // Half-cycle → sin(π) = 0 → no shift.
        let surf_half = temperature_at_z(50.0, half, 0, period / 2);
        assert!((surf_half - THERMAL_TOP).abs() < 0.05);
        // Three-quarter-cycle → sin(3π/2) = -1 → -SEASONAL_AMP shift.
        let surf_3q = temperature_at_z(50.0, half, 0, 3 * period / 4);
        assert!((surf_3q - (THERMAL_TOP - THERMAL_SEASONAL_AMP)).abs() < 0.05);
    }

    #[test]
    fn temperature_combined_seasonal_and_diurnal() {
        // Sprint 86: seasonal i diurnal jsou aditivní. Quarter-day +
        // quarter-season → surface = TOP + DIURNAL_AMP + SEASONAL_AMP,
        // bottom = BOTTOM + SEASONAL_AMP.
        let half = [960.0, 540.0, 50.0];
        let t_q = THERMAL_DIURNAL_PERIOD_TICKS / 4;
        let g_q = CYCLE_GEN_PERIOD / 4;
        let surf = temperature_at_z(50.0, half, t_q, g_q);
        let expected = THERMAL_TOP + THERMAL_DIURNAL_AMP + THERMAL_SEASONAL_AMP;
        assert!(
            (surf - expected).abs() < 0.05,
            "combined surface {} ≠ expected {}",
            surf,
            expected
        );
        let bot = temperature_at_z(-50.0, half, t_q, g_q);
        let expected_bot = THERMAL_BOTTOM + THERMAL_SEASONAL_AMP;
        assert!((bot - expected_bot).abs() < 0.05);
    }

    #[test]
    fn metabolism_factor_q10_ratio() {
        // Q10 = 2.0 → biologické rychlosti přesně 2× per +10 sim-units T.
        let m_ref = metabolism_factor(THERMAL_REF_TEMP);
        assert!((m_ref - 1.0).abs() < 1e-4, "ref temp musí dát factor 1.0");
        let m_plus_10 = metabolism_factor(THERMAL_REF_TEMP + 10.0);
        assert!(
            (m_plus_10 - THERMAL_Q10).abs() < 1e-4,
            "+10 musí dát Q10 (= 2.0), got {m_plus_10}"
        );
        let m_minus_10 = metabolism_factor(THERMAL_REF_TEMP - 10.0);
        assert!(
            (m_minus_10 - 1.0 / THERMAL_Q10).abs() < 1e-4,
            "-10 musí dát 1/Q10 (= 0.5), got {m_minus_10}"
        );
        // Endpoints by měly dát ratio top:bottom = Q10^((TOP-BOT)/10)
        let m_top = metabolism_factor(THERMAL_TOP);
        let m_bot = metabolism_factor(THERMAL_BOTTOM);
        let expected_ratio = THERMAL_Q10.powf((THERMAL_TOP - THERMAL_BOTTOM) / 10.0);
        assert!(
            ((m_top / m_bot) - expected_ratio).abs() < 1e-3,
            "top/bottom ratio {} vs expected {}",
            m_top / m_bot,
            expected_ratio
        );
    }

    #[test]
    fn apply_energy_costs_scales_with_temperature() {
        // Sprint 85: cell na warm depth (z = +half) drain rychleji než cell na
        // cold depth (z = -half). Při shodné velocity / body / vision platí
        // ratio drain = metabolism(top) / metabolism(bottom) ≈ 2.46 / 0.41 ≈ 6×.
        let half = [1000.0, 1000.0, 50.0];
        let physics = no_drag_physics(0.001, 0.0);
        let mut warm = base_cell();
        warm.position = [0.0, 0.0, 50.0]; // top → warmest
        warm.velocity = [60.0, 0.0, 0.0];
        let mut cold = base_cell();
        cold.position = [0.0, 0.0, -50.0]; // bottom → coldest
        cold.velocity = [60.0, 0.0, 0.0];
        warm.step(1.0, half, 0, 0, &physics);
        cold.step(1.0, half, 0, 0, &physics);
        let warm_drain = 100.0 - warm.energy;
        let cold_drain = 100.0 - cold.energy;
        let ratio = warm_drain / cold_drain;
        let expected = metabolism_factor(THERMAL_TOP) / metabolism_factor(THERMAL_BOTTOM);
        assert!(
            (ratio - expected).abs() < 0.05,
            "warm/cold drain ratio {ratio} ≠ expected {expected}"
        );
    }

    #[test]
    fn vision_fov_factor_endpoints() {
        // Full sphere (theta = π) → solid angle = 4π str → factor = 1.0.
        assert!((vision_fov_factor(MAX_VISION_FOV) - 1.0).abs() < 1e-6);
        // Hemisphere (theta = π/2) → solid angle = 2π str → factor = 0.5.
        let half = vision_fov_factor(core::f32::consts::PI * 0.5);
        assert!((half - 0.5).abs() < 1e-6, "got {half}");
        // Narrow cone (theta = 0) → factor = 0.
        assert!(vision_fov_factor(0.0).abs() < 1e-6);
        // Clamp: above π saturates na 1.0 (kdyby někdo poslal 2π omylem).
        assert!((vision_fov_factor(core::f32::consts::PI * 2.0) - 1.0).abs() < 1e-6);
        // Monotonic mezi krajními body.
        let q = vision_fov_factor(core::f32::consts::PI * 0.25);
        assert!(q > 0.0 && q < 0.5);
    }

    #[test]
    fn fov_cone_accept_basic_directions() {
        let fwd = [1.0_f32, 0.0, 0.0];
        // Quarter-circle FOV: half-angle = π/4 → cos = ~0.707.
        let cos_q = (core::f32::consts::PI * 0.25).cos();
        // Target přímo vpředu — vždy uvnitř.
        let front = [10.0_f32, 0.0, 0.0];
        assert!(fov_cone_accept(front, 100.0, fwd, cos_q));
        // Target přímo vpravo (90° offset) — mimo π/4 kuželu.
        let side = [0.0_f32, 10.0, 0.0];
        assert!(!fov_cone_accept(side, 100.0, fwd, cos_q));
        // Target přímo vzadu — mimo.
        let back = [-10.0_f32, 0.0, 0.0];
        assert!(!fov_cone_accept(back, 100.0, fwd, cos_q));
        // Hemisphere FOV (cos = 0) — front + side accepted, back rejected.
        let cos_h = 0.0_f32;
        assert!(fov_cone_accept(front, 100.0, fwd, cos_h));
        // Side je přesně na hranici (dot = 0 = cos_h) → accept.
        assert!(fov_cone_accept(side, 100.0, fwd, cos_h));
        assert!(!fov_cone_accept(back, 100.0, fwd, cos_h));
        // Degenerate target na cell pozici — vždy accept.
        assert!(fov_cone_accept([0.0, 0.0, 0.0], 0.0, fwd, cos_q));
        // Full sphere (cos = -1) — vše accept včetně back.
        assert!(fov_cone_accept(back, 100.0, fwd, -1.0));
    }

    #[test]
    fn fov_cone_works_in_3d() {
        // Heading podél +X, cell s pitch +π/4 → forward má kladnou Z komponentu.
        // Test, že target nahoře-vpředu projde, target dole-vpředu padne ven
        // u úzkého kuželu.
        let fwd = forward_vector(0.0, core::f32::consts::PI * 0.25);
        let cos_q = (core::f32::consts::PI * 0.25).cos();
        let up_front = [10.0_f32, 0.0, 10.0];
        let down_front = [10.0_f32, 0.0, -10.0];
        let d2 = 200.0;
        assert!(fov_cone_accept(up_front, d2, fwd, cos_q));
        assert!(!fov_cone_accept(down_front, d2, fwd, cos_q));
    }

    #[test]
    fn vision_fov_narrows_energy_cost() {
        // Sprint 82: užší FOV → menší cost. Hemisphere (factor 0.5) drained
        // přesně poloviční energy než full sphere (factor 1.0) při stejném
        // vision_radius a stejném dt.
        let mut wide = base_cell();
        wide.genome.vision_fov = MAX_VISION_FOV;
        let mut narrow = base_cell();
        narrow.genome.vision_fov = core::f32::consts::PI * 0.5;
        let physics = no_drag_physics(0.0, 0.05);
        wide.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
        narrow.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
        let wide_drain = 100.0 - wide.energy;
        let narrow_drain = 100.0 - narrow.energy;
        // Vision part: wide = 40 × 0.05 × 1.0 = 2.0, narrow = 40 × 0.05 × 0.5 = 1.0.
        // Ostatní drain (body, motion, …) je 0 v no_drag_physics.
        assert!((wide_drain - 2.0).abs() < 1e-4, "wide drain {wide_drain}");
        assert!((narrow_drain - 1.0).abs() < 1e-4, "narrow drain {narrow_drain}");
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
            cell_state: 0.5,
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
        cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &no_drag_physics(0.001, 0.05));
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
        cell.step(1.0, [100.0, 100.0, 0.0], 0, 0, &no_drag_physics(0.0, 0.0));
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
        cell.step(1.0, [100.0, 100.0, 0.0], 0, 0, &no_drag_physics(0.0, 0.0));
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
        cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
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
        cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
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
        cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
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
        cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
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
            vision_fov: MIN_VISION_FOV,
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
            vision_fov: MAX_VISION_FOV,
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
            assert!(c.vision_fov == MIN_VISION_FOV || c.vision_fov == MAX_VISION_FOV);
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
        // hidden = [1.0; hidden_n], output = [1.0; OUT], reward = 1.0, lr = 0.1
        // Δb1[i] = 0.1 × 1.0 × hidden[i] = 0.1 pro i < hidden_n; 0 jinak
        // Δb2[i] = 0.1 × 1.0 × output[i] = 0.1
        brain.hebbian_update(
            &[0.0; BRAIN_INPUTS],
            &[1.0; BRAIN_HIDDEN],
            &[1.0; BRAIN_OUTPUTS],
            1.0,
            0.1,
        );
        // Sprint 80: hebbian bounded by hidden_n. Dead zone b1 stays at init (0).
        let h_n = brain.hidden_n as usize;
        for &b in &brain.b1[..h_n] {
            assert!((b - 0.1).abs() < 1e-5, "active b1 got {}", b);
        }
        for &b in &brain.b1[h_n..] {
            assert_eq!(b, 0.0, "dead-zone b1 must stay 0");
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
        // Sprint 79: seeded RNG (pre-S57 era používalo `rand::rng()` thread-local
        // → flaky napříč CI, ~5 % run failures kdyby gaussian sampling ojediněle
        // posunul mean pod 0.3). Fixed seed dělá test deterministický.
        let mut rng = StdRng::seed_from_u64(42);
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
            hidden_n: BRAIN_HIDDEN_DEFAULT as u32,
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
    fn brain_random_sets_default_hidden_n() {
        let mut rng = StdRng::seed_from_u64(7);
        let b = Brain::random(&mut rng);
        assert_eq!(b.hidden_n as usize, BRAIN_HIDDEN_DEFAULT);
    }

    #[test]
    fn brain_random_with_hidden_zeros_dead_zone() {
        let mut rng = StdRng::seed_from_u64(7);
        let h_n: u32 = 8;
        assert!((h_n as usize) < BRAIN_HIDDEN, "test assumes h_n < storage");
        let b = Brain::random_with_hidden(&mut rng, h_n);
        assert_eq!(b.hidden_n, h_n);
        // Dead zone w1[h_n..] / b1[h_n..] / w2[*][h_n..] must stay 0 — random
        // initialization touched only active region.
        for i in (h_n as usize)..BRAIN_HIDDEN {
            assert_eq!(b.b1[i], 0.0, "b1[{}] should be 0", i);
            for &w in b.w1[i].iter() {
                assert_eq!(w, 0.0, "w1[{}][..] should be 0", i);
            }
        }
        for o in 0..BRAIN_OUTPUTS {
            for j in (h_n as usize)..BRAIN_HIDDEN {
                assert_eq!(b.w2[o][j], 0.0, "w2[{}][{}] should be 0", o, j);
            }
        }
    }

    #[test]
    fn brain_mutate_preserves_hidden_n_and_dead_zone() {
        let mut rng = StdRng::seed_from_u64(11);
        let h_n: u32 = 6;
        let parent = Brain::random_with_hidden(&mut rng, h_n);
        let child = parent.mutate(&mut rng, 0.5);
        assert_eq!(child.hidden_n, h_n, "hidden_n must survive mutation");
        // Dead zone untouched (no gaussian draws applied to inactive rows).
        for i in (h_n as usize)..BRAIN_HIDDEN {
            assert_eq!(child.b1[i], parent.b1[i]);
            assert_eq!(child.w1[i], parent.w1[i]);
        }
        for o in 0..BRAIN_OUTPUTS {
            for j in (h_n as usize)..BRAIN_HIDDEN {
                assert_eq!(child.w2[o][j], parent.w2[o][j]);
            }
        }
    }

    #[test]
    #[should_panic(expected = "matching hidden_n")]
    fn brain_crossover_panics_on_mismatched_hidden_n() {
        let mut rng = StdRng::seed_from_u64(13);
        let a = Brain::random_with_hidden(&mut rng, 8);
        let b = Brain::random_with_hidden(&mut rng, 12);
        let _ = Brain::crossover(&a, &b, &mut rng);
    }

    #[test]
    fn brain_storage_cap_above_default_with_room_for_growth() {
        // Sprint 80 (storage bump): BRAIN_HIDDEN je storage cap, default je
        // initial active. Rozdíl = headroom pro structural mutace.
        assert!(
            BRAIN_HIDDEN > BRAIN_HIDDEN_DEFAULT,
            "BRAIN_HIDDEN ({}) must be > BRAIN_HIDDEN_DEFAULT ({}) to leave room for add_neuron",
            BRAIN_HIDDEN,
            BRAIN_HIDDEN_DEFAULT
        );
        assert!(
            BRAIN_HIDDEN >= BRAIN_HIDDEN_DEFAULT + 8,
            "headroom < 8 neurons: structural mutace bude rychle narážet na cap"
        );
    }

    #[test]
    fn add_neuron_increments_hidden_n() {
        let mut rng = StdRng::seed_from_u64(31);
        let mut b = Brain::random_with_hidden(&mut rng, BRAIN_HIDDEN_DEFAULT as u32);
        let h_before = b.hidden_n;
        let added = b.add_neuron(&mut rng, ADD_NEURON_SIGMA);
        assert!(added);
        assert_eq!(b.hidden_n, h_before + 1);
    }

    #[test]
    fn add_neuron_returns_false_at_storage_cap() {
        let mut rng = StdRng::seed_from_u64(33);
        let mut b = Brain::random_with_hidden(&mut rng, BRAIN_HIDDEN as u32);
        assert_eq!(b.hidden_n as usize, BRAIN_HIDDEN);
        let added = b.add_neuron(&mut rng, ADD_NEURON_SIGMA);
        assert!(!added, "add_neuron at cap must return false");
        assert_eq!(b.hidden_n as usize, BRAIN_HIDDEN, "cap respected");
    }

    #[test]
    fn add_neuron_initializes_active_region_only() {
        let mut rng = StdRng::seed_from_u64(37);
        let mut b = Brain::random_with_hidden(&mut rng, BRAIN_HIDDEN_DEFAULT as u32);
        let new_idx = b.hidden_n as usize;
        let active_inputs = BRAIN_INPUTS_SENSORY + new_idx + 1;
        let _ = b.add_neuron(&mut rng, ADD_NEURON_SIGMA);
        // New neuron's row [new_idx] active part [0..active_inputs] should be
        // gaussian-initialized (some non-zero values expected). Dead-zone of
        // that same row [active_inputs..BRAIN_INPUTS] should remain 0.
        let any_active_nonzero = b.w1[new_idx][..active_inputs]
            .iter()
            .any(|&w| w != 0.0);
        assert!(any_active_nonzero, "new neuron active w1 row all-zero");
        for &w in &b.w1[new_idx][active_inputs..] {
            assert_eq!(w, 0.0, "new neuron dead-cols must stay 0");
        }
    }

    #[test]
    fn add_neuron_preserves_existing_neurons() {
        let mut rng = StdRng::seed_from_u64(41);
        let mut b = Brain::random_with_hidden(&mut rng, BRAIN_HIDDEN_DEFAULT as u32);
        let snapshot_w1: Vec<_> = b.w1[..BRAIN_HIDDEN_DEFAULT].to_vec();
        let snapshot_b1 = b.b1;
        let snapshot_b2 = b.b2;
        // Snapshot w2 active cols only — dead col at new_idx will get populated.
        let snapshot_w2_active: Vec<Vec<f32>> = b
            .w2
            .iter()
            .map(|row| row[..BRAIN_HIDDEN_DEFAULT].to_vec())
            .collect();
        let _ = b.add_neuron(&mut rng, ADD_NEURON_SIGMA);
        // Existing neurons (rows 0..BRAIN_HIDDEN_DEFAULT) untouched.
        for (i, expected) in snapshot_w1.iter().enumerate() {
            assert_eq!(&b.w1[i], expected, "w1[{}] should be unchanged", i);
        }
        for i in 0..BRAIN_HIDDEN_DEFAULT {
            assert_eq!(b.b1[i], snapshot_b1[i], "b1[{}] should be unchanged", i);
        }
        // b2 unchanged (no contribution from add_neuron).
        assert_eq!(b.b2, snapshot_b2);
        // w2 active cols (existing neurons' connections) unchanged.
        for o in 0..BRAIN_OUTPUTS {
            for h in 0..BRAIN_HIDDEN_DEFAULT {
                assert_eq!(b.w2[o][h], snapshot_w2_active[o][h]);
            }
        }
    }

    #[test]
    fn genome_mutate_with_rate_one_grows_brain_to_cap() {
        let mut rng = StdRng::seed_from_u64(43);
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
            sigma_shell: 0.0,
            sigma_brain: 0.0,
            adhesion_flip_rate: 0.0,
            sigma_bond_stiffness: 0.0,
            sigma_bond_damping: 0.0,
            add_neuron_rate: 1.0,
            sigma_vision_fov: 0.0,
        };
        let mut current = g;
        // Iteruj N >= (BRAIN_HIDDEN - DEFAULT). Po naplnění capu se další
        // add_neuron volá, ale vrací false → hidden_n se nezvyšuje.
        for _ in 0..(BRAIN_HIDDEN - BRAIN_HIDDEN_DEFAULT + 4) {
            current = current.mutate(&mut rng, &cfg);
        }
        assert_eq!(
            current.brain.hidden_n as usize,
            BRAIN_HIDDEN,
            "brain musí dosáhnout cap (BRAIN_HIDDEN={}) ale hidden_n={}",
            BRAIN_HIDDEN,
            current.brain.hidden_n
        );
    }

    #[test]
    fn brain_hidden_n_above_default_forward_uses_padded_storage() {
        // Sprint B: brain s hidden_n > BRAIN_HIDDEN_DEFAULT používá rozšířený
        // storage. Forward output musí brát v potaz nové aktivní neurony.
        let h_n: u32 = (BRAIN_HIDDEN_DEFAULT as u32) + 4;
        assert!(
            (h_n as usize) <= BRAIN_HIDDEN,
            "test config: h_n {} must be ≤ BRAIN_HIDDEN {}",
            h_n,
            BRAIN_HIDDEN
        );
        let mut rng = StdRng::seed_from_u64(17);
        let brain_default = Brain::random_with_hidden(&mut rng, BRAIN_HIDDEN_DEFAULT as u32);
        let brain_extended = Brain::random_with_hidden(&mut rng, h_n);
        let inputs = [0.5_f32; BRAIN_INPUTS];
        let out_default = brain_default.forward(&inputs);
        let out_extended = brain_extended.forward(&inputs);
        // Two different brains with non-overlapping random init should produce
        // different outputs. Sanity check že padded storage není no-op.
        let any_diff = out_default
            .iter()
            .zip(out_extended.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff, "extended hidden_n produced identical output");
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
        // Width clampuje na MIN_BODY_WIDTH (0.8), takže |Δ| pro width je
        // 1.0 - 0.8 = 0.2. Total |Δ| = 0.8 (length) + 0.2 (width clamped)
        // + 0.0 (height: signal=0) + 0.8 (spike) = 1.8.
        let delta = phen.apply_morph([0.8, -0.8, 0.0, 0.8], 1.0, 1.0);
        assert!((delta - 1.8).abs() < 1e-5, "got {}", delta);
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
        forward.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
        sideways.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
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
        cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
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
        cell.step(1.0, [1000.0, 1000.0, 1000.0], 0, 0, &no_drag_physics(0.0, 0.0));
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
        cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
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
        cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
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
            add_neuron_rate: 0.0,
            sigma_vision_fov: 0.0,
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
        young.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
        let young_drain = young_energy_before - young.energy;

        // Cell at age 600 (= 10s) → factor 1 + 0.005×10 = 1.05.
        let mut old = base_cell();
        old.age = 600;
        let old_energy_before = old.energy;
        old.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
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
        cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &no_drag_physics(0.0, 0.0));
        assert_eq!(cell.age, 1);
        cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &no_drag_physics(0.0, 0.0));
        assert_eq!(cell.age, 2);
    }

    #[test]
    fn cooldown_decrements_per_step() {
        let mut cell = base_cell();
        cell.reproduce_cooldown_ticks = 5;
        cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &no_drag_physics(0.0, 0.0));
        assert_eq!(cell.reproduce_cooldown_ticks, 4);
    }

    #[test]
    fn cooldown_does_not_underflow() {
        let mut cell = base_cell();
        cell.reproduce_cooldown_ticks = 0;
        cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &no_drag_physics(0.0, 0.0));
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
    fn pick_cluster_parent_prefers_bonded_matching_adhesion() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut a = Cell::random(&mut rng, [960.0, 540.0, 50.0], 0, 0, 0);
        let mut b = Cell::random(&mut rng, [960.0, 540.0, 50.0], 1, 0, 1);
        a.genome.adhesion_type = 3;
        b.genome.adhesion_type = 5;
        a.bonds[0] = Some(Bond {
            other_cell_id: 99,
            rest_length: 5.0,
            stiffness: BOND_STIFFNESS,
            damping: 0.6,
            age_ticks: 0,
        });
        // Child adhesion=3 → match s parent_a, který má bond.
        let pick = pick_cluster_parent(&a, &b, 3);
        assert!(pick.is_some());
        assert_eq!(pick.unwrap().cell_id, a.cell_id);
    }

    #[test]
    fn pick_cluster_parent_falls_back_to_any_bonded() {
        let mut rng = StdRng::seed_from_u64(8);
        let mut a = Cell::random(&mut rng, [960.0, 540.0, 50.0], 0, 0, 0);
        let mut b = Cell::random(&mut rng, [960.0, 540.0, 50.0], 1, 0, 1);
        a.genome.adhesion_type = 3;
        b.genome.adhesion_type = 5;
        b.bonds[0] = Some(Bond {
            other_cell_id: 99,
            rest_length: 5.0,
            stiffness: BOND_STIFFNESS,
            damping: 0.6,
            age_ticks: 0,
        });
        // Child adhesion=7 — match neither — ale b má bondy → fallback.
        let pick = pick_cluster_parent(&a, &b, 7);
        assert!(pick.is_some());
        assert_eq!(pick.unwrap().cell_id, b.cell_id);
    }

    #[test]
    fn pick_cluster_parent_returns_none_when_neither_bonded() {
        let mut rng = StdRng::seed_from_u64(9);
        let a = Cell::random(&mut rng, [960.0, 540.0, 50.0], 0, 0, 0);
        let b = Cell::random(&mut rng, [960.0, 540.0, 50.0], 1, 0, 1);
        assert!(pick_cluster_parent(&a, &b, 0).is_none());
    }

    #[test]
    fn mating_child_spawns_near_bonded_parent() {
        let mut rng = StdRng::seed_from_u64(123);
        let mut a = Cell::random(&mut rng, [960.0, 540.0, 50.0], 0, 0, 0);
        let mut b = Cell::random(&mut rng, [960.0, 540.0, 50.0], 1, 0, 1);
        a.position = [100.0, 100.0, 0.0];
        b.position = [-100.0, -100.0, 0.0];
        a.genome.adhesion_type = 3;
        b.genome.adhesion_type = 3;
        a.bonds[0] = Some(Bond {
            other_cell_id: 99,
            rest_length: 5.0,
            stiffness: BOND_STIFFNESS,
            damping: 0.6,
            age_ticks: 0,
        });
        // Force child adhesion=3 by making both parents have type 3 →
        // crossover deterministic na typu (pre-mutation), mutation sice může
        // flipnout, ale 5% rate × seed 123 to nestane (test verifuje deterministic).
        let child = make_mating_child(&a, &b, &mut rng, 42);
        // Child adhesion mohl mutovat, ale spawn pozice byla rozhodnuta podle
        // child.genome.adhesion_type. Pokud child má type 3, spawn by měl
        // být blízko a (jediný bonded). Pokud mutoval, jeden z parents stejně
        // má bondy → fallback. Tedy child by měl být v každém případě blízko
        // parent_a (= jediný bonded), max do CLUSTER_SPAWN_RADIUS od něho.
        let dx = (child.position[0] - a.position[0]).abs();
        let dy = (child.position[1] - a.position[1]).abs();
        let dz = (child.position[2] - a.position[2]).abs();
        assert!(
            dx <= CLUSTER_SPAWN_RADIUS && dy <= CLUSTER_SPAWN_RADIUS
                && dz <= CLUSTER_SPAWN_RADIUS * 0.3 + 1e-3,
            "child spawn pozice mimo cluster jitter range — dx={} dy={} dz={}",
            dx,
            dy,
            dz
        );
    }

    #[test]
    fn mating_child_spawns_at_midpoint_when_neither_parent_bonded() {
        let mut rng = StdRng::seed_from_u64(456);
        let mut a = Cell::random(&mut rng, [960.0, 540.0, 50.0], 0, 0, 0);
        let mut b = Cell::random(&mut rng, [960.0, 540.0, 50.0], 1, 0, 1);
        a.position = [100.0, 100.0, 0.0];
        b.position = [-100.0, -100.0, 0.0];
        // Žádný bond → midpoint (0, 0, 0).
        let child = make_mating_child(&a, &b, &mut rng, 42);
        assert!(
            child.position[0].abs() < 1e-3 && child.position[1].abs() < 1e-3,
            "child spawn pozice měla být midpoint (0, 0), got {:?}",
            child.position
        );
    }

    #[test]
    fn hunter_seeks_nearest_attackable_cell() {
        let mut rng = StdRng::seed_from_u64(11);
        let half = [960.0, 540.0, 50.0];
        let mut cells: Vec<Cell> = Vec::new();
        // Cell 0 daleko (>vision), cell 1 blízko + bez bondů.
        let mut c0 = Cell::random(&mut rng, half, 0, 0, 0);
        c0.position = [500.0, 500.0, 0.0];
        cells.push(c0);
        let mut c1 = Cell::random(&mut rng, half, 1, 0, 1);
        c1.position = [50.0, 0.0, 0.0];
        cells.push(c1);
        let hunter_pos = [0.0, 0.0, 0.0];
        // Sprint 84: idle hunter (velocity 0) → cone filter disabled, omni.
        let pick = nearest_attackable_cell(hunter_pos, [0.0; 3], &cells, half);
        assert_eq!(pick, Some(1));
    }

    #[test]
    fn hunter_skips_immune_cluster_cells() {
        let mut rng = StdRng::seed_from_u64(12);
        let half = [960.0, 540.0, 50.0];
        let mut cells: Vec<Cell> = Vec::new();
        // Cell 0 nejbližší ale immune (3 bondy).
        let mut c0 = Cell::random(&mut rng, half, 0, 0, 0);
        c0.position = [10.0, 0.0, 0.0];
        for slot in 0..3 {
            c0.bonds[slot] = Some(Bond {
                other_cell_id: 100 + slot as u64,
                rest_length: 5.0,
                stiffness: BOND_STIFFNESS,
                damping: 0.6,
                age_ticks: 0,
            });
        }
        cells.push(c0);
        // Cell 1 dál ale solo → attackable.
        let mut c1 = Cell::random(&mut rng, half, 1, 0, 1);
        c1.position = [60.0, 0.0, 0.0];
        cells.push(c1);
        let pick = nearest_attackable_cell([0.0, 0.0, 0.0], [0.0; 3], &cells, half);
        assert_eq!(pick, Some(1));
    }

    #[test]
    fn hunter_returns_none_when_only_immune_in_range() {
        let mut rng = StdRng::seed_from_u64(13);
        let half = [960.0, 540.0, 50.0];
        let mut c = Cell::random(&mut rng, half, 0, 0, 0);
        c.position = [10.0, 0.0, 0.0];
        for slot in 0..3 {
            c.bonds[slot] = Some(Bond {
                other_cell_id: 100 + slot as u64,
                rest_length: 5.0,
                stiffness: BOND_STIFFNESS,
                damping: 0.6,
                age_ticks: 0,
            });
        }
        let cells = vec![c];
        assert!(nearest_attackable_cell([0.0, 0.0, 0.0], [0.0; 3], &cells, half).is_none());
    }

    #[test]
    fn hunter_cone_filters_blind_spot() {
        // Sprint 84: hunter pohybující se +X má forward = +X. Target přímo
        // vzadu (-X) je v blind spotu pro 60° half-angle FOV.
        let mut rng = StdRng::seed_from_u64(84);
        let half = [960.0, 540.0, 50.0];
        let mut behind = Cell::random(&mut rng, half, 0, 0, 0);
        behind.position = [-50.0, 0.0, 0.0];
        let cells = vec![behind];
        let hunter_pos = [0.0, 0.0, 0.0];
        let hunter_vel = [50.0, 0.0, 0.0]; // forward = +X (speed_sq = 2500 > threshold)
        // Cone aktivní, target vzadu → None.
        assert!(nearest_attackable_cell(hunter_pos, hunter_vel, &cells, half).is_none());
        // Idle hunter (velocity 0) → cone disabled → target nalezen.
        assert!(nearest_attackable_cell(hunter_pos, [0.0; 3], &cells, half).is_some());
    }

    #[test]
    fn hunter_cone_sees_front_target() {
        // Hunter pohybující se +X, target přímo vpředu — uvnitř cone.
        let mut rng = StdRng::seed_from_u64(85);
        let half = [960.0, 540.0, 50.0];
        let mut front = Cell::random(&mut rng, half, 0, 0, 0);
        front.position = [50.0, 0.0, 0.0];
        let cells = vec![front];
        let pick = nearest_attackable_cell([0.0, 0.0, 0.0], [50.0, 0.0, 0.0], &cells, half);
        assert_eq!(pick, Some(0));
    }

    #[test]
    fn hunter_cone_filters_flank_target() {
        // Target přímo vpravo (90° offset) — mimo 60° cone.
        let mut rng = StdRng::seed_from_u64(86);
        let half = [960.0, 540.0, 50.0];
        let mut side = Cell::random(&mut rng, half, 0, 0, 0);
        side.position = [0.0, 50.0, 0.0];
        let cells = vec![side];
        let pick = nearest_attackable_cell([0.0, 0.0, 0.0], [50.0, 0.0, 0.0], &cells, half);
        assert!(pick.is_none());
    }

    #[test]
    fn hunter_step_moves_toward_target() {
        let mut rng = StdRng::seed_from_u64(14);
        let half = [960.0, 540.0, 50.0];
        let mut h = Hunter {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            hunter_id: 0,
        };
        let target = [100.0, 0.0, 0.0];
        // Krok 1: velocity začne přibližovat k +x (toward target).
        h.step(Some(target), &mut rng, 1.0 / 60.0, half);
        assert!(h.velocity[0] > 0.0, "expected +x velocity, got {:?}", h.velocity);
        assert!(h.position[0] > 0.0, "expected +x position, got {:?}", h.position);
    }

    #[test]
    fn hunter_step_random_walks_when_no_target() {
        let mut rng = StdRng::seed_from_u64(15);
        let half = [960.0, 540.0, 50.0];
        let mut h = Hunter {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            hunter_id: 0,
        };
        // Bez targetu se velocity nenuluje (idle drift).
        for _ in 0..30 {
            h.step(None, &mut rng, 1.0 / 60.0, half);
        }
        let speed_sq = h.velocity[0] * h.velocity[0]
            + h.velocity[1] * h.velocity[1]
            + h.velocity[2] * h.velocity[2];
        assert!(speed_sq > 1e-3, "idle drift should produce nonzero velocity, got {:?}", h.velocity);
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
