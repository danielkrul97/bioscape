//! Bioscape simulation core — pure logic, no rendering.
//!
//! Keeping this layer free of Bevy types lets us drive the same world
//! from a windowed renderer (`main.rs`) or a headless batch run later.

use core::f32::consts::{FRAC_PI_2, PI, TAU};
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
// 10=heading_y, 11=pheromone_grad_x (ch0), 12=pheromone_grad_y (ch0),
// 13=local_density, 14=damage. Sprint 33 přidává 3D rozšíření:
// 15=food_dz, 16=cell_dz, 17=smell_grad_z, 18=heading_z, 19=pheromone_grad_z (ch0).
// heading_x/y jsou nově xy projekce 3D forward vektoru (násobeno cos(pitch)),
// heading_z = sin(pitch). Pro pitch=0 jsou identické s pre-Sprint-33 cos/sin
// yaw — mozky natrénované v 2D zachovají chování při horizontálním letu.
// Sprint 28 přidává recurrent kanál: indexy [BRAIN_INPUTS_SENSORY..BRAIN_INPUTS]
// = previous tick `last_hidden` activations (Elman RNN). Genom drží sjednocený
// `w1` matrix 28×8, mutace + Hebbian pracují bez rozlišení sensory vs recurrent.
// Sprint 87: 20 → 21 — slot 20 = `temperature_local`. Brain inputs šířka
// shiftne 52 → 53 (sensory + recurrent). Breaking change pre-S87 baseline:
// w1 matice resize, GPU shader hardcoded constants update (brain_forward.wgsl,
// hebbian.wgsl), all brain weights re-randomized při Genome::random.
// Sprint 126 (multi-channel pheromones): 21 → 27 — sloty 21,22,23 = ch1
// pheromone gradient xyz, 24,25,26 = ch2 pheromone gradient xyz. Nové kanály
// umožňují diskriminovanou komunikaci (cells emitují mixturu, sensors rozliší).
pub const BRAIN_INPUTS_SENSORY: usize = 27;
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
//
// Sprint 103 (HyperNEAT decade prep): 32 → 50 storage. BRAIN_INPUTS shift
// 53 → 71. w1 z 32×53=1696 → 50×71=3550 floats (2.1×). Per cell brain
// ~16 KB → 2500 cells = ~41 MB. Forward 4050 ops/cell × 2500 × 60Hz =
// 600M ops/sec single-core, viable. Storage prostor pro NEAT add_neuron
// (S104) + HyperNEAT CPPN-derived weights (S106).
pub const BRAIN_HIDDEN: usize = 50;
/// Initial active hidden count při spawn. Cell s `hidden_n = BRAIN_HIDDEN_DEFAULT`
/// reprodukuje pre-Sprint-80 behavior byte-identical (dead zone weights = 0,
/// nepřispívá). Structural mutace později rozhojnou na ≤ `BRAIN_HIDDEN`.
///
/// Sprint 103: 16 → 25 (~50 % storage). Větší startovní kapacita pro
/// post-S103 brain landscape; add_neuron mutace pak roste k 50.
pub const BRAIN_HIDDEN_DEFAULT: usize = 25;
/// Floor pro budoucí remove_neuron mutaci. Pod 4 neurony brain nedělá nic
/// užitečného (sensory→hidden kompresor potřebuje minimum kapacity).
pub const BRAIN_HIDDEN_MIN: usize = 4;
/// Sprint 80 Sprint C: gaussian sigma pro inicializaci nových neuronů v
/// `Brain::add_neuron`. Menší než `sigma_brain` (0.2) — NEAT-style minimal
/// disruption: nový neuron startuje s near-zero kontribucí k outputům, takže
/// add_neuron sám o sobě neporouchá existující funkční mozek. Selekce ho
/// pak buď posílí (subsequent weight mutace), nebo nechá v latentní formě.
pub const ADD_NEURON_SIGMA: f32 = 0.1;
/// Sprint 104: minimum |w| pro link aby se kvalifikoval pro split_link.
/// Linky pod threshold považujeme za "neaktivní" (nemá smysl je splitnout
/// — jejich phenotype nese ~0 informaci). 0.05 ≈ noise level pro sigma 0.2
/// init.
pub const SPLIT_LINK_THRESHOLD: f32 = 0.05;
/// Sprint 28: kolik dimenzí předchozího hidden state se feeduje zpátky jako
/// input. = `BRAIN_HIDDEN` znamená 1:1 mapping (každý neuron má vlastní paměť
/// slot). Menší než HIDDEN by exponoval jen subset hidden state; větší by
/// vyžadoval delay buffer. 1:1 je nejjednodušší a kapacita stačí.
pub const BRAIN_RECURRENT: usize = BRAIN_HIDDEN;
/// Total brain input width = sensory + recurrent. Genom + forward pass +
/// Hebbian + mutace pracují s touto velikostí transparentně.
pub const BRAIN_INPUTS: usize = BRAIN_INPUTS_SENSORY + BRAIN_RECURRENT;
// Brain outputs: 0=turn (yaw rate), 1=thrust, 2=pheromone ch0 emit
// (positive = emit more above baseline, costs energy), 3=morph_length,
// 4=morph_width, 5=morph_spike, 6=attack, 7=turn_pitch (Sprint 33),
// 8=morph_height (Sprint 34, appended kvůli zachování existujících indexů).
// Sprint 66: 9=bond_signal — pozitivní (>BOND_FORM_THRESHOLD) povoluje vznik
// spring bondů, silně negativní (<BOND_BREAK_THRESHOLD) trhá existující bondy.
// Indexy 0–8 zachované, jen append.
// Sprint 26 morph signals: signal × MORPH_RATE × dt přičteno k phenotype dim
// každý tick, energy cost ∝ |delta|. Sprint 27 attack: gating signál pro
// `predate` — bez aktivního output[6] > THRESHOLD se predace nestane.
// Sprint 126 (multi-channel pheromones): 10=ch1 emit, 11=ch2 emit. Trojice
// kanálů s různým decay (ch0 slow, ch1 medium, ch2 fast) → temporal patterning
// + diskriminace.
pub const BRAIN_OUTPUTS: usize = 12;
/// Inicializační bias na thrust output bin v `Brain::random`. Bez něj má ~½
/// random brainů thrust output blízko nuly (cell se sotva hýbe), což vytvářelo
/// hluboké bottlenecky v ranných generacích. Po prvním selekčním tlaku evoluce
/// hodnotu doladí — bias je jen jumpstart.
pub const INNATE_THRUST_BIAS: f32 = 2.0;
/// Inicializační bias na pheromone ch0 output (b2[2]). Sprint 25 vyžaduje
/// aktivní emisi pro reprodukci — bez biasu jen ~25 % párů projde threshold.
/// S bias 1.0 většina random brainů emituje nad threshold v gen 0; selekce
/// pak ladí.
pub const INNATE_PHEROMONE_BIAS: f32 = 1.0;
/// Sprint 126: slabší bias na ch1 / ch2 emit slots (b2[10], b2[11]). Bez
/// biasu by random brainy startovaly s ~0 emisi na nových kanálech a evoluce
/// by měla dlouhý cold-start (signal must emerge from noise alone). 0.5 dává
/// non-zero baseline, ale ne tak silný jako ch0 (mating gating ho potřebuje).
pub const INNATE_PHEROMONE_AUX_BIAS: f32 = 0.5;
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
pub const INNATE_BOND_BIAS: f32 = 2.5;

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
///
/// Sprint 100: 2500 → 5000 (proportional s z=50 → z=100 expansion). Density
/// 1.2e-5 cells/unit³ zachovaná. 5k testováno v S63 benchmarku jako
/// 870 ticks/s na CPU paralelní cestě.
pub const MAX_POPULATION: usize = 1500;

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
/// Sprint 87: range pro per-cell `thermal_optimum` gen. Init populace draws
/// uniform across [MIN, MAX] = [BOTTOM, TOP] → speciace mezi cold-prefer
/// a warm-prefer fenotypy. Range matches static gradient endpoints — cells
/// nemůžou preferovat teplotu mimo dostupný z-rozsah.
pub const MIN_THERMAL_OPTIMUM: f32 = THERMAL_BOTTOM;
pub const MAX_THERMAL_OPTIMUM: f32 = THERMAL_TOP;
/// Sprint 87: peak penalty rate při |dev|/13 = 1.0 (= max single-direction
/// deviation v [BOTTOM, TOP] od optima blízko opačného konce). Quadratic
/// penalty: `((temp - opt) / 13.0)² × PENALTY × dt`. Při PENALTY = 1.0:
/// extreme-deviation cell platí ~1.0 energy/sec navíc — comparable s body
/// maintenance cost. Independentní od metabolism factor (thermal stress je
/// extra cost, ne reduction enzymové aktivity).
pub const THERMAL_OPTIMUM_PENALTY: f32 = 1.0;
/// Sprint 31 spatial clustering: rejection sampling síla. Per uniformně
/// vzorkovaný food candidate je pravděpodobnost zamítnutí
/// `STRENGTH × (1 - richness)`. Při richness=1 (rich zone) nikdy nezamítá,
/// při richness=0 (poor zone) zamítá s pravděpodobností STRENGTH. Sprint 21
/// v1–v5 zkoušel plnou sílu (`1 × (1 - richness)`) → extinkce gen 70–110.
/// Sprint 100: 0.3 → 0.6 — silnější spatial gradient. Poor zone má 40 % šanci
/// spawnu per candidate (vs. 70 % před), což zostřuje hranici mezi rich a poor
/// zonami a posiluje selekční signál pro spatial preference. Stále hluboko pod
/// extinction threshold ze Sprintu 21 (1.0 = full reject v poor zone). Plus
/// value modulace `WORLD_MAP_FOOD_FLOOR/AMP` jako safety net (i food v poor
/// zone má hodnotu ~85 % baselinu). MAX_SPAWN_ATTEMPTS retry budget na uniform
/// resample zajišťuje, že clustering nikdy úplně nezablokuje spawn.
pub const FOOD_REJECTION_STRENGTH: f32 = 0.6;
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
/// Sprint 126: počet nezávislých pheromone kanálů. Multi-channel umožňuje
/// emergence diskriminované komunikace (cells emitují mixturu, sensors
/// rozliší). 3 = uprostřed rozsahu 3-8 — dost pro discrimination, méně
/// invazivní vs Brain dim expansion.
pub const N_PHEROMONE_CHANNELS: usize = 3;
/// Sprint 126: per-channel decay (1/s). ch0 = existing slow (mating-friendly),
/// ch1 medium, ch2 fast (bursty / temporal patterning).
pub const PHEROMONE_DECAY_PER_CH: [f32; N_PHEROMONE_CHANNELS] = [0.3, 1.5, 5.0];
/// Sprint 126: per-channel diffusion. Slow channels difunduji víc (cumulative
/// spread), rychlé méně (lokalizovaná spike).
pub const PHEROMONE_DIFFUSION_PER_CH: [f32; N_PHEROMONE_CHANNELS] = [0.15, 0.12, 0.08];
/// Backward-compat: ch0 (slow) decay/diffusion. GPU shaders + headless GPU
/// path stále používají single-channel scalar.
pub const PHEROMONE_DIFFUSION: f32 = PHEROMONE_DIFFUSION_PER_CH[0];
pub const PHEROMONE_DECAY: f32 = PHEROMONE_DECAY_PER_CH[0];
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

/// Half-extents simulačního světa (toroidal). Sdíleno mezi rendererem
/// (`src/main.rs`) a headlessem (`src/bin/headless.rs`) — single source
/// of truth, jinak by sim tiše divergoval při změně jen na jedné straně.
///
/// Sprint 53: WORLD_HALF[2] expanded z=2 → z=20. SmellField + WorldMap +
/// Pheromone jsou plně 3D (volumetric grid + 7-point Jacobi diffusion + 3D
/// gradient). Cells získávají vertikální environmental sensing (smell_grad_z,
/// pheromone_grad_z přes inputs[17,19]).
///
/// Sprint 64: z=20 → z=50 expansion + MAX_POPULATION 1000 → 2500
/// (proportional volumetric scaling, cells/unit³ ≈ 1.2e-5).
///
/// Sprint 100: z=50 → z=100 expansion + MAX_POPULATION 2500 → 5000
/// (drží density 1.2e-5 cells/unit³). Grid resolutions (`*_GRID_RES_Z=16`)
/// nezměněné — cell_size_z naroste 6.25 → 12.5, stále jemnější než xy
/// (cell_size_x=30, cell_size_y≈17), takže thin-world aspect rationale
/// ze S53 platí dál. Food count auto-scaluje přes `food_target` z_factor.
pub const WORLD_HALF: [f32; 3] = [960.0, 540.0, 100.0];

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
/// Sprint 121: max počet spike na buňku. `Genome` i `Phenotype` drží
/// `[Spike; SPIKE_SLOTS]` plus `spike_count: u8 ∈ [0, SPIKE_SLOTS]`. Aktivní
/// sloty 0..spike_count, neaktivní zero-init. Fixed array (ne Vec) drží
/// `Cell: Copy` pro Bevy par_iter snapshoty + GPU storage buffer s pevným
/// layoutem.
pub const SPIKE_SLOTS: usize = 5;
pub const MIN_SPIKE_AZIMUTH: f32 = -PI;
pub const MAX_SPIKE_AZIMUTH: f32 = PI;
pub const MIN_SPIKE_ELEVATION: f32 = -FRAC_PI_2;
pub const MAX_SPIKE_ELEVATION: f32 = FRAC_PI_2;
pub const MIN_SPIKE_COMPLEXITY: f32 = 0.0;
pub const MAX_SPIKE_COMPLEXITY: f32 = 1.0;
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

/// Sprint 123: complexity multiplikátor na attack bonus.
/// `effective_attack = base × (1 + COMPLEXITY_ATTACK_GAIN × complexity)`.
/// 0.5 = max +50 % bonus (complexity=1).
pub const COMPLEXITY_ATTACK_GAIN: f32 = 0.5;
/// Sprint 123: quadratic complexity multiplikátor na maintenance cost.
/// `cost_factor = 1 + COMPLEXITY_COST_GAIN × complexity²`. 3.0 = complexity=1
/// platí ×4, complexity=0.5 ×1.75. Quadratic záměrně — max-complexity je
/// vzácný, sweet-spot kolem 0.4-0.6.
pub const COMPLEXITY_COST_GAIN: f32 = 3.0;
/// Sprint 122: half-angle (rad) per-spike eat grab cone u tipu spike.
/// 0.3 rad ≈ 17°. Cell může jíst food bod, který spadne do tohoto kuželu
/// (vrchol = tip spike, osa = spike direction, range ∝ length).
pub const SPIKE_GRAB_HALF_ANGLE: f32 = 0.3;
/// Sprint 123: complexity multiplikátor na grab cone half-angle.
/// `effective_half_angle = SPIKE_GRAB_HALF_ANGLE × (1 + COMPLEXITY_GRAB_GAIN × complexity)`.
/// 1.0 = complexity=1 dvojnásobí kužel (širší branching = větší reach).
pub const COMPLEXITY_GRAB_GAIN: f32 = 1.0;
/// Sprint 122: násobitel `effective_radius`, kterým se měří range spike
/// grab cone od cell centra: `tip_distance = effective_radius + spike.length`.
/// Šířka cone u tipu = `tip_distance × tan(half_angle)`.
pub const SPIKE_GRAB_REACH_BONUS: f32 = 1.0;

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
// ─── Sprint 105: HyperNEAT CPPN scaffolding ─────────────────────────────────
//
// CPPN (Compositional Pattern-Producing Network) je malá heterogenní NN
// s diverse activation functions. V S106 bude generovat Brain weights na
// základě geometric coords substrate neuronů. V S105 je standalone — datová
// struktura, mutace, crossover, forward pass, tests.

/// Activation functions for CPPN nodes. HyperNEAT-classic library —
/// rozmanité tvary vedou k symetrickým / periodic patterns ve weight space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationFn {
    Linear,
    Sigmoid,
    Tanh,
    Gaussian,
    Sine,
    Abs,
    Step,
}

impl ActivationFn {
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            ActivationFn::Linear => x,
            ActivationFn::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFn::Tanh => x.tanh(),
            ActivationFn::Gaussian => (-x * x).exp(),
            ActivationFn::Sine => x.sin(),
            ActivationFn::Abs => x.abs(),
            ActivationFn::Step => {
                if x >= 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    pub fn random(rng: &mut impl Rng) -> Self {
        match rng.random_range(0..7) {
            0 => ActivationFn::Linear,
            1 => ActivationFn::Sigmoid,
            2 => ActivationFn::Tanh,
            3 => ActivationFn::Gaussian,
            4 => ActivationFn::Sine,
            5 => ActivationFn::Abs,
            _ => ActivationFn::Step,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CppnNode {
    pub id: u32,
    pub activation: ActivationFn,
    pub bias: f32,
    /// Layer index pro topological sort. Inputs = 0, outputs = max_layer,
    /// hidden ∈ (0, max_layer). Add_node split insertem dostane layer mezi
    /// from a to.
    pub layer: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CppnLink {
    /// Innovation id — monotonic per-Cppn. Per-Cppn lokální (ne globální
    /// jako v classic NEAT). Stačí pro speciation distance v rámci jedné
    /// linie; cross-population alignment je proxy via crossover.
    pub innovation: u32,
    pub from: u32,
    pub to: u32,
    pub weight: f32,
    pub enabled: bool,
}

/// Sprint 105: CPPN config. CPPN_INPUTS=6 stačí pro 3D substrate (x1,y1,z1,
/// x2,y2,z2 = coords obou neuronů co spojuje). Plus volitelný bias input
/// (1.0 const). CPPN_OUTPUTS=2: weight + link_existence (gate via threshold).
pub const CPPN_INPUTS: usize = 7; // 6 coords + 1 bias-const
pub const CPPN_OUTPUTS: usize = 2; // weight + link_exists
/// Initial CPPN nodes count při random init: CPPN_INPUTS + CPPN_OUTPUTS
/// + 1 hidden neuron na startup. Growable přes add_node mutace.
pub const CPPN_INITIAL_HIDDEN: usize = 1;
/// Maximum CPPN nodes celkem. Soft cap k zabránění memory blow-up. 64 dává
/// (CPPN_INPUTS + CPPN_OUTPUTS + ~55 hidden), což pokryje phenotype rozsah
/// většiny HyperNEAT studií.
pub const CPPN_MAX_NODES: usize = 64;
/// Maximum CPPN links — quadratic-ish growth s nodes, soft cap.
pub const CPPN_MAX_LINKS: usize = 256;

/// Sprint 106: fixed-size arrays místo Vec — preserves Copy trait pro Genome.
/// Packed layout: nodes[0..num_nodes], links[0..num_links] valid; zbytek None.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Cppn {
    #[serde(with = "serde_arr_cppn_nodes")]
    pub nodes: [Option<CppnNode>; CPPN_MAX_NODES],
    #[serde(with = "serde_arr_cppn_links")]
    pub links: [Option<CppnLink>; CPPN_MAX_LINKS],
    pub num_nodes: u8,
    pub num_links: u16,
    pub next_innovation: u32,
}

mod serde_arr_cppn_nodes {
    use super::{CppnNode, CPPN_MAX_NODES};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(
        a: &[Option<CppnNode>; CPPN_MAX_NODES],
        s: S,
    ) -> Result<S::Ok, S::Error> {
        a.as_slice().serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<[Option<CppnNode>; CPPN_MAX_NODES], D::Error> {
        let v: Vec<Option<CppnNode>> = Vec::deserialize(d)?;
        if v.len() != CPPN_MAX_NODES {
            return Err(serde::de::Error::custom("cppn nodes length mismatch"));
        }
        let mut a: [Option<CppnNode>; CPPN_MAX_NODES] = [None; CPPN_MAX_NODES];
        for (i, x) in v.into_iter().enumerate() {
            a[i] = x;
        }
        Ok(a)
    }
}

mod serde_arr_cppn_links {
    use super::{CppnLink, CPPN_MAX_LINKS};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(
        a: &[Option<CppnLink>; CPPN_MAX_LINKS],
        s: S,
    ) -> Result<S::Ok, S::Error> {
        a.as_slice().serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<[Option<CppnLink>; CPPN_MAX_LINKS], D::Error> {
        let v: Vec<Option<CppnLink>> = Vec::deserialize(d)?;
        if v.len() != CPPN_MAX_LINKS {
            return Err(serde::de::Error::custom("cppn links length mismatch"));
        }
        let mut a: [Option<CppnLink>; CPPN_MAX_LINKS] = [None; CPPN_MAX_LINKS];
        for (i, x) in v.into_iter().enumerate() {
            a[i] = x;
        }
        Ok(a)
    }
}

impl Cppn {
    /// Active nodes iterator (skip None slots).
    pub fn iter_nodes(&self) -> impl Iterator<Item = &CppnNode> {
        self.nodes
            .iter()
            .take(self.num_nodes as usize)
            .filter_map(|n| n.as_ref())
    }

    /// Active links iterator.
    pub fn iter_links(&self) -> impl Iterator<Item = &CppnLink> {
        self.links
            .iter()
            .take(self.num_links as usize)
            .filter_map(|l| l.as_ref())
    }

    fn push_node(&mut self, n: CppnNode) -> bool {
        if (self.num_nodes as usize) >= CPPN_MAX_NODES {
            return false;
        }
        self.nodes[self.num_nodes as usize] = Some(n);
        self.num_nodes += 1;
        true
    }

    fn push_link(&mut self, l: CppnLink) -> bool {
        if (self.num_links as usize) >= CPPN_MAX_LINKS {
            return false;
        }
        self.links[self.num_links as usize] = Some(l);
        self.num_links += 1;
        true
    }
}

impl Cppn {
    /// Random init: CPPN_INPUTS input nodes (Linear, layer 0) +
    /// CPPN_OUTPUTS output nodes (Tanh, layer 2) + 1 hidden (random fn,
    /// layer 1). Initial links: každý input → hidden + hidden → output
    /// s random gaussian weight.
    pub fn random(rng: &mut impl Rng) -> Self {
        let mut cppn = Cppn {
            nodes: [None; CPPN_MAX_NODES],
            links: [None; CPPN_MAX_LINKS],
            num_nodes: 0,
            num_links: 0,
            next_innovation: 0,
        };
        let mut next_id: u32 = 0;
        // Inputs (layer 0).
        let mut input_ids: [u32; CPPN_INPUTS] = [0; CPPN_INPUTS];
        for slot in input_ids.iter_mut() {
            cppn.push_node(CppnNode {
                id: next_id,
                activation: ActivationFn::Linear,
                bias: 0.0,
                layer: 0,
            });
            *slot = next_id;
            next_id += 1;
        }
        // Outputs (layer 2). Tanh dává weights ∈ [-1,1] a link_exists ∈ [-1,1].
        let mut output_ids: [u32; CPPN_OUTPUTS] = [0; CPPN_OUTPUTS];
        for slot in output_ids.iter_mut() {
            cppn.push_node(CppnNode {
                id: next_id,
                activation: ActivationFn::Tanh,
                bias: 0.0,
                layer: 2,
            });
            *slot = next_id;
            next_id += 1;
        }
        // Hidden seed neurons (layer 1).
        let mut hidden_ids: [u32; CPPN_INITIAL_HIDDEN] = [0; CPPN_INITIAL_HIDDEN];
        for slot in hidden_ids.iter_mut() {
            cppn.push_node(CppnNode {
                id: next_id,
                activation: ActivationFn::random(rng),
                bias: gaussian(rng) * 0.5,
                layer: 1,
            });
            *slot = next_id;
            next_id += 1;
        }
        // Initial bipartite links.
        let mut innovation: u32 = 0;
        for &i in &input_ids {
            for &h in &hidden_ids {
                cppn.push_link(CppnLink {
                    innovation,
                    from: i,
                    to: h,
                    weight: gaussian(rng),
                    enabled: true,
                });
                innovation += 1;
            }
        }
        for &h in &hidden_ids {
            for &o in &output_ids {
                cppn.push_link(CppnLink {
                    innovation,
                    from: h,
                    to: o,
                    weight: gaussian(rng),
                    enabled: true,
                });
                innovation += 1;
            }
        }
        cppn.next_innovation = innovation;
        cppn
    }

    /// Forward pass — feed-forward by layer. Inputs are mapped do prvních
    /// CPPN_INPUTS nodů. Outputs returned ze posledních CPPN_OUTPUTS.
    /// Layer-wise computation; cycles unsupported (add_link mutace
    /// preventuje cykly — viz `mutate_add_link`).
    pub fn forward(&self, inputs: [f32; CPPN_INPUTS]) -> [f32; CPPN_OUTPUTS] {
        let mut activations: rustc_hash::FxHashMap<u32, f32> =
            rustc_hash::FxHashMap::default();
        // Inputs occupy nodes[0..CPPN_INPUTS] per random() layout.
        for i in 0..CPPN_INPUTS {
            if let Some(n) = self.nodes[i] {
                activations.insert(n.id, inputs[i]);
            }
        }
        let max_layer = self.iter_nodes().map(|n| n.layer).max().unwrap_or(0);
        for layer in 1..=max_layer {
            for n in self.iter_nodes() {
                if n.layer != layer {
                    continue;
                }
                let mut sum = n.bias;
                for link in self.iter_links() {
                    if !link.enabled || link.to != n.id {
                        continue;
                    }
                    if let Some(&x) = activations.get(&link.from) {
                        sum += link.weight * x;
                    }
                }
                activations.insert(n.id, n.activation.apply(sum));
            }
        }
        let mut out = [0.0; CPPN_OUTPUTS];
        // Outputs occupy nodes[CPPN_INPUTS..CPPN_INPUTS+CPPN_OUTPUTS].
        for o in 0..CPPN_OUTPUTS {
            if let Some(n) = self.nodes[CPPN_INPUTS + o] {
                out[o] = *activations.get(&n.id).unwrap_or(&0.0);
            }
        }
        out
    }

    /// Mutate weight of random enabled link gaussian-style.
    pub fn mutate_weight(&mut self, rng: &mut impl Rng, sigma: f32) {
        let active: Vec<usize> = (0..self.num_links as usize)
            .filter(|&i| self.links[i].as_ref().map_or(false, |l| l.enabled))
            .collect();
        if active.is_empty() {
            return;
        }
        let pick = active[rng.random_range(0..active.len())];
        if let Some(l) = self.links[pick].as_mut() {
            l.weight += gaussian(rng) * sigma;
        }
    }

    /// Add_node: split random enabled link, insert nový hidden node.
    pub fn mutate_add_node(&mut self, rng: &mut impl Rng) {
        if (self.num_nodes as usize) >= CPPN_MAX_NODES
            || (self.num_links as usize) + 2 > CPPN_MAX_LINKS
        {
            return;
        }
        let active: Vec<usize> = (0..self.num_links as usize)
            .filter(|&i| self.links[i].as_ref().map_or(false, |l| l.enabled))
            .collect();
        if active.is_empty() {
            return;
        }
        let pick = active[rng.random_range(0..active.len())];
        let original = match self.links[pick] {
            Some(l) => l,
            None => return,
        };
        let from_layer = self
            .iter_nodes()
            .find(|n| n.id == original.from)
            .map(|n| n.layer)
            .unwrap_or(0);
        let to_layer = self
            .iter_nodes()
            .find(|n| n.id == original.to)
            .map(|n| n.layer)
            .unwrap_or(0);
        let new_layer = if from_layer + 1 < to_layer {
            from_layer + 1
        } else {
            let new_l = from_layer + 1;
            for slot in self.nodes.iter_mut() {
                if let Some(n) = slot.as_mut() {
                    if n.layer >= new_l {
                        n.layer += 1;
                    }
                }
            }
            new_l
        };
        let new_id = self.iter_nodes().map(|n| n.id).max().unwrap_or(0) + 1;
        self.push_node(CppnNode {
            id: new_id,
            activation: ActivationFn::random(rng),
            bias: 0.0,
            layer: new_layer,
        });
        if let Some(l) = self.links[pick].as_mut() {
            l.enabled = false;
        }
        let inv1 = self.next_innovation;
        let inv2 = self.next_innovation + 1;
        self.next_innovation += 2;
        self.push_link(CppnLink {
            innovation: inv1,
            from: original.from,
            to: new_id,
            weight: 1.0,
            enabled: true,
        });
        self.push_link(CppnLink {
            innovation: inv2,
            from: new_id,
            to: original.to,
            weight: original.weight,
            enabled: true,
        });
    }

    /// Add_link: pick random pair (from, to) with from.layer < to.layer.
    pub fn mutate_add_link(&mut self, rng: &mut impl Rng, sigma: f32) {
        if (self.num_links as usize) >= CPPN_MAX_LINKS {
            return;
        }
        if self.num_nodes < 2 {
            return;
        }
        let n = self.num_nodes as usize;
        for _ in 0..16 {
            let i_idx = rng.random_range(0..n);
            let j_idx = rng.random_range(0..n);
            if i_idx == j_idx {
                continue;
            }
            let from_node = match self.nodes[i_idx] {
                Some(x) => x,
                None => continue,
            };
            let to_node = match self.nodes[j_idx] {
                Some(x) => x,
                None => continue,
            };
            if from_node.layer >= to_node.layer {
                continue;
            }
            let exists = self
                .iter_links()
                .any(|l| l.from == from_node.id && l.to == to_node.id);
            if exists {
                continue;
            }
            let inv = self.next_innovation;
            self.next_innovation += 1;
            self.push_link(CppnLink {
                innovation: inv,
                from: from_node.id,
                to: to_node.id,
                weight: gaussian(rng) * sigma,
                enabled: true,
            });
            return;
        }
    }

    /// Toggle enable/disable bit of random link.
    pub fn mutate_toggle_link(&mut self, rng: &mut impl Rng) {
        if self.num_links == 0 {
            return;
        }
        let pick = rng.random_range(0..self.num_links as usize);
        if let Some(l) = self.links[pick].as_mut() {
            l.enabled = !l.enabled;
        }
    }

    /// Mutate activation function of random hidden node (skip inputs/outputs).
    pub fn mutate_activation(&mut self, rng: &mut impl Rng) {
        let hidden: Vec<usize> = (0..self.num_nodes as usize)
            .filter(|&i| {
                self.nodes[i]
                    .as_ref()
                    .map_or(false, |n| n.layer != 0 && n.layer != 2)
            })
            .collect();
        if hidden.is_empty() {
            return;
        }
        let pick = hidden[rng.random_range(0..hidden.len())];
        if let Some(n) = self.nodes[pick].as_mut() {
            n.activation = ActivationFn::random(rng);
        }
    }

    pub fn mutate(&self, rng: &mut impl Rng, cfg: &CppnMutationConfig) -> Self {
        let mut out = *self;
        if rng.random::<f32>() < cfg.weight_rate {
            out.mutate_weight(rng, cfg.sigma_weight);
        }
        if rng.random::<f32>() < cfg.add_node_rate {
            out.mutate_add_node(rng);
        }
        if rng.random::<f32>() < cfg.add_link_rate {
            out.mutate_add_link(rng, cfg.sigma_weight);
        }
        if rng.random::<f32>() < cfg.toggle_link_rate {
            out.mutate_toggle_link(rng);
        }
        if rng.random::<f32>() < cfg.activation_rate {
            out.mutate_activation(rng);
        }
        out
    }

    /// Sprint 107: NEAT compatibility distance metric. Speciation gate.
    ///   δ(a, b) = c1 × E/N + c2 × D/N + c3 × W̄
    /// kde:
    ///   E = excess gene count (innovations beyond max in other parent)
    ///   D = disjoint gene count (within range, not matching)
    ///   N = max(genome size) — normalizace
    ///   W̄ = average |weight diff| pro matching genes
    /// Constants follow classical NEAT defaults.
    pub fn compatibility_distance(a: &Cppn, b: &Cppn) -> f32 {
        const C_EXCESS: f32 = 1.0;
        const C_DISJOINT: f32 = 1.0;
        const C_WEIGHT: f32 = 0.4;
        let max_inv_a = a.iter_links().map(|l| l.innovation).max().unwrap_or(0);
        let max_inv_b = b.iter_links().map(|l| l.innovation).max().unwrap_or(0);
        let cutoff = max_inv_a.min(max_inv_b);
        let mut excess: u32 = 0;
        let mut disjoint: u32 = 0;
        let mut weight_diff_sum: f32 = 0.0;
        let mut matching: u32 = 0;
        let a_links: rustc_hash::FxHashMap<u32, &CppnLink> =
            a.iter_links().map(|l| (l.innovation, l)).collect();
        let b_links: rustc_hash::FxHashMap<u32, &CppnLink> =
            b.iter_links().map(|l| (l.innovation, l)).collect();
        for (inv, la) in a_links.iter() {
            if let Some(lb) = b_links.get(inv) {
                weight_diff_sum += (la.weight - lb.weight).abs();
                matching += 1;
            } else if *inv > cutoff {
                excess += 1;
            } else {
                disjoint += 1;
            }
        }
        for inv in b_links.keys() {
            if a_links.contains_key(inv) {
                continue;
            }
            if *inv > cutoff {
                excess += 1;
            } else {
                disjoint += 1;
            }
        }
        let n = (a.num_links.max(b.num_links) as f32).max(1.0);
        let w_avg = if matching > 0 {
            weight_diff_sum / matching as f32
        } else {
            0.0
        };
        C_EXCESS * (excess as f32) / n + C_DISJOINT * (disjoint as f32) / n + C_WEIGHT * w_avg
    }

    /// Crossover: align matching innovations + nodes by id. Random pick na
    /// matching, inherit from both na disjoint. Cap respektován (CPPN_MAX_*).
    pub fn crossover(a: &Cppn, b: &Cppn, rng: &mut impl Rng) -> Cppn {
        let mut nodes_map: rustc_hash::FxHashMap<u32, CppnNode> =
            rustc_hash::FxHashMap::default();
        for n in a.iter_nodes() {
            nodes_map.insert(n.id, *n);
        }
        for n in b.iter_nodes() {
            match nodes_map.get(&n.id) {
                Some(_) if rng.random::<bool>() => {
                    nodes_map.insert(n.id, *n);
                }
                None => {
                    nodes_map.insert(n.id, *n);
                }
                _ => {}
            }
        }
        let mut sorted_nodes: Vec<CppnNode> = nodes_map.into_values().collect();
        sorted_nodes.sort_by_key(|n| n.id);

        let mut links_map: rustc_hash::FxHashMap<u32, CppnLink> =
            rustc_hash::FxHashMap::default();
        for l in a.iter_links() {
            links_map.insert(l.innovation, *l);
        }
        for l in b.iter_links() {
            match links_map.get(&l.innovation) {
                Some(_) if rng.random::<bool>() => {
                    links_map.insert(l.innovation, *l);
                }
                None => {
                    links_map.insert(l.innovation, *l);
                }
                _ => {}
            }
        }
        let mut sorted_links: Vec<CppnLink> = links_map.into_values().collect();
        sorted_links.sort_by_key(|l| l.innovation);

        let next_innovation = sorted_links
            .iter()
            .map(|l| l.innovation + 1)
            .max()
            .unwrap_or(0);

        let mut out = Cppn {
            nodes: [None; CPPN_MAX_NODES],
            links: [None; CPPN_MAX_LINKS],
            num_nodes: 0,
            num_links: 0,
            next_innovation,
        };
        for n in sorted_nodes.into_iter().take(CPPN_MAX_NODES) {
            out.push_node(n);
        }
        for l in sorted_links.into_iter().take(CPPN_MAX_LINKS) {
            out.push_link(l);
        }
        out
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CppnMutationConfig {
    pub weight_rate: f32,
    pub sigma_weight: f32,
    pub add_node_rate: f32,
    pub add_link_rate: f32,
    pub toggle_link_rate: f32,
    pub activation_rate: f32,
}

pub const CPPN_MUTATION_CONFIG: CppnMutationConfig = CppnMutationConfig {
    weight_rate: 0.8,    // time per child má change weight
    sigma_weight: 0.5,
    add_node_rate: 0.03, // structural growth, NEAT default ~0.03
    add_link_rate: 0.05, // higher than node — more new connections than nodes
    toggle_link_rate: 0.01,
    activation_rate: 0.02,
};

// ─── Sprint 106: HyperNEAT substrate + Brain phenotype generation ───────────
//
// Substrate je geometrické rozložení sensor / hidden / output neuronů ve
// 3D prostoru. CPPN (S105) přijímá coords obou neuronů jako 6 vstupů +
// 1 bias a vrací [weight, link_exists]. Brain::from_cppn projde všechny
// possible (input, hidden) a (hidden, output) páry, populuje weights.
//
// Substrate je jednoduchý 1D: každá vrstva má z-coord (input z=-1, hidden
// z=0, output z=+1) a x-coord normalizován do [-1, 1] podle slot indexu.
// y-coord = 0 (1D substrate, scope-reduced).

/// Spočítá substrate coords pro brain input slot. Slot < BRAIN_INPUTS_SENSORY
/// jsou sensory inputs (mapovány do x-axis); slot ≥ BRAIN_INPUTS_SENSORY
/// jsou recurrent inputs (sdílí coord s hidden neuronem stejného indexu —
/// recurrent slot k mapuje na hidden neuron k coords).
pub fn substrate_input_coords(slot: usize) -> [f32; 3] {
    if slot < BRAIN_INPUTS_SENSORY {
        let x = -1.0 + 2.0 * (slot as f32) / (BRAIN_INPUTS_SENSORY as f32 - 1.0).max(1.0);
        [x, 0.0, -1.0]
    } else {
        let h_idx = slot - BRAIN_INPUTS_SENSORY;
        substrate_hidden_coords(h_idx)
    }
}

pub fn substrate_hidden_coords(slot: usize) -> [f32; 3] {
    let x = -1.0 + 2.0 * (slot as f32) / (BRAIN_HIDDEN as f32 - 1.0).max(1.0);
    [x, 0.0, 0.0]
}

pub fn substrate_output_coords(slot: usize) -> [f32; 3] {
    let x = -1.0 + 2.0 * (slot as f32) / (BRAIN_OUTPUTS as f32 - 1.0).max(1.0);
    [x, 0.0, 1.0]
}

/// CPPN_LINK_EXISTS_THRESHOLD: pokud CPPN output[1] < threshold, link je
/// "expressed off" — weight = 0 (no connection). 0.0 (Tanh midpoint) dává
/// ~50 % links by default. Posun threshold mění density derived networks.
pub const CPPN_LINK_EXISTS_THRESHOLD: f32 = 0.0;

impl Brain {
    /// Sprint 106: derive brain weights z CPPN substrate query. Pro každý
    /// (input slot, hidden slot) pár zavolá cppn.forward([from.coord, to.coord, 1.0]),
    /// extrahuje weight + link_exists. Stejně pro (hidden, output).
    /// Biases (b1, b2) odvozeny z CPPN query s "self-loop" coord (oba inputs
    /// stejné). hidden_n = BRAIN_HIDDEN_DEFAULT.
    pub fn from_cppn(cppn: &Cppn) -> Brain {
        let mut w1 = [[0.0_f32; BRAIN_INPUTS]; BRAIN_HIDDEN];
        let mut b1 = [0.0_f32; BRAIN_HIDDEN];
        let mut w2 = [[0.0_f32; BRAIN_HIDDEN]; BRAIN_OUTPUTS];
        let mut b2 = [0.0_f32; BRAIN_OUTPUTS];

        // w1: input → hidden
        for h in 0..BRAIN_HIDDEN {
            let to_c = substrate_hidden_coords(h);
            for i in 0..BRAIN_INPUTS {
                let from_c = substrate_input_coords(i);
                let inputs = [
                    from_c[0], from_c[1], from_c[2],
                    to_c[0], to_c[1], to_c[2],
                    1.0,
                ];
                let out = cppn.forward(inputs);
                if out[1] >= CPPN_LINK_EXISTS_THRESHOLD {
                    w1[h][i] = out[0];
                }
            }
            // b1[h]: query CPPN with from = hidden (self-loop sentinel)
            let inputs = [to_c[0], to_c[1], to_c[2], to_c[0], to_c[1], to_c[2], 0.0];
            let out = cppn.forward(inputs);
            b1[h] = out[0] * 0.5; // dampen bias
        }

        // w2: hidden → output
        for o in 0..BRAIN_OUTPUTS {
            let to_c = substrate_output_coords(o);
            for h in 0..BRAIN_HIDDEN {
                let from_c = substrate_hidden_coords(h);
                let inputs = [
                    from_c[0], from_c[1], from_c[2],
                    to_c[0], to_c[1], to_c[2],
                    1.0,
                ];
                let out = cppn.forward(inputs);
                if out[1] >= CPPN_LINK_EXISTS_THRESHOLD {
                    w2[o][h] = out[0];
                }
            }
            // b2[o]: self-loop sentinel
            let inputs = [to_c[0], to_c[1], to_c[2], to_c[0], to_c[1], to_c[2], 0.0];
            let out = cppn.forward(inputs);
            b2[o] = out[0] * 0.5;
        }

        Brain {
            hidden_n: BRAIN_HIDDEN_DEFAULT as u32,
            w1,
            b1,
            w2,
            b2,
        }
    }
}

// ─── End Sprint 106 ──────────────────────────────────────────────────────────

// ─── End Sprint 105 CPPN scaffolding ─────────────────────────────────────────

pub const PHYSICS_CONFIG: PhysicsConfig = PhysicsConfig {
    drag: DRAG_COEFFICIENT,
    angular_drag: ANGULAR_DRAG,
    energy_cost_per_v_sq: ENERGY_COST_PER_V_SQ,
    angular_energy_cost: ANGULAR_ENERGY_COST,
    vision_cost_per_radius: VISION_COST_PER_RADIUS,
    body_cost_factor: BODY_COST_FACTOR,
    thermal_optimum_penalty: THERMAL_OPTIMUM_PENALTY,
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
    #[serde(with = "serde_arr_hidden")]
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

// Sprint 103: BRAIN_HIDDEN > 32 → serde's native `[T; N]` impl nepokrývá.
// Wrapper sloučí pole do Vec<f32> na serializaci, resp. roundtripuje zpět.
pub mod serde_arr_hidden {
    use super::BRAIN_HIDDEN;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(a: &[f32; BRAIN_HIDDEN], s: S) -> Result<S::Ok, S::Error> {
        a.as_slice().serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<[f32; BRAIN_HIDDEN], D::Error> {
        let v: Vec<f32> = Vec::deserialize(d)?;
        if v.len() != BRAIN_HIDDEN {
            return Err(serde::de::Error::custom("hidden length mismatch"));
        }
        let mut a = [0.0_f32; BRAIN_HIDDEN];
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
        // Sprint 126: ch1, ch2 emit biases. Slabší než ch0 (mating gating
        // tam potřebuje high baseline) — jen aby evoluce neměla cold-start
        // s output ≈ 0.
        b2[10] += INNATE_PHEROMONE_AUX_BIAS;
        b2[11] += INNATE_PHEROMONE_AUX_BIAS;
        Self { hidden_n, w1, b1, w2, b2 }
    }

    pub fn forward(&self, inputs: &[f32; BRAIN_INPUTS]) -> [f32; BRAIN_OUTPUTS] {
        self.forward_with_state(inputs).1
    }

    /// Same forward pass as `forward`, but also returns hidden activations
    /// (needed for Hebbian updates).
    ///
    /// Sprint 112: `wide::f32x8` přes 10 lanes (padded BRAIN_INPUTS 71→80) v
    /// L1, 7 lanes (padded BRAIN_HIDDEN 50→56) v L2. Dead zone (h_n..BRAIN_HIDDEN)
    /// padding zero → mul_add se zapíše jako akumulace přes celou pad šíři bez
    /// větvení. `target-cpu=native` (Sprint 111) zapne FMA, takže jeden chunk
    /// dot product je 1× vfmadd231ps.
    pub fn forward_with_state(
        &self,
        inputs: &[f32; BRAIN_INPUTS],
    ) -> ([f32; BRAIN_HIDDEN], [f32; BRAIN_OUTPUTS]) {
        use wide::f32x8;
        const L1_PAD: usize = ((BRAIN_INPUTS + 7) / 8) * 8;
        const L1_LANES: usize = L1_PAD / 8;
        const L2_PAD: usize = ((BRAIN_HIDDEN + 7) / 8) * 8;
        const L2_LANES: usize = L2_PAD / 8;
        let h_n = self.hidden_n as usize;

        let mut padded_inputs = [0.0_f32; L1_PAD];
        padded_inputs[..BRAIN_INPUTS].copy_from_slice(inputs);
        let mut input_lanes = [f32x8::ZERO; L1_LANES];
        for (lane, chunk) in input_lanes.iter_mut().zip(padded_inputs.chunks_exact(8)) {
            *lane = f32x8::new(chunk.try_into().unwrap());
        }

        let mut hidden = [0.0_f32; BRAIN_HIDDEN];
        let mut padded_w1_row = [0.0_f32; L1_PAD];
        for i in 0..h_n {
            padded_w1_row[..BRAIN_INPUTS].copy_from_slice(&self.w1[i]);
            let mut acc = f32x8::ZERO;
            for (lane_idx, chunk) in padded_w1_row.chunks_exact(8).enumerate() {
                let w = f32x8::new(chunk.try_into().unwrap());
                acc = w.mul_add(input_lanes[lane_idx], acc);
            }
            let sum = self.b1[i] + acc.reduce_add();
            hidden[i] = sum.tanh();
        }

        let mut padded_hidden = [0.0_f32; L2_PAD];
        padded_hidden[..BRAIN_HIDDEN].copy_from_slice(&hidden);
        let mut hidden_lanes = [f32x8::ZERO; L2_LANES];
        for (lane, chunk) in hidden_lanes.iter_mut().zip(padded_hidden.chunks_exact(8)) {
            *lane = f32x8::new(chunk.try_into().unwrap());
        }

        let mut out = [0.0_f32; BRAIN_OUTPUTS];
        let mut padded_w2_row = [0.0_f32; L2_PAD];
        for ((o, row), &bias) in out.iter_mut().zip(self.w2.iter()).zip(self.b2.iter()) {
            padded_w2_row[..BRAIN_HIDDEN].copy_from_slice(row);
            let mut acc = f32x8::ZERO;
            for (lane_idx, chunk) in padded_w2_row.chunks_exact(8).enumerate() {
                let w = f32x8::new(chunk.try_into().unwrap());
                acc = w.mul_add(hidden_lanes[lane_idx], acc);
            }
            let sum = bias + acc.reduce_add();
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

    /// Sprint 104: classic NEAT split_link. Vyber random (input, hidden) pár
    /// s |w|>threshold, deaktivuj přímou cestu (w → 0), insert nový hidden
    /// neuron k mezi nimi: w1[k][input] = 1.0, w2[output][k] = original_w
    /// (resp. pro hidden→hidden link: posuneme přes prostřední neuron).
    /// Vrací `true` pokud mutace proběhla.
    ///
    /// **Topology-preserving:** forward output je při split exactly stejný
    /// jako pre-split (pre-tanh: 1.0 × x = x, post tanh × original_w =
    /// original × tanh(x) — pro malé x ≈ original × x). Drobná nelinearity
    /// drift, ale zhruba zachovává funkci.
    pub fn split_link(&mut self, rng: &mut impl Rng, threshold: f32) -> bool {
        let new_idx = self.hidden_n as usize;
        if new_idx >= BRAIN_HIDDEN {
            return false;
        }
        // Find candidates: w1 entries (h, i) s |w|>threshold (active links).
        let h_n = self.hidden_n as usize;
        let active_inputs = BRAIN_INPUTS_SENSORY + h_n;
        let mut candidates: Vec<(usize, usize, f32)> = Vec::new();
        for h in 0..h_n {
            for i in 0..active_inputs {
                let w = self.w1[h][i];
                if w.abs() > threshold {
                    candidates.push((h, i, w));
                }
            }
        }
        if candidates.is_empty() {
            return false;
        }
        let pick = rng.random_range(0..candidates.len());
        let (h_target, i_src, w_orig) = candidates[pick];
        // Disable direct path
        self.w1[h_target][i_src] = 0.0;
        // Wire through new node k = new_idx
        // input i_src → k weight = 1.0
        // k → h_target via recurrent slot (h_target is fed from inputs[BRAIN_INPUTS_SENSORY + k])
        // But w1 connects inputs to hidden, not hidden to hidden.
        // Original link was input i_src → hidden h_target via w1.
        // New: input i_src → hidden k (= new_idx) via w1[k][i_src] = 1.0
        // Then hidden k must influence hidden h_target. Recurrent path:
        //   k's tanh output → next tick's inputs[BRAIN_INPUTS_SENSORY + k]
        //   → w1[h_target][BRAIN_INPUTS_SENSORY + k] propagates to h_target.
        // Set that recurrent weight to w_orig.
        self.w1[new_idx][i_src] = 1.0;
        self.b1[new_idx] = 0.0;
        // recurrent index for k:
        let rec_idx = BRAIN_INPUTS_SENSORY + new_idx;
        if rec_idx < BRAIN_INPUTS {
            self.w1[h_target][rec_idx] = w_orig;
        }
        // Output side: leave existing w2 (k contributes only via recurrent
        // path, drives same downstream signal next tick).
        self.hidden_n += 1;
        true
    }

    /// Sprint 104: structural mutation — odeber nejnižší prioritní hidden
    /// neuron. Decrement hidden_n a zero-out jeho weights (oba w1 row +
    /// w2 column + b1 entry). Vrací `true` pokud proběhlo (jinak při
    /// hidden_n ≤ BRAIN_HIDDEN_MIN).
    pub fn remove_neuron(&mut self, rng: &mut impl Rng) -> bool {
        let h_n = self.hidden_n as usize;
        if h_n <= BRAIN_HIDDEN_MIN {
            return false;
        }
        let pick = rng.random_range(0..h_n);
        // Zero out w1 row + b1 + w2 column
        for j in 0..BRAIN_INPUTS {
            self.w1[pick][j] = 0.0;
        }
        self.b1[pick] = 0.0;
        for o in 0..BRAIN_OUTPUTS {
            self.w2[o][pick] = 0.0;
        }
        // Compact: pokud pick je poslední, jen decrementuj. Jinak swap-remove
        // (last neuron → pick slot) k zachování dense [0..hidden_n] layout.
        let last = h_n - 1;
        if pick != last {
            self.w1[pick] = self.w1[last];
            self.b1[pick] = self.b1[last];
            for o in 0..BRAIN_OUTPUTS {
                self.w2[o][pick] = self.w2[o][last];
            }
            // Zero out the (now duplicate) last slot.
            for j in 0..BRAIN_INPUTS {
                self.w1[last][j] = 0.0;
            }
            self.b1[last] = 0.0;
            for o in 0..BRAIN_OUTPUTS {
                self.w2[o][last] = 0.0;
            }
            // NOTE: recurrent slot remap — ostatní neurony, které měly
            // vstup z [BRAIN_INPUTS_SENSORY + last] (= last's recurrent feed)
            // teď čtou prázdný slot (0). Ostatní s inputem z [SENSORY+pick]
            // teď čtou last's signal. Je to slight semantic drift; pro tichou
            // kompatibilitu by chtělo plné remapping, ale acceptujeme drobný
            // disruption — selekce kompenzuje.
        }
        self.hidden_n -= 1;
        true
    }

    /// Per-row uniform crossover. Each hidden neuron's `w1` row + `b1`
    /// scalar comes from one parent (50/50); same for output neurons. Per-row
    /// rather than per-weight preserves coordinated patterns within a single
    /// neuron's receptive field.
    ///
    /// Sprint 104: structural mutace mohou rozejít `hidden_n` rodičů. Pokud
    /// neshoda, vezmi menší size (= disjoint hidden slots z většího parenta
    /// nesharedily — drop). Child = min(a.hidden_n, b.hidden_n), per-row
    /// crossover přes shared rozsah, base = parent s menším hidden_n
    /// (zachovává jeho dead-zone weights v 0).
    pub fn crossover(a: &Brain, b: &Brain, rng: &mut impl Rng) -> Brain {
        let h_n = a.hidden_n.min(b.hidden_n) as usize;
        // Base = parent s menším hidden_n (jeho rows beyond h_n jsou zero
        // díky add_neuron / remove_neuron logice).
        let (base, other) = if a.hidden_n <= b.hidden_n {
            (a, b)
        } else {
            (b, a)
        };
        let mut out = *base;
        out.hidden_n = h_n as u32;
        for i in 0..h_n {
            if rng.random::<bool>() {
                out.w1[i] = other.w1[i];
                out.b1[i] = other.b1[i];
            }
        }
        for i in 0..BRAIN_OUTPUTS {
            if rng.random::<bool>() {
                out.w2[i] = other.w2[i];
                out.b2[i] = other.b2[i];
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

fn default_cppn() -> Cppn {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    Cppn::random(&mut rng)
}

fn default_sensor_gains() -> [f32; N_SENSOR_CATEGORIES] {
    [1.0; N_SENSOR_CATEGORIES]
}

fn default_thermal_optimum() -> f32 {
    THERMAL_REF_TEMP
}

fn default_pooled_hidden() -> [f32; BRAIN_HIDDEN] {
    [0.0; BRAIN_HIDDEN]
}

fn default_vision_fov() -> f32 {
    INITIAL_VISION_FOV
}

fn default_spikes() -> [Spike; SPIKE_SLOTS] {
    [Spike::ZERO; SPIKE_SLOTS]
}

fn default_spike_count() -> u8 {
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
    /// Sprint 87: drain rate koeficient pro thermal_optimum penalty.
    /// `dev² × penalty × dt` kde dev = (temp − optimum) / 13.0 (normalized
    /// half-range). Default 1.0; tests mohou override 0.0 pro disable
    /// (např. `step_gpu_matches_cpu` parita — GPU shader nepočítá penalty).
    pub thermal_optimum_penalty: f32,
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
    /// Sprint 121: per-spike runtime stav. `length` je morphable přes brain
    /// output[5] (Sprint 122 aggregate signal); `azimuth_offset`,
    /// `elevation_offset`, `complexity` jsou snapshot z genomu (pure-genetic,
    /// žádný runtime morph). `spike_count` je snapshot — discrete add/remove
    /// se děje jen na reprodukci (mutace).
    #[serde(default = "default_spikes")]
    pub spikes: [Spike; SPIKE_SLOTS],
    #[serde(default = "default_spike_count")]
    pub spike_count: u8,
    /// Sprint 41: snapshot z genomu, runtime morph zatím neexistuje.
    pub shell_thickness: f32,
}

impl Phenotype {
    pub fn from_genome(genome: &Genome) -> Self {
        Self {
            body_length: genome.body_length,
            body_width: genome.body_width,
            body_height: genome.body_height,
            spikes: genome.spikes,
            spike_count: genome.spike_count,
            shell_thickness: genome.shell_thickness,
        }
    }

    /// Sprint 121: primary spike length (slot 0). Pre-S121 callers které
    /// četly `phenotype.spike_length` čtou tohle.
    pub fn primary_spike_length(&self) -> f32 {
        if self.spike_count > 0 {
            self.spikes[0].length
        } else {
            0.0
        }
    }

    /// Sprint 121: sum length přes všechny aktivní spiky. V S121 (spike_count=1)
    /// identické s `primary_spike_length`. Sprint 122+ začne divergovat.
    pub fn total_spike_length(&self) -> f32 {
        let n = self.spike_count.min(SPIKE_SLOTS as u8) as usize;
        self.spikes[..n].iter().map(|s| s.length).sum()
    }

    pub fn active_spikes(&self) -> &[Spike] {
        let n = self.spike_count.min(SPIKE_SLOTS as u8) as usize;
        &self.spikes[..n]
    }

    /// Sprint 124: aggregate spike maintenance cost factor pro GPU step shader
    /// aux[0]. CPU energy drain semantika:
    /// `total_spike_cost_factor × SPIKE_COST_PER_SEC × dt_eff`. S spike_count=1
    /// a complexity=0 redukuje na pre-S121 `spike_length`.
    pub fn total_spike_cost_factor(&self) -> f32 {
        let mut acc = 0.0;
        for spike in self.active_spikes() {
            acc += spike.length * spike_complexity_cost_factor(spike.complexity);
        }
        acc
    }

    /// Sprint 124: primary spike attack factor pro GPU predate shader
    /// `spike_lengths[i]` semantiku. `length × attack_complexity_factor`
    /// pro slot 0 (single-direction predicate). Multi-spike non-primary
    /// sloty na GPU nedostávají bonus — CPU path je multi-spike-faithful.
    pub fn primary_spike_attack_factor(&self) -> f32 {
        if self.spike_count == 0 {
            return 0.0;
        }
        let s = self.spikes[0];
        s.length * spike_complexity_attack_factor(s.complexity)
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

        let actual_dl = (new_len - self.body_length).abs();
        let actual_dw = (new_wid - self.body_width).abs();
        let actual_dh = (new_hgt - self.body_height).abs();

        self.body_length = new_len;
        self.body_width = new_wid;
        self.body_height = new_hgt;

        // Sprint 121: morph[3] aggregate spike length signal — proporčně přes
        // všechny aktivní spiky (per-spike rate ∝ length / sum_lengths). S121
        // s spike_count=1 redukuje na pre-S121 single spike. S122 multi-spike
        // smysluplně rozvrhuje delta.
        let actual_ds = self.apply_spike_morph(raw_ds);

        actual_dl + actual_dw + actual_dh + actual_ds
    }

    fn apply_spike_morph(&mut self, raw_ds: f32) -> f32 {
        let n = self.spike_count.min(SPIKE_SLOTS as u8) as usize;
        if n == 0 || raw_ds == 0.0 {
            return 0.0;
        }
        let sum_lengths: f32 = self.spikes[..n].iter().map(|s| s.length).sum();
        let mut total_delta = 0.0;
        for i in 0..n {
            let weight = if sum_lengths > f32::EPSILON {
                self.spikes[i].length / sum_lengths
            } else {
                1.0 / n as f32
            };
            // Sprint 123: high-complexity spike morphuje pomaleji — geometric
            // structure je commitment, ne behavioral knob. complexity=1 → 50 %
            // rate, complexity=0 → 100 % (pre-S123 sémantika).
            let rate_factor = 1.0 - 0.5 * self.spikes[i].complexity.clamp(0.0, 1.0);
            let delta = raw_ds * weight * rate_factor;
            let new_len = (self.spikes[i].length + delta)
                .clamp(MIN_SPIKE_LENGTH, MAX_SPIKE_LENGTH);
            total_delta += (new_len - self.spikes[i].length).abs();
            self.spikes[i].length = new_len;
        }
        total_delta
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
    #[serde(with = "serde_arr_hidden")]
    pub last_hidden: [f32; BRAIN_HIDDEN],
    pub last_outputs: [f32; BRAIN_OUTPUTS],
    /// Sprint 126: per-channel emission z minulého ticku. Updated v
    /// `emit_pheromones` po výpočtu nového emit value (dříve čteno pro burst).
    #[serde(default)]
    pub last_emit: [f32; N_PHEROMONE_CHANNELS],
    /// Sprint 126: per-channel akumulátor squared tick-to-tick deltas.
    /// `burst_accum[ch] += (current - last_emit[ch])²` per emit_pheromones.
    /// Resetuje se v end-of-gen write_stats. Vyšší hodnota = víc bursty
    /// (continuous emit má small frame-to-frame deltas → low burst score).
    #[serde(default)]
    pub burst_accum: [f32; N_PHEROMONE_CHANNELS],
    /// Sprint 94: cluster-shared brain. Pre-tick mean `last_hidden` přes
    /// bond network (self + bonded partners). Brain recurrent input slots
    /// 21..52 čtou `pooled_hidden` místo `last_hidden` — cluster cells
    /// získají přístup ke kolektivní paměti (proto-distributed cognition).
    /// Solo cells: pooled_hidden == last_hidden (no neighbors). Bonded cells:
    /// average over cluster → bigger effective context window.
    /// `serde(default)` returns zeros pro backward-compat (= same as fresh
    /// cell s žádnou prior activity, behavior matches pre-Sprint-94).
    #[serde(default = "default_pooled_hidden", with = "serde_arr_hidden")]
    pub pooled_hidden: [f32; BRAIN_HIDDEN],
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
            last_emit: [0.0; N_PHEROMONE_CHANNELS],
            burst_accum: [0.0; N_PHEROMONE_CHANNELS],
            pooled_hidden: [0.0; BRAIN_HIDDEN],
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
        self.step_with_climate(dt, world_half, tick, generation, physics, 0.0);
    }

    /// Sprint 112: step se shock-aware climate offsetem. Caller (headless / main)
    /// předem spočítá `climate_offset` přes `climate_shock_offset(...)` na cell
    /// pozici a předá sem; vnitřní `apply_energy_costs` ho přičte k baseline
    /// temperatuře. `climate_offset = 0.0` → byte-identical chování s `step`.
    pub fn step_with_climate(
        &mut self,
        dt: f32,
        world_half: [f32; 3],
        tick: u64,
        generation: u64,
        physics: &PhysicsConfig,
        climate_offset: f32,
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
        // Sprint 86: tick + generation propagace pro time-varying thermal.
        // Sprint 112: + climate_offset (default 0.0) z aktivních ClimateShift
        // shocků, předem spočítaný callerem.
        self.apply_energy_costs(dt, world_half, tick, generation, physics, climate_offset);
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
        climate_offset: f32,
    ) {
        let temp =
            temperature_at_z(self.position[2], world_half, tick, generation) + climate_offset;
        let metabolism = metabolism_factor(temp);
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
        // Sprint 121: sum maintenance přes aktivní spiky. Sprint 123: per-spike
        // quadratic complexity multiplier. S spike_count=1, complexity=0
        // redukuje na pre-S121 single spike cost.
        let mut spike_cost = 0.0;
        for spike in self.phenotype.active_spikes() {
            spike_cost += spike.length * spike_complexity_cost_factor(spike.complexity);
        }
        self.energy -= spike_cost * SPIKE_COST_PER_SEC * dt_eff;
        // Sprint 41: shell maintenance — defensive armor stojí víc než spike,
        // protože pokrývá celý povrch.
        self.energy -= self.phenotype.shell_thickness * SHELL_COST_PER_SEC * dt_eff;
        // Sprint 27 attack maintenance: cost ∝ max(0, output[6]).
        let attack_strength = self.last_outputs[6].max(0.0);
        self.energy -= attack_strength * ATTACK_COST_PER_SEC * dt_eff;
        // Sprint 87: thermal stress penalty. Quadratic na deviation od optima,
        // independent metabolism (tepelný stres = extra cost, ne reduced rate).
        // `dt`, ne `dt_eff` — penalty není Q10-modulated.
        let dev = (temp - self.genome.thermal_optimum) / 13.0;
        self.energy -= dev * dev * physics.thermal_optimum_penalty * dt;
        // Sprint 97: sensor gain drain. Sum gains × cost × dt_eff (Q10-modulated
        // jako ostatní biological costs). Cluster cells which off-load sensor
        // duties to specialists (gain → 0 v category) save energy proportionally.
        let total_sensor_gain: f32 = self.genome.sensor_gains.iter().sum();
        self.energy -= total_sensor_gain * SENSOR_GAIN_COST * dt_eff;
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

    /// Bonus predation gain pokud má attacker aspoň jeden spike, který směřuje
    /// na cíl (cosine > `SPIKE_DOT_THRESHOLD`). Sprint 121: iteruje všechny
    /// aktivní spiky — každý kontribuuje vlastní cone test. S spike_count=1
    /// a azimuth/elev=0 redukuje na pre-S121 single-spike behavior.
    pub fn spike_bonus_against(&self, target_pos: [f32; 3]) -> f32 {
        let dx = target_pos[0] - self.position[0];
        let dy = target_pos[1] - self.position[1];
        let dz = target_pos[2] - self.position[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        if dist_sq < f32::EPSILON {
            return 0.0;
        }
        let dist = dist_sq.sqrt();
        let to_target = [dx / dist, dy / dist, dz / dist];
        let mut total = 0.0;
        for spike in self.phenotype.active_spikes() {
            if spike.length <= 0.0 {
                continue;
            }
            let dir = spike_direction(self.heading, self.pitch, spike);
            let cos_angle = dir[0] * to_target[0] + dir[1] * to_target[1] + dir[2] * to_target[2];
            if cos_angle < SPIKE_DOT_THRESHOLD {
                continue;
            }
            total += PREDATION_GAIN_PER_TICK
                * spike.length
                * spike_complexity_attack_factor(spike.complexity)
                * SPIKE_PREDATION_BONUS;
        }
        total
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

    /// Sprint 122: rozšířený eat test — kombinuje ellipsoid (`eat_test`)
    /// a per-spike forward grab cones u tipu spike. Spike kuželek má vrchol
    /// `cell_pos + dir × (eff_radius + length)` (tip), osu = spike direction,
    /// half-angle `SPIKE_GRAB_HALF_ANGLE × (1 + complexity × COMPLEXITY_GRAB_GAIN)`,
    /// range `length × SPIKE_GRAB_REACH_BONUS` od tipu. Cell sní food pokud
    /// projde ELLIPSOID NEBO kterýkoli aktivní spike kuželek. Pre-S122
    /// (`spike_count = 0` nebo všechny `length = 0`) redukuje na `eat_test`.
    pub fn eat_test_with_spikes(&self, food: &Food, eat_factor: f32) -> bool {
        if self.eat_test(food, eat_factor) {
            return true;
        }
        let eff_r = self.phenotype.effective_radius();
        for spike in self.phenotype.active_spikes() {
            if spike.length <= 0.0 {
                continue;
            }
            let dir = spike_direction(self.heading, self.pitch, spike);
            let tip_dist = eff_r + spike.length;
            let tip = [
                self.position[0] + dir[0] * tip_dist,
                self.position[1] + dir[1] * tip_dist,
                self.position[2] + dir[2] * tip_dist,
            ];
            let dx = food.position[0] - tip[0];
            let dy = food.position[1] - tip[1];
            let dz = food.position[2] - tip[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let max_range = spike.length * SPIKE_GRAB_REACH_BONUS;
            if dist_sq > max_range * max_range {
                continue;
            }
            if dist_sq < f32::EPSILON {
                return true;
            }
            let dist = dist_sq.sqrt();
            let cos_food = (dx * dir[0] + dy * dir[1] + dz * dir[2]) / dist;
            let half_angle =
                SPIKE_GRAB_HALF_ANGLE * spike_complexity_grab_factor(spike.complexity);
            if cos_food >= half_angle.cos() {
                return true;
            }
        }
        false
    }

    /// Sprint 41: ellipsoidální acceptance + energy gain při hitu.
    /// Sprint 122: zahrnuje per-spike grab cones (viz `eat_test_with_spikes`).
    pub fn try_eat(&mut self, food: &Food, eat_factor: f32, food_value: f32) -> bool {
        if self.eat_test_with_spikes(food, eat_factor) {
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
            spikes: [Spike::ZERO; SPIKE_SLOTS],
            spike_count: 1,
            shell_thickness: 0.0,
            adhesion_type: 0,
            bond_stiffness: BOND_STIFFNESS,
            bond_damping: BOND_DAMPING,
            vision_fov: INITIAL_VISION_FOV,
            thermal_optimum: THERMAL_REF_TEMP,
            carnivore_score: 0.0,
            // Sprint 97: zero gains v test fixture aby legacy energy-drain testy
            // (pre-S97) neviděly sensor_gain cost. Per-test override když test
            // sensor pooling testuje.
            sensor_gains: [0.0; N_SENSOR_CATEGORIES],
            brain: dummy_brain(),
            cppn: default_cppn(),
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
            split_link_rate: 0.0,
            remove_neuron_rate: 0.0,
            sigma_vision_fov: 0.0,
            sigma_thermal_optimum: 0.0,
            sigma_carnivore_score: 0.0,
            sigma_sensor_gain: 0.0,
            spike_count_mutation_rate: 0.0,
            sigma_spike_orientation: 0.0,
            sigma_spike_complexity: 0.0,
            sigma_spike_length_secondary: 0.0,
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
            spikes: {
                let mut s = [Spike::ZERO; SPIKE_SLOTS];
                s[0].length = 0.4;
                s
            },
            spike_count: 1,
            shell_thickness: 0.0,
            adhesion_type: 0,
            bond_stiffness: BOND_STIFFNESS,
            bond_damping: BOND_DAMPING,
            vision_fov: INITIAL_VISION_FOV,
            thermal_optimum: THERMAL_REF_TEMP,
            carnivore_score: 0.0,
            sensor_gains: [1.0; N_SENSOR_CATEGORIES],
            brain: Brain {
                hidden_n: BRAIN_HIDDEN_DEFAULT as u32,
                w1: [[1.0; BRAIN_INPUTS]; BRAIN_HIDDEN],
                b1: [0.3; BRAIN_HIDDEN],
                w2: [[1.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
                b2: [0.5; BRAIN_OUTPUTS],
            },
            cppn: default_cppn(),
        };
        let m = g.mutate(&mut rng, &zero_cfg());
        assert_eq!(m.max_speed, 50.0);
        assert_eq!(m.color_hue, 120.0);
        assert_eq!(m.vision_radius, 40.0);
        assert_eq!(m.turn_rate, 2.5);
        assert_eq!(m.body_length, 1.1);
        assert_eq!(m.body_width, 0.9);
        assert_eq!(m.spikes[0].length, 0.4);
        assert_eq!(m.spike_count, 1);
        // Sprint 106: brain je derived z mutated CPPN. S sigma=0 v zero_cfg,
        // ale CPPN má vlastní mutation rates (CPPN_MUTATION_CONFIG) které
        // jsou non-zero — brain weights NEZACHOVANÉ identity. Test now
        // validates structural compatibility místo identity.
        assert_eq!(m.brain.w1.len(), g.brain.w1.len());
        assert_eq!(m.brain.b1.len(), g.brain.b1.len());
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
            split_link_rate: 0.0,
            remove_neuron_rate: 0.0,
            sigma_vision_fov: 10.0,
            sigma_thermal_optimum: 100.0,
            sigma_carnivore_score: 100.0,
            sigma_sensor_gain: 100.0,
            spike_count_mutation_rate: 0.5,
            sigma_spike_orientation: 10.0,
            sigma_spike_complexity: 10.0,
            sigma_spike_length_secondary: 10.0,
        };
        for _ in 0..1000 {
            let m = g.mutate(&mut rng, &cfg);
            for spike in m.spikes.iter() {
                assert!((MIN_SPIKE_AZIMUTH..=MAX_SPIKE_AZIMUTH).contains(&spike.azimuth_offset));
                assert!(
                    (MIN_SPIKE_ELEVATION..=MAX_SPIKE_ELEVATION).contains(&spike.elevation_offset)
                );
                assert!((MIN_SPIKE_COMPLEXITY..=MAX_SPIKE_COMPLEXITY).contains(&spike.complexity));
            }
            assert!(m.spike_count <= SPIKE_SLOTS as u8);
            assert!(m.max_speed >= MIN_SPEED);
            assert!(m.max_speed <= MAX_SPEED, "Sprint 73: speed cap respected");
            assert!(m.color_hue >= 0.0 && m.color_hue < HUE_RANGE);
            assert!(m.vision_radius >= MIN_VISION);
            assert!(m.turn_rate >= MIN_TURN_RATE);
            assert!((MIN_BODY_LENGTH..=MAX_BODY_LENGTH).contains(&m.body_length));
            assert!((MIN_BODY_WIDTH..=MAX_BODY_WIDTH).contains(&m.body_width));
            for spike in m.spikes.iter() {
                assert!((MIN_SPIKE_LENGTH..=MAX_SPIKE_LENGTH).contains(&spike.length));
            }
            assert!((MIN_VISION_FOV..=MAX_VISION_FOV).contains(&m.vision_fov));
            assert!((MIN_THERMAL_OPTIMUM..=MAX_THERMAL_OPTIMUM).contains(&m.thermal_optimum));
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
    fn climate_offset_default_zero() {
        let pos_xy = [123.0, -45.0];
        // Empty events → 0.0.
        let off = climate_shock_offset(&[], 50, pos_xy, WORLD_HALF);
        assert!(off.abs() < 1e-6, "empty events must give 0.0, got {}", off);
        // Non-ClimateShift event (HazardPulse) → 0.0.
        let event = ShockEvent {
            kind: ShockKind::HazardPulse,
            start_gen: 0,
            duration_gen: 10,
            ramp_gens: 2,
            intensity: 1.0,
            center_xy: None,
            radius: None,
        };
        let off = climate_shock_offset(&[event], 5, pos_xy, WORLD_HALF);
        assert!(off.abs() < 1e-6, "HazardPulse must not affect climate, got {}", off);
    }

    #[test]
    fn climate_offset_global_shift_at_peak() {
        // Sprint 112: 1 global ClimateShift, intensity = 1, peak ramp = 1, no
        // spatial → offset = CLIMATE_SHIFT_MAX_OFFSET (= 5.0).
        let event = ShockEvent {
            kind: ShockKind::ClimateShift,
            start_gen: 100,
            duration_gen: 10,
            ramp_gens: 2,
            intensity: 1.0,
            center_xy: None,
            radius: None,
        };
        // Plateau (gen 102..=107) → ramp = 1.0.
        let off = climate_shock_offset(&[event], 105, [50.0, -10.0], WORLD_HALF);
        assert!(
            (off - CLIMATE_SHIFT_MAX_OFFSET).abs() < 1e-5,
            "global peak must give CLIMATE_SHIFT_MAX_OFFSET, got {}",
            off
        );
        // Pre-start: 0.0.
        let off_before = climate_shock_offset(&[event], 99, [50.0, -10.0], WORLD_HALF);
        assert!(off_before.abs() < 1e-6);
        // Post-end: 0.0.
        let off_after = climate_shock_offset(&[event], 110, [50.0, -10.0], WORLD_HALF);
        assert!(off_after.abs() < 1e-6);
    }

    #[test]
    fn temperature_with_shocks_matches_baseline_when_no_events() {
        // Sprint 112: temperature_at_z_with_shocks musí být byte-identical
        // s temperature_at_z když events.empty (default off path).
        let half = [960.0, 540.0, 50.0];
        let pos_xy = [200.0, -100.0];
        for &(z, tick, gen) in &[
            (0.0_f32, 0_u64, 0_u64),
            (50.0, 100, 5),
            (-50.0, 1000, 25),
            (25.0, THERMAL_DIURNAL_PERIOD_TICKS / 4, CYCLE_GEN_PERIOD / 4),
            (-25.0, 3 * THERMAL_DIURNAL_PERIOD_TICKS / 4, CYCLE_GEN_PERIOD / 2),
        ] {
            let base = temperature_at_z(z, half, tick, gen);
            let with_shocks = temperature_at_z_with_shocks(z, half, tick, gen, &[], pos_xy);
            assert_eq!(
                base.to_bits(),
                with_shocks.to_bits(),
                "byte-identical required: z={}, tick={}, gen={}",
                z,
                tick,
                gen
            );
        }
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
    fn pool_bonded_hidden_solo_cell_returns_self() {
        // Sprint 94: solo cell s no bonds → pooled == last_hidden.
        let mut cell = base_cell();
        for k in 0..BRAIN_HIDDEN {
            cell.last_hidden[k] = (k as f32) * 0.1;
        }
        let pooled = pool_bonded_hidden(&cell, |_| None);
        assert_eq!(pooled, cell.last_hidden);
    }

    #[test]
    fn pool_bonded_hidden_pair_averages() {
        // Sprint 94: pair cell A bonded to B → A.pooled = (A.last + B.last) / 2.
        let mut cell = base_cell();
        cell.cell_id = 1;
        cell.bonds[0] = Some(Bond {
            other_cell_id: 2,
            rest_length: 5.0,
            stiffness: BOND_STIFFNESS,
            damping: BOND_DAMPING,
            age_ticks: 0,
        });
        for k in 0..BRAIN_HIDDEN {
            cell.last_hidden[k] = 1.0;
        }
        let mut partner_hidden = [0.0; BRAIN_HIDDEN];
        for k in 0..BRAIN_HIDDEN {
            partner_hidden[k] = 3.0;
        }
        let pooled = pool_bonded_hidden(&cell, |id| {
            if id == 2 { Some(partner_hidden) } else { None }
        });
        for k in 0..BRAIN_HIDDEN {
            assert!((pooled[k] - 2.0).abs() < 1e-6, "expected 2.0, got {}", pooled[k]);
        }
    }

    #[test]
    fn pool_bonded_hidden_skips_dead_partners() {
        // Sprint 94: missing partner (despawned mid-tick) skipped, pool jen
        // s alive bonded.
        let mut cell = base_cell();
        cell.bonds[0] = Some(Bond {
            other_cell_id: 99,
            rest_length: 5.0,
            stiffness: BOND_STIFFNESS,
            damping: BOND_DAMPING,
            age_ticks: 0,
        });
        for k in 0..BRAIN_HIDDEN {
            cell.last_hidden[k] = 5.0;
        }
        // Dead partner returns None.
        let pooled = pool_bonded_hidden(&cell, |_| None);
        // Pool falls back to self only.
        assert_eq!(pooled, cell.last_hidden);
    }

    #[test]
    fn sensor_slot_category_covers_known_indices() {
        // Sprint 97: každý sensory slot v 0..BRAIN_INPUTS_SENSORY musí buď
        // vrátit Some(category) nebo None (proprio). Žádný slot nevypadne.
        // Defensive (damage_norm) je slot 14, density slot 20.
        assert_eq!(sensor_slot_category(0), Some(SENSOR_CATEGORY_VISION));
        assert_eq!(sensor_slot_category(7), Some(SENSOR_CATEGORY_CHEMISTRY));
        assert_eq!(sensor_slot_category(14), Some(SENSOR_CATEGORY_DEFENSIVE));
        assert_eq!(sensor_slot_category(20), Some(SENSOR_CATEGORY_DEFENSIVE));
        // Proprio slot (energy/speed/heading) → None.
        assert!(sensor_slot_category(4).is_none());
    }

    #[test]
    fn apply_sensor_gains_scales_only_categorized_slots() {
        // Sprint 97: gains aplikuje na sensory slots, proprio nedotčeno.
        let mut inputs = [1.0_f32; BRAIN_INPUTS];
        let gains = [2.0, 0.5, 0.0];
        apply_sensor_gains(&mut inputs, &gains);
        // Vision slot 0 = 2× gain
        assert!((inputs[0] - 2.0).abs() < 1e-6);
        // Chemistry slot 7 = 0.5× gain
        assert!((inputs[7] - 0.5).abs() < 1e-6);
        // Defensive slot 14 = 0× gain
        assert!((inputs[14] - 0.0).abs() < 1e-6);
        // Proprio slot 4 → unchanged.
        assert!((inputs[4] - 1.0).abs() < 1e-6);
        // Recurrent slot mimo BRAIN_INPUTS_SENSORY → unchanged.
        assert!((inputs[BRAIN_INPUTS_SENSORY] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pool_bonded_sensors_solo_cell_returns_own() {
        // Sprint 97: solo cell bez bonds → pooled == own (žádný partner).
        let cell = base_cell();
        let mut own = [0.0; BRAIN_INPUTS];
        own[0] = 0.5;
        own[7] = -0.3;
        let pooled = pool_bonded_sensors(&cell, &own, |_| None);
        assert_eq!(pooled, own);
    }

    #[test]
    fn pool_bonded_sensors_takes_max_magnitude_from_partner() {
        // Sprint 97: partner má silnější vision signal → pooled převezme partner.
        // Magnitude-based pooling (abs()) — invertovaný gradient (-0.9) přebije
        // slabý kladný (0.2).
        let mut cell = base_cell();
        cell.cell_id = 1;
        cell.bonds[0] = Some(Bond {
            other_cell_id: 2,
            rest_length: 5.0,
            stiffness: BOND_STIFFNESS,
            damping: BOND_DAMPING,
            age_ticks: 0,
        });
        let mut own = [0.0; BRAIN_INPUTS];
        own[0] = 0.2;
        own[7] = 0.5;
        let mut partner = [0.0; BRAIN_INPUTS];
        partner[0] = -0.9;
        partner[7] = 0.1;
        let pooled = pool_bonded_sensors(&cell, &own, |id| {
            if id == 2 { Some(partner) } else { None }
        });
        // Vision slot: |-0.9| > |0.2| → partner wins
        assert!((pooled[0] - (-0.9)).abs() < 1e-6);
        // Chemistry slot: |0.5| > |0.1| → own wins
        assert!((pooled[7] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn pool_bonded_sensors_ignores_proprio_slots() {
        // Sprint 97: proprio (energy, speed, heading) NESMÍ poolnout — každá
        // buňka má svůj vlastní stav.
        let mut cell = base_cell();
        cell.bonds[0] = Some(Bond {
            other_cell_id: 7,
            rest_length: 5.0,
            stiffness: BOND_STIFFNESS,
            damping: BOND_DAMPING,
            age_ticks: 0,
        });
        let mut own = [0.0; BRAIN_INPUTS];
        own[4] = 0.1; // proprio
        let mut partner = [0.0; BRAIN_INPUTS];
        partner[4] = 0.99; // partner higher proprio
        let pooled = pool_bonded_sensors(&cell, &own, |id| {
            if id == 7 { Some(partner) } else { None }
        });
        // Proprio slot 4 zůstává own — nebyl poolen.
        assert!((pooled[4] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn cell_exposure_endpoints() {
        // Sprint 96: quadratic falloff — `(linear)²`. Solo cell fully exposed,
        // first bondy mají dramatic defense bonus, ≥4 plně immune.
        assert!((cell_exposure(0) - 1.0).abs() < 1e-6);
        // 1 bond → 0.75² = 0.5625
        assert!((cell_exposure(1) - 0.5625).abs() < 1e-6);
        // 2 bonds → 0.5² = 0.25
        assert!((cell_exposure(2) - 0.25).abs() < 1e-6);
        // 3 bonds → 0.25² = 0.0625
        assert!((cell_exposure(3) - 0.0625).abs() < 1e-6);
        // 4+ bonds → 0 (effectively immune)
        assert!(cell_exposure(4).abs() < 1e-6);
        assert!(cell_exposure(10).abs() < 1e-6);
    }

    #[test]
    fn eat_efficiency_diet_specialization() {
        // Pure herbivore preference for plant.
        assert!((eat_efficiency(FoodKind::Plant, 0.0) - 1.0).abs() < 1e-6);
        assert!(eat_efficiency(FoodKind::HunterCarrion, 0.0).abs() < 1e-6);
        // Pure carnivore preference for hunter carrion.
        assert!(eat_efficiency(FoodKind::Plant, 1.0).abs() < 1e-6);
        assert!((eat_efficiency(FoodKind::HunterCarrion, 1.0) - 1.0).abs() < 1e-6);
        // Mixed (0.5) — plant 0.5, carrion 0.5, hunter carrion 0.5.
        assert!((eat_efficiency(FoodKind::Plant, 0.5) - 0.5).abs() < 1e-6);
        assert!((eat_efficiency(FoodKind::Carrion, 0.5) - 0.5).abs() < 1e-6);
        assert!((eat_efficiency(FoodKind::HunterCarrion, 0.5) - 0.5).abs() < 1e-6);
        // Cell carrion: universally 0.5 — compromise food.
        assert!((eat_efficiency(FoodKind::Carrion, 0.0) - 0.5).abs() < 1e-6);
        assert!((eat_efficiency(FoodKind::Carrion, 1.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn food_base_value_per_kind() {
        // Sprint 92: kind-dependent base values. Hunter carrion richest.
        assert!((food_base_value(FoodKind::Plant) - PLANT_FOOD_VALUE).abs() < 1e-6);
        assert!((food_base_value(FoodKind::Carrion) - CARRION_FOOD_VALUE).abs() < 1e-6);
        assert!(
            (food_base_value(FoodKind::HunterCarrion) - HUNTER_CARRION_FOOD_VALUE).abs() < 1e-6
        );
        assert!(food_base_value(FoodKind::HunterCarrion) > food_base_value(FoodKind::Plant));
    }

    #[test]
    fn coop_food_lifecycle_no_arrivals_expires() {
        // Sprint 128: bez arrivals coop node prošlý TIME_WINDOW musí vrátit
        // is_expired = true → caller cleanup, no reward.
        let mut coop = CoopFood::new([0.0, 0.0, 0.0], 0);
        assert!(!coop.is_expired(0));
        assert!(!coop.is_expired((COOP_FOOD_TIME_WINDOW_TICKS as u64).saturating_sub(1)));
        assert!(coop.is_expired(COOP_FOOD_TIME_WINDOW_TICKS as u64));
        let mut cells: [Cell; 0] = [];
        assert!(!try_trigger_coop(&mut coop, &mut cells));
        assert!(!coop.triggered);
    }

    #[test]
    fn coop_food_threshold_triggers_reward() {
        // Sprint 128: 3 cells s unique cell_id v arrivals → trigger distribuuje
        // COOP_FOOD_REWARD_PER_CELL na každého. Caller pak coop odstraní.
        let mut coop = CoopFood::new([0.0, 0.0, 0.0], 0);
        let mut cells = [
            Cell {
                cell_id: 1,
                energy: 50.0,
                ..base_cell()
            },
            Cell {
                cell_id: 2,
                energy: 30.0,
                ..base_cell()
            },
            Cell {
                cell_id: 3,
                energy: 70.0,
                ..base_cell()
            },
        ];
        register_coop_arrival(&mut coop, 1);
        register_coop_arrival(&mut coop, 2);
        register_coop_arrival(&mut coop, 3);
        // Duplicate id ignored.
        assert!(!register_coop_arrival(&mut coop, 1));
        assert_eq!(coop.arrivals.len(), 3);
        assert!(try_trigger_coop(&mut coop, &mut cells));
        assert!(coop.triggered);
        assert!((cells[0].energy - (50.0 + COOP_FOOD_REWARD_PER_CELL)).abs() < 1e-4);
        assert!((cells[1].energy - (30.0 + COOP_FOOD_REWARD_PER_CELL)).abs() < 1e-4);
        assert!((cells[2].energy - (70.0 + COOP_FOOD_REWARD_PER_CELL)).abs() < 1e-4);
        // Idempotent: druhý try_trigger nesmí znovu rozdat reward.
        assert!(!try_trigger_coop(&mut coop, &mut cells));
        assert!((cells[0].energy - (50.0 + COOP_FOOD_REWARD_PER_CELL)).abs() < 1e-4);
    }

    #[test]
    fn coop_food_below_threshold_no_reward() {
        // Sprint 128: 2 < REQUIRED_ARRIVALS (=3) → trigger nefiringuje, energie
        // beze změny, coop stále alive.
        let mut coop = CoopFood::new([0.0, 0.0, 0.0], 0);
        let mut cells = [
            Cell {
                cell_id: 1,
                energy: 50.0,
                ..base_cell()
            },
            Cell {
                cell_id: 2,
                energy: 30.0,
                ..base_cell()
            },
        ];
        register_coop_arrival(&mut coop, 1);
        register_coop_arrival(&mut coop, 2);
        assert!(!try_trigger_coop(&mut coop, &mut cells));
        assert!(!coop.triggered);
        assert!((cells[0].energy - 50.0).abs() < 1e-4);
        assert!((cells[1].energy - 30.0).abs() < 1e-4);
        assert!(!coop.is_expired(0));
    }

    #[test]
    fn carnivore_score_in_genome_random_initial_range() {
        let mut rng = StdRng::seed_from_u64(0xCA12);
        for _ in 0..100 {
            let g = Genome::random(&mut rng);
            assert!(
                (0.0..0.5).contains(&g.carnivore_score),
                "carnivore_score {} out of init range [0, 0.5]",
                g.carnivore_score
            );
        }
    }

    #[test]
    fn thermal_optimum_random_in_range() {
        // Sprint 87: Genome::random by měl init thermal_optimum uniform v range.
        let mut rng = StdRng::seed_from_u64(0x7E0);
        for _ in 0..100 {
            let g = Genome::random(&mut rng);
            assert!(
                (MIN_THERMAL_OPTIMUM..=MAX_THERMAL_OPTIMUM).contains(&g.thermal_optimum),
                "optimum {} out of range",
                g.thermal_optimum
            );
        }
    }

    #[test]
    fn apply_energy_costs_thermal_stress_quadratic() {
        // Sprint 87: penalty kvadratický v |temp - optimum|. Cell s optimum
        // matching local temp platí 0 penalty; cell s extreme deviation platí
        // PENALTY × (dev/13)². Compare 3 cells: matched, half-deviation,
        // extreme.
        let half = [1000.0, 1000.0, 50.0];
        let physics = PhysicsConfig {
            drag: 0.0,
            angular_drag: 0.0,
            energy_cost_per_v_sq: 0.0,
            angular_energy_cost: 0.0,
            vision_cost_per_radius: 0.0,
            body_cost_factor: 0.0,
            thermal_optimum_penalty: 1.0,
        };
        // Cell at z=0 → temp = REF = 17. Optimum = 17 → no penalty.
        let mut matched = base_cell();
        matched.position = [0.0, 0.0, 0.0];
        matched.genome.thermal_optimum = THERMAL_REF_TEMP;
        matched.step(1.0, half, 0, 0, &physics);
        let matched_drain = 100.0 - matched.energy;
        assert!(matched_drain.abs() < 0.01, "matched drain {matched_drain}");
        // Cell at z=0 (temp=17), optimum=4 (= BOTTOM, dev=13). penalty/sec =
        // (13/13)² × 1.0 = 1.0.
        let mut extreme = base_cell();
        extreme.position = [0.0, 0.0, 0.0];
        extreme.genome.thermal_optimum = MIN_THERMAL_OPTIMUM;
        extreme.step(1.0, half, 0, 0, &physics);
        let extreme_drain = 100.0 - extreme.energy;
        assert!(
            (extreme_drain - 1.0).abs() < 0.01,
            "extreme drain {extreme_drain}"
        );
        // Cell at z=0, optimum = 17 + 6.5 = 23.5 (= half-deviation). penalty/sec
        // = (6.5/13)² × 1.0 = 0.25.
        let mut half_dev = base_cell();
        half_dev.position = [0.0, 0.0, 0.0];
        half_dev.genome.thermal_optimum = THERMAL_REF_TEMP + 6.5;
        half_dev.step(1.0, half, 0, 0, &physics);
        let half_dev_drain = 100.0 - half_dev.energy;
        assert!(
            (half_dev_drain - 0.25).abs() < 0.01,
            "half-dev drain {half_dev_drain}"
        );
    }

    #[test]
    fn populate_brain_inputs_writes_temperature_slot() {
        // Sprint 87: slot 20 = tanh((temp - REF) / 10).
        let mut cell = base_cell();
        let sensors = BrainSensors {
            nearest_food: None,
            nearest_cell: None,
            neighbors_in_vision: 0,
            smell_grad: [0.0; 3],
            pheromone_grads: [[0.0; 3]; N_PHEROMONE_CHANNELS],
            temperature_local: THERMAL_REF_TEMP, // exact REF → tanh(0) = 0
        };
        let inputs = populate_brain_inputs(&mut cell, &sensors, 50.0);
        assert!((inputs[20] - 0.0).abs() < 1e-4, "REF should be 0, got {}", inputs[20]);
        // Test top temp.
        let sensors_top = BrainSensors {
            temperature_local: THERMAL_TOP,
            ..sensors
        };
        let inputs_top = populate_brain_inputs(&mut cell, &sensors_top, 50.0);
        // tanh(13/10) = tanh(1.3) ≈ 0.86
        assert!(
            (inputs_top[20] - 1.3_f32.tanh()).abs() < 1e-4,
            "TOP got {}",
            inputs_top[20]
        );
        // Test bottom temp.
        let sensors_bot = BrainSensors {
            temperature_local: THERMAL_BOTTOM,
            ..sensors
        };
        let inputs_bot = populate_brain_inputs(&mut cell, &sensors_bot, 50.0);
        assert!(
            (inputs_bot[20] - (-1.3_f32).tanh()).abs() < 1e-4,
            "BOTTOM got {}",
            inputs_bot[20]
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
            thermal_optimum_penalty: 0.0,
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
            last_emit: [0.0; N_PHEROMONE_CHANNELS],
            burst_accum: [0.0; N_PHEROMONE_CHANNELS],
            pooled_hidden: [0.0; BRAIN_HIDDEN],
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
            thermal_optimum_penalty: 0.0,
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
            thermal_optimum_penalty: 0.0,
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
            thermal_optimum_penalty: 0.0,
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
            thermal_optimum_penalty: 0.0,
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
        let food = Food { position: [5.0, 0.0, 0.0], age_ticks: 0, kind: FoodKind::Plant };
        assert!(cell.try_eat(&food, 8.0, 20.0));
        assert_eq!(cell.energy, 70.0);
    }

    #[test]
    fn try_eat_outside_radius_returns_false_and_keeps_energy() {
        let mut cell = Cell {
            energy: 50.0,
            ..base_cell()
        };
        let food = Food { position: [20.0, 0.0, 0.0], age_ticks: 0, kind: FoodKind::Plant };
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
            spikes: [Spike::ZERO; SPIKE_SLOTS],
            spike_count: 1,
            shell_thickness: 0.0,
            adhesion_type: 1,
            bond_stiffness: 2.0,
            bond_damping: 0.3,
            vision_fov: MIN_VISION_FOV,
            thermal_optimum: MIN_THERMAL_OPTIMUM,
            carnivore_score: 0.0,
            sensor_gains: [MIN_SENSOR_GAIN; N_SENSOR_CATEGORIES],
            brain: dummy_brain(),
            cppn: default_cppn(),
        };
        let b = Genome {
            max_speed: 90.0,
            color_hue: 200.0,
            vision_radius: 80.0,
            turn_rate: 5.0,
            body_length: 1.5,
            body_width: 1.4,
            body_height: 1.3,
            spikes: {
                let mut s = [Spike::ZERO; SPIKE_SLOTS];
                s[0].length = 0.8;
                s
            },
            spike_count: 1,
            shell_thickness: 0.5,
            adhesion_type: 5,
            bond_stiffness: 8.0,
            bond_damping: 1.0,
            vision_fov: MAX_VISION_FOV,
            thermal_optimum: MAX_THERMAL_OPTIMUM,
            carnivore_score: 1.0,
            sensor_gains: [MAX_SENSOR_GAIN; N_SENSOR_CATEGORIES],
            brain: dummy_brain(),
            cppn: default_cppn(),
        };
        for _ in 0..100 {
            let c = Genome::crossover(&a, &b, &mut rng);
            assert!(c.max_speed == 30.0 || c.max_speed == 90.0);
            assert!(c.color_hue == 10.0 || c.color_hue == 200.0);
            assert!(c.vision_radius == 20.0 || c.vision_radius == 80.0);
            assert!(c.turn_rate == 1.0 || c.turn_rate == 5.0);
            assert!(c.body_length == 0.5 || c.body_length == 1.5);
            assert!(c.body_width == 0.6 || c.body_width == 1.4);
            assert!(c.spikes[0].length == 0.0 || c.spikes[0].length == 0.8);
            assert!(c.vision_fov == MIN_VISION_FOV || c.vision_fov == MAX_VISION_FOV);
            assert!(
                c.thermal_optimum == MIN_THERMAL_OPTIMUM
                    || c.thermal_optimum == MAX_THERMAL_OPTIMUM
            );
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
        // Sprint 126: BRAIN_INPUTS 71 → 77 zvýšilo input variance (víc gaussian
        // weight noise feeded do hidden → větší tail variance v output[1]).
        // INNATE_THRUST_BIAS posune mean kladně, ale fraction positive se snížila
        // z >75 % na ~70 %. Mean je dál >0.3, evolutionary jumpstart funguje.
        assert!(
            count_positive > n * 2 / 3,
            "expected >66% positive, got {}/{}",
            count_positive,
            n
        );
    }

    #[test]
    fn brain_forward_zero_weights_outputs_tanh_of_output_biases() {
        // Zero weights kill signal flow at both layers — output equals tanh(b2),
        // independent of b1 (the hidden activations get zeroed by w2).
        // Sprint 126: BRAIN_OUTPUTS = 12 (+2 ch1/ch2 emit), test still passes
        // because we read just outputs[0] and [1].
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
        assert_eq!(outputs.len(), BRAIN_OUTPUTS);
        assert!((outputs[0] - 0.5_f32.tanh()).abs() < 1e-6);
        assert!((outputs[1] - (-0.5_f32).tanh()).abs() < 1e-6);
        // ch1, ch2 (sloty 10, 11) → b2 = 0 → output = tanh(0) = 0.
        assert!(outputs[10].abs() < 1e-6);
        assert!(outputs[11].abs() < 1e-6);
    }

    #[test]
    fn multi_channel_pheromone_emit_costs_proportionally() {
        // Sprint 126 sanity: tři kanály emit at full strength = 3× cost vs.
        // jeden. Validates summed cost model: cost = total_emit × cost_rate × dt.
        // Test je jen formální (cost rovnice je v emit_pheromones binárky, ne v
        // lib), takže testujeme přímo equation v isolation.
        let cost_rate = PHEROMONE_COST_PER_RATE;
        let dt = 1.0 / FIXED_TIMESTEP_HZ;
        let emit_single = [1.0_f32, 0.0, 0.0];
        let emit_triple = [1.0_f32, 1.0, 1.0];
        let total_single: f32 = emit_single.iter().sum();
        let total_triple: f32 = emit_triple.iter().sum();
        let cost_single = total_single * cost_rate * dt;
        let cost_triple = total_triple * cost_rate * dt;
        assert!((cost_triple / cost_single - 3.0).abs() < 1e-6);
    }

    #[test]
    fn pheromone_field_array_independent_decay() {
        // Sprint 126: 3 fields s rozdílnými decay rates. Po jednom kroku step
        // má ch2 (decay 5.0) ztratit > ch1 (1.5) > ch0 (0.3) signálu.
        let world_half = [100.0_f32, 100.0, 50.0];
        let mut fields: [SmellField; N_PHEROMONE_CHANNELS] =
            std::array::from_fn(|_| SmellField::new([8, 8, 4], world_half));
        for f in fields.iter_mut() {
            f.add_source([0.0, 0.0, 0.0], 1.0);
        }
        let dt = 1.0 / FIXED_TIMESTEP_HZ;
        for ch in 0..N_PHEROMONE_CHANNELS {
            for _ in 0..30 {
                fields[ch].step(PHEROMONE_DIFFUSION_PER_CH[ch], PHEROMONE_DECAY_PER_CH[ch], dt);
            }
        }
        let signal_ch0 = fields[0].sample([0.0, 0.0, 0.0]);
        let signal_ch1 = fields[1].sample([0.0, 0.0, 0.0]);
        let signal_ch2 = fields[2].sample([0.0, 0.0, 0.0]);
        assert!(
            signal_ch0 > signal_ch1,
            "ch0 (slow decay) should retain více signal než ch1: ch0={signal_ch0} ch1={signal_ch1}"
        );
        assert!(
            signal_ch1 > signal_ch2,
            "ch1 should retain více než ch2 (rychlejší decay): ch1={signal_ch1} ch2={signal_ch2}"
        );
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
    fn brain_crossover_handles_mismatched_hidden_n() {
        // Sprint 104: structural mutace mohou rozejít hidden_n. Crossover
        // teď vezme menší size + per-row mix přes shared rozsah, místo paniky.
        let mut rng = StdRng::seed_from_u64(13);
        let a = Brain::random_with_hidden(&mut rng, 8);
        let b = Brain::random_with_hidden(&mut rng, 12);
        let c = Brain::crossover(&a, &b, &mut rng);
        assert_eq!(c.hidden_n, 8, "child takes smaller parent's hidden_n");
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
    #[ignore = "Sprint 106 HyperNEAT: brain.hidden_n je deterministicky BRAIN_HIDDEN_DEFAULT \
                z Brain::from_cppn — direct add_neuron mutace dead, brain re-derived z CPPN \
                na každý mutate() call. Topologie evoluuje teď přes CPPN structural mutations \
                (mutate_add_node v Cppn), test by se nastavoval jinak."]
    fn genome_mutate_with_rate_one_grows_brain_to_cap() {}

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
            spikes: {
                let mut s = [Spike::ZERO; SPIKE_SLOTS];
                s[0].length = 0.3;
                s
            },
            spike_count: 1,
            shell_thickness: 0.0,
        };
        let delta = phen.apply_morph([0.0, 0.0, 0.0, 0.0], MORPH_RATE, 0.5);
        assert_eq!(delta, 0.0);
        assert_eq!(phen.body_length, 1.5);
        assert_eq!(phen.body_width, 0.8);
        assert_eq!(phen.body_height, 1.0);
        assert_eq!(phen.spikes[0].length, 0.3);
    }

    #[test]
    fn morph_clamps_to_min_max_bounds() {
        let mut phen = Phenotype {
            body_length: MAX_BODY_LENGTH,
            body_width: MIN_BODY_WIDTH,
            body_height: MAX_BODY_HEIGHT,
            spikes: {
                let mut s = [Spike::ZERO; SPIKE_SLOTS];
                s[0].length = MAX_SPIKE_LENGTH;
                s
            },
            spike_count: 1,
            shell_thickness: 0.0,
        };
        // Strong positive signal on length, height & spike (already at max) → no change.
        // Strong negative signal on width (already at min) → no change.
        let delta = phen.apply_morph([1.0, -1.0, 1.0, 1.0], 100.0, 1.0);
        assert_eq!(delta, 0.0);
        assert_eq!(phen.body_length, MAX_BODY_LENGTH);
        assert_eq!(phen.body_width, MIN_BODY_WIDTH);
        assert_eq!(phen.body_height, MAX_BODY_HEIGHT);
        assert_eq!(phen.spikes[0].length, MAX_SPIKE_LENGTH);
    }

    #[test]
    fn morph_returns_total_absolute_delta() {
        let mut phen = Phenotype {
            body_length: 1.0,
            body_width: 1.0,
            body_height: 1.0,
            spikes: {
                let mut s = [Spike::ZERO; SPIKE_SLOTS];
                s[0].length = 0.5;
                s
            },
            spike_count: 1,
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
        assert!((phen.spikes[0].length - 1.3).abs() < 1e-5);
    }

    #[test]
    fn morph_signal_below_threshold_is_deadzoned() {
        let mut phen = Phenotype {
            body_length: 1.0,
            body_width: 1.0,
            body_height: 1.0,
            spikes: [Spike::ZERO; SPIKE_SLOTS],
            spike_count: 1,
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
        assert_eq!(phen.spikes[0].length, 0.0);
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
            thermal_optimum_penalty: 0.0,
        };
        let make_cell = |vel: [f32; 3]| {
            let mut c = base_cell();
            c.phenotype = Phenotype {
                body_length: 2.0,
                body_width: 1.0,
                body_height: 1.0,
                spikes: [Spike::ZERO; SPIKE_SLOTS],
                spike_count: 1,
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
            thermal_optimum_penalty: 0.0,
        };
        cell.step(1.0, [1000.0, 1000.0, 0.0], 0, 0, &physics);
        assert!((cell.velocity[0] - 9.0).abs() < 1e-4);
    }

    #[test]
    fn spike_bonus_only_when_target_in_front_cone() {
        let mut cell = base_cell();
        cell.position = [0.0, 0.0, 0.0];
        cell.heading = 0.0; // pointing +x
        cell.phenotype.spikes[0].length = 1.0;
        cell.phenotype.spike_count = 1;

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
        cell.phenotype.spikes[0].length = 0.0;
        cell.phenotype.spike_count = 1;
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
        cell.phenotype.spikes[0].length = 0.5;
        cell.phenotype.spike_count = 1;
        let physics = PhysicsConfig {
            drag: 0.0,
            angular_drag: 0.0,
            energy_cost_per_v_sq: 0.0,
            angular_energy_cost: 0.0,
            vision_cost_per_radius: 0.0,
            body_cost_factor: 0.0,
            thermal_optimum_penalty: 0.0,
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
        let inside = Food { position: [5.0, 0.0, 0.0], age_ticks: 0, kind: FoodKind::Plant };
        let outside = Food { position: [10.0, 0.0, 0.0], age_ticks: 0, kind: FoodKind::Plant };
        let lateral_inside = Food { position: [0.0, 5.0, 0.0], age_ticks: 0, kind: FoodKind::Plant };
        let vertical_inside = Food { position: [0.0, 0.0, 5.0], age_ticks: 0, kind: FoodKind::Plant };
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
            spikes: [Spike::ZERO; SPIKE_SLOTS],
            spike_count: 1,
            shell_thickness: 0.0,
        };
        // Forward at +14: inside ellipsoid (14/16 = 0.875).
        let forward_inside = Food { position: [14.0, 0.0, 0.0], age_ticks: 0, kind: FoodKind::Plant };
        // Lateral at +3.5: inside (3.5/4 = 0.875).
        let lateral_inside = Food { position: [0.0, 3.5, 0.0], age_ticks: 0, kind: FoodKind::Plant };
        // Forward at +17: outside (17/16 > 1).
        let forward_outside = Food { position: [17.0, 0.0, 0.0], age_ticks: 0, kind: FoodKind::Plant };
        // Lateral at +5: outside (5/4 > 1).
        let lateral_outside = Food { position: [0.0, 5.0, 0.0], age_ticks: 0, kind: FoodKind::Plant };
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
            spikes: [Spike::ZERO; SPIKE_SLOTS],
            spike_count: 1,
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
            thermal_optimum_penalty: 0.0,
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
            split_link_rate: 0.0,
            remove_neuron_rate: 0.0,
            sigma_vision_fov: 0.0,
            sigma_thermal_optimum: 0.0,
            sigma_carnivore_score: 0.0,
            sigma_sensor_gain: 0.0,
            spike_count_mutation_rate: 0.0,
            sigma_spike_orientation: 0.0,
            sigma_spike_complexity: 0.0,
            sigma_spike_length_secondary: 0.0,
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
            thermal_optimum_penalty: 0.0,
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
        let outputs = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
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
        let mut food = Food { position: [0.0, 0.0, 0.0], age_ticks: 0, kind: FoodKind::Plant };
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
        let mut fresh = Food { position: [0.0, 0.0, 0.0], age_ticks: 0, kind: FoodKind::Plant };
        assert!(fresh.age_step());
        // Past lifetime: age_step bump → value_factor = 0 → returns false.
        // F32 precision: použijeme age daleko za bod expirace, abychom se vyhli
        // ULP edge case (60.0/0.0005 jako u32 rounds k 119999, ne 120000).
        let mut expired = Food {
            position: [0.0, 0.0, 0.0],
            age_ticks: ((FIXED_TIMESTEP_HZ / CARRION_DECAY_PER_SEC) as u32) + 100,
            kind: FoodKind::Plant,
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
    fn hunter_genome_random_in_range() {
        let mut rng = StdRng::seed_from_u64(0xA0);
        for _ in 0..100 {
            let g = HunterGenome::random(&mut rng);
            assert!(g.vision_radius >= MIN_HUNTER_VISION_RADIUS);
            assert!(g.vision_radius <= MAX_HUNTER_VISION_RADIUS);
            assert!(g.vision_fov >= MIN_HUNTER_VISION_FOV);
            assert!(g.vision_fov <= MAX_HUNTER_VISION_FOV);
            assert!(g.max_speed >= MIN_HUNTER_MAX_SPEED);
            assert!(g.max_speed <= MAX_HUNTER_MAX_SPEED);
            assert!(g.acceleration >= MIN_HUNTER_ACC);
            assert!(g.acceleration <= MAX_HUNTER_ACC);
            assert!(g.attack_radius >= MIN_HUNTER_ATTACK_RADIUS);
            assert!(g.attack_radius <= MAX_HUNTER_ATTACK_RADIUS);
            assert!(g.damage_per_tick >= MIN_HUNTER_DAMAGE);
            assert!(g.damage_per_tick <= MAX_HUNTER_DAMAGE);
            assert!(g.body_size >= MIN_HUNTER_BODY_SIZE);
            assert!(g.body_size <= MAX_HUNTER_BODY_SIZE);
            assert!(g.color_hue >= 0.0 && g.color_hue < HUE_RANGE);
        }
    }

    #[test]
    fn hunter_mutate_clamps_to_range() {
        let mut rng = rand::rng();
        let g = HunterGenome::random(&mut rng);
        let cfg = HunterMutationConfig {
            sigma_vision_radius: 1000.0,
            sigma_vision_fov: 100.0,
            sigma_max_speed: 1000.0,
            sigma_acceleration: 1000.0,
            sigma_attack_radius: 100.0,
            sigma_damage: 100.0,
            sigma_body_size: 10.0,
            sigma_color_hue: 1000.0,
            sigma_brain: 100.0,
            adhesion_flip_rate: 0.0,
        };
        for _ in 0..500 {
            let m = g.mutate(&mut rng, &cfg);
            assert!(
                (MIN_HUNTER_VISION_RADIUS..=MAX_HUNTER_VISION_RADIUS).contains(&m.vision_radius)
            );
            assert!((MIN_HUNTER_VISION_FOV..=MAX_HUNTER_VISION_FOV).contains(&m.vision_fov));
            assert!((MIN_HUNTER_MAX_SPEED..=MAX_HUNTER_MAX_SPEED).contains(&m.max_speed));
            assert!((MIN_HUNTER_ACC..=MAX_HUNTER_ACC).contains(&m.acceleration));
            assert!(
                (MIN_HUNTER_ATTACK_RADIUS..=MAX_HUNTER_ATTACK_RADIUS).contains(&m.attack_radius)
            );
            assert!((MIN_HUNTER_DAMAGE..=MAX_HUNTER_DAMAGE).contains(&m.damage_per_tick));
            assert!((MIN_HUNTER_BODY_SIZE..=MAX_HUNTER_BODY_SIZE).contains(&m.body_size));
            assert!(m.color_hue >= 0.0 && m.color_hue < HUE_RANGE);
        }
    }

    #[test]
    fn hunter_crossover_picks_from_either_parent() {
        let mut rng = rand::rng();
        let a = HunterGenome {
            vision_radius: MIN_HUNTER_VISION_RADIUS,
            vision_fov: MIN_HUNTER_VISION_FOV,
            max_speed: MIN_HUNTER_MAX_SPEED,
            acceleration: MIN_HUNTER_ACC,
            attack_radius: MIN_HUNTER_ATTACK_RADIUS,
            damage_per_tick: MIN_HUNTER_DAMAGE,
            body_size: MIN_HUNTER_BODY_SIZE,
            color_hue: 10.0,
            adhesion_type: 0,
            brain: dummy_brain(),
        };
        let b = HunterGenome {
            vision_radius: MAX_HUNTER_VISION_RADIUS,
            vision_fov: MAX_HUNTER_VISION_FOV,
            max_speed: MAX_HUNTER_MAX_SPEED,
            acceleration: MAX_HUNTER_ACC,
            attack_radius: MAX_HUNTER_ATTACK_RADIUS,
            damage_per_tick: MAX_HUNTER_DAMAGE,
            body_size: MAX_HUNTER_BODY_SIZE,
            color_hue: 200.0,
            adhesion_type: 5,
            brain: dummy_brain(),
        };
        for _ in 0..100 {
            let c = HunterGenome::crossover(&a, &b, &mut rng);
            assert!(c.vision_radius == a.vision_radius || c.vision_radius == b.vision_radius);
            assert!(c.max_speed == a.max_speed || c.max_speed == b.max_speed);
            assert!(c.damage_per_tick == a.damage_per_tick || c.damage_per_tick == b.damage_per_tick);
            assert!(c.adhesion_type == a.adhesion_type || c.adhesion_type == b.adhesion_type);
        }
    }

    #[test]
    fn hunter_apply_energy_costs_drains() {
        // Static hunter — no motion drain, only vision + body + attack upkeep.
        let mut h = make_test_hunter([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let initial = h.energy;
        h.apply_energy_costs(1.0);
        assert!(h.energy < initial, "energy should drain in 1 sec, got {}", h.energy);
        // Moving hunter drains more (motion adds v² × MOTION_COST × dt).
        let mut moving = make_test_hunter([0.0, 0.0, 0.0], [100.0, 0.0, 0.0]);
        moving.apply_energy_costs(1.0);
        assert!(
            moving.energy < h.energy,
            "moving hunter should drain more: still={} moving={}",
            h.energy,
            moving.energy
        );
    }

    // ─── Sprint 105 CPPN tests ──────────────────────────────────────────

    #[test]
    fn cppn_random_has_correct_topology() {
        let mut rng = StdRng::seed_from_u64(7);
        let c = Cppn::random(&mut rng);
        assert_eq!(
            c.iter_nodes().filter(|n| n.layer == 0).count(),
            CPPN_INPUTS,
            "CPPN_INPUTS input nodes at layer 0"
        );
        assert_eq!(
            c.iter_nodes().filter(|n| n.layer == 2).count(),
            CPPN_OUTPUTS,
            "CPPN_OUTPUTS output nodes at layer 2"
        );
        assert_eq!(
            c.iter_nodes().filter(|n| n.layer == 1).count(),
            CPPN_INITIAL_HIDDEN,
            "CPPN_INITIAL_HIDDEN hidden nodes at layer 1"
        );
        let expected_links = CPPN_INPUTS * CPPN_INITIAL_HIDDEN
            + CPPN_INITIAL_HIDDEN * CPPN_OUTPUTS;
        assert_eq!(c.iter_links().count(), expected_links);
    }

    #[test]
    fn cppn_forward_deterministic() {
        let mut rng = StdRng::seed_from_u64(11);
        let c = Cppn::random(&mut rng);
        let inputs = [0.5, -0.3, 0.7, 0.1, -0.4, 0.0, 1.0];
        let out1 = c.forward(inputs);
        let out2 = c.forward(inputs);
        assert_eq!(out1, out2, "deterministic forward");
        for o in out1.iter() {
            assert!(o.is_finite() && (-1.0..=1.0).contains(o), "out {} oob", o);
        }
    }

    #[test]
    fn cppn_add_node_grows_topology() {
        let mut rng = StdRng::seed_from_u64(13);
        let mut c = Cppn::random(&mut rng);
        let n_pre = c.num_nodes;
        let l_pre = c.num_links;
        c.mutate_add_node(&mut rng);
        assert_eq!(c.num_nodes, n_pre + 1, "add_node adds 1 node");
        assert_eq!(c.num_links, l_pre + 2, "add_node adds 2 links");
    }

    #[test]
    fn cppn_add_link_no_cycle() {
        let mut rng = StdRng::seed_from_u64(17);
        let mut c = Cppn::random(&mut rng);
        for _ in 0..50 {
            c.mutate_add_link(&mut rng, 0.5);
        }
        for l in c.iter_links() {
            let from_layer = c
                .iter_nodes()
                .find(|n| n.id == l.from)
                .map(|n| n.layer)
                .unwrap();
            let to_layer = c
                .iter_nodes()
                .find(|n| n.id == l.to)
                .map(|n| n.layer)
                .unwrap();
            assert!(
                from_layer < to_layer,
                "no cycles allowed: from layer {} >= to layer {}",
                from_layer,
                to_layer
            );
        }
    }

    #[test]
    fn cppn_crossover_preserves_matching_innovations() {
        let mut rng = StdRng::seed_from_u64(19);
        let a = Cppn::random(&mut rng);
        let mut b = a;
        b.mutate_weight(&mut rng, 0.5);
        let c = Cppn::crossover(&a, &b, &mut rng);
        for la in a.iter_links() {
            assert!(
                c.iter_links().any(|lc| lc.innovation == la.innovation),
                "innovation {} preserved",
                la.innovation
            );
        }
    }

    #[test]
    fn cppn_compatibility_distance_self_zero() {
        let mut rng = StdRng::seed_from_u64(31);
        let c = Cppn::random(&mut rng);
        let d = Cppn::compatibility_distance(&c, &c);
        assert!(d < 1e-3, "self-distance ≈ 0, got {}", d);
    }

    #[test]
    fn cppn_compatibility_distance_grows_with_mutation() {
        let mut rng = StdRng::seed_from_u64(37);
        let a = Cppn::random(&mut rng);
        let mut b = a;
        // Heavy mutation pushuje distance výš
        for _ in 0..20 {
            b.mutate_weight(&mut rng, 1.0);
            b.mutate_add_node(&mut rng);
        }
        let d_self = Cppn::compatibility_distance(&a, &a);
        let d_far = Cppn::compatibility_distance(&a, &b);
        assert!(
            d_far > d_self + 0.05,
            "mutation grows distance: self={:.3}, mutated={:.3}",
            d_self,
            d_far
        );
    }

    #[test]
    fn cppn_mutate_drives_diversity() {
        let mut rng = StdRng::seed_from_u64(23);
        let initial = Cppn::random(&mut rng);
        let cfg = CppnMutationConfig {
            weight_rate: 1.0,
            sigma_weight: 0.5,
            add_node_rate: 1.0,
            add_link_rate: 1.0,
            toggle_link_rate: 0.0,
            activation_rate: 1.0,
        };
        let mutated = initial.mutate(&mut rng, &cfg);
        assert!(mutated.num_nodes > initial.num_nodes, "topology grew");
    }

    #[test]
    fn make_hunter_child_splits_energy() {
        let mut rng = StdRng::seed_from_u64(0xCAB);
        let half = [960.0, 540.0, 50.0];
        let mut parent = make_test_hunter([10.0, 20.0, 0.0], [0.0; 3]);
        parent.energy = 600.0;
        parent.lineage_id = 42;
        let child = make_hunter_child(&parent, &mut rng, half, 100, 5);
        // Child energy = half of parent.
        assert!((child.energy - 300.0).abs() < 1e-3, "child energy {}", child.energy);
        // Child lineage_id inherits.
        assert_eq!(child.lineage_id, 42);
        // Child birth_gen = current_gen.
        assert_eq!(child.lineage_birth_gen, 5);
        // Child cooldown set.
        assert_eq!(child.reproduce_cooldown_ticks, HUNTER_REPRODUCE_COOLDOWN_TICKS);
        // Child position = parent position.
        assert_eq!(child.position, parent.position);
    }

    #[test]
    fn make_hunter_mating_child_sums_parent_energies() {
        let mut rng = StdRng::seed_from_u64(0xBEEF);
        let half = [960.0, 540.0, 50.0];
        let mut a = make_test_hunter([10.0, 20.0, 5.0], [0.0; 3]);
        let mut b = make_test_hunter([30.0, 40.0, -5.0], [0.0; 3]);
        a.energy = 200.0;
        b.energy = 150.0;
        a.lineage_id = 7;
        b.lineage_id = 11;
        let child = make_hunter_mating_child(&a, &b, &mut rng, half, 99, 12);
        // Child energy = a.energy + b.energy (caller halves oba pre-call).
        assert!((child.energy - 350.0).abs() < 1e-3, "child energy {}", child.energy);
        // Lineage from parent_a (mirror cell semantics).
        assert_eq!(child.lineage_id, 7);
        assert_eq!(child.lineage_birth_gen, 12);
        // Cooldown applied.
        assert_eq!(child.reproduce_cooldown_ticks, HUNTER_REPRODUCE_COOLDOWN_TICKS);
        // Position = midpoint.
        assert!((child.position[0] - 20.0).abs() < 1e-3);
        assert!((child.position[1] - 30.0).abs() < 1e-3);
        assert!((child.position[2] - 0.0).abs() < 1e-3);
    }

    #[test]
    fn make_hunter_mating_child_genes_come_from_either_parent() {
        let half = [960.0, 540.0, 50.0];
        let a = HunterGenome {
            vision_radius: 100.0,
            vision_fov: 1.0,
            max_speed: 200.0,
            acceleration: 50.0,
            attack_radius: 10.0,
            damage_per_tick: 5.0,
            body_size: 0.8,
            color_hue: 0.0,
            adhesion_type: 0,
            brain: dummy_brain(),
        };
        let b = HunterGenome {
            vision_radius: 300.0,
            vision_fov: 2.0,
            max_speed: 400.0,
            acceleration: 90.0,
            attack_radius: 30.0,
            damage_per_tick: 12.0,
            body_size: 1.5,
            color_hue: 0.5,
            adhesion_type: 3,
            brain: dummy_brain(),
        };
        let mut h_a = make_test_hunter([0.0; 3], [0.0; 3]);
        let mut h_b = make_test_hunter([0.0; 3], [0.0; 3]);
        h_a.genome = a.clone();
        h_b.genome = b.clone();
        // Vary RNG seed; with crossover (50/50 per gene + zero-mutation cfg
        // applied below) child fields must each match jednoho z rodičů.
        for seed in 0u64..16 {
            let mut rng = StdRng::seed_from_u64(seed);
            let child = make_hunter_mating_child(&h_a, &h_b, &mut rng, half, seed, 0);
            let g = &child.genome;
            // Mutace narůstá hodnoty σ-měřítky; nemůžem testovat ==. Místo toho
            // ověřujeme, že každé pole je blíž k jednomu z rodičů než ke druhému
            // (po lehké mutaci mid-parent crossover by rozhodil pořadí).
            let near = |child_v: f32, av: f32, bv: f32| -> bool {
                (child_v - av).abs() < (av - bv).abs() * 0.5
                    || (child_v - bv).abs() < (av - bv).abs() * 0.5
            };
            assert!(near(g.vision_radius, a.vision_radius, b.vision_radius));
            assert!(near(g.max_speed, a.max_speed, b.max_speed));
            assert!(near(g.attack_radius, a.attack_radius, b.attack_radius));
            assert!(near(g.damage_per_tick, a.damage_per_tick, b.damage_per_tick));
        }
    }

    #[test]
    fn pair_fertile_hunters_respects_radius() {
        let r = HUNTER_MATING_RADIUS;
        let r2 = r * r;
        let half = [960.0, 540.0, 50.0];
        // 3 fertile hunters: a + b within radius (distance 50), c far (distance 500).
        let fertile: Vec<(usize, [f32; 3])> = vec![
            (0, [0.0, 0.0, 0.0]),
            (1, [50.0, 0.0, 0.0]),
            (2, [500.0, 0.0, 0.0]),
        ];
        let matings = pair_fertile(&fertile, r2, 10, half);
        // Single pair (0,1); 2 paired with no one in range.
        assert_eq!(matings.len(), 1);
        let (a, b) = matings[0];
        assert!(
            (a == 0 && b == 1) || (a == 1 && b == 0),
            "unexpected pair {:?}",
            matings[0]
        );
    }

    fn build_test_cell_grid(cells: &[Cell]) -> SpatialGrid<usize, ()> {
        let mut g: SpatialGrid<usize, ()> = SpatialGrid::new(GRID_CELL_SIZE);
        g.rebuild(cells.iter().enumerate().map(|(i, c)| (i, c.position, ())));
        g
    }

    /// Sprint 89: helper pro test hunter setup. Default genome (S71-S84
    /// const-equivalent middle ranges), full energy, no cooldown.
    /// Sprint 90: + dummy brain (zero weights), heading 0, pitch 0.
    fn make_test_hunter(pos: [f32; 3], vel: [f32; 3]) -> Hunter {
        let genome = HunterGenome {
            vision_radius: HUNTER_VISION_RADIUS,
            vision_fov: HUNTER_VISION_FOV,
            max_speed: HUNTER_MAX_SPEED,
            acceleration: HUNTER_ACC,
            attack_radius: HUNTER_ATTACK_RADIUS,
            damage_per_tick: HUNTER_DAMAGE_PER_TICK,
            body_size: 1.0,
            color_hue: 0.0,
            adhesion_type: 0,
            brain: dummy_brain(),
        };
        Hunter {
            position: pos,
            velocity: vel,
            hunter_id: 0,
            genome,
            energy: HUNTER_INITIAL_ENERGY,
            age: 0,
            reproduce_cooldown_ticks: 0,
            lineage_id: 0,
            lineage_birth_gen: 0,
            heading: 0.0,
            pitch: 0.0,
            angular_velocity: 0.0,
            pitch_velocity: 0.0,
            last_inputs: [0.0; BRAIN_INPUTS],
            last_hidden: [0.0; BRAIN_HIDDEN],
            last_outputs: [0.0; BRAIN_OUTPUTS],
            bonds: [None; MAX_BONDS_PER_CELL],
            pooled_hidden: [0.0; BRAIN_HIDDEN],
        }
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
        // Sprint 84: idle hunter (velocity 0) → cone filter disabled, omni.
        let h = make_test_hunter([0.0, 0.0, 0.0], [0.0; 3]);
        let grid = build_test_cell_grid(&cells);
        let pick = nearest_attackable_cell(&h, &cells, &grid, half);
        assert_eq!(pick, Some(1));
    }

    #[test]
    fn hunter_skips_immune_cluster_cells() {
        let mut rng = StdRng::seed_from_u64(12);
        let half = [960.0, 540.0, 50.0];
        let mut cells: Vec<Cell> = Vec::new();
        // Cell 0 nejbližší ale immune (Sprint 92: ≥ HUNTER_BOND_IMMUNITY_THRESHOLD = 4 bondy).
        let mut c0 = Cell::random(&mut rng, half, 0, 0, 0);
        c0.position = [10.0, 0.0, 0.0];
        for slot in 0..(HUNTER_BOND_IMMUNITY_THRESHOLD as usize) {
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
        let h = make_test_hunter([0.0, 0.0, 0.0], [0.0; 3]);
        let grid = build_test_cell_grid(&cells);
        let pick = nearest_attackable_cell(&h, &cells, &grid, half);
        assert_eq!(pick, Some(1));
    }

    #[test]
    fn hunter_returns_none_when_only_immune_in_range() {
        let mut rng = StdRng::seed_from_u64(13);
        let half = [960.0, 540.0, 50.0];
        let mut c = Cell::random(&mut rng, half, 0, 0, 0);
        c.position = [10.0, 0.0, 0.0];
        for slot in 0..(HUNTER_BOND_IMMUNITY_THRESHOLD as usize) {
            c.bonds[slot] = Some(Bond {
                other_cell_id: 100 + slot as u64,
                rest_length: 5.0,
                stiffness: BOND_STIFFNESS,
                damping: 0.6,
                age_ticks: 0,
            });
        }
        let cells = vec![c];
        let h = make_test_hunter([0.0, 0.0, 0.0], [0.0; 3]);
        let grid = build_test_cell_grid(&cells);
        assert!(nearest_attackable_cell(&h, &cells, &grid, half).is_none());
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
        let grid = build_test_cell_grid(&cells);
        // Hunter moving +X; target behind → cone reject.
        let active_h = make_test_hunter([0.0, 0.0, 0.0], [50.0, 0.0, 0.0]);
        assert!(nearest_attackable_cell(&active_h, &cells, &grid, half).is_none());
        // Idle hunter (velocity 0) → cone disabled → target nalezen.
        let idle_h = make_test_hunter([0.0, 0.0, 0.0], [0.0; 3]);
        assert!(nearest_attackable_cell(&idle_h, &cells, &grid, half).is_some());
    }

    #[test]
    fn hunter_cone_sees_front_target() {
        // Hunter pohybující se +X, target přímo vpředu — uvnitř cone.
        let mut rng = StdRng::seed_from_u64(85);
        let half = [960.0, 540.0, 50.0];
        let mut front = Cell::random(&mut rng, half, 0, 0, 0);
        front.position = [50.0, 0.0, 0.0];
        let cells = vec![front];
        let h = make_test_hunter([0.0, 0.0, 0.0], [50.0, 0.0, 0.0]);
        let grid = build_test_cell_grid(&cells);
        let pick = nearest_attackable_cell(&h, &cells, &grid, half);
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
        let h = make_test_hunter([0.0, 0.0, 0.0], [50.0, 0.0, 0.0]);
        let grid = build_test_cell_grid(&cells);
        let pick = nearest_attackable_cell(&h, &cells, &grid, half);
        assert!(pick.is_none());
    }

    #[test]
    fn hunter_step_integrates_velocity() {
        // Sprint 90: step je teď čistě integration. Set velocity manuálně
        // (replikuje brain motor output), step posune position.
        let half = [960.0, 540.0, 50.0];
        let mut h = make_test_hunter([0.0, 0.0, 0.0], [60.0, 0.0, 0.0]);
        h.step(1.0 / 60.0, half);
        assert!(
            (h.position[0] - 1.0).abs() < 0.05,
            "expected pos.x ≈ 1.0 po 1 ticku s v=60, got {}",
            h.position[0]
        );
    }

    #[test]
    fn hunter_apply_brain_motor_thrusts_forward() {
        // Sprint 90: positive thrust output → velocity gain podél forward
        // (heading=0, pitch=0 → forward = +X).
        let half = [1000.0, 1000.0, 50.0];
        let mut h = make_test_hunter([0.0, 0.0, 0.0], [0.0; 3]);
        let mut outputs = [0.0_f32; BRAIN_OUTPUTS];
        outputs[1] = 1.0; // full thrust
        h.apply_brain_motor(&outputs, None, 1.0 / 60.0, half);
        assert!(
            h.velocity[0] > 0.0,
            "expected +x velocity po thrust, got {:?}",
            h.velocity
        );
    }

    #[test]
    fn hunter_apply_brain_motor_turn_yaw_sets_angular() {
        // Sprint 90: brain turn output (no seek target) → angular_velocity ×
        // (1.0 - seek_mix) = brain × 0.4. Bez seek targetu seek_turn = 0.
        let half = [1000.0, 1000.0, 50.0];
        let mut h = make_test_hunter([0.0, 0.0, 0.0], [0.0; 3]);
        let mut outputs = [0.0_f32; BRAIN_OUTPUTS];
        outputs[0] = 1.0;
        h.apply_brain_motor(&outputs, None, 1.0 / 60.0, half);
        // turn = 1.0 × 0.4 + 0.0 × 0.6 = 0.4 → angular_velocity = 0.4 × HUNTER_TURN_RATE.
        let expected = 0.4 * HUNTER_TURN_RATE;
        assert!(
            (h.angular_velocity - expected).abs() < 1e-4,
            "expected angular_velocity ≈ {}, got {}",
            expected,
            h.angular_velocity
        );
    }

    #[test]
    fn hunter_apply_brain_motor_seek_dominates_chase() {
        // Sprint 90: hybrid mix — seek (60 %) + brain (40 %). Target přímo
        // vlevo (90° from forward = +X), neutral brain output → angular_velocity
        // toward +Y.
        let half = [1000.0, 1000.0, 50.0];
        let mut h = make_test_hunter([0.0, 0.0, 0.0], [0.0; 3]);
        let outputs = [0.0_f32; BRAIN_OUTPUTS]; // neutral brain
        let target = [0.0, 100.0, 0.0]; // +Y direction
        h.apply_brain_motor(&outputs, Some(target), 1.0 / 60.0, half);
        // Seek wants yaw toward +Y (= π/2). yaw_diff = π/2 - 0 = π/2 → seek_turn
        // = (π/2)/π = 0.5 → mixed turn = 0.0×0.4 + 0.5×0.6 = 0.3 → angular_velocity
        // = 0.3 × HUNTER_TURN_RATE.
        assert!(
            h.angular_velocity > 0.0,
            "expected positive angular_velocity (turning toward +Y), got {}",
            h.angular_velocity
        );
    }

    #[test]
    fn populate_hunter_brain_inputs_writes_prey_delta() {
        let mut h = make_test_hunter([0.0, 0.0, 0.0], [0.0; 3]);
        let sensors = HunterBrainSensors {
            nearest_prey: Some([100.0, 50.0, 10.0]),
            nearest_prey_size: 1.5,
            neighbors_in_vision: 3,
            smell_grad: [0.0; 3],
            nearest_pack_member: None,
            same_type_in_vision: 0,
        };
        let inputs = populate_hunter_brain_inputs(&mut h, &sensors);
        // vision_radius = HUNTER_VISION_RADIUS = 200; prey_dx/200 = 0.5.
        assert!((inputs[0] - 0.5).abs() < 1e-4);
        assert!((inputs[1] - 0.25).abs() < 1e-4);
        assert!((inputs[15] - 0.05).abs() < 1e-4);
        // density 3 / DENSITY_NORM_COUNT = 1.0 → tanh(1.0) ≈ 0.762
        assert!((inputs[13] - 1.0_f32.tanh()).abs() < 1e-4);
        // prey_size_relative = (1.5 - 1.0) / 1.0 = 0.5
        assert!((inputs[6] - 0.5).abs() < 1e-4);
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

    fn shock_cfg_active() -> ShockScheduleConfig {
        ShockScheduleConfig {
            mean_gens_between: 20,
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

    #[test]
    fn event_calendar_default_is_empty() {
        let cfg = ShockScheduleConfig::default();
        let cal = EventCalendar::generate(123, &cfg, 1000);
        assert!(cal.events.is_empty());
        assert_eq!(cal.seed, 123);
    }

    #[test]
    fn event_calendar_is_deterministic_for_seed() {
        let cfg = shock_cfg_active();
        let a = EventCalendar::generate(42, &cfg, 500);
        let b = EventCalendar::generate(42, &cfg, 500);
        assert_eq!(a.events.len(), b.events.len());
        assert!(!a.events.is_empty(), "active cfg must produce events");
        for (ea, eb) in a.events.iter().zip(b.events.iter()) {
            assert_eq!(ea.kind, eb.kind);
            assert_eq!(ea.start_gen, eb.start_gen);
            assert_eq!(ea.duration_gen, eb.duration_gen);
            assert_eq!(ea.ramp_gens, eb.ramp_gens);
            assert!((ea.intensity - eb.intensity).abs() < 1e-6);
            assert_eq!(ea.center_xy.is_some(), eb.center_xy.is_some());
            assert_eq!(ea.radius.is_some(), eb.radius.is_some());
        }
    }

    #[test]
    fn event_calendar_different_seeds_differ() {
        let cfg = shock_cfg_active();
        let a = EventCalendar::generate(42, &cfg, 1000);
        let b = EventCalendar::generate(43, &cfg, 1000);
        // Drobný risk shody, ale s mean=20 a 1000 gens je to >>50 eventů —
        // collision pravděpodobnost zanedbatelná.
        let identical = a.events.len() == b.events.len()
            && a.events
                .iter()
                .zip(b.events.iter())
                .all(|(x, y)| x.start_gen == y.start_gen && x.kind == y.kind);
        assert!(!identical, "different seeds should produce different schedules");
    }

    #[test]
    fn event_calendar_respects_max_gens() {
        let cfg = shock_cfg_active();
        let max_gens = 500;
        let cal = EventCalendar::generate(7, &cfg, max_gens);
        for e in &cal.events {
            assert!(e.start_gen < max_gens, "start_gen {} >= max {}", e.start_gen, max_gens);
        }
    }

    #[test]
    fn event_calendar_events_sorted() {
        let cfg = shock_cfg_active();
        let cal = EventCalendar::generate(11, &cfg, 1000);
        for w in cal.events.windows(2) {
            assert!(w[0].start_gen <= w[1].start_gen);
        }
    }

    #[test]
    fn shock_ramp_factor_trapezoid() {
        let trap = ShockEvent {
            kind: ShockKind::HazardPulse,
            start_gen: 100,
            duration_gen: 10,
            ramp_gens: 2,
            intensity: 1.0,
            center_xy: None,
            radius: None,
        };
        assert_eq!(shock_ramp_factor(&trap, 99), 0.0);
        assert_eq!(shock_ramp_factor(&trap, 110), 0.0);
        // Mid plateau (gen 104..=107) musí být 1.0.
        assert!((shock_ramp_factor(&trap, 105) - 1.0).abs() < 1e-6);
        // První gen rampy: monotonně rostoucí, < 1.
        let f0 = shock_ramp_factor(&trap, 100);
        let f1 = shock_ramp_factor(&trap, 101);
        assert!(f0 > 0.0 && f0 < 1.0);
        assert!(f1 > f0);
        // Poslední gen rampy: < 1, > 0.
        let f_end = shock_ramp_factor(&trap, 109);
        assert!(f_end > 0.0 && f_end < 1.0);

        // Triangle case: duration <= 2 * ramp.
        let tri = ShockEvent {
            kind: ShockKind::HazardPulse,
            start_gen: 0,
            duration_gen: 4,
            ramp_gens: 3,
            intensity: 1.0,
            center_xy: None,
            radius: None,
        };
        assert_eq!(shock_ramp_factor(&tri, 4), 0.0);
        let peaks: Vec<f32> = (0..4).map(|g| shock_ramp_factor(&tri, g)).collect();
        let max_peak = peaks.iter().cloned().fold(0.0_f32, f32::max);
        assert!(max_peak > 0.0 && max_peak <= 1.0);
        // Triangle musí mít jeden inner peak — okraje nižší než střed.
        assert!(peaks[0] < max_peak);
        assert!(peaks[3] < max_peak);
    }

    #[test]
    fn event_calendar_intensity_in_range() {
        let cfg = shock_cfg_active();
        let cal = EventCalendar::generate(99, &cfg, 1000);
        assert!(!cal.events.is_empty());
        for e in &cal.events {
            assert!(
                e.intensity >= cfg.intensity_min - 1e-6
                    && e.intensity <= cfg.intensity_max + 1e-6,
                "intensity {} out of range",
                e.intensity
            );
            assert!(e.duration_gen >= cfg.duration_min_gens);
            assert!(e.duration_gen <= cfg.duration_max_gens);
        }
    }

    #[test]
    fn event_calendar_global_vs_spatial_split() {
        let cfg = shock_cfg_active();
        let cal = EventCalendar::generate(2024, &cfg, 4000);
        assert!(
            cal.events.len() >= 20,
            "need enough events for split test, got {}",
            cal.events.len()
        );
        let global = cal.events.iter().filter(|e| e.center_xy.is_none()).count();
        let spatial = cal.events.iter().filter(|e| e.center_xy.is_some()).count();
        assert!(global > 0, "expected at least one global event");
        assert!(spatial > 0, "expected at least one spatial event");
        for e in cal.events.iter().filter(|e| e.radius.is_some()) {
            let r = e.radius.unwrap();
            let lo = cfg.spatial_radius_min_frac * WORLD_HALF[0];
            let hi = cfg.spatial_radius_max_frac * WORLD_HALF[0];
            assert!(r >= lo - 1e-3 && r <= hi + 1e-3, "radius {} out of range", r);
        }
    }

    #[test]
    fn hazard_multiplier_default_one() {
        let pos = [0.0, 0.0, 0.0];
        let m = hazard_shock_multiplier(pos, &[], 50, 0, WORLD_HALF);
        assert!((m - 1.0).abs() < 1e-6, "empty events must give 1.0, got {}", m);
    }

    #[test]
    fn hazard_multiplier_global_pulse_doubles_at_peak() {
        let event = ShockEvent {
            kind: ShockKind::HazardPulse,
            start_gen: 100,
            duration_gen: 10,
            ramp_gens: 2,
            intensity: 1.0,
            center_xy: None,
            radius: None,
        };
        let pos = [123.0, -45.0, 7.0];
        // Plateau (gen 102..=107) → ramp = 1.0, mask = 1.0 → 1 + 1 * 1 * 1 * 1 = 2.0.
        let m = hazard_shock_multiplier(pos, &[event], 105, 0, WORLD_HALF);
        assert!((m - 2.0).abs() < 1e-5, "global peak must give 2.0, got {}", m);
        // Pre-start: no contribution.
        let m_before = hazard_shock_multiplier(pos, &[event], 99, 0, WORLD_HALF);
        assert!((m_before - 1.0).abs() < 1e-6);
        // Post-end: no contribution.
        let m_after = hazard_shock_multiplier(pos, &[event], 110, 0, WORLD_HALF);
        assert!((m_after - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hazard_multiplier_spatial_mask_falls_off() {
        let center = [0.0, 0.0];
        let radius = 100.0;
        let event = ShockEvent {
            kind: ShockKind::HazardPulse,
            start_gen: 0,
            duration_gen: 10,
            ramp_gens: 2,
            intensity: 1.0,
            center_xy: Some(center),
            radius: Some(radius),
        };
        let gen = 5;
        // Plateau, ramp = 1.0.
        // Center → mask = 1.0 → multiplier = 2.0.
        let m_center = hazard_shock_multiplier([0.0, 0.0, 0.0], &[event], gen, 0, WORLD_HALF);
        assert!((m_center - 2.0).abs() < 1e-5, "center must be 2.0, got {}", m_center);
        // At edge (dist = radius) → mask = 0 → multiplier = 1.0.
        let m_edge = hazard_shock_multiplier([radius, 0.0, 0.0], &[event], gen, 0, WORLD_HALF);
        assert!((m_edge - 1.0).abs() < 1e-5, "edge must be 1.0, got {}", m_edge);
        // Beyond radius → mask = 0.
        let m_outside = hazard_shock_multiplier(
            [radius * 1.5, 0.0, 0.0],
            &[event],
            gen,
            0,
            WORLD_HALF,
        );
        assert!((m_outside - 1.0).abs() < 1e-5, "outside must be 1.0, got {}", m_outside);
        // Mid-radius → strictly between 1.0 and 2.0 (smoothstep monotone).
        let m_mid = hazard_shock_multiplier(
            [radius * 0.5, 0.0, 0.0],
            &[event],
            gen,
            0,
            WORLD_HALF,
        );
        assert!(
            m_mid > 1.0 && m_mid < 2.0,
            "mid must be in (1, 2), got {}",
            m_mid
        );
        // Smoothstep monotone: closer point → higher multiplier.
        let m_near = hazard_shock_multiplier(
            [radius * 0.25, 0.0, 0.0],
            &[event],
            gen,
            0,
            WORLD_HALF,
        );
        assert!(m_near > m_mid, "near {} should exceed mid {}", m_near, m_mid);
    }

    #[test]
    fn food_multiplier_default_one() {
        // Empty events → 1.0.
        let m = food_density_shock_multiplier(&[], 50);
        assert!((m - 1.0).abs() < 1e-6, "empty events must give 1.0, got {}", m);
        // Non-FoodCrash event (HazardPulse) → 1.0.
        let event = ShockEvent {
            kind: ShockKind::HazardPulse,
            start_gen: 0,
            duration_gen: 10,
            ramp_gens: 2,
            intensity: 1.0,
            center_xy: None,
            radius: None,
        };
        let m = food_density_shock_multiplier(&[event], 5);
        assert!((m - 1.0).abs() < 1e-6, "HazardPulse must not affect food, got {}", m);
    }

    #[test]
    fn food_multiplier_global_crash_drops() {
        // Sprint 113: 1 global FoodCrash, intensity = 1, peak ramp = 1
        // → mult = 1.0 - 1.0 × 1.0 × 0.5 = 0.5.
        let event = ShockEvent {
            kind: ShockKind::FoodCrash,
            start_gen: 100,
            duration_gen: 10,
            ramp_gens: 2,
            intensity: 1.0,
            center_xy: None,
            radius: None,
        };
        // Plateau (gen 102..=107) → ramp = 1.0.
        let m = food_density_shock_multiplier(&[event], 105);
        let expected = 1.0 - FOOD_CRASH_MAX_DROP;
        assert!(
            (m - expected).abs() < 1e-5,
            "global peak must give 1 - FOOD_CRASH_MAX_DROP = {}, got {}",
            expected,
            m
        );
        // Pre-start: 1.0.
        let m_before = food_density_shock_multiplier(&[event], 99);
        assert!((m_before - 1.0).abs() < 1e-6, "pre-start must be 1.0, got {}", m_before);
        // Post-end: 1.0.
        let m_after = food_density_shock_multiplier(&[event], 110);
        assert!((m_after - 1.0).abs() < 1e-6, "post-end must be 1.0, got {}", m_after);
    }

    #[test]
    fn food_multiplier_compound_clamped() {
        // Sprint 113: 3× FoodCrash s intensity=1, peak ramp současně:
        // 0.5 × 0.5 × 0.5 = 0.125 (> 0.1 floor → no clamp, return raw).
        let mk = |start: u64| ShockEvent {
            kind: ShockKind::FoodCrash,
            start_gen: start,
            duration_gen: 10,
            ramp_gens: 2,
            intensity: 1.0,
            center_xy: None,
            radius: None,
        };
        let three = [mk(100), mk(100), mk(100)];
        let m3 = food_density_shock_multiplier(&three, 105);
        assert!(
            (m3 - 0.125).abs() < 1e-5,
            "3 crashes compound to 0.125, got {}",
            m3
        );
        // 4× FoodCrash: 0.5^4 = 0.0625 < 0.1 floor → clamp to FOOD_CRASH_MIN_FACTOR.
        let four = [mk(100), mk(100), mk(100), mk(100)];
        let m4 = food_density_shock_multiplier(&four, 105);
        assert!(
            (m4 - FOOD_CRASH_MIN_FACTOR).abs() < 1e-5,
            "4 crashes must clamp to FOOD_CRASH_MIN_FACTOR = {}, got {}",
            FOOD_CRASH_MIN_FACTOR,
            m4
        );
    }
}
