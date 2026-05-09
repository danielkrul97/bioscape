use core::f32::consts::PI;

pub const HUE_RANGE: f32 = 360.0;
pub const MIN_SPEED: f32 = 1.0;
/// Sprint 73: horní cap pro `genome.max_speed`. Pre-Sprint-73 nebyl žádný cap,
/// jen MIN_SPEED floor — sigma_speed=3.0 mutation drift v Sprint 72 1000-gen
/// smoke produkoval mean spd 344 (12 % nad HUNTER_MAX_SPEED=300). Bez cap
/// je arms race degenerative — outrun je vždy levnější než cluster path,
/// takže multicelularita nikdy nepřijde. 200 = mírně pod Sprint 71 baseline
/// 218 → HUNTER_MAX_SPEED=300 reálně neutekatelný → cluster path se stává
/// jediná viable defense. Nominal cap, ne hard cieling — sigma_speed pořád
/// dovoluje fluktuace v rámci capu.
pub const MAX_SPEED: f32 = 200.0;
pub const MIN_VISION: f32 = 1.0;
/// Sprint 82: minimum half-angle směrového FOV (radiány). Pod ~17° je vidění
/// degenerované — buňka skoro nic neuvidí, nemá smysl evolvovat dál.
pub const MIN_VISION_FOV: f32 = PI / 12.0;
/// Sprint 82: maximum half-angle FOV = π = full sphere (4π str solid angle).
/// `vision_fov_factor(π) = 1.0` → energy cost identický s pre-Sprint-82
/// omnidirectional vision; výchozí hodnota při Genome::random.
pub const MAX_VISION_FOV: f32 = PI;
/// Sprint 82: initial vision_fov při Genome::random = full sphere baseline.
/// Cone filter v sensor gather je no-op pro fov ≥ π (cos π = −1, dot vždy ≥ −1).
/// Selekční tlak (cost ∝ fov_factor) může FOV zúžit, pokud info-loss < energy
/// savings; v Sprint 82 je `sigma_vision_fov = 0` → gen drift dormant.
pub const INITIAL_VISION_FOV: f32 = PI;
pub const MIN_TURN_RATE: f32 = 0.1;
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
pub const BRAIN_HIDDEN: usize = 45;
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
