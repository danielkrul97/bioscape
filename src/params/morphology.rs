use core::f32::consts::{FRAC_PI_2, PI};

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

// ─── Axis C: morphogenesis (S213) ────────────────────────────────────────────
// Positional information for emergent multicellular body-plans. A per-cell
// scalar "morphogen" diffuses over the bond graph (one Jacobi step per tick),
// each cell sourcing in proportion to its bond degree. Steady state grades from
// high (deeply interior cells, surrounded by bonded neighbours) to ~0
// (surface/tip/solo cells) — giving cells a sense of *where* they sit in the
// cluster, the precondition for spatially-organised differentiation.
//
// Stability needs `MORPHOGEN_DECAY > MORPHOGEN_DIFFUSION`; the uniform fixed
// point is `SOURCE·deg / (DECAY − DIFFUSION)`.
pub const MORPHOGEN_DECAY: f32 = 0.3;
pub const MORPHOGEN_DIFFUSION: f32 = 0.2;
pub const MORPHOGEN_SOURCE: f32 = 0.1;
pub const MORPHOGEN_MAX: f32 = 1.5;
/// Differentiation coupling: per-second rate at which a cluster cell's
/// `cell_state` is pulled toward its positional role. SURFACE foragers (low
/// morphogen) become altruists — they can reach food and share it inward to
/// feed the blocked interior; DEEP interior cells (high morphogen) stay
/// self-focused (they are fed). Without this division of labour the interior
/// of a dense organism starves. Gentle so the bistable feedback in
/// `update_cell_state` still amplifies. Only applied to bonded cells.
pub const MORPHOGEN_DIFFERENTIATION_K: f32 = 0.3;

/// Mutation sigma for the heritable `division_angle` gene (oriented division).
/// ~2 % of the 2π range per generation — slow drift so a lineage's body-build
/// rule is strategically stable.
pub const SIGMA_DIVISION_ANGLE: f32 = 0.13;
