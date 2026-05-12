pub const REPRODUCE_THRESHOLD: f32 = 150.0;
pub const SIZE_RATIO_THRESHOLD: f32 = 1.3;
/// Post-Hunter rebalance: cell-on-cell predace je teď jediný predační režim.
/// Drain ×2 + gain ×1.5 oproti pre-Hunter baseline → defense matters → bonded
/// clusters jsou viable strategy → arms race mezi predátorskými lineages a
/// defenzivními clustery emerguje napříč seedy (siege sweep, 6/7 seedů).
pub const PREDATION_DRAIN_PER_TICK: f32 = 6.0;
pub const PREDATION_GAIN_PER_TICK: f32 = 2.25;
/// Sprint 27: attacker.last_outputs[6].max(0) musí být > THRESHOLD aby se
/// predate-on-contact spustila. Bez aktivního brain signálu jsou `cell × cell`
/// kolize jen kolize — energy se nepřevádí. Mirroruje semantiku
/// `MATING_PHEROMONE_THRESHOLD` (input → behaviorální gate).
pub const ATTACK_THRESHOLD: f32 = 0.15;
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

/// Sprint 134: predator reward = `gain_this_tick × PREDATION_REWARD_SCALE`,
/// then clamped at `PREDATION_REWARD_MAX` before the global accumulator clamp.
/// Tuned so a typical multi-tick attack (per-tick gain ~2.25 across 3–5 ticks)
/// stays under saturation but is well above the eat-event magnitude (1.0) —
/// predation is a stronger learning signal because it's costlier to execute.
pub const PREDATION_REWARD_SCALE: f32 = 0.4;
pub const PREDATION_REWARD_MAX: f32 = 1.0;

/// Sprint 134: escape reward fires when a cell that was being damaged makes
/// it through a tick clean. `MAGNITUDE` smaller than predation/eat — escape
/// is a secondary signal, the primary motivator is avoiding death.
/// `STREAK_THRESHOLD = 30 ticks` (~0.5 s @ 60 Hz) — random one-tick contacts
/// don't qualify; the cell must have been under attack long enough that
/// surviving represents a learned behaviour.
/// `COOLDOWN_TICKS = 60` (~1 s) prevents reward farming via damage flicker
/// (in / out / in / out of an attack cone).
pub const ESCAPE_REWARD_MAGNITUDE: f32 = 0.3;
pub const ESCAPE_STREAK_THRESHOLD: u16 = 30;
pub const ESCAPE_COOLDOWN_TICKS: u16 = 60;

/// Sprint 135: damage = negative reinforcer. Reward magnitude = `-drain ×
/// DAMAGE_REWARD_GAIN`. With per-tick predation drain ~6 and gain `0.1`,
/// one attacker delivers ~`-0.6`/tick — under the global accumulator clamp
/// of `-2.0`, two simultaneous attackers saturate. Hazard drains (scale
/// ~0.01/tick) stay near noise floor. Asymmetric vs attacker reward
/// (`PREDATION_REWARD_MAX = 1.0`) so an attack is net-negative for the
/// victim even after the clamp.
pub const DAMAGE_REWARD_GAIN: f32 = 0.1;

/// Sprint 135: positive reward when a bond forms. Both partners credited.
/// Magnitude smaller than predation/eat — bond formation is a single
/// transient event, not a per-tick stream, so the credit needs to be
/// substantial enough that the eligibility trace catches it but not so
/// large it dominates foraging.
pub const BOND_FORMED_REWARD_MAGNITUDE: f32 = 0.6;

/// Sprint 135: positive reward for both parents when a mating produces a
/// child. Larger than bond formation — successful reproduction is the
/// principal evolutionary signal, and pre-S135 it had no per-lifetime
/// Hebbian credit at all.
pub const MATING_REWARD_MAGNITUDE: f32 = 0.5;

/// Sprint 136: per-cell evolvable learning rate range. Below the min a
/// lineage's Hebbian apply is a no-op; above the max the trace-amplified
/// `Δw` saturates the weight magnitudes within a few reward events. Default
/// `LEARNING_RATE = 0.005` sits in the middle so neutral drift goes either way.
pub const MIN_LEARNING_RATE: f32 = 0.0;
pub const MAX_LEARNING_RATE: f32 = 0.02;

/// Sprint 136: per-cell evolvable eligibility-trace decay range, in
/// `1 / second`. Lower decay = longer credit-assignment window (slower
/// forget), higher decay = sharper attribution. Default
/// `HEBBIAN_TRACE_DECAY_PER_SEC = 0.5` corresponds to a 2 s window.
pub const MIN_TRACE_DECAY_PER_SEC: f32 = 0.1;
pub const MAX_TRACE_DECAY_PER_SEC: f32 = 5.0;

/// Sprint 138: homeostatic synaptic scaling cap. Each `SCALING_PERIOD_TICKS`
/// we walk every cell's `w1` and `w2` row by row; rows whose L2 norm
/// exceeds `W_NORM_CAP` are scaled back to the cap. Pre-S138 Hebbian-only
/// runs let `||w1[i]||_2` drift past 20 on aggressive reward streaks,
/// saturating tanh and locking the brain in a single action. Cap chosen
/// well above gen-0 random init norm (≈ 2–3 with `sigma_brain = 0.2`) so
/// healthy weight growth isn't clipped — only runaway accumulation is.
pub const W_NORM_CAP: f32 = 8.0;

/// Sprint 138: 600 ticks ≈ 10 s @ 60 Hz. Hebbian's trace eligibility
/// window is ~2 s, so 10 s gives selection-driven weights a chance to
/// settle between scaling passes; smaller periods would clip Hebbian
/// gains before they're consolidated.
pub const SCALING_PERIOD_TICKS: u64 = 600;

/// Sprint 139: signed-EMA blend factor for per-neuron activity tracking.
/// `0.01` gives an effective window of ~100 ticks (~1.7 s @ 60 Hz) — long
/// enough to filter motor-output bursts but short enough that selection
/// can still drive activity-bias divergence within a generation.
pub const ACTIVITY_EMA_ALPHA: f32 = 0.01;

/// Sprint 139: linear excitability regulator gain. Per tick we shift each
/// hidden neuron's bias by `−DRIFT · activity_avg`, pushing the EMA toward
/// zero. With `activity_avg ∈ [−1, +1]` and `DRIFT = 0.001`, the
/// per-tick adjustment is bounded by `±0.001` and the time constant is
/// ~1000 ticks (~16 s). Slow enough that Hebbian learning consolidates
/// before homeostasis nulls it out.
pub const EXCITABILITY_DRIFT_PER_TICK: f32 = 0.001;

/// Sprint 148: STDP magnitude defaults. Standard asymmetric STDP has
/// LTP (`a_plus`, pre-before-post) slightly weaker than LTD (`a_minus`,
/// post-before-pre) so net effect is stabilising. Both bounded by genome
/// range `[0, MAX_STDP_A]` so STDP gain stays in a sane regime even after
/// long evolutionary drift.
pub const DEFAULT_STDP_A_PLUS: f32 = 0.001;
pub const DEFAULT_STDP_A_MINUS: f32 = 0.0012;
pub const MAX_STDP_A: f32 = 0.02;

/// Sprint 148: STDP temporal window, expressed in simulation ticks. Default
/// 5 ticks ≈ 83 ms @ 60 Hz, on the longer end of biological windows
/// (~20–100 ms reported in cortex). Range `[1, MAX_STDP_TAU]` keeps the
/// trace decay numerically stable.
pub const DEFAULT_STDP_TAU_TICKS: f32 = 5.0;
pub const MIN_STDP_TAU_TICKS: f32 = 1.0;
pub const MAX_STDP_TAU_TICKS: f32 = 30.0;

/// Sprint 153: threshold for continuous brain input → binary spike event.
/// An input channel emits a spike on a given tick when its normalised
/// value exceeds this threshold. Choice of 0.5 splits the [-1, +1] tanh
/// output mid-range — strong sensory signals fire, weak ones stay silent.
/// Future S163+ may swap this for a rate-coded Bernoulli encoder.
pub const SPIKE_ENCODE_THRESHOLD: f32 = 0.5;
