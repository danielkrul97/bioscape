//! Vibration field — mechanosensory layer. Each cell emits an amplitude
//! proportional to its own motion (linear + angular). The field propagates
//! via the same 7-point Jacobi stencil + multiplicative decay as `SmellField`;
//! diffusion is parabolic (not the hyperbolic wave equation), which is a
//! deliberate approximation: cheap, numerically stable, and still preserves
//! the two signals brains need — spatial range limited by decay, and a
//! gradient that points back toward the loudest neighbor.
//!
//! Emission is NOT a brain output. It is a byproduct of motion, so evolution
//! has to repurpose motor patterns to carry information through this channel
//! — that is the hypothesis under test.

/// 3D grid resolution. Matches `PHEROMONE_GRID_RES` / `SMELL_GRID_RES` so
/// `gradient_at` epsilon and addressing semantics line up with other fields.
pub const VIBRATION_GRID_RES: usize = 64;
pub const VIBRATION_GRID_RES_Z: usize = 16;

/// Diffusion coefficient. Must stay `< 1/6` for the 3D 7-point Jacobi
/// stencil to be numerically stable; values at or above the cap cause
/// exponential growth on each tick. Same as `SMELL_DIFFUSION=0.15`.
pub const VIBRATION_DIFFUSION: f32 = 0.15;

/// Decay rate (1/s). Higher than `SMELL_DECAY=0.3` — mechanical waves
/// dissipate quickly, but the original 4.0 was so aggressive that steady-state
/// amplitude landed around 0.02 (sub-noise once tanh-normalized). 1.5 keeps
/// vibrations clearly "of the moment" while letting the brain register them.
pub const VIBRATION_DECAY: f32 = 0.7;

/// Linear-motion contribution to emission. `K_LINEAR * speed / max_speed`
/// reaches ~1.0 for a cell cruising at its own maximum velocity.
pub const VIBRATION_K_LINEAR: f32 = 1.0;

/// Angular-motion contribution. Rotation stirs the medium independently of
/// linear translation; clamp + a smaller K keeps spinning-in-place cells
/// from dominating the field.
pub const VIBRATION_K_ANGULAR: f32 = 0.5;

/// Pre-tanh gain on the amplitude brain-input slot [32]. With the 1.5/s decay
/// the typical steady-state amp lands around 0.05–0.10 in dense scenes;
/// `GAIN = 10` puts that into tanh's responsive zone (0.4–0.7) instead of the
/// noise floor. Much higher than `SMELL_NORMALIZATION_GAIN = 0.5` — vibration
/// is a sharper short-range cue.
pub const VIBRATION_AMP_GAIN: f32 = 10.0;

/// Pre-tanh gain on gradient brain-input slots [29..31] = grad_{x,y,z}. The
/// gradient is ~100× smaller than the amp because diffusion (0.15) smooths
/// the field across the sample-epsilon (10 world units) — measured grad
/// magnitudes land around 0.001 in healthy runs. With a 1000 gain, that maps
/// to tanh(1.0) ≈ 0.76, in tanh's responsive zone. Pre-split (when the same
/// gain=10 fed both amp and grad) the gradient slots sat at tanh(0.01)=0.01,
/// effective noise floor → brains had no directional info, only "how loud."
pub const VIBRATION_GRAD_GAIN: f32 = 1000.0;

/// Central-differences epsilon for `gradient_at`. Same scale as
/// `SMELL_SAMPLE_EPSILON` and `PHEROMONE_SAMPLE_EPSILON`.
pub const VIBRATION_SAMPLE_EPSILON: f32 = 10.0;

/// Brain input slot count owned by the vibration channel: `[grad_x, grad_y,
/// grad_z, amp]` → 4 slots in `BRAIN_INPUTS_SENSORY`.
pub const N_VIBRATION_INPUTS: usize = 4;

/// Brain output index controlling active (brain-driven) vibration emission.
/// `last_outputs[VIBRATION_EMIT_OUTPUT]` is rectified (max 0) and scaled by
/// `MAX_ACTIVE_EMIT` before being added on top of passive motion emission.
/// See `vibration_emit_for_cell` for the formula.
pub const VIBRATION_EMIT_OUTPUT: usize = 14;

/// Ceiling on the active emit component. Matches the passive emit max
/// (`K_LINEAR + 2·K_ANGULAR = 2.0`) in order of magnitude so neither term
/// can dominate the field on its own — selection has to combine them.
pub const MAX_ACTIVE_EMIT: f32 = 1.5;

/// Per-tick energy drain for brain-controlled emission. Drain at full output
/// is `MAX_ACTIVE_EMIT × VIBRATION_EMIT_COST = 0.075`/s — comparable with
/// `SENSOR_GAIN_COST × 1.0 = 0.1`/s but intentionally lower, so selection
/// can carry the channel through the cold-start before listening evolves.
pub const VIBRATION_EMIT_COST: f32 = 0.05;
