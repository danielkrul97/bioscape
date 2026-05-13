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
pub const VIBRATION_DECAY: f32 = 1.5;

/// Linear-motion contribution to emission. `K_LINEAR * speed / max_speed`
/// reaches ~1.0 for a cell cruising at its own maximum velocity.
pub const VIBRATION_K_LINEAR: f32 = 1.0;

/// Angular-motion contribution. Rotation stirs the medium independently of
/// linear translation; clamp + a smaller K keeps spinning-in-place cells
/// from dominating the field.
pub const VIBRATION_K_ANGULAR: f32 = 0.5;

/// Pre-tanh gain on brain-input slots [29..32] = grad_{x,y,z}, amp. With the
/// 1.5/s decay the typical steady-state amp lands around 0.05–0.10 in dense
/// scenes; `GAIN = 10` puts that into tanh's responsive zone (0.4–0.7) instead
/// of the noise floor. Much higher than `SMELL_NORMALIZATION_GAIN = 0.5` —
/// vibration is a sharper short-range cue.
pub const VIBRATION_NORMALIZATION_GAIN: f32 = 10.0;

/// Central-differences epsilon for `gradient_at`. Same scale as
/// `SMELL_SAMPLE_EPSILON` and `PHEROMONE_SAMPLE_EPSILON`.
pub const VIBRATION_SAMPLE_EPSILON: f32 = 10.0;

/// Brain input slot count owned by the vibration channel: `[grad_x, grad_y,
/// grad_z, amp]` → 4 slots in `BRAIN_INPUTS_SENSORY`.
pub const N_VIBRATION_INPUTS: usize = 4;
