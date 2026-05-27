//! Planet SPH thermal model — per-particle internal energy `u`,
//! viscous heating, conduction, radiation. Normalised units
//! (G = M = R = 1); `T = u / HEAT_CAPACITY_CV` so cv = 1 makes
//! temperature equal to internal energy. See `docs/sprints/`
//! S202–S210 for the staged build-up.

/// Specific heat capacity at constant volume (energy / mass / T).
/// `cv = 1` makes the temperature numerically equal to the internal
/// energy per unit mass. Real silicate rock: ~800 J/kg/K.
pub const HEAT_CAPACITY_CV: f32 = 1.0;

/// Default initial internal energy per unit mass. With `cv = 1` this
/// equals the seed temperature. 0.01 starts "cold" so viscous heating
/// during gravitational collapse is visible in the log.
pub const INITIAL_INTERNAL_ENERGY: f32 = 0.01;

/// Lower safety clamp on `u`. Numerical noise can otherwise produce
/// tiny negative values that break `sqrt(γ P / ρ)` and similar.
pub const U_MIN: f32 = 1e-6;

/// Upper safety clamp on `u`. Stops thermal runaway from blowing past
/// representable floats; well above any physically reasonable state
/// in normalised units so legitimate evolution isn't capped.
pub const U_MAX: f32 = 1.0e3;

/// Sprint 204 — thermal conductivity κ in the Cleary–Monaghan SPH
/// conduction term. Diffusion time scale τ_diff ≈ R² ρ c_v / κ ≈ 1/κ
/// in normalised units; κ = 0.1 ⇒ τ_diff ≈ 10 t_ff (slow conduction)
/// so the planet retains thermal gradients over many free-fall times.
pub const THERMAL_CONDUCTIVITY_KAPPA: f32 = 0.1;

/// Sprint 205 — surface detection threshold. A particle counts as
/// "surface" when its density drops below this fraction of the mean
/// initial density `ρ_mean`.
pub const SURFACE_DENSITY_FRAC: f32 = 0.5;

/// Sprint 205 — Stefan–Boltzmann coefficient σ in normalised units.
/// Real σ ≈ 5.67e-8 W·m⁻²·K⁻⁴; here rescaled so radiation produces
/// visible cooling over one `t_ff` at the seed temperatures used in
/// `INITIAL_INTERNAL_ENERGY`-based runs.
pub const STEFAN_BOLTZMANN_SIGMA: f32 = 1.0e-3;

/// Sprint 205 — surface emissivity ε ∈ [0, 1]. 1 = perfect blackbody;
/// rocky surfaces are typically ~0.9.
pub const RADIATION_EMISSIVITY: f32 = 0.9;

/// Sprint 205 — cosmic background temperature in normalised units.
/// With `cv = 1` and a seed `u ≈ 0.01`, T_space = 3e-3 keeps the
/// background well below the planet surface so net radiation > 0.
pub const SPACE_TEMPERATURE: f32 = 3.0e-3;

/// Sprint 205 — per-step safety fraction on the radiation term. T⁴
/// scaling means one anomalously hot particle could otherwise drain
/// its full `u` in a single substep; the integrator clamps the
/// radiative loss to this fraction of `u_i` per tick.
pub const RADIATION_MAX_FRAC: f32 = 0.1;

/// Convert internal energy per unit mass to temperature.
#[inline]
pub fn temperature_of(u: f32) -> f32 {
    u / HEAT_CAPACITY_CV
}

/// Convert temperature to internal energy per unit mass.
#[inline]
pub fn internal_energy_of(t: f32) -> f32 {
    t * HEAT_CAPACITY_CV
}
