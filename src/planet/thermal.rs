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

/// Sprint 223 — melt onset (solidus). With `cv = 1` the solidus internal
/// energy equals `T_m`: a particle melts once collapse heating drives `u`
/// past this. Above the cold-start `INITIAL_INTERNAL_ENERGY = 0.01` so a
/// cold body must heat before it melts; below `U_MAX`.
pub const MELT_TEMPERATURE_T_M: f32 = 0.30;

/// Sprint 223 — specific latent heat of fusion. Liquidus `u_liq = T_m + L`.
/// The dimensionless Stefan number `L / (cv · T_m) = 0.5` is the physically
/// meaningful group; tune that ratio rather than `L` alone.
pub const LATENT_HEAT_FUSION_L: f32 = 0.15;

/// Phase state derived from internal energy via the enthalpy split.
/// `t` is the sensible temperature (flat across the melt band — the
/// latent plateau); `phi` is the solid fraction in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseState {
    pub t: f32,
    pub phi: f32,
}

/// Map internal energy per unit mass to `(sensible temperature, solid
/// fraction)`. `u` is the conserved quantity and *carries* latent heat;
/// `T` plateaus at `T_m` while `u` climbs through `[u_sol, u_liq]` (the
/// energy goes into melting, raising `phi`, not temperature). Single
/// source of truth — the WGSL mirror in `shaders/planet_phase_common.wgsl`
/// must stay identical (checked by `gpu_phase_map_matches_cpu`).
#[inline]
pub fn phase_of(u: f32) -> PhaseState {
    let u_sol = MELT_TEMPERATURE_T_M;
    let u_liq = u_sol + LATENT_HEAT_FUSION_L;
    if u <= u_sol {
        PhaseState { t: u, phi: 1.0 }
    } else if u < u_liq {
        PhaseState {
            t: MELT_TEMPERATURE_T_M,
            phi: 1.0 - (u - u_sol) / LATENT_HEAT_FUSION_L,
        }
    } else {
        PhaseState {
            t: MELT_TEMPERATURE_T_M + (u - u_liq),
            phi: 0.0,
        }
    }
}

/// Sensible temperature for thermal terms (conduction flux, radiation
/// `T⁴`, diagnostics). Goes through the phase map so the latent plateau
/// is respected — NOT the same as the raw `u / cv` used pre-S223.
#[inline]
pub fn sensible_temperature_of(u: f32) -> f32 {
    phase_of(u).t
}

/// Sprint 224 — vaporisation energy. Above this the ideal-gas EoS governs
/// (hot core / atmosphere); below it the condensed Tait/Murnaghan EoS
/// gives cold matter real bulk stiffness. Well above the liquidus.
pub const VAPORIZATION_ENERGY_U_VAP: f32 = 5.0;

/// Sprint 224 — condensed-phase reference sound speed `c0`; bulk modulus
/// `K0 = ρ0 · c0²` with `ρ0 = rho_mean_init`. Bounded by the elastic CFL.
pub const TAIT_REF_SOUND_SPEED_C0: f32 = 1.0;

/// Sprint 224 — Tait/Murnaghan stiffening exponent. Rock/water cold curves
/// use ~7, but the condensed sound speed scales as `(ρ/ρ0)^((n−1)/2)`, so
/// at fixed `dt` a high `n` blows the elastic CFL the moment the body
/// compresses. `n = 3` keeps the global CFL satisfied up to ~4·ρ0 in the
/// soft (path A) regime; stiffer bodies (`n→7`) need stress sub-cycling.
pub const TAIT_EXPONENT_N: f32 = 3.0;

/// Sprint 225 — solid shear modulus at full solidity. The deviatoric
/// stress evolves as `dS/dt = 2G·dev(ε̇) + spin`. Bounded jointly with the
/// bulk `K0` by the elastic CFL `c_el = √((K0+4G/3)/ρ)`; with `c0 = 1`,
/// `G0 = 1` keeps the soft (path A) body inside `dt = 1e-3` up to ~4·ρ0.
/// Phase-gated as `G_i = G0·φ²` from S227 (constant `G0` in S225).
pub const SHEAR_MODULUS_G0: f32 = 1.0;

/// Sprint 229 — per-step cap on plastic-work heating, as a fraction of the
/// particle's `u`. Plastic dissipation `≈ J2_trial·(1−f²)/(2Gρ)` releases
/// the elastic energy shed by the von Mises return into heat; the cap keeps
/// the explicit Euler stable if a particle yields hard in one step.
pub const PLASTIC_HEAT_MAX_FRAC: f32 = 0.1;

/// Sprint 227 — von Mises yield strength at full solidity; `Y_i = Y0·φ²`.
/// Sets the elastic→plastic transition (the block dents/cracks above it).
/// `Y0/G0 = 0.5` is the (intentionally high, stable-demo) yield strain;
/// lower it toward ~1e-2 as stiffness rises. `Y → 0` at `φ = 0` is what
/// drives `S → 0` on remelt, so no separate relaxation constant is needed.
pub const YIELD_STRENGTH_Y0: f32 = 0.5;

/// Sprint 225 — fixed Tikhonov regularisation added to the Bonet–Lok
/// moment matrix (`M + λI`) before inversion. Applied unconditionally
/// (not a branch on `det ≈ 0`) so surface/rank-deficient particles get a
/// finite, deterministic, cross-platform-identical correction matrix.
pub const GRAD_CORRECTION_LAMBDA: f32 = 1.0e-4;

/// Sprint 228 — maximum sustainable tension (negative pressure) for fully
/// solid condensed matter (φ = 1). The EoS clamps the cohesive floor to
/// `P ≥ −P_tens·(MELT_COHESION_FRAC + (1−MELT_COHESION_FRAC)·φ)`, so a solid
/// particle cohesively resists being pulled apart up to `P_tens`. This is the
/// continuum cohesion that lets cold clumps fuse into a block.
pub const TENSILE_STRENGTH_P_TENS: f32 = 0.5;

/// Melt cohesion: the fraction of `P_tens` that survives into the fully
/// molten phase (φ = 0). A liquid is condensed matter, not a gas, so it stays
/// cohesive when it melts — the melt holds together (surface-tension-like) and
/// fuses contacting molten regions instead of cavitating at `P = 0`. The EoS
/// interpolates the cohesive tension floor linearly in φ from `P_tens` (solid)
/// down to `MELT_COHESION_FRAC·P_tens` (liquid); only the gas branch
/// (`u ≥ u_vap`) is fully cohesionless. `0.0` recovers the pre-fusion
/// solid-only cohesion (`−P_tens·φ`). Passed as the `melt_coh_frac` uniform to
/// the EoS and artificial-stress passes.
pub const MELT_COHESION_FRAC: f32 = 0.5;

/// Sprint 228 — Monaghan (2000) artificial-stress coefficient ε. Cures the
/// tensile (pairing) instability that negative pressure would otherwise
/// trigger, and supplies the short-range cohesion. Literature value ~0.3.
pub const ARTIFICIAL_STRESS_EPSILON: f32 = 0.3;

/// Sprint 228 — exponent `m` in the artificial-stress kernel ratio
/// `(W(r)/W(Δp))^m`. Standard Monaghan-2000 value.
pub const ARTIFICIAL_STRESS_EXPONENT_M: f32 = 4.0;

/// SPH smoothing ratio `η` in `h = η·(m/ρ)^(1/3)`; the reference particle
/// spacing for the artificial-stress kernel ratio is `Δp = h/η`. Mirrors
/// `init::SPH_SMOOTHING_ETA`.
pub const SPH_SMOOTHING_ETA: f32 = 1.3;

/// Phase-selected isotropic (bulk) sound speed. Gas branch: ideal-gas
/// `√(γ(γ−1)u)`; condensed branch: Tait `c0·√n·(ρ/ρ0)^((n−1)/2)`.
/// Mirror of the WGSL EoS in `planet_sph_force.wgsl`.
#[inline]
pub fn sound_speed_of(u: f32, rho: f32, eos_gamma: f32, rho0: f32) -> f32 {
    let u = u.max(U_MIN);
    if u >= VAPORIZATION_ENERGY_U_VAP {
        let gm1 = eos_gamma - 1.0;
        (eos_gamma * gm1 * u).max(0.0).sqrt()
    } else {
        let ratio = (rho / rho0.max(1e-30)).max(1e-6);
        TAIT_REF_SOUND_SPEED_C0 * TAIT_EXPONENT_N.sqrt() * ratio.powf(0.5 * (TAIT_EXPONENT_N - 1.0))
    }
}

/// Longitudinal elastic-wave speed `c_el = √(c_bulk² + 4G/(3ρ))` — the
/// signal speed that actually constrains the explicit timestep once the
/// deviatoric stress couples into momentum (S226). Uses the worst-case
/// `G0` (full solidity) so the CFL bound is conservative. Gas branch has
/// no shear, so it reduces to the bulk sound speed.
#[inline]
pub fn elastic_sound_speed_of(u: f32, rho: f32, eos_gamma: f32, rho0: f32) -> f32 {
    let c_bulk = sound_speed_of(u, rho, eos_gamma, rho0);
    if u.max(U_MIN) >= VAPORIZATION_ENERGY_U_VAP {
        c_bulk
    } else {
        let shear = 4.0 * SHEAR_MODULUS_G0 / (3.0 * rho.max(1e-30));
        (c_bulk * c_bulk + shear).sqrt()
    }
}

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
