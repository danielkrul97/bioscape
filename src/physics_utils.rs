use core::f32::consts::TAU;

use crate::*;

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

/// Per-tick precomputed thermal terms shared across all cells. The diurnal
/// and seasonal phases depend only on `tick` and `generation`, so they're
/// computed once per dispatch rather than per cell. `temperature_at_z` builds
/// one of these on every call; hot loops should call `ThermalCtx::for_tick`
/// once and pass it to `temperature_at_z_with_ctx` instead.
#[derive(Debug, Clone, Copy)]
pub struct ThermalCtx {
    pub seasonal_offset: f32,
    /// `sin(TAU * diurnal_phase)` — kept un-scaled by `THERMAL_DIURNAL_AMP`
    /// so the per-cell evaluation order matches the original
    /// `temperature_at_z` formula bit-for-bit.
    pub diurnal_phase_sin: f32,
}

impl ThermalCtx {
    #[inline]
    pub fn for_tick(tick: u64, generation: u64) -> Self {
        let seasonal_phase = (generation % CYCLE_GEN_PERIOD) as f32 / CYCLE_GEN_PERIOD as f32;
        let seasonal_offset = THERMAL_SEASONAL_AMP * (TAU * seasonal_phase).sin();
        let diurnal_phase =
            (tick % THERMAL_DIURNAL_PERIOD_TICKS) as f32 / THERMAL_DIURNAL_PERIOD_TICKS as f32;
        let diurnal_phase_sin = (TAU * diurnal_phase).sin();
        Self {
            seasonal_offset,
            diurnal_phase_sin,
        }
    }
}

/// Linear z-gradient temperature: warm at the top (`+world_half[2]`), cold at
/// the bottom (`-world_half[2]`). When `world_half[2] == 0` (2D baseline) the
/// function returns `THERMAL_REF_TEMP` so `metabolism_factor` collapses to 1.
/// Layered on top: a uniform seasonal shift driven by `generation`, and a
/// surface-weighted diurnal oscillation driven by `tick`. With `tick = 0` and
/// `generation = 0` both sins are 0 and the result reduces to the pure
/// stratification.
#[inline]
pub fn temperature_at_z(z: f32, world_half: [f32; 3], tick: u64, generation: u64) -> f32 {
    temperature_at_z_with_ctx(z, world_half, &ThermalCtx::for_tick(tick, generation))
}

/// Per-cell core of `temperature_at_z`. Hot loops compute the `ThermalCtx`
/// once per tick and call this for each cell, saving two sin/modulo pairs
/// per cell.
#[inline]
pub fn temperature_at_z_with_ctx(z: f32, world_half: [f32; 3], ctx: &ThermalCtx) -> f32 {
    if world_half[2] <= 0.0 {
        return THERMAL_REF_TEMP;
    }
    let normalized = ((z / world_half[2]) + 1.0) * 0.5;
    let normalized = normalized.clamp(0.0, 1.0);
    let base = THERMAL_BOTTOM + (THERMAL_TOP - THERMAL_BOTTOM) * normalized;
    // Same evaluation order as the original `THERMAL_DIURNAL_AMP * normalized
    // * (TAU * diurnal_phase).sin()` so per-cell results stay bit-identical.
    let diurnal_offset = THERMAL_DIURNAL_AMP * normalized * ctx.diurnal_phase_sin;
    base + ctx.seasonal_offset + diurnal_offset
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
