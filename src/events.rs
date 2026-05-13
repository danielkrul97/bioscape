use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::*;

/// Salt mixed into the world seed when seeding the shock RNG. Keeps the
/// shock schedule on an independent stream so changing it doesn't shift
/// the cell-logic RNG sequence.
pub const SHOCK_SCHEDULE_SALT: u64 = 0xCAFE_F00D;

/// Must match the number of `ShockKind` variants. Adding a variant requires
/// bumping this and extending `ShockScheduleConfig.type_weights`.
pub const SHOCK_KIND_COUNT: usize = 3;

/// Discrete environmental shocks with a ramped envelope — not smooth cycles
/// — to keep selection pressure non-stationary across long runs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ShockKind {
    HazardPulse,
    ClimateShift,
    FoodCrash,
}

/// One scheduled shock. Active over `[start_gen, start_gen + duration_gen)`;
/// `ramp_gens` controls the rise/fall of the envelope. `None` for both
/// `center_xy` and `radius` marks a globally-applied event.
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

/// Parameters for the shock scheduler. `mean_gens_between == 0` is a no-op
/// (the default): the calendar stays empty and the per-tick lookups become
/// trivial empty-slice iterations.
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

/// Deterministically generated shock calendar for the whole run. Stores
/// the originating `seed` so reproducibility checks can verify it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventCalendar {
    pub events: Vec<ShockEvent>,
    pub seed: u64,
}

impl EventCalendar {
    /// Returns an empty calendar when `cfg.mean_gens_between == 0`.
    /// Otherwise deterministically samples shocks up to `max_gens` using
    /// Poisson-like inter-arrival times (mean = `mean_gens_between`) and
    /// `StdRng::seed_from_u64(seed ^ SHOCK_SCHEDULE_SALT)`. The returned
    /// `events` vector is sorted by `start_gen` ascending.
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

    /// Iterator of shocks active in the window `[start, start + duration)`.
    /// `tick` is ignored — the ramp works in gen units so it doesn't depend
    /// on `FIXED_TIMESTEP_HZ`. The parameter stays in the signature for
    /// future tick-level shocks.
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

/// Trapezoidal envelope (or triangle when `duration <= 2 × ramp_gens`).
/// Returns 0.0 outside `[start, start + duration)`, otherwise a value in
/// `[0, 1]`. The ramp lives in gen units, so `FIXED_TIMESTEP_HZ` doesn't
/// change its shape.
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

/// Peak `drain_factor` bonus from a single HazardPulse:
/// `1.0 + intensity × ramp × HAZARD_PULSE_MAX_MULTIPLIER_BONUS`. With
/// intensity=1 and ramp=1, drain doubles.
pub const HAZARD_PULSE_MAX_MULTIPLIER_BONUS: f32 = 1.0;

/// Peak temperature offset (°C) added by a single ClimateShift at full
/// intensity / ramp / mask. Sign is positive (warming); cooling would need
/// per-event signed intensity.
pub const CLIMATE_SHIFT_MAX_OFFSET: f32 = 5.0;

/// Per-cell hazard-drain multiplier at `pos` for the given `(generation, tick)`.
/// Returns 1.0 when no HazardPulse is active. Each active pulse contributes
/// `1.0 + intensity × ramp × spatial_mask × HAZARD_PULSE_MAX_MULTIPLIER_BONUS`,
/// compounded multiplicatively. Spatial mask is a smoothstep falloff in XY
/// (z is uniform), with toroidal wrap via `min_image_delta`.
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
                // V8: full 3D spherical falloff. The center sits on the world
                // floor (`z = 0`) so the strongest hit lands at the bottom and
                // surface cells feel only the outer skirt — matches the visual
                // wireframe sphere drawn by `draw_hazard_pulse_gizmos`.
                let center3 = [center[0], center[1], 0.0];
                let d_vec = min_image_delta(center3, pos, world_half);
                let dist = (d_vec[0] * d_vec[0]
                    + d_vec[1] * d_vec[1]
                    + d_vec[2] * d_vec[2])
                    .sqrt();
                if dist >= radius {
                    0.0
                } else {
                    let t = (1.0 - dist / radius).clamp(0.0, 1.0);
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

/// Signed temperature offset (°C) at `pos_xy` from active ClimateShift
/// events. Returns 0.0 when none is active. Each active event adds
/// `intensity × ramp × spatial_mask × CLIMATE_SHIFT_MAX_OFFSET` (additive).
/// Spatial mask is a smoothstep falloff over XY (toroidal); global events
/// without a center use mask = 1.0.
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

/// Peak fractional drop a single FoodCrash applies. With intensity=1 and
/// ramp=1, the density factor halves.
pub const FOOD_CRASH_MAX_DROP: f32 = 0.5;

/// Hard floor on the density factor — compound crashes can't drive food
/// to zero. 10 % of baseline still leaves room for an adapted population.
pub const FOOD_CRASH_MIN_FACTOR: f32 = 0.1;

/// Global food-density multiplier from active FoodCrash events. Returns
/// 1.0 when none is active; otherwise compounds
/// `multiplier *= 1.0 - intensity × ramp × FOOD_CRASH_MAX_DROP` across
/// active events and clamps to `FOOD_CRASH_MIN_FACTOR`. No spatial mask —
/// FoodCrash is a global per-tick scalar.
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

/// Shock-aware variant of `temperature_at_z` — adds the active ClimateShift
/// offset to the baseline gradient. With no active ClimateShift this is
/// bit-identical with `temperature_at_z`. Production callers (renderer +
/// headless) use this wrapper; tests still call the bare `temperature_at_z`.
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
