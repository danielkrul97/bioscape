use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::*;

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
