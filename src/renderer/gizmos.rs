use bevy::prelude::*;
use bioscape::WORLD_HALF;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::components::{BondSnapshot, CellEntity, Dying};
use super::material::adhesion_hue;
use super::resources::{Clock, EventCalendarResource, VibrationResource};

/// Renderer-side toggle for the vibration overlay. Default **on** — vibration
/// is a primary feature of the simulation, so the overlay greets the user
/// at startup. `V` flips it off (e.g. to declutter the view during cluster
/// observation). Stored as a `Resource` so the keypress system can write
/// and the gizmo system can read in the same tick.
#[derive(Resource, Debug, Clone, Copy)]
pub(super) struct ShowVibration(pub(super) bool);

impl Default for ShowVibration {
    fn default() -> Self {
        Self(true)
    }
}

/// Sprint 69: render persistent spring bonds jako gizmo lines. Hue podle
/// `adhesion_type` (bondy se tvoří jen mezi same-type páry, takže obě cells
/// sdílí hue). Toroidal wrap-aware: skip line, pokud raw distance > poloviny
/// world (znamená že bond jde "přes okraj", straight line by visuálně lhala).
pub(super) fn draw_bond_gizmos(
    cells: Query<&CellEntity, Without<Dying>>,
    mut gizmos: Gizmos,
    mut id_to_pos: Local<FxHashMap<u64, Vec3>>,
    mut snapshot: Local<Vec<BondSnapshot>>,
    mut segments: Local<Vec<(Vec3, Vec3, Color)>>,
) {
    id_to_pos.clear();
    snapshot.clear();
    for cell in &cells {
        let start = Vec3::new(cell.0.position[0], cell.0.position[1], cell.0.position[2]);
        id_to_pos.insert(cell.0.cell_id, start);
        let hue = adhesion_hue(cell.0.genome.adhesion_type);
        // Sprint 85: saturation 0.85 → 1.0, match s body color v adhesion_material.
        // Sprint 88: linear color × 3.0 multiplier — Bevy gizmos render do HDR
        // backbufferu, super-bright hodnoty Bloom catches → bondy svítí jako
        // skutečné spring laser-lines.
        let base = Color::hsl(hue, 1.0, 0.6).to_linear();
        let color = Color::linear_rgba(base.red * 3.0, base.green * 3.0, base.blue * 3.0, 1.0);
        let mut partners = [None; bioscape::MAX_BONDS_PER_CELL];
        for (i, slot) in cell.0.bonds.iter().enumerate() {
            partners[i] = slot.as_ref().map(|b| b.other_cell_id);
        }
        snapshot.push(BondSnapshot {
            cell_id: cell.0.cell_id,
            start,
            color,
            partners,
        });
    }
    let half_x = WORLD_HALF[0];
    let half_y = WORLD_HALF[1];
    let id_to_pos_ref = &*id_to_pos;
    segments.clear();
    segments.par_extend(snapshot.par_iter().flat_map_iter(|s| {
        s.partners.iter().filter_map(move |partner| {
            let other_id = (*partner)?;
            if s.cell_id >= other_id {
                return None;
            }
            let end = *id_to_pos_ref.get(&other_id)?;
            let dx = (s.start.x - end.x).abs();
            let dy = (s.start.y - end.y).abs();
            if dx > half_x || dy > half_y {
                return None;
            }
            Some((s.start, end, s.color))
        })
    }));
    for (a, b, c) in segments.iter() {
        gizmos.line(*a, *b, *c);
    }
}

/// Per-cell ring whose radius and brightness encode the local vibration
/// field amplitude. Off by default; toggle with `V`. The ring lives in
/// the cell's xy plane so it reads as a "wave front" emanating from the
/// cell, even when zoomed out. We use a constant `VIZ_AMP_SCALE` because
/// raw field amplitudes tend to be small (≤ ~0.1 in dense scenes) — the
/// scale gain makes the signal visible without saturating.
pub(super) fn draw_vibration_gizmos(
    show: Res<ShowVibration>,
    vibration: Res<VibrationResource>,
    cells: Query<&CellEntity, Without<Dying>>,
    mut gizmos: Gizmos,
) {
    if !show.0 {
        return;
    }
    const VIZ_AMP_SCALE: f32 = 80.0;
    const RING_MIN_R: f32 = 1.0;
    const RING_MAX_R: f32 = 25.0;
    for cell in &cells {
        let pos = Vec3::new(
            cell.0.position[0],
            cell.0.position[1],
            cell.0.position[2],
        );
        let amp = vibration
            .0
            .sample([pos.x, pos.y, pos.z])
            .max(0.0)
            .min(1.0);
        let r = (amp * VIZ_AMP_SCALE).clamp(RING_MIN_R, RING_MAX_R);
        // Teal — distinct from bond hue (variable) and cell_state line
        // (blue↔red). Brightness scales with amplitude so faint background
        // rings don't drown the loud spots.
        let intensity = (amp * VIZ_AMP_SCALE).clamp(0.2, 2.0);
        let color = Color::linear_rgba(0.2 * intensity, 0.9 * intensity, 1.0 * intensity, 1.0);
        gizmos.circle(
            Isometry3d::from_translation(pos),
            r,
            color,
        );
    }
}

/// Per active local HazardPulse, draw a 3D wireframe sphere (outline) plus
/// phase-offset expanding shockwave spheres. Anchored on the world floor
/// (`z = 0`), matching `bioscape::hazard_shock_multiplier`'s spherical
/// falloff. Globální pulses (`center_xy = None`) se nekreslí — bez středu
/// nemají smysl jako lokalizovaný overlay. Fáze shockwave je vázaná na
/// `Time<Virtual>`, takže animace mrzne při pauze a škáluje se speed
/// multiplierem — konzistentní s pohybem cells.
pub(super) fn draw_hazard_pulse_gizmos(
    calendar: Res<EventCalendarResource>,
    clock: Res<Clock>,
    time: Res<Time<Virtual>>,
    mut gizmos: Gizmos,
) {
    const PERIOD_SEC: f32 = 1.4;
    const SHOCKWAVE_COUNT: usize = 2;
    let gen = clock.0.generation;
    let t = time.elapsed_secs();

    for event in calendar.0.active(gen, 0) {
        if event.kind != bioscape::ShockKind::HazardPulse {
            continue;
        }
        let (Some([cx, cy]), Some(radius)) = (event.center_xy, event.radius) else {
            continue;
        };
        if radius <= 0.0 {
            continue;
        }
        let env = event.intensity * bioscape::shock_ramp_factor(event, gen);
        if env <= 0.0 {
            continue;
        }

        let iso = Isometry3d::from_translation(Vec3::new(cx, cy, 0.0));
        let throb = 0.85 + 0.15 * (t / PERIOD_SEC * std::f32::consts::TAU).sin();
        let go = 2.5 * env * throb;
        // Static outer sphere — three great circles via `gizmos.sphere`. Reads
        // as a volumetric danger zone instead of a flat ring on the floor.
        gizmos.sphere(
            iso,
            radius,
            Color::linear_rgba(go, 0.25 * go, 0.05 * go, 1.0),
        );

        for k in 0..SHOCKWAVE_COUNT {
            let off = k as f32 / SHOCKWAVE_COUNT as f32;
            let phase = ((t / PERIOD_SEC) + off).fract();
            let r = (phase * radius).max(0.01);
            let alpha = (1.0 - phase) * env;
            let gw = 1.5 * alpha;
            gizmos.sphere(
                iso,
                r,
                Color::linear_rgba(gw, 0.35 * gw, 0.10 * gw, 1.0),
            );
        }
    }
}

/// Sprint 80: vertical marker per cell colored by `cell_state`. Modrá =
/// selfish (state≈0), červená = altruist (state≈1). Per-cell StandardMaterial
/// rebind by byl drahý (každý tick allocate handle), gizmo line je free.
pub(super) fn draw_cell_state_gizmos(
    cells: Query<&CellEntity, Without<Dying>>,
    mut gizmos: Gizmos,
) {
    for cell in &cells {
        let s = cell.0.cell_state.clamp(0.0, 1.0);
        let pos = Vec3::new(cell.0.position[0], cell.0.position[1], cell.0.position[2]);
        // Marker výška: 1.5× max body axis nad cell, viditelné nezávisle
        // na velikosti těla.
        let h = cell.0.phenotype.max_axis() * 1.5 + 1.0;
        let top = pos + Vec3::new(0.0, 0.0, h);
        // Lerp blue → red v sRGB. Mezistav kolem 0.5 = magenta = vidíme,
        // jak cells přecházejí přes attractor boundary.
        let color = Color::srgb(s, 0.05, 1.0 - s);
        gizmos.line(pos, top, color);
    }
}
