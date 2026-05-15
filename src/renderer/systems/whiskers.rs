use bevy::prelude::*;
use bioscape::{whisker_directions, CELL_RADIUS, WHISKER_RANGE};

use super::super::components::{CellEntity, Dying, Pooled};

/// Renderer-side toggle for the whisker overlay. Default **off** — whiskers
/// are a maze-navigation debugging aid, not a primary view, and drawing them
/// for every cell is busy. `K` flips it.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub(crate) struct ShowWhiskers(pub(crate) bool);

pub(crate) fn toggle_whiskers(
    keys: Res<ButtonInput<KeyCode>>,
    mut show: ResMut<ShowWhiskers>,
) {
    if !keys.just_pressed(KeyCode::KeyK) {
        return;
    }
    show.0 = !show.0;
    info!("whisker overlay: {}", if show.0 { "on" } else { "off" });
}

/// Per-cell whisker overlay: one gizmo line per body-frame raycast direction,
/// length encoding free distance (`last_whisker_distances` ∈ [0,1] × range)
/// and color going green → red as a wall closes in. Directions mirror
/// `ObstacleField::whisker_distances` exactly — the ±z rays always read clear
/// (xy-only walls), so they show as constant full-length lines.
pub(crate) fn draw_whiskers(
    show: Res<ShowWhiskers>,
    cells: Query<&CellEntity, (Without<Dying>, Without<Pooled>)>,
    mut gizmos: Gizmos,
) {
    if !show.0 {
        return;
    }
    for cell in &cells {
        let cell = &cell.0;
        let pos = Vec3::new(cell.position[0], cell.position[1], cell.position[2]);
        let surface = CELL_RADIUS * cell.phenotype.effective_radius();
        for (k, dir) in whisker_directions(cell.heading, cell.pitch).iter().enumerate() {
            let dir = Vec3::new(dir[0], dir[1], dir[2]);
            let d = cell.last_whisker_distances[k].clamp(0.0, 1.0);
            let start = pos + dir * surface;
            let end = pos + dir * (surface + d * WHISKER_RANGE);
            // Slight HDR boost so the line catches bloom and stays readable
            // against the ocean-blue scene.
            let color = Color::linear_rgba((1.0 - d) * 1.5, d * 1.5, 0.15, 1.0);
            gizmos.line(start, end, color);
        }
    }
}
