//! Subtle gizmo rings around the selected cell (and a fainter pair around
//! the hovered cell). Two great circles on the XY and XZ planes give a
//! 3D-aware silhouette without requiring a custom shader or extra mesh.
//! Bond / vibration / state gizmos already share the gizmo pipeline; this
//! adds two more circles per frame, negligible cost.

use bevy::prelude::*;

use super::super::resources::SimWorld;
use super::{HoverCell, SelectedCell};

const SELECTED_COLOR: Color = Color::linear_rgba(0.30, 1.00, 0.95, 1.0);
const HOVER_COLOR: Color = Color::linear_rgba(1.00, 0.85, 0.35, 0.6);
/// How far outside the bounding sphere the outline sits. 1.25× max axis
/// keeps it clear of the cell body even when emissive bloom blurs the
/// rendered silhouette outward.
const OUTLINE_SCALE: f32 = 1.25;

pub(super) fn draw_outline(
    selected: Res<SelectedCell>,
    sim_world: Res<SimWorld>,
    mut gizmos: Gizmos,
) {
    let Some(id) = selected.cell_id else {
        return;
    };
    // Prefer live position; fall back to snapshot when deceased so the
    // outline freezes with the dialog rather than disappearing the moment
    // the cell dies.
    let (center, radius) = if let Some(idx) = sim_world.0.find_cell_idx_by_id(id) {
        let cell = &sim_world.0.cells[idx];
        (
            Vec3::new(cell.position[0], cell.position[1], cell.position[2]),
            cell.phenotype.max_axis(),
        )
    } else if let Some(cell) = selected.snapshot.as_ref() {
        (
            Vec3::new(cell.position[0], cell.position[1], cell.position[2]),
            cell.phenotype.max_axis(),
        )
    } else {
        return;
    };
    draw_double_ring(&mut gizmos, center, radius * OUTLINE_SCALE, SELECTED_COLOR);
}

pub(super) fn draw_hover_outline(
    selected: Res<SelectedCell>,
    hover: Res<HoverCell>,
    sim_world: Res<SimWorld>,
    mut gizmos: Gizmos,
) {
    let Some(id) = hover.cell_id else {
        return;
    };
    // Don't draw a hover ring on the cell that's already selected.
    if selected.cell_id == Some(id) {
        return;
    }
    let Some(idx) = sim_world.0.find_cell_idx_by_id(id) else {
        return;
    };
    let cell = &sim_world.0.cells[idx];
    let center = Vec3::new(cell.position[0], cell.position[1], cell.position[2]);
    let radius = cell.phenotype.max_axis() * OUTLINE_SCALE;
    draw_double_ring(&mut gizmos, center, radius, HOVER_COLOR);
}

fn draw_double_ring(gizmos: &mut Gizmos, center: Vec3, radius: f32, color: Color) {
    // XY circle — readable from a top-down camera.
    gizmos.circle(Isometry3d::from_translation(center), radius, color);
    // XZ circle — gives a 3D feel when the camera is tilted; `Quat::from_rotation_x(PI/2)`
    // rotates the default XY-plane circle into the XZ plane.
    gizmos.circle(
        Isometry3d::new(center, Quat::from_rotation_x(core::f32::consts::FRAC_PI_2)),
        radius,
        color,
    );
}
