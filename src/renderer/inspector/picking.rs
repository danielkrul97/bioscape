//! LMB ray-from-cursor → closest-cell hit test. Uses a bounding-sphere
//! approximation around the rendered ellipsoid (max body axis × slack)
//! so picking stays well-defined under non-uniform scale without solving
//! the ray-ellipsoid quadratic. Selection clears on Escape or a click
//! that hits no cell. Hover preview runs the same query every frame.

use bevy::prelude::*;
use bioscape::Cell;

use super::super::components::CellEntity;
use super::super::resources::SimWorld;
use super::{HoverCell, SelectedCell};

/// Multiplier on `phenotype.max_axis()` to set the click hit radius. The
/// rendered mesh is a unit sphere scaled to `(length, width, height)`; the
/// bounding sphere wrapping the ellipsoid has radius `max_axis`. The
/// 1.15× slack makes small cells comfortable to click without making
/// large neighbours steal nearby clicks (closest-along-ray wins anyway).
const PICK_SLACK: f32 = 1.15;

pub(super) fn pick_cell(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    mut sim_world: ResMut<SimWorld>,
    mut selected: ResMut<SelectedCell>,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }
    let Some(hit) = cast_pick_ray(&windows, &cameras, &sim_world.0.cells) else {
        selected.clear();
        return;
    };
    // Pull live brain weights from GPU into the CPU mirror before taking
    // the snapshot. Without this the inspector would show stale weights
    // (CPU `cell.genome.brain` is only written at spawn / reproduce; the
    // Hebbian / STDP / synaptic-scaling pipeline runs entirely on GPU).
    sim_world.0.sync_cell_brain_from_gpu(hit);
    let cell = &sim_world.0.cells[hit];
    selected.cell_id = Some(cell.cell_id);
    selected.snapshot = Some(*cell);
    selected.deceased = false;
}

pub(super) fn hover_cell(
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    sim_world: Res<SimWorld>,
    mut hover: ResMut<HoverCell>,
) {
    hover.cell_id = cast_pick_ray(&windows, &cameras, &sim_world.0.cells)
        .map(|idx| sim_world.0.cells[idx].cell_id);
}

pub(super) fn clear_on_escape(
    keys: Res<ButtonInput<KeyCode>>,
    mut selected: ResMut<SelectedCell>,
) {
    if keys.just_pressed(KeyCode::Escape) && selected.is_active() {
        selected.clear();
    }
}

/// Each tick, refresh the snapshot from `SimWorld.cells` if the cell is
/// still alive. If the cell_id disappeared (cell died / was swap-removed),
/// flip the `deceased` flag and freeze the last snapshot so the dialog can
/// keep rendering it. Independent of `CellEntity` queries — `SimWorld` is
/// the source of truth.
///
/// Also pulls the selected cell's brain weights from GPU into the CPU
/// mirror each tick so the inspector's weight heatmap reflects live
/// Hebbian / STDP / synaptic-scaling state. Cost: one `Wait` barrier on
/// `BRAIN_WEIGHTS_PER_CELL × 4 ≈ 18 KB` per tick — negligible for a
/// single selected cell.
pub(super) fn sync_selection_snapshot(
    mut sim_world: ResMut<SimWorld>,
    mut selected: ResMut<SelectedCell>,
) {
    let Some(id) = selected.cell_id else {
        return;
    };
    let idx = sim_world.0.find_cell_idx_by_id(id);
    match idx {
        Some(i) => {
            sim_world.0.sync_cell_brain_from_gpu(i);
            selected.snapshot = Some(sim_world.0.cells[i]);
            selected.deceased = false;
        }
        None => {
            selected.deceased = true;
        }
    }
}

/// Returns the index of the closest hit cell in `cells`, or `None` if the
/// ray missed everything (or the cursor is outside the window / no
/// camera). "Closest" means smallest ray parameter `t`, not Euclidean
/// distance from camera — handles overlapping cells the way the user
/// expects (frontmost wins).
fn cast_pick_ray(
    windows: &Query<&Window>,
    cameras: &Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    cells: &[Cell],
) -> Option<usize> {
    let window = windows.single().ok()?;
    let cursor = window.cursor_position()?;
    let (camera, camera_transform) = cameras.single().ok()?;
    let ray = camera.viewport_to_world(camera_transform, cursor).ok()?;
    let origin = ray.origin;
    let dir = *ray.direction;

    let mut best: Option<(usize, f32)> = None;
    for (idx, cell) in cells.iter().enumerate() {
        let center = Vec3::new(cell.position[0], cell.position[1], cell.position[2]);
        let radius = cell.phenotype.max_axis() * PICK_SLACK;
        let to_center = center - origin;
        let t = to_center.dot(dir);
        let closest = origin + dir * t;
        if closest.distance_squared(center) <= radius * radius {
            match best {
                Some((_, best_t)) if best_t <= t => {}
                _ => best = Some((idx, t)),
            }
        }
    }
    best.map(|(idx, _)| idx)
}

/// Hover-only suppression for the cell entity query — exported so the
/// outline system can read `HoverCell` without re-running the picking
/// test. Kept here next to the producer for clarity.
#[allow(dead_code)]
pub(super) fn cell_center_radius(cell_entity: &CellEntity) -> (Vec3, f32) {
    let cell = &cell_entity.0;
    let center = Vec3::new(cell.position[0], cell.position[1], cell.position[2]);
    let radius = cell.phenotype.max_axis();
    (center, radius)
}
