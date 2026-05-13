use bevy::diagnostic::Diagnostics;
use bevy::prelude::*;
use std::time::Instant;

use super::super::components::{CellEntity, Dying};
use super::super::config::{DEATH_FADE_TICKS, DIAG_SYNC_TRANSFORMS};
use super::super::material::{cell_rotation, cell_scale};

pub(crate) fn sync_transforms(
    mut cells: Query<(&CellEntity, &mut Transform), Without<Dying>>,
    mut diag: Diagnostics,
) {
    let t = Instant::now();
    cells.par_iter_mut().for_each(|(cell, mut transform)| {
        transform.translation.x = cell.0.position[0];
        transform.translation.y = cell.0.position[1];
        transform.translation.z = cell.0.position[2];
        transform.rotation = cell_rotation(cell.0.heading, cell.0.pitch);
        let target_scale = cell_scale(&cell.0.phenotype);
        if (transform.scale.x - target_scale.x).abs() > 1e-3
            || (transform.scale.y - target_scale.y).abs() > 1e-3
            || (transform.scale.z - target_scale.z).abs() > 1e-3
        {
            transform.scale = target_scale;
        }
    });
    diag.add_measurement(&DIAG_SYNC_TRANSFORMS, || t.elapsed().as_secs_f64() * 1000.0);
}

/// Sprint 36 (recreated v S181 cleanup): visual fade-out for entities
/// marked `Dying`. Shrinks scale linearly across `DEATH_FADE_TICKS`,
/// despawns when the timer expires. Pure visual; SimWorld lifecycle
/// (Sprint 176-178) handles the underlying cell death.
pub(crate) fn tick_death_fade(
    mut dying: Query<(Entity, &mut Dying, &CellEntity, &mut Transform)>,
    mut commands: Commands,
) {
    for (entity, mut d, cell, mut transform) in &mut dying {
        if d.ticks_left == 0 {
            commands.entity(entity).despawn();
            continue;
        }
        d.ticks_left -= 1;
        let progress = d.ticks_left as f32 / DEATH_FADE_TICKS as f32;
        transform.scale = cell_scale(&cell.0.phenotype) * progress;
    }
}
