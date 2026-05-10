use bevy::prelude::*;
use bioscape::{spike_direction, CELL_RADIUS, MIN_SPIKE_LENGTH};

use super::super::components::{CellEntity, SpikeEntity};

/// Spike thickness as a fraction of CELL_RADIUS — controls cone radius.
/// Length comes from `phenotype.spikes[slot].length` directly (sim units).
const SPIKE_THICKNESS_FRAC: f32 = 0.25;
/// Visibility floor — spikes below this length stay hidden to declutter.
const SPIKE_VISIBLE_MIN: f32 = MIN_SPIKE_LENGTH + 0.05;

/// Per-frame sync of spike transforms from owner cells. Hidden when:
/// - owner cell despawned (also despawns the spike entity)
/// - slot is inactive (slot >= spike_count)
/// - length below threshold
pub(crate) fn sync_spikes(
    mut spikes: Query<(Entity, &SpikeEntity, &mut Transform, &mut Visibility)>,
    cells: Query<&CellEntity>,
    mut commands: Commands,
) {
    for (entity, spike, mut transform, mut vis) in &mut spikes {
        let Ok(cell) = cells.get(spike.owner) else {
            commands.entity(entity).despawn();
            continue;
        };
        let cell = &cell.0;
        let slot_idx = spike.slot as usize;
        if slot_idx >= cell.phenotype.spike_count as usize {
            *vis = Visibility::Hidden;
            continue;
        }
        let s = cell.phenotype.spikes[slot_idx];
        if s.length < SPIKE_VISIBLE_MIN {
            *vis = Visibility::Hidden;
            continue;
        }
        let dir = spike_direction(cell.heading, cell.pitch, &s);
        let dir = Vec3::new(dir[0], dir[1], dir[2]).normalize_or_zero();
        // Body surface offset along spike direction. Approximate ellipsoid as
        // sphere of effective radius — exact ellipsoid surface intersection is
        // expensive and unnecessary for visual feedback.
        let surface = CELL_RADIUS * cell.phenotype.effective_radius();
        let half_len = s.length * 0.5 * CELL_RADIUS;
        let center = Vec3::new(
            cell.position[0],
            cell.position[1],
            cell.position[2],
        ) + dir * (surface + half_len);
        let thickness = SPIKE_THICKNESS_FRAC * CELL_RADIUS;
        let length = s.length * CELL_RADIUS;
        transform.translation = center;
        transform.rotation = Quat::from_rotation_arc(Vec3::Y, dir);
        transform.scale = Vec3::new(thickness, length, thickness);
        *vis = Visibility::Visible;
    }
}
