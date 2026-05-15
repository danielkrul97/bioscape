use bevy::prelude::*;
use bioscape::SYMBIONT_UPKEEP_DEFICIT_TICKS;

use super::super::components::{CellEntity, Pooled, SymbiontMarker};
use super::super::material::cell_scale;

const BASE_SCALE_FRAC: f32 = 0.40;
const PULSE_AMP_REST: f32 = 0.04;
const PULSE_FREQ_REST: f32 = 1.5;
const PULSE_AMP_STRESS: f32 = 0.10;
const PULSE_FREQ_STRESS: f32 = 5.0;
const ORBIT_RADIUS_FRAC: f32 = 0.15;
const ORBIT_SPEED: f32 = 0.45;

pub(crate) fn sync_symbionts(
    time: Res<Time>,
    mut symbionts: Query<(&SymbiontMarker, &mut Transform, &mut Visibility)>,
    cells: Query<
        (&CellEntity, &Visibility, &Transform),
        (Without<SymbiontMarker>, Without<Pooled>),
    >,
) {
    let t = time.elapsed_secs();
    for (marker, mut transform, mut vis) in &mut symbionts {
        let Ok((cell_entity, owner_vis, cell_transform)) = cells.get(marker.owner) else {
            vis.set_if_neq(Visibility::Hidden);
            continue;
        };
        if matches!(owner_vis, Visibility::Hidden) {
            vis.set_if_neq(Visibility::Hidden);
            continue;
        }
        let cell = &cell_entity.0;
        let Some(sym) = cell.symbiont.as_ref() else {
            vis.set_if_neq(Visibility::Hidden);
            continue;
        };
        // Desync the population so bearers pulse out of phase instead of strobing in unison.
        let phase = (cell.cell_id as f32 * 0.7) % std::f32::consts::TAU;
        let stress = (sym.deficit_streak as f32 / SYMBIONT_UPKEEP_DEFICIT_TICKS as f32)
            .clamp(0.0, 1.0);
        let freq = PULSE_FREQ_REST + (PULSE_FREQ_STRESS - PULSE_FREQ_REST) * stress;
        let amp = PULSE_AMP_REST + (PULSE_AMP_STRESS - PULSE_AMP_REST) * stress;
        let pulse = (t * freq + phase).sin() * amp;
        let scale_frac = BASE_SCALE_FRAC + pulse;
        let host_axes = cell_scale(&cell.phenotype);
        let orbit_angle = t * ORBIT_SPEED + phase;
        // Offset is expressed in the host's local frame so it tracks host rotation.
        let offset_local = Vec3::new(
            orbit_angle.cos() * host_axes.x * ORBIT_RADIUS_FRAC,
            orbit_angle.sin() * host_axes.y * ORBIT_RADIUS_FRAC,
            (t * ORBIT_SPEED * 0.7 + phase).sin() * host_axes.z * ORBIT_RADIUS_FRAC * 0.5,
        );
        let world_offset = cell_transform.rotation * offset_local;
        transform.translation = cell_transform.translation + world_offset;
        transform.scale = host_axes * scale_frac;
        vis.set_if_neq(Visibility::Visible);
    }
}
