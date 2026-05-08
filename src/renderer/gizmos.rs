use bevy::prelude::*;
use bioscape::WORLD_HALF;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::components::{BondSnapshot, CellEntity, Dying};
use super::material::adhesion_hue;

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
