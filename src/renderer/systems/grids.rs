use bevy::prelude::*;

use super::super::components::{CellEntity, Dying};
use super::super::resources::CellEntityLookups;

/// R-#2: build `id_to_entity` raz/tick. Cells layout (entity sady) je stable
/// uvnitř ticku až do reproduce/die_and_drop_carrion na konci, takže jediný
/// rebuild stačí. Konzumuje `cell_eats_food`; další systémy jako
/// `resolve_cell_collisions` mají vlastní snapshot-based `entity_to_idx`
/// (potřebují snapshot order indexing) a zůstávají na Local<FxHashMap>.
pub(crate) fn rebuild_cell_entity_lookups(
    mut lookups: ResMut<CellEntityLookups>,
    cells: Query<(Entity, &CellEntity), Without<Dying>>,
) {
    lookups.rebuild(cells.iter().map(|(e, c)| (e, c.0.cell_id, c.0.position)));
}
