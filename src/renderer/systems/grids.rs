use bevy::diagnostic::Diagnostics;
use bevy::prelude::*;
use std::time::Instant;

use super::super::components::{CellEntity, Dying, FoodEntity};
use super::super::config::DIAG_GRID_REBUILD;
use super::super::resources::{CellEntityLookups, CellGrid, FoodGrid};

pub(crate) fn rebuild_cell_grid(
    mut grid: ResMut<CellGrid>,
    cells: Query<(Entity, &CellEntity), Without<Dying>>,
    mut diag: Diagnostics,
) {
    let t = Instant::now();
    grid.0.rebuild(
        cells
            .iter()
            .map(|(e, c)| (e, c.0.position, c.0.phenotype.effective_radius())),
    );
    diag.add_measurement(&DIAG_GRID_REBUILD, || t.elapsed().as_secs_f64() * 1000.0);
}

pub(crate) fn rebuild_food_grid(
    mut grid: ResMut<FoodGrid>,
    foods: Query<(Entity, &FoodEntity)>,
) {
    grid.0.rebuild(foods.iter().map(|(e, f)| (e, f.0.position, f.0.kind)));
}

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
