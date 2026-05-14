//! Headless JSON dump pipeline. Opt-in via `--dump-dir <PATH>`.
//!
//! Output layout per dump event:
//! ```text
//! <dump-dir>/
//!   gen_000050/
//!     manifest.json              population summary (all alive cells)
//!     cell_<id>_age<a>_lin<l>.json   full Cell JSON for top-K (by age desc, energy desc tiebreak)
//!   gen_000100/...
//!   final/
//!     manifest.json
//!     cell_<id>_age<a>_lin<l>.json   × final-top-K
//! ```
//!
//! Manifest holds a per-cell summary (~200 B/cell) plus run-level fields
//! (seed, gen, tick, density, maze state). The full per-cell JSON only
//! exists for the selected top-K, marked with `has_full_dump: true` in the
//! summary; the rest of the population appears in the summary with
//! `has_full_dump: false` so a downstream reader can scan the manifest for
//! lineage / age distribution without loading 1500 full Cell files.

use bioscape::json_export;
use bioscape::sim::World;
use bioscape::Cell;
use serde::Serialize;
use std::cmp::Ordering;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct DumpConfig {
    pub dir: PathBuf,
    /// Period in generations between periodic dumps. 0 disables periodic
    /// dumps entirely (final-only mode).
    pub every: u64,
    pub periodic_top_k: usize,
    pub final_top_k: usize,
}

/// Static per-run metadata embedded in every manifest. The headless main
/// builds this once at startup and passes it into each dump call.
#[derive(Debug, Clone)]
pub struct RunMeta {
    pub seed: u64,
    pub map_seed: u64,
    pub maze_label: Option<String>,
}

#[derive(Serialize)]
struct CellSummary {
    cell_id: u64,
    lineage_id: u64,
    lineage_birth_gen: u64,
    age: u64,
    energy: f32,
    position: [f32; 3],
    heading: f32,
    pitch: f32,
    bond_count: u32,
    adhesion_type: u8,
    cell_state: f32,
    damage_accum: f32,
    has_full_dump: bool,
}

#[derive(Serialize)]
struct Manifest<'a> {
    seed: u64,
    map_seed: u64,
    maze_label: Option<&'a str>,
    generation: u64,
    tick: u64,
    selection: &'a str,
    alive_count: usize,
    foods_count: usize,
    coop_foods_count: usize,
    density_factor: f32,
    maze_active: bool,
    full_dump_count: usize,
    cells: Vec<CellSummary>,
}

/// Periodic dump trigger: called from main after each generation boundary.
/// Returns `Ok(Some(path))` when a dump was written, `Ok(None)` when the
/// generation does not match the periodic schedule, or an error if the
/// write failed.
pub fn maybe_dump_generation(
    world: &World,
    run: &RunMeta,
    cfg: &DumpConfig,
    generation: u64,
) -> io::Result<Option<PathBuf>> {
    if cfg.every == 0 || generation == 0 || !generation.is_multiple_of(cfg.every) {
        return Ok(None);
    }
    if world.cells.is_empty() {
        return Ok(None);
    }
    let subdir = cfg.dir.join(format!("gen_{:06}", generation));
    let path = write_dump(world, run, cfg.periodic_top_k, &subdir, "periodic_top_k")?;
    Ok(Some(path))
}

/// End-of-run dump. Always writes when called — caller decides whether to
/// invoke it (typically gated on `--dump-dir` being set).
pub fn dump_final(world: &World, run: &RunMeta, cfg: &DumpConfig) -> io::Result<PathBuf> {
    let subdir = cfg.dir.join("final");
    write_dump(world, run, cfg.final_top_k, &subdir, "final_top_k")
}

/// Shared dump body: builds the manifest, writes top-K full Cell JSONs,
/// then the manifest itself. Returns the directory path that was written.
fn write_dump(
    world: &World,
    run: &RunMeta,
    top_k: usize,
    subdir: &Path,
    selection_label: &str,
) -> io::Result<PathBuf> {
    fs::create_dir_all(subdir)?;

    let top_indices = select_top_k(&world.cells, top_k);
    let top_id_set: std::collections::HashSet<u64> =
        top_indices.iter().map(|&i| world.cells[i].cell_id).collect();

    let cells_summary: Vec<CellSummary> = world
        .cells
        .iter()
        .map(|c| CellSummary {
            cell_id: c.cell_id,
            lineage_id: c.lineage_id,
            lineage_birth_gen: c.lineage_birth_gen,
            age: c.age,
            energy: c.energy,
            position: c.position,
            heading: c.heading,
            pitch: c.pitch,
            bond_count: c.n_bonds(),
            adhesion_type: c.genome.adhesion_type,
            cell_state: c.cell_state,
            damage_accum: c.damage_accum,
            has_full_dump: top_id_set.contains(&c.cell_id),
        })
        .collect();

    for &idx in &top_indices {
        let cell = &world.cells[idx];
        let filename = json_export::stable_filename(cell);
        let json = json_export::serialize_cell(cell)
            .map_err(|e| io::Error::other(format!("serialize_cell: {e}")))?;
        fs::write(subdir.join(filename), json)?;
    }

    let manifest = Manifest {
        seed: run.seed,
        map_seed: run.map_seed,
        maze_label: run.maze_label.as_deref(),
        generation: world.clock.generation,
        tick: world.clock.tick,
        selection: selection_label,
        alive_count: world.cells.len(),
        foods_count: world.foods.len(),
        coop_foods_count: world.coop_foods.len(),
        density_factor: world.density_factor,
        maze_active: world.obstacles.is_some(),
        full_dump_count: top_indices.len(),
        cells: cells_summary,
    };
    let manifest_value = serde_json::to_value(&manifest)
        .map_err(|e| io::Error::other(format!("manifest to_value: {e}")))?;
    let manifest_text = json_export::format_human(&manifest_value);
    fs::write(subdir.join("manifest.json"), manifest_text)?;

    Ok(subdir.to_path_buf())
}

/// Returns indices into `cells` of the top-K cells sorted by age desc with
/// energy desc as the tiebreak. `k` is clamped to `cells.len()`.
fn select_top_k(cells: &[Cell], k: usize) -> Vec<usize> {
    let k = k.min(cells.len());
    if k == 0 {
        return Vec::new();
    }
    let mut indices: Vec<usize> = (0..cells.len()).collect();
    indices.sort_unstable_by(|&a, &b| {
        let ca = &cells[a];
        let cb = &cells[b];
        match cb.age.cmp(&ca.age) {
            Ordering::Equal => cb
                .energy
                .partial_cmp(&ca.energy)
                .unwrap_or(Ordering::Equal),
            other => other,
        }
    });
    indices.truncate(k);
    indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_cell(rng: &mut impl rand::Rng, cell_id: u64, age: u64, energy: f32) -> Cell {
        let mut c = Cell::random(rng, [100.0, 100.0, 0.0], 0, 0, cell_id);
        c.age = age;
        c.energy = energy;
        c
    }

    #[test]
    fn top_k_sorts_by_age_then_energy() {
        let mut rng = StdRng::seed_from_u64(1);
        let cells = vec![
            make_cell(&mut rng, 1, 100, 0.5),
            make_cell(&mut rng, 2, 200, 0.1),
            make_cell(&mut rng, 3, 200, 0.9),
            make_cell(&mut rng, 4, 50, 1.0),
        ];
        let top = select_top_k(&cells, 3);
        // age 200 cells come first, with energy=0.9 (id=3) before energy=0.1 (id=2),
        // then age 100 (id=1).
        let ids: Vec<u64> = top.iter().map(|&i| cells[i].cell_id).collect();
        assert_eq!(ids, vec![3, 2, 1]);
    }

    #[test]
    fn top_k_clamps_to_len() {
        let mut rng = StdRng::seed_from_u64(1);
        let cells = vec![make_cell(&mut rng, 1, 10, 0.5)];
        let top = select_top_k(&cells, 50);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn top_k_zero_returns_empty() {
        let mut rng = StdRng::seed_from_u64(1);
        let cells = vec![make_cell(&mut rng, 1, 10, 0.5)];
        let top = select_top_k(&cells, 0);
        assert!(top.is_empty());
    }
}
