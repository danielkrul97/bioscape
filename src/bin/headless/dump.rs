//! Headless JSON dump pipeline. Opt-in via `--dump-dir <PATH>`.
//!
//! Output layout per dump event:
//! ```text
//! <dump-dir>/
//!   gen_000050/
//!     manifest.json              population summary (all alive cells)
//!     brain_stats.csv            per-cell hidden-layer utilization stats
//!     cell_<id>_age<a>_lin<l>.json   full Cell JSON for top-K (by age desc, energy desc tiebreak)
//!   gen_000100/...
//!   final/
//!     manifest.json
//!     brain_stats.csv
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
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Thresholds for classifying hidden-neuron activations across the
/// population. `|act| > SAT_THRESHOLD` → tanh-saturated; `|act| < DEAD_THRESHOLD`
/// → effectively silent. Cells with many saturated/dead neurons indicate
/// underutilized brain capacity vs. evolved `hidden_n`.
const SAT_THRESHOLD: f32 = 0.95;
const DEAD_THRESHOLD: f32 = 0.05;

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

    let brain_csv = render_brain_stats_csv(&world.cells);
    fs::write(subdir.join("brain_stats.csv"), brain_csv)?;

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

fn render_brain_stats_csv(cells: &[Cell]) -> String {
    let mut out = String::with_capacity(64 + cells.len() * 96);
    out.push_str(
        "cell_id,lineage_id,age,energy,hidden_n,sat_count,dead_count,active_count,frac_sat,frac_dead,max_abs_act,mean_abs_act\n",
    );
    for c in cells {
        let stats = BrainStats::compute(c);
        let _ = writeln!(
            out,
            "{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6}",
            c.cell_id,
            c.lineage_id,
            c.age,
            c.energy,
            stats.hidden_n,
            stats.sat_count,
            stats.dead_count,
            stats.active_count,
            stats.frac_sat,
            stats.frac_dead,
            stats.max_abs_act,
            stats.mean_abs_act,
        );
    }
    out
}

struct BrainStats {
    hidden_n: u32,
    sat_count: u32,
    dead_count: u32,
    active_count: u32,
    frac_sat: f32,
    frac_dead: f32,
    max_abs_act: f32,
    mean_abs_act: f32,
}

impl BrainStats {
    fn compute(cell: &Cell) -> Self {
        let hidden_n = cell.genome.brain.hidden_n;
        let n = (hidden_n as usize).min(cell.last_hidden.len());
        let mut sat: u32 = 0;
        let mut dead: u32 = 0;
        let mut max_abs: f32 = 0.0;
        let mut sum_abs: f32 = 0.0;
        for &a in &cell.last_hidden[..n] {
            let abs = a.abs();
            if abs > SAT_THRESHOLD {
                sat += 1;
            } else if abs < DEAD_THRESHOLD {
                dead += 1;
            }
            if abs > max_abs {
                max_abs = abs;
            }
            sum_abs += abs;
        }
        let denom = n.max(1) as f32;
        let active = (n as u32).saturating_sub(sat).saturating_sub(dead);
        BrainStats {
            hidden_n,
            sat_count: sat,
            dead_count: dead,
            active_count: active,
            frac_sat: sat as f32 / denom,
            frac_dead: dead as f32 / denom,
            max_abs_act: max_abs,
            mean_abs_act: sum_abs / denom,
        }
    }
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

    #[test]
    fn brain_stats_classifies_sat_dead_active() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut cell = make_cell(&mut rng, 42, 100, 5.0);
        cell.genome.brain.hidden_n = 5;
        cell.last_hidden = [0.0; bioscape::BRAIN_HIDDEN];
        cell.last_hidden[0] = 0.99;   // sat
        cell.last_hidden[1] = -0.97;  // sat
        cell.last_hidden[2] = 0.5;    // active
        cell.last_hidden[3] = 0.01;   // dead
        cell.last_hidden[4] = -0.02;  // dead
        // index 5+ ignored (outside hidden_n)
        cell.last_hidden[6] = 0.99;

        let s = BrainStats::compute(&cell);
        assert_eq!(s.hidden_n, 5);
        assert_eq!(s.sat_count, 2);
        assert_eq!(s.dead_count, 2);
        assert_eq!(s.active_count, 1);
        assert!((s.frac_sat - 0.4).abs() < 1e-6);
        assert!((s.frac_dead - 0.4).abs() < 1e-6);
        assert!((s.max_abs_act - 0.99).abs() < 1e-6);
    }

    #[test]
    fn brain_stats_csv_has_header_and_one_row_per_cell() {
        let mut rng = StdRng::seed_from_u64(1);
        let cells = vec![
            make_cell(&mut rng, 1, 10, 0.5),
            make_cell(&mut rng, 2, 20, 0.8),
        ];
        let csv = render_brain_stats_csv(&cells);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3, "header + 2 rows");
        assert!(lines[0].starts_with("cell_id,lineage_id,age,energy,hidden_n"));
        assert!(lines[1].starts_with("1,"));
        assert!(lines[2].starts_with("2,"));
    }
}
