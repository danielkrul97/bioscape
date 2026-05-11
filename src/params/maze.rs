//! Maze world parameters. When the runtime maze toggle is off (no `--maze`
//! flag in headless, `KeyL` not pressed in renderer), this whole subsystem
//! is skipped — `ObstacleField` is not allocated and the world behaves
//! identically to the pre-maze toroidal homogeneous space.

/// Maze difficulty controls cell-grid size and post-carving wall removal.
/// Voxel grid resolution per xy axis = `2 * cells + 1` (interiors at odd
/// indices, wall channels at even). z resolution is fixed at 1 — walls are
/// infinite-height pillars to prevent vertical circumvention given the
/// ±15° pitch cap.
pub const MAZE_EASY_CELLS: (usize, usize) = (16, 9);
pub const MAZE_MEDIUM_CELLS: (usize, usize) = (24, 14);
pub const MAZE_HARD_CELLS: (usize, usize) = (32, 18);

/// Fraction of inner walls to randomly open AFTER perfect-maze carving,
/// per difficulty. 0.0 = perfect maze (single path between any pair of
/// cells), higher = more loops / shortcuts. Easy mode is closer to an
/// open arena with hint-walls; hard is a pure recursive-backtracker maze.
pub const MAZE_EASY_LOOP_FRAC: f32 = 0.45;
pub const MAZE_MEDIUM_LOOP_FRAC: f32 = 0.10;
pub const MAZE_HARD_LOOP_FRAC: f32 = 0.0;

/// Per-tick energy drain when a cell's collision-resolve pushes it out of
/// a wall. Discourages wall-grinding strategies without being lethal.
pub const MAZE_WALL_BUMP_DAMAGE: f32 = 0.5;

/// xy radius around the deepest maze cell counted as the goal zone. Cells
/// inside this radius count as "reached" for CSV navigation metrics.
pub const MAZE_GOAL_RADIUS: f32 = 40.0;

/// Goal-zone food-spawn weighting. Food candidates with `richness <
/// MAZE_GOAL_FOOD_BIAS × baseline_richness` near the goal are accepted
/// preferentially, biasing food toward the maze terminus and giving
/// reaching it real fitness payoff.
pub const MAZE_GOAL_FOOD_BIAS: f32 = 5.0;
