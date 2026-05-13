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

/// Whisker proximity sensor — number of raycast directions in body frame.
/// Six covers ±forward, ±right, ±up adequately for xy navigation; cheap
/// enough to compute every tick (one short raycast each). Always-on brain
/// input slot count regardless of maze toggle: when no obstacles exist the
/// helper returns 1.0 (no wall within range) for every direction.
pub const WHISKER_COUNT: usize = 6;

/// Maximum whisker raycast range, in world units. Past this distance the
/// returned signal saturates at 1.0 ("no wall in range"). Tuned to a few
/// voxel widths at default world size (voxel size ≈ 60 units in Medium maze
/// → 90 covers 1.5 voxels, plenty for "wall ahead" warning at navigation
/// speed).
pub const WHISKER_RANGE: f32 = 90.0;

/// Eligibility-trace decay per second. Hebbian update changes from
/// instantaneous `Δw = lr · reward · pre · post` (1-tick myopic) to trace-
/// based: `e[i,j] *= decay; e[i,j] += pre · post; w[i,j] += lr · reward · e[i,j]`.
/// At dt=1/60 s and `decay_per_sec=0.5`, per-tick decay ≈ 0.99 → effective
/// reward window ~120 ticks (2 s). That spans a maze-corridor transit at
/// typical speed, so cells reaching food can credit the choice they made
/// at the previous corridor junction.
pub const HEBBIAN_TRACE_DECAY_PER_SEC: f32 = 0.5;

/// Per-cell episodic novelty: ring buffer of recently visited coarse-grid
/// voxel indices. New voxel entry → novelty boost; revisit → no boost.
/// Buffer length controls how far back "recent" reaches; 32 covers ~1/2
/// generation at typical movement speed without dominating cell memory
/// footprint.
pub const NOVELTY_HISTORY_LEN: usize = 32;

/// Coarse grid cell size for novelty bucketing, in world units. Larger =
/// "I've been roughly here" granularity; smaller = each step counts as
/// novel. 80 ≈ one maze-corridor cell.
pub const NOVELTY_GRID_CELL_SIZE: f32 = 80.0;

/// Novelty reward magnitude. Added to the Hebbian `reward` term whenever a
/// cell visits a voxel not in its history buffer. Stays small relative to
/// food/predation rewards (~1.0–10.0) so the cell isn't pulled away from
/// goal-seeking by pure exploration. 0.05 ≈ 5 % of a food event.
pub const NOVELTY_REWARD_MAGNITUDE: f32 = 0.05;
