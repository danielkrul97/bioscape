//! Static voxel obstacle field for maze experiments. Toggleable via
//! `--maze[=easy|medium|hard]` (headless) or `KeyL` (renderer); when absent,
//! no allocation happens and the simulation runs the homogeneous toroidal
//! path unchanged.
//!
//! Topology: 2D xy maze projected through the full z-extent. Walls are
//! infinite-height pillars — given the ±15° pitch cap and the thin slab
//! aspect ratio, vertical circumvention is impractical so the simpler 2D
//! voxel model is sufficient.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::params::*;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MazeDifficulty {
    Easy,
    Medium,
    Hard,
}

impl MazeDifficulty {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "easy" => Some(Self::Easy),
            "medium" | "med" => Some(Self::Medium),
            "hard" => Some(Self::Hard),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Easy => "easy",
            Self::Medium => "medium",
            Self::Hard => "hard",
        }
    }

    fn cells(self) -> (usize, usize) {
        match self {
            Self::Easy => MAZE_EASY_CELLS,
            Self::Medium => MAZE_MEDIUM_CELLS,
            Self::Hard => MAZE_HARD_CELLS,
        }
    }

    fn loop_frac(self) -> f32 {
        match self {
            Self::Easy => MAZE_EASY_LOOP_FRAC,
            Self::Medium => MAZE_MEDIUM_LOOP_FRAC,
            Self::Hard => MAZE_HARD_LOOP_FRAC,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObstacleField {
    pub resolution: [usize; 3],
    pub world_half: [f32; 3],
    pub goal_position: [f32; 3],
    pub start_position: [f32; 3],
    pub difficulty: MazeDifficulty,
    pub seed: u64,
    occupied: Vec<bool>,
}

impl ObstacleField {
    /// Build a maze-topology obstacle field via recursive-backtracker on the
    /// cell graph, with optional loop-creation pass for easier difficulties.
    /// Voxel grid resolution is `2 * cells + 1` per xy axis. Cell interiors
    /// (odd indices) are always passable; wall channels (even indices) start
    /// occupied and are carved per the algorithm below.
    pub fn new_maze(world_half: [f32; 3], seed: u64, difficulty: MazeDifficulty) -> Self {
        let (cells_x, cells_y) = difficulty.cells();
        let res_x = 2 * cells_x + 1;
        let res_y = 2 * cells_y + 1;
        let res_z = 1usize;
        let n = res_x * res_y * res_z;
        let mut occupied = vec![true; n];

        for cy in 0..cells_y {
            for cx in 0..cells_x {
                let vx = 2 * cx + 1;
                let vy = 2 * cy + 1;
                occupied[vy * res_x + vx] = false;
            }
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let mut visited = vec![false; cells_x * cells_y];
        let mut stack: Vec<(usize, usize)> = Vec::with_capacity(cells_x * cells_y);
        visited[0] = true;
        stack.push((0, 0));

        while let Some(&(cx, cy)) = stack.last() {
            let mut neighbors: [(usize, usize, i32, i32); 4] = [(0, 0, 0, 0); 4];
            let mut count = 0usize;
            if cx > 0 && !visited[cy * cells_x + cx - 1] {
                neighbors[count] = (cx - 1, cy, -1, 0);
                count += 1;
            }
            if cx + 1 < cells_x && !visited[cy * cells_x + cx + 1] {
                neighbors[count] = (cx + 1, cy, 1, 0);
                count += 1;
            }
            if cy > 0 && !visited[(cy - 1) * cells_x + cx] {
                neighbors[count] = (cx, cy - 1, 0, -1);
                count += 1;
            }
            if cy + 1 < cells_y && !visited[(cy + 1) * cells_x + cx] {
                neighbors[count] = (cx, cy + 1, 0, 1);
                count += 1;
            }

            if count == 0 {
                stack.pop();
            } else {
                let pick = rng.random_range(0..count);
                let (nx, ny, dx, dy) = neighbors[pick];
                let wx = (2 * cx) as i32 + 1 + dx;
                let wy = (2 * cy) as i32 + 1 + dy;
                occupied[(wy as usize) * res_x + (wx as usize)] = false;
                visited[ny * cells_x + nx] = true;
                stack.push((nx, ny));
            }
        }

        let loop_frac = difficulty.loop_frac();
        if loop_frac > 0.0 {
            let mut wall_voxels: Vec<usize> = Vec::new();
            for vy in 1..res_y - 1 {
                for vx in 1..res_x - 1 {
                    let on_x_wall = vx % 2 == 0 && vy % 2 == 1;
                    let on_y_wall = vx % 2 == 1 && vy % 2 == 0;
                    if (on_x_wall || on_y_wall) && occupied[vy * res_x + vx] {
                        wall_voxels.push(vy * res_x + vx);
                    }
                }
            }
            wall_voxels.shuffle(&mut rng);
            let to_open = (wall_voxels.len() as f32 * loop_frac) as usize;
            for &idx in wall_voxels.iter().take(to_open) {
                occupied[idx] = false;
            }
        }

        let cs_x = (2.0 * world_half[0]) / res_x as f32;
        let cs_y = (2.0 * world_half[1]) / res_y as f32;
        let goal_vx = 2 * (cells_x - 1) + 1;
        let goal_vy = 2 * (cells_y - 1) + 1;
        let goal_position = [
            -world_half[0] + (goal_vx as f32 + 0.5) * cs_x,
            -world_half[1] + (goal_vy as f32 + 0.5) * cs_y,
            0.0,
        ];
        let start_position = [
            -world_half[0] + 1.5 * cs_x,
            -world_half[1] + 1.5 * cs_y,
            0.0,
        ];

        Self {
            resolution: [res_x, res_y, res_z],
            world_half,
            goal_position,
            start_position,
            difficulty,
            seed,
            occupied,
        }
    }

    pub fn voxel_size(&self) -> [f32; 2] {
        [
            2.0 * self.world_half[0] / self.resolution[0] as f32,
            2.0 * self.world_half[1] / self.resolution[1] as f32,
        ]
    }

    fn voxel_index(&self, pos_x: f32, pos_y: f32) -> Option<(usize, usize)> {
        let cs = self.voxel_size();
        let nx = self.resolution[0] as i32;
        let ny = self.resolution[1] as i32;
        let xi = ((pos_x + self.world_half[0]) / cs[0]).floor() as i32;
        let yi = ((pos_y + self.world_half[1]) / cs[1]).floor() as i32;
        if xi < 0 || xi >= nx || yi < 0 || yi >= ny {
            return None;
        }
        Some((xi as usize, yi as usize))
    }

    /// True if the voxel containing `pos` is occupied. Out-of-xy-bounds is
    /// treated as occupied — in maze mode the world boundary acts as solid
    /// wall (no toroidal wrap).
    pub fn sample(&self, pos: [f32; 3]) -> bool {
        match self.voxel_index(pos[0], pos[1]) {
            Some((xi, yi)) => self.occupied[yi * self.resolution[0] + xi],
            None => true,
        }
    }

    /// xy raycast — true if any voxel between `origin` and `target` is
    /// occupied. Uniform sampling at half-voxel pitch. z is single-pillar so
    /// only xy is traversed.
    pub fn raycast_blocked(&self, origin: [f32; 3], target: [f32; 3]) -> bool {
        let dx = target[0] - origin[0];
        let dy = target[1] - origin[1];
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < 1e-3 {
            return false;
        }
        let cs = self.voxel_size();
        let step = cs[0].min(cs[1]) * 0.5;
        let n = (dist / step).ceil() as usize;
        if n <= 1 {
            return false;
        }
        for i in 1..n {
            let t = i as f32 * step / dist;
            let p = [origin[0] + dx * t, origin[1] + dy * t, origin[2]];
            if self.sample(p) {
                return true;
            }
        }
        false
    }

    /// Returns the displacement that resolves overlap between a sphere at
    /// `pos` (radius `r`) and any of the 9 nearby voxels. Caller adds this
    /// to position post-step. xy-only — vertical motion is not resolved
    /// since walls span full z. Cell-center-inside-wall (post-spawn or
    /// teleport) escapes through the closer face axis.
    pub fn collision_push(&self, pos: [f32; 3], radius: f32) -> [f32; 3] {
        let cs = self.voxel_size();
        let nx = self.resolution[0] as i32;
        let ny = self.resolution[1] as i32;
        let xi = ((pos[0] + self.world_half[0]) / cs[0]).floor() as i32;
        let yi = ((pos[1] + self.world_half[1]) / cs[1]).floor() as i32;
        let mut push = [0.0_f32; 3];
        for dvy in -1..=1 {
            for dvx in -1..=1 {
                let vx = xi + dvx;
                let vy = yi + dvy;
                if vx < 0 || vx >= nx || vy < 0 || vy >= ny {
                    continue;
                }
                if !self.occupied[(vy as usize) * self.resolution[0] + (vx as usize)] {
                    continue;
                }
                let cx = -self.world_half[0] + (vx as f32 + 0.5) * cs[0];
                let cy = -self.world_half[1] + (vy as f32 + 0.5) * cs[1];
                let half_x = cs[0] * 0.5;
                let half_y = cs[1] * 0.5;
                let closest_x = pos[0].clamp(cx - half_x, cx + half_x);
                let closest_y = pos[1].clamp(cy - half_y, cy + half_y);
                let diff_x = pos[0] - closest_x;
                let diff_y = pos[1] - closest_y;
                let d2 = diff_x * diff_x + diff_y * diff_y;
                if d2 >= radius * radius {
                    continue;
                }
                let d = d2.sqrt();
                if d < 1e-4 {
                    let axis_x_overlap = half_x + radius - (pos[0] - cx).abs();
                    let axis_y_overlap = half_y + radius - (pos[1] - cy).abs();
                    if axis_x_overlap < axis_y_overlap {
                        let sign = if pos[0] >= cx { 1.0 } else { -1.0 };
                        push[0] += axis_x_overlap * sign;
                    } else {
                        let sign = if pos[1] >= cy { 1.0 } else { -1.0 };
                        push[1] += axis_y_overlap * sign;
                    }
                } else {
                    let pen = radius - d;
                    push[0] += diff_x / d * pen;
                    push[1] += diff_y / d * pen;
                }
            }
        }
        push
    }

    /// True if `pos` is within `MAZE_GOAL_RADIUS` of the goal cell center
    /// (xy distance only — z is uniform).
    pub fn at_goal(&self, pos: [f32; 3]) -> bool {
        let dx = pos[0] - self.goal_position[0];
        let dy = pos[1] - self.goal_position[1];
        dx * dx + dy * dy <= MAZE_GOAL_RADIUS * MAZE_GOAL_RADIUS
    }

    pub fn occupied(&self) -> &[bool] {
        &self.occupied
    }

    /// Wave 4: pack the boolean occupancy array into a `Vec<u32>` (one u32
    /// per voxel, value 0 or 1) for GPU upload. Direct one-bool-per-u32
    /// avoids bit-twiddling on the shader side; storage cost is small (a
    /// Hard-mode 65×37 maze packs to ~2.4 KB).
    pub fn packed_for_gpu(&self) -> Vec<u32> {
        self.occupied
            .iter()
            .map(|&b| if b { 1 } else { 0 })
            .collect()
    }

    /// Whisker raycast: shoots `WHISKER_COUNT` short rays from `pos` in the
    /// six body-frame cardinal directions (forward, back, right, left, up,
    /// down) and returns per-ray free-distance normalized to `[0, 1]`.
    /// `1.0` = no wall within `WHISKER_RANGE`; `0.0` = wall touching the
    /// origin. Always xy-plane (z directions return 1.0 since walls span
    /// full z). Pure read.
    pub fn whisker_distances(
        &self,
        pos: [f32; 3],
        heading: f32,
        pitch: f32,
    ) -> [f32; WHISKER_COUNT] {
        let dirs = whisker_directions(heading, pitch);
        let cs = self.voxel_size();
        let step = cs[0].min(cs[1]) * 0.5;
        let n_steps = (WHISKER_RANGE / step).ceil() as usize;
        let mut out = [1.0_f32; WHISKER_COUNT];
        for (k, dir) in dirs.iter().enumerate() {
            // z-axis whiskers always clear (xy-only walls). Skip raycast.
            if dir[0].abs() < 1e-6 && dir[1].abs() < 1e-6 {
                continue;
            }
            for s in 1..=n_steps {
                let t = s as f32 * step;
                let p = [pos[0] + dir[0] * t, pos[1] + dir[1] * t, pos[2]];
                if self.sample(p) {
                    out[k] = (t / WHISKER_RANGE).clamp(0.0, 1.0);
                    break;
                }
            }
        }
        out
    }

    /// Sample this obstacle field at every voxel center of an external
    /// `[res_x, res_y, res_z]` grid (e.g. SmellField / pheromone field) and
    /// return a flat boolean mask in the same row-major layout the diffusion
    /// stencil uses. True = wall / Neumann boundary. Computed once at world
    /// setup; the diffusion step then reads it without per-voxel lookups
    /// against this field.
    pub fn mask_for_grid(&self, resolution: [usize; 3]) -> Vec<bool> {
        let nx = resolution[0];
        let ny = resolution[1];
        let nz = resolution[2];
        let cs_x = (2.0 * self.world_half[0]) / nx as f32;
        let cs_y = (2.0 * self.world_half[1]) / ny as f32;
        let cs_z = if nz > 0 {
            (2.0 * self.world_half[2]) / nz as f32
        } else {
            1.0
        };
        let mut mask = vec![false; nx * ny * nz];
        for k in 0..nz {
            let z = -self.world_half[2] + (k as f32 + 0.5) * cs_z;
            for j in 0..ny {
                let y = -self.world_half[1] + (j as f32 + 0.5) * cs_y;
                for i in 0..nx {
                    let x = -self.world_half[0] + (i as f32 + 0.5) * cs_x;
                    if self.sample([x, y, z]) {
                        mask[k * nx * ny + j * nx + i] = true;
                    }
                }
            }
        }
        mask
    }
}

/// The six body-frame whisker ray directions — forward, back, right, left,
/// up, down — as unit vectors in world frame. Shared by
/// `ObstacleField::whisker_distances` (the raycast) and the renderer's
/// whisker overlay so the two cannot drift out of order.
pub fn whisker_directions(heading: f32, pitch: f32) -> [[f32; 3]; WHISKER_COUNT] {
    let fwd = crate::forward_vector(heading, pitch);
    let right = [-heading.sin(), heading.cos(), 0.0];
    [
        fwd,
        [-fwd[0], -fwd[1], -fwd[2]],
        right,
        [-right[0], -right[1], -right[2]],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
}

/// Sprint 195 — deterministic stateless transduction noise for the whisker
/// spring-damper model. Hashes `(cell_index, tick, whisker_k)` to `[-1, 1]`.
/// Stateless (no RNG buffer, no reproduction reset); mirrored with identical
/// integer arithmetic in `shaders/sensor_gather.wgsl` so the CPU and GPU
/// whisker paths stay in bit-comparable parity.
pub fn whisker_noise(cell_index: u32, tick: u32, whisker_k: u32) -> f32 {
    let mut h = cell_index.wrapping_mul(0x9E37_79B9);
    h ^= tick.wrapping_mul(0x85EB_CA6B);
    h ^= whisker_k.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    // Top 24 bits → [-1, 1).
    (h >> 8) as f32 / 8_388_608.0 - 1.0
}

/// Sprint 195 — one semi-implicit Euler step of the whisker spring-damper
/// for a single whisker. `raw` is the raycast free-distance ∈ [0, 1]; the
/// wall-imposed target deflection is `1 − raw`. Mutates `(deflection,
/// velocity)` in place and returns the sensed value `clamp(1 − deflection)
/// + noise`. Mirrors the per-whisker body in `shaders/sensor_gather.wgsl`
/// — `dt` is the fixed sim step `1/60` on both sides.
pub fn whisker_step(deflection: &mut f32, velocity: &mut f32, raw: f32, noise: f32) -> f32 {
    let dt = 1.0_f32 / 60.0;
    let target = 1.0 - raw;
    let accel = WHISKER_STIFFNESS * (target - *deflection) - WHISKER_DAMPING * *velocity;
    *velocity += accel * dt;
    *deflection += *velocity * dt;
    ((1.0 - *deflection).clamp(0.0, 1.0) + noise).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HALF: [f32; 3] = [960.0, 540.0, 100.0];

    #[test]
    fn maze_is_deterministic_per_seed() {
        let a = ObstacleField::new_maze(HALF, 42, MazeDifficulty::Medium);
        let b = ObstacleField::new_maze(HALF, 42, MazeDifficulty::Medium);
        assert_eq!(a.occupied, b.occupied);
        assert_eq!(a.goal_position, b.goal_position);
    }

    #[test]
    fn different_seeds_diverge() {
        let a = ObstacleField::new_maze(HALF, 1, MazeDifficulty::Medium);
        let b = ObstacleField::new_maze(HALF, 2, MazeDifficulty::Medium);
        assert_ne!(a.occupied, b.occupied);
    }

    #[test]
    fn cell_interiors_always_passable() {
        let m = ObstacleField::new_maze(HALF, 0, MazeDifficulty::Hard);
        let res_x = m.resolution[0];
        let res_y = m.resolution[1];
        for vy in 0..res_y {
            for vx in 0..res_x {
                if vx % 2 == 1 && vy % 2 == 1 {
                    assert!(
                        !m.occupied[vy * res_x + vx],
                        "cell interior occupied at ({vx},{vy})"
                    );
                }
            }
        }
    }

    #[test]
    fn outer_boundary_is_solid() {
        let m = ObstacleField::new_maze(HALF, 0, MazeDifficulty::Medium);
        assert!(m.sample([HALF[0] * 2.0, 0.0, 0.0]));
        assert!(m.sample([0.0, -HALF[1] * 2.0, 0.0]));
    }

    #[test]
    fn raycast_inside_passable_cell_returns_false() {
        let m = ObstacleField::new_maze(HALF, 0, MazeDifficulty::Medium);
        let origin = m.start_position;
        let target = [m.start_position[0] + 5.0, m.start_position[1] + 5.0, 0.0];
        assert!(!m.raycast_blocked(origin, target));
    }

    #[test]
    fn raycast_across_perfect_maze_hits_walls() {
        let m = ObstacleField::new_maze(HALF, 0, MazeDifficulty::Hard);
        // Long diagonal across the whole maze in HARD mode (perfect maze, no
        // loops opened) is overwhelmingly likely to cross at least one wall.
        let blocked = m.raycast_blocked(m.start_position, m.goal_position);
        assert!(blocked, "long diagonal in hard maze unexpectedly clear");
    }

    #[test]
    fn collision_push_zero_at_passable_cell() {
        let m = ObstacleField::new_maze(HALF, 0, MazeDifficulty::Easy);
        let push = m.collision_push(m.start_position, 1.0);
        assert!(push[0].abs() < 1e-3);
        assert!(push[1].abs() < 1e-3);
    }

    #[test]
    fn collision_push_repels_from_wall() {
        let m = ObstacleField::new_maze(HALF, 0, MazeDifficulty::Medium);
        // Find any occupied voxel and place a sphere just inside its boundary
        let res_x = m.resolution[0];
        let res_y = m.resolution[1];
        let cs = m.voxel_size();
        let mut found_repel = false;
        'outer: for vy in 1..res_y - 1 {
            for vx in 1..res_x - 1 {
                if !m.occupied[vy * res_x + vx] {
                    continue;
                }
                let cx = -HALF[0] + (vx as f32 + 0.5) * cs[0];
                let cy = -HALF[1] + (vy as f32 + 0.5) * cs[1];
                // Place sphere slightly past wall edge so there's overlap
                let pos = [cx + cs[0] * 0.5 + 1.0, cy, 0.0];
                let push = m.collision_push(pos, 5.0);
                if push[0].abs() > 1e-3 || push[1].abs() > 1e-3 {
                    found_repel = true;
                    break 'outer;
                }
            }
        }
        assert!(
            found_repel,
            "expected at least one wall to push back a nearby sphere"
        );
    }

    #[test]
    fn easy_more_open_than_hard() {
        let easy = ObstacleField::new_maze(HALF, 7, MazeDifficulty::Easy);
        let hard = ObstacleField::new_maze(HALF, 7, MazeDifficulty::Hard);
        let easy_open =
            easy.occupied.iter().filter(|&&o| !o).count() as f32 / easy.occupied.len() as f32;
        let hard_open =
            hard.occupied.iter().filter(|&&o| !o).count() as f32 / hard.occupied.len() as f32;
        assert!(
            easy_open > hard_open,
            "easy={easy_open:.3} should be more open than hard={hard_open:.3}"
        );
    }

    #[test]
    fn goal_position_is_passable() {
        let m = ObstacleField::new_maze(HALF, 0, MazeDifficulty::Medium);
        assert!(!m.sample(m.goal_position));
        assert!(m.at_goal(m.goal_position));
    }

    #[test]
    fn parse_difficulty_strings() {
        assert_eq!(MazeDifficulty::parse("easy"), Some(MazeDifficulty::Easy));
        assert_eq!(
            MazeDifficulty::parse("MEDIUM"),
            Some(MazeDifficulty::Medium)
        );
        assert_eq!(MazeDifficulty::parse("hard"), Some(MazeDifficulty::Hard));
        assert_eq!(MazeDifficulty::parse("nope"), None);
    }
}
