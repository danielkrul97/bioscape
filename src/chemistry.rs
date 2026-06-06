use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// 3D volumetric scalar field with explicit-Jacobi diffusion + decay. Per-axis
/// resolution `[res_x, res_y, res_z]` — typically `[64, 64, 16]` to match the
/// thin-slab aspect ratio (`world_half_z << world_half_xy`). Grid layout:
/// `idx = z*W*H + y*W + x`. The 7-point Laplacian stencil is stable for
/// `diffusion < 1/6` (the 2D version was `< 1/4`); `SMELL_DIFFUSION = 0.15`
/// sits below both bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmellField {
    pub resolution: [usize; 3],
    pub world_half: [f32; 3],
    grid: Vec<f32>,
    scratch: Vec<f32>,
}

impl SmellField {
    pub fn new(resolution: [usize; 3], world_half: [f32; 3]) -> Self {
        let n = resolution[0] * resolution[1] * resolution[2];
        Self {
            resolution,
            world_half,
            grid: vec![0.0; n],
            scratch: vec![0.0; n],
        }
    }

    fn cell_size(&self, axis: usize) -> f32 {
        (2.0 * self.world_half[axis]) / self.resolution[axis] as f32
    }

    /// XY wraps toroidally; Z is bounded. Returns `None` when `pos.z` is
    /// outside the z-volume; XY positions always project into the grid.
    fn idx_of(&self, pos: [f32; 3]) -> Option<usize> {
        let cs_x = self.cell_size(0);
        let cs_y = self.cell_size(1);
        let cs_z = self.cell_size(2);
        let nx = self.resolution[0] as i32;
        let ny = self.resolution[1] as i32;
        let nz = self.resolution[2] as i32;
        let xi = ((pos[0] + self.world_half[0]) / cs_x).floor() as i32;
        let yi = ((pos[1] + self.world_half[1]) / cs_y).floor() as i32;
        let zi = ((pos[2] + self.world_half[2]) / cs_z).floor() as i32;
        if zi < 0 || zi >= nz {
            return None;
        }
        let xi_mod = xi.rem_euclid(nx) as usize;
        let yi_mod = yi.rem_euclid(ny) as usize;
        let nx_us = self.resolution[0];
        let ny_us = self.resolution[1];
        Some((zi as usize) * nx_us * ny_us + yi_mod * nx_us + xi_mod)
    }

    pub fn add_source(&mut self, pos: [f32; 3], amount: f32) {
        if let Some(idx) = self.idx_of(pos) {
            self.grid[idx] += amount;
        }
    }

    /// 7-point Jacobi stencil + multiplicative decay; stable for
    /// `diffusion < 1/6`. XY uses toroidal wrap (left at i=0 reads column
    /// i=nx-1, etc.); Z is Neumann zero-flux (z=0 and z=nz-1 fall back to
    /// the center plane — ground/ceiling, not wrap).
    ///
    /// Parallelized over z-planes via rayon: each plane reads its own xy
    /// stencil plus the back/front planes and writes only its slice of the
    /// scratch buffer, so there's no write conflict.
    ///
    /// Inner loop is SIMD via `wide::f32x8`. Per row (k, j) we pre-extract
    /// the row offsets for center/up/down/back/front (back/front with
    /// Neumann fallback to the current plane), then process 8-wide chunks
    /// over the interior `i ∈ [1, simd_end)`. Boundary cells (i=0 and
    /// i ∈ [simd_end, nx-1]) fall back to the scalar path — these are the
    /// only places where left/right wrap across the x-boundary is possible.
    /// The sequential `((((l+r)+u)+d)+b)+f` add chain is preserved so the
    /// SIMD result stays bit-identical with the scalar reference (no
    /// `reduce_add` reordering).
    pub fn step(&mut self, diffusion: f32, decay_per_sec: f32, dt: f32) {
        use wide::f32x8;
        let nx = self.resolution[0];
        let ny = self.resolution[1];
        let nz = self.resolution[2];
        let decay = (1.0 - decay_per_sec * dt).max(0.0);
        let plane = nx * ny;
        let grid = &self.grid;
        let diffusion_v = f32x8::splat(diffusion);
        let decay_v = f32x8::splat(decay);
        let six_v = f32x8::splat(6.0);
        // Largest multiple-of-8 + 1 such that i+7 ≤ nx-2 (right read at i+8
        // must stay ≤ nx-1). For nx=64 → simd_end = 57, covering chunks at
        // i = 1, 9, …, 49.
        let simd_end = if nx >= 9 {
            1 + ((nx - 9) / 8 + 1) * 8
        } else {
            1
        };
        self.scratch
            .par_chunks_mut(plane)
            .enumerate()
            .for_each(|(k, scratch_plane)| {
                let center_plane = k * plane;
                let back_plane = if k > 0 { (k - 1) * plane } else { center_plane };
                let front_plane = if k + 1 < nz {
                    (k + 1) * plane
                } else {
                    center_plane
                };
                for j in 0..ny {
                    let j_up = if j == 0 { ny - 1 } else { j - 1 };
                    let j_down = if j + 1 == ny { 0 } else { j + 1 };
                    let center_row = center_plane + j * nx;
                    let up_row = center_plane + j_up * nx;
                    let down_row = center_plane + j_down * nx;
                    let back_row = back_plane + j * nx;
                    let front_row = front_plane + j * nx;
                    let scalar_cell = |i: usize| -> f32 {
                        let i_left = if i == 0 { nx - 1 } else { i - 1 };
                        let i_right = if i + 1 == nx { 0 } else { i + 1 };
                        let center = grid[center_row + i];
                        let left = grid[center_row + i_left];
                        let right = grid[center_row + i_right];
                        let up = grid[up_row + i];
                        let down = grid[down_row + i];
                        let back = grid[back_row + i];
                        let front = grid[front_row + i];
                        let new = center
                            + diffusion * (left + right + up + down + back + front - 6.0 * center);
                        new * decay
                    };
                    scratch_plane[j * nx] = scalar_cell(0);
                    let mut i = 1;
                    while i < simd_end {
                        let center = f32x8::new(
                            grid[center_row + i..center_row + i + 8].try_into().unwrap(),
                        );
                        let left = f32x8::new(
                            grid[center_row + i - 1..center_row + i + 7]
                                .try_into()
                                .unwrap(),
                        );
                        let right = f32x8::new(
                            grid[center_row + i + 1..center_row + i + 9]
                                .try_into()
                                .unwrap(),
                        );
                        let up = f32x8::new(grid[up_row + i..up_row + i + 8].try_into().unwrap());
                        let down =
                            f32x8::new(grid[down_row + i..down_row + i + 8].try_into().unwrap());
                        let back =
                            f32x8::new(grid[back_row + i..back_row + i + 8].try_into().unwrap());
                        let front =
                            f32x8::new(grid[front_row + i..front_row + i + 8].try_into().unwrap());
                        let mut acc = left + right;
                        acc += up;
                        acc += down;
                        acc += back;
                        acc += front;
                        acc -= six_v * center;
                        let new = (center + diffusion_v * acc) * decay_v;
                        let arr: [f32; 8] = new.into();
                        scratch_plane[j * nx + i..j * nx + i + 8].copy_from_slice(&arr);
                        i += 8;
                    }
                    while i < nx {
                        scratch_plane[j * nx + i] = scalar_cell(i);
                        i += 1;
                    }
                }
            });
        std::mem::swap(&mut self.grid, &mut self.scratch);
    }

    /// Maze-mode diffusion step. `mask[idx] == true` marks Neumann-boundary
    /// (wall) voxels — they hold value 0 and any neighbor reading them in
    /// the stencil substitutes the center cell's own value instead (zero
    /// flux through walls). Scalar-only path; the SIMD fast path in `step`
    /// is bypassed because mask checks vary per-voxel. Caller pays one
    /// branch per voxel — acceptable since this only runs in maze mode.
    pub fn step_masked(&mut self, diffusion: f32, decay_per_sec: f32, dt: f32, mask: &[bool]) {
        debug_assert_eq!(mask.len(), self.grid.len());
        let nx = self.resolution[0];
        let ny = self.resolution[1];
        let nz = self.resolution[2];
        let plane = nx * ny;
        let decay = (1.0 - decay_per_sec * dt).max(0.0);
        let grid = &self.grid;
        self.scratch
            .par_chunks_mut(plane)
            .enumerate()
            .for_each(|(k, scratch_plane)| {
                let center_plane = k * plane;
                let back_plane = if k > 0 { (k - 1) * plane } else { center_plane };
                let front_plane = if k + 1 < nz {
                    (k + 1) * plane
                } else {
                    center_plane
                };
                for j in 0..ny {
                    let j_up = if j == 0 { ny - 1 } else { j - 1 };
                    let j_down = if j + 1 == ny { 0 } else { j + 1 };
                    for i in 0..nx {
                        let idx = center_plane + j * nx + i;
                        if mask[idx] {
                            scratch_plane[j * nx + i] = 0.0;
                            continue;
                        }
                        let i_left = if i == 0 { nx - 1 } else { i - 1 };
                        let i_right = if i + 1 == nx { 0 } else { i + 1 };
                        let center = grid[idx];
                        let read = |neighbor_idx: usize| -> f32 {
                            if mask[neighbor_idx] {
                                center
                            } else {
                                grid[neighbor_idx]
                            }
                        };
                        let left = read(center_plane + j * nx + i_left);
                        let right = read(center_plane + j * nx + i_right);
                        let up = read(center_plane + j_up * nx + i);
                        let down = read(center_plane + j_down * nx + i);
                        let back = read(back_plane + j * nx + i);
                        let front = read(front_plane + j * nx + i);
                        let new = center
                            + diffusion * (left + right + up + down + back + front - 6.0 * center);
                        scratch_plane[j * nx + i] = new * decay;
                    }
                }
            });
        std::mem::swap(&mut self.grid, &mut self.scratch);
    }

    pub fn sample(&self, pos: [f32; 3]) -> f32 {
        self.idx_of(pos).map(|i| self.grid[i]).unwrap_or(0.0)
    }

    pub fn grid_ref(&self) -> &[f32] {
        &self.grid
    }

    /// Overwrite grid contents from an external source. Used for the
    /// `FieldGpu` wire-up: GPU computes diffuse + deposit, downloads the
    /// snapshot, and the CPU `SmellField` holds it so `gradient_at` and
    /// `sample` keep working from the sensor stage.
    pub fn replace_grid_from(&mut self, data: &[f32]) {
        debug_assert_eq!(data.len(), self.grid.len());
        self.grid.copy_from_slice(data);
    }

    /// 3D central differences at `pos ± epsilon` along each axis. Returns
    /// `[d/dx, d/dy, d/dz]`. Out-of-bounds samples count as 0.
    pub fn gradient_at(&self, pos: [f32; 3], epsilon: f32) -> [f32; 3] {
        let f_xp = self.sample([pos[0] + epsilon, pos[1], pos[2]]);
        let f_xm = self.sample([pos[0] - epsilon, pos[1], pos[2]]);
        let f_yp = self.sample([pos[0], pos[1] + epsilon, pos[2]]);
        let f_ym = self.sample([pos[0], pos[1] - epsilon, pos[2]]);
        let f_zp = self.sample([pos[0], pos[1], pos[2] + epsilon]);
        let f_zm = self.sample([pos[0], pos[1], pos[2] - epsilon]);
        let inv = 1.0 / (2.0 * epsilon);
        [
            (f_xp - f_xm) * inv,
            (f_yp - f_ym) * inv,
            (f_zp - f_zm) * inv,
        ]
    }
}
