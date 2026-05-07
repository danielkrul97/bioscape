use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Sprint 53: 3D volumetric scalar field s explicit-Jacobi diffusion + decay.
/// Resolution per-axis (`[res_x, res_y, res_z]`) — typicky `[64, 64, 16]` aby
/// matchne aspect rátia tenkého z-sliceu (`world_half_z << world_half_xy`).
/// Grid layout: `idx = z*W*H + y*W + x`. 7-point stencil pro 3D Laplacian.
/// Stabilní při `diffusion < 1/6` (vs `< 1/4` v 2D — pre-Sprint-53 SmellField
/// měl 2D stencil). `SMELL_DIFFUSION = 0.15` zůstává pod oběma limity.
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

    /// Sprint 54: xy wrap (toroidal), z bounded. Mimo z-volume vrací `None`;
    /// xy je vždy modulo zarovnaný do gridu.
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

    /// 7-point Jacobi stencil + multiplicative decay. Sprint 54: toroidal v
    /// xy (left at i=0 čte sloupec i=nx-1, atd.), Neumann zero-flux na z
    /// (z=0 a z=nz-1 fallback na center — odpovídá ground/ceiling, ne wrap).
    /// Stable pro `diffusion < 1/6`.
    ///
    /// Sprint 57: paralelizováno přes z-roviny — každá rovina čte své okolí
    /// (xy stencil + back/front z grid) a zapisuje pouze do své části scratch,
    /// takže žádný write conflict. Pro 12-core CPU + 16 rovin je load balanced.
    ///
    /// Sprint 117: SIMD inner loop přes `wide::f32x8`. Per row (k, j) si
    /// pre-extract row offsets pro center/up/down/back/front (back/front s
    /// Neumann fallback na current plane), pak SIMD chunks po 8 buňkách na
    /// interior `i ∈ [1, nx-9]` (8 lanes × 7 chunks = 56 cells s nx=64).
    /// Boundary cells `i=0` a `i ∈ [nx-7, nx-1]` (8 z 64) scalar fallback —
    /// jediná místa, kde left/right wrap přes x-boundary. Sequential adds
    /// `(((l+r)+u)+d)+b)+f` → bit-identical s pre-S117 scalar verzí (žádný
    /// reduce_add); FP drift jen pokud nx<9.
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
        // SIMD pokrývá `i ∈ [1, simd_end)`, kde simd_end je největší
        // násobek 8 + 1 takový, že i+7 ≤ nx-2 (right read at i+8 ≤ nx-1).
        // Pro nx=64: simd_end = 1 + 7*8 = 57 → chunky i = 1, 9, …, 49.
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
                            + diffusion
                                * (left + right + up + down + back + front - 6.0 * center);
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
                        let up = f32x8::new(
                            grid[up_row + i..up_row + i + 8].try_into().unwrap(),
                        );
                        let down = f32x8::new(
                            grid[down_row + i..down_row + i + 8].try_into().unwrap(),
                        );
                        let back = f32x8::new(
                            grid[back_row + i..back_row + i + 8].try_into().unwrap(),
                        );
                        let front = f32x8::new(
                            grid[front_row + i..front_row + i + 8].try_into().unwrap(),
                        );
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

    pub fn sample(&self, pos: [f32; 3]) -> f32 {
        self.idx_of(pos).map(|i| self.grid[i]).unwrap_or(0.0)
    }

    pub fn grid_ref(&self) -> &[f32] {
        &self.grid
    }

    /// Sprint 59: replace grid contents from external source (GPU readback).
    /// Used for FieldGpu wire-up — GPU computes diffuse+deposit, downloads
    /// snapshot, CPU SmellField holds it pro sensor gather (`gradient_at` + `sample`).
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
