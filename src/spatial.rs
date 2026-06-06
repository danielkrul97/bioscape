/// Sprint 54: minimum-image displacement na toroidal xy + bounded z.
/// Vrátí signed delta `b - a` adjustnuté tak, že |dx|, |dy| ≤ `world_half`,
/// dz beze změny (z-osa není wrapped — gravita + food sink + carrion drop
/// vyžadují pevný strop/dno).
///
/// Pro toroidal world: dva body na opačných koncích světa (např. x=−950 a
/// x=+950 při half=960) jsou minimum-image-blízké (Δx=20, ne 1900).
pub fn min_image_delta(a: [f32; 3], b: [f32; 3], world_half: [f32; 3]) -> [f32; 3] {
    let mut dx = b[0] - a[0];
    let mut dy = b[1] - a[1];
    let dz = b[2] - a[2];
    let wx = 2.0 * world_half[0];
    let wy = 2.0 * world_half[1];
    if dx > world_half[0] {
        dx -= wx;
    } else if dx < -world_half[0] {
        dx += wx;
    }
    if dy > world_half[1] {
        dy -= wy;
    } else if dy < -world_half[1] {
        dy += wy;
    }
    [dx, dy, dz]
}

/// Sprint 54: wrap pos.xy do `[-half, half)` (toroidal). z se nepojí.
pub fn wrap_position_xy(pos: [f32; 3], world_half: [f32; 3]) -> [f32; 3] {
    let wx = 2.0 * world_half[0];
    let wy = 2.0 * world_half[1];
    let mut x = pos[0];
    let mut y = pos[1];
    while x >= world_half[0] {
        x -= wx;
    }
    while x < -world_half[0] {
        x += wx;
    }
    while y >= world_half[1] {
        y -= wy;
    }
    while y < -world_half[1] {
        y += wy;
    }
    [x, y, pos[2]]
}

/// Sprint 43: 3D uniform spatial hash. Generic přes `Id` (Bevy `Entity` v
/// rendereru, `usize` v headless) a `P` (per-item payload, např. radius).
///
/// Storage je flat `Vec<Vec<…>>` indexovaný `bx + by*nx + bz*nx*ny`; bucket
/// počty jsou odvozené z `world_half` + `cell_size` při konstrukci. xy je
/// toroidal (modulo wrap), z bounded (`bz` clamped na `[0, nz)`). Pro typický
/// svět (1920×1080×200, cs=64) máme 30×17×4 ≈ 2 040 bucketů — flat Vec
/// vyhrává nad `FxHashMap` per-bucket lookup tým, že se vyhne hashing/
/// reprobing.
///
/// **Determinismus:** rebuild iteruje vstup v pořadí, ve kterém přijde, a Vec
/// v každém bucketu drží push-order. `for_each_in_radius` iteruje (dx,dy,dz)
/// ve fixním pořadí. Caller, který předá rebuild items ve stable order
/// (např. `cells.iter().enumerate()`), dostane reprodukovatelný traversal.
pub struct SpatialGrid<Id: Copy, P: Copy> {
    cell_size: f32,
    nx: i32,
    ny: i32,
    nz: i32,
    buckets: Vec<Vec<(Id, [f32; 3], P)>>,
}

impl<Id: Copy, P: Copy> SpatialGrid<Id, P> {
    /// Build a grid sized to wrap a world of `±world_half[i]` along each axis.
    /// `world_half[2] == 0.0` collapses z to a single bucket (2D mode).
    pub fn new(cell_size: f32, world_half: [f32; 3]) -> Self {
        let nx = ((2.0 * world_half[0] / cell_size).ceil() as i32).max(1);
        let ny = ((2.0 * world_half[1] / cell_size).ceil() as i32).max(1);
        let nz = if world_half[2] > 0.0 {
            ((2.0 * world_half[2] / cell_size).ceil() as i32).max(1)
        } else {
            1
        };
        let total = (nx * ny * nz) as usize;
        Self {
            cell_size,
            nx,
            ny,
            nz,
            buckets: (0..total).map(|_| Vec::new()).collect(),
        }
    }

    /// Toroidal-wrap a position component into `[0, n)`. Floor-divide makes
    /// negative coords land in the correct (positive) bucket.
    #[inline]
    fn axis_bucket(pos_axis: f32, cell_size: f32, n: i32) -> i32 {
        let raw = (pos_axis / cell_size).floor() as i32 + n / 2;
        ((raw % n) + n) % n
    }

    #[inline]
    fn bucket_idx(&self, bx: i32, by: i32, bz: i32) -> usize {
        debug_assert!(bx >= 0 && bx < self.nx);
        debug_assert!(by >= 0 && by < self.ny);
        debug_assert!(bz >= 0 && bz < self.nz);
        (bx + by * self.nx + bz * self.nx * self.ny) as usize
    }

    /// Drops stale entries z předchozího rebuildu, ale zachová bucket Vec
    /// kapacity — populace je per-tick relativně stabilní, takže reuse alokace
    /// vyhrává nad realokací.
    pub fn rebuild<I: IntoIterator<Item = (Id, [f32; 3], P)>>(&mut self, items: I) {
        for bucket in self.buckets.iter_mut() {
            bucket.clear();
        }
        for (id, pos, payload) in items {
            let bx = Self::axis_bucket(pos[0], self.cell_size, self.nx);
            let by = Self::axis_bucket(pos[1], self.cell_size, self.ny);
            let bz = if self.nz == 1 {
                0
            } else {
                let raw = (pos[2] / self.cell_size).floor() as i32 + self.nz / 2;
                raw.clamp(0, self.nz - 1)
            };
            let idx = self.bucket_idx(bx, by, bz);
            self.buckets[idx].push((id, pos, payload));
        }
    }

    /// Volá `f(id, pos, payload)` pro každý item v 3³ buckets okolo `pos`.
    /// Caller musí narrow-phase distance test dělat sám (grid vrací overestimate).
    /// Bucket walk auto-wraps v xy (toroidal); v z clampuje, takže items mimo
    /// vertical range se nikdy nevidí.
    pub fn for_each_in_radius<F: FnMut(Id, [f32; 3], P)>(
        &self,
        pos: [f32; 3],
        radius: f32,
        mut f: F,
    ) {
        let r_cells = (radius / self.cell_size).ceil() as i32;
        let cx_raw = (pos[0] / self.cell_size).floor() as i32 + self.nx / 2;
        let cy_raw = (pos[1] / self.cell_size).floor() as i32 + self.ny / 2;
        let cz_raw = if self.nz == 1 {
            0
        } else {
            (pos[2] / self.cell_size).floor() as i32 + self.nz / 2
        };
        for dz in -r_cells..=r_cells {
            let bz_raw = cz_raw + dz;
            if bz_raw < 0 || bz_raw >= self.nz {
                continue;
            }
            for dy in -r_cells..=r_cells {
                let by = (((cy_raw + dy) % self.ny) + self.ny) % self.ny;
                for dx in -r_cells..=r_cells {
                    let bx = (((cx_raw + dx) % self.nx) + self.nx) % self.nx;
                    let idx = self.bucket_idx(bx, by, bz_raw);
                    for &(id, p, payload) in &self.buckets[idx] {
                        f(id, p, payload);
                    }
                }
            }
        }
    }

    /// Toroidal query — kept as a thin wrapper for backward compatibility.
    /// `for_each_in_radius` already wraps xy internally now (the grid is
    /// bounded), so the explicit ghost-position walk that the FxHashMap
    /// version needed is redundant.
    pub fn for_each_in_radius_toroidal<F: FnMut(Id, [f32; 3], P)>(
        &self,
        pos: [f32; 3],
        radius: f32,
        _world_half: [f32; 3],
        f: F,
    ) {
        self.for_each_in_radius(pos, radius, f);
    }
}

/// Sprint 43: defaultní velikost buňky spatial gridu. ~1.3× max vision_radius
/// (50). Větší = méně buckets, víc kandidátů per query; menší = víc buckets,
/// méně kandidátů. Renderer v `main.rs` má svůj vlastní knob.
pub const GRID_CELL_SIZE: f32 = 64.0;
