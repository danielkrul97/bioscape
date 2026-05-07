use std::hash::Hash;

use rustc_hash::FxHashMap;

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
/// **Determinismus:** rebuild iteruje vstup v pořadí, ve kterém přijde, a Vec
/// v každém bucketu drží push-order. `for_each_in_radius` iteruje 3³ buckets ve
/// fixním (dx, dy, dz) pořadí; `HashMap::get(&key)` je lookup-deterministic.
/// Caller, který předá rebuild items ve stable order (např. `cells.iter().enumerate()`),
/// dostane reprodukovatelný traversal napříč runy. Floats z následných sumací
/// nejsou bit-identical s O(N²) baseline kvůli jinému pořadí akumulace.
pub struct SpatialGrid<Id: Copy + Eq + Hash, P: Copy> {
    cell_size: f32,
    buckets: FxHashMap<(i32, i32, i32), Vec<(Id, [f32; 3], P)>>,
}

impl<Id: Copy + Eq + Hash, P: Copy> SpatialGrid<Id, P> {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            buckets: FxHashMap::default(),
        }
    }

    fn key_of(&self, pos: [f32; 3]) -> (i32, i32, i32) {
        (
            (pos[0] / self.cell_size).floor() as i32,
            (pos[1] / self.cell_size).floor() as i32,
            (pos[2] / self.cell_size).floor() as i32,
        )
    }

    /// Drops stale entries z předchozího rebuildu, ale zachová bucket Vec
    /// kapacity — populace je per-tick relativně stabilní, takže reuse alokace
    /// vyhrává nad clear() celé HashMap.
    pub fn rebuild<I: IntoIterator<Item = (Id, [f32; 3], P)>>(&mut self, items: I) {
        for bucket in self.buckets.values_mut() {
            bucket.clear();
        }
        for (id, pos, payload) in items {
            let key = self.key_of(pos);
            self.buckets.entry(key).or_default().push((id, pos, payload));
        }
    }

    /// Volá `f(id, pos, payload)` pro každý item v 3³ buckets okolo `pos`.
    /// Caller musí narrow-phase distance test dělat sám (grid vrací overestimate).
    pub fn for_each_in_radius<F: FnMut(Id, [f32; 3], P)>(
        &self,
        pos: [f32; 3],
        radius: f32,
        mut f: F,
    ) {
        let r_cells = (radius / self.cell_size).ceil() as i32;
        let (cx, cy, cz) = self.key_of(pos);
        for dx in -r_cells..=r_cells {
            for dy in -r_cells..=r_cells {
                for dz in -r_cells..=r_cells {
                    if let Some(bucket) = self.buckets.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &(id, p, payload) in bucket {
                            f(id, p, payload);
                        }
                    }
                }
            }
        }
    }

    /// Sprint 54: toroidal-aware query přes ghost positions. Pokud je `pos`
    /// blízko xy-boundary (do `radius`), vyšleme dodatečné lookup queries do
    /// "ghost" pozic na opačné straně světa. Z není wrapped (cylinder topology).
    /// Stejný `f` callback se může volat na duplicate items pokud je radius
    /// > world_half — caller musí narrow-phase použít `min_image_delta` aby
    /// duplicates filtroval.
    pub fn for_each_in_radius_toroidal<F: FnMut(Id, [f32; 3], P)>(
        &self,
        pos: [f32; 3],
        radius: f32,
        world_half: [f32; 3],
        mut f: F,
    ) {
        // Center query.
        self.for_each_in_radius(pos, radius, &mut f);
        let wx = 2.0 * world_half[0];
        let wy = 2.0 * world_half[1];
        let near_left = pos[0] < -world_half[0] + radius;
        let near_right = pos[0] > world_half[0] - radius;
        let near_bot = pos[1] < -world_half[1] + radius;
        let near_top = pos[1] > world_half[1] - radius;
        // Edges (4 ghost positions).
        if near_left {
            self.for_each_in_radius([pos[0] + wx, pos[1], pos[2]], radius, &mut f);
        }
        if near_right {
            self.for_each_in_radius([pos[0] - wx, pos[1], pos[2]], radius, &mut f);
        }
        if near_bot {
            self.for_each_in_radius([pos[0], pos[1] + wy, pos[2]], radius, &mut f);
        }
        if near_top {
            self.for_each_in_radius([pos[0], pos[1] - wy, pos[2]], radius, &mut f);
        }
        // Corners (4 ghost positions).
        if near_left && near_bot {
            self.for_each_in_radius([pos[0] + wx, pos[1] + wy, pos[2]], radius, &mut f);
        }
        if near_left && near_top {
            self.for_each_in_radius([pos[0] + wx, pos[1] - wy, pos[2]], radius, &mut f);
        }
        if near_right && near_bot {
            self.for_each_in_radius([pos[0] - wx, pos[1] + wy, pos[2]], radius, &mut f);
        }
        if near_right && near_top {
            self.for_each_in_radius([pos[0] - wx, pos[1] - wy, pos[2]], radius, &mut f);
        }
    }
}

/// Sprint 43: defaultní velikost buňky spatial gridu. ~1.3× max vision_radius
/// (50). Větší = méně buckets, víc kandidátů per query; menší = víc buckets,
/// méně kandidátů. Renderer v `main.rs` má svůj vlastní knob.
pub const GRID_CELL_SIZE: f32 = 64.0;

/// Sprint 102: hunter cell-grid bucket size. Hunter vision_radius je řádově
/// větší než typická cell-cell interakce (100–400 vs ~20), takže `GRID_CELL_SIZE
/// = 64` by dělalo r_cells = 5–7 → 1300+ HashMap lookupů per query a grid
/// by byl pomalejší než brute force při běžné populaci. 200 odpovídá median
/// hunter vision → r_cells = 1–2 → 27–125 lookupů.
pub const HUNTER_GRID_CELL_SIZE: f32 = 200.0;
