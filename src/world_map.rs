use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Sprint 53: deterministic 3D volumetric scalar field. `resolution` per axis,
/// hodnoty v `[0, 1]` z value-noise: `base_resolution³` random uniform grid,
/// smoothstep trilinear interp do plné resolution. Generováno jednou při
/// startu, pak jen čtení — žádný update per tick.
///
/// Use case: prostorová modulace mechaniky, která má být 3D-nehomogenní —
/// food_richness (xy projekce stačí pro food spawn floor), hazard (3D field
/// pro vertikální hazard layers), terrain drag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldMap {
    pub resolution: [usize; 3],
    pub world_half: [f32; 3],
    field: Vec<f32>,
}

impl WorldMap {
    pub fn new(
        resolution: [usize; 3],
        base_resolution: [usize; 3],
        world_half: [f32; 3],
        seed: u64,
    ) -> Self {
        assert!(resolution.iter().all(|&r| r >= 2) && base_resolution.iter().all(|&r| r >= 2));
        let mut rng = StdRng::seed_from_u64(seed);
        let base_n = base_resolution[0] * base_resolution[1] * base_resolution[2];
        let base: Vec<f32> = (0..base_n).map(|_| rng.random()).collect();

        let nx = resolution[0];
        let ny = resolution[1];
        let nz = resolution[2];
        let bnx = base_resolution[0];
        let bny = base_resolution[1];
        let bnz = base_resolution[2];
        let scale_x = (bnx as f32 - 1.0) / nx as f32;
        let scale_y = (bny as f32 - 1.0) / ny as f32;
        let scale_z = (bnz as f32 - 1.0) / nz as f32;

        let mut field = vec![0.0_f32; nx * ny * nz];
        for k in 0..nz {
            let w = (k as f32 + 0.5) * scale_z;
            let z0 = (w.floor() as usize).min(bnz - 1);
            let z1 = (z0 + 1).min(bnz - 1);
            let fz = (w - z0 as f32).clamp(0.0, 1.0);
            let sz = fz * fz * (3.0 - 2.0 * fz);
            for j in 0..ny {
                let v = (j as f32 + 0.5) * scale_y;
                let y0 = (v.floor() as usize).min(bny - 1);
                let y1 = (y0 + 1).min(bny - 1);
                let fy = (v - y0 as f32).clamp(0.0, 1.0);
                let sy = fy * fy * (3.0 - 2.0 * fy);
                for i in 0..nx {
                    let u = (i as f32 + 0.5) * scale_x;
                    let x0 = (u.floor() as usize).min(bnx - 1);
                    let x1 = (x0 + 1).min(bnx - 1);
                    let fx = (u - x0 as f32).clamp(0.0, 1.0);
                    let sx = fx * fx * (3.0 - 2.0 * fx);
                    // Trilinear interp s smoothstep blend.
                    let i000 = base[z0 * bnx * bny + y0 * bnx + x0];
                    let i100 = base[z0 * bnx * bny + y0 * bnx + x1];
                    let i010 = base[z0 * bnx * bny + y1 * bnx + x0];
                    let i110 = base[z0 * bnx * bny + y1 * bnx + x1];
                    let i001 = base[z1 * bnx * bny + y0 * bnx + x0];
                    let i101 = base[z1 * bnx * bny + y0 * bnx + x1];
                    let i011 = base[z1 * bnx * bny + y1 * bnx + x0];
                    let i111 = base[z1 * bnx * bny + y1 * bnx + x1];
                    let c00 = i000 * (1.0 - sx) + i100 * sx;
                    let c10 = i010 * (1.0 - sx) + i110 * sx;
                    let c01 = i001 * (1.0 - sx) + i101 * sx;
                    let c11 = i011 * (1.0 - sx) + i111 * sx;
                    let c0 = c00 * (1.0 - sy) + c10 * sy;
                    let c1 = c01 * (1.0 - sy) + c11 * sy;
                    field[k * nx * ny + j * nx + i] = c0 * (1.0 - sz) + c1 * sz;
                }
            }
        }

        Self {
            resolution,
            world_half,
            field,
        }
    }

    /// Sprint 54: xy wrap (toroidal), z clamp (bounded).
    pub fn sample(&self, pos: [f32; 3]) -> f32 {
        let nx = self.resolution[0];
        let ny = self.resolution[1];
        let nz = self.resolution[2];
        let cs_x = (2.0 * self.world_half[0]) / nx as f32;
        let cs_y = (2.0 * self.world_half[1]) / ny as f32;
        let cs_z = (2.0 * self.world_half[2]) / nz as f32;
        let xi = ((pos[0] + self.world_half[0]) / cs_x).floor() as i32;
        let yi = ((pos[1] + self.world_half[1]) / cs_y).floor() as i32;
        let zi = ((pos[2] + self.world_half[2]) / cs_z).floor() as i32;
        let xi = xi.rem_euclid(nx as i32) as usize;
        let yi = yi.rem_euclid(ny as i32) as usize;
        let zi = zi.clamp(0, nz as i32 - 1) as usize;
        self.field[zi * nx * ny + yi * nx + xi]
    }

    pub fn field(&self) -> &[f32] {
        &self.field
    }
}
