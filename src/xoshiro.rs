//! Pure-Rust port of the per-cell xoshiro128++ stream used in
//! `shaders/brownian.wgsl`. Byte-identical with the GPU dispatch when the
//! state is identical — that's the whole point of the port: CPU
//! `apply_brownian` and GPU brownian shader now share the same RNG
//! sequence per cell, so motor perturbations match across compute paths
//! and trajectories diverge only through brain-forward ULP drift (which is
//! orders of magnitude slower than thread-local `rand::rng()` drift).
//!
//! State expansion (`from_cell_id`) mirrors `CellsGpu::upload_xoshiro_seeds`
//! / `upload_xoshiro_seed_at`: SplitMix64 on `seed + 0x9E3779B97F4A7C15`,
//! split two u64s into four u32s, force non-zero. Seed is `cell_id` so
//! every cell — including children spawned mid-run — gets a deterministic,
//! reproducible stream tied to its stable identifier.

use serde::{Deserialize, Serialize};

/// 128-bit xoshiro state. The `xoshiro128++` algorithm requires at least
/// one non-zero word; the seeding helpers enforce that invariant.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Xoshiro128PlusPlus {
    pub state: [u32; 4],
}

impl Default for Xoshiro128PlusPlus {
    fn default() -> Self {
        // Sentinel non-zero seed. Real cells override via `from_cell_id` at
        // spawn / reproduce; this default exists only so `serde(default)` on
        // older checkpoints doesn't deserialize an all-zero (invalid) state.
        Self {
            state: [1, 0, 0, 0],
        }
    }
}

impl Xoshiro128PlusPlus {
    /// Derive a fresh state from a 64-bit seed. Mirrors the GPU upload path
    /// in `CellsGpu::upload_xoshiro_seeds` — SplitMix64 produces two u64s,
    /// each split into low/high u32, then the all-zero state is patched to
    /// `[1, 0, 0, 0]`. Keep this byte-identical with the GPU helper.
    #[inline]
    pub fn from_cell_id(cell_id: u64) -> Self {
        fn splitmix(z: &mut u64) -> u64 {
            *z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = *z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        }
        let mut z = cell_id.wrapping_add(0x9E3779B97F4A7C15);
        let a = splitmix(&mut z);
        let b = splitmix(&mut z);
        let mut s0 = a as u32;
        let s1 = (a >> 32) as u32;
        let s2 = b as u32;
        let s3 = (b >> 32) as u32;
        if s0 == 0 && s1 == 0 && s2 == 0 && s3 == 0 {
            s0 = 1;
        }
        Self {
            state: [s0, s1, s2, s3],
        }
    }

    /// One step of the xoshiro128++ generator. Verbatim port of the GPU
    /// shader's `xoshiro_next` (same op order, same rotl amounts).
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let s = self.state;
        let result = rotl_u32(s[0].wrapping_add(s[3]), 7).wrapping_add(s[0]);
        let t = s[1] << 9;
        let mut s2 = s[2] ^ s[0];
        let mut s3 = s[3] ^ s[1];
        let s1 = s[1] ^ s2;
        let s0 = s[0] ^ s3;
        s2 ^= t;
        s3 = rotl_u32(s3, 11);
        self.state = [s0, s1, s2, s3];
        result
    }

    /// `[0, 1)` via the IEEE-754 bit pattern trick used by the shader:
    /// build a float in `[1, 2)` with sign=0, exp=127, mantissa from the
    /// upper 23 bits of the rng output, then subtract 1. 23-bit precision.
    /// Bit-identical with `shaders/brownian.wgsl::uniform01`.
    #[inline]
    pub fn next_f32_uniform01(&mut self) -> f32 {
        let bits = self.next_u32();
        f32::from_bits((bits >> 9) | 0x3F800000) - 1.0
    }

    /// Box-Muller — one rng → one pair of independent N(0,1) gaussians.
    /// Caller uses both, or discards the second (mirroring the shader's
    /// z-axis path that consumes a pair but writes only one component).
    /// `EPSILON_CLAMP` matches the shader's `1.1920929e-7` guard against
    /// `log(0) = -inf`.
    #[inline]
    pub fn next_gaussian_pair(&mut self) -> (f32, f32) {
        const EPSILON_CLAMP: f32 = 1.1920929e-7;
        const TWO_PI: f32 = 6.283_185_307_179_586;
        let u1 = self.next_f32_uniform01().max(EPSILON_CLAMP);
        let u2 = self.next_f32_uniform01();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = TWO_PI * u2;
        (r * theta.cos(), r * theta.sin())
    }
}

#[inline]
fn rotl_u32(x: u32, k: u32) -> u32 {
    (x << k) | (x >> (32 - k))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_cell_id_avoids_all_zero() {
        // The seed expansion path patches an all-zero outcome to [1, 0, 0, 0].
        // Pick a cell_id we know upstream — there is no public way to hit the
        // patch path otherwise, since SplitMix64 of any non-zero u64 doesn't
        // reach 0. This is a guard test: if the magic constants ever change
        // and an input *did* yield 0, the patch keeps us out of the dead loop.
        let rng = Xoshiro128PlusPlus::from_cell_id(0xDEADBEEF);
        assert_ne!(rng.state, [0; 4]);
    }

    #[test]
    fn deterministic_stream_for_same_seed() {
        let mut a = Xoshiro128PlusPlus::from_cell_id(42);
        let mut b = Xoshiro128PlusPlus::from_cell_id(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn distinct_streams_for_distinct_seeds() {
        let mut a = Xoshiro128PlusPlus::from_cell_id(1);
        let mut b = Xoshiro128PlusPlus::from_cell_id(2);
        // Compare first 4 outputs — extremely unlikely to coincide unless
        // the seed expansion collapsed.
        let mut same_count = 0;
        for _ in 0..4 {
            if a.next_u32() == b.next_u32() {
                same_count += 1;
            }
        }
        assert!(same_count < 4, "neighbouring seeds produced identical stream");
    }

    #[test]
    fn uniform01_in_range() {
        let mut rng = Xoshiro128PlusPlus::from_cell_id(7);
        for _ in 0..10_000 {
            let x = rng.next_f32_uniform01();
            assert!(x >= 0.0 && x < 1.0, "uniform01 out of range: {x}");
        }
    }

    #[test]
    fn gaussian_pair_mean_near_zero() {
        // Quick sanity: 4000 pairs → 8000 samples, mean should be tight.
        let mut rng = Xoshiro128PlusPlus::from_cell_id(13);
        let mut sum = 0.0f64;
        let n = 4000;
        for _ in 0..n {
            let (a, b) = rng.next_gaussian_pair();
            sum += a as f64 + b as f64;
        }
        let mean = sum / (2 * n) as f64;
        assert!(mean.abs() < 0.05, "gaussian mean too biased: {mean}");
    }
}
