//! CPU O(N²) self-gravity reference. Used pre-S206 inside
//! `PlanetWorld::tick`, and as the unit-test reference once the GPU
//! shader lands (S206 verifies abs-error against this).
//!
//! Force: `a_i = G · Σ_{j≠i} m_j · (x_j − x_i) / (|x_j − x_i|² + ε²)^(3/2)`
//! (Plummer softening — see Aarseth, "Gravitational N-Body
//! Simulations", §2.2). The same ε is used in the potential to keep
//! energy diagnostics self-consistent.

use crate::planet::particle::Particles;

pub fn compute_acceleration(particles: &mut Particles, g: f32, softening: f32) {
    let n = particles.len();
    let eps2 = softening * softening;
    for i in 0..n {
        let xi = particles.positions[i];
        let mut ax = 0.0_f32;
        let mut ay = 0.0_f32;
        let mut az = 0.0_f32;
        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = particles.positions[j];
            let dx = xj[0] - xi[0];
            let dy = xj[1] - xi[1];
            let dz = xj[2] - xi[2];
            let r2 = dx * dx + dy * dy + dz * dz + eps2;
            let inv = 1.0 / (r2 * r2.sqrt());
            let f = g * particles.masses[j] * inv;
            ax += f * dx;
            ay += f * dy;
            az += f * dz;
        }
        particles.accelerations[i] = [ax, ay, az];
    }
}

/// Total gravitational potential energy with Plummer softening:
/// `U = -G/2 · Σ_{i ≠ j} m_i m_j / √(|r_ij|² + ε²)`.
///
/// O(N²). Parallelised over `i` with rayon, but summed deterministically:
/// each `i` reduces its own `j > i` tail sequentially, the per-`i` partials
/// are collected in index order, then folded in order — so the result is
/// bit-stable run-to-run (the diagnostic CSV stays reproducible) despite the
/// parallel map.
pub fn potential_energy(particles: &Particles, g: f32, softening: f32) -> f64 {
    use rayon::prelude::*;
    let n = particles.len();
    let eps2 = (softening * softening) as f64;
    let g = g as f64;
    let positions = &particles.positions;
    let masses = &particles.masses;
    let partials: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let xi = positions[i];
            let mi = masses[i] as f64;
            let mut ui = 0.0_f64;
            for j in (i + 1)..n {
                let xj = positions[j];
                let dx = (xj[0] - xi[0]) as f64;
                let dy = (xj[1] - xi[1]) as f64;
                let dz = (xj[2] - xi[2]) as f64;
                let r = (dx * dx + dy * dy + dz * dz + eps2).sqrt();
                ui -= g * mi * (masses[j] as f64) / r;
            }
            ui
        })
        .collect();
    partials.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_body_symmetric_force() {
        let mut p = Particles::with_capacity(2);
        p.push([-1.0, 0.0, 0.0], [0.0; 3], 1.0);
        p.push([1.0, 0.0, 0.0], [0.0; 3], 1.0);
        compute_acceleration(&mut p, 1.0, 0.0);
        // a_0 points toward +x, a_1 points toward -x, equal magnitude.
        assert!(p.accelerations[0][0] > 0.0);
        assert!(p.accelerations[1][0] < 0.0);
        assert!((p.accelerations[0][0] + p.accelerations[1][0]).abs() < 1e-6);
        assert!(p.accelerations[0][1].abs() < 1e-6);
        assert!(p.accelerations[0][2].abs() < 1e-6);
        // |a| = G·m/r² with r=2 ⇒ 1/4
        assert!((p.accelerations[0][0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn softened_zero_separation_finite() {
        let mut p = Particles::with_capacity(2);
        p.push([0.0, 0.0, 0.0], [0.0; 3], 1.0);
        p.push([0.0, 0.0, 0.0], [0.0; 3], 1.0);
        compute_acceleration(&mut p, 1.0, 0.1);
        for a in p.accelerations[0] {
            assert!(a.is_finite());
        }
    }

    #[test]
    fn potential_energy_two_body() {
        let mut p = Particles::with_capacity(2);
        p.push([-1.0, 0.0, 0.0], [0.0; 3], 1.0);
        p.push([1.0, 0.0, 0.0], [0.0; 3], 1.0);
        let u = potential_energy(&p, 1.0, 0.0);
        // -G·m₁m₂/r = -1·1·1/2 = -0.5
        assert!((u + 0.5).abs() < 1e-9, "u = {u}");
    }
}
