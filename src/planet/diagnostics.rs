//! Stability metrics for the torus planet experiment.
//!
//! Live on the CPU side; the GPU pipelines do not include reduction
//! kernels for these yet (S213+ may add `planet_inertia.wgsl` once
//! the per-tick download cost shows up on the profiler). For 100k
//! particles at 100 Hz a position+velocity readback is ~2.4 MB →
//! well within PCIe budget for sub-second diagnostic cadence.

use crate::planet::gravity_cpu::potential_energy;
use crate::planet::particle::Particles;

#[derive(Debug, Clone, Copy, Default)]
pub struct ScalarDiagnostics {
    pub total_mass: f64,
    pub kinetic_energy: f64,
    pub angular_momentum_z: f64,
    /// Sprint 202: total internal energy `Σ m_i · u_i`. Combined with
    /// kinetic + potential gives the closed-system energy that conduction
    /// + viscous heating must conserve (radiation is a sink).
    pub internal_energy: f64,
    /// Sprint 202: mass-weighted mean temperature. Equals
    /// `Σ m_i u_i / (Σ m_i · c_v)`.
    pub mean_temperature: f64,
}

impl ScalarDiagnostics {
    pub fn compute(particles: &Particles) -> Self {
        let mut t_kin = 0.0_f64;
        let mut l_z = 0.0_f64;
        let mut m_sum = 0.0_f64;
        let mut u_sum = 0.0_f64;
        let have_u = particles.internal_energies.len() == particles.positions.len();
        for (i, ((p, v), &m)) in particles
            .positions
            .iter()
            .zip(&particles.velocities)
            .zip(&particles.masses)
            .enumerate()
        {
            let m64 = m as f64;
            let vx = v[0] as f64;
            let vy = v[1] as f64;
            let vz = v[2] as f64;
            t_kin += 0.5 * m64 * (vx * vx + vy * vy + vz * vz);
            l_z += m64 * (p[0] as f64 * vy - p[1] as f64 * vx);
            m_sum += m64;
            if have_u {
                u_sum += m64 * particles.internal_energies[i] as f64;
            }
        }
        let cv = crate::planet::thermal::HEAT_CAPACITY_CV as f64;
        let mean_t = if m_sum > 0.0 && cv > 0.0 {
            u_sum / (m_sum * cv)
        } else {
            0.0
        };
        Self {
            total_mass: m_sum,
            kinetic_energy: t_kin,
            angular_momentum_z: l_z,
            internal_energy: u_sum,
            mean_temperature: mean_t,
        }
    }
}

/// Symmetric 3×3 inertia tensor `I_ab = Σ m (r² δ_ab − r_a r_b)`.
/// Returns `[I_xx, I_yy, I_zz, I_xy, I_xz, I_yz]`.
pub fn inertia_tensor(particles: &Particles) -> [f64; 6] {
    let mut i_xx = 0.0_f64;
    let mut i_yy = 0.0_f64;
    let mut i_zz = 0.0_f64;
    let mut i_xy = 0.0_f64;
    let mut i_xz = 0.0_f64;
    let mut i_yz = 0.0_f64;
    for (pos, &m) in particles.positions.iter().zip(&particles.masses) {
        let (x, y, z, mm) = (pos[0] as f64, pos[1] as f64, pos[2] as f64, m as f64);
        let r2 = x * x + y * y + z * z;
        i_xx += mm * (r2 - x * x);
        i_yy += mm * (r2 - y * y);
        i_zz += mm * (r2 - z * z);
        i_xy += mm * (-x * y);
        i_xz += mm * (-x * z);
        i_yz += mm * (-y * z);
    }
    [i_xx, i_yy, i_zz, i_xy, i_xz, i_yz]
}

/// Principal moments of inertia — eigenvalues of the symmetric 3×3
/// inertia tensor obtained via cyclic Jacobi rotations. Returned
/// sorted descending. Convergence is typically < 10 sweeps for a
/// physical inertia tensor.
pub fn principal_moments(particles: &Particles) -> [f64; 3] {
    let [i_xx, i_yy, i_zz, i_xy, i_xz, i_yz] = inertia_tensor(particles);
    let mut m = [
        [i_xx, i_xy, i_xz],
        [i_xy, i_yy, i_yz],
        [i_xz, i_yz, i_zz],
    ];
    for _sweep in 0..30 {
        let mut off = 0.0;
        for a in 0..3 {
            for b in a + 1..3 {
                off += m[a][b].abs();
            }
        }
        if off < 1e-12 {
            break;
        }
        for a in 0..3 {
            for b in a + 1..3 {
                if m[a][b].abs() < 1e-15 {
                    continue;
                }
                let theta = (m[b][b] - m[a][a]) / (2.0 * m[a][b]);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let m_aa = m[a][a];
                let m_bb = m[b][b];
                let m_ab = m[a][b];
                m[a][a] = m_aa - t * m_ab;
                m[b][b] = m_bb + t * m_ab;
                m[a][b] = 0.0;
                m[b][a] = 0.0;
                for k in 0..3 {
                    if k != a && k != b {
                        let m_ak = m[a][k];
                        let m_bk = m[b][k];
                        m[a][k] = c * m_ak - s * m_bk;
                        m[b][k] = s * m_ak + c * m_bk;
                        m[k][a] = m[a][k];
                        m[k][b] = m[b][k];
                    }
                }
            }
        }
    }
    let mut v = [m[0][0], m[1][1], m[2][2]];
    v.sort_by(|a, b| b.total_cmp(a));
    v
}

/// Kinetic + potential energy from the current particle state.
/// Uses the same Plummer-softened pair sum as the GPU gravity for
/// consistency, but in f64 — slower but suitable for end-of-tick
/// diagnostic cadence, not every step.
pub fn total_energy(particles: &Particles, g: f32, softening: f32) -> (f64, f64, f64) {
    let mut t = 0.0_f64;
    for (v, &m) in particles.velocities.iter().zip(&particles.masses) {
        let m64 = m as f64;
        let v2 = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) as f64;
        t += 0.5 * m64 * v2;
    }
    let u = potential_energy(particles, g, softening);
    (t, u, t + u)
}

/// CFL-stable timestep estimate. For each particle:
///   dt_i = C · h_i / (c_s_i + |v_i|)
/// Reduces to the global minimum. `c_s = √(γ K ρ^(γ-1))`.
pub fn cfl_dt(particles: &Particles, eos_k: f32, eos_gamma: f32, c_courant: f32) -> f32 {
    let mut dt_min = f32::INFINITY;
    for ((v, &h), &rho) in particles
        .velocities
        .iter()
        .zip(&particles.smoothing_lengths)
        .zip(&particles.densities)
    {
        let v_mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let c_s = (eos_gamma * eos_k * rho.max(1e-30).powf(eos_gamma - 1.0)).sqrt();
        let denom = c_s + v_mag;
        if denom > 0.0 {
            let dt = c_courant * h / denom;
            if dt < dt_min {
                dt_min = dt;
            }
        }
    }
    if dt_min.is_finite() {
        dt_min
    } else {
        c_courant
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::init::torus_uniform;
    use crate::planet::world::PlanetConfig;

    #[test]
    fn principal_moments_axis_aligned_torus() {
        let cfg = PlanetConfig {
            n_particles: 20_000,
            r_major: 1.0,
            r_minor: 0.2,
            total_mass: 1.0,
            seed: 7,
            ..PlanetConfig::default()
        };
        let p = torus_uniform(&cfg);
        let mom = principal_moments(&p);
        let m = cfg.total_mass as f64;
        let r = cfg.r_major as f64;
        let a = cfg.r_minor as f64;
        let i_zz_expected = m * (r * r + 0.75 * a * a);
        let i_xx_expected = m * (0.5 * r * r + 0.625 * a * a);
        assert!((mom[0] - i_zz_expected).abs() / i_zz_expected < 0.02);
        assert!((mom[1] - i_xx_expected).abs() / i_xx_expected < 0.02);
        assert!((mom[2] - i_xx_expected).abs() / i_xx_expected < 0.02);
    }
}
