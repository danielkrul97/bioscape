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
    /// Sprint 207: extremes of the per-particle temperature distribution.
    /// `min_t` flags cold spots that may stall conduction; `max_t` flags
    /// hot anomalies that could drive runaway radiation.
    pub min_temperature: f32,
    pub max_temperature: f32,
    /// Sprint 230: mass-weighted mean solid fraction `Σ m φ / Σ m` and the
    /// mass fraction that is fully solid (`φ > 0.5`). Track the melt/freeze
    /// state of the body.
    pub mean_phase_frac: f64,
    pub solid_mass_frac: f64,
}

impl ScalarDiagnostics {
    pub fn compute(particles: &Particles) -> Self {
        let mut t_kin = 0.0_f64;
        let mut l_z = 0.0_f64;
        let mut m_sum = 0.0_f64;
        let mut u_sum = 0.0_f64;
        let mut t_min = f32::INFINITY;
        let mut t_max = f32::NEG_INFINITY;
        let mut t_sum = 0.0_f64;
        let mut phi_sum = 0.0_f64;
        let mut solid_sum = 0.0_f64;
        let have_u = particles.internal_energies.len() == particles.positions.len();
        let have_phi = particles.phase_fracs.len() == particles.positions.len();
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
                let u_i = particles.internal_energies[i];
                u_sum += m64 * u_i as f64;
                // Sensible temperature via the enthalpy map so the melt
                // plateau is reflected in min/max/mean T (S223).
                let t_i = crate::planet::thermal::sensible_temperature_of(u_i);
                t_sum += m64 * t_i as f64;
                if t_i < t_min { t_min = t_i; }
                if t_i > t_max { t_max = t_i; }
            }
            if have_phi {
                let phi = particles.phase_fracs[i] as f64;
                phi_sum += m64 * phi;
                if phi > 0.5 {
                    solid_sum += m64;
                }
            }
        }
        if !have_u {
            t_min = 0.0;
            t_max = 0.0;
        }
        let mean_t = if m_sum > 0.0 { t_sum / m_sum } else { 0.0 };
        let mean_phi = if m_sum > 0.0 { phi_sum / m_sum } else { 0.0 };
        let solid_frac = if m_sum > 0.0 { solid_sum / m_sum } else { 0.0 };
        Self {
            total_mass: m_sum,
            kinetic_energy: t_kin,
            angular_momentum_z: l_z,
            internal_energy: u_sum,
            mean_temperature: mean_t,
            min_temperature: t_min,
            max_temperature: t_max,
            mean_phase_frac: mean_phi,
            solid_mass_frac: solid_frac,
        }
    }
}

/// Count emergent solid "blocks": connected components of the
/// solid-fraction graph. Nodes are particles with `φ > 0.5`; two solid
/// particles are linked when within `link_factor · ½(h_i + h_j)`. Returns
/// `(n_blocks, largest_block_size)`. O(N²) — for diagnostic cadence, not
/// the hot loop.
pub fn count_solid_blocks(particles: &Particles, link_factor: f32) -> (usize, usize) {
    let n = particles.positions.len();
    if particles.phase_fracs.len() != n {
        return (0, 0);
    }
    let solid: Vec<usize> = (0..n).filter(|&i| particles.phase_fracs[i] > 0.5).collect();
    let m = solid.len();
    if m == 0 {
        return (0, 0);
    }
    // Find linked (a, b) pairs in parallel — each `a` scans its `b > a` tail.
    // Collected in `a` order; union-find then runs sequentially. Connected-
    // component sizes are invariant to union order, so the result is
    // deterministic regardless of how rayon schedules the search.
    use rayon::prelude::*;
    let solid_ref = &solid;
    let positions = &particles.positions;
    let h = &particles.smoothing_lengths;
    let pairs: Vec<(usize, usize)> = (0..m)
        .into_par_iter()
        .flat_map_iter(|a| {
            let i = solid_ref[a];
            let xi = positions[i];
            let hi = h[i];
            (a + 1..m).filter_map(move |b| {
                let j = solid_ref[b];
                let xj = positions[j];
                let hj = h[j];
                let dx = xi[0] - xj[0];
                let dy = xi[1] - xj[1];
                let dz = xi[2] - xj[2];
                let r2 = dx * dx + dy * dy + dz * dz;
                let link = link_factor * 0.5 * (hi + hj);
                if r2 <= link * link { Some((a, b)) } else { None }
            })
        })
        .collect();

    let mut parent: Vec<usize> = (0..m).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for (a, b) in pairs {
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }
    let mut sizes = std::collections::HashMap::new();
    for a in 0..m {
        let r = find(&mut parent, a);
        *sizes.entry(r).or_insert(0usize) += 1;
    }
    let largest = sizes.values().copied().max().unwrap_or(0);
    (sizes.len(), largest)
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
///   dt_i = C · h_i / (c_eff_i + |v_i|)
/// reduced to the global minimum. `c_eff` is the phase-selected sound
/// speed (`thermal::sound_speed_of`): ideal-gas `√(γ(γ−1)u)` above the
/// vaporisation energy, condensed Tait `c0√n·(ρ/ρ0)^((n−1)/2)` below.
/// The condensed branch is the S224 fix — the old gas-only estimate
/// reported a falsely huge safe dt for cold low-`u` solids (`√(γ(γ−1)u)`
/// → 0), exactly the stiffest, least stable particles.
pub fn cfl_dt(particles: &Particles, eos_gamma: f32, rho0: f32, c_courant: f32) -> f32 {
    let have_u = particles.internal_energies.len() == particles.positions.len();
    let have_rho = particles.densities.len() == particles.positions.len();
    let mut dt_min = f32::INFINITY;
    for (i, (v, &h)) in particles
        .velocities
        .iter()
        .zip(&particles.smoothing_lengths)
        .enumerate()
    {
        let v_mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let u = if have_u {
            particles.internal_energies[i].max(crate::planet::thermal::U_MIN)
        } else {
            crate::planet::thermal::INITIAL_INTERNAL_ENERGY
        };
        let rho = if have_rho {
            particles.densities[i].max(1e-30)
        } else {
            rho0
        };
        let c_s = crate::planet::thermal::elastic_sound_speed_of(u, rho, eos_gamma, rho0);
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
