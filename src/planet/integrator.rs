//! Symplectic leapfrog integrator (KDK — kick-drift-kick).
//!
//! Why leapfrog: energy error is bounded and oscillatory rather than
//! secular, which makes it the standard choice for long-running
//! orbital integrations. Explicit Euler (the integrator in the biology
//! sim `shaders/step.wgsl`) accumulates energy and is unusable for
//! Kepler-class problems on the timescales we care about here.
//!
//! Convention: state `(x_n, v_n, a_n)` stays synchronous in time.
//! One step:
//!   1. `v ← v + (dt/2) a_old`
//!   2. `x ← x + dt v`
//!   3. recompute `a` at new `x`
//!   4. `v ← v + (dt/2) a_new`

use crate::planet::particle::Particles;

/// One full KDK leapfrog step. The caller supplies a closure that
/// refreshes `particles.accelerations` from the new positions.
pub fn leapfrog_step<F: FnMut(&mut Particles)>(
    particles: &mut Particles,
    dt: f32,
    mut recompute_acc: F,
) {
    let n = particles.len();
    let half = 0.5 * dt;
    for i in 0..n {
        for d in 0..3 {
            particles.velocities[i][d] += half * particles.accelerations[i][d];
        }
    }
    for i in 0..n {
        for d in 0..3 {
            particles.positions[i][d] += dt * particles.velocities[i][d];
        }
    }
    recompute_acc(particles);
    for i in 0..n {
        for d in 0..3 {
            particles.velocities[i][d] += half * particles.accelerations[i][d];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    /// Analytic 1/r² acceleration toward origin for a fixed central
    /// mass `GM = 1`. `eps² = 0` — no softening for the integrator-only
    /// test.
    fn kepler_acc(pos: [f32; 3]) -> [f32; 3] {
        let r2 = pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2];
        let r = r2.sqrt();
        let inv_r3 = 1.0 / (r2 * r);
        [-pos[0] * inv_r3, -pos[1] * inv_r3, -pos[2] * inv_r3]
    }

    fn kepler_energy(pos: [f32; 3], vel: [f32; 3]) -> f32 {
        let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
        let v2 = vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2];
        0.5 * v2 - 1.0 / r
    }

    #[test]
    fn kepler_circular_orbit_energy_conservation() {
        // Circular orbit, r=1, GM=1 ⇒ v=1, T=2π, E=-1/2.
        let mut p = Particles::with_capacity(1);
        p.push([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0);
        p.accelerations[0] = kepler_acc(p.positions[0]);
        let e0 = kepler_energy(p.positions[0], p.velocities[0]);
        let period = 2.0 * PI;
        let steps_per_period = 200;
        let n_periods = 10;
        let dt = period / steps_per_period as f32;
        for _ in 0..(steps_per_period * n_periods) {
            leapfrog_step(&mut p, dt, |pp| {
                pp.accelerations[0] = kepler_acc(pp.positions[0]);
            });
        }
        let ef = kepler_energy(p.positions[0], p.velocities[0]);
        let rel = ((ef - e0) / e0).abs();
        assert!(rel < 1e-3, "energy drift {rel} too large; e0={e0} ef={ef}");
    }

    #[test]
    fn kepler_eccentric_orbit_energy_bounded() {
        // Eccentric orbit (e=0.5): r=1, v=0.5√(GM/r) tangential plus
        // small extra at apocenter. Check energy oscillation stays bounded.
        let mut p = Particles::with_capacity(1);
        // Apocenter at (1.5, 0, 0), v=√(GM(2/r - 1/a)) where a=1, r=1.5
        // ⇒ v² = 2/1.5 - 1 = 1/3, v ≈ 0.577 tangential.
        let v_apo = (1.0_f32 / 3.0).sqrt();
        p.push([1.5, 0.0, 0.0], [0.0, v_apo, 0.0], 0.0);
        p.accelerations[0] = kepler_acc(p.positions[0]);
        let e0 = kepler_energy(p.positions[0], p.velocities[0]);

        let period = 2.0 * PI; // semi-major a=1 ⇒ T=2π
        let dt = period / 1000.0; // tighter dt for eccentric orbit
        let n_steps = (5.0 * period / dt) as usize;
        let mut max_drift = 0.0_f32;
        for _ in 0..n_steps {
            leapfrog_step(&mut p, dt, |pp| {
                pp.accelerations[0] = kepler_acc(pp.positions[0]);
            });
            let e = kepler_energy(p.positions[0], p.velocities[0]);
            max_drift = max_drift.max(((e - e0) / e0).abs());
        }
        assert!(max_drift < 5e-3, "eccentric orbit max drift {max_drift}");
    }
}
