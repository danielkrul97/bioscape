//! Particle storage — Struct-of-Arrays layout for direct GPU upload.
//!
//! Parallel `Vec`s share index `i`. SoA is the layout the GPU shaders
//! expect (`positions[i*3..i*3+3]`, etc.); CPU diagnostics also benefit
//! from contiguous f32 streams for reductions.

#[derive(Debug, Clone, Default)]
pub struct Particles {
    pub positions: Vec<[f32; 3]>,
    pub velocities: Vec<[f32; 3]>,
    pub accelerations: Vec<[f32; 3]>,
    pub masses: Vec<f32>,
    /// Per-particle SPH smoothing length. Initialised in S209 once SPH lands;
    /// pre-S209 stays at the default `0.0` and is ignored.
    pub smoothing_lengths: Vec<f32>,
    /// Per-particle SPH density. Same lifecycle as `smoothing_lengths`.
    pub densities: Vec<f32>,
}

impl Particles {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            positions: Vec::with_capacity(n),
            velocities: Vec::with_capacity(n),
            accelerations: Vec::with_capacity(n),
            masses: Vec::with_capacity(n),
            smoothing_lengths: Vec::with_capacity(n),
            densities: Vec::with_capacity(n),
        }
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn push(&mut self, pos: [f32; 3], vel: [f32; 3], mass: f32) {
        self.positions.push(pos);
        self.velocities.push(vel);
        self.accelerations.push([0.0; 3]);
        self.masses.push(mass);
        self.smoothing_lengths.push(0.0);
        self.densities.push(0.0);
    }

    pub fn clear(&mut self) {
        self.positions.clear();
        self.velocities.clear();
        self.accelerations.clear();
        self.masses.clear();
        self.smoothing_lengths.clear();
        self.densities.clear();
    }

    pub fn total_mass(&self) -> f64 {
        self.masses.iter().map(|&m| m as f64).sum()
    }

    pub fn center_of_mass(&self) -> [f64; 3] {
        if self.is_empty() {
            return [0.0; 3];
        }
        let mut com = [0.0_f64; 3];
        let mut m_sum = 0.0_f64;
        for (p, &m) in self.positions.iter().zip(&self.masses) {
            let m64 = m as f64;
            com[0] += p[0] as f64 * m64;
            com[1] += p[1] as f64 * m64;
            com[2] += p[2] as f64 * m64;
            m_sum += m64;
        }
        if m_sum > 0.0 {
            [com[0] / m_sum, com[1] / m_sum, com[2] / m_sum]
        } else {
            [0.0; 3]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_particles() {
        let p = Particles::default();
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
        assert_eq!(p.total_mass(), 0.0);
        assert_eq!(p.center_of_mass(), [0.0; 3]);
    }

    #[test]
    fn push_and_com() {
        let mut p = Particles::with_capacity(2);
        p.push([1.0, 0.0, 0.0], [0.0; 3], 2.0);
        p.push([-1.0, 0.0, 0.0], [0.0; 3], 2.0);
        assert_eq!(p.len(), 2);
        assert_eq!(p.total_mass(), 4.0);
        let com = p.center_of_mass();
        assert!(com[0].abs() < 1e-9);
    }
}
