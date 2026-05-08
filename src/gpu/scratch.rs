//! Sdílená persistent-scratch struktura pro `--gpu-full` pipeline (headless +
//! renderer). Pre-fix path měla 17 fresh `Vec::collect()` per tick + 9 fresh
//! `Vec` po readbacku. Persistent reuse zachovává kapacitu napříč ticky → 0
//! alloc/free v hot loop (bar capacity grow při pop spike).

use crate::{BRAIN_HIDDEN, BRAIN_OUTPUTS};

#[derive(Default)]
pub struct GpuFullScratch {
    // Upload scratch — naplňuje `brain_act_gpu_full` Phase 1 single pass přes cells.
    pub positions: Vec<[f32; 3]>,
    pub eff_radii: Vec<f32>,
    pub vision_radii: Vec<f32>,
    pub food_positions: Vec<[f32; 3]>,
    pub energies: Vec<f32>,
    pub headings: Vec<f32>,
    pub pitches: Vec<f32>,
    pub damage_accums: Vec<f32>,
    pub max_speeds: Vec<f32>,
    pub velocities: Vec<[f32; 3]>,
    pub angular_vels: Vec<f32>,
    pub pitch_vels: Vec<f32>,
    pub turn_rates: Vec<f32>,
    pub ages: Vec<u32>,
    pub cooldowns: Vec<u32>,
    pub body_dims: Vec<[f32; 3]>,
    pub aux: Vec<[f32; 4]>,
    // Readback scratch — `CellsGpu::download_full_batch_into` zapisuje do těchto.
    pub dl_hiddens: Vec<[f32; BRAIN_HIDDEN]>,
    pub dl_outputs: Vec<[f32; BRAIN_OUTPUTS]>,
    pub dl_velocities: Vec<[f32; 3]>,
    pub dl_angular: Vec<f32>,
    pub dl_pitch: Vec<f32>,
    pub dl_positions: Vec<[f32; 3]>,
    pub dl_ages: Vec<u32>,
    pub dl_cooldowns: Vec<u32>,
    pub dl_energies: Vec<f32>,
}

impl GpuFullScratch {
    pub fn clear_and_reserve(&mut self, n: usize, food_n: usize) {
        macro_rules! cr {
            ($v:expr, $cap:expr) => {{
                $v.clear();
                $v.reserve($cap);
            }};
        }
        cr!(self.positions, n);
        cr!(self.eff_radii, n);
        cr!(self.vision_radii, n);
        cr!(self.food_positions, food_n);
        cr!(self.energies, n);
        cr!(self.headings, n);
        cr!(self.pitches, n);
        cr!(self.damage_accums, n);
        cr!(self.max_speeds, n);
        cr!(self.velocities, n);
        cr!(self.angular_vels, n);
        cr!(self.pitch_vels, n);
        cr!(self.turn_rates, n);
        cr!(self.ages, n);
        cr!(self.cooldowns, n);
        cr!(self.body_dims, n);
        cr!(self.aux, n);
    }
}
