//! Persistent scratch buffers for the `--gpu-full` pipeline (headless +
//! renderer). The naive path allocated ~17 upload `Vec`s per tick plus ~9
//! more on readback; this struct reuses them across ticks so the hot loop
//! does zero allocations except when a population spike forces a capacity
//! grow.

use crate::{BRAIN_HIDDEN, BRAIN_INPUTS, BRAIN_OUTPUTS, N_BOND_MSG_CHANNELS};

#[derive(Default)]
pub struct GpuFullScratch {
    // Upload scratch — naplňuje `brain_act_gpu_full` Phase 1 single pass přes cells.
    pub positions: Vec<[f32; 3]>,
    pub eff_radii: Vec<f32>,
    /// Sprint 202 — inertial mass `volume × MASS_DENSITY`. Uploaded alongside
    /// `eff_radii`; consumed by motor / collision / brownian shaders for the
    /// `Δv = impulse / mass` conversion.
    pub masses: Vec<f32>,
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
    pub ages: Vec<u32>,
    pub cooldowns: Vec<u32>,
    pub body_dims: Vec<[f32; 3]>,
    pub aux: Vec<[f32; 4]>,
    pub hidden_ns: Vec<u32>,
    pub bonded_inboxes: Vec<[f32; N_BOND_MSG_CHANNELS]>,
    /// Sprint 198: per-cell symbiont presence (0/1) — read by populate_inputs
    /// to drive brain input slot 39 (has_symbiont).
    pub sym_has: Vec<u32>,
    /// Sprint 198: per-cell symbiont `deficit_streak` (u32 ticks). Shader
    /// normalises by SYMBIONT_UPKEEP_DEFICIT_TICKS for input slot 40.
    pub sym_deficit: Vec<u32>,
    // Readback scratch — `CellsGpu::download_full_batch_into` zapisuje do těchto.
    /// Sprint 188: GPU now mirrors `last_inputs_buf` back to the CPU so
    /// `Cell.last_inputs` reflects what the brain actually saw this tick
    /// (pre-S188 it was frozen at spawn-time zeros — see the GPU pipeline
    /// in `world.rs` Phase 5 → 10).
    pub dl_inputs: Vec<[f32; BRAIN_INPUTS]>,
    pub dl_hiddens: Vec<[f32; BRAIN_HIDDEN]>,
    pub dl_outputs: Vec<[f32; BRAIN_OUTPUTS]>,
    pub dl_velocities: Vec<[f32; 3]>,
    pub dl_angular: Vec<f32>,
    pub dl_pitch: Vec<f32>,
    pub dl_positions: Vec<[f32; 3]>,
    pub dl_ages: Vec<u32>,
    pub dl_cooldowns: Vec<u32>,
    pub dl_energies: Vec<f32>,
    // Late-tick phase scratch — populated AFTER brain_act_gpu_full
    // (i.e. after the step shader has written new positions/velocities).
    // resolve_collisions / predate / eat_food clear + refill these every
    // tick, so the capacity persists and per-tick allocations drop to zero.
    pub lt_positions: Vec<[f32; 3]>,
    pub lt_velocities: Vec<[f32; 3]>,
    pub lt_headings: Vec<f32>,
    pub lt_pitches: Vec<f32>,
    pub lt_eff_radii: Vec<f32>,
    /// Sprint 202 — late-tick inertial mass snapshot (volume × DENSITY) used
    /// by `resolve_collisions_gpu_pass1` so bond / damping / adhesion impulses
    /// convert correctly via `Δv = impulse · dt / mass`.
    pub lt_masses: Vec<f32>,
    pub lt_max_axes: Vec<f32>,
    pub lt_body_dims: Vec<[f32; 3]>,
    pub lt_carnivore: Vec<f32>,
    pub lt_attack_signals: Vec<f32>,
    /// Sprint 187: per-cell aggression genome traits and aggregated defense
    /// pool — refilled from `cell.genome.*` and bonded-partner walk every
    /// predate dispatch.
    pub lt_attack_gates: Vec<f32>,
    pub lt_predation_size_ratios: Vec<f32>,
    pub lt_defense_pool: Vec<f32>,
    pub lt_adhesion_types: Vec<u32>,
    pub lt_spike_counts: Vec<u32>,
    pub lt_spikes_packed: Vec<[f32; 4]>,
    pub lt_partner_idx: Vec<i32>,
    pub lt_bond_rest: Vec<f32>,
    pub lt_bond_stiff: Vec<f32>,
    pub lt_bond_damp: Vec<f32>,
    /// Sprint 202: BPC² flattened rest cosines per cell for the bond-bend
    /// angle-spring term in `collision.wgsl`.
    pub lt_bond_rest_cos: Vec<f32>,
    pub lt_rewards: Vec<f32>,
    // Per-food scratch (eat_food). Lengths == self.foods.len(), variable
    // across ticks as carrion drops/decays.
    pub lt_food_positions: Vec<[f32; 3]>,
    pub lt_food_kinds: Vec<u32>,
    pub lt_food_age_ticks: Vec<u32>,
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
        cr!(self.masses, n);
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
        cr!(self.ages, n);
        cr!(self.cooldowns, n);
        cr!(self.body_dims, n);
        cr!(self.aux, n);
        cr!(self.hidden_ns, n);
        cr!(self.bonded_inboxes, n);
        cr!(self.sym_has, n);
        cr!(self.sym_deficit, n);
    }

    /// Resize all per-cell snapshot fields to exactly `n` elements (filling
    /// new slots with zero/default). Used when the caller writes by slot
    /// index instead of pushing — see `cells_brain_act_gpu_full`'s parallel
    /// snapshot path.
    pub fn resize_snapshot(&mut self, n: usize) {
        self.positions.clear();
        self.positions.resize(n, [0.0; 3]);
        self.eff_radii.clear();
        self.eff_radii.resize(n, 0.0);
        self.masses.clear();
        self.masses.resize(n, 0.0);
        self.vision_radii.clear();
        self.vision_radii.resize(n, 0.0);
        self.energies.clear();
        self.energies.resize(n, 0.0);
        self.headings.clear();
        self.headings.resize(n, 0.0);
        self.pitches.clear();
        self.pitches.resize(n, 0.0);
        self.damage_accums.clear();
        self.damage_accums.resize(n, 0.0);
        self.max_speeds.clear();
        self.max_speeds.resize(n, 0.0);
        self.velocities.clear();
        self.velocities.resize(n, [0.0; 3]);
        self.angular_vels.clear();
        self.angular_vels.resize(n, 0.0);
        self.pitch_vels.clear();
        self.pitch_vels.resize(n, 0.0);
        self.ages.clear();
        self.ages.resize(n, 0);
        self.cooldowns.clear();
        self.cooldowns.resize(n, 0);
        self.body_dims.clear();
        self.body_dims.resize(n, [0.0; 3]);
        self.aux.clear();
        self.aux.resize(n, [0.0; 4]);
        self.hidden_ns.clear();
        self.hidden_ns.resize(n, 0);
        self.bonded_inboxes.clear();
        self.bonded_inboxes.resize(n, [0.0; N_BOND_MSG_CHANNELS]);
    }
}
