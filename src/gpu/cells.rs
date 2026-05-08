use std::sync::Arc;


use crate::*;
use super::*;

// ============================================================================
// Sprint 51: CellsGpu persistent SoA state — drží brain weights + last_*
// + xoshiro RNG na GPU mezi ticky. Eliminuje 30 MB/tick brain upload bottleneck
// ze Sprintu 44.
// ============================================================================

/// Persistent SoA cell state na GPU. Drží brain forward state (last_inputs,
/// last_hidden, last_outputs, brain weights) + velocities (pro brownian
/// mutation) + per-cell xoshiro128++ RNG state. Per Sprint 51 scope:
/// **NE-drží** position/heading/etc. (ty zůstávají na CPU pro sensor/motor/
/// step/collision/predate fáze — Sprint 50 standalone shadery jsou ready
/// pro plnou migraci, kdyby se rozhodlo).
///
/// Lifecycle:
/// 1. `new(ctx, capacity)` alokuje buffers + initializuje xoshiro state.
/// 2. `upload_brains(brains, init_xoshiro_seed)` na sim init.
/// 3. Hot loop:
///    - `upload_inputs(last_inputs)` před brain forward.
///    - `forward_batch_persistent(brain_gpu)` — channels/persistent.
///    - `download_hidden_outputs() -> (Vec<hidden>, Vec<outputs>)` po brain.
///    - `upload_velocities(velocities)` před brownian.
///    - `brownian_persistent(brownian_gpu, ...)` — mutuje velocities + state.
///    - `download_velocities() -> Vec<velocities>` po brownian.
///    - `upload_rewards(rewards)` po eat_food.
///    - `hebbian_persistent(hebbian_gpu, lr)` — mutuje brain weights in-place.
/// 4. `upload_brain_at(idx, brain)` po reproduce (nová cell na slot idx).
/// 5. `download_brains() -> Vec<Brain>` pro checkpoint nebo introspection.
pub struct CellsGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    capacity: usize,
    last_inputs_buf: wgpu::Buffer,
    last_hidden_buf: wgpu::Buffer,
    last_outputs_buf: wgpu::Buffer,
    brain_weights_buf: wgpu::Buffer,
    velocities_buf: wgpu::Buffer,
    xoshiro_state_buf: wgpu::Buffer,
    rewards_buf: wgpu::Buffer,
    /// Sprint 61: cell metadata pro GPU populate_brain_inputs shader.
    /// energy / heading / pitch / damage_accum mutable per tick (CPU upload),
    /// max_speed / eff_radius zpravidla constant po reproduce/morph (CPU upload
    /// per tick je ale levný — ~4 KB × 6 = 24 KB při N=1000).
    energy_buf: wgpu::Buffer,
    heading_buf: wgpu::Buffer,
    pitch_buf: wgpu::Buffer,
    damage_accum_buf: wgpu::Buffer,
    max_speed_buf: wgpu::Buffer,
    eff_radius_buf: wgpu::Buffer,
    /// Sprint 62: motor on GPU — turn_rate per-cell konstanta (genome),
    /// angular/pitch velocities mutated by motor shader.
    turn_rate_buf: wgpu::Buffer,
    angular_velocity_buf: wgpu::Buffer,
    pitch_velocity_buf: wgpu::Buffer,
    /// Sprint 62: motor batch readback. velocity_rb už existuje (Sprint 51).
    angular_velocity_rb: wgpu::Buffer,
    pitch_velocity_rb: wgpu::Buffer,
    /// Sprint 63: step on GPU — kinematics + drag + energy + bounce per cell.
    /// Position mutated each tick (integrate_kinematics + bounce). Age/cooldown
    /// incremented. Body_dims (length/width/height) constant per cell post-morph.
    /// Aux (spike/shell/vision/attack) per-tick recomputed (attack from outputs).
    position_buf: wgpu::Buffer,
    age_buf: wgpu::Buffer,
    cooldown_buf: wgpu::Buffer,
    body_dims_buf: wgpu::Buffer,
    aux_buf: wgpu::Buffer,
    position_rb: wgpu::Buffer,
    age_rb: wgpu::Buffer,
    cooldown_rb: wgpu::Buffer,
    energy_rb: wgpu::Buffer,
    last_hidden_rb: wgpu::Buffer,
    last_outputs_rb: wgpu::Buffer,
    velocities_rb: wgpu::Buffer,
    brain_weights_rb: wgpu::Buffer,
    /// Sprint 51: staging pro `swap_to` — wgpu zakazuje same-buffer copy.
    swap_brain_temp: wgpu::Buffer,
    swap_xoshiro_temp: wgpu::Buffer,
    swap_turn_rate_temp: wgpu::Buffer,
}

impl CellsGpu {
    pub fn with_context(ctx: &GpuContext, capacity: usize) -> Self {
        Self::with_device_inner(Arc::clone(&ctx.device), Arc::clone(&ctx.queue), capacity)
    }

    fn with_device_inner(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        capacity: usize,
    ) -> Self {
        assert!(capacity > 0);
        let f = std::mem::size_of::<f32>() as u64;
        let n = capacity as u64;
        let stor_dst_src = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let last_inputs_buf = mk("cells-last-inputs", n * (BRAIN_INPUTS as u64) * f, stor_dst_src);
        let last_hidden_buf = mk("cells-last-hidden", n * (BRAIN_HIDDEN as u64) * f, stor_dst_src);
        let last_outputs_buf = mk("cells-last-outputs", n * (BRAIN_OUTPUTS as u64) * f, stor_dst_src);
        let brain_weights_buf = mk("cells-brain-weights", n * (BRAIN_WEIGHTS_PER_CELL as u64) * f, stor_dst_src);
        let velocities_buf = mk("cells-velocities", n * 3 * f, stor_dst_src);
        let xoshiro_state_buf = mk("cells-xoshiro", n * 4 * 4, stor_dst_src);
        let rewards_buf = mk(
            "cells-rewards",
            n * f,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        // Sprint 61: cell metadata pro populate_inputs shader.
        let energy_buf = mk("cells-energy", n * f, stor_dst_src);
        let heading_buf = mk("cells-heading", n * f, stor_dst_src);
        let pitch_buf = mk("cells-pitch", n * f, stor_dst_src);
        let damage_accum_buf = mk("cells-damage", n * f, stor_dst_src);
        let max_speed_buf = mk("cells-max-speed", n * f, stor_dst_src);
        let eff_radius_buf = mk("cells-eff-radius", n * f, stor_dst_src);
        // Sprint 62: motor shader buffers.
        let turn_rate_buf = mk("cells-turn-rate", n * f, stor_dst_src);
        let angular_velocity_buf = mk("cells-ang-vel", n * f, stor_dst_src);
        let pitch_velocity_buf = mk("cells-pitch-vel", n * f, stor_dst_src);
        let angular_velocity_rb = mk("cells-ang-vel-rb", n * f, read);
        let pitch_velocity_rb = mk("cells-pitch-vel-rb", n * f, read);
        // Sprint 63: step shader buffers.
        let position_buf = mk("cells-position", n * 3 * f, stor_dst_src);
        let age_buf = mk("cells-age", n * 4, stor_dst_src);
        let cooldown_buf = mk("cells-cooldown", n * 4, stor_dst_src);
        let body_dims_buf = mk("cells-body-dims", n * 3 * f, stor_dst_src);
        let aux_buf = mk("cells-aux", n * 4 * f, stor_dst_src);
        let position_rb = mk("cells-position-rb", n * 3 * f, read);
        let age_rb = mk("cells-age-rb", n * 4, read);
        let cooldown_rb = mk("cells-cooldown-rb", n * 4, read);
        let energy_rb = mk("cells-energy-rb", n * f, read);
        let last_hidden_rb = mk("cells-hidden-rb", n * (BRAIN_HIDDEN as u64) * f, read);
        let last_outputs_rb = mk("cells-outputs-rb", n * (BRAIN_OUTPUTS as u64) * f, read);
        let velocities_rb = mk("cells-velocities-rb", n * 3 * f, read);
        let brain_weights_rb = mk("cells-weights-rb", n * (BRAIN_WEIGHTS_PER_CELL as u64) * f, read);
        let swap_brain_temp = mk(
            "cells-swap-brain-temp",
            (BRAIN_WEIGHTS_PER_CELL as u64) * f,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let swap_xoshiro_temp = mk(
            "cells-swap-xoshiro-temp",
            16,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let swap_turn_rate_temp = mk(
            "cells-swap-turn-rate-temp",
            f,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let _ = mk;
        Self {
            device,
            queue,
            capacity,
            last_inputs_buf,
            last_hidden_buf,
            last_outputs_buf,
            brain_weights_buf,
            velocities_buf,
            xoshiro_state_buf,
            rewards_buf,
            energy_buf,
            heading_buf,
            pitch_buf,
            damage_accum_buf,
            max_speed_buf,
            eff_radius_buf,
            turn_rate_buf,
            angular_velocity_buf,
            pitch_velocity_buf,
            angular_velocity_rb,
            pitch_velocity_rb,
            position_buf,
            age_buf,
            cooldown_buf,
            body_dims_buf,
            aux_buf,
            position_rb,
            age_rb,
            cooldown_rb,
            energy_rb,
            last_hidden_rb,
            last_outputs_rb,
            velocities_rb,
            brain_weights_rb,
            swap_brain_temp,
            swap_xoshiro_temp,
            swap_turn_rate_temp,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn last_inputs_buffer(&self) -> &wgpu::Buffer { &self.last_inputs_buf }
    pub fn last_hidden_buffer(&self) -> &wgpu::Buffer { &self.last_hidden_buf }
    pub fn last_outputs_buffer(&self) -> &wgpu::Buffer { &self.last_outputs_buf }
    pub fn brain_weights_buffer(&self) -> &wgpu::Buffer { &self.brain_weights_buf }
    pub fn velocities_buffer(&self) -> &wgpu::Buffer { &self.velocities_buf }
    pub fn xoshiro_state_buffer(&self) -> &wgpu::Buffer { &self.xoshiro_state_buf }
    pub fn rewards_buffer(&self) -> &wgpu::Buffer { &self.rewards_buf }
    /// Sprint 61: cell metadata buffery pro populate_inputs shader binding.
    pub fn energy_buffer(&self) -> &wgpu::Buffer { &self.energy_buf }
    pub fn heading_buffer(&self) -> &wgpu::Buffer { &self.heading_buf }
    pub fn pitch_buffer(&self) -> &wgpu::Buffer { &self.pitch_buf }
    pub fn damage_accum_buffer(&self) -> &wgpu::Buffer { &self.damage_accum_buf }
    pub fn max_speed_buffer(&self) -> &wgpu::Buffer { &self.max_speed_buf }
    pub fn eff_radius_buffer(&self) -> &wgpu::Buffer { &self.eff_radius_buf }
    /// Sprint 62: motor shader buffery (turn_rate read, angular/pitch velocities rw).
    pub fn turn_rate_buffer(&self) -> &wgpu::Buffer { &self.turn_rate_buf }
    pub fn angular_velocity_buffer(&self) -> &wgpu::Buffer { &self.angular_velocity_buf }
    pub fn pitch_velocity_buffer(&self) -> &wgpu::Buffer { &self.pitch_velocity_buf }
    /// Sprint 63: step shader buffery.
    pub fn position_buffer(&self) -> &wgpu::Buffer { &self.position_buf }
    pub fn age_buffer(&self) -> &wgpu::Buffer { &self.age_buf }
    pub fn cooldown_buffer(&self) -> &wgpu::Buffer { &self.cooldown_buf }
    pub fn body_dims_buffer(&self) -> &wgpu::Buffer { &self.body_dims_buf }
    pub fn aux_buffer(&self) -> &wgpu::Buffer { &self.aux_buf }

    /// Sprint 63: upload positions (3D per cell, packed).
    pub fn upload_positions(&self, positions: &[[f32; 3]]) {
        let mut packed: Vec<f32> = Vec::with_capacity(positions.len() * 3);
        for p in positions { packed.extend_from_slice(p); }
        self.queue.write_buffer(&self.position_buf, 0, bytemuck::cast_slice(&packed));
    }

    /// Sprint 63: upload age + cooldown (u32 per cell).
    pub fn upload_age_cooldown(&self, ages: &[u32], cooldowns: &[u32]) {
        debug_assert_eq!(ages.len(), cooldowns.len());
        self.queue.write_buffer(&self.age_buf, 0, bytemuck::cast_slice(ages));
        self.queue.write_buffer(&self.cooldown_buf, 0, bytemuck::cast_slice(cooldowns));
    }

    /// Sprint 63: upload body dimensions (3 × f32 per cell: length, width, height).
    pub fn upload_body_dims(&self, body_dims: &[[f32; 3]]) {
        let mut packed: Vec<f32> = Vec::with_capacity(body_dims.len() * 3);
        for d in body_dims { packed.extend_from_slice(d); }
        self.queue.write_buffer(&self.body_dims_buf, 0, bytemuck::cast_slice(&packed));
    }

    /// Sprint 63: upload aux (4 × f32 per cell: spike, shell, vision, attack).
    /// Attack je per-tick recomputed z brain output[6]; ostatní jsou per-cell
    /// konstanty (genome). Lazy per-tick upload pro consistency.
    pub fn upload_aux(&self, aux: &[[f32; 4]]) {
        let mut packed: Vec<f32> = Vec::with_capacity(aux.len() * 4);
        for a in aux { packed.extend_from_slice(a); }
        self.queue.write_buffer(&self.aux_buf, 0, bytemuck::cast_slice(&packed));
    }

    /// Sprint 63: combined batch readback po brain_act + step pipeline.
    /// Single Wait barrier pro all 9 buffers, výsledek se zapisuje do volajícím
    /// poskytnutých scratch Vec slotů (clear+extend pattern). Pre-fix path
    /// alokovala 9 fresh Vec na výstupu — nyní 0 alloc/free per call při
    /// stable capacity.
    #[allow(clippy::too_many_arguments)]
    pub fn download_full_batch_into(
        &self,
        n: usize,
        hidden_out: &mut Vec<[f32; BRAIN_HIDDEN]>,
        outputs_out: &mut Vec<[f32; BRAIN_OUTPUTS]>,
        velocities_out: &mut Vec<[f32; 3]>,
        angular_out: &mut Vec<f32>,
        pitch_out: &mut Vec<f32>,
        positions_out: &mut Vec<[f32; 3]>,
        ages_out: &mut Vec<u32>,
        cooldowns_out: &mut Vec<u32>,
        energies_out: &mut Vec<f32>,
    ) {
        if n == 0 {
            hidden_out.clear();
            outputs_out.clear();
            velocities_out.clear();
            angular_out.clear();
            pitch_out.clear();
            positions_out.clear();
            ages_out.clear();
            cooldowns_out.clear();
            energies_out.clear();
            return;
        }
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cells-download-full"),
        });
        self.download_full_copy_into(&mut encoder, n);
        self.queue.submit(Some(encoder.finish()));
        self.download_full_read_into(
            n,
            hidden_out,
            outputs_out,
            velocities_out,
            angular_out,
            pitch_out,
            positions_out,
            ages_out,
            cooldowns_out,
            energies_out,
        );
    }

    pub fn download_full_copy_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        n: usize,
    ) {
        if n == 0 {
            return;
        }
        assert!(n <= self.capacity);
        let h_bytes = (n * BRAIN_HIDDEN * 4) as u64;
        let o_bytes = (n * BRAIN_OUTPUTS * 4) as u64;
        let v_bytes = (n * 3 * 4) as u64;
        let a_bytes = (n * 4) as u64;
        let p_bytes = (n * 4) as u64;
        let pos_bytes = (n * 3 * 4) as u64;
        let age_bytes = (n * 4) as u64;
        let cd_bytes = (n * 4) as u64;
        let e_bytes = (n * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.last_hidden_buf, 0, &self.last_hidden_rb, 0, h_bytes);
        encoder.copy_buffer_to_buffer(&self.last_outputs_buf, 0, &self.last_outputs_rb, 0, o_bytes);
        encoder.copy_buffer_to_buffer(&self.velocities_buf, 0, &self.velocities_rb, 0, v_bytes);
        encoder.copy_buffer_to_buffer(&self.angular_velocity_buf, 0, &self.angular_velocity_rb, 0, a_bytes);
        encoder.copy_buffer_to_buffer(&self.pitch_velocity_buf, 0, &self.pitch_velocity_rb, 0, p_bytes);
        encoder.copy_buffer_to_buffer(&self.position_buf, 0, &self.position_rb, 0, pos_bytes);
        encoder.copy_buffer_to_buffer(&self.age_buf, 0, &self.age_rb, 0, age_bytes);
        encoder.copy_buffer_to_buffer(&self.cooldown_buf, 0, &self.cooldown_rb, 0, cd_bytes);
        encoder.copy_buffer_to_buffer(&self.energy_buf, 0, &self.energy_rb, 0, e_bytes);
    }

    /// Map readback buffers, Wait, copy out into caller scratch. MUST be called
    /// after the encoder containing `download_full_copy_into` has been submitted.
    #[allow(clippy::too_many_arguments)]
    pub fn download_full_read_into(
        &self,
        n: usize,
        hidden_out: &mut Vec<[f32; BRAIN_HIDDEN]>,
        outputs_out: &mut Vec<[f32; BRAIN_OUTPUTS]>,
        velocities_out: &mut Vec<[f32; 3]>,
        angular_out: &mut Vec<f32>,
        pitch_out: &mut Vec<f32>,
        positions_out: &mut Vec<[f32; 3]>,
        ages_out: &mut Vec<u32>,
        cooldowns_out: &mut Vec<u32>,
        energies_out: &mut Vec<f32>,
    ) {
        hidden_out.clear();
        outputs_out.clear();
        velocities_out.clear();
        angular_out.clear();
        pitch_out.clear();
        positions_out.clear();
        ages_out.clear();
        cooldowns_out.clear();
        energies_out.clear();
        if n == 0 {
            return;
        }
        assert!(n <= self.capacity);
        let h_bytes = (n * BRAIN_HIDDEN * 4) as u64;
        let o_bytes = (n * BRAIN_OUTPUTS * 4) as u64;
        let v_bytes = (n * 3 * 4) as u64;
        let a_bytes = (n * 4) as u64;
        let p_bytes = (n * 4) as u64;
        let pos_bytes = (n * 3 * 4) as u64;
        let age_bytes = (n * 4) as u64;
        let cd_bytes = (n * 4) as u64;
        let e_bytes = (n * 4) as u64;
        let h_s = self.last_hidden_rb.slice(0..h_bytes);
        let o_s = self.last_outputs_rb.slice(0..o_bytes);
        let v_s = self.velocities_rb.slice(0..v_bytes);
        let a_s = self.angular_velocity_rb.slice(0..a_bytes);
        let p_s = self.pitch_velocity_rb.slice(0..p_bytes);
        let pos_s = self.position_rb.slice(0..pos_bytes);
        let age_s = self.age_rb.slice(0..age_bytes);
        let cd_s = self.cooldown_rb.slice(0..cd_bytes);
        let e_s = self.energy_rb.slice(0..e_bytes);
        for s in [&h_s, &o_s, &v_s, &a_s, &p_s, &pos_s, &age_s, &cd_s, &e_s] {
            s.map_async(wgpu::MapMode::Read, |_| {});
        }
        self.device.poll(wgpu::Maintain::Wait);
        let h_data = h_s.get_mapped_range();
        let o_data = o_s.get_mapped_range();
        let v_data = v_s.get_mapped_range();
        let a_data = a_s.get_mapped_range();
        let p_data = p_s.get_mapped_range();
        let pos_data = pos_s.get_mapped_range();
        let age_data = age_s.get_mapped_range();
        let cd_data = cd_s.get_mapped_range();
        let e_data = e_s.get_mapped_range();
        let h_f: &[f32] = bytemuck::cast_slice(&h_data);
        let o_f: &[f32] = bytemuck::cast_slice(&o_data);
        let v_f: &[f32] = bytemuck::cast_slice(&v_data);
        let a_f: &[f32] = bytemuck::cast_slice(&a_data);
        let p_f: &[f32] = bytemuck::cast_slice(&p_data);
        let pos_f: &[f32] = bytemuck::cast_slice(&pos_data);
        let age_u: &[u32] = bytemuck::cast_slice(&age_data);
        let cd_u: &[u32] = bytemuck::cast_slice(&cd_data);
        let e_f: &[f32] = bytemuck::cast_slice(&e_data);
        hidden_out.reserve(n);
        outputs_out.reserve(n);
        velocities_out.reserve(n);
        positions_out.reserve(n);
        for i in 0..n {
            let mut h = [0.0_f32; BRAIN_HIDDEN];
            h.copy_from_slice(&h_f[i * BRAIN_HIDDEN..(i + 1) * BRAIN_HIDDEN]);
            hidden_out.push(h);
            let mut o = [0.0_f32; BRAIN_OUTPUTS];
            o.copy_from_slice(&o_f[i * BRAIN_OUTPUTS..(i + 1) * BRAIN_OUTPUTS]);
            outputs_out.push(o);
            velocities_out.push([v_f[i * 3], v_f[i * 3 + 1], v_f[i * 3 + 2]]);
            positions_out.push([pos_f[i * 3], pos_f[i * 3 + 1], pos_f[i * 3 + 2]]);
        }
        angular_out.extend_from_slice(&a_f[..n]);
        pitch_out.extend_from_slice(&p_f[..n]);
        ages_out.extend_from_slice(&age_u[..n]);
        cooldowns_out.extend_from_slice(&cd_u[..n]);
        energies_out.extend_from_slice(&e_f[..n]);
        drop(h_data); drop(o_data); drop(v_data); drop(a_data); drop(p_data);
        drop(pos_data); drop(age_data); drop(cd_data); drop(e_data);
        self.last_hidden_rb.unmap();
        self.last_outputs_rb.unmap();
        self.velocities_rb.unmap();
        self.angular_velocity_rb.unmap();
        self.pitch_velocity_rb.unmap();
        self.position_rb.unmap();
        self.age_rb.unmap();
        self.cooldown_rb.unmap();
        self.energy_rb.unmap();
    }

    /// Sprint 62: turn_rate je per-cell genome konstanta. Upload na sim init +
    /// při reproduce (sparse). Sprint 61 `upload_metadata` je per-tick mutable
    /// (energy/heading/pitch/damage/max_speed/eff_radius).
    pub fn upload_turn_rates(&self, turn_rates: &[f32]) {
        self.queue.write_buffer(&self.turn_rate_buf, 0, bytemuck::cast_slice(turn_rates));
    }

    pub fn upload_turn_rate_at(&self, slot: usize, turn_rate: f32) {
        assert!(slot < self.capacity);
        let offset = (slot * std::mem::size_of::<f32>()) as u64;
        self.queue.write_buffer(&self.turn_rate_buf, offset, bytemuck::cast_slice(&[turn_rate]));
    }

    /// Sprint 62: upload current angular + pitch velocity (Cell::angular_velocity,
    /// Cell::pitch_velocity). Volá se před brain_act_gpu_full.
    pub fn upload_angular_pitch(&self, angular: &[f32], pitches: &[f32]) {
        debug_assert_eq!(angular.len(), pitches.len());
        self.queue.write_buffer(&self.angular_velocity_buf, 0, bytemuck::cast_slice(angular));
        self.queue.write_buffer(&self.pitch_velocity_buf, 0, bytemuck::cast_slice(pitches));
    }

    /// Sprint 62: batch readback motor results (velocities + angular + pitch).
    /// Single Wait barrier místo 3× separate downloads. Volá se po motor +
    /// brownian dispatch v brain_act_gpu_full.
    pub fn download_motor_state(
        &self,
        n: usize,
    ) -> (Vec<[f32; 3]>, Vec<f32>, Vec<f32>) {
        if n == 0 {
            return (Vec::new(), Vec::new(), Vec::new());
        }
        assert!(n <= self.capacity);
        let v_bytes = (n * 3 * 4) as u64;
        let a_bytes = (n * 4) as u64;
        let p_bytes = (n * 4) as u64;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cells-download-motor"),
        });
        encoder.copy_buffer_to_buffer(&self.velocities_buf, 0, &self.velocities_rb, 0, v_bytes);
        encoder.copy_buffer_to_buffer(&self.angular_velocity_buf, 0, &self.angular_velocity_rb, 0, a_bytes);
        encoder.copy_buffer_to_buffer(&self.pitch_velocity_buf, 0, &self.pitch_velocity_rb, 0, p_bytes);
        self.queue.submit(Some(encoder.finish()));
        let v_s = self.velocities_rb.slice(0..v_bytes);
        let a_s = self.angular_velocity_rb.slice(0..a_bytes);
        let p_s = self.pitch_velocity_rb.slice(0..p_bytes);
        v_s.map_async(wgpu::MapMode::Read, |_| {});
        a_s.map_async(wgpu::MapMode::Read, |_| {});
        p_s.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let v_data = v_s.get_mapped_range();
        let a_data = a_s.get_mapped_range();
        let p_data = p_s.get_mapped_range();
        let v_f: &[f32] = bytemuck::cast_slice(&v_data);
        let a_f: &[f32] = bytemuck::cast_slice(&a_data);
        let p_f: &[f32] = bytemuck::cast_slice(&p_data);
        let velocities: Vec<[f32; 3]> = (0..n)
            .map(|i| [v_f[i * 3], v_f[i * 3 + 1], v_f[i * 3 + 2]])
            .collect();
        let angular: Vec<f32> = a_f[..n].to_vec();
        let pitch_vels: Vec<f32> = p_f[..n].to_vec();
        drop(v_data);
        drop(a_data);
        drop(p_data);
        self.velocities_rb.unmap();
        self.angular_velocity_rb.unmap();
        self.pitch_velocity_rb.unmap();
        (velocities, angular, pitch_vels)
    }

    /// Sprint 62: combined batch readback hidden + outputs + motor state v
    /// jediném Wait barrier. Volá se na konci brain_act_gpu_full po
    /// brain.forward + motor.dispatch + brownian.dispatch sequence.
    pub fn download_brain_motor_batch(
        &self,
        n: usize,
    ) -> (
        Vec<[f32; BRAIN_HIDDEN]>,
        Vec<[f32; BRAIN_OUTPUTS]>,
        Vec<[f32; 3]>,
        Vec<f32>,
        Vec<f32>,
    ) {
        if n == 0 {
            return (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        }
        assert!(n <= self.capacity);
        let h_bytes = (n * BRAIN_HIDDEN * 4) as u64;
        let o_bytes = (n * BRAIN_OUTPUTS * 4) as u64;
        let v_bytes = (n * 3 * 4) as u64;
        let a_bytes = (n * 4) as u64;
        let p_bytes = (n * 4) as u64;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cells-download-batch"),
        });
        encoder.copy_buffer_to_buffer(&self.last_hidden_buf, 0, &self.last_hidden_rb, 0, h_bytes);
        encoder.copy_buffer_to_buffer(&self.last_outputs_buf, 0, &self.last_outputs_rb, 0, o_bytes);
        encoder.copy_buffer_to_buffer(&self.velocities_buf, 0, &self.velocities_rb, 0, v_bytes);
        encoder.copy_buffer_to_buffer(&self.angular_velocity_buf, 0, &self.angular_velocity_rb, 0, a_bytes);
        encoder.copy_buffer_to_buffer(&self.pitch_velocity_buf, 0, &self.pitch_velocity_rb, 0, p_bytes);
        self.queue.submit(Some(encoder.finish()));
        let h_s = self.last_hidden_rb.slice(0..h_bytes);
        let o_s = self.last_outputs_rb.slice(0..o_bytes);
        let v_s = self.velocities_rb.slice(0..v_bytes);
        let a_s = self.angular_velocity_rb.slice(0..a_bytes);
        let p_s = self.pitch_velocity_rb.slice(0..p_bytes);
        h_s.map_async(wgpu::MapMode::Read, |_| {});
        o_s.map_async(wgpu::MapMode::Read, |_| {});
        v_s.map_async(wgpu::MapMode::Read, |_| {});
        a_s.map_async(wgpu::MapMode::Read, |_| {});
        p_s.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let h_data = h_s.get_mapped_range();
        let o_data = o_s.get_mapped_range();
        let v_data = v_s.get_mapped_range();
        let a_data = a_s.get_mapped_range();
        let p_data = p_s.get_mapped_range();
        let h_f: &[f32] = bytemuck::cast_slice(&h_data);
        let o_f: &[f32] = bytemuck::cast_slice(&o_data);
        let v_f: &[f32] = bytemuck::cast_slice(&v_data);
        let a_f: &[f32] = bytemuck::cast_slice(&a_data);
        let p_f: &[f32] = bytemuck::cast_slice(&p_data);
        let hidden: Vec<[f32; BRAIN_HIDDEN]> = (0..n)
            .map(|i| {
                let mut a = [0.0_f32; BRAIN_HIDDEN];
                a.copy_from_slice(&h_f[i * BRAIN_HIDDEN..(i + 1) * BRAIN_HIDDEN]);
                a
            })
            .collect();
        let outputs: Vec<[f32; BRAIN_OUTPUTS]> = (0..n)
            .map(|i| {
                let mut a = [0.0_f32; BRAIN_OUTPUTS];
                a.copy_from_slice(&o_f[i * BRAIN_OUTPUTS..(i + 1) * BRAIN_OUTPUTS]);
                a
            })
            .collect();
        let velocities: Vec<[f32; 3]> = (0..n)
            .map(|i| [v_f[i * 3], v_f[i * 3 + 1], v_f[i * 3 + 2]])
            .collect();
        let angular: Vec<f32> = a_f[..n].to_vec();
        let pitch_vels: Vec<f32> = p_f[..n].to_vec();
        drop(h_data);
        drop(o_data);
        drop(v_data);
        drop(a_data);
        drop(p_data);
        self.last_hidden_rb.unmap();
        self.last_outputs_rb.unmap();
        self.velocities_rb.unmap();
        self.angular_velocity_rb.unmap();
        self.pitch_velocity_rb.unmap();
        (hidden, outputs, velocities, angular, pitch_vels)
    }

    /// Sprint 61: bulk upload metadata pro populate_inputs shader. Sloučeno
    /// do jediného call pro jasnost — všechna pole jsou per-cell f32.
    pub fn upload_metadata(
        &self,
        energies: &[f32],
        headings: &[f32],
        pitches: &[f32],
        damage_accums: &[f32],
        max_speeds: &[f32],
        eff_radii: &[f32],
    ) {
        debug_assert_eq!(energies.len(), headings.len());
        debug_assert_eq!(energies.len(), pitches.len());
        debug_assert_eq!(energies.len(), damage_accums.len());
        debug_assert_eq!(energies.len(), max_speeds.len());
        debug_assert_eq!(energies.len(), eff_radii.len());
        self.queue.write_buffer(&self.energy_buf, 0, bytemuck::cast_slice(energies));
        self.queue.write_buffer(&self.heading_buf, 0, bytemuck::cast_slice(headings));
        self.queue.write_buffer(&self.pitch_buf, 0, bytemuck::cast_slice(pitches));
        self.queue.write_buffer(&self.damage_accum_buf, 0, bytemuck::cast_slice(damage_accums));
        self.queue.write_buffer(&self.max_speed_buf, 0, bytemuck::cast_slice(max_speeds));
        self.queue.write_buffer(&self.eff_radius_buf, 0, bytemuck::cast_slice(eff_radii));
    }

    /// Uploaduje brain weights pro N cells. Volá se na sim init + po reproduce
    /// (re-upload všech, nebo per-slot přes `upload_brain_at`).
    pub fn upload_brains<'a, I>(&self, brains: I)
    where
        I: IntoIterator<Item = &'a Brain>,
    {
        let mut packed: Vec<f32> = Vec::with_capacity(self.capacity * BRAIN_WEIGHTS_PER_CELL);
        for brain in brains {
            for row in brain.w1.iter() { packed.extend_from_slice(row); }
            packed.extend_from_slice(&brain.b1);
            for row in brain.w2.iter() { packed.extend_from_slice(row); }
            packed.extend_from_slice(&brain.b2);
        }
        self.queue.write_buffer(&self.brain_weights_buf, 0, bytemuck::cast_slice(&packed));
    }

    /// Uploaduje brain weights pro jeden slot (idx). Použito po reproduce —
    /// nová cell se zapíše na konec Vec, její brain na slot idx = old_len.
    pub fn upload_brain_at(&self, idx: usize, brain: &Brain) {
        assert!(idx < self.capacity);
        let mut packed: Vec<f32> = Vec::with_capacity(BRAIN_WEIGHTS_PER_CELL);
        for row in brain.w1.iter() { packed.extend_from_slice(row); }
        packed.extend_from_slice(&brain.b1);
        for row in brain.w2.iter() { packed.extend_from_slice(row); }
        packed.extend_from_slice(&brain.b2);
        let offset = (idx * BRAIN_WEIGHTS_PER_CELL * 4) as u64;
        self.queue.write_buffer(&self.brain_weights_buf, offset, bytemuck::cast_slice(&packed));
    }

    /// Sync brains z GPU zpátky na CPU. Pomalá operace — kvůli checkpoint
    /// nebo introspekci. Hot loop ji nevolá.
    pub fn download_brains(&self, n: usize) -> Vec<Brain> {
        assert!(n <= self.capacity);
        if n == 0 { return Vec::new(); }
        let bytes = (n * BRAIN_WEIGHTS_PER_CELL * 4) as u64;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cells-download-brains"),
        });
        encoder.copy_buffer_to_buffer(&self.brain_weights_buf, 0, &self.brain_weights_rb, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let s = self.brain_weights_rb.slice(0..bytes);
        s.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = s.get_mapped_range();
        let f: &[f32] = bytemuck::cast_slice(&data);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * BRAIN_WEIGHTS_PER_CELL;
            let mut b = Brain {
                hidden_n: BRAIN_HIDDEN_DEFAULT as u32,
                w1: [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN],
                b1: [0.0; BRAIN_HIDDEN],
                w2: [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
                b2: [0.0; BRAIN_OUTPUTS],
            };
            for h in 0..BRAIN_HIDDEN {
                for in_i in 0..BRAIN_INPUTS {
                    b.w1[h][in_i] = f[off + h * BRAIN_INPUTS + in_i];
                }
            }
            for h in 0..BRAIN_HIDDEN { b.b1[h] = f[off + B1_OFFSET + h]; }
            for o in 0..BRAIN_OUTPUTS {
                for h in 0..BRAIN_HIDDEN {
                    b.w2[o][h] = f[off + W2_OFFSET + o * BRAIN_HIDDEN + h];
                }
            }
            for o in 0..BRAIN_OUTPUTS { b.b2[o] = f[off + B2_OFFSET + o]; }
            out.push(b);
        }
        drop(data);
        self.brain_weights_rb.unmap();
        out
    }

    pub fn upload_inputs(&self, inputs: &[[f32; BRAIN_INPUTS]]) {
        let flat: Vec<f32> = inputs.iter().flatten().copied().collect();
        self.queue.write_buffer(&self.last_inputs_buf, 0, bytemuck::cast_slice(&flat));
    }

    pub fn upload_velocities(&self, velocities: &[[f32; 3]]) {
        let flat: Vec<f32> = velocities.iter().flatten().copied().collect();
        self.queue.write_buffer(&self.velocities_buf, 0, bytemuck::cast_slice(&flat));
    }

    pub fn upload_rewards(&self, rewards: &[f32]) {
        self.queue.write_buffer(&self.rewards_buf, 0, bytemuck::cast_slice(rewards));
    }

    /// Sprint 51: GPU-side copy slot[src] → slot[dst] pro brain_weights +
    /// xoshiro_state. Použito v die_and_drop_carrion swap_remove pattern —
    /// keď cell v dst slotu zemřela, src je poslední živá cell, která se
    /// přesune. NIC se ne-stahuje, NIC se ne-uploaduje — pure GPU memcpy.
    pub fn swap_to(&self, dst: usize, src: usize) {
        assert!(dst < self.capacity && src < self.capacity);
        if dst == src {
            return;
        }
        let brain_bytes = (BRAIN_WEIGHTS_PER_CELL * 4) as u64;
        let brain_src = (src * BRAIN_WEIGHTS_PER_CELL * 4) as u64;
        let brain_dst = (dst * BRAIN_WEIGHTS_PER_CELL * 4) as u64;
        let xosh_bytes = 4u64 * 4;
        let xosh_src = (src * 4 * 4) as u64;
        let xosh_dst = (dst * 4 * 4) as u64;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cells-swap"),
        });
        // wgpu nepovoluje same-buffer copy → routujeme přes staging temps.
        encoder.copy_buffer_to_buffer(
            &self.brain_weights_buf,
            brain_src,
            &self.swap_brain_temp,
            0,
            brain_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &self.swap_brain_temp,
            0,
            &self.brain_weights_buf,
            brain_dst,
            brain_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &self.xoshiro_state_buf,
            xosh_src,
            &self.swap_xoshiro_temp,
            0,
            xosh_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &self.swap_xoshiro_temp,
            0,
            &self.xoshiro_state_buf,
            xosh_dst,
            xosh_bytes,
        );
        let tr_bytes = std::mem::size_of::<f32>() as u64;
        let tr_src = src as u64 * tr_bytes;
        let tr_dst = dst as u64 * tr_bytes;
        encoder.copy_buffer_to_buffer(
            &self.turn_rate_buf,
            tr_src,
            &self.swap_turn_rate_temp,
            0,
            tr_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &self.swap_turn_rate_temp,
            0,
            &self.turn_rate_buf,
            tr_dst,
            tr_bytes,
        );
        self.queue.submit(Some(encoder.finish()));
    }

    /// Sprint 51: seed xoshiro state pro konkrétní slot. Použito po reproduce
    /// (nová cell potřebuje fresh state).
    pub fn upload_xoshiro_seed_at(&self, slot: usize, seed: u64) {
        assert!(slot < self.capacity);
        fn splitmix(z: &mut u64) -> u64 {
            *z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = *z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        }
        let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
        let a = splitmix(&mut z);
        let b = splitmix(&mut z);
        let mut s0 = a as u32;
        let s1 = (a >> 32) as u32;
        let s2 = b as u32;
        let s3 = (b >> 32) as u32;
        let mut state = [s0, s1, s2, s3];
        if state == [0u32; 4] {
            s0 = 1;
            state[0] = s0;
        }
        let offset = (slot * 4 * 4) as u64;
        self.queue.write_buffer(&self.xoshiro_state_buf, offset, bytemuck::cast_slice(&state));
    }

    /// Inicializuj per-cell xoshiro state z deterministic seeds. SplitMix64
    /// rozšíří 64-bit seed na 4× 32-bit xoshiro state. Protect proti all-zero
    /// state (xoshiro vyžaduje aspoň jednu non-zero word).
    pub fn upload_xoshiro_seeds<I>(&self, seeds: I)
    where
        I: IntoIterator<Item = u64>,
    {
        fn splitmix(z: &mut u64) -> u64 {
            *z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = *z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        }
        let mut state: Vec<u32> = Vec::with_capacity(self.capacity * 4);
        for s in seeds {
            let mut z = s.wrapping_add(0x9E3779B97F4A7C15);
            let a = splitmix(&mut z);
            let b = splitmix(&mut z);
            let mut s0 = a as u32;
            let s1 = (a >> 32) as u32;
            let s2 = b as u32;
            let s3 = (b >> 32) as u32;
            if s0 == 0 && s1 == 0 && s2 == 0 && s3 == 0 {
                s0 = 1;
            }
            state.push(s0);
            state.push(s1);
            state.push(s2);
            state.push(s3);
        }
        self.queue.write_buffer(&self.xoshiro_state_buf, 0, bytemuck::cast_slice(&state));
    }

    /// Stáhne (last_hidden, last_outputs) jako Vec — caller je potřebuje pro
    /// motor + apply_morph fáze (CPU). Per-tick, kritická pro --gpu-full.
    pub fn download_hidden_outputs(
        &self,
        n: usize,
    ) -> (Vec<[f32; BRAIN_HIDDEN]>, Vec<[f32; BRAIN_OUTPUTS]>) {
        if n == 0 { return (Vec::new(), Vec::new()); }
        assert!(n <= self.capacity);
        let h_bytes = (n * BRAIN_HIDDEN * 4) as u64;
        let o_bytes = (n * BRAIN_OUTPUTS * 4) as u64;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cells-download-ho"),
        });
        encoder.copy_buffer_to_buffer(&self.last_hidden_buf, 0, &self.last_hidden_rb, 0, h_bytes);
        encoder.copy_buffer_to_buffer(&self.last_outputs_buf, 0, &self.last_outputs_rb, 0, o_bytes);
        self.queue.submit(Some(encoder.finish()));
        let h_s = self.last_hidden_rb.slice(0..h_bytes);
        let o_s = self.last_outputs_rb.slice(0..o_bytes);
        h_s.map_async(wgpu::MapMode::Read, |_| {});
        o_s.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let h_data = h_s.get_mapped_range();
        let o_data = o_s.get_mapped_range();
        let h_f: &[f32] = bytemuck::cast_slice(&h_data);
        let o_f: &[f32] = bytemuck::cast_slice(&o_data);
        let hidden: Vec<[f32; BRAIN_HIDDEN]> = (0..n)
            .map(|i| {
                let mut a = [0.0_f32; BRAIN_HIDDEN];
                a.copy_from_slice(&h_f[i * BRAIN_HIDDEN..(i + 1) * BRAIN_HIDDEN]);
                a
            })
            .collect();
        let outputs: Vec<[f32; BRAIN_OUTPUTS]> = (0..n)
            .map(|i| {
                let mut a = [0.0_f32; BRAIN_OUTPUTS];
                a.copy_from_slice(&o_f[i * BRAIN_OUTPUTS..(i + 1) * BRAIN_OUTPUTS]);
                a
            })
            .collect();
        drop(h_data);
        drop(o_data);
        self.last_hidden_rb.unmap();
        self.last_outputs_rb.unmap();
        (hidden, outputs)
    }

    /// Stáhne brain weights pro jeden slot. Použito v reproduce phase
    /// (download parent brains z GPU pro crossover/mutate na CPU).
    pub fn download_brain_at(&self, idx: usize) -> Brain {
        assert!(idx < self.capacity);
        let bytes = (BRAIN_WEIGHTS_PER_CELL * 4) as u64;
        let offset = (idx * BRAIN_WEIGHTS_PER_CELL * 4) as u64;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cells-download-brain-slot"),
        });
        encoder.copy_buffer_to_buffer(&self.brain_weights_buf, offset, &self.brain_weights_rb, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let s = self.brain_weights_rb.slice(0..bytes);
        s.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = s.get_mapped_range();
        let f: &[f32] = bytemuck::cast_slice(&data);
        let mut b = Brain {
            hidden_n: BRAIN_HIDDEN_DEFAULT as u32,
            w1: [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN],
            b1: [0.0; BRAIN_HIDDEN],
            w2: [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
            b2: [0.0; BRAIN_OUTPUTS],
        };
        for h in 0..BRAIN_HIDDEN {
            for in_i in 0..BRAIN_INPUTS {
                b.w1[h][in_i] = f[h * BRAIN_INPUTS + in_i];
            }
        }
        for h in 0..BRAIN_HIDDEN { b.b1[h] = f[B1_OFFSET + h]; }
        for o in 0..BRAIN_OUTPUTS {
            for h in 0..BRAIN_HIDDEN {
                b.w2[o][h] = f[W2_OFFSET + o * BRAIN_HIDDEN + h];
            }
        }
        for o in 0..BRAIN_OUTPUTS { b.b2[o] = f[B2_OFFSET + o]; }
        drop(data);
        self.brain_weights_rb.unmap();
        b
    }

    pub fn download_velocities(&self, n: usize) -> Vec<[f32; 3]> {
        if n == 0 { return Vec::new(); }
        assert!(n <= self.capacity);
        let bytes = (n * 3 * 4) as u64;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cells-download-vel"),
        });
        encoder.copy_buffer_to_buffer(&self.velocities_buf, 0, &self.velocities_rb, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let s = self.velocities_rb.slice(0..bytes);
        s.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = s.get_mapped_range();
        let f: &[f32] = bytemuck::cast_slice(&data);
        let out: Vec<[f32; 3]> = (0..n).map(|i| [f[i*3], f[i*3+1], f[i*3+2]]).collect();
        drop(data);
        self.velocities_rb.unmap();
        out
    }
}

