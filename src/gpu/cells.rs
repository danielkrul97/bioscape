use std::sync::Arc;


use crate::*;
use super::*;

// CellsGpu persistent SoA state. Keeps brain weights, last_inputs/hidden/
// outputs, velocities, and the per-cell xoshiro128++ RNG resident on the GPU
// across ticks. The original upload-each-tick path moved 30 MB/tick of brain
// weights — this struct removes that bottleneck.

/// Persistent per-cell GPU state. Holds the brain forward state
/// (`last_inputs`, `last_hidden`, `last_outputs`, brain weights) plus
/// velocities (for brownian) and the per-cell xoshiro128++ RNG. Subsequent
/// fields were added so the full-GPU pipeline (motor, step, sensor gather)
/// can also bind shared buffers without internal duplicates.
///
/// Lifecycle:
/// 1. `new(ctx, capacity)` allocates buffers and seeds the xoshiro state.
/// 2. `upload_brains(brains, init_xoshiro_seed)` at simulation init.
/// 3. Hot loop:
///    - `upload_inputs(last_inputs)` before brain forward.
///    - `forward_batch_persistent(brain_gpu)`.
///    - `download_hidden_outputs() -> (Vec<hidden>, Vec<outputs>)`.
///    - `upload_velocities(velocities)` before brownian.
///    - `brownian_persistent(...)` — mutates velocities + RNG in place.
///    - `download_velocities()` after brownian.
///    - `upload_rewards(rewards)` after eat_food.
///    - `hebbian_persistent(lr)` — mutates brain weights in place.
/// 4. `upload_brain_at(idx, brain)` after a reproduction event.
/// 5. `download_brains()` for checkpointing or introspection.
pub struct CellsGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    capacity: usize,
    last_inputs_buf: wgpu::Buffer,
    last_hidden_buf: wgpu::Buffer,
    last_outputs_buf: wgpu::Buffer,
    brain_weights_buf: wgpu::Buffer,
    /// Wave 7: per-cell eligibility-trace shadow, parallel to
    /// `brain_weights_buf`. Same `BRAIN_WEIGHTS_PER_CELL` stride; w1/w2
    /// regions are decayed+accumulated each tick by `hebbian_step.wgsl`,
    /// b1/b2 slots stay at zero (biases ride on activations at reward time).
    /// Initialized to zero so first step starts clean.
    brain_traces_buf: wgpu::Buffer,
    /// Sprint 195: per-cell whisker spring-damper state — 12 f32 per cell,
    /// layout `[deflection0..6, velocity0..6]`. Persistent across ticks,
    /// mutated in place by `sensor_gather.wgsl`. Zeroed on allocation and on
    /// every slot recycle (`reset_persistent_brain_state_at`).
    whisker_state_buf: wgpu::Buffer,
    velocities_buf: wgpu::Buffer,
    xoshiro_state_buf: wgpu::Buffer,
    rewards_buf: wgpu::Buffer,
    /// Sprint 137: per-cell Hebbian learning rate, read by
    /// `hebbian_apply_reward.wgsl`. Initialised at allocation to the
    /// pre-S137 global `LEARNING_RATE`; uploaded from `Genome.learning_rate`
    /// every reproduce phase so newborns and post-swap slots track the
    /// child's gene.
    learning_rates_buf: wgpu::Buffer,
    /// Sprint 137: per-cell eligibility-trace decay (1/s), read by
    /// `hebbian_step.wgsl`. Same init/upload story as `learning_rates_buf`.
    trace_decays_buf: wgpu::Buffer,
    /// Sprint 187: per-cell `genome.reproduce_at_energy`, read by
    /// `populate_inputs.wgsl` for the energy-fullness brain input (slot 4).
    /// Initialised at allocation to the pre-S187 global `REPRODUCE_THRESHOLD`;
    /// uploaded from genome every reproduce phase so newborns track their
    /// per-cell threshold from tick 0.
    reproduce_at_energies_buf: wgpu::Buffer,
    /// Sprint 198: per-cell symbiont presence flag (0 = no symbiont, 1 = has).
    /// Read by `populate_inputs.wgsl` for brain input slot 39 (has_symbiont).
    /// Default fill = 0; CPU uploads each tick before the brain pass.
    symbiont_has_buf: wgpu::Buffer,
    /// Sprint 198: per-cell `symbiont.deficit_streak` (u32 ticks). Read by
    /// `populate_inputs.wgsl`; divided by SYMBIONT_UPKEEP_DEFICIT_TICKS in the
    /// shader to normalize to [0, 1] for brain input slot 40 (deficit_norm).
    symbiont_deficit_buf: wgpu::Buffer,
    /// Sprint 145: per-cell, per-hidden-neuron Izhikevich membrane
    /// potential `v`. Layout: `[capacity × BRAIN_HIDDEN]` f32, row-major
    /// (cell i hidden h at index `i * BRAIN_HIDDEN + h`). Initialised at
    /// allocation to `IZH_V_REST = -65.0`; Izhikevich forward (S146/S147)
    /// reads and mutates; Perceptron path ignores it.
    membrane_buf: wgpu::Buffer,
    /// Sprint 145: matching recovery variable `u`. Default 0.0.
    recovery_buf: wgpu::Buffer,
    /// Sprint 147: per-cell `NeuronModel` discriminator (u32: 0 = Perceptron,
    /// 1 = Izhikevich). The Izhikevich forward shader early-exits for cells
    /// whose entry isn't `Izhikevich`. Initialised to all-zeros so a fresh
    /// capacity defaults to Perceptron (matches pre-S147 behavior).
    neuron_models_buf: wgpu::Buffer,
    /// Sprint 163: per-cell, per-input last spike tick (u32). Layout
    /// `[capacity × BRAIN_INPUTS]`. Pre-spike encoder shader (S165)
    /// stamps this when `inputs[i] > SPIKE_ENCODE_THRESHOLD`. STDP rule
    /// reads to determine LTD timing windows.
    pre_spike_times_buf: wgpu::Buffer,
    /// Sprint 163: per-cell, per-hidden last spike tick (u32). Layout
    /// `[capacity × BRAIN_HIDDEN]`. Izhikevich forward shader (S164)
    /// stamps this when membrane crosses threshold. STDP rule reads to
    /// determine LTP timing windows.
    post_spike_times_buf: wgpu::Buffer,
    /// Sprint 163: per-cell, per-input STDP trace (f32). Layout
    /// `[capacity × BRAIN_INPUTS]`. Decayed + accumulated by stdp_step
    /// shader (S166). LTP magnitude on post-spike = `a_plus × pre_trace[i]`.
    pre_trace_buf: wgpu::Buffer,
    /// Sprint 163: per-cell, per-hidden STDP trace (f32). Layout
    /// `[capacity × BRAIN_HIDDEN]`. LTD magnitude on pre-spike = `a_minus
    /// × post_trace[h]`.
    post_trace_buf: wgpu::Buffer,
    /// Cell metadata fed to the `populate_brain_inputs` shader.
    /// `energy`, `heading`, `pitch`, `damage_accum` are uploaded per tick;
    /// `max_speed` and `eff_radius` only change on reproduce / morph but the
    /// per-tick upload is cheap (~24 KB total at N=1000).
    energy_buf: wgpu::Buffer,
    heading_buf: wgpu::Buffer,
    pitch_buf: wgpu::Buffer,
    damage_accum_buf: wgpu::Buffer,
    max_speed_buf: wgpu::Buffer,
    eff_radius_buf: wgpu::Buffer,
    /// Bond-mediated communication inbox per cell, written each tick from CPU
    /// (mean of bonded peers' last_outputs message channels). Read by
    /// populate_inputs shader into brain inputs slots [27..29].
    bonded_inbox_buf: wgpu::Buffer,
    /// Motor pipeline buffers. `turn_rate` is a per-cell genome constant;
    /// `angular_velocity` and `pitch_velocity` are mutated by the motor
    /// shader.
    turn_rate_buf: wgpu::Buffer,
    angular_velocity_buf: wgpu::Buffer,
    pitch_velocity_buf: wgpu::Buffer,
    angular_velocity_rb: wgpu::Buffer,
    pitch_velocity_rb: wgpu::Buffer,
    /// Step pipeline buffers. Position is mutated each tick by integrate +
    /// bounce; age/cooldown are incremented; `body_dims` only changes on
    /// morph; `aux` (spike/shell/vision/attack) is recomputed each tick.
    position_buf: wgpu::Buffer,
    age_buf: wgpu::Buffer,
    cooldown_buf: wgpu::Buffer,
    body_dims_buf: wgpu::Buffer,
    aux_buf: wgpu::Buffer,
    position_rb: wgpu::Buffer,
    age_rb: wgpu::Buffer,
    cooldown_rb: wgpu::Buffer,
    energy_rb: wgpu::Buffer,
    last_inputs_rb: wgpu::Buffer,
    last_hidden_rb: wgpu::Buffer,
    last_outputs_rb: wgpu::Buffer,
    velocities_rb: wgpu::Buffer,
    brain_weights_rb: wgpu::Buffer,
    /// Staging buffers for `swap_to` — wgpu disallows same-buffer copies.
    swap_brain_temp: wgpu::Buffer,
    swap_xoshiro_temp: wgpu::Buffer,
    swap_turn_rate_temp: wgpu::Buffer,
    epoch: u64,
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
        // Wave 7: per-cell eligibility-trace shadow buffer, parallel to
        // brain_weights. Same WEIGHTS_PER_CELL stride; only the w1 / w2
        // sub-regions are actively updated by `hebbian_step.wgsl` (bias
        // slots are written zero on init and never touched). Initial value
        // is zero everywhere — `cells_buffer_init_zero` sets that on
        // allocation by virtue of `mk`'s zeroed default.
        let brain_traces_buf = mk("cells-brain-traces", n * (BRAIN_WEIGHTS_PER_CELL as u64) * f, stor_dst_src);
        // Zero the trace buffer explicitly so the first hebbian_step starts
        // from a clean state (initial decay·0 + pre·post = pre·post).
        queue.write_buffer(
            &brain_traces_buf,
            0,
            &vec![0u8; (n * (BRAIN_WEIGHTS_PER_CELL as u64) * f) as usize],
        );
        // Sprint 195: whisker spring-damper state, 12 f32 per cell
        // (deflection ×6, velocity ×6). Zeroed so the first sensor_gather
        // dispatch starts every whisker at rest (deflection 0 = straight).
        let whisker_state_buf = mk("cells-whisker-state", n * 12 * f, stor_dst_src);
        queue.write_buffer(&whisker_state_buf, 0, &vec![0u8; (n * 12 * f) as usize]);
        let velocities_buf = mk("cells-velocities", n * 3 * f, stor_dst_src);
        let xoshiro_state_buf = mk("cells-xoshiro", n * 4 * 4, stor_dst_src);
        let rewards_buf = mk(
            "cells-rewards",
            n * f,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        // Sprint 137: per-cell Hebbian rates. COPY_SRC included so `swap_to`
        // can route slot data through the staging temp. Initial fill at the
        // pre-S137 globals so a fresh capacity (no upload yet) still produces
        // the legacy uniform-rate behaviour at the first dispatch.
        let learning_rates_buf = mk("cells-learning-rates", n * f, stor_dst_src);
        let trace_decays_buf = mk("cells-trace-decays", n * f, stor_dst_src);
        let default_lr_fill = vec![LEARNING_RATE; capacity];
        queue.write_buffer(
            &learning_rates_buf,
            0,
            bytemuck::cast_slice(&default_lr_fill),
        );
        let default_decay_fill = vec![HEBBIAN_TRACE_DECAY_PER_SEC; capacity];
        queue.write_buffer(
            &trace_decays_buf,
            0,
            bytemuck::cast_slice(&default_decay_fill),
        );
        let reproduce_at_energies_buf =
            mk("cells-reproduce-at-energies", n * f, stor_dst_src);
        let default_repro_fill = vec![REPRODUCE_THRESHOLD; capacity];
        queue.write_buffer(
            &reproduce_at_energies_buf,
            0,
            bytemuck::cast_slice(&default_repro_fill),
        );
        // Sprint 198: symbiont state buffers. Both default-filled to zero so
        // a population with no bearers reads has=0, deficit=0 on every cell.
        let u32_size = std::mem::size_of::<u32>() as u64;
        let symbiont_has_buf =
            mk("cells-symbiont-has", n * u32_size, stor_dst_src);
        let symbiont_deficit_buf =
            mk("cells-symbiont-deficit", n * u32_size, stor_dst_src);
        let default_zero_u32 = vec![0u32; capacity];
        queue.write_buffer(
            &symbiont_has_buf,
            0,
            bytemuck::cast_slice(&default_zero_u32),
        );
        queue.write_buffer(
            &symbiont_deficit_buf,
            0,
            bytemuck::cast_slice(&default_zero_u32),
        );
        // Sprint 145: Izhikevich (v, u) buffers, sized like `last_hidden`.
        // Initial fill at resting potential / zero recovery so any cell
        // that's flipped to `NeuronModel::Izhikevich` mid-run starts the
        // membrane dynamics from rest.
        let membrane_buf = mk(
            "cells-membrane",
            n * (BRAIN_HIDDEN as u64) * f,
            stor_dst_src,
        );
        let recovery_buf = mk(
            "cells-recovery",
            n * (BRAIN_HIDDEN as u64) * f,
            stor_dst_src,
        );
        let default_membrane_fill = vec![IZH_V_REST; capacity * BRAIN_HIDDEN];
        queue.write_buffer(
            &membrane_buf,
            0,
            bytemuck::cast_slice(&default_membrane_fill),
        );
        // Zero-init recovery — write_buffer with zero slice is cheap.
        let zero_recovery = vec![0.0_f32; capacity * BRAIN_HIDDEN];
        queue.write_buffer(&recovery_buf, 0, bytemuck::cast_slice(&zero_recovery));
        // Sprint 147: neuron_models default all-zero = Perceptron.
        let neuron_models_buf = mk(
            "cells-neuron-models",
            (capacity as u64) * (std::mem::size_of::<u32>() as u64),
            stor_dst_src,
        );
        let zero_models = vec![0_u32; capacity];
        queue.write_buffer(
            &neuron_models_buf,
            0,
            bytemuck::cast_slice(&zero_models),
        );
        // Sprint 163: STDP spike-time + trace buffers, zero-initialised.
        let u32sz = std::mem::size_of::<u32>() as u64;
        let pre_spike_times_buf = mk(
            "cells-pre-spike-times",
            n * (BRAIN_INPUTS as u64) * u32sz,
            stor_dst_src,
        );
        let post_spike_times_buf = mk(
            "cells-post-spike-times",
            n * (BRAIN_HIDDEN as u64) * u32sz,
            stor_dst_src,
        );
        let pre_trace_buf = mk(
            "cells-pre-trace",
            n * (BRAIN_INPUTS as u64) * f,
            stor_dst_src,
        );
        let post_trace_buf = mk(
            "cells-post-trace",
            n * (BRAIN_HIDDEN as u64) * f,
            stor_dst_src,
        );
        let zero_pre_u32 = vec![0_u32; capacity * BRAIN_INPUTS];
        let zero_post_u32 = vec![0_u32; capacity * BRAIN_HIDDEN];
        let zero_pre_f32 = vec![0.0_f32; capacity * BRAIN_INPUTS];
        let zero_post_f32 = vec![0.0_f32; capacity * BRAIN_HIDDEN];
        queue.write_buffer(&pre_spike_times_buf, 0, bytemuck::cast_slice(&zero_pre_u32));
        queue.write_buffer(&post_spike_times_buf, 0, bytemuck::cast_slice(&zero_post_u32));
        queue.write_buffer(&pre_trace_buf, 0, bytemuck::cast_slice(&zero_pre_f32));
        queue.write_buffer(&post_trace_buf, 0, bytemuck::cast_slice(&zero_post_f32));
        // populate_inputs metadata.
        let energy_buf = mk("cells-energy", n * f, stor_dst_src);
        let heading_buf = mk("cells-heading", n * f, stor_dst_src);
        let pitch_buf = mk("cells-pitch", n * f, stor_dst_src);
        let damage_accum_buf = mk("cells-damage", n * f, stor_dst_src);
        let max_speed_buf = mk("cells-max-speed", n * f, stor_dst_src);
        let eff_radius_buf = mk("cells-eff-radius", n * f, stor_dst_src);
        let bonded_inbox_buf = mk(
            "cells-bonded-inbox",
            n * (N_BOND_MSG_CHANNELS as u64) * f,
            stor_dst_src,
        );
        // Motor.
        let turn_rate_buf = mk("cells-turn-rate", n * f, stor_dst_src);
        let angular_velocity_buf = mk("cells-ang-vel", n * f, stor_dst_src);
        let pitch_velocity_buf = mk("cells-pitch-vel", n * f, stor_dst_src);
        let angular_velocity_rb = mk("cells-ang-vel-rb", n * f, read);
        let pitch_velocity_rb = mk("cells-pitch-vel-rb", n * f, read);
        // Step.
        let position_buf = mk("cells-position", n * 3 * f, stor_dst_src);
        let age_buf = mk("cells-age", n * 4, stor_dst_src);
        let cooldown_buf = mk("cells-cooldown", n * 4, stor_dst_src);
        let body_dims_buf = mk("cells-body-dims", n * 3 * f, stor_dst_src);
        let aux_buf = mk("cells-aux", n * 4 * f, stor_dst_src);
        let position_rb = mk("cells-position-rb", n * 3 * f, read);
        let age_rb = mk("cells-age-rb", n * 4, read);
        let cooldown_rb = mk("cells-cooldown-rb", n * 4, read);
        let energy_rb = mk("cells-energy-rb", n * f, read);
        let last_inputs_rb = mk("cells-inputs-rb", n * (BRAIN_INPUTS as u64) * f, read);
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
            brain_traces_buf,
            whisker_state_buf,
            velocities_buf,
            xoshiro_state_buf,
            rewards_buf,
            learning_rates_buf,
            trace_decays_buf,
            reproduce_at_energies_buf,
            symbiont_has_buf,
            symbiont_deficit_buf,
            membrane_buf,
            recovery_buf,
            neuron_models_buf,
            pre_spike_times_buf,
            post_spike_times_buf,
            pre_trace_buf,
            post_trace_buf,
            energy_buf,
            heading_buf,
            pitch_buf,
            damage_accum_buf,
            max_speed_buf,
            eff_radius_buf,
            bonded_inbox_buf,
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
            last_inputs_rb,
            last_hidden_rb,
            last_outputs_rb,
            velocities_rb,
            brain_weights_rb,
            swap_brain_temp,
            swap_xoshiro_temp,
            swap_turn_rate_temp,
            epoch: 0,
        }
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
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
    /// Wave 7: per-cell eligibility-trace buffer (parallel to brain weights).
    pub fn brain_traces_buffer(&self) -> &wgpu::Buffer { &self.brain_traces_buf }
    /// Sprint 195: per-cell whisker spring-damper state (12 f32/cell).
    pub fn whisker_state_buffer(&self) -> &wgpu::Buffer { &self.whisker_state_buf }
    pub fn velocities_buffer(&self) -> &wgpu::Buffer { &self.velocities_buf }
    pub fn xoshiro_state_buffer(&self) -> &wgpu::Buffer { &self.xoshiro_state_buf }
    pub fn rewards_buffer(&self) -> &wgpu::Buffer { &self.rewards_buf }
    pub fn learning_rates_buffer(&self) -> &wgpu::Buffer { &self.learning_rates_buf }
    pub fn trace_decays_buffer(&self) -> &wgpu::Buffer { &self.trace_decays_buf }
    pub fn reproduce_at_energies_buffer(&self) -> &wgpu::Buffer { &self.reproduce_at_energies_buf }
    pub fn symbiont_has_buffer(&self) -> &wgpu::Buffer { &self.symbiont_has_buf }
    pub fn symbiont_deficit_buffer(&self) -> &wgpu::Buffer { &self.symbiont_deficit_buf }
    pub fn membrane_buffer(&self) -> &wgpu::Buffer { &self.membrane_buf }
    pub fn recovery_buffer(&self) -> &wgpu::Buffer { &self.recovery_buf }
    pub fn neuron_models_buffer(&self) -> &wgpu::Buffer { &self.neuron_models_buf }
    pub fn pre_spike_times_buffer(&self) -> &wgpu::Buffer { &self.pre_spike_times_buf }
    pub fn post_spike_times_buffer(&self) -> &wgpu::Buffer { &self.post_spike_times_buf }
    pub fn pre_trace_buffer(&self) -> &wgpu::Buffer { &self.pre_trace_buf }
    pub fn post_trace_buffer(&self) -> &wgpu::Buffer { &self.post_trace_buf }
    /// `populate_inputs` metadata buffer accessors.
    pub fn energy_buffer(&self) -> &wgpu::Buffer { &self.energy_buf }
    pub fn heading_buffer(&self) -> &wgpu::Buffer { &self.heading_buf }
    pub fn pitch_buffer(&self) -> &wgpu::Buffer { &self.pitch_buf }
    pub fn damage_accum_buffer(&self) -> &wgpu::Buffer { &self.damage_accum_buf }
    pub fn max_speed_buffer(&self) -> &wgpu::Buffer { &self.max_speed_buf }
    pub fn eff_radius_buffer(&self) -> &wgpu::Buffer { &self.eff_radius_buf }
    pub fn bonded_inbox_buffer(&self) -> &wgpu::Buffer { &self.bonded_inbox_buf }
    /// Motor accessors. `turn_rate` is read-only; angular/pitch velocities
    /// are read-write.
    pub fn turn_rate_buffer(&self) -> &wgpu::Buffer { &self.turn_rate_buf }
    pub fn angular_velocity_buffer(&self) -> &wgpu::Buffer { &self.angular_velocity_buf }
    pub fn pitch_velocity_buffer(&self) -> &wgpu::Buffer { &self.pitch_velocity_buf }
    /// Step accessors.
    pub fn position_buffer(&self) -> &wgpu::Buffer { &self.position_buf }
    pub fn age_buffer(&self) -> &wgpu::Buffer { &self.age_buf }
    pub fn cooldown_buffer(&self) -> &wgpu::Buffer { &self.cooldown_buf }
    pub fn body_dims_buffer(&self) -> &wgpu::Buffer { &self.body_dims_buf }
    pub fn aux_buffer(&self) -> &wgpu::Buffer { &self.aux_buf }

    /// Upload positions (3 × f32 per cell, packed AoS).
    pub fn upload_positions(&self, positions: &[[f32; 3]]) {
        // `[f32; 3]` is Pod; reinterpret the slice directly to avoid a per-tick
        // repacking Vec.
        self.queue.write_buffer(&self.position_buf, 0, bytemuck::cast_slice(positions));
    }

    /// Upload age + cooldown (u32 per cell).
    pub fn upload_age_cooldown(&self, ages: &[u32], cooldowns: &[u32]) {
        debug_assert_eq!(ages.len(), cooldowns.len());
        self.queue.write_buffer(&self.age_buf, 0, bytemuck::cast_slice(ages));
        self.queue.write_buffer(&self.cooldown_buf, 0, bytemuck::cast_slice(cooldowns));
    }

    /// Upload body dimensions (length, width, height per cell).
    pub fn upload_body_dims(&self, body_dims: &[[f32; 3]]) {
        self.queue.write_buffer(&self.body_dims_buf, 0, bytemuck::cast_slice(body_dims));
    }

    /// Upload aux (spike, shell, vision, attack per cell). Attack is the
    /// only per-tick element (recomputed from `output[6]`); the rest are
    /// genome constants but uploaded together for cache simplicity.
    pub fn upload_aux(&self, aux: &[[f32; 4]]) {
        self.queue.write_buffer(&self.aux_buf, 0, bytemuck::cast_slice(aux));
    }

    /// Combined batch readback after the brain_act + step pipeline. Single
    /// `Wait` barrier covers all 10 buffers; results are written into the
    /// caller-provided scratch slots (clear + extend pattern) so a stable
    /// capacity yields zero allocations per tick.
    ///
    /// `inputs_out` carries the per-cell `last_inputs` array straight from
    /// the GPU `populate_inputs` shader — pre-S188 this stayed GPU-only,
    /// leaving the CPU mirror frozen at spawn-time zeros and the inspector
    /// brain panel showing a lie. Same `Wait` barrier as the rest.
    #[allow(clippy::too_many_arguments)]
    pub fn download_full_batch_into(
        &self,
        n: usize,
        inputs_out: &mut Vec<[f32; BRAIN_INPUTS]>,
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
            inputs_out.clear();
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
            inputs_out,
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
        let i_bytes = (n * BRAIN_INPUTS * 4) as u64;
        let h_bytes = (n * BRAIN_HIDDEN * 4) as u64;
        let o_bytes = (n * BRAIN_OUTPUTS * 4) as u64;
        let v_bytes = (n * 3 * 4) as u64;
        let a_bytes = (n * 4) as u64;
        let p_bytes = (n * 4) as u64;
        let pos_bytes = (n * 3 * 4) as u64;
        let age_bytes = (n * 4) as u64;
        let cd_bytes = (n * 4) as u64;
        let e_bytes = (n * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.last_inputs_buf, 0, &self.last_inputs_rb, 0, i_bytes);
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
        inputs_out: &mut Vec<[f32; BRAIN_INPUTS]>,
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
        inputs_out.clear();
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
        let i_bytes = (n * BRAIN_INPUTS * 4) as u64;
        let h_bytes = (n * BRAIN_HIDDEN * 4) as u64;
        let o_bytes = (n * BRAIN_OUTPUTS * 4) as u64;
        let v_bytes = (n * 3 * 4) as u64;
        let a_bytes = (n * 4) as u64;
        let p_bytes = (n * 4) as u64;
        let pos_bytes = (n * 3 * 4) as u64;
        let age_bytes = (n * 4) as u64;
        let cd_bytes = (n * 4) as u64;
        let e_bytes = (n * 4) as u64;
        let i_s = self.last_inputs_rb.slice(0..i_bytes);
        let h_s = self.last_hidden_rb.slice(0..h_bytes);
        let o_s = self.last_outputs_rb.slice(0..o_bytes);
        let v_s = self.velocities_rb.slice(0..v_bytes);
        let a_s = self.angular_velocity_rb.slice(0..a_bytes);
        let p_s = self.pitch_velocity_rb.slice(0..p_bytes);
        let pos_s = self.position_rb.slice(0..pos_bytes);
        let age_s = self.age_rb.slice(0..age_bytes);
        let cd_s = self.cooldown_rb.slice(0..cd_bytes);
        let e_s = self.energy_rb.slice(0..e_bytes);
        for s in [&i_s, &h_s, &o_s, &v_s, &a_s, &p_s, &pos_s, &age_s, &cd_s, &e_s] {
            s.map_async(wgpu::MapMode::Read, |_| {});
        }
        self.device.poll(wgpu::Maintain::Wait);
        let i_data = i_s.get_mapped_range();
        let h_data = h_s.get_mapped_range();
        let o_data = o_s.get_mapped_range();
        let v_data = v_s.get_mapped_range();
        let a_data = a_s.get_mapped_range();
        let p_data = p_s.get_mapped_range();
        let pos_data = pos_s.get_mapped_range();
        let age_data = age_s.get_mapped_range();
        let cd_data = cd_s.get_mapped_range();
        let e_data = e_s.get_mapped_range();
        let i_f: &[f32] = bytemuck::cast_slice(&i_data);
        let h_f: &[f32] = bytemuck::cast_slice(&h_data);
        let o_f: &[f32] = bytemuck::cast_slice(&o_data);
        let v_f: &[f32] = bytemuck::cast_slice(&v_data);
        let a_f: &[f32] = bytemuck::cast_slice(&a_data);
        let p_f: &[f32] = bytemuck::cast_slice(&p_data);
        let pos_f: &[f32] = bytemuck::cast_slice(&pos_data);
        let age_u: &[u32] = bytemuck::cast_slice(&age_data);
        let cd_u: &[u32] = bytemuck::cast_slice(&cd_data);
        let e_f: &[f32] = bytemuck::cast_slice(&e_data);
        // Reinterpret flat readback slices as arrays of fixed-size chunks; one
        // memcpy per output Vec instead of per-element pushes.
        let inputs_chunks: &[[f32; BRAIN_INPUTS]] = bytemuck::cast_slice(i_f);
        inputs_out.extend_from_slice(&inputs_chunks[..n]);
        let hidden_chunks: &[[f32; BRAIN_HIDDEN]] = bytemuck::cast_slice(h_f);
        hidden_out.extend_from_slice(&hidden_chunks[..n]);
        let outputs_chunks: &[[f32; BRAIN_OUTPUTS]] = bytemuck::cast_slice(o_f);
        outputs_out.extend_from_slice(&outputs_chunks[..n]);
        let vel_chunks: &[[f32; 3]] = bytemuck::cast_slice(v_f);
        velocities_out.extend_from_slice(&vel_chunks[..n]);
        let pos_chunks: &[[f32; 3]] = bytemuck::cast_slice(pos_f);
        positions_out.extend_from_slice(&pos_chunks[..n]);
        angular_out.extend_from_slice(&a_f[..n]);
        pitch_out.extend_from_slice(&p_f[..n]);
        ages_out.extend_from_slice(&age_u[..n]);
        cooldowns_out.extend_from_slice(&cd_u[..n]);
        energies_out.extend_from_slice(&e_f[..n]);
        drop(i_data); drop(h_data); drop(o_data); drop(v_data); drop(a_data); drop(p_data);
        drop(pos_data); drop(age_data); drop(cd_data); drop(e_data);
        self.last_inputs_rb.unmap();
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

    /// `turn_rate` is a per-cell genome constant — uploaded once at sim
    /// init and again on reproduce (sparse). For per-tick mutable metadata
    /// (energy / heading / pitch / damage / max_speed / eff_radius) use
    /// `upload_metadata`.
    pub fn upload_turn_rates(&self, turn_rates: &[f32]) {
        self.queue.write_buffer(&self.turn_rate_buf, 0, bytemuck::cast_slice(turn_rates));
    }

    pub fn upload_turn_rate_at(&self, slot: usize, turn_rate: f32) {
        assert!(slot < self.capacity);
        let offset = (slot * std::mem::size_of::<f32>()) as u64;
        self.queue.write_buffer(&self.turn_rate_buf, offset, bytemuck::cast_slice(&[turn_rate]));
    }

    /// Upload current angular + pitch velocities. Run before
    /// `brain_act_gpu_full`.
    pub fn upload_angular_pitch(&self, angular: &[f32], pitches: &[f32]) {
        debug_assert_eq!(angular.len(), pitches.len());
        self.queue.write_buffer(&self.angular_velocity_buf, 0, bytemuck::cast_slice(angular));
        self.queue.write_buffer(&self.pitch_velocity_buf, 0, bytemuck::cast_slice(pitches));
    }

    /// Batch readback of motor results (velocities + angular + pitch) under
    /// a single `Wait` barrier instead of three separate downloads. Run
    /// after the motor + brownian dispatches in `brain_act_gpu_full`.
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

    /// Combined batch readback of hidden + outputs + motor state under one
    /// `Wait` barrier. Run at the end of `brain_act_gpu_full` after the
    /// `brain.forward` + `motor.dispatch` + `brownian.dispatch` sequence.
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

    /// Bulk upload of `populate_inputs` metadata. Six per-cell f32 arrays
    /// in one call — slice lengths must all match.
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
                trace_w1: [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN],
                trace_w2: [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
                membrane: [IZH_V_REST; BRAIN_HIDDEN],
                recovery: [0.0; BRAIN_HIDDEN],
                last_pre_spike_ticks: [0; BRAIN_INPUTS],
                last_post_spike_ticks: [0; BRAIN_HIDDEN],
                pre_trace: [0.0; BRAIN_INPUTS],
                post_trace: [0.0; BRAIN_HIDDEN],
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
        self.queue.write_buffer(&self.last_inputs_buf, 0, bytemuck::cast_slice(inputs));
    }

    pub fn upload_bonded_inboxes(
        &self,
        inboxes: &[[f32; N_BOND_MSG_CHANNELS]],
    ) {
        self.queue.write_buffer(
            &self.bonded_inbox_buf,
            0,
            bytemuck::cast_slice(inboxes),
        );
    }

    pub fn upload_velocities(&self, velocities: &[[f32; 3]]) {
        self.queue.write_buffer(&self.velocities_buf, 0, bytemuck::cast_slice(velocities));
    }

    pub fn upload_rewards(&self, rewards: &[f32]) {
        self.queue.write_buffer(&self.rewards_buf, 0, bytemuck::cast_slice(rewards));
    }

    /// Sprint 137: full-population upload of per-cell `genome.learning_rate`
    /// and `genome.trace_decay_per_sec`. Called once per reproduce phase
    /// (after births append) so newborn slots reflect their genome before
    /// the next Hebbian dispatch fires.
    pub fn upload_learning_rates(&self, rates: &[f32]) {
        self.queue
            .write_buffer(&self.learning_rates_buf, 0, bytemuck::cast_slice(rates));
    }

    pub fn upload_trace_decays(&self, decays: &[f32]) {
        self.queue
            .write_buffer(&self.trace_decays_buf, 0, bytemuck::cast_slice(decays));
    }

    /// Sprint 187: full-population upload of per-cell `genome.reproduce_at_energy`.
    /// Mirrors the `upload_learning_rates` cadence — once per reproduce phase
    /// so newborn slots track their per-cell threshold before the next
    /// `populate_inputs` dispatch reads the value.
    pub fn upload_reproduce_at_energies(&self, thresholds: &[f32]) {
        self.queue.write_buffer(
            &self.reproduce_at_energies_buf,
            0,
            bytemuck::cast_slice(thresholds),
        );
    }

    pub fn upload_reproduce_at_energy_at(&self, slot: usize, value: f32) {
        assert!(slot < self.capacity);
        let f = std::mem::size_of::<f32>() as u64;
        self.queue.write_buffer(
            &self.reproduce_at_energies_buf,
            slot as u64 * f,
            bytemuck::bytes_of(&value),
        );
    }

    /// Sprint 198: full-population upload of symbiont state. `has[i]` = 1 if
    /// cell i is a bearer, 0 otherwise. `deficit[i]` = bearer's `deficit_streak`,
    /// 0 for non-bearers. Called from `World::tick` before the brain dispatch.
    pub fn upload_symbiont_state(&self, has: &[u32], deficit: &[u32]) {
        assert_eq!(has.len(), deficit.len());
        self.queue
            .write_buffer(&self.symbiont_has_buf, 0, bytemuck::cast_slice(has));
        self.queue
            .write_buffer(&self.symbiont_deficit_buf, 0, bytemuck::cast_slice(deficit));
    }

    /// Sprint 137: write the rates for a single slot. Used by post-swap_to
    /// path so the moved cell's gene-encoded rates land in its new slot
    /// without re-uploading the entire population.
    pub fn upload_rates_at(&self, slot: usize, learning_rate: f32, trace_decay: f32) {
        assert!(slot < self.capacity);
        let f = std::mem::size_of::<f32>() as u64;
        self.queue.write_buffer(
            &self.learning_rates_buf,
            slot as u64 * f,
            bytemuck::bytes_of(&learning_rate),
        );
        self.queue.write_buffer(
            &self.trace_decays_buf,
            slot as u64 * f,
            bytemuck::bytes_of(&trace_decay),
        );
    }

    /// Sprint 147: full-population upload of per-cell neuron model
    /// discriminators. Called at `init_gpu_full` and per reproduce phase.
    pub fn upload_neuron_models(&self, models: &[u32]) {
        self.queue
            .write_buffer(&self.neuron_models_buf, 0, bytemuck::cast_slice(models));
    }

    pub fn upload_neuron_model_at(&self, slot: usize, model: u32) {
        assert!(slot < self.capacity);
        let stride = std::mem::size_of::<u32>() as u64;
        self.queue.write_buffer(
            &self.neuron_models_buf,
            slot as u64 * stride,
            bytemuck::bytes_of(&model),
        );
    }

    /// Zero per-slot buffers that accumulate state across ticks: Hebbian
    /// eligibility traces, last_inputs / last_hidden / last_outputs (the
    /// BRAIN_RECURRENT feedback path reads `last_hidden` next tick), rewards,
    /// and bonded inboxes. Used when the simulation is restarted in place
    /// and the same slot is re-assigned to a fresh cell.
    pub fn zero_persistent_state(&self) {
        let n = self.capacity;
        let f = std::mem::size_of::<f32>();
        let traces_bytes = n * BRAIN_WEIGHTS_PER_CELL * f;
        let inputs_bytes = n * BRAIN_INPUTS * f;
        let hidden_bytes = n * BRAIN_HIDDEN * f;
        let outputs_bytes = n * BRAIN_OUTPUTS * f;
        let rewards_bytes = n * f;
        let inbox_bytes = n * N_BOND_MSG_CHANNELS * f;
        let max_bytes = traces_bytes
            .max(inputs_bytes)
            .max(hidden_bytes)
            .max(outputs_bytes)
            .max(rewards_bytes)
            .max(inbox_bytes);
        let zeros = vec![0u8; max_bytes];
        self.queue.write_buffer(&self.brain_traces_buf, 0, &zeros[..traces_bytes]);
        self.queue.write_buffer(&self.last_inputs_buf, 0, &zeros[..inputs_bytes]);
        self.queue.write_buffer(&self.last_hidden_buf, 0, &zeros[..hidden_bytes]);
        self.queue.write_buffer(&self.last_outputs_buf, 0, &zeros[..outputs_bytes]);
        self.queue.write_buffer(&self.rewards_buf, 0, &zeros[..rewards_bytes]);
        self.queue.write_buffer(&self.bonded_inbox_buf, 0, &zeros[..inbox_bytes]);
    }

    /// Zero per-slot persistent brain state that does NOT get refreshed each
    /// tick from the CPU: `brain_traces` (Hebbian eligibility, decayed over
    /// ~120 ticks), `last_hidden` (read next tick as recurrent input by
    /// populate_inputs), plus `last_inputs` / `last_outputs` for hygiene.
    /// Must be called whenever a slot changes occupant — either after
    /// `swap_to` (the moved cell starts fresh in its new slot) or after a
    /// new cell is allocated into a previously-used slot. Without this,
    /// the new occupant inherits the previous cell's eligibility traces
    /// and gets spurious Hebbian weight updates during the trace decay
    /// window.
    pub fn reset_persistent_brain_state_at(&self, slot: usize) {
        assert!(slot < self.capacity);
        let f = std::mem::size_of::<f32>();
        let u32sz = std::mem::size_of::<u32>();
        let traces_bytes = BRAIN_WEIGHTS_PER_CELL * f;
        let inputs_bytes = BRAIN_INPUTS * f;
        let hidden_bytes = BRAIN_HIDDEN * f;
        let outputs_bytes = BRAIN_OUTPUTS * f;
        let pre_u32_bytes = BRAIN_INPUTS * u32sz;
        let post_u32_bytes = BRAIN_HIDDEN * u32sz;
        let zeros = vec![0u8; traces_bytes];
        self.queue
            .write_buffer(&self.brain_traces_buf, (slot * traces_bytes) as u64, &zeros);
        self.queue
            .write_buffer(&self.last_inputs_buf, (slot * inputs_bytes) as u64, &zeros[..inputs_bytes]);
        self.queue
            .write_buffer(&self.last_hidden_buf, (slot * hidden_bytes) as u64, &zeros[..hidden_bytes]);
        self.queue
            .write_buffer(&self.last_outputs_buf, (slot * outputs_bytes) as u64, &zeros[..outputs_bytes]);
        // Sprint 163: clear STDP state so a recycled slot doesn't inherit
        // the previous cell's spike timings / traces.
        self.queue.write_buffer(
            &self.pre_spike_times_buf,
            (slot * pre_u32_bytes) as u64,
            &zeros[..pre_u32_bytes],
        );
        self.queue.write_buffer(
            &self.post_spike_times_buf,
            (slot * post_u32_bytes) as u64,
            &zeros[..post_u32_bytes],
        );
        self.queue.write_buffer(
            &self.pre_trace_buf,
            (slot * inputs_bytes) as u64,
            &zeros[..inputs_bytes],
        );
        self.queue.write_buffer(
            &self.post_trace_buf,
            (slot * hidden_bytes) as u64,
            &zeros[..hidden_bytes],
        );
        // Sprint 195: clear the whisker spring-damper state so a recycled
        // slot doesn't inherit the previous occupant's ringing whiskers.
        let whisker_bytes = 12 * f;
        self.queue.write_buffer(
            &self.whisker_state_buf,
            (slot * whisker_bytes) as u64,
            &zeros[..whisker_bytes],
        );
    }

    /// GPU-side copy `slot[src] → slot[dst]` for `brain_weights`,
    /// `xoshiro_state`, and `turn_rate`. Used by the swap-remove pattern in
    /// `die_and_drop_carrion`: when the cell at `dst` dies, `src` is the
    /// last live cell moved into its place. Pure on-device memcpy — no
    /// upload or download.
    ///
    /// The moved cell's `brain_traces` / `last_hidden` / `last_outputs` /
    /// `last_inputs` do NOT follow — the previous occupant of `dst` left
    /// stale data there, so we zero the destination after the copy. The
    /// moved cell loses at most ~2 s of Hebbian trace context (which the
    /// trace decay would have wiped anyway), but does not inherit the
    /// dead cell's pre×post accumulator.
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
        // Sprint 137: route per-cell Hebbian rates through the same swap.
        // Reuse `swap_turn_rate_temp` — all three buffers are f32 / 4 B
        // wide so the staging slot fits, and encoder calls serialise.
        encoder.copy_buffer_to_buffer(
            &self.learning_rates_buf,
            tr_src,
            &self.swap_turn_rate_temp,
            0,
            tr_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &self.swap_turn_rate_temp,
            0,
            &self.learning_rates_buf,
            tr_dst,
            tr_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &self.trace_decays_buf,
            tr_src,
            &self.swap_turn_rate_temp,
            0,
            tr_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &self.swap_turn_rate_temp,
            0,
            &self.trace_decays_buf,
            tr_dst,
            tr_bytes,
        );
        self.queue.submit(Some(encoder.finish()));
        // Hebbian traces + recurrent state belong to the slot's occupant;
        // the dead cell's leftovers must not bleed into the moved cell.
        self.reset_persistent_brain_state_at(dst);
    }

    /// Seed the xoshiro state for a single slot. Used after reproduction so
    /// the new cell starts on an independent RNG stream.
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
            trace_w1: [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN],
            trace_w2: [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
            membrane: [IZH_V_REST; BRAIN_HIDDEN],
            recovery: [0.0; BRAIN_HIDDEN],
            last_pre_spike_ticks: [0; BRAIN_INPUTS],
            last_post_spike_ticks: [0; BRAIN_HIDDEN],
            pre_trace: [0.0; BRAIN_INPUTS],
            post_trace: [0.0; BRAIN_HIDDEN],
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

#[cfg(test)]
#[path = "cells_tests.rs"]
mod tests;

