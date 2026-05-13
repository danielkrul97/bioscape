use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::*;
use super::*;

// GPU predate — runs the herd_count and attack compute passes; atomic
// float CAS handles the multi-attacker → victim energy/damage accumulation.

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
pub struct PredateParamsGpu {
    pub num_cells: u32,
    pub cell_size: f32,
    pub cell_radius_const: f32,
    /// Sprint 187: floor for the defense multiplier (was implicit `0.4` from
    /// `BOND_DEFENSE_FRAC × BOND_DEFENSE_CAP` cap). Per-genome defense pool
    /// can sum past `1.0`, so the shader needs an explicit clamp.
    pub defense_floor: f32,
    pub herd_radius_sq: f32,
    /// Sprint 187: was `attack_threshold` global; now per-cell via
    /// `attack_gates_buf` — slot kept as pad for back-compat layout.
    pub _pad_attack_threshold: f32,
    pub predation_gain: f32,
    pub predation_drain: f32,
    pub spike_dot_threshold: f32,
    pub spike_bonus: f32,
    pub dilution_k: f32,
    pub _pad0: u32,
    pub world_half_x: f32,
    pub world_half_y: f32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Per-victim attacker tracking depth — must match `predate.wgsl::MAX_ATTACKERS_PER_VICTIM`.
/// Pack metric requires a bonded pair among the attackers; K = 8 is enough to
/// catch one when a victim is mobbed without blowing up storage.
pub const MAX_ATTACKERS_PER_VICTIM: usize = 8;

#[derive(Debug, Clone)]
pub struct PredateResult {
    pub herd_counts: Vec<u32>,
    pub energy_delta: Vec<f32>,
    pub damage_delta: Vec<f32>,
    /// Wave H regression mitigation: per-tick total number of (attacker,
    /// victim) attack hits. Mirrors CPU `attack_events.len()` so
    /// `predation_events_gen` stays accurate on the GPU path.
    pub total_events: u32,
    /// Sprint 186: per-attacker hit count this dispatch. Used by CPU to
    /// split `bonded_attacks_gen` vs `solo_attacks_gen` by `cell.n_bonds()`.
    pub attacker_event_count: Vec<u32>,
    /// Sprint 186: per-attacker summed gain (post-spike-bonus, post-dilution).
    /// Feeds `bonded_attack_gain_sum_gen` / `solo_attack_gain_sum_gen`.
    pub attacker_gain_sum: Vec<f32>,
    /// Sprint 186: per-victim count of distinct attackers this dispatch.
    /// `>= 2` → swarm victim; `>= 1` contributes to `attack_victims_gen`.
    pub victim_attacker_count: Vec<u32>,
    /// Sprint 186: per-victim, the first `MAX_ATTACKERS_PER_VICTIM` attacker
    /// indices that hit. Stored as `idx + 1`; `0` = empty slot. Flat layout
    /// `victim_attackers[j * K + k]`.
    pub victim_attackers: Vec<u32>,
}

pub struct PredateGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline_bucket_min: wgpu::ComputePipeline,
    pipeline_herd: wgpu::ComputePipeline,
    pipeline_attack: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    capacity: usize,
    params_buf: wgpu::Buffer,
    pos_buf: wgpu::Buffer,
    eff_radii_buf: wgpu::Buffer,
    headings_buf: wgpu::Buffer,
    /// Per-cell `SPIKE_SLOTS × vec4<f32>` (length, azim, elev, complexity).
    spikes_packed_buf: wgpu::Buffer,
    attack_buf: wgpu::Buffer,
    herd_buf: wgpu::Buffer,
    energy_gain_buf: wgpu::Buffer,
    damage_delta_buf: wgpu::Buffer,
    /// Per-cell active spike count (`u32`).
    spike_counts_buf: wgpu::Buffer,
    /// Per-cell pitch (rad) — multi-spike needs the full 3D forward
    /// direction (yaw + azim, pitch + elev).
    pitches_buf: wgpu::Buffer,
    attacker_event_count_buf: wgpu::Buffer,
    attacker_gain_sum_buf: wgpu::Buffer,
    victim_attacker_count_buf: wgpu::Buffer,
    victim_attackers_buf: wgpu::Buffer,
    /// NUM_BUCKETS × f32 — per-bucket minimum eff_radius used by the attack
    /// kernel's whole-bucket size-ratio prefilter. Recomputed each tick by
    /// the `bucket_min_radius` kernel right after the spatial hash rebuild.
    bucket_min_radius_buf: wgpu::Buffer,
    /// Sprint 187: per-cell attacker gate (replaces uniform attack_threshold).
    attack_gates_buf: wgpu::Buffer,
    /// Sprint 187: per-cell attacker size-ratio threshold (replaces uniform).
    predation_size_ratios_buf: wgpu::Buffer,
    /// Sprint 187: per-cell victim defense pool — CPU-aggregated sum of
    /// bonded partners' `genome.defense_contribution` (capped at
    /// `BOND_DEFENSE_CAP` partners). Shader applies
    /// `max(1.0 − defense_pool[j], defense_floor)` to drain + gain.
    defense_pool_buf: wgpu::Buffer,
    herd_rb: wgpu::Buffer,
    energy_gain_rb: wgpu::Buffer,
    damage_rb: wgpu::Buffer,
    attacker_event_count_rb: wgpu::Buffer,
    attacker_gain_sum_rb: wgpu::Buffer,
    victim_attacker_count_rb: wgpu::Buffer,
    victim_attackers_rb: wgpu::Buffer,
    pos_packed: Vec<f32>,
}

impl PredateGpu {
    pub fn with_context(ctx: &GpuContext, capacity: usize) -> Result<Self, String> {
        Self::with_device_inner(Arc::clone(&ctx.device), Arc::clone(&ctx.queue), capacity)
    }

    pub fn new(capacity: usize) -> Result<Self, String> {
        let ctx = GpuContext::new()?;
        Self::with_device_inner(ctx.device, ctx.queue, capacity)
    }

    fn with_device_inner(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        capacity: usize,
    ) -> Result<Self, String> {
        assert!(capacity > 0);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("predate"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/predate.wgsl").into()),
        });
        // 21 bindings (S187 added 3 per-cell aggression buffers): 1 uniform +
        // 20 storage. Read-only: 1..=7, 11..=12, 18..=20 (per-cell aggression).
        // Read_write: 8 (herd_counts), 9..=10 (energy/damage atomic), 13..=16
        // (Sprint 186 diagnostics), 17 (bucket min radius).
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..21)
            .map(|i| {
                let ty = if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else if (1..=7).contains(&i)
                    || (11..=12).contains(&i)
                    || (18..=20).contains(&i)
                {
                    wgpu::BufferBindingType::Storage { read_only: true }
                } else {
                    wgpu::BufferBindingType::Storage { read_only: false }
                };
                wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            })
            .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("predate-bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("predate-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline_bucket_min = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("predate-bucket-min-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("bucket_min_radius"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_herd = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("predate-herd-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("herd_count"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_attack = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("predate-attack-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("attack"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("predate-params"),
            contents: bytemuck::bytes_of(&PredateParamsGpu::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let f = std::mem::size_of::<f32>() as u64;
        let n = capacity as u64;
        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let stor_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let stor_dst_src =
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let pos_buf = mk("predate-pos", n * 3 * f, stor_dst);
        let eff_radii_buf = mk("predate-eff", n * f, stor_dst);
        let headings_buf = mk("predate-heading", n * f, stor_dst);
        // SPIKE_SLOTS × vec4<f32> = 5 × 16 B = 80 B per cell.
        let spike_slots = SPIKE_SLOTS as u64;
        let spikes_packed_buf = mk("predate-spikes-packed", n * spike_slots * 4 * f, stor_dst);
        let attack_buf = mk("predate-attack", n * f, stor_dst);
        let herd_buf = mk("predate-herd", n * f, stor_dst_src);
        let energy_gain_buf = mk("predate-energy-gain", n * f, stor_dst_src);
        let damage_delta_buf = mk("predate-damage-delta", n * f, stor_dst_src);
        let spike_counts_buf = mk("predate-spike-counts", n * f, stor_dst);
        let pitches_buf = mk("predate-pitches", n * f, stor_dst);
        let attack_gates_buf = mk("predate-attack-gates", n * f, stor_dst);
        let predation_size_ratios_buf =
            mk("predate-predation-size-ratios", n * f, stor_dst);
        let defense_pool_buf = mk("predate-defense-pool", n * f, stor_dst);
        let attacker_event_count_buf =
            mk("predate-attacker-event-count", n * f, stor_dst_src);
        let attacker_gain_sum_buf =
            mk("predate-attacker-gain-sum", n * f, stor_dst_src);
        let victim_attacker_count_buf =
            mk("predate-victim-attacker-count", n * f, stor_dst_src);
        let victim_attackers_buf = mk(
            "predate-victim-attackers",
            n * MAX_ATTACKERS_PER_VICTIM as u64 * f,
            stor_dst_src,
        );
        // NUM_BUCKETS = GRID_NX * GRID_NY * GRID_NZ = 64 * 32 * 4 = 8192.
        // Keep in sync with the const in `predate.wgsl` / `spatial_hash.wgsl`.
        const NUM_BUCKETS: u64 = 8192;
        let bucket_min_radius_buf = mk(
            "predate-bucket-min-radius",
            NUM_BUCKETS * f,
            stor_dst,
        );
        let herd_rb = mk("predate-herd-rb", n * f, read);
        let energy_gain_rb = mk("predate-energy-gain-rb", n * f, read);
        let damage_rb = mk("predate-damage-rb", n * f, read);
        let attacker_event_count_rb =
            mk("predate-attacker-event-count-rb", n * f, read);
        let attacker_gain_sum_rb =
            mk("predate-attacker-gain-sum-rb", n * f, read);
        let victim_attacker_count_rb =
            mk("predate-victim-attacker-count-rb", n * f, read);
        let victim_attackers_rb = mk(
            "predate-victim-attackers-rb",
            n * MAX_ATTACKERS_PER_VICTIM as u64 * f,
            read,
        );

        Ok(Self {
            device,
            queue,
            pipeline_bucket_min,
            pipeline_herd,
            pipeline_attack,
            bind_group_layout,
            capacity,
            params_buf,
            pos_buf,
            eff_radii_buf,
            headings_buf,
            spikes_packed_buf,
            attack_buf,
            herd_buf,
            energy_gain_buf,
            damage_delta_buf,
            spike_counts_buf,
            pitches_buf,
            attacker_event_count_buf,
            attacker_gain_sum_buf,
            victim_attacker_count_buf,
            victim_attackers_buf,
            bucket_min_radius_buf,
            attack_gates_buf,
            predation_size_ratios_buf,
            defense_pool_buf,
            herd_rb,
            energy_gain_rb,
            damage_rb,
            attacker_event_count_rb,
            attacker_gain_sum_rb,
            victim_attacker_count_rb,
            victim_attackers_rb,
            pos_packed: Vec::new(),
        })
    }

    /// Multi-spike attack pass. `spikes_packed[i*SPIKE_SLOTS + slot]` is
    /// `[length, azimuth_offset, elevation_offset, complexity]`;
    /// `spike_counts[i]` marks how many slots are active (the rest stay
    /// zero-initialized); `pitches[i]` provides the 3D forward direction.
    /// Sprint 187: `attack_gates[i]` and `predation_size_ratios[i]` are
    /// per-attacker (replace pre-S187 uniform Params fields); `defense_pool[j]`
    /// is per-victim sum of bonded partners' `defense_contribution` (CPU
    /// aggregated each tick).
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &mut self,
        positions: &[[f32; 3]],
        eff_radii: &[f32],
        headings: &[f32],
        pitches: &[f32],
        spikes_packed: &[[f32; 4]],
        spike_counts: &[u32],
        attack_signals: &[f32],
        attack_gates: &[f32],
        predation_size_ratios: &[f32],
        defense_pool: &[f32],
        cell_hash: &SpatialHashGpu,
        params: PredateParamsGpu,
    ) -> PredateResult {
        let n = positions.len();
        assert!(n <= self.capacity, "predate capacity overflow");
        assert_eq!(pitches.len(), n);
        assert_eq!(spike_counts.len(), n);
        assert_eq!(spikes_packed.len(), n * SPIKE_SLOTS);
        assert_eq!(attack_gates.len(), n);
        assert_eq!(predation_size_ratios.len(), n);
        assert_eq!(defense_pool.len(), n);
        if n == 0 {
            return PredateResult {
                herd_counts: Vec::new(),
                energy_delta: Vec::new(),
                damage_delta: Vec::new(),
                total_events: 0,
                attacker_event_count: Vec::new(),
                attacker_gain_sum: Vec::new(),
                victim_attacker_count: Vec::new(),
                victim_attackers: Vec::new(),
            };
        }
        let mut params = params;
        params.num_cells = n as u32;

        self.pos_packed.clear();
        self.pos_packed.reserve(n * 3);
        for p in positions {
            self.pos_packed.extend_from_slice(p);
        }

        // Reset herd_counts + energy_delta + damage_delta na 0 (atomic add must
        // start from clean slate per dispatch).
        let zero_bytes = vec![0u8; (n * 4) as usize];
        self.queue.write_buffer(&self.herd_buf, 0, &zero_bytes);
        self.queue.write_buffer(&self.energy_gain_buf, 0, &zero_bytes);
        self.queue.write_buffer(&self.damage_delta_buf, 0, &zero_bytes);
        // Sprint 186: per-attacker / per-victim diagnostics zero-init.
        self.queue
            .write_buffer(&self.attacker_event_count_buf, 0, &zero_bytes);
        self.queue
            .write_buffer(&self.attacker_gain_sum_buf, 0, &zero_bytes);
        self.queue
            .write_buffer(&self.victim_attacker_count_buf, 0, &zero_bytes);
        let victim_attackers_zero = vec![0u8; n * MAX_ATTACKERS_PER_VICTIM * 4];
        self.queue
            .write_buffer(&self.victim_attackers_buf, 0, &victim_attackers_zero);

        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&self.pos_buf, 0, bytemuck::cast_slice(&self.pos_packed));
        self.queue
            .write_buffer(&self.eff_radii_buf, 0, bytemuck::cast_slice(eff_radii));
        self.queue
            .write_buffer(&self.headings_buf, 0, bytemuck::cast_slice(headings));
        self.queue.write_buffer(
            &self.spikes_packed_buf,
            0,
            bytemuck::cast_slice(spikes_packed),
        );
        self.queue
            .write_buffer(&self.attack_buf, 0, bytemuck::cast_slice(attack_signals));
        self.queue.write_buffer(
            &self.spike_counts_buf,
            0,
            bytemuck::cast_slice(spike_counts),
        );
        self.queue
            .write_buffer(&self.pitches_buf, 0, bytemuck::cast_slice(pitches));
        self.queue.write_buffer(
            &self.attack_gates_buf,
            0,
            bytemuck::cast_slice(attack_gates),
        );
        self.queue.write_buffer(
            &self.predation_size_ratios_buf,
            0,
            bytemuck::cast_slice(predation_size_ratios),
        );
        self.queue.write_buffer(
            &self.defense_pool_buf,
            0,
            bytemuck::cast_slice(defense_pool),
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("predate-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.eff_radii_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.headings_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.spikes_packed_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.attack_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: cell_hash.offsets_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: cell_hash.sorted_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.herd_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.energy_gain_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: self.damage_delta_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: self.spike_counts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: self.pitches_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: self.attacker_event_count_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 14, resource: self.attacker_gain_sum_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 15, resource: self.victim_attacker_count_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 16, resource: self.victim_attackers_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 17, resource: self.bucket_min_radius_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 18, resource: self.attack_gates_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 19, resource: self.predation_size_ratios_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 20, resource: self.defense_pool_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("predate-encoder"),
        });
        let workgroups = ((n as u32) + 63) / 64;
        // NUM_BUCKETS = 8192 → 128 workgroups of 64 threads each.
        const NUM_BUCKETS_WG: u32 = 8192 / 64;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("predate-bucket-min-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_bucket_min);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(NUM_BUCKETS_WG, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("predate-herd-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_herd);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("predate-attack-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_attack);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        let bytes = (n as u64) * 4;
        let victim_attackers_bytes = bytes * MAX_ATTACKERS_PER_VICTIM as u64;
        encoder.copy_buffer_to_buffer(&self.herd_buf, 0, &self.herd_rb, 0, bytes);
        encoder.copy_buffer_to_buffer(&self.energy_gain_buf, 0, &self.energy_gain_rb, 0, bytes);
        encoder.copy_buffer_to_buffer(&self.damage_delta_buf, 0, &self.damage_rb, 0, bytes);
        encoder.copy_buffer_to_buffer(
            &self.attacker_event_count_buf,
            0,
            &self.attacker_event_count_rb,
            0,
            bytes,
        );
        encoder.copy_buffer_to_buffer(
            &self.attacker_gain_sum_buf,
            0,
            &self.attacker_gain_sum_rb,
            0,
            bytes,
        );
        encoder.copy_buffer_to_buffer(
            &self.victim_attacker_count_buf,
            0,
            &self.victim_attacker_count_rb,
            0,
            bytes,
        );
        encoder.copy_buffer_to_buffer(
            &self.victim_attackers_buf,
            0,
            &self.victim_attackers_rb,
            0,
            victim_attackers_bytes,
        );
        self.queue.submit(Some(encoder.finish()));

        let h = self.herd_rb.slice(0..bytes);
        let e = self.energy_gain_rb.slice(0..bytes);
        let d = self.damage_rb.slice(0..bytes);
        let aec = self.attacker_event_count_rb.slice(0..bytes);
        let ags = self.attacker_gain_sum_rb.slice(0..bytes);
        let vac = self.victim_attacker_count_rb.slice(0..bytes);
        let va = self.victim_attackers_rb.slice(0..victim_attackers_bytes);
        h.map_async(wgpu::MapMode::Read, |_| {});
        e.map_async(wgpu::MapMode::Read, |_| {});
        d.map_async(wgpu::MapMode::Read, |_| {});
        aec.map_async(wgpu::MapMode::Read, |_| {});
        ags.map_async(wgpu::MapMode::Read, |_| {});
        vac.map_async(wgpu::MapMode::Read, |_| {});
        va.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let attacker_event_count: Vec<u32> =
            bytemuck::cast_slice::<u8, u32>(&aec.get_mapped_range()).to_vec();
        let total_events: u32 = attacker_event_count.iter().sum();
        // GPU now writes only the per-attacker self_gain (`energy_gain`) and
        // per-victim drain (`damage_delta`). Net per-cell `energy_delta` is
        // derived host-side so the shader can skip its second CAS smyčka.
        let energy_gain: Vec<f32> =
            bytemuck::cast_slice::<u8, f32>(&e.get_mapped_range()).to_vec();
        let damage_delta: Vec<f32> =
            bytemuck::cast_slice::<u8, f32>(&d.get_mapped_range()).to_vec();
        let energy_delta: Vec<f32> = energy_gain
            .iter()
            .zip(damage_delta.iter())
            .map(|(g, d)| g - d)
            .collect();
        let res = PredateResult {
            herd_counts: bytemuck::cast_slice::<u8, u32>(&h.get_mapped_range()).to_vec(),
            energy_delta,
            damage_delta,
            total_events,
            attacker_event_count,
            attacker_gain_sum: bytemuck::cast_slice::<u8, f32>(&ags.get_mapped_range()).to_vec(),
            victim_attacker_count: bytemuck::cast_slice::<u8, u32>(&vac.get_mapped_range())
                .to_vec(),
            victim_attackers: bytemuck::cast_slice::<u8, u32>(&va.get_mapped_range()).to_vec(),
        };
        self.herd_rb.unmap();
        self.energy_gain_rb.unmap();
        self.damage_rb.unmap();
        self.attacker_event_count_rb.unmap();
        self.attacker_gain_sum_rb.unmap();
        self.victim_attacker_count_rb.unmap();
        self.victim_attackers_rb.unmap();
        res
    }
}

