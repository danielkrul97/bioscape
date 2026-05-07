use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::*;
use super::*;

// ============================================================================
// Sprint 50: GPU predate — herd_count + attack passes, atomic float CAS
// ============================================================================

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
pub struct PredateParamsGpu {
    pub num_cells: u32,
    pub cell_size: f32,
    pub cell_radius_const: f32,
    pub size_ratio_threshold: f32,
    pub herd_radius_sq: f32,
    pub attack_threshold: f32,
    pub predation_gain: f32,
    pub predation_drain: f32,
    pub spike_dot_threshold: f32,
    pub spike_bonus: f32,
    pub dilution_k: f32,
    pub _pad0: u32,
    /// Sprint 55: toroidal world bounds.
    pub world_half_x: f32,
    pub world_half_y: f32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[derive(Debug, Clone)]
pub struct PredateResult {
    pub herd_counts: Vec<u32>,
    pub energy_delta: Vec<f32>,
    pub damage_delta: Vec<f32>,
}

pub struct PredateGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline_herd: wgpu::ComputePipeline,
    pipeline_attack: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    capacity: usize,
    params_buf: wgpu::Buffer,
    pos_buf: wgpu::Buffer,
    eff_radii_buf: wgpu::Buffer,
    headings_buf: wgpu::Buffer,
    /// Sprint 126: per-cell n × SPIKE_SLOTS × vec4<f32> (length, azim, elev, complexity).
    spikes_packed_buf: wgpu::Buffer,
    attack_buf: wgpu::Buffer,
    herd_buf: wgpu::Buffer,
    energy_delta_buf: wgpu::Buffer,
    damage_delta_buf: wgpu::Buffer,
    /// Sprint 126: per-cell aktivní spike count (`u32`).
    spike_counts_buf: wgpu::Buffer,
    /// Sprint 126: per-cell pitch (rad) — multi-spike potřebuje 3D forward
    /// direction (yaw + azim, pitch + elev).
    pitches_buf: wgpu::Buffer,
    herd_rb: wgpu::Buffer,
    energy_rb: wgpu::Buffer,
    damage_rb: wgpu::Buffer,
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
        // Sprint 126: 13 bindings total (1 uniform + 12 storage). Storage 1..=7
        // a 11..=12 jsou read-only (positions, eff_radii, headings, spikes_packed,
        // attack_signals, hash_offsets, hash_sorted, spike_counts, pitches);
        // 8..=10 jsou read_write (herd_counts, energy_delta atomic, damage_delta
        // atomic).
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..13)
            .map(|i| {
                let ty = if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else if (1..=7).contains(&i) || (11..=12).contains(&i) {
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
        // Sprint 126: SPIKE_SLOTS × vec4<f32> = 5 × 4 × 4 B = 80 B per cell.
        let spike_slots = SPIKE_SLOTS as u64;
        let spikes_packed_buf = mk("predate-spikes-packed", n * spike_slots * 4 * f, stor_dst);
        let attack_buf = mk("predate-attack", n * f, stor_dst);
        let herd_buf = mk("predate-herd", n * f, stor_dst_src);
        let energy_delta_buf = mk("predate-energy-delta", n * f, stor_dst_src);
        let damage_delta_buf = mk("predate-damage-delta", n * f, stor_dst_src);
        let spike_counts_buf = mk("predate-spike-counts", n * f, stor_dst);
        let pitches_buf = mk("predate-pitches", n * f, stor_dst);
        let herd_rb = mk("predate-herd-rb", n * f, read);
        let energy_rb = mk("predate-energy-rb", n * f, read);
        let damage_rb = mk("predate-damage-rb", n * f, read);

        Ok(Self {
            device,
            queue,
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
            energy_delta_buf,
            damage_delta_buf,
            spike_counts_buf,
            pitches_buf,
            herd_rb,
            energy_rb,
            damage_rb,
            pos_packed: Vec::new(),
        })
    }

    /// Sprint 126: full multi-spike. `spikes_packed[i*SPIKE_SLOTS + slot]` =
    /// `[length, azimuth_offset, elevation_offset, complexity]`. `spike_counts[i]`
    /// určuje aktivní sloty (rest zero-init). `pitches[i]` per cell pro 3D
    /// forward direction.
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
        cell_hash: &SpatialHashGpu,
        params: PredateParamsGpu,
    ) -> PredateResult {
        let n = positions.len();
        assert!(n <= self.capacity, "predate capacity overflow");
        assert_eq!(pitches.len(), n);
        assert_eq!(spike_counts.len(), n);
        assert_eq!(spikes_packed.len(), n * SPIKE_SLOTS);
        if n == 0 {
            return PredateResult {
                herd_counts: Vec::new(),
                energy_delta: Vec::new(),
                damage_delta: Vec::new(),
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
        self.queue.write_buffer(&self.energy_delta_buf, 0, &zero_bytes);
        self.queue.write_buffer(&self.damage_delta_buf, 0, &zero_bytes);

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
                wgpu::BindGroupEntry { binding: 9, resource: self.energy_delta_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: self.damage_delta_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: self.spike_counts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: self.pitches_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("predate-encoder"),
        });
        let workgroups = ((n as u32) + 63) / 64;
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
        encoder.copy_buffer_to_buffer(&self.herd_buf, 0, &self.herd_rb, 0, bytes);
        encoder.copy_buffer_to_buffer(&self.energy_delta_buf, 0, &self.energy_rb, 0, bytes);
        encoder.copy_buffer_to_buffer(&self.damage_delta_buf, 0, &self.damage_rb, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let h = self.herd_rb.slice(0..bytes);
        let e = self.energy_rb.slice(0..bytes);
        let d = self.damage_rb.slice(0..bytes);
        h.map_async(wgpu::MapMode::Read, |_| {});
        e.map_async(wgpu::MapMode::Read, |_| {});
        d.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let res = PredateResult {
            herd_counts: bytemuck::cast_slice::<u8, u32>(&h.get_mapped_range()).to_vec(),
            energy_delta: bytemuck::cast_slice::<u8, f32>(&e.get_mapped_range()).to_vec(),
            damage_delta: bytemuck::cast_slice::<u8, f32>(&d.get_mapped_range()).to_vec(),
        };
        self.herd_rb.unmap();
        self.energy_rb.unmap();
        self.damage_rb.unmap();
        res
    }
}

