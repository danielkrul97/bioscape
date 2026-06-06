use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::*;

// GPU sensor gather. Chains 2× `SpatialHashGpu` (cells, foods) + 2×
// `FieldGpu` (smell, pheromone) into a single per-cell pass.

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
pub struct SensorParamsGpu {
    pub num_cells: u32,
    pub num_foods: u32,
    pub hash_cell_size: f32,
    pub world_half_x: f32,
    pub world_half_y: f32,
    pub world_half_z: f32,
    pub field_res_x: u32,
    pub field_res_y: u32,
    pub field_res_z: u32,
    pub field_eps: f32,
    pub field_world_half_x: f32,
    pub field_world_half_y: f32,
    pub field_world_half_z: f32,
    /// Wave 5: maze fields (LOS raycast). `maze_active != 0` enables LOS
    /// filtering of food/cell candidates against the obstacle mask at
    /// binding 13.
    pub maze_active: u32,
    pub maze_res_x: u32,
    pub maze_res_y: u32,
    /// Sprint 195: current sim tick — seeds the deterministic whisker
    /// transduction noise. `_pad` rounds the struct to 80 bytes so the
    /// uniform layout matches naga's std140 rounding of the WGSL struct.
    pub tick: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SensorRow {
    pub nearest_food: Option<[f32; 3]>,
    pub nearest_cell: Option<([f32; 3], f32)>,
    pub neighbors_in_vision: u32,
    pub smell_grad: [f32; 3],
    pub pheromone_grad: [f32; 3],
    pub pheromone_grad_ch1: [f32; 3],
    pub pheromone_grad_ch2: [f32; 3],
    pub vibration_grad: [f32; 3],
    pub vibration_amp: f32,
}

/// Output stride per cell in `sensor_gather.wgsl` — keep in lock-step with
/// the shader's `let off = i * 31u;` block. V7 bumped 15 → 19 (+4 vibration);
/// Wave 6 bumped 19 → 25 (+6 whisker raycast); Wave L bumped 25 → 31
/// (+6 for ch1/ch2 pheromone gradients).
pub const SENSOR_OUTPUT_STRIDE: usize = 31;

pub struct SensorGatherGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    capacity_cells: usize,
    capacity_foods: usize,
    params_buf: wgpu::Buffer,
    positions_buf: wgpu::Buffer,
    eff_radii_buf: wgpu::Buffer,
    vision_radii_buf: wgpu::Buffer,
    food_positions_buf: wgpu::Buffer,
    /// Wave 5: per-voxel maze occupancy mask (binding 13). Same packing as
    /// `StepGpu::maze_mask_buf`. Pre-allocated at MAZE_MASK_CAPACITY u32s.
    maze_mask_buf: wgpu::Buffer,
    /// Wave 6: per-cell heading + pitch for whisker raycast direction
    /// derivation (bindings 14, 15). Uploaded each `dispatch_no_readback`.
    headings_buf: wgpu::Buffer,
    pitches_buf: wgpu::Buffer,
    output_buf: wgpu::Buffer,
    output_rb: wgpu::Buffer,
    pos_packed: Vec<f32>,
    food_packed: Vec<f32>,
    epoch: u64,
}

impl SensorGatherGpu {
    pub fn with_context(
        ctx: &GpuContext,
        capacity_cells: usize,
        capacity_foods: usize,
    ) -> Result<Self, String> {
        Self::with_device_inner(
            Arc::clone(&ctx.device),
            Arc::clone(&ctx.queue),
            capacity_cells,
            capacity_foods,
        )
    }

    pub fn new(capacity_cells: usize, capacity_foods: usize) -> Result<Self, String> {
        let ctx = GpuContext::new()?;
        Self::with_device_inner(ctx.device, ctx.queue, capacity_cells, capacity_foods)
    }

    fn with_device_inner(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        capacity_cells: usize,
        capacity_foods: usize,
    ) -> Result<Self, String> {
        assert!(capacity_cells > 0);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sensor_gather"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/sensor_gather.wgsl").into(),
            ),
        });
        // Wave 6: 14 → 16 bindings (bindings 14, 15 = headings, pitches
        // read-only for whisker raycast direction).
        // Wave L: 16 → 18 bindings (bindings 16, 17 = pheromone ch1/ch2
        // grids read-only for multi-channel gradient sampling).
        // Sprint 195: 18 → 19 bindings (binding 18 = whisker spring-damper
        // state, read_write persistent — owned by `CellsGpu`).
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..19)
            .map(|i| {
                let ty = if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else if i == 11 || i == 18 {
                    wgpu::BufferBindingType::Storage { read_only: false }
                } else {
                    wgpu::BufferBindingType::Storage { read_only: true }
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
            label: Some("sensor-bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sensor-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sensor-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("sensor_gather"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sensor-params"),
            contents: bytemuck::bytes_of(&SensorParamsGpu::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let f = std::mem::size_of::<f32>() as u64;
        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let stor_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let stor_src = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let nc = capacity_cells as u64;
        let nf = capacity_foods.max(1) as u64;
        let positions_buf = mk("sensor-pos", nc * 3 * f, stor_dst);
        let eff_radii_buf = mk("sensor-eff", nc * f, stor_dst);
        let vision_radii_buf = mk("sensor-vision", nc * f, stor_dst);
        let food_positions_buf = mk("sensor-food-pos", nf * 3 * f, stor_dst);
        // Wave 5: maze mask buffer. Same MAZE_MASK_CAPACITY as StepGpu so
        // the same packed mask payload feeds both shaders.
        let maze_mask_buf = mk(
            "sensor-maze-mask",
            MAZE_MASK_CAPACITY as u64 * std::mem::size_of::<u32>() as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let headings_buf = mk("sensor-headings", nc * f, stor_dst);
        let pitches_buf = mk("sensor-pitches", nc * f, stor_dst);
        // V7: 19 floats per cell (was 15). Layout — nearest_food (3) +
        // has_food (1) + nearest_cell delta (3) + radius (1) + smell_grad (3)
        // + phero_grad (3) + neighbors_count (1, bitcast<u32>) + vibration_grad (3)
        // + vibration_amp (1). Keep in sync with shader's `let off = i * 19u`.
        let stride_bytes = (SENSOR_OUTPUT_STRIDE as u64) * f;
        let output_buf = mk("sensor-output", nc * stride_bytes, stor_src);
        let output_rb = mk("sensor-output-rb", nc * stride_bytes, read);

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            capacity_cells,
            capacity_foods,
            params_buf,
            positions_buf,
            eff_radii_buf,
            vision_radii_buf,
            food_positions_buf,
            maze_mask_buf,
            headings_buf,
            pitches_buf,
            output_buf,
            output_rb,
            pos_packed: Vec::new(),
            food_packed: Vec::new(),
            epoch: 0,
        })
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Output buffer accessor for chained shaders (`populate_inputs`).
    pub fn output_buffer(&self) -> &wgpu::Buffer {
        &self.output_buf
    }
    pub fn vision_radii_buffer(&self) -> &wgpu::Buffer {
        &self.vision_radii_buf
    }

    /// Wave 5: upload the maze mask. Same packing as `StepGpu::upload_maze`.
    /// `params.maze_active = 0` next dispatch makes the shader skip the LOS
    /// check; the mask buffer contents are then irrelevant.
    pub fn upload_maze(&self, mask: &[u32]) {
        debug_assert!(
            mask.len() <= MAZE_MASK_CAPACITY,
            "sensor maze mask {} exceeds capacity {}",
            mask.len(),
            MAZE_MASK_CAPACITY
        );
        if mask.is_empty() {
            return;
        }
        self.queue
            .write_buffer(&self.maze_mask_buf, 0, bytemuck::cast_slice(mask));
    }

    /// Variant of `compute()` that skips the `Vec<SensorRow>` readback —
    /// the sensor output stays in `output_buf` for the chained
    /// `PopulateInputsGpu` shader to consume. Saves a ~60 KB readback /
    /// device round-trip per tick.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_no_readback(
        &mut self,
        positions: &[[f32; 3]],
        eff_radii: &[f32],
        vision_radii: &[f32],
        food_positions: &[[f32; 3]],
        headings: &[f32],
        pitches: &[f32],
        cell_hash: &SpatialHashGpu,
        food_hash: &SpatialHashGpu,
        smell: &FieldGpu,
        pheromone: &FieldGpu,
        pheromone_ch1: &FieldGpu,
        pheromone_ch2: &FieldGpu,
        vibration: &FieldGpu,
        whisker_state: &wgpu::Buffer,
        params: SensorParamsGpu,
    ) {
        if positions.is_empty() {
            return;
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sensor-dispatch-encoder"),
            });
        self.dispatch_no_readback_into(
            &mut encoder,
            positions,
            eff_radii,
            vision_radii,
            food_positions,
            headings,
            pitches,
            cell_hash,
            food_hash,
            smell,
            pheromone,
            pheromone_ch1,
            pheromone_ch2,
            vibration,
            whisker_state,
            params,
        );
        self.queue.submit(Some(encoder.finish()));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_no_readback_into(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        positions: &[[f32; 3]],
        eff_radii: &[f32],
        vision_radii: &[f32],
        food_positions: &[[f32; 3]],
        headings: &[f32],
        pitches: &[f32],
        cell_hash: &SpatialHashGpu,
        food_hash: &SpatialHashGpu,
        smell: &FieldGpu,
        pheromone: &FieldGpu,
        pheromone_ch1: &FieldGpu,
        pheromone_ch2: &FieldGpu,
        vibration: &FieldGpu,
        whisker_state: &wgpu::Buffer,
        params: SensorParamsGpu,
    ) {
        let n = positions.len();
        let nf = food_positions.len();
        assert!(n <= self.capacity_cells, "sensor cell capacity overflow");
        assert!(nf <= self.capacity_foods, "sensor food capacity overflow");
        if n == 0 {
            return;
        }
        let mut params = params;
        params.num_cells = n as u32;
        params.num_foods = nf as u32;

        self.pos_packed.clear();
        self.pos_packed.reserve(n * 3);
        for p in positions {
            self.pos_packed.extend_from_slice(p);
        }
        self.food_packed.clear();
        self.food_packed.reserve((nf.max(1)) * 3);
        for p in food_positions {
            self.food_packed.extend_from_slice(p);
        }
        if self.food_packed.is_empty() {
            self.food_packed.extend_from_slice(&[0.0, 0.0, 0.0]);
        }

        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(
            &self.positions_buf,
            0,
            bytemuck::cast_slice(&self.pos_packed),
        );
        self.queue
            .write_buffer(&self.eff_radii_buf, 0, bytemuck::cast_slice(eff_radii));
        self.queue.write_buffer(
            &self.vision_radii_buf,
            0,
            bytemuck::cast_slice(vision_radii),
        );
        self.queue.write_buffer(
            &self.food_positions_buf,
            0,
            bytemuck::cast_slice(&self.food_packed),
        );
        self.queue
            .write_buffer(&self.headings_buf, 0, bytemuck::cast_slice(headings));
        self.queue
            .write_buffer(&self.pitches_buf, 0, bytemuck::cast_slice(pitches));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sensor-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.positions_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.eff_radii_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.vision_radii_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.food_positions_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cell_hash.offsets_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: cell_hash.sorted_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: food_hash.offsets_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: food_hash.sorted_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: smell.current_grid_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: pheromone.current_grid_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: vibration.current_grid_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: self.maze_mask_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: self.headings_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: self.pitches_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: pheromone_ch1.current_grid_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: pheromone_ch2.current_grid_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: whisker_state.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sensor-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
    }

    /// Spustí celý sensor gather pipeline. `cell_hash` musí být rebuildnut z
    /// `positions`, `food_hash` z `food_positions`. `smell` + `pheromone`
    /// FieldGpu musí mít stepnuté za current tick.
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &mut self,
        positions: &[[f32; 3]],
        eff_radii: &[f32],
        vision_radii: &[f32],
        food_positions: &[[f32; 3]],
        headings: &[f32],
        pitches: &[f32],
        cell_hash: &SpatialHashGpu,
        food_hash: &SpatialHashGpu,
        smell: &FieldGpu,
        pheromone: &FieldGpu,
        pheromone_ch1: &FieldGpu,
        pheromone_ch2: &FieldGpu,
        vibration: &FieldGpu,
        whisker_state: &wgpu::Buffer,
        params: SensorParamsGpu,
    ) -> Vec<SensorRow> {
        let n = positions.len();
        let nf = food_positions.len();
        assert!(n <= self.capacity_cells, "sensor cell capacity overflow");
        assert!(nf <= self.capacity_foods, "sensor food capacity overflow");
        if n == 0 {
            return Vec::new();
        }
        let mut params = params;
        params.num_cells = n as u32;
        params.num_foods = nf as u32;

        self.pos_packed.clear();
        self.pos_packed.reserve(n * 3);
        for p in positions {
            self.pos_packed.extend_from_slice(p);
        }
        self.food_packed.clear();
        self.food_packed.reserve((nf.max(1)) * 3);
        for p in food_positions {
            self.food_packed.extend_from_slice(p);
        }
        if self.food_packed.is_empty() {
            // Sentinel one-element pad — buffer alokovaný s capacity ≥ 1.
            self.food_packed.extend_from_slice(&[0.0, 0.0, 0.0]);
        }

        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(
            &self.positions_buf,
            0,
            bytemuck::cast_slice(&self.pos_packed),
        );
        self.queue
            .write_buffer(&self.eff_radii_buf, 0, bytemuck::cast_slice(eff_radii));
        self.queue.write_buffer(
            &self.vision_radii_buf,
            0,
            bytemuck::cast_slice(vision_radii),
        );
        self.queue.write_buffer(
            &self.food_positions_buf,
            0,
            bytemuck::cast_slice(&self.food_packed),
        );
        // Wave 6: per-cell heading + pitch for in-shader whisker raycast.
        self.queue
            .write_buffer(&self.headings_buf, 0, bytemuck::cast_slice(headings));
        self.queue
            .write_buffer(&self.pitches_buf, 0, bytemuck::cast_slice(pitches));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sensor-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.positions_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.eff_radii_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.vision_radii_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.food_positions_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cell_hash.offsets_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: cell_hash.sorted_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: food_hash.offsets_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: food_hash.sorted_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: smell.current_grid_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: pheromone.current_grid_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: vibration.current_grid_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: self.maze_mask_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: self.headings_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: self.pitches_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: pheromone_ch1.current_grid_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: pheromone_ch2.current_grid_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: whisker_state.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sensor-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sensor-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
        }
        let bytes = (n as u64) * (SENSOR_OUTPUT_STRIDE as u64) * 4;
        encoder.copy_buffer_to_buffer(&self.output_buf, 0, &self.output_rb, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let s = self.output_rb.slice(0..bytes);
        s.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let data = s.get_mapped_range();
        let f: &[f32] = bytemuck::cast_slice(&data);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * SENSOR_OUTPUT_STRIDE;
            let has_food = f[off + 3] > 0.5;
            // `SensorRow` carries a signed min-image delta (toroidal-aware,
            // matches `BrainSensors.nearest_food/cell`). Reconstructing an
            // absolute position via `positions[i] + delta` was wrong across
            // a wrap — e.g. cell at x=-950 with delta=60 lands at -890
            // instead of the ghost +1010 — so callers store the delta as-is.
            let nearest_food = if has_food {
                Some([f[off], f[off + 1], f[off + 2]])
            } else {
                None
            };
            let radius = f[off + 7];
            let nearest_cell = if radius >= 0.0 {
                Some(([f[off + 4], f[off + 5], f[off + 6]], radius))
            } else {
                None
            };
            let _ = positions;
            let count_bits = f[off + 14].to_bits();
            out.push(SensorRow {
                nearest_food,
                nearest_cell,
                neighbors_in_vision: count_bits,
                smell_grad: [f[off + 8], f[off + 9], f[off + 10]],
                pheromone_grad: [f[off + 11], f[off + 12], f[off + 13]],
                pheromone_grad_ch1: [f[off + 25], f[off + 26], f[off + 27]],
                pheromone_grad_ch2: [f[off + 28], f[off + 29], f[off + 30]],
                vibration_grad: [f[off + 15], f[off + 16], f[off + 17]],
                vibration_amp: f[off + 18],
            });
        }
        drop(data);
        self.output_rb.unmap();
        out
    }
}
