use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::{GpuContext, SpatialHashGpu};

// GPU eat_food candidate selection. Mirror of headless `eat_food` Pass 1:
// per cell, walk `food_hash` buckets within `EAT_RADIUS × max_axis` and
// return the FIRST food whose center is inside the pose-rotated ellipsoid
// (eat_factor = EAT_RADIUS). CPU then runs Pass 2 (first-cell-wins race
// resolution + cluster food share + reward) sequentially against the
// downloaded `(food_idx, value)` arrays.
//
// Per-tick uploads are small and bounded: food positions/kinds/ages
// (variable count, max ~ field_sources_cap) + per-cell pose/body/carnivore
// (n_cells × ~24 bytes). Compared to the legacy CPU path this eliminates
// the ~27% of CPU time spent in `SpatialGrid::for_each_in_radius`.

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
pub struct EatFoodParamsGpu {
    pub num_cells: u32,
    pub num_foods: u32,
    pub cell_size: f32,
    pub eat_radius: f32,
    pub world_half_x: f32,
    pub world_half_y: f32,
    pub world_half_z: f32,
    pub world_map_nx: u32,
    pub world_map_ny: u32,
    pub world_map_nz: u32,
    pub fixed_timestep_hz: f32,
    pub plant_food_value: f32,
    pub carrion_food_value: f32,
    pub carrion_decay_per_sec: f32,
    pub world_map_food_floor: f32,
    pub world_map_food_amp: f32,
}

#[derive(Debug, Clone, Default)]
pub struct EatFoodResult {
    /// Per-cell candidate food index. `num_foods` (one past last valid
    /// idx) is the sentinel for "this cell ate nothing this tick".
    pub food_idx: Vec<u32>,
    /// Per-cell candidate food value (energy gain pre-share).
    pub value: Vec<f32>,
}

pub struct EatFoodGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    cell_capacity: usize,
    food_capacity: usize,
    params_buf: wgpu::Buffer,
    // Per-tick CPU → GPU uploads. All live in the same struct so capacity
    // doesn't have to reallocate when the food count drifts during a run.
    cell_positions_buf: wgpu::Buffer,
    cell_headings_buf: wgpu::Buffer,
    cell_pitches_buf: wgpu::Buffer,
    cell_body_dims_buf: wgpu::Buffer,
    cell_carnivore_buf: wgpu::Buffer,
    cell_max_axes_buf: wgpu::Buffer,
    food_positions_buf: wgpu::Buffer,
    food_kinds_buf: wgpu::Buffer,
    food_age_ticks_buf: wgpu::Buffer,
    world_map_buf: wgpu::Buffer,
    out_food_idx_buf: wgpu::Buffer,
    out_value_buf: wgpu::Buffer,
    out_food_idx_rb: wgpu::Buffer,
    out_value_rb: wgpu::Buffer,
}

impl EatFoodGpu {
    pub fn with_context(
        ctx: &GpuContext,
        cell_capacity: usize,
        food_capacity: usize,
        world_map_size: u64,
    ) -> Result<Self, String> {
        Self::with_device_inner(
            Arc::clone(&ctx.device),
            Arc::clone(&ctx.queue),
            cell_capacity,
            food_capacity,
            world_map_size,
        )
    }

    fn with_device_inner(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        cell_capacity: usize,
        food_capacity: usize,
        world_map_size: u64,
    ) -> Result<Self, String> {
        assert!(cell_capacity > 0, "eat_food cell capacity must be > 0");
        assert!(food_capacity > 0, "eat_food food capacity must be > 0");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("eat-food"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/eat_food.wgsl").into()),
        });
        // 15 bindings: 1 uniform + 14 storage. Read-write: out_food_idx (13),
        // out_value (14). Read-only: everything else.
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..15)
            .map(|i| {
                let ty = if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else if i == 13 || i == 14 {
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
            label: Some("eat-food-bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("eat-food-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("eat-food-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("eat_food"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("eat-food-params"),
            contents: bytemuck::bytes_of(&EatFoodParamsGpu::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let f = std::mem::size_of::<f32>() as u64;
        let u4 = std::mem::size_of::<u32>() as u64;
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
        let n = cell_capacity as u64;
        let m = food_capacity as u64;
        let cell_positions_buf = mk("eat-food-cell-positions", n * 3 * f, stor_dst);
        let cell_headings_buf = mk("eat-food-cell-headings", n * f, stor_dst);
        let cell_pitches_buf = mk("eat-food-cell-pitches", n * f, stor_dst);
        let cell_body_dims_buf = mk("eat-food-cell-body-dims", n * 3 * f, stor_dst);
        let cell_carnivore_buf = mk("eat-food-cell-carnivore", n * f, stor_dst);
        let cell_max_axes_buf = mk("eat-food-cell-max-axes", n * f, stor_dst);
        let food_positions_buf = mk("eat-food-food-positions", m * 3 * f, stor_dst);
        let food_kinds_buf = mk("eat-food-food-kinds", m * u4, stor_dst);
        let food_age_ticks_buf = mk("eat-food-food-age-ticks", m * u4, stor_dst);
        let world_map_buf = mk("eat-food-world-map", world_map_size * f, stor_dst);
        let out_food_idx_buf = mk("eat-food-out-food-idx", n * u4, stor_dst_src);
        let out_value_buf = mk("eat-food-out-value", n * f, stor_dst_src);
        let out_food_idx_rb = mk("eat-food-out-food-idx-rb", n * u4, read);
        let out_value_rb = mk("eat-food-out-value-rb", n * f, read);

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            cell_capacity,
            food_capacity,
            params_buf,
            cell_positions_buf,
            cell_headings_buf,
            cell_pitches_buf,
            cell_body_dims_buf,
            cell_carnivore_buf,
            cell_max_axes_buf,
            food_positions_buf,
            food_kinds_buf,
            food_age_ticks_buf,
            world_map_buf,
            out_food_idx_buf,
            out_value_buf,
            out_food_idx_rb,
            out_value_rb,
        })
    }

    pub fn cell_capacity(&self) -> usize {
        self.cell_capacity
    }

    pub fn food_capacity(&self) -> usize {
        self.food_capacity
    }

    /// Upload the world-map richness field. Called once at init — the field
    /// only changes on `WorldMap` rebuild (not a thing today).
    pub fn upload_world_map(&self, field: &[f32]) {
        self.queue
            .write_buffer(&self.world_map_buf, 0, bytemuck::cast_slice(field));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &mut self,
        cell_positions: &[[f32; 3]],
        cell_headings: &[f32],
        cell_pitches: &[f32],
        cell_body_dims: &[[f32; 3]],
        cell_carnivore: &[f32],
        cell_max_axes: &[f32],
        food_positions: &[[f32; 3]],
        food_kinds: &[u32],
        food_age_ticks: &[u32],
        food_hash: &SpatialHashGpu,
        params: EatFoodParamsGpu,
    ) -> EatFoodResult {
        let n_cells = cell_positions.len();
        let n_foods = food_positions.len();
        assert_eq!(cell_headings.len(), n_cells);
        assert_eq!(cell_pitches.len(), n_cells);
        assert_eq!(cell_body_dims.len(), n_cells);
        assert_eq!(cell_carnivore.len(), n_cells);
        assert_eq!(cell_max_axes.len(), n_cells);
        assert_eq!(food_kinds.len(), n_foods);
        assert_eq!(food_age_ticks.len(), n_foods);
        assert!(n_cells <= self.cell_capacity, "eat_food cell overflow");
        assert!(n_foods <= self.food_capacity, "eat_food food overflow");
        if n_cells == 0 {
            return EatFoodResult::default();
        }

        // Sentinel "no match" output for every cell when there are no foods.
        // The shader will write the sentinel regardless, but skipping the
        // dispatch lets us avoid uploading empty food buffers (and matches
        // the n_cells == 0 fast path above).
        let mut params = params;
        params.num_cells = n_cells as u32;
        params.num_foods = n_foods as u32;
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        // Per-tick uploads.
        self.queue
            .write_buffer(&self.cell_positions_buf, 0, bytemuck::cast_slice(cell_positions));
        self.queue
            .write_buffer(&self.cell_headings_buf, 0, bytemuck::cast_slice(cell_headings));
        self.queue
            .write_buffer(&self.cell_pitches_buf, 0, bytemuck::cast_slice(cell_pitches));
        self.queue
            .write_buffer(&self.cell_body_dims_buf, 0, bytemuck::cast_slice(cell_body_dims));
        self.queue
            .write_buffer(&self.cell_carnivore_buf, 0, bytemuck::cast_slice(cell_carnivore));
        self.queue
            .write_buffer(&self.cell_max_axes_buf, 0, bytemuck::cast_slice(cell_max_axes));
        if n_foods > 0 {
            self.queue.write_buffer(
                &self.food_positions_buf,
                0,
                bytemuck::cast_slice(food_positions),
            );
            self.queue
                .write_buffer(&self.food_kinds_buf, 0, bytemuck::cast_slice(food_kinds));
            self.queue.write_buffer(
                &self.food_age_ticks_buf,
                0,
                bytemuck::cast_slice(food_age_ticks),
            );
        }

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("eat-food-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.cell_positions_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.cell_headings_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.cell_pitches_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.cell_body_dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.cell_carnivore_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.cell_max_axes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.food_positions_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.food_kinds_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.food_age_ticks_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: food_hash.offsets_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: food_hash.sorted_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: self.world_map_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: self.out_food_idx_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 14, resource: self.out_value_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("eat-food-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("eat-food-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = ((n_cells as u32) + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        let idx_bytes = (n_cells as u64) * 4;
        let val_bytes = (n_cells as u64) * 4;
        encoder.copy_buffer_to_buffer(
            &self.out_food_idx_buf,
            0,
            &self.out_food_idx_rb,
            0,
            idx_bytes,
        );
        encoder.copy_buffer_to_buffer(&self.out_value_buf, 0, &self.out_value_rb, 0, val_bytes);
        self.queue.submit(Some(encoder.finish()));

        let idx_slice = self.out_food_idx_rb.slice(0..idx_bytes);
        let val_slice = self.out_value_rb.slice(0..val_bytes);
        idx_slice.map_async(wgpu::MapMode::Read, |_| {});
        val_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let idx_data = idx_slice.get_mapped_range();
        let val_data = val_slice.get_mapped_range();
        let food_idx: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&idx_data).to_vec();
        let value: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&val_data).to_vec();
        drop(idx_data);
        drop(val_data);
        self.out_food_idx_rb.unmap();
        self.out_value_rb.unmap();
        EatFoodResult { food_idx, value }
    }
}
