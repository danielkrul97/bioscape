use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::*;

// Wave J: GPU food rejection sampling. K parallel attempts → CPU picks valid
// candidates up to per-tick spawn budget. World map + obstacle mask uploaded
// once at init (rebuilt on maze curriculum changes via `upload_obstacle`);
// per-thread xoshiro state mutates across dispatches for fresh streams.

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
pub struct FoodSpawnParamsGpu {
    pub num_attempts: u32,
    pub rejection_strength: f32,
    pub eat_radius: f32,
    pub cell_size: f32,
    pub world_half_x: f32,
    pub world_half_y: f32,
    pub world_half_z: f32,
    pub num_cells: u32,
    pub world_map_nx: u32,
    pub world_map_ny: u32,
    pub world_map_nz: u32,
    pub obstacle_active: u32,
    pub obstacle_nx: u32,
    pub obstacle_ny: u32,
    pub obstacle_nz: u32,
    pub _pad0: u32,
}

#[derive(Debug, Clone)]
pub struct FoodSpawnResult {
    pub candidate_positions: Vec<[f32; 3]>,
    pub valid_mask: Vec<u32>,
}

pub struct FoodSpawnGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Per-attempt xoshiro128++ state (vec4<u32>). Sized at construction;
    /// callers reseed via `seed_attempts` when the dispatch count changes
    /// or when reproducibility resets (e.g. `--checkpoint` load).
    capacity: usize,
    /// Number of u32 entries currently sitting in `world_map_buf` /
    /// `obstacle_mask_buf`. Both buffers are sized to the maximum supported
    /// grid res; uploads update a leading slice.
    params_buf: wgpu::Buffer,
    xoshiro_buf: wgpu::Buffer,
    world_map_buf: wgpu::Buffer,
    obstacle_buf: wgpu::Buffer,
    candidates_buf: wgpu::Buffer,
    valid_mask_buf: wgpu::Buffer,
    candidates_rb: wgpu::Buffer,
    valid_mask_rb: wgpu::Buffer,
}

impl FoodSpawnGpu {
    pub fn with_context(
        ctx: &GpuContext,
        capacity: usize,
        world_map_size: u64,
        obstacle_mask_size: u64,
    ) -> Result<Self, String> {
        Self::with_device_inner(
            Arc::clone(&ctx.device),
            Arc::clone(&ctx.queue),
            capacity,
            world_map_size,
            obstacle_mask_size,
        )
    }

    pub fn new(
        capacity: usize,
        world_map_size: u64,
        obstacle_mask_size: u64,
    ) -> Result<Self, String> {
        let ctx = GpuContext::new()?;
        Self::with_device_inner(
            ctx.device,
            ctx.queue,
            capacity,
            world_map_size,
            obstacle_mask_size,
        )
    }

    fn with_device_inner(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        capacity: usize,
        world_map_size: u64,
        obstacle_mask_size: u64,
    ) -> Result<Self, String> {
        assert!(capacity > 0, "food_spawn capacity must be > 0");
        assert!(world_map_size > 0, "world_map_size must be > 0");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("food-spawn"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/food_spawn.wgsl").into()),
        });
        // 10 bindings: 1 uniform + 9 storage. Read-write: xoshiro (1),
        // candidates (8), valid_mask (9). Read-only: everything else.
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..10)
            .map(|i| {
                let ty = if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else if i == 1 || i == 8 || i == 9 {
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
            label: Some("food-spawn-bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("food-spawn-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("food-spawn-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("food_spawn"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("food-spawn-params"),
            contents: bytemuck::bytes_of(&FoodSpawnParamsGpu::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let f = std::mem::size_of::<f32>() as u64;
        let u4 = std::mem::size_of::<[u32; 4]>() as u64;
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
        let k = capacity as u64;
        let xoshiro_buf = mk("food-spawn-xoshiro", k * u4, stor_dst);
        let world_map_buf = mk("food-spawn-world-map", world_map_size * f, stor_dst);
        // Obstacle mask may be empty when --maze is not used; allocate at
        // least 4 bytes so the binding is valid even when unused.
        let mask_size = obstacle_mask_size.max(1) * 4;
        let obstacle_buf = mk("food-spawn-obstacle", mask_size, stor_dst);
        let candidates_buf = mk("food-spawn-candidates", k * 3 * f, stor_dst_src);
        let valid_mask_buf = mk("food-spawn-valid-mask", k * 4, stor_dst_src);
        let candidates_rb = mk("food-spawn-candidates-rb", k * 3 * f, read);
        let valid_mask_rb = mk("food-spawn-valid-mask-rb", k * 4, read);

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            capacity,
            params_buf,
            xoshiro_buf,
            world_map_buf,
            obstacle_buf,
            candidates_buf,
            valid_mask_buf,
            candidates_rb,
            valid_mask_rb,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Upload `K × vec4<u32>` xoshiro seeds. Caller derives these from a
    /// CPU RNG so the per-attempt streams stay deterministic with the seed.
    pub fn seed_attempts(&self, seeds: &[[u32; 4]]) {
        assert!(seeds.len() <= self.capacity, "food-spawn seed overflow");
        self.queue
            .write_buffer(&self.xoshiro_buf, 0, bytemuck::cast_slice(seeds));
    }

    /// Upload the static world-map richness field. Called once per init
    /// (and again on `WorldMap` rebuild — none today). Layout is flat
    /// `field[zi * nx * ny + yi * nx + xi]`, matching `WorldMap.field`.
    pub fn upload_world_map(&self, field: &[f32]) {
        self.queue
            .write_buffer(&self.world_map_buf, 0, bytemuck::cast_slice(field));
    }

    /// Upload an obstacle mask (`1u` = blocked). Skipped when maze is off —
    /// the buffer still exists (1 u32 placeholder) and `obstacle_active`
    /// param flag gates the shader-side sample.
    pub fn upload_obstacle(&self, mask: &[u32]) {
        if mask.is_empty() {
            return;
        }
        self.queue
            .write_buffer(&self.obstacle_buf, 0, bytemuck::cast_slice(mask));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &mut self,
        num_attempts: usize,
        cell_positions: &[[f32; 3]],
        cell_max_axes: &[f32],
        cell_hash: &SpatialHashGpu,
        params: FoodSpawnParamsGpu,
    ) -> FoodSpawnResult {
        assert!(num_attempts <= self.capacity, "food-spawn dispatch overflow");
        let n_cells = cell_positions.len();
        assert_eq!(cell_max_axes.len(), n_cells, "max_axes length mismatch");
        if num_attempts == 0 {
            return FoodSpawnResult {
                candidate_positions: Vec::new(),
                valid_mask: Vec::new(),
            };
        }

        let mut params = params;
        params.num_attempts = num_attempts as u32;
        params.num_cells = n_cells as u32;
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        // Per-tick read-only inputs. cell_positions/max_axes are CPU-side
        // because the spawn dispatch runs before the rest of the GPU
        // pipeline this tick — caller hasn't refreshed the persistent
        // cells.position buffer at this point.
        let mut pos_packed: Vec<f32> = Vec::with_capacity(n_cells * 3);
        for p in cell_positions {
            pos_packed.extend_from_slice(p);
        }
        // We reuse the candidates / valid_mask buffers as read-write
        // outputs, so they don't need clearing — every thread unconditionally
        // writes to its slot.
        let cell_positions_buf =
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("food-spawn-cell-positions-tick"),
                contents: bytemuck::cast_slice(&pos_packed),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let cell_max_axes_buf =
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("food-spawn-cell-max-axes-tick"),
                contents: bytemuck::cast_slice(cell_max_axes),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("food-spawn-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.xoshiro_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.world_map_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.obstacle_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cell_positions_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cell_max_axes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: cell_hash.offsets_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: cell_hash.sorted_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.candidates_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.valid_mask_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("food-spawn-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("food-spawn-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = ((num_attempts as u32) + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        let pos_bytes = (num_attempts as u64) * 3 * 4;
        let mask_bytes = (num_attempts as u64) * 4;
        encoder.copy_buffer_to_buffer(&self.candidates_buf, 0, &self.candidates_rb, 0, pos_bytes);
        encoder.copy_buffer_to_buffer(&self.valid_mask_buf, 0, &self.valid_mask_rb, 0, mask_bytes);
        self.queue.submit(Some(encoder.finish()));

        let pos_slice = self.candidates_rb.slice(0..pos_bytes);
        let mask_slice = self.valid_mask_rb.slice(0..mask_bytes);
        pos_slice.map_async(wgpu::MapMode::Read, |_| {});
        mask_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let pos_data = pos_slice.get_mapped_range();
        let mask_data = mask_slice.get_mapped_range();
        let pf: &[f32] = bytemuck::cast_slice(&pos_data);
        let mf: &[u32] = bytemuck::cast_slice(&mask_data);
        let candidates: Vec<[f32; 3]> = (0..num_attempts)
            .map(|i| [pf[i * 3], pf[i * 3 + 1], pf[i * 3 + 2]])
            .collect();
        let mask: Vec<u32> = mf.to_vec();
        drop(pos_data);
        drop(mask_data);
        self.candidates_rb.unmap();
        self.valid_mask_rb.unmap();
        FoodSpawnResult {
            candidate_positions: candidates,
            valid_mask: mask,
        }
    }
}
