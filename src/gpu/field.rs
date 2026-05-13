use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::*;

// GPU field diffusion shared by smell and pheromone fields — same compute
// path, separate `FieldGpu` instances.

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct FieldParams {
    res_x: u32,
    res_y: u32,
    res_z: u32,
    num_sources: u32,
    diffusion: f32,
    decay: f32,
    cell_size_x: f32,
    cell_size_y: f32,
    cell_size_z: f32,
    world_half_x: f32,
    world_half_y: f32,
    world_half_z: f32,
    /// Wave 4: when non-zero, the diffuse shader treats voxels with
    /// `mask[idx] != 0u` as Neumann zero-flux walls (mirror of CPU
    /// `SmellField::step_masked`). Mask buffer is binding 4.
    mask_active: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct FieldGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline_deposit: wgpu::ComputePipeline,
    pipeline_diffuse: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    grid_a: wgpu::Buffer,
    grid_b: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    sources_buf: wgpu::Buffer,
    /// Wave 4: per-voxel obstacle mask (binding 4). One u32 per voxel,
    /// matching grid layout (`k * res_x * res_y + j * res_x + i`). Value 0
    /// = passable, non-zero = wall. Sized = grid size; zeroed at
    /// allocation so the no-maze path stays neutral.
    mask_buf: wgpu::Buffer,
    /// Wave 4: cached active-mask state. When `false`, the params'
    /// `mask_active` flag is also 0 and shader skips the mask check.
    mask_is_active: bool,
    grid_readback: wgpu::Buffer,
    bg_round_a: wgpu::BindGroup,
    bg_round_b: wgpu::BindGroup,
    pending_sources: Vec<f32>, // [px, py, pz, amount] × N
    capacity_sources: usize,
    current_is_a: bool,
    /// 3D resolution and world bounds. XY is toroidal, Z is bounded.
    resolution: [usize; 3],
    world_half: [f32; 3],
}

impl FieldGpu {
    pub fn new(
        resolution: [usize; 3],
        world_half: [f32; 3],
        sources_capacity: usize,
    ) -> Result<Self, String> {
        assert!(resolution.iter().all(|&r| r >= 2) && sources_capacity > 0);
        let ctx = GpuContext::new()?;
        Self::with_device_inner(ctx.device, ctx.queue, resolution, world_half, sources_capacity)
    }

    pub fn with_context(
        ctx: &GpuContext,
        resolution: [usize; 3],
        world_half: [f32; 3],
        sources_capacity: usize,
    ) -> Result<Self, String> {
        assert!(resolution.iter().all(|&r| r >= 2) && sources_capacity > 0);
        Self::with_device_inner(
            Arc::clone(&ctx.device),
            Arc::clone(&ctx.queue),
            resolution,
            world_half,
            sources_capacity,
        )
    }

    fn with_device_inner(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        resolution: [usize; 3],
        world_half: [f32; 3],
        sources_capacity: usize,
    ) -> Result<Self, String> {
        let shader_deposit = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("field_deposit"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/field_deposit.wgsl").into(),
            ),
        });
        let shader_diffuse = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("field_diffuse"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/field_diffuse.wgsl").into(),
            ),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("field-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("field-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let make_pipe = |module: &wgpu::ShaderModule, entry: &str, label: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let pipeline_deposit = make_pipe(&shader_deposit, "deposit", "field-deposit");
        let pipeline_diffuse = make_pipe(&shader_diffuse, "diffuse", "field-diffuse");

        let grid_size_bytes =
            (resolution[0] * resolution[1] * resolution[2] * std::mem::size_of::<u32>()) as u64;
        let grid_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field-grid-a"),
            size: grid_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grid_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field-grid-b"),
            size: grid_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Zero both grids so the first ping-pong reads from a clean slate.
        queue.write_buffer(&grid_a, 0, &vec![0u8; grid_size_bytes as usize]);
        queue.write_buffer(&grid_b, 0, &vec![0u8; grid_size_bytes as usize]);

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("field-params"),
            contents: bytemuck::bytes_of(&FieldParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // 4 floats per source (px, py, pz, amount).
        let sources_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field-sources"),
            size: (sources_capacity * 4 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Wave 4: per-voxel obstacle mask buffer (binding 4). Same size as
        // grid; zeroed initially so without an `upload_obstacle_mask` call
        // the shader behaves identically to the pre-Wave-4 (no-mask) path.
        let mask_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field-mask"),
            size: grid_size_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&mask_buf, 0, &vec![0u8; grid_size_bytes as usize]);

        let make_bg = |grid_in: &wgpu::Buffer, grid_out: &wgpu::Buffer, label: &str| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: sources_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: grid_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: grid_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: mask_buf.as_entire_binding(),
                    },
                ],
            })
        };
        let bg_round_a = make_bg(&grid_a, &grid_b, "field-bg-round-a");
        let bg_round_b = make_bg(&grid_b, &grid_a, "field-bg-round-b");

        let grid_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field-grid-readback"),
            size: grid_size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            pipeline_deposit,
            pipeline_diffuse,
            bind_group_layout,
            grid_a,
            grid_b,
            params_buf,
            sources_buf,
            mask_buf,
            mask_is_active: false,
            grid_readback,
            bg_round_a,
            bg_round_b,
            pending_sources: Vec::new(),
            capacity_sources: sources_capacity,
            current_is_a: true,
            resolution,
            world_half,
        })
    }

    pub fn resolution(&self) -> [usize; 3] {
        self.resolution
    }

    pub fn world_half(&self) -> [f32; 3] {
        self.world_half
    }

    fn cell_size(&self, axis: usize) -> f32 {
        (2.0 * self.world_half[axis]) / self.resolution[axis] as f32
    }

    /// 3D mirror of `SmellField::add_source` — 4 floats per source
    /// (px, py, pz, amount). Pending sources are flushed to the GPU on the
    /// next `step()` call.
    pub fn add_source(&mut self, pos: [f32; 3], amount: f32) {
        self.pending_sources.push(pos[0]);
        self.pending_sources.push(pos[1]);
        self.pending_sources.push(pos[2]);
        self.pending_sources.push(amount);
    }

    /// Reallocates the sources buffer when pending source count exceeds
    /// current capacity. Geometric (×2) growth.
    fn ensure_sources_capacity(&mut self, num_sources: usize) {
        if num_sources <= self.capacity_sources {
            return;
        }
        let new_cap = (self.capacity_sources * 2).max(num_sources);
        self.sources_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field-sources"),
            size: (new_cap * 4 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.capacity_sources = new_cap;
        // Bind groups capture `sources_buf`, so they must be rebuilt.
        let make_bg = |grid_in: &wgpu::Buffer, grid_out: &wgpu::Buffer, label: &str| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.sources_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: grid_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: grid_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.mask_buf.as_entire_binding(),
                    },
                ],
            })
        };
        self.bg_round_a = make_bg(&self.grid_a, &self.grid_b, "field-bg-round-a");
        self.bg_round_b = make_bg(&self.grid_b, &self.grid_a, "field-bg-round-b");
    }

    /// Wave 4: upload an obstacle mask sized to this field's grid (one bool
    /// per voxel). Switches the diffuse shader to the masked path until
    /// `clear_obstacle_mask` is called. The mask is zero-padded if shorter
    /// than expected; oversize is rejected.
    pub fn upload_obstacle_mask(&mut self, mask: &[bool]) {
        let n = self.resolution[0] * self.resolution[1] * self.resolution[2];
        debug_assert_eq!(mask.len(), n, "field mask len mismatch");
        let packed: Vec<u32> = mask.iter().map(|&b| if b { 1 } else { 0 }).collect();
        self.queue
            .write_buffer(&self.mask_buf, 0, bytemuck::cast_slice(&packed));
        self.mask_is_active = true;
    }

    /// Wave 4: deactivate the obstacle mask (homogeneous diffusion path).
    pub fn clear_obstacle_mask(&mut self) {
        self.mask_is_active = false;
    }

    /// 3D mirror of `SmellField::step`: 7-point Jacobi stencil with
    /// toroidal XY and bounded Z. `decay_per_sec` is converted to a
    /// per-step factor `(1 - decay × dt).max(0)`.
    pub fn step(&mut self, diffusion: f32, decay_per_sec: f32, dt: f32) {
        let num_sources = self.pending_sources.len() / 4;
        self.ensure_sources_capacity(num_sources.max(1));
        if num_sources > 0 {
            self.queue.write_buffer(
                &self.sources_buf,
                0,
                bytemuck::cast_slice(&self.pending_sources),
            );
        }
        let params = FieldParams {
            res_x: self.resolution[0] as u32,
            res_y: self.resolution[1] as u32,
            res_z: self.resolution[2] as u32,
            num_sources: num_sources as u32,
            diffusion,
            decay: (1.0 - decay_per_sec * dt).max(0.0),
            cell_size_x: self.cell_size(0),
            cell_size_y: self.cell_size(1),
            cell_size_z: self.cell_size(2),
            world_half_x: self.world_half[0],
            world_half_y: self.world_half[1],
            world_half_z: self.world_half[2],
            mask_active: if self.mask_is_active { 1 } else { 0 },
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let bg = if self.current_is_a {
            &self.bg_round_a
        } else {
            &self.bg_round_b
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("field-encoder"),
            });
        if num_sources > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("field-deposit-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_deposit);
            pass.set_bind_group(0, bg, &[]);
            let workgroups = ((num_sources as u32) + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("field-diffuse-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_diffuse);
            pass.set_bind_group(0, bg, &[]);
            // 3D dispatch — shader uses `@workgroup_size(4, 4, 4)`.
            let wg_x = ((self.resolution[0] as u32) + 3) / 4;
            let wg_y = ((self.resolution[1] as u32) + 3) / 4;
            let wg_z = ((self.resolution[2] as u32) + 3) / 4;
            pass.dispatch_workgroups(wg_x, wg_y, wg_z);
        }
        self.queue.submit(Some(encoder.finish()));

        self.current_is_a = !self.current_is_a;
        self.pending_sources.clear();
    }

    /// Accessor used by chained shaders (sensor gather) that sample the
    /// field inline. Returns the buffer holding the latest post-step state.
    pub fn current_grid_buffer(&self) -> &wgpu::Buffer {
        if self.current_is_a {
            &self.grid_a
        } else {
            &self.grid_b
        }
    }

    /// Download the current grid (post-diffuse output) as a `Vec<f32>`.
    /// Slow — used by tests and visualization. The hot sensor path samples
    /// the buffer on the GPU instead and never reads it back.
    pub fn download(&mut self) -> Vec<f32> {
        let n = self.resolution[0] * self.resolution[1] * self.resolution[2];
        let bytes = (n * 4) as u64;
        // After `step()` the swap has flipped `current_is_a`, so "current"
        // refers to whichever grid was the *output* of the latest pass.
        let src = if self.current_is_a {
            &self.grid_a
        } else {
            &self.grid_b
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("field-readback-encoder"),
            });
        encoder.copy_buffer_to_buffer(src, 0, &self.grid_readback, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let slice = self.grid_readback.slice(0..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        self.grid_readback.unmap();
        result
    }
}

