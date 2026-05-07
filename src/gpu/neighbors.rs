use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::*;
use super::*;

// ============================================================================
// Sprint 49: GPU broad-phase neighbors query (chains SpatialHashGpu output)
// ============================================================================

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct NeighborsParams {
    num_cells: u32,
    cell_size: f32,
    /// Sprint 55: toroidal bounds.
    world_half_x: f32,
    world_half_y: f32,
}

/// Per-cell broad-phase result. `nearest_cell` = None pokud žádný neighbor
/// uvnitř `vision_radius` neexistuje.
#[derive(Debug, Default, Clone, Copy)]
pub struct NeighborResult {
    pub nearest_cell: Option<([f32; 3], f32)>,
    pub neighbors_in_vision: u32,
}

pub struct NeighborsGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    capacity: usize,
    cell_size: f32,
    /// Sprint 55: toroidal bounds.
    world_half_xy: [f32; 2],
    params_buf: wgpu::Buffer,
    positions_buf: wgpu::Buffer,
    radii_buf: wgpu::Buffer,
    vision_buf: wgpu::Buffer,
    output_buf: wgpu::Buffer,
    output_readback: wgpu::Buffer,
    pos_packed: Vec<f32>,
}

impl NeighborsGpu {
    pub fn with_context(
        ctx: &GpuContext,
        capacity: usize,
        cell_size: f32,
        world_half_xy: [f32; 2],
    ) -> Result<Self, String> {
        Self::with_device_inner(
            Arc::clone(&ctx.device),
            Arc::clone(&ctx.queue),
            capacity,
            cell_size,
            world_half_xy,
        )
    }

    pub fn new(capacity: usize, cell_size: f32, world_half_xy: [f32; 2]) -> Result<Self, String> {
        let ctx = GpuContext::new()?;
        Self::with_device_inner(ctx.device, ctx.queue, capacity, cell_size, world_half_xy)
    }

    fn with_device_inner(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        capacity: usize,
        cell_size: f32,
        world_half_xy: [f32; 2],
    ) -> Result<Self, String> {
        assert!(capacity > 0);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cell_neighbors"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/cell_neighbors.wgsl").into(),
            ),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("neighbors-bgl"),
                entries: &[
                    // 0: params uniform
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
                    // 1-5: read-only storage (positions, radii, vision, hash offsets, hash sorted)
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 6: output (read-write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("neighbors-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("neighbors-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("neighbors"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("neighbors-params"),
            contents: bytemuck::bytes_of(&NeighborsParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let (positions_buf, radii_buf, vision_buf, output_buf, output_readback) =
            Self::alloc_buffers(&device, capacity);

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            capacity,
            cell_size,
            world_half_xy,
            params_buf,
            positions_buf,
            radii_buf,
            vision_buf,
            output_buf,
            output_readback,
            pos_packed: Vec::new(),
        })
    }

    fn alloc_buffers(
        device: &wgpu::Device,
        capacity: usize,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        let pos_size = (capacity * 3 * std::mem::size_of::<f32>()) as u64;
        let scalar_size = (capacity * std::mem::size_of::<f32>()) as u64;
        let output_size = (capacity * 5 * std::mem::size_of::<f32>()) as u64;
        let positions_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("neighbors-positions"),
            size: pos_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let radii_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("neighbors-radii"),
            size: scalar_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vision_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("neighbors-vision"),
            size: scalar_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("neighbors-output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("neighbors-readback"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (positions_buf, radii_buf, vision_buf, output_buf, output_readback)
    }

    fn ensure_capacity(&mut self, n: usize) {
        if n <= self.capacity {
            return;
        }
        let new_cap = (self.capacity * 2).max(n);
        let (p, r, v, o, or_) = Self::alloc_buffers(&self.device, new_cap);
        self.positions_buf = p;
        self.radii_buf = r;
        self.vision_buf = v;
        self.output_buf = o;
        self.output_readback = or_;
        self.capacity = new_cap;
    }

    /// Sprint 49: bind cell hash + cells data, dispatch, readback.
    /// Caller je zodpovědný za to, že `cell_hash` byl rebuildnut s těmi
    /// samými positions (jinak GPU dostane mismatch indices).
    pub fn compute(
        &mut self,
        positions: &[[f32; 3]],
        radii: &[f32],
        vision_radii: &[f32],
        cell_hash: &SpatialHashGpu,
    ) -> Vec<NeighborResult> {
        let n = positions.len();
        assert_eq!(radii.len(), n);
        assert_eq!(vision_radii.len(), n);
        if n == 0 {
            return Vec::new();
        }
        self.ensure_capacity(n);

        self.pos_packed.clear();
        self.pos_packed.reserve(n * 3);
        for p in positions {
            self.pos_packed.extend_from_slice(p);
        }

        let params = NeighborsParams {
            num_cells: n as u32,
            cell_size: self.cell_size,
            world_half_x: self.world_half_xy[0],
            world_half_y: self.world_half_xy[1],
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(
            &self.positions_buf,
            0,
            bytemuck::cast_slice(&self.pos_packed),
        );
        self.queue
            .write_buffer(&self.radii_buf, 0, bytemuck::cast_slice(radii));
        self.queue
            .write_buffer(&self.vision_buf, 0, bytemuck::cast_slice(vision_radii));

        // Bind group musí refekrencovat cell_hash buffery — vytváří se per call.
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("neighbors-bg"),
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
                    resource: self.radii_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.vision_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cell_hash.offsets_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cell_hash.sorted_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.output_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("neighbors-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("neighbors-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = ((n as u32) + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        let bytes = (n * 5 * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.output_buf, 0, &self.output_readback, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.output_readback.slice(0..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * 5;
            let dx = floats[off];
            let dy = floats[off + 1];
            let dz = floats[off + 2];
            let radius = floats[off + 3];
            let count_bits = floats[off + 4].to_bits();
            let nearest = if radius < 0.0 {
                None
            } else {
                Some(([positions[i][0] + dx, positions[i][1] + dy, positions[i][2] + dz], radius))
            };
            out.push(NeighborResult {
                nearest_cell: nearest,
                neighbors_in_vision: count_bits,
            });
        }
        drop(data);
        self.output_readback.unmap();
        out
    }
}

