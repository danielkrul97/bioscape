use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::*;

// GPU spatial hash via counting sort. Layout pinned to
// `shaders/spatial_hash.wgsl`; the bucket grid is fixed at
// 64 × 32 × 4 = 8192 buckets covering ±2048 / ±512 / ±128 world units at
// `GRID_CELL_SIZE = 64`.

pub const GPU_HASH_GRID_NX: i32 = 64;
pub const GPU_HASH_GRID_NY: i32 = 32;
pub const GPU_HASH_GRID_NZ: i32 = 4;
pub const GPU_HASH_NUM_BUCKETS: usize =
    (GPU_HASH_GRID_NX * GPU_HASH_GRID_NY * GPU_HASH_GRID_NZ) as usize;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct HashParams {
    num_cells: u32,
    cell_size: f32,
    world_half_x: f32,
    world_half_y: f32,
}

pub struct SpatialHashGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline_count: wgpu::ComputePipeline,
    pipeline_prefix: wgpu::ComputePipeline,
    pipeline_scatter: wgpu::ComputePipeline,
    pipeline_sort: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    capacity: usize,
    cell_size: f32,
    world_half_xy: [f32; 2],
    params_buf: wgpu::Buffer,
    positions_buf: wgpu::Buffer,
    counts_buf: wgpu::Buffer,
    offsets_buf: wgpu::Buffer,
    sorted_buf: wgpu::Buffer,
    offsets_readback: wgpu::Buffer,
    sorted_readback: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Persistent zero buffer used to reset `counts` between rebuilds.
    /// Allocating 32 KB per rebuild (× 2 grids per tick) showed up in
    /// profiles, so we keep one buffer alive for the lifetime of the hash.
    counts_zero: Vec<u8>,
    epoch: u64,
}

impl SpatialHashGpu {
    pub fn new(capacity: usize, cell_size: f32, world_half_xy: [f32; 2]) -> Result<Self, String> {
        assert!(capacity > 0);
        let ctx = GpuContext::new()?;
        Self::with_device_inner(ctx.device, ctx.queue, capacity, cell_size, world_half_xy)
    }

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

    fn with_device_inner(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        capacity: usize,
        cell_size: f32,
        world_half_xy: [f32; 2],
    ) -> Result<Self, String> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("spatial_hash"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/spatial_hash.wgsl").into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hash-bgl"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hash-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let make_pipe = |entry: &str, label: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let pipeline_count = make_pipe("count", "hash-count");
        let pipeline_prefix = make_pipe("prefix_sum", "hash-prefix");
        let pipeline_scatter = make_pipe("scatter", "hash-scatter");
        let pipeline_sort = make_pipe("sort_buckets", "hash-sort");

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hash-params"),
            contents: bytemuck::bytes_of(&HashParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let (positions_buf, counts_buf, offsets_buf, sorted_buf, offsets_readback, sorted_readback) =
            Self::alloc_buffers(&device, capacity);
        let bind_group = Self::make_bind_group(
            &device,
            &bind_group_layout,
            &params_buf,
            &positions_buf,
            &counts_buf,
            &offsets_buf,
            &sorted_buf,
        );

        Ok(Self {
            device,
            queue,
            pipeline_count,
            pipeline_prefix,
            pipeline_scatter,
            pipeline_sort,
            bind_group_layout,
            capacity,
            cell_size,
            world_half_xy,
            params_buf,
            positions_buf,
            counts_buf,
            offsets_buf,
            sorted_buf,
            offsets_readback,
            sorted_readback,
            bind_group,
            counts_zero: vec![0u8; GPU_HASH_NUM_BUCKETS * 4],
            epoch: 0,
        })
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    fn alloc_buffers(
        device: &wgpu::Device,
        capacity: usize,
    ) -> (
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::Buffer,
    ) {
        let pos_size = (capacity * 3 * std::mem::size_of::<f32>()) as u64;
        let counts_size = (GPU_HASH_NUM_BUCKETS * std::mem::size_of::<u32>()) as u64;
        let offsets_size = ((GPU_HASH_NUM_BUCKETS + 1) * std::mem::size_of::<u32>()) as u64;
        let sorted_size = (capacity * std::mem::size_of::<u32>()) as u64;
        let positions_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hash-positions"),
            size: pos_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let counts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hash-counts"),
            size: counts_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let offsets_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hash-offsets"),
            size: offsets_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let sorted_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hash-sorted"),
            size: sorted_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let offsets_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hash-offsets-readback"),
            size: offsets_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sorted_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hash-sorted-readback"),
            size: sorted_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (
            positions_buf,
            counts_buf,
            offsets_buf,
            sorted_buf,
            offsets_readback,
            sorted_readback,
        )
    }

    fn make_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        params: &wgpu::Buffer,
        positions: &wgpu::Buffer,
        counts: &wgpu::Buffer,
        offsets: &wgpu::Buffer,
        sorted: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hash-bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sorted.as_entire_binding(),
                },
            ],
        })
    }

    fn ensure_capacity(&mut self, n: usize) {
        if n <= self.capacity {
            return;
        }
        let new_cap = (self.capacity * 2).max(n);
        let (p, c, o, s, or_, sr) = Self::alloc_buffers(&self.device, new_cap);
        self.positions_buf = p;
        self.counts_buf = c;
        self.offsets_buf = o;
        self.sorted_buf = s;
        self.offsets_readback = or_;
        self.sorted_readback = sr;
        self.bind_group = Self::make_bind_group(
            &self.device,
            &self.bind_group_layout,
            &self.params_buf,
            &self.positions_buf,
            &self.counts_buf,
            &self.offsets_buf,
            &self.sorted_buf,
        );
        self.capacity = new_cap;
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// Vrátí `(offsets[NUM_BUCKETS+1], sorted_cells[N])`. offsets[b] je
    /// inclusive začátek a offsets[b+1] exclusive konec range v sorted_cells
    /// pro bucket b. offsets[NUM_BUCKETS] = N (total).
    pub fn rebuild(&mut self, positions: &[[f32; 3]]) -> (Vec<u32>, Vec<u32>) {
        let n = positions.len();
        if n == 0 {
            return (vec![0; GPU_HASH_NUM_BUCKETS + 1], Vec::new());
        }
        self.ensure_capacity(n);

        // Reset counts to 0 (rebuild expects fresh state). Persistent
        // counts_zero buffer — pre-fix alokoval čerstvě 32 KB / dispatch.
        self.queue
            .write_buffer(&self.counts_buf, 0, &self.counts_zero);

        // Direct cast `&[[f32; 3]]` → `&[f32]` přes bytemuck — `[f32;3]` má
        // identický layout, takže positions_packed kopie odpadá. Necháváme
        // pole ve struktu (potenciální fallback), ale nevyplňujeme.
        let params = HashParams {
            num_cells: n as u32,
            cell_size: self.cell_size,
            world_half_x: self.world_half_xy[0],
            world_half_y: self.world_half_xy[1],
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&self.positions_buf, 0, bytemuck::cast_slice(positions));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hash-encoder"),
            });
        let workgroups = ((n as u32) + 63) / 64;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hash-count-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_count);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hash-prefix-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_prefix);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hash-scatter-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_scatter);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hash-sort-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_sort);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups((GPU_HASH_NUM_BUCKETS as u32 + 63) / 64, 1, 1);
        }

        let offsets_bytes = ((GPU_HASH_NUM_BUCKETS + 1) * 4) as u64;
        let sorted_bytes = (n * 4) as u64;
        encoder.copy_buffer_to_buffer(
            &self.offsets_buf,
            0,
            &self.offsets_readback,
            0,
            offsets_bytes,
        );
        encoder.copy_buffer_to_buffer(&self.sorted_buf, 0, &self.sorted_readback, 0, sorted_bytes);
        self.queue.submit(Some(encoder.finish()));

        let off_slice = self.offsets_readback.slice(0..offsets_bytes);
        let sor_slice = self.sorted_readback.slice(0..sorted_bytes);
        off_slice.map_async(wgpu::MapMode::Read, |_| {});
        sor_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let offsets: Vec<u32> = {
            let data = off_slice.get_mapped_range();
            bytemuck::cast_slice::<u8, u32>(&data).to_vec()
        };
        let sorted: Vec<u32> = {
            let data = sor_slice.get_mapped_range();
            bytemuck::cast_slice::<u8, u32>(&data).to_vec()
        };
        self.offsets_readback.unmap();
        self.sorted_readback.unmap();
        (offsets, sorted)
    }

    /// `rebuild()` variant that skips the per-tick readback — submits the
    /// count → prefix → scatter dispatch and returns. Used by the full-GPU
    /// sensor pipeline where the chained `SensorGatherGpu` reads
    /// `offsets`/`sorted` directly via the accessor methods. Eliminates
    /// the two `device.poll(Wait)` round-trips that `rebuild` does for
    /// the cell hash and food hash each tick.
    pub fn dispatch(&mut self, positions: &[[f32; 3]]) {
        if positions.is_empty() {
            return;
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hash-dispatch-encoder"),
            });
        self.dispatch_into(&mut encoder, positions);
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn dispatch_into(&mut self, encoder: &mut wgpu::CommandEncoder, positions: &[[f32; 3]]) {
        let n = positions.len();
        if n == 0 {
            return;
        }
        self.ensure_capacity(n);

        self.queue
            .write_buffer(&self.counts_buf, 0, &self.counts_zero);
        let params = HashParams {
            num_cells: n as u32,
            cell_size: self.cell_size,
            world_half_x: self.world_half_xy[0],
            world_half_y: self.world_half_xy[1],
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&self.positions_buf, 0, bytemuck::cast_slice(positions));

        let workgroups = ((n as u32) + 63) / 64;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hash-count-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_count);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hash-prefix-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_prefix);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hash-scatter-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_scatter);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hash-sort-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_sort);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups((GPU_HASH_NUM_BUCKETS as u32 + 63) / 64, 1, 1);
        }
    }

    /// Accessors for chained shaders (`NeighborsGpu`, `SensorGatherGpu`)
    /// — bind the hash buffers as read-only inputs to a downstream pass.
    /// The buffers stay valid for the lifetime of `SpatialHashGpu`.
    pub fn offsets_buffer(&self) -> &wgpu::Buffer {
        &self.offsets_buf
    }

    pub fn sorted_buffer(&self) -> &wgpu::Buffer {
        &self.sorted_buf
    }

    /// CPU mirror of the shader's `bucket_id_of`. Useful for tests and for
    /// CPU code that needs to match the GPU bucket layout.
    pub fn bucket_id_of(pos: [f32; 3], cell_size: f32) -> u32 {
        let bx = (pos[0] / cell_size).floor() as i32 + GPU_HASH_GRID_NX / 2;
        let by = (pos[1] / cell_size).floor() as i32 + GPU_HASH_GRID_NY / 2;
        let bz = (pos[2] / cell_size).floor() as i32 + GPU_HASH_GRID_NZ / 2;
        let bx_c = bx.clamp(0, GPU_HASH_GRID_NX - 1);
        let by_c = by.clamp(0, GPU_HASH_GRID_NY - 1);
        let bz_c = bz.clamp(0, GPU_HASH_GRID_NZ - 1);
        (bx_c + by_c * GPU_HASH_GRID_NX + bz_c * GPU_HASH_GRID_NX * GPU_HASH_GRID_NY) as u32
    }
}
