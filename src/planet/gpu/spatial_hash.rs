//! GPU spatial hash for SPH neighbour search. 32³ uniform-grid
//! counting sort. Cell size chosen so the 3×3×3 stencil covers
//! `2 h_max` — i.e., `cell_size ≥ 4 h_max / 3` for full coverage.

use crate::gpu::GpuContext;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub const GRID_N: u32 = 32;
pub const NUM_BUCKETS: u32 = GRID_N * GRID_N * GRID_N;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct HashParams {
    num_particles: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    world_half: f32,
    cell_size: f32,
    pad_a0: f32,
    pad_a1: f32,
}

pub struct SpatialHashGpu {
    pub ctx: Arc<GpuContext>,
    pub capacity: usize,
    pub world_half: f32,

    params_buf: wgpu::Buffer,
    counts_buf: wgpu::Buffer,
    offsets_buf: wgpu::Buffer,
    sorted_buf: wgpu::Buffer,

    offsets_rb: wgpu::Buffer,
    sorted_rb: wgpu::Buffer,

    count_pipeline: wgpu::ComputePipeline,
    prefix_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
    sort_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl SpatialHashGpu {
    pub fn new(
        ctx: Arc<GpuContext>,
        capacity: usize,
        world_half: f32,
        positions_buf: &wgpu::Buffer,
    ) -> Result<Self, String> {
        let device = &ctx.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planet-spatial-hash"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/planet_spatial_hash.wgsl").into(),
            ),
        });

        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..5)
            .map(|i| {
                let ty = if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else if i == 1 {
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
        let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("planet-hash-bgl"),
            entries: &entries,
        });
        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("planet-hash-pl"),
            bind_group_layouts: &[&bg_layout],
            push_constant_ranges: &[],
        });
        let mk_pipeline = |entry: &str, label: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let count_pipeline = mk_pipeline("count", "planet-hash-count");
        let prefix_pipeline = mk_pipeline("prefix_sum", "planet-hash-prefix");
        let scatter_pipeline = mk_pipeline("scatter", "planet-hash-scatter");
        let sort_pipeline = mk_pipeline("sort_buckets", "planet-hash-sort");

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet-hash-params"),
            contents: bytemuck::bytes_of(&HashParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let u = std::mem::size_of::<u32>() as u64;
        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let stor = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let counts_buf = mk("planet-hash-counts", (NUM_BUCKETS as u64) * u, stor);
        let offsets_buf = mk("planet-hash-offsets", (NUM_BUCKETS as u64 + 1) * u, stor);
        let sorted_buf = mk("planet-hash-sorted", (capacity as u64) * u, stor);
        let offsets_rb = mk("planet-hash-offsets-rb", (NUM_BUCKETS as u64 + 1) * u, read);
        let sorted_rb = mk("planet-hash-sorted-rb", (capacity as u64) * u, read);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet-hash-bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: positions_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: counts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: offsets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: sorted_buf.as_entire_binding() },
            ],
        });

        Ok(Self {
            ctx,
            capacity,
            world_half,
            params_buf,
            counts_buf,
            offsets_buf,
            sorted_buf,
            offsets_rb,
            sorted_rb,
            count_pipeline,
            prefix_pipeline,
            scatter_pipeline,
            sort_pipeline,
            bind_group,
        })
    }

    pub fn cell_size(&self) -> f32 {
        2.0 * self.world_half / GRID_N as f32
    }

    /// Maximum smoothing length this grid can support with a full 3×3×3
    /// neighbour scan: `h_max = (3/4) · cell_size`.
    pub fn max_supported_h(&self) -> f32 {
        0.75 * self.cell_size()
    }

    pub fn counts_buffer(&self) -> &wgpu::Buffer {
        &self.counts_buf
    }
    pub fn offsets_buffer(&self) -> &wgpu::Buffer {
        &self.offsets_buf
    }
    pub fn sorted_buffer(&self) -> &wgpu::Buffer {
        &self.sorted_buf
    }

    pub fn rebuild(&self, n: usize) {
        if n == 0 {
            return;
        }
        let params = HashParams {
            num_particles: n as u32,
            world_half: self.world_half,
            cell_size: self.cell_size(),
            ..HashParams::default()
        };
        self.ctx
            .queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let device = &self.ctx.device;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("planet-hash-encoder"),
        });
        // Zero counts before count pass.
        encoder.clear_buffer(&self.counts_buf, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("planet-hash-count-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.count_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let wg = ((n as u32) + 63) / 64;
            pass.dispatch_workgroups(wg, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("planet-hash-prefix-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.prefix_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("planet-hash-scatter-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scatter_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let wg = ((n as u32) + 63) / 64;
            pass.dispatch_workgroups(wg, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("planet-hash-sort-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sort_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let wg = (NUM_BUCKETS + 63) / 64;
            pass.dispatch_workgroups(wg, 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn download_offsets(&self) -> Vec<u32> {
        let bytes = (NUM_BUCKETS as u64 + 1) * 4;
        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("planet-hash-offsets-readback"),
        });
        encoder.copy_buffer_to_buffer(&self.offsets_buf, 0, &self.offsets_rb, 0, bytes);
        self.ctx.queue.submit(Some(encoder.finish()));
        let slice = self.offsets_rb.slice(0..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.ctx.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let out: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.offsets_rb.unmap();
        out
    }

    pub fn download_sorted(&self, n: usize) -> Vec<u32> {
        if n == 0 {
            return Vec::new();
        }
        let bytes = (n as u64) * 4;
        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("planet-hash-sorted-readback"),
        });
        encoder.copy_buffer_to_buffer(&self.sorted_buf, 0, &self.sorted_rb, 0, bytes);
        self.ctx.queue.submit(Some(encoder.finish()));
        let slice = self.sorted_rb.slice(0..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.ctx.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let out: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.sorted_rb.unmap();
        out
    }
}

/// Helper: compute the bucket id for a position, matching the WGSL
/// `bucket_id_of`. Used by CPU-side reference tests.
pub fn bucket_id_cpu(pos: [f32; 3], world_half: f32) -> u32 {
    let cell_size = 2.0 * world_half / GRID_N as f32;
    let half_n = (GRID_N as i32) / 2;
    let bx = ((pos[0] / cell_size).floor() as i32 + half_n).clamp(0, GRID_N as i32 - 1);
    let by = ((pos[1] / cell_size).floor() as i32 + half_n).clamp(0, GRID_N as i32 - 1);
    let bz = ((pos[2] / cell_size).floor() as i32 + half_n).clamp(0, GRID_N as i32 - 1);
    (bx + by * GRID_N as i32 + bz * (GRID_N * GRID_N) as i32) as u32
}
