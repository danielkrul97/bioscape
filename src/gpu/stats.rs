use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::*;
use super::*;

// ============================================================================
// Sprint 47: GPU stats reduction (single-workgroup tree reduce)
// ============================================================================

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct StatsParams {
    num_cells: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CellStats {
    pub sum_x: f32,
    pub sum_y: f32,
    pub sum_z: f32,
    pub sum_speed_sq: f32,
    pub sum_energy: f32,
}

pub struct StatsGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    capacity: usize,
    params_buf: wgpu::Buffer,
    positions_buf: wgpu::Buffer,
    velocities_buf: wgpu::Buffer,
    energies_buf: wgpu::Buffer,
    output_buf: wgpu::Buffer,
    output_readback: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pos_packed: Vec<f32>,
    vel_packed: Vec<f32>,
}

impl StatsGpu {
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
            label: Some("cell_stats"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/cell_stats.wgsl").into()),
        });
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("stats-bgl"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stats-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("stats-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("reduce"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stats-params"),
            contents: bytemuck::bytes_of(&StatsParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let (positions_buf, velocities_buf, energies_buf, output_buf, output_readback) =
            Self::alloc_buffers(&device, capacity);
        let bind_group = Self::make_bind_group(
            &device,
            &bind_group_layout,
            &params_buf,
            &positions_buf,
            &velocities_buf,
            &energies_buf,
            &output_buf,
        );
        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            capacity,
            params_buf,
            positions_buf,
            velocities_buf,
            energies_buf,
            output_buf,
            output_readback,
            bind_group,
            pos_packed: Vec::new(),
            vel_packed: Vec::new(),
        })
    }

    fn alloc_buffers(
        device: &wgpu::Device,
        capacity: usize,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        let pos_size = (capacity * 3 * std::mem::size_of::<f32>()) as u64;
        let energy_size = (capacity * std::mem::size_of::<f32>()) as u64;
        let output_size = (8 * std::mem::size_of::<f32>()) as u64;
        let positions_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stats-positions"),
            size: pos_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let velocities_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stats-velocities"),
            size: pos_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let energies_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stats-energies"),
            size: energy_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stats-output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stats-readback"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (positions_buf, velocities_buf, energies_buf, output_buf, output_readback)
    }

    fn make_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        params: &wgpu::Buffer,
        positions: &wgpu::Buffer,
        velocities: &wgpu::Buffer,
        energies: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stats-bg"),
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
                    resource: velocities.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: energies.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output.as_entire_binding(),
                },
            ],
        })
    }

    fn ensure_capacity(&mut self, n: usize) {
        if n <= self.capacity {
            return;
        }
        let new_cap = (self.capacity * 2).max(n);
        let (p, v, e, o, or_) = Self::alloc_buffers(&self.device, new_cap);
        self.positions_buf = p;
        self.velocities_buf = v;
        self.energies_buf = e;
        self.output_buf = o;
        self.output_readback = or_;
        self.bind_group = Self::make_bind_group(
            &self.device,
            &self.bind_group_layout,
            &self.params_buf,
            &self.positions_buf,
            &self.velocities_buf,
            &self.energies_buf,
            &self.output_buf,
        );
        self.capacity = new_cap;
    }

    /// Compute reduction over per-cell SoA. Single workgroup tree reduce → 5
    /// floats v `output[0..5]`. Caller obvykle dělí sumy `n` aby získal mean.
    pub fn compute(
        &mut self,
        positions: &[[f32; 3]],
        velocities: &[[f32; 3]],
        energies: &[f32],
    ) -> CellStats {
        let n = positions.len();
        assert_eq!(velocities.len(), n);
        assert_eq!(energies.len(), n);
        if n == 0 {
            return CellStats::default();
        }
        self.ensure_capacity(n);

        self.pos_packed.clear();
        self.pos_packed.reserve(n * 3);
        for p in positions {
            self.pos_packed.extend_from_slice(p);
        }
        self.vel_packed.clear();
        self.vel_packed.reserve(n * 3);
        for v in velocities {
            self.vel_packed.extend_from_slice(v);
        }

        let params = StatsParams {
            num_cells: n as u32,
            ..StatsParams::default()
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(
            &self.positions_buf,
            0,
            bytemuck::cast_slice(&self.pos_packed),
        );
        self.queue.write_buffer(
            &self.velocities_buf,
            0,
            bytemuck::cast_slice(&self.vel_packed),
        );
        self.queue
            .write_buffer(&self.energies_buf, 0, bytemuck::cast_slice(energies));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("stats-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stats-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let bytes = (8 * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.output_buf, 0, &self.output_readback, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.output_readback.slice(0..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let stats = CellStats {
            sum_x: floats[0],
            sum_y: floats[1],
            sum_z: floats[2],
            sum_speed_sq: floats[3],
            sum_energy: floats[4],
        };
        drop(data);
        self.output_readback.unmap();
        stats
    }
}

