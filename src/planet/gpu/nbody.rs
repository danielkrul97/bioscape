//! Direct N² gravitational acceleration on the GPU. See
//! `shaders/planet_nbody.wgsl` for the tiled algorithm.

use crate::gpu::GpuContext;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct NBodyParams {
    num_particles: u32,
    pad_a0: u32,
    pad_a1: u32,
    pad_a2: u32,
    g: f32,
    eps2: f32,
    pad_b0: f32,
    pad_b1: f32,
}

pub struct NBodyGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    capacity: usize,
    params_buf: wgpu::Buffer,
    positions_buf: wgpu::Buffer,
    masses_buf: wgpu::Buffer,
    accelerations_buf: wgpu::Buffer,
    accelerations_rb: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl NBodyGpu {
    pub fn with_context(ctx: &GpuContext, capacity: usize) -> Result<Self, String> {
        Self::new_inner(Arc::clone(&ctx.device), Arc::clone(&ctx.queue), capacity)
    }

    fn new_inner(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        capacity: usize,
    ) -> Result<Self, String> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planet-nbody"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/planet_nbody.wgsl").into(),
            ),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..4)
            .map(|i| {
                let ty = if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else if i == 3 {
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
            label: Some("planet-nbody-bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("planet-nbody-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("planet-nbody-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("nbody"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet-nbody-params"),
            contents: bytemuck::bytes_of(&NBodyParams::default()),
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
        let stor = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let positions_buf = mk("planet-nbody-pos", n * 3 * f, stor);
        let masses_buf = mk("planet-nbody-mass", n * f, stor);
        let accelerations_buf = mk("planet-nbody-acc", n * 3 * f, stor);
        let accelerations_rb = mk("planet-nbody-acc-rb", n * 3 * f, read);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet-nbody-bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: positions_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: masses_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: accelerations_buf.as_entire_binding() },
            ],
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            capacity,
            params_buf,
            positions_buf,
            masses_buf,
            accelerations_buf,
            accelerations_rb,
            bind_group,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Convenience wrapper — upload positions/masses, dispatch, read
    /// back accelerations. Used by tests and any callsite that doesn't
    /// share a command encoder.
    pub fn compute(
        &mut self,
        positions: &[[f32; 3]],
        masses: &[f32],
        g: f32,
        softening: f32,
    ) -> Vec<[f32; 3]> {
        let n = positions.len();
        assert_eq!(masses.len(), n);
        assert!(n <= self.capacity);
        if n == 0 {
            return Vec::new();
        }

        let params = NBodyParams {
            num_particles: n as u32,
            g,
            eps2: softening * softening,
            ..NBodyParams::default()
        };
        let pos_flat: Vec<f32> = positions.iter().flatten().copied().collect();

        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&self.positions_buf, 0, bytemuck::cast_slice(&pos_flat));
        self.queue
            .write_buffer(&self.masses_buf, 0, bytemuck::cast_slice(masses));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-nbody-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("planet-nbody-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let wg = ((n as u32) + 63) / 64;
            pass.dispatch_workgroups(wg, 1, 1);
        }
        let bytes = (n as u64) * 3 * std::mem::size_of::<f32>() as u64;
        encoder.copy_buffer_to_buffer(&self.accelerations_buf, 0, &self.accelerations_rb, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.accelerations_rb.slice(0..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let f: &[f32] = bytemuck::cast_slice(&data);
        let result: Vec<[f32; 3]> = (0..n)
            .map(|i| [f[i * 3], f[i * 3 + 1], f[i * 3 + 2]])
            .collect();
        drop(data);
        self.accelerations_rb.unmap();
        result
    }

    pub fn positions_buffer(&self) -> &wgpu::Buffer {
        &self.positions_buf
    }
    pub fn masses_buffer(&self) -> &wgpu::Buffer {
        &self.masses_buf
    }
    pub fn accelerations_buffer(&self) -> &wgpu::Buffer {
        &self.accelerations_buf
    }
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Hot-loop entry — record dispatch into a caller-owned encoder.
    /// Used by the S207 leapfrog driver. Caller is responsible for
    /// uploading buffers and submitting.
    pub fn dispatch_into(&self, encoder: &mut wgpu::CommandEncoder, n: usize) {
        if n == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("planet-nbody-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        let wg = ((n as u32) + 63) / 64;
        pass.dispatch_workgroups(wg, 1, 1);
    }

    /// Update the uniform params block. Call before `dispatch_into`
    /// when `n`, `g`, or `softening` change.
    pub fn upload_params(&self, n: usize, g: f32, softening: f32) {
        let params = NBodyParams {
            num_particles: n as u32,
            g,
            eps2: softening * softening,
            ..NBodyParams::default()
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
    }
}
