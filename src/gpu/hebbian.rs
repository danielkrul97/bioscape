use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::*;
use super::*;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct HebbianParams {
    num_cells: u32,
    learning_rate: f32,
    _pad0: u32,
    _pad1: u32,
}

pub struct HebbianGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    capacity: usize,
    params_buf: wgpu::Buffer,
    inputs_buf: wgpu::Buffer,
    hidden_buf: wgpu::Buffer,
    outputs_buf: wgpu::Buffer,
    rewards_buf: wgpu::Buffer,
    weights_buf: wgpu::Buffer,
    weights_rb: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl HebbianGpu {
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
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hebbian"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/hebbian.wgsl").into()),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..6)
            .map(|i| {
                let ty = if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else if i == 5 {
                    wgpu::BufferBindingType::Storage { read_only: false }
                } else {
                    wgpu::BufferBindingType::Storage { read_only: true }
                };
                wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                }
            })
            .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hebbian-bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hebbian-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hebbian-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("hebbian"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hebbian-params"),
            contents: bytemuck::bytes_of(&HebbianParams::default()),
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
        let stor_dst_src = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let inputs_buf = mk("hebbian-inputs", n * (BRAIN_INPUTS as u64) * f, stor_dst);
        let hidden_buf = mk("hebbian-hidden", n * (BRAIN_HIDDEN as u64) * f, stor_dst);
        let outputs_buf = mk("hebbian-outputs", n * (BRAIN_OUTPUTS as u64) * f, stor_dst);
        let rewards_buf = mk("hebbian-rewards", n * f, stor_dst);
        let weights_buf = mk("hebbian-weights", n * (BRAIN_WEIGHTS_PER_CELL as u64) * f, stor_dst_src);
        let weights_rb = mk("hebbian-weights-rb", n * (BRAIN_WEIGHTS_PER_CELL as u64) * f, read);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hebbian-bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: inputs_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: hidden_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: outputs_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: rewards_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: weights_buf.as_entire_binding() },
            ],
        });
        Ok(Self {
            device, queue, pipeline, bind_group_layout, capacity,
            params_buf, inputs_buf, hidden_buf, outputs_buf, rewards_buf, weights_buf, weights_rb, bind_group,
        })
    }

    pub fn compute<'a, I>(
        &mut self,
        last_inputs: &[[f32; BRAIN_INPUTS]],
        last_hidden: &[[f32; BRAIN_HIDDEN]],
        last_outputs: &[[f32; BRAIN_OUTPUTS]],
        rewards: &[f32],
        brains: I,
        learning_rate: f32,
    ) -> Vec<Brain>
    where
        I: IntoIterator<Item = &'a Brain>,
    {
        let n = last_inputs.len();
        assert_eq!(last_hidden.len(), n);
        assert_eq!(last_outputs.len(), n);
        assert_eq!(rewards.len(), n);
        assert!(n <= self.capacity);
        if n == 0 {
            return Vec::new();
        }

        let inputs_flat: Vec<f32> = last_inputs.iter().flatten().copied().collect();
        let hidden_flat: Vec<f32> = last_hidden.iter().flatten().copied().collect();
        let outputs_flat: Vec<f32> = last_outputs.iter().flatten().copied().collect();
        let mut weights_flat: Vec<f32> = Vec::with_capacity(n * BRAIN_WEIGHTS_PER_CELL);
        let mut count = 0;
        for brain in brains.into_iter().take(n) {
            for row in brain.w1.iter() { weights_flat.extend_from_slice(row); }
            weights_flat.extend_from_slice(&brain.b1);
            for row in brain.w2.iter() { weights_flat.extend_from_slice(row); }
            weights_flat.extend_from_slice(&brain.b2);
            count += 1;
        }
        assert_eq!(count, n, "brains iterator length mismatch");

        let params = HebbianParams {
            num_cells: n as u32,
            learning_rate,
            ..HebbianParams::default()
        };
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(&self.inputs_buf, 0, bytemuck::cast_slice(&inputs_flat));
        self.queue.write_buffer(&self.hidden_buf, 0, bytemuck::cast_slice(&hidden_flat));
        self.queue.write_buffer(&self.outputs_buf, 0, bytemuck::cast_slice(&outputs_flat));
        self.queue.write_buffer(&self.rewards_buf, 0, bytemuck::cast_slice(rewards));
        self.queue.write_buffer(&self.weights_buf, 0, bytemuck::cast_slice(&weights_flat));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hebbian-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hebbian-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
        }
        let bytes = (n as u64) * (BRAIN_WEIGHTS_PER_CELL as u64) * 4;
        encoder.copy_buffer_to_buffer(&self.weights_buf, 0, &self.weights_rb, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let s = self.weights_rb.slice(0..bytes);
        s.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = s.get_mapped_range();
        let f: &[f32] = bytemuck::cast_slice(&data);
        let mut out: Vec<Brain> = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * BRAIN_WEIGHTS_PER_CELL;
            let mut b = Brain {
                hidden_n: BRAIN_HIDDEN_DEFAULT as u32,
                w1: [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN],
                b1: [0.0; BRAIN_HIDDEN],
                w2: [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
                b2: [0.0; BRAIN_OUTPUTS],
            };
            for h in 0..BRAIN_HIDDEN {
                for in_i in 0..BRAIN_INPUTS {
                    b.w1[h][in_i] = f[off + h * BRAIN_INPUTS + in_i];
                }
            }
            for h in 0..BRAIN_HIDDEN {
                b.b1[h] = f[off + B1_OFFSET + h];
            }
            for o in 0..BRAIN_OUTPUTS {
                for h in 0..BRAIN_HIDDEN {
                    b.w2[o][h] = f[off + W2_OFFSET + o * BRAIN_HIDDEN + h];
                }
            }
            for o in 0..BRAIN_OUTPUTS {
                b.b2[o] = f[off + B2_OFFSET + o];
            }
            out.push(b);
        }
        drop(data);
        self.weights_rb.unmap();
        out
    }

    pub fn weights_buffer(&self) -> &wgpu::Buffer {
        &self.weights_buf
    }

    /// Persistent-mode dispatch: binds the `CellsGpu` buffers
    /// (`last_inputs`, `last_hidden`, `last_outputs`, `rewards`,
    /// `brain_weights`). Brain weights stay resident on the GPU across
    /// ticks; this call mutates them in place. Run after `upload_rewards()`.
    pub fn compute_persistent(
        &self,
        cells_gpu: &CellsGpu,
        n: usize,
        learning_rate: f32,
    ) {
        if n == 0 {
            return;
        }
        let params = HebbianParams {
            num_cells: n as u32,
            learning_rate,
            ..HebbianParams::default()
        };
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hebbian-bg-persistent"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cells_gpu.last_inputs_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cells_gpu.last_hidden_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cells_gpu.last_outputs_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cells_gpu.rewards_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cells_gpu.brain_weights_buffer().as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hebbian-encoder-persistent"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hebbian-pass-persistent"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}

