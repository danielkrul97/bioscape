use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::*;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct IzhikevichParams {
    num_cells: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Sprint 147: GPU Izhikevich forward. Runs after `BrainGpu` (Perceptron)
/// per tick — for cells whose `neuron_models[i] == 1` the shader overwrites
/// `last_hidden` / `last_outputs` with spike-rate-derived values. Mutates
/// `membrane` / `recovery` in place.
pub struct IzhikevichGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
}

impl IzhikevichGpu {
    pub fn with_context(ctx: &GpuContext) -> Result<Self, String> {
        Self::with_device_inner(Arc::clone(&ctx.device), Arc::clone(&ctx.queue))
    }

    pub fn new() -> Result<Self, String> {
        let ctx = GpuContext::new()?;
        Self::with_device_inner(ctx.device, ctx.queue)
    }

    fn with_device_inner(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Result<Self, String> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brain_forward_izhikevich"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/brain_forward_izhikevich.wgsl").into(),
            ),
        });
        // 8 bindings: params + inputs + weights + hidden + outputs +
        // membrane + recovery + neuron_models. hidden/outputs/membrane/
        // recovery are rw; inputs/weights/neuron_models are ro.
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..8)
            .map(|i| {
                let ty = match i {
                    0 => wgpu::BufferBindingType::Uniform,
                    3 | 4 | 5 | 6 => wgpu::BufferBindingType::Storage { read_only: false },
                    _ => wgpu::BufferBindingType::Storage { read_only: true },
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
            label: Some("izhikevich-bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("izhikevich-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("izhikevich-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("forward_izhikevich"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("izhikevich-params"),
            contents: bytemuck::bytes_of(&IzhikevichParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            params_buf,
        })
    }

    pub fn dispatch(&self, cells_gpu: &CellsGpu, n: usize) {
        if n == 0 {
            return;
        }
        let params = IzhikevichParams {
            num_cells: n as u32,
            ..IzhikevichParams::default()
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("izhikevich-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cells_gpu.last_inputs_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cells_gpu.brain_weights_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cells_gpu.last_hidden_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cells_gpu.last_outputs_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cells_gpu.membrane_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: cells_gpu.recovery_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: cells_gpu.neuron_models_buffer().as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("izhikevich-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("izhikevich-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}
