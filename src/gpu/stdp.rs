use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::*;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct StdpParams {
    num_cells: u32,
    tick: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct StdpStepParams {
    num_cells: u32,
    tick: u32,
    /// `exp(-1 / max(1, tau_ticks))`, precomputed on the CPU so the shader
    /// doesn't repeat the `exp` on every cell every tick.
    decay: f32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct StdpApplyParams {
    num_cells: u32,
    tick: u32,
    a_plus: f32,
    a_minus: f32,
}

/// Sprint 165: pre-spike encoder. Per Izhikevich cell, write
/// `pre_spike_times[i] = tick` when an input crosses the encoding
/// threshold.
pub struct StdpEncodePreGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
}

/// Sprint 166: trace decay + accumulate. Mirror of `Brain::stdp_step`.
pub struct StdpStepGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
}

/// Sprint 167: reward-modulated STDP weight update. Mirror of
/// `Brain::stdp_apply_rewarded` — LTP / LTD via per-neuron traces.
pub struct StdpApplyGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
}

impl StdpEncodePreGpu {
    pub fn with_context(ctx: &GpuContext) -> Result<Self, String> {
        let device = Arc::clone(&ctx.device);
        let queue = Arc::clone(&ctx.queue);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stdp_encode_pre"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/stdp_encode_pre.wgsl").into(),
            ),
        });
        let entries = build_layout(&[
            BindKind::Uniform,
            BindKind::StorageRo,
            BindKind::StorageRo,
            BindKind::StorageRw,
        ]);
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stdp-encode-pre-bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stdp-encode-pre-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("stdp-encode-pre-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("stdp_encode_pre"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stdp-encode-pre-params"),
            contents: bytemuck::bytes_of(&StdpParams::default()),
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

    pub fn dispatch(&self, cells_gpu: &CellsGpu, n: usize, tick: u32) {
        if n == 0 {
            return;
        }
        let params = StdpParams {
            num_cells: n as u32,
            tick,
            ..StdpParams::default()
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stdp-encode-pre-bg"),
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
                    resource: cells_gpu.neuron_models_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cells_gpu.pre_spike_times_buffer().as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("stdp-encode-pre-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stdp-encode-pre-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}

impl StdpStepGpu {
    pub fn with_context(ctx: &GpuContext) -> Result<Self, String> {
        let device = Arc::clone(&ctx.device);
        let queue = Arc::clone(&ctx.queue);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stdp_step"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/stdp_step.wgsl").into()),
        });
        let entries = build_layout(&[
            BindKind::Uniform,
            BindKind::StorageRo, // pre_spike_times
            BindKind::StorageRo, // post_spike_times
            BindKind::StorageRw, // pre_trace
            BindKind::StorageRw, // post_trace
            BindKind::StorageRo, // neuron_models
        ]);
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stdp-step-bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stdp-step-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("stdp-step-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("stdp_step"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stdp-step-params"),
            contents: bytemuck::bytes_of(&StdpStepParams::default()),
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

    pub fn dispatch(&self, cells_gpu: &CellsGpu, n: usize, tick: u32, tau_ticks: f32) {
        if n == 0 {
            return;
        }
        let tau = tau_ticks.max(1.0);
        let params = StdpStepParams {
            num_cells: n as u32,
            tick,
            decay: (-1.0 / tau).exp(),
            ..StdpStepParams::default()
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stdp-step-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cells_gpu.pre_spike_times_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cells_gpu.post_spike_times_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cells_gpu.pre_trace_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cells_gpu.post_trace_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cells_gpu.neuron_models_buffer().as_entire_binding() },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("stdp-step-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stdp-step-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}

impl StdpApplyGpu {
    pub fn with_context(ctx: &GpuContext) -> Result<Self, String> {
        let device = Arc::clone(&ctx.device);
        let queue = Arc::clone(&ctx.queue);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stdp_apply"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/stdp_apply.wgsl").into()),
        });
        // 8 bindings: params + weights rw + pre_trace ro + post_trace ro +
        // pre_spike_times ro + post_spike_times ro + rewards ro +
        // neuron_models ro.
        let entries = build_layout(&[
            BindKind::Uniform,
            BindKind::StorageRw,
            BindKind::StorageRo,
            BindKind::StorageRo,
            BindKind::StorageRo,
            BindKind::StorageRo,
            BindKind::StorageRo,
            BindKind::StorageRo,
        ]);
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stdp-apply-bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stdp-apply-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("stdp-apply-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("stdp_apply"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stdp-apply-params"),
            contents: bytemuck::bytes_of(&StdpApplyParams::default()),
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

    pub fn dispatch(
        &self,
        cells_gpu: &CellsGpu,
        n: usize,
        tick: u32,
        a_plus: f32,
        a_minus: f32,
    ) {
        if n == 0 {
            return;
        }
        let params = StdpApplyParams {
            num_cells: n as u32,
            tick,
            a_plus,
            a_minus,
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stdp-apply-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cells_gpu.brain_weights_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cells_gpu.pre_trace_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cells_gpu.post_trace_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cells_gpu.pre_spike_times_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cells_gpu.post_spike_times_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: cells_gpu.rewards_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: cells_gpu.neuron_models_buffer().as_entire_binding() },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("stdp-apply-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stdp-apply-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}

#[derive(Copy, Clone)]
enum BindKind {
    Uniform,
    StorageRo,
    StorageRw,
}

fn build_layout(kinds: &[BindKind]) -> Vec<wgpu::BindGroupLayoutEntry> {
    kinds
        .iter()
        .enumerate()
        .map(|(i, kind)| {
            let ty = match kind {
                BindKind::Uniform => wgpu::BufferBindingType::Uniform,
                BindKind::StorageRo => wgpu::BufferBindingType::Storage { read_only: true },
                BindKind::StorageRw => wgpu::BufferBindingType::Storage { read_only: false },
            };
            wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        })
        .collect()
}
