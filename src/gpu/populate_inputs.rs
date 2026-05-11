use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::*;

// `PopulateInputsGpu` — fuses the sensor stage into the brain forward
// pipeline by writing brain inputs directly into `last_inputs` on the GPU.

/// Shader params for `populate_inputs.wgsl`. Constants mirror
/// `lib::populate_brain_inputs` (BRAIN_INPUTS layout, normalization gains,
/// `DENSITY_NORM_COUNT`).
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
pub struct PopulateInputsParams {
    pub num_cells: u32,
    pub brain_inputs: u32,
    pub brain_inputs_sensory: u32,
    pub brain_hidden: u32,
    pub brain_recurrent: u32,
    pub smell_norm_gain: f32,
    pub phero_norm_gain: f32,
    pub damage_norm_gain: f32,
    pub density_norm: f32,
    pub reproduce_threshold: f32,
    pub vibration_norm_gain: f32,
    pub _pad0: u32,
}

pub struct PopulateInputsGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    cached_bg: Option<wgpu::BindGroup>,
    cached_cells_epoch: u64,
    cached_sensor_epoch: u64,
}

impl PopulateInputsGpu {
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
            label: Some("populate_inputs"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/populate_inputs.wgsl").into(),
            ),
        });
        // 13 bindings: 0 uniform, 1-10 + 12 storage read, 11 storage read_write,
        // 6 (damage_accums) je read_write, 11 (last_inputs) read_write,
        // 12 (bonded_inbox) je read-only.
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..13u32)
            .map(|i| {
                let ty = if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else if i == 6 || i == 11 {
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
            label: Some("populate-inputs-bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("populate-inputs-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("populate-inputs-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("populate_inputs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("populate-inputs-params"),
            contents: bytemuck::bytes_of(&PopulateInputsParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            params_buf,
            cached_bg: None,
            cached_cells_epoch: 0,
            cached_sensor_epoch: 0,
        })
    }

    /// Dispatch the `populate_inputs` shader. Inputs:
    /// - `cells`: holds energy / heading / pitch / damage / max_speed /
    ///   eff_radius / last_hidden / last_inputs / velocities buffers.
    /// - `sensor`: provides the per-cell sensor output and vision radii.
    /// Output is written into `cells.last_inputs_buf`, which the brain
    /// forward dispatch then reads directly.
    pub fn dispatch(
        &mut self,
        cells: &CellsGpu,
        sensor: &SensorGatherGpu,
        params: PopulateInputsParams,
    ) {
        if params.num_cells == 0 {
            return;
        }
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("populate-inputs-encoder"),
        });
        self.dispatch_into(&mut encoder, cells, sensor, params);
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn dispatch_into(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        cells: &CellsGpu,
        sensor: &SensorGatherGpu,
        params: PopulateInputsParams,
    ) {
        if params.num_cells == 0 {
            return;
        }
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        let cells_epoch = cells.epoch();
        let sensor_epoch = sensor.epoch();
        if self.cached_bg.is_none()
            || self.cached_cells_epoch != cells_epoch
            || self.cached_sensor_epoch != sensor_epoch
        {
            self.cached_bg = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("populate-inputs-bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: sensor.output_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: cells.velocities_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: cells.energy_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: cells.heading_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: cells.pitch_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: cells.damage_accum_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: cells.max_speed_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: cells.eff_radius_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: sensor.vision_radii_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 10, resource: cells.last_hidden_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 11, resource: cells.last_inputs_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 12, resource: cells.bonded_inbox_buffer().as_entire_binding() },
                ],
            }));
            self.cached_cells_epoch = cells_epoch;
            self.cached_sensor_epoch = sensor_epoch;
        }
        let bind_group = self.cached_bg.as_ref().unwrap();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("populate-inputs-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        let workgroups = (params.num_cells + 63) / 64;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
}

