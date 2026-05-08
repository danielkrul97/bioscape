use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::*;
use super::*;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct Params {
    num_cells: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct BrainGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    capacity: usize,
    params_buf: wgpu::Buffer,
    inputs_buf: wgpu::Buffer,
    weights_buf: wgpu::Buffer,
    hidden_buf: wgpu::Buffer,
    outputs_buf: wgpu::Buffer,
    hidden_readback: wgpu::Buffer,
    outputs_readback: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    // Persistent CPU staging — reused per forward_batch, ne realokuje.
    inputs_packed: Vec<f32>,
    weights_packed: Vec<f32>,
    cached_persistent_bg: Option<wgpu::BindGroup>,
    cached_persistent_epoch: u64,
}

impl BrainGpu {
    pub fn with_context(ctx: &GpuContext, capacity: usize) -> Result<Self, String> {
        Self::with_device_inner(Arc::clone(&ctx.device), Arc::clone(&ctx.queue), capacity)
    }

    pub fn new(capacity: usize) -> Result<Self, String> {
        assert!(capacity > 0, "BrainGpu capacity must be > 0");
        let ctx = GpuContext::new()?;
        Self::with_device_inner(ctx.device, ctx.queue, capacity)
    }

    fn with_device_inner(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        capacity: usize,
    ) -> Result<Self, String> {
        assert!(capacity > 0, "BrainGpu capacity must be > 0");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brain_forward"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/brain_forward.wgsl").into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brain-bgl"),
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
            label: Some("brain-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("brain-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("forward"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brain-params"),
            contents: bytemuck::bytes_of(&Params::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let (
            inputs_buf,
            weights_buf,
            hidden_buf,
            outputs_buf,
            hidden_readback,
            outputs_readback,
        ) = Self::alloc_buffers(&device, capacity);

        let bind_group = Self::make_bind_group(
            &device,
            &bind_group_layout,
            &params_buf,
            &inputs_buf,
            &weights_buf,
            &hidden_buf,
            &outputs_buf,
        );

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            capacity,
            params_buf,
            inputs_buf,
            weights_buf,
            hidden_buf,
            outputs_buf,
            hidden_readback,
            outputs_readback,
            bind_group,
            inputs_packed: Vec::new(),
            weights_packed: Vec::new(),
            cached_persistent_bg: None,
            cached_persistent_epoch: 0,
        })
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
        let inputs_size = (capacity * BRAIN_INPUTS * std::mem::size_of::<f32>()) as u64;
        let weights_size = (capacity * BRAIN_WEIGHTS_PER_CELL * std::mem::size_of::<f32>()) as u64;
        let hidden_size = (capacity * BRAIN_HIDDEN * std::mem::size_of::<f32>()) as u64;
        let outputs_size = (capacity * BRAIN_OUTPUTS * std::mem::size_of::<f32>()) as u64;
        let inputs_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-inputs"),
            size: inputs_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let weights_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-weights"),
            size: weights_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hidden_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-hidden"),
            size: hidden_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let outputs_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-outputs"),
            size: outputs_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let hidden_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-hidden-readback"),
            size: hidden_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let outputs_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-outputs-readback"),
            size: outputs_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (
            inputs_buf,
            weights_buf,
            hidden_buf,
            outputs_buf,
            hidden_readback,
            outputs_readback,
        )
    }

    fn make_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        params: &wgpu::Buffer,
        inputs: &wgpu::Buffer,
        weights: &wgpu::Buffer,
        hidden: &wgpu::Buffer,
        outputs: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brain-bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: inputs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: hidden.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: outputs.as_entire_binding(),
                },
            ],
        })
    }

    /// Roste storage buffery, pokud `n` přesahuje aktuální `capacity`. Volá se
    /// na začátku `forward_batch` před uploadem.
    fn ensure_capacity(&mut self, n: usize) {
        if n <= self.capacity {
            return;
        }
        let new_cap = (self.capacity * 2).max(n);
        let (i, w, h, o, hr, or_) = Self::alloc_buffers(&self.device, new_cap);
        self.inputs_buf = i;
        self.weights_buf = w;
        self.hidden_buf = h;
        self.outputs_buf = o;
        self.hidden_readback = hr;
        self.outputs_readback = or_;
        self.bind_group = Self::make_bind_group(
            &self.device,
            &self.bind_group_layout,
            &self.params_buf,
            &self.inputs_buf,
            &self.weights_buf,
            &self.hidden_buf,
            &self.outputs_buf,
        );
        self.capacity = new_cap;
    }

    /// Spočítá forward pass všem `inputs.len()` cells. Délky všech vstupů musí
    /// být shodné. `brains` je iterátor (typicky `cells.iter().map(|c| &c.genome.brain)`)
    /// aby se vyhnula intermediate `Vec<Brain>` (~3 KB/cell × 10 k = 30 MB).
    /// Výsledky se zapíšou do `hiddens_out` + `outputs_out`.
    pub fn forward_batch<'a, I>(
        &mut self,
        inputs: &[[f32; BRAIN_INPUTS]],
        brains: I,
        hiddens_out: &mut [[f32; BRAIN_HIDDEN]],
        outputs_out: &mut [[f32; BRAIN_OUTPUTS]],
    ) where
        I: IntoIterator<Item = &'a Brain>,
    {
        let n = inputs.len();
        assert_eq!(hiddens_out.len(), n);
        assert_eq!(outputs_out.len(), n);
        if n == 0 {
            return;
        }
        self.ensure_capacity(n);

        // Pack inputs (AoS f32 stream).
        self.inputs_packed.clear();
        self.inputs_packed.reserve(n * BRAIN_INPUTS);
        for inp in inputs {
            self.inputs_packed.extend_from_slice(inp);
        }

        // Pack weights per-cell. Layout musí matchnout WGSL shader.
        self.weights_packed.clear();
        self.weights_packed.reserve(n * BRAIN_WEIGHTS_PER_CELL);
        let mut brains_seen = 0usize;
        for brain in brains.into_iter().take(n) {
            for row in brain.w1.iter() {
                self.weights_packed.extend_from_slice(row);
            }
            self.weights_packed.extend_from_slice(&brain.b1);
            for row in brain.w2.iter() {
                self.weights_packed.extend_from_slice(row);
            }
            self.weights_packed.extend_from_slice(&brain.b2);
            brains_seen += 1;
        }
        assert_eq!(brains_seen, n, "brains iterator length mismatch");
        debug_assert_eq!(self.inputs_packed.len(), n * BRAIN_INPUTS);
        debug_assert_eq!(self.weights_packed.len(), n * BRAIN_WEIGHTS_PER_CELL);

        let params = Params {
            num_cells: n as u32,
            ..Params::default()
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(
            &self.inputs_buf,
            0,
            bytemuck::cast_slice(&self.inputs_packed),
        );
        self.queue.write_buffer(
            &self.weights_buf,
            0,
            bytemuck::cast_slice(&self.weights_packed),
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("brain-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("brain-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let workgroups = (n as u32 + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        let hidden_bytes = (n * BRAIN_HIDDEN * std::mem::size_of::<f32>()) as u64;
        let outputs_bytes = (n * BRAIN_OUTPUTS * std::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(&self.hidden_buf, 0, &self.hidden_readback, 0, hidden_bytes);
        encoder.copy_buffer_to_buffer(
            &self.outputs_buf,
            0,
            &self.outputs_readback,
            0,
            outputs_bytes,
        );
        self.queue.submit(Some(encoder.finish()));

        let hidden_slice = self.hidden_readback.slice(0..hidden_bytes);
        let outputs_slice = self.outputs_readback.slice(0..outputs_bytes);
        hidden_slice.map_async(wgpu::MapMode::Read, |_| {});
        outputs_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        {
            let h_data = hidden_slice.get_mapped_range();
            let h_slice: &[f32] = bytemuck::cast_slice(&h_data);
            for (i, dst) in hiddens_out.iter_mut().enumerate() {
                let off = i * BRAIN_HIDDEN;
                dst.copy_from_slice(&h_slice[off..off + BRAIN_HIDDEN]);
            }
        }
        {
            let o_data = outputs_slice.get_mapped_range();
            let o_slice: &[f32] = bytemuck::cast_slice(&o_data);
            for (i, dst) in outputs_out.iter_mut().enumerate() {
                let off = i * BRAIN_OUTPUTS;
                dst.copy_from_slice(&o_slice[off..off + BRAIN_OUTPUTS]);
            }
        }
        self.hidden_readback.unmap();
        self.outputs_readback.unmap();
    }

    /// Sprint 51: persistent-mode dispatch — bindujeme `CellsGpu` buffers
    /// (last_inputs jako vstup, brain_weights jako persistent storage,
    /// last_hidden + last_outputs jako write-back). **NULL upload weights**
    /// — to je hlavní win sprintu.
    pub fn forward_persistent(&mut self, cells_gpu: &CellsGpu, n: usize) {
        if n == 0 {
            return;
        }
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("brain-encoder-persistent"),
        });
        self.forward_persistent_into(&mut encoder, cells_gpu, n);
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn forward_persistent_into(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        cells_gpu: &CellsGpu,
        n: usize,
    ) {
        if n == 0 {
            return;
        }
        let params = Params {
            num_cells: n as u32,
            ..Params::default()
        };
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        let cells_epoch = cells_gpu.epoch();
        if self.cached_persistent_bg.is_none() || self.cached_persistent_epoch != cells_epoch {
            self.cached_persistent_bg = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("brain-bg-persistent"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: cells_gpu.last_inputs_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: cells_gpu.brain_weights_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: cells_gpu.last_hidden_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: cells_gpu.last_outputs_buffer().as_entire_binding() },
                ],
            }));
            self.cached_persistent_epoch = cells_epoch;
        }
        let bind_group = self.cached_persistent_bg.as_ref().unwrap();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("brain-pass-persistent"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        let workgroups = (n as u32 + 63) / 64;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
}
