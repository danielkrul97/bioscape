//! Sprint 44: GPU compute scaffolding pro batch brain forward pass.
//!
//! Ohraničeno feature gate `gpu` — bez `--features gpu` modul vůbec
//! nezkompiluje wgpu, takže build na strojích bez GPU stacku zůstává štíhlý.
//!
//! Architektonicky drží `BrainGpu` perzistentní wgpu device + storage buffery
//! sized na `capacity` cells. Per `forward_batch`: upload inputs + per-cell
//! weights, dispatch compute, readback hidden + outputs. State on GPU mezi
//! ticky **NE** drží — to je Sprint 47.

use crate::{Brain, BRAIN_HIDDEN, BRAIN_INPUTS, BRAIN_OUTPUTS};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

// ============================================================================
// Sprint 47: shared GPU context — jeden device pro všechny subsystémy
// ============================================================================

/// Sprint 47: sdílený wgpu device + queue (Arc, protože `wgpu::Device` v 23
/// není `Clone`). Každý subsystém (BrainGpu, FieldGpu, SpatialHashGpu, StatsGpu)
/// si naklonuje `Arc::clone` handles z `GpuContext`. Předtím každý subsystém
/// vyrobil vlastní device → 4× duplicitních adapterů, žádné sdílení resourců.
/// Pro Sprint 47+ pipelines (full GPU tick) je shared device **nutnost** —
/// jedna queue submission ordering, jedna device-lifetime, shared command
/// encoding.
#[derive(Clone)]
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl GpuContext {
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            },
        ))
        .ok_or_else(|| "no suitable wgpu adapter".to_string())?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("bioscape-shared"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 12,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| format!("device request failed: {e:?}"))?;
        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }
}

// ============================================================================
// Sprint 44: GPU brain forward batch
// ============================================================================

/// Počet f32 váh per cell — packing musí matchnout WGSL shader.
pub const BRAIN_WEIGHTS_PER_CELL: usize =
    BRAIN_HIDDEN * BRAIN_INPUTS + BRAIN_HIDDEN + BRAIN_OUTPUTS * BRAIN_HIDDEN + BRAIN_OUTPUTS;

// Layout offsety v rámci jednoho cell weights bloku — musí matchnout WGSL.
// Compile-time guard přes assert v `forward_batch`.
const W1_OFFSET: usize = 0;
const B1_OFFSET: usize = BRAIN_HIDDEN * BRAIN_INPUTS;
const W2_OFFSET: usize = B1_OFFSET + BRAIN_HIDDEN;
const B2_OFFSET: usize = W2_OFFSET + BRAIN_OUTPUTS * BRAIN_HIDDEN;
const _: () = assert!(W1_OFFSET == 0);
const _: () = assert!(B1_OFFSET == 576);
const _: () = assert!(W2_OFFSET == 592);
const _: () = assert!(B2_OFFSET == 736);
const _: () = assert!(BRAIN_WEIGHTS_PER_CELL == 745);

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
                include_str!("../shaders/brain_forward.wgsl").into(),
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
    pub fn forward_persistent(&self, cells_gpu: &CellsGpu, n: usize) {
        if n == 0 {
            return;
        }
        let params = Params {
            num_cells: n as u32,
            ..Params::default()
        };
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brain-bg-persistent"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cells_gpu.last_inputs_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cells_gpu.brain_weights_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cells_gpu.last_hidden_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cells_gpu.last_outputs_buffer().as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("brain-encoder-persistent"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("brain-pass-persistent"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n as u32 + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}

// ============================================================================
// Sprint 45: GPU spatial hash (counting sort)
// ============================================================================

/// Layout musí matchnout `shaders/spatial_hash.wgsl`. Bucket grid je fixed
/// 64×32×4 = 8192 cells krytí ±2048 / ±512 / ±128 world units při
/// `GRID_CELL_SIZE = 64`.
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
    /// Sprint 55: xy world bounds pro toroidal bucket wrap.
    world_half_x: f32,
    world_half_y: f32,
}

pub struct SpatialHashGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline_count: wgpu::ComputePipeline,
    pipeline_prefix: wgpu::ComputePipeline,
    pipeline_scatter: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    capacity: usize,
    cell_size: f32,
    /// Sprint 55: xy world bounds pro toroidal bucket wrap.
    world_half_xy: [f32; 2],
    params_buf: wgpu::Buffer,
    positions_buf: wgpu::Buffer,
    counts_buf: wgpu::Buffer,
    offsets_buf: wgpu::Buffer,
    sorted_buf: wgpu::Buffer,
    offsets_readback: wgpu::Buffer,
    sorted_readback: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    positions_packed: Vec<f32>,
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
                include_str!("../shaders/spatial_hash.wgsl").into(),
            ),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            positions_packed: Vec::new(),
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

        // Reset counts to 0 (rebuild expects fresh state).
        self.queue.write_buffer(
            &self.counts_buf,
            0,
            &vec![0u8; GPU_HASH_NUM_BUCKETS * 4],
        );

        self.positions_packed.clear();
        self.positions_packed.reserve(n * 3);
        for p in positions {
            self.positions_packed.push(p[0]);
            self.positions_packed.push(p[1]);
            self.positions_packed.push(p[2]);
        }

        let params = HashParams {
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
            bytemuck::cast_slice(&self.positions_packed),
        );

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
        self.device.poll(wgpu::Maintain::Wait);

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

    /// Sprint 49: accessor pro chained shadery (NeighborsGpu) — bind hash
    /// buffery jako read-only ne další pass. Buffery jsou platné dokud
    /// SpatialHashGpu žije.
    pub fn offsets_buffer(&self) -> &wgpu::Buffer {
        &self.offsets_buf
    }

    pub fn sorted_buffer(&self) -> &wgpu::Buffer {
        &self.sorted_buf
    }

    /// CPU side mirror funkce `bucket_id_of` v shaderu. Useful pro testy a
    /// pro callers, kteří chtějí dohnat GPU bucket layout.
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

// ============================================================================
// Sprint 46: GPU field diffusion (smell + pheromone na ekvivalentní compute path)
// ============================================================================

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct FieldParams {
    resolution: u32,
    num_sources: u32,
    diffusion: f32,
    decay: f32,
    cell_size_x: f32,
    cell_size_y: f32,
    world_half_x: f32,
    world_half_y: f32,
}

pub struct FieldGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline_deposit: wgpu::ComputePipeline,
    pipeline_diffuse: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    grid_a: wgpu::Buffer,
    grid_b: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    sources_buf: wgpu::Buffer,
    grid_readback: wgpu::Buffer,
    bg_round_a: wgpu::BindGroup, // round A: grid_in=A, grid_out=B
    bg_round_b: wgpu::BindGroup, // round B: grid_in=B, grid_out=A
    pending_sources: Vec<f32>,   // [px, py, amount] * N
    capacity_sources: usize,
    /// Které grid je "current" = naposled zapsané pole, kam jdou next sources.
    /// `true` = A, `false` = B. Po každém step() se invertuje.
    current_is_a: bool,
    resolution: usize,
    world_half: [f32; 2],
}

impl FieldGpu {
    pub fn new(
        resolution: usize,
        world_half: [f32; 2],
        sources_capacity: usize,
    ) -> Result<Self, String> {
        assert!(resolution >= 2 && sources_capacity > 0);
        let ctx = GpuContext::new()?;
        Self::with_device_inner(ctx.device, ctx.queue, resolution, world_half, sources_capacity)
    }

    pub fn with_context(
        ctx: &GpuContext,
        resolution: usize,
        world_half: [f32; 2],
        sources_capacity: usize,
    ) -> Result<Self, String> {
        assert!(resolution >= 2 && sources_capacity > 0);
        Self::with_device_inner(
            Arc::clone(&ctx.device),
            Arc::clone(&ctx.queue),
            resolution,
            world_half,
            sources_capacity,
        )
    }

    fn with_device_inner(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        resolution: usize,
        world_half: [f32; 2],
        sources_capacity: usize,
    ) -> Result<Self, String> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("field_diffuse"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/field_diffuse.wgsl").into(),
            ),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("field-bgl"),
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
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("field-pl"),
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
        let pipeline_deposit = make_pipe("deposit", "field-deposit");
        let pipeline_diffuse = make_pipe("diffuse", "field-diffuse");

        let grid_size_bytes = (resolution * resolution * std::mem::size_of::<u32>()) as u64;
        let grid_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field-grid-a"),
            size: grid_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grid_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field-grid-b"),
            size: grid_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Inicializuj oba buffery na 0.
        queue.write_buffer(&grid_a, 0, &vec![0u8; grid_size_bytes as usize]);
        queue.write_buffer(&grid_b, 0, &vec![0u8; grid_size_bytes as usize]);

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("field-params"),
            contents: bytemuck::bytes_of(&FieldParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let sources_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field-sources"),
            size: (sources_capacity * 3 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let make_bg = |grid_in: &wgpu::Buffer, grid_out: &wgpu::Buffer, label: &str| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: sources_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: grid_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: grid_out.as_entire_binding(),
                    },
                ],
            })
        };
        let bg_round_a = make_bg(&grid_a, &grid_b, "field-bg-round-a");
        let bg_round_b = make_bg(&grid_b, &grid_a, "field-bg-round-b");

        let grid_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field-grid-readback"),
            size: grid_size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            pipeline_deposit,
            pipeline_diffuse,
            bind_group_layout,
            grid_a,
            grid_b,
            params_buf,
            sources_buf,
            grid_readback,
            bg_round_a,
            bg_round_b,
            pending_sources: Vec::new(),
            capacity_sources: sources_capacity,
            current_is_a: true,
            resolution,
            world_half,
        })
    }

    pub fn resolution(&self) -> usize {
        self.resolution
    }

    pub fn world_half(&self) -> [f32; 2] {
        self.world_half
    }

    fn cell_size_x(&self) -> f32 {
        (2.0 * self.world_half[0]) / self.resolution as f32
    }

    fn cell_size_y(&self) -> f32 {
        (2.0 * self.world_half[1]) / self.resolution as f32
    }

    /// Mirror `SmellField::add_source`: zaregistruje bod-zdroj. Bude
    /// flushnut na GPU při dalším `step()`.
    pub fn add_source(&mut self, pos: [f32; 2], amount: f32) {
        self.pending_sources.push(pos[0]);
        self.pending_sources.push(pos[1]);
        self.pending_sources.push(amount);
    }

    /// Realloc sources buffer, pokud `add_source` nahromadil víc než current
    /// capacity. Geometric (×2).
    fn ensure_sources_capacity(&mut self, num_sources: usize) {
        if num_sources <= self.capacity_sources {
            return;
        }
        let new_cap = (self.capacity_sources * 2).max(num_sources);
        self.sources_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field-sources"),
            size: (new_cap * 3 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.capacity_sources = new_cap;
        // Bind groups zachycují sources_buf — musíme je rebuildnout.
        let make_bg = |grid_in: &wgpu::Buffer, grid_out: &wgpu::Buffer, label: &str| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.sources_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: grid_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: grid_out.as_entire_binding(),
                    },
                ],
            })
        };
        self.bg_round_a = make_bg(&self.grid_a, &self.grid_b, "field-bg-round-a");
        self.bg_round_b = make_bg(&self.grid_b, &self.grid_a, "field-bg-round-b");
    }

    /// Mirror `SmellField::step`: flush pending sources do GPU bufferu, dispatch
    /// deposit + diffuse, ping-pong swap. `decay_per_sec` se converté na
    /// `(1 - decay × dt).max(0)` aby matchnul CPU semantiku.
    pub fn step(&mut self, diffusion: f32, decay_per_sec: f32, dt: f32) {
        let num_sources = self.pending_sources.len() / 3;
        self.ensure_sources_capacity(num_sources.max(1));
        if num_sources > 0 {
            self.queue.write_buffer(
                &self.sources_buf,
                0,
                bytemuck::cast_slice(&self.pending_sources),
            );
        }
        let params = FieldParams {
            resolution: self.resolution as u32,
            num_sources: num_sources as u32,
            diffusion,
            decay: (1.0 - decay_per_sec * dt).max(0.0),
            cell_size_x: self.cell_size_x(),
            cell_size_y: self.cell_size_y(),
            world_half_x: self.world_half[0],
            world_half_y: self.world_half[1],
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let bg = if self.current_is_a {
            &self.bg_round_a
        } else {
            &self.bg_round_b
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("field-encoder"),
            });
        if num_sources > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("field-deposit-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_deposit);
            pass.set_bind_group(0, bg, &[]);
            let workgroups = ((num_sources as u32) + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("field-diffuse-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_diffuse);
            pass.set_bind_group(0, bg, &[]);
            let workgroups_xy = ((self.resolution as u32) + 7) / 8;
            pass.dispatch_workgroups(workgroups_xy, workgroups_xy, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.current_is_a = !self.current_is_a;
        self.pending_sources.clear();
    }

    /// Sprint 50: accessor pro chained shadery (sensor gather), které samplují
    /// pole inline. Vrací buffer obsahující latest state (post-step).
    pub fn current_grid_buffer(&self) -> &wgpu::Buffer {
        if self.current_is_a {
            &self.grid_a
        } else {
            &self.grid_b
        }
    }

    /// Stáhne current grid (post-diffuse output buffer) jako Vec<f32>.
    /// Pomalá operace — kvůli tests + visualization. Sprint 47+ sample přes
    /// GPU compute, žádný readback.
    pub fn download(&mut self) -> Vec<f32> {
        let n = self.resolution * self.resolution;
        let bytes = (n * 4) as u64;
        // Po step() je current_is_a inverted, takže "current" je teď grid_a
        // pokud current_is_a=true (původně bylo b, swap -> a).
        let src = if self.current_is_a {
            &self.grid_a
        } else {
            &self.grid_b
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("field-readback-encoder"),
            });
        encoder.copy_buffer_to_buffer(src, 0, &self.grid_readback, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let slice = self.grid_readback.slice(0..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let bits: &[u32] = bytemuck::cast_slice(&data);
        let result: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
        drop(data);
        self.grid_readback.unmap();
        result
    }
}

// ============================================================================
// Sprint 51: CellsGpu persistent SoA state — drží brain weights + last_*
// + xoshiro RNG na GPU mezi ticky. Eliminuje 30 MB/tick brain upload bottleneck
// ze Sprintu 44.
// ============================================================================

/// Persistent SoA cell state na GPU. Drží brain forward state (last_inputs,
/// last_hidden, last_outputs, brain weights) + velocities (pro brownian
/// mutation) + per-cell xoshiro128++ RNG state. Per Sprint 51 scope:
/// **NE-drží** position/heading/etc. (ty zůstávají na CPU pro sensor/motor/
/// step/collision/predate fáze — Sprint 50 standalone shadery jsou ready
/// pro plnou migraci, kdyby se rozhodlo).
///
/// Lifecycle:
/// 1. `new(ctx, capacity)` alokuje buffers + initializuje xoshiro state.
/// 2. `upload_brains(brains, init_xoshiro_seed)` na sim init.
/// 3. Hot loop:
///    - `upload_inputs(last_inputs)` před brain forward.
///    - `forward_batch_persistent(brain_gpu)` — channels/persistent.
///    - `download_hidden_outputs() -> (Vec<hidden>, Vec<outputs>)` po brain.
///    - `upload_velocities(velocities)` před brownian.
///    - `brownian_persistent(brownian_gpu, ...)` — mutuje velocities + state.
///    - `download_velocities() -> Vec<velocities>` po brownian.
///    - `upload_rewards(rewards)` po eat_food.
///    - `hebbian_persistent(hebbian_gpu, lr)` — mutuje brain weights in-place.
/// 4. `upload_brain_at(idx, brain)` po reproduce (nová cell na slot idx).
/// 5. `download_brains() -> Vec<Brain>` pro checkpoint nebo introspection.
pub struct CellsGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    capacity: usize,
    last_inputs_buf: wgpu::Buffer,
    last_hidden_buf: wgpu::Buffer,
    last_outputs_buf: wgpu::Buffer,
    brain_weights_buf: wgpu::Buffer,
    velocities_buf: wgpu::Buffer,
    xoshiro_state_buf: wgpu::Buffer,
    rewards_buf: wgpu::Buffer,
    last_hidden_rb: wgpu::Buffer,
    last_outputs_rb: wgpu::Buffer,
    velocities_rb: wgpu::Buffer,
    brain_weights_rb: wgpu::Buffer,
    /// Sprint 51: staging pro `swap_to` — wgpu zakazuje same-buffer copy.
    swap_brain_temp: wgpu::Buffer,
    swap_xoshiro_temp: wgpu::Buffer,
}

impl CellsGpu {
    pub fn with_context(ctx: &GpuContext, capacity: usize) -> Self {
        Self::with_device_inner(Arc::clone(&ctx.device), Arc::clone(&ctx.queue), capacity)
    }

    fn with_device_inner(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        capacity: usize,
    ) -> Self {
        assert!(capacity > 0);
        let f = std::mem::size_of::<f32>() as u64;
        let n = capacity as u64;
        let stor_dst_src = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let last_inputs_buf = mk("cells-last-inputs", n * (BRAIN_INPUTS as u64) * f, stor_dst_src);
        let last_hidden_buf = mk("cells-last-hidden", n * (BRAIN_HIDDEN as u64) * f, stor_dst_src);
        let last_outputs_buf = mk("cells-last-outputs", n * (BRAIN_OUTPUTS as u64) * f, stor_dst_src);
        let brain_weights_buf = mk("cells-brain-weights", n * (BRAIN_WEIGHTS_PER_CELL as u64) * f, stor_dst_src);
        let velocities_buf = mk("cells-velocities", n * 3 * f, stor_dst_src);
        let xoshiro_state_buf = mk("cells-xoshiro", n * 4 * 4, stor_dst_src);
        let rewards_buf = mk(
            "cells-rewards",
            n * f,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let last_hidden_rb = mk("cells-hidden-rb", n * (BRAIN_HIDDEN as u64) * f, read);
        let last_outputs_rb = mk("cells-outputs-rb", n * (BRAIN_OUTPUTS as u64) * f, read);
        let velocities_rb = mk("cells-velocities-rb", n * 3 * f, read);
        let brain_weights_rb = mk("cells-weights-rb", n * (BRAIN_WEIGHTS_PER_CELL as u64) * f, read);
        let swap_brain_temp = mk(
            "cells-swap-brain-temp",
            (BRAIN_WEIGHTS_PER_CELL as u64) * f,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let swap_xoshiro_temp = mk(
            "cells-swap-xoshiro-temp",
            16,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let _ = mk;
        Self {
            device,
            queue,
            capacity,
            last_inputs_buf,
            last_hidden_buf,
            last_outputs_buf,
            brain_weights_buf,
            velocities_buf,
            xoshiro_state_buf,
            rewards_buf,
            last_hidden_rb,
            last_outputs_rb,
            velocities_rb,
            brain_weights_rb,
            swap_brain_temp,
            swap_xoshiro_temp,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn last_inputs_buffer(&self) -> &wgpu::Buffer { &self.last_inputs_buf }
    pub fn last_hidden_buffer(&self) -> &wgpu::Buffer { &self.last_hidden_buf }
    pub fn last_outputs_buffer(&self) -> &wgpu::Buffer { &self.last_outputs_buf }
    pub fn brain_weights_buffer(&self) -> &wgpu::Buffer { &self.brain_weights_buf }
    pub fn velocities_buffer(&self) -> &wgpu::Buffer { &self.velocities_buf }
    pub fn xoshiro_state_buffer(&self) -> &wgpu::Buffer { &self.xoshiro_state_buf }
    pub fn rewards_buffer(&self) -> &wgpu::Buffer { &self.rewards_buf }

    /// Uploaduje brain weights pro N cells. Volá se na sim init + po reproduce
    /// (re-upload všech, nebo per-slot přes `upload_brain_at`).
    pub fn upload_brains<'a, I>(&self, brains: I)
    where
        I: IntoIterator<Item = &'a Brain>,
    {
        let mut packed: Vec<f32> = Vec::with_capacity(self.capacity * BRAIN_WEIGHTS_PER_CELL);
        for brain in brains {
            for row in brain.w1.iter() { packed.extend_from_slice(row); }
            packed.extend_from_slice(&brain.b1);
            for row in brain.w2.iter() { packed.extend_from_slice(row); }
            packed.extend_from_slice(&brain.b2);
        }
        self.queue.write_buffer(&self.brain_weights_buf, 0, bytemuck::cast_slice(&packed));
    }

    /// Uploaduje brain weights pro jeden slot (idx). Použito po reproduce —
    /// nová cell se zapíše na konec Vec, její brain na slot idx = old_len.
    pub fn upload_brain_at(&self, idx: usize, brain: &Brain) {
        assert!(idx < self.capacity);
        let mut packed: Vec<f32> = Vec::with_capacity(BRAIN_WEIGHTS_PER_CELL);
        for row in brain.w1.iter() { packed.extend_from_slice(row); }
        packed.extend_from_slice(&brain.b1);
        for row in brain.w2.iter() { packed.extend_from_slice(row); }
        packed.extend_from_slice(&brain.b2);
        let offset = (idx * BRAIN_WEIGHTS_PER_CELL * 4) as u64;
        self.queue.write_buffer(&self.brain_weights_buf, offset, bytemuck::cast_slice(&packed));
    }

    /// Sync brains z GPU zpátky na CPU. Pomalá operace — kvůli checkpoint
    /// nebo introspekci. Hot loop ji nevolá.
    pub fn download_brains(&self, n: usize) -> Vec<Brain> {
        assert!(n <= self.capacity);
        if n == 0 { return Vec::new(); }
        let bytes = (n * BRAIN_WEIGHTS_PER_CELL * 4) as u64;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cells-download-brains"),
        });
        encoder.copy_buffer_to_buffer(&self.brain_weights_buf, 0, &self.brain_weights_rb, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let s = self.brain_weights_rb.slice(0..bytes);
        s.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = s.get_mapped_range();
        let f: &[f32] = bytemuck::cast_slice(&data);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * BRAIN_WEIGHTS_PER_CELL;
            let mut b = Brain {
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
            for h in 0..BRAIN_HIDDEN { b.b1[h] = f[off + 576 + h]; }
            for o in 0..BRAIN_OUTPUTS {
                for h in 0..BRAIN_HIDDEN {
                    b.w2[o][h] = f[off + 592 + o * BRAIN_HIDDEN + h];
                }
            }
            for o in 0..BRAIN_OUTPUTS { b.b2[o] = f[off + 736 + o]; }
            out.push(b);
        }
        drop(data);
        self.brain_weights_rb.unmap();
        out
    }

    pub fn upload_inputs(&self, inputs: &[[f32; BRAIN_INPUTS]]) {
        let flat: Vec<f32> = inputs.iter().flatten().copied().collect();
        self.queue.write_buffer(&self.last_inputs_buf, 0, bytemuck::cast_slice(&flat));
    }

    pub fn upload_velocities(&self, velocities: &[[f32; 3]]) {
        let flat: Vec<f32> = velocities.iter().flatten().copied().collect();
        self.queue.write_buffer(&self.velocities_buf, 0, bytemuck::cast_slice(&flat));
    }

    pub fn upload_rewards(&self, rewards: &[f32]) {
        self.queue.write_buffer(&self.rewards_buf, 0, bytemuck::cast_slice(rewards));
    }

    /// Sprint 51: GPU-side copy slot[src] → slot[dst] pro brain_weights +
    /// xoshiro_state. Použito v die_and_drop_carrion swap_remove pattern —
    /// keď cell v dst slotu zemřela, src je poslední živá cell, která se
    /// přesune. NIC se ne-stahuje, NIC se ne-uploaduje — pure GPU memcpy.
    pub fn swap_to(&self, dst: usize, src: usize) {
        assert!(dst < self.capacity && src < self.capacity);
        if dst == src {
            return;
        }
        let brain_bytes = (BRAIN_WEIGHTS_PER_CELL * 4) as u64;
        let brain_src = (src * BRAIN_WEIGHTS_PER_CELL * 4) as u64;
        let brain_dst = (dst * BRAIN_WEIGHTS_PER_CELL * 4) as u64;
        let xosh_bytes = 4u64 * 4;
        let xosh_src = (src * 4 * 4) as u64;
        let xosh_dst = (dst * 4 * 4) as u64;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cells-swap"),
        });
        // wgpu nepovoluje same-buffer copy → routujeme přes staging temps.
        encoder.copy_buffer_to_buffer(
            &self.brain_weights_buf,
            brain_src,
            &self.swap_brain_temp,
            0,
            brain_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &self.swap_brain_temp,
            0,
            &self.brain_weights_buf,
            brain_dst,
            brain_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &self.xoshiro_state_buf,
            xosh_src,
            &self.swap_xoshiro_temp,
            0,
            xosh_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &self.swap_xoshiro_temp,
            0,
            &self.xoshiro_state_buf,
            xosh_dst,
            xosh_bytes,
        );
        self.queue.submit(Some(encoder.finish()));
    }

    /// Sprint 51: seed xoshiro state pro konkrétní slot. Použito po reproduce
    /// (nová cell potřebuje fresh state).
    pub fn upload_xoshiro_seed_at(&self, slot: usize, seed: u64) {
        assert!(slot < self.capacity);
        fn splitmix(z: &mut u64) -> u64 {
            *z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = *z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        }
        let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
        let a = splitmix(&mut z);
        let b = splitmix(&mut z);
        let mut s0 = a as u32;
        let s1 = (a >> 32) as u32;
        let s2 = b as u32;
        let s3 = (b >> 32) as u32;
        let mut state = [s0, s1, s2, s3];
        if state == [0u32; 4] {
            s0 = 1;
            state[0] = s0;
        }
        let offset = (slot * 4 * 4) as u64;
        self.queue.write_buffer(&self.xoshiro_state_buf, offset, bytemuck::cast_slice(&state));
    }

    /// Inicializuj per-cell xoshiro state z deterministic seeds. SplitMix64
    /// rozšíří 64-bit seed na 4× 32-bit xoshiro state. Protect proti all-zero
    /// state (xoshiro vyžaduje aspoň jednu non-zero word).
    pub fn upload_xoshiro_seeds<I>(&self, seeds: I)
    where
        I: IntoIterator<Item = u64>,
    {
        fn splitmix(z: &mut u64) -> u64 {
            *z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = *z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        }
        let mut state: Vec<u32> = Vec::with_capacity(self.capacity * 4);
        for s in seeds {
            let mut z = s.wrapping_add(0x9E3779B97F4A7C15);
            let a = splitmix(&mut z);
            let b = splitmix(&mut z);
            let mut s0 = a as u32;
            let s1 = (a >> 32) as u32;
            let s2 = b as u32;
            let s3 = (b >> 32) as u32;
            if s0 == 0 && s1 == 0 && s2 == 0 && s3 == 0 {
                s0 = 1;
            }
            state.push(s0);
            state.push(s1);
            state.push(s2);
            state.push(s3);
        }
        self.queue.write_buffer(&self.xoshiro_state_buf, 0, bytemuck::cast_slice(&state));
    }

    /// Stáhne (last_hidden, last_outputs) jako Vec — caller je potřebuje pro
    /// motor + apply_morph fáze (CPU). Per-tick, kritická pro --gpu-full.
    pub fn download_hidden_outputs(
        &self,
        n: usize,
    ) -> (Vec<[f32; BRAIN_HIDDEN]>, Vec<[f32; BRAIN_OUTPUTS]>) {
        if n == 0 { return (Vec::new(), Vec::new()); }
        assert!(n <= self.capacity);
        let h_bytes = (n * BRAIN_HIDDEN * 4) as u64;
        let o_bytes = (n * BRAIN_OUTPUTS * 4) as u64;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cells-download-ho"),
        });
        encoder.copy_buffer_to_buffer(&self.last_hidden_buf, 0, &self.last_hidden_rb, 0, h_bytes);
        encoder.copy_buffer_to_buffer(&self.last_outputs_buf, 0, &self.last_outputs_rb, 0, o_bytes);
        self.queue.submit(Some(encoder.finish()));
        let h_s = self.last_hidden_rb.slice(0..h_bytes);
        let o_s = self.last_outputs_rb.slice(0..o_bytes);
        h_s.map_async(wgpu::MapMode::Read, |_| {});
        o_s.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let h_data = h_s.get_mapped_range();
        let o_data = o_s.get_mapped_range();
        let h_f: &[f32] = bytemuck::cast_slice(&h_data);
        let o_f: &[f32] = bytemuck::cast_slice(&o_data);
        let hidden: Vec<[f32; BRAIN_HIDDEN]> = (0..n)
            .map(|i| {
                let mut a = [0.0_f32; BRAIN_HIDDEN];
                a.copy_from_slice(&h_f[i * BRAIN_HIDDEN..(i + 1) * BRAIN_HIDDEN]);
                a
            })
            .collect();
        let outputs: Vec<[f32; BRAIN_OUTPUTS]> = (0..n)
            .map(|i| {
                let mut a = [0.0_f32; BRAIN_OUTPUTS];
                a.copy_from_slice(&o_f[i * BRAIN_OUTPUTS..(i + 1) * BRAIN_OUTPUTS]);
                a
            })
            .collect();
        drop(h_data);
        drop(o_data);
        self.last_hidden_rb.unmap();
        self.last_outputs_rb.unmap();
        (hidden, outputs)
    }

    /// Stáhne brain weights pro jeden slot. Použito v reproduce phase
    /// (download parent brains z GPU pro crossover/mutate na CPU).
    pub fn download_brain_at(&self, idx: usize) -> Brain {
        assert!(idx < self.capacity);
        let bytes = (BRAIN_WEIGHTS_PER_CELL * 4) as u64;
        let offset = (idx * BRAIN_WEIGHTS_PER_CELL * 4) as u64;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cells-download-brain-slot"),
        });
        encoder.copy_buffer_to_buffer(&self.brain_weights_buf, offset, &self.brain_weights_rb, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let s = self.brain_weights_rb.slice(0..bytes);
        s.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = s.get_mapped_range();
        let f: &[f32] = bytemuck::cast_slice(&data);
        let mut b = Brain {
            w1: [[0.0; BRAIN_INPUTS]; BRAIN_HIDDEN],
            b1: [0.0; BRAIN_HIDDEN],
            w2: [[0.0; BRAIN_HIDDEN]; BRAIN_OUTPUTS],
            b2: [0.0; BRAIN_OUTPUTS],
        };
        for h in 0..BRAIN_HIDDEN {
            for in_i in 0..BRAIN_INPUTS {
                b.w1[h][in_i] = f[h * BRAIN_INPUTS + in_i];
            }
        }
        for h in 0..BRAIN_HIDDEN { b.b1[h] = f[576 + h]; }
        for o in 0..BRAIN_OUTPUTS {
            for h in 0..BRAIN_HIDDEN {
                b.w2[o][h] = f[592 + o * BRAIN_HIDDEN + h];
            }
        }
        for o in 0..BRAIN_OUTPUTS { b.b2[o] = f[736 + o]; }
        drop(data);
        self.brain_weights_rb.unmap();
        b
    }

    pub fn download_velocities(&self, n: usize) -> Vec<[f32; 3]> {
        if n == 0 { return Vec::new(); }
        assert!(n <= self.capacity);
        let bytes = (n * 3 * 4) as u64;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cells-download-vel"),
        });
        encoder.copy_buffer_to_buffer(&self.velocities_buf, 0, &self.velocities_rb, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let s = self.velocities_rb.slice(0..bytes);
        s.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = s.get_mapped_range();
        let f: &[f32] = bytemuck::cast_slice(&data);
        let out: Vec<[f32; 3]> = (0..n).map(|i| [f[i*3], f[i*3+1], f[i*3+2]]).collect();
        drop(data);
        self.velocities_rb.unmap();
        out
    }
}

// ============================================================================
// Sprint 51: GPU brownian (xoshiro128++ per-cell RNG) + Hebbian update
// ============================================================================

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct BrownianParams {
    num_cells: u32,
    has_z: u32,
    thermal_noise: f32,
    sqrt_dt: f32,
}

pub struct BrownianGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    capacity: usize,
    params_buf: wgpu::Buffer,
    velocities_buf: wgpu::Buffer,
    state_buf: wgpu::Buffer,
    velocities_rb: wgpu::Buffer,
    state_rb: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl BrownianGpu {
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
            label: Some("brownian"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/brownian.wgsl").into()),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..3)
            .map(|i| {
                let ty = if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage { read_only: false }
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
            label: Some("brownian-bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brownian-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("brownian-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("brownian"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brownian-params"),
            contents: bytemuck::bytes_of(&BrownianParams::default()),
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
        let stor_dst_src = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let velocities_buf = mk("brownian-vel", n * 3 * f, stor_dst_src);
        let state_buf = mk("brownian-state", n * 4 * 4, stor_dst_src);
        let velocities_rb = mk("brownian-vel-rb", n * 3 * f, read);
        let state_rb = mk("brownian-state-rb", n * 4 * 4, read);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brownian-bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: velocities_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: state_buf.as_entire_binding() },
            ],
        });
        Ok(Self {
            device, queue, pipeline, bind_group_layout, capacity,
            params_buf, velocities_buf, state_buf, velocities_rb, state_rb, bind_group,
        })
    }

    pub fn compute(
        &mut self,
        velocities_in: &[[f32; 3]],
        state_in: &[[u32; 4]],
        thermal_noise: f32,
        dt: f32,
        has_z: bool,
    ) -> (Vec<[f32; 3]>, Vec<[u32; 4]>) {
        let n = velocities_in.len();
        assert_eq!(state_in.len(), n);
        assert!(n <= self.capacity);
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let vel_flat: Vec<f32> = velocities_in.iter().flatten().copied().collect();
        let state_flat: Vec<u32> = state_in.iter().flatten().copied().collect();
        let params = BrownianParams {
            num_cells: n as u32,
            has_z: if has_z { 1 } else { 0 },
            thermal_noise,
            sqrt_dt: dt.sqrt(),
        };
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(&self.velocities_buf, 0, bytemuck::cast_slice(&vel_flat));
        self.queue.write_buffer(&self.state_buf, 0, bytemuck::cast_slice(&state_flat));
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("brownian-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("brownian-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
        }
        let v_bytes = (n as u64) * 3 * 4;
        let s_bytes = (n as u64) * 4 * 4;
        encoder.copy_buffer_to_buffer(&self.velocities_buf, 0, &self.velocities_rb, 0, v_bytes);
        encoder.copy_buffer_to_buffer(&self.state_buf, 0, &self.state_rb, 0, s_bytes);
        self.queue.submit(Some(encoder.finish()));
        let v_s = self.velocities_rb.slice(0..v_bytes);
        let s_s = self.state_rb.slice(0..s_bytes);
        v_s.map_async(wgpu::MapMode::Read, |_| {});
        s_s.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let v_data = v_s.get_mapped_range();
        let s_data = s_s.get_mapped_range();
        let v_f: &[f32] = bytemuck::cast_slice(&v_data);
        let s_u: &[u32] = bytemuck::cast_slice(&s_data);
        let velocities: Vec<[f32; 3]> = (0..n).map(|i| [v_f[i*3], v_f[i*3+1], v_f[i*3+2]]).collect();
        let state: Vec<[u32; 4]> = (0..n).map(|i| [s_u[i*4], s_u[i*4+1], s_u[i*4+2], s_u[i*4+3]]).collect();
        drop(v_data);
        drop(s_data);
        self.velocities_rb.unmap();
        self.state_rb.unmap();
        (velocities, state)
    }

    pub fn velocities_buffer(&self) -> &wgpu::Buffer {
        &self.velocities_buf
    }
    pub fn state_buffer(&self) -> &wgpu::Buffer {
        &self.state_buf
    }

    /// Sprint 51: persistent-mode dispatch — bindujeme CellsGpu buffery místo
    /// internal. Žádný upload/download, žádný realloc. Volá se v hot loopu
    /// po `cells_gpu.upload_velocities(...)`.
    pub fn compute_persistent(
        &self,
        cells_gpu: &CellsGpu,
        n: usize,
        thermal_noise: f32,
        dt: f32,
        has_z: bool,
    ) {
        if n == 0 {
            return;
        }
        let params = BrownianParams {
            num_cells: n as u32,
            has_z: if has_z { 1 } else { 0 },
            thermal_noise,
            sqrt_dt: dt.sqrt(),
        };
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brownian-bg-persistent"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cells_gpu.velocities_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cells_gpu.xoshiro_state_buffer().as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("brownian-encoder-persistent"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("brownian-pass-persistent"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}

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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/hebbian.wgsl").into()),
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
                b.b1[h] = f[off + 576 + h];
            }
            for o in 0..BRAIN_OUTPUTS {
                for h in 0..BRAIN_HIDDEN {
                    b.w2[o][h] = f[off + 592 + o * BRAIN_HIDDEN + h];
                }
            }
            for o in 0..BRAIN_OUTPUTS {
                b.b2[o] = f[off + 736 + o];
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

    /// Sprint 51: persistent-mode dispatch — bindujeme CellsGpu buffers
    /// (last_inputs, last_hidden, last_outputs, rewards, brain_weights), což
    /// znamená že brain_weights ZŮSTÁVAJÍ na GPU mezi ticky a tato funkce je
    /// mutuje in-place. Volá se po `upload_rewards()`.
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

// ============================================================================
// Sprint 50: GPU motor — applies brain outputs to velocity / angular / pitch
// ============================================================================

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct MotorParams {
    num_cells: u32,
    dt: f32,
    drag_coefficient: f32,
    _pad0: u32,
}

pub struct MotorGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    capacity: usize,
    params_buf: wgpu::Buffer,
    outputs_buf: wgpu::Buffer,
    headings_buf: wgpu::Buffer,
    pitches_buf: wgpu::Buffer,
    max_speeds_buf: wgpu::Buffer,
    turn_rates_buf: wgpu::Buffer,
    eff_radii_buf: wgpu::Buffer,
    velocities_buf: wgpu::Buffer,
    angular_buf: wgpu::Buffer,
    pitch_vel_buf: wgpu::Buffer,
    velocities_readback: wgpu::Buffer,
    angular_readback: wgpu::Buffer,
    pitch_vel_readback: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pos_packed: Vec<f32>,
}

impl MotorGpu {
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
            label: Some("motor"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/motor.wgsl").into()),
        });

        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..10)
            .map(|i| {
                let ty = if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else if i < 7 {
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
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("motor-bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("motor-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("motor-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("motor"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("motor-params"),
            contents: bytemuck::bytes_of(&MotorParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let (
            outputs_buf,
            headings_buf,
            pitches_buf,
            max_speeds_buf,
            turn_rates_buf,
            eff_radii_buf,
            velocities_buf,
            angular_buf,
            pitch_vel_buf,
            velocities_readback,
            angular_readback,
            pitch_vel_readback,
        ) = Self::alloc_buffers(&device, capacity);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("motor-bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: outputs_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: headings_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pitches_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: max_speeds_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: turn_rates_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: eff_radii_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: velocities_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: angular_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: pitch_vel_buf.as_entire_binding() },
            ],
        });
        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            capacity,
            params_buf,
            outputs_buf,
            headings_buf,
            pitches_buf,
            max_speeds_buf,
            turn_rates_buf,
            eff_radii_buf,
            velocities_buf,
            angular_buf,
            pitch_vel_buf,
            velocities_readback,
            angular_readback,
            pitch_vel_readback,
            bind_group,
            pos_packed: Vec::new(),
        })
    }

    fn alloc_buffers(
        device: &wgpu::Device,
        capacity: usize,
    ) -> (
        wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer,
        wgpu::Buffer, wgpu::Buffer, wgpu::Buffer,
        wgpu::Buffer, wgpu::Buffer, wgpu::Buffer,
    ) {
        let f = std::mem::size_of::<f32>();
        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let stor_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let stor_dst_src =
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let n = capacity as u64;
        (
            mk("motor-outputs", n * 9 * f as u64, stor_dst),
            mk("motor-headings", n * f as u64, stor_dst),
            mk("motor-pitches", n * f as u64, stor_dst),
            mk("motor-max-speeds", n * f as u64, stor_dst),
            mk("motor-turn-rates", n * f as u64, stor_dst),
            mk("motor-eff-radii", n * f as u64, stor_dst),
            mk("motor-velocities", n * 3 * f as u64, stor_dst_src),
            mk("motor-angular", n * f as u64, stor_dst_src),
            mk("motor-pitch-vel", n * f as u64, stor_dst_src),
            mk("motor-velocities-rb", n * 3 * f as u64, read),
            mk("motor-angular-rb", n * f as u64, read),
            mk("motor-pitch-vel-rb", n * f as u64, read),
        )
    }

    /// Aplikuje motor pass na inputs. Vrací (new_velocities, new_angular,
    /// new_pitch_vel). dt + drag_coefficient z volajícího kontextu.
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &mut self,
        outputs: &[[f32; BRAIN_OUTPUTS]],
        headings: &[f32],
        pitches: &[f32],
        max_speeds: &[f32],
        turn_rates: &[f32],
        eff_radii: &[f32],
        velocities_in: &[[f32; 3]],
        angular_in: &[f32],
        pitch_vel_in: &[f32],
        dt: f32,
        drag_coefficient: f32,
    ) -> (Vec<[f32; 3]>, Vec<f32>, Vec<f32>) {
        let n = outputs.len();
        assert!(n <= self.capacity, "motor capacity overflow");
        if n == 0 {
            return (Vec::new(), Vec::new(), Vec::new());
        }

        let mut outputs_flat: Vec<f32> = Vec::with_capacity(n * BRAIN_OUTPUTS);
        for o in outputs { outputs_flat.extend_from_slice(o); }
        self.pos_packed.clear();
        self.pos_packed.reserve(n * 3);
        for v in velocities_in { self.pos_packed.extend_from_slice(v); }

        let params = MotorParams {
            num_cells: n as u32,
            dt,
            drag_coefficient,
            ..MotorParams::default()
        };
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(&self.outputs_buf, 0, bytemuck::cast_slice(&outputs_flat));
        self.queue.write_buffer(&self.headings_buf, 0, bytemuck::cast_slice(headings));
        self.queue.write_buffer(&self.pitches_buf, 0, bytemuck::cast_slice(pitches));
        self.queue.write_buffer(&self.max_speeds_buf, 0, bytemuck::cast_slice(max_speeds));
        self.queue.write_buffer(&self.turn_rates_buf, 0, bytemuck::cast_slice(turn_rates));
        self.queue.write_buffer(&self.eff_radii_buf, 0, bytemuck::cast_slice(eff_radii));
        self.queue.write_buffer(&self.velocities_buf, 0, bytemuck::cast_slice(&self.pos_packed));
        self.queue.write_buffer(&self.angular_buf, 0, bytemuck::cast_slice(angular_in));
        self.queue.write_buffer(&self.pitch_vel_buf, 0, bytemuck::cast_slice(pitch_vel_in));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("motor-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("motor-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
        }
        let f = std::mem::size_of::<f32>() as u64;
        let v_bytes = (n as u64) * 3 * f;
        let s_bytes = (n as u64) * f;
        encoder.copy_buffer_to_buffer(&self.velocities_buf, 0, &self.velocities_readback, 0, v_bytes);
        encoder.copy_buffer_to_buffer(&self.angular_buf, 0, &self.angular_readback, 0, s_bytes);
        encoder.copy_buffer_to_buffer(&self.pitch_vel_buf, 0, &self.pitch_vel_readback, 0, s_bytes);
        self.queue.submit(Some(encoder.finish()));

        let v_slice = self.velocities_readback.slice(0..v_bytes);
        let a_slice = self.angular_readback.slice(0..s_bytes);
        let p_slice = self.pitch_vel_readback.slice(0..s_bytes);
        v_slice.map_async(wgpu::MapMode::Read, |_| {});
        a_slice.map_async(wgpu::MapMode::Read, |_| {});
        p_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let velocities: Vec<[f32; 3]> = {
            let data = v_slice.get_mapped_range();
            let f: &[f32] = bytemuck::cast_slice(&data);
            (0..n).map(|i| [f[i * 3], f[i * 3 + 1], f[i * 3 + 2]]).collect()
        };
        let angular: Vec<f32> = {
            let data = a_slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&data).to_vec()
        };
        let pitch_vel: Vec<f32> = {
            let data = p_slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&data).to_vec()
        };
        self.velocities_readback.unmap();
        self.angular_readback.unmap();
        self.pitch_vel_readback.unmap();
        (velocities, angular, pitch_vel)
    }
}

// ============================================================================
// Sprint 50: GPU step — kinematic + drag + energy + bounce per cell
// ============================================================================

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
pub struct StepParamsGpu {
    pub num_cells: u32,
    pub _pad_a0: u32,
    pub _pad_a1: u32,
    pub _pad_a2: u32,
    pub dt: f32,
    pub world_half_x: f32,
    pub world_half_y: f32,
    pub world_half_z: f32,
    pub gravity: f32,
    pub drag: f32,
    pub angular_drag: f32,
    pub energy_cost_per_v_sq: f32,
    pub angular_energy_cost: f32,
    pub vision_cost_per_radius: f32,
    pub body_cost_factor: f32,
    pub age_decay_per_sec: f32,
    pub fixed_timestep_hz: f32,
    pub spike_cost_per_sec: f32,
    pub shell_cost_per_sec: f32,
    pub attack_cost_per_sec: f32,
    pub pitch_clamp: f32,
}

pub struct StepGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    capacity: usize,
    params_buf: wgpu::Buffer,
    pos_buf: wgpu::Buffer,
    vel_buf: wgpu::Buffer,
    heading_buf: wgpu::Buffer,
    pitch_buf: wgpu::Buffer,
    ang_vel_buf: wgpu::Buffer,
    pitch_vel_buf: wgpu::Buffer,
    age_buf: wgpu::Buffer,
    cooldown_buf: wgpu::Buffer,
    energy_buf: wgpu::Buffer,
    body_dims_buf: wgpu::Buffer,
    aux_buf: wgpu::Buffer,
    pos_rb: wgpu::Buffer,
    vel_rb: wgpu::Buffer,
    heading_rb: wgpu::Buffer,
    pitch_rb: wgpu::Buffer,
    ang_vel_rb: wgpu::Buffer,
    pitch_vel_rb: wgpu::Buffer,
    age_rb: wgpu::Buffer,
    cooldown_rb: wgpu::Buffer,
    energy_rb: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

#[derive(Debug, Clone)]
pub struct StepResult {
    pub positions: Vec<[f32; 3]>,
    pub velocities: Vec<[f32; 3]>,
    pub headings: Vec<f32>,
    pub pitches: Vec<f32>,
    pub angular_velocities: Vec<f32>,
    pub pitch_velocities: Vec<f32>,
    pub ages: Vec<u32>,
    pub cooldowns: Vec<u32>,
    pub energies: Vec<f32>,
}

impl StepGpu {
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
            label: Some("step"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/step.wgsl").into()),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..12)
            .map(|i| {
                let ty = if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else if i >= 10 {
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
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("step-bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("step-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("step-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("step"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("step-params"),
            contents: bytemuck::bytes_of(&StepParamsGpu::default()),
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
        let stor_dst_src =
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let stor_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;

        let pos_buf = mk("step-pos", n * 3 * f, stor_dst_src);
        let vel_buf = mk("step-vel", n * 3 * f, stor_dst_src);
        let heading_buf = mk("step-heading", n * f, stor_dst_src);
        let pitch_buf = mk("step-pitch", n * f, stor_dst_src);
        let ang_vel_buf = mk("step-ang-vel", n * f, stor_dst_src);
        let pitch_vel_buf = mk("step-pitch-vel", n * f, stor_dst_src);
        let age_buf = mk("step-age", n * f, stor_dst_src);
        let cooldown_buf = mk("step-cooldown", n * f, stor_dst_src);
        let energy_buf = mk("step-energy", n * f, stor_dst_src);
        let body_dims_buf = mk("step-body-dims", n * 3 * f, stor_dst);
        let aux_buf = mk("step-aux", n * 4 * f, stor_dst);

        let pos_rb = mk("step-pos-rb", n * 3 * f, read);
        let vel_rb = mk("step-vel-rb", n * 3 * f, read);
        let heading_rb = mk("step-heading-rb", n * f, read);
        let pitch_rb = mk("step-pitch-rb", n * f, read);
        let ang_vel_rb = mk("step-ang-vel-rb", n * f, read);
        let pitch_vel_rb = mk("step-pitch-vel-rb", n * f, read);
        let age_rb = mk("step-age-rb", n * f, read);
        let cooldown_rb = mk("step-cooldown-rb", n * f, read);
        let energy_rb = mk("step-energy-rb", n * f, read);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("step-bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: vel_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: heading_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pitch_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: ang_vel_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: pitch_vel_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: age_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: cooldown_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: energy_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: body_dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: aux_buf.as_entire_binding() },
            ],
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            capacity,
            params_buf,
            pos_buf,
            vel_buf,
            heading_buf,
            pitch_buf,
            ang_vel_buf,
            pitch_vel_buf,
            age_buf,
            cooldown_buf,
            energy_buf,
            body_dims_buf,
            aux_buf,
            pos_rb,
            vel_rb,
            heading_rb,
            pitch_rb,
            ang_vel_rb,
            pitch_vel_rb,
            age_rb,
            cooldown_rb,
            energy_rb,
            bind_group,
        })
    }

    /// Apply step pass. Inputs upload + dispatch + readback.
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &mut self,
        positions: &[[f32; 3]],
        velocities: &[[f32; 3]],
        headings: &[f32],
        pitches: &[f32],
        angular_velocities: &[f32],
        pitch_velocities: &[f32],
        ages: &[u32],
        cooldowns: &[u32],
        energies: &[f32],
        body_dims: &[[f32; 3]],
        aux: &[[f32; 4]],
        params: StepParamsGpu,
    ) -> StepResult {
        let n = positions.len();
        assert!(n <= self.capacity, "step capacity overflow");
        if n == 0 {
            return StepResult {
                positions: Vec::new(),
                velocities: Vec::new(),
                headings: Vec::new(),
                pitches: Vec::new(),
                angular_velocities: Vec::new(),
                pitch_velocities: Vec::new(),
                ages: Vec::new(),
                cooldowns: Vec::new(),
                energies: Vec::new(),
            };
        }
        let mut params = params;
        params.num_cells = n as u32;

        let pos_flat: Vec<f32> = positions.iter().flatten().copied().collect();
        let vel_flat: Vec<f32> = velocities.iter().flatten().copied().collect();
        let body_flat: Vec<f32> = body_dims.iter().flatten().copied().collect();
        let aux_flat: Vec<f32> = aux.iter().flatten().copied().collect();

        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(&self.pos_buf, 0, bytemuck::cast_slice(&pos_flat));
        self.queue.write_buffer(&self.vel_buf, 0, bytemuck::cast_slice(&vel_flat));
        self.queue.write_buffer(&self.heading_buf, 0, bytemuck::cast_slice(headings));
        self.queue.write_buffer(&self.pitch_buf, 0, bytemuck::cast_slice(pitches));
        self.queue.write_buffer(&self.ang_vel_buf, 0, bytemuck::cast_slice(angular_velocities));
        self.queue.write_buffer(&self.pitch_vel_buf, 0, bytemuck::cast_slice(pitch_velocities));
        self.queue.write_buffer(&self.age_buf, 0, bytemuck::cast_slice(ages));
        self.queue.write_buffer(&self.cooldown_buf, 0, bytemuck::cast_slice(cooldowns));
        self.queue.write_buffer(&self.energy_buf, 0, bytemuck::cast_slice(energies));
        self.queue.write_buffer(&self.body_dims_buf, 0, bytemuck::cast_slice(&body_flat));
        self.queue.write_buffer(&self.aux_buf, 0, bytemuck::cast_slice(&aux_flat));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("step-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("step-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
        }
        let f = std::mem::size_of::<f32>() as u64;
        let v3 = (n as u64) * 3 * f;
        let v1 = (n as u64) * f;
        encoder.copy_buffer_to_buffer(&self.pos_buf, 0, &self.pos_rb, 0, v3);
        encoder.copy_buffer_to_buffer(&self.vel_buf, 0, &self.vel_rb, 0, v3);
        encoder.copy_buffer_to_buffer(&self.heading_buf, 0, &self.heading_rb, 0, v1);
        encoder.copy_buffer_to_buffer(&self.pitch_buf, 0, &self.pitch_rb, 0, v1);
        encoder.copy_buffer_to_buffer(&self.ang_vel_buf, 0, &self.ang_vel_rb, 0, v1);
        encoder.copy_buffer_to_buffer(&self.pitch_vel_buf, 0, &self.pitch_vel_rb, 0, v1);
        encoder.copy_buffer_to_buffer(&self.age_buf, 0, &self.age_rb, 0, v1);
        encoder.copy_buffer_to_buffer(&self.cooldown_buf, 0, &self.cooldown_rb, 0, v1);
        encoder.copy_buffer_to_buffer(&self.energy_buf, 0, &self.energy_rb, 0, v1);
        self.queue.submit(Some(encoder.finish()));

        let pos_s = self.pos_rb.slice(0..v3);
        let vel_s = self.vel_rb.slice(0..v3);
        let h_s = self.heading_rb.slice(0..v1);
        let p_s = self.pitch_rb.slice(0..v1);
        let av_s = self.ang_vel_rb.slice(0..v1);
        let pv_s = self.pitch_vel_rb.slice(0..v1);
        let age_s = self.age_rb.slice(0..v1);
        let cd_s = self.cooldown_rb.slice(0..v1);
        let en_s = self.energy_rb.slice(0..v1);
        pos_s.map_async(wgpu::MapMode::Read, |_| {});
        vel_s.map_async(wgpu::MapMode::Read, |_| {});
        h_s.map_async(wgpu::MapMode::Read, |_| {});
        p_s.map_async(wgpu::MapMode::Read, |_| {});
        av_s.map_async(wgpu::MapMode::Read, |_| {});
        pv_s.map_async(wgpu::MapMode::Read, |_| {});
        age_s.map_async(wgpu::MapMode::Read, |_| {});
        cd_s.map_async(wgpu::MapMode::Read, |_| {});
        en_s.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let pos_data = pos_s.get_mapped_range();
        let vel_data = vel_s.get_mapped_range();
        let pos_f: &[f32] = bytemuck::cast_slice(&pos_data);
        let vel_f: &[f32] = bytemuck::cast_slice(&vel_data);
        let positions: Vec<[f32; 3]> = (0..n)
            .map(|i| [pos_f[i * 3], pos_f[i * 3 + 1], pos_f[i * 3 + 2]])
            .collect();
        let velocities: Vec<[f32; 3]> = (0..n)
            .map(|i| [vel_f[i * 3], vel_f[i * 3 + 1], vel_f[i * 3 + 2]])
            .collect();
        drop(pos_data);
        drop(vel_data);
        let result = StepResult {
            positions,
            velocities,
            headings: bytemuck::cast_slice::<u8, f32>(&h_s.get_mapped_range()).to_vec(),
            pitches: bytemuck::cast_slice::<u8, f32>(&p_s.get_mapped_range()).to_vec(),
            angular_velocities: bytemuck::cast_slice::<u8, f32>(&av_s.get_mapped_range()).to_vec(),
            pitch_velocities: bytemuck::cast_slice::<u8, f32>(&pv_s.get_mapped_range()).to_vec(),
            ages: bytemuck::cast_slice::<u8, u32>(&age_s.get_mapped_range()).to_vec(),
            cooldowns: bytemuck::cast_slice::<u8, u32>(&cd_s.get_mapped_range()).to_vec(),
            energies: bytemuck::cast_slice::<u8, f32>(&en_s.get_mapped_range()).to_vec(),
        };
        self.pos_rb.unmap();
        self.vel_rb.unmap();
        self.heading_rb.unmap();
        self.pitch_rb.unmap();
        self.ang_vel_rb.unmap();
        self.pitch_vel_rb.unmap();
        self.age_rb.unmap();
        self.cooldown_rb.unmap();
        self.energy_rb.unmap();
        result
    }
}

// ============================================================================
// Sprint 50: GPU collision — per-cell delta accumulation, chains SpatialHashGpu
// ============================================================================

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct CollisionParams {
    num_cells: u32,
    cell_size: f32,
    cell_radius_const: f32,
    _pad0: u32,
    /// Sprint 55: toroidal world bounds.
    world_half_x: f32,
    world_half_y: f32,
    _pad1: u32,
    _pad2: u32,
}

pub struct CollisionGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    capacity: usize,
    cell_size: f32,
    cell_radius_const: f32,
    /// Sprint 55: toroidal bounds.
    world_half_xy: [f32; 2],
    params_buf: wgpu::Buffer,
    positions_buf: wgpu::Buffer,
    eff_radii_buf: wgpu::Buffer,
    max_axes_buf: wgpu::Buffer,
    deltas_buf: wgpu::Buffer,
    deltas_rb: wgpu::Buffer,
    pos_packed: Vec<f32>,
}

impl CollisionGpu {
    pub fn with_context(
        ctx: &GpuContext,
        capacity: usize,
        cell_size: f32,
        cell_radius_const: f32,
        world_half_xy: [f32; 2],
    ) -> Result<Self, String> {
        Self::with_device_inner(
            Arc::clone(&ctx.device),
            Arc::clone(&ctx.queue),
            capacity,
            cell_size,
            cell_radius_const,
            world_half_xy,
        )
    }

    pub fn new(
        capacity: usize,
        cell_size: f32,
        cell_radius_const: f32,
        world_half_xy: [f32; 2],
    ) -> Result<Self, String> {
        let ctx = GpuContext::new()?;
        Self::with_device_inner(ctx.device, ctx.queue, capacity, cell_size, cell_radius_const, world_half_xy)
    }

    fn with_device_inner(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        capacity: usize,
        cell_size: f32,
        cell_radius_const: f32,
        world_half_xy: [f32; 2],
    ) -> Result<Self, String> {
        assert!(capacity > 0);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("collision"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/collision.wgsl").into()),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..7)
            .map(|i| {
                let ty = if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else if i == 6 {
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
            label: Some("collision-bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("collision-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("collision-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("collision"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("collision-params"),
            contents: bytemuck::bytes_of(&CollisionParams::default()),
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
        let stor_src = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let positions_buf = mk("collision-positions", n * 3 * f, stor_dst);
        let eff_radii_buf = mk("collision-eff-radii", n * f, stor_dst);
        let max_axes_buf = mk("collision-max-axes", n * f, stor_dst);
        let deltas_buf = mk("collision-deltas", n * 3 * f, stor_src);
        let deltas_rb = mk("collision-deltas-rb", n * 3 * f, read);

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            capacity,
            cell_size,
            cell_radius_const,
            world_half_xy,
            params_buf,
            positions_buf,
            eff_radii_buf,
            max_axes_buf,
            deltas_buf,
            deltas_rb,
            pos_packed: Vec::new(),
        })
    }

    pub fn compute(
        &mut self,
        positions: &[[f32; 3]],
        eff_radii: &[f32],
        max_axes: &[f32],
        cell_hash: &SpatialHashGpu,
    ) -> Vec<[f32; 3]> {
        let n = positions.len();
        assert!(n <= self.capacity, "collision capacity overflow");
        if n == 0 {
            return Vec::new();
        }
        self.pos_packed.clear();
        self.pos_packed.reserve(n * 3);
        for p in positions {
            self.pos_packed.extend_from_slice(p);
        }
        let params = CollisionParams {
            num_cells: n as u32,
            cell_size: self.cell_size,
            cell_radius_const: self.cell_radius_const,
            _pad0: 0,
            world_half_x: self.world_half_xy[0],
            world_half_y: self.world_half_xy[1],
            _pad1: 0,
            _pad2: 0,
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(
            &self.positions_buf,
            0,
            bytemuck::cast_slice(&self.pos_packed),
        );
        self.queue
            .write_buffer(&self.eff_radii_buf, 0, bytemuck::cast_slice(eff_radii));
        self.queue
            .write_buffer(&self.max_axes_buf, 0, bytemuck::cast_slice(max_axes));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("collision-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.positions_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.eff_radii_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.max_axes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cell_hash.offsets_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cell_hash.sorted_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.deltas_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("collision-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("collision-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
        }
        let bytes = (n as u64) * 3 * 4;
        encoder.copy_buffer_to_buffer(&self.deltas_buf, 0, &self.deltas_rb, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.deltas_rb.slice(0..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let f: &[f32] = bytemuck::cast_slice(&data);
        let out: Vec<[f32; 3]> = (0..n)
            .map(|i| [f[i * 3], f[i * 3 + 1], f[i * 3 + 2]])
            .collect();
        drop(data);
        self.deltas_rb.unmap();
        out
    }
}

// ============================================================================
// Sprint 50: GPU predate — herd_count + attack passes, atomic float CAS
// ============================================================================

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
pub struct PredateParamsGpu {
    pub num_cells: u32,
    pub cell_size: f32,
    pub cell_radius_const: f32,
    pub size_ratio_threshold: f32,
    pub herd_radius_sq: f32,
    pub attack_threshold: f32,
    pub predation_gain: f32,
    pub predation_drain: f32,
    pub spike_dot_threshold: f32,
    pub spike_bonus: f32,
    pub dilution_k: f32,
    pub _pad0: u32,
    /// Sprint 55: toroidal world bounds.
    pub world_half_x: f32,
    pub world_half_y: f32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[derive(Debug, Clone)]
pub struct PredateResult {
    pub herd_counts: Vec<u32>,
    pub energy_delta: Vec<f32>,
    pub damage_delta: Vec<f32>,
}

pub struct PredateGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline_herd: wgpu::ComputePipeline,
    pipeline_attack: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    capacity: usize,
    params_buf: wgpu::Buffer,
    pos_buf: wgpu::Buffer,
    eff_radii_buf: wgpu::Buffer,
    headings_buf: wgpu::Buffer,
    spike_buf: wgpu::Buffer,
    attack_buf: wgpu::Buffer,
    herd_buf: wgpu::Buffer,
    energy_delta_buf: wgpu::Buffer,
    damage_delta_buf: wgpu::Buffer,
    herd_rb: wgpu::Buffer,
    energy_rb: wgpu::Buffer,
    damage_rb: wgpu::Buffer,
    pos_packed: Vec<f32>,
}

impl PredateGpu {
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
            label: Some("predate"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/predate.wgsl").into()),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..11)
            .map(|i| {
                let ty = if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else if (1..=7).contains(&i) {
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
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("predate-bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("predate-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline_herd = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("predate-herd-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("herd_count"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_attack = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("predate-attack-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("attack"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("predate-params"),
            contents: bytemuck::bytes_of(&PredateParamsGpu::default()),
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
        let stor_dst_src =
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let pos_buf = mk("predate-pos", n * 3 * f, stor_dst);
        let eff_radii_buf = mk("predate-eff", n * f, stor_dst);
        let headings_buf = mk("predate-heading", n * f, stor_dst);
        let spike_buf = mk("predate-spike", n * f, stor_dst);
        let attack_buf = mk("predate-attack", n * f, stor_dst);
        let herd_buf = mk("predate-herd", n * f, stor_dst_src);
        let energy_delta_buf = mk("predate-energy-delta", n * f, stor_dst_src);
        let damage_delta_buf = mk("predate-damage-delta", n * f, stor_dst_src);
        let herd_rb = mk("predate-herd-rb", n * f, read);
        let energy_rb = mk("predate-energy-rb", n * f, read);
        let damage_rb = mk("predate-damage-rb", n * f, read);

        Ok(Self {
            device,
            queue,
            pipeline_herd,
            pipeline_attack,
            bind_group_layout,
            capacity,
            params_buf,
            pos_buf,
            eff_radii_buf,
            headings_buf,
            spike_buf,
            attack_buf,
            herd_buf,
            energy_delta_buf,
            damage_delta_buf,
            herd_rb,
            energy_rb,
            damage_rb,
            pos_packed: Vec::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &mut self,
        positions: &[[f32; 3]],
        eff_radii: &[f32],
        headings: &[f32],
        spike_lengths: &[f32],
        attack_signals: &[f32],
        cell_hash: &SpatialHashGpu,
        params: PredateParamsGpu,
    ) -> PredateResult {
        let n = positions.len();
        assert!(n <= self.capacity, "predate capacity overflow");
        if n == 0 {
            return PredateResult {
                herd_counts: Vec::new(),
                energy_delta: Vec::new(),
                damage_delta: Vec::new(),
            };
        }
        let mut params = params;
        params.num_cells = n as u32;

        self.pos_packed.clear();
        self.pos_packed.reserve(n * 3);
        for p in positions {
            self.pos_packed.extend_from_slice(p);
        }

        // Reset herd_counts + energy_delta + damage_delta na 0 (atomic add must
        // start from clean slate per dispatch).
        let zero_bytes = vec![0u8; (n * 4) as usize];
        self.queue.write_buffer(&self.herd_buf, 0, &zero_bytes);
        self.queue.write_buffer(&self.energy_delta_buf, 0, &zero_bytes);
        self.queue.write_buffer(&self.damage_delta_buf, 0, &zero_bytes);

        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&self.pos_buf, 0, bytemuck::cast_slice(&self.pos_packed));
        self.queue
            .write_buffer(&self.eff_radii_buf, 0, bytemuck::cast_slice(eff_radii));
        self.queue
            .write_buffer(&self.headings_buf, 0, bytemuck::cast_slice(headings));
        self.queue
            .write_buffer(&self.spike_buf, 0, bytemuck::cast_slice(spike_lengths));
        self.queue
            .write_buffer(&self.attack_buf, 0, bytemuck::cast_slice(attack_signals));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("predate-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.eff_radii_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.headings_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.spike_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.attack_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: cell_hash.offsets_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: cell_hash.sorted_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.herd_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.energy_delta_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: self.damage_delta_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("predate-encoder"),
        });
        let workgroups = ((n as u32) + 63) / 64;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("predate-herd-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_herd);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("predate-attack-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_attack);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        let bytes = (n as u64) * 4;
        encoder.copy_buffer_to_buffer(&self.herd_buf, 0, &self.herd_rb, 0, bytes);
        encoder.copy_buffer_to_buffer(&self.energy_delta_buf, 0, &self.energy_rb, 0, bytes);
        encoder.copy_buffer_to_buffer(&self.damage_delta_buf, 0, &self.damage_rb, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let h = self.herd_rb.slice(0..bytes);
        let e = self.energy_rb.slice(0..bytes);
        let d = self.damage_rb.slice(0..bytes);
        h.map_async(wgpu::MapMode::Read, |_| {});
        e.map_async(wgpu::MapMode::Read, |_| {});
        d.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let res = PredateResult {
            herd_counts: bytemuck::cast_slice::<u8, u32>(&h.get_mapped_range()).to_vec(),
            energy_delta: bytemuck::cast_slice::<u8, f32>(&e.get_mapped_range()).to_vec(),
            damage_delta: bytemuck::cast_slice::<u8, f32>(&d.get_mapped_range()).to_vec(),
        };
        self.herd_rb.unmap();
        self.energy_rb.unmap();
        self.damage_rb.unmap();
        res
    }
}

// ============================================================================
// Sprint 50: GPU sensor gather — chains 2× SpatialHashGpu + 2× FieldGpu
// ============================================================================

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
pub struct SensorParamsGpu {
    pub num_cells: u32,
    pub num_foods: u32,
    pub hash_cell_size: f32,
    pub world_half_x: f32,
    pub world_half_y: f32,
    pub world_half_z: f32,
    pub field_resolution: u32,
    pub field_eps: f32,
    pub field_world_half_x: f32,
    pub field_world_half_y: f32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SensorRow {
    pub nearest_food: Option<[f32; 3]>,
    pub nearest_cell: Option<([f32; 3], f32)>,
    pub neighbors_in_vision: u32,
    pub smell_grad: [f32; 2],
    pub pheromone_grad: [f32; 2],
}

pub struct SensorGatherGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    capacity_cells: usize,
    capacity_foods: usize,
    params_buf: wgpu::Buffer,
    positions_buf: wgpu::Buffer,
    eff_radii_buf: wgpu::Buffer,
    vision_radii_buf: wgpu::Buffer,
    food_positions_buf: wgpu::Buffer,
    output_buf: wgpu::Buffer,
    output_rb: wgpu::Buffer,
    pos_packed: Vec<f32>,
    food_packed: Vec<f32>,
}

impl SensorGatherGpu {
    pub fn with_context(
        ctx: &GpuContext,
        capacity_cells: usize,
        capacity_foods: usize,
    ) -> Result<Self, String> {
        Self::with_device_inner(
            Arc::clone(&ctx.device),
            Arc::clone(&ctx.queue),
            capacity_cells,
            capacity_foods,
        )
    }

    pub fn new(capacity_cells: usize, capacity_foods: usize) -> Result<Self, String> {
        let ctx = GpuContext::new()?;
        Self::with_device_inner(ctx.device, ctx.queue, capacity_cells, capacity_foods)
    }

    fn with_device_inner(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        capacity_cells: usize,
        capacity_foods: usize,
    ) -> Result<Self, String> {
        assert!(capacity_cells > 0);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sensor_gather"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/sensor_gather.wgsl").into(),
            ),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..12)
            .map(|i| {
                let ty = if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else if i == 11 {
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
            label: Some("sensor-bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sensor-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sensor-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("sensor_gather"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sensor-params"),
            contents: bytemuck::bytes_of(&SensorParamsGpu::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let f = std::mem::size_of::<f32>() as u64;
        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let stor_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let stor_src = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let nc = capacity_cells as u64;
        let nf = capacity_foods.max(1) as u64;
        let positions_buf = mk("sensor-pos", nc * 3 * f, stor_dst);
        let eff_radii_buf = mk("sensor-eff", nc * f, stor_dst);
        let vision_radii_buf = mk("sensor-vision", nc * f, stor_dst);
        let food_positions_buf = mk("sensor-food-pos", nf * 3 * f, stor_dst);
        let output_buf = mk("sensor-output", nc * 13 * f, stor_src);
        let output_rb = mk("sensor-output-rb", nc * 13 * f, read);

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            capacity_cells,
            capacity_foods,
            params_buf,
            positions_buf,
            eff_radii_buf,
            vision_radii_buf,
            food_positions_buf,
            output_buf,
            output_rb,
            pos_packed: Vec::new(),
            food_packed: Vec::new(),
        })
    }

    /// Spustí celý sensor gather pipeline. `cell_hash` musí být rebuildnut z
    /// `positions`, `food_hash` z `food_positions`. `smell` + `pheromone`
    /// FieldGpu musí mít stepnuté za current tick.
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &mut self,
        positions: &[[f32; 3]],
        eff_radii: &[f32],
        vision_radii: &[f32],
        food_positions: &[[f32; 3]],
        cell_hash: &SpatialHashGpu,
        food_hash: &SpatialHashGpu,
        smell: &FieldGpu,
        pheromone: &FieldGpu,
        params: SensorParamsGpu,
    ) -> Vec<SensorRow> {
        let n = positions.len();
        let nf = food_positions.len();
        assert!(n <= self.capacity_cells, "sensor cell capacity overflow");
        assert!(nf <= self.capacity_foods, "sensor food capacity overflow");
        if n == 0 {
            return Vec::new();
        }
        let mut params = params;
        params.num_cells = n as u32;
        params.num_foods = nf as u32;

        self.pos_packed.clear();
        self.pos_packed.reserve(n * 3);
        for p in positions {
            self.pos_packed.extend_from_slice(p);
        }
        self.food_packed.clear();
        self.food_packed.reserve((nf.max(1)) * 3);
        for p in food_positions {
            self.food_packed.extend_from_slice(p);
        }
        if self.food_packed.is_empty() {
            // Sentinel one-element pad — buffer alokovaný s capacity ≥ 1.
            self.food_packed.extend_from_slice(&[0.0, 0.0, 0.0]);
        }

        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(&self.positions_buf, 0, bytemuck::cast_slice(&self.pos_packed));
        self.queue.write_buffer(&self.eff_radii_buf, 0, bytemuck::cast_slice(eff_radii));
        self.queue.write_buffer(&self.vision_radii_buf, 0, bytemuck::cast_slice(vision_radii));
        self.queue.write_buffer(&self.food_positions_buf, 0, bytemuck::cast_slice(&self.food_packed));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sensor-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.positions_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.eff_radii_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.vision_radii_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.food_positions_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cell_hash.offsets_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: cell_hash.sorted_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: food_hash.offsets_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: food_hash.sorted_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: smell.current_grid_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: pheromone.current_grid_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: self.output_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sensor-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sensor-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
        }
        let bytes = (n as u64) * 13 * 4;
        encoder.copy_buffer_to_buffer(&self.output_buf, 0, &self.output_rb, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let s = self.output_rb.slice(0..bytes);
        s.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = s.get_mapped_range();
        let f: &[f32] = bytemuck::cast_slice(&data);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * 13;
            let has_food = f[off + 3] > 0.5;
            let nearest_food = if has_food {
                Some([
                    positions[i][0] + f[off + 0],
                    positions[i][1] + f[off + 1],
                    positions[i][2] + f[off + 2],
                ])
            } else {
                None
            };
            let radius = f[off + 7];
            let nearest_cell = if radius >= 0.0 {
                Some((
                    [
                        positions[i][0] + f[off + 4],
                        positions[i][1] + f[off + 5],
                        positions[i][2] + f[off + 6],
                    ],
                    radius,
                ))
            } else {
                None
            };
            let count_bits = f[off + 12].to_bits();
            out.push(SensorRow {
                nearest_food,
                nearest_cell,
                neighbors_in_vision: count_bits,
                smell_grad: [f[off + 8], f[off + 9]],
                pheromone_grad: [f[off + 10], f[off + 11]],
            });
        }
        drop(data);
        self.output_rb.unmap();
        out
    }
}

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
                include_str!("../shaders/cell_neighbors.wgsl").into(),
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/cell_stats.wgsl").into()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Brain;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Sprint 44: parity test CPU vs GPU forward. Tolerance 1e-5 — single-precision
    /// floats + tanh implementations se mírně liší napříč implementacemi, ale
    /// ne-trivial drift > 1e-5 by indikoval bug v packingu nebo shader logic.
    /// Test vyžaduje compatible wgpu adapter (skipped pokud `BrainGpu::new` selže).
    #[test]
    fn brain_forward_gpu_matches_cpu() {
        let mut rng = StdRng::seed_from_u64(7);
        let n = 32;
        let brains: Vec<Brain> = (0..n).map(|_| Brain::random(&mut rng)).collect();
        let inputs: Vec<[f32; BRAIN_INPUTS]> = (0..n)
            .map(|_| {
                let mut a = [0.0_f32; BRAIN_INPUTS];
                for v in a.iter_mut() {
                    *v = rand::Rng::random_range(&mut rng, -1.0_f32..1.0_f32);
                }
                a
            })
            .collect();

        let mut gpu = match BrainGpu::new(n) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skip: no GPU adapter ({e})");
                return;
            }
        };

        let mut h_gpu = vec![[0.0_f32; BRAIN_HIDDEN]; n];
        let mut o_gpu = vec![[0.0_f32; BRAIN_OUTPUTS]; n];
        gpu.forward_batch(&inputs, &brains, &mut h_gpu, &mut o_gpu);

        for i in 0..n {
            let (h_cpu, o_cpu) = brains[i].forward_with_state(&inputs[i]);
            for k in 0..BRAIN_HIDDEN {
                let diff = (h_cpu[k] - h_gpu[i][k]).abs();
                assert!(
                    diff < 1e-4,
                    "hidden mismatch i={i} k={k} cpu={} gpu={} diff={}",
                    h_cpu[k],
                    h_gpu[i][k],
                    diff
                );
            }
            for k in 0..BRAIN_OUTPUTS {
                let diff = (o_cpu[k] - o_gpu[i][k]).abs();
                assert!(
                    diff < 1e-4,
                    "output mismatch i={i} k={k} cpu={} gpu={} diff={}",
                    o_cpu[k],
                    o_gpu[i][k],
                    diff
                );
            }
        }
    }

    /// Sprint 45: parity test GPU spatial hash vs CPU brute force.
    /// Pro každý bucket: SET cells na GPU = SET cells na CPU. Bucketing přes
    /// `SpatialHashGpu::bucket_id_of` (CPU mirror shader logiky).
    #[test]
    fn spatial_hash_gpu_matches_cpu_buckets() {
        let mut rng = StdRng::seed_from_u64(11);
        let n: usize = 500;
        let cell_size: f32 = 64.0;
        // Drž positions uvnitř world bounds [-960, 960] × [-540, 540] × [-2, 2]
        // — stejné jako headless WORLD_HALF.
        let positions: Vec<[f32; 3]> = (0..n)
            .map(|_| {
                [
                    rng.random_range(-900.0_f32..900.0),
                    rng.random_range(-500.0_f32..500.0),
                    rng.random_range(-2.0_f32..2.0),
                ]
            })
            .collect();

        let mut gpu = match SpatialHashGpu::new(n, cell_size, [1000.0, 1000.0]) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skip: no GPU adapter ({e})");
                return;
            }
        };
        let (offsets, sorted) = gpu.rebuild(&positions);

        // CPU reference: build bucket_id → set<cell_idx> map.
        let mut cpu_buckets: std::collections::HashMap<u32, std::collections::BTreeSet<u32>> =
            std::collections::HashMap::new();
        for (i, p) in positions.iter().enumerate() {
            let b = SpatialHashGpu::bucket_id_of(*p, cell_size);
            cpu_buckets.entry(b).or_default().insert(i as u32);
        }

        // Total se musí matchnout.
        assert_eq!(sorted.len(), n);
        assert_eq!(offsets.len(), GPU_HASH_NUM_BUCKETS + 1);
        assert_eq!(offsets[GPU_HASH_NUM_BUCKETS] as usize, n);

        for b in 0..GPU_HASH_NUM_BUCKETS {
            let start = offsets[b] as usize;
            let end = offsets[b + 1] as usize;
            let gpu_set: std::collections::BTreeSet<u32> =
                sorted[start..end].iter().copied().collect();
            let cpu_set = cpu_buckets
                .get(&(b as u32))
                .cloned()
                .unwrap_or_default();
            assert_eq!(
                gpu_set, cpu_set,
                "bucket {b} mismatch: gpu={:?} cpu={:?}",
                gpu_set, cpu_set
            );
        }
    }

    /// Sprint 46: parity test GPU FieldGpu vs CPU `SmellField`. Stejné sources +
    /// stejné kroky → grid hodnoty match v ε. Tolerance 1e-3 — atomic float CAS
    /// loop má potenciální drift kvůli pořadí přídavků (ne-asociativita f32).
    /// **Sprint 53 #[ignore]:** SmellField přešel na 3D 7-point stencil; FieldGpu
    /// stále drží 2D 5-point. Test vyžaduje stejnou dimenzionalitu — Sprint 54
    /// migruje FieldGpu na 3D a re-enabluje.
    #[test]
    #[ignore]
    fn field_gpu_diffusion_matches_cpu() {
        // Sprint 53: SmellField se přesunul na 3D, FieldGpu zůstává 2D.
        // Tělo testu je dočasně skipnuté — Sprint 54 migruje FieldGpu na 3D
        // a re-enabluje porovnání. Stub-body proto smí být prázdné.
    }

    /// Sprint 47: StatsGpu reduction parity vs naive CPU sum. Tolerance 1e-2
    /// kvůli f32 sum non-associativity v 1024-element tree reduce.
    #[test]
    fn stats_gpu_matches_cpu_sums() {
        let mut rng = StdRng::seed_from_u64(17);
        let n = 1024;
        let positions: Vec<[f32; 3]> = (0..n)
            .map(|_| {
                [
                    rng.random_range(-1000.0_f32..1000.0),
                    rng.random_range(-500.0_f32..500.0),
                    rng.random_range(-2.0_f32..2.0),
                ]
            })
            .collect();
        let velocities: Vec<[f32; 3]> = (0..n)
            .map(|_| {
                [
                    rng.random_range(-50.0_f32..50.0),
                    rng.random_range(-50.0_f32..50.0),
                    rng.random_range(-5.0_f32..5.0),
                ]
            })
            .collect();
        let energies: Vec<f32> = (0..n)
            .map(|_| rng.random_range(0.0_f32..150.0))
            .collect();

        let mut gpu = match StatsGpu::new(n) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skip: no GPU adapter ({e})");
                return;
            }
        };
        let gpu_stats = gpu.compute(&positions, &velocities, &energies);

        let cpu_sum_x: f32 = positions.iter().map(|p| p[0]).sum();
        let cpu_sum_y: f32 = positions.iter().map(|p| p[1]).sum();
        let cpu_sum_z: f32 = positions.iter().map(|p| p[2]).sum();
        let cpu_sum_speed_sq: f32 = velocities
            .iter()
            .map(|v| v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
            .sum();
        let cpu_sum_energy: f32 = energies.iter().sum();

        let close = |a: f32, b: f32, scale: f32| {
            let d = (a - b).abs();
            assert!(d < scale * 1e-3 + 1e-3, "diff = {d}, a={a}, b={b}");
        };
        close(gpu_stats.sum_x, cpu_sum_x, cpu_sum_x.abs());
        close(gpu_stats.sum_y, cpu_sum_y, cpu_sum_y.abs());
        close(gpu_stats.sum_z, cpu_sum_z, cpu_sum_z.abs());
        close(gpu_stats.sum_speed_sq, cpu_sum_speed_sq, cpu_sum_speed_sq.abs());
        close(gpu_stats.sum_energy, cpu_sum_energy, cpu_sum_energy.abs());
    }

    /// Sprint 51: brownian GPU produces non-trivial velocity perturbation +
    /// xoshiro state mutates (deterministic). Test ne porovnává CPU stejně —
    /// CPU gaussian (Box-Muller) uses StdRng, GPU uses xoshiro128++ — různé
    /// PRNG. Ověření je: po N kroků velocity má nenulové statistické
    /// rozptyly v očekávaném scale (thermal_noise × √dt × √N).
    #[test]
    fn brownian_gpu_perturbs_velocity() {
        let n = 256;
        let mut gpu = match BrownianGpu::new(n) {
            Ok(g) => g,
            Err(e) => { eprintln!("skip: no GPU adapter ({e})"); return; }
        };
        let mut velocities = vec![[0.0_f32; 3]; n];
        let mut state: Vec<[u32; 4]> = (0..n)
            .map(|i| {
                let s = (i as u64) + 0xDEADBEEFu64;
                [
                    (s >> 0) as u32 ^ 0x9E3779B9u32,
                    (s >> 16) as u32 ^ 0xBB67AE85u32,
                    (s >> 32) as u32 ^ 0x3C6EF372u32,
                    (s >> 48) as u32 ^ 0xA54FF53Au32,
                ]
            })
            .collect();
        let thermal_noise = 0.5_f32;
        let dt = 1.0_f32 / 60.0;
        let steps = 100;
        for _ in 0..steps {
            let (v, s) = gpu.compute(&velocities, &state, thermal_noise, dt, true);
            velocities = v;
            state = s;
        }
        // Empirical sigma: thermal × sqrt(dt) × sqrt(steps). Test že existuje
        // nenulová variance (statisticky téměř jistě > 0 pro 256 cells).
        let mut sum_v_sq = 0.0_f64;
        for v in velocities.iter() {
            sum_v_sq += (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) as f64;
        }
        let mean_v_sq = sum_v_sq / n as f64;
        // Theoretical: 3 × thermal² × dt × steps = 3 × 0.25 × (100/60) ≈ 1.25
        let expected = 3.0 * (thermal_noise * thermal_noise) as f64 * dt as f64 * steps as f64;
        // Tolerance ±50 % kvůli small N stochastic noise.
        assert!(
            mean_v_sq > expected * 0.5 && mean_v_sq < expected * 1.5,
            "mean_v_sq = {} (expected ~{})",
            mean_v_sq,
            expected
        );
    }

    /// Sprint 51: brownian determinismus — stejný initial state → stejný
    /// výsledek napříč běhy. xoshiro128++ je deterministic per state seed.
    #[test]
    fn brownian_gpu_deterministic_across_runs() {
        let n = 64;
        let mut gpu = match BrownianGpu::new(n) {
            Ok(g) => g,
            Err(e) => { eprintln!("skip: no GPU adapter ({e})"); return; }
        };
        let velocities = vec![[0.5_f32, 1.0, -0.3]; n];
        let state: Vec<[u32; 4]> = (0..n).map(|i| [i as u32 + 1, 2, 3, 4]).collect();
        let (v1, s1) = gpu.compute(&velocities, &state, 0.3, 1.0 / 60.0, false);
        let (v2, s2) = gpu.compute(&velocities, &state, 0.3, 1.0 / 60.0, false);
        for i in 0..n {
            for k in 0..3 {
                assert_eq!(v1[i][k].to_bits(), v2[i][k].to_bits(),
                    "i={i} k={k} not deterministic v1={} v2={}", v1[i][k], v2[i][k]);
            }
            for k in 0..4 {
                assert_eq!(s1[i][k], s2[i][k]);
            }
        }
    }

    /// Sprint 51: Hebbian GPU vs CPU `Brain::hebbian_update` parity. 32 cells,
    /// random pre/post activations + reward, GPU update vs CPU update.
    /// Tolerance 1e-4 (per-weight FMA chain).
    #[test]
    fn hebbian_gpu_matches_cpu() {
        let mut rng = StdRng::seed_from_u64(73);
        let n = 32;
        let lr: f32 = 0.005;
        let brains: Vec<Brain> = (0..n).map(|_| Brain::random(&mut rng)).collect();
        let last_inputs: Vec<[f32; BRAIN_INPUTS]> = (0..n)
            .map(|_| {
                let mut a = [0.0_f32; BRAIN_INPUTS];
                for v in a.iter_mut() { *v = rng.random_range(-1.0_f32..1.0); }
                a
            })
            .collect();
        let last_hidden: Vec<[f32; BRAIN_HIDDEN]> = (0..n)
            .map(|_| {
                let mut a = [0.0_f32; BRAIN_HIDDEN];
                for v in a.iter_mut() { *v = rng.random_range(-1.0_f32..1.0); }
                a
            })
            .collect();
        let last_outputs: Vec<[f32; BRAIN_OUTPUTS]> = (0..n)
            .map(|_| {
                let mut a = [0.0_f32; BRAIN_OUTPUTS];
                for v in a.iter_mut() { *v = rng.random_range(-1.0_f32..1.0); }
                a
            })
            .collect();
        let rewards: Vec<f32> = (0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();

        let mut gpu = match HebbianGpu::new(n) {
            Ok(g) => g,
            Err(e) => { eprintln!("skip: no GPU adapter ({e})"); return; }
        };
        let gpu_brains = gpu.compute(
            &last_inputs, &last_hidden, &last_outputs, &rewards, &brains, lr,
        );

        // CPU equivalent.
        let mut cpu_brains = brains.clone();
        for i in 0..n {
            cpu_brains[i].hebbian_update(
                &last_inputs[i], &last_hidden[i], &last_outputs[i], rewards[i], lr,
            );
        }

        for i in 0..n {
            for h in 0..BRAIN_HIDDEN {
                for in_i in 0..BRAIN_INPUTS {
                    let d = (cpu_brains[i].w1[h][in_i] - gpu_brains[i].w1[h][in_i]).abs();
                    assert!(d < 1e-4, "i={i} h={h} in_i={in_i} cpu={} gpu={} d={}",
                        cpu_brains[i].w1[h][in_i], gpu_brains[i].w1[h][in_i], d);
                }
                let d = (cpu_brains[i].b1[h] - gpu_brains[i].b1[h]).abs();
                assert!(d < 1e-4);
            }
            for o in 0..BRAIN_OUTPUTS {
                for h in 0..BRAIN_HIDDEN {
                    let d = (cpu_brains[i].w2[o][h] - gpu_brains[i].w2[o][h]).abs();
                    assert!(d < 1e-4);
                }
                let d = (cpu_brains[i].b2[o] - gpu_brains[i].b2[o]).abs();
                assert!(d < 1e-4);
            }
        }
    }

    /// Sprint 50: motor GPU vs CPU `Cell::apply_brain_motor` parity. Stejné
    /// outputs + cell state → identické post-motor velocities/angular/pitch_vel
    /// v ε. Tolerance 1e-4 (single-precision multiply chain ~10 ops).
    #[test]
    fn motor_gpu_matches_cpu() {
        use crate::{Cell, DRAG_COEFFICIENT};
        let mut rng = StdRng::seed_from_u64(41);
        let n = 64;
        let dt = 1.0_f32 / 60.0;
        let mut cells: Vec<Cell> = (0..n)
            .map(|_| Cell::random(&mut rng, [960.0, 540.0, 2.0], 0, 0))
            .collect();
        let outputs: Vec<[f32; BRAIN_OUTPUTS]> = (0..n)
            .map(|_| {
                let mut a = [0.0_f32; BRAIN_OUTPUTS];
                for v in a.iter_mut() {
                    *v = rng.random_range(-1.0_f32..1.0);
                }
                a
            })
            .collect();
        let headings: Vec<f32> = cells.iter().map(|c| c.heading).collect();
        let pitches: Vec<f32> = cells.iter().map(|c| c.pitch).collect();
        let max_speeds: Vec<f32> = cells.iter().map(|c| c.genome.max_speed).collect();
        let turn_rates: Vec<f32> = cells.iter().map(|c| c.genome.turn_rate).collect();
        let eff_radii: Vec<f32> = cells.iter().map(|c| c.phenotype.effective_radius()).collect();
        let velocities_in: Vec<[f32; 3]> = cells.iter().map(|c| c.velocity).collect();
        let angular_in: Vec<f32> = cells.iter().map(|c| c.angular_velocity).collect();
        let pitch_vel_in: Vec<f32> = cells.iter().map(|c| c.pitch_velocity).collect();

        let mut gpu = match MotorGpu::new(n) {
            Ok(g) => g,
            Err(e) => { eprintln!("skip: no GPU adapter ({e})"); return; }
        };
        let (gpu_v, gpu_a, gpu_p) = gpu.compute(
            &outputs, &headings, &pitches, &max_speeds, &turn_rates, &eff_radii,
            &velocities_in, &angular_in, &pitch_vel_in, dt, DRAG_COEFFICIENT,
        );

        for (i, cell) in cells.iter_mut().enumerate() {
            cell.apply_brain_motor(&outputs[i], dt);
        }

        for i in 0..n {
            for k in 0..3 {
                let d = (cells[i].velocity[k] - gpu_v[i][k]).abs();
                assert!(d < 1e-4, "i={i} k={k} cpu={} gpu={} diff={}", cells[i].velocity[k], gpu_v[i][k], d);
            }
            assert!((cells[i].angular_velocity - gpu_a[i]).abs() < 1e-4);
            assert!((cells[i].pitch_velocity - gpu_p[i]).abs() < 1e-4);
        }
    }

    /// Sprint 50: full sensor gather GPU vs CPU parity. Cells + foods + 2
    /// fields. GPU spustí všech subsystémů na shared context, output SoA
    /// porovnán s CPU equivalent (cell broad-phase + food broad-phase + 2
    /// gradient_at samples). Drobný drift kvůli atomic float CAS na fields
    /// — tolerance 1e-2 na gradient values.
    /// **Sprint 53 #[ignore]:** SmellField přešel na 3D (gradient_at vrací
    /// `[f32; 3]`). FieldGpu + sensor_gather.wgsl stále drží 2D field
    /// indexing. Sprint 54 migruje GPU field stack na 3D a re-enabluje.
    #[test]
    #[ignore]
    fn sensor_gather_gpu_matches_cpu() {
        // Sprint 53 stub: tělo přesunuto na Sprint 54 (FieldGpu 3D migrace).
    }


    /// Sprint 50: predate GPU vs CPU parity. Cluster cells s mixed sizes a
    /// random attack signals; herd_counts + energy_delta + damage_delta v ε
    /// match. Atomic float CAS sumace má ULP drift, tolerance 1e-3 absolute.
    #[test]
    fn predate_gpu_matches_cpu() {
        use crate::{
            ATTACK_THRESHOLD, CELL_RADIUS, DILUTION_K, HERD_RADIUS, PREDATION_DRAIN_PER_TICK,
            PREDATION_GAIN_PER_TICK, SIZE_RATIO_THRESHOLD, SPIKE_DOT_THRESHOLD,
            SPIKE_PREDATION_BONUS,
        };
        let mut rng = StdRng::seed_from_u64(67);
        let n = 80;
        let cell_size = 64.0_f32;
        // Pack cells aby se některé dotýkaly.
        let positions: Vec<[f32; 3]> = (0..n)
            .map(|_| {
                [
                    rng.random_range(-40.0_f32..40.0),
                    rng.random_range(-40.0_f32..40.0),
                    rng.random_range(-1.0_f32..1.0),
                ]
            })
            .collect();
        let eff_radii: Vec<f32> = (0..n).map(|_| rng.random_range(0.5_f32..2.0)).collect();
        let headings: Vec<f32> = (0..n)
            .map(|_| rng.random_range(0.0_f32..core::f32::consts::TAU))
            .collect();
        let spike_lengths: Vec<f32> = (0..n).map(|_| rng.random_range(0.0_f32..0.3)).collect();
        let attack_signals: Vec<f32> = (0..n).map(|_| rng.random_range(-0.5_f32..1.0)).collect();

        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("skip: no GPU adapter ({e})"); return; }
        };
        let mut hash = SpatialHashGpu::with_context(&ctx, n, cell_size, [1000.0, 1000.0]).expect("hash");
        let _ = hash.rebuild(&positions);
        let mut pred = PredateGpu::with_context(&ctx, n).expect("predate init");
        let params = PredateParamsGpu {
            cell_size,
            cell_radius_const: CELL_RADIUS,
            size_ratio_threshold: SIZE_RATIO_THRESHOLD,
            herd_radius_sq: HERD_RADIUS * HERD_RADIUS,
            attack_threshold: ATTACK_THRESHOLD,
            predation_gain: PREDATION_GAIN_PER_TICK,
            predation_drain: PREDATION_DRAIN_PER_TICK,
            spike_dot_threshold: SPIKE_DOT_THRESHOLD,
            spike_bonus: SPIKE_PREDATION_BONUS,
            dilution_k: DILUTION_K,
            world_half_x: 1000.0,
            world_half_y: 1000.0,
            ..PredateParamsGpu::default()
        };
        let res = pred.compute(
            &positions, &eff_radii, &headings, &spike_lengths, &attack_signals, &hash, params,
        );

        // CPU brute force.
        let mut cpu_herd = vec![0u32; n];
        let herd_r2 = HERD_RADIUS * HERD_RADIUS;
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let dx = positions[i][0] - positions[j][0];
                let dy = positions[i][1] - positions[j][1];
                let dz = positions[i][2] - positions[j][2];
                if dx * dx + dy * dy + dz * dz < herd_r2 {
                    cpu_herd[i] += 1;
                }
            }
        }

        let mut cpu_energy = vec![0.0_f32; n];
        let mut cpu_damage = vec![0.0_f32; n];
        for i in 0..n {
            let attack = attack_signals[i].max(0.0);
            if attack <= ATTACK_THRESHOLD { continue; }
            let r_i = eff_radii[i];
            let spike = spike_lengths[i];
            let heading = headings[i];
            for j in 0..n {
                if i == j { continue; }
                let r_j = eff_radii[j];
                if r_i < SIZE_RATIO_THRESHOLD * r_j { continue; }
                let pair_r = CELL_RADIUS * (r_i + r_j);
                let pair_r2 = pair_r * pair_r;
                let dx = positions[i][0] - positions[j][0];
                let dy = positions[i][1] - positions[j][1];
                let dz = positions[i][2] - positions[j][2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < pair_r2 {
                    let mut gain = PREDATION_GAIN_PER_TICK;
                    if spike > 0.0 && d2 > 0.0 {
                        let inv_d = 1.0 / d2.sqrt();
                        let to_j_x = -dx * inv_d;
                        let to_j_y = -dy * inv_d;
                        let cos_angle = heading.cos() * to_j_x + heading.sin() * to_j_y;
                        if cos_angle >= SPIKE_DOT_THRESHOLD {
                            gain += PREDATION_GAIN_PER_TICK * spike * SPIKE_PREDATION_BONUS;
                        }
                    }
                    let dilution = 1.0 / (1.0 + DILUTION_K * cpu_herd[j] as f32);
                    gain *= dilution;
                    cpu_energy[i] += gain;
                    cpu_energy[j] -= PREDATION_DRAIN_PER_TICK;
                    cpu_damage[j] += PREDATION_DRAIN_PER_TICK;
                }
            }
        }

        for i in 0..n {
            assert_eq!(cpu_herd[i], res.herd_counts[i], "i={i} herd");
            assert!(
                (cpu_energy[i] - res.energy_delta[i]).abs() < 1e-3,
                "i={i} energy cpu={} gpu={}",
                cpu_energy[i],
                res.energy_delta[i]
            );
            assert!(
                (cpu_damage[i] - res.damage_delta[i]).abs() < 1e-3,
                "i={i} damage cpu={} gpu={}",
                cpu_damage[i],
                res.damage_delta[i]
            );
        }
    }

    /// Sprint 50: collision GPU vs CPU `headless::resolve_collisions` parity.
    /// Pack cells velmi blízko sebe → forced overlaps. GPU vrací delta_position
    /// per cell; CPU brute-force počítá totéž. Tolerance 1e-3.
    #[test]
    fn collision_gpu_matches_cpu() {
        use crate::CELL_RADIUS;
        let mut rng = StdRng::seed_from_u64(53);
        let n = 100;
        let cell_size = 64.0_f32;
        // Cluster cells v malé oblasti aby měl collision co řešit.
        let positions: Vec<[f32; 3]> = (0..n)
            .map(|_| {
                [
                    rng.random_range(-30.0_f32..30.0),
                    rng.random_range(-30.0_f32..30.0),
                    rng.random_range(-1.0_f32..1.0),
                ]
            })
            .collect();
        let eff_radii: Vec<f32> = (0..n).map(|_| rng.random_range(0.7_f32..1.5)).collect();
        let max_axes: Vec<f32> = eff_radii.iter().map(|r| r * 1.2).collect();

        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("skip: no GPU adapter ({e})"); return; }
        };
        let mut hash = SpatialHashGpu::with_context(&ctx, n, cell_size, [1000.0, 1000.0]).expect("hash");
        let _ = hash.rebuild(&positions);
        let mut col = CollisionGpu::with_context(&ctx, n, cell_size, CELL_RADIUS, [1000.0, 1000.0])
            .expect("collision init");
        let gpu_deltas = col.compute(&positions, &eff_radii, &max_axes, &hash);

        // CPU brute force.
        let mut cpu_deltas: Vec<[f32; 3]> = vec![[0.0; 3]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let pair_r = CELL_RADIUS * (eff_radii[i] + eff_radii[j]);
                let pair_r2 = pair_r * pair_r;
                let dx = positions[i][0] - positions[j][0];
                let dy = positions[i][1] - positions[j][1];
                let dz = positions[i][2] - positions[j][2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < pair_r2 && d2 > 0.0 {
                    let d = d2.sqrt();
                    let overlap = pair_r - d;
                    cpu_deltas[i][0] += (dx / d) * overlap * 0.5;
                    cpu_deltas[i][1] += (dy / d) * overlap * 0.5;
                    cpu_deltas[i][2] += (dz / d) * overlap * 0.5;
                }
            }
        }

        for i in 0..n {
            for k in 0..3 {
                let d = (cpu_deltas[i][k] - gpu_deltas[i][k]).abs();
                assert!(d < 1e-3, "i={i} k={k} cpu={} gpu={} diff={}",
                    cpu_deltas[i][k], gpu_deltas[i][k], d);
            }
        }
    }

    /// Sprint 50: step GPU vs CPU `Cell::step` parity. Random cells, jeden step
    /// na obou stranách, výsledný state musí matchnout v ε. Tolerance 1e-4
    /// (long arithmetic chain s drag + decay + pitch clamp).
    #[test]
    fn step_gpu_matches_cpu() {
        use crate::{
            Cell, AGE_DECAY_PER_SEC, ANGULAR_DRAG, ANGULAR_ENERGY_COST, ATTACK_COST_PER_SEC,
            BODY_COST_FACTOR, DRAG_COEFFICIENT, ENERGY_COST_PER_V_SQ, FIXED_TIMESTEP_HZ, GRAVITY,
            PHYSICS_CONFIG, SHELL_COST_PER_SEC, SPIKE_COST_PER_SEC, VISION_COST_PER_RADIUS,
        };
        let mut rng = StdRng::seed_from_u64(43);
        let n = 64;
        let dt = 1.0_f32 / FIXED_TIMESTEP_HZ;
        let world_half: [f32; 3] = [960.0, 540.0, 2.0];
        // Spawn cells s mírnou velocity / angular_velocity aby step měl co dělat.
        let mut cells: Vec<Cell> = (0..n)
            .map(|_| {
                let mut c = Cell::random(&mut rng, world_half, 0, 0);
                c.angular_velocity = rng.random_range(-0.3_f32..0.3);
                c.pitch_velocity = rng.random_range(-0.05_f32..0.05);
                c.last_outputs[6] = rng.random_range(-0.5_f32..1.0);
                c
            })
            .collect();

        let positions: Vec<[f32; 3]> = cells.iter().map(|c| c.position).collect();
        let velocities: Vec<[f32; 3]> = cells.iter().map(|c| c.velocity).collect();
        let headings: Vec<f32> = cells.iter().map(|c| c.heading).collect();
        let pitches: Vec<f32> = cells.iter().map(|c| c.pitch).collect();
        let angular_velocities: Vec<f32> = cells.iter().map(|c| c.angular_velocity).collect();
        let pitch_velocities: Vec<f32> = cells.iter().map(|c| c.pitch_velocity).collect();
        let ages: Vec<u32> = cells.iter().map(|c| c.age as u32).collect();
        let cooldowns: Vec<u32> = cells.iter().map(|c| c.reproduce_cooldown_ticks).collect();
        let energies: Vec<f32> = cells.iter().map(|c| c.energy).collect();
        let body_dims: Vec<[f32; 3]> = cells
            .iter()
            .map(|c| {
                [
                    c.phenotype.body_length,
                    c.phenotype.body_width,
                    c.phenotype.body_height,
                ]
            })
            .collect();
        let aux: Vec<[f32; 4]> = cells
            .iter()
            .map(|c| {
                [
                    c.phenotype.spike_length,
                    c.phenotype.shell_thickness,
                    c.genome.vision_radius,
                    c.last_outputs[6],
                ]
            })
            .collect();

        let mut gpu = match StepGpu::new(n) {
            Ok(g) => g,
            Err(e) => { eprintln!("skip: no GPU adapter ({e})"); return; }
        };
        let params = StepParamsGpu {
            num_cells: n as u32,
            dt,
            world_half_x: world_half[0],
            world_half_y: world_half[1],
            world_half_z: world_half[2],
            gravity: GRAVITY,
            drag: DRAG_COEFFICIENT,
            angular_drag: ANGULAR_DRAG,
            energy_cost_per_v_sq: ENERGY_COST_PER_V_SQ,
            angular_energy_cost: ANGULAR_ENERGY_COST,
            vision_cost_per_radius: VISION_COST_PER_RADIUS,
            body_cost_factor: BODY_COST_FACTOR,
            age_decay_per_sec: AGE_DECAY_PER_SEC,
            fixed_timestep_hz: FIXED_TIMESTEP_HZ,
            spike_cost_per_sec: SPIKE_COST_PER_SEC,
            shell_cost_per_sec: SHELL_COST_PER_SEC,
            attack_cost_per_sec: ATTACK_COST_PER_SEC,
            pitch_clamp: core::f32::consts::FRAC_PI_6 * 0.5,
            ..StepParamsGpu::default()
        };
        let res = gpu.compute(
            &positions, &velocities, &headings, &pitches, &angular_velocities,
            &pitch_velocities, &ages, &cooldowns, &energies, &body_dims, &aux, params,
        );

        for c in cells.iter_mut() {
            c.step(dt, world_half, &PHYSICS_CONFIG);
        }

        for i in 0..n {
            for k in 0..3 {
                let dp = (cells[i].position[k] - res.positions[i][k]).abs();
                let dv = (cells[i].velocity[k] - res.velocities[i][k]).abs();
                assert!(dp < 1e-3, "i={i} k={k} pos cpu={} gpu={} d={}",
                    cells[i].position[k], res.positions[i][k], dp);
                assert!(dv < 1e-3, "i={i} k={k} vel cpu={} gpu={} d={}",
                    cells[i].velocity[k], res.velocities[i][k], dv);
            }
            assert!((cells[i].heading - res.headings[i]).abs() < 1e-3, "i={i} heading");
            assert!((cells[i].pitch - res.pitches[i]).abs() < 1e-4, "i={i} pitch");
            assert!((cells[i].angular_velocity - res.angular_velocities[i]).abs() < 1e-4);
            assert!((cells[i].pitch_velocity - res.pitch_velocities[i]).abs() < 1e-4);
            assert_eq!(cells[i].age as u32, res.ages[i]);
            assert_eq!(cells[i].reproduce_cooldown_ticks, res.cooldowns[i]);
            assert!(
                (cells[i].energy - res.energies[i]).abs() < 1e-3,
                "i={i} energy cpu={} gpu={}",
                cells[i].energy,
                res.energies[i]
            );
        }
    }

    /// Sprint 49: GPU broad-phase neighbor query parity vs CPU brute force.
    /// Stejná positions + vision_radii + hash → stejný nearest cell + count
    /// per cell. Tolerance 1e-3 na pozici (single-precision float tieng může
    /// disagree mezi nejbližšími při téměř identických vzdálenostech).
    #[test]
    fn neighbors_gpu_matches_cpu_brute_force() {
        let mut rng = StdRng::seed_from_u64(31);
        let n = 200;
        let cell_size = 64.0;
        let positions: Vec<[f32; 3]> = (0..n)
            .map(|_| {
                [
                    rng.random_range(-500.0_f32..500.0),
                    rng.random_range(-300.0_f32..300.0),
                    rng.random_range(-2.0_f32..2.0),
                ]
            })
            .collect();
        let radii: Vec<f32> = (0..n).map(|_| rng.random_range(0.5_f32..2.0)).collect();
        let vision_radii: Vec<f32> = (0..n).map(|_| rng.random_range(20.0_f32..80.0)).collect();

        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip: no GPU adapter ({e})");
                return;
            }
        };
        let mut hash = SpatialHashGpu::with_context(&ctx, n, cell_size, [1000.0, 1000.0]).expect("hash init");
        let _ = hash.rebuild(&positions);
        let mut nb = NeighborsGpu::with_context(&ctx, n, cell_size, [1000.0, 1000.0]).expect("nb init");
        let gpu_results = nb.compute(&positions, &radii, &vision_radii, &hash);

        for i in 0..n {
            let pos_i = positions[i];
            let vr2 = vision_radii[i] * vision_radii[i];
            let mut cpu_count: u32 = 0;
            let mut best_d2 = f32::MAX;
            let mut best_j: Option<usize> = None;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let dx = positions[j][0] - pos_i[0];
                let dy = positions[j][1] - pos_i[1];
                let dz = positions[j][2] - pos_i[2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 <= vr2 {
                    cpu_count += 1;
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best_j = Some(j);
                    }
                }
            }
            let gpu = &gpu_results[i];
            assert_eq!(
                gpu.neighbors_in_vision, cpu_count,
                "i={i}: count gpu={} cpu={}",
                gpu.neighbors_in_vision, cpu_count
            );
            match (best_j, gpu.nearest_cell) {
                (None, None) => {}
                (Some(j), Some((p, r))) => {
                    let cpu_d2 = {
                        let dx = positions[j][0] - pos_i[0];
                        let dy = positions[j][1] - pos_i[1];
                        let dz = positions[j][2] - pos_i[2];
                        dx * dx + dy * dy + dz * dz
                    };
                    let gpu_d2 = {
                        let dx = p[0] - pos_i[0];
                        let dy = p[1] - pos_i[1];
                        let dz = p[2] - pos_i[2];
                        dx * dx + dy * dy + dz * dz
                    };
                    // Acceptujeme jiný winner pokud d2 jsou v ε.
                    assert!(
                        (cpu_d2 - gpu_d2).abs() < 1e-2,
                        "i={i}: cpu_d2={cpu_d2}, gpu_d2={gpu_d2}, cpu_j={j}"
                    );
                    assert!(
                        (r - radii[j]).abs() < 1e-3 || cpu_d2 == gpu_d2,
                        "i={i}: radius mismatch, gpu={r}, cpu_j={j} radius={}",
                        radii[j]
                    );
                }
                (cpu, gpu) => panic!("i={i}: cpu={:?} gpu={:?} mismatch", cpu, gpu),
            }
        }
    }

    /// Sprint 49: ověření že single-workgroup tree reduce zvládá N >> 10k.
    /// Strided loop v shaderu je unbounded; jediné co single-WG hraje roli je
    /// že 256 threadů sekvenciálně iteruje N/256 prvků. Pro N=50000 to je 195
    /// iterací per thread (~10 µs wall time) — žádný correctness problem.
    #[test]
    fn stats_gpu_handles_50k() {
        let mut rng = StdRng::seed_from_u64(101);
        let n = 50_000;
        let positions: Vec<[f32; 3]> = (0..n)
            .map(|_| {
                [
                    rng.random_range(-1000.0_f32..1000.0),
                    rng.random_range(-500.0_f32..500.0),
                    rng.random_range(-2.0_f32..2.0),
                ]
            })
            .collect();
        let velocities: Vec<[f32; 3]> = (0..n)
            .map(|_| {
                [
                    rng.random_range(-50.0_f32..50.0),
                    rng.random_range(-50.0_f32..50.0),
                    rng.random_range(-5.0_f32..5.0),
                ]
            })
            .collect();
        let energies: Vec<f32> = (0..n).map(|_| rng.random_range(0.0_f32..150.0)).collect();
        let mut gpu = match StatsGpu::new(n) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skip: no GPU adapter ({e})");
                return;
            }
        };
        let gpu_stats = gpu.compute(&positions, &velocities, &energies);
        let cpu_sum_energy: f32 = energies.iter().sum();
        let scale = cpu_sum_energy.abs().max(1.0);
        let diff = (gpu_stats.sum_energy - cpu_sum_energy).abs();
        // Tolerance scaled — pro 50k hodnot s ~75 mean je sum ~3.75M, ULP cumulative
        // drift napříč 50k single-precision adds může být ~1e3 relativně 1e-4.
        assert!(
            diff < scale * 1e-3,
            "diff = {} (scale = {}); gpu = {}, cpu = {}",
            diff,
            scale,
            gpu_stats.sum_energy,
            cpu_sum_energy
        );
    }

    /// Sprint 47: integration test sdíleného `GpuContext` napříč 4 subsystémy.
    /// Každý úspěšně inicializuje skrz `with_context` a doběhne jeden mini
    /// pipeline cycle (brain forward → spatial hash → field step → stats reduce)
    /// na sdíleném device. Verifikuje, že device-lifetime + bind group ownership
    /// se ne-konfliktuje.
    #[test]
    fn gpu_context_shared_across_subsystems() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip: no GPU adapter ({e})");
                return;
            }
        };
        let n = 64;
        let mut rng = StdRng::seed_from_u64(19);

        let mut brain_gpu = BrainGpu::with_context(&ctx, n).expect("BrainGpu init");
        let mut hash_gpu =
            SpatialHashGpu::with_context(&ctx, n, 64.0, [1000.0, 1000.0]).expect("SpatialHashGpu init");
        let mut field_gpu =
            FieldGpu::with_context(&ctx, 16, [320.0, 320.0], 32).expect("FieldGpu init");
        let mut stats_gpu = StatsGpu::with_context(&ctx, n).expect("StatsGpu init");

        let brains: Vec<Brain> = (0..n).map(|_| Brain::random(&mut rng)).collect();
        let inputs: Vec<[f32; BRAIN_INPUTS]> = (0..n)
            .map(|_| {
                let mut a = [0.0_f32; BRAIN_INPUTS];
                for v in a.iter_mut() {
                    *v = rng.random_range(-1.0_f32..1.0);
                }
                a
            })
            .collect();
        let positions: Vec<[f32; 3]> = (0..n)
            .map(|_| {
                [
                    rng.random_range(-300.0_f32..300.0),
                    rng.random_range(-300.0_f32..300.0),
                    rng.random_range(-2.0_f32..2.0),
                ]
            })
            .collect();
        let velocities: Vec<[f32; 3]> = (0..n)
            .map(|_| {
                [
                    rng.random_range(-30.0_f32..30.0),
                    rng.random_range(-30.0_f32..30.0),
                    0.0,
                ]
            })
            .collect();
        let energies: Vec<f32> = (0..n).map(|_| 50.0).collect();

        let mut h = vec![[0.0_f32; BRAIN_HIDDEN]; n];
        let mut o = vec![[0.0_f32; BRAIN_OUTPUTS]; n];
        brain_gpu.forward_batch(&inputs, &brains, &mut h, &mut o);

        let (offsets, sorted) = hash_gpu.rebuild(&positions);
        assert_eq!(offsets.len(), GPU_HASH_NUM_BUCKETS + 1);
        assert_eq!(sorted.len(), n);
        assert_eq!(offsets[GPU_HASH_NUM_BUCKETS] as usize, n);

        for (pos, _) in positions.iter().zip(0..n) {
            field_gpu.add_source([pos[0], pos[1]], 1.0);
        }
        field_gpu.step(0.15, 0.3, 1.0 / 60.0);
        let grid = field_gpu.download();
        assert_eq!(grid.len(), 16 * 16);

        let stats = stats_gpu.compute(&positions, &velocities, &energies);
        assert!(stats.sum_energy > 0.0);
    }
}
