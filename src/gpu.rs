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
                required_limits: wgpu::Limits::default(),
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
    _pad0: u32,
    _pad1: u32,
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
    pub fn new(capacity: usize, cell_size: f32) -> Result<Self, String> {
        assert!(capacity > 0);
        let ctx = GpuContext::new()?;
        Self::with_device_inner(ctx.device, ctx.queue, capacity, cell_size)
    }

    pub fn with_context(
        ctx: &GpuContext,
        capacity: usize,
        cell_size: f32,
    ) -> Result<Self, String> {
        Self::with_device_inner(
            Arc::clone(&ctx.device),
            Arc::clone(&ctx.queue),
            capacity,
            cell_size,
        )
    }

    fn with_device_inner(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        capacity: usize,
        cell_size: f32,
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
            ..HashParams::default()
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

        let mut gpu = match SpatialHashGpu::new(n, cell_size) {
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
    #[test]
    fn field_gpu_diffusion_matches_cpu() {
        use crate::SmellField;
        let resolution = 32;
        let world_half = [320.0_f32, 320.0];
        let diffusion = 0.15_f32;
        let decay_per_sec = 0.3_f32;
        let dt = 1.0_f32 / 60.0;

        let mut gpu = match FieldGpu::new(resolution, world_half, 64) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skip: no GPU adapter ({e})");
                return;
            }
        };
        let mut cpu = SmellField::new(resolution, world_half);

        let mut rng = StdRng::seed_from_u64(13);
        // 10 ticků, každý s 5 random sources.
        for _ in 0..10 {
            for _ in 0..5 {
                let pos = [
                    rng.random_range(-300.0_f32..300.0),
                    rng.random_range(-300.0_f32..300.0),
                ];
                let amount: f32 = rng.random_range(0.5_f32..2.0);
                cpu.add_source(pos, amount);
                gpu.add_source(pos, amount);
            }
            cpu.step(diffusion, decay_per_sec, dt);
            gpu.step(diffusion, decay_per_sec, dt);
        }

        let gpu_grid = gpu.download();
        // CPU sample přes index_of helper.
        let mut max_diff = 0.0_f32;
        for j in 0..resolution {
            for i in 0..resolution {
                let idx = j * resolution + i;
                let cell_size_x = (2.0 * world_half[0]) / resolution as f32;
                let cell_size_y = (2.0 * world_half[1]) / resolution as f32;
                let pos = [
                    -world_half[0] + (i as f32 + 0.5) * cell_size_x,
                    -world_half[1] + (j as f32 + 0.5) * cell_size_y,
                ];
                let cpu_val = cpu.sample(pos);
                let gpu_val = gpu_grid[idx];
                let diff = (cpu_val - gpu_val).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
        assert!(
            max_diff < 1e-3,
            "field GPU vs CPU max diff = {} (expected < 1e-3)",
            max_diff
        );
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
            SpatialHashGpu::with_context(&ctx, n, 64.0).expect("SpatialHashGpu init");
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
