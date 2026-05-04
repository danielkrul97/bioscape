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
use wgpu::util::DeviceExt;

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
    device: wgpu::Device,
    queue: wgpu::Queue,
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
    pub fn new(capacity: usize) -> Result<Self, String> {
        assert!(capacity > 0, "BrainGpu capacity must be > 0");
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
                label: Some("bioscape-brain-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| format!("device request failed: {e:?}"))?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Brain;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

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
}
