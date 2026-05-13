use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::*;
use super::*;

/// Per-child substrate query count: 45×72 + 45 + 12×45 + 12 = 3837.
pub const CPPN_QUERIES_PER_CHILD: u32 =
    (BRAIN_HIDDEN * BRAIN_INPUTS + BRAIN_HIDDEN + BRAIN_OUTPUTS * BRAIN_HIDDEN + BRAIN_OUTPUTS)
        as u32;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct CppnFromCppnParams {
    num_children: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct CppnNodePacked {
    id: u32,
    activation: u32,
    bias: f32,
    layer: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct CppnLinkPacked {
    from_id: u32,
    to_id: u32,
    weight: f32,
    enabled: u32,
}

fn activation_code(a: ActivationFn) -> u32 {
    match a {
        ActivationFn::Linear => 0,
        ActivationFn::Sigmoid => 1,
        ActivationFn::Tanh => 2,
        ActivationFn::Gaussian => 3,
        ActivationFn::Sine => 4,
        ActivationFn::Abs => 5,
        ActivationFn::Step => 6,
    }
}

/// GPU mirror of `Brain::from_cppn`. Materialises per-child brain weights
/// directly into `CellsGpu::brain_weights_buf` at the given slot offsets,
/// skipping the per-child `upload_brain_at` round-trip the CPU path would
/// otherwise need. One dispatch covers all children produced in one
/// reproduction phase.
/// CSR offsets per cell: `CPPN_MAX_NODES + 1` u32 slots. `link_offsets[id]`
/// = inclusive start index in the cell's `cppn_links` slice; `link_offsets[id + 1]`
/// = end. Lets the shader walk only links incoming to a given node instead of
/// scanning all 256 slots per node × per layer pass.
const CPPN_LINK_OFFSETS_PER_CELL: usize = CPPN_MAX_NODES + 1;

pub struct CppnGpu {
    #[allow(dead_code)]
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    capacity: usize,
    params_buf: wgpu::Buffer,
    meta_buf: wgpu::Buffer,
    nodes_buf: wgpu::Buffer,
    links_buf: wgpu::Buffer,
    link_offsets_buf: wgpu::Buffer,
    slots_buf: wgpu::Buffer,
    cached_bg: Option<wgpu::BindGroup>,
    cached_cells_epoch: u64,

    // CPU scratch — preserved across dispatches to skip per-tick allocs.
    meta_scratch: Vec<[u32; 4]>,
    nodes_scratch: Vec<CppnNodePacked>,
    links_scratch: Vec<CppnLinkPacked>,
    link_offsets_scratch: Vec<u32>,
    slots_scratch: Vec<u32>,
    sort_scratch: Vec<CppnLinkPacked>,
}

impl CppnGpu {
    pub fn with_context(ctx: &GpuContext, capacity: usize) -> Self {
        Self::with_device_inner(Arc::clone(&ctx.device), Arc::clone(&ctx.queue), capacity)
    }

    fn with_device_inner(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        capacity: usize,
    ) -> Self {
        assert!(capacity > 0);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cppn-from-cppn"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/cppn_from_cppn.wgsl").into(),
            ),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..7)
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
            label: Some("cppn-from-cppn-bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cppn-from-cppn-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cppn-from-cppn-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cppn_from_cppn"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cppn-from-cppn-params"),
            contents: bytemuck::bytes_of(&CppnFromCppnParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let cap = capacity as u64;
        let mk = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let meta_size = cap * 16; // vec4<u32>
        let nodes_size = cap * (CPPN_MAX_NODES as u64) * 16; // CppnNode = 16 bytes
        let links_size = cap * (CPPN_MAX_LINKS as u64) * 16; // CppnLink = 16 bytes
        let link_offsets_size = cap * (CPPN_LINK_OFFSETS_PER_CELL as u64) * 4;
        let slots_size = cap * 4;
        let meta_buf = mk("cppn-meta", meta_size);
        let nodes_buf = mk("cppn-nodes", nodes_size);
        let links_buf = mk("cppn-links", links_size);
        let link_offsets_buf = mk("cppn-link-offsets", link_offsets_size);
        let slots_buf = mk("cppn-slots", slots_size);

        Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            capacity,
            params_buf,
            meta_buf,
            nodes_buf,
            links_buf,
            link_offsets_buf,
            slots_buf,
            cached_bg: None,
            cached_cells_epoch: 0,
            meta_scratch: Vec::with_capacity(capacity),
            nodes_scratch: Vec::with_capacity(capacity * CPPN_MAX_NODES),
            links_scratch: Vec::with_capacity(capacity * CPPN_MAX_LINKS),
            link_offsets_scratch: Vec::with_capacity(capacity * CPPN_LINK_OFFSETS_PER_CELL),
            slots_scratch: Vec::with_capacity(capacity),
            sort_scratch: Vec::with_capacity(CPPN_MAX_LINKS),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn pack(&mut self, children: &[(usize, &Cppn)]) {
        self.meta_scratch.clear();
        self.nodes_scratch.clear();
        self.links_scratch.clear();
        self.link_offsets_scratch.clear();
        self.slots_scratch.clear();
        for &(slot, cppn) in children {
            self.slots_scratch.push(slot as u32);
            self.meta_scratch
                .push([cppn.num_nodes as u32, cppn.num_links as u32, 0, 0]);
            // Pack nodes 0..CPPN_MAX_NODES; unused slots stay zero (caller
            // never reads past `num_nodes`, so zeros are harmless).
            let node_start = self.nodes_scratch.len();
            self.nodes_scratch
                .resize(node_start + CPPN_MAX_NODES, CppnNodePacked::default());
            for (i, slot_node) in cppn.nodes.iter().enumerate() {
                if let Some(n) = slot_node {
                    self.nodes_scratch[node_start + i] = CppnNodePacked {
                        id: n.id,
                        activation: activation_code(n.activation),
                        bias: n.bias,
                        layer: n.layer,
                    };
                }
            }
            // CSR pack: collect valid links, sort by `to_id`, then write a
            // sorted run + per-target offsets so the shader walks only the
            // incoming-edge slice for each node instead of all 256 slots.
            self.sort_scratch.clear();
            for slot_link in cppn.links.iter().take(cppn.num_links as usize) {
                if let Some(l) = slot_link {
                    self.sort_scratch.push(CppnLinkPacked {
                        from_id: l.from,
                        to_id: l.to,
                        weight: l.weight,
                        // `enabled = 0` makes the shader skip the link, so
                        // disabled links don't need pre-filtering.
                        enabled: if l.enabled { 1 } else { 0 },
                    });
                }
            }
            self.sort_scratch.sort_by_key(|l| l.to_id);
            let link_start = self.links_scratch.len();
            self.links_scratch
                .resize(link_start + CPPN_MAX_LINKS, CppnLinkPacked::default());
            for (i, l) in self.sort_scratch.iter().enumerate() {
                self.links_scratch[link_start + i] = *l;
            }
            // Build per-target counts in offsets[id + 1], then prefix-sum.
            let offsets_start = self.link_offsets_scratch.len();
            self.link_offsets_scratch
                .resize(offsets_start + CPPN_LINK_OFFSETS_PER_CELL, 0u32);
            for l in &self.sort_scratch {
                if (l.to_id as usize) < CPPN_MAX_NODES {
                    self.link_offsets_scratch[offsets_start + l.to_id as usize + 1] += 1;
                }
            }
            for i in 1..CPPN_LINK_OFFSETS_PER_CELL {
                self.link_offsets_scratch[offsets_start + i] +=
                    self.link_offsets_scratch[offsets_start + i - 1];
            }
        }
    }

    fn ensure_bind_group(&mut self, cells: &CellsGpu) {
        let epoch = cells.epoch();
        if self.cached_bg.is_some() && self.cached_cells_epoch == epoch {
            return;
        }
        self.cached_bg = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cppn-from-cppn-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.meta_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.nodes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.links_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.slots_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cells.brain_weights_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.link_offsets_buf.as_entire_binding(),
                },
            ],
        }));
        self.cached_cells_epoch = epoch;
    }

    /// Dispatch `from_cppn` for the given children, writing brain weights
    /// directly into `cells.brain_weights_buf[slot * WEIGHTS_PER_CELL …]`.
    /// `children` is a slice of `(slot, &cppn)` pairs.
    pub fn dispatch_into(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        children: &[(usize, &Cppn)],
        cells: &CellsGpu,
    ) {
        if children.is_empty() {
            return;
        }
        assert!(children.len() <= self.capacity);
        self.pack(children);
        let n = children.len() as u32;
        let params = CppnFromCppnParams {
            num_children: n,
            ..Default::default()
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&self.meta_buf, 0, bytemuck::cast_slice(&self.meta_scratch));
        self.queue
            .write_buffer(&self.nodes_buf, 0, bytemuck::cast_slice(&self.nodes_scratch));
        self.queue
            .write_buffer(&self.links_buf, 0, bytemuck::cast_slice(&self.links_scratch));
        self.queue.write_buffer(
            &self.link_offsets_buf,
            0,
            bytemuck::cast_slice(&self.link_offsets_scratch),
        );
        self.queue
            .write_buffer(&self.slots_buf, 0, bytemuck::cast_slice(&self.slots_scratch));

        self.ensure_bind_group(cells);
        let bg = self.cached_bg.as_ref().unwrap();
        let total_threads = n * CPPN_QUERIES_PER_CHILD;
        let workgroups = (total_threads + 63) / 64;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cppn-from-cppn-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Convenience wrapper: build encoder, dispatch, submit.
    pub fn dispatch(&mut self, children: &[(usize, &Cppn)], cells: &CellsGpu) {
        if children.is_empty() {
            return;
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cppn-from-cppn-encoder"),
            });
        self.dispatch_into(&mut encoder, children, cells);
        self.queue.submit(Some(encoder.finish()));
    }
}
