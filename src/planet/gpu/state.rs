//! `PlanetGpu` — unified GPU state for the leapfrog hot loop.
//!
//! Owns the shared particle data buffers (positions, velocities,
//! accelerations, masses) and the three pipelines that operate on
//! them (nbody, kick, drift). Each pipeline has its own uniform
//! params buffer + bind group, but all bind groups reference the
//! same data buffers.
//!
//! Hot loop per tick (KDK leapfrog):
//!   1. kick (dt/2, a_old)
//!   2. drift (dt)
//!   3. nbody → new accelerations
//!   4. kick (dt/2, a_new)
//!
//! All four dispatches are recorded into one command encoder and
//! submitted with a single `queue.submit`. Readback only happens
//! when the caller asks for it (`download_state`).

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

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct StepParams {
    num_particles: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    dt: f32,
    pad_a0: f32,
    pad_a1: f32,
    pad_a2: f32,
}

pub struct PlanetGpu {
    pub ctx: Arc<GpuContext>,
    pub capacity: usize,

    positions_buf: wgpu::Buffer,
    velocities_buf: wgpu::Buffer,
    accelerations_buf: wgpu::Buffer,
    masses_buf: wgpu::Buffer,
    smoothing_lengths_buf: wgpu::Buffer,
    densities_buf: wgpu::Buffer,

    positions_rb: wgpu::Buffer,
    velocities_rb: wgpu::Buffer,
    accelerations_rb: wgpu::Buffer,
    smoothing_lengths_rb: wgpu::Buffer,
    densities_rb: wgpu::Buffer,

    nbody_pipeline: wgpu::ComputePipeline,
    nbody_params_buf: wgpu::Buffer,
    nbody_bg: wgpu::BindGroup,

    kick_pipeline: wgpu::ComputePipeline,
    kick_params_buf: wgpu::Buffer,
    kick_bg: wgpu::BindGroup,

    drift_pipeline: wgpu::ComputePipeline,
    drift_params_buf: wgpu::Buffer,
    drift_bg: wgpu::BindGroup,
}

impl PlanetGpu {
    pub fn new(ctx: Arc<GpuContext>, capacity: usize) -> Result<Self, String> {
        let device = &ctx.device;
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

        let positions_buf = mk("planet-positions", n * 3 * f, stor);
        let velocities_buf = mk("planet-velocities", n * 3 * f, stor);
        let accelerations_buf = mk("planet-accelerations", n * 3 * f, stor);
        let masses_buf = mk("planet-masses", n * f, stor);
        let smoothing_lengths_buf = mk("planet-h", n * f, stor);
        let densities_buf = mk("planet-rho", n * f, stor);
        let positions_rb = mk("planet-positions-rb", n * 3 * f, read);
        let velocities_rb = mk("planet-velocities-rb", n * 3 * f, read);
        let accelerations_rb = mk("planet-accelerations-rb", n * 3 * f, read);
        let smoothing_lengths_rb = mk("planet-h-rb", n * f, read);
        let densities_rb = mk("planet-rho-rb", n * f, read);

        // NBody pipeline
        let nbody_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planet-nbody"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/planet_nbody.wgsl").into(),
            ),
        });
        let nbody_layout = make_layout(device, "planet-nbody-bgl", &[false, true, true, false]);
        let (nbody_pipeline, _) = build_pipeline(device, "planet-nbody", &nbody_shader, "nbody", &nbody_layout);
        let nbody_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet-nbody-params"),
            contents: bytemuck::bytes_of(&NBodyParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let nbody_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet-nbody-bg"),
            layout: &nbody_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: nbody_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: positions_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: masses_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: accelerations_buf.as_entire_binding() },
            ],
        });

        // Kick pipeline
        let kick_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planet-kick"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/planet_kick.wgsl").into(),
            ),
        });
        let kick_layout = make_layout(device, "planet-kick-bgl", &[false, false, true]);
        let (kick_pipeline, _) = build_pipeline(device, "planet-kick", &kick_shader, "kick", &kick_layout);
        let kick_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet-kick-params"),
            contents: bytemuck::bytes_of(&StepParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let kick_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet-kick-bg"),
            layout: &kick_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: kick_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: velocities_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: accelerations_buf.as_entire_binding() },
            ],
        });

        // Drift pipeline
        let drift_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planet-drift"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/planet_drift.wgsl").into(),
            ),
        });
        let drift_layout = make_layout(device, "planet-drift-bgl", &[false, false, true]);
        let (drift_pipeline, _) = build_pipeline(device, "planet-drift", &drift_shader, "drift", &drift_layout);
        let drift_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet-drift-params"),
            contents: bytemuck::bytes_of(&StepParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let drift_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet-drift-bg"),
            layout: &drift_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: drift_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: positions_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: velocities_buf.as_entire_binding() },
            ],
        });

        Ok(Self {
            ctx,
            capacity,
            positions_buf,
            velocities_buf,
            accelerations_buf,
            masses_buf,
            smoothing_lengths_buf,
            densities_buf,
            positions_rb,
            velocities_rb,
            accelerations_rb,
            smoothing_lengths_rb,
            densities_rb,
            nbody_pipeline,
            nbody_params_buf,
            nbody_bg,
            kick_pipeline,
            kick_params_buf,
            kick_bg,
            drift_pipeline,
            drift_params_buf,
            drift_bg,
        })
    }

    /// Upload initial particle state. Caller is responsible for the
    /// initial acceleration (call `compute_accelerations` once after
    /// this to seed `a_0`).
    pub fn upload_state(
        &self,
        positions: &[[f32; 3]],
        velocities: &[[f32; 3]],
        masses: &[f32],
    ) {
        let n = positions.len();
        assert!(n <= self.capacity);
        assert_eq!(velocities.len(), n);
        assert_eq!(masses.len(), n);
        if n == 0 {
            return;
        }
        let pos: Vec<f32> = positions.iter().flatten().copied().collect();
        let vel: Vec<f32> = velocities.iter().flatten().copied().collect();
        self.ctx
            .queue
            .write_buffer(&self.positions_buf, 0, bytemuck::cast_slice(&pos));
        self.ctx
            .queue
            .write_buffer(&self.velocities_buf, 0, bytemuck::cast_slice(&vel));
        self.ctx
            .queue
            .write_buffer(&self.masses_buf, 0, bytemuck::cast_slice(masses));
    }

    /// One-shot nbody dispatch — computes accelerations from current
    /// positions/masses. Used for `a_0` seeding and the S206-style
    /// correctness test.
    pub fn compute_accelerations(&self, n: usize, g: f32, softening: f32) {
        if n == 0 {
            return;
        }
        let params = NBodyParams {
            num_particles: n as u32,
            g,
            eps2: softening * softening,
            ..NBodyParams::default()
        };
        self.ctx
            .queue
            .write_buffer(&self.nbody_params_buf, 0, bytemuck::bytes_of(&params));
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-accel-encoder"),
            });
        self.encode_nbody(&mut encoder, n);
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    /// Half-kick dispatch — one of the two `v += dt/2 · a` updates in
    /// a KDK leapfrog. Owns its own encoder + submit; for SPH the
    /// caller invokes `kick → drift → density → nbody → pressure →
    /// viscosity → kick` directly so each force pass can refresh the
    /// shared accelerations buffer.
    pub fn kick(&self, n: usize, dt_half: f32) {
        if n == 0 {
            return;
        }
        let params = StepParams {
            num_particles: n as u32,
            dt: dt_half,
            ..StepParams::default()
        };
        self.ctx
            .queue
            .write_buffer(&self.kick_params_buf, 0, bytemuck::bytes_of(&params));
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-kick-encoder"),
            });
        self.encode_kick(&mut encoder, n);
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    /// Drift dispatch — `x += dt · v`.
    pub fn drift(&self, n: usize, dt: f32) {
        if n == 0 {
            return;
        }
        let params = StepParams {
            num_particles: n as u32,
            dt,
            ..StepParams::default()
        };
        self.ctx
            .queue
            .write_buffer(&self.drift_params_buf, 0, bytemuck::bytes_of(&params));
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-drift-encoder"),
            });
        self.encode_drift(&mut encoder, n);
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    /// Full KDK leapfrog step on the GPU. Records all four dispatches
    /// into a single command encoder and submits.
    pub fn step_leapfrog(&self, n: usize, dt: f32, g: f32, softening: f32) {
        if n == 0 {
            return;
        }
        let kick_params = StepParams {
            num_particles: n as u32,
            dt: 0.5 * dt,
            ..StepParams::default()
        };
        let drift_params = StepParams {
            num_particles: n as u32,
            dt,
            ..StepParams::default()
        };
        let nbody_params = NBodyParams {
            num_particles: n as u32,
            g,
            eps2: softening * softening,
            ..NBodyParams::default()
        };
        let q = &self.ctx.queue;
        q.write_buffer(&self.kick_params_buf, 0, bytemuck::bytes_of(&kick_params));
        q.write_buffer(&self.drift_params_buf, 0, bytemuck::bytes_of(&drift_params));
        q.write_buffer(&self.nbody_params_buf, 0, bytemuck::bytes_of(&nbody_params));

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-step-encoder"),
            });
        self.encode_kick(&mut encoder, n);
        self.encode_drift(&mut encoder, n);
        self.encode_nbody(&mut encoder, n);
        self.encode_kick(&mut encoder, n);
        q.submit(Some(encoder.finish()));
    }

    pub fn positions_buffer(&self) -> &wgpu::Buffer {
        &self.positions_buf
    }
    pub fn velocities_buffer(&self) -> &wgpu::Buffer {
        &self.velocities_buf
    }
    pub fn accelerations_buffer(&self) -> &wgpu::Buffer {
        &self.accelerations_buf
    }
    pub fn masses_buffer(&self) -> &wgpu::Buffer {
        &self.masses_buf
    }
    pub fn smoothing_lengths_buffer(&self) -> &wgpu::Buffer {
        &self.smoothing_lengths_buf
    }
    pub fn densities_buffer(&self) -> &wgpu::Buffer {
        &self.densities_buf
    }

    pub fn upload_smoothing_lengths(&self, h: &[f32]) {
        if h.is_empty() {
            return;
        }
        assert!(h.len() <= self.capacity);
        self.ctx
            .queue
            .write_buffer(&self.smoothing_lengths_buf, 0, bytemuck::cast_slice(h));
    }

    pub fn upload_densities(&self, rho: &[f32]) {
        if rho.is_empty() {
            return;
        }
        assert!(rho.len() <= self.capacity);
        self.ctx
            .queue
            .write_buffer(&self.densities_buf, 0, bytemuck::cast_slice(rho));
    }

    pub fn download_smoothing_lengths(&self, n: usize) -> Vec<f32> {
        self.download_f32(&self.smoothing_lengths_buf, &self.smoothing_lengths_rb, n)
    }

    pub fn download_densities(&self, n: usize) -> Vec<f32> {
        self.download_f32(&self.densities_buf, &self.densities_rb, n)
    }

    fn download_f32(&self, src: &wgpu::Buffer, rb: &wgpu::Buffer, n: usize) -> Vec<f32> {
        if n == 0 {
            return Vec::new();
        }
        let bytes = (n as u64) * 4;
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-f32-readback"),
            });
        encoder.copy_buffer_to_buffer(src, 0, rb, 0, bytes);
        self.ctx.queue.submit(Some(encoder.finish()));
        let slice = rb.slice(0..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.ctx.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        rb.unmap();
        out
    }

    fn encode_nbody(&self, encoder: &mut wgpu::CommandEncoder, n: usize) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("planet-nbody-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.nbody_pipeline);
        pass.set_bind_group(0, &self.nbody_bg, &[]);
        // nbody shader uses workgroup_size(128) — match the divisor.
        let wg = ((n as u32) + 127) / 128;
        pass.dispatch_workgroups(wg, 1, 1);
    }

    fn encode_kick(&self, encoder: &mut wgpu::CommandEncoder, n: usize) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("planet-kick-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.kick_pipeline);
        pass.set_bind_group(0, &self.kick_bg, &[]);
        let wg = ((n as u32) + 63) / 64;
        pass.dispatch_workgroups(wg, 1, 1);
    }

    fn encode_drift(&self, encoder: &mut wgpu::CommandEncoder, n: usize) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("planet-drift-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.drift_pipeline);
        pass.set_bind_group(0, &self.drift_bg, &[]);
        let wg = ((n as u32) + 63) / 64;
        pass.dispatch_workgroups(wg, 1, 1);
    }

    pub fn download_positions(&self, n: usize) -> Vec<[f32; 3]> {
        self.download_vec3(&self.positions_buf, &self.positions_rb, n)
    }

    pub fn download_velocities(&self, n: usize) -> Vec<[f32; 3]> {
        self.download_vec3(&self.velocities_buf, &self.velocities_rb, n)
    }

    pub fn download_accelerations(&self, n: usize) -> Vec<[f32; 3]> {
        self.download_vec3(&self.accelerations_buf, &self.accelerations_rb, n)
    }

    fn download_vec3(
        &self,
        src: &wgpu::Buffer,
        rb: &wgpu::Buffer,
        n: usize,
    ) -> Vec<[f32; 3]> {
        if n == 0 {
            return Vec::new();
        }
        let bytes = (n as u64) * 3 * std::mem::size_of::<f32>() as u64;
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet-readback"),
            });
        encoder.copy_buffer_to_buffer(src, 0, rb, 0, bytes);
        self.ctx.queue.submit(Some(encoder.finish()));
        let slice = rb.slice(0..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.ctx.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let f: &[f32] = bytemuck::cast_slice(&data);
        let out: Vec<[f32; 3]> = (0..n)
            .map(|i| [f[i * 3], f[i * 3 + 1], f[i * 3 + 2]])
            .collect();
        drop(data);
        rb.unmap();
        out
    }
}

fn make_layout(
    device: &wgpu::Device,
    label: &str,
    storage_read_only: &[bool],
) -> wgpu::BindGroupLayout {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = storage_read_only
        .iter()
        .enumerate()
        .map(|(i, &read_only)| {
            let ty = if i == 0 {
                wgpu::BufferBindingType::Uniform
            } else {
                wgpu::BufferBindingType::Storage { read_only }
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
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn build_pipeline(
    device: &wgpu::Device,
    label: &str,
    shader: &wgpu::ShaderModule,
    entry: &str,
    layout: &wgpu::BindGroupLayout,
) -> (wgpu::ComputePipeline, wgpu::PipelineLayout) {
    let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}-pl")),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl_layout),
        module: shader,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    (pipeline, pl_layout)
}
