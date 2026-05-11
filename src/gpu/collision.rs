use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::*;

// GPU collision pass — per-cell delta accumulation, chained off `SpatialHashGpu`.

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct CollisionParams {
    num_cells: u32,
    cell_size: f32,
    cell_radius_const: f32,
    collision_restitution: f32,
    world_half_x: f32,
    world_half_y: f32,
    _pad0: u32,
    _pad1: u32,
}

pub struct CollisionGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    capacity: usize,
    cell_size: f32,
    cell_radius_const: f32,
    collision_restitution: f32,
    world_half_xy: [f32; 2],
    params_buf: wgpu::Buffer,
    positions_buf: wgpu::Buffer,
    velocities_buf: wgpu::Buffer,
    eff_radii_buf: wgpu::Buffer,
    max_axes_buf: wgpu::Buffer,
    deltas_buf: wgpu::Buffer,
    vel_deltas_buf: wgpu::Buffer,
    deltas_rb: wgpu::Buffer,
    vel_deltas_rb: wgpu::Buffer,
    pos_packed: Vec<f32>,
    vel_packed: Vec<f32>,
}

impl CollisionGpu {
    pub fn with_context(
        ctx: &GpuContext,
        capacity: usize,
        cell_size: f32,
        cell_radius_const: f32,
        collision_restitution: f32,
        world_half_xy: [f32; 2],
    ) -> Result<Self, String> {
        Self::with_device_inner(
            Arc::clone(&ctx.device),
            Arc::clone(&ctx.queue),
            capacity,
            cell_size,
            cell_radius_const,
            collision_restitution,
            world_half_xy,
        )
    }

    pub fn new(
        capacity: usize,
        cell_size: f32,
        cell_radius_const: f32,
        collision_restitution: f32,
        world_half_xy: [f32; 2],
    ) -> Result<Self, String> {
        let ctx = GpuContext::new()?;
        Self::with_device_inner(
            ctx.device,
            ctx.queue,
            capacity,
            cell_size,
            cell_radius_const,
            collision_restitution,
            world_half_xy,
        )
    }

    fn with_device_inner(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        capacity: usize,
        cell_size: f32,
        cell_radius_const: f32,
        collision_restitution: f32,
        world_half_xy: [f32; 2],
    ) -> Result<Self, String> {
        assert!(capacity > 0);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("collision"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/collision.wgsl").into()),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..9)
            .map(|i| {
                let ty = if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else if i == 6 || i == 8 {
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
        let velocities_buf = mk("collision-velocities", n * 3 * f, stor_dst);
        let eff_radii_buf = mk("collision-eff-radii", n * f, stor_dst);
        let max_axes_buf = mk("collision-max-axes", n * f, stor_dst);
        let deltas_buf = mk("collision-deltas", n * 3 * f, stor_src);
        let vel_deltas_buf = mk("collision-vel-deltas", n * 3 * f, stor_src);
        let deltas_rb = mk("collision-deltas-rb", n * 3 * f, read);
        let vel_deltas_rb = mk("collision-vel-deltas-rb", n * 3 * f, read);

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            capacity,
            cell_size,
            cell_radius_const,
            collision_restitution,
            world_half_xy,
            params_buf,
            positions_buf,
            velocities_buf,
            eff_radii_buf,
            max_axes_buf,
            deltas_buf,
            vel_deltas_buf,
            deltas_rb,
            vel_deltas_rb,
            pos_packed: Vec::new(),
            vel_packed: Vec::new(),
        })
    }

    pub fn compute(
        &mut self,
        positions: &[[f32; 3]],
        velocities: &[[f32; 3]],
        eff_radii: &[f32],
        max_axes: &[f32],
        cell_hash: &SpatialHashGpu,
    ) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
        let n = positions.len();
        assert!(n <= self.capacity, "collision capacity overflow");
        assert_eq!(velocities.len(), n, "velocities length mismatch");
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
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
        let params = CollisionParams {
            num_cells: n as u32,
            cell_size: self.cell_size,
            cell_radius_const: self.cell_radius_const,
            collision_restitution: self.collision_restitution,
            world_half_x: self.world_half_xy[0],
            world_half_y: self.world_half_xy[1],
            _pad0: 0,
            _pad1: 0,
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
                wgpu::BindGroupEntry { binding: 7, resource: self.velocities_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.vel_deltas_buf.as_entire_binding() },
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
        encoder.copy_buffer_to_buffer(&self.vel_deltas_buf, 0, &self.vel_deltas_rb, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let pos_slice = self.deltas_rb.slice(0..bytes);
        let vel_slice = self.vel_deltas_rb.slice(0..bytes);
        pos_slice.map_async(wgpu::MapMode::Read, |_| {});
        vel_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let pos_data = pos_slice.get_mapped_range();
        let vel_data = vel_slice.get_mapped_range();
        let pf: &[f32] = bytemuck::cast_slice(&pos_data);
        let vf: &[f32] = bytemuck::cast_slice(&vel_data);
        let pos_out: Vec<[f32; 3]> = (0..n)
            .map(|i| [pf[i * 3], pf[i * 3 + 1], pf[i * 3 + 2]])
            .collect();
        let vel_out: Vec<[f32; 3]> = (0..n)
            .map(|i| [vf[i * 3], vf[i * 3 + 1], vf[i * 3 + 2]])
            .collect();
        drop(pos_data);
        drop(vel_data);
        self.deltas_rb.unmap();
        self.vel_deltas_rb.unmap();
        (pos_out, vel_out)
    }
}

