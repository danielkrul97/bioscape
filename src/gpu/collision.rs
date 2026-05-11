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
    adhesion_strength: f32,
    adhesion_cross_type: f32,
    adhesion_range_factor: f32,
    bond_break_factor: f32,
    bonds_per_cell: u32,
    max_contacts_per_cell: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct AdhesionParams {
    pub strength: f32,
    pub cross_type: f32,
    pub range_factor: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct BondParams {
    pub bonds_per_cell: u32,
    pub break_factor: f32,
}

#[derive(Debug, Clone)]
pub struct CollisionResult {
    pub position_deltas: Vec<[f32; 3]>,
    pub velocity_deltas: Vec<[f32; 3]>,
    /// `contact_count[i]` = number of in-contact partners with idx > i.
    /// Truncated at `max_contacts_per_cell`; overflow drops events but the
    /// counter still reflects the true count for diagnostics.
    pub contact_count: Vec<u32>,
    /// Flat `contact_partners[i * max_contacts_per_cell + slot]` = partner
    /// idx (only the first `contact_count[i]` slots are valid).
    pub contact_partners: Vec<u32>,
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
    adhesion: AdhesionParams,
    bonds: BondParams,
    max_contacts_per_cell: u32,
    world_half_xy: [f32; 2],
    params_buf: wgpu::Buffer,
    positions_buf: wgpu::Buffer,
    velocities_buf: wgpu::Buffer,
    eff_radii_buf: wgpu::Buffer,
    max_axes_buf: wgpu::Buffer,
    adhesion_types_buf: wgpu::Buffer,
    bond_partner_idx_buf: wgpu::Buffer,
    bond_rest_buf: wgpu::Buffer,
    bond_stiffness_buf: wgpu::Buffer,
    bond_damping_buf: wgpu::Buffer,
    contact_count_buf: wgpu::Buffer,
    contact_partners_buf: wgpu::Buffer,
    deltas_buf: wgpu::Buffer,
    vel_deltas_buf: wgpu::Buffer,
    deltas_rb: wgpu::Buffer,
    vel_deltas_rb: wgpu::Buffer,
    contact_count_rb: wgpu::Buffer,
    contact_partners_rb: wgpu::Buffer,
    pos_packed: Vec<f32>,
    vel_packed: Vec<f32>,
}

impl CollisionGpu {
    #[allow(clippy::too_many_arguments)]
    pub fn with_context(
        ctx: &GpuContext,
        capacity: usize,
        cell_size: f32,
        cell_radius_const: f32,
        collision_restitution: f32,
        adhesion: AdhesionParams,
        bonds: BondParams,
        max_contacts_per_cell: u32,
        world_half_xy: [f32; 2],
    ) -> Result<Self, String> {
        Self::with_device_inner(
            Arc::clone(&ctx.device),
            Arc::clone(&ctx.queue),
            capacity,
            cell_size,
            cell_radius_const,
            collision_restitution,
            adhesion,
            bonds,
            max_contacts_per_cell,
            world_half_xy,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        capacity: usize,
        cell_size: f32,
        cell_radius_const: f32,
        collision_restitution: f32,
        adhesion: AdhesionParams,
        bonds: BondParams,
        max_contacts_per_cell: u32,
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
            adhesion,
            bonds,
            max_contacts_per_cell,
            world_half_xy,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn with_device_inner(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        capacity: usize,
        cell_size: f32,
        cell_radius_const: f32,
        collision_restitution: f32,
        adhesion: AdhesionParams,
        bonds: BondParams,
        max_contacts_per_cell: u32,
        world_half_xy: [f32; 2],
    ) -> Result<Self, String> {
        assert!(capacity > 0);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("collision"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/collision.wgsl").into()),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..16)
            .map(|i| {
                let ty = if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else if i == 6 || i == 8 || i == 14 || i == 15 {
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
        let adhesion_types_buf = mk("collision-adhesion-types", n * f, stor_dst);
        let bonds_total = n * (bonds.bonds_per_cell.max(1) as u64);
        let bond_partner_idx_buf = mk("collision-bond-partner-idx", bonds_total * f, stor_dst);
        let bond_rest_buf = mk("collision-bond-rest", bonds_total * f, stor_dst);
        let bond_stiffness_buf = mk("collision-bond-stiffness", bonds_total * f, stor_dst);
        let bond_damping_buf = mk("collision-bond-damping", bonds_total * f, stor_dst);
        let stor_dst_src = stor_dst | wgpu::BufferUsages::COPY_SRC;
        let contacts_total = n * (max_contacts_per_cell.max(1) as u64);
        let contact_count_buf = mk("collision-contact-count", n * f, stor_dst_src);
        let contact_partners_buf =
            mk("collision-contact-partners", contacts_total * f, stor_dst_src);
        let deltas_buf = mk("collision-deltas", n * 3 * f, stor_src);
        let vel_deltas_buf = mk("collision-vel-deltas", n * 3 * f, stor_src);
        let deltas_rb = mk("collision-deltas-rb", n * 3 * f, read);
        let vel_deltas_rb = mk("collision-vel-deltas-rb", n * 3 * f, read);
        let contact_count_rb = mk("collision-contact-count-rb", n * f, read);
        let contact_partners_rb = mk("collision-contact-partners-rb", contacts_total * f, read);

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            capacity,
            cell_size,
            cell_radius_const,
            collision_restitution,
            adhesion,
            bonds,
            max_contacts_per_cell,
            world_half_xy,
            params_buf,
            positions_buf,
            velocities_buf,
            eff_radii_buf,
            max_axes_buf,
            adhesion_types_buf,
            bond_partner_idx_buf,
            bond_rest_buf,
            bond_stiffness_buf,
            bond_damping_buf,
            contact_count_buf,
            contact_partners_buf,
            deltas_buf,
            vel_deltas_buf,
            deltas_rb,
            vel_deltas_rb,
            contact_count_rb,
            contact_partners_rb,
            pos_packed: Vec::new(),
            vel_packed: Vec::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &mut self,
        positions: &[[f32; 3]],
        velocities: &[[f32; 3]],
        eff_radii: &[f32],
        max_axes: &[f32],
        adhesion_types: &[u32],
        bond_partner_idx: &[i32],
        bond_rest: &[f32],
        bond_stiffness: &[f32],
        bond_damping: &[f32],
        cell_hash: &SpatialHashGpu,
    ) -> CollisionResult {
        let n = positions.len();
        assert!(n <= self.capacity, "collision capacity overflow");
        assert_eq!(velocities.len(), n, "velocities length mismatch");
        assert_eq!(adhesion_types.len(), n, "adhesion_types length mismatch");
        let bonds_total = n * self.bonds.bonds_per_cell as usize;
        assert_eq!(bond_partner_idx.len(), bonds_total, "bond_partner_idx length mismatch");
        assert_eq!(bond_rest.len(), bonds_total, "bond_rest length mismatch");
        assert_eq!(bond_stiffness.len(), bonds_total, "bond_stiffness length mismatch");
        assert_eq!(bond_damping.len(), bonds_total, "bond_damping length mismatch");
        if n == 0 {
            return CollisionResult {
                position_deltas: Vec::new(),
                velocity_deltas: Vec::new(),
                contact_count: Vec::new(),
                contact_partners: Vec::new(),
            };
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
            adhesion_strength: self.adhesion.strength,
            adhesion_cross_type: self.adhesion.cross_type,
            adhesion_range_factor: self.adhesion.range_factor,
            bond_break_factor: self.bonds.break_factor,
            bonds_per_cell: self.bonds.bonds_per_cell,
            max_contacts_per_cell: self.max_contacts_per_cell,
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
        self.queue.write_buffer(
            &self.adhesion_types_buf,
            0,
            bytemuck::cast_slice(adhesion_types),
        );
        self.queue.write_buffer(
            &self.bond_partner_idx_buf,
            0,
            bytemuck::cast_slice(bond_partner_idx),
        );
        self.queue
            .write_buffer(&self.bond_rest_buf, 0, bytemuck::cast_slice(bond_rest));
        self.queue.write_buffer(
            &self.bond_stiffness_buf,
            0,
            bytemuck::cast_slice(bond_stiffness),
        );
        self.queue.write_buffer(
            &self.bond_damping_buf,
            0,
            bytemuck::cast_slice(bond_damping),
        );
        // Atomic counters and the partners array start clean each dispatch.
        let zero_count = vec![0u8; (n * 4) as usize];
        self.queue
            .write_buffer(&self.contact_count_buf, 0, &zero_count);
        let contacts_total = n * self.max_contacts_per_cell as usize;
        let zero_partners = vec![0u8; contacts_total * 4];
        self.queue
            .write_buffer(&self.contact_partners_buf, 0, &zero_partners);

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
                wgpu::BindGroupEntry { binding: 9, resource: self.adhesion_types_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: self.bond_partner_idx_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: self.bond_rest_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: self.bond_stiffness_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: self.bond_damping_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 14, resource: self.contact_count_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 15, resource: self.contact_partners_buf.as_entire_binding() },
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
        let count_bytes = (n as u64) * 4;
        let partners_bytes = (n as u64) * (self.max_contacts_per_cell as u64) * 4;
        encoder.copy_buffer_to_buffer(&self.deltas_buf, 0, &self.deltas_rb, 0, bytes);
        encoder.copy_buffer_to_buffer(&self.vel_deltas_buf, 0, &self.vel_deltas_rb, 0, bytes);
        encoder.copy_buffer_to_buffer(
            &self.contact_count_buf,
            0,
            &self.contact_count_rb,
            0,
            count_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &self.contact_partners_buf,
            0,
            &self.contact_partners_rb,
            0,
            partners_bytes,
        );
        self.queue.submit(Some(encoder.finish()));

        let pos_slice = self.deltas_rb.slice(0..bytes);
        let vel_slice = self.vel_deltas_rb.slice(0..bytes);
        let count_slice = self.contact_count_rb.slice(0..count_bytes);
        let partners_slice = self.contact_partners_rb.slice(0..partners_bytes);
        pos_slice.map_async(wgpu::MapMode::Read, |_| {});
        vel_slice.map_async(wgpu::MapMode::Read, |_| {});
        count_slice.map_async(wgpu::MapMode::Read, |_| {});
        partners_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let pos_data = pos_slice.get_mapped_range();
        let vel_data = vel_slice.get_mapped_range();
        let count_data = count_slice.get_mapped_range();
        let partners_data = partners_slice.get_mapped_range();
        let pf: &[f32] = bytemuck::cast_slice(&pos_data);
        let vf: &[f32] = bytemuck::cast_slice(&vel_data);
        let pos_out: Vec<[f32; 3]> = (0..n)
            .map(|i| [pf[i * 3], pf[i * 3 + 1], pf[i * 3 + 2]])
            .collect();
        let vel_out: Vec<[f32; 3]> = (0..n)
            .map(|i| [vf[i * 3], vf[i * 3 + 1], vf[i * 3 + 2]])
            .collect();
        let contact_count: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&count_data).to_vec();
        let contact_partners: Vec<u32> =
            bytemuck::cast_slice::<u8, u32>(&partners_data).to_vec();
        drop(pos_data);
        drop(vel_data);
        drop(count_data);
        drop(partners_data);
        self.deltas_rb.unmap();
        self.vel_deltas_rb.unmap();
        self.contact_count_rb.unmap();
        self.contact_partners_rb.unmap();
        CollisionResult {
            position_deltas: pos_out,
            velocity_deltas: vel_out,
            contact_count,
            contact_partners,
        }
    }
}

