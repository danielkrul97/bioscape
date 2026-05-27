// Per-cell `Cell::step` mirror: integrate kinematics, anisotropic + angular
// drag, thermal-modulated energy costs, world bounce. No neighbor lookup.
//
// Pipeline order on the host: apply_brain_motor (motor.wgsl) → apply_morph
// (CPU) → apply_brownian (CPU, RNG-bound) → this shader.
//
// Bindings: 11 storage + 1 uniform = 12 total — at the per-pipeline limit
// (`GpuContext` caps at 12). Adding a binding here forces a SoA refactor.


struct StepParams {
    num_cells: u32,
    pad_a0: u32, pad_a1: u32, pad_a2: u32,
    dt: f32,
    world_half_x: f32,
    world_half_y: f32,
    world_half_z: f32,
    gravity: f32,
    drag: f32,
    angular_drag: f32,
    energy_cost_per_v_sq: f32,
    angular_energy_cost: f32,
    vision_cost_per_radius: f32,
    body_cost_factor: f32,
    age_decay_per_sec: f32,
    fixed_timestep_hz: f32,
    spike_cost_per_sec: f32,
    shell_cost_per_sec: f32,
    attack_cost_per_sec: f32,
    pitch_clamp: f32,
    thermal_top: f32,
    thermal_bottom: f32,
    thermal_q10: f32,
    thermal_ref_temp: f32,
    thermal_diurnal_amp: f32,
    thermal_seasonal_amp: f32,
    thermal_diurnal_sin: f32,
    thermal_seasonal_sin: f32,
    thermal_log2_q10: f32,
    // Wave 4 maze fields. `maze_active != 0` enables the collision_push pass
    // after world_bounce. Voxel resolution is read from `maze_res_x/y`; the
    // mask itself comes from binding 12 (row-major `vy * res_x + vx`,
    // non-zero = occupied). `maze_active = 0` skips the entire block so
    // non-maze runs stay byte-identical.
    maze_active: u32,
    maze_res_x: u32,
    maze_res_y: u32,
    // Sprint 202: thermal perturbation field (binding 13). Sampled at cell
    // position and added to the analytic temperature base; lets warm /
    // cool currents propagate spatially instead of being a strict function
    // of z. `thermal_pert_active = 0` skips the sample so the legacy path
    // stays byte-identical.
    thermal_pert_active: u32,
    thermal_res_x: u32,
    thermal_res_y: u32,
    thermal_res_z: u32,
}

@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocities: array<f32>;
@group(0) @binding(3) var<storage, read_write> headings: array<f32>;
@group(0) @binding(4) var<storage, read_write> pitches: array<f32>;
@group(0) @binding(5) var<storage, read_write> angular_velocities: array<f32>;
@group(0) @binding(6) var<storage, read_write> pitch_velocities: array<f32>;
@group(0) @binding(7) var<storage, read_write> ages: array<u32>;
@group(0) @binding(8) var<storage, read_write> cooldowns: array<u32>;
@group(0) @binding(9) var<storage, read_write> energies: array<f32>;
// body_dims is `array<vec4<f32>>` (length / width / height + 4-byte pad)
// instead of three scalar slots — one 16B load replaces 3 separate scalar
// loads per cell. aux is naturally `vec4` (spike / shell / vision / attack).
@group(0) @binding(10) var<storage, read> body_dims: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> aux: array<vec4<f32>>;
// Wave 4: maze occupancy mask. One u32 per voxel, row-major
// `vy * maze_res_x + vx`. Value 0 = passable, non-zero = wall. Allocated
// at `MAZE_MASK_CAPACITY` u32s; only the first `maze_res_x * maze_res_y`
// are read when `maze_active != 0`.
@group(0) @binding(12) var<storage, read> maze_mask: array<u32>;
// Sprint 202: thermal perturbation grid. Each voxel = f32 (bitcast<u32> on
// upload by FieldGpu). Optional: skipped when `thermal_pert_active = 0`.
@group(0) @binding(13) var<storage, read> thermal_pert_grid: array<u32>;

@compute @workgroup_size(64)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }
    let is_3d = params.world_half_z > 0.0;
    let dt = params.dt;
    let new_age = ages[i] + 1u;
    ages[i] = new_age;
    if (cooldowns[i] > 0u) {
        cooldowns[i] = cooldowns[i] - 1u;
    }

    var pos = vec3<f32>(
        positions[i * 3u + 0u],
        positions[i * 3u + 1u],
        positions[i * 3u + 2u],
    );
    var vel = vec3<f32>(
        velocities[i * 3u + 0u],
        velocities[i * 3u + 1u],
        velocities[i * 3u + 2u],
    );
    var heading = headings[i];
    var pitch = pitches[i];
    var ang_vel = angular_velocities[i];
    var pitch_vel = pitch_velocities[i];
    var energy = energies[i];
    let body = body_dims[i];
    let body_l = body.x;
    let body_w = body.y;
    let body_h = body.z;
    let aux_i = aux[i];
    let spike = aux_i.x;
    let shell = aux_i.y;
    let vision = aux_i.z;
    let attack = aux_i.w;

    pos = pos + vel * dt;
    heading = heading + ang_vel * dt;
    if (is_3d) {
        vel.z = vel.z - params.gravity * dt;
    }
    pitch = clamp(pitch + pitch_vel * dt, -params.pitch_clamp, params.pitch_clamp);

    let cp = cos(pitch);
    let fwd = vec3<f32>(cos(heading) * cp, sin(heading) * cp, sin(pitch));
    let v_par = dot(vel, fwd);
    let v_perp = vel - v_par * fwd;
    let v_perp_mag = length(v_perp);
    let drag_par_factor = params.drag * abs(v_par) * body_w * dt;
    let drag_perp_factor = params.drag * v_perp_mag * body_l * dt;
    let new_v_par = v_par - drag_par_factor * v_par;
    let new_v_perp = v_perp - drag_perp_factor * v_perp;
    vel = new_v_par * fwd + new_v_perp;

    let ang_drag_factor = max(1.0 - params.angular_drag * dt, 0.0);
    ang_vel = ang_vel * ang_drag_factor;
    pitch_vel = pitch_vel * ang_drag_factor;

    // Thermal-modulated metabolism: z-stratified base temp + seasonal (per gen)
    // and diurnal (per tick) oscillations, fed through Q10. In 2D mode
    // (world_half_z = 0) temp = ref → metabolism = 1.0; fold the entire branch
    // away to skip the pow when the simulation is non-thermal.
    var dt_eff = dt;
    if (is_3d) {
        var norm = (pos.z / params.world_half_z + 1.0) * 0.5;
        norm = clamp(norm, 0.0, 1.0);
        let base = params.thermal_bottom + (params.thermal_top - params.thermal_bottom) * norm;
        let seasonal_offset = params.thermal_seasonal_amp * params.thermal_seasonal_sin;
        let diurnal_offset = params.thermal_diurnal_amp * norm * params.thermal_diurnal_sin;
        var temp = base + seasonal_offset + diurnal_offset;
        // Sprint 202: nearest-voxel sample of the advected thermal pert field.
        if (params.thermal_pert_active != 0u) {
            let nx = params.thermal_res_x;
            let ny = params.thermal_res_y;
            let nz = params.thermal_res_z;
            let cs_x = (2.0 * params.world_half_x) / f32(nx);
            let cs_y = (2.0 * params.world_half_y) / f32(ny);
            let cs_z = (2.0 * params.world_half_z) / f32(nz);
            let xi = i32(floor((pos.x + params.world_half_x) / cs_x));
            let yi = i32(floor((pos.y + params.world_half_y) / cs_y));
            let zi = i32(floor((pos.z + params.world_half_z) / cs_z));
            let xim = ((xi % i32(nx)) + i32(nx)) % i32(nx);
            let yim = ((yi % i32(ny)) + i32(ny)) % i32(ny);
            let zim = clamp(zi, 0, i32(nz) - 1);
            let idx = u32(zim) * nx * ny + u32(yim) * nx + u32(xim);
            temp = temp + bitcast<f32>(thermal_pert_grid[idx]);
        }
        // `pow(q10, x) == exp2(x * log2(q10))`; CPU-side `thermal_log2_q10`
        // turns the per-cell pow into a single mul + exp2.
        let metabolism = exp2((temp - params.thermal_ref_temp) * 0.1 * params.thermal_log2_q10);
        dt_eff = dt * metabolism;
    }
    let v_mag_sq = dot(vel, vel);
    energy = energy - v_mag_sq * params.energy_cost_per_v_sq * dt_eff;
    let eff_r = (body_l + body_w + body_h) / 3.0;
    energy = energy - eff_r * eff_r * ang_vel * ang_vel * params.angular_energy_cost * dt_eff;
    energy = energy - vision * params.vision_cost_per_radius * dt_eff;
    let age_sec = f32(new_age) / params.fixed_timestep_hz;
    let aging_factor = 1.0 + params.age_decay_per_sec * age_sec;
    let volume = body_l * body_w * body_h;
    energy = energy - volume * params.body_cost_factor * aging_factor * dt_eff;
    energy = energy - spike * params.spike_cost_per_sec * dt_eff;
    energy = energy - shell * params.shell_cost_per_sec * dt_eff;
    let attack_strength = max(attack, 0.0);
    energy = energy - attack_strength * params.attack_cost_per_sec * dt_eff;

    // Toroidal XY wrap (cylinder topology) + Z bounce. Mirror of CPU
    // `Cell::apply_world_bounce`. In maze mode (Wave 4), XY uses bounded
    // walls instead — clamp + zero outward velocity (Cell::apply_world_bounce_bounded).
    if (params.maze_active != 0u) {
        if (pos.x > params.world_half_x) {
            pos.x = params.world_half_x;
            if (vel.x > 0.0) { vel.x = 0.0; }
        } else if (pos.x < -params.world_half_x) {
            pos.x = -params.world_half_x;
            if (vel.x < 0.0) { vel.x = 0.0; }
        }
        if (pos.y > params.world_half_y) {
            pos.y = params.world_half_y;
            if (vel.y > 0.0) { vel.y = 0.0; }
        } else if (pos.y < -params.world_half_y) {
            pos.y = -params.world_half_y;
            if (vel.y < 0.0) { vel.y = 0.0; }
        }
    } else {
        let wx = 2.0 * params.world_half_x;
        let wy = 2.0 * params.world_half_y;
        if (pos.x >= params.world_half_x || pos.x < -params.world_half_x) {
            let p = pos.x + params.world_half_x;
            pos.x = p - floor(p / wx) * wx - params.world_half_x;
        }
        if (pos.y >= params.world_half_y || pos.y < -params.world_half_y) {
            let p = pos.y + params.world_half_y;
            pos.y = p - floor(p / wy) * wy - params.world_half_y;
        }
    }
    if (is_3d && abs(pos.z) > params.world_half_z) {
        vel.z = -vel.z;
        pos.z = clamp(pos.z, -params.world_half_z, params.world_half_z);
    }

    // Wave 4 maze collision_push: mirror of `Cell::apply_obstacle_collision`.
    // Sample 9 voxels around `pos` (xy 3×3 stencil), accumulate push out of
    // any occupied one, apply to position, zero outward velocity component.
    // Damage to `damage_accum` is NOT written here (binding budget) — CPU/GPU
    // diverge in the brain's wall-bump signal but motion stays identical.
    if (params.maze_active != 0u && params.maze_res_x > 0u && params.maze_res_y > 0u) {
        let radius = max((body_l + body_w + body_h) / 3.0, 2.5);
        let cs_x = (2.0 * params.world_half_x) / f32(params.maze_res_x);
        let cs_y = (2.0 * params.world_half_y) / f32(params.maze_res_y);
        let xi_f = floor((pos.x + params.world_half_x) / cs_x);
        let yi_f = floor((pos.y + params.world_half_y) / cs_y);
        let xi = i32(xi_f);
        let yi = i32(yi_f);
        let nx = i32(params.maze_res_x);
        let ny = i32(params.maze_res_y);
        var push_x: f32 = 0.0;
        var push_y: f32 = 0.0;
        for (var dvy: i32 = -1; dvy <= 1; dvy = dvy + 1) {
            for (var dvx: i32 = -1; dvx <= 1; dvx = dvx + 1) {
                let vx = xi + dvx;
                let vy = yi + dvy;
                if (vx < 0 || vx >= nx || vy < 0 || vy >= ny) {
                    continue;
                }
                let idx = u32(vy) * params.maze_res_x + u32(vx);
                if (maze_mask[idx] == 0u) {
                    continue;
                }
                let cx = -params.world_half_x + (f32(vx) + 0.5) * cs_x;
                let cy = -params.world_half_y + (f32(vy) + 0.5) * cs_y;
                let half_x = cs_x * 0.5;
                let half_y = cs_y * 0.5;
                let closest_x = clamp(pos.x, cx - half_x, cx + half_x);
                let closest_y = clamp(pos.y, cy - half_y, cy + half_y);
                let diff_x = pos.x - closest_x;
                let diff_y = pos.y - closest_y;
                let d2 = diff_x * diff_x + diff_y * diff_y;
                if (d2 >= radius * radius) {
                    continue;
                }
                if (d2 < 1e-8) {
                    let ax = half_x + radius - abs(pos.x - cx);
                    let ay = half_y + radius - abs(pos.y - cy);
                    if (ax < ay) {
                        let s = select(-1.0, 1.0, pos.x >= cx);
                        push_x = push_x + ax * s;
                    } else {
                        let s = select(-1.0, 1.0, pos.y >= cy);
                        push_y = push_y + ay * s;
                    }
                } else {
                    let inv_d = inverseSqrt(d2);
                    let pen = radius - d2 * inv_d;
                    push_x = push_x + diff_x * inv_d * pen;
                    push_y = push_y + diff_y * inv_d * pen;
                }
            }
        }
        if (abs(push_x) > 1e-4 || abs(push_y) > 1e-4) {
            pos.x = pos.x + push_x;
            pos.y = pos.y + push_y;
            let pmag2 = push_x * push_x + push_y * push_y;
            if (pmag2 > 1e-8) {
                let inv = inverseSqrt(pmag2);
                let nxn = push_x * inv;
                let nyn = push_y * inv;
                let v_dot_n = vel.x * nxn + vel.y * nyn;
                if (v_dot_n < 0.0) {
                    vel.x = vel.x - v_dot_n * nxn;
                    vel.y = vel.y - v_dot_n * nyn;
                }
            }
        }
    }

    positions[i * 3u + 0u] = pos.x;
    positions[i * 3u + 1u] = pos.y;
    positions[i * 3u + 2u] = pos.z;
    velocities[i * 3u + 0u] = vel.x;
    velocities[i * 3u + 1u] = vel.y;
    velocities[i * 3u + 2u] = vel.z;
    headings[i] = heading;
    pitches[i] = pitch;
    angular_velocities[i] = ang_vel;
    pitch_velocities[i] = pitch_vel;
    energies[i] = energy;
}
