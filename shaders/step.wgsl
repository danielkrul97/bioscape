// Sprint 50: GPU mirror Cell::step — integrate_kinematics + anisotropic drag
// + angular drag + energy costs + world bounce. Per-cell, no neighbor lookup.
//
// CPU sekvence: apply_brain_motor (Sprint 50 motor.wgsl) → apply_morph
// (CPU stays) → apply_brownian (CPU stays — RNG) → step (THIS SHADER).
//
// Bindings: 11 storage (limit_max=12 v GpuContext) + 1 uniform = 12 total.

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
    thermal_diurnal_phase: f32,
    thermal_seasonal_phase: f32,
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
@group(0) @binding(10) var<storage, read> body_dims: array<f32>;
@group(0) @binding(11) var<storage, read> aux: array<f32>;

@compute @workgroup_size(64)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }
    ages[i] = ages[i] + 1u;
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
    let body_l = body_dims[i * 3u + 0u];
    let body_w = body_dims[i * 3u + 1u];
    let body_h = body_dims[i * 3u + 2u];
    let spike = aux[i * 4u + 0u];
    let shell = aux[i * 4u + 1u];
    let vision = aux[i * 4u + 2u];
    let attack = aux[i * 4u + 3u];

    // integrate_kinematics
    pos = pos + vel * params.dt;
    heading = heading + ang_vel * params.dt;
    if (params.world_half_z > 0.0) {
        vel.z = vel.z - params.gravity * params.dt;
    }
    pitch = clamp(pitch + pitch_vel * params.dt, -params.pitch_clamp, params.pitch_clamp);

    // apply_anisotropic_drag
    let cp = cos(pitch);
    let fwd = vec3<f32>(cos(heading) * cp, sin(heading) * cp, sin(pitch));
    let v_par = dot(vel, fwd);
    let v_perp = vel - v_par * fwd;
    let v_perp_mag = length(v_perp);
    let drag_par_factor = params.drag * abs(v_par) * body_w * params.dt;
    let drag_perp_factor = params.drag * v_perp_mag * body_l * params.dt;
    let new_v_par = v_par - drag_par_factor * v_par;
    let new_v_perp = v_perp - drag_perp_factor * v_perp;
    vel = new_v_par * fwd + new_v_perp;

    // apply_angular_drag
    let ang_drag_factor = max(1.0 - params.angular_drag * params.dt, 0.0);
    ang_vel = ang_vel * ang_drag_factor;
    pitch_vel = pitch_vel * ang_drag_factor;

    // apply_energy_costs
    // Sprint 85: thermal stratification — z-gradient teplota × Q10 metabolism
    // multiplikátor na all drains. Mirror CPU `temperature_at_z` +
    // `metabolism_factor`. Při world_half_z = 0 fallback na ref temp = 1.0×.
    // Sprint 86: time-varying — seasonal uniform shift (per gen) + diurnal
    // surface-weighted oscilace (per tick). Phases pre-computed CPU-side.
    let TAU = 6.28318530717958647692;
    var temp = params.thermal_ref_temp;
    if (params.world_half_z > 0.0) {
        var norm = (pos.z / params.world_half_z + 1.0) * 0.5;
        norm = clamp(norm, 0.0, 1.0);
        let base = params.thermal_bottom + (params.thermal_top - params.thermal_bottom) * norm;
        let seasonal_offset = params.thermal_seasonal_amp * sin(TAU * params.thermal_seasonal_phase);
        let diurnal_offset = params.thermal_diurnal_amp * norm * sin(TAU * params.thermal_diurnal_phase);
        temp = base + seasonal_offset + diurnal_offset;
    }
    let metabolism = pow(params.thermal_q10, (temp - params.thermal_ref_temp) / 10.0);
    let dt_eff = params.dt * metabolism;
    let v_mag_sq = dot(vel, vel);
    energy = energy - v_mag_sq * params.energy_cost_per_v_sq * dt_eff;
    let eff_r = (body_l + body_w + body_h) / 3.0;
    energy = energy - eff_r * eff_r * ang_vel * ang_vel * params.angular_energy_cost * dt_eff;
    energy = energy - vision * params.vision_cost_per_radius * dt_eff;
    let age_sec = f32(ages[i]) / params.fixed_timestep_hz;
    let aging_factor = 1.0 + params.age_decay_per_sec * age_sec;
    let volume = body_l * body_w * body_h;
    energy = energy - volume * params.body_cost_factor * aging_factor * dt_eff;
    energy = energy - spike * params.spike_cost_per_sec * dt_eff;
    energy = energy - shell * params.shell_cost_per_sec * dt_eff;
    let attack_strength = max(attack, 0.0);
    energy = energy - attack_strength * params.attack_cost_per_sec * dt_eff;

    // Sprint 54: toroidal xy wrap (cylinder topology), z bounce. Matches
    // CPU `Cell::apply_world_bounce` Sprint 54 semantiku.
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
    if (params.world_half_z > 0.0 && abs(pos.z) > params.world_half_z) {
        vel.z = -vel.z;
        pos.z = clamp(pos.z, -params.world_half_z, params.world_half_z);
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
