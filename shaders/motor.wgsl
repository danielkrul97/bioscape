// Per-cell motor update — mirrors CPU `Cell::apply_brain_motor`. Reads
// brain output slots 0 (turn), 1 (thrust), 7 (pitch); the remaining slots
// are written by other consumers. Mass = `effective_radius` (the arithmetic
// mean of the three body axes), not full volume — smoke-tuned for inertia
// scaling without quadratic cost shock on untrained brains.

const BRAIN_OUTPUTS: u32 = 14u; // 12 motor/morph + 2 bond message (V7)

struct Params {
    num_cells: u32,
    dt: f32,
    drag_coefficient: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> outputs_in: array<f32>;
@group(0) @binding(2) var<storage, read> headings: array<f32>;
@group(0) @binding(3) var<storage, read> pitches: array<f32>;
@group(0) @binding(4) var<storage, read> max_speeds: array<f32>;
@group(0) @binding(5) var<storage, read> turn_rates: array<f32>;
@group(0) @binding(6) var<storage, read> effective_radii: array<f32>;
@group(0) @binding(7) var<storage, read_write> velocities: array<f32>;
@group(0) @binding(8) var<storage, read_write> angular_velocities: array<f32>;
@group(0) @binding(9) var<storage, read_write> pitch_velocities: array<f32>;

fn forward_vector(yaw: f32, pitch: f32) -> vec3<f32> {
    let cp = cos(pitch);
    return vec3<f32>(cos(yaw) * cp, sin(yaw) * cp, sin(pitch));
}

@compute @workgroup_size(64)
fn motor(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }
    let mass = max(effective_radii[i], 0.01);
    let inv_mass = 1.0 / mass;
    let turn_rate = turn_rates[i];
    let max_speed = max_speeds[i];
    let dt = params.dt;
    let turn_signal = outputs_in[i * BRAIN_OUTPUTS + 0u];
    let thrust_norm = (outputs_in[i * BRAIN_OUTPUTS + 1u] + 1.0) * 0.5;
    let pitch_signal = outputs_in[i * BRAIN_OUTPUTS + 7u];

    let turn_scale = turn_rate * inv_mass;
    angular_velocities[i] = angular_velocities[i] + turn_signal * turn_scale * dt;
    pitch_velocities[i] = pitch_velocities[i] + pitch_signal * turn_scale * dt;

    let a_dt = thrust_norm * params.drag_coefficient * max_speed * max_speed * inv_mass * dt;
    let fwd = forward_vector(headings[i], pitches[i]);
    velocities[i * 3u + 0u] = velocities[i * 3u + 0u] + a_dt * fwd.x;
    velocities[i * 3u + 1u] = velocities[i * 3u + 1u] + a_dt * fwd.y;
    velocities[i * 3u + 2u] = velocities[i * 3u + 2u] + a_dt * fwd.z;
}
