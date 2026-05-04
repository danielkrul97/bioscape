// Sprint 50: GPU mirror Cell::apply_brain_motor.
// Sprint 42 mass = effective_radius (smoke-tuned z volume()).
// Mutuje per-cell velocity (3D), angular_velocity, pitch_velocity.
// Outputs read-only — výsledky jiného passu (typicky brain forward).

const BRAIN_OUTPUTS: u32 = 9u;

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
    let turn_rate = turn_rates[i];
    let max_speed = max_speeds[i];
    let turn_signal = outputs_in[i * BRAIN_OUTPUTS + 0u];
    let thrust_norm = (outputs_in[i * BRAIN_OUTPUTS + 1u] + 1.0) * 0.5;
    let pitch_signal = outputs_in[i * BRAIN_OUTPUTS + 7u];

    let ang_acc = turn_signal * turn_rate / mass;
    angular_velocities[i] = angular_velocities[i] + ang_acc * params.dt;
    let pitch_acc = pitch_signal * turn_rate / mass;
    pitch_velocities[i] = pitch_velocities[i] + pitch_acc * params.dt;

    let a_max = params.drag_coefficient * max_speed * max_speed / mass;
    let a = thrust_norm * a_max;
    let fwd = forward_vector(headings[i], pitches[i]);
    velocities[i * 3u + 0u] = velocities[i * 3u + 0u] + a * fwd.x * params.dt;
    velocities[i * 3u + 1u] = velocities[i * 3u + 1u] + a * fwd.y * params.dt;
    velocities[i * 3u + 2u] = velocities[i * 3u + 2u] + a * fwd.z * params.dt;
}
