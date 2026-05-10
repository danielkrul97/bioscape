// Sprint 61: GPU populate_brain_inputs. Merges sensor output buffer (Sprint 60
// SensorGatherGpu, stride 15) + cell metadata (energy, velocity, heading,
// pitch, damage_accum, max_speed, eff_radius, last_hidden) into the brain
// inputs buffer.
//
// Mirrors lib::populate_brain_inputs semantics 1:1 (slot order, sensor field
// mapping, normalization). Workgroup 64.
//
// Side effect: damage_accums[i] is reset to 0 after read, mirroring the CPU
// path. Without readback, the in-shader reset is the only consumption signal.
//
// Sprint 126: BRAIN_INPUTS_SENSORY 21→27 (slots 21–26 = ch1/ch2 pheromone
// gradient xyz; GPU path writes 0 — multi-channel sensor gather is CPU-only).
// BRAIN_INPUTS 71→77, BRAIN_OUTPUTS 10→12.

struct Params {
    num_cells: u32,
    brain_inputs: u32,           // Sprint 126: 77
    brain_inputs_sensory: u32,   // Sprint 126: 27
    brain_hidden: u32,           // 50
    brain_recurrent: u32,        // 50 (== brain_hidden)
    smell_norm_gain: f32,
    phero_norm_gain: f32,
    damage_norm_gain: f32,
    density_norm: f32,
    reproduce_threshold: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sensor_output: array<f32>;          // stride 15
@group(0) @binding(2) var<storage, read> velocities: array<f32>;             // n × 3
@group(0) @binding(3) var<storage, read> energies: array<f32>;
@group(0) @binding(4) var<storage, read> headings: array<f32>;
@group(0) @binding(5) var<storage, read> pitches: array<f32>;
@group(0) @binding(6) var<storage, read_write> damage_accums: array<f32>;
@group(0) @binding(7) var<storage, read> max_speeds: array<f32>;
@group(0) @binding(8) var<storage, read> eff_radii: array<f32>;
@group(0) @binding(9) var<storage, read> vision_radii: array<f32>;
@group(0) @binding(10) var<storage, read> last_hidden: array<f32>;           // n × hidden
@group(0) @binding(11) var<storage, read_write> last_inputs: array<f32>;     // n × inputs
@group(0) @binding(12) var<storage, read> bonded_inbox: array<f32>;          // n × N_BOND_MSG_CHANNELS

fn forward_vector(yaw: f32, pitch: f32) -> vec3<f32> {
    let cy = cos(yaw);
    let sy = sin(yaw);
    let cp = cos(pitch);
    let sp = sin(pitch);
    return vec3<f32>(cy * cp, sy * cp, sp);
}

@compute @workgroup_size(64)
fn populate_inputs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }

    let sensor_off = i * 15u;
    let inputs_off = i * params.brain_inputs;
    let hidden_off = i * params.brain_hidden;

    // Sensor row.
    let food_dx = sensor_output[sensor_off];
    let food_dy = sensor_output[sensor_off + 1u];
    let food_dz = sensor_output[sensor_off + 2u];
    let has_food = sensor_output[sensor_off + 3u] > 0.5;
    let cell_dx = sensor_output[sensor_off + 4u];
    let cell_dy = sensor_output[sensor_off + 5u];
    let cell_dz = sensor_output[sensor_off + 6u];
    let cell_radius = sensor_output[sensor_off + 7u]; // -1 sentinel if no neighbor
    let smell_x = sensor_output[sensor_off + 8u];
    let smell_y = sensor_output[sensor_off + 9u];
    let smell_z = sensor_output[sensor_off + 10u];
    let phero_x = sensor_output[sensor_off + 11u];
    let phero_y = sensor_output[sensor_off + 12u];
    let phero_z = sensor_output[sensor_off + 13u];
    let neighbor_count = f32(bitcast<u32>(sensor_output[sensor_off + 14u]));

    // Cell metadata.
    let vx = velocities[i * 3u];
    let vy = velocities[i * 3u + 1u];
    let energy = energies[i];
    let heading = headings[i];
    let pitch = pitches[i];
    let damage = damage_accums[i];
    let max_speed = max(max_speeds[i], 1e-3);
    let my_radius = max(eff_radii[i], 0.01);
    let vision_r = max(vision_radii[i], 1.0);

    let speed_xy = sqrt(vx * vx + vy * vy);
    let speed_norm = clamp(speed_xy / max_speed, 0.0, 1.0);
    let energy_norm = clamp(energy / params.reproduce_threshold, 0.0, 1.5);

    // No init loop: every slot below is written exactly once. Conditional
    // slots use `select` to fold the absent-target case into the same write.
    let inv_vision = 1.0 / vision_r;

    // Food (slots 0, 1, 15) — zero when no food in vision.
    let food_scale = select(0.0, inv_vision, has_food);
    last_inputs[inputs_off + 0u]  = food_dx * food_scale;
    last_inputs[inputs_off + 1u]  = food_dy * food_scale;
    last_inputs[inputs_off + 15u] = food_dz * food_scale;

    // Nearest cell (slots 2, 3, 6, 16) — zero when no neighbor in vision.
    let cell_present = cell_radius >= 0.0;
    let cell_scale = select(0.0, inv_vision, cell_present);
    last_inputs[inputs_off + 2u]  = cell_dx * cell_scale;
    last_inputs[inputs_off + 3u]  = cell_dy * cell_scale;
    last_inputs[inputs_off + 6u]  = select(0.0, (cell_radius - my_radius) / my_radius, cell_present);
    last_inputs[inputs_off + 16u] = cell_dz * cell_scale;

    // Always-written sensory inputs.
    last_inputs[inputs_off + 4u]  = energy_norm;
    last_inputs[inputs_off + 5u]  = speed_norm;
    last_inputs[inputs_off + 7u]  = tanh(smell_x * params.smell_norm_gain);
    last_inputs[inputs_off + 8u]  = tanh(smell_y * params.smell_norm_gain);
    last_inputs[inputs_off + 17u] = tanh(smell_z * params.smell_norm_gain);
    let fwd = forward_vector(heading, pitch);
    last_inputs[inputs_off + 9u]  = fwd.x;
    last_inputs[inputs_off + 10u] = fwd.y;
    last_inputs[inputs_off + 18u] = fwd.z;
    last_inputs[inputs_off + 11u] = tanh(phero_x * params.phero_norm_gain);
    last_inputs[inputs_off + 12u] = tanh(phero_y * params.phero_norm_gain);
    last_inputs[inputs_off + 19u] = tanh(phero_z * params.phero_norm_gain);
    last_inputs[inputs_off + 13u] = tanh(neighbor_count / params.density_norm);
    last_inputs[inputs_off + 14u] = tanh(damage * params.damage_norm_gain);

    // Reserved sensory slots: 20 (legacy gap) and 21–26 (ch1/ch2 pheromone
    // gradient xyz, Sprint 126). GPU sensor gather is single-channel only,
    // so these stay at 0 until multi-channel gather lands on GPU.
    for (var k: u32 = 20u; k <= 26u; k = k + 1u) {
        last_inputs[inputs_off + k] = 0.0;
    }
    // Bond-mediated communication inbox: slots 27..29 (N_BOND_MSG_CHANNELS=2).
    // CPU pre-tick aggregates partner messages into bonded_inbox buffer.
    let inbox_off = i * 2u;
    last_inputs[inputs_off + 27u] = bonded_inbox[inbox_off];
    last_inputs[inputs_off + 28u] = bonded_inbox[inbox_off + 1u];

    // Sprint 30: damage_accum reset after consume.
    damage_accums[i] = 0.0;

    // Recurrent state: copy last_hidden[..BRAIN_RECURRENT] into
    // inputs[INPUTS_SENSORY..]. Overwrites slots 27..71, no pre-zero needed.
    let recurrent = min(params.brain_recurrent, params.brain_hidden);
    for (var k: u32 = 0u; k < recurrent; k = k + 1u) {
        last_inputs[inputs_off + params.brain_inputs_sensory + k] = last_hidden[hidden_off + k];
    }
}