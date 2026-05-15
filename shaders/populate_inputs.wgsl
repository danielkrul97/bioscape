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
// V7: vibration sensing — slots 29..32 (grad_x, grad_y, grad_z, amp). GPU
// sensor gather has no vibration field readback, so these stay 0 — the GPU
// brain path is effectively deaf to the mechanosensory channel.
// BRAIN_INPUTS_SENSORY 29→33, BRAIN_INPUTS 74→78, BRAIN_OUTPUTS unchanged.

struct Params {
    num_cells: u32,
    brain_inputs: u32,           // V7: 78
    brain_inputs_sensory: u32,   // V7: 33
    brain_hidden: u32,           // 50
    brain_recurrent: u32,        // 50 (== brain_hidden)
    smell_norm_gain: f32,
    phero_norm_gain: f32,
    damage_norm_gain: f32,
    density_norm: f32,
    vibration_amp_gain: f32,     // tanh gain on slot 32 (amplitude)
    vibration_grad_gain: f32,    // tanh gain on slots 29..31 (gradient — needs ~100× more than amp)
    _pad1: u32,                  // S187: was reproduce_threshold (now per-cell)
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
// Sprint 187: per-cell reproduce threshold (was a uniform Params field).
@group(0) @binding(13) var<storage, read> reproduce_at_energies: array<f32>; // n

fn forward_vector(yaw: f32, pitch: f32) -> vec3<f32> {
    let cy = cos(yaw);
    let sy = sin(yaw);
    let cp = cos(pitch);
    let sp = sin(pitch);
    return vec3<f32>(cy * cp, sy * cp, sp);
}

// Padé(3,2) tanh — mirror of CPU `tanh_fast_scalar` and the same helper
// in `brain_forward.wgsl`. CPU `populate_brain_inputs` uses the same
// approximation on every sensor input, so this keeps GPU/CPU parity.
fn tanh_fast(x: f32) -> f32 {
    let cx = clamp(x, -3.0, 3.0);
    let x2 = cx * cx;
    return cx * (27.0 + x2) / (27.0 + 9.0 * x2);
}

@compute @workgroup_size(64)
fn populate_inputs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }

    // Sensor stride: V7 = 19 (4 vibration), Wave 6 = 25 (+6 whisker raycast).
    let sensor_off = i * 31u;
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
    // V7: vibration grad + amp, written by sensor_gather.wgsl at offsets 15..19.
    let vib_grad_x = sensor_output[sensor_off + 15u];
    let vib_grad_y = sensor_output[sensor_off + 16u];
    let vib_grad_z = sensor_output[sensor_off + 17u];
    let vib_amp    = sensor_output[sensor_off + 18u];

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
    let energy_norm = clamp(energy / max(reproduce_at_energies[i], 1e-3), 0.0, 1.5);

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
    last_inputs[inputs_off + 7u]  = tanh_fast(smell_x * params.smell_norm_gain);
    last_inputs[inputs_off + 8u]  = tanh_fast(smell_y * params.smell_norm_gain);
    last_inputs[inputs_off + 17u] = tanh_fast(smell_z * params.smell_norm_gain);
    let fwd = forward_vector(heading, pitch);
    last_inputs[inputs_off + 9u]  = fwd.x;
    last_inputs[inputs_off + 10u] = fwd.y;
    last_inputs[inputs_off + 18u] = fwd.z;
    last_inputs[inputs_off + 11u] = tanh_fast(phero_x * params.phero_norm_gain);
    last_inputs[inputs_off + 12u] = tanh_fast(phero_y * params.phero_norm_gain);
    last_inputs[inputs_off + 19u] = tanh_fast(phero_z * params.phero_norm_gain);
    last_inputs[inputs_off + 13u] = tanh_fast(neighbor_count / params.density_norm);
    last_inputs[inputs_off + 14u] = tanh_fast(damage * params.damage_norm_gain);

    // Slot 20 is a legacy reserved gap.
    last_inputs[inputs_off + 20u] = 0.0;
    // Wave L: ch1/ch2 pheromone gradients from sensor_gather slots 25..30.
    // Most runs leave multi-channel pheromone unused (only ch0 is populated)
    // → the whole 3-float chunk is zero and `tanh_fast(0) = 0`, so we can skip
    // 6 `tanh_fast` calls when the chunk is all-zero. The branch is uniform within
    // each warp whenever the simulation doesn't use ch1/ch2 pheromones.
    let p25 = sensor_output[sensor_off + 25u];
    let p26 = sensor_output[sensor_off + 26u];
    let p27 = sensor_output[sensor_off + 27u];
    if (p25 != 0.0 || p26 != 0.0 || p27 != 0.0) {
        last_inputs[inputs_off + 21u] = tanh_fast(p25 * params.phero_norm_gain);
        last_inputs[inputs_off + 22u] = tanh_fast(p26 * params.phero_norm_gain);
        last_inputs[inputs_off + 23u] = tanh_fast(p27 * params.phero_norm_gain);
    } else {
        last_inputs[inputs_off + 21u] = 0.0;
        last_inputs[inputs_off + 22u] = 0.0;
        last_inputs[inputs_off + 23u] = 0.0;
    }
    let p28 = sensor_output[sensor_off + 28u];
    let p29 = sensor_output[sensor_off + 29u];
    let p30 = sensor_output[sensor_off + 30u];
    if (p28 != 0.0 || p29 != 0.0 || p30 != 0.0) {
        last_inputs[inputs_off + 24u] = tanh_fast(p28 * params.phero_norm_gain);
        last_inputs[inputs_off + 25u] = tanh_fast(p29 * params.phero_norm_gain);
        last_inputs[inputs_off + 26u] = tanh_fast(p30 * params.phero_norm_gain);
    } else {
        last_inputs[inputs_off + 24u] = 0.0;
        last_inputs[inputs_off + 25u] = 0.0;
        last_inputs[inputs_off + 26u] = 0.0;
    }
    // Bond-mediated communication inbox: slots 27..29 (N_BOND_MSG_CHANNELS=2).
    // CPU pre-tick aggregates partner messages into bonded_inbox buffer.
    let inbox_off = i * 2u;
    last_inputs[inputs_off + 27u] = bonded_inbox[inbox_off];
    last_inputs[inputs_off + 28u] = bonded_inbox[inbox_off + 1u];

    // V7: vibration slots 29..32 (grad_xyz + amp). GPU sensor_gather samples
    // the vibration FieldGpu directly, mirroring smell/pheromone semantics
    // (tanh-normalize through the per-channel gain so saturation behaviour
    // matches `lib::populate_brain_inputs`).
    last_inputs[inputs_off + 29u] = tanh_fast(vib_grad_x * params.vibration_grad_gain);
    last_inputs[inputs_off + 30u] = tanh_fast(vib_grad_y * params.vibration_grad_gain);
    last_inputs[inputs_off + 31u] = tanh_fast(vib_grad_z * params.vibration_grad_gain);
    last_inputs[inputs_off + 32u] = tanh_fast(vib_amp    * params.vibration_amp_gain);

    // Wave 6: whisker raycast slots 33..38. sensor_gather.wgsl writes
    // normalized free-distances at sensor_off + 19..25; map [0, 1] →
    // [-1, 1] so "clear" reads as +1 and "wall touching" as -1, matching
    // CPU `populate_brain_inputs` whisker formatting.
    last_inputs[inputs_off + 33u] = sensor_output[sensor_off + 19u] * 2.0 - 1.0;
    last_inputs[inputs_off + 34u] = sensor_output[sensor_off + 20u] * 2.0 - 1.0;
    last_inputs[inputs_off + 35u] = sensor_output[sensor_off + 21u] * 2.0 - 1.0;
    last_inputs[inputs_off + 36u] = sensor_output[sensor_off + 22u] * 2.0 - 1.0;
    last_inputs[inputs_off + 37u] = sensor_output[sensor_off + 23u] * 2.0 - 1.0;
    last_inputs[inputs_off + 38u] = sensor_output[sensor_off + 24u] * 2.0 - 1.0;

    // Sprint 30: damage_accum reset after consume.
    damage_accums[i] = 0.0;

    // Recurrent state: copy last_hidden[..BRAIN_RECURRENT] into
    // inputs[INPUTS_SENSORY..]. Overwrites slots 27..71, no pre-zero needed.
    //
    // Manual 4-wide unroll: gives naga a clean window of 4 adjacent loads +
    // 4 adjacent stores so it can emit 128-bit memory ops where the hardware
    // supports them. Full `array<vec4<f32>>` packing would require padding
    // `last_hidden` from BRAIN_HIDDEN=45 to 48 across every plasticity shader,
    // so we settle for the unroll-only path here.
    let recurrent = min(params.brain_recurrent, params.brain_hidden);
    let chunks = recurrent / 4u;
    let out_base = inputs_off + params.brain_inputs_sensory;
    for (var c: u32 = 0u; c < chunks; c = c + 1u) {
        let k = c * 4u;
        let h0 = last_hidden[hidden_off + k];
        let h1 = last_hidden[hidden_off + k + 1u];
        let h2 = last_hidden[hidden_off + k + 2u];
        let h3 = last_hidden[hidden_off + k + 3u];
        last_inputs[out_base + k]      = h0;
        last_inputs[out_base + k + 1u] = h1;
        last_inputs[out_base + k + 2u] = h2;
        last_inputs[out_base + k + 3u] = h3;
    }
    for (var k: u32 = chunks * 4u; k < recurrent; k = k + 1u) {
        last_inputs[out_base + k] = last_hidden[hidden_off + k];
    }
}