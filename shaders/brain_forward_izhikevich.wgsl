// Sprint 147: per-cell Izhikevich forward. One thread per cell; cells
// whose `neuron_models[i] != 1` are skipped (the perceptron shader handled
// them earlier in the tick). For Izhikevich cells:
//   1. L1 matvec: pre_hidden = w1·inputs + b1 → injection current I.
//   2. 32 Euler sub-steps integrate (v, u) over a 16 ms simulated tick.
//      v' = 0.04 v² + 5 v + 140 − u + I
//      u' = a (b v − u)
//      if v ≥ 30: v = c, u += d, spike_count++.
//   3. hidden = 2 × spike_count / IZH_SUBSTEPS − 1 (maps to [-1, +1]).
//   4. L2 matvec + tanh → outputs.
//
// Mirrors `Brain::forward_izhikevich_with_state` exactly for CPU/GPU parity.

const BRAIN_INPUTS: u32 = 84u;
const BRAIN_HIDDEN: u32 = 45u;
const BRAIN_OUTPUTS: u32 = 14u;
const W1_OFFSET: u32 = 0u;
const B1_OFFSET: u32 = 3780u;
const W2_OFFSET: u32 = 3825u;
const B2_OFFSET: u32 = 4455u;
const WEIGHTS_PER_CELL: u32 = 4469u;

const IZH_A: f32 = 0.02;
const IZH_B: f32 = 0.2;
const IZH_C: f32 = -65.0;
const IZH_D: f32 = 8.0;
const IZH_SPIKE_THRESHOLD: f32 = 30.0;
const IZH_SUBSTEPS: u32 = 32u;
const IZH_DT_PER_SUBSTEP_MS: f32 = 0.5;
const NEURON_MODEL_IZHIKEVICH: u32 = 1u;

struct Params {
    num_cells: u32,
    tick: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(4) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(5) var<storage, read_write> membrane: array<f32>;
@group(0) @binding(6) var<storage, read_write> recovery: array<f32>;
@group(0) @binding(7) var<storage, read> neuron_models: array<u32>;
@group(0) @binding(8) var<storage, read_write> post_spike_times: array<u32>;

@compute @workgroup_size(64)
fn forward_izhikevich(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= params.num_cells) {
        return;
    }
    if (neuron_models[cell] != NEURON_MODEL_IZHIKEVICH) {
        return;
    }

    let w_off = cell * WEIGHTS_PER_CELL;
    let in_off = cell * BRAIN_INPUTS;
    let hid_off = cell * BRAIN_HIDDEN;
    let out_off = cell * BRAIN_OUTPUTS;
    let w1_base = w_off + W1_OFFSET;
    let b1_base = w_off + B1_OFFSET;
    let w2_base = w_off + W2_OFFSET;
    let b2_base = w_off + B2_OFFSET;

    // L1 matvec — injection current per hidden neuron.
    var current: array<f32, BRAIN_HIDDEN>;
    for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
        let row_base = w1_base + h * BRAIN_INPUTS;
        var acc: f32 = 0.0;
        for (var in_i: u32 = 0u; in_i < BRAIN_INPUTS; in_i = in_i + 1u) {
            acc = acc + weights[row_base + in_i] * inputs[in_off + in_i];
        }
        current[h] = acc + weights[b1_base + h];
    }

    // Cache (v, u) in registers; flush back after sub-stepping.
    var v_local: array<f32, BRAIN_HIDDEN>;
    var u_local: array<f32, BRAIN_HIDDEN>;
    var spike_counts: array<u32, BRAIN_HIDDEN>;
    for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
        v_local[h] = membrane[hid_off + h];
        u_local[h] = recovery[hid_off + h];
        spike_counts[h] = 0u;
    }

    // Euler sub-step integration.
    for (var step: u32 = 0u; step < IZH_SUBSTEPS; step = step + 1u) {
        for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
            let v = v_local[h];
            let u = u_local[h];
            let dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current[h];
            let du = IZH_A * (IZH_B * v - u);
            let v_new = v + dv * IZH_DT_PER_SUBSTEP_MS;
            let u_new = u + du * IZH_DT_PER_SUBSTEP_MS;
            if (v_new >= IZH_SPIKE_THRESHOLD) {
                v_local[h] = IZH_C;
                u_local[h] = u_new + IZH_D;
                spike_counts[h] = spike_counts[h] + 1u;
                // Sprint 164: record post-spike timing for STDP.
                post_spike_times[hid_off + h] = params.tick;
            } else {
                v_local[h] = v_new;
                u_local[h] = u_new;
            }
        }
    }

    // Flush (v, u) back to global, compute hidden activation.
    let scale = 2.0 / f32(IZH_SUBSTEPS);
    var hidden_local: array<f32, BRAIN_HIDDEN>;
    for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
        membrane[hid_off + h] = v_local[h];
        recovery[hid_off + h] = u_local[h];
        let activation = f32(spike_counts[h]) * scale - 1.0;
        hidden_local[h] = activation;
        hidden[hid_off + h] = activation;
    }

    // L2 matvec + tanh.
    for (var o: u32 = 0u; o < BRAIN_OUTPUTS; o = o + 1u) {
        let row_base = w2_base + o * BRAIN_HIDDEN;
        var acc: f32 = 0.0;
        for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
            acc = acc + weights[row_base + h] * hidden_local[h];
        }
        outputs[out_off + o] = tanh(acc + weights[b2_base + o]);
    }
}
