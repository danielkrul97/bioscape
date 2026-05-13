// Sprint 167: reward-modulated STDP weight update. Per Izhikevich cell
// with non-zero reward this tick:
//   - For every post-neuron that spiked this tick → walk pre, apply LTP
//     (`Δw[h][j] += a_plus × reward × pre_trace[j]`).
//   - For every pre-input that spiked this tick → walk post, apply LTD
//     (`Δw[h][j] -= a_minus × reward × post_trace[h]`).
// Mirror of `Brain::stdp_apply_rewarded`.
//
// Workgroup_size 64 = one thread per cell; no inter-cell weight sharing
// → no atomics needed (weights are per-cell).

const BRAIN_INPUTS: u32 = 84u;
const BRAIN_HIDDEN: u32 = 45u;
const W1_OFFSET: u32 = 0u;
const WEIGHTS_PER_CELL: u32 = 4469u;
const NEURON_MODEL_IZHIKEVICH: u32 = 1u;

struct Params {
    num_cells: u32,
    tick: u32,
    a_plus: f32,
    a_minus: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> brain_weights: array<f32>;
@group(0) @binding(2) var<storage, read> pre_trace: array<f32>;
@group(0) @binding(3) var<storage, read> post_trace: array<f32>;
@group(0) @binding(4) var<storage, read> pre_spike_times: array<u32>;
@group(0) @binding(5) var<storage, read> post_spike_times: array<u32>;
@group(0) @binding(6) var<storage, read> rewards: array<f32>;
@group(0) @binding(7) var<storage, read> neuron_models: array<u32>;

@compute @workgroup_size(64)
fn stdp_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= params.num_cells) {
        return;
    }
    if (neuron_models[cell] != NEURON_MODEL_IZHIKEVICH) {
        return;
    }
    let reward = rewards[cell];
    if (reward == 0.0) {
        return;
    }
    let tick = params.tick;
    let a_plus = params.a_plus * reward;
    let a_minus = params.a_minus * reward;
    let in_off = cell * BRAIN_INPUTS;
    let hid_off = cell * BRAIN_HIDDEN;
    let w1_base = cell * WEIGHTS_PER_CELL + W1_OFFSET;

    // LTP: post-spike this tick → walk pre, add a_plus × pre_trace.
    for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
        if (post_spike_times[hid_off + h] != tick) {
            continue;
        }
        let row_base = w1_base + h * BRAIN_INPUTS;
        for (var j: u32 = 0u; j < BRAIN_INPUTS; j = j + 1u) {
            brain_weights[row_base + j] =
                brain_weights[row_base + j] + a_plus * pre_trace[in_off + j];
        }
    }

    // LTD: pre-spike this tick → walk post, subtract a_minus × post_trace.
    for (var j: u32 = 0u; j < BRAIN_INPUTS; j = j + 1u) {
        if (pre_spike_times[in_off + j] != tick) {
            continue;
        }
        for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
            let row_base = w1_base + h * BRAIN_INPUTS;
            brain_weights[row_base + j] =
                brain_weights[row_base + j] - a_minus * post_trace[hid_off + h];
        }
    }
}
