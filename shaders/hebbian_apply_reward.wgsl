// Wave 7: per-event trace-based Hebbian reward apply (mirror of CPU
// `Brain::hebbian_apply_reward`). Runs after `hebbian_step` has decayed +
// accumulated traces over recent ticks; only cells with non-zero reward
// (this tick's eat / predation / novelty events) actually mutate weights.
//
//   w1[h][in] += lr · reward · trace_w1[h][in];
//   b1[h]     += lr · reward · last_hidden[h]    // bias rides on activations
//   w2[o][h]  += lr · reward · trace_w2[o][h];
//   b2[o]     += lr · reward · last_outputs[o]
//
// Sprint 137: `lr` comes from a per-cell `learning_rates` storage binding,
// not a uniform scalar — each cell scales its Hebbian update with its
// genome `learning_rate`. The legacy `hebbian.wgsl` parity-test path keeps
// the uniform scalar.

const BRAIN_INPUTS: u32 = 84u;
const BRAIN_HIDDEN: u32 = 45u;
const BRAIN_OUTPUTS: u32 = 14u;
const W1_OFFSET: u32 = 0u;
const B1_OFFSET: u32 = 3780u;
const W2_OFFSET: u32 = 3825u;
const B2_OFFSET: u32 = 4455u;
const WEIGHTS_PER_CELL: u32 = 4469u;

struct ApplyParams {
    num_cells: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: ApplyParams;
@group(0) @binding(1) var<storage, read> last_hidden: array<f32>;
@group(0) @binding(2) var<storage, read> last_outputs: array<f32>;
@group(0) @binding(3) var<storage, read> rewards: array<f32>;
@group(0) @binding(4) var<storage, read_write> brain_weights: array<f32>;
@group(0) @binding(5) var<storage, read> brain_traces: array<f32>;
@group(0) @binding(6) var<storage, read> learning_rates: array<f32>;

@compute @workgroup_size(64)
fn hebbian_apply_reward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }
    let reward = rewards[i];
    if (reward == 0.0) {
        return;
    }
    let lr = learning_rates[i] * reward;
    let w_off = i * WEIGHTS_PER_CELL;
    let t_off = i * WEIGHTS_PER_CELL;
    let hid_off = i * BRAIN_HIDDEN;
    let out_off = i * BRAIN_OUTPUTS;

    // L1 weights + biases. Bias term uses last_hidden directly (no trace).
    let w1_base = w_off + W1_OFFSET;
    let b1_base = w_off + B1_OFFSET;
    let t1_base = t_off + W1_OFFSET;
    for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
        let row_base = w1_base + h * BRAIN_INPUTS;
        let trow_base = t1_base + h * BRAIN_INPUTS;
        for (var in_i: u32 = 0u; in_i < BRAIN_INPUTS; in_i = in_i + 1u) {
            let w_idx = row_base + in_i;
            brain_weights[w_idx] = brain_weights[w_idx] + lr * brain_traces[trow_base + in_i];
        }
        brain_weights[b1_base + h] = brain_weights[b1_base + h] + lr * last_hidden[hid_off + h];
    }

    // L2 weights + biases.
    let w2_base = w_off + W2_OFFSET;
    let b2_base = w_off + B2_OFFSET;
    let t2_base = t_off + W2_OFFSET;
    for (var o: u32 = 0u; o < BRAIN_OUTPUTS; o = o + 1u) {
        let row_base = w2_base + o * BRAIN_HIDDEN;
        let trow_base = t2_base + o * BRAIN_HIDDEN;
        for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
            let w_idx = row_base + h;
            brain_weights[w_idx] = brain_weights[w_idx] + lr * brain_traces[trow_base + h];
        }
        brain_weights[b2_base + o] = brain_weights[b2_base + o] + lr * last_outputs[out_off + o];
    }
}
