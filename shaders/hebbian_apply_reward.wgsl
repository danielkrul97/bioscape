// Wave 7: per-event trace-based Hebbian reward apply (mirror of CPU
// `Brain::hebbian_apply_reward`). Runs after `hebbian_step` has decayed +
// accumulated traces over recent ticks; only cells with non-zero reward
// (this tick's eat / predation / novelty events) actually mutate weights.
//
// Sprint 190: Oja's rule — the additive Hebbian term gets a multiplicative
// regularisation that pulls each weight toward `trace / post²`:
//
//   w1[h][in] += lr · reward · (trace_w1[h][in] − last_hidden[h]² · w1[h][in]);
//   b1[h]     += lr · reward · last_hidden[h]    // biases unchanged (no Oja term)
//   w2[o][h]  += lr · reward · (trace_w2[o][h]  − last_outputs[o]² · w2[o][h]);
//   b2[o]     += lr · reward · last_outputs[o]
//
// Mathematical effect: Δw = 0 at `w = trace / post²`, so weights converge
// to a finite equilibrium even without the L2 cap / scaling pair. Side
// effect across hidden neurons: each tends to align with a *different*
// principal component of the input distribution (Oja's classical PCA
// result), which is the decorrelation pressure absent from plain Hebbian.
// Combined with the existing `synaptic_scale + weight_decay` defense, the
// stack now (a) bounds weights mathematically, (b) bounds them via a
// hard L2 cap, (c) drifts them toward zero via decay, and (d) decorrelates
// neurons across the population.
//
// Sprint 137: `lr` comes from a per-cell `learning_rates` storage binding,
// not a uniform scalar — each cell scales its Hebbian update with its
// genome `learning_rate`. The legacy `hebbian.wgsl` parity-test path keeps
// the uniform scalar.

const BRAIN_INPUTS: u32 = 86u;
const BRAIN_HIDDEN: u32 = 45u;
const BRAIN_OUTPUTS: u32 = 14u;
const W1_OFFSET: u32 = 0u;
const B1_OFFSET: u32 = 3870u;
const W2_OFFSET: u32 = 3915u;
const B2_OFFSET: u32 = 4545u;
const WEIGHTS_PER_CELL: u32 = 4559u;

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
    // Sprint 190: Oja correction `−lr · reward · hidden[h]² · w` folded
    // into the same loop — single read of `brain_weights[w_idx]` covers
    // both the accumulator and the regulariser, so this is one extra
    // multiply-add per weight (free on GPU).
    let w1_base = w_off + W1_OFFSET;
    let b1_base = w_off + B1_OFFSET;
    let t1_base = t_off + W1_OFFSET;
    for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
        let row_base = w1_base + h * BRAIN_INPUTS;
        let trow_base = t1_base + h * BRAIN_INPUTS;
        let post = last_hidden[hid_off + h];
        let oja_coef = post * post;
        for (var in_i: u32 = 0u; in_i < BRAIN_INPUTS; in_i = in_i + 1u) {
            let w_idx = row_base + in_i;
            let w_old = brain_weights[w_idx];
            brain_weights[w_idx] =
                w_old + lr * (brain_traces[trow_base + in_i] - oja_coef * w_old);
        }
        brain_weights[b1_base + h] = brain_weights[b1_base + h] + lr * post;
    }

    // L2 weights + biases. Same Oja correction shape.
    let w2_base = w_off + W2_OFFSET;
    let b2_base = w_off + B2_OFFSET;
    let t2_base = t_off + W2_OFFSET;
    for (var o: u32 = 0u; o < BRAIN_OUTPUTS; o = o + 1u) {
        let row_base = w2_base + o * BRAIN_HIDDEN;
        let trow_base = t2_base + o * BRAIN_HIDDEN;
        let post = last_outputs[out_off + o];
        let oja_coef = post * post;
        for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
            let w_idx = row_base + h;
            let w_old = brain_weights[w_idx];
            brain_weights[w_idx] =
                w_old + lr * (brain_traces[trow_base + h] - oja_coef * w_old);
        }
        brain_weights[b2_base + o] = brain_weights[b2_base + o] + lr * post;
    }
}
