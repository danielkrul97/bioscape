// Wave 7: per-tick eligibility-trace decay + accumulate (mirror of CPU
// `Brain::hebbian_step`). Runs every tick once `last_inputs/hidden/outputs`
// reflect this tick's brain forward pass; no weight changes happen here,
// only trace bookkeeping. Reward application is a separate dispatch
// (`hebbian_apply_reward.wgsl`) that fires on event.
//
//   trace_w1[h][in] = decay · trace_w1[h][in] + last_hidden[h] · last_inputs[in]
//   trace_w2[o][h]  = decay · trace_w2[o][h]  + last_outputs[o] · last_hidden[h]
//
// `decay` arrives precomputed as `(1 - decay_per_sec * dt).max(0)` so the
// shader has no per-cell mul. Bias terms (`b1`, `b2`) skip traces — they
// ride directly on `last_hidden` / `last_outputs` at reward time, matching
// the CPU implementation.

const BRAIN_INPUTS: u32 = 84u;       // 27 + 2 bond inbox + 4 vibration + 6 whisker + 45 recurrent
const BRAIN_HIDDEN: u32 = 45u;
const BRAIN_OUTPUTS: u32 = 14u;      // 12 motor/morph + 2 bond message
// Trace buffer layout mirrors brain_weights: w1 first (rows × inputs), then
// w2 (rows × hidden). Bias slots are present (so the same WEIGHTS_PER_CELL
// stride works) but always zero — caller MUST NOT rely on traces for biases.
const W1_OFFSET: u32 = 0u;
const B1_OFFSET: u32 = 3780u;        // BRAIN_HIDDEN * BRAIN_INPUTS
const W2_OFFSET: u32 = 3825u;        // B1_OFFSET + BRAIN_HIDDEN
const B2_OFFSET: u32 = 4455u;        // W2_OFFSET + BRAIN_OUTPUTS * BRAIN_HIDDEN
const WEIGHTS_PER_CELL: u32 = 4469u; // B2_OFFSET + BRAIN_OUTPUTS

struct StepParams {
    num_cells: u32,
    decay: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> last_inputs: array<f32>;
@group(0) @binding(2) var<storage, read> last_hidden: array<f32>;
@group(0) @binding(3) var<storage, read> last_outputs: array<f32>;
@group(0) @binding(4) var<storage, read_write> brain_traces: array<f32>;

@compute @workgroup_size(64)
fn hebbian_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }
    let decay = params.decay;
    let t_off = i * WEIGHTS_PER_CELL;
    let inp_off = i * BRAIN_INPUTS;
    let hid_off = i * BRAIN_HIDDEN;
    let out_off = i * BRAIN_OUTPUTS;

    // Cache pre-activations once.
    var in_local: array<f32, BRAIN_INPUTS>;
    for (var k: u32 = 0u; k < BRAIN_INPUTS; k = k + 1u) {
        in_local[k] = last_inputs[inp_off + k];
    }
    var hid_local: array<f32, BRAIN_HIDDEN>;
    for (var k: u32 = 0u; k < BRAIN_HIDDEN; k = k + 1u) {
        hid_local[k] = last_hidden[hid_off + k];
    }

    // L1 trace_w1[h][in] = decay · trace + hidden[h] · inputs[in]
    let w1_base = t_off + W1_OFFSET;
    for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
        let post = hid_local[h];
        let row_base = w1_base + h * BRAIN_INPUTS;
        for (var in_i: u32 = 0u; in_i < BRAIN_INPUTS; in_i = in_i + 1u) {
            let idx = row_base + in_i;
            brain_traces[idx] = decay * brain_traces[idx] + post * in_local[in_i];
        }
    }

    // L2 trace_w2[o][h] = decay · trace + outputs[o] · hidden[h]
    let w2_base = t_off + W2_OFFSET;
    for (var o: u32 = 0u; o < BRAIN_OUTPUTS; o = o + 1u) {
        let post = last_outputs[out_off + o];
        let row_base = w2_base + o * BRAIN_HIDDEN;
        for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
            let idx = row_base + h;
            brain_traces[idx] = decay * brain_traces[idx] + post * hid_local[h];
        }
    }
}
