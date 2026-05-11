// Per-cell reward-modulated Hebbian update — mirrors CPU
// `Brain::hebbian_update`. For cells with non-zero reward:
//   w1[h][in] += lr · last_hidden[h] · last_inputs[in];   b1[h] += lr · last_hidden[h]
//   w2[o][h] += lr · last_outputs[o] · last_hidden[h];    b2[o] += lr · last_outputs[o]
// where lr = params.learning_rate · reward. Weight buffer layout must match
// the brain_forward shader; see those constants below.
//
// The shader iterates the full BRAIN_HIDDEN range regardless of `hidden_n`:
// dead-zone neurons have `last_hidden = 0`, so their updates resolve to
// `lr · 0 · x = 0` and leave weights untouched.

const BRAIN_INPUTS: u32 = 84u;       // Wave 2: 27 + 2 bond inbox + 4 vibration + 6 whisker + 45 recurrent
const BRAIN_HIDDEN: u32 = 45u;
const BRAIN_OUTPUTS: u32 = 14u;      // 12 motor/morph + 2 bond message
const W1_OFFSET: u32 = 0u;
const B1_OFFSET: u32 = 3780u;        // BRAIN_HIDDEN * BRAIN_INPUTS
const W2_OFFSET: u32 = 3825u;        // B1_OFFSET + BRAIN_HIDDEN
const B2_OFFSET: u32 = 4455u;        // W2_OFFSET + BRAIN_OUTPUTS * BRAIN_HIDDEN
const WEIGHTS_PER_CELL: u32 = 4469u; // B2_OFFSET + BRAIN_OUTPUTS

struct HebbianParams {
    num_cells: u32,
    learning_rate: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: HebbianParams;
@group(0) @binding(1) var<storage, read> last_inputs: array<f32>;
@group(0) @binding(2) var<storage, read> last_hidden: array<f32>;
@group(0) @binding(3) var<storage, read> last_outputs: array<f32>;
@group(0) @binding(4) var<storage, read> rewards: array<f32>;
@group(0) @binding(5) var<storage, read_write> brain_weights: array<f32>;

@compute @workgroup_size(64)
fn hebbian(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }
    let reward = rewards[i];
    if (reward == 0.0) {
        return;
    }
    let lr = params.learning_rate * reward;
    let w_off = i * WEIGHTS_PER_CELL;
    let inp_off = i * BRAIN_INPUTS;
    let hid_off = i * BRAIN_HIDDEN;
    let out_off = i * BRAIN_OUTPUTS;

    // Cache pre-activations once. Without this, the L1 inner loop re-reads
    // last_inputs BRAIN_HIDDEN× and L2 re-reads last_hidden BRAIN_OUTPUTS×.
    var in_local: array<f32, BRAIN_INPUTS>;
    for (var k: u32 = 0u; k < BRAIN_INPUTS; k = k + 1u) {
        in_local[k] = last_inputs[inp_off + k];
    }
    var hid_local: array<f32, BRAIN_HIDDEN>;
    for (var k: u32 = 0u; k < BRAIN_HIDDEN; k = k + 1u) {
        hid_local[k] = last_hidden[hid_off + k];
    }

    let w1_base = w_off + W1_OFFSET;
    let b1_base = w_off + B1_OFFSET;
    for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
        let lr_h = lr * hid_local[h];
        let row_base = w1_base + h * BRAIN_INPUTS;
        for (var in_i: u32 = 0u; in_i < BRAIN_INPUTS; in_i = in_i + 1u) {
            let w_idx = row_base + in_i;
            brain_weights[w_idx] = brain_weights[w_idx] + lr_h * in_local[in_i];
        }
        let b_idx = b1_base + h;
        brain_weights[b_idx] = brain_weights[b_idx] + lr_h;
    }

    let w2_base = w_off + W2_OFFSET;
    let b2_base = w_off + B2_OFFSET;
    for (var o: u32 = 0u; o < BRAIN_OUTPUTS; o = o + 1u) {
        let lr_o = lr * last_outputs[out_off + o];
        let row_base = w2_base + o * BRAIN_HIDDEN;
        for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
            let w_idx = row_base + h;
            brain_weights[w_idx] = brain_weights[w_idx] + lr_o * hid_local[h];
        }
        let b_idx = b2_base + o;
        brain_weights[b_idx] = brain_weights[b_idx] + lr_o;
    }
}
