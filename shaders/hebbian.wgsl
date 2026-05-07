// Sprint 51: GPU mirror Brain::hebbian_update. Per-cell pokud reward != 0,
// updates brain weights (w1, b1, w2, b2) in-place podle reward × pre × post
// rule. CPU equivalent v lib.rs Brain::hebbian_update.
//
// Layout brain weights musí matchnout `lib::gpu::BRAIN_WEIGHTS_PER_CELL`
// packing (Sprint 44 brain_forward.wgsl). Sprint 80 storage bump HIDDEN 16→32:
//   [0..1696)    w1 row-major (HIDDEN × INPUTS)
//   [1696..1728) b1
//   [1728..2048) w2 row-major (OUTPUTS × HIDDEN) — 10*32 = 320
//   [2048..2058) b2
// Dead zone neurons (hidden_n..BRAIN_HIDDEN) bezpečně self-bound: mají
// last_hidden = 0, takže `lr × h_val × x = 0`, weights nemodifikují.

const BRAIN_INPUTS: u32 = 77u;       // Sprint 126: 27 sensory + 50 recurrent
const BRAIN_HIDDEN: u32 = 50u;
const BRAIN_OUTPUTS: u32 = 12u;      // Sprint 126: +2 (ch1, ch2 emit)
const W1_OFFSET: u32 = 0u;
const B1_OFFSET: u32 = 3850u;        // Sprint 126: BRAIN_HIDDEN * BRAIN_INPUTS = 50*77
const W2_OFFSET: u32 = 3900u;
const B2_OFFSET: u32 = 4500u;
const WEIGHTS_PER_CELL: u32 = 4512u;

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

    // w1[h][in] += lr × hidden[h] × inputs[in]; b1[h] += lr × hidden[h]
    for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
        let h_val = last_hidden[hid_off + h];
        for (var in_i: u32 = 0u; in_i < BRAIN_INPUTS; in_i = in_i + 1u) {
            let x = last_inputs[inp_off + in_i];
            let w_idx = w_off + W1_OFFSET + h * BRAIN_INPUTS + in_i;
            brain_weights[w_idx] = brain_weights[w_idx] + lr * h_val * x;
        }
        let b_idx = w_off + B1_OFFSET + h;
        brain_weights[b_idx] = brain_weights[b_idx] + lr * h_val;
    }

    // w2[o][h] += lr × output[o] × hidden[h]; b2[o] += lr × output[o]
    for (var o: u32 = 0u; o < BRAIN_OUTPUTS; o = o + 1u) {
        let o_val = last_outputs[out_off + o];
        for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
            let h_val = last_hidden[hid_off + h];
            let w_idx = w_off + W2_OFFSET + o * BRAIN_HIDDEN + h;
            brain_weights[w_idx] = brain_weights[w_idx] + lr * o_val * h_val;
        }
        let b_idx = w_off + B2_OFFSET + o;
        brain_weights[b_idx] = brain_weights[b_idx] + lr * o_val;
    }
}
