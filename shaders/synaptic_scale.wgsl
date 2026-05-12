// Sprint 138: homeostatic synaptic scaling. One thread per cell; per cell
// we walk every w1 row (size BRAIN_INPUTS) and every w2 row (size
// BRAIN_HIDDEN). Rows whose L2 norm exceeds `cap` get scaled down to `cap`.
// Bias slots (`b1`, `b2`) are untouched — homeostasis is on the multiplicative
// weight, not the additive threshold.
//
// Layout mirrors `hebbian_apply_reward.wgsl`'s weight buffer; we share the
// same `brain_weights_buf` binding from `CellsGpu`. Pre-S138 there was no
// runtime weight regularization, so unbounded Hebbian drift could push
// `||w1[i]||_2` past 20.

const BRAIN_INPUTS: u32 = 84u;
const BRAIN_HIDDEN: u32 = 45u;
const BRAIN_OUTPUTS: u32 = 14u;
const W1_OFFSET: u32 = 0u;
const B1_OFFSET: u32 = 3780u;        // BRAIN_HIDDEN * BRAIN_INPUTS
const W2_OFFSET: u32 = 3825u;        // B1_OFFSET + BRAIN_HIDDEN
const B2_OFFSET: u32 = 4455u;        // W2_OFFSET + BRAIN_OUTPUTS * BRAIN_HIDDEN
const WEIGHTS_PER_CELL: u32 = 4469u; // B2_OFFSET + BRAIN_OUTPUTS

struct Params {
    num_cells: u32,
    cap: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> brain_weights: array<f32>;

@compute @workgroup_size(64)
fn synaptic_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }
    let cap = params.cap;
    let cap_sq = cap * cap;
    let w_off = i * WEIGHTS_PER_CELL;
    let w1_base = w_off + W1_OFFSET;
    let w2_base = w_off + W2_OFFSET;

    for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
        let row_base = w1_base + h * BRAIN_INPUTS;
        var sum_sq: f32 = 0.0;
        for (var in_i: u32 = 0u; in_i < BRAIN_INPUTS; in_i = in_i + 1u) {
            let v = brain_weights[row_base + in_i];
            sum_sq = sum_sq + v * v;
        }
        if (sum_sq > cap_sq) {
            let scale = cap / sqrt(sum_sq);
            for (var in_i: u32 = 0u; in_i < BRAIN_INPUTS; in_i = in_i + 1u) {
                let idx = row_base + in_i;
                brain_weights[idx] = brain_weights[idx] * scale;
            }
        }
    }

    for (var o: u32 = 0u; o < BRAIN_OUTPUTS; o = o + 1u) {
        let row_base = w2_base + o * BRAIN_HIDDEN;
        var sum_sq: f32 = 0.0;
        for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
            let v = brain_weights[row_base + h];
            sum_sq = sum_sq + v * v;
        }
        if (sum_sq > cap_sq) {
            let scale = cap / sqrt(sum_sq);
            for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
                let idx = row_base + h;
                brain_weights[idx] = brain_weights[idx] * scale;
            }
        }
    }
}
