// Sprint 138: homeostatic synaptic scaling. One thread per cell; per cell
// we walk every w1 row (size BRAIN_INPUTS) and every w2 row (size
// BRAIN_HIDDEN). Rows whose L2 norm exceeds `cap` get scaled down to `cap`.
//
// Sprint 188: combined with per-tick multiplicative weight decay (`decay`).
// The fused scale `(1 - decay) × min(1, cap × rsqrt(sum_sq))` runs every
// tick — decay regularises the Hebbian growth even when rows are well
// inside the cap (the previous "scaling only" design clipped only after
// overshoot, which let one neuron monopolise output weights). Biases
// receive the same multiplicative decay so symmetry is preserved; the
// observed bias range stays well inside `[-2, 2]` so no additive clamp
// is needed yet.
//
// Layout mirrors `hebbian_apply_reward.wgsl`'s weight buffer; we share the
// same `brain_weights_buf` binding from `CellsGpu`.

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
    decay: f32,
    _pad0: u32,
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
    let decay_keep = 1.0 - params.decay;
    let w_off = i * WEIGHTS_PER_CELL;
    let w1_base = w_off + W1_OFFSET;
    let b1_base = w_off + B1_OFFSET;
    let w2_base = w_off + W2_OFFSET;
    let b2_base = w_off + B2_OFFSET;

    // Fused single-pass: compute sum_sq, then apply `(1 - decay) × min(1,
    // cap × rsqrt(sum_sq))`. Below-cap rows still get the decay term —
    // that's the whole point of S188's per-tick decay (`w_eq = Δ /
    // decay`, holds weights at biologically plausible magnitudes even
    // when L2 has plenty of headroom). Path stays branchless across the
    // workgroup.
    for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
        let row_base = w1_base + h * BRAIN_INPUTS;
        var sum_sq: f32 = 0.0;
        for (var in_i: u32 = 0u; in_i < BRAIN_INPUTS; in_i = in_i + 1u) {
            let v = brain_weights[row_base + in_i];
            sum_sq = sum_sq + v * v;
        }
        let scale = decay_keep * min(1.0, cap * inverseSqrt(max(sum_sq, 1e-30)));
        for (var in_i: u32 = 0u; in_i < BRAIN_INPUTS; in_i = in_i + 1u) {
            let idx = row_base + in_i;
            brain_weights[idx] = brain_weights[idx] * scale;
        }
    }

    for (var o: u32 = 0u; o < BRAIN_OUTPUTS; o = o + 1u) {
        let row_base = w2_base + o * BRAIN_HIDDEN;
        var sum_sq: f32 = 0.0;
        for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
            let v = brain_weights[row_base + h];
            sum_sq = sum_sq + v * v;
        }
        let scale = decay_keep * min(1.0, cap * inverseSqrt(max(sum_sq, 1e-30)));
        for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
            let idx = row_base + h;
            brain_weights[idx] = brain_weights[idx] * scale;
        }
    }

    // Bias decay (no L2 cap — biases are 45 + 14 scalars, not a row, and
    // observed magnitudes stay sane). Pure `b *= (1 - decay)` so a bias
    // not actively reinforced by Hebbian drifts back toward 0.
    for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
        let idx = b1_base + h;
        brain_weights[idx] = brain_weights[idx] * decay_keep;
    }
    for (var o: u32 = 0u; o < BRAIN_OUTPUTS; o = o + 1u) {
        let idx = b2_base + o;
        brain_weights[idx] = brain_weights[idx] * decay_keep;
    }
}
