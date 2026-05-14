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
const BRAIN_INPUTS_SENSORY: u32 = 39u; // 27 + 2 bond inbox + 4 vibration + 6 whisker
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
    gs_strength: f32,
    floor: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> brain_weights: array<f32>;
@group(0) @binding(2) var<storage, read> hidden_n: array<u32>;

@compute @workgroup_size(64)
fn synaptic_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }
    let cap = params.cap;
    let decay_keep = 1.0 - params.decay;
    let floor = params.floor;
    let w_off = i * WEIGHTS_PER_CELL;
    let w1_base = w_off + W1_OFFSET;
    let b1_base = w_off + B1_OFFSET;
    let w2_base = w_off + W2_OFFSET;
    let b2_base = w_off + B2_OFFSET;

    // Gram-Schmidt decorrelation (modified GS, in place). Each active row is
    // partially orthogonalised against the lower-indexed rows, then
    // renormalised to its pre-GS L2 norm so no neuron is zeroed. This is the
    // explicit weight-space deflation term the Oja + lateral-inhibition stack
    // was missing: per-neuron Oja drives every hidden neuron to the *same*
    // principal component, so the layer collapses to rank 1; lateral
    // inhibition is 96 % common-mode (deleted by the next LayerNorm) and
    // cannot decorrelate weight directions. GS acts directly on that
    // collapsed dimension. Active range only — dead-zone rows / recurrent
    // columns carry CPPN-materialised garbage on freshly-born cells and must
    // not enter the dot products. Reads of rows `< h` see this invocation's
    // own earlier writes (program-order coherent within a single invocation).
    let gs = params.gs_strength;
    let h_n = min(hidden_n[i], BRAIN_HIDDEN);
    if (gs > 0.0 && h_n > 1u) {
        let active_inputs = BRAIN_INPUTS_SENSORY + h_n;
        // L1: w1 rows [1, h_n) against rows [0, h), columns [0, active_inputs).
        for (var h: u32 = 1u; h < h_n; h = h + 1u) {
            let row_base = w1_base + h * BRAIN_INPUTS;
            var rowh: array<f32, BRAIN_INPUTS>;
            var orig_sq: f32 = 0.0;
            for (var j: u32 = 0u; j < active_inputs; j = j + 1u) {
                let v = brain_weights[row_base + j];
                rowh[j] = v;
                orig_sq = orig_sq + v * v;
            }
            for (var k: u32 = 0u; k < h; k = k + 1u) {
                let k_base = w1_base + k * BRAIN_INPUTS;
                var dot: f32 = 0.0;
                var ksq: f32 = 0.0;
                for (var j: u32 = 0u; j < active_inputs; j = j + 1u) {
                    let kv = brain_weights[k_base + j];
                    dot = dot + rowh[j] * kv;
                    ksq = ksq + kv * kv;
                }
                if (ksq > 1e-12) {
                    let coef = gs * dot / ksq;
                    for (var j: u32 = 0u; j < active_inputs; j = j + 1u) {
                        rowh[j] = rowh[j] - coef * brain_weights[k_base + j];
                    }
                }
            }
            var new_sq: f32 = 0.0;
            for (var j: u32 = 0u; j < active_inputs; j = j + 1u) {
                new_sq = new_sq + rowh[j] * rowh[j];
            }
            if (new_sq > 1e-8) {
                let renorm = min(sqrt(orig_sq / new_sq), 3.0);
                for (var j: u32 = 0u; j < active_inputs; j = j + 1u) {
                    brain_weights[row_base + j] = rowh[j] * renorm;
                }
            }
        }
        // L2: w2 rows [1, BRAIN_OUTPUTS) against rows [0, o), columns [0, h_n).
        for (var o: u32 = 1u; o < BRAIN_OUTPUTS; o = o + 1u) {
            let row_base = w2_base + o * BRAIN_HIDDEN;
            var rowo: array<f32, BRAIN_HIDDEN>;
            var orig_sq2: f32 = 0.0;
            for (var j: u32 = 0u; j < h_n; j = j + 1u) {
                let v = brain_weights[row_base + j];
                rowo[j] = v;
                orig_sq2 = orig_sq2 + v * v;
            }
            for (var k: u32 = 0u; k < o; k = k + 1u) {
                let k_base = w2_base + k * BRAIN_HIDDEN;
                var dot2: f32 = 0.0;
                var ksq2: f32 = 0.0;
                for (var j: u32 = 0u; j < h_n; j = j + 1u) {
                    let kv = brain_weights[k_base + j];
                    dot2 = dot2 + rowo[j] * kv;
                    ksq2 = ksq2 + kv * kv;
                }
                if (ksq2 > 1e-12) {
                    let coef = gs * dot2 / ksq2;
                    for (var j: u32 = 0u; j < h_n; j = j + 1u) {
                        rowo[j] = rowo[j] - coef * brain_weights[k_base + j];
                    }
                }
            }
            var new_sq2: f32 = 0.0;
            for (var j: u32 = 0u; j < h_n; j = j + 1u) {
                new_sq2 = new_sq2 + rowo[j] * rowo[j];
            }
            if (new_sq2 > 1e-8) {
                let renorm = min(sqrt(orig_sq2 / new_sq2), 3.0);
                for (var j: u32 = 0u; j < h_n; j = j + 1u) {
                    brain_weights[row_base + j] = rowo[j] * renorm;
                }
            }
        }
    }

    // Fused single-pass: compute sum_sq, then apply `(1 - decay) × min(1,
    // cap × rsqrt(sum_sq))`. Below-cap rows still get the decay term —
    // that's the whole point of S188's per-tick decay (`w_eq = Δ /
    // decay`, holds weights at biologically plausible magnitudes even
    // when L2 has plenty of headroom). Path stays branchless across the
    // workgroup.
    //
    // Cache the row in a private array so the apply pass doesn't re-read
    // every weight from storage — drops the per-weight memory ops from
    // 1R(sum) + 1R(apply) + 1W to 1R + 1W (storage) plus on-chip private
    // R/W which folds into registers on modern GPUs.
    var w1_row: array<f32, BRAIN_INPUTS>;
    for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
        let row_base = w1_base + h * BRAIN_INPUTS;
        var sum_sq: f32 = 0.0;
        for (var in_i: u32 = 0u; in_i < BRAIN_INPUTS; in_i = in_i + 1u) {
            let v = brain_weights[row_base + in_i];
            w1_row[in_i] = v;
            sum_sq = sum_sq + v * v;
        }
        // Clamp row L2 norm into [floor, cap], then apply per-tick decay.
        // The floor rescues reward-starved rows from decaying to ~0; decay is
        // uniform scaling so it preserves direction — a floored row keeps its
        // learnt orientation, just at a minimally-responsive magnitude.
        let norm = sqrt(max(sum_sq, 1e-30));
        let scale = max(min(norm, cap) * decay_keep, floor) / norm;
        for (var in_i: u32 = 0u; in_i < BRAIN_INPUTS; in_i = in_i + 1u) {
            brain_weights[row_base + in_i] = w1_row[in_i] * scale;
        }
    }

    var w2_row: array<f32, BRAIN_HIDDEN>;
    for (var o: u32 = 0u; o < BRAIN_OUTPUTS; o = o + 1u) {
        let row_base = w2_base + o * BRAIN_HIDDEN;
        var sum_sq: f32 = 0.0;
        for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
            let v = brain_weights[row_base + h];
            w2_row[h] = v;
            sum_sq = sum_sq + v * v;
        }
        // Clamp row L2 norm into [floor, cap], then apply per-tick decay.
        // The floor rescues reward-starved rows from decaying to ~0; decay is
        // uniform scaling so it preserves direction — a floored row keeps its
        // learnt orientation, just at a minimally-responsive magnitude.
        let norm = sqrt(max(sum_sq, 1e-30));
        let scale = max(min(norm, cap) * decay_keep, floor) / norm;
        for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
            brain_weights[row_base + h] = w2_row[h] * scale;
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
