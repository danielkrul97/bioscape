// Per-cell brain forward pass. Weights layout must match the Rust packing
// in `lib::gpu` — keep BRAIN_WEIGHTS_PER_CELL in sync.
//
// Bindings:
//   1 inputs    N × BRAIN_INPUTS  (AoS)
//   2 weights   N × WEIGHTS_PER_CELL (AoS); per-cell layout:
//       [0..3240)    w1 row-major (HIDDEN × INPUTS)
//       [3240..3285) b1
//       [3285..3825) w2 row-major (OUTPUTS × HIDDEN)
//       [3825..3837) b2
//   3 hidden    N × BRAIN_HIDDEN  (write-back)
//   4 outputs   N × BRAIN_OUTPUTS (write-back)
//   5 hidden_n  N × u32 — per-cell active neuron count
//
// The shader iterates the full BRAIN_HIDDEN range regardless of `hidden_n`:
// dead-zone weights are zero (enforced CPU-side), so inactive neurons
// contribute nothing. `hidden_n` only gates which tanh path the activation
// takes — see `tanh_fast` below.
//
// Sprint 189: LayerNorm before tanh. Pre-activations (`w·x + b`) are
// normalized to mean=0 / std=1 over the active range (L1: [0, h_n),
// L2: [0, BRAIN_OUTPUTS)) before passing through tanh. This breaks the
// recurrent-saturation feedback loop where saturated hidden activations
// (±1) fed back as recurrent inputs and saturated the next-tick preact.
// Math: `normed = (pre - mean) / sqrt(var + eps)`; the tanh that follows
// only saturates on |normed| > ~2.3, which (since std=1) is a tail event
// not the typical case.

const BRAIN_INPUTS: u32 = 84u;       // Wave 2: 27 + 2 bond inbox + 4 vibration + 6 whisker + 45 recurrent
const BRAIN_HIDDEN: u32 = 45u;
const BRAIN_OUTPUTS: u32 = 14u;      // 12 motor/morph + 2 bond message
const W1_OFFSET: u32 = 0u;
const B1_OFFSET: u32 = 3780u;        // BRAIN_HIDDEN * BRAIN_INPUTS
const W2_OFFSET: u32 = 3825u;        // B1_OFFSET + BRAIN_HIDDEN
const B2_OFFSET: u32 = 4455u;        // W2_OFFSET + BRAIN_OUTPUTS * BRAIN_HIDDEN
const WEIGHTS_PER_CELL: u32 = 4469u; // B2_OFFSET + BRAIN_OUTPUTS

struct Params {
    num_cells: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(4) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(5) var<storage, read> hidden_n: array<u32>;

// Padé(3,2) tanh — mirrors the CPU `tanh_fast` SIMD path. Active neurons
// in full 8-chunks use this; the scalar tail uses the WGSL builtin `tanh`
// to match CPU's scalar fallback. Without the split the ~2 % Padé error
// per neuron busts the 1e-4 CPU/GPU parity bound.
fn tanh_fast(x: f32) -> f32 {
    let cx = clamp(x, -3.0, 3.0);
    let x2 = cx * cx;
    return cx * (27.0 + x2) / (27.0 + 9.0 * x2);
}

@compute @workgroup_size(64)
fn forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= params.num_cells) {
        return;
    }

    let w_off = cell * WEIGHTS_PER_CELL;
    let i_off = cell * BRAIN_INPUTS;
    let h_off = cell * BRAIN_HIDDEN;
    let o_off = cell * BRAIN_OUTPUTS;

    let h_n = hidden_n[cell];
    let chunk_end = (h_n / 8u) * 8u;

    // Without this cache, every BRAIN_HIDDEN outer iteration re-reads the
    // same 77 storage slots — 50× redundant load instructions per cell.
    var in_local: array<f32, BRAIN_INPUTS>;
    for (var i: u32 = 0u; i < BRAIN_INPUTS; i = i + 1u) {
        in_local[i] = inputs[i_off + i];
    }

    // Plain dot-product sum. Pre-S190 used Kahan compensated summation
    // (4 FLOPs/iter vs 1) to keep the parity test's 1e-4 tolerance — but
    // CPU `Brain::forward_with_state` uses plain summation too, so the only
    // CPU↔GPU drift left to chase was tanh-implementation noise. Plain sum
    // here stays within parity bounds in practice and saves ~3× FLOPs on the
    // hottest kernel.
    //
    // Two-pass per layer: (1) compute pre-activations; (2) layer-norm over
    // the active range, then tanh. Inactive (`h ≥ h_n`) slots stay at zero
    // and are excluded from mean / variance to avoid biasing the norm.
    let w1_base = w_off + W1_OFFSET;
    let b1_base = w_off + B1_OFFSET;

    // Pass 1: L1 pre-activations.
    var pre_hid: array<f32, BRAIN_HIDDEN>;
    for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
        let row_base = w1_base + h * BRAIN_INPUTS;
        var sum: f32 = weights[b1_base + h];
        for (var i: u32 = 0u; i < BRAIN_INPUTS; i = i + 1u) {
            sum = sum + weights[row_base + i] * in_local[i];
        }
        pre_hid[h] = sum;
    }

    // LayerNorm stats over the active range [0, h_n).
    var sum_h: f32 = 0.0;
    for (var h: u32 = 0u; h < h_n; h = h + 1u) {
        sum_h = sum_h + pre_hid[h];
    }
    let inv_hn = 1.0 / max(f32(h_n), 1.0);
    let mean_h = sum_h * inv_hn;
    var var_h_sum: f32 = 0.0;
    for (var h: u32 = 0u; h < h_n; h = h + 1u) {
        let d = pre_hid[h] - mean_h;
        var_h_sum = var_h_sum + d * d;
    }
    let inv_std_h = inverseSqrt(var_h_sum * inv_hn + 1e-6);

    // Pass 2: apply normalization + tanh. Dead-zone stays at zero.
    var hid: array<f32, BRAIN_HIDDEN>;
    for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
        var act: f32;
        if (h < h_n) {
            let normed = (pre_hid[h] - mean_h) * inv_std_h;
            if (h < chunk_end) {
                act = tanh_fast(normed);
            } else {
                act = tanh(normed);
            }
        } else {
            act = 0.0;
        }
        hid[h] = act;
        hidden[h_off + h] = act;
    }

    // L2: same two-pass treatment over the fixed BRAIN_OUTPUTS range.
    let w2_base = w_off + W2_OFFSET;
    let b2_base = w_off + B2_OFFSET;
    var pre_out: array<f32, BRAIN_OUTPUTS>;
    for (var o: u32 = 0u; o < BRAIN_OUTPUTS; o = o + 1u) {
        let row_base = w2_base + o * BRAIN_HIDDEN;
        var sum: f32 = weights[b2_base + o];
        for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
            sum = sum + weights[row_base + h] * hid[h];
        }
        pre_out[o] = sum;
    }
    var sum_o: f32 = 0.0;
    for (var o: u32 = 0u; o < BRAIN_OUTPUTS; o = o + 1u) {
        sum_o = sum_o + pre_out[o];
    }
    let inv_on = 1.0 / f32(BRAIN_OUTPUTS);
    let mean_o = sum_o * inv_on;
    var var_o_sum: f32 = 0.0;
    for (var o: u32 = 0u; o < BRAIN_OUTPUTS; o = o + 1u) {
        let d = pre_out[o] - mean_o;
        var_o_sum = var_o_sum + d * d;
    }
    let inv_std_o = inverseSqrt(var_o_sum * inv_on + 1e-6);
    for (var o: u32 = 0u; o < BRAIN_OUTPUTS; o = o + 1u) {
        let normed = (pre_out[o] - mean_o) * inv_std_o;
        outputs[o_off + o] = tanh(normed);
    }
}
