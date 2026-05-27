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
//
// Sprint 192: lateral inhibition on L1 preacts. Before LayerNorm,
// subtract `α × mean_{h' ≠ h}(softplus(pre[h']))` from each pre[h].
// Effect: strongly active neurons cast a larger inhibitory shadow on
// the rest of the layer; weakly active neurons get suppressed more
// than they inhibit. Tiny preact differences at the input of this
// stage get amplified at its output, which keeps the Hebbian update
// per-neuron-distinct on the next tick — without this, runtime
// convergence to a 14+11 bipolar split (one cluster +0.72, other
// −0.83) reasserted itself within ~6000 ticks even after S188-S191.

const BRAIN_INPUTS: u32 = 86u;       // S198: + 2 symbiont (has + deficit_norm)
const BRAIN_HIDDEN: u32 = 45u;
const BRAIN_OUTPUTS: u32 = 15u;      // 12 motor/morph + 2 bond message + 1 vibration emit
const W1_OFFSET: u32 = 0u;
const B1_OFFSET: u32 = 3870u;        // BRAIN_HIDDEN * BRAIN_INPUTS
const W2_OFFSET: u32 = 3915u;        // B1_OFFSET + BRAIN_HIDDEN
const B2_OFFSET: u32 = 4590u;        // W2_OFFSET + BRAIN_OUTPUTS * BRAIN_HIDDEN
const WEIGHTS_PER_CELL: u32 = 4605u; // B2_OFFSET + BRAIN_OUTPUTS

struct Params {
    num_cells: u32,
    lateral_inhibition_alpha: f32,
    pad1: u32,
    pad2: u32,
}

// Numerically stable softplus. For x > 20 the exp() saturates; cap to
// `x` directly so we never overflow. Below that, `log(1 + exp(x))` is
// the canonical form — log1p is not in WGSL.
fn softplus(x: f32) -> f32 {
    let capped = min(x, 20.0);
    return log(1.0 + exp(capped));
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(4) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(5) var<storage, read> hidden_n: array<u32>;

// Padé(3,2) tanh — mirrors CPU `tanh_fast_scalar` / `tanh_fast_simd`.
// Both CPU SIMD chunks and scalar tail use the same approximation, so
// the GPU path can apply it uniformly without busting the 1e-4 parity bound.
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

    // Sprint 192: lateral inhibition. Compute `softplus(pre)` per active
    // neuron, sum, then subtract `α × (sum − self_softplus) / (n − 1)`
    // from each preact. `n_safe` avoids divide-by-zero on degenerate
    // `h_n ≤ 1`; with `h_n = 1` there are no neighbours to inhibit, so
    // the term is 0 and the path collapses to pre-S192 behaviour.
    let alpha = params.lateral_inhibition_alpha;
    if (alpha > 0.0 && h_n > 1u) {
        // Cache softplus per neuron so the second pass doesn't recompute
        // (45 × log+exp saved per cell per tick).
        var sp_cache: array<f32, BRAIN_HIDDEN>;
        var sp_sum: f32 = 0.0;
        for (var h: u32 = 0u; h < h_n; h = h + 1u) {
            let sp = softplus(pre_hid[h]);
            sp_cache[h] = sp;
            sp_sum = sp_sum + sp;
        }
        let inv_others = 1.0 / f32(h_n - 1u);
        for (var h: u32 = 0u; h < h_n; h = h + 1u) {
            let others = sp_sum - sp_cache[h];
            pre_hid[h] = pre_hid[h] - alpha * others * inv_others;
        }
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
            act = tanh_fast(normed);
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
        outputs[o_off + o] = tanh_fast(normed);
    }
}
