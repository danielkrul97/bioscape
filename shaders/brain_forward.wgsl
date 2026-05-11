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

const BRAIN_INPUTS: u32 = 78u;       // 27 sensory + 2 bond inbox + 4 vibration + 45 recurrent
const BRAIN_HIDDEN: u32 = 45u;
const BRAIN_OUTPUTS: u32 = 14u;      // 12 motor/morph + 2 bond message
const W1_OFFSET: u32 = 0u;
const B1_OFFSET: u32 = 3510u;        // BRAIN_HIDDEN * BRAIN_INPUTS
const W2_OFFSET: u32 = 3555u;        // B1_OFFSET + BRAIN_HIDDEN
const B2_OFFSET: u32 = 4185u;        // W2_OFFSET + BRAIN_OUTPUTS * BRAIN_HIDDEN
const WEIGHTS_PER_CELL: u32 = 4199u; // B2_OFFSET + BRAIN_OUTPUTS

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

    // V8: Kahan compensated summation in the dot-product. Bounds error to
    // O(ε) per dot regardless of float-add order, which (combined with the
    // unified brownian RNG stream) keeps CPU `Brain::forward_with_state`
    // and this shader within a tight ULP envelope per tick. The bias is
    // applied AFTER the compensated sum so it does not pollute the running
    // compensation term.
    let w1_base = w_off + W1_OFFSET;
    let b1_base = w_off + B1_OFFSET;
    var hid: array<f32, BRAIN_HIDDEN>;
    for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
        let row_base = w1_base + h * BRAIN_INPUTS;
        var sum: f32 = 0.0;
        var c: f32 = 0.0;
        for (var i: u32 = 0u; i < BRAIN_INPUTS; i = i + 1u) {
            let y = weights[row_base + i] * in_local[i] - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        sum = sum + weights[b1_base + h];
        var act: f32;
        if (h < chunk_end) {
            act = tanh_fast(sum);
        } else if (h < h_n) {
            act = tanh(sum);
        } else {
            act = 0.0;
        }
        hid[h] = act;
        hidden[h_off + h] = act;
    }

    let w2_base = w_off + W2_OFFSET;
    let b2_base = w_off + B2_OFFSET;
    for (var o: u32 = 0u; o < BRAIN_OUTPUTS; o = o + 1u) {
        let row_base = w2_base + o * BRAIN_HIDDEN;
        var sum: f32 = 0.0;
        var c: f32 = 0.0;
        for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
            let y = weights[row_base + h] * hid[h] - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        sum = sum + weights[b2_base + o];
        outputs[o_off + o] = tanh(sum);
    }
}
