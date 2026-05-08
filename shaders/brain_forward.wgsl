// Sprint 44: per-cell forward pass mozku.
// Layout musí matchnout `lib::gpu::BRAIN_WEIGHTS_PER_CELL` packing v Rustu.
//
// Vstupy (binding 1): N × BRAIN_INPUTS f32, AoS po cells.
// Váhy (binding 2): N × WEIGHTS_PER_CELL f32, AoS po cells. Per-cell layout:
//   [0..3850)    w1 row-major (HIDDEN rows × INPUTS cols)
//   [3850..3900) b1
//   [3900..4500) w2 row-major (OUTPUTS rows × HIDDEN cols)
//   [4500..4512) b2
// Hidden (binding 3): N × BRAIN_HIDDEN f32, write-back.
// Outputs (binding 4): N × BRAIN_OUTPUTS f32, write-back.
// Per-cell active hidden count (`Brain.hidden_n`) je pro shader IRRELEVANT —
// dead zone weights jsou zero (CPU-side init), tedy zero contribution k
// hidden / output. Pre-Sprint-80 cells (hidden_n=16) produkují identický
// výstup pre a post bump.

const BRAIN_INPUTS: u32 = 77u;       // Sprint 126: 27 sensory + 50 recurrent
const BRAIN_HIDDEN: u32 = 50u;       // Sprint 103: 32 → 50 storage cap
const BRAIN_OUTPUTS: u32 = 12u;      // Sprint 126: +2 (ch1, ch2 emit)
const W1_OFFSET: u32 = 0u;
const B1_OFFSET: u32 = 3850u;        // Sprint 126: BRAIN_HIDDEN * BRAIN_INPUTS = 50*77
const W2_OFFSET: u32 = 3900u;        // B1 + BRAIN_HIDDEN
const B2_OFFSET: u32 = 4500u;        // W2 + BRAIN_OUTPUTS * BRAIN_HIDDEN = 3900+12*50
const WEIGHTS_PER_CELL: u32 = 4512u; // B2 + BRAIN_OUTPUTS

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

// Padé(3,2) tanh approximation matching CPU `tanh_fast` for the active
// hidden range that lands in full chunks of 8. Tail neurons within the
// active region use the WGSL builtin `tanh` to mirror the scalar fallback
// on CPU. Without this split, CPU and GPU diverge by up to ~2 % per neuron
// (Padé approximation error), which busts the 1e-4 parity threshold.
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

    var hid: array<f32, 64>;
    for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
        var sum: f32 = weights[w_off + B1_OFFSET + h];
        for (var i: u32 = 0u; i < BRAIN_INPUTS; i = i + 1u) {
            sum = sum + weights[w_off + W1_OFFSET + h * BRAIN_INPUTS + i] * inputs[i_off + i];
        }
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

    for (var o: u32 = 0u; o < BRAIN_OUTPUTS; o = o + 1u) {
        var sum: f32 = weights[w_off + B2_OFFSET + o];
        for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
            sum = sum + weights[w_off + W2_OFFSET + o * BRAIN_HIDDEN + h] * hid[h];
        }
        outputs[o_off + o] = tanh(sum);
    }
}
