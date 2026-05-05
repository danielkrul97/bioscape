// Sprint 44: per-cell forward pass mozku.
// Layout musí matchnout `lib::gpu::BRAIN_WEIGHTS_PER_CELL` packing v Rustu.
//
// Vstupy (binding 1): N × BRAIN_INPUTS f32, AoS po cells.
// Váhy (binding 2): N × WEIGHTS_PER_CELL f32, AoS po cells. Per-cell:
//   [0..576)   w1 row-major (HIDDEN rows × INPUTS cols)
//   [576..592) b1
//   [592..752) w2 row-major (OUTPUTS rows × HIDDEN cols) — Sprint 66 OUTPUTS=10
//   [752..762) b2 — Sprint 66 OUTPUTS=10
// Hidden (binding 3): N × BRAIN_HIDDEN f32, write-back.
// Outputs (binding 4): N × BRAIN_OUTPUTS f32, write-back.

const BRAIN_INPUTS: u32 = 36u;
const BRAIN_HIDDEN: u32 = 16u;
const BRAIN_OUTPUTS: u32 = 10u; // Sprint 66: +1 (bond signal output[9])
const W1_OFFSET: u32 = 0u;
const B1_OFFSET: u32 = 576u;        // BRAIN_HIDDEN * BRAIN_INPUTS
const W2_OFFSET: u32 = 592u;        // B1 + BRAIN_HIDDEN
const B2_OFFSET: u32 = 752u;        // W2 + BRAIN_OUTPUTS * BRAIN_HIDDEN (10*16)
const WEIGHTS_PER_CELL: u32 = 762u; // B2 + BRAIN_OUTPUTS

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

    var hid: array<f32, 16>;
    for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
        var sum: f32 = weights[w_off + B1_OFFSET + h];
        for (var i: u32 = 0u; i < BRAIN_INPUTS; i = i + 1u) {
            sum = sum + weights[w_off + W1_OFFSET + h * BRAIN_INPUTS + i] * inputs[i_off + i];
        }
        let act = tanh(sum);
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
