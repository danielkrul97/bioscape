// Sprint 139: per-neuron intrinsic excitability regulator. Each tick we
// update a signed EMA of `last_hidden` (the tanh output of the brain
// forward) and shift the corresponding `b1` bias by `−DRIFT × activity_avg`.
// This is a linear BCM-lite — a chronically saturated-positive neuron
// drifts its bias down (less positive pre-activation), a saturated-negative
// neuron drifts its bias up. The dead zone `hidden_n..BRAIN_HIDDEN` has
// `last_hidden = 0`, so its `activity_avg` stays at 0 and the shader is a
// no-op there (no need to read `hidden_n`).
//
// Reads: `last_hidden` (already populated by brain forward this tick).
// Writes: `activity_avg` and `brain_weights` (only the `b1` slots).

const BRAIN_HIDDEN: u32 = 45u;
const W1_OFFSET: u32 = 0u;
const B1_OFFSET: u32 = 3780u;        // BRAIN_HIDDEN * BRAIN_INPUTS
const WEIGHTS_PER_CELL: u32 = 4469u;

struct Params {
    num_cells: u32,
    alpha: f32,
    drift: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> last_hidden: array<f32>;
@group(0) @binding(2) var<storage, read_write> activity_avg: array<f32>;
@group(0) @binding(3) var<storage, read_write> brain_weights: array<f32>;

@compute @workgroup_size(64)
fn excitability(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }
    let hid_off = i * BRAIN_HIDDEN;
    let act_off = i * BRAIN_HIDDEN;
    let b1_base = i * WEIGHTS_PER_CELL + B1_OFFSET;
    let alpha = params.alpha;
    let drift = params.drift;
    for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
        let post = last_hidden[hid_off + h];
        let a_prev = activity_avg[act_off + h];
        let a_new = (1.0 - alpha) * a_prev + alpha * post;
        activity_avg[act_off + h] = a_new;
        brain_weights[b1_base + h] = brain_weights[b1_base + h] - drift * a_new;
    }
}
