// Sprint 166: STDP trace step. Per Izhikevich cell, decay
// `pre_trace[i]` and `post_trace[h]` by `exp(-1 / tau_ticks)`, then bump
// any trace slot whose corresponding spike-time array equals the
// current tick by +1.0. Mirror of CPU `Brain::stdp_step`.
//
// Perceptron cells are early-exited.

const BRAIN_INPUTS: u32 = 86u;
const BRAIN_HIDDEN: u32 = 45u;
const NEURON_MODEL_IZHIKEVICH: u32 = 1u;

struct Params {
    num_cells: u32,
    tick: u32,
    // CPU computes `decay = exp(-1 / max(1, tau_ticks))` once per tick and
    // uploads here — saves one `exp` per cell per tick (was per-shader).
    decay: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pre_spike_times: array<u32>;
@group(0) @binding(2) var<storage, read> post_spike_times: array<u32>;
@group(0) @binding(3) var<storage, read_write> pre_trace: array<f32>;
@group(0) @binding(4) var<storage, read_write> post_trace: array<f32>;
@group(0) @binding(5) var<storage, read> neuron_models: array<u32>;

@compute @workgroup_size(64)
fn stdp_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= params.num_cells) {
        return;
    }
    if (neuron_models[cell] != NEURON_MODEL_IZHIKEVICH) {
        return;
    }
    let decay = params.decay;
    let tick = params.tick;
    let in_off = cell * BRAIN_INPUTS;
    let hid_off = cell * BRAIN_HIDDEN;
    for (var j: u32 = 0u; j < BRAIN_INPUTS; j = j + 1u) {
        let idx = in_off + j;
        let bump = select(0.0, 1.0, pre_spike_times[idx] == tick);
        pre_trace[idx] = pre_trace[idx] * decay + bump;
    }
    for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
        let idx = hid_off + h;
        let bump = select(0.0, 1.0, post_spike_times[idx] == tick);
        post_trace[idx] = post_trace[idx] * decay + bump;
    }
}
