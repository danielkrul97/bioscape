// Sprint 165: pre-spike encoder for STDP. Per Izhikevich cell, per input
// channel: if the input value crossed `SPIKE_ENCODE_THRESHOLD` this tick,
// stamp `pre_spike_times[cell * BRAIN_INPUTS + i] = tick`. Perceptron
// cells are early-exited (neuron_models[cell] != 1).
//
// Pre-S153 STDP design assumed pre-spikes; this shader is the GPU
// realisation of `Brain::forward_izhikevich_with_state`'s CPU encoder.

const BRAIN_INPUTS: u32 = 86u;
const SPIKE_ENCODE_THRESHOLD: f32 = 0.5;
const NEURON_MODEL_IZHIKEVICH: u32 = 1u;

struct Params {
    num_cells: u32,
    tick: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read> neuron_models: array<u32>;
@group(0) @binding(3) var<storage, read_write> pre_spike_times: array<u32>;

@compute @workgroup_size(64)
fn stdp_encode_pre(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= params.num_cells) {
        return;
    }
    if (neuron_models[cell] != NEURON_MODEL_IZHIKEVICH) {
        return;
    }
    let in_off = cell * BRAIN_INPUTS;
    let times_off = cell * BRAIN_INPUTS;
    for (var j: u32 = 0u; j < BRAIN_INPUTS; j = j + 1u) {
        if (inputs[in_off + j] > SPIKE_ENCODE_THRESHOLD) {
            pre_spike_times[times_off + j] = params.tick;
        }
    }
}
