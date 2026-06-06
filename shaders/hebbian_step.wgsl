// Wave 7: per-tick eligibility-trace decay + accumulate (mirror of CPU
// `Brain::hebbian_step`). Runs every tick once `last_inputs/hidden/outputs`
// reflect this tick's brain forward pass; no weight changes happen here,
// only trace bookkeeping. Reward application is a separate dispatch
// (`hebbian_apply_reward.wgsl`) that fires on event.
//
//   trace_w1[h][in] = decay · trace_w1[h][in] + last_hidden[h] · last_inputs[in]
//   trace_w2[o][h]  = decay · trace_w2[o][h]  + last_outputs[o] · last_hidden[h]
//
// Decomposition (post-S189): 1 cell = 1 workgroup. The 64 threads cooperate
// on loading `last_inputs` / `last_hidden` into workgroup-shared memory once
// per cell, then each thread owns one L1 / L2 row. Pre-S189 layout was
// 1 cell = 1 thread, so every thread re-read the full input+hidden vectors
// from global storage on every (h, in) outer iteration.

const BRAIN_INPUTS: u32 = 85u;       // 27 + 2 bond inbox + 4 vibration + 6 whisker + 45 recurrent
const BRAIN_HIDDEN: u32 = 45u;
const BRAIN_OUTPUTS: u32 = 15u;      // 12 motor/morph + 2 bond message + 1 vibration emit
// Trace buffer layout mirrors brain_weights: w1 first (rows × inputs), then
// w2 (rows × hidden). Bias slots are present (so the same WEIGHTS_PER_CELL
// stride works) but always zero — caller MUST NOT rely on traces for biases.
const W1_OFFSET: u32 = 0u;
const B1_OFFSET: u32 = 3825u;        // BRAIN_HIDDEN * BRAIN_INPUTS
const W2_OFFSET: u32 = 3870u;        // B1_OFFSET + BRAIN_HIDDEN
const B2_OFFSET: u32 = 4545u;        // W2_OFFSET + BRAIN_OUTPUTS * BRAIN_HIDDEN
const WEIGHTS_PER_CELL: u32 = 4560u; // B2_OFFSET + BRAIN_OUTPUTS

const WG_SIZE: u32 = 64u;

struct StepParams {
    num_cells: u32,
    dt: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> last_inputs: array<f32>;
@group(0) @binding(2) var<storage, read> last_hidden: array<f32>;
@group(0) @binding(3) var<storage, read> last_outputs: array<f32>;
@group(0) @binding(4) var<storage, read_write> brain_traces: array<f32>;
@group(0) @binding(5) var<storage, read> trace_decays: array<f32>;

var<workgroup> in_shared: array<f32, BRAIN_INPUTS>;
var<workgroup> hid_shared: array<f32, BRAIN_HIDDEN>;

@compute @workgroup_size(64)
fn hebbian_step(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = wg.x;
    if (i >= params.num_cells) {
        return;
    }
    let tid = lid.x;
    let inp_off = i * BRAIN_INPUTS;
    let hid_off = i * BRAIN_HIDDEN;
    let out_off = i * BRAIN_OUTPUTS;
    let t_off = i * WEIGHTS_PER_CELL;

    // Cooperative load: each thread fetches ceil(BRAIN_INPUTS / 64) = 2 inputs
    // and ceil(BRAIN_HIDDEN / 64) = 1 hidden into workgroup-shared memory.
    if (tid < BRAIN_INPUTS) {
        in_shared[tid] = last_inputs[inp_off + tid];
    }
    let tid2 = tid + WG_SIZE;
    if (tid2 < BRAIN_INPUTS) {
        in_shared[tid2] = last_inputs[inp_off + tid2];
    }
    if (tid < BRAIN_HIDDEN) {
        hid_shared[tid] = last_hidden[hid_off + tid];
    }
    workgroupBarrier();

    let decay = max(0.0, 1.0 - trace_decays[i] * params.dt);

    // L1 trace_w1[h][in] = decay · trace + hidden[h] · inputs[in].
    // Thread `tid` owns row `h = tid` (only the first BRAIN_HIDDEN=45 threads
    // do work; threads 45..63 fall through to the L2 stage).
    if (tid < BRAIN_HIDDEN) {
        let post = hid_shared[tid];
        let row_base = t_off + W1_OFFSET + tid * BRAIN_INPUTS;
        for (var in_i: u32 = 0u; in_i < BRAIN_INPUTS; in_i = in_i + 1u) {
            let idx = row_base + in_i;
            brain_traces[idx] = decay * brain_traces[idx] + post * in_shared[in_i];
        }
    }

    // L2 trace_w2[o][h] = decay · trace + outputs[o] · hidden[h].
    // First BRAIN_OUTPUTS=14 threads each own one output row.
    if (tid < BRAIN_OUTPUTS) {
        let post = last_outputs[out_off + tid];
        let row_base = t_off + W2_OFFSET + tid * BRAIN_HIDDEN;
        for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
            let idx = row_base + h;
            brain_traces[idx] = decay * brain_traces[idx] + post * hid_shared[h];
        }
    }
}
