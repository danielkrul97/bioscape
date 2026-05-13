// Sprint 147: per-cell Izhikevich forward.
//
// Decomposition (post-S190): 1 cell = 1 workgroup. 64 threads cooperate
// per cell — the first 45 threads each own one hidden neuron's (v, u, spike
// count) in scalar registers, the rest help with cooperative loads. The 32
// Euler sub-steps run in registers per thread, no private-memory `array`
// indexing in the hot loop. Pre-S190 layout was 1 thread = 1 cell, which
// spilled 5× `array<f32, 45>` into off-chip private memory and made the
// integrator the most expensive shader by far on Izhikevich-heavy runs.
//
// Pipeline (mirrors `Brain::forward_izhikevich_with_state` for CPU/GPU parity):
//   1. L1 matvec: pre_hidden = w1·inputs + b1 → injection current I.
//   2. 32 Euler sub-steps integrate (v, u) over a 16 ms simulated tick.
//      v' = 0.04 v² + 5 v + 140 − u + I
//      u' = a (b v − u)
//      if v ≥ 30: v = c, u += d, spike_count++.
//   3. hidden = 2 × spike_count / IZH_SUBSTEPS − 1 (maps to [-1, +1]).
//   4. L2 matvec + tanh → outputs.

const BRAIN_INPUTS: u32 = 84u;
const BRAIN_HIDDEN: u32 = 45u;
const BRAIN_OUTPUTS: u32 = 14u;
const W1_OFFSET: u32 = 0u;
const B1_OFFSET: u32 = 3780u;
const W2_OFFSET: u32 = 3825u;
const B2_OFFSET: u32 = 4455u;
const WEIGHTS_PER_CELL: u32 = 4469u;

const IZH_A: f32 = 0.02;
const IZH_B: f32 = 0.2;
const IZH_C: f32 = -65.0;
const IZH_D: f32 = 8.0;
const IZH_SPIKE_THRESHOLD: f32 = 30.0;
const IZH_SUBSTEPS: u32 = 32u;
const IZH_DT_PER_SUBSTEP_MS: f32 = 0.5;
const NEURON_MODEL_IZHIKEVICH: u32 = 1u;

const WG_SIZE: u32 = 64u;

struct Params {
    num_cells: u32,
    tick: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(4) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(5) var<storage, read_write> membrane: array<f32>;
@group(0) @binding(6) var<storage, read_write> recovery: array<f32>;
@group(0) @binding(7) var<storage, read> neuron_models: array<u32>;
@group(0) @binding(8) var<storage, read_write> post_spike_times: array<u32>;

// Workgroup-shared per-cell scratch — replaces the pre-S190 private-memory
// `array<f32, 45>` per thread. On-chip, ~10-100× faster than private DRAM.
var<workgroup> in_shared: array<f32, BRAIN_INPUTS>;
var<workgroup> current_shared: array<f32, BRAIN_HIDDEN>;
var<workgroup> hidden_shared: array<f32, BRAIN_HIDDEN>;
// Sprint 189: output preacts before LayerNorm + tanh. One thread per
// output computes its preact; lane 0 reduces over all 14 to derive
// mean/inv_std, then every output thread normalizes and applies tanh.
var<workgroup> pre_out_shared: array<f32, BRAIN_OUTPUTS>;
var<workgroup> norm_mean: f32;
var<workgroup> norm_inv_std: f32;

@compute @workgroup_size(64)
fn forward_izhikevich(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let cell = wg.x;
    if (cell >= params.num_cells) {
        return;
    }
    // Whole-workgroup early-out on non-Izhikevich cells — uniform branch.
    if (neuron_models[cell] != NEURON_MODEL_IZHIKEVICH) {
        return;
    }
    let tid = lid.x;

    let w_off = cell * WEIGHTS_PER_CELL;
    let in_off = cell * BRAIN_INPUTS;
    let hid_off = cell * BRAIN_HIDDEN;
    let out_off = cell * BRAIN_OUTPUTS;
    let w1_base = w_off + W1_OFFSET;
    let b1_base = w_off + B1_OFFSET;
    let w2_base = w_off + W2_OFFSET;
    let b2_base = w_off + B2_OFFSET;

    // Cooperative load of BRAIN_INPUTS=84 floats — each thread loads up to
    // ceil(84/64) = 2 slots.
    if (tid < BRAIN_INPUTS) {
        in_shared[tid] = inputs[in_off + tid];
    }
    let tid2 = tid + WG_SIZE;
    if (tid2 < BRAIN_INPUTS) {
        in_shared[tid2] = inputs[in_off + tid2];
    }
    workgroupBarrier();

    // L1 matvec — thread `tid` owns neuron `h = tid` for tid < BRAIN_HIDDEN.
    if (tid < BRAIN_HIDDEN) {
        let row_base = w1_base + tid * BRAIN_INPUTS;
        var acc: f32 = weights[b1_base + tid];
        for (var in_i: u32 = 0u; in_i < BRAIN_INPUTS; in_i = in_i + 1u) {
            acc = acc + weights[row_base + in_i] * in_shared[in_i];
        }
        current_shared[tid] = acc;
    }
    workgroupBarrier();

    // 32 Euler sub-steps integrating one neuron per active thread. (v, u)
    // and spike_count live in scalar registers across the whole 32-step
    // loop — no `array` indexing in the hot path.
    if (tid < BRAIN_HIDDEN) {
        var v: f32 = membrane[hid_off + tid];
        var u: f32 = recovery[hid_off + tid];
        var spike_count: u32 = 0u;
        let injection = current_shared[tid];
        for (var step: u32 = 0u; step < IZH_SUBSTEPS; step = step + 1u) {
            let dv = 0.04 * v * v + 5.0 * v + 140.0 - u + injection;
            let du = IZH_A * (IZH_B * v - u);
            let v_new = v + dv * IZH_DT_PER_SUBSTEP_MS;
            let u_new = u + du * IZH_DT_PER_SUBSTEP_MS;
            if (v_new >= IZH_SPIKE_THRESHOLD) {
                v = IZH_C;
                u = u_new + IZH_D;
                spike_count = spike_count + 1u;
                // Sprint 164: record post-spike timing for STDP.
                post_spike_times[hid_off + tid] = params.tick;
            } else {
                v = v_new;
                u = u_new;
            }
        }
        membrane[hid_off + tid] = v;
        recovery[hid_off + tid] = u;
        let activation = f32(spike_count) * (2.0 / f32(IZH_SUBSTEPS)) - 1.0;
        hidden_shared[tid] = activation;
        hidden[hid_off + tid] = activation;
    }
    workgroupBarrier();

    // L2 matvec — thread `tid` owns output `o = tid` for tid < BRAIN_OUTPUTS.
    if (tid < BRAIN_OUTPUTS) {
        let row_base = w2_base + tid * BRAIN_HIDDEN;
        var acc: f32 = weights[b2_base + tid];
        for (var h: u32 = 0u; h < BRAIN_HIDDEN; h = h + 1u) {
            acc = acc + weights[row_base + h] * hidden_shared[h];
        }
        pre_out_shared[tid] = acc;
    }
    workgroupBarrier();

    // LayerNorm reduction: lane 0 walks all BRAIN_OUTPUTS=14 preacts to
    // compute mean / inv_std. Cheap enough to keep serial — BRAIN_OUTPUTS
    // is small, and a 14-way warp shuffle reduction would not pay back
    // the dispatch cost. Result lands in workgroup-shared scalars.
    if (tid == 0u) {
        var sum: f32 = 0.0;
        for (var o: u32 = 0u; o < BRAIN_OUTPUTS; o = o + 1u) {
            sum = sum + pre_out_shared[o];
        }
        let inv_n = 1.0 / f32(BRAIN_OUTPUTS);
        let mean = sum * inv_n;
        var var_sum: f32 = 0.0;
        for (var o: u32 = 0u; o < BRAIN_OUTPUTS; o = o + 1u) {
            let d = pre_out_shared[o] - mean;
            var_sum = var_sum + d * d;
        }
        norm_mean = mean;
        norm_inv_std = inverseSqrt(var_sum * inv_n + 1e-6);
    }
    workgroupBarrier();

    if (tid < BRAIN_OUTPUTS) {
        let normed = (pre_out_shared[tid] - norm_mean) * norm_inv_std;
        outputs[out_off + tid] = tanh(normed);
    }
}
