// GPU equivalent of `Brain::from_cppn` (neural/brain.rs). Each compute thread
// evaluates one substrate query for one child cell — the 3837 weights of a
// child brain are produced by 3837 threads, batched across all children of
// the current reproduce phase. Result is written **directly** into the
// `CellsGpu::brain_weights_buf` at the child's slot offset, skipping the
// per-child `upload_brain_at` round-trip.
//
// Layout of brain_weights[slot * WEIGHTS_PER_CELL + …] mirrors the CPU /
// brain_forward shader convention:
//   [0..3240)    w1 (HIDDEN × INPUTS, row-major)
//   [3240..3285) b1
//   [3285..3825) w2 (OUTPUTS × HIDDEN, row-major)
//   [3825..3837) b2
//
// CPPN encoding (1 buffer slot per child, indexed by `cell_idx`):
//   `cppn_meta[cell_idx]   = (num_nodes, num_links, _, _)`
//   `cppn_nodes[cell_idx * CPPN_MAX_NODES + i]`   — i-th node (i < num_nodes)
//   `cppn_links[cell_idx * CPPN_MAX_LINKS + j]`   — j-th link (j < num_links)
//
// Convention from `Cppn::random` — preserved by `Cppn::crossover` /
// `mutate_add_node`:
//   nodes[0..CPPN_INPUTS]        = input nodes (ids 0..6)
//   nodes[CPPN_INPUTS..+CPPN_OUT] = output nodes (ids 7..8)
//   nodes[CPPN_INPUTS+CPPN_OUT..] = hidden nodes
// node `id` is dense across [0, num_nodes) at random init but can have small
// gaps after crossover combines two parents with diverged add_node histories.
// Activations are indexed by `id` (max < CPPN_MAX_NODES per `mutate_add_node`
// guard), not by array index.

const CPPN_MAX_NODES: u32 = 64u;
const CPPN_MAX_LINKS: u32 = 256u;
const CPPN_INPUTS: u32 = 7u;
const CPPN_OUTPUTS: u32 = 2u;
const CPPN_LINK_EXISTS_THRESHOLD: f32 = 0.0;

const BRAIN_INPUTS_SENSORY: u32 = 41u;  // S198: + 2 symbiont (has + deficit_norm)
const BRAIN_INPUTS: u32 = 86u;          // + 45 recurrent
const BRAIN_HIDDEN: u32 = 45u;
const BRAIN_OUTPUTS: u32 = 15u;         // 12 motor/morph + 2 bond message + 1 vibration emit
const W1_OFFSET: u32 = 0u;
const B1_OFFSET: u32 = 3870u;
const W2_OFFSET: u32 = 3915u;
const B2_OFFSET: u32 = 4590u;
const WEIGHTS_PER_CELL: u32 = 4605u;
const VIBRATION_EMIT_OUTPUT: u32 = 14u;
const INNATE_VIBRATION_EMIT_BIAS: f32 = -2.0;

const NUM_W1: u32 = 3870u;     // BRAIN_HIDDEN × BRAIN_INPUTS
const NUM_B1: u32 = 45u;       // BRAIN_HIDDEN
const NUM_W2: u32 = 675u;      // BRAIN_OUTPUTS × BRAIN_HIDDEN
const NUM_B2: u32 = 15u;       // BRAIN_OUTPUTS
const QUERIES_PER_CHILD: u32 = 4605u; // NUM_W1 + NUM_B1 + NUM_W2 + NUM_B2

struct Params {
    num_children: u32,
    pad0: u32, pad1: u32, pad2: u32,
}

struct CppnNode {
    id: u32,
    activation: u32,    // 0=Linear, 1=Sigmoid, 2=Tanh, 3=Gaussian, 4=Sine, 5=Abs, 6=Step
    bias: f32,
    layer: u32,
}

struct CppnLink {
    from_id: u32,
    to_id: u32,
    weight: f32,
    enabled: u32,       // 0=false, 1=true
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cppn_meta: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> cppn_nodes: array<CppnNode>;
@group(0) @binding(3) var<storage, read> cppn_links: array<CppnLink>;
@group(0) @binding(4) var<storage, read> child_slots: array<u32>;
@group(0) @binding(5) var<storage, read_write> brain_weights: array<f32>;
// CSR per-target offsets — `link_offsets[cell * (CPPN_MAX_NODES+1) + to_id]`
// is the start index in `cppn_links` for links ending at `to_id`. Lets us
// walk only the incoming-edge slice instead of all CPPN_MAX_LINKS slots.
@group(0) @binding(6) var<storage, read> link_offsets: array<u32>;

const CPPN_LINK_OFFSETS_PER_CELL: u32 = 65u; // CPPN_MAX_NODES + 1

// 1D substrate coords (mirror of `neural::cppn::substrate_*_coords`).
fn substrate_input(slot: u32) -> vec3<f32> {
    if (slot < BRAIN_INPUTS_SENSORY) {
        let denom = max(f32(BRAIN_INPUTS_SENSORY) - 1.0, 1.0);
        let x = -1.0 + 2.0 * f32(slot) / denom;
        return vec3<f32>(x, 0.0, -1.0);
    }
    let h = slot - BRAIN_INPUTS_SENSORY;
    let denom = max(f32(BRAIN_HIDDEN) - 1.0, 1.0);
    let x = -1.0 + 2.0 * f32(h) / denom;
    return vec3<f32>(x, 0.0, 0.0);
}

fn substrate_hidden(slot: u32) -> vec3<f32> {
    let denom = max(f32(BRAIN_HIDDEN) - 1.0, 1.0);
    let x = -1.0 + 2.0 * f32(slot) / denom;
    return vec3<f32>(x, 0.0, 0.0);
}

fn substrate_output(slot: u32) -> vec3<f32> {
    let denom = max(f32(BRAIN_OUTPUTS) - 1.0, 1.0);
    let x = -1.0 + 2.0 * f32(slot) / denom;
    return vec3<f32>(x, 0.0, 1.0);
}

// Padé(3,2) tanh — matches the CPU `tanh_fast_scalar` and the SIMD lane in
// `forward_batch_x8`.
fn tanh_fast(x: f32) -> f32 {
    let cx = clamp(x, -3.0, 3.0);
    let x2 = cx * cx;
    return cx * (27.0 + x2) / (27.0 + 9.0 * x2);
}

fn activate(act: u32, x: f32) -> f32 {
    switch act {
        case 0u: { return x; }                                  // Linear
        case 1u: { return 1.0 / (1.0 + exp(-x)); }              // Sigmoid
        case 2u: { return tanh_fast(x); }                       // Tanh
        case 3u: { return exp(-x * x); }                        // Gaussian
        case 4u: { return sin(x); }                             // Sine
        case 5u: { return abs(x); }                             // Abs
        default: { return select(0.0, 1.0, x >= 0.0); }         // Step (case 6u)
    }
}

// Forward pass through the cell's CPPN. `inputs` has the 7 substrate values
// (from_xyz, to_xyz, marker). Returns (weight, link_exists).
fn cppn_forward(cell_idx: u32, inputs: array<f32, 7>) -> vec2<f32> {
    let node_base = cell_idx * CPPN_MAX_NODES;
    let link_base = cell_idx * CPPN_MAX_LINKS;
    let offsets_base = cell_idx * CPPN_LINK_OFFSETS_PER_CELL;
    let cell_meta = cppn_meta[cell_idx];
    let num_nodes = cell_meta.x;
    // `cell_meta.z` carries the precomputed max_layer (CPU-side `pack`).
    // Replaces the per-query O(num_nodes) scan over `cppn_nodes`.
    let max_layer = cell_meta.z;

    // WGSL zero-initializes function-scope `var` arrays — no explicit loop
    // needed. Inputs immediately overwrite the slots they care about; the
    // CPPN topological order ensures every `activations[from_id]` read in
    // the layer pass has been written before.
    var activations: array<f32, 64>;
    // Inputs occupy nodes[0..CPPN_INPUTS] per `Cppn::random` layout.
    for (var k: u32 = 0u; k < CPPN_INPUTS; k = k + 1u) {
        let n = cppn_nodes[node_base + k];
        activations[n.id] = inputs[k];
    }

    for (var layer: u32 = 1u; layer <= max_layer; layer = layer + 1u) {
        for (var k: u32 = 0u; k < num_nodes; k = k + 1u) {
            let n = cppn_nodes[node_base + k];
            if (n.layer != layer) {
                continue;
            }
            var sum: f32 = n.bias;
            // CSR walk: only links incoming to `n.id` instead of all 256.
            let link_start = link_offsets[offsets_base + n.id];
            let link_end = link_offsets[offsets_base + n.id + 1u];
            for (var j: u32 = link_start; j < link_end; j = j + 1u) {
                let l = cppn_links[link_base + j];
                if (l.enabled == 0u) {
                    continue;
                }
                sum = sum + l.weight * activations[l.from_id];
            }
            activations[n.id] = activate(n.activation, sum);
        }
    }

    // Outputs at nodes[CPPN_INPUTS..CPPN_INPUTS+CPPN_OUTPUTS].
    let n_out0 = cppn_nodes[node_base + CPPN_INPUTS];
    let n_out1 = cppn_nodes[node_base + CPPN_INPUTS + 1u];
    return vec2<f32>(activations[n_out0.id], activations[n_out1.id]);
}

@compute @workgroup_size(64)
fn cppn_from_cppn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let cell_idx = tid / QUERIES_PER_CHILD;
    if (cell_idx >= params.num_children) {
        return;
    }
    let q = tid % QUERIES_PER_CHILD;
    let slot = child_slots[cell_idx];
    let weight_base = slot * WEIGHTS_PER_CELL;

    var inputs: array<f32, 7>;
    var write_offset: u32 = 0u;
    var is_bias: bool = false;
    var apply_gate: bool = true;

    if (q < NUM_W1) {
        // L1 weight: q = h * BRAIN_INPUTS + i
        let h = q / BRAIN_INPUTS;
        let i = q % BRAIN_INPUTS;
        let from_c = substrate_input(i);
        let to_c = substrate_hidden(h);
        inputs[0] = from_c.x; inputs[1] = from_c.y; inputs[2] = from_c.z;
        inputs[3] = to_c.x;   inputs[4] = to_c.y;   inputs[5] = to_c.z;
        inputs[6] = 1.0;
        write_offset = W1_OFFSET + h * BRAIN_INPUTS + i;
    } else if (q < NUM_W1 + NUM_B1) {
        // L1 bias: self-loop sentinel (from = to, marker = 0).
        let h = q - NUM_W1;
        let to_c = substrate_hidden(h);
        inputs[0] = to_c.x; inputs[1] = to_c.y; inputs[2] = to_c.z;
        inputs[3] = to_c.x; inputs[4] = to_c.y; inputs[5] = to_c.z;
        inputs[6] = 0.0;
        write_offset = B1_OFFSET + h;
        is_bias = true;
        apply_gate = false;
    } else if (q < NUM_W1 + NUM_B1 + NUM_W2) {
        // L2 weight: q' = q - NUM_W1 - NUM_B1; o = q' / BRAIN_HIDDEN; h = q' % BRAIN_HIDDEN
        let q2 = q - NUM_W1 - NUM_B1;
        let o = q2 / BRAIN_HIDDEN;
        let h = q2 % BRAIN_HIDDEN;
        let from_c = substrate_hidden(h);
        let to_c = substrate_output(o);
        inputs[0] = from_c.x; inputs[1] = from_c.y; inputs[2] = from_c.z;
        inputs[3] = to_c.x;   inputs[4] = to_c.y;   inputs[5] = to_c.z;
        inputs[6] = 1.0;
        write_offset = W2_OFFSET + o * BRAIN_HIDDEN + h;
    } else {
        // L2 bias.
        let o = q - (NUM_W1 + NUM_B1 + NUM_W2);
        let to_c = substrate_output(o);
        inputs[0] = to_c.x; inputs[1] = to_c.y; inputs[2] = to_c.z;
        inputs[3] = to_c.x; inputs[4] = to_c.y; inputs[5] = to_c.z;
        inputs[6] = 0.0;
        write_offset = B2_OFFSET + o;
        is_bias = true;
        apply_gate = false;
    }

    let out = cppn_forward(cell_idx, inputs);
    var value: f32 = 0.0;
    if (is_bias) {
        value = out.x * 0.5;
        // Vibration emit: default silent — bias the b2[14] slot strongly
        // negative so rectified output starts at 0. Mirrors CPU
        // `Brain::from_cppn`'s post-jitter bias addition.
        if (write_offset == B2_OFFSET + VIBRATION_EMIT_OUTPUT) {
            value = value + INNATE_VIBRATION_EMIT_BIAS;
        }
    } else if (out.y >= CPPN_LINK_EXISTS_THRESHOLD) {
        value = out.x;
    }

    brain_weights[weight_base + write_offset] = value;
}
