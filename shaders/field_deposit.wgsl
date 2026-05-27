// Source deposit kernel for FieldGpu. Mirror of `lib::SmellField::deposit`.
//
// Determinism: sources accumulate into a fixed-point integer buffer via
// `atomicAdd`. Integer add is associative, so the per-voxel sum is identical
// regardless of the order threads land — unlike the previous CAS-based
// atomic *float* add, where FP non-associativity + race-ordered CAS made the
// grid (and every sensory gradient derived from it) non-deterministic.
// `field_diffuse` decodes this buffer and folds it into the f32 grid.

// Fixed-point scale: `amount * SCALE` truncated to u32. 2^20 keeps ~6
// decimal digits of precision with headroom against per-voxel overflow.
const DEPOSIT_FIXED_SCALE: f32 = 1048576.0;

struct Params {
    res_x: u32,
    res_y: u32,
    res_z: u32,
    num_sources: u32,
    diffusion: f32,
    decay: f32,
    cell_size_x: f32,
    cell_size_y: f32,
    cell_size_z: f32,
    world_half_x: f32,
    world_half_y: f32,
    world_half_z: f32,
    mask_active: u32,
    flow_active: u32,
    dt: f32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sources: array<vec4<f32>>;          // xyz + amount
// Bindings 2, 3, 4 are unused by deposit but declared so the layout matches
// the shared `field-bgl` (diffuse pipeline needs them). Naga prunes unused
// storage buffers from the SPIR-V so there's no runtime overhead.
@group(0) @binding(2) var<storage, read_write> grid_in_unused: array<u32>;
@group(0) @binding(3) var<storage, read_write> grid_out_unused: array<u32>;
@group(0) @binding(4) var<storage, read> obstacle_mask_unused: array<u32>;
// Fixed-point deposit accumulator (binding 5). Zeroed by the host each step.
@group(0) @binding(5) var<storage, read_write> deposit_accum: array<atomic<u32>>;
// Sprint 202: flow_field declared so the bind group layout matches diffuse;
// not read here. Naga prunes unused storage buffers.
@group(0) @binding(6) var<storage, read> flow_field_unused: array<vec4<f32>>;

@compute @workgroup_size(64)
fn deposit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_sources) {
        return;
    }
    let s = sources[i];
    let pos = s.xyz;
    let amount = s.w;

    let nx = i32(params.res_x);
    let ny = i32(params.res_y);
    let nz = i32(params.res_z);
    let zi = i32(floor((pos.z + params.world_half_z) / params.cell_size_z));
    if (zi < 0 || zi >= nz) {
        return;
    }
    let xi_raw = i32(floor((pos.x + params.world_half_x) / params.cell_size_x));
    let yi_raw = i32(floor((pos.y + params.world_half_y) / params.cell_size_y));
    // Normalize for negative modulo (WGSL i32 % follows truncated division).
    let xi = ((xi_raw % nx) + nx) % nx;
    let yi = ((yi_raw % ny) + ny) % ny;
    let idx = u32(zi * nx * ny + yi * nx + xi);

    // Deterministic accumulation: integer atomicAdd is associative, so the
    // per-voxel sum does not depend on thread race order.
    atomicAdd(&deposit_accum[idx], u32(max(amount, 0.0) * DEPOSIT_FIXED_SCALE));
}
