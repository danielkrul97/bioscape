// Source deposit kernel for FieldGpu. Mirror of `lib::SmellField::deposit`.
// Splits from the diffuse path so this is the only place that needs atomic
// types on `grid_in` — the diffuse shader treats the same buffer as plain
// `array<u32>` and saves 6× `atomicLoad` per voxel per pass.

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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sources: array<vec4<f32>>;          // xyz + amount
@group(0) @binding(2) var<storage, read_write> grid_in: array<atomic<u32>>;
// Bindings 3 + 4 are unused by deposit but declared so the layout matches
// the shared `field-bgl` (diffuse pipeline needs both). Naga prunes unused
// storage buffers from the SPIR-V so there's no runtime overhead.
@group(0) @binding(3) var<storage, read_write> grid_out_unused: array<u32>;
@group(0) @binding(4) var<storage, read> obstacle_mask_unused: array<u32>;

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

    // Atomic float add via CAS — no native atomic<f32> in WGSL.
    var old_bits: u32 = atomicLoad(&grid_in[idx]);
    loop {
        let new_bits: u32 = bitcast<u32>(bitcast<f32>(old_bits) + amount);
        let res = atomicCompareExchangeWeak(&grid_in[idx], old_bits, new_bits);
        if (res.exchanged) {
            break;
        }
        old_bits = res.old_value;
    }
}
