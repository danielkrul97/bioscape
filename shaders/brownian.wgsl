// Sprint 51: GPU mirror Cell::apply_brownian. Per-cell xoshiro128++ RNG state
// (4× u32) v `xoshiro_state` buffer, mutuje velocities[N×3] adicí
// `gaussian × thermal_noise × sqrt(dt)` na každou složku (z jen pokud
// world_half_z > 0).
//
// xoshiro128++ je deterministic per-cell — stejný state seed → stejná RNG
// sekvence napříč běhy (řeší part-of Sprint 48 GPU determinismus goalu pro
// stochastic phases).

struct BrownianParams {
    num_cells: u32,
    has_z: u32,
    thermal_noise: f32,
    sqrt_dt: f32,
}

@group(0) @binding(0) var<uniform> params: BrownianParams;
@group(0) @binding(1) var<storage, read_write> velocities: array<f32>;
@group(0) @binding(2) var<storage, read_write> xoshiro_state: array<u32>;

fn rotl_u32(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

// Inline xoshiro128++ next. Vrací u32, mutuje state in place.
fn xoshiro_next(state: ptr<function, vec4<u32>>) -> u32 {
    let s = *state;
    let result = rotl_u32(s.x + s.w, 7u) + s.x;
    let t = s.y << 9u;
    var s2 = s.z ^ s.x;
    var s3 = s.w ^ s.y;
    let s1 = s.y ^ s2;
    let s0 = s.x ^ s3;
    s2 = s2 ^ t;
    s3 = rotl_u32(s3, 11u);
    *state = vec4<u32>(s0, s1, s2, s3);
    return result;
}

fn uniform01(state: ptr<function, vec4<u32>>) -> f32 {
    let bits = xoshiro_next(state);
    // 24-bit mantisa precision; shift na [0, 1).
    return f32(bits >> 8u) * (1.0 / 16777216.0);
}

// Box-Muller gaussian. epsilon na u1 brání log(0).
fn gaussian(state: ptr<function, vec4<u32>>) -> f32 {
    let u1 = max(uniform01(state), 1.1920929e-7);
    let u2 = uniform01(state);
    return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}

@compute @workgroup_size(64)
fn brownian(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }
    var s = vec4<u32>(
        xoshiro_state[i * 4u + 0u],
        xoshiro_state[i * 4u + 1u],
        xoshiro_state[i * 4u + 2u],
        xoshiro_state[i * 4u + 3u],
    );
    let scale = params.thermal_noise * params.sqrt_dt;
    let g_x = gaussian(&s);
    let g_y = gaussian(&s);
    velocities[i * 3u + 0u] = velocities[i * 3u + 0u] + g_x * scale;
    velocities[i * 3u + 1u] = velocities[i * 3u + 1u] + g_y * scale;
    if (params.has_z != 0u) {
        let g_z = gaussian(&s);
        velocities[i * 3u + 2u] = velocities[i * 3u + 2u] + g_z * scale;
    }
    xoshiro_state[i * 4u + 0u] = s.x;
    xoshiro_state[i * 4u + 1u] = s.y;
    xoshiro_state[i * 4u + 2u] = s.z;
    xoshiro_state[i * 4u + 3u] = s.w;
}
