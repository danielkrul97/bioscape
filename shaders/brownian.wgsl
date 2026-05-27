// Sprint 51: GPU mirror of Cell::apply_brownian. Per-cell xoshiro128++ RNG
// state (vec4<u32>) mutates velocities[N×3] by adding
// gaussian × thermal_noise × sqrt(dt) / sqrt(mass) on each axis (z only when
// has_z != 0).
//
// Sprint 202: Brownian impulse scaled by 1/sqrt(mass) — small cells get
// more wobble, large cells stay still. Matches the equipartition theorem
// for thermal motion (⟨½mv²⟩ = ½kT → ⟨v²⟩ ∝ 1/m), where larger inertia
// damps the same heat-bath kick into a smaller velocity excursion.
//
// Deterministic per-cell: same seed → same sequence across runs
// (part of Sprint 48 GPU determinism goal for stochastic phases).

struct BrownianParams {
    num_cells: u32,
    has_z: u32,
    thermal_noise: f32,
    sqrt_dt: f32,
}

@group(0) @binding(0) var<uniform> params: BrownianParams;
@group(0) @binding(1) var<storage, read_write> velocities: array<f32>;
@group(0) @binding(2) var<storage, read_write> xoshiro_state: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> masses: array<f32>;

fn rotl_u32(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

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

// Build a float in [1, 2) via IEEE-754 bit pattern (sign=0, exp=127,
// mantissa from RNG), then subtract 1. Avoids the FP multiply of the
// classic `bits * 2^-24` form. 23-bit precision in [0, 1).
fn uniform01(state: ptr<function, vec4<u32>>) -> f32 {
    let bits = xoshiro_next(state);
    return bitcast<f32>((bits >> 9u) | 0x3F800000u) - 1.0;
}

// Box-Muller: two independent gaussians per pair of uniforms.
// epsilon clamp on u1 prevents log(0) → -inf.
fn gaussian_pair(state: ptr<function, vec4<u32>>) -> vec2<f32> {
    let u1 = max(uniform01(state), 1.1920929e-7);
    let u2 = uniform01(state);
    let r = sqrt(-2.0 * log(u1));
    let theta = 6.28318530718 * u2;
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

@compute @workgroup_size(64)
fn brownian(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_cells) {
        return;
    }
    var s = xoshiro_state[i];
    let inv_sqrt_mass = inverseSqrt(max(masses[i], 0.01));
    let scale = params.thermal_noise * params.sqrt_dt * inv_sqrt_mass;

    let g_xy = gaussian_pair(&s);
    velocities[i * 3u + 0u] = velocities[i * 3u + 0u] + g_xy.x * scale;
    velocities[i * 3u + 1u] = velocities[i * 3u + 1u] + g_xy.y * scale;

    if (params.has_z != 0u) {
        // Second pair yields two gaussians; we use one and drop the other
        // to keep the RNG sequence stateless across dispatches.
        let g_z = gaussian_pair(&s);
        velocities[i * 3u + 2u] = velocities[i * 3u + 2u] + g_z.x * scale;
    }

    xoshiro_state[i] = s;
}