// Leapfrog half-kick: v ← v + dt_half · a.
//
// Dispatched twice per step (KDK): once with a_old before drift, once
// with a_new after drift. `dt_half = dt / 2` is computed CPU-side and
// uploaded to params.

struct KickParams {
    num_particles: u32,
    pad0: u32, pad1: u32, pad2: u32,
    dt_half: f32,
    pad_a0: f32, pad_a1: f32, pad_a2: f32,
}

@group(0) @binding(0) var<uniform> params: KickParams;
@group(0) @binding(1) var<storage, read_write> velocities: array<f32>;
@group(0) @binding(2) var<storage, read> accelerations: array<f32>;

@compute @workgroup_size(64)
fn kick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }
    let h = params.dt_half;
    velocities[i * 3u + 0u] = velocities[i * 3u + 0u] + h * accelerations[i * 3u + 0u];
    velocities[i * 3u + 1u] = velocities[i * 3u + 1u] + h * accelerations[i * 3u + 1u];
    velocities[i * 3u + 2u] = velocities[i * 3u + 2u] + h * accelerations[i * 3u + 2u];
}
