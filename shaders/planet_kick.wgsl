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
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> accelerations: array<f32>;

@compute @workgroup_size(64)
fn kick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }
    let h = params.dt_half;
    velocities[i].x = velocities[i].x + h * accelerations[i * 3u + 0u];
    velocities[i].y = velocities[i].y + h * accelerations[i * 3u + 1u];
    velocities[i].z = velocities[i].z + h * accelerations[i * 3u + 2u];
}
