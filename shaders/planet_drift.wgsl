// Leapfrog drift: x ← x + dt · v. Called once per KDK step between
// the two half-kicks.

struct DriftParams {
    num_particles: u32,
    pad0: u32, pad1: u32, pad2: u32,
    dt: f32,
    pad_a0: f32, pad_a1: f32, pad_a2: f32,
}

@group(0) @binding(0) var<uniform> params: DriftParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4<f32>>;

@compute @workgroup_size(64)
fn drift(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }
    let dt = params.dt;
    positions[i].x = positions[i].x + dt * velocities[i].x;
    positions[i].y = positions[i].y + dt * velocities[i].y;
    positions[i].z = positions[i].z + dt * velocities[i].z;
}
