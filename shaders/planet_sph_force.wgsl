// Merged SPH non-gravity force: polytropic pressure + Monaghan
// artificial viscosity in a single neighbour scan. Replaces the
// previously separate `planet_pressure.wgsl` and `planet_viscosity.wgsl`.
//
// Per neighbour j (one bucket scan per particle i):
//   1. compute r, q, Wendland-C2 gradient `∇_i W_ij` — needed by both terms
//   2. pressure: dv/dt += −m_j (P_i/ρ_i² + P_j/ρ_j²) ∇_i W_ij
//   3. viscosity (only if v_ij·r_ij < 0):
//          μ_ij = h̄ (v_ij·r_ij) / (r² + 0.01 h̄²)
//          Π_ij = (−α c̄ μ_ij + β μ_ij²) / ρ̄
//          dv/dt += −m_j Π_ij ∇_i W_ij
//   4. sound speed `c = √(γ P / ρ)` reuses `P_j` from the pressure step
//      (one pow per neighbour instead of two).
//
// Polytropic EOS: `P = K · ρ^γ`. Kernel: Wendland C2 with `h_i` (Newton's
// 3rd law holds for equal h, see notes in `planet_pressure.wgsl`).
// Contribution is **added** to `accelerations`; caller writes gravity
// first.

const PI: f32 = 3.14159265358979;
const GRID_N: i32 = 32;
const HALF_N: i32 = 16;
const VISC_EPS: f32 = 0.01;

struct SphForceParams {
    num_particles: u32,
    pad0: u32, pad1: u32, pad2: u32,
    world_half: f32,
    cell_size: f32,
    eos_k: f32,
    eos_gamma: f32,
    alpha: f32,
    beta: f32,
    pad_a0: f32, pad_a1: f32,
}

@group(0) @binding(0) var<uniform> params: SphForceParams;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> velocities: array<f32>;
@group(0) @binding(3) var<storage, read> masses: array<f32>;
@group(0) @binding(4) var<storage, read> smoothing_lengths: array<f32>;
@group(0) @binding(5) var<storage, read> densities: array<f32>;
@group(0) @binding(6) var<storage, read_write> accelerations: array<f32>;
@group(0) @binding(7) var<storage, read> hash_offsets: array<u32>;
@group(0) @binding(8) var<storage, read> sorted_particles: array<u32>;

fn bucket_xyz(pos: vec3<f32>) -> vec3<i32> {
    let bx = i32(floor(pos.x / params.cell_size)) + HALF_N;
    let by = i32(floor(pos.y / params.cell_size)) + HALF_N;
    let bz = i32(floor(pos.z / params.cell_size)) + HALF_N;
    return vec3<i32>(
        clamp(bx, 0, GRID_N - 1),
        clamp(by, 0, GRID_N - 1),
        clamp(bz, 0, GRID_N - 1),
    );
}

@compute @workgroup_size(64)
fn sph_force(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let xi = vec3<f32>(
        positions[i * 3u + 0u],
        positions[i * 3u + 1u],
        positions[i * 3u + 2u],
    );
    let vi = vec3<f32>(
        velocities[i * 3u + 0u],
        velocities[i * 3u + 1u],
        velocities[i * 3u + 2u],
    );
    let hi = smoothing_lengths[i];
    let rho_i = max(densities[i], 1e-30);
    let p_i = params.eos_k * pow(rho_i, params.eos_gamma);
    let inv_rho_i2 = 1.0 / (rho_i * rho_i);
    let c_i = sqrt(params.eos_gamma * p_i / rho_i);

    var ax: f32 = 0.0;
    var ay: f32 = 0.0;
    var az: f32 = 0.0;

    let b = bucket_xyz(xi);
    let kernel_coeff = -105.0 / (16.0 * PI);
    let h4 = hi * hi * hi * hi;

    for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            for (var dx_b: i32 = -1; dx_b <= 1; dx_b = dx_b + 1) {
                let nbx = clamp(b.x + dx_b, 0, GRID_N - 1);
                let nby = clamp(b.y + dy, 0, GRID_N - 1);
                let nbz = clamp(b.z + dz, 0, GRID_N - 1);
                let bucket = u32(nbx + nby * GRID_N + nbz * GRID_N * GRID_N);
                let start = hash_offsets[bucket];
                let end = hash_offsets[bucket + 1u];
                for (var slot: u32 = start; slot < end; slot = slot + 1u) {
                    let j = sorted_particles[slot];
                    if (j == i) { continue; }
                    let xj = vec3<f32>(
                        positions[j * 3u + 0u],
                        positions[j * 3u + 1u],
                        positions[j * 3u + 2u],
                    );
                    let dvec = xi - xj;
                    let r2 = dot(dvec, dvec);
                    if (r2 < 1e-20) { continue; }
                    let r = sqrt(r2);
                    let q = r / hi;
                    if (q >= 2.0) { continue; }

                    // Wendland C2 radial gradient + direction. Shared by
                    // both pressure and viscosity contributions.
                    let one_minus_half_q = 1.0 - 0.5 * q;
                    let cube = one_minus_half_q * one_minus_half_q * one_minus_half_q;
                    let dW_dr = kernel_coeff * q * cube / h4;
                    let inv_r = 1.0 / r;
                    let grad_x = dW_dr * dvec.x * inv_r;
                    let grad_y = dW_dr * dvec.y * inv_r;
                    let grad_z = dW_dr * dvec.z * inv_r;

                    // Shared neighbour loads — pressure needs ρ_j and m_j,
                    // viscosity needs the same plus v_j and h_j.
                    let mj = masses[j];
                    let rho_j = max(densities[j], 1e-30);
                    let p_j = params.eos_k * pow(rho_j, params.eos_gamma);

                    // Pressure: always contributes.
                    let inv_rho_j2 = 1.0 / (rho_j * rho_j);
                    var factor = mj * (p_i * inv_rho_i2 + p_j * inv_rho_j2);

                    // Viscosity: only for approaching pairs.
                    let vj = vec3<f32>(
                        velocities[j * 3u + 0u],
                        velocities[j * 3u + 1u],
                        velocities[j * 3u + 2u],
                    );
                    let v_rel = vi - vj;
                    let v_dot_r = dot(v_rel, dvec);
                    if (v_dot_r < 0.0) {
                        let hj = smoothing_lengths[j];
                        let h_bar = 0.5 * (hi + hj);
                        let mu = h_bar * v_dot_r / (r2 + VISC_EPS * h_bar * h_bar);
                        let c_j = sqrt(params.eos_gamma * p_j / rho_j);
                        let c_bar = 0.5 * (c_i + c_j);
                        let rho_bar = 0.5 * (rho_i + rho_j);
                        let pi_ij = (-params.alpha * c_bar * mu + params.beta * mu * mu)
                            / max(rho_bar, 1e-30);
                        factor = factor + mj * pi_ij;
                    }

                    ax = ax - factor * grad_x;
                    ay = ay - factor * grad_y;
                    az = az - factor * grad_z;
                }
            }
        }
    }

    accelerations[i * 3u + 0u] = accelerations[i * 3u + 0u] + ax;
    accelerations[i * 3u + 1u] = accelerations[i * 3u + 1u] + ay;
    accelerations[i * 3u + 2u] = accelerations[i * 3u + 2u] + az;
}
