// Cleary–Monaghan SPH thermal conduction. Per particle i, scans the
// 27 neighbour buckets and accumulates:
//
//   du_i/dt += Σ_j  m_j · κ_ij · (T_i − T_j) · F_ij / (ρ_i ρ_j)
//
// with the standard symmetric form:
//   κ_ij = 4 κ_i κ_j / (κ_i + κ_j)       (uniform-material: κ_ij = 2κ)
//   F_ij = (r_ij · ∇_i W̄_ij) / |r_ij|² · 2   (Wendland-C2 form below)
//
// The kernel uses the *averaged* smoothing length h̄ = (h_i + h_j)/2 so
// that F_ij viewed from i and F_ji viewed from j are equal. Without
// this symmetry pair contributions don't cancel and conduction silently
// injects (or sinks) total energy — see notes in S204.
//
// Sign: dW/dr < 0 for q < 2, so F_ij < 0; for T_i > T_j the contribution
// is negative → i cools. Symmetric (j heats by the same amount).
//
// Reads `u` (binding 5), writes incrementally to `du_dt` (binding 6,
// `+=` not `=`). The integrator clears `du_dt` so accumulation across
// passes within one tick is well-defined: sph_force overwrites with
// `du_visc + du_pdv`; this shader adds conduction on top.

const PI: f32 = 3.14159265358979;
const GRID_N: i32 = 32;
const HALF_N: i32 = 16;
const U_MIN: f32 = 1.0e-6;

struct ConductionParams {
    num_particles: u32,
    pad0: u32, pad1: u32, pad2: u32,
    world_half: f32,
    cell_size: f32,
    kappa: f32,
    inv_cv: f32,
}

@group(0) @binding(0) var<uniform> params: ConductionParams;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> masses: array<f32>;
@group(0) @binding(3) var<storage, read> smoothing_lengths: array<f32>;
@group(0) @binding(4) var<storage, read> densities: array<f32>;
@group(0) @binding(5) var<storage, read> internal_energies: array<f32>;
@group(0) @binding(6) var<storage, read_write> du_dt: array<f32>;
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
fn thermal_conduction(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let xi = vec3<f32>(
        positions[i * 3u + 0u],
        positions[i * 3u + 1u],
        positions[i * 3u + 2u],
    );
    let hi = smoothing_lengths[i];
    let rho_i = max(densities[i], 1e-30);
    let u_i = max(internal_energies[i], U_MIN);
    let t_i = u_i * params.inv_cv;

    // Uniform-material: κ_ij = 2 κ.
    let two_kappa = 2.0 * params.kappa;

    var du_cond: f32 = 0.0;

    let b = bucket_xyz(xi);
    let kernel_coeff = -105.0 / (16.0 * PI);

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

                    let hj = smoothing_lengths[j];
                    let h_bar = 0.5 * (hi + hj);
                    let q = r / h_bar;
                    if (q >= 2.0) { continue; }
                    let h4 = h_bar * h_bar * h_bar * h_bar;

                    let one_minus_half_q = 1.0 - 0.5 * q;
                    let cube = one_minus_half_q * one_minus_half_q * one_minus_half_q;
                    let dW_dr = kernel_coeff * q * cube / h4;
                    // F_ij = 2 · (r_ij · ∇_i W) / r² = 2 · dW/dr / r.
                    // Symmetric in h via h̄, so F_ij = F_ji ⇒ pair-wise
                    // energy conservation holds.
                    let f_ij = 2.0 * dW_dr / r;

                    let mj = masses[j];
                    let rho_j = max(densities[j], 1e-30);
                    let u_j = max(internal_energies[j], U_MIN);
                    let t_j = u_j * params.inv_cv;

                    du_cond = du_cond + mj * two_kappa * (t_i - t_j) * f_ij
                        / (rho_i * rho_j);
                }
            }
        }
    }

    du_dt[i] = du_dt[i] + du_cond;
}
