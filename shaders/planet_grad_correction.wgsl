// Bonet–Lok kernel-gradient correction matrix (S225). Per particle i:
//
//   M_i = Σ_j V_j (x_j − x_i) ⊗ ∇_i W_ij,    V_j = m_j / ρ_j
//   B_i = (M_i + λI)^-1
//
// The corrected gradient B_i ∇_i W_ij reproduces a constant strain field
// exactly, so a rigidly rotating/orbiting solid develops no spurious
// deviatoric stress (the objectivity property the stress model needs).
// λ is added UNCONDITIONALLY (Tikhonov) rather than branching on det≈0,
// so surface/rank-deficient particles stay finite and deterministic
// across platforms. M is symmetric PSD (sum of V·(x⊗x)·(−dW/dr / r)), so
// M + λI is SPD and invertible.
//
// Writes the 3×3 B row-major (9 floats per particle). Same Wendland-C2
// gradient + h_i convention as planet_sph_force.

const PI: f32 = 3.14159265358979;
const GRID_N: i32 = 32;
const HALF_N: i32 = 16;

struct GradCorrParams {
    num_particles: u32,
    pad0: u32, pad1: u32, pad2: u32,
    world_half: f32,
    cell_size: f32,
    lambda: f32,
    pad_a0: f32,
}

@group(0) @binding(0) var<uniform> params: GradCorrParams;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> masses: array<f32>;
@group(0) @binding(3) var<storage, read> smoothing_lengths: array<f32>;
@group(0) @binding(4) var<storage, read> densities: array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_corr: array<f32>;
@group(0) @binding(6) var<storage, read> hash_offsets: array<u32>;
@group(0) @binding(7) var<storage, read> sorted_particles: array<u32>;

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
fn grad_correction(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let xi = vec3<f32>(
        positions[i].x,
        positions[i].y,
        positions[i].z,
    );
    let hi = smoothing_lengths[i];
    let h4 = hi * hi * hi * hi;
    let kernel_coeff = -105.0 / (16.0 * PI);

    // Row-major accumulation: m[r][c].
    var m00: f32 = 0.0; var m01: f32 = 0.0; var m02: f32 = 0.0;
    var m10: f32 = 0.0; var m11: f32 = 0.0; var m12: f32 = 0.0;
    var m20: f32 = 0.0; var m21: f32 = 0.0; var m22: f32 = 0.0;

    let b = bucket_xyz(xi);
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
                        positions[j].x,
                        positions[j].y,
                        positions[j].z,
                    );
                    let dvec = xi - xj;
                    let r2 = dot(dvec, dvec);
                    if (r2 < 1e-20) { continue; }
                    let r = sqrt(r2);
                    let q = r / hi;
                    if (q >= 2.0) { continue; }

                    let one_minus_half_q = 1.0 - 0.5 * q;
                    let cube = one_minus_half_q * one_minus_half_q * one_minus_half_q;
                    let dW_dr = kernel_coeff * q * cube / h4;
                    let inv_r = 1.0 / r;
                    let grad = vec3<f32>(dW_dr * dvec.x * inv_r, dW_dr * dvec.y * inv_r, dW_dr * dvec.z * inv_r);

                    let vj = masses[j] / max(densities[j], 1e-30);
                    let dxji = -dvec; // x_j − x_i
                    m00 = m00 + vj * dxji.x * grad.x;
                    m01 = m01 + vj * dxji.x * grad.y;
                    m02 = m02 + vj * dxji.x * grad.z;
                    m10 = m10 + vj * dxji.y * grad.x;
                    m11 = m11 + vj * dxji.y * grad.y;
                    m12 = m12 + vj * dxji.y * grad.z;
                    m20 = m20 + vj * dxji.z * grad.x;
                    m21 = m21 + vj * dxji.z * grad.y;
                    m22 = m22 + vj * dxji.z * grad.z;
                }
            }
        }
    }

    // Tikhonov regularisation.
    let lam = params.lambda;
    m00 = m00 + lam; m11 = m11 + lam; m22 = m22 + lam;

    // 3×3 inverse via cofactors.
    let c00 = m11 * m22 - m12 * m21;
    let c01 = m12 * m20 - m10 * m22;
    let c02 = m10 * m21 - m11 * m20;
    let det = m00 * c00 + m01 * c01 + m02 * c02;

    var b00: f32 = 1.0; var b01: f32 = 0.0; var b02: f32 = 0.0;
    var b10: f32 = 0.0; var b11: f32 = 1.0; var b12: f32 = 0.0;
    var b20: f32 = 0.0; var b21: f32 = 0.0; var b22: f32 = 1.0;
    if (abs(det) > 1e-20) {
        let inv_det = 1.0 / det;
        b00 = c00 * inv_det;
        b01 = (m02 * m21 - m01 * m22) * inv_det;
        b02 = (m01 * m12 - m02 * m11) * inv_det;
        b10 = c01 * inv_det;
        b11 = (m00 * m22 - m02 * m20) * inv_det;
        b12 = (m02 * m10 - m00 * m12) * inv_det;
        b20 = c02 * inv_det;
        b21 = (m01 * m20 - m00 * m21) * inv_det;
        b22 = (m00 * m11 - m01 * m10) * inv_det;
    }

    let base = i * 9u;
    grad_corr[base + 0u] = b00; grad_corr[base + 1u] = b01; grad_corr[base + 2u] = b02;
    grad_corr[base + 3u] = b10; grad_corr[base + 4u] = b11; grad_corr[base + 5u] = b12;
    grad_corr[base + 6u] = b20; grad_corr[base + 7u] = b21; grad_corr[base + 8u] = b22;
}
