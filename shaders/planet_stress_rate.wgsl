// Deviatoric stress rate (Gray–Monaghan–Swift 2001), S225. Per particle i:
//
//   corrected velocity gradient  L_ab = Σ_j V_j (v_j − v_i)_a (B_i ∇_iW_ij)_b
//   strain rate   ε̇ = ½(L + Lᵀ)          spin  R = ½(L − Lᵀ)
//   dS/dt = 2G·dev(ε̇) + (R·S − S·R)       (Jaumann corotational rate)
//
// B_i is the Bonet–Lok correction matrix (read from grad_corr) making the
// gradient first-order exact, so a rigidly rotating block (ε̇ = 0) produces
// dS/dt = 0 and develops no spurious stress. G is the constant shear
// modulus G0 in S225 (phase-gated G0·φ² lands in S227). The rotation term
// vanishes when S = 0, so the S225 oracles (which start unstressed) test
// the strain-rate + Hooke path directly.
//
// Writes ds_dt[6] = [Sxx,Syy,Szz,Sxy,Sxz,Syz] rate (OVERWRITE). Neighbour
// data is read-only; per-i accumulation over the deterministic walk.

const PI: f32 = 3.14159265358979;
const GRID_N: i32 = 32;
const HALF_N: i32 = 16;

struct StressRateParams {
    num_particles: u32,
    pad0: u32, pad1: u32, pad2: u32,
    world_half: f32,
    cell_size: f32,
    g0: f32,
    t_m: f32,
    l: f32,
    pad_b0: f32, pad_b1: f32, pad_b2: f32,
}

@group(0) @binding(0) var<uniform> params: StressRateParams;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> masses: array<f32>;
@group(0) @binding(4) var<storage, read> smoothing_lengths: array<f32>;
@group(0) @binding(5) var<storage, read> densities: array<f32>;
@group(0) @binding(6) var<storage, read> dev_stress: array<f32>;
@group(0) @binding(7) var<storage, read> grad_corr: array<f32>;
@group(0) @binding(8) var<storage, read_write> ds_dt: array<f32>;
@group(0) @binding(9) var<storage, read> hash_offsets: array<u32>;
@group(0) @binding(10) var<storage, read> sorted_particles: array<u32>;
@group(0) @binding(11) var<storage, read> internal_energies: array<f32>;
@group(0) @binding(12) var<storage, read> mat_t_m: array<f32>;

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
fn stress_rate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let xi = vec3<f32>(positions[i].x, positions[i].y, positions[i].z);
    let vi = vec3<f32>(velocities[i].x, velocities[i].y, velocities[i].z);
    let hi = smoothing_lengths[i];
    let h4 = hi * hi * hi * hi;
    let kernel_coeff = -105.0 / (16.0 * PI);

    let gb = i * 9u;
    let brow0 = vec3<f32>(grad_corr[gb + 0u], grad_corr[gb + 1u], grad_corr[gb + 2u]);
    let brow1 = vec3<f32>(grad_corr[gb + 3u], grad_corr[gb + 4u], grad_corr[gb + 5u]);
    let brow2 = vec3<f32>(grad_corr[gb + 6u], grad_corr[gb + 7u], grad_corr[gb + 8u]);

    // Velocity gradient L_ab (row a = velocity component, col b = space).
    var l00: f32 = 0.0; var l01: f32 = 0.0; var l02: f32 = 0.0;
    var l10: f32 = 0.0; var l11: f32 = 0.0; var l12: f32 = 0.0;
    var l20: f32 = 0.0; var l21: f32 = 0.0; var l22: f32 = 0.0;

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
                    let xj = vec3<f32>(positions[j].x, positions[j].y, positions[j].z);
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
                    let gradw = vec3<f32>(dW_dr * dvec.x * inv_r, dW_dr * dvec.y * inv_r, dW_dr * dvec.z * inv_r);

                    // Corrected gradient g̃ = B_i · ∇W.
                    let gt = vec3<f32>(dot(brow0, gradw), dot(brow1, gradw), dot(brow2, gradw));

                    let vj = vec3<f32>(velocities[j].x, velocities[j].y, velocities[j].z);
                    let dv = vj - vi;
                    let volj = masses[j] / max(densities[j], 1e-30);

                    l00 = l00 + volj * dv.x * gt.x; l01 = l01 + volj * dv.x * gt.y; l02 = l02 + volj * dv.x * gt.z;
                    l10 = l10 + volj * dv.y * gt.x; l11 = l11 + volj * dv.y * gt.y; l12 = l12 + volj * dv.y * gt.z;
                    l20 = l20 + volj * dv.z * gt.x; l21 = l21 + volj * dv.z * gt.y; l22 = l22 + volj * dv.z * gt.z;
                }
            }
        }
    }

    // Strain rate (symmetric) and spin (antisymmetric).
    let exx = l00; let eyy = l11; let ezz = l22;
    let exy = 0.5 * (l01 + l10);
    let exz = 0.5 * (l02 + l20);
    let eyz = 0.5 * (l12 + l21);
    let r01 = 0.5 * (l01 - l10);
    let r02 = 0.5 * (l02 - l20);
    let r12 = 0.5 * (l12 - l21);

    // Phase-gated shear modulus G_i = G0·φ² (S227): mush is soft, liquid
    // (φ = 0) carries no shear stiffness.
    let phi_i = phase_of(internal_energies[i], mat_t_m[i], params.l).phi;
    let tr3 = (exx + eyy + ezz) / 3.0;
    let two_g = 2.0 * params.g0 * phi_i * phi_i;
    let dxx = two_g * (exx - tr3);
    let dyy = two_g * (eyy - tr3);
    let dzz = two_g * (ezz - tr3);
    let dxy = two_g * exy;
    let dxz = two_g * exz;
    let dyz = two_g * eyz;

    // Jaumann rotation term rot = R·S − S·R (symmetric). Column-major mats.
    let sb = i * 6u;
    let sxx = dev_stress[sb + 0u]; let syy = dev_stress[sb + 1u]; let szz = dev_stress[sb + 2u];
    let sxy = dev_stress[sb + 3u]; let sxz = dev_stress[sb + 4u]; let syz = dev_stress[sb + 5u];
    let smat = mat3x3<f32>(
        vec3<f32>(sxx, sxy, sxz),
        vec3<f32>(sxy, syy, syz),
        vec3<f32>(sxz, syz, szz),
    );
    let rmat = mat3x3<f32>(
        vec3<f32>(0.0, -r01, -r02),
        vec3<f32>(r01, 0.0, -r12),
        vec3<f32>(r02, r12, 0.0),
    );
    let rotm = rmat * smat - smat * rmat;

    ds_dt[sb + 0u] = dxx + rotm[0][0];
    ds_dt[sb + 1u] = dyy + rotm[1][1];
    ds_dt[sb + 2u] = dzz + rotm[2][2];
    ds_dt[sb + 3u] = dxy + rotm[1][0];
    ds_dt[sb + 4u] = dxz + rotm[2][0];
    ds_dt[sb + 5u] = dyz + rotm[2][1];
}
