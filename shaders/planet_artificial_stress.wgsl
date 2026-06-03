// Monaghan-2000 artificial-stress tensor (S228). Per particle (no
// neighbours): form the total stress σ = −P·I + S, diagonalise it (cyclic
// Jacobi, fixed sweeps for determinism), take the artificial stress in the
// PRINCIPAL frame
//
//   r̂_a = −ε · max(λ_a, 0) / ρ²        (only TENSILE principal stresses)
//
// and rotate back: R̂ = V·diag(r̂)·Vᵀ. Working in the principal frame (not
// the raw xyz diagonal) is what makes R̂ transform as a tensor — the
// objectivity the rotating-block oracle needs. sph_force then applies the
// pairwise (W(r)/W(Δp))^m·(R̂_i+R̂_j) force.
//
// `phase_of` (for the tension-clamp on P) is prepended from
// shaders/planet_phase_common.wgsl.

struct ArtStressParams {
    num_particles: u32,
    pad0: u32, pad1: u32, pad2: u32,
    rho0: f32,
    eos_gamma: f32,
    u_vap: f32,
    c0: f32,
    tait_n: f32,
    t_m: f32,
    l: f32,
    p_tens: f32,
    eps_art: f32,
    melt_coh_frac: f32, pad_c1: f32, pad_c2: f32,
}

@group(0) @binding(0) var<uniform> params: ArtStressParams;
@group(0) @binding(1) var<storage, read> densities: array<f32>;
@group(0) @binding(2) var<storage, read> internal_energies: array<f32>;
@group(0) @binding(3) var<storage, read> dev_stress: array<f32>;
@group(0) @binding(4) var<storage, read_write> art_stress: array<f32>;
@group(0) @binding(5) var<storage, read> mat_rho0: array<f32>;
@group(0) @binding(6) var<storage, read> mat_t_m: array<f32>;

const U_MIN: f32 = 1.0e-6;

fn eos_pressure(rho: f32, u: f32, rho0_p: f32, t_m_p: f32) -> f32 {
    if (u >= params.u_vap) {
        return rho * u * (params.eos_gamma - 1.0);
    }
    let r0 = max(rho0_p, 1e-30);
    let ratio = max(rho / r0, 1e-6);
    let k0 = r0 * params.c0 * params.c0;
    let p_raw = (k0 / params.tait_n) * (pow(ratio, params.tait_n) - 1.0);
    let phi = phase_of(u, t_m_p, params.l).phi;
    let coh = params.melt_coh_frac + (1.0 - params.melt_coh_frac) * phi;
    return max(p_raw, -params.p_tens * coh);
}

@compute @workgroup_size(64)
fn artificial_stress(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let rho = max(densities[i], 1e-30);
    let u = max(internal_energies[i], U_MIN);
    let p = eos_pressure(rho, u, mat_rho0[i], mat_t_m[i]);
    let sb = i * 6u;

    // Total stress σ = −P·I + S (row-major a[r*3+c], symmetric).
    var a: array<f32, 9>;
    a[0] = -p + dev_stress[sb + 0u]; a[4] = -p + dev_stress[sb + 1u]; a[8] = -p + dev_stress[sb + 2u];
    a[1] = dev_stress[sb + 3u]; a[3] = a[1];
    a[2] = dev_stress[sb + 4u]; a[6] = a[2];
    a[5] = dev_stress[sb + 5u]; a[7] = a[5];

    // Eigenvectors accumulate in v (identity init).
    var v: array<f32, 9> = array<f32, 9>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);

    // Cyclic Jacobi: fixed sweeps over the 3 off-diagonal pairs.
    for (var sweep: i32 = 0; sweep < 8; sweep = sweep + 1) {
        var pq: array<u32, 6> = array<u32, 6>(0u, 1u, 0u, 2u, 1u, 2u);
        for (var t: u32 = 0u; t < 3u; t = t + 1u) {
            let pp = pq[t * 2u];
            let qq = pq[t * 2u + 1u];
            let apq = a[pp * 3u + qq];
            if (abs(apq) < 1e-20) { continue; }
            let app = a[pp * 3u + pp];
            let aqq = a[qq * 3u + qq];
            let theta = (aqq - app) / (2.0 * apq);
            let sgn = select(-1.0, 1.0, theta >= 0.0);
            let tt = sgn / (abs(theta) + sqrt(theta * theta + 1.0));
            let c = 1.0 / sqrt(tt * tt + 1.0);
            let s = tt * c;
            // a ← Jᵀ a J : rotate columns pp,qq then rows pp,qq.
            for (var k: u32 = 0u; k < 3u; k = k + 1u) {
                let akp = a[k * 3u + pp];
                let akq = a[k * 3u + qq];
                a[k * 3u + pp] = c * akp - s * akq;
                a[k * 3u + qq] = s * akp + c * akq;
            }
            for (var k: u32 = 0u; k < 3u; k = k + 1u) {
                let apk = a[pp * 3u + k];
                let aqk = a[qq * 3u + k];
                a[pp * 3u + k] = c * apk - s * aqk;
                a[qq * 3u + k] = s * apk + c * aqk;
            }
            for (var k: u32 = 0u; k < 3u; k = k + 1u) {
                let vkp = v[k * 3u + pp];
                let vkq = v[k * 3u + qq];
                v[k * 3u + pp] = c * vkp - s * vkq;
                v[k * 3u + qq] = s * vkp + c * vkq;
            }
        }
    }

    // Principal artificial stress r̂_a = −ε·max(λ_a,0)/ρ²  (λ_a = a[a][a]).
    let inv_rho2 = 1.0 / (rho * rho);
    let r0 = -params.eps_art * max(a[0], 0.0) * inv_rho2;
    let r1 = -params.eps_art * max(a[4], 0.0) * inv_rho2;
    let r2 = -params.eps_art * max(a[8], 0.0) * inv_rho2;

    // R̂_ab = Σ_k V_ak r̂_k V_bk (V columns = eigenvectors).
    let v00 = v[0]; let v01 = v[1]; let v02 = v[2];
    let v10 = v[3]; let v11 = v[4]; let v12 = v[5];
    let v20 = v[6]; let v21 = v[7]; let v22 = v[8];
    art_stress[sb + 0u] = r0 * v00 * v00 + r1 * v01 * v01 + r2 * v02 * v02;
    art_stress[sb + 1u] = r0 * v10 * v10 + r1 * v11 * v11 + r2 * v12 * v12;
    art_stress[sb + 2u] = r0 * v20 * v20 + r1 * v21 * v21 + r2 * v22 * v22;
    art_stress[sb + 3u] = r0 * v00 * v10 + r1 * v01 * v11 + r2 * v02 * v12;
    art_stress[sb + 4u] = r0 * v00 * v20 + r1 * v01 * v21 + r2 * v02 * v22;
    art_stress[sb + 5u] = r0 * v10 * v20 + r1 * v11 * v21 + r2 * v12 * v22;
}
