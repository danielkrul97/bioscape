// Merged SPH non-gravity force: ideal-gas pressure + Monaghan
// artificial viscosity + adiabatic compression + viscous heating in a
// single neighbour scan.
//
// EOS (S224): phase-selected — ideal gas `P = ρ u (γ−1)` for `u ≥ u_vap`,
// else condensed Tait/Murnaghan `P = (K0/n)((ρ/ρ0)ⁿ − 1)` clamped to
// `P ≥ 0`. See `eos_pc` below. Reduces to the pre-S224 ideal gas when all
// `u ≥ u_vap`. Sound speed comes from the same branch.
//
// Per neighbour j:
//   1. compute r, q, Wendland-C2 gradient `∇_i W_ij` — shared by all
//   2. pressure: dv/dt += −m_j (P_i/ρ_i² + P_j/ρ_j²) ∇_i W_ij
//   3. adiabatic heating (always):
//          du_i/dt += (P_i/ρ_i²) m_j (v_i − v_j) · ∇_i W_ij
//      (derived from −P/ρ · ∇·v with SPH continuity; compression heats,
//       expansion cools.)
//   4. viscosity (only if v_ij·r_ij < 0):
//          μ_ij = h̄ (v_ij·r_ij) / (r² + 0.01 h̄²)
//          Π_ij = (−α c̄ μ_ij + β μ_ij²) / ρ̄
//          dv/dt += −m_j Π_ij ∇_i W_ij
//          du_i/dt += ½ m_j Π_ij (v_i − v_j) · ∇_i W_ij     (Monaghan 1992)
//
// Kernel: Wendland C2 with `h_i`.
// `accelerations` (binding 6) is **added** to (gravity wrote first);
// `du_dt` (binding 10) is **overwritten** so the integrator sees only
// this tick's source terms.

const PI: f32 = 3.14159265358979;
const GRID_N: i32 = 32;
const HALF_N: i32 = 16;
const VISC_EPS: f32 = 0.01;
const U_MIN: f32 = 1.0e-6;
// Monaghan-2000 artificial-stress kernel-ratio exponent + SPH spacing
// ratio. Mirror of thermal::ARTIFICIAL_STRESS_EXPONENT_M / SPH_SMOOTHING_ETA.
// NOTE: the force loop inlines M_ART = 4 as (ratio²)² — keep them in sync.
const M_ART: f32 = 4.0;
const ETA_ART: f32 = 1.3;

fn wendland_w(r: f32, h: f32) -> f32 {
    let q = r / h;
    if (q >= 2.0) { return 0.0; }
    let h3 = h * h * h;
    let omq = 1.0 - 0.5 * q;
    let f = omq * omq * omq * omq;
    return (21.0 / (16.0 * PI * h3)) * f * (1.0 + 2.0 * q);
}

struct SphForceParams {
    num_particles: u32,
    pad0: u32, pad1: u32, pad2: u32,
    world_half: f32,
    cell_size: f32,
    rho0: f32,
    eos_gamma: f32,
    alpha: f32,
    beta: f32,
    u_vap: f32,
    c0: f32,
    tait_n: f32,
    t_m: f32,
    l: f32,
    p_tens: f32,
}

// Pressure `P` and sound speed `c` are precomputed per particle by the EoS
// pass (planet_eos.wgsl) — phase-selected ideal-gas / condensed Tait, with
// the per-material ρ0 / T_m (S232) and the S228 cohesion clamp baked in — and
// read here from `pressure` / `sound_speed`. This removes the two `pow` that
// the old inline `eos_pc` evaluated for every neighbour pair.

@group(0) @binding(0) var<uniform> params: SphForceParams;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> masses: array<f32>;
@group(0) @binding(4) var<storage, read> smoothing_lengths: array<f32>;
@group(0) @binding(5) var<storage, read> densities: array<f32>;
@group(0) @binding(6) var<storage, read_write> accelerations: array<f32>;
@group(0) @binding(7) var<storage, read> hash_offsets: array<u32>;
@group(0) @binding(8) var<storage, read> sorted_particles: array<u32>;
@group(0) @binding(9) var<storage, read> internal_energies: array<f32>;
@group(0) @binding(10) var<storage, read_write> du_dt: array<f32>;
@group(0) @binding(11) var<storage, read> dev_stress: array<f32>;
@group(0) @binding(12) var<storage, read> art_stress: array<f32>;
@group(0) @binding(13) var<storage, read> mat_rho0: array<f32>;
@group(0) @binding(14) var<storage, read> mat_t_m: array<f32>;
@group(0) @binding(15) var<storage, read> pressure: array<f32>;
@group(0) @binding(16) var<storage, read> sound_speed: array<f32>;

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
        positions[i].x,
        positions[i].y,
        positions[i].z,
    );
    let vi = vec3<f32>(
        velocities[i].x,
        velocities[i].y,
        velocities[i].z,
    );
    let hi = smoothing_lengths[i];
    let rho_i = max(densities[i], 1e-30);
    let u_i = max(internal_energies[i], U_MIN);
    let tm_i = mat_t_m[i];
    let p_i = pressure[i];
    let inv_rho_i2 = 1.0 / (rho_i * rho_i);
    let c_i = sound_speed[i];

    // Deviatoric stress of i (S226). σ = −P·I + S; the −P part is the
    // existing pressure factor, S adds the shear-bearing contraction.
    let sbi = i * 6u;
    let sixx = dev_stress[sbi + 0u]; let siyy = dev_stress[sbi + 1u]; let sizz = dev_stress[sbi + 2u];
    let sixy = dev_stress[sbi + 3u]; let sixz = dev_stress[sbi + 4u]; let siyz = dev_stress[sbi + 5u];
    let phi_i = phase_of(u_i, tm_i, params.l).phi;
    // Adiabatic pdV heating applies only where the EoS is THERMAL (gas
    // branch, P depends on u). The condensed Tait branch is barotropic
    // (P = P(ρ)), so its compression work is recoverable cold-curve elastic
    // energy carried conservatively by the force — dumping it into thermal
    // `u` would spuriously heat (and melt) a cold solid. (S231/S232 fix.)
    let gas_i = select(0.0, 1.0, u_i >= params.u_vap);

    // Artificial-stress tensor of i (S228) + the reference kernel value at
    // particle spacing Δp = h_i/η, for the Monaghan-2000 (W(r)/W(Δp))^m ratio.
    let raixx = art_stress[sbi + 0u]; let raiyy = art_stress[sbi + 1u]; let raizz = art_stress[sbi + 2u];
    let raixy = art_stress[sbi + 3u]; let raixz = art_stress[sbi + 4u]; let raiyz = art_stress[sbi + 5u];
    let w_dp = wendland_w(hi / ETA_ART, hi);

    var ax: f32 = 0.0;
    var ay: f32 = 0.0;
    var az: f32 = 0.0;
    var du_visc: f32 = 0.0;
    var du_pdv: f32 = 0.0;

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

                    // Wendland C2 radial gradient + direction.
                    let one_minus_half_q = 1.0 - 0.5 * q;
                    let cube = one_minus_half_q * one_minus_half_q * one_minus_half_q;
                    let dW_dr = kernel_coeff * q * cube / h4;
                    let inv_r = 1.0 / r;
                    let grad_x = dW_dr * dvec.x * inv_r;
                    let grad_y = dW_dr * dvec.y * inv_r;
                    let grad_z = dW_dr * dvec.z * inv_r;

                    let mj = masses[j];
                    let rho_j = max(densities[j], 1e-30);
                    let u_j = max(internal_energies[j], U_MIN);
                    let p_j = pressure[j];

                    // Pressure: always contributes.
                    let inv_rho_j2 = 1.0 / (rho_j * rho_j);
                    var factor = mj * (p_i * inv_rho_i2 + p_j * inv_rho_j2);

                    // Relative velocity is shared by adiabatic pdV and viscosity.
                    let vj = vec3<f32>(
                        velocities[j].x,
                        velocities[j].y,
                        velocities[j].z,
                    );
                    let v_rel = vi - vj;
                    let v_dot_grad = v_rel.x * grad_x + v_rel.y * grad_y + v_rel.z * grad_z;

                    // Adiabatic pdV heating. Positive on compression
                    // (v_rel · ∇W > 0) ⇒ warms; negative on expansion.
                    du_pdv = du_pdv + gas_i * p_i * inv_rho_i2 * mj * v_dot_grad;

                    // Viscosity: only for approaching pairs.
                    let v_dot_r = dot(v_rel, dvec);
                    if (v_dot_r < 0.0) {
                        let hj = smoothing_lengths[j];
                        let h_bar = 0.5 * (hi + hj);
                        let mu = h_bar * v_dot_r / (r2 + VISC_EPS * h_bar * h_bar);
                        let c_j = sound_speed[j];
                        let c_bar = 0.5 * (c_i + c_j);
                        let rho_bar = 0.5 * (rho_i + rho_j);
                        // Phase-scaled AV (S227): solid pairs (φ→1) get little
                        // shock viscosity — their mechanics are carried by the
                        // deviatoric stress, and full AV would viscously heat a
                        // cold solid into a spurious melt. Fluid pairs (φ→0)
                        // keep the standard Monaghan α, β.
                        let phi_j = phase_of(u_j, mat_t_m[j], params.l).phi;
                        let visc_scale = clamp(1.0 - 0.5 * (phi_i + phi_j), 0.0, 1.0);
                        let pi_ij = (-params.alpha * visc_scale * c_bar * mu
                                     + params.beta * visc_scale * mu * mu)
                            / max(rho_bar, 1e-30);
                        factor = factor + mj * pi_ij;
                        du_visc = du_visc + 0.5 * mj * pi_ij * v_dot_grad;
                    }

                    ax = ax - factor * grad_x;
                    ay = ay - factor * grad_y;
                    az = az - factor * grad_z;

                    // Deviatoric stress contraction (S226):
                    //   a_a += m_j (S_i,ab/ρ_i² + S_j,ab/ρ_j²) ∇W_b.
                    // Symmetric in i↔j ⇒ Newton's 3rd law holds for equal h.
                    let sbj = j * 6u;
                    let sjxx = dev_stress[sbj + 0u]; let sjyy = dev_stress[sbj + 1u]; let sjzz = dev_stress[sbj + 2u];
                    let sjxy = dev_stress[sbj + 3u]; let sjxz = dev_stress[sbj + 4u]; let sjyz = dev_stress[sbj + 5u];
                    ax = ax + mj * (inv_rho_i2 * (sixx * grad_x + sixy * grad_y + sixz * grad_z)
                                  + inv_rho_j2 * (sjxx * grad_x + sjxy * grad_y + sjxz * grad_z));
                    ay = ay + mj * (inv_rho_i2 * (sixy * grad_x + siyy * grad_y + siyz * grad_z)
                                  + inv_rho_j2 * (sjxy * grad_x + sjyy * grad_y + sjyz * grad_z));
                    az = az + mj * (inv_rho_i2 * (sixz * grad_x + siyz * grad_y + sizz * grad_z)
                                  + inv_rho_j2 * (sjxz * grad_x + sjyz * grad_y + sjzz * grad_z));

                    // Monaghan-2000 artificial stress (S228): the R̂ tensors
                    // already carry the −ε·tensile/ρ² factor; here apply the
                    // (W(r)/W(Δp))^m kernel ratio. Cures tensile pairing and
                    // binds cohered solid clumps into one block.
                    let w_r = wendland_w(r, hi);
                    // (W(r)/W(Δp))^M_ART with M_ART = 4: two muls instead of a
                    // pow() (exp/log) in the innermost pair loop.
                    let wr_ratio = w_r / max(w_dp, 1e-30);
                    let wr2 = wr_ratio * wr_ratio;
                    let fac = wr2 * wr2;
                    let rajxx = art_stress[sbj + 0u]; let rajyy = art_stress[sbj + 1u]; let rajzz = art_stress[sbj + 2u];
                    let rajxy = art_stress[sbj + 3u]; let rajxz = art_stress[sbj + 4u]; let rajyz = art_stress[sbj + 5u];
                    ax = ax + mj * fac * ((raixx + rajxx) * grad_x + (raixy + rajxy) * grad_y + (raixz + rajxz) * grad_z);
                    ay = ay + mj * fac * ((raixy + rajxy) * grad_x + (raiyy + rajyy) * grad_y + (raiyz + rajyz) * grad_z);
                    az = az + mj * fac * ((raixz + rajxz) * grad_x + (raiyz + rajyz) * grad_y + (raizz + rajzz) * grad_z);
                }
            }
        }
    }

    accelerations[i * 3u + 0u] = accelerations[i * 3u + 0u] + ax;
    accelerations[i * 3u + 1u] = accelerations[i * 3u + 1u] + ay;
    accelerations[i * 3u + 2u] = accelerations[i * 3u + 2u] + az;

    // Overwrite du/dt with this tick's SPH-side source terms.
    // Thermal conduction (S204) adds on top; the integrator
    // applies dt + radiation (S205) and clears the buffer.
    du_dt[i] = du_visc + du_pdv;
}
