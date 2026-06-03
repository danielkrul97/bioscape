// Single source of truth for the enthalpy phase map (WGSL side).
// Mirror of `bioscape::planet::thermal::phase_of` — the two MUST stay
// identical; `gpu_phase_map_matches_cpu` asserts agreement on a grid.
//
// Prepended (string-concatenated) into every shader that needs the
// sensible temperature or solid fraction, so the map lives in one place.
// Declares only a struct + a pure function — no bindings, no entry point
// — so it is safe to concatenate ahead of any shader.
//
// `u` is the conserved internal energy per unit mass (it carries latent
// heat). With cv = 1 the sensible temperature equals `u` below melt; it
// plateaus at `t_m` while `u` climbs through `[t_m, t_m + l]` (energy goes
// into melting, raising `phi`, not temperature), then rises again above.

struct PhaseState {
    t: f32,
    phi: f32,
}

fn phase_of(u: f32, t_m: f32, l: f32) -> PhaseState {
    let u_sol = t_m;
    let u_liq = t_m + l;
    var r: PhaseState;
    if (u <= u_sol) {
        r.t = u;
        r.phi = 1.0;
    } else if (u < u_liq) {
        r.t = t_m;
        r.phi = 1.0 - (u - u_sol) / l;
    } else {
        r.t = t_m + (u - u_liq);
        r.phi = 0.0;
    }
    return r;
}
