# Planet thermal model — Sprints 202–211

Per-particle internal energy `u` plus the thermodynamic feedback loop:
viscous + adiabatic heating → conduction → ideal-gas pressure → motion,
with Stefan–Boltzmann radiation on surface particles as the sole energy
sink. Built on top of the SPH engine introduced in `203-212-torus-planet-sph.md`.

## Numerical model

```
u_i ← clamp(u_i + dt · (du_visc + du_pdv + du_cond + du_rad), u_min, u_max)

du_visc = ½ Σⱼ mⱼ Πᵢⱼ (vᵢ − vⱼ)·∇Wᵢⱼ              (S202)
du_pdv  = (Pᵢ/ρᵢ²) Σⱼ mⱼ (vᵢ − vⱼ)·∇Wᵢⱼ          (S203)
du_cond = Σⱼ mⱼ · 2κ · (Tᵢ − Tⱼ) · Fᵢⱼ / (ρᵢρⱼ)   (S204, h̄-symmetric)
du_rad  = −εσ(Tᵢ⁴ − T_space⁴) · surface_flagᵢ      (S205, floor-clamped)

P = ρ u (γ−1)        ideal gas (S203)
c_s = √(γ(γ−1) u)    sound speed
T = u / c_v
```

`HEAT_CAPACITY_CV = 1` makes `T ≡ u` for visual + log convenience.

## Sprint roll-up

### S202 — internal energy buffer + viscous heating
- `Particles.internal_energies: Vec<f32>` + `PlanetGpu` storage buffer.
- `planet_sph_force.wgsl` accumulates `du_visc` from the existing
  Monaghan `Πᵢⱼ` term during the same neighbour scan.
- New `planet_thermal_integrate.wgsl` + `ThermalIntegrateGpu` apply
  `dt · du/dt` and clamp `u ∈ [U_MIN, U_MAX]`.
- CSV picks up `u_total`, `e_full`, `mean_t`.
- Verified: KE drop ≈ `u` rise in cold-init collapse.

### S203 — ideal gas EoS + adiabatic compression
- Pressure now `P = ρ u (γ−1)`; sound speed `√(γ(γ−1) u)`.
- pdV heating in the same SPH-force loop.
- `eos_k` retained in `PlanetConfig`/CLI for binary compatibility,
  ignored by the shader.
- Total energy drift drops from ~1.6 % (S202) to ~0.07 % over 200 ticks
  — pdV closes the KE↔U budget that viscosity alone couldn't.

### S204 — Cleary–Monaghan thermal conduction
- New `planet_thermal_conduction.wgsl` + `ThermalConductionGpu`
  slotted between sph_force and thermal_integrate.
- Symmetric kernel `h̄ = (hᵢ + hⱼ)/2` ensures `Fᵢⱼ = Fⱼᵢ` so pair
  contributions cancel — without this conduction silently injected
  ~30 % extra `u` over 200 ticks (h-asymmetry in the gather form).
- `κ = 0.1` ⇒ diffusion timescale `τ_diff ≈ 10·t_ff`.
- Verified energy-neutral on uniform-T smoke run (byte-identical
  trace to S203).

### S205 — Stefan–Boltzmann radiation
- Surface flag inline in `planet_thermal_integrate.wgsl`:
  `ρᵢ < SURFACE_DENSITY_FRAC · ρ_mean_init ⇒ surface`.
- `du_rad = −εσ(T⁴ − T_space⁴)`, floored at `−u · max_rad_frac / dt`
  so an anomalously hot particle can't drain `u_i` in one substep.
- `PlanetWorld.rho_mean_init` caches the analytic `M/V` once at
  construction; threshold reuses it every tick.

### S206 — initial temperature profiles
- `TemperatureProfile::{Uniform, HotCore, Differentiated}` enum +
  `apply_temperature_profile()` post-init hook.
- Both binaries pick up `--init-temp-{profile,core,surface}` flags.
- HotCore = quadratic radial falloff, smooth. Differentiated = sharp
  step at `R/2`. Uniform overrides `INITIAL_INTERNAL_ENERGY` for the
  whole field.

### S207 — diagnostics + drift detector
- `ScalarDiagnostics.{min,max}_temperature`.
- CSV gains `min_t`, `max_t`, `drift_pct`.
- `planet_headless` warns to stderr on each new whole-percent drift
  milestone — radiation pushes drift negative, leapfrog accumulated
  error can push positive, so the warning is informational, not a
  hard error.

### S208 — temperature visualisation (planet_view)
- F8 toggles Rock ↔ Temperature mode. Rock keeps the original earth-
  tone palette; Temperature swaps to a 16-step viridis-like ramp.
- Per-frame auto-scale on `u_min/u_max` so contrast stays full as the
  field evolves. Material handle swap (not mesh rebuild) so Bevy
  batching stays tight (≤ 16 draw groups).
- HUD shows `U`, `E+U`, `T̄`, `T_min`, `T_max`, and the active mode.

### S209 — deferred
Reserved for biology coupling (planet surface T → main sim
`THERMAL_TOP/BOTTOM` seed). Currently the SPH planet engine and the
biology `World` are independent; the bridge will land once the planet
becomes a substrate the cells live on.

### S210 — tuning sweep + this doc
5-seed × 1·t_ff headless sweep with HotCore profile (core=1.0,
surface=0.05, κ=0.1, σ=1e-3, ε=0.9):

```
seed  u_total   mean_T    min_T     max_T     drift
1     0.1589    0.1589    0.135     0.198     +0.25 %
2     0.1626    0.1626    0.135     0.225     +0.20 %
3     0.1618    0.1618    0.130     0.200     +0.16 %
4     0.1585    0.1585    0.135     0.191     +0.28 %
5     0.1663    0.1663    0.131     0.239     +0.30 %
```

Cross-seed variation ≈ 5 % on `mean_T`; drift bounded under 0.3 % at
1·t_ff. Conduction has visibly shrunk the initial 0.05–1.0 T range to
0.13–0.20 ± seed noise. The slight positive drift is leapfrog error
exceeding the (cold-surface, weak) radiation sink at these parameters.

### S211 — reserved
For future thermal extensions: per-material `κ` once heterogeneous
composition lands; semi-implicit integrator if explicit Euler starts
oscillating at lower `dt`; CFL clamp on `dt_thermal`.

## Files touched

```
src/planet/thermal.rs                       (new — constants + helpers)
src/planet/particle.rs                      (+ internal_energies)
src/planet/world.rs                         (+ thermal pipelines + rho_mean_init)
src/planet/init.rs                          (+ TemperatureProfile, applier)
src/planet/diagnostics.rs                   (+ u_total, min/max T, ideal-gas CFL)
src/planet/gpu/state.rs                     (+ u + du_dt buffers, clear/upload/dl)
src/planet/gpu/sph_force.rs                 (bindings 9, 10)
src/planet/gpu/thermal_integrate.rs         (new — ThermalIntegrateGpu)
src/planet/gpu/thermal_conduction.rs        (new — ThermalConductionGpu)
src/planet/gpu/mod.rs                       (re-exports)

shaders/planet_sph_force.wgsl               (ideal gas + adiabatic + viscous heating)
shaders/planet_thermal_integrate.wgsl       (new — Euler + radiation + clamp)
shaders/planet_thermal_conduction.wgsl      (new — Cleary–Monaghan, h̄-symmetric)

src/bin/planet_headless/main.rs             (CSV columns, drift detector, CLI flags)
src/bin/planet_view/main.rs                 (F8 toggle, ThermalPalette, HUD)
```

## Constants reference

| Constant | Value | Meaning |
|---|---|---|
| `HEAT_CAPACITY_CV` | 1.0 | T ≡ u in sim units |
| `INITIAL_INTERNAL_ENERGY` | 0.01 | default cold-start `u` |
| `U_MIN` / `U_MAX` | 1e-6 / 1e3 | safety clamps |
| `THERMAL_CONDUCTIVITY_KAPPA` | 0.1 | τ_diff ≈ 10·t_ff |
| `SURFACE_DENSITY_FRAC` | 0.5 | `ρ < frac · ρ_mean ⇒ surface` |
| `STEFAN_BOLTZMANN_SIGMA` | 1e-3 | sim-normalised σ |
| `RADIATION_EMISSIVITY` | 0.9 | rocky surface ε |
| `SPACE_TEMPERATURE` | 3e-3 | cosmic background sink |
| `RADIATION_MAX_FRAC` | 0.1 | per-tick radiation cap |
