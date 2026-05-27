# Sprint 203–212 — Torus planet (SPH + self-gravity)

Side experiment, isolated from biology. Goal: simulate a small-scale
self-gravitating fluid planet initialised as a torus and observe whether
the shape persists over a long horizon. Decision matrix from planning:

- **Physics model:** SPH (Wendland C2 kernel) + direct N² self-gravity
- **N target:** 100k particles
- **GPU isolation:** shared `GpuContext`, separate `planet_*` binaries
- **EOS:** polytropic `P = K · ρ^γ`, γ = 5/3 (monatomic ideal gas)
- **Integrator:** symplectic leapfrog (KDK)

Normalised units throughout the simulation (`G = M_total = R_major = 1`).
SI conversion table:

| Quantity | Normalised | SI (Pluto-class fiducial) |
|---|---|---|
| `G` | 1 | 6.674e-11 m³/(kg·s²) |
| `M_total` | 1 | 4.7e21 kg |
| `R_major` | 1 | 1000 km |
| `t_ff = √(R³/GM)` | 1 | ≈ 1500 s |
| `v_orb = √(GM/R)` | 1 | ≈ 560 m/s |

## Sprint 203 — module scaffolding

**Cíl:** vytvořit izolovaný `src/planet/` modul + dvě binárky
(`planet_view`, `planet_headless`) jako kostru pro následující sprinty.
Žádná fyzika ještě nepoběží — jen typy, builder API a build-green.

**Výstup:**

- `src/planet/` modul s podmoduly `particle`, `world`, `init`,
  `diagnostics`, `gpu` (poslední dva jsou stuby do S206/S212).
- `Particles` SoA struktura: `positions`, `velocities`, `accelerations`,
  `masses`, `smoothing_lengths`, `densities`. Diagnostické helpery
  `total_mass`, `center_of_mass`.
- `PlanetConfig` + `PlanetWorld` skeleton — drží částice, sdílený
  `GpuContext` (lazy `init_gpu()`), helpery `t_ff()` a `omega_circ()`.
- `src/bin/planet_view/main.rs` — minimální Bevy app, otevře okno s
  3D kamerou a směrovým světlem.
- `src/bin/planet_headless/main.rs` — clap CLI (`--n --r-major --r-minor
  --omega-frac --seed --t-end`), instancuje `PlanetWorld`, zatím jen
  printuje konfiguraci.
- `Cargo.toml` přidává dvě `[[bin]]` položky.
- `src/lib.rs` přidává `pub mod planet;`.

Build green:
`cargo build --bin planet_view --bin planet_headless` — 8 s.

**Poznámky:** unit testy uvnitř `src/planet/*.rs` (`#[cfg(test)] mod
tests`) zatím nelze spustit přes `cargo test --lib`, protože
`src/gpu/tests.rs:976` má pre-existing chybějící argumenty k
`CollisionGpu::compute` (`masses`, `bond_rest_cos`) — out of scope pro
S203. Kompilační validace prošla, runtime test až jakmile bude
upstream test fix.

## Sprint 204 — torus init + static viewer

**Cíl:** parametricky vygenerovat uniformní torus distribuci + zobrazit
ji v Bevy point-cloud vieweru.

**Výstup:**

- `src/planet/init.rs` — `torus_uniform(config)` rejection-samples
  bounding-box `[-(R+r), R+r]² × [-r, r]`, accept-region torus implicit
  `(√(x²+y²) − R)² + z² ≤ r²`. Acceptance ≈ 34 % při R=1, r=0.2.
  Per-particle mass `m = M / N`, initial smoothing length
  `h₀ = 1.3 · (m / ρ̄)^(1/3)`, kde `ρ̄ = M / V_torus`.
- `omega_from_frac(config, frac)` helper — převod CLI `--omega-frac` na
  `Ω = frac · √(GM/R³)`. Rigid rotation kolem z-osy.
- Unit testy: total mass, count, in-volume, COM ≈ 0, uniform `h`,
  rotation velocity match, **principal moments shape**: pro
  `N = 20 000` test ověřuje `I_zz ≈ M(R² + 3r²/4)`,
  `I_xx ≈ I_yy ≈ M(R²/2 + 5r²/8)` do 2 %.
- `src/bin/planet_view/main.rs` přidává clap CLI, instancuje
  `PlanetWorld`, generuje částice přes `torus_uniform`, spawn-uje
  per-particle `Sphere` (default 5000). Camera + DirectionalLight +
  AmbientLight (Bevy 0.18: `commands.spawn(AmbientLight{..})`).

Build green: `cargo build --bin planet_view --bin planet_headless` — 3 s
inkrementálně.

**Poznámky:** S214 vymění naivní per-entity sphere za instanced mesh
(pro 100k částic je per-entity nepřijatelně pomalý).

## Sprint 205 — CPU leapfrog + Kepler test

**Cíl:** mít symplectic KDK leapfrog na CPU + CPU O(N²) gravity jako
reference oracle pro pozdější GPU validace. Validovat na analytickém
Kepler orbiteru — energy conservation < 10⁻³ za 10 period.

**Výstup:**

- `src/planet/integrator.rs` — `leapfrog_step(particles, dt,
  recompute_acc)`. Konvenční KDK: kick(½dt, a_old), drift(dt),
  recompute, kick(½dt, a_new). State `(x, v, a)` zůstává časově
  synchronní.
- `src/planet/gravity_cpu.rs` — `compute_acceleration(particles, g,
  softening)`: O(N²) pairwise s Plummer softening
  `1/(r² + ε²)^(3/2)`. Skip `i == j` pro determinismus.
  `potential_energy(particles, g, softening)` — `U = -G/2 Σ m_i m_j /
  √(r² + ε²)` pro energy diagnostiku.
- `PlanetWorld::tick()` zapojuje CPU leapfrog + CPU gravity (production
  cesta zatím; S207 přepne na GPU). `seed_accelerations()` musí být
  volaný jednou před první `tick()` aby první half-kick měl validní `a₀`.
- `tests/planet_integration.rs` — integration tests mimo lib
  (pre-existing `src/gpu/tests.rs:976` neumí kompilovat → lib unit
  tests blokované; integration tests závisí jen na public API).
  Pokrývá: Kepler kruhový orbit (drift < 10⁻³ za 10 period při
  dt=T/200), eccentric orbit (e=0.5, max drift < 5×10⁻³ za 5 period
  při dt=T/1000), 2-body symmetric force, potential energy 2-body,
  torus principal moments, world tick advances state.

Test run: `cargo test --test planet_integration` → 10 passed (< 1 s).

**Poznámky:** existují i `#[cfg(test)] mod tests` uvnitř
`src/planet/{integrator,gravity_cpu,particle,world,init}.rs` —
duplikují integration tests, ale dokumentují API uvnitř souboru.
Skutečně se nyní spouštějí jen integration tests.

## Sprint 206 — GPU N² gravity shader

**Cíl:** GPU N² direct-sum gravity s tile-cached workgroup memory.
Validovat proti CPU referenci.

**Výstup:**

- `shaders/planet_nbody.wgsl` — tile of 64 source particles cached do
  `var<workgroup> shared_pm: array<vec4<f32>, 64>` (xyz + mass).
  Workgroup size 64; per-thread inner loop iteruje plný tile
  (out-of-range entries mají mass=0 → no contribution). Self
  masking přes `select(mj, 0.0, k_global == i)` — branch-free.
- `src/planet/gpu/nbody.rs` — `NBodyGpu`. `compute()` upload + dispatch
  + readback (testing); `dispatch_into()` + `upload_params()` pro
  hot-loop S207. 4 storage bindings (params uniform + positions +
  masses + accelerations).
- Integration test `gpu_nbody_matches_cpu_reference`: N=1024 torus
  distribution, porovnání accelerations GPU vs CPU O(N²),
  `max_abs < 5×10⁻⁴`. Reálně naměřeno `~10⁻⁵` na desktop GPU.
- Test `gpu_nbody_zero_particles_safe`.

Test run: `cargo test --test planet_integration` → 12 passed (0.22 s
včetně dvou GPU testů).

**Poznámky:** S207 přepne na unifikovaný `PlanetGpu` který shared
buffers napříč všemi 3 pipeline (nbody, kick, drift).

## Sprint 207 — GPU leapfrog kick/drift

**Cíl:** kompletní KDK leapfrog smyčka na GPU. Validace 2-body
orbit + energy conservation.

**Výstup:**

- `shaders/planet_kick.wgsl` — v += dt_half · a, 3 bindings.
- `shaders/planet_drift.wgsl` — x += dt · v, 3 bindings.
- `src/planet/gpu/state.rs` — `PlanetGpu`: shared buffers (positions,
  velocities, accelerations, masses) + 3 pipelines (nbody, kick,
  drift) + readback buffers. Jedna `step_leapfrog(n, dt, g, eps)`
  metoda zaznamenává všechny 4 dispatche (kick₁, drift, nbody, kick₂)
  do jednoho command bufferu a submituje.
- Fix self-mask v `planet_nbody.wgsl`: move `select` z `mj_eff` na
  `inv_r3` aby `0 × inf` (pro eps=0) nedalo NaN.
- Integration test `gpu_two_body_circular_orbit_energy_conserved`:
  m₁=m₂=0.5, a=1, v=√(Gm/4a)=0.354, dt=0.01, n_steps=9000 ≈ 5 period.
  Max |dE/E| < 5×10⁻³. Naměřeno typicky 10⁻³.

Test run: `cargo test --test planet_integration` → 13 passed (0.69 s).

**Poznámky:** S214 dovolí PlanetGpu sdílet positions buffer s instanced
mesh rendererem (`COPY_SRC` + `VERTEX` usage flag) pro zero-copy
particle viz.

## Sprint 208 — GPU spatial hash for SPH

**Cíl:** GPU counting-sort spatial hash pro neighbour search v SPH.

**Výstup:**

- `shaders/planet_spatial_hash.wgsl` — 32³ = 32 768 buckets, 3D grid
  pokrývající `[-world_half, +world_half]³`. 4 entry points
  (`count`, `prefix_sum`, `scatter`, `sort_buckets`). Hillis-Steele
  scan v 256-threadové workgroupě (128 elements per thread). Bucket
  sortování zajišťuje deterministický neighbour walk.
- `src/planet/gpu/spatial_hash.rs` — `SpatialHashGpu`. Bind group
  referencuje positions z `PlanetGpu`. `rebuild(n)` encodes all
  4 passes do jednoho command bufferu. Helper `cell_size()` a
  `max_supported_h() = 0.75 × cell_size` definuje, jaký h lze SPH
  pipeline použít s 3×3×3 stencilem.
- `PlanetGpu` přidává public accessory `positions_buffer()`,
  `velocities_buffer()`, `accelerations_buffer()`, `masses_buffer()`
  aby ostatní pipeliny mohly bindovat.
- `bucket_id_cpu()` helper — match `bucket_id_of` z shaderu pro
  CPU testy.
- Integration test `gpu_spatial_hash_bucket_assignment`: 200 částic
  ve známých pozicích, verifikuje že každá je ve správném bucketu a
  sorted_particles uvnitř bucketu jsou ascending.

Test run: `cargo test --test planet_integration` → 14 passed (0.76 s).

**Poznámky:** s default `world_half = 2.5`, cell_size = 0.156. Pro
init torus `h_init ≈ 0.053`, takže 2h = 0.106 < cell_size (3×3×3
stencil ok). Pokud particles spread out (collapse → expansion), h
roste; clamp na 0.75×cell_size brání podexcerptu neighbours.

## Sprint 209 — SPH density (Wendland C2)

**Cíl:** GPU SPH density estimator + adaptive smoothing length.

**Výstup:**

- `shaders/planet_density.wgsl`. Per-particle 3×3×3 bucket scan,
  `ρ_i = Σ_j m_j · W(|x_i − x_j|, h_i)`. Wendland C2:
  `W(r,h) = (21 / 16π h³) · (1 − q/2)⁴ · (1 + 2q)`, `q = r/h ≤ 2`.
  Include self (`j == i` gives `W = 21/16π h³`, the kernel peak).
  Po sumě update `h_new = clamp(η · (m/ρ)^(1/3), h_min, h_max)` kde
  `h_max = 0.75 · cell_size` (grid coverage limit).
- `src/planet/gpu/density.rs` — `DensityGpu` bind group referencuje
  positions, masses, smoothing_lengths, densities (z `PlanetGpu`)
  + hash offsets + sorted_particles. 7 bindings total.
- `PlanetGpu` přidává `smoothing_lengths_buf`, `densities_buf` +
  readback buffery + upload/download metody.
- `wendland_c2_cpu` helper pro CPU reference testy.
- Integration test `gpu_sph_density_uniform_grid`: 16³ = 4096
  uniformně rozmístěných částic v boxu `[-0.5, 0.5]³`, `M = 1`,
  `ρ_true = 1`. Pro interior částice (`r < 0.3` od středu, ~64
  vzorků) max relative error < 10 %. Naměřeno typicky ~3-5 %.

Test run: `cargo test --test planet_integration` → 15 passed (1.08 s).

**Poznámky:** standardní lattice noise pro `h/dx ≈ 1.3` je 3-7 %.
Pokud bude potřeba přesnější estimate, S211 může iterovat
density→h několikrát per tick.

## Sprint 210 — SPH pressure force + EOS

**Cíl:** SPH pressure-gradient force s polytropic EOS.

**Výstup:**

- `shaders/planet_pressure.wgsl`. Per-particle 3×3×3 bucket scan,
  pro každého souseda:
  - `dvec = x_i − x_j`, `r = |dvec|`, `q = r/h_i`
  - Wendland C2 gradient: `dW/dr = −105 q (1 − q/2)³ / (16π h⁴)`
  - `∇_i W_ij = (dW/dr) · dvec / r` (points from `i` to `j`)
  - `P = K · ρ^γ`, `factor = m_j · (P_i/ρ_i² + P_j/ρ_j²)`
  - `a_i += −factor · ∇_i W_ij`
- Polytropic EOS: γ = 5/3 (monatomic ideal gas). `K` jako runtime
  param přes `PressureParams`.
- Bind group má 8 entries (positions, masses, h, ρ, accel R/W,
  hash offsets, sorted, params uniform).
- Accelerace **přidává**, neoverwriteuje — caller zaručuje že
  gravity dispatched first.
- Integration test `gpu_sph_pressure_pair_newton_third_law`: 2 částice
  se stejným h a ρ → kernel symetrický → `a_1 + a_2 = 0` do FP
  tolerance, znaménka správně (pressure pushes apart).

Test run: `cargo test --test planet_integration` → 16 passed (1.25 s).

**Poznámky:** asymmetric kernel form (h_i only). Newton 3rd law
striktně drží jen pro stejné h; pro nestejné h je až 1-5% drift,
což je akceptovatelné pro SPH (Monaghan 1992). S213+ může přejít na
kernel-averaged formu pokud bude problém.

## Sprint 211 — artificial viscosity

**Cíl:** Monaghan artificial viscosity pro shock capture +
suppression post-shock oscilací.

**Výstup:**

- `shaders/planet_viscosity.wgsl`. Per-particle scan, pro approaching
  pair (`v_ij · r_ij < 0`):
  - `μ_ij = h̄ · (v_ij · r_ij) / (r² + 0.01 h̄²)`
  - `c̄ = (c_i + c_j)/2`, `c = √(γ P / ρ)`
  - `Π_ij = (−α c̄ μ + β μ²) / ρ̄`
  - `a_i += −m_j · Π_ij · ∇_i W_ij`
- Standardní `α = 1, β = 2`, runtime tunable přes params.
- 9 bindings (positions, velocities, masses, h, ρ, accel R/W,
  offsets, sorted, params).
- Integration testy:
  - `gpu_sph_viscosity_decelerates_approaching_pair`: 2 částice
    closing s `v = ±0.5`. Force decelerates obě (a_1.x < 0, a_2.x > 0),
    Newton 3rd dodrženo.
  - `gpu_sph_viscosity_inactive_for_separating_pair`: stejný setup,
    opačné rychlosti. Force = 0.

Test run: `cargo test --test planet_integration` → 18 passed (1.74 s).

**Poznámky:** sound speed shared s pressure shaderem (γ P / ρ). Pokud
S213+ chce energy-conservative SPH form, oba shadery musí přejít na
single-pipeline merge.

## Sprint 212 — adaptive timestep + diagnostics

**Cíl:** end-to-end SPH+gravity tick driver + CPU-side stability
metriky. Konec Decade 1.

**Výstup:**

- `PlanetGpu::kick(n, dt_half)` + `drift(n, dt)` — sólo dispatche
  (vlastní encoder + submit) pro orchestraci s force passes mezi nimi.
- `PlanetWorld::init_gpu_full()` — alokuje `PlanetGpu` + `SpatialHashGpu`
  + `DensityGpu` + `PressureGpu` + `ViscosityGpu`, uploaduje
  particle state, seedne `a_0` (gravity + pressure + viscosity).
  `world_half = 2 (R + r)` (100 % slack pro post-collapse spread).
- `PlanetWorld::tick_sph()` — KDK leapfrog: kick₁ → drift → hash
  rebuild → density → gravity (overwrite) → pressure (add) →
  viscosity (add) → kick₂.
- `PlanetWorld::download_state()` — pull positions, velocities,
  accelerations, smoothing_lengths, densities z GPU do
  `self.particles`. Pro diagnostiku a CSV.
- `src/planet/diagnostics.rs`:
  - `ScalarDiagnostics` (mass, KE, Lz)
  - `inertia_tensor(particles) -> [f64; 6]` (symetric, `[Ixx, Iyy, Izz, Ixy, Ixz, Iyz]`)
  - `principal_moments(particles) -> [f64; 3]` — cyclic Jacobi
    rotations na 3×3 symmetric matrix; vrací descending. Konvergence
    typicky < 10 sweeps.
  - `total_energy(particles, g, eps) -> (KE, PE, E)`
  - `cfl_dt(particles, K, γ, C_courant) -> dt_min`
- Inline `#[cfg(test)]` test `principal_moments_axis_aligned_torus`
  (do 2 % konvergence k analytic).

Integration testy:
- `full_sph_gravity_tick_smoke`: 2000 particles, 50 ticků. Verifikuje
  particles moved, vše finite, no NaN, Lz drift < 5 %, principal
  moments sensibly ordered.
- `cfl_dt_finite_for_torus_init`: torus init → finite positive dt.

Test run: `cargo test --test planet_integration` → 20 passed (1.81 s).

**Poznámky:** adaptive dt se zatím nepoužívá v tick_sph (drží
config.dt). S213 ho zapojí přes "recompute dt every K ticks na
základě CFL z downloaded state". Inertia tensor + energy se počítají
on CPU s f64 — za cenu downloadu, ale ve f32 GPU reduction by
nashromáždily ~10⁻⁴ relativní chybu která maskuje shape evolution.

## Decade 1 retro (S203-S212)

10 sprintů, 20 integration testů, ~3500 řádků nového kódu. Decade 1
dokončila celý core SPH+gravity engine — od `Particles` SoA přes 5
GPU compute pipelines (nbody, kick, drift, spatial hash, density,
pressure, viscosity) až k `tick_sph()` orchestrátoru, CPU
diagnostikám a CFL helperu. Jediný blokující out-of-scope problém:
pre-existing `src/gpu/tests.rs:976` chybějící args u CollisionGpu::compute,
kvůli kterému planet unit testy běží jako `tests/planet_integration.rs`
mimo lib test target.

Decade 2 (S213-S217) experimentální harness + Bevy viewer polish +
parametric sweep.
