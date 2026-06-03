# Sprint 223–232 — Tání + slučování do bloků (elasto-plastické SPH)

Decade 3 torus/planet experimentu. Rozšiřuje SPH+gravity+termální engine
(S202–S222) o **fyzikálně realistické tání** (fázový přechod řízený
teplotou + latentní teplo) a **slučování elementů do pevných bloků**
(koheze/agregace). Návrh vzešel z multi-agent design workflow + adversariální
red-team; rozhodnutí a opravy viz níže.

## Architektura (rozhodovací matice)

| Tradice | Verdikt |
|---|---|
| **Kontinuální elasto-plastické SPH (Gray–Monaghan–Swift 2001)** | ✅ zvoleno — jediná dává *smykovou tuhost* (= co fyzikálně znamená „blok") |
| Bonded-particle / DEM | ✗ formování vazeb není symetrické při adaptivním `h` → rozbíjí Newton 3 + determinismus |
| Multiphase EoS + koheze (bez tensoru) | ✗ bloky bez smyku tečou; částečně zachováno (Tait EoS) |
| Pragmatic minimal | ✗ tání nemá nezávislý mechanický podpis |

**Model:** per-částicový deviatorický stress tensor `S` (elastická paměť),
buzený Jaumann/Hooke rate rovnicí s Bonet–Lok korekcí gradientu a von
Misesovým plastickým mezí. Fáze z entalpické mapy `phase_of(u) → (T, φ)`.
Tlak z fázově přepínané EoS (plyn = ideální, kondenzát = Tait/Murnaghan).
Koheze = kontinuální tah (`P < 0`) + Monaghanův 2000 artificial stress.
„Blok" = emergentní souvislá oblast `φ≈1`, nikdy materializovaný objekt.

**Cesta tuhosti = A (měkké soudržné těleso, fixní `dt`).** EoS exponent
`n=3` + `c0=1`, `G0=1` drží globální CFL při `dt=1e-3` do ~4·ρ0. Tužší
„skalní" těleso (`n→7`, `G0~30`) vyžaduje inner sub-cyklus — rezervováno
pro S231 (path C).

**Red-team opravy zapracované:** (1) latentní teplo se NEřeší gatováním
pdV (rozbilo by první zákon) — `u` zůstává konzervovaná, plateau je jen v
mapě `T(u)`; (2) *všichni* konzumenti teploty (vedení, radiace, diagnostika)
jdou přes `phase_of(u).T`; (3) Tait `c` exploduje při kolapsu → `cfl_dt`
generalizováno na elastickou rychlost + zvolen mírný `n`; (4) Bonet–Lok
inverze regularizována bezpodmínečně (`M+λI`), ne větvením na `det≈0`; (5)
φ se počítá inline z `u` ve všech mechanických passech (žádný one-tick lag
na melt frontu); (6) artificial stress v *hlavní soustavě* (Jacobi
eigensolve), ne xyz-diagonální aproximace; (7) plastické teplo přes
separátní `du_plastic` scratch (žádný du_dt konflikt).

## Per-tick pořadí passů

```
kick(dt/2) → drift → hash → density
 → grad_correction (Bonet–Lok B) → stress_rate (Jaumann) → stress_integrate (von Mises → du_plastic)
 → artificial_stress (principal-frame R̂) → gravity → sph_force (P + S + R̂; φ-škálovaná AV)
 → thermal_conduction → thermal_integrate (+ du_plastic) → phase → kick(dt/2)
```

`φ` buffer je CPU cache (render/CSV/blocky), psaný na konci ticku; GPU
mechanika počítá φ inline. Stress passy se redukují na no-op (`S=0,
G0=Y0=0`), takže byte-identický determinismus test hlídá každý sprint.

## Konstanty (`src/planet/thermal.rs`, normalizované)

| Konstanta | Hodnota | Význam |
|---|---|---|
| `MELT_TEMPERATURE_T_M` | 0.30 | solidus `u_sol = T_m` |
| `LATENT_HEAT_FUSION_L` | 0.15 | latentní teplo; `u_liq=0.45`; Stefan `L/T_m=0.5` |
| `VAPORIZATION_ENERGY_U_VAP` | 5.0 | přepínač plyn/kondenzát |
| `TAIT_REF_SOUND_SPEED_C0` | 1.0 | `K0=ρ0·c0²` |
| `TAIT_EXPONENT_N` | 3.0 | tuhost (path A; viz CFL) |
| `SHEAR_MODULUS_G0` | 1.0 | `G_i=G0·φ²` |
| `YIELD_STRENGTH_Y0` | 0.5 | von Mises mez `Y_i=Y0·φ²`; `Y0/G0=0.5` |
| `TENSILE_STRENGTH_P_TENS` | 0.5 | max tah pevné větve (`P≥−P_tens·φ`) |
| `ARTIFICIAL_STRESS_EPSILON` / `_M` | 0.3 / 4.0 | Monaghan 2000 |
| `GRAD_CORRECTION_LAMBDA` | 1e-4 | Tikhonov regularizace B |
| `PLASTIC_HEAT_MAX_FRAC` | 0.1 | cap plastického ohřevu / tick |

## Sprinty

### S223a — determinism harness + phase single-source
**Cíl:** committed byte-identický rerun test (chyběl) + jednozdrojová `phase_of`.
**Výstup:** `gpu_planet_tick_deterministic_rerun` (2× 30 ticků, bit-identické
pos/vel/u). `thermal::phase_of` + WGSL mirror `shaders/planet_phase_common.wgsl`
(konkatenováno do konzumentů). CPU unit test mapy.
**Validace:** rerun bit-identický; plateau + monotonie φ.

### S223 — phase-map latent heat + unifikace teploty
**Cíl:** entalpická mapa + `φ`, bez mechaniky; plateau `T=T_m` přes pásmo tání.
**Výstup:** `phase_frac` buffer (PlanetGpu + Particles mirror + readback,
7-tuple `download_full`). `planet_phase` pass. Vedení + radiace +
diagnostika přepnuty na `phase_of(u).T`. `u` zůstává konzervovaná.
**Validace:** `gpu_phase_map_matches_cpu` (WGSL↔Rust), `gpu_conduction_plateau_no_intra_band_flux`,
`gpu_phase_frac_spans_and_matches_u`.

### S224 — kondenzovaná EoS (Tait/Murnaghan) + CFL
**Cíl:** fázová tlaková větev + honest CFL (cold solid měl falešně velký dt).
**Výstup:** `eos_pc` v sph_force (plyn ≥ u_vap; jinak Tait, S224 clamp `P≥0`).
`diagnostics::cfl_dt` → `thermal::sound_speed_of`. Plyn větev = bit-identická
s pre-S224 (reduction).
**Validace:** `gpu_condensed_eos_compression_and_tension`, `cfl_dt_uses_condensed_sound_speed`
(c_eff ≈ c0√n).

### S225 — stress storage + Bonet–Lok + Jaumann (elastic, bez momentum)
**Cíl:** persistentní `S[6N]` + `ds_dt[6N]` + `B[9N]`; korigovaný gradient +
Jaumann rate + explicit integrate. Bez momentum couplingu.
**Výstup:** `planet_grad_correction` (M assembly + Tikhonov inverze),
`planet_stress_rate`, `planet_stress_integrate`. `stress.rs` wrappery.
**Validace:** `gpu_stress_rate_linear_shear_matches_analytic` (dSxy=G·γ),
`gpu_stress_rate_rigid_rotation_objective` (max|dS/dt|=0 — Bonet–Lok).

### S226 — stress→momentum coupling + elastická CFL
**Cíl:** `+S` kontrakce do sph_force (bloky získají tuhost); CFL = elastická rychlost.
**Výstup:** sph_force binding 11 (dev_stress) + deviatorická síla.
`thermal::elastic_sound_speed_of` (`√(c_bulk²+4G/3ρ)`) v cfl_dt.
**Validace:** `gpu_sph_force_deviatoric_newton_third_law`,
`gpu_elastic_solid_resists_and_stays_finite` (max|S|>0, max_r bounded).

### S227 — fázově řízené moduly + von Mises + AV
**Cíl:** `G=G0·φ²`, `Y=Y0·φ²`, radial return (= remelt mechanismus).
**Výstup:** stress_rate gate G; stress_integrate von Mises clamp
`S*=min(1,Y/√3J2)`. Fázově škálovaná AV (solid se viskózně neroztaví).
**Validace:** `gpu_stress_integrate_von_mises_yield_and_remelt` (solid → σ_vm=Y0;
liquid φ=0 → S→0).

### S228 — Monaghan-2000 artificial stress + koheze (principal frame)
**Cíl:** tah na pevné větvi (`P≥−P_tens·φ`) + artificial stress → fúze bloků.
**Výstup:** `planet_artificial_stress` (σ=−P I+S → Jacobi eigensolve →
`R̂=V diag(−ε max(λ,0)/ρ²) Vᵀ`). sph_force binding 12 + `(W(r)/W(Δp))^m·(R̂_i+R̂_j)`.
**Validace:** `gpu_artificial_stress_principal_frame` (analytic R̂ pro rotovaný
σ), `gpu_cohesion_cold_pair_attracts` (cold tah přitahuje, hot odpuzuje).

### S229 — remelt/break + plastické teplo
**Cíl:** plný melt cyklus + plastická disipace do `u` (správný vzorec).
**Výstup:** `du_plastic` buffer; `J2_trial(1−f²)/(2Gρ)`, cap `PLASTIC_HEAT_MAX_FRAC`,
přičteno v thermal_integrate.
**Validace:** `gpu_remelt_dissolves_stress` (zahřátý solid → φ→0, max|S| spadne
na <10 % během 5 ticků).

### S230 — diagnostika, bloky, vizualizace, 5-seed sweep
**Cíl:** měřit bloky + skupenství; vizualizace; cross-seed validace.
**Výstup:** `ScalarDiagnostics.{mean_phase_frac, solid_mass_frac}`,
`diagnostics::count_solid_blocks` (union-find na φ>0.5 v 1.5·h̄). CSV sloupce
`mean_phi, solid_frac, largest_block`. planet_view `ColorMode::Phase` (F8
cyklus Rock→Temperature→Phase).
**Validace:** `gpu_block_detection_solid_vs_molten` (cold = 1 velký blok;
molten = 0). 5-seed sweep (seedy 1–5, t_end=0.3): mass=1.000000 exact,
mean_phi≈1, largest_block≈2000/2000, max_r≈1.2 (bez kolapsu), bez NaN.

### S231 — inner leapfrog sub-cyklus (cesta C) + konfigurovatelná tuhost
**Cíl:** tuhé „skalní" bloky za fixní vnější `dt` přes operator-split
sub-cyklování (gravitace O(N²) na vnějším kroku, tuhá fyzika na `dt_sub`).
**Výstup:** `PlanetConfig` + `n_substeps, shear_modulus, tait_c0,
tait_exponent, yield_strength` (defaulty = thermal consty → default cesta
bit-identická). Pipeliny mají `set_stiffness`/`set_g0`. `PlanetGpu` +
`grav_accel` buffer + `nbody_grav_bg` (gravitace do separátního bufferu) +
`encode_gravity_into_grav` / `encode_copy_grav_to_accel`.
`PlanetWorld::tick_sph_substepped`: gravitace jednou do `grav_accel` (držena
fixní přes `dt ≪ t_ff`), pak `n_sub`× vnitřní KDK smyčka (kick/drift/hash/
density/grad/stress/sph_force/thermal na `dt_sub`), s re-seedem
`accelerations = grav_accel` před každým force evalem. `tick_sph` dispatchuje
podle `n_substeps`. Klíč (red-team): sub-cyklovat jen stress pass je no-op —
rate závisí na `∇v`, které mění jen kick; nutná celá vnitřní smyčka.
CLI flagy v obou binárkách.
**Validace:** `gpu_stiff_solid_needs_substepping` — tuhost (max|S|=31.5
vs měkký 0.33, 96×), nutnost (`n_sub=1` při CFL-porušujícím `dt` exploduje
na r=2062) i dostatečnost (`n_sub=20` drží r=7 ohraničeně).
`gpu_substepped_tick_deterministic` (bit-identický rerun).

### S232 — multi-materiál (per-částicové ρ0/T_m) + diferenciace
**Cíl:** koexistence materiálů s různými body tání a hustotami.
**Výstup:** per-částicové buffery `mat_rho0` (Tait reference → diferenciace)
a `mat_t_m` (bod tání) v `PlanetGpu` + `Particles` mirror + `upload_materials`
(s default-seed ρ0=1, T_m=melt v `PlanetGpu::new` pro standalone testy).
Generátory plní `mat_rho0 = ρ_mean`. `init::assign_core_material` přiřadí
hutnější/žáruvzdornější materiál do jádra. EoS (`eos_pc`/`eos_pressure`),
phase pass, vedení, radiace, stress_rate, stress_integrate a artificial_stress
čtou per-částicový materiál (soused `j` svůj vlastní). Y0 zůstává globální,
g0 z S231 configu. CLI `--core-radius-frac/--core-rho0-mult/--core-t-m`.
**Validace:** `gpu_multimaterial_heterogeneous_melting` (při stejném `u`:
žáruvzdorné jádro φ=1.000 pevné, těkavá kůra φ=0.000 roztavená),
`gpu_multimaterial_rho0_controls_eos` (per-částicové ρ0 řídí EoS: nízké ρ0 →
komprese/odpuzování, vysoké ρ0 → tah/přitahování = hnací síla diferenciace),
`gpu_multimaterial_deterministic`.

### Poznámky k S231/S232 — tuhost, pdV gate, ekvilibrace

- **pdV gate (oprava v `sph_force`):** kondenzovaná Tait větev je
  *barotropní* (`P = P(ρ)`), takže adiabatická kompresní práce je
  zotavitelná elastická energie cold-curve nesená konzervativní silou — ne
  teplo. Adiabatický `du_pdv` se proto aplikuje jen na **plynnou větev**
  (`u ≥ u_vap`, kde `P` reálně závisí na `u`); jinak by spuriózně zahříval
  (a tavil) studené pevné těleso. Všech 48 testů zelených i po této změně.
- **Známé omezení (ekvilibrace):** studené *tuhé* těleso nastartované daleko
  od hydrostatické rovnováhy (cold start `u=0.01`, `P≈0` při `ρ0` → kolaps →
  prudký bounce při tuhé EoS) se rozkmitá; jakmile se část roztaví, φ-škálovaná
  AV se znovu zapne a disipace toto těleso postupně **přehřeje a roztaví**
  (~0.1–0.3 t_ff). Měkké těleso (`G0=1`) drží pevné neomezeně; tuhý blok je
  prokazatelně rigidní *krátkodobě* (test: max|S|=31.5 @ 60 ticků). Dlouho
  žijící tuhý blok vyžaduje **inicializační relaxaci/ekvilibraci** (původně
  plánováno jako budoucí práce) nebo energeticky-konzervativní stress
  integrátor — mimo rozsah S231/S232. Multi-materiál při dostatečném
  rozlišení (cube N=30k) je stabilní (jeden souvislý blok, ohraničený).
- **Rozlišení:** non-torus tvary (cube/pancake) jsou při nízkém N
  pod-vzorkované pro SPH dynamiku (`h_init > h_max` kvůli těsnému
  `world_half`); pro dynamiku multi-materiálu volit torus nebo vyšší N.

## Decade 3 retro

8 sprintů + prerekvizita (S223a). 16 nových integration testů (27 → 43),
9 nových WGSL shaderů/passů (phase, grad_correction, stress_rate,
stress_integrate, artificial_stress + sdílený phase_common), 6 nových
per-částicových bufferů (φ, S, ds_dt, B, R̂, du_plastic). Engine teď
simuluje **vícefázové samogravitující těleso s tavením a kohezí**: studená
hmota tuhne do souvislého elastického bloku se smykovou tuhostí, kohezí a
mezí kluzu; ohřev přes bod tání absorbuje latentní teplo (T plateau),
rozpustí stress (von Mises Y→0) a vrátí těleso na tekutinu. Vše GPU-resident,
deterministické (byte-identické rerun), validované analytickými oraclemi.

## Soubory

```
src/planet/thermal.rs           (+ phase_of, EoS/stress/artificial konstanty + helpers)
src/planet/particle.rs          (+ phase_fracs)
src/planet/world.rs             (+ phase, stress, artificial passy v tick_sph/init/reset)
src/planet/diagnostics.rs       (+ mean_phi/solid_frac, count_solid_blocks, cfl_dt)
src/planet/gpu/state.rs         (+ phase_frac, dev_stress, ds_dt, grad_corr, art_stress, du_plastic)
src/planet/gpu/phase.rs         (new)
src/planet/gpu/stress.rs        (new — GradCorrection/StressRate/StressIntegrate/ArtificialStress)
src/planet/gpu/{sph_force,thermal_conduction,thermal_integrate}.rs  (fáze + EoS + stress bindings)
shaders/planet_phase_common.wgsl, planet_phase.wgsl            (new)
shaders/planet_{grad_correction,stress_rate,stress_integrate,artificial_stress}.wgsl  (new)
shaders/planet_sph_force.wgsl   (fázová EoS + deviator + artificial stress + φ-AV)
shaders/planet_thermal_{conduction,integrate}.wgsl  (phase_of T + plastické teplo)
src/bin/planet_headless/main.rs (CSV: mean_phi, solid_frac, largest_block)
src/bin/planet_view/main.rs     (ColorMode::Phase)
tests/planet_integration.rs     (+16 testů)
```
