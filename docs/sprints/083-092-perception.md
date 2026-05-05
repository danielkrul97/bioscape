# Sprinty 83–92: Perception

Decade přechází z multicelularitního výzkumu (S73-82) k senzorické
specializaci. Pre-Sprint-83 bylo vidění čistě sférické (4π str), bez
směrového rozlišení — buňka „viděla" stejně dobře dopředu, dozadu i do
stran. Sprint 82 zavedl genovou infrastrukturu (`vision_fov` half-angle +
energy cost faktor) a Sprint 83 aktivuje cone filter v sensor gather, čímž
poprvé vzniká skutečný **trade-off mezi periferním vědomím a energetickou
efektivitou**: úzký kužel platí menší vision drain, ale ztrácí informaci v
slepém úhlu. Sprint 84 přenáší stejnou mechaniku na Hunter — predátor s
forward-eye konfigurací (60° half-angle, 120° kužel) má slepou skvrnu za
sebou, kterou cells mohou flank-uniknout.

## Sprint 83 — cone filter v sensor gather (cells)

- **Cíl:** aktivovat směrový FOV v cell sensor gather. Sprint 82 přidal
  `vision_fov` gen + cost factor s `MUTATION_CONFIG.sigma_vision_fov = 0.0`
  (drift dormant) — neměl ještě behavior change. Sprint 83 zapne mutaci
  (`sigma_vision_fov = 0.05`) a přidá cone filter do
  `for_each_in_radius_toroidal` callbacks v obou binárkách. Cells s užším
  FOV uvidí jen kandidáty v `dot(d/|d|, forward) >= cos(theta)` výseči;
  smell + pheromone gradients zůstávají omni (chemické modality jsou
  biologicky izotropní + samy nesou směr přes gradient).

- **Mechanismus:**
  - `bioscape::fov_cone_accept(delta, d2, forward, cos_fov)` helper:
    `dot(delta, forward) >= cos_fov × |delta|` (jediný sqrt per kandidát).
    Degenerate `|delta| ≈ 0` → accept (cell-overlap zone).
  - Per-cell precompute: `cos_fov = vision_fov.cos()`, `skip_cone = fov >=
    MAX_VISION_FOV` (krátké okno pro full-sphere FOV), `fwd =
    forward_vector(heading, pitch)`.
  - Aplikace v `main.rs::cells_brain_act::gather` + `headless.rs::brain_act`
    + `headless.rs::brain_act_gpu` Phase 1: filtr **uvnitř** radius testu
    pro `nearest_food`, `nearest_cell`, `neighbors_in_vision`. Kandidát mimo
    cone se přeskočí (early return v callback), nepřispívá k density count.
  - `MUTATION_CONFIG.sigma_vision_fov: 0.0 → 0.05` (~1.7 % FOV range per
    gen, conservative).

- **CSV diagnostika (`headless.rs`):** přidány 2 sloupce `fov_avg` + `fov_dev`
  (51 + 52 z 50 + 2 = 52 total). Header + extinction-row updated. Per-cell
  sum/sumsq agregace v `for c in &world.cells` loop.

- **Smoke (seed=0, 30 gen):**
  - Gen 0: fov_avg = π = 3.1416 (initial), fov_dev = 0 (homogenní).
  - Gen 30: fov_avg = 2.832 (~10 % zúžení), fov_dev = 0.219 (rozptyl roste).
  - Population stabilní 500-1000 cells; cone filter neporouchá baseline
    survival, jen vytváří selekční tlak.
  - Drift biased downward — `MAX_VISION_FOV = π` clamp znamená, že mutace
    může jen zužovat. Selekce může lineage push zpět k π přes přežití
    cells s širokým FOV; observed drift potvrzuje, že úzký FOV je
    competitive (energy savings > info-loss v current ekosystému).

- **Determinismus:** Sprint 83 je **první new baseline** — pre-S83 CSV
  reprodukovat nelze. Sprint 82 (pure infra, sigma=0) byl byte-identical
  s pre-Sprint-82 díky short-circuit pattern v mutate/crossover. Sprint 83
  rozejde díky (1) `sigma_vision_fov: 0 → 0.05` (gaussian draws v mutate
  aktivují se), (2) divergující fov hodnoty po prvních generacích spustí
  i bool draw v crossover, (3) cone filter v sensor gather mění behavior.
  Uvnitř Sprint 83 je deterministic (seed=0 byte-identical měření
  součástí dev workflow, ne test suite).

- **Test suite:** 112/112 pass (110 baseline z Sprint 82 + 2 nové:
  `fov_cone_accept_basic_directions`, `fov_cone_works_in_3d`).

- **Výstup:**
  - `src/lib.rs`: `fov_cone_accept` helper, `MUTATION_CONFIG.sigma_vision_fov`
    aktivován (0.0 → 0.05), 2 nové cone tests.
  - `src/main.rs`: cone filter v `cells_brain_act::gather` closure (food + cell
    neighbors).
  - `src/bin/headless.rs`: cone filter v `brain_act` + `brain_act_gpu` Phase 1.
    CSV header + extinction-row + write_stats writeln rozšířen o
    `fov_avg`/`fov_dev`.

- **Co Sprint 83 NEŘEŠÍ (S84+):**
  - Hunter směrový FOV — Sprint 84.
  - Brain awareness `vision_fov` (feed jako další input). Mozek už dostává
    `forward_vector` (inputs 9/10/18); explicit `vision_fov` input je
    pravděpodobně redundantní a otestuje se jen pokud Sprint 85+ ukáže
    informační deficit.
  - Multiple eyes (binocular FOV s overlap). Natural extension, čeká na
    selekční důkaz, že 1 eye optimální.
  - Brain output pro „active gaze" (cell může otáčet hlavou nezávisle na
    body heading). Aktuálně forward = body heading, takže FOV jde tam,
    kam jde tělo. Decoupling vision from body orientation by byl S88+
    feature.

## Sprint 84 — Hunter směrový FOV (forward-eye predator)

- **Cíl:** přenést směrový FOV na Hunter entitu. Pre-Sprint-84 měl Hunter
  omnidirectional vision (`HUNTER_VISION_RADIUS = 200` koule), takže žádný
  blind spot a žádná flanking strategy pro prey. Sprint 84 zavádí
  `HUNTER_VISION_FOV = π/3` (60° half-angle, 120° kužel) s forward derived
  from velocity. Cells mohou nyní uniknout do hunter's blind spotu, což
  vytváří selekční tlak na lateral evasion behavior.

- **Mechanismus:**
  - Konstanty (`src/lib.rs`): `HUNTER_VISION_FOV = PI / 3.0`,
    `HUNTER_FORWARD_SPEED_THRESHOLD_SQ = 1.0`. Druhá konstanta řeší startovní
    edge case: hunter s velocity = 0 nemá definovaný forward; pod threshold
    se cone filter vypne (omni fallback) aby idle hunter nikdy nezasekl bez
    targetu.
  - `nearest_attackable_cell` signature rozšířen o `hunter_velocity:
    [f32; 3]`. Implementace: speed_sq → cone_active flag → forward derived
    z normalized velocity (single sqrt). Filter aplikován po radius test +
    immunity test.
  - Vector orientace: pre-Sprint-84 byl `d = min_image_delta(c.position,
    hunter_pos, ...)` (= hunter − cell, sloužilo jen k d²); Sprint 84 swap
    na `min_image_delta(hunter_pos, c.position, ...)` (= cell − hunter)
    aby `dot(d, forward)` měl správné znaménko pro „target in front of
    hunter".
  - Call sites: `main.rs::step_hunters` + `headless.rs::hunt`.

- **Bez genomu, bez evoluce:** Hunter zůstává non-evolving world feature
  (Sprint 71 design). `HUNTER_VISION_FOV` je konstanta, ne gen — fixní
  predator profile, environment property.

- **Smoke (seed=0, 30 gen):**
  - Cells population stabilní (685 v gen 30). Hunter attacks/gen v
    rozsahu 228-494 — srovnatelné s pre-Sprint-84 baseline.
  - immune_frac < 1 % (cluster cells stále vzácné v krátkém runu).
  - cell `fov_avg` drift 3.14 → 3.00 (méně než Sprint 83 sám 3.14 → 2.90)
    — flanking advantage cells konkuruje energy-saving advantage úzkého
    FOV, lehce zpomalí drift. Nutný delší run (500+ gen) pro robustnější
    sledování.

- **Test suite:** 115/115 pass (112 z Sprint 83 + 3 nové: existing 3
  hunter tests aktualizovány o `[0.0; 3]` velocity argument; nové
  `hunter_cone_filters_blind_spot`, `hunter_cone_sees_front_target`,
  `hunter_cone_filters_flank_target`).

- **Výstup:**
  - `src/lib.rs`: 2 nové konstanty, `nearest_attackable_cell` signature
    expansion, vector orientation swap, 3 nové tests.
  - `src/main.rs`: 1 call site update v `step_hunters`.
  - `src/bin/headless.rs`: 1 call site update v `hunt`.

- **Co Sprint 84 NEŘEŠÍ (S85+):**
  - Hunter heading jako field (nezávisle na velocity). Aktuálně velocity
    = forward; pokud bude potřeba decouple (např. hunter sleduje target,
    ale „dívá se" jinam), bude třeba dedicated `Hunter.heading` + brain
    nebo policy logic.
  - Multiple hunters s overlapping FOVs (pack hunting). Aktuálně každý
    hunter je nezávislý.
  - Long-run (500+ gen) sweep `vision_fov` evoluce s + bez Hunter cone
    filter — porovnat selekční tlak.

## Sprint 85 — thermal stratification + sytější barvy

- **Cíl:** zavést první environmentální gradient ve vertikální ose. Pre-Sprint-85
  bylo cell prostředí prostorově homogenní (až na food noise field) — jediný
  z-faktor byla gravita (= 0 od Sprint 65) a food sink (8 u/s dolů). Sprint
  85 přidává **thermal stratification** přes Q10 metabolism multiplikátor:
  warm at top, cold at bottom. Cells nahoře platí ~2.46× více energie za
  všechny per-tick drains, cells dole ~0.41× — niche separation by depth
  emergne behaviorálně přes diferenciální přežití.

  Druhá část je vizuální: bump saturation 0.85 → 1.0 v `adhesion_material`
  a bond gizmo. Sytější barvy lépe kontrastují proti light ClearColor a
  ostřeji rozlišují adhesion-type clusters.

- **Mechanismus:**
  - Konstanty (`src/lib.rs`): `THERMAL_TOP = 30.0`, `THERMAL_BOTTOM = 4.0`,
    `THERMAL_Q10 = 2.0`, `THERMAL_REF_TEMP = 17.0` (mid-water = 1× faktor).
  - Helper `temperature_at_z(z, world_half) -> f32` — lineární gradient,
    fallback na REF_TEMP pro `world_half[2] = 0` (pre-3D backward-compat).
  - Helper `metabolism_factor(temp) -> f32 = Q10^((T - REF) / 10)`.
  - `Cell::apply_energy_costs` rozšířen o `world_half` parametr; per-tick
    multiplicates `dt_eff = dt × metabolism_factor` aplikovaný na **všechny**
    drains (motion, rotation, vision, body maintenance, spike, shell, attack
    upkeep). Jednotná Q10 sémantika — teplota škáluje rychlost biochemie.
  - `step()` call site update — nové `world_half` argument do
    `apply_energy_costs`.

- **GPU parita (`shaders/step.wgsl` + `gpu.rs`):**
  - `StepParamsGpu` rozšířen o 4 thermal pole (`thermal_top/bottom/q10/ref_temp`).
  - Shader `step.wgsl` mirror CPU `temperature_at_z` + `metabolism_factor`,
    všechny drain řádky × `dt_eff`.
  - `step_gpu_matches_cpu` parity test pass — cells s varying z dostávají
    correct metabolism factor na obou stranách.
  - `headless.rs --gpu-full` step_params populace o nové pole.
  - **Známý debt:** GPU shader nepočítá `vision_fov_factor` (Sprint 82). Při
    Sprint 82+83 testy passnuly náhodou — `Cell::random` nastaví fov = π
    všude → factor = 1.0 → no-op. Pokud by se kdy parity test rozšířil o
    cells s varying fov, shader by se rozejel s CPU. Aux buffer by potřeboval
    expansion ([f32; 4] → [f32; 5]) s rebind layoutem. Out-of-scope Sprint 85.

- **Renderer (`src/main.rs`):**
  - `adhesion_material`: `Color::hsl(hue, 0.85, 0.55)` → `Color::hsl(hue, 1.0, 0.55)`.
  - Bond gizmo (`draw_bond_gizmos`): `Color::hsl(hue, 0.85, 0.65)` → `1.0`.
  - Žádný visual API change, jen větší color saturation.

- **CSV diagnostika (`headless.rs`):** přidán sloupec `temp_avg` (52 → 53
  total). Per-cell `temperature_at_z(c.position[2], WORLD_HALF)` mean přes
  populaci. Sledování vertikální migrace — drift od REF_TEMP=17 indikuje
  evoluci niche preference.

- **Smoke (seed=0, 30 gen):**
  - Gen 0: temp_avg = 16.98 (init pop uniformně rozsazená v z, mean ≈ REF).
  - Gen 30: temp_avg = 10.92 (~6 sim-units pokles, populace migruje dolů
    k chladnějším zónám). Confirms behavioral selection — bez explicit
    thermal brain input populace najde cooler depth jen přes diferenciální
    energy drain → přežití.
  - Pop stabilní (200 → 712), žádný kolaps. fov_avg paralelně drift
    (3.14 → 2.82) — dvě nezávislé selekční síly aktivní současně.

- **Determinismus:** žádné nové RNG draws (analytické temperature, multiplicative
  drain). Sprint 85 je nový baseline kvůli energy formula change, ale uvnitř
  deterministic. CSV reprodukovatelný se stejným seedem.

- **Test suite:** 118/118 pass (115 z Sprint 84 + 3 nové: `temperature_at_z_endpoints`,
  `metabolism_factor_q10_ratio`, `apply_energy_costs_scales_with_temperature`).
  GPU parity test (`step_gpu_matches_cpu`) re-passing po shader update.

- **Výstup:**
  - `src/lib.rs`: 4 nové konstanty (`THERMAL_TOP/BOTTOM/Q10/REF_TEMP`),
    `temperature_at_z` + `metabolism_factor` helpers, `apply_energy_costs`
    signature + body update, 3 nové tests.
  - `src/main.rs`: 2 saturation bumps (body + bond gizmo).
  - `src/bin/headless.rs`: thermal_* fields v `StepParamsGpu` setup, CSV
    column + extinction-row + writeln rozšířen, per-cell temp_sum agregace.
  - `src/gpu.rs`: `StepParamsGpu` 4 nové pole, parity test setup updated.
  - `shaders/step.wgsl`: mirror temperature + metabolism × dt_eff.

- **Co Sprint 85 NEŘEŠÍ (S86+):**
  - Brain input pro thermal sensing — `thermal_norm` jako 21. sensory slot
    by vyžadoval `BRAIN_INPUTS_SENSORY: 20 → 21`, w1 matice resize, breaking
    change. Sprint 86+ kandidát pokud behavioral evolution (skrz z-pozici)
    příliš pomalá.
  - `thermal_optimum` gen — per-cell preferovaná teplota, evoluce hledá
    deviation z absolute Q10 univerzální křivky. S86+ pokud Sprint 85 long-run
    ukáže "all cells migrate down" trivial outcome.
  - Photic stratification — natural pair s thermal (depth-coupled niches).
  - GPU `vision_fov_factor` — latentní debt z Sprint 82.

## Sprint 86 — thermal day/night + seasonal cycle

- **Cíl:** static z-gradient teploty (Sprint 85) rozšířit o time-varying
  oscilace. Reálný oceán má (1) **diurnal** cycle — surface warms ve dne,
  cools v noci, hloubka stabilní (thermokline buffer); (2) **seasonal**
  cycle — celé volume warms/cools across měsíců. Sprint 86 obě v
  deterministickém analytickém formě (žádný field grid).

- **Mechanismus:**
  - Konstanty (`src/lib.rs`): `THERMAL_DIURNAL_AMP = 5.0` (peak surface
    oscilace ±5°), `THERMAL_DIURNAL_PERIOD_TICKS = TICKS_PER_GENERATION` (1
    day = 1 gen = 600 ticks = 10 s real-time), `THERMAL_SEASONAL_AMP = 4.0`
    (peak uniform shift). Seasonal period reusne `CYCLE_GEN_PERIOD = 50 gen`
    — sdílený s food density cyklem, takže warm season = abundant food,
    cold = scarce. Natural ecological coupling.
  - `temperature_at_z(z, world_half, tick, generation)`:
    1. **Base** = lineární z-gradient (Sprint 85 unchanged).
    2. **Seasonal** = `AMP × sin(TAU × (gen mod period) / period)`. Uniform.
    3. **Diurnal** = `AMP × normalized_z × sin(TAU × (tick mod period) / period)`.
       Surface-weighted (bottom = no oscillation, mirror reálné termokliny).
    Modulo přes period drží phase v [0,1) bez f32 precision loss long runs.
  - `metabolism_factor` time-agnostic (čistý Q10 power law).
  - `Cell::step` signature rozšířen o `tick: u64, generation: u64` před
    `physics`. Propagace do `apply_energy_costs` → `temperature_at_z`.
    Tests passing `0, 0` → sin(0) = 0 → zachována Sprint 85 behavior.

- **GPU parita (`shaders/step.wgsl` + `gpu.rs`):**
  - `StepParamsGpu` rozšířen o 4 nové f32 pole: `thermal_diurnal_amp/seasonal_amp`
    + `thermal_diurnal_phase/seasonal_phase`. Phases pre-computed CPU-side
    (`tick mod period / period`) aby shader nemusel řešit u64 modulo.
  - Shader mirror diurnal + seasonal offsets. Parity test passuje
    s phase = 0.

- **Smoke (seed=0, 60 gen — > 1 seasonal cycle):**
  - Gen 0: temp_avg = 16.98 (REF baseline).
  - Gen 25-40: temp_avg drop k 7.34 (peak winter cycle ~ gen 37.5 sin = -1
    + populace deep z-migrate). Sezónní + niche selection synergize.
  - Gen 50-60: temp_avg recovery 10.55 → 12.26 (warming back, populace
    drží deep niche).
  - fov_avg paralelně 3.14 → 2.41 (3 nezávislé selekční síly).
  - Pop stabilní (200 → 633), žádný kolaps.

- **Determinismus:** žádné nové RNG draws — diurnal i seasonal pure funkce
  `tick` a `generation`. Sprint 86 = nový baseline kvůli temp formula
  change, ale uvnitř deterministic. `tick = gen = 0` → output identical
  s Sprint 85.

- **Test suite:** 121/121 pass (118 z S85 + 3 nové: `temperature_diurnal_surface_oscillates`,
  `temperature_seasonal_uniform_shift`, `temperature_combined_seasonal_and_diurnal`).
  17 existing test step callsites updated o `0, 0` placeholder.

- **Výstup:**
  - `src/lib.rs`: 3 nové konstanty (`THERMAL_DIURNAL_AMP/PERIOD_TICKS/SEASONAL_AMP`),
    `temperature_at_z` + `apply_energy_costs` + `Cell::step` signature
    expansion, propagace tick/gen, 3 nové tests.
  - `src/main.rs`: `step_cells` přebírá `Res<Clock>`, propaguje tick/gen.
  - `src/bin/headless.rs`: `World::step` propaguje `self.clock.tick/gen`,
    `--gpu-full` `StepParamsGpu` populace o phase fractions, CSV
    `temperature_at_z` použije aktuální clock.
  - `src/gpu.rs`: `StepParamsGpu` 4 nové f32 (amps + phases), parity test
    setup s tick=0 fallback.
  - `shaders/step.wgsl`: seasonal + diurnal offset, identický s CPU.
  - `benches/headless_phases.rs`: 1 step callsite update.

- **Co Sprint 86 NEŘEŠÍ (S87+):**
  - Brain sensor pro thermal/temporal awareness. Cells žijí v time-varying
    prostředí ale neví o čase. Kandidát: `time_of_day` nebo
    `temperature_local`. Vyžaduje `BRAIN_INPUTS_SENSORY` resize.
  - Stochastic noise field — random temperature perturbace nad analytickou
    base.
  - Climate trend (monotonic warming) — open-ended evolution stress test.
  - Per-cell `thermal_optimum` gen.

## Sprint 87 — thermal awareness (sensor input + optimum gene)

- **Cíl:** dát buňkám "thermometer" — pre-Sprint-87 cells žily v thermal
  prostředí ale neměly k němu sensory access. Selekce probíhala pomalu
  čistě behaviorálně přes diferenciální energy drain (= „survive longer
  in cold zones"). Sprint 87 přidává (1) per-cell `temperature_local`
  brain input — cells "vidí" svou aktuální teplotu, mohou na ni reagovat
  v rámci života; (2) per-cell `thermal_optimum` gen — preferovaná
  teplota, drain kvadraticky penalizuje deviation. Společně tvoří
  thermal niche framework: brain učí "kde mě bolí", genom evolvuje
  "kde se cítím dobře".

- **Mechanismus:**
  - **Konstanty (`src/lib.rs`):** `MIN_THERMAL_OPTIMUM = THERMAL_BOTTOM`
    (4), `MAX_THERMAL_OPTIMUM = THERMAL_TOP` (30), `THERMAL_OPTIMUM_PENALTY
    = 1.0` (peak penalty/sec při |dev|/13 = 1.0, comparable s body cost).
  - **Genome:** `thermal_optimum: f32` ∈ [4, 30]. Init populace uniform
    random across range → speciation potential. Mutate gaussian s
    `sigma_thermal_optimum = 0.5` (~1.9 % range/gen) + clamp. Crossover
    standard bool draw.
  - **PhysicsConfig:** `thermal_optimum_penalty: f32` (default = const
    1.0). Tests + GPU parity override na 0.0 pro disable.
  - **Penalty drain (`apply_energy_costs`):** `((temp − optimum) / 13)² ×
    penalty × dt`. Independent of metabolism (thermal stress = extra
    cost, ne reduced enzyme rate). Cell s matching optimum platí 0;
    extreme deviation ~1.0/sec.
  - **Brain input slot 20:** `tanh((temp − REF) / 10)` ∈ [-1, +1]. Q10-
    aware škálování — REF→0, TOP→+0.86, BOTTOM→−0.86. Diurnal/seasonal
    swings mohou krátkodobě saturovat k ±1.
  - **`BRAIN_INPUTS_SENSORY: 20 → 21`**, `BRAIN_INPUTS: 52 → 53`. Breaking
    change — w1 matice resize, all brain weights re-randomized při
    `Genome::random`.
  - **`BrainSensors`:** + field `temperature_local: f32` (caller spočítá
    z pos[2] + clock).

- **Sensor gather (CPU):** main.rs `cells_brain_act` přebírá `Res<Clock>`,
  počítá `temperature_at_z(pos[2], world_half, tick, gen)` per cell před
  `populate_brain_inputs`. Headless `brain_act` + `brain_act_gpu` Phase 1
  capture `self.clock.tick/generation` před par_iter, computes per cell.

- **GPU shader updates:**
  - `brain_forward.wgsl` + `hebbian.wgsl`: hardcoded constants update
    (`BRAIN_INPUTS = 53u`, B1_OFFSET=1696u, W2_OFFSET=1728u, B2_OFFSET=2048u,
    `WEIGHTS_PER_CELL = 2058u`). Compile-time asserts v `gpu.rs` updated.
  - **`step.wgsl` thermal_optimum penalty: latentní debt** — aux buffer
    `[f32; 4]` by potřeboval expand na `[f32; 5]` pro per-cell optimum.
    Out-of-scope. Parity test (`step_gpu_matches_cpu`) override `physics.thermal_optimum_penalty
    = 0.0` aby se vyhne CPU↔GPU drift.
  - **`populate_inputs.wgsl` slot 20: latentní debt** — GPU shader nemá
    positions binding (12 bindings cap), nemůže spočítat thermal. Init
    loop zeroes all inputs, takže slot 20 = 0 v `--gpu-full` mode. Cells
    v `--gpu-full` mají thermal weights v brain (slot 20 column existuje),
    ale input vždy 0 → effectively zero contribution. Default headless
    (CPU-only) je unaffected.

- **CSV diagnostika (`headless.rs`):** přidány 2 sloupce `topt_avg`, `topt_dev`
  (53 → 55 total). Speciation tracking: gen 0 bude wide (uniform init,
  std ~7.5), narrowing přes selection.

- **Smoke (seed=0, 60 gen):**
  - Gen 0: `topt_avg=17.66`, `topt_dev=7.66` (uniform random init).
  - Gen 30: `topt_avg=16.80`, `topt_dev=4.17` (selection narrows).
  - Gen 60: `topt_avg=14.91`, `topt_dev=2.51` — population convergence
    k stabilizing-selection optimum mírně pod REF=17. **Bez bimodální
    speciace** (cold-prefer × warm-prefer split): seasonal cycle homogenizes
    populace — extrémní optima umírají v opačné půlce roku, mid-range
    survives. Přesně očekávaný outcome ve well-mixed environment bez
    barrier proti migraci.
  - `temp_avg` paralelně sleduje seasonal cycle (gen 35-40 winter trough
    6.0, gen 60 recovery 10.8).
  - `fov_avg` 3.14 → 2.86 (FOV evolution unchanged).
  - Pop stabilní (200 → 521).

- **Determinismus:** Sprint 87 = nový baseline kvůli BRAIN_INPUTS shape
  change (52 → 53) — `Brain::random` produces different weights per
  identical seed. Tato breaking change ekonomicky nevyhnutelná pro nový
  sensory slot. Žádné nové RNG draws v `apply_energy_costs` (deterministic
  Q10 + penalty math).

- **Test suite:** 124/124 pass (121 z S86 + 3 nové: `thermal_optimum_random_in_range`,
  `apply_energy_costs_thermal_stress_quadratic`,
  `populate_brain_inputs_writes_temperature_slot`). `mutation_keeps_genes_in_valid_ranges`
  + `crossover_picks_genes_from_either_parent` rozšířeny. GPU parity test
  passuje s thermal_optimum_penalty=0 override.

- **Výstup:**
  - `src/lib.rs`: 3 nové konstanty, BRAIN_INPUTS_SENSORY 20→21,
    `thermal_optimum` field na Genome, `sigma_thermal_optimum` na MutationConfig,
    `thermal_optimum_penalty` na PhysicsConfig, `temperature_local` na
    BrainSensors, populate_brain_inputs slot 20, apply_energy_costs penalty,
    3 nové tests, 6 literal callsites updated.
  - `src/main.rs`: `cells_brain_act` přebírá `Res<Clock>`, propaguje tick/gen
    do gather closure, vytváří `temperature_local` per cell.
  - `src/bin/headless.rs`: 2 sensor gather sites (CPU + GPU Phase 1)
    capture self.clock.tick/gen, vytváří `temperature_local`. CSV header
    + extinction-row + writeln rozšířen o `topt_avg`/`topt_dev`.
  - `src/gpu.rs`: BRAIN offsety asserts updated (1696/1728/2048/2058),
    parity test override `thermal_optimum_penalty = 0`.
  - `shaders/brain_forward.wgsl`, `shaders/hebbian.wgsl`: BRAIN_INPUTS=53u
    + offsets update.
  - `benches/headless_phases.rs`: BrainSensors literal +temperature_local field.

- **Co Sprint 87 NEŘEŠÍ (S88+):**
  - GPU step.wgsl thermal_optimum penalty (latentní debt, aux buffer
    expansion).
  - GPU populate_inputs.wgsl slot 20 (latentní debt, positions binding).
  - Spatial barrier (z-discontinuity) proti homogenizaci. Bez něj
    seasonal cycle drives populace k mid-temp optimum místo speciace.
  - Photic stratification (light gradient + photoreceptor sensor).
  - Brain output „active gaze".
  - Multiple eyes.

## Sprint 88 — atmospheric pass (renderer eye-candy v1)

- **Cíl:** transformovat dev-tool look (white background, flat plastic
  spheres) na art-piece look (deep ocean, bioluminescent cells). Quick-win
  renderer pass, žádný sim impact — pouze visual layer. HDR+Bloom dovolí
  emissive > 1.0 hodnoty bloom-out na halos; depth fog tints far objects
  modře (atmospheric perspective); cells & hunter dostávají emissive
  glow.

- **Mechanismus:**
  - **Cargo.toml:** `bevy_post_process` feature flag pro `Bloom`
    component access (Bevy 0.18 přesunul Bloom z `core_pipeline` do
    `bevy_post_process`).
  - **Camera (`main.rs:setup`):**
    - `Hdr` marker component — zero-sized, enables HDR backbuffer.
    - `Tonemapping::TonyMcMapface` — modern filmic curve, dobrý dynamic
      range pro biologické scény.
    - `Bloom::NATURAL` — energy-conserving preset, intensity 0.15. Cells
      s emissive > base získají soft halo.
    - `DistanceFog` color `srgb(0.04, 0.10, 0.18)`, density 0.0004
      ExponentialSquared. Fade-out far cells/floor do deep blue ambiance.
  - **ClearColor:** `Color::WHITE` → `srgb(0.02, 0.06, 0.12)` (deep
    blue-black). Match s fog color → no harsh edges between fog horizon
    a background.
  - **Lights:**
    - AmbientLight: `WHITE` brightness 1500 → `srgb(0.5, 0.7, 1.0)` (cool
      bluish), brightness 600. Underwater feel.
    - DirectionalLight: illuminance 12000 → 6000, color slight cool
      tint. Reduced kvůli HDR + Bloom — pre-S88 hodnoty by oversaturovaly
      bloom highlights k pure white.
  - **Cell material (`adhesion_material`):**
    - HSL lightness 0.55 → 0.45 (darker base, lets emissive dominate).
    - `emissive: hue × 0.8` linear — cells září vlastní hue (hue-coded
      bioluminescence). 8 distinct adhesion-type "tribes" jsou ostře
      rozlišené i v slabém ambient light.
  - **Hunter material:**
    - Base color stays dark red, emissive bumped na `LinearRgba(2.5, 0.2,
      0.1)` — super-saturated red glow (HDR linear > 1.0). Bloom catches
      → hunter visible jako menacing red beacon i z dálky přes fog.
  - **Bond gizmo lines:**
    - `Color::hsl(hue, 1.0, 0.6).to_linear() × 3.0` — multiplied linear
      space. Bevy gizmos render do HDR backbuffer, super-bright values
      Bloom catches → bondy svítí jako spring laser-lines, viditelné
      i v denním ambient.

- **Sim impact:** **žádný**. Visual-only change. Headless mode (no
  renderer) unaffected. CSV/determinism preserved. Žádné nové RNG draws.

- **GPU cost:** HDR backbuffer (RGBA16F místo RGBA8) + Bloom (mip pyramid
  generation) + fog (additional uniform) + ExtendedMaterial ne-používáme,
  takže emissive je v rámci stejného StandardMaterial pipeline. Estimated
  +1-2 ms/frame na GTX 1060+; minor cost vs sim hot path.

- **Test suite:** 124/124 pass — pure visual change, žádné sim regression.

- **NETESTOVÁNO vizuálně** — implementační agent nemá GUI access, nemůže
  spustit `cargo run` a porovnat output. User akce: `cargo run --features
  dev` a vizuální ověření. Pokud:
  - Cells nejsou vidět: bloom intensity příliš agresivní → snížit
    `Bloom::NATURAL.intensity` nebo lower emissive multiplier.
  - Fog příliš opaque: snížit `density` z 0.0004 na 0.0002.
  - Background příliš tmavé: ClearColor `srgb(0.02, 0.06, 0.12)` →
    bump na `(0.05, 0.10, 0.18)`.
  - Hunter "exploduje" v bloom: emissive `LinearRgba(2.5, ..)` → 1.5.

- **Výstup:**
  - `Cargo.toml`: + `bevy_post_process` feature.
  - `src/main.rs`: imports (Bloom, Tonemapping, DistanceFog, FogFalloff,
    Hdr), camera spawn rozšířen o 4 components, ClearColor change, lights
    re-tuned, `adhesion_material` emissive, hunter_material emissive boost,
    bond gizmo HDR-multiplied color.

- **Co Sprint 88 NEŘEŠÍ (S89+):**
  - Per-cell energy-modulated emissive (cells dim/bright dle energy).
    Vyžaduje per-tick material swap nebo ExtendedMaterial s instance
    attributes — větší feature.
  - Cell_state visual coupling (selfish blue ↔ altruist red blend).
    Vyžaduje multi-axis material cache.
  - Plankton particles / dust v pozadí.
  - Volumetric smell/pheromone field overlay.
  - God rays / light shafts.
  - Cell trails (motion streaks).
  - Predation flash / death ripple particles.
  - Subsurface scattering on cells (jelly translucency).
  - Cinematic camera modes / HUD graphs.

## Sprint 89 — Hunter evolution v1 (parametric genome, biological arms race)

- **Cíl:** Pre-Sprint-89 byl Hunter non-evolving environmental feature
  (S71 design): fixed konstanty řídily chování, žádný genom/energy/lifecycle.
  Cells evolvovaly proti hunteru, hunter zpět nikdy → asymmetric selection.
  Sprint 89 udělá z hunteru **evolvable entitu**: 8-gene heritable parameters,
  lifecycle (energy, reprodukce, smrt, floor respawn). Bez brain v této fázi —
  chování zůstává „seek nearest attackable prey" (S84), ale parametry jsou
  per-hunter genové. Coevolution emerge přes diferenciální přežití.

- **Mechanismus:**
  - **HunterGenome (`src/lib.rs`):** 8 genes — `vision_radius` ([50, 400]),
    `vision_fov` ([π/12, π]), `max_speed` ([100, 500]), `acceleration`
    ([40, 160]), `attack_radius` ([10, 40]), `damage_per_tick` ([2, 16]),
    `body_size` ([0.5, 2.5]), `color_hue` ([0, 360)). Init draws kolem S71-S84
    const middle ranges + ~30 % spread (initial diversity). `random()` /
    `mutate()` / `crossover()` jako Cell `Genome`.
  - **HunterMutationConfig:** sigma per-gene ~3 % range/gen — vyšší než cell
    `MUTATION_CONFIG` aby evolution signal byl viditelný v menší populaci.
  - **Hunter struct expansion:** + `genome`, `energy`, `age`, `reproduce_cooldown_ticks`,
    `lineage_id`, `lineage_birth_gen`. Constructor `Hunter::from_genome` +
    `Hunter::random`.
  - **Energy mechanics:** per-tick drain v `apply_energy_costs` — vision
    (`radius × fov_factor × VISION_COST`), motion (`v² × MOTION_COST`),
    body (`size³ × BODY_COST`), attack upkeep (`damage × ATTACK_UPKEEP`).
    Gain v hunt phase: `damage_dealt × ENERGY_PER_DAMAGE` per attack tick.
  - **Lifecycle:** death (energy ≤ 0 → drop `HUNTER_CARRION_DROP=2` carrion +
    despawn), reproduce (energy ≥ `HUNTER_REPRODUCE_THRESHOLD=800` AND
    cooldown 0 → split energy 50/50 + clone-with-mutate child), floor respawn
    (n_hunters == 0 → 1 fresh random genome aby nedošlo k total predator
    extinction blokující arms race), MAX_POP cap 50.
  - **`nearest_attackable_cell` signature:** `&Hunter` místo `(pos, vel)` —
    direct genome access pro vision_radius + vision_fov.

- **Tuning v2 (po initial smoke):** Pre-tuning hunters mass-died gen 1 (motion
  cost při v=300 dominated). Adjusted:
  - `HUNTER_INITIAL_ENERGY`: 300 → 500 (delší survival window)
  - `HUNTER_VISION_COST`: 0.03 → 0.01
  - `HUNTER_MOTION_COST`: 0.0015 → 0.0001 (klíčový fix — pre-tuning 1350
    energy/gen drain při v=300, post-tuning 90/gen)
  - `HUNTER_BODY_COST`: 1.0 → 0.5
  - `HUNTER_ATTACK_UPKEEP`: 0.05 → 0.02
  - `HUNTER_ENERGY_PER_DAMAGE`: 1.0 → 3.0 (attack net-positive při contact)
  - `HUNTER_REPRODUCE_THRESHOLD`: 600 → 800

- **Lifecycle systems:**
  - `main.rs::hunters_lifecycle` — Bevy system po `step_hunters`. Death pass
    (despawn + spawn carrion FoodEntity), reproduce pass (commands.spawn child
    HunterEntity), floor respawn (1 hunter pokud 0 alive). Resource
    `NextHunterId` monotonic counter.
  - `headless.rs::hunter_lifecycle` — mirror (Vec mutation místo ECS), volá
    se po `hunt()` v `tick()`.

- **CSV diagnostika (`headless.rs`):** 7 nových sloupců — `hunter_births`,
  `hunter_deaths`, `h_spd_avg`, `h_vis_avg`, `h_fov_avg`, `h_dmg_avg`,
  `h_size_avg`. CSV total 55 → 62 columns. Per-gen reset births/deaths
  counters v generation transition.

- **Smoke (seed=0, 100 gen) — ARMS RACE EMERGENCE:**

  | gen | cells | n_hunters | h_spd | h_dmg | h_size | cells_spd | ratio |
  |-----|-------|-----------|-------|-------|--------|-----------|-------|
  | 0   | 200   | 12        | 282   | 8.59  | 1.22   | 59.3      | 4.78× |
  | 30  | 691   | 6         | 324   | 11.27 | 1.41   | 124.2     | 2.61× |
  | 60  | 506   | 9         | 347   | 11.59 | 1.50   | 151.2     | 2.29× |
  | 100 | 540   | 18        | 348   | 11.71 | 1.49   | 191.4     | 1.82× |

  **Arms race signal:** hunter:cell speed ratio **4.78× → 1.82×** za 100 gens
  — cells caught up dramaticky (speed 59 → 191, +222 %), hunters partially
  caught up (282 → 348, +23 %). Damage drift (+36 %) ukazuje selekci na
  silnější útoky kompenzující rychlejší kořist. Pop dynamic (cells 200 →
  500-700, hunters 4-18) je Lotka-Volterra-like — žádná extinction events
  (díky floor respawn + MAX_POP cap), žádná runaway growth.

- **Determinismus:** Sprint 89 = nový baseline. `Hunter::random` má extra
  RNG draws (8 genome floats) per init — RNG sequence shifted od S88. Uvnitř
  S89 deterministic.

- **Test suite:** 129/129 pass (124 z S88 + 5 nových: `hunter_genome_random_in_range`,
  `hunter_mutate_clamps_to_range`, `hunter_crossover_picks_from_either_parent`,
  `hunter_apply_energy_costs_drains`, `make_hunter_child_splits_energy`).
  Existing 7 hunter tests aktualizovány (Hunter literals + nearest_attackable_cell
  signature change) — `make_test_hunter` helper s default genome.

- **Výstup:**
  - `src/lib.rs`: `HunterGenome` struct + impl, `HunterMutationConfig` +
    `HUNTER_MUTATION_CONFIG`, 8 gene-range konstant (MIN/MAX_HUNTER_*),
    9 lifecycle/energy konstant (HUNTER_INITIAL_ENERGY, REPRODUCE_THRESHOLD,
    MAX_POP, VISION/MOTION/BODY/ATTACK_UPKEEP costs, ENERGY_PER_DAMAGE,
    CARRION_DROP, REPRODUCE_COOLDOWN_TICKS), `Hunter` struct expansion,
    `Hunter::from_genome` + `apply_energy_costs`, `make_hunter_child` helper,
    `nearest_attackable_cell` signature, 5 nové tests + helper.
  - `src/main.rs`: `Hunter::random` call sites, `step_hunters` refactor
    (genome params, energy gain/drain), `hunters_lifecycle` system,
    `NextHunterId` resource, system registration.
  - `src/bin/headless.rs`: `World` + `next_hunter_id`/`hunter_births_gen`/
    `hunter_deaths_gen` fields, init + checkpoint paths updated, `hunt()`
    refactor, `hunter_lifecycle()` method, CSV writer + extinction-row + header
    + per-gen reset.

- **Co Sprint 89 NEŘEŠÍ (S90+):**
  - **Hunter brain** — adaptive ambush/chase tactics, prey selection,
    coordinated hunting. Big lift (Brain struct na Hunter, BRAIN_INPUTS
    layout, GPU shader update). Pokud S89 long-run ukáže stable dynamics,
    Sprint 90 přidá brain pro deeper coevolution.
  - **Sexual reproduction for hunters** — pairing logic + crossover.
    Asexual v1 dostatek pro arms race signal.
  - **Hunter shell / armor gene** — defensive evolution. Cells mají shell,
    hunters by mohli taky.
  - **Inter-hunter cannibalism / pack hunting** — currently hunters ignorují
    sebe. Group dynamics by mohly emerge přes brain (S90+).
  - **Hunter checkpoint serialization** — current load_checkpoint resets
    hunters s random genomy (lineage reset). Bincode serialize HunterGenome
    by zachoval evolution napříč session.

## Sprint 90 — Hunter brain (hybrid seek + brain modulation)

- **Cíl:** Sprint 89 ukázal hunter parameter evolution (genome drift). Brain
  přidává adaptive chase tactics — random cell brain s INNATE_THRUST_BIAS
  startuje s forward motion, evolution tuneuje turn/pitch outputs k
  prey-coordinated chase.

- **Mechanismus:**
  - **HunterGenome.brain:** reuse cell `Brain` struct (BRAIN_INPUTS=53,
    HIDDEN=32, OUTPUTS=10). Slot semantics re-mapped pro hunter:
    0/1/15 = nearest_prey delta, 4 = own_energy, 5 = own_speed, 6 =
    prey_size_relative, 7-8/17 = smell_grad, 9-10/18 = heading, 13 =
    density. Used outputs: 0 (turn), 1 (thrust), 7 (pitch). Cell-only
    outputs (morph, attack, bond) ignored.
  - **Hunter struct:** + `heading`, `pitch`, `angular_velocity`, `pitch_velocity`,
    `last_inputs/hidden/outputs`. Hunter::from_genome inits heading random,
    pitch 0, brain state zero.
  - **`HunterBrainSensors`:** `nearest_prey` delta, `nearest_prey_size`,
    `neighbors_in_vision`, `smell_grad`. Linear scan cells (n_hunters max
    50, cell pop ~500 → 25k pair compares × 50 hunters = 1.25M ops/tick).
  - **`gather_hunter_sensors`:** filter cells by vision_radius + cone
    (genome) + `n_bonds < HUNTER_BOND_IMMUNITY_THRESHOLD`. Returns nearest
    + count + smell.
  - **`populate_hunter_brain_inputs`:** maps sensors → `[f32; BRAIN_INPUTS]`
    s hunter-specific slot semantics (= cell layout, repurposed). Recurrent
    last_hidden copied to slots 21..52.
  - **`Hunter::apply_brain_motor` HYBRID design:** `seek_mix = 0.6` —
    deterministic seek-toward-prey direction mixed s brain output (40 %).
    Bez tohoto random initial brain neumí chase (random turn output =
    spinning), populace kolabuje do floor respawn loop. S hybridem brain
    moduluje dominantní seek (např. learned prey selection, retreat při
    low energy). Když brain weights evolvují k matching seek, mix se stane
    redundant; když brain learnuje jiný strategy, brain dominuje.
  - **`Hunter::step` refaktor:** pure kinematic integration (position +
    heading + pitch), žádný seek logic. Caller volá `apply_brain_motor`
    před step.
  - **HUNTER_TURN_RATE = 3.0, HUNTER_PITCH_RATE = 1.0** (mid-cell range).
    Sprint 91+ může přidat jako gene.

- **Energy economics tuning v3:**
  - Pre-tune (V1 pure brain): pop crashed to 1, no reproduction —
    random brain s thrust ale random turn nestíhá chase.
  - V2 (hybrid 60/40): better chase ale energy still pod-tuned, single
    hunter survives via floor respawn.
  - **V3 final:** `HUNTER_ENERGY_PER_DAMAGE: 3.0 → 6.0`,
    `HUNTER_REPRODUCE_THRESHOLD: 800 → 700` — predace teď net-positive,
    hunter populace dosáhne carrying capacity 50 v ~30 gens.

- **Smoke seed=0 100 gen — predator-prey equilibrium s arms race:**

  | gen | cells | c_spd | hunters | atk/gen | h_spd | h_dmg |
  |-----|-------|-------|---------|---------|-------|-------|
  | 0   | 200   | 59.3  | 12      | 0       | 285   | 8.95  |
  | 10  | 845   | 87.3  | 8       | 506     | 290   | 8.85  |
  | 30  | 730   | 134.9 | 17      | 2066    | 262   | 11.85 |
  | 40  | 718   | 156.4 | **50**  | 6675    | 249   | 11.78 |
  | 60  | 514   | 183.1 | 50      | 6985    | 249   | 11.78 |
  | 100 | 730   | 191.4 | 50      | 7731    | 249   | 11.78 |

  - **Cells +222 %** speed (59 → 191) — strong selection pod intense predation
    (2000-7000 attacks/gen vs S89 200/gen).
  - **Hunters hit MAX_POP** v gen 40, sustainable equilibrium (0 deaths
    + 0 births since cap blokuje reproduction).
  - **Hunter genome frozen** at gen 30 progenitor (h_spd=249, h_dmg=11.8) —
    cap reached před selection diversity. **Limitace:** asexual reproduction +
    pop cap → evolution stagnates po dosažení carrying capacity.

- **Determinismus:** Sprint 90 = nový baseline (Brain init na hunter +
  brain forward weights v každém ticku). Brain forward je deterministic
  given inputs/weights. RNG draws posunuty.

- **Test suite:** 132/132 pass (129 z S89 + 3 nové: `hunter_apply_brain_motor_thrusts_forward`,
  `hunter_apply_brain_motor_turn_yaw_sets_angular`, `hunter_apply_brain_motor_seek_dominates_chase`,
  + replaced 2 stale tests s pure-seek pattern). Plus
  `populate_hunter_brain_inputs_writes_prey_delta`.

- **Výstup:**
  - `src/lib.rs`: `HUNTER_TURN_RATE` + `HUNTER_PITCH_RATE` consts,
    `HunterGenome.brain` field, brain integration v random/mutate/crossover,
    `HunterMutationConfig.sigma_brain` field, `Hunter` struct expansion
    (heading/pitch/angular_velocity/pitch_velocity/last_inputs/hidden/outputs),
    `Hunter::from_genome` init nová pole, `Hunter::apply_brain_motor` (hybrid),
    `Hunter::step` refaktor (pure integration), `HunterBrainSensors` struct,
    `gather_hunter_sensors` + `populate_hunter_brain_inputs` helpers,
    energy economics tune (PER_DAMAGE 3→6, REPRODUCE_THRESHOLD 800→700),
    4 nové tests + replaced 2.
  - `src/main.rs`: `step_hunters` přebírá `Res<SmellResource>`, brain
    forward + hybrid motor + step pipeline.
  - `src/bin/headless.rs`: `hunt()` — sensor gather + brain forward +
    hybrid motor + step.

- **Co Sprint 90 NEŘEŠÍ (S91+):**
  - **Sexual reproduction** — pairing logic + crossover mezi dvěma
    hunters. Asexual + cap → genome stagnates po carrying capacity reached.
    Sexual by zachoval diversity přes crossover.
  - **Hebbian learning na hunter brain** — currently brain weights jen
    from genome inheritance. Hebbian by dovolil within-life learning.
  - **GPU brain forward pro hunters** — currently CPU-only. S 50 hunters
    × forward pass každý tick je ~OK, ale GPU integration by sjednotila
    pipeline s cells.
  - **Larger hunter brain inputs** — některé sensory slots jsou nevyužité
    (cell-cell delta, pheromone, damage). Mohly by být repurposed pro
    predator-specific signály (např. inter-hunter awareness pro pack
    hunting).
  - **Higher MAX_POP** — 50 cap je tight, evolution stagnates rychle.
    Vyšší cap (100-200) by dovolil delší selekční signal před equilibrium.

## Sprint 91 — procedural bio-textures (cell + hunter)

- **Cíl:** Sprint 88 atmospheric pass dal HDR + bloom + fog. Sprint 91 přidává
  „sexy" surface detail — Voronoi-based procedural pattern přes `ExtendedMaterial<StandardMaterial,
  BioMaterialExt>`. Single shader handles obě cell + hunter, parametrizováno
  `pattern_kind` uniformem.

- **Mechanismus:**
  - **Shader (`assets/shaders/bio_material.wgsl`):** PBR pipeline reuse —
    fragment override calls `pbr_input_from_standard_material`, then modifies
    `pbr_input.material.{base_color, emissive, perceptual_roughness, metallic}`,
    then calls `apply_pbr_lighting + main_pass_post_lighting_processing`.
    Procedural coord = `world_normal × scale` (texture rotates s mesh, ne
    drift při motion). 3D Voronoi (3-loop neighbor scan) → F1, F2 distances
    → `edge = smoothstep(0.0, 0.2, F2 - F1)` (peaks na cell borders).
  - **Pattern_kind 0 (CELL — jelly membrane):** bright Voronoi edges
    (membrane web), dim cores (cytoplasm). Emissive boost +250% na edges →
    bioluminescent border glow pod bloom. Roughness 0.3 (smooth) v core,
    0.7 (rougher) na edges.
  - **Pattern_kind 1 (HUNTER — chitinous scales):** inverse — dark edges
    (between scales), bright centers (scale plates). Partial metallic 0.4
    na edges → reflective armor look. Higher emissive na centers (2.5×).
  - **Rust Material (`main.rs`):**
    - `BioMaterialExt` struct s 4 uniforms na binding 100 (pattern_kind,
      scale, intensity, _pad). `Asset + AsBindGroup + Reflect + Default`.
    - `MaterialExtension` trait impl — `fragment_shader()` + `deferred_fragment_shader()`
      vrátí `BIO_SHADER_PATH = "shaders/bio_material.wgsl"`.
    - Type alias `BioMaterial = ExtendedMaterial<StandardMaterial, BioMaterialExt>`.
    - `MaterialPlugin::<BioMaterial>::default()` registrován v App builderu.
  - **Resource type changes:**
    - `AdhesionMaterials([Option<Handle<StandardMaterial>>; 8])` → `Handle<BioMaterial>`.
    - `HunterMaterial(Handle<StandardMaterial>)` → `Handle<BioMaterial>`.
    - `setup` system + `cell_reproduces_on_threshold` přebírají `ResMut<Assets<BioMaterial>>`
      navíc. `adhesion_material` function vrací `Handle<BioMaterial>`.
  - **Shader scale tuning:** cell scale=6 (medium voronoi tiles na povrchu
    sphere — viditelné membrane segments), hunter scale=14 (denser scales
    pro chitinous look).

- **Determinismus:** Pure visual change — žádný sim impact. Headless mode
  (no renderer) unaffected. CSV/test suite preserved.

- **Test suite:** 132/132 pass — žádná sim regression. `cargo check --bins
  --benches` clean.

- **NETESTOVÁNO vizuálně** — implementační agent nemá GUI access. User akce:
  `cargo run --features dev` a vizuální ověření. Pokud:
  - Shader compile error: Bevy 0.18 PBR shader naming conventions se mohly
    změnit oproti starší verzi. Check `bevy_pbr-0.18.1/src/render/pbr_fragment.wgsl`
    pro správné import paths.
  - Pattern příliš noisy: snížit `scale` (cell 6 → 4, hunter 14 → 10).
  - Pattern invisible: zvýšit `scale` nebo `intensity`.
  - Edge brightness wrong: tweak `mix(0.4, 1.4, edge)` ranges v shaderu.
  - Hunter scales vypadají flat: bump `metallic` mix range.
  - Black artifacts: některé mesh edges nemají world_normal interpolated
    správně — `Sphere::ico(2)` má 162 verts, hrubé na denser pattern; bump
    `ico(3)` (642 verts) by pomohl.

- **Výstup:**
  - `assets/shaders/bio_material.wgsl`: nový procedural shader (~80 řádků,
    Voronoi + PBR moduulace).
  - `src/main.rs`: imports (ExtendedMaterial, MaterialExtension, AsBindGroup,
    ShaderRef), `BioMaterialExt` struct, `BioMaterial` type alias,
    `MaterialPlugin` registration, `AdhesionMaterials` + `HunterMaterial`
    type changes, `setup` + `cell_reproduces_on_threshold` system signatures
    (+ `ResMut<Assets<BioMaterial>>`), `adhesion_material` function rewrite,
    hunter material spawn rewrite.

- **Co Sprint 91 NEŘEŠÍ (S92+):**
  - **Time-animated patterns** — current shader je statický (žádný `time`
    uniform). Dodal by pulsing/breathing effect. Vyžaduje globals binding.
  - **Per-cell pattern variation** — všechny cells s adhesion_type X mají
    identický pattern. Mohly by mít unique seed (cell_id) přes vertex
    instance attributes.
  - **Subsurface scattering** — translucent jelly look (Bevy 0.18 má
    `StandardMaterial.subsurface_intensity`).
  - **Animated UV warp** — vertex shader manipulace pro pulsing.
  - **Energy-modulated emissive** — material reacts to cell.energy (low =
    dim, high = bright pulse). Vyžaduje per-instance uniform updates.

## Sprint 92 — edge-vulnerability + multi-trophic food

- **Cíl:** dva souběžné continuous-gradient pressures driving cluster
  complexity. Pre-Sprint-92: binary `n_bonds ≥ 2` immunity threshold (S76)
  byl tipping point — po dosažení žádný další tlak na complexity. Sprint 92
  replaces s **gradient damage scaling** (více bondů = méně damage) +
  **multi-trophic food chain** (cluster diversifikace přes diet specialization).

- **Mechanismus #1 — Edge-vulnerability:**
  - Threshold `HUNTER_BOND_IMMUNITY_THRESHOLD: 2 → 4` — repurpose jako
    discoverability cap (cells s ≥4 bondy hunter ignoruje, čistě efficiency
    rozhodnutí).
  - `cell_exposure(n_bonds) = max(0, 1 - n_bonds × EXPOSURE_PER_BOND)`,
    `EXPOSURE_PER_BOND = 0.25`. Damage applied = `damage_per_tick × exposure × dt`:
    - 0 bonds → 1.0 (full damage, solo cell)
    - 1 bond → 0.75
    - 2 bonds → 0.5
    - 3 bonds → 0.25
    - ≥4 bonds → 0.0 (effectively immune; threshold skipne i target lookup)
  - Selection pressure: maximalizovat surface-area-to-volume ratio cluster
    (= sphere). Drive toward větší 3D clusters s many bonds, ne jen S78 line-of-pairs.

- **Mechanismus #3 — Multi-trophic food chain:**
  - **`FoodKind` enum** (Plant=0, Carrion=1, HunterCarrion=2). Food struct +
    `kind` field s serde default = Plant pro backward-compat.
  - **Per-kind base value:** `PLANT_FOOD_VALUE = 20.0`, `CARRION_FOOD_VALUE
    = 30.0`, `HUNTER_CARRION_FOOD_VALUE = 50.0`. Cell carrion má větší value
    (concentrated biomass), hunter carrion ještě víc (apex predator drop).
  - **`carnivore_score: f32` ∈ [0, 1]** gen na `Genome`. 0 = pure herbivore
    (plant only), 1 = pure carnivore (hunter carrion only). Init range
    [0.0, 0.3] — bias herbivore aby cells survive cold start.
  - **`eat_efficiency(kind, score)`:** continuous trade-off:
    - `Plant + score` → `1 - score` (herbivore best)
    - `Carrion + score` → `0.5` (universal compromise food)
    - `HunterCarrion + score` → `score` (carnivore best)
  - **Drop tagging:** cell death → `Carrion`, hunter death → `HunterCarrion`,
    ambient spawn → `Plant`. SpatialGrid<usize, FoodKind> carries kind v
    grid lookup → eat path má kind bez extra Query.
  - **Synergy:** mixed-diet cluster outperforms monoculture protože má
    access ke všem food types. Per-cell specialization cooperates v
    cluster — herbivore cells na plant patches, carnivore cells na hunter
    carrion drop sites.

- **Genome / MutationConfig changes:**
  - `Genome.carnivore_score: f32` (serde default 0.0).
  - `MutationConfig.sigma_carnivore_score: f32`, default `0.02` (~2 %
    range/gen, pomalejší než ostatní geny aby selekce přes food availability
    signal stihla operate).
  - Genome::random init `[0.0, 0.3]`, mutate short-circuit pattern (sigma=0
    → no draw).
  - Crossover: standard bool draw.

- **Determinismus:** Sprint 92 = nový baseline. RNG draws shifted o
  carnivore_score initial draw. Inside S92 deterministic.

- **Test suite:** 136/136 pass (132 z S91 + 4 nové: `cell_exposure_endpoints`,
  `eat_efficiency_diet_specialization`, `food_base_value_per_kind`,
  `carnivore_score_in_genome_random_initial_range`). Existing 2 hunter
  immunity tests aktualizovány na nové threshold (HUNTER_BOND_IMMUNITY_THRESHOLD
  místo hardcoded 3).

- **Smoke seed=0 60 gen:**
  - Mechanically funguje — cells evolve normálně, hunters lifecycle běží,
    food kinds taggované.
  - `immune_frac = 0.000` napříč gens — **selekční signal: žádný cluster
    nedosáhne 4-bond threshold**. Pre-S92 binary immunity (≥2) byla
    dosažitelná (~3-5 % cells), nový 4+ je hard cap.
  - Gradient damage funguje implicitně — cells s 1-3 bondy berou redukovaný
    damage proporčně k exposure. Bez explicit metric ale viditelné v
    survival rate.

- **Tuning concerns:**
  - Hunter pop dropped k 1 (vs S90 cap 50) — exposure scaling reduces avg
    damage per attack tick → hunter energy gain klesá → reproduce threshold
    není dosažen často → pop nečerpá nahoru. Sprint 93+ bude tunit
    HUNTER_ENERGY_PER_DAMAGE up nebo HUNTER_REPRODUCE_THRESHOLD down.
  - Initial carnivore_score range [0, 0.3] může být too restrictive —
    žádné carnivore cells initial → hunter carrion useless waste of compute
    until mutation drift produces score > 0.5. Možný bump initial range to
    [0, 0.5].
  - CSV column `immune_frac` semantics shifted (≥4 bondy místo ≥2). Sprint
    93 by mohl přidat `mean_exposure` nebo `n_bonds_avg` jako lepší
    diagnostic.

- **Výstup:**
  - `src/lib.rs`: `EXPOSURE_PER_BOND` const, threshold bump 2→4, `cell_exposure`
    helper, `FoodKind` enum + Food struct field, `PLANT/CARRION/HUNTER_CARRION_FOOD_VALUE`
    consts, `food_base_value` + `eat_efficiency` helpers, `carnivore_score`
    on Genome, `sigma_carnivore_score` on MutationConfig, mutate/crossover/random
    integration, 4 nové tests, 2 hunter tests aktualizovány.
  - `src/main.rs`: `FoodGrid<Entity, FoodKind>`, food spawn sites tagují
    kind (Carrion na cell death, HunterCarrion na hunter death), eat path
    používá `eat_efficiency(food_kind, carnivore_score)` a `food_base_value(kind)`,
    hunter damage scales s `cell_exposure`.
  - `src/bin/headless.rs`: stejné — `food_grid: SpatialGrid<usize, FoodKind>`,
    eat path s efficiency, damage exposure, hunter death drops HunterCarrion.

- **Co Sprint 92 NEŘEŠÍ (S93+):**
  - **Visual differentiation food kinds** — všechny food entities mají
    same green material. Plant/Carrion/HunterCarrion by mohly mít distinct
    colors (green/brown/red).
  - **Cluster-shape evolution metric** — surface:volume ratio diagnostic
    pro CSV.
  - **Carnivore_avg + food_kind_distribution v CSV** — track diet
    specialization přes generations.
  - **Hunter economy re-tune** — exposure scaling reduces avg damage per
    attack → hunter pop kolabuje. Sprint 93 bump ENERGY_PER_DAMAGE.
  - **Bond formation incentive** — current bond formation requires brain
    output[9] threshold. Cluster grows pomalu; gradient defense reward
    není guaranteed selekční tah pokud bond formation cost dominates.

## Sprinty 93+ — open-ended

- **Sprint 87+:** Long-run sweep (500-1000 gen) s monitoring `fov_avg` +
  `temp_avg` trajektorie. Hypotézy: úzký FOV (~π/4 .. π/2) emergne pokud
  cone filter vytváří dostatečný informační deficit; populace stabilizuje
  v cold-deep niche pokud Q10 selekce dominuje, nebo udrží warm-shallow
  lineage pokud food gradient kompenzuje thermal cost.
- **Sprint 87+:** Brain input pro thermal/temporal sensing — `thermal_norm`
  + `time_of_day_phase` jako 21./22. sensory sloty. Vyžaduje
  `BRAIN_INPUTS_SENSORY: 20 → 22`, w1 resize.
- **Sprint 87+:** `thermal_optimum` gen — per-cell preference. Driver pro
  diversifikaci napříč z-vrstvami místo trivial "all migrate down".
- **Sprint 87+:** Photic stratification (z-gradient light field +
  photoreceptor sensor input). Natural pair s thermal — depth-coupled
  niches.
- **Sprint 87+:** Climate trend (monotonic warming) — open-ended evolution
  stress test, populace musí evolvovat rychleji než klima.
- **Sprint 87+:** Stochastic temperature noise field — random perturbace
  nad analytickou base, lokální hot/cold spots.
- **Sprint 87+:** GPU `vision_fov_factor` v step.wgsl (latentní debt
  z Sprint 82) — vyžaduje aux buffer expansion.
- **Sprint 87+:** renderer screencast + HUD bond/state stats (z S80
  odsunuto).
- **Sprint 87+:** Cluster reproduction (S70 retry s S78 baseline).
- **Sprint 87+:** `BOND_FOOD_SHARE_FRAC` sweep (0.1, 0.5, 0.7).
- **Sprint 87+:** GPU collision shader, anisotropic collision.
- **Sprint 87+:** Spatial autocorrelation adhesion_type clustering metric.
- **Sprint 87+:** Brain output pro „active gaze" (decouple FOV direction
  od body heading).
- **Sprint 87+:** Brain `vision_fov` input — feed half-angle do mozku.
- **Sprint 87+:** Multiple eyes (2 sensory cones s overlap).
