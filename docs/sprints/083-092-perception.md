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

## Sprinty 87+ — open-ended

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
