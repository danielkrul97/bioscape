# Sprinty 63–72: Scale-up

Decade rozšiřuje sim scope — víc cells, větší volume — ne další performance
work. Sprint 60-62 série ukázala že GPU offload je net-negative pro current
sim shape (1k–10k cells, 64×64×16 grid) napříč všemi paths kvůli per-tick
GPU dispatch overhead. CPU paralelní cesta (Sprint 57+58) drží excellent
scaling. Sprint 63 wired step na GPU jako infrastructure (single Wait barrier),
Sprint 64 expandoval volume (z=50) + populaci (2500). Sprint 65+ pokračuje
horizontálně (thermal/light fields, bigger world) místo vertical perf-pursuit.

## Sprint 63 — step on GPU + bigger workload benchmark

- **Cíl:** wire `StepGpu` (Sprint 50 standalone shader) do `brain_act_gpu_full`
  pipeline po Sprint 62 motor + brownian fuze. Plus změřit bigger workload
  (5k a 10k cells) pro CPU vs GPU comparison — určit GPU break-even point.

  **Plán implementace:**

  *Body 0 — Bigger workload baseline (CPU):*
  - Headless 60 gen smoke s `MAX_POPULATION + INITIAL_CELLS`:
    - 5k cells: 41.4 s = **870 ticks/s** (vs 1k = 977, –11 % linear scaling).
    - 10k cells: 49.6 s = **726 ticks/s** (vs 5k = 870, –17 %, super-linear).
  - CPU paralelní cesta drží excellent scaling napříč 1k–10k cells díky
    Sprint 57 FxHashMap + rayon stencil. Per-fáze 10k:
    - collisions 1269 µs (super-linear),
    - predate 2018 µs,
    - eat_food 745 µs (food count nezávislý na N).

  *Body 1 — `CellsGpu` rozšíření o step buffery:*
  - 9 nových buffers + readbacks: `position_buf`, `age_buf`, `cooldown_buf`,
    `body_dims_buf`, `aux_buf` (4 floats: spike, shell, vision, attack);
    + `position_rb`, `age_rb`, `cooldown_rb`, `energy_rb`.
  - Upload helpers: `upload_positions`, `upload_age_cooldown`,
    `upload_body_dims`, `upload_aux`.
  - `download_full_batch` extends Sprint 62 `download_brain_motor_batch` na
    9 buffers v jediném Wait barrier (hidden + outputs + velocity + ang_vel
    + pitch_vel + position + age + cooldown + energy).

  *Body 2 — `StepGpu::dispatch_with_cells`:*
  - Variant of `compute()` co bind shared `&CellsGpu` buffery místo own
    duplicates (mirror MotorGpu Sprint 62 pattern). Step shader (Sprint 50
    `step.wgsl`) beze změny — mirror lib::Cell::step kinematics + drag +
    energy + bounce.

  *Body 3 — `brain_act_gpu_full` 11-fáze pipeline:*
  - Phase 1: CPU snapshot rozšířen o positions/ages/cooldowns/body_dims/
    aux (5 nových Vec extracts).
  - Phase 2: 4 nové uploads (positions, age+cooldown, body_dims, aux).
  - Phase 9: `step.dispatch_with_cells` dispatched po brownian. Mutuje
    position/velocity/heading/pitch/ang_vel/pitch_vel/age/cooldown/energy
    GPU-side.
  - Phase 10: `download_full_batch` (single Wait, 9 buffers).
  - Phase 11: CPU writeback all 9 vec do cell state. NO Cell::step CPU
    (work bylo GPU-side).

  *Body 4 — `step` fáze v `--gpu-full` skip:*
  - Pokud `gpu_full.is_some()`, fáze early-return. Mirror Sprint 62
    `apply_brownian` skip pattern. CPU `Cell::step` lib helper zachován.

- **Konstanty:** žádné nové. Re-export `AGE_DECAY_PER_SEC`,
  `ATTACK_COST_PER_SEC`, `GRAVITY`, `SPIKE_COST_PER_SEC`, `SHELL_COST_PER_SEC`
  z lib pro StepParamsGpu fill.

- **Výstup:**
  - `src/gpu.rs`:
    - `CellsGpu` rozšířen o 5 step buffers + 4 readbacks + 4 upload helpers
      + `download_full_batch`.
    - `StepGpu::dispatch_with_cells` variant.
  - `src/bin/headless.rs`:
    - `GpuFullState` přidává `step: StepGpu`. Init.
    - `brain_act_gpu_full` 11-fáze pipeline. CPU snapshot rozšířen.
    - `step` fáze early-return v `--gpu-full`.
  - **Test suite: 73/73 pass** (1 flaky `random_brain_average_thrust_is_positive`
    — pre-Sprint-57 issue, unseeded `rand::rng()`).

  **Smoke seed=0, 60 gen, default world s `--gpu-full`:**

  | Cells | Sprint 62 ticks/s | Sprint 63 ticks/s | brain_act+step µs S62→S63 |
  |-------|-------------------|-------------------|---------------------------|
  | 1000  | 270               | **277 (+2.6 %)**  | (3463+17) → (3384+0) = -96 µs |
  | 5000  | 262               | **263 (+0.4 %)**  | (4407+156) → (4760+0) = +197 µs |

  Pop final 1k: 514 (Sprint 62 487, CPU 572 — drift v noise rangi).
  Pop final 5k: 553 (Sprint 62 591, CPU 538 — drift acceptable).

- **Závěr:**
  - **GPU step win pro 1k cells** (CPU step jen 17 µs, fuze do brain_act
    eliminuje rayon spawn + L1 cache touch = -96 µs).
  - **GPU step lose pro 5k cells** (+197 µs regrese — per-tick metadata
    upload 320 KB + GPU dispatch overhead překonal CPU step 156 µs).
  - **Trend: GPU full-pipeline degraduje s rostoucím N**, opačně než
    initial expectation. Per-tick upload bandwidth (positions 5k×3×4 =
    60 KB, body_dims 60 KB, aux 80 KB, atd.) + 7+ dispatch passes overhead
    sčítá rychleji než CPU paralelní per-fáze čas.

  **5k full-stack benchmark final:**

  | Path | Wall-clock | Ticks/s | vs CPU 870 |
  |------|------------|---------|------------|
  | CPU paralelní (Sprint 57+58) | 41 s | **870** | baseline |
  | GPU `--gpu-full` (Sprint 63) | 137 s | 263 | **3.3× slower** |

  10k extrapolation: CPU 726 ticks/s, GPU expected ~150-200 ticks/s
  (4-5× slower).

- **Poznámky:**
  - **Round-trip status: 1 `device.poll(Wait)` per tick zachován** přes
    Sprint 62→63 fuze step. Achievable minimum bez full GPU loop (eat_food,
    predate, collisions stále CPU).
  - **Per-tick upload cost je hot:** 5k cells = ~324 KB upload/tick = ~20 MB/s
    při 60 Hz target. PCIe bandwidth pohodlně, ale CPU snapshot extract
    cost (Vec::collect × 5 = ~50 µs/tick @ 5k) přidává to brain_act fáze.
  - **GPU break-even nepřišel.** Pre-Sprint-63 očekávání "5-10k cells = GPU
    win" bylo wrong. CPU paralelní cesta drží excellent scaling díky
    L1/L2 cache friendly access pattern (rayon par_iter_mut na contiguous
    Vec<Cell>) + FxHashMap fast bucket lookup. GPU pipeline má fixed
    per-tick overhead (5+ dispatch passes × 50-200 µs each = 250-1000 µs)
    který nikdy neamortizuje.
  - **Co Sprint 63 NEŘEŠÍ (Sprint 64+):**
    - GPU collision wire-up — atomic delta scratch + 2-pass apply.
    - GPU predate wire-up — 2-pass herd_count + attack with atomic floats.
    - Eat_food / reproduce / spawn_food / die_and_drop_carrion — sparse
      Vec mutations, hard to GPU-ify.
    - GPU per-tick upload elimination — držet CPU positions stale, pouze
      sync na konci epochy. Vyžaduje plný GPU loop pro **all** mutating
      phases, jinak CPU sync per phase je ne-skipnutelný.
    - Bigger workload (50k+) benchmark — tam by GPU teoreticky mohl válčit,
      ale current scaling trend (5k 0.4% improvement, 10k expected loss)
      naznačuje že CPU stack drží napříč N.
    - Renderer mirror Sprint 60-63 GPU pipeline (Bevy ECS).

## Sprint 64 — z=50 expansion + pop=2500 (proportional volumetric scale-up)

- **Cíl:** rozšířit z-rozsah simulace 2.5× (z=20 → z=50) + zvětšit
  populaci 2.5× (1000 → 2500) tak, aby cell density per volume zůstala
  konstantní. Volumetric food scaling (Sprint 53 `z_factor`) auto-rozšíří
  food count: ~8000 → ~20000 (2.5×). Pre-Sprint-64 z=20 byl conservative
  bump kvůli initial-smoke extinkci u z=50; po Sprintu 53 food scaling +
  Sprint 57+58 paralelní path je z=50 stable.

  **Plán implementace:**

  *Body 1 — `lib::MAX_POPULATION`: 1000 → 2500.*
  - Mass change v const. Headless + renderer používají hodnotu přes import.
  - 2.5× faktor matchuje z-volume scaling (z_extent 40 → 100 = 2.5×).
  - CPU paralelní cesta (Sprint 57+58) drží: Sprint 63 5k cells = 870 ticks/s,
    extrapolace 2500 cells ≈ 900-950 ticks/s při z=50.

  *Body 2 — `WORLD_HALF[2]` / `SIMULATION_HALF[2]`: 20.0 → 50.0.*
  - Headless + main shared change. Field GPU sources capacity auto-adjusts
    přes `food_target(peak_density) × 2` (Sprint 59 init).
  - GPU SpatialHash bucket grid (Sprint 55) má fixed `GRID_NZ=4`,
    `HALF_NZ=2`, `cell_size=64` → krytí ±128 v z. z=50 fits comfortably.
  - FieldGpu z-resolution: `SMELL_GRID_RES_Z=16`. Při world_half_z=50:
    cell_size_z = 100/16 = 6.25 (vs pre-Sprint-64 2.5 = 2.5× coarser).
    Diffusion stability: `< 1/6 ≈ 0.167`; `SMELL_DIFFUSION = 0.15` ✓.
  - Carrion floor sink: drops k z = -50 (post-Sprint-38 gravity semantika).

  *Body 3 — Food count auto-scaling check:*
  - `food_target(factor)` = `(area / 2600) × factor × z_factor` kde
    `z_factor = z_extent / 4`. Při z=50: `z_factor = 25` → ~20000 food
    pro `factor=1.0`. Pre-Sprint-64 z=20: `z_factor = 10` → ~8000 food.
    Žádný code change — auto-scales přes WORLD_HALF[2] bump.

  *Body 4 — Cell density per volume zachována:*
  - Pre-Sprint-64 (z=20, pop=1000): `1000 / (1920 × 1080 × 40) = 1.2e-5
    cells/unit³`.
  - Post-Sprint-64 (z=50, pop=2500): `2500 / (1920 × 1080 × 100) = 1.2e-5
    cells/unit³` ✓.
  - Plus food density: `8000 / (1920 × 1080 × 40) = 9.6e-5 food/unit³`
    → `20000 / (1920 × 1080 × 100) = 9.6e-5 food/unit³` ✓.
  - Encounter rates (predation, mating, collisions) by měly držet stejné
    statistiky per cell.

- **Konstanty:**
  - `lib::MAX_POPULATION`: `1000` → `2500`.
  - `headless::WORLD_HALF[2]`: `20.0` → `50.0`.
  - `main::SIMULATION_HALF[2]`: `20.0` → `50.0`.

- **Výstup:**
  - `src/lib.rs`: `MAX_POPULATION = 2500` + comment.
  - `src/bin/headless.rs`: `WORLD_HALF[2] = 50.0` + Sprint 64 expansion comment.
  - `src/main.rs`: `SIMULATION_HALF[2] = 50.0` + comment.
  - **Smoke seed=0, 60 gen, default world (z=50, pop=2500), CPU:**
    - **Wall-clock 44.4 s = 811 ticks/s.** (Sprint 58 1k = 977, –17 %;
      Sprint 63 5k = 870 = roughly same workload as 2.5× cells.)
    - Pop saturuje na ~1946 cells (pod 2500 cap — cells sparser per volume,
      mating density mírně klesá vůči pre-Sprint-64 saturated 1000/1000).
      Pop final gen 60 = 559 (oscilace zdravá, žádná extinkce).
    - Initial food = 19938 (z_factor scaling ✓).
  - **Renderer launch:** Bevy app starts s z=50 world OK, žádný panic.
    8 s SIGTERM exit clean.

- **Poznámky:**
  - **Pop saturated < cap:** ~1946 / 2500 = 78 % saturation. Sprint 53
    pre-z-bump měl pop saturated ~1000 / 1000 = 100 %. Pravděpodobné
    důvody:
    1. **Mating density:** cells sparser per volume → encounter rate per
       tick klesá → mating success klesá → birth rate < cap.
    2. **Food spread:** food density per volume konstant, ale food clusters
       (Sprint 21+ richness biome stratification) se rozprostřou v 3D —
       cells musí navigovat 3D pro food. Některé brainy nestihnou
       evolve 3D navigation (Sprint 53 vertical sensing už ano, ale
       2.5× větší prostor zase znamená delší search). Selection může
       trvat víc generací.

    Toto je expected behavior — Sprint 64 cílí "scale-up" ne "saturate".
    Při delším runu (200+ gen) by selekce favorizovala 3D navigators a
    pop by mohlo dál růst.
  - **Per-fáze breakdown z=50 / pop=2500 vs Sprint 58 (z=20 / pop=1000):**

    | Fáze | Sprint 58 (z=20, n=1000) | Sprint 64 (z=50, n=1946) | Scale |
    |------|--------------------------|---------------------------|-------|
    | update_smell | 211 µs | 502 µs | 2.4× (foods 8k → 20k) |
    | brain_act | 333 µs | 665 µs | 2× (cells) |
    | eat_food | 278 µs | 797 µs | 2.9× (foods + cells) |
    | predate | 105 µs | 117 µs | 1.1× (sparse) |
    | resolve_collisions | 66 µs | 106 µs | 1.6× |
    | apply_food_gravity | 13 µs | 35 µs | 2.7× |
    | reproduce | 24 µs | 49 µs | 2× |

    Predate + collisions škálují **sub-linearly** — sparse cell density
    snižuje encounter rate. Eat_food dominantní (foods × cells).

  - **Co Sprint 64 NEŘEŠÍ (Sprint 65+):**
    - Renderer overlay validace v 3D z=50 — uživatel manually verify že
      kamera + ortho projection ukazují plné z-rozsah.
    - INITIAL_CELLS bump (200 → 500) pokud chce rychlejší pop ramp.
    - GPU SpatialHash z-bucket capacity bump (`GRID_NZ` 4 → 8) pro extreme
      z (z=100+). Aktuálně OK pro z=50.
    - Thermal stratification (temperature field z-gradient — Sprint 60+ deferred).
    - Light field z-attenuation (photic vs aphotic zones).
    - Bigger horizontal world (xy bigger than ±960 / ±540) — vyžaduje
      `GpuSpatialHash::GRID_NX/NY` bump nebo dynamic sizing.

## Sprinty 65–72 — open-ended

- **Sprint 65+:** Thermal stratification (temperature field z-gradient).
- **Sprint 65+:** Light field z-attenuation (photic vs aphotic zones).
- **Sprint 65+:** GPU collision/predate wire (Sprint 64 deferred).
- **Sprint 65+:** Renderer mirror Sprint 60-63 GPU pipeline.
- **Sprint 65+:** Bigger horizontal world expansion + dynamic GPU
  SpatialHash sizing.
- **Sprint 65+:** INITIAL_CELLS bump pro rychlejší pop ramp.
- **Sprint 65+:** Long-run smoke (200+ gen) ověření že 3D navigator brainy
  evolve a populace satureuje ke cap pri z=50.
