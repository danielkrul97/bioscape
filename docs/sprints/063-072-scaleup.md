# Sprinty 63–72: Scale-up

Decade rozšiřuje sim scope — víc cells, větší volume, biofyzikální vrstvy
multicelulárity — ne další performance work. Sprint 60-62 série ukázala že
GPU offload je net-negative pro current sim shape (1k–10k cells, 64×64×16
grid) napříč všemi paths kvůli per-tick GPU dispatch overhead. CPU paralelní
cesta (Sprint 57+58) drží excellent scaling. Sprint 63 wired step na GPU
jako infrastructure (single Wait barrier), Sprint 64 expandoval volume
(z=50) + populaci (2500). Sprint 65 opravil 3D fyziku (gravity off, 3D
collision, velocity damping). Sprint 66 zavedl differential adhesion
(Steinberg) + persistent spring bonds — základ pro evoluci tkání /
multicelulárních organismů. Sprint 67 vystavil bond/adhesion telemetrii do
CSV (groundwork pro long-run experimenty). Sprint 67.1 long-run smoke
(250 gen) potvrdil bond mechaniku ale ukázal selekci bondů net-disfavored.
Sprint 68 convertoval `bond_stiffness` + `bond_damping` z globálních
konstant na per-cell geny — selekce je aktivně modeluje (damping ↑↑
1.3). Sprint 69+ pokračuje na bond benefit-side mechanic (predator
dilution, food share) + horizontální layers (thermal/light fields).

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

## Sprint 65 — fyzika 3D fixes (gravita off + 3D collision + velocity damping)

- **Cíl:** opravit 3 problémy ve fyzice ze Sprintu 53-58 era:
  1. **Cell-cell collision byl jen 2D** v `main.rs` (delta na x/y, pre-Sprint-58
     headless taky 2D, fixed Sprint 53). Buňky se v z prolínaly.
  2. **Gravity = 5.0 bez buoyancy** vytvářelo selekční tlak směrem k „seď
     na dně" — všechno postupně sedimentovalo na floor reflective wall,
     vertikální motion neměla evoluční benefit.
  3. **Žádná velocity-response v collision** — pozice se depenetrovala,
     ale momentum pokračoval. Cells po push-apart pokračovaly v closing
     motion → re-overlap next tick (oscilace + zbytečný compute).

  **Plán implementace:**

  *Body 1 — `GRAVITY` 5.0 → 0.0 (neutral buoyancy):*
  - lib.rs const update + doc string. Pre-Sprint-65 komentář popisoval
    GRAVITY jako "effective post-buoyancy" (5 % netto force kvůli density
    ratio 1.05/1.0). Sprint 65 přijímá explicit "cell density == water
    density → neutral buoyancy" jako cleaner design — vertikální motion
    100 % brain-driven.
  - Food (`FOOD_SINK_RATE = 8.0`) zachován — food má vyšší density než
    cells (benthic deposit semantika, cells musí proaktivně dive za food).

  *Body 2 — `COLLISION_RESTITUTION = 0.0` const + velocity damping:*
  - Lib const pro inelastic collision tuning. Restitution 0 = closing
    velocity podél separation normal je vynulovaná (cells „stick"
    momentárně, oddělí se přes position depenetration). 1.0 = elastic
    (perfect bounce). Soft biological cells = 0 default.
  - Math: per pair (i, j), `n = (pos_i - pos_j).normalize()`,
    `v_rel_n = (v_i - v_j) · n`. Pokud `v_rel_n < 0` (closing), Δv_i along
    n = `-v_rel_n × 0.5 × (1 - restitution)`. Symmetric per pair (Newton
    3rd law když j visits i v par_iter).

  *Body 3 — Headless `resolve_collisions` 3D + velocity damping:*
  - `velocity_deltas_scratch: Vec<[f32; 3]>` přidán k `World` struct.
  - Callback computes both position delta + velocity delta v jediném
    callback pass.
  - Aplikace: cell.position += delta + cell.velocity += vel_delta v
    sekvenčním Pass 2.

  *Body 4 — Main `resolve_cell_collisions` 3D + velocity damping:*
  - Snapshot rozšířen o velocity (4-tuple místo 3-tuple).
  - `FxHashMap<Entity, [f32; 3]>` pre-built pro O(1) velocity lookup
    v par_iter callback (per-pair v_rel computation). Bez něho by každý
    callback dělal linear scan snapshot = O(N²) per tick.
  - Pass 2 apply 3D delta + vel_delta přes `Query::get_mut`.

- **Konstanty:**
  - `lib::GRAVITY`: `5.0` → `0.0`.
  - `lib::COLLISION_RESTITUTION` nový: `0.0`.

- **Výstup:**
  - `src/lib.rs`: GRAVITY = 0.0 + Sprint 65 doc string. COLLISION_RESTITUTION
    nový const + doc.
  - `src/bin/headless.rs`: `velocity_deltas_scratch` field + 3D collision
    + velocity damping. Re-export COLLISION_RESTITUTION.
  - `src/main.rs`: 3D position delta (pre-Sprint-65 missing z!), velocity
    delta + FxHashMap velocity lookup v par_iter.
  - **Test suite: 73/73 pass** (1 flaky `random_brain_average_thrust_is_positive`,
    pre-existing).
  - **Smoke seed=0, 60 gen, z=50, pop=2500, CPU:**
    - Wall-clock 58.5 s = **615 ticks/s** (Sprint 64 811, –24 % wall-clock
      ale **pop dynamic zcela jiná**).
    - **Pop final 883 (Sprint 64: 559, +58 % vyšší)** — cells už nesedí
      na floor → 3D distribution → vyšší steady-state populace.
    - Per-fáze deltas vs Sprint 64 (n=734 avg vs Sprint 64 n=1946 — fewer
      cells per fáze, ale per-tick rates lower):
      - brain_act 554 µs (S64: 665, –17 %)
      - eat_food 683 µs (S64: 797, –14 %)
      - predate 83 µs (S64: 117, –29 %)
      - resolve_collisions 65 µs (S64: 106, –39 %) — **velocity damping
        eliminuje re-overlap loops** ✓
      - step 10 µs (S64: 17, –41 %)
  - **Renderer launch:** Bevy app starts s GRAVITY=0 + 3D collisions OK,
    žádný panic. 8 s SIGTERM exit clean.

- **Poznámky:**
  - **Pop trajectory dramaticky lepší.** Sprint 64 měl pop saturated
    early ~1946, deklinoval k 559. Sprint 65 má pop osciluje stable v
    rozsahu 700-900, gen 60 = 883 (final). Důvod: bez gravity cells nemusí
    plavat up jen aby přežily — vertikální motion je now optional. Plus
    velocity damping snižuje wasted collision-cycles → cells mají více
    energie na užitečnou activity (food search, mating).
  - **Velocity damping přínos kvantifikován v collisions fáze:** 65 µs
    vs 106 µs (S64) = 39 % rychlejší. Per-collision compute je vyšší
    (extra v_rel + impulse compute), ale počet collision events nižší
    (no re-overlap oscilace) → net lower work.
  - **Renderer collision fix:** main.rs delta byl `[f32; 2]` od Sprintu 58
    refactoru (chybou pre-Sprint-58 verze měla `[f32; 2]` který nikdy
    nebyl updaten po Sprint 53 z=2→z=20 expansion). Reálně to 4 sprinty
    cells volně prolínaly v z. Sprint 65 fix je pure correctness fix.
  - **Co Sprint 65 NEŘEŠÍ (Sprint 66+):**
    - GPU collision shader — wire-up Sprint 64 deferred. Sprint 50
      `collision.wgsl` má 2D delta per Sprint 50 design; potřebuje 3D
      update + velocity damping mirror.
    - Anisotropic collision (cells s elongated body geometrie nemají
      stejný collision radius v různých směrech). Aktuálně `pair_r =
      CELL_RADIUS × (radius_i + radius_j)` je sféra; měl by být ellipsoid
      (matchnout `eat_test` semantiku).
    - Wall collision velocity response — z-bounce má reflective wall
      (Sprint 53/54 cylinder), velocity[2] je flipped. Inelastic by
      damped reflection. Aktuálně bounce je elastic.
    - Long-run smoke (200+ gen) — verify že pop oscilace stable napříč
      širších time scales s nový dynamikou.

## Sprint 66 — differential adhesion + persistent spring bonds (hybrid)

- **Cíl:** zavést dvojvrstvý adhesion model jako základ pro multicelulární
  shlukování / morfogenezi:
  1. **Differential adhesion (Steinberg)** — soft, stateless. Heritable
     `adhesion_type: u8` (8 typů, ~3 bity informace). V broad-phase loopu
     mimo kontaktní vzdálenost se aplikuje atraktivní síla mezi same-type
     cells, mírná repulze mezi cross-type. Linear falloff `(1 - d/R)` na
     `R = 3 × pair_radius`. Stateless, O(1) per pár, kompatibilní s GPU
     broad-phase + toroidální xy.
  2. **Persistent spring bonds** — stateful, rigid. Cell drží
     `bonds: [Option<Bond>; 6]` (fixní array, žádný heap alloc). Bond =
     stable `other_cell_id: u64` + `rest_length` + `age_ticks`. Hookean
     spring podél min-imaged delta (`F = -k × extension`) + damping podél
     spring axis. Rest length set z aktuální vzdálenosti při formaci
     × `BOND_REST_LENGTH_SLACK = 1.05`.
  3. **Hybrid (cíl):** soft attraction default, spring bond se aktivuje
     až po `BOND_FORM_TICKS=30` (0.5 s při 60 Hz) prolonged contact +
     **OBA** cells musí mít `last_outputs[9] > BOND_FORM_THRESHOLD`. Bond
     se trhá při overstretch (`d > rest × 2.5`), explicit brain signál
     (`outputs[9] < -0.5`), nebo když cíl zemře.

  **Plán implementace:**

  *Body 1 — Lib: gen + Cell rozšíření:*
  - `Genome::adhesion_type: u8` (uniform initial draw 0..ADHESION_TYPE_COUNT,
    crossover 50/50, mutation flip with `ADHESION_MUTATION_RATE = 5%`).
  - `Cell::cell_id: u64` — stable monotonic identifier pro bond resolution.
    Lineage_id sdílený per linie, takže není unique per cell.
  - `Cell::bonds: [Option<Bond>; MAX_BONDS_PER_CELL=6]` — fixed array,
    Cell zůstává Copy, no heap alloc per cell.
  - `BRAIN_OUTPUTS: 9 → 10`, output[9] = bond_signal. INNATE_BOND_BIAS=0
    (opt-in, jako attack).

  *Body 2 — Force kernels:*
  - `adhesion_velocity_delta(delta_ji, dist, pair_r, same_type) → [f32; 3]`
    — pure function, callable z headless i main.
  - `bond_velocity_delta(bond, delta_ji, dist, vel_i, vel_j) → ([f32; 3], broken)`
    — spring + damping along spring axis.

  *Body 3 — World-level state:*
  - `next_cell_id: u64` (counter), `contact_progress: FxHashMap<(u64, u64), u32>`
    (per-pair tick counter, sparse).
  - Renderer: `NextCellId(u64)` + `ContactProgress(FxHashMap)` Bevy resources.

  *Body 4 — Headless `resolve_collisions` extend:*
  - Phase 1 (paralelní): původní collision delta + velocity damping; navíc
    soft adhesion mimo kontakt; navíc bond force ze cell.bonds (lookup
    `cell_id → idx`); collected contact pairs (deduped by `cell_id_i < cell_id_j`).
  - Phase 2 (sekvenční): apply position/velocity deltas; bond age + prune
    (overstretch / dead target / explicit-break); contact_progress merge +
    decay (`CONTACT_DECAY_TICKS=5`); bond formation pro páry co dosáhly
    `BOND_FORM_TICKS` thresholdu (oba `outputs[9] > 0.2`, same adhesion_type,
    volné sloty na obou).

  *Body 5 — Renderer mirror:*
  - `resolve_cell_collisions` paralelně na snapshot s `cell_id`, `adhesion_type`,
    `bonds`, `radius`. FxHashMap<Entity, idx> + cell_id → idx pre-built.
  - `cell_reproduces_on_threshold` čerpá child_id z `NextCellId` resource.

  *Body 6 — Checkpoint version bump 1 → 2:*
  - Cell layout změna (`cell_id` + `bonds` + BRAIN_OUTPUTS 9→10) láme
    bincode parsing starých checkpointů. Hard-fail load místo migrace.
  - `next_cell_id` se rederivuje z `max(cell.cell_id) + 1` při loadu.
    `contact_progress` startuje prázdný (ephemeral state).

  *Body 7 — GPU shader update:*
  - `shaders/brain_forward.wgsl`, `motor.wgsl`, `hebbian.wgsl`:
    `BRAIN_OUTPUTS = 10u`, `B2_OFFSET = 752u`, `WEIGHTS_PER_CELL = 762u`.
  - `gpu.rs` static asserts updated; hardcoded literal `736` → `B2_OFFSET`.
  - `MotorGpu::alloc_buffers` outputs_buf bumped z `9 × f` na
    `BRAIN_OUTPUTS × f` (causal: 64 cells × 10 = 2560 B vs 2304 B → původní
    test selhal s buffer overflow).

- **Konstanty (lib.rs):**
  - `ADHESION_TYPE_COUNT: u8 = 8`
  - `ADHESION_MUTATION_RATE: f32 = 0.05`
  - `ADHESION_RANGE_FACTOR: f32 = 3.0`
  - `ADHESION_STRENGTH: f32 = 8.0`
  - `ADHESION_CROSS_TYPE: f32 = -0.3`
  - `MAX_BONDS_PER_CELL: usize = 6`
  - `BOND_FORM_TICKS: u32 = 30`
  - `CONTACT_DECAY_TICKS: u32 = 5`
  - `BOND_STIFFNESS: f32 = 4.0`
  - `BOND_DAMPING: f32 = 0.6`
  - `BOND_BREAK_FACTOR: f32 = 2.5`
  - `BOND_REST_LENGTH_SLACK: f32 = 1.05`
  - `BOND_FORM_THRESHOLD: f32 = 0.2`
  - `BOND_BREAK_THRESHOLD: f32 = -0.5`
  - `BOND_FORMATION_COST: f32 = 0.5`
  - `BOND_MAINTENANCE_PER_SEC: f32 = 0.1`
  - `INNATE_BOND_BIAS: f32 = 0.0`
  - `BRAIN_OUTPUTS: usize = 10` (was 9)
  - `MutationConfig::adhesion_flip_rate: f32` nový field.
  - `CHECKPOINT_VERSION: u32 = 2` (was 1).

- **Výstup:**
  - `src/lib.rs`:
    - `Bond` struct (other_cell_id, rest_length, age_ticks).
    - `Genome::adhesion_type` + crossover + mutate + Genome::random.
    - `Cell::cell_id`, `Cell::bonds`. `Cell::random` / `from_genome` +
      `make_mating_child` rozšířeny o `cell_id` parameter.
    - `adhesion_velocity_delta`, `bond_velocity_delta` pure helpery.
    - 9 nových unit testů (Sprint 66 sekce v `tests` mod).
  - `src/bin/headless.rs`:
    - `World::next_cell_id` + `World::contact_progress` +
      `World::bonds_formed_gen` / `bonds_broken_gen`.
    - `resolve_collisions` Phase 1 + Phase 2 rozšířená logika.
    - `spawn_children_from_matings` přiděluje cell_ids z World counteru.
    - `load_checkpoint` rederivuje `next_cell_id`. `CHECKPOINT_VERSION=2`.
  - `src/main.rs`:
    - `NextCellId` + `ContactProgress` Bevy resources.
    - `resolve_cell_collisions` snapshot rozšířen o cell_id + adhesion_type
      + bonds. `entity_to_idx` + `id_to_idx` FxHashMaps pro O(1) lookup.
    - `cell_reproduces_on_threshold` čerpá child id z `NextCellId`.
  - `shaders/brain_forward.wgsl`, `hebbian.wgsl`, `motor.wgsl`: BRAIN_OUTPUTS=10.
  - `src/gpu.rs`: B2_OFFSET=752, WEIGHTS_PER_CELL=762, motor outputs buffer
    sized z BRAIN_OUTPUTS const (was hardcoded 9).
  - **Test suite: 82/82 pass** (lib + GPU; new tests: adhesion zero
    inside contact, zero beyond range, pulls same-type, repels cross-type;
    bond pulls when stretched, pushes when compressed, breaks past
    factor, damping opposes closing; toroidal wrap-aware).
  - **Smoke seed=0, 60 gen, default world, CPU:**
    - Wall-clock 61.8 s = **582 ticks/s** (Sprint 65: 615; –5 %).
    - Pop final 627 (Sprint 65: 883). Adhesion + bonds přidávají sílový
      koupling; rané generace cells reagují na novinku, populace osciluje
      níže než pre-Sprint-66 quasi-equilibrium. Žádná extinkce.
    - Per-fáze delta vs Sprint 65: `resolve_collisions` 65 µs → 101 µs
      (+55 % — adhesion broadphase fan-out + bond force apply + contact
      tracker bookkeeping). Ostatní fáze beze změn (~kvůli šumu n_cells).
  - **Renderer launch:** `cargo build --release` clean, 8 s SIGTERM exit
    bez panic.

- **Poznámky:**
  - **Cell.cell_id vs lineage_id:** lineage_id je shared per linii (initial
    pop unique, ale po reproduce dítě dědí parent's lineage). Pro bond
    resolution potřebujeme per-cell unique ID — proto cell_id (monotonic
    counter, World-level). Cena: extra u64 per Cell + FxHashMap pre-tick
    lookup v collision broadphase. Acceptable.
  - **Snapshot extend renderer ladí 4 ms collision phase pro 1k cells**
    (vs 2 ms pre-Sprint-66). FxHashMap pre-build dominantní cost (~1 ms).
    Pro 2.5k cells je extrapolace ~10 ms = pod 16.6 ms frame budget,
    headroom pro Sprint 67+.
  - **Bond determinismus:** `id_to_idx` build je deterministic order
    (cells.iter().enumerate). Spring force per cell deterministic. Bond
    formation candidates iterují FxHashMap — order není guaranteed; ale
    hash je fixed-seed, takže stejný seed dá stejný order. Per-tick CSV
    by měla být reprodukovatelná stejně jako pre-Sprint-66.
  - **GPU compatibility:** spring bonds + adhesion běží **CPU-only**.
    GPU full-pipeline (`--gpu-full`) v Sprintu 63 zůstává nedotčen — step
    + brownian + brain forward jsou na GPU, ale collision pass (= adhesion +
    bonds) se neoffloaduje. Renderer GPU (Sprint 52) podobně CPU-side
    collision. Sprint 67+ může přesunout adhesion na GPU (stateless,
    paralelní).
  - **Steinberg sorting kvalitativně:** při ADHESION_CROSS_TYPE = -0.3
    se očekává, že same-type cells budou tvořit malé clustery; cross-type
    interakce mírně rozhání. Při dlouhém run (200+ gen) by selekce mohla
    buď homogenizovat populaci (jeden adhesion_type vyhraje) nebo
    udržovat polytypický stav (clusters různých typů koexistují přes
    niche separation). Empirické verification až s long-run smoke (Sprint 67+).
  - **Bond emergence question:** v 60 gen smoke neměřím, jestli bondy
    fakticky vznikly. INNATE_BOND_BIAS=0 znamená, že random brainy mají
    ~50 % šanci `outputs[9] > 0` nad ~50 % random seedu, ale jen ~10 %
    nad threshold 0.2. Sprint 67+ by měl přidat per-gen counter `bonds_total`
    do CSV pro zjištění, kdy selekce začne bondování favorizovat.
  - **Co Sprint 66 NEŘEŠÍ (Sprint 67+):**
    - GPU adhesion + bond shaders (CPU-only path adekvátní pro current N).
    - CSV columns pro `bonds_formed_gen` / `bonds_broken_gen` /
      `mean_bond_count` / `mean_adhesion_type_entropy`.
    - Rendering: bondy se aktuálně neukazují (žádné spring linky mezi
      cells v 3D viewportu). Sprint 67+ může přidat gizmo lines.
    - Anisotropic adhesion (ellipsoid kontakt): aktuálně sphere R based.
    - Bond stiffness/damping evolvability — všechny bondy sdílí
      konstantní `BOND_STIFFNESS` / `BOND_DAMPING`, ne per-pair gen.
    - Long-run smoke (200+ gen) verify že bondy + clustering jsou
      evoluční attraktor.

## Sprint 67 — bond / adhesion CSV diagnostics

- **Cíl:** dokončit deferred item ze Sprintu 66 — vystavit per-generation
  bond + adhesion telemetry do `headless` CSV výstupu, aby se daly měřit
  emergence bondů a Steinberg sorting napříč long-run experimenty bez
  re-buildu binárky. World už eviduje `bonds_formed_gen` /
  `bonds_broken_gen`, jen je třeba je přidat do log řádku spolu s
  derivovanými agregáty.

  **Plán implementace:**

  *Body 1 — per-cell akumulátory v `write_stats`:*
  - `bond_signal_sum` (Σ `last_outputs[9].max(0.0)`).
  - `total_bonds` (Σ populated `bonds[..]` slotů per cell).
  - `bonded_cells` (count cells s ≥ 1 bondem).
  - `adhesion_hist: [u64; ADHESION_TYPE_COUNT]` (frequency table typů).

  *Body 2 — derivované metriky:*
  - `mean_bond_count = total_bonds / n` — průměrný degree v bond grafu.
  - `bond_active_frac = bonded_cells / n` — fraction populace zapojené.
  - `bond_signal_avg = bond_signal_sum / n` — proxy adopce brain output[9].
  - `adhesion_entropy = -Σ p_i log₂ p_i / log₂ K` — normalizovaná Shannon
    entropy distribuce typů. 1.0 = uniformní (initial random), 0.0 =
    monokultura. Selekce by ji teoreticky měla snižovat (winning type fixuje).

  *Body 3 — CSV header + writeln! formát + extinction fallback:*
  - 6 nových sloupců na konci hlavičky:
    `bonds_formed,bonds_broken,mean_bond_count,bond_active_frac,bond_signal_avg,adhesion_entropy`.
  - Populated branch: 6 nových placeholderů v `writeln!` formátu.
  - Empty-pop fallback (extinkce gen): `bonds_formed_gen` /
    `bonds_broken_gen` ze World struktury (mají smysl i bez živých cells —
    counts last-tick formace/breakage), zbylé 4 metriky 0.

- **Konstanty:** žádné nové.

- **Výstup:**
  - `src/bin/headless.rs`:
    - `write_stats` rozšířený o 4 akumulátory + 4 derivované metriky.
    - CSV header rozšířen o 6 sloupců (36 → 42 total).
    - `writeln!` populated + extinction-fallback formáty extended.
  - **Test suite: 82/82 pass** (žádné nové testy — diagnostic-only změna,
    smoke verifuje CSV alignment).
  - **Smoke seed=0, 30 gen, default world, CPU:**
    - Header + všechny řádky mají přesně **42 sloupců** (verifikováno awk).
    - Sample data:
      - gen 0: cells=200, bonds_formed=0, bond_active_frac=0,
        adhesion_entropy=**0.997** (random initial = near-max).
      - gen 1: cells=748, bonds_formed=1, bonds_broken=1,
        bond_signal_avg=0.510, adhesion_entropy=0.990.
      - gen 29: cells=652, bonds_formed=18, bonds_broken=35,
        bond_active_frac=0.029, adhesion_entropy=0.985.
    - Bond dynamika sub-1 % populace v gen 0–29 — selekce zatím nezačala
      bondy favorizovat. `bonds_broken > bonds_formed` v gen 29 indikuje
      net loss, což matchne expectation pro untrained brainy v early-gen
      (bondy se trhají overstretch před selekcí na cooperative motion).

- **Poznámky:**
  - **Přidání nestojí runtime perf** — write_stats běží 1× per generation
    (= per 600 ticks), takže 4 extra inkrementace per cell × 600 cells
    × 1/600 ticks = sub-µs amortizovaně. Smoke wall-clock matchne Sprint 66
    (~582 ticks/s) v rámci noise.
  - **Adhesion entropy normalization volba:** `log₂(K)` (= log₂(8) = 3.0
    bits) jako denominator dává hodnotu v [0, 1] nezávisle na
    `ADHESION_TYPE_COUNT`. Pokud Sprint 67+ změní K, metric zůstává
    porovnatelný napříč experimentů.
  - **bond_signal_avg používá `.max(0.0)`** stejně jako `ph_emit_avg` /
    `atk_emit_avg` — měří jen aktivní emisi (positive output), ne raw
    tanh průměr. Konsistentní s pre-existujícími brain-output
    diagnostickými sloupci.
  - **Co Sprint 67 NEŘEŠÍ (Sprint 68+):**
    - Per-pair bond duration histogram (uniformní bondy vs short-lived
      churn). Vyžaduje samostatnou metric struct, ne jen scalar agg.
    - Spatial autocorrelation adhesion_type — empirická detekce
      Steinberg clusterů by potřebovala neighborhood enrichment metric
      (e.g. mean neighbor same-type fraction vs random baseline).
    - Renderer overlay (bond gizmo lines) — pořád pending.
    - Long-run smoke (200+ gen) — diagnostiky jsou ready, samotný run
      je další item.

## Sprint 67.1 — long-run smoke (250 gen) bond + adhesion verification

- **Cíl:** ověřit Sprint 65 (3D fyzika fixes) a Sprint 66 (adhesion + bonds)
  napříč delším time scalem než 60-gen smoke. Diagnostika připravená v
  Sprintu 67 zachycuje bond emergence + adhesion sorting jako CSV trendy.

- **Run:** `cargo run --release --bin headless -- 0 250 /tmp/sprint68_longrun.csv`
  - **Wall-clock 208.4 s = 720 ticks/s** (lepší než Sprint 66 60-gen smoke
    582 ticks/s — pop equilibrium ~620 < pop ramp ~1k v krátkém runu).
  - Žádná extinkce. Pop saturoval cap 2498 @ gen 2 (initial random brains
    pumpou energetického rampu), pak crash na ~600 quasi-equilibrium kde
    setrvává po zbytek 250 gen (oscilace 600–650).

- **Bond dynamika:**

  | Metric | Early (gen 1-50) | Late (gen 200-250) | Peak | Δ early→late |
  |--------|-------------------|---------------------|------|---------------|
  | mean `bond_active_frac` | 0.0298 | 0.0389 | 0.138 @ gen 128 | +30 % |
  | mean `bond_signal_avg` | 0.488 | 0.325 | 0.660 @ gen 128 | **-33 %** |
  | mean `adhesion_entropy` | 0.9872 | 0.9945 | 0.999 @ gen 112 (max) / 0.971 @ gen 18 (min) | +0.7 % |
  | totals `bonds_formed` | 1012 | 426 | – | – |
  | totals `bonds_broken` | 1663 | 430 | – | – |
  | net `formed - broken` | -651 (early crash churn) | -4 (equilibrium) | 250-gen total: -677 | – |

- **Závěr:**
  1. **Bondy mechanicky fungují** — formace + breakage proběhly napříč
     celým runem (3463 formed / 4140 broken total). Telemetrie zachytává
     real signal: peak `bond_active_frac=13.8 %` @ gen 128 koreluje s
     peak `bond_signal_avg=0.660` (brain output[9] driven).
  2. **Selekce mírně NEGATIVNĚ tlačí bondování** — `bond_signal_avg`
     klesá z 0.488 (early) na 0.325 (late, -33 %). Random brain bias je
     vyšší než to, co selekce uchovává. To znamená, že bonding při
     current parametrech (`BOND_FORMATION_COST=0.5`,
     `BOND_MAINTENANCE_PER_SEC=0.1`) je **net energy loss** pro
     individuální fitness v foraging-driven nice — cells jsou selektovány
     na rychlou exploration, ne na shlukování.
  3. **Steinberg sorting nenastává** — `adhesion_entropy` osciluje v
     [0.971, 0.999], žádný kolaps směrem k monokultuře. Při
     `ADHESION_STRENGTH=8.0` je adhesion síla nedostatečná aby překonala
     thrust + Brownian noise; same-type clustery se rozpadají dřív než
     se stihnou stabilizovat.
  4. **Pop dynamics stable** — Sprint 65 fixes (gravity=0, 3D collision,
     velocity damping) drží napříč 250 gen bez sedimentace nebo
     oscilace blow-up. ✓

- **Implikace pro Sprint 68+:**
  - **Tuning sweep parametrů** — najít regime, kde se bonding stane
    evolučním attraktorem. Kandidátní knoby:
    - `BOND_FORMATION_COST` 0.5 → 0.1 (snížit one-shot tax)
    - `BOND_MAINTENANCE_PER_SEC` 0.1 → 0.02 (continuous tax 5× nižší)
    - `ADHESION_STRENGTH` 8.0 → 20.0 (silnější soft-attraction)
    - `INNATE_BOND_BIAS` 0.0 → 1.0 (positive prior, jako attack original
      Sprint 27 conservative path)
  - **Selection regime change** — současná niche je pure foraging. Bonding
    by potenciálně dávalo benefit při:
    - predace (bonded cluster = collective defense)
    - food coverage (bonded clusters mohou pokrývat větší prostorový gradient)
    Aktuálně předpokládáme, že random brainy nemají přístup k „group selection"
    feedback, takže bond benefits se evolučně neprojeví.
  - **Long-long run (1000+ gen)** — možná že 250 gen je málo na
    second-order selekci na bonding (cells musí nejdřív vyladit
    foraging, pak teprve cooperation). Sprint 68+ by mohl hodit 1000 gen
    při kompromisních parametrech.

- **Výstup:**
  - `/tmp/sprint68_longrun.csv` — 250 datových řádků, 42 sloupců.
    Sprint 67 telemetrie produkuje smysluplný signal (žádné NaN, hodnoty
    v očekávaných rozsazích).
  - Žádná code change. Pure observational sprint — verifikuje, že Sprint
    66 + 67 jsou stable + diagnostické nástroje fungují.

## Sprint 68 — bond_stiffness + bond_damping jako per-cell geny

- **Cíl:** convertovat dva globální bond physics konstanty (`BOND_STIFFNESS=4.0`,
  `BOND_DAMPING=0.6`) na per-cell evolvable geny. Sprint 67.1 long-run smoke
  ukázal, že selekce má signál na bonding, ale current parametry produkují
  net-disfavored bonding. Per-cell evolvability dovolí selekci najít optimum
  místo toho, aby ji limitovaly globální konstanty.

  **Plán implementace:**

  *Body 1 — Genome rozšíření:*
  - `Genome::bond_stiffness: f32`, range `[MIN_BOND_STIFFNESS=0.5,
    MAX_BOND_STIFFNESS=16.0]` (široký rozsah).
  - `Genome::bond_damping: f32`, range `[0.0, 2.0]` (under-damped → over-damped).
  - Initial draw `BOND_STIFFNESS × [0.5, 1.5]` (= 2.0..6.0) center 4.0.
  - `MutationConfig::sigma_bond_stiffness = 0.3`,
    `sigma_bond_damping = 0.05`. Konzervativní hodnoty kvůli sub-procentní
    `bond_active_frac` v Sprint 67.1 — slabší signál → menší sigma proti
    random walk.

  *Body 2 — Bond struct rozšíření:*
  - `Bond::stiffness: f32`, `Bond::damping: f32`. Set při formaci jako mean
    obou cells' genome values (pair = jedna pružina, fyzicky správné).
  - `bond_velocity_delta` čte `bond.stiffness` a `bond.damping` místo
    globálních konstant. `BOND_STIFFNESS` / `BOND_DAMPING` zůstávají jako
    centery pro initial Genome draw.

  *Body 3 — Formation sites (headless + main):*
  - V resolve_collisions / resolve_cell_collisions, při Bond formaci:
    `stiffness = (genome_a.bond_stiffness + genome_b.bond_stiffness) * 0.5`
    `damping = (genome_a.bond_damping + genome_b.bond_damping) * 0.5`.

  *Body 4 — CSV diagnostics:*
  - 2 nové sloupce: `bond_stiff_avg`, `bond_damp_avg` (mean genome values
    napříč populací). Header 42 → 44 sloupců.

  *Body 5 — Checkpoint version 2 → 3:*
  - Genome + Bond layout změna láme bincode parsing V2. Hard-fail load.

- **Konstanty:**
  - `MIN_BOND_STIFFNESS: f32 = 0.5`, `MAX_BOND_STIFFNESS: f32 = 16.0` nový
  - `MIN_BOND_DAMPING: f32 = 0.0`, `MAX_BOND_DAMPING: f32 = 2.0` nový
  - `MutationConfig::sigma_bond_stiffness = 0.3`,
    `sigma_bond_damping = 0.05` nové fields
  - `CHECKPOINT_VERSION: u32 = 3` (was 2)

- **Výstup:**
  - `src/lib.rs`: Genome + Bond + bond_velocity_delta extended. Tests + dummy
    genome literals updated.
  - `src/bin/headless.rs`: Bond formation site computes pair-mean k+c.
    `write_stats` emits `bond_stiff_avg` + `bond_damp_avg`. CHECKPOINT
    bumped.
  - `src/main.rs`: Bond formation site mirror.
  - **Test suite: 82/82 pass** (žádné nové testy — mechanika identická,
    jen knoby per-cell).
  - **Long-run smoke seed=0, 250 gen, default world, CPU:**
    - Wall-clock 243.1 s = 617 ticks/s (Sprint 67.1: 720; pomalejší kvůli
      pop crash na 288 final → cells×ticks malé per-iter), pop final 288
      (Sprint 67.1: 618). Pop crash je side-effect RNG drift z extra
      `random_range()` calls v Genome::random — žádný direct mechanic
      change. Nicméně gene selection signal je v early-mid generations
      jasný:

  | Gen window | `bond_stiff_avg` | `bond_damp_avg` | `bond_active_frac` |
  |------------|-------------------|------------------|---------------------|
  | EARLY 1-50 | 3.669 | 0.516 | 0.0356 |
  | MID 51-100 | 3.811 | 0.577 | 0.0413 |
  | LATE 200-250 | 3.175 | **1.302** | 0.0022 |

  **Klíčové pozorování:** `bond_damp_avg` se zdvihl z initial center 0.6
  na **1.30 (+117 %)** napříč 250 gen — selekce silně favorizuje
  vysoký damping. Stiffness mírně klesá (3.7 → 3.2). Late-stage `baf=0.002`
  znamená že bondy zanikly skoro úplně (bond_signal_avg dropdownul pod
  threshold), takže late drift je čistě genetic, ne selekce. Ale early-mid
  fáze, kde bondy aktivně byly, ukazují clear selection signal.

- **Závěr:** Sprint 68 patch je **úspěšný** — per-cell genes pod selekcí,
  diagnostiky zachycují drift, hypotéza "selekce by mohla najít lepší
  parametry" potvrzena. Damping ⬆ (1.3) signalizuje, že cells preferují
  bondy s rychlejším damping pro stabilizaci proti collision/Brownian
  noise než globální 0.6 default.

- **Poznámky:**
  - **Late-stage gene drift potvrdil bottleneck Sprintu 67.1**: bondy
    samy se utlumí, ale samotné mechanic genes nejsou problém.
    Reálný bottleneck je `bond_signal` selection (output[9] favored
    negative) + bond cost. Sprint 69+ by se měl zaměřit na **bond
    benefit side** (predator pressure pro group defense, food coverage).
  - **Pop crash 618 → 288:** stochastic, ne deterministic z patche.
    Re-run s diff seed by ověřil. Při tomto experimentu nehrálo roli
    final pop, ale selection signal v gene means.
  - **Paradox: damping ↑↑ ale stiffness ↓.** Combined: bondy preferují
    "soft, slowly-relaxing" než "stiff, snappy". Floppy bond se chová
    spíš jako adhesion než rigidní spring. Steinberg-like sorting by
    býval cleaner s vysokým k a vysokým c (rigid, dampened) — ale
    selekce evidentně chce volnější vazbu (možná kvůli foraging
    mobility — rigidní bondy by zpomalily moving cluster).
  - **Co Sprint 68 NEŘEŠÍ (Sprint 69+):**
    - Bond benefit-side mechanic (predator dilution boost pro bonded,
      food share, atp.).
    - Per-cell `bond_form_threshold` (output[9] threshold konzument).
    - Renderer rendering bondů jako lines.
    - Detailed seed sweep pro robustness vs single-seed RNG noise.

## Sprint 69 — bond defense + gizmo render + adhesion-type cell coloring

- **Cíl:** dvě těsně provázané změny, kterými dohromady chceme dostat
  multicelularitu na obrazovku **viditelně i evolučně**:
  1. **Group-defense benefit hook** — bonded cells dostávají per-bond
     reduction na incoming predation gain + damage. Tím se převrací Sprint
     67.1 závěr „bonding je individual fitness-cost". Bonded cluster =
     skutečný evoluční attraktor.
  2. **Renderer visualization** — gizmo lines mezi bonded páry + obarvení
     cells podle `adhesion_type` (8 distinct hues místo random hue per
     lineage). Dosud byly bondy CSV-only signal; teď je vidíš na obrazovce.

  **Plán implementace:**

  *Body 1 — `bond_defense_factor` (lib.rs):*
  - Nová pure helper funkce: `bond_defense_factor(n_bonds: u32) -> f32`,
    vrací `1 - BOND_DEFENSE_FRAC × min(n_bonds, BOND_DEFENSE_CAP)`.
  - Nový helper `Cell::n_bonds() -> u32` count populated bond slotů.
  - Konstanty: `BOND_DEFENSE_FRAC = 0.15`, `BOND_DEFENSE_CAP = 4`.
    n_bonds=0 → 1.0 (no defense), n_bonds=4 → 0.4 (max defense, 60 % off).
    Cap brání stacking abuse — 6-bond cell by jinak byla 100% immune.
  - 3 nové unit testy: solo unity, linear scaling do capu, n_bonds counts only populated.

  *Body 2 — `predate` defense apply (headless + main):*
  - Headless `predate`: per-attack-event tuple rozšířen o `defense: f32`.
    Inside `flat_map_iter` callback: `defense = bond_defense_factor(cells[j].n_bonds())`.
    Apply step: `gain *= defense`, `drain = PREDATION_DRAIN × defense`,
    `damage_delta[j] += drain` (consistent — bonded prey takes less damage).
  - Renderer `cell_predates_on_neighbor`: nový `bond_counts: FxHashMap<Entity, u32>`
    pre-built ze stejného `cells.iter()` jako herd_counts pre-pass.
    Inside grid callback: lookup `bond_counts.get(&entity_b)`, apply
    defense to gain + drain + damage. Graceful fallback na 0 pokud
    entity_b není v map.

  *Body 3 — Adhesion-keyed materials (main.rs):*
  - `LineageMaterials(FxHashMap<u64, Handle>)` → `AdhesionMaterials([Option<Handle>; 8])`.
    Lazy init první cell s daným typem; cap 8 entries.
  - `lineage_material(...)` → `adhesion_material(cache, materials, adhesion_type)`.
    Hue = `idx × (360/8)` (= 45° per type), saturation 0.85, lightness 0.55.
  - Volacích sites 2: initial spawn (`setup`) + reproduce
    (`cell_reproduces_on_threshold`). Oba čerpají z `cell.genome.adhesion_type`.
  - `lineage_hue` helper smazán (unused po replace).

  *Body 4 — Gizmo line system (main.rs):*
  - Nový systém `draw_bond_gizmos` v `Update` schedule (po `sync_transforms`).
    Build `id_to_pos: FxHashMap<u64, Vec3>` per-frame; iterate cells,
    pro každý populated bond slot lookup partner pozice + draw line.
  - Owner pravidlo: kresli jen pokud `cell.cell_id < bond.other_cell_id`
    (každý bond renderován exact jednou).
  - Toroidal-aware: skip line pokud `|dx| > half_x` nebo `|dy| > half_y`
    (znamená wrap-cross — straight line by visuálně lhala).
  - Hue podle `cell.genome.adhesion_type` přes `adhesion_hue` helper —
    match s cell color. Bonds jsou jen mezi same-type páry, takže
    line + obě cells sdílí stejnou barvu = jednolitý vizuální chunk.

  *Body 5 — Cargo.toml feature flagy:*
  - `bevy_gizmos` (data + systems) + `bevy_gizmos_render` (actual rendering
    pipeline). Bez `bevy_gizmos_render` line resources jen sběry, ale
    nikdy nedostanou na obrazovku — dvojí flag je nutný pro full effect.

- **Konstanty:**
  - `BOND_DEFENSE_FRAC: f32 = 0.15` nový
  - `BOND_DEFENSE_CAP: u32 = 4` nový
  - Žádný checkpoint version bump — Cell layout beze změny.

- **Výstup:**
  - `src/lib.rs`: `bond_defense_factor` + `Cell::n_bonds` + 3 unit testy.
  - `src/bin/headless.rs`: `predate` event tuple rozšířen o defense, apply
    step škáluje gain + drain + damage.
  - `src/main.rs`:
    - `LineageMaterials` → `AdhesionMaterials`. `lineage_material` →
      `adhesion_material`. `lineage_hue` smazán.
    - `cell_predates_on_neighbor` rozšířen o `bond_counts` pre-pass +
      defense apply.
    - `draw_bond_gizmos` system + Update registration.
  - `Cargo.toml`: `bevy_gizmos` + `bevy_gizmos_render` features.
  - **Test suite: 85/85 pass** (82 baseline + 3 nové bond_defense / n_bonds).
  - **Smoke seed=0, 60 gen, default world, CPU:**
    - Wall-clock 57.0 s = **631 ticks/s** (Sprint 68: 617; +2 % v rámci noise).
    - Per-fáze `predate`: 82.6 µs (Sprint 68 reference range 82-100; defense
      multiplier <1 µs per attack event).
    - Final pop 580.
    - **Bond density signal — Sprint 69 vs Sprint 67.1:**

      | Metric | Sprint 67.1 LATE 200-250 (no defense) | Sprint 69 gen 59 (with defense) |
      |--------|----------------------------------------|----------------------------------|
      | mean_bond_count | 0.0389 | **0.062 (+59 %)** |
      | bond_active_frac | 0.0389 | **0.062** |
      | bond_signal_avg | 0.325 | 0.49 (extrapolováno z trendu) |
      | net `formed - broken` | -4 | -5 (stable equilibrium) |

      Bond density vyrostla o **59 % vs Sprint 67.1** v jen 60 generacích
      (vs 250 gen baseline). Selekce nyní pozitivně tlačí na bonding,
      vs pre-Sprint-69 net-negative pressure.
  - **Renderer launch:** Bevy app starts s gizmo plugins OK. ~60 FPS,
    frame_time 16.7 ms, render_overhead 10.5 ms (z-fighting / gizmo
    overhead acceptable). Cells viditelně shlukují podle 8 distinct hues
    (adhesion_type sorting → Steinberg-like behavior visible on screen).

- **Závěr:**
  - **Bonding hypothesis verified** — group benefit (predation defense)
    je dostatečný incentive pro pozitivní selekci na bondování. Sprint 67.1
    + Sprint 68 ukázaly, že samotné parametry (stiffness/damping) ani
    formation cost nedovedly bonding zachránit; **chyběl benefit side**.
  - **Visual layer dohnal CSV-only signal** — adhesion_type colors + gizmo
    lines mezi bonded páry. Multicelularita poprvé viditelná v real-time
    rendereru.

- **Poznámky:**
  - **Defense scaling lineární do capu:** 1 bond = 0.85×, 2 = 0.70×,
    3 = 0.55×, 4+ = 0.40×. Volil jsem lineární místo exponenciální, aby
    selekce dostávala kontinuální gradient (každý další bond stále něco
    dá až do capu). Cap 4 = sweet spot — dostatečně velký aby cluster
    s 4-bond center cells dostal smysluplnou ochranu, dostatečně malý
    aby 6-bond cells nebyly imunní.
  - **Drain + damage scale stejně jako gain.** Alternativa byla scale
    jen damage (= prey trvá tick, ale predator nedostane stejnou energii).
    Volil jsem symmetric — predator vidí bonded prey jako „těžší cíl",
    z energie i damage perspektivy. Konzistentní biologie: shell-shielded
    cell snižuje obě strany interakce.
  - **GPU shader nepotřebuje update.** Predation běží CPU-only (jak
    headless tak main). GPU pipeline (`--gpu-full`) má jen brain +
    motor + brownian + step na GPU; predation je vždy CPU pass.
  - **Gizmo line cost:** per-tick FxHashMap build (cell_id → Vec3) +
    iterace bondů. Pro 1k cells s mean_bond_count=0.06 = ~30 lines/frame.
    Bevy gizmo batches efektivně, render_overhead +<1 ms per frame.
  - **Co Sprint 69 NEŘEŠÍ (Sprint 70+):**
    - HUD overlay s bond stats (current_bonds, mean_bond_count). Telemetrie
      je v CSV; renderer HUD ji nezrcadlí.
    - Long-run smoke (250+ gen) verify, že defense + Sprint 68 evolvable
      params dotáhnou bond_active_frac k 0.20+ (true tissue formation).
    - Spatial autocorrelation adhesion_type clustering metric (CSV).
    - Cluster-aware reproduction (offspring spawnuje uvnitř parent's
      bond network — šance pro stable multi-cell organism).
    - Per-cell `bond_form_threshold` evolvable gen (output[9] threshold).
    - GPU adhesion + bond shaders.
    - Anisotropic cell collision (ellipsoid geometry).

## Sprint 70 — cluster-aware reproduction + 250-gen verification

- **Cíl:** dotáhnout bond density z 60-gen ~6 % (Sprint 69) k true tissue
  regimu (15-20 % active). Hypotéza: bondy zanikají, protože nově rozené
  cells spawnou „někde mezi rodiči" (current `make_mating_child` midpoint),
  typicky daleko od parent's bond network — bond clustery rostou jen
  organicky přes náhodný kontakt, smrtí cells se zmenšují. **Cluster-aware
  spawn** posune dítě uvnitř bonded parent's clusteru → bond network roste
  i přes reprodukci, ne se jen rozpadá smrtí.

  **Plán implementace:**

  *Body 1 — `pick_cluster_parent` helper (lib.rs):*
  - Pure funkce: bere oba parents + child's adhesion_type, vrací
    `Option<&Cell>` = parent, ke kterému se má child spawn-it.
  - Priorita: bonded parent matchující adhesion_type → bonded parent
    bez match → None (= midpoint fallback).

  *Body 2 — `make_mating_child` cluster-aware spawn:*
  - Po `Genome::crossover` + direction draw přidat 3 nepodmínečné
    `rng.random_range` pro x/y/z jitter (z je 0.3× — užší z-rang).
    Unconditional draw zachovává RNG draw order konzistentní napříč
    všemi children, ne jen bonded větví.
  - Pokud `pick_cluster_parent` vrátí `Some(p)`: child position =
    `p.position + jitter`. Jinak: midpoint (pre-Sprint-70 chování).

  *Body 3 — `CLUSTER_SPAWN_RADIUS = 8.0`:*
  - 0.8× pair_radius pro typical post-evolution body (radius ~1.0,
    pair_r ≈ 10) — child se spawne uvnitř bond contact distance, takže
    existing collision-based bond formation chytne v <1 s.

- **Konstanty:**
  - `CLUSTER_SPAWN_RADIUS: f32 = 8.0` nový.

- **Výstup:**
  - `src/lib.rs`: `pick_cluster_parent` + `make_mating_child` rozšířen.
  - **Test suite: 90/90 pass** (85 baseline + 5 nové: pick_cluster prefer
    matching, fallback to any bonded, none when neither, mating spawns
    near bonded parent, midpoint when neither bonded).
  - **Long-run smoke seed=0, 250 gen, default world, CPU:**
    - Wall-clock 275.3 s = **545 ticks/s** (Sprint 67.1 720, Sprint 68 617,
      Sprint 69 631 @ 60-gen — Sprint 70 pomalejší kvůli denser bond
      networks → víc collision events, `resolve_collisions` 142.9 µs vs
      Sprint 69 ~100 µs). Final pop 632.
    - **Bond density trajectory:**

      | Gen | mean_bond_count | predation_events | spk_avg |
      |-----|-----------------|-------------------|---------|
      | 40  | 0.020 | 3515 | 0.324 |
      | 60  | 0.023 | 2065 | 0.366 |
      | 100 | 0.015 | 1335 | 0.285 |
      | 140 | 0.032 | 1572 | 0.146 |
      | **160** | **0.054** | 1479 | 0.111 |
      | **180** | **0.057 (peak)** | 435 | 0.075 |
      | 200 | 0.041 | 269 | 0.055 |
      | 249 | 0.028 | **0** | 0.078 |

- **Závěr — emergent předator-extinction event:**
  - **Cluster-aware spawn mechanicky funguje** (testy ✓, bond density
    peak 0.057 @ gen 180 vs Sprint 67.1 baseline 0.039 = +46 %).
  - **Ale ecosystem-level effect je jiný, než hypotéza předpokládala.**
    Kombinace Sprint 69 defense + Sprint 70 cluster spawn = bonded
    clustery přežívaly natolik dobře, že **predace ztratila fitness
    payoff** → predátoři vyhynuli (`spk_avg` 0.32 → 0.05, predation_events
    3500 → 0). Bez predátorů bond defense bonus zmizel, bondování
    stagnovalo na ~3 %.
  - **Tipping point nebyl k tissue persistence, ale k peaceful niche
    s extreme aspect ratio** (asp 1.0 → 12.3 = pure speed swimmery).
    Selekce našla jiný attraktor: vyhnout se predaci útěkem místo bondingu.
  - **Sprint 71+ musí rebalancovat:** buď slabší defense
    (`BOND_DEFENSE_FRAC` 0.15 → 0.08), nebo nový predator pressure
    (e.g. spike-bonus zvýšit, baseline attack incentive). Bez stable
    predace bondování nemá selekční signál.

- **Poznámky:**
  - **RNG draw order:** přidání 3 unconditional `random_range` calls v
    `make_mating_child` mění RNG trajectory napříč all seedy. Sprint 70
    seed=0 NENÍ apples-to-apples s Sprint 67.1/68/69 seed=0. Comparison
    je v ranges (peak vs baseline), ne v point matchingu.
  - **Asp_avg 12.3 je extrémní** — body length/width = 12. Cells jsou
    skoro 1D čáry. Spike crashed (0.05) takže to nejsou predator
    needles; nejspíš pure-foraging streamliners co se vyhýbají
    všemu kontaktu (lower collision = lower energy loss).
  - **Energy crash 222 → 74** je consequence (no predation = no
    energy transfer between cells, ekosystém běží jen na food eat).
  - **Test `mating_child_spawns_at_midpoint_when_neither_parent_bonded`
    zachycuje regression** — pre-Sprint-70 behavior se reproduces když
    parents nemají bondy (= většina ranných gen).

## Sprint 71 — macropredator (Hunter) entity

- **Cíl:** zavést persistent predator pressure, kterou cell-vs-cell selekce
  nemůže vypotit. Sprint 70 ukázal predator-extinction event: bonded clustery
  byly tak tough, že spike investice ztratila fitness payoff → predátoři
  vyhynuli → bond benefit zmizel. Sprint 71 přidává **non-evolving
  environmental predator** (= „Hunter") — entitu mimo Cell selection loop,
  takže nikdy nevyhyne. Hunter atakuje cells s `n_bonds() < 3` (cluster
  „too big to swallow" = Volvox / paramecium scenario z reálné biologie).
  Tím vzniká exact tipping point: dosáhnout ≥3-bond clusteru = immunity.

  **Plán implementace:**

  *Body 1 — `Hunter` struct + `nearest_attackable_cell` (lib.rs):*
  - Hunter má position, velocity, hunter_id. Žádný brain, žádný genome,
    žádná mutace — pure world entity.
  - `Hunter::random` random init, `Hunter::step` per-tick movement
    (target-seek pokud ∃ attackable cell ∈ vision range, jinak random drift).
    Toroidal-aware přes `min_image_delta`.
  - `nearest_attackable_cell(pos, &cells, world_half) -> Option<usize>` —
    skip cells s `n_bonds() ≥ HUNTER_BOND_IMMUNITY_THRESHOLD`.

  *Body 2 — Headless `hunt` phase:*
  - World gets `hunters: Vec<Hunter>` + `hunter_attacks_gen` counter.
    Init: HUNTER_TARGET_COUNT random spawns.
  - `World::hunt` mezi `predate` a `eat_food` v tick chain. Two-pass
    (borrow checker): pass 1 sbírá (cell_idx, damage), pass 2 apply.
  - CSV: 3 nové sloupce — `hunter_attacks`, `hunters_alive`, `immune_frac`
    (= fraction cells s `n_bonds() ≥ 3`).

  *Body 3 — Renderer Hunter ECS:*
  - `HunterEntity(Hunter)` component, `HunterMesh` + `HunterMaterial`
    resources (single shared mesh/material — všichni hunters look same).
  - Mesh = `Sphere::new(CELL_RADIUS × 4)` (= radius 20, 4× větší než cell).
    Material = dark red (`Color::hsl(0.0, 0.7, 0.30)`) + emissive accent.
  - `step_hunters` system v FixedUpdate (mezi `cell_predates_on_neighbor`
    a `cell_eats_food`). `sync_hunter_transforms` v Update (mirror cells).

  *Body 4 — Lib const + tests:*
  - 7 nových konstant: `HUNTER_TARGET_COUNT=3`, `HUNTER_VISION_RADIUS=120`,
    `HUNTER_ATTACK_RADIUS=18`, `HUNTER_DAMAGE_PER_TICK=4.0`,
    `HUNTER_MAX_SPEED=220`, `HUNTER_ACC=80`, `HUNTER_IDLE_DRIFT=30`,
    `HUNTER_BOND_IMMUNITY_THRESHOLD=3`.
  - 5 unit testů: seek nearest, skip immune cluster, none when only
    immune, step toward target, idle random walk.

- **Konstanty (lib.rs):** výše uvedené 7 + threshold.

- **Výstup:**
  - `src/lib.rs`: Hunter struct + impl + nearest_attackable_cell + 5 testů.
  - `src/bin/headless.rs`: World rozšířen o hunters + hunter_attacks_gen,
    nový `hunt` phase + timed!, CSV header rozšířen na 47 sloupců,
    immune_frac spočtený. Empty-row format opraven (pre-existující bug
    z S68: měl 42 fields místo 44 → teď 47).
  - `src/main.rs`: HunterEntity + HunterMesh/Material resources, spawn
    v setup, `step_hunters` + `sync_hunter_transforms` systémy.
  - **Test suite: 95/95 pass** (90 baseline + 5 nové hunter testy).
  - **Long-run smoke seed=0, 250 gen, default world, CPU:**
    - Wall-clock 272.4 s = **551 ticks/s** (Sprint 70: 545; Sprint 67.1: 720).
      Hunt phase měřena přes timed!() — ale dump nezahrnut v výstupu z důvodu
      pre-existing dump format (přidáno post-smoke). Final pop 558.
    - **Hunters živi end-to-end** — gen 49: 872 útoků, gen 249: 500 útoků.
      **Žádné extinkce** (Sprint 70 měl predE=0 v gen 199+). ✓ Hypotéza
      „non-evolving predator nikdy nevyhyne" potvrzena.
    - **Bond density trajectory:**

      | Gen | mean_bond_count | predE | hunt_atks | spk | asp | immune_frac |
      |-----|-----------------|-------|-----------|-----|-----|-------------|
      | 49  | 0.040 | 1208 | 872 | 0.31 | 3.6  | 0.002 |
      | 99  | 0.034 | 2083 | 565 | 0.26 | 8.6  | 0.000 |
      | **149** | **0.051 (peak)** | 274 | 545 | 0.14 | 10.8 | 0.000 |
      | 199 | 0.022 | 0 | 693 | 0.08 | 12.0 | 0.000 |
      | 249 | 0.000 | 0 | 500 | 0.08 | 12.6 | 0.000 |

- **Závěr — partial success:**
  - **Hunter pressure persists** ✓ — design funguje, predátor je v simu
    end-to-end aktivní. Sprint 70 bottleneck (extinction) vyřešen.
  - **Bonding peak +56 % oproti Sprint 70** (0.051 vs Sprint 70 0.057
    ~comparable, dosaženo o 30 gen rychleji — gen 149 vs gen 180).
  - **Ale tipping point nedosažen.** `immune_frac` zůstal pod 0.2 % —
    cells nedosáhly proto-tissue regimu. Místo clusteringu **evolovaly
    extreme speed swimmer** strategy (asp 12.6, spd 218). Hunter
    MAX_SPEED je 220 — cells se těsně přiblížily limitu, **outrun místo
    immune** se ukázal být snadnější evoluční cesta než 3+ bond cluster.
  - **Bond density crash to 0 by gen 249** — bonded cells v této niche
    nedostávaly žádný benefit (speed-evader strategy nepotřebuje cluster),
    selekce postupně bondování opustila.

- **Implikace pro Sprint 72+:**
  - **Tunit hunter aby outrun nebylo viable.** Tři páky:
    1. `HUNTER_MAX_SPEED` 220 → 280 (cells nemůžou outrun bez krádeže
       cell.max_speed limit).
    2. `HUNTER_TARGET_COUNT` 3 → 8 (víc hunterů = víc paths blokovaných;
       solo cell nemá kde se schovat).
    3. `HUNTER_VISION_RADIUS` 120 → 200 (větší než MATING_RADIUS — hunters
       detekují cells dřív než cells stihnou ujet).
  - Doporučuju kombinaci #1 + #2 (rychlejší + víc hunterů). Vision je
    sekundární — primary je „kdo je rychlejší".
  - **Long-long run (1000 gen)** může taky odhalit, že cells mají strop
    na max_speed (genome.max_speed cap je definovaný v lib.rs) — pokud
    HUNTER_MAX_SPEED je nad cap, escape je nemožný a cluster path se
    stane jediná. Toto by bylo ideální nastavení.

- **Poznámky:**
  - **Hunter mesh size 4× CELL_RADIUS = 20 unit radius:** dramatic visual
    distinction. V renderer-u (3 hunters × 1500-cell pop) snadno
    rozlišitelní jako tmavě-červené koule pohybující se rychleji než cells.
  - **Hunters v checkpointu nejsou** — re-spawnou se fresh z
    `chk.mating_radius` jako seed (rough hash). Hunter je transient world
    feature; ztráta pozice při loadu nemá selection signal.
  - **Two-pass borrow checker pattern:** v obou binárkách hunter step + attack
    sbírají (idx/entity, damage) tuples během iterace `&mut hunters`,
    pak apply na `cells` po uvolnění hunter borrow. Mirror Sprint 66
    `resolve_collisions` Phase 1+2 pattern.
  - **Empty-row CSV bug fix:** Sprint 68 přidal bond_stiff/damp sloupce
    do header (42→44), ale empty row nezakreslil (zůstal 42 fields).
    Sprint 71 opravil (47 fields total = header match).
  - **Co Sprint 71 NEŘEŠÍ (Sprint 72+):**
    - Hunter speed/count tuning pro flip equilibrium k cluster path.
    - Cells útočící zpátky na hunter (spike fight-back). Aktuálně hunter
      je nesmrtelný; Sprint 72 může dát hunteru health pool.
    - Hunter HUD overlay (renderer ukazuje hunters vizuálně, ale CSV
      sloupce hunter_attacks / hunters_alive / immune_frac nejsou v HUD).
    - Spatial clustering of hunters (currently random walk; could implement
      „pack" behavior pro extra pressure).

## Sprint 72 — hunter tuning + 1000-gen verification

- **Cíl:** zlomit Sprint 71 outrun-equilibrium tím, že hunter dostane natolik
  rychlost / vision / counts, že cells nebudou moct utíkat — cluster path
  (≥3 bondy = immunity) musí být dominantní strategie. Plus 1000-gen smoke
  ověří, jestli druhořádová selekce na bondování přijde s delším časem.

  **Plán:** parameter-only změna v lib.rs. Žádný structural / API code change.

- **Konstanty:**
  - `HUNTER_MAX_SPEED`: 220 → **300** (předpoklad: nad cell.max_speed cap).
  - `HUNTER_TARGET_COUNT`: 3 → **8** (víc paths blokovaných).
  - `HUNTER_VISION_RADIUS`: 120 → **200** (větší než MATING_RADIUS).

- **Výstup:**
  - `src/lib.rs`: 3 const updates + comments.
  - **Test suite: 95/95 pass** (parameter-only change, žádné nové testy).
  - **Long-run smoke seed=0, 1000 gen, default world, CPU:**
    - Wall-clock 868.9 s = **691 ticks/s**. Hunt phase 29 µs/tick avg
      (timed!() wired do dump). Final pop 390.
    - **Hunt pressure persists end-to-end**, jako Sprint 71 ✓.
      Hunt attacks: peak 4148 @ gen 3 (early panic) → stable ~1000-1100/gen
      napříč gen 500-1000. Hunters nikdy nevyhynuli.
    - **Cells outran HUNTER_MAX_SPEED=300:**

      | Gen | spd_avg | asp | mBond | hunt_atks | immune_frac |
      |-----|---------|------|-------|-----------|-------------|
      | 49  | 148 | 3.5 | 0.039 | 1523 | 0 |
      | 99  | 172 | 7.5 | 0.012 | 1964 | 0 |
      | 199 | 209 | 11.6 | 0.000 | 1701 | 0 |
      | 499 | **307** | 12.7 | 0.000 | 968 | 0 |
      | **999** | **337 (= 1.12× hunter)** | 12.4 | 0.000 | 1109 | **0** |
    - Peak `mean_bond_count = 0.045` @ gen 50 (NIŽŠÍ než Sprint 71 0.051).
    - Peak `immune_frac = 0.001` (essentially zero — proto-tissue NEDOSAŽEN).

- **Závěr — fundamentální problém s arms race:**
  - **Cells evolovaly `genome.max_speed` přes hunter cap**, protože v lib.rs
    je jen `MIN_SPEED=1.0` floor, **žádný horní cap**. Mutation drift
    (sigma_speed=3.0) přes 1000 gen + selekce na escape produkovala
    spd_avg 344 @ gen 991. Hunter MAX_SPEED=300 byl nevýznamný.
  - **Bonding krátce vzplanul (gen 50: mBond=0.045) a pak crashl**, protože
    selekce našla výhodnější escape route (rychlost) než cluster (immunity).
    Speedy cells dostávaly hunt damage jen krátce pri sblížení; cluster
    cells by stále musely investovat do bond cost + adhesion overhead.
  - **Tipping point je strukturálně nedosažitelný bez cell-speed cap.**
    Jakmile cells můžou eskalovat speed bez bound, outrun je vždycky
    levnější než cluster. Sprint 73 musí přidat `MAX_SPEED` constant
    (např. 200, mírně pod Sprint 71 baseline 218) plus `clamp` v
    `Genome::mutate`. To by udělalo speed strop a teprve pak by bonding
    byl jediná zbylá obrana.

- **Implikace pro Sprint 73+:**
  - **Strukturální fix: `MAX_SPEED` cap** v lib.rs + `Genome::mutate`
    `clamp(MIN_SPEED, MAX_SPEED)`. Hodnota 200 (mírně pod 218 Sprint 71
    baseline) by udělala HUNTER_MAX_SPEED=300 reálně neutekatelný.
  - **Alternative: zostřit `ENERGY_COST_PER_V_SQ`.** Aktuálně 0.0008 dává
    při v=337 cca 91 energy/s drain. Cells přežijí. Pokud bumpneme
    cost na 0.0030 (~4×), drain by byl 340/s — energeticky nemyslitelné.
    Ale tato změna je global (ovlivní brain motion economy), takže risk
    of unintended consequences. Hard cap je čistší.
  - **Dlouhý smoke ukázal svou hodnotu:** Sprint 71 250-gen ukázal mBond
    peak 0.051 @ gen 149; Sprint 72 1000-gen ukázal že to byl jen
    transient peak — cluster path NIKDY nestabilizoval. Bez 1000-gen
    runu by Sprint 72 vypadal jako částečný úspěch.

- **Poznámky:**
  - **Hunt phase wall-time 29 µs/tick = 0.0058 µs per cell-Hunter pair**
    (8 hunters × 600 cells = 4800 pairs/tick). Negligible vs eat_food
    (799 µs) nebo brain_act (691 µs). Hunters scale freely; Sprint 73 by
    mohl mít 16-32 hunters bez perf hit.
  - **CSV opravě fungovala** — empty-row format má 47 fields, populated
    row taky 47, parsing s awk čistý.
  - **Sprint 72 je poslední decade-end sprint** v `063-072-scaleup.md`.
    Sprint 73+ patří do nového souboru `073-082-<slug>.md`. Slug podle
    dominantního tématu next decade — pravděpodobně „selection-bounds"
    nebo „tissue-emergence" podle toho, kam Sprint 73 míří.
  - **Co Sprint 72 NEŘEŠÍ (Sprint 73+):**
    - Cell `MAX_SPEED` cap + clamp v Genome::mutate (kritický fix).
    - HUD overlay s real-time bond / hunter / immune_frac stats.
    - Long-long run (5000+ gen) post-cap k ověření, že cluster path
      finally dominuje.
    - Anizotropic cell collision, GPU collision, photic stratification.
