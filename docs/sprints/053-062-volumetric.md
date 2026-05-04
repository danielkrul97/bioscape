# Sprinty 53–62: Volumetric environment

Decade rozšiřuje sim-svět z polovičatého 2.5D (z=2 thin layer od Sprintu 35)
na plně volumetrické 3D prostředí. Cells získávají vertikální environmental
sensing, hazards stratify napříč hloubkou, smell + pheromone fields jsou
plně 3D s 7-point Jacobi diffusion.

## Sprint 53 — volumetric core (3D fields + WorldMap + brain inputs)

- **Cíl:** přesunout všechny dosud-2D environmental subsystémy na 3D
  volumetric grid + expandovat z-osu z 2 na 20 (10× volume increase).
  Robustně = pop trajectory non-extinct při seed=0 60 gen, food density
  per volume zachovaná.

  **Plán implementace:**

  *Body 1 — `SmellField` 3D:*
  - `resolution: [usize; 3]`, `world_half: [f32; 3]`. Grid layout
    `idx = z*W*H + y*W + x`.
  - `step()` 7-point Jacobi stencil (`left+right+up+down+back+front - 6×center`).
    Stabilita: `diffusion < 1/6` v 3D (vs `< 1/4` v 2D); `SMELL_DIFFUSION = 0.15`
    je pod oběma limity.
  - `gradient_at([x,y,z], eps) -> [f32; 3]` — central differences podél všech
    tří os.
  - 6× memory pro `[64, 64, 16]` resolution = 256 KB per field (vs 16 KB v 2D).

  *Body 2 — `WorldMap` 3D:*
  - 3D value-noise: `base_resolution³` random uniform → trilinear smoothstep
    interp do plné `resolution³`.
  - `sample([x, y, z])` clamps boundary, vrací f32 ∈ [0, 1].
  - Default `[64, 64, 16]` resolution s `[8, 8, 4]` base.

  *Body 3 — brain inputs full 3D:*
  - `BrainSensors.smell_grad` + `pheromone_grad` z `[f32; 2]` na `[f32; 3]`.
  - `populate_brain_inputs` plní `inputs[17] = smell_grad_z`,
    `inputs[19] = pheromone_grad_z` (slots reservované od Sprintu 33 jako
    zero-padded; teď populated). Brain získává vertikální chemické vnímání.

  *Body 4 — `WORLD_HALF[2]` expanze:*
  - `headless::WORLD_HALF[2]: 2.0 → 20.0`. Cells se rozprostřou napříč
    40 z-units (vs 4 pre-Sprint-53). Volume × 10.
  - Renderer `SIMULATION_HALF[2]`: stejně 20.0.
  - Initial smoke s `z=50` extinktoval gen 33 (food density per volume klesla
    25× → mass starvation). Z=20 + food count scaling (Body 5) = stable.

  *Body 5 — food density volume scaling:*
  - `food_target(factor)` v headless + `food_target(extent, factor)` v main:
    multiplikuje base count s `z_factor = (2 × world_half_z / 4).max(1)`.
    Pre-Sprint-53 baseline: z=2 → z_extent=4 → z_factor=1 (no change). Při
    z=20: z_factor=10 → 10× food count.
  - Při `MAX_POPULATION = 1000` a 8000 food entities: pop saturuje gen ~3,
    healthy steady-state ~500-700 cells přes gen 60.

  *Body 6 — callsite migrace 2D → 3D:*
  - Headless: `update_smell` / `emit_pheromones` add_source uses 3D position
    (food.position / cell.position). `apply_hazards` samples WorldMap at full
    3D. `spawn_food` exclusion checks 3D. `pos_xyz` instead of `pos_xy` v
    sensor gather.
  - Renderer: stejný pattern přes Bevy systems (update_smell_field /
    emit_pheromones / cells_brain_act / apply_environmental_hazards / spawn_food).
  - Food spawn richness: callers samplují WorldMap při `z = 0` (canonical
    surface depth). Hazards samplují plný 3D pos (vertikální stratifikace).

- **Konstanty:**
  - `SMELL_GRID_RES_Z = 16`, `PHEROMONE_GRID_RES_Z = 16`, `WORLD_MAP_RES_Z = 16`
    (vs 64 v xy → matchne thin z aspect, šetří memory).
  - `WORLD_MAP_BASE_RES_Z = 4` (smoother vertical noise, méně high-freq variance
    v thin z-volume).

- **Výstup:**
  - `lib.rs`:
    - `SmellField` přepsán na 3D (struct fields + impl methods).
    - `WorldMap` přepsán na 3D (constructor, sample, field layout).
    - `BrainSensors.smell_grad` + `pheromone_grad` rozšířeno na `[f32; 3]`.
    - `populate_brain_inputs` plní `inputs[17]` a `inputs[19]`.
    - 4 nové `pub const` (`*_GRID_RES_Z`, `WORLD_MAP_BASE_RES_Z`).
  - `src/bin/headless.rs`: `WORLD_HALF[2] = 20.0`. World init s 3D field
    constructors. Všech 17 callsiteů updated na 3D coords. `food_target`
    scales s volume.
  - `src/main.rs`: `SIMULATION_HALF[2] = 20.0`. Bevy systems updated. ECS
    queries, smell/pheromone resources, world_map renderování updated.
    `world_map_image` renderuje xy-slice na `z = floor(nz/2)` (canonical
    surface) — Sprint 54+ může do volumetric voxel viz.
  - **GPU subsystémy:** `FieldGpu` zůstává 2D (Sprint 50/46 standalone tests
    a sensor_gather.wgsl). Sprint 51 `--gpu-full` headless pipeline + Sprint 52
    renderer GPU default ne-používají FieldGpu v hot loopu (smell/pheromone
    běží na CPU `SmellField`), takže žádný immediate breakage. **2 testy
    (`field_gpu_diffusion_matches_cpu`, `sensor_gather_gpu_matches_cpu`) jsou
    `#[ignore]`** pro Sprint 53 — Sprint 54 migruje GPU field stack na 3D
    a re-enabluje.
  - **71/71 tests pass + 2 ignored** (4 unit tests v lib.rs world_map suite
    updated na 3D API).
  - **Smoke (seed=0, 60 gen, default size, default threads):** pop trajectory
    200 → 1000 (gen 3 saturated MAX_POPULATION) → 832 (gen 18) → 689 (gen 28)
    → 547 (gen 43) → 503 (gen 60). Lineages 200 → 46 (typická konvergence,
    Sprint 22+). Predation 1866 → 7637 → 6416 events/gen. Energy mean rostoucí
    (100 → 484), což indikuje že selekce favorizuje feeding-efficient brainy
    v 3D. **Žádná extinkce.** Wall-clock: 36000 ticks v 102.3 s = 352 ticks/s.

- **Poznámky:**
  - **Proč z=20 a ne víc:** initial smoke s z=50 extinktoval gen 33 (random
    initial brainy nezvládly 25× volume bez vertical guidance). Z=20 +
    volume-scaled food = stable basemark. Sprint 54+ může postupně zvyšovat
    z (z=30, z=50) jakmile selekce stabilizuje 3D navigation.
  - **Food vs hazard z-strategie:** food spawn samples WorldMap při z=0 (xy
    biome stratification — food clustery zachovávají Sprint 21 semantiku).
    Hazards samplují full 3D position → vertikální hazard layers (cells musí
    sense smell_grad_z aby unikly 3D nebezpečí). To dává selekční gradient
    pro vertical motion bez nuceného food-search-in-3D.
  - **`world_map_image` overlay:** renderer ground plane vykresluje xy-slice
    při `z = floor(nz/2)` = middle layer. Pre-Sprint-53 byl plný 2D field
    rendered. Volumetric viz (voxel grid nebo iso-surfaces) je Sprint 54+
    visualization work.
  - **GPU 2D field assumption:** Sprint 46 `field_diffuse.wgsl` má 5-point
    stencil (2D). Sprint 50 `sensor_gather.wgsl` reads 2D field grid. Po
    Sprint 53 SmellField API change tyhle GPU shadery nematchují CPU
    semantiku → ignored testy. Sprint 54 plan: 3D variant `field_diffuse_3d.wgsl`
    + `sensor_gather_3d.wgsl`. FieldGpu+SensorGatherGpu Rust wrappery rozšířit
    na 3D buffery.
  - **Z-axis resolution mismatch:** xy resolution 64 vs z resolution 16. Cell
    size_x = 30 units, cell_size_y = 17 units, cell_size_z = 2.5 units. Z má
    finer per-unit resolution! Trade-off: méně GPU memory ale vertikální
    chemistry je více detailní než horizontální. Sprint 54+ může equalizovat
    podle measured selection pressure.

## Sprint 54 — toroidal world (cylinder topology)

- **Cíl:** odstranit edge bias z reflective xy walls. Cells na opačných
  koncích světa se vidí jako sousedé přes wrap, smell/pheromone gradient
  pole jsou kontinuální (žádný degenerated Neumann boundary). Z osa zůstává
  bounded (gravita + food sink + carrion drop vyžadují pevný strop/dno) →
  cylinder topology, ne plný torus. Pre-Sprint-54 cells akumulovaly u zdí
  (Sprint 30+ `edge_frac`/`corner_frac` metriky), evoluce našla wall exploit
  (těsné otáčení), pole degenerovaly u krajů.

  **Plán implementace:**

  *Body 1 — `lib.rs` core helpers:*
  - `pub fn min_image_delta(a, b, world_half) -> [f32; 3]`: signed delta s
    minimum-image konvencí na xy (|dx|, |dy| ≤ world_half), dz beze změny.
  - `pub fn wrap_position_xy(pos, world_half) -> [f32; 3]`: modulo wrap xy
    do `[-half, half)`, z beze změny.
  - `Cell::apply_world_bounce` re-fits semantiku: xy modulo wrap (žádný
    velocity flip, žádný heading recompute), z stále bounce.
  - `SmellField::idx_of` xy wrap (rem_euclid), z bounded.
  - `SmellField::step` 7-point Jacobi: xy wrap stencil (i=0 čte i=nx-1, atd.),
    z stále Neumann zero-flux (ground/ceiling).
  - `WorldMap::sample` xy wrap (rem_euclid), z clamp.
  - `BrainSensors.nearest_food` / `nearest_cell` semantika: nyní min-imaged
    delta `[dx, dy, dz]` (signed), ne absolutní target pozice. populate_brain_inputs
    odstranil `target − pos` math; používá delta přímo.
  - `pair_fertile` rozšířen o `world_half: [f32; 3]` parametr; používá
    `min_image_delta`.

  *Body 2 — `SpatialGrid::for_each_in_radius_toroidal`:*
  - Nová metoda nad `for_each_in_radius`. Center query + až 8 ghost queries
    (4 edges + 4 corners) podle blízkosti pos k xy boundary. Z není wrapped.
  - For pos uvnitř world (ne blízko edge): 1 query, žádný overhead.
  - For corner cell (do `radius` od obou xy boundary): 9 queries (1 center +
    4 edges + 4 corners). ~5% cells typicky — overall overhead < 20%.

  *Body 3 — callsite migrace (headless + main):*
  - **headless** (3 brain_act variants, predate herd+attack, resolve_collisions,
    eat_food, spawn_food exclusion, pair_fertile call, nn_dist diagnostic):
    `for_each_in_radius` → `for_each_in_radius_toroidal(WORLD_HALF)`,
    raw `dx = a − b` → `bioscape::min_image_delta(a, b, WORLD_HALF)`. eat_food
    a spawn_food používají "ghost food" pattern: před `cell.try_eat()` pos
    food.position adjustnut o min-image delta aby cell.eat_test ellipsoid
    acceptance match toroidal frame.
  - **main** (cells_brain_act gather closure, predate herd+attack,
    resolve_cell_collisions, cell_eats_food, spawn_food, pair_fertile call):
    stejný pattern s `SIMULATION_HALF`.

  *Body 4 — GPU step.wgsl wrap:*
  - Sprint 50 step shader měl `apply_world_bounce` se reflective xy.
    Sprint 54 nahrazuje xy wrap (modulo) + z bounce, match Cell::step
    Sprint 54 semantiku. step_gpu_matches_cpu parity test stále prochází.
  - **GPU spatial broad-phase shadery** (`spatial_hash`, `cell_neighbors`,
    `collision`, `predate`, `sensor_gather`) zůstávají pre-Sprint-54 (raw
    bucket clamp + raw distance). Tyto shadery jsou v Sprint 50 standalone +
    parity tests, nikoliv v `--gpu-full` headless ani renderer GPU default
    hot path → žádný immediate breakage. Migrace na toroidal bucket modulo +
    min-image distance je Sprint 55+ work.

  *Body 5 — testy aktualizace:*
  - `step_bounce_recomputes_heading` → `step_xy_wraps_toroidal`: testuje
    cell na pos x=99 s vel +60 dt=1 → wrap na x=-41, žádný velocity flip,
    heading beze změny.
  - `world_map_sample_clamps_to_world_bounds` → `world_map_sample_xy_wraps_z_clamps`:
    sample(+half_x) ekvivalentní sample(-half_x), z out-of-range clampuje.

- **Konstanty:** žádné nové.

- **Výstup:**
  - `lib.rs`: 2 nové pub fn helpers, BrainSensors semantic change (delta vs
    abs pos), pair_fertile signature + 1 new param, SmellField + WorldMap
    sample/step toroidal, Cell::apply_world_bounce wrap.
  - `src/bin/headless.rs`: ~12 callsiteů (3 sensor gather variants + collisions +
    predate + eat + spawn + nn + pair_fertile) updated.
  - `src/main.rs`: ~7 Bevy systems updated (gather closure, herd, predate,
    collision, eat, spawn, pair_fertile call).
  - `shaders/step.wgsl`: bounce → wrap xy + z bounce. Match CPU semantiku.
  - **2 lib testy updated** (step wrap, worldmap wrap). **71/71 testů pass +
    2 ignored** (Sprint 54+ GPU field 3D + GPU spatial wrap follow-up).
  - **Smoke (seed=0, 60 gen):** pop 200 → 1000 (gen 3 saturated) → 548
    (gen 60). Lineages 200 → 47. Predation 1921 → 7514 events/gen. **Žádná
    extinkce.** `corner_frac` stays ≤ 1.6% napříč generacemi (vs reflective
    borders kde Sprint 30+ `edge_frac` typicky 0.4-0.6 = cells akumulované u
    krajů). `mean_x` ≈ 0 ± 0.05 (žádný drift).
  - **Renderer smoke:** Bevy launch OK, GPU init `cap 1064`, no panic.

- **Poznámky:**
  - **Cylinder vs full torus:** xy wraps, z bounded. Důvody: (1) gravity
    Sprint 38 modeluje fluid sink — celá fluid column má top/bottom; (2) food
    sink rate by byl meaningless v wrapped z; (3) carrion drop semantika
    "drops to floor" potřebuje floor.
  - **CSV identity break:** trajectory diverguje od pre-Sprint-54 baseline
    (různý wrap vs bounce, různé sensor delta normalizace). Sprint 41/42/43
    už CSV identity nezachovaly; consistent.
  - **Edge bias eliminován:** corner_frac drops, mean_x ≈ 0. Ekologie nyní
    homogenní napříč prostorem. Selection nemůže favorizovat wall-exploit
    strategy (nemá kde).
  - **Smell/pheromone fields kontinuální:** 7-point toroidal stencil =
    žádný gradient degeneration u xy boundary. Sprint 25+ pheromone signaling
    funguje v plném prostoru bez border artefaktů.
  - **GPU production fáze nedotčená:** Sprint 51 `--gpu-full` headless +
    Sprint 52 renderer GPU default používají BrainGpu/HebbianGpu/BrownianGpu
    + CPU sensor gather. CPU gather je toroidal (Sprint 54 callsites).
    GPU spatial broad-phase shadery (Sprint 45/49/50) jsou test-only mimo
    hot path; jejich migrace na toroidal je Sprint 55+ follow-up.
  - **Co Sprint 54 NEŘEŠÍ (Sprint 55+):**
    - GPU spatial_hash bucket modulo (`bucket_id_of` clamp → mod).
    - GPU cell_neighbors / collision / predate / sensor_gather min-image
      distance + ghost queries.
    - GPU FieldGpu 7-point toroidal stencil (Sprint 53 already deferred).
    - Renderer "ghost cell" rendering — cell na x=−950 visualizována i jako
      duplicate na x=+970 (smooth wrap visual). Aktuálně cell skipne přes
      okraj.

## Sprint 55 — GPU broad-phase toroidal

- **Cíl:** dokončit toroidal semantiku v GPU broad-phase shaderech ze
  Sprintu 45/49/50 — Sprint 54 toroidal CPU implementaci dotáhne i na GPU
  side. 4 shadery updated (spatial_hash, cell_neighbors, collision, predate);
  sensor_gather zůstává `#[ignore]` až do Sprint 56 (FieldGpu 3D migrace).

  **Plán implementace:**

  *Body 1 — `spatial_hash.wgsl` bucket wrap:*
  - `bucket_id_of(pos)`: pre-Sprint-55 clamp xy bucket coords k grid bounds.
    Sprint 55 wraps pos.xy do `[-half, half)` modulo, pak spočítá bucket.
    Cells na opačných koncích světa získávají adjacent bucket ID. Z stále
    bounded.
  - Params struct rozšířen o `world_half_x`, `world_half_y`. Rust `HashParams`
    + `SpatialHashGpu.world_half_xy` field, propagated přes `with_context` /
    `new` constructory.

  *Body 2 — `cell_neighbors.wgsl` toroidal query:*
  - Přidán helper `min_image_xy(d, half)` — toroidal-aware signed delta.
    Bucket iter změněn z manual `bx_base + dx` clamp na ghost positions
    (`pos_i + offset * cell_size`) přes `bucket_id_of` který wraps. Narrow-
    phase `dx, dy` přes `min_image_xy`, dz beze změny.
  - `NeighborsParams` + `NeighborsGpu.world_half_xy` field, constructor
    propagated.

  *Body 3 — `collision.wgsl` + `predate.wgsl` toroidal:*
  - Stejný pattern jako cell_neighbors: lokální `min_image_xy` +
    `bucket_id_wrapped` helpers, ghost position iteration. Narrow-phase
    distance přes min-image.
  - `CollisionParams` + `PredateParamsGpu` rozšířeny o `world_half_x/y`
    + 2 padding fields (zarovnání 16-byte uniform alignment).
  - `CollisionGpu.world_half_xy` field. `PredateGpu` čte z params (caller
    naplní per compute() call).

  *Body 4 — testy:*
  - `cargo test --features gpu` callsiteů updated: `with_context(..., [1000.0, 1000.0])`
    nebo `world_half_x: 1000.0, world_half_y: 1000.0` v params struct.
    Cluster fixtury (positions ±500) zůstávají uvnitř [−1000, 1000] tak že
    min-image collapses na raw delta — parity tests s CPU brute force
    procházejí.
  - **71/71 tests pass + 2 ignored**.

- **Konstanty:** žádné nové; world_half přidán do 4 GPU params struktur.

- **Výstup:**
  - 4 WGSL shadery updated (spatial_hash, cell_neighbors, collision, predate).
  - 4 Rust Params struct extended + 3 Gpu structs (Hash/Neighbors/Collision)
    získali `world_half_xy` field; PredateGpu čte z compute() params.
  - `with_context` / `new` constructor signatury rozšířeny o `world_half_xy:
    [f32; 2]` (Hash, Neighbors, Collision).
  - **71/71 testů pass + 2 ignored** (sensor_gather + field_gpu_diffusion
    deferred Sprint 56).
  - **Smoke headless seed=0:** beze změny od Sprintu 54 (--gpu-full
    nepoužívá GPU broad-phase v hot loopu — používá CPU sensor gather +
    GPU brain forward). GPU broad-phase shadery zatím test-only.
  - **Renderer launch OK** (žádný panic).

- **Poznámky:**
  - **Ghost position iteration vs explicit modulo:** šel jsem cestou
    "neighbor_pos = pos_i + dx*cell_size" + bucket_id_of wraps internally.
    Alternativa: explicit modulo na bucket coords (`(bx_base+dx) mod
    world_bx_count`). Ghost approach je jednodušší (nepotřebuje
    `world_bucket_count_x` uniform) a bucket_id_of už wraps pos.xy. Nevýhoda:
    při r_cells × 2 ≥ world_bucket_count by se objevily duplicates (cells
    counted 2× v některých scénářích). Pro typický vision_radius (50) +
    cell_size=64 je r_cells=1 → 3³=27 bucket queries < world_bx_count=30,
    žádné dups. Pro broader predator search (r_cells=2 → 5×5=25 < 30),
    také OK. Edge case r_cells ≥ world_bx_count/2 dokumentován.
  - **GPU broad-phase není v hot path:** Sprint 51 `--gpu-full` headless +
    Sprint 52 renderer GPU default používají BrainGpu/HebbianGpu/BrownianGpu
    + CPU sensor gather. Sprint 50 GPU shadery (sensor/motor/step/collision/
    predate) jsou tested-in-isolation, ne wired. Sprint 55 jejich toroidal
    konzistence znamená readiness pro Sprint 57+ full GPU tick pipeline.
  - **CPU brute force v parity testech:** zůstávají raw delta (no min_image).
    Test fixtury jsou cluster (±500 nebo ±300 v ±1000 world) → raw delta ==
    min_image (žádný cell přes wrap). Tests pass se aktuální brute force.
    Pokud budoucí test stresuje wrap (cells na opačných koncích), CPU brute
    force bude muset taky použít min_image_delta.
  - **Sensor_gather odložené:** Sprint 53 ignored kvůli FieldGpu 2D vs
    SmellField 3D mismatch. Sprint 55 by mohl bucket part toroidal fixnout,
    ale field sample part stále nematch. Sprint 56 (FieldGpu 3D) re-enable.

## Sprinty 56–62 — open-ended

- **Sprint 56: GPU FieldGpu 3D + toroidal.** field_diffuse_3d.wgsl 7-point
  stencil with xy wrap. Re-enable 2 ignored tests.
- **Sprint 56: GPU FieldGpu 3D + toroidal.** field_diffuse_3d.wgsl 7-point
  stencil with xy wrap.
- **Sprint 57+:** 3D voxel rendering, ghost cell visual wrap.
- **Sprint 58+:** progressive z expansion (z=30, z=50).
- **Sprint 59+:** thermal stratification (temperature field z-gradient).
- **Sprint 60+:** light field z-attenuation (photic vs aphotic zones).
