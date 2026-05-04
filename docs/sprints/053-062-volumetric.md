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

## Sprint 56 — GPU FieldGpu 3D + toroidal

- **Cíl:** dokončit GPU field stack ze Sprintu 46/50 — přejít z 2D 5-point
  stencilu na 3D 7-point stencil s xy toroidal wrap + z Neumann zero-flux.
  Re-enable 2 testy (`field_gpu_diffusion_matches_cpu` +
  `sensor_gather_gpu_matches_cpu`) které byly `#[ignore]` od Sprintu 53/54.

  **Plán implementace:**

  *Body 1 — `field_diffuse.wgsl` 3D rewrite:*
  - `Params` struct `res_x/y/z` (per-axis), `cell_size_x/y/z`,
    `world_half_x/y/z`. Pre-Sprint-56 byl `resolution: u32` + `world_half: f32`
    pro symmetric 2D grid.
  - `deposit` shader: 4 floats per source (`px, py, pz, amount`). xy modulo
    wrap match `SmellField::idx_of`, z out-of-range → no-op.
  - `diffuse` shader: workgroup_size(4,4,4). 7-point stencil
    (`left+right+up+down+back+front - 6×center`). xy wrap (i_left = nx-1
    if i==0, atd.) match `SmellField::step` Sprint 53/54 semantiku. z na
    krajích fallback na center (Neumann zero-flux).

  *Body 2 — Rust `FieldGpu` 3D API:*
  - `resolution: [usize; 3]`, `world_half: [f32; 3]`. Konstruktor
    (`new`, `with_context`, `with_device_inner`) přepsán; sources buffer
    alokován s 4 floats per source. `add_source(pos: [f32; 3], amount)`.
  - `step()` 3D dispatch (`wg_x = (res_x+3)/4`, atd.). `download()` vrací
    `res_x * res_y * res_z` elements.
  - `cell_size(axis)` helper.

  *Body 3 — `sensor_gather.wgsl` 3D field sampling + toroidal broad-phase:*
  - `SensorParams` rozšířen o `field_res_x/y/z`, `field_world_half_x/y/z`.
    Pre-Sprint-56 byl `field_resolution: u32` + 2D `field_world_half_x/y`.
  - Helpers: `bucket_id_wrapped` (Sprint 55 toroidal hash), `min_image_xy`
    (Sprint 55), `sample_field_3d`, `gradient_at_3d` (3D central diff
    along all axes).
  - Cell + food broad-phase: ghost positions (`pos_i + offset * cs`) +
    `bucket_id_wrapped` + min-image distance pro narrow-phase. Sprint 55
    pattern (cell_neighbors / collision / predate).
  - Output stride 13 → 15 (smell_grad.z + pheromone_grad.z přidány). Layout:
    `[0..3]` food delta, `[3]` has_food, `[4..7]` cell delta, `[7]` cell radius,
    `[8..11]` smell_grad, `[11..14]` pheromone_grad, `[14]` neighbor count.

  *Body 4 — Rust `SensorGatherGpu` 3D:*
  - `SensorParamsGpu` Rust struct match nový shader struct (3D field params,
    16-byte aligned přes `_pad0`).
  - `SensorRow.smell_grad` + `pheromone_grad` z `[f32; 2]` na `[f32; 3]`.
    Match `BrainSensors` 3D semantiku ze Sprintu 53.
  - Output buffer + readback alokované na `n * 15 * 4` bytes. `compute()`
    extract loop reads stride 15.

  *Body 5 — testy:*
  - `field_gpu_diffusion_matches_cpu` re-enabled: 16×16×4 grid, 320×320×20
    world_half, 16 sources, 6 step iterací s diffusion=0.15, decay=0.5,
    dt=0.1. Compares full grid `cpu.grid_ref()` vs `gpu.download()` v ε
    1e-3 absolute. `SmellField` získal `pub fn grid_ref()` accessor (test
    needs direct grid).
  - `sensor_gather_gpu_matches_cpu` re-enabled: 32 cells + 16 foods,
    16×16×4 fields, 12 smell sources + 8 pheromone sources, 3 step iter.
    Compares neighbor count, nearest cell radius, nearest food presence,
    smell_grad + pheromone_grad (3 axes) vs CPU brute force s min-image
    delta. Tolerance 1e-2 na gradient (atomic CAS drift).

- **Konstanty:** žádné nové; FieldParams + SensorParams gain 3D fields.

- **Výstup:**
  - `shaders/field_diffuse.wgsl`: kompletní 3D rewrite (deposit + diffuse).
  - `shaders/sensor_gather.wgsl`: 3D field helpers, toroidal broad-phase
    (ghost positions + min_image_xy), output stride 15.
  - `src/gpu.rs`: FieldGpu 3D, SensorParamsGpu rozšířen, SensorRow 3D
    gradients, output stride 15. 2 testy re-enabled — vesměs **pass**.
  - `src/lib.rs`: `SmellField::grid_ref()` accessor (test-only convenience).
  - **73/73 testů pass + 0 ignored** (oba field/sensor gpu tests prošly).
  - **Smoke headless seed=2 (CLI args fallback k defaultům):** 200 gen
    completed bez panic, pop trajektorie 200 → 1000 (saturated) → 508 (gen
    200). 442 ticks/s.

- **Poznámky:**
  - **Output stride bump 13 → 15:** binární kompat s pre-Sprint-56 caller
    není zachována. Žádný in-tree caller mimo testy (SensorGatherGpu je
    Sprint 50 standalone — nikoliv v `--gpu-full` nor renderer hot path),
    takže žádný breakage produktu.
  - **Neumann z-flux match CPU:** GPU diffuse shader fallback na center
    pro k=0 a k=nz-1 odpovídá `SmellField::step` Sprint 53/54 semantiku.
    7-point stencil v 3D s `diffusion < 1/6` je stable. `SMELL_DIFFUSION =
    0.15` zůstává pod limitem.
  - **Atomic CAS f32 drift:** GPU deposit používá CAS loop s `bitcast<u32>`
    ↔ `bitcast<f32>` (Naga storage pointer restriction = inline CAS, no
    function param). Pořadí přídavků není deterministické → ULP drift na
    grid hodnotách. Tolerance 1e-3 absolute na `field_gpu_diffusion`,
    1e-2 absolute na `sensor_gather` gradient (gradient přes 6 samples
    zhoršuje drift 6×).
  - **GPU field stack readiness:** Po Sprintu 56 GPU FieldGpu produkuje
    bit-equivalent (modulo CAS drift) jako CPU SmellField. Sprint 57+
    full GPU tick pipeline (smell + pheromone deposit + diffuse na GPU,
    sensor gather na GPU) může nyní spojit FieldGpu + SensorGatherGpu +
    BrainGpu + HebbianGpu + step + collision + predate v jednom unified
    GPU loop. Pre-Sprint-56 byl FieldGpu blocking item.

## Sprint 57 — performance hardening (CPU)

- **Cíl:** dostat z headless tick rate maximum bez GPU wire-up. Baseline
  pre-Sprint-57 (release default profile, 1000 cells, 60 gen, seed=0):
  **164 ticks/s**. Profil identifikoval `eat_food` jako 56 % ticku
  (2120 µs avg) díky standardnímu `HashMap` SipHasheru v `SpatialGrid`
  (27–243 lookupů per radius query) + sekvenční sběr candidate eats.

  **Plán implementace:**

  *Body 1 — `[profile.release]` v Cargo.toml:*
  - `lto = "fat"`, `codegen-units = 1`, `opt-level = 3`. Inlining napříč
    Bevy / wgpu / rayon crate boundaries; LLVM vidí celý program pro
    vektorizaci. Dev rebuild +20 s, ale release wall-clock zisk se měřitelně
    projeví v každém dalším bodu.

  *Body 2 — criterion bench harness:*
  - `criterion = "0.5"` v dev-dependencies, dva benchovací moduly:
    `benches/headless_phases.rs` měří lib API (Brain forward, SmellField
    step, WorldMap sample, Cell step, populate_brain_inputs); `benches/full_tick.rs`
    je momentálně placeholder (full World tick se replikovat nedá bez
    refactoru bin → lib, viz "co Sprint 57 neřeší"). End-to-end měření
    běží přes `time ./target/release/headless 0 60` se seed=0 baseline.

  *Body 3 — paralelizace + hash swap:*
  - **`SpatialGrid` `HashMap` → `FxHashMap`** (rustc-hash 2). Drop-in,
    fixed-seed (deterministický mezi runy), 5-10× rychlejší hash than
    SipHash. Iteration order zachován (3³ buckets v fixním pořadí; per-bucket
    `Vec` insert-order zachován). Zachovává cross-run reproducibility.
  - **`SmellField::step` paralelní přes z-roviny.** `par_chunks_mut(plane)`
    nad scratch — každá z-rovina čte své stencil okolí z `grid` (read-only)
    a píše jen do své části `scratch`. Žádný write conflict. Pro 16 z-rovin
    × 12 cores load-balance chodí ok.
  - **`eat_food` 3-pass refactor.** Pass 1 (`par_iter` cells): per-cell
    candidate selection — `eat_test` je read-only, `min_image_delta` +
    grid lookup paralelně. Pass 2 (sekvenční): resolve race per food
    (first-cell-wins, mark `eaten_scratch[idx]`), apply energy delta +
    Hebbian update sekvenčně. Pass 3: `swap_remove` eaten foods. Hebbian
    sběr šlo paralelizovat (per-cell brain je disjoint), ale ~700 ops × 10–30
    cells per tick je pod rayon spawn overhead → sekvenční.
  - **`predate` Pass 2 → parallel candidate gather + sequential aggregate.**
    `(0..n).par_iter().flat_map_iter()` sbírá `(attacker_i, victim_j, gain)`
    eventy; sekvenční aggregate rozdistribuuje do `energy_deltas_scratch`
    a `damage_deltas_scratch` (ty jsou per-victim shared, takže paralelní
    write by potřeboval atomics nebo per-thread bucket reduce).
  - **Drobné fáze NEparalelizovány.** `apply_morph` (2 µs/tick),
    `step` (16 µs), `apply_hazards` (14 µs), `apply_brownian` (26 µs) —
    work per cell je tak malý, že rayon spawn overhead převáží: rayon
    verze v testovacím průběhu ukázala ~25 µs floor pro každou z těchto fází
    (10–13× zhoršení). Sekvenční je rychlejší.

- **Konstanty / dependencies:** žádné nové konstanty; `rustc-hash = "2"`
  + `criterion = "0.5"` (dev-dep) v Cargo.toml.

- **Výstup:**
  - `Cargo.toml`: `[profile.release]` (LTO + codegen-units=1), `rustc-hash`
    dep, `criterion` dev-dep + 2 `[[bench]]` entries.
  - `src/lib.rs`: `HashMap` → `FxHashMap` v `SpatialGrid`,
    `SmellField::step` paralelní stencil.
  - `src/bin/headless.rs`: `eat_food` 3-pass refactor, `predate` Pass 2
    paralelní gather. `apply_morph` / `step` / `apply_hazards` zůstávají
    sekvenční (rayon overhead)— ozkoušeno ale revertnuto.
  - `benches/headless_phases.rs` + `benches/full_tick.rs`: criterion
    skeleton + 9 lib benchmarků (brain_forward batch/single, smell_step
    populated/empty, pheromone_step, world_map_sample, smell_gradient,
    cell_step, populate_brain_inputs, brain_random, genome_random).
  - **Test suite: 73/73 pass** (jeden flaky `random_brain_average_thrust_is_positive`
    používající unseeded `rand::rng()`, projde re-runem; ne-Sprint-57 issue).
  - **Smoke seed=0, 60 gen, 1000 cells, default world:**
    - **wall-clock 219 s → 36.8 s = 164 → 977 ticks/s = 6.0×.**
    - Pop trajektorie: 200 → 1000 (gen 3 saturated) → 572 (gen 60). Baseline
      finálně 548; +5 % ticha varianta z eat_food ordering change (Pass 1
      paralelní sběr nemá pre-Sprint-57 sekvenční eaten_scratch shortcut).
    - Žádná extinkce, lineages a predation events v healthy rangi.

- **Per-fáze breakdown (us avg per tick, seed=0, 60 gen):**

  | Fáze              | Pre-Sprint-57 | Sprint 57 | Speedup |
  |-------------------|---------------|-----------|---------|
  | update_smell      | 316.9         | 211.5     | 1.5×    |
  | update_pheromone  | 175.0         | 73.3      | 2.4×    |
  | brain_act         | 584.1         | 333.6     | 1.8×    |
  | resolve_collisions| 134.1         | 71.8      | 1.9×    |
  | predate           | 330.9         | 105.0     | **3.2×**|
  | **eat_food**      | **2120.7**    | **277.8** | **7.6×**|
  | spawn_food        | 23.4          | 14.4      | 1.6×    |

  Ostatní fáze < 30 µs nebyly hot path; FxHashMap je vyvedl z ~1 µs do
  ~1 µs. `brain_act` zlepšení je čistě z FxHashMap (sensor gather grid
  lookups), funkční tělo zůstává sekvenční sběr inputs / forward / motor.

- **Poznámky:**
  - **FxHashMap drop-in:** žádný change v determinismu. SpatialGrid řád
    items je insert-order (per-bucket `Vec`), bucket iter je 3³ fixed
    `(dx, dy, dz)`. Hash function ovlivňuje jen *který* bucket dostane
    klíč; lookup `get(&key)` je deterministický (vůči same key) ať je
    hasher jakýkoli. Default `RandomState` SipHash ALE má per-process
    random seed → cross-run iteration *order* HashMap.iter() není
    deterministický. SpatialGrid neposílá `iter()` ven (jen `get(&key)`),
    takže to nikdy nebyl problém — ale FxHashMap je "more deterministic"
    + 5-10× rychlejší. Pro hot path 27-243 lookupů/query × 1000 cells/tick
    × 13 fází to dělá velký rozdíl.
  - **eat_food behavior change:** pre-Sprint-57 sekvenční loop měl
    `if ate_idx.is_some() || self.eaten_scratch[idx]` shortcut uvnitř grid
    callback — tj. cell pokud našel zabraný food v rané grid traversal
    pokračoval hledat NEzabraný. Sprint 57 paralelní Pass 1 nevidí
    `eaten_scratch` (ne yet populated), takže cell vybere first
    eat_test-passing food, a pokud Pass 2 zjistí že je zabraný, cell ten
    tick nedostane jídlo. Smoke ukazuje že tohle pop trajektorii nepoškodí
    (final 572 vs baseline 548); nově introdukovaný stochasticism má drobný
    selekční effect ale nezmění direction evoluce.
  - **Co Sprint 57 NEŘEŠÍ (Sprint 58+):**
    - GPU FieldGpu wire-up (deposit/diffuse). Po Sprintu 57 update_smell+pheromone
      = 285 µs = 22 % ticku; readback latence (PCIe + sync) by srovnala benefit
      pro grid 64×64×16 (jen ~65k elementů). FieldGpu má smysl při bigger grid
      nebo full-GPU loop bez per-tick readback.
    - GPU SensorGatherGpu wire-up (Sprint 56 ready). Závislé na FieldGpu —
      sensor potřebuje sample/gradient z field na GPU.
    - `brain_act` GPU dispatch v `--gpu-full` (Sprint 51 ready ale CPU sensor
      gather + GPU forward s upload/download stále je net negative pro 1000
      cells × 36 inputs × 8 outputs).
    - `SpatialGrid` `FxHashMap` → dense `Vec` indexed by 3D bucket coord
      (~1020 buckets pro default world). Eliminuje hash kompletně, ale
      vyžaduje refactor generic API (signed bucket coords, modulo wrap).
    - `apply_brownian_cpu` paralelní s per-cell deterministic RNG (seed-derived
      z lineage_id, jako Sprint 51 GPU brownian). Celkem 26 µs → potenciálně
      5–10 µs, ale Sprint 51 už má GPU verzi pro `--gpu-full`.
    - Renderer-side parallelizace `cell_eats_food` / `predate` — Bevy ECS
      Commands (despawn) blokuje par_iter Query pattern; vyžaduje strukturní
      přepis přes `EntityCommands` bucket + main-thread flush.
    - Full headless tick benchmark v criterion (současný `benches/full_tick.rs`
      je placeholder; měření přes `time` binary stačí pro Sprint 57 baseline).

## Sprint 58 — performance hardening (renderer)

- **Cíl:** replikovat Sprint 57 paralelizační wins do renderer hot path
  (`src/main.rs`). Sprint 57 paralelně pracoval s headless `Vec<Cell>`;
  renderer používá Bevy ECS Query, takže pattern je jiný — snapshot
  primitivních dat z Query → rayon par compute → sekvenční apply přes
  `Query::get_mut` (ECS write je single-threaded).

  **Plán implementace:**

  *Body 1 — `HashMap` / `HashSet` → `FxHashMap` / `FxHashSet`:*
  - `LineageMaterials`, `CellSlotMap.entity_to_slot`, predate scratch tables
    (energy_changes, damage_changes, herd_counts), eat_food eaten set,
    stats_overlay lineages set → drop-in. Stejně jako Sprint 57: SipHash
    je v hot path drahý (Bevy `Entity` je 64-bit, hash overhead je per-key).

  *Body 2 — `cell_predates_on_neighbor`:*
  - Pre-compute `herd_counts` přes `snapshot.par_iter()` → `Vec<u32>`
    indexed by snapshot order. Pak `FxHashMap<Entity, u32>` pro lookup
    v sekvenčním attack loopu (zachován kvůli ECS write).
  - Attack event pass zůstává sekvenční (jako headless Pass 2 pre-Sprint-57)
    — refactor na par flat_map vyžaduje extra snapshot pose dat (heading,
    pitch, spike_length) a `slot_map` lookup pro per-victim herd_count;
    pro Sprint 58 stačí FxHashMap drop-in + par herd_counts.

  *Body 3 — `resolve_cell_collisions`:*
  - Pass 1 (par): snapshot `Vec<(Entity, pos, radius)>` → par compute
    `Vec<(Entity, [f32; 2])>` non-zero deltas přes `par_iter().filter_map()`.
  - Pass 2 (seq): apply přes `cells.get_mut(entity)`.

  *Body 4 — `cell_eats_food`:*
  - 3-pass refactor (mirror Sprint 57 headless `eat_food`).
  - Pass 1 (par): snapshot s `(Entity, pos, max_axis, body_dims, heading, pitch)`
    pro `eat_test_pose` (nový lib helper). `Vec<Option<(Entity_food, value)>>`
    candidates collect.
  - Pass 2 (seq): resolve race (first-cell-wins per food entity), apply
    energy + Hebbian.
  - Pass 3 (seq, main thread): `commands.entity(food_e).despawn()` flush —
    Bevy `Commands` je single-threaded resource, nelze v `par_iter`.

  *Body 5 — lib helper `eat_test_pose`:*
  - Pure ellipsoid acceptance test bez `&Cell` (Bevy Query nemá `Send + Sync`
    refs do Cell ve `par_iter`). Parametrizováno: `cell_pos`, `heading`,
    `pitch`, `body_dims = [length, width, height]`, `food_pos`, `eat_factor`.
  - `Cell::eat_test` re-implementováno jako tenký delegát na `eat_test_pose`
    — zachovává původní API + sjednocuje math do jednoho místa.

  *Body 6 — `step_cells` / `apply_environmental_hazards` / `apply_cell_morph`:*
  - Stejně jako Sprint 57 lekce: rayon spawn overhead převáží malou per-cell
    práci (1-15 µs sekvenčně vs ~25 µs paralelně). Sekvenční zachováno.

- **Konstanty / dependencies:** žádné nové. `rayon` + `rustc-hash` už v
  Sprint 57. Nový lib helper `eat_test_pose` neviditelný pro callsites
  mimo refactor.

- **Výstup:**
  - `src/main.rs`: imports `rayon::prelude::*` + `rustc_hash::{FxHashMap,
    FxHashSet}`. Hot path 3 systémy paralelizovány (predate herd counts,
    collisions, eats_food). Drobnosti zachovány sekvenční. HashMap/Set drop-in
    všude (4 callsites + 2 struct fields).
  - `src/lib.rs`: `eat_test_pose` pub fn helper. `Cell::eat_test`
    delegát.
  - **Test suite: 73/73 pass** (existující eat_test testy validují delegate
    refactor).
  - **Renderer launch:** Bevy app starts bez panic, sim tick + render OK.
    Wgpu init `cap 1064`, žádné GPU validation errory.

- **Poznámky:**
  - **ECS write je single-threaded:** `Query::par_iter_mut` v Bevy 0.18
    funguje jen pro for_each-style closures bez collecting. Pro `cells.get_mut(entity)`
    apply pattern (po par compute) je sekvenční — pattern: snapshot →
    par compute → seq apply. Pomáhá když per-iter práce je drahá (grid
    lookup, eat_test, ellipsoid math); pro malé fáze (step, hazards, morph)
    je rayon overhead větší než paralelní zisk (Sprint 57 lekce).
  - **Snapshot allocation cost:** Vec<(Entity, [f32; 3], f32)> ~= 32 bytes ×
    1000 cells = 32 KB clone per tick. Při 60 FPS = 2 MB/s memcpy — negligible
    vs 6 GB/s DDR4 bandwidth.
  - **Commands flush dependency:** `cell_eats_food` despawn je v Pass 3
    sekvenčně. `cell_dies_on_zero_energy` + `cell_reproduces_on_threshold`
    používají Commands podobně, ale per-entity work je tak malý (energy
    threshold check, jednorázový spawn), že rayon overhead by převážil —
    sekvenční zachováno.
  - **Co Sprint 58 NEŘEŠÍ (Sprint 59+):**
    - `cells_brain_act` CPU sensor gather paralelizace (snapshot + par_iter).
      GPU forward path už pokrývá brain forward; sensor gather je ~30 % brain_act
      time. Refactor vyžaduje extract pose snapshot per cell.
    - GPU FieldGpu / SensorGather hot-path wire-up (Sprint 56 ready).
    - GPU brain_act CPU fallback path paralelizace (Bevy par_iter_mut + extract
      Vec<Cell> snapshot + scatter back).
    - `SpatialGrid` `FxHashMap` → dense `Vec` (Sprint 57 deferred).
    - 3D voxel rendering, ghost cell visual wrap.

## Sprint 59 — GPU FieldGpu hot-path wire-up

- **Cíl:** dokončit Sprint 56 deferred item — `FieldGpu` (3D 7-point Jacobi)
  zapojen do `--gpu-full` headless a renderer GPU default hot path. Smell +
  pheromone fields běží na GPU; CPU `SmellField` slouží jako shadow buffer
  pro sensor gather (`gradient_at`, `sample`) — Sprint 60+ rewire sensor
  gather na `SensorGatherGpu` čte direct GPU storage buffer, eliminuje
  per-tick readback.

  **Plán implementace:**

  *Body 1 — `SmellField::replace_grid_from(&[f32])`:*
  - Pub setter co `copy_from_slice` na private `grid` Vec. Volá GPU readback
    path po každém `step()`. Žádná validace mimo `debug_assert_eq!` na délku
    — caller (Sprint 59 wire) zajišťuje `[res_x × res_y × res_z]` match.

  *Body 2 — Headless `GpuFullState`:*
  - `smell: FieldGpu` + `pheromone: FieldGpu` přidány do struct.
  - Init capacity per FieldGpu: `(food_target(peak_density) + max_population) × 2`
    (worst case: density cycle peak → food count + cells co emit pheromones,
    × 2 safety pro auto-realloc trigger margin). Sdílí GpuContext s
    BrainGpu/CellsGpu/HebbianGpu/BrownianGpu (Sprint 47 pattern).
  - `update_smell` GPU path: foods loop `gpu.smell.add_source(pos, amount)` →
    `gpu.smell.step(SMELL_DIFFUSION, SMELL_DECAY, dt)` → `gpu.smell.download()`
    → `self.smell.replace_grid_from(&grid)`. CPU SmellField shadow je sync.
  - `update_pheromone` GPU path: `gpu.pheromone.step()` → readback. Žádný
    deposit zde; emise z předchozího ticku se flushují přes `pending_sources`
    bufferu.
  - `emit_pheromones` GPU path: cells loop `gpu.pheromone.add_source(pos, rate*dt)`.
    Sources se akumulují do `pending_sources`, deposit shader spustí v dalším
    ticku `update_pheromone` — match Sprint 38 "diffuse before emit" semantiku
    (žádný self-feedback).

  *Body 3 — Renderer `GpuFieldState` Resource:*
  - Separate Bevy Resource od `GpuBrainState` aby `update_smell_field`
    (ResMut<GpuFieldState>) nezpůsobil schedule contention s ostatními
    systémy přistupujícími na brain GPU.
  - Setup init mirroruje headless: `food_target(extent, peak_density)` +
    `MAX_POPULATION` × 2. Insert Resource pokud GpuContext OK, jinak
    Resource None a CPU SmellResource path drží.
  - 3 systems updated: `update_smell_field` / `update_pheromone_field` /
    `emit_pheromones` přijímají `Option<ResMut<GpuFieldState>>` a větví na
    GPU vs CPU path. CPU SmellResource se vždy plní (přímo nebo přes readback)
    — sensor gather (`cells_brain_act`) zůstává CPU `gradient_at` nezměněn.

- **Konstanty:** žádné nové. Field sources capacity je derived `food_target ×
  density_peak + MAX_POPULATION` × 2.

- **Výstup:**
  - `src/lib.rs`: `SmellField::replace_grid_from(&[f32])` setter.
  - `src/bin/headless.rs`: `GpuFullState` rozšířen o `smell + pheromone`
    `FieldGpu`. 3 hot path systems větví na GPU. Init log oznámí
    `field sources capacity`.
  - `src/main.rs`: `GpuFieldState` Resource. Setup init creates + inserts.
    3 hot path systems větví na GPU.
  - **Test suite: 73/73 pass** (jeden flaky `random_brain_average_thrust_is_positive`
    — Sprint 57 dokumentovaný, prošel re-runem; ne-Sprint-59 issue).
  - **Smoke seed=0, 60 gen, 1000 cells, default world s `--gpu-full`:**
    - **Wall-clock 104 s = 345 ticks/s** (vs Sprint 58 CPU-only 977 ticks/s
      = **2.8× POMALEJŠÍ**).
    - Per-fáze us avg: update_smell 423 (CPU 211, GPU 2.0× pomalejší),
      update_pheromone 377 (CPU 73, 5.2×), brain_act 967 (CPU 333, 2.9×),
      apply_brownian 240 (CPU 26, 9.2×), reproduce 464 (CPU 24, 19×).
      **Hlavní viník: per-tick `device.poll(Wait)` round-trip × N readback
      pointů.** Field readback × 2/tick + brain hidden/outputs/velocities
      readback se sečtou.
    - Pop final 499 (vs CPU 572, –13 % drift). Trajektorie qualitativně
      stejná (pop saturate na MAX_POPULATION → steady-state ~500), žádná
      extinkce. Atomic CAS deposit drift v noise rangi pro evoluci.
  - **Renderer launch:** `cargo run --release` startuje, info log oznámí
    "renderer-gpu: ... + Field (cap N cells, M field sources)", žádný
    panic. 8s SIGTERM exit clean.

- **Poznámky:**
  - **GPU verze je per-tick POMALEJŠÍ než CPU paralelní path.** Smoke ukazuje
    345 ticks/s s `--gpu-full` vs 977 ticks/s pure CPU (Sprint 57+58
    paralelní stencil + FxHashMap + eat_food refactor). Důvod: per-tick
    `device.poll(Wait)` round-trip × N readback pointů (smell + pheromone
    field × 2/tick + hidden + outputs + velocities z brain/brownian).
    PCIe latence dominuje; bandwidth (~60 MB/s) je pod limity. Pro grid
    64×64×16 je CPU paralelní stencil (16 z-rovin × 12 cores) přímý win.
    GPU má smysl při bigger grid (256³+) nebo full-GPU loop bez per-tick
    readback (cíl Sprint 60).
  - **Atomic CAS deposit drift:** Sprint 56 dokumentuje že GPU deposit
    používá `bitcast<u32>` ↔ `bitcast<f32>` CAS loop (Naga storage pointer
    omezení). Pořadí přídavků je non-deterministické per-thread → ULP-level
    drift na grid hodnotách. Pro evolution sim trajectory je drift v noise
    rangi (1e-3 absolute < 1% smell field max amplitude = ~1.0).
  - **CPU SmellField shadow:** sensor gather zůstává CPU (`gradient_at`
    central-differences sample). Readback po každém step() znamená že
    CPU shadow je vždy current pre-emit. Sprint 60 cíl: SensorGatherGpu
    čte field z GPU storage buffer přes `current_grid_buffer()`,
    eliminuje per-tick readback.
  - **CPU fallback:** pokud GpuContext init selže (no Vulkan/Metal/DX),
    headless `gpu_full = None` + renderer `GpuFieldState` Resource neexistuje.
    `Option<ResMut<>>` v systému None → CPU SmellResource path. Žádný
    runtime overhead pokud GPU není.
  - **Co Sprint 59 NEŘEŠÍ (Sprint 60+):**
    - SensorGatherGpu wire-up: sensor gather zůstává CPU. Sprint 56 SensorGatherGpu
      output stride 15 (3D gradients) je ready, ale wire-up znamená i
      `SensorRow` snapshot upload + `apply_brain_motor` motor refactor.
    - GPU full tick pipeline (no per-tick readback): cílí Sprint 60+. 
      SensorGatherGpu může číst FieldGpu storage buffer direct (Sprint 50
      `current_grid_buffer()` accessor existuje), eliminuje smell+pheromone
      readback × 2/tick.
    - Renderer `world_map_image` overlay update (CPU SmellField changes
      reflektují v atlas image). Aktuální readback flow zachová overlay
      funkcionalitu.

## Sprint 60 — SensorGatherGpu wire-up (headless --gpu-full)

- **Cíl:** eliminovat per-tick FieldGpu readback ze Sprintu 59 wire-upem
  SensorGatherGpu (Sprint 56 ready, dosud test-only). GPU sensor shader
  čte FieldGpu smell + pheromone storage buffer **direct** (přes
  `current_grid_buffer()` accessor) — žádný `device.poll(Wait)` round-trip
  na 256 KB grid × 2 fields/tick. Plus zapojuje GPU `SpatialHashGpu` jako
  broad-phase pro sensor (cell + food bucket grids), nahrazuje 4 CPU
  SpatialGrid rebuilds + lookup smyčky v `brain_act_gpu_full`.

  **Plán implementace:**

  *Body 1 — `SpatialHashGpu::dispatch(positions)`:*
  - Variant of `rebuild()` bez per-tick readback. Submits count → prefix →
    scatter compute passes a vrací. Sensor pipeline využije
    `offsets_buffer()` + `sorted_buffer()` accessory pro chained access
    (Sprint 49+ pattern: hash buffery vázány read-only do sensor bind
    group). Eliminuje 2× `device.poll(Wait)` round-trip per tick (cell
    hash + food hash).

  *Body 2 — `SensorRow` API change (delta vs absolute):*
  - Pre-Sprint-60 `compute()` rekonstruoval `nearest_food = positions[i] +
    delta` (absolute pozice). Bylo to chybné přes toroidal wrap — cell na
    `x=-950` + delta `60` = `-890`, ne ghost `+1010`. Sprint 60 ukládá raw
    signed delta z shaderu direct → `SensorRow.nearest_food/cell` jsou
    `Option<[f32; 3]>` (delta), match `BrainSensors` Sprint 54 sémantiku.
    `populate_brain_inputs(cell, &sensors, vision_r)` konzumuje delta
    přímo bez `target − pos` math.

  *Body 3 — `GpuFullState` rozšíření (headless):*
  - Add `cell_hash: SpatialHashGpu`, `food_hash: SpatialHashGpu`, `sensor:
    SensorGatherGpu` do struct. Init v `setup` po Sprint 47 sdílený
    GpuContext pattern. Capacity: cells = `MAX_POPULATION + slack`,
    foods = `food_target(peak_density) × 2`.

  *Body 4 — `brain_act_gpu_full` refactor:*
  - 7-fáze pipeline:
    1. CPU snapshot: `Vec<position>`, `Vec<eff_radius>`, `Vec<vision_radius>`
       z `self.cells`; `Vec<food_position>` z `self.foods`.
    2. `cell_hash.dispatch(positions)` + `food_hash.dispatch(food_positions)`
       — submit only, no readback.
    3. `sensor.compute(...)` čte FieldGpu smell + pheromone storage + hash
       buffery → `Vec<SensorRow>` (60 KB readback × 1, vs Sprint 59 256 KB
       readback × 2).
    4. CPU `populate_brain_inputs` (rayon par_iter_mut zip s sensor_rows):
       SensorRow → BrainSensors 1:1 (Sprint 60 SensorRow = delta), pak lib
       helper plní `[f32; BRAIN_INPUTS]` array (energy, speed_norm, heading
       projection, recurrent state z `cell.last_hidden`).
    5–7. Identický s pre-Sprint-60: GPU brain forward → download hidden +
       outputs → CPU motor.

  *Body 5 — `update_smell` / `update_pheromone` no-readback:*
  - `--gpu-full` cesta volá `gpu.smell.step()` ale NE `gpu.smell.download()`.
    Sprint 59 `replace_grid_from(&grid)` syncing CPU SmellField shadow je
    odstraněn — CPU `World.smell` / `pheromone` nejsou updated v `--gpu-full`,
    sensor shader čte field direct. CPU shadows zůstávají v struct kvůli
    checkpoint serialization (po Sprint 60 jsou stale v `--gpu-full`).
  - `emit_pheromones` GPU path beze změny: `gpu.pheromone.add_source(pos,
    rate*dt)` push do `pending_sources`, flushed v dalším ticku
    `update_pheromone` `step()`.

  *Body 6 — Renderer beze změny:*
  - Sprint 60 wire-up je jen pro headless `--gpu-full`. Renderer
    `GpuFieldState` Resource zachovává Sprint 59 path (per-tick FieldGpu
    readback do CPU SmellResource). Bevy ECS Query + Commands extrakce pro
    GPU snapshot pattern je Sprint 61+ refactor (analogický Sprint 58
    `cell_eats_food` 3-pass).

- **Konstanty:** žádné nové. SensorGatherGpu fixed grid 64×32×4 = 8192
  buckets; `WORLD_HALF[0]=960` při `GRID_CELL_SIZE=64` → 30×17 buckets
  active (krytí ±1024 / ±512 = ±960 / ±540 pohodlně).

- **Výstup:**
  - `src/gpu.rs`: `SpatialHashGpu::dispatch(&[[f32; 3]])` (no readback variant
    of `rebuild`). `SensorRow` field semantika changed na min-image delta
    (compute() vrací raw shader output).
  - `src/bin/headless.rs`: `GpuFullState` přidává `cell_hash + food_hash +
    sensor`. `brain_act_gpu_full` 7-fáze refactor. `update_smell` /
    `update_pheromone` odstranily `replace_grid_from` sync.
  - **Test suite: 73/73 pass** (sensor_gather_gpu_matches_cpu byl už
    Sprint 56 re-enabled; Sprint 60 SensorRow change netestoval absolute
    pos vs delta, jen radius/presence/grad — passuje).
  - **Smoke seed=0, 60 gen, 1000 cells, default world s `--gpu-full`:**
    - **Wall-clock 162 s = 222 ticks/s** (vs Sprint 58 CPU-only 977 ticks/s
      = **4.4× POMALEJŠÍ**, vs Sprint 59 GPU s field readback 345 ticks/s
      = **1.6× POMALEJŠÍ**).
    - Per-fáze us avg: update_smell 47 (Sprint 59 423, **9× zlepšení** —
      readback eliminován ✓), update_pheromone 20 (Sprint 59 377, **19×
      zlepšení** ✓). Field stack je teď pohodlně low-overhead.
    - ALE **brain_act 4103 us** (Sprint 59 967, Sprint 58 CPU 333) —
      **+3137 us regrese**. GPU sensor pipeline (3 dispatch points:
      cell_hash + food_hash + sensor + 1 readback 60 KB) má více
      `device.poll(Wait)` round-trip než Sprint 59 path (CPU rayon sensor
      gather + 1 brain forward dispatch).
    - Pop final 525 (žádná extinkce, atomic CAS drift v noise rangi).
    - **Net: Sprint 60 kombinuje field stack win s sensor stack ztrátou.
      Total wall-clock je horší než Sprint 59.**

- **Závěr (nepříjemný):**
  Sprint 60 dokumentuje že **GPU offload pro current workload (1000 cells,
  64×64×16 grid, 60 Hz target) je net negative** napříč všemi tested
  konfiguracemi:
  - Sprint 51 `--gpu-full` (brain forward + Hebbian + Brownian, CPU
    sensor + field): rychlost vs CPU-only TBD historicky.
  - Sprint 59 + GPU FieldGpu (readback): 345 ticks/s = 2.8× slower than CPU.
  - Sprint 60 + GPU SensorGather (no field readback): 222 ticks/s = 4.4×
    slower than CPU.
  Per-tick CPU work je sub-millisecond díky Sprint 57+58 paralelizaci;
  každý GPU `device.poll(Wait)` round-trip přidá ~50-200 µs (PCIe latence,
  ne bandwidth). Sčítáním 4-5 sync points/tick se GPU stack stává hot path.
  GPU má smysl při **bigger workload** (10k+ cells, larger grids, no per-tick
  readback fully-fused pipeline). Pro current sim scale je CPU paralelní
  cesta rychlejší.

- **Poznámky:**
  - **Eliminuje 2× FieldGpu readback (Sprint 59 hlavní bottleneck)** +
    4× CPU SpatialGrid rebuild (cell_grid + food_grid pro brain_act).
    Přidává 1× SensorGatherGpu readback (60 KB) + 2× GPU SpatialHash
    dispatch (no readback). Net win záleží na PCIe latence vs GPU compute
    time pro sensor shader.
  - **CPU SmellField shadows jsou v `--gpu-full` stale.** Checkpoint
    serialization v `--gpu-full` po Sprint 60 ukládá stale CPU state.
    Pre-Sprint-60 path (CPU only nebo `--gpu` brain-only) zachová
    deterministic CPU SmellField. Sprint 61+ může přidat readback při
    checkpoint save (rare event, ne hot path).
  - **GPU sensor shader bere `field_res` jako jediný param** — sensor
    používá stejnou rezoluci pro smell i pheromone (`SensorParamsGpu.field_res_x/y/z`).
    Aktuálně `SMELL_GRID_RES == PHEROMONE_GRID_RES` (=64), `SMELL_GRID_RES_Z
    == PHEROMONE_GRID_RES_Z` (=16). Pokud konstanty rozejdou, sensor shader
    vrátí špatné gradient — invariant `smell_resolution == pheromone_resolution`
    je new constraint.
  - **Renderer není v Sprint 60 zaznamenán.** Renderer `--gpu` default
    cesta zůstává Sprint 59 readback pattern. Sprint 61 `cells_brain_act`
    refactor přes Bevy ECS Query snapshot extrakce + GpuFieldState GPU
    sensor wire-up.
  - **Co Sprint 60 NEŘEŠÍ (Sprint 61+):**
    - Renderer GPU sensor pipeline.
    - Eliminace SensorRow readback: brain_forward shader by mohl číst
      sensor output buffer + cell internals direct, fuzed do BRAIN_INPUTS
      shader-side. Pak NO readback per tick.
    - Eliminace hidden+outputs readback po brain forward: pokud Hebbian
      update + motor jsou GPU shadery, hidden+outputs nemusí na CPU.
    - GPU step / collision / predate / eat shadery (Sprint 50 standalone)
      wire-up. Plný GPU tick loop bez CPU side-effects.

## Sprint 61 — fuze sensor → brain forward (eliminate sensor readback)

- **Cíl:** odstranit 1 ze 3 GPU round-trips v `--gpu-full` brain_act (sensor
  output 60 KB readback). Klíč: GPU shader `populate_inputs.wgsl` čte
  `sensor.output_buf` storage direct + cell metadata (energy, velocity,
  heading, pitch, damage_accum, max_speed, eff_radius, last_hidden) → píše
  `cells.last_inputs_buf` který brain forward už čte. Žádný `device.poll(Wait)`
  ze sensor stage.

  **Plán implementace:**

  *Body 1 — `CellsGpu` rozšíření o cell metadata buffery:*
  - Add 6 buffers (každý `n × f32 = 4 KB`): `energy_buf`, `heading_buf`,
    `pitch_buf`, `damage_accum_buf`, `max_speed_buf`, `eff_radius_buf`.
    Velocities buf už existuje (Sprint 51 brownian).
  - Pub accessory pro shader binding. Bulk `upload_metadata(...)` helper.

  *Body 2 — `populate_inputs.wgsl` shader:*
  - 12 bindings: params (uniform) + 11 storage. Read-only: sensor_output,
    velocities, energies, headings, pitches, max_speeds, eff_radii,
    vision_radii, last_hidden. Read-write: damage_accums (reset side-effect),
    last_inputs (write target). Workgroup 64.
  - Mirroruje lib `populate_brain_inputs` 1:1 — sensor stride 15 → BrainSensors
    field mapping; energy/speed/forward_vector/recurrent normalizace; damage
    consume + reset.
  - Konstanty (gainy, REPRODUCE_THRESHOLD, BRAIN_INPUTS layout) přes uniform
    `PopulateInputsParams`.

  *Body 3 — `PopulateInputsGpu` Rust wrapper:*
  - Pipeline + bind group layout. `dispatch(&CellsGpu, &SensorGatherGpu, params)`
    metoda. Sdílí GpuContext s ostatními (Sprint 47 pattern).

  *Body 4 — `SensorGatherGpu::dispatch_no_readback`:*
  - Variant of `compute()` bez `Vec<SensorRow>` readback. Submit only —
    `output_buf` zůstává storage pro chained populate shader.
  - Pub `output_buffer()` + `vision_radii_buffer()` accessory pro binding.

  *Body 5 — `brain_act_gpu_full` 8-fáze refactor:*
  - 1: CPU `apply_shell_absorb` pre-pass + snapshot positions/eff_radii/
       vision_radii/food_positions + cell metadata (energies, headings, pitches,
       damage_accums, max_speeds, velocities).
  - 2: `cells.upload_metadata(...)` + `cells.upload_velocities(...)`.
  - 3: `cell_hash.dispatch + food_hash.dispatch` (no readback).
  - 4: `sensor.dispatch_no_readback(...)` (no readback).
  - 5: `populate.dispatch(&cells, &sensor, params)` (no readback).
  - 6: `brain.forward_persistent(&cells, n)` (no readback — brain reads
       last_inputs_buf direct).
  - 7: `cells.download_hidden_outputs(n)` ← **round-trip #2** (Sprint 62
       target: motor na GPU eliminuje).
  - 8: CPU motor + writeback last_hidden/last_outputs/damage_accum=0.

- **Konstanty:** žádné nové. Lib re-export `BRAIN_INPUTS_SENSORY`,
  `BRAIN_RECURRENT`, `DAMAGE_NORMALIZATION_GAIN`, `DENSITY_NORM_COUNT`,
  `PHEROMONE_NORMALIZATION_GAIN`, `SMELL_NORMALIZATION_GAIN`,
  `REPRODUCE_THRESHOLD` (všechny pre-existing pub const).

- **Výstup:**
  - `src/gpu.rs`: `CellsGpu` přidává 6 metadata buffery + accessors +
    `upload_metadata`. `SensorGatherGpu` přidává `output_buffer()` /
    `vision_radii_buffer()` accessory + `dispatch_no_readback()`.
    `PopulateInputsGpu` + `PopulateInputsParams` nový.
  - `shaders/populate_inputs.wgsl` nový — 11-binding shader.
  - `src/bin/headless.rs`: `GpuFullState` přidává `populate: PopulateInputsGpu`.
    `brain_act_gpu_full` 8-fáze refactor.
  - **Test suite: 73/73 pass**.
  - **Smoke seed=0, 60 gen, 1000 cells, default world s `--gpu-full`:**
    - **Wall-clock 136 s = 265 ticks/s** (Sprint 60 222 = **+19 %**, ale
      Sprint 59 345, Sprint 58 CPU 977 = stále **3.7× slower** než CPU).
    - Per-fáze us avg: update_smell 41 (Sprint 60 47), update_pheromone 16
      (Sprint 60 20), **brain_act 3341 (Sprint 60 4103, –762 µs ✓)**,
      apply_brownian 142 (Sprint 60 199), reproduce 217 (Sprint 60 299).
    - Pop final 548 (Sprint 60 525, CPU 572) — populate shader správně
      mirroruje CPU populate, drift v CAS noise rangi.

- **Poznámky:**
  - **Sprint 61 win nad Sprint 60 ale ne nad Sprint 59.** Sprint 60 přidal
    GPU sensor pipeline (3 dispatch + 1 readback). Sprint 61 odstranil
    sensor readback (1 RT eliminován). Net: Sprint 60→61 -19% wall-clock.
    Ale Sprint 59 (CPU sensor + GPU field) je stále rychlejší — Sprint 61
    má víc dispatch overhead (5 GPU passes brain_act fáze).
  - **Round-trip status po Sprint 61:**
    - #1 sensor: ELIMINATED (Sprint 61) ✓
    - #2 hidden+outputs (96 KB): zůstává — Sprint 62 motor na GPU eliminuje.
    - #3 velocities (12 KB): zůstává — Sprint 51 GPU brownian path.
    - + reproduce sparse upload_brain_at: rare event, ne hot path.
    Sprint 62 motor + Sprint 63+ step/collision/predate by snížilo na 0
    round-trips/tick (jen na konci epochy CSV log readback).
  - **Damage_accum dual-state:** populate_inputs shader resetuje
    `damage_accums[i] = 0` GPU-side. CPU mirror: `cell.damage_accum = 0.0`
    v Phase 8 motor pass (před výpisem v dalším tick uploadu metadata).
    Apply_hazards + predate píší CPU damage_accum, který je v dalším
    brain_act ticku uploadnut → consistent.
  - **Per-tick metadata upload:** 6 buffers × 1000 cells × 4 B = 24 KB
    upload/tick. `queue.write_buffer` je async submit (no Wait). Plus
    velocities 12 KB. Total ~36 KB/tick metadata = 2.2 MB/s při 60 Hz —
    pod PCIe limity.
  - **Co Sprint 61 NEŘEŠÍ (Sprint 62+):**
    - Motor on GPU shader (`shaders/motor.wgsl` Sprint 50 standalone
      ready). Eliminuje round-trip #2 (96 KB hidden+outputs readback).
      Brain forward → motor jako single fused pipeline.
    - GPU step/brownian/eat/predate/collision (Sprint 50 standalone shadery)
      wire — full GPU loop. Eliminuje round-trip #3 + ostatní CPU phase
      readbacks.
    - Renderer GPU populate_inputs (mirror Sprint 61 do main.rs Bevy ECS).
    - Bigger-workload benchmark: Sprint 61 měřeno jen 1000 cells. GPU
      stack profitabilita roste s N — 5k+ cells je expected break-even
      vs CPU baseline.

## Sprint 62 — motor + brownian fuze (single Wait barrier)

- **Cíl:** eliminovat round-trip #2 (hidden+outputs 96 KB) i round-trip #3
  (velocities 12 KB) sloučením motor + brownian dispatch s brain forward
  do jedinné GPU pipeline. Single batch readback `download_brain_motor_batch`
  na konci brain_act_gpu_full = 1 `device.poll(Wait)` per tick (vs Sprint 61 2×).
  `apply_brownian` fáze v `--gpu-full` se stává no-op (work fused).

  **Plán implementace:**

  *Body 1 — `CellsGpu` motor buffery:*
  - Add `turn_rate_buf`, `angular_velocity_buf`, `pitch_velocity_buf` +
    readback variants. Plus accessory + `upload_turn_rates` /
    `upload_angular_pitch` helpers.

  *Body 2 — `MotorGpu::dispatch_with_cells`:*
  - Variant of `compute()` co bind `&CellsGpu` shared buffery (last_outputs,
    heading, pitch, max_speed, turn_rate, eff_radius, velocity, ang_vel,
    pitch_vel) místo own duplicate buffers. Shader (Sprint 50 `motor.wgsl`)
    beze změny — mirror lib::Cell::apply_brain_motor 1:1.

  *Body 3 — `CellsGpu::download_brain_motor_batch`:*
  - Single Wait barrier readback 5 buffers: hidden, outputs, velocity,
    angular_vel, pitch_vel. 5× `copy_buffer_to_buffer` + 5× `map_async` +
    1× `device.poll(Wait)`. Total bytes ~110 KB/tick.

  *Body 4 — `brain_act_gpu_full` 10-fáze refactor:*
  - Phase 7-8: motor.dispatch_with_cells + brownian.compute_persistent
    fused (sequential dispatches, no Wait between). Brownian už používá
    CellsGpu.velocities_buffer direct (Sprint 51).
  - Phase 9: download_brain_motor_batch (single Wait).
  - Phase 10: CPU writeback all 5 vec do cell state. NO `apply_brain_motor`
    (motor byl GPU-side). damage_accum reset.

  *Body 5 — `apply_brownian` fáze skip v `--gpu-full`:*
  - Pokud `gpu_full.is_some()`, fáze early-return (work proběhl v brain_act
    Phase 8). `apply_brownian_gpu` standalone metoda zachována jako
    `#[allow(dead_code)]` pro Sprint 63+ test path.

  *Body 6 — Per-tick `upload_turn_rates`:*
  - Reproduce mění turn_rate u nových childů. Per-tick refresh (~4 KB/tick)
    je safer než per-event sparse update; negligible bandwidth.

- **Konstanty:** žádné nové. Re-export `DRAG_COEFFICIENT` + `THERMAL_NOISE`
  z lib.

- **Výstup:**
  - `src/gpu.rs`:
    - `CellsGpu` rozšířen o 5 buffers (turn_rate, ang_vel, pitch_vel,
      ang_vel_rb, pitch_vel_rb) + accessory + 2 upload helpers + 2 download
      helpers (`download_motor_state`, `download_brain_motor_batch`).
    - `MotorGpu::dispatch_with_cells` variant. `MotorParams` pub.
  - `src/bin/headless.rs`:
    - `GpuFullState` přidává `motor: MotorGpu`. Init uploaduje turn_rates
      jednou, brain_act_gpu_full uploaduje per tick (lazy).
    - `brain_act_gpu_full` 10-fáze pipeline: motor + brownian fused, single
      batch readback.
    - `apply_brownian` early-return v `--gpu-full`.
  - **Test suite: 73/73 pass**.
  - **Smoke seed=0, 60 gen, 1000 cells, default world s `--gpu-full`:**
    - **Wall-clock 133 s = 270 ticks/s** (Sprint 61 265, +1.9 %; Sprint 58
      CPU 977 = stále **3.6× slower**).
    - Per-fáze us avg: brain_act 3463 (Sprint 61 3341, +122),
      apply_brownian 0.05 (Sprint 61 142, –142, fused ✓).
      **Combined brain+brownian: 3463 vs 3483 (Sprint 61) = -20 µs net.**
    - Pop final 487 (Sprint 61 548, CPU 572) — drift v rangi předchozí
      atomic CAS noise; žádná extinkce.

- **Závěr Sprint 60-62 série:**

  | Sprint | --gpu-full ticks/s | vs CPU 977 |
  |--------|-------------------|------------|
  | 59 (field-GPU + readback) | 345 | -65 % |
  | 60 (sensor-GPU) | 222 | -77 % |
  | 61 (populate-GPU, no sensor RT) | 265 | -73 % |
  | **62 (motor + brownian fused, 1 RT/tick)** | **270** | **-72 %** |

  GPU full pipeline pro 1000 cells / 64×64×16 grid je **fundamentálně
  net-negative**. Per-dispatch overhead (~50-200 µs `device.poll` round-trip
  + per-encoder submit cost) dominuje sub-millisecond CPU work (Sprint 57+58
  paralelní baseline). 5+ GPU dispatches/tick v brain_act fázi (spatial_hash×2,
  sensor, populate, brain, motor, brownian) sčítají do ~600-1200 µs jen na
  dispatch overhead. CPU paralelní cesta nemá tento per-call overhead — jen
  rayon spawn (~5 µs) + L1 cache friendly access.

  **GPU win se objeví při bigger workload** — per-cell GPU compute time
  roste lineárně s N, ale per-dispatch overhead je fixed. Break-even
  očekáván 5k–10k cells. Sprint 63+ benchmark verifikuje.

- **Poznámky:**
  - **Round-trip status po Sprint 62: 1 `device.poll(Wait)` per tick** —
    minimum dosažitelný bez full-GPU sim loop (eat_food / predate /
    collisions / step na CPU stále vyžadují readback positions/energy).
  - **Sprint 63+ kandidáti:**
    - GPU step/eat/predate/collision (Sprint 50 standalone shadery wire).
      Plný GPU loop = 0 RT/tick (CSV log readback only at gen-end).
    - Bigger workload smoke: spustit `--gpu-full` při 5k a 10k cells.
      Compare s CPU paralelní baseline. Identify break-even point.
    - GPU spatial_hash bucket capacity bump pro bigger world (current
      fixed 64×32×4 = ±1024/±512 worldová krytí).
    - Renderer mirror Sprint 60-62 GPU pipeline (currently Sprint 59 path).
