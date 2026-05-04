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

## Sprinty 59–62 — open-ended

- **Sprint 59+:** GPU FieldGpu/SensorGather hot-path wire-up, případně
  `SpatialGrid` dense Vec layout (žádný hash).
- **Sprint 59+:** 3D voxel rendering, ghost cell visual wrap.
- **Sprint 60+:** progressive z expansion (z=30, z=50).
- **Sprint 61+:** thermal stratification (temperature field z-gradient).
- **Sprint 62+:** light field z-attenuation (photic vs aphotic zones).
