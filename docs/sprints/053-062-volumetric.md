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

## Sprinty 54–62 — open-ended

- **Sprint 54: GPU field stack 3D migrace.** `field_diffuse_3d.wgsl` 7-point
  stencil + atomic float CAS. SensorGatherGpu (Sprint 50) read 3D field grid.
  Re-enable 2 ignored tests.
- **Sprint 55+:** 3D voxel rendering renderer overlay (Bevy `Mesh3d`
  voxelmesh nebo iso-surfaces).
- **Sprint 56+:** progressive z expansion (z=30, z=50) s adaptive selection
  monitoring.
- **Sprint 57+:** thermal stratification — temperature field z-gradient
  ovlivňuje cell metabolism / behavior.
- **Sprint 58+:** light field (z-attenuation) — photic vs aphotic zones,
  chemosynthesis vs photosynthesis evolutionary niche.
