# Plán: odstranění CPU compute, GPU jako jediný SOT

## 0) Definice rozsahu

**„CPU část"** = veškerý sim compute kód, který má dnes GPU paritu (dual-path) i CPU-only fáze bez GPU shaderu.

Cíl: **headless a renderer používají stejný GPU compute SOT**. Data žijí v GPU bufferech. CPU drží jen I/O + control plane.

**Co po refaktoru zůstává na CPU** (= I/O + control plane):

- `src/params/*` — `pub const` SOT pro CPU↔GPU upload
- `src/lib.rs` — type definice (`Cell`, `Genome`, `Spike`, `Bond`, `Food`, …) jako **data layouts** pro bincode/CSV/uniform upload
- `src/clock.rs`, `src/events.rs` — sim time + shock kalendář (CPU řídí tick driver)
- `src/world_map.rs` — Perlin generace mapy (jednou na startu, upload do GPU texture)
- `src/xoshiro.rs` — seed RNG před uploadem do GPU bufferu
- `src/bin/headless/{main,csv}.rs` — CLI, CSV writer, checkpoint serializace
- `src/renderer/*` — Bevy app shell, kamera, gizmos, ECS Transform sync z mapped GPU bufferu
- `src/genetics/mod.rs` + `src/neural/{brain,cppn}.rs` — **jen** struct definice + serde (forward/mutate/crossover impls jdou pryč)

**Co jde pryč:**

- 31 CPU compute callsites (`Brain::forward`, `populate_brain_inputs`, `step_with_thermal_maze`, `SmellField::step`, `resolve_collisions`, …)
- 33 dual-path `if let Some(mut gpu) = gpu_full { … } else { CPU }` větví
- 128 `#[cfg(feature = "gpu")]` gatů — `gpu` feature flag mizí, `wgpu` je core dep
- `src/spatial.rs` (nahrazeno `SpatialHashGpu`)
- Větší část `src/cell.rs`, `src/sensors.rs`, `src/chemistry.rs`, `src/physics_utils.rs`, `src/reproduction.rs`

## 1) Mezery, které musí být zaplněné GPU shaderem

Bez nich nelze vymazat CPU:

| Fáze | Dnešní stav | Co potřebuje |
|---|---|---|
| Reproduce | CPU `World::reproduce()` — variable-length spawn child cells, mutate genome | `reproduce.wgsl` — scan fertile pairs (parallel), per-pair child slot allocation přes atomic counter, CPPN respawn (už máme `cppn_from_cppn.wgsl`), upload child weights |
| Die + carrion | CPU `die_and_drop_carrion()` — swap_remove cells, push carrion food | `compact.wgsl` — mark-and-stream-compact dead cells; carrion → food slot přes atomic |
| Food spawn | CPU `spawn_food()` — rejection sampling vs world_map richness + obstacles | `food_spawn.wgsl` — per-attempt thread, sample world_map texture, obstacle mask, atomic write do free slot |
| Coop food | CPU `spawn_coop_food()` + `update_coop_food()` — Poisson + arrival registration | `coop_food.wgsl` — Poisson lottery + arrival markers per spot |
| Vibration field | CPU `update_vibration()` — žádný GPU shader neexistuje | `vibration.wgsl` — varianta `field_diffuse.wgsl` s motion-driven emit kernelem |
| Predation production | GPU `predate.wgsl` existuje + parity test, ale produkční cesta je CPU `World::predate()` (S127: GPU režie 5× pro N<10k) | Flip switch + akceptovat režii pro malé N (nebo gate na N>threshold) |
| Collision production | Stejně — GPU `collision.wgsl` test-only, produkce CPU | Flip switch |
| Eat food race | CPU 3-pass: parallel candidates → sequential resolve → reward | `eat_food.wgsl` — atomic CAS na free food slot, deterministická tiebreak rule (lowest cell_id) |
| Pheromone ch 1–2 | CPU `SmellField::step` ch1/ch2 (jen ch0 na GPU) | Rozšířit `FieldGpu` na N kanálů (jen storage limit gate) |

## 2) Fázování — wave H–N (navazuje na maze wave 1–7)

Každá vlna = 1 sprint, jedna PR, smoke run `--seed=0 --max-gens=30` na konci.

### Wave H — Predation + Collision na produkční GPU
- Smaž CPU `World::predate()` a `World::resolve_collisions()`.
- Dispatch `PredateGpu` a `CollisionGpu` v každém ticku (už existují, jen test-wired).
- Renderer `systems/collisions.rs` ztrácí CPU větev.
- Akceptujeme 5× overhead pro N<10k (research projekt, ne real-time).
- Smoke: seed=0, 30 gen → final pop sane.

### Wave I — `vibration.wgsl` + `coop_food.wgsl`
- Vibration: rozšířit `FieldGpu` o motion-emit step (per-cell write speed²·gain do nejbližšího gridu) + diffuse + decay reuse.
- Coop food: Poisson + arrival registration jako WGSL kernel; rozšířit `FoodGpu` o coop slots.
- Drop CPU `update_vibration()`, `spawn_coop_food()`, `update_coop_food()`.

### Wave J — Variable-length: food spawn + die compact
- `food_spawn.wgsl`: K-attempts × M-threads, atomic_add na `free_food_count` → write do slotu.
- `compact.wgsl`: mark-dead → stream-compact (parallel prefix sum nebo two-pass scatter). Dead → carrion food slot.
- Smaž CPU `spawn_food()`, `die_and_drop_carrion()`.
- Riziko: pokud compact je pomalejší než CPU `swap_remove` pro N~1500, kompromis = CPU pošle "dead indices" do GPU, GPU dělá swap. **Měřit, ne hádat.**

### Wave K — Reproduce na GPU
- `reproduce.wgsl`: scan fertile cells (energy ≥ threshold), atomic-pair sloty, per-pair dispatch `cppn_from_cppn.wgsl` pro child brain weights.
- Lineage ID counter: CPU si drží atomic counter, GPU dostane base+offset, zapíše `child_lineage_id = base + tid`.
- Smaž CPU `World::reproduce()`, `make_mating_child()` v `reproduction.rs`.
- Genome mutation/crossover na CPU dnes — zvážit jestli to zůstane (jen pre-tick batch) nebo migrovat (probably zůstane jako "control plane", protože je to per-child sekvenční).

### Wave L — Eat food race + ostatní zbytky
- `eat_food.wgsl`: atomic-CAS na food slot, lowest `cell_id` wins.
- Pheromone ch1/ch2 do `FieldGpu` (rozšíření existujícího pipelinu).
- Whisker/novelty/goal — wave 6 už whisker raycast přesunul, ale `apply_episodic_novelty()` a `track_goal_metrics()` jsou stále CPU. Buď migrovat (novelty má voxel ring buffer per cell — fits GPU), nebo nechat jako CPU-side diagnostika (run jednou za N tiků mimo hot path).

### Wave M — Data ownership flip
- `World::cells: Vec<Cell>` → `World::cell_count: AtomicU32` + GPU handle.
- `World::foods: Vec<Food>` → totéž.
- Cell struct zůstává jen jako (a) layout pro `CellsGpu` (`#[repr(C)]`, `bytemuck::Pod`), (b) bincode snapshot pro checkpoint.
- Bevy ECS: `Entity` per cell ostane, ale `Transform` sync system čte z mapped GPU bufferu once per frame (mechanika už existuje, jen se z ní stává jediná cesta).
- Headless CSV: gen-boundary readback do `Vec<Cell>` snapshot → CSV writer.
- Checkpoint: V9 bump. Save = readback GPU → bincode. Load = bincode → upload do GPU.

### Wave N — Vymazání mrtvého CPU kódu + odstranění `gpu` feature
- Smaž CPU implementace:
  - `src/cell.rs` `impl Cell { apply_*, populate_*, body_basis, … }` → zůstává jen struct + `Cell::new` + serde
  - `src/sensors.rs` celé compute
  - `src/chemistry.rs` `impl SmellField { step }` → struct zůstane pro checkpoint readback formát
  - `src/physics_utils.rs` `step_with_thermal_maze`, `apply_brownian`, `forward_vector`, …
  - `src/spatial.rs` celé
  - `src/neural/brain.rs` `forward`, `hebbian_step`, `apply_eligibility_step` → struct + serde stay
  - `src/reproduction.rs` `pair_fertile`, `make_mating_child` → CPU helpers ven, jen RNG seed gen zůstane
- Smaž 33 dual-path větví v `renderer/systems/*` a `bin/headless/world.rs`.
- `Cargo.toml`: smaž `gpu` feature, `wgpu`/`bytemuck`/`pollster` → not optional.
- Smaž 128 `#[cfg(feature = "gpu")]` gatů (greppable check viz §4).
- `BIOSCAPE_GPU_FULL` env var → smaž, dispatch je vždy GPU.

## 3) Verifikace per-wave

- `cargo build --release` (musí projít bez warningů)
- `cargo test` — některé testy v `tests_phase{1,2,3}.rs` testují CPU sim; ty se postupně mažou s každou vlnou
- **Smoke**: `cargo run --release --bin headless -- --seed=0 --max-gens=30 --max-pop=200` → final_pop ∈ [50, 200] (sanity, ne bit-exact)
- Renderer smoke: `cargo run --release` na 1 min, vizuální check (cells žijí, food spawnuje, žádná teleportace/freeze)
- Po wave N: `cargo bench` proti baseline (S128) → pokud regrese > 30 %, gate je open question

## 4) Ověření, že nic CPU nezůstalo

Po wave N musí všechny tyto greps vrátit 0 řádků (mimo `params/`, `lib.rs` struct defs, `xoshiro.rs`):

```bash
grep -rn "cfg(feature = \"gpu\")" src/                # 0
grep -rn "cfg(not(feature = \"gpu\"))" src/            # 0
grep -rn "BIOSCAPE_GPU_FULL\|--gpu-full" src/          # 0
grep -rn "if let Some(.*gpu_full" src/                 # 0
grep -rn "Brain::forward\|brain\.forward(" src/        # 0 (jen jako test/data, ne call)
grep -rn "step_with_thermal_maze\|apply_brownian\|SmellField::step" src/  # 0
grep -rn "fn resolve_collisions\|fn predate\b" src/    # 0 (CPU verze)
grep -rn "use bioscape::\(SpatialGrid\|ObstacleField\)" src/  # 0 (ObstacleField → GpuObstacleField)
```

Plus `Cargo.toml`:
- `wgpu`, `bytemuck`, `pollster` v `[dependencies]`, **ne** `optional = true`
- `[features]` jen `dev` zůstává, `gpu` smazaný, `default` prázdný array nebo smazaný

## 5) Rizika a mitigace

| Riziko | Mitigace |
|---|---|
| GPU reproduce/die kernel je pomalejší než CPU pro N~1500 | Měřit per wave. Fallback hybridní: CPU pre-alloc child sloty + RNG seedy, GPU dělá compute work nad nimi |
| Stream compact je nestabilní (závisí na driver implementaci prefix sum) | Začít s naivním two-pass scatter (atomic counters); optimalizovat až po měření |
| Checkpoint V9 nelze loadovat z V8 | Akceptovat (V5→V6 už bylo breaking) — research save files nejsou produkční |
| Bevy Transform sync stall (mapped buffer wait) | Použít persistently-mapped buffer (wgpu má `MapMode::Read` + `Maintain::Poll`) — staging path už existuje pro debug overlays |
| Velký diff (~5k řádků smazaných) → nepřezkoumatelný PR | Každá vlna samostatný PR; deletion (wave N) split do několika commitů: (a) systems, (b) module impl bodies, (c) feature gates, (d) Cargo.toml |
| Recent commits (`wave 7 hebbian`) jsou na `main` — refactor diverguje s rozpracovanou prací | Provést wave H–N na branch `gpu-only`, rebase before merge |

## 6) Co to umožní

- Jediný compute path → konec parity test maintenance (S132 ukázal, že CPU vs GPU drift je živý problém)
- ~3–5k řádků smazaných (28 % z `src/cell.rs`, 60 % z `src/physics_utils.rs`, celé `spatial.rs`, většinu dual-path systems)
- Headless a renderer **přesně stejný tick sequence** — share `World::tick()` jako tenkou funkci nad `GpuFullPipeline`
- Otevírá Sprint 138+ na čistou GPU optimalizaci (fuse kernels, persistent compute encoder, multi-queue)

## Otevřené otázky

1. **Genome mutate/crossover** — zůstává CPU (per-child sekvenční, low N pre-tick), nebo migrovat? Doporučení: zůstává, je to "control plane".
2. **`apply_episodic_novelty` + `track_goal_metrics`** — jsou to maze-experiment diagnostika, ne hot path. Buď migrovat (voxel ring buffer fits), nebo nechat jako CPU sample-rate kernel. Doporučení: nechat, pokud nestihne deadline.
3. **`MIN_BONDS_PER_CELL` a `Bond` struct** — bond formace má dnes contact_lists scratch hash map per-cell. Migrace na GPU vyžaduje fixed-size bond table per cell. Layout `Cell` už má `bonds: [Bond; MAX_BONDS_PER_CELL]`, takže je to jen otázka kernelu — zařadit do wave H s collision.
4. **GPU pro initial world map generation** — Perlin generation je single-shot CPU. Není to hot path, ale „nic CPU compute" striktně znamená i tohle migrovat. Doporučení: nechat CPU (run-once init).
