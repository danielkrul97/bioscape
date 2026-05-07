# Refaktor: rozbít monolitní soubory

## Stav

Čtyři soubory tvoří 99 % zdrojáku a každý je nad přívětivou hranicí:

| Soubor | Řádky | Obsah |
|---|---:|---|
| `src/lib.rs` | 9 402 | Sim core, 17 doménových bloků, ~3 400 ř. testů |
| `src/gpu.rs` | 6 687 | 13 GPU pipeline + sdílený `GpuContext` + testy |
| `src/bin/headless.rs` | 3 830 | `World` struct, tick loop, checkpoint v5, CSV logging |
| `src/main.rs` | 3 418 | Bevy app, ECS systémy, kamera, UI, screencast |

## Cíl

Rozdělit do menších modulů s jasnou doménovou hranicí. **Behaviorálně neutrální** — stejný seed → stejný běh, byte-for-byte identické CSV. Žádné funkční změny během refaktoru.

`lib.rs` zůstává **single source of truth** pro sim parametry a typy (per CLAUDE.md). Po refaktoru je `lib.rs` tenká fasáda s `pub use` re-exporty — stávající importy (`use bioscape::Cell`, `use bioscape::INNATE_THRUST_BIAS`) **se nemění**.

## Cílová struktura

```
src/
├── lib.rs                  # mod-deklarace + pub use re-exporty (~200 ř.)
├── tests.rs                # přesunuté #[cfg(test)] mod tests (~3 400 ř.)
├── params/                 # sim parametry (single source of truth)
│   ├── mod.rs
│   ├── brain.rs            # BRAIN_INPUTS, BRAIN_HIDDEN, INNATE_*_BIAS
│   ├── physics.rs          # CELL_RADIUS, DRAG_*, ENERGY_*, FOOD_*
│   ├── thermal.rs          # THERMAL_TOP/BOTTOM/Q10, diurnal/seasonal
│   ├── pheromone.rs        # PHEROMONE_GRID_*, N_PHEROMONE_CHANNELS
│   ├── world.rs            # WORLD_HALF, WORLD_MAP_*, hazard
│   ├── morphology.rs       # MIN/MAX_BODY_*, MIN/MAX_SPIKE_*
│   ├── adhesion.rs         # ADHESION_*, BOND_*
│   └── hunter.rs           # HUNTER_*
├── morphology.rs           # struct Spike + Spike::ZERO
├── neural/
│   ├── mod.rs
│   ├── brain.rs            # struct Brain, forward, mutate, crossover, hebbian
│   └── cppn.rs             # ActivationFn, CppnNode, CppnLink, Cppn + mutace
├── genetics/
│   ├── mod.rs              # Genome, MutationConfig, MUTATION_CONFIG, defaults
│   ├── mutate.rs           # Genome::mutate (132 ř.)
│   ├── crossover.rs        # Genome::crossover (117 ř.)
│   └── phenotype.rs        # PhysicsConfig, Phenotype
├── cell/
│   ├── mod.rs              # struct Cell, struct Bond, konstruktory
│   ├── lifecycle.rs        # apply_energy_drain, apply_damage, apply_spike_morph
│   ├── physics.rs          # apply_body_forces, body_basis, eat_test_pose
│   ├── sensors.rs          # BrainSensors, populate_brain_inputs, pool_*
│   └── adhesion.rs         # adhesion_velocity_delta, bond_velocity_delta, ...
├── predator/
│   ├── mod.rs              # HunterGenome, HunterMutationConfig, Hunter
│   ├── sensors.rs          # gather_hunter_sensors, populate_hunter_brain_inputs
│   ├── targeting.rs        # nearest_attackable_cell (134 ř.)
│   └── reproduce.rs        # make_hunter_child, make_hunter_mating_child
├── food.rs                 # FoodKind, Food, CoopFood
├── reproduction.rs         # pair_fertile, pick_cluster_parent, make_mating_child
├── chemistry/
│   ├── mod.rs
│   └── smell_field.rs      # SmellField + diffuse step
├── world_map.rs            # WorldMap (Perlin)
├── spatial.rs              # SpatialGrid, min_image_delta, wrap_position_xy
├── clock.rs                # SimClock, ClockTransitions
├── events.rs               # ShockKind, EventCalendar, ramp/multipliers
├── physics_utils.rs        # forward_vector, temperature_at_z, metabolism_factor, ...
├── gpu/
│   ├── mod.rs
│   ├── context.rs          # GpuContext + sdílené konstanty (BRAIN_WEIGHTS_PER_CELL, ...)
│   ├── brain.rs            # BrainGpu (brain_forward.wgsl)
│   ├── spatial_hash.rs     # SpatialHashGpu (spatial_hash.wgsl)
│   ├── field.rs            # FieldGpu (field_diffuse.wgsl)
│   ├── cells.rs            # CellsGpu (perzistentní SoA)
│   ├── brownian.rs         # BrownianGpu (brownian.wgsl)
│   ├── hebbian.rs          # HebbianGpu (hebbian.wgsl)
│   ├── motor.rs            # MotorGpu (motor.wgsl)
│   ├── step.rs             # StepGpu (step.wgsl)
│   ├── collision.rs        # CollisionGpu (collision.wgsl)
│   ├── predate.rs          # PredateGpu (predate.wgsl)
│   ├── sensor_gather.rs    # SensorGatherGpu (sensor_gather.wgsl)
│   ├── populate_inputs.rs  # PopulateInputsGpu (populate_inputs.wgsl)
│   ├── neighbors.rs        # NeighborsGpu (cell_neighbors.wgsl)
│   ├── stats.rs            # StatsGpu (cell_stats.wgsl)
│   └── tests.rs            # GPU integrační testy
├── renderer/               # binary-only modul pro Bevy app
│   ├── mod.rs              # pub fn run()
│   ├── components.rs       # ECS komponenty + resources
│   ├── setup.rs            # setup() startup, materiály, kamera, lighting
│   ├── camera.rs           # orbit kamera input
│   ├── ui.rs               # stats overlay, world map overlay, toggles
│   ├── screencast.rs       # PNG capture, CLI parse
│   ├── material.rs         # BioMaterialExt + ExtendedMaterial
│   └── systems/
│       ├── mod.rs
│       ├── fields.rs       # update_smell_field, update_pheromone_field
│       ├── brain.rs        # cells_brain_act, pool_bonded_hidden_cells
│       ├── lifecycle.rs    # spawn_food, eat, reproduce, die, fade
│       ├── hunters.rs      # step_hunters, hunters_lifecycle
│       ├── collisions.rs   # resolve_cell_collisions, predation
│       └── render.rs       # gizmos, transform sync
├── main.rs                 # `mod renderer; fn main() { renderer::run() }` (~30 ř.)
└── bin/
    └── headless/
        ├── main.rs         # CLI parsing + main loop (~400 ř.)
        ├── world.rs        # struct World + new + tick (orchestrace)
        ├── checkpoint.rs   # Checkpoint v5 (save/load)
        ├── csv_logging.rs  # write_stats (141 sloupců) + analytika
        ├── timings.rs      # PhaseTimings + dump
        └── gpu_full.rs     # GpuFullState (perzistentní GPU stav)
```

## Re-export strategie

`lib.rs` po refaktoru:

```rust
pub mod params;
pub mod neural;
pub mod genetics;
pub mod cell;
pub mod predator;
pub mod gpu;  // už existuje
// ...

// Ploché re-exporty pro zpětnou kompatibilitu
pub use cell::{Cell, Bond};
pub use genetics::{Genome, Phenotype, MutationConfig, MUTATION_CONFIG};
pub use neural::brain::Brain;
pub use neural::cppn::{Cppn, ActivationFn};
pub use predator::{Hunter, HunterGenome};
pub use params::*;  // všechny pub const
// ...
```

Externí kód (`gpu.rs`, `main.rs`, `headless.rs`, `benches/`) nemusí měnit importy. Po dokončení refaktoru lze případně postupně migrovat na namespace-aware importy (`bioscape::genetics::Genome`), ale to je separátní úloha.

## Fázování

Každá fáze = jeden samostatný commit/PR + zelený `cargo test` + smoke run + bit-identický CSV diff vs. baseline. Hrubě každá fáze ≈ 1 sprint, větší možná 2.

### Fáze 0: Baseline
- Commit nebo stash rozpracovaných změn (`M src/gpu.rs`, `M src/main.rs`).
- Zaznamenat referenční výstup: `cargo run --bin headless -- --seed=0 --max-gens=30 --max-pop=200`, totéž pro `--seed=42`. Uložit CSV jako baseline (commit do branche).
- `cargo bench` pro headless_phases / full_tick / predate_gpu → baseline časy do dokumentu.
- Vytvořit branch `refactor/split-modules`.

### Fáze 1: Tests out
**Cíl:** Sundat ~3 400 ř. testů z `lib.rs`, aby další fáze pracovaly nad ~6 000 ř. souborem.

- Přesun: `#[cfg(test)] mod tests { ... }` (lib.rs řádky 5963–9402) → nový `src/tests.rs`.
- V `lib.rs` zůstane jen `#[cfg(test)] mod tests;`.
- Privátní položky lib.rs zůstávají dostupné pro tests modul automaticky.

### Fáze 2: Sim parametry → `params/`
**Cíl:** Vytáhnout všechny `pub const` clustery do `src/params/`.

- Mapování (přibližné, doladit při implementaci):
  - `params/brain.rs` ← lines 17–151 (vision + brain + innate biases)
  - `params/physics.rs` ← lines 157–208 (timing + cell + food)
  - `params/thermal.rs` ← lines 217–253
  - `params/pheromone.rs` ← lines 266–367 (hazard + pheromone + smell + density)
  - `params/world.rs` ← lines 388–400
  - `params/morphology.rs` ← lines 410–499 (body + spike rozsahy)
  - `params/adhesion.rs` ← lines 528–705 (shell + adhesion + bond + cluster)
  - `params/hunter.rs` ← lines 756–870
- Cross-references jako `BRAIN_INPUTS = SENSORY + RECURRENT` musí žít v jednom souboru.
- `lib.rs` přidá `pub mod params; pub use params::*;`.
- Helpery jako `bond_defense_factor`, `cell_exposure`, `sensor_slot_category` zůstávají zatím v lib.rs (přesun až s relevantní doménou).

### Fáze 3: Neural (Brain + CPPN)
**Cíl:** `neural/brain.rs` + `neural/cppn.rs`.

- Přesun CPPN bloku (lib.rs 1773–2410): `ActivationFn`, `CppnNode`, `CppnLink`, `Cppn`, serde helpery, substrate coord fns → `neural/cppn.rs`.
- Přesun Brain bloku (lib.rs 2424–2964): `struct Brain`, impl (forward, mutate, crossover, hebbian), serde → `neural/brain.rs`.
- `Brain::crossover` (159 ř.) zůstává monolitická — případné rozdělení pošli do separátního follow-up.
- Re-export přes `pub use neural::brain::Brain; pub use neural::cppn::*;`.

### Fáze 4: Genetics
**Cíl:** `genetics/{mod, mutate, crossover, phenotype}.rs`.

- `genetics/mod.rs` ← `MutationConfig` + `MUTATION_CONFIG` const + `Genome` struct + defaults + `Genome::random`/`Genome::new` (lib.rs 2965–3190 + 3447–3645).
- `genetics/mutate.rs` ← `impl Genome { fn mutate(...) }` (lib.rs 3191–3322, 132 ř.).
- `genetics/crossover.rs` ← `impl Genome { fn crossover(...) }` (lib.rs 3323–3439, 117 ř.).
- `genetics/phenotype.rs` ← `PhysicsConfig`, `Phenotype`, `Phenotype::from_genome` (lib.rs 3447–3645).

### Fáze 5: Cell entita
**Cíl:** `cell/{mod, lifecycle, physics, sensors, adhesion}.rs`. **Největší a nejrizikovější fáze.**

- `cell/mod.rs` ← `struct Bond`, `struct Cell`, konstruktory `new`/`from_genome`/`child_from_parents`, `mutate`/`crossover` (lib.rs 3646–3745).
- `cell/lifecycle.rs` ← `apply_energy_drain`, `apply_damage`, `apply_spike_morph` (133 ř.).
- `cell/physics.rs` ← `apply_body_forces`, `body_basis`, `eat_test_pose`, `fov_cone_accept`.
- `cell/sensors.rs` ← `BrainSensors`, `populate_brain_inputs` (70 ř.), `apply_sensor_gains`, `pool_bonded_sensors`, `pool_bonded_hidden`.
- `cell/adhesion.rs` ← `adhesion_velocity_delta`, `bond_velocity_delta`, `bond_defense_factor`, `cell_exposure`, `sensor_slot_category`.

Cyklické závislosti Cell ↔ Adhesion ↔ Reproduction řeší **volné funkce** přijímající `&mut [Cell]` (Rust nedovolí impl-blok rozdělit napříč různými moduly, ale dovolí více impl-bloků na stejný typ ve stejném crate). Pro logiku, která vyžaduje multi-cell přístup, použít free fn v relevantním submodulu.

### Fáze 6: Predátoři
**Cíl:** `predator/{mod, sensors, targeting, reproduce}.rs`.

- `predator/mod.rs` ← `HunterGenome`, `HunterMutationConfig`, `HUNTER_MUTATION_CONFIG`, `Hunter` + impl, `HunterSnapshotMin` (lib.rs 952–1377).
- `predator/sensors.rs` ← `HunterBrainSensors`, `gather_hunter_sensors` (108 ř.), `populate_hunter_brain_inputs`, `pool_bonded_hunter_hidden`.
- `predator/targeting.rs` ← `nearest_attackable_cell` (134 ř.).
- `predator/reproduce.rs` ← `make_hunter_child`, `make_hunter_mating_child`.

### Fáze 7: Food, reproduce, chemistry, svět, čas, eventy
**Cíl:** Rozkrojit zbytek lib.rs do menších doménových souborů.

- `food.rs` ← `FoodKind`, `Food`, `CoopFood` + helpery (lib.rs 4220–4400).
- `reproduction.rs` ← `pair_fertile`, `pick_cluster_parent`, `make_mating_child` (103 ř.), `reject_food_for_richness` (lib.rs 4794–5063).
- `chemistry/smell_field.rs` ← `SmellField` + diffuse step (lib.rs 5064–5274).
- `world_map.rs` ← `WorldMap` + Perlin (lib.rs 5275–5428).
- `spatial.rs` ← `SpatialGrid` + `min_image_delta` + `wrap_position_xy` (lib.rs 5380–5548).
- `clock.rs` ← `SimClock`, `ClockTransitions` (lib.rs 5549–5599).
- `events.rs` ← `ShockKind`, `ShockEvent`, `ShockScheduleConfig`, `EventCalendar`, multipliers (lib.rs 5600–5961).

### Fáze 8: Physics utilities
**Cíl:** Drobné helpery do `physics_utils.rs`.

- `forward_vector`, `spike_direction`, `vision_fov_factor`, `temperature_at_z`, `metabolism_factor` (lib.rs 4442–4611).
- Možná alternativa: slít do `cell/physics.rs` pokud nikdo jiný nepoužívá. Pokud ano (např. `pheromone_field` sampling), pak samostatný soubor.

Po fázi 8 by `lib.rs` měla mít cca 200 ř. (jen mod-deklarace + re-exporty).

### Fáze 9: GPU modul
**Cíl:** Rozbít `gpu.rs` na per-pipeline soubory zrcadlící `shaders/`.

- `gpu/context.rs` ← `GpuContext` + sdílené konstanty (`BRAIN_WEIGHTS_PER_CELL`, `GPU_HASH_GRID_*`).
- 14 per-pipeline souborů (300–500 ř. každý) — viz cílová struktura.
- Common buffer/dispatch patterny: pokud se opakují, do `gpu/helpers.rs`. Pokud jsou per-pipeline různé, nech.
- `gpu/tests.rs` ← integrační testy GPU (parita CPU vs GPU; gpu.rs 5424–6687).

Cargo struktura: `src/gpu/mod.rs` deklaruje submoduly; `lib.rs` má `pub mod gpu;` (už existuje, pouze přejde na directory module).

### Fáze 10: Headless binárka
**Cíl:** `src/bin/headless.rs` → `src/bin/headless/{main, world, checkpoint, csv_logging, timings, gpu_full}.rs`.

- `headless/main.rs` ← CLI parsing (Sprint args), main loop, GPU init wiring.
- `headless/world.rs` ← `struct World` + `World::new()` + `World::tick()` (orchestrace fází).
- `headless/checkpoint.rs` ← `Checkpoint` v5, magic+version validation, save/load. **Pozor:** Bincode serializace musí zůstat byte-identická.
- `headless/csv_logging.rs` ← `write_stats` (141 sloupců), `shock_summary`, `attack_entropy`, `w1_frobenius_std`, events sidecar.
- `headless/timings.rs` ← `PhaseTimings` + per-phase microsecond accumulators + dump.
- `headless/gpu_full.rs` ← `GpuFullState` (gated `cfg(feature = "gpu")`).

### Fáze 11: Renderer binárka
**Cíl:** `src/main.rs` → `src/main.rs` (tenký entry) + `src/renderer/`.

- `main.rs` zůstane `~30 ř.`: `mod renderer; fn main() { renderer::run() }` (+ screencast CLI parse, viz fáze 0 výstup).
- `renderer/mod.rs` ← `pub fn run()` builder Bevy app, registrace pluginů a systémů.
- `renderer/components.rs` ← ECS komponenty + resources (main.rs 130–405).
- `renderer/setup.rs` ← `setup()` startup systém + materiály + kamera + lighting (main.rs 635–930).
- `renderer/camera.rs` ← orbit kamera state + input handling (část main.rs 2191–2421).
- `renderer/ui.rs` ← stats overlay + world map overlay + toggles (main.rs 2432–2588 + 3389–3418).
- `renderer/screencast.rs` ← PNG capture, CLI parse (main.rs 593–634).
- `renderer/material.rs` ← `BioMaterialExt` + `ExtendedMaterial` (main.rs 114–127).
- `renderer/systems/{fields, brain, lifecycle, hunters, collisions, render}.rs` ← ECS systémy (main.rs 1028–3083 podle bloků).

Cargo nepotřebuje úpravu — `[[bin]] path = "src/main.rs"` zůstává; submoduly jsou pouze viditelné z `main.rs`.

### Fáze 12 (volitelná): Sjednocení duplicit binárek
**Cíl:** Lift společné logiky z `renderer/systems/*` a `bin/headless/world.rs` do `lib.rs`.

Kandidáti:
- `resolve_collisions` (broad-phase + Newton damping + contact tracking) — duplikováno
- `cell_eats_food` (kontaktní kontrola + energy gain) — duplikováno
- `food_multiplier`, `food_target`, `hazard_drain` (3× duplikováno verbatim)

Bevy ECS používá `Query<&mut Cell>`, headless `Vec<Cell>`. Lift musí mít obě varianty: free fn nad `&mut [Cell]` v lib + tenká wrapper v rendereru. **Zvážit po fázi 11** — pokud se duplikace jeví jako menší zlo než trait-based abstrakce, nechat.

## Verifikace (každá fáze)

1. `cargo build --features gpu`
2. `cargo build --no-default-features` (bez GPU)
3. `cargo test --features gpu` — **primární signál**, 176 testů deterministicky pokrývá sim core
4. `cargo test --no-default-features`
5. Smoke headless: seed=0, 30 gen, max_pop=200 → musí doběhnout bez crash, sane final pop
6. Totéž seed=42
7. (Každé 3 fáze) `cargo bench` → drift > 5 % značí regrese inlinování

**Pozn.:** CSV výstup není byte-deterministický napříč běhy (S113 konvence — `rand::rng()` thread-local + par_iter ordering). Bit-for-bit diff není validní signál; primární verifikace je test suite.

## Rizika

| Riziko | Mitigace |
|---|---|
| Subtle změna chování (pořadí volání, viditelnost, inlining) | Bit-for-bit CSV diff každé fáze; fail-fast |
| Cyklické závislosti Cell ↔ Adhesion ↔ Reproduction | Volné fn s `&mut [Cell]` místo metod; impl-bloky rozděleny per soubor |
| Checkpoint v5 binární kompatibilita (Fáze 10) | Před fází uložit baseline checkpoint; po fázi načíst → diff |
| GPU testy parita CPU/GPU (Fáze 9) | Per-pipeline tests izolovaně; full-tick parita až nakonec |
| Velké diffy → nečitelné PR | Každá fáze samostatný PR; aim < 1 500 ř. změn |
| Bevy `dynamic_linking` interaguje s rozbitím main.rs (Fáze 11) | Testovat i s `cargo build --features dev` |
| Regression na `pub` viditelnost (něco původně privátní teď musí být `pub(crate)`) | Před commit `cargo +nightly clippy --workspace -- -W unreachable_pub` |

## Otevřené otázky

1. **Tests granularity:** Single `src/tests.rs` (Fáze 1) nebo postupná migrace do per-modul `mod tests`? Druhá idiomatičtější, ale dražší. **Doporučení:** Fáze 1 = single file; po Fázi 8 zvážit per-modul migraci jako separátní úlohu.
2. **Re-export politika po dokončení:** Zachovat ploché `pub use` (`bioscape::Cell`), nebo migrovat na namespace (`bioscape::cell::Cell`)? **Doporučení:** ploché, namespace jako follow-up.
3. **Sprint mapování:** Fáze 5, 9, 11 možná každá 2 sprinty kvůli objemu. Doladit při plánování.
4. **Fáze 12 (lift duplicit):** Provést, nebo nechat? Bevy ECS vs. plain `Vec` rozdíl. **Doporučení:** rozhodnout po Fázi 11 podle konkrétních kandidátů.
5. **Helper soubory granularity:** `physics_utils.rs` jako jeden soubor, nebo rozdělit (`geometry.rs`, `thermal.rs`, ...)? **Doporučení:** jeden soubor; pokud naroste nad ~300 ř., rozdělit.

## Souhrn

Po refaktoru:
- ~50 souborů místo 4
- Žádný soubor nad ~800 ř. (medián ~300 ř.)
- `lib.rs` jako tenká fasáda (~200 ř.)
- Behaviorálně identické — všechny existující testy + smoke runs procházejí
- Stávající importy nezměněny (re-exporty)
- GPU pipeline mapováno 1:1 na shadery v `shaders/`
- Renderer a headless každá vlastní subdir, sdílené sim core přes `lib.rs`
