# Sprinty 173–182: Shared sim driver

Předchozí 4 desítky (133-172) přinesly plasticity infrastructure
(reward funnel, neuromodulace, homeostat, Izhikevich+STDP) **jen
v headless**. Renderer (`src/main.rs` + `src/renderer/systems/*`) drží
**pre-S133 stav** — žádné předace/damage/bond/mate rewards, žádný
Izhikevich, žádný STDP. ~30 sprintů divergence.

**Decade cíl:** vytáhnout tick orchestration z `src/bin/headless/world.rs`
do shared lib modulu `bioscape::sim::World`. Headless + renderer oba
volají identickou simulační logiku. **Eliminuje budoucí divergenci
permanentně** — každý nový sprint mění jen jedno místo, renderer
automaticky vidí.

## Sprint 173 — Extract `World` to `bioscape::sim`

**Cíl:** přesunout `World` struct + `init_gpu_full` + `tick` + supporting
methods z `src/bin/headless/world.rs` do `src/sim/world.rs`. Headless
binary jen importuje. Behavioral byte-identical s pre-S173.

**Výstup:** new module `src/sim/mod.rs` + `src/sim/world.rs` (3309 LOC,
exported via `pub use world::*`). `lib.rs` přidá `pub mod sim;`. Headless
main `mod world;` → `use bioscape::sim as world;` (preserved alias
keeps `world::World` / `world::Checkpoint` namespacing v binary).
`csv.rs` import změna na `bioscape::sim::{World, EDGE_FRAC_THRESHOLD}`.
**100+ `bioscape::` refs uvnitř moved file přepsané na `crate::`**
přes sed (file's now uvnitř lib crate). `world_tests.rs` zůstal
v `src/bin/headless/` jako before — GPU-adapter-heavy tests vs `cargo
test --lib` fan-out-fail collision unresolved. Lib testy 442 passed
(1 ignored) — **same as pre-S173 baseline**. Headless 2-gen smoke
runs cleanly @ 95 ticks/s, pop 354. Renderer binary kompiluje (no
behavior change yet).

**Poznámky:** Test fixtures (`world_tests.rs`) zůstaly v binary místo
move do sim/. Důvod: 43 z těch testů call `init_gpu_full()` který fails
pod parallel test runner. Pre-existing issue (tests were in `cargo test
--bin headless` running serial, masked by lib-test filter). Move into
lib reveal failures. Workaround = keep tests v binary. Proper fix
(shared GPU context across tests) odložené do S181 cleanup.

## Sprint 174 — Renderer `World` resource

**Cíl:** renderer instantiates `bioscape::sim::World` jako Bevy Resource.
Existing renderer ECS systems zatím zůstávají (parallel path), `World`
just sits unused.

**Výstup (foundational scope):** `#[derive(Resource)] pub(super) struct
SimWorld(pub(super) bioscape::sim::World);` newtype wrapper v
`renderer/resources.rs`. World je `Send + Sync + 'static` (Cell, Bond,
SimClock, wgpu types vše Send+Sync), takže Resource derive funguje bez
`Arc<Mutex<_>>` overhead. Renderer kompiluje (warning: dead_code, expected
— actual init + wire-up přichází v S175).

**Poznámky:** Init `World::new_with_maze(...)` ponechán do S175 spolu s
tick system wire-up. Důvod: bare type declaration je 5-řádkový change,
kompletní init + tick + sync je další 200+ řádků a vlastní commit.

## Sprint 175 — SimWorld instantiation (scope-cut: tick system → S176)

**Cíl original:** add Bevy system `sim_tick_system` calling `world.tick`
+ `sync_cells_to_entities` system po tick.

**Výstup (instantiation only):** v `renderer/setup.rs` se vytvoří
`SimWorld(World::new_with_maze(&mut rng, WORLD_MAP_SEED, MATING_RADIUS,
INITIAL_CELLS, MAX_POPULATION, EventCalendar::default(), None))` a
inserts as Resource. GPU init záměrně skipped (`world.gpu_full = None`)
— renderer drží vlastní `GpuFullPipeline` jako canonical pipeline až do
S176. Renderer kompiluje + boots; SimWorld sits as passive CPU mirror.

**Poznámky scope-cut:** tick system + sync_cells_to_entities přesunuto
do S176. Důvod: renderer má 2050 LOC vlastních tick systémů; substituce
za world.tick() je atomic surgery (compile must stay green) která se
nehodí inkrementálně. Sprint 176 spojuje tick wire-up + legacy system
deletion do single coordinated commit. Sprint 175 zde dodává jen
foundational World init.

## Sprint 176 — Shared-driver tick wired (parallel to legacy)

**Cíl original:** remove `apply_eligibility_step`, `apply_episodic_novelty`,
`cells_brain_act`, `cell_predates_on_neighbor`, motor/step/brownian
systems v renderer. World::tick handles vše.

**Výstup (scope-pivot — add, then delete):** rather than atomic
substitution (high risk), add `sim_tick` system that calls
`world.tick(&mut rng)` per `FixedUpdate`, **sequenced after legacy
`tick_end`**. Both pipelines tick concurrently — SimWorld evolves on
its own state, legacy cells stay canonical for rendering. World GPU
init happens at setup (creates 2nd wgpu Instance alongside Bevy's
RenderPlugin's own — wasteful but functional). `SimRng` Resource added
for deterministic seed. `sync_simworld_to_cellentity` (position copy
from SimWorld → CellEntity components) + actual deletion of legacy
systems pushed to S177-S178.

**Poznámky:** S176 přinesl WORKING WIRE-UP — World tickne každý frame
v renderer process. Vizuální payoff (renderer mirror plasticity z
133-172) přijde s S177 sync + S178 delete. Decade scope-revised:
S176 = "wire", S177 = "sync", S178 = "delete legacy". Memory: 2 wgpu
instances + 2× cell populations + ~5-10% perf hit (acceptable for
this transition decade).

## Sprint 177 — Entity sync (position copy only)

**Cíl original:** when `world.cells` grows / shrinks, renderer entities
follow (spawn / despawn).

**Výstup (scope-cut to "position sync only"):** new system
`sync_simworld_to_cellentity` v `renderer/systems/sim_tick.rs`. Per
FixedUpdate po `sim_tick`: iterates `CellSlotMap.slot_to_entity` 0..min(N,
world.cells.len()) a kopíruje `world.cells[slot]` přes `&mut CellEntity`.
Index-based pairing — předpokládá že legacy lifecycle drives slot
allocation/release v sync s SimWorld swap_remove. Out-of-range slots
silently skipped (no panic). Pop changes from SimWorld nyní viditelné
v renderer visual pipeline (sync_transforms reads `CellEntity` which
SimWorld now drives).

**Poznámky scope-cut:** plný spawn/despawn (entity lifecycle synchronized
with SimWorld cell birth/death events) odložené do S178+. Důvod: renderer
legacy `cell_reproduces_on_threshold` + `cell_dies_on_zero_energy`
systems still allocate/release slots; deleting them (S178) by současně
přesměrovalo lifecycle k SimWorld events. Index-based sync v S177 funguje
v overlapping regime; po S178 legacy je gone a SimWorld diverging length
becomes a real issue → S178 must include lifecycle migration.

## Sprint 178 — Unschedule legacy renderer tick systems

**Cíl original:** wire renderer affordances (camera, UI, keyboard) k novému
shared driver.

**Výstup (scope-pivot k legacy deletion):** Bevy FixedUpdate schedule
zredukován z **25+ systémů přes 3 chained tuples + 4 standalone**
registrací na **single 6-system chain**:
- `tick_start` → frame-timing diagnostic open
- `advance_clock` → renderer's SimClock resource (UI dependency)
- `sim_tick` → `world.tick(rng)` — full S133-S172 plasticity
- `sync_simworld_to_cellentity` → cell positions/state copy
- `tick_death_fade` → visual fade-out for Dying entities
- `tick_end` → frame-timing diagnostic close

Code v `src/renderer/systems/{brains,brains_gpu,collisions,fields,food,
lifecycle,physics}.rs` zůstává v tree (dead_code warnings emerge, S181
will sweep them). Renderer **kompiluje green** s SimWorld jako jediným
sim driverem. Pre-S173 dual-tick (legacy + SimWorld) je gone — single
canonical pipeline.

**Poznámky:** GPU memory: 2 wgpu instances stále — Bevy's RenderPlugin
+ SimWorld's GpuContext. Legacy `GpuFullPipeline` Resource zůstává
inserted (nikdo ji nečte). S181 cleanup smaže Resource + delete dead-code
files. Camera, UI, gizmos, screencast v Update schedule netknuty —
fungují stejně jako pre-S173 protože čtou `CellEntity` (now driven by
SimWorld).

## Sprint 179 — Performance + Resource ergonomics

**Cíl:** minimize per-frame sync overhead. World tick je ~10 ms; sync
~1 ms; render 16 ms (60fps). Budget tight.

**Plán:** profile sync_cells_to_entities. Batch updates. Skip transforms
when position unchanged (rare).

**Acceptance:** renderer 60fps na 500 cells. STDP-augmented pop boom
(1500 cells) ≥ 30fps.

## Sprint 180 — Headless ↔ renderer parity test

**Cíl:** same seed/params/initial-izh-frac, headless CSV vs renderer-
driven World CSV. Should be byte-identical (modulo wallclock).

**Plán:** renderer dump CSV at gen end (mirror headless logger). Diff
gen N rows.

**Acceptance:** byte-identical on `ticks_per_sec` column exception.

## Sprint 181 — Code cleanup

**Cíl:** remove `pub` exports že už nikdo nepoužívá. Update CLAUDE.md
sekce **Architektura**.

**Plán:** dead code analysis. CLAUDE.md update z "Split kódu" sekce —
World je nyní v lib, ne v `src/bin/headless/`.

**Acceptance:** clippy clean, CLAUDE.md aktuální.

## Sprint 182 — Decade retro + 183+ outline

**Cíl:** retrospektiva.

**Plán:** decade retro v tomto souboru. Outline 183+ work: per-cell
STDP evolution (S171 deferred z 163-172), longer-run stability
validation, dendritic compartments, multi-channel STDP, replicate
runs pro statistical confidence.

## Velká varování pro desítku

1. **Bevy Resource thread-safety.** `World` obsahuje GPU resources
   (wgpu Device/Queue z Arc), Vec<Cell>, atd. Send + Sync constraints
   na Resource by mohly vyžadovat refactor některých nested types.

2. **CellSlotMap vs cells indexing.** Renderer existing model =
   per-entity components; shared World = Vec<Cell> indexed by slot.
   Mapping musí být robust přes reproduce/swap_remove.

3. **GPU context sharing.** Headless dnes vlastní GpuContext;
   renderer Bevy taky chce GPU. Mohl by collide. Pravděpodobně
   sdílet Bevy's wgpu instance — risk šachy s renderer rendering
   queue.

4. **Existing renderer features** (orbit camera, world map rendering,
   pheromone field viz, fog) — co z toho přežije refactor a co se
   musí rewritnout.
