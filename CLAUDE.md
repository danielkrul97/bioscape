# Projekt

Cílem výzkumného projektu Bioscape je vytvořit simulaci evoluce, abych pochopil, jak vzniká inteligence. Projekt je napsán v Rustu. **Veškerý sim compute běží na GPU** přes wgpu compute shadery — CPU drží jen control plane (CLI, CSV, checkpoint, ECS sync, tick driver).

**Architektura:**

- **Engine:** Bevy 0.18 (ECS, 3D rendering přes wgpu — orbit Camera3d, StandardMaterial, sphere mesh + non-uniform scale pro ellipsoid těla). `default-features = false` plus ručně vybraný feature set bez `bevy_gilrs` a `audio` — projekt nepotřebuje gamepad a Bevy built-in audio.
- **Audio sonifikace vibrací:** renderer-only modul `src/renderer/audio.rs` (Bevy plugin) za Cargo feature `audio` (default on). Otevírá vlastní `cpal` output stream a sonifikuje `VibrationResource` na pozici kamery přes `fundsp` procedurální graph (pink noise → SVF lowpass → stereo pan). Headless ho nikdy nepoužívá. Pro CI / boxy bez `libasound2-dev`: `cargo build --no-default-features --bin headless`. F8 toggle on/off za běhu.
- **Split kódu:**
  - `src/lib.rs` — sim type definitions (`Cell`, `Genome`, `Bond`, `Spike`, …) + `params/*` `pub const` parametry + `MUTATION_CONFIG` / `PHYSICS_CONFIG`. **Single source of truth** pro renderer i headless — uploaduje se přímo do GPU bufferů. Nové tuneables patří sem.
  - `src/gpu/*` — GPU compute layer: per-pipeline Rust API kolem `shaders/*.wgsl` (brain, sensor_gather, populate_inputs, motor, brownian, step, hebbian, collision, predate, food_spawn, field, izhikevich, stdp_*, synaptic_scale, excitability, …). Tahle vrstva je sdílená renderer + headless.
  - `src/sim/world.rs` — **shared simulation driver** (od Sprint 173). Obsahuje `World` struct + `init_gpu_full` + `tick()` orchestraci. Volaný z obou binárek; nové plasticity sprints mění jen tady a oba binaries to vidí automaticky.
  - `src/main.rs` — Bevy app, ECS rendering. Per-frame `sim_tick` system (S176) volá `world.tick()`; `sync_simworld_to_cellentity` (S177) kopíruje `world.cells` do `CellEntity` componentů pro visual pipeline. Legacy ECS tick systems unscheduled v S178 (jejich code v `src/renderer/systems/*.rs` zůstává v tree pro reference; mass delete v 183+ cleanup).
  - `src/bin/headless/` — bezokenní harness pro batch experimenty (CSV log per generaci, deterministický seed). Tick loop = `world.tick()` v tight CPU loop. Konzumuje stejný `bioscape::sim::World` jako renderer — semantická parita je structural guarantee. Opt-in per-cell JSON dump přes `--dump-dir` (periodic top-K po N gen + final dump; viz `bin/headless/dump.rs`).
  - `src/json_export.rs` — pure-Rust serializace `Cell` do human-readable JSON (custom pretty-printer, primitive arrays na jednom řádku). Sdílená mezi renderer inspector "Save…" / "Copy" akcemi a headless dumpem. Žádné Bevy/egui deps.
- **GPU:** mandatory, ne opt-in. `wgpu` + `bytemuck` + `pollster` jsou core dependencies (žádná `gpu` feature). Spuštění vyžaduje wgpu-compatible adapter s compute supportem + ≥ 20 storage buffers per shader stage. Init failure je fatal (panic).
- **Dev smyčka:** `cargo run --features dev` zapne `bevy/dynamic_linking` pro rychlou inkrementální iteraci po prvním buildu.

**Zdroj pravdy pro biologii a evoluci:** Pokud prompt řeší evoluční mechanismy, genotyp/fenotyp, selekci, fitness, mutace nebo podobné koncepty, primárně čerpej z `docs/` — odráží to specifický framing tohoto výzkumu. Generické znalosti z tréninku použij jen jako doplněk a při konfliktu s `docs/` na to upozorni.

# Vývoj

- Vývoj vedeme po sprintech.
- 10 sprintů = jeden markdown dokument v `docs/sprints/`.
- Pojmenování: `NNN-MMM-slug.md`, kde `NNN` a `MMM` jsou zero-padded čísla prvního a posledního sprintu v souboru a `slug` je krátký název shrnující téma té desítky sprintů (např. `001-010-bootstrap.md`, `011-020-genome.md`).
- Každý sprint v dokumentu má sekci `## Sprint NN — krátký slug` se třemi řádky:
  - **Cíl:** co chceme za sprint dosáhnout.
  - **Výstup:** co reálně vzniklo (může odkazovat na commity / soubory).
  - **Poznámky:** volitelné — pozorování, slepé uličky, otázky do dalších sprintů.

# Code style

- Screenshoty VŽDY ukládej do složky screenshots/

## Commit messages

- Conventional Commits (`fix:`, `feat:`, `chore:`, …)
- Subject ≤ 50 znaků, bez tečky na konci
- Body jen když "proč" není zřejmé z diffů — max 2–3 řádky
- Žádné bullet-listy změněných souborů ani vyčerpávající popisy — kdo chce detail, přečte si diff

## Komentáře

- **Default: žádný komentář.** Dobře pojmenovaná proměnná, funkce a typ popisují *co* kód dělá — komentář to jen duplikuje a zastarává.
- Komentář piš **jen když vysvětluje WHY** — skrytý constraint, netriviální invariant, workaround pro konkrétní bug, překvapivé chování pro čtenáře.
- Pokud by odstranění komentáře nikoho nezmátlo, nepiš ho.
- **Stručně**: typicky 1–3 řádky. Odstavce v komentářích jsou varovný signál (kód by měl být čitelnější, ne okomentovanější).
- **Nepiš komentáře u self-explanatory kódu** — getter, mapovací funkce, zřejmá validace, pojmenovaný boolean výraz, typická CRUD operace. I krátký komentář tady jen přidává šum.
- **Nereferencuj kontext, který komentář přežije**: žádné „used by X", „added for the Y flow", „handles case from issue #123". To patří do PR description / commit message, ne do souboru.
- Komentáře piš anglicky.
