# Bioscape

Simulace evoluce, která má pomoct pochopit, jak vzniká inteligence. Žádný předem daný cíl, žádná „buď chytrý" fitness — jen prostředí, replikace, mutace, selekce a hodně času.

## Cíl

Postavit otevřený systém, ve kterém se z primitivních agentů (něco jako buňky) vyvine smysluplné chování — sami od sebe. Tři velké otázky, které projekt sleduje:

1. Jak se z chemie stane buňka?
2. Jak se z buněk stanou těla a mozky?
3. Jak se z mozků stane inteligence?

## Architektura

- **Jazyk:** Rust (CPU i GPU)
- **Engine:** [Bevy 0.18](https://bevyengine.org) (ECS + 3D rendering přes wgpu — orbit Camera3d, StandardMaterial, ellipsoid těla), trimmed feature set bez `bevy_gilrs` a `audio` (žádné systémové `libudev-dev` / `libasound2-dev`)
- **Split kódu:**
  - `src/lib.rs` — čistá simulační logika (`Cell`, `step()`, brain, world map, smell/pheromone fields …) bez Bevy. Single source of truth pro simulační konstanty + helpery (`populate_brain_inputs`, `pair_fertile`, `make_mating_child`).
  - `src/main.rs` — Bevy app, ECS svět + `Transform` synchronizace, 3D renderer.
  - `src/bin/headless.rs` — bezokenní harness pro batch experimenty (CSV log per generaci, deterministický seed).
- **GPU compute:** opt-in feature `gpu` (přímý wgpu + bytemuck + pollster) pro vlastní kernely; Bevy už wgpu táhne pro rendering

## Spuštění

**Renderer (3D viz):**

```bash
# Default běh
cargo run --release

# Rychlejší inkrementální iterace (dynamic linking Bevy)
cargo run --features dev

# S GPU compute kernely
cargo run --features gpu
```

Ovládání: **levé tlačítko + drag** = orbit, **střední tlačítko + drag** = pan, **kolečko** = zoom (orthographic scale), **WASD/šipky** = pan z klávesnice.

**Headless (batch experimenty, CSV log):**

```bash
# Argumenty: <seed> <max_gens> <out.csv>
cargo build --release --bin headless
./target/release/headless 0 200 /tmp/run.csv
```

CSV obsahuje per-generaci stats: pop, lineages, body morphology, predation events, brain adoption metriky a další.

## Dokumentace

Rešerše vědeckého kontextu projektu (česky, srozumitelně i pro laika) je v [`docs/`](docs/README.md):

- [Úvod a motivace](docs/00-uvod.md)
- [Základy evoluce](docs/01-evoluce-zakladny.md)
- [Umělý život: Tierra, Avida, Karl Sims, Lenia, Stanford DERL](docs/02-umely-zivot.md)
- [Neuroevoluce: NEAT, HyperNEAT, MAP-Elites](docs/03-neuroevoluce.md)
- [Buňky a morfogeneze: Neural CA, Levin, bioelektřina](docs/04-bunky-a-morfogeneze.md)
- [Neurony a mozek: Izhikevich, spiking sítě, plasticita](docs/05-neurony-a-mozek.md)
- [Inteligence a embodiment: Free Energy Principle, Baldwin effect](docs/06-inteligence-a-embodiment.md)
- [Open-ended evolution: novelty search, quality-diversity](docs/07-open-ended-evolution.md)
- [Implementace v Rustu na GPU](docs/08-implementace-rust-gpu.md)

## Status

40+ sprintů. Plná 3D simulace: ellipsoid morfologie (length × width × height + spike), 3D pohyb (yaw + pitch), gravitace s vztlakem, predace s attack-gate, mating přes pheromone signaling, food clustering ve world map, hazard zóny, recurrent brain (20 sensory + 16 recurrent vstupů × 16 hidden × 9 výstupů, Elman feedback). Headless harness pro deterministické batch experimenty, 3D renderer s orbit kamerou.

Detailní stav po sprintech: [`docs/sprints/`](docs/sprints/).

## Licence

MIT OR Apache-2.0
