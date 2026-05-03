# Bioscape

Simulace evoluce, která má pomoct pochopit, jak vzniká inteligence. Žádný předem daný cíl, žádná „buď chytrý" fitness — jen prostředí, replikace, mutace, selekce a hodně času.

## Cíl

Postavit otevřený systém, ve kterém se z primitivních agentů (něco jako buňky) vyvine smysluplné chování — sami od sebe. Tři velké otázky, které projekt sleduje:

1. Jak se z chemie stane buňka?
2. Jak se z buněk stanou těla a mozky?
3. Jak se z mozků stane inteligence?

## Architektura

- **Jazyk:** Rust (CPU i GPU)
- **Engine:** [Bevy 0.18](https://bevyengine.org) (ECS + 2D rendering přes wgpu), trimmed feature set bez `bevy_gilrs` a `audio` (žádné systémové `libudev-dev` / `libasound2-dev`)
- **Split kódu:**
  - `src/lib.rs` — čistá simulační logika (`Cell`, `step()`, …) bez Bevy, pojede i headless
  - `src/main.rs` — Bevy app, ECS svět + `Transform` synchronizace
- **GPU compute:** opt-in feature `gpu` (přímý wgpu + bytemuck + pollster) pro vlastní kernely; Bevy už wgpu táhne pro rendering

## Spuštění

```bash
# Default běh (debug, bez GPU compute kernelů)
cargo run

# Rychlejší inkrementální iterace (dynamic linking Bevy)
cargo run --features dev

# S GPU compute kernely
cargo run --features gpu

# Release pro reálné experimenty
cargo run --release
```

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

Early stage. Architektura je rozhodnutá, simulační logika v `src/lib.rs` se rozjíždí.

## Licence

MIT OR Apache-2.0
