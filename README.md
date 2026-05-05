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

80+ sprintů. Plná 3D simulace: ellipsoid morfologie (length × width × height + spike), 3D pohyb (yaw + pitch), gravitace s vztlakem, predace s attack-gate, mating přes pheromone signaling, food clustering ve world map, hazard zóny, recurrent brain (20 sensory + 16 recurrent vstupů × 16 hidden × 9 výstupů, Elman feedback), persistentní spring bondy → tissue regime, makropredátor (Hunter), bistabilní cell-state. Headless harness pro deterministické batch experimenty, 3D renderer s orbit kamerou.

Detailní stav po sprintech: [`docs/sprints/`](docs/sprints/).

## Emergent behaviors

Measured empirically in the **current code state** — three 300-generation headless runs (seeds 0, 1, 42) plus a 5-minute renderer screencast (`screenshots/screencast.mp4`). What selection actually produces, none of it directly coded. Each entry includes a biological analogy.

**Highly reproducible across seeds (CV < 5 %):**

1. **Speed locks at the cap** — `spd_avg` converges to ~189 (cap = 200) by gen 100. Without the cap, mutation drift would carry it past hunter speed (historically observed). *Like the cheetah hitting its respiratory ceiling — selection presses hard against a hard physical wall.*
2. **Pheromone emission saturates to 1.0** — `ph_emit` rises 0.6 → 0.92 → 1.00. Mating-gated emission selects for "shouting"; by gen 100 every fertile cell is fully loud. *Like peacock tails or frog calls — a runaway signaling spiral where the only stable state is "maximum volume," even though it draws predators.*
3. **Lineage collapse** — 198 → 54 (gen 1–30) → 17 (gen 31–100) → 5 (gen 101–300). Diversity is destroyed monotonically. *Population bottleneck, like cheetahs (~10 effective lineages globally) or post-glacial European trees — a single dominant strategy crowds out the rest.*
4. **Hunter pressure persists** — 12 hunters alive throughout; attacks grow 368 → 454 → 573/gen as cells lose defenses. *Keystone predator dynamics — wolves in Yellowstone keep elk in check; remove them and the whole trophic web reorganizes.*
5. **Energy plateau** — `energy_avg` stabilizes at ~80 in late phase regardless of seed. *Carrying capacity — population finds the steady-state balance of food intake vs. metabolic cost.*
6. **Adhesion-type diversity preserved** — entropy ~0.99 throughout (all 8 types active) despite lineage collapse. Type is decoupled from lineage. *Like blood-group polymorphism in humans — multiple alleles persist because none has a universal advantage.*

**Trajectories — what gets selected away:**

7. **Vision is abandoned** — `vis_avg` 47 → 28 → 13. Long-range sensing becomes redundant once the pheromone field is everywhere. *Like blind cave fish — eyes are expensive and pointless when the environment provides equivalent info more cheaply.*
8. **Spikes are abandoned** — `spk_avg` 0.42 → 0.11. With predation gating dampened and clusters collapsed, weapons have no payoff. *Like flightless birds on predator-free islands (dodo, kakapo) — drop costly traits the moment they stop earning their keep.*

**Seed-dependent (high variance):**

9. **Body shape bifurcates** — late `asp_avg` is either elongated (~4.5, seeds 0 & 42) or round (~1.2, seed 1). Two attractors, not one; the earlier "needle convergence" (asp ≈ 12) does not reproduce at this scale. *Like polymorphic butterflies — different starting conditions lock into different stable morphs.*

**Visual-only finding (from screencast):**

10. **Differential adhesion clustering** — same-color cells (= same `adhesion_type`) physically aggregate into 2–5 cell clumps even when no formal bonds are present (`bond_active_frac` ≈ 0). The screencast shows multiple yellow clusters, red pairs, green/teal/purple triplets — sorting purely from spatial dynamics + cluster-spawn placement. CSV doesn't capture this (entropy is global, not local autocorrelation); only the visual reveals it. *Steinberg's differential adhesion hypothesis from embryology — like-attracts-like cells spontaneously sort like oil and water, the historical foundation of tissue self-assembly.*

**Failures of recent mechanisms:**

11. **Tissue regime collapses long-term** — `bond_active_frac` peaks at 0.06–0.11 (gen 30–60), decays to **0.00 by gen 200** in every seed. Sprint 78's food-share produces transient bonding but cannot sustain it. *Like reversion in some choanoflagellates — multicellular phases form, then revert under selection for solo speed.*
12. **Cell-state goes fully selfish** — Sprint 80 `state_avg` drops 0.06 → 0.01 → 0.00; `altruist_frac` follows. The bistable system is bimodal early (`state_dev` ≈ 0.18, both attractors populated) and collapses onto the selfish attractor by gen 200. The 30-generation smoke that showed altruist dominance was a **transient**. *Tragedy of the commons / cheater takeover in microbial communities — without enforcement, free-riders out-compete cooperators every time.*

**Reading these results:** the simulation reliably reproduces predator-driven simplifying dynamics — diversity collapse, signal saturation, sensor abandonment, cooperation breakdown. Mechanisms aimed at multicellularity (food-sharing, phenotypic memory) work as transients but do not yet hold a tissue regime against long-run hunter pressure. The one cooperative-looking pattern that *does* persist — same-type spatial clustering — comes for free from physics, not from any of the explicit cooperation machinery. That gap is the open research question.

## Licence

MIT OR Apache-2.0
