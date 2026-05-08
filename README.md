# Bioscape

An evolution simulation built to help understand how intelligence emerges. No predefined goal, no "be smart" fitness — just an environment, replication, mutation, selection, and a lot of time.

## Goal

Build an open-ended system in which meaningful behavior evolves on its own from primitive agents (something like cells). Three large questions the project pursues:

1. How does a cell arise from chemistry?
2. How do bodies and brains arise from cells?
3. How does intelligence arise from brains?

## Architecture

- **Language:** Rust (CPU + GPU)
- **Engine:** [Bevy 0.18](https://bevyengine.org) (ECS + 3D rendering via wgpu — orbit Camera3d, StandardMaterial, ellipsoid bodies), trimmed feature set without `bevy_gilrs` and `audio` (no system `libudev-dev` / `libasound2-dev`)
- **Code split:**
  - `src/lib.rs` — pure simulation logic (`Cell`, `step()`, brain, world map, smell/pheromone fields …) without Bevy. Single source of truth for simulation constants + helpers (`populate_brain_inputs`, `pair_fertile`, `make_mating_child`).
  - `src/main.rs` — Bevy app, ECS world + `Transform` synchronization, 3D renderer.
  - `src/bin/headless.rs` — windowless harness for batch experiments (CSV log per generation, deterministic seed).
- **GPU compute:** opt-in `gpu` feature (raw wgpu + bytemuck + pollster) for custom kernels; Bevy already pulls in wgpu for rendering

## Running

**Renderer (3D viz):**

```bash
# Default run — `gpu` feature active, GPU compute pipeline default-on
# (sensor + populate + brain + motor + brownian + step on GPU,
# single-Wait readback per tick).
cargo run --release

# Force CPU SIMD path (for comparison or on adapters without compute).
BIOSCAPE_GPU_FULL=0 cargo run --release

# Faster incremental iteration (Bevy dynamic linking).
cargo run --features dev
```

Controls: **left button + drag** = orbit, **middle button + drag** = pan, **scroll wheel** = zoom (orthographic scale), **WASD/arrows** = pan from keyboard.

**Headless (batch experiments, CSV log):**

```bash
# Arguments: <seed> <max_gens> <out.csv>
cargo build --release --bin headless
./target/release/headless 0 200 /tmp/run.csv
```

The CSV contains per-generation stats: pop, lineages, body morphology, predation events, brain adoption metrics, and more.

## Documentation

A scientific-context survey for the project (in Czech, accessible to laypeople too) lives in [`docs/`](docs/README.md):

- [Introduction and motivation](docs/00-uvod.md)
- [Foundations of evolution](docs/01-evoluce-zakladny.md)
- [Artificial life: Tierra, Avida, Karl Sims, Lenia, Stanford DERL](docs/02-umely-zivot.md)
- [Neuroevolution: NEAT, HyperNEAT, MAP-Elites](docs/03-neuroevoluce.md)
- [Cells and morphogenesis: Neural CA, Levin, bioelectricity](docs/04-bunky-a-morfogeneze.md)
- [Neurons and the brain: Izhikevich, spiking networks, plasticity](docs/05-neurony-a-mozek.md)
- [Intelligence and embodiment: Free Energy Principle, Baldwin effect](docs/06-inteligence-a-embodiment.md)
- [Open-ended evolution: novelty search, quality-diversity](docs/07-open-ended-evolution.md)
- [Implementation in Rust on GPU](docs/08-implementace-rust-gpu.md)

## Status

120+ sprints. Full 3D simulation: ellipsoid morphology (length × width × height + spike), 3D motion (yaw + pitch), gravity with buoyancy, predation with attack-gate + gradient exposure based on bond count, mating via pheromone signaling, food clustering in the world map, hazard zones, thermal field (vertical gradient + diurnal/seasonal oscillation, per-cell `thermal_optimum` gene), multi-trophic food (plant / carrion / hunter-carrion + evolutionary `carnivore_score`), recurrent brain (21 sensory + recurrent inputs × NEAT-grown hidden × 10 outputs, Elman feedback), HyperNEAT CPPN templates, cluster-shared brain pooling across bonded peers (proto-distributed cognition), persistent spring bonds → tissue regime, evolving Hunter with its own brain (biological arms race), bistable cell-state, periodic environmental shocks (HazardPulse / ClimateShift / FoodCrash). Headless harness for deterministic batch experiments, 3D renderer with orbit camera (HDR + bloom + fog + procedural bio-textures).

**Performance decade 111–120** delivered **~9× ticks/s @ 2,500 cells**
via target-cpu=native (S111), SIMD brain forward (S112, 2.1×), Bevy
rayon parallelization (S113, 4.6× per phase), SIMD field diffusion (S117,
4×), plus PGO infrastructure (S119, opt-in). Baseline on i5-12400F
(12 threads, target-cpu=native, lto=fat): 1k cells = 2,994 ticks/s,
2.5k = 1,408, 5k = 598. Detail: [`docs/sprints/111-120-perf.md`](docs/sprints/111-120-perf.md).

**Decade 128–137 (renderer-side perf)** continues: per-system scratch
reuse in hot loops (Local + persistent Resources, ~30–40 alloc/tick →
0), `--gpu-full` single-Wait pipeline default-on in the renderer (replaces
the fragmented S132 path), `eat_food` solo-skip via the sensor cache.
Detail: [`docs/sprints/128-137-perf.md`](docs/sprints/128-137-perf.md).

Detailed sprint-by-sprint status: [`docs/sprints/`](docs/sprints/).

## Emergent behaviors

Measured empirically — three 300-generation headless runs (seeds 0, 1, 42) plus a 5-minute renderer screencast (`screenshots/screencast.mp4`), **at the sprint-86 code state**. Subsequent sprints (87+: thermal sensor + per-cell optimum gene, Hunter brain + evolution, multi-trophic food, gradient hunter exposure, cluster-shared brain pooling, NEAT-style brain growth) changed selection pressure substantially — these findings would need remeasurement against the current code. What selection actually produces, none of it directly coded. Each entry includes a biological analogy.

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

## License

MIT OR Apache-2.0
