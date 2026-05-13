# Validation report — Sprinty 153–162 STDP + spiking populace

Decade přinesl:
1. **CPU STDP rule** (S153-S156) — pre-spike encoder, per-neuron traces,
   classical pair-based STDP, reward-modulated variant.
2. **Pre-seeded mixed pop** (S159) — `--initial-izhikevich-frac` CLI
   flag pro bypass mutation bootstrap.
3. **Cross-seed sweep** (S161) — 3 seedy × 50 gen s 50% pre-seeded
   Izhikevich.

**GPU STDP shader (S157)** + **production wire-up reward dispatch**
**odložené do 163+**. Bez nich CPU STDP rule není v hot path
production simu — všechny S161 výsledky pocházejí z Izhikevich +
S133-142 Hebbian (rate-mapped přes spike-count → [-1, +1] hidden).

## Cross-seed S161 výsledky (3 × 50 gen, `--initial-izhikevich-frac 0.5`)

| seed | izh @ gen 0 | izh peak | izh @ gen 49 | final pop | lin @ 49 |
|------|-------------|----------|--------------|-----------|----------|
| 0    | 0.500       | **0.944 (gen 19)** | 0.495 | 428 | 4 |
| 42   | 0.500       | 0.464 (gen 9) | 0.153 | 353 | 3 |
| 100  | 0.500       | 0.500 (gen 0) | 0.127 | 393 | 4 |

**Acceptance criterion** ("≥30 % v steady state ≥ 1 z 3 seedů") **splněn**
— seed=0 ends at 49.5 % steady state s peak 94.4 %.

## Klíčový insight: pre-seeding > mutation bootstrap

| metric | S149 (mutation bootstrap) | S161 (pre-seeded 50 %) |
|--------|--------------------------|------------------------|
| seed=0 izh peak | 0.202 (gen 19) | **0.944 (gen 19)** |
| seed=0 izh @ 49 | 0.070 | **0.495** |
| seed=42 izh peak | 0.045 | 0.464 |
| seed=42 izh @ 49 | 0.045 | 0.153 |
| seed=100 izh peak | 0.071 | 0.500 |
| seed=100 izh @ 49 | 0.071 | 0.127 |

Pre-seeding dává Izhikevich **early-mover advantage**: před tím než
Perceptron drift najde stable strategy, Izhikevich už má funkční lineages.
S149 mutation bootstrap (~0.5 %/gen flip) naopak Izhikevich nikdy nedosáhne
critical mass — Perceptron majority dominate first.

## Seed-dependent niche divergence

3 seedy vykazují kvalitativně RŮZNÉ outcomes — to je big finding:

- **seed=0**: Izhikevich populace najde dominant policy, drží 70-94 %
  pop pro 20+ generations, settles to 50 % coexistence v late game.
- **seed=42**: Izhikevich loses early competition, Perceptron majority
  85-95 % gen 19+, slow recovery jen do 15 %.
- **seed=100**: Worst case pro Izhikevich — populace crash na 4 % gen 9,
  near-extinction (0.4 % gen 19), slow comeback k 13 %.

Tahle variance napříč seedy ukazuje že:
1. **Izhikevich + Hebbian je viable competitor** — when initial conditions
   favor it, Izhikevich can dominate.
2. **Selekce má locally bistable attractors** — same parameters, different
   outcomes. To je sám o sobě interesting finding pro Bioscape biology.

## Co STDP MOHL přinést navíc

Pre-decade hypotéza: STDP enables Izhikevich to **win** in seeds where
rate-Hebbian alone loses. S161 data ukazuje:
- seed=0 už wins WITHOUT STDP → STDP by mohlo posunout > 50 % steady state
- seedy 42/100 stuck v Perceptron-majority → STDP by mohlo "unlock"
  Izhikevich path

Aby tuhle hypotézu otestovali, **potřebujeme GPU STDP integration**
(S157 odložené). Bez ní CPU rule existuje jako spec, ale production
sim neběží.

## Cross-decade trajectory

| metric | post-S142 | post-S152 | post-S162 |
|--------|-----------|-----------|-----------|
| neuron model diversity | 1 (Perceptron) | 2 (~5 % Izh) | 2 (5-50 % Izh) |
| ecological turnover | baseline | +S143 quadratic | same |
| brain saturation | spike_frac ≈ 1.0 | spike_frac ≈ 1.0 | spike_frac ≈ 1.0 |
| STDP capability | ❌ | ❌ | ✅ CPU only |
| niche bistability | ❌ | ❌ | **✅ observed** |

## Recommendations pro 163+

1. **GPU STDP shader (S157 done properly).** Mirror CPU `stdp_step` +
   `stdp_apply_rewarded`. Per-cell branching on `neuron_models[i]`.
   Storage budget OK pod 12 bindings.
2. **Reward dispatch routing.** World tick loop must dispatch CPU/GPU
   STDP for Izhikevich cells AND existing Hebbian for Perceptron — dual
   plasticity pipeline.
3. **STDP-specific CSV breakdown.** Per-model lr / weight stats, spike
   rate distribution per cell.
4. **Targeted reproducibility study.** Why does seed=0 favor Izhikevich
   but seed=42 favor Perceptron? Investigate initial weight distribution
   correlations.
5. **Longer runs (150-300 gen)** s pre-seeded mix — verify if mid-game
   coexistence (seed=0 ~50 %) holds, or if one model eventually wins.
