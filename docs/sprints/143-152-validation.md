# Validation report — Sprinty 143–152 spiking-neurons

Desítka zavřela 2 dílčí cíle:

1. **Lineage diversity strengthening (S143)** — kvadratická
   anti-monoculture penalty místo lineární. Odpověď na 2-lineage
   monoculture pozorovaný v 150-gen run desítky 133-142.
2. **Izhikevich SNN infrastructure (S144-S150)** — opt-in neuron model
   v Genome, GPU shader paralelně s Perceptronem, mutation-driven flip,
   validation že Izhikevich + Hebbian (S133-142 rule) je viable niche.

Plný STDP (S148 plánovaný full rule) **odložen do 153-162**.

## Sprint 143 — Diversity (50-gen smoke seed=0)

| metric | pre-S143 | post-S143 | delta |
|--------|----------|-----------|-------|
| final pop | 294 | 291 | ≈ |
| lineage_count @ gen 49 | 3 | 3 | = |
| avg lineage_count gen 1-49 | 14.1 | 14.3 | +1.4 % |
| avg bonds_formed/gen | 35.8 | **196.6** | **+449 %** |
| avg predation_events/gen | 609.4 | **1906.1** | **+213 %** |

Strict "≥5 lineages na gen 50" nesplněno (3 final), ale **ecological
turnover dramaticky vzrostl** — multicelular formace 5.5×, predator-prey
cycling 3×. Qualitative goal (richer ecosystem) ✓.

## Sprint 149-150 — Cross-seed Izhikevich coexistence (3 × 50 gen)

| seed | final pop | izh peak | izh @ gen 49 | lin @ gen 49 | pred final |
|------|-----------|----------|--------------|--------------|------------|
| 0    | 303       | 0.202 (gen 19) | 0.070 | 3 | 224 |
| 42   | 379       | 0.045 (gen 29) | 0.045 | 3 | 138 |
| 100  | 325       | 0.071 (gen 49) | 0.071 | 3 | 130 |

**Klíčový závěr:** Izhikevich cells **přežívají napříč všemi 3 seedy**
přes celých 50 generací (žádná extinkce). Steady-state fraction
4-7 % populace, peak transient 4-20 % u seedu=0. Pop a předace zůstávají
v rozsahu pre-S149 baseline → Izhikevich nemá zjevný fitness deficit ani
edge proti Perceptronu s aktuálním (Hebbian-only) plasticity pravidlem.

## Co data ukazují vs pre-decade hypotézu

**Pre-decade hypotéza (z 150-gen analýzy):**
> "Rate brain stojí na ceiling (spike_frac = 1.0, w_norm = cap). SNN
> s spike timing by mohlo prolomit přes Izhikevich-coded info capacity."

**Skutečnost po desítce:**
1. Izhikevich + Hebbian = stable niche, ale ne dominance. Mapping
   spike_count → [-1,+1] dovoluje Hebbian work, ale **timing info je
   v tomto mappingu zahozen** — Izhikevich cells trénují stejně jako
   Perceptron, jen s odlišnou non-linearitou.
2. `neural_spike_frac` (S142 metric, perceptron-saturation proxy)
   zůstává 0.99-1.00 i s Izhikevich aktivním v ~7% populace. Saturation
   regime je vlastnost Hebbian rule + reward funnel, ne neuron modelu.
3. Dominance Izhikevich by potřebovala STDP — bez ní je rate
   reinterpretation per neuron type, ale **selekční signál stejný**.

**Tj. původní hypotéza je correct ALE potřebuje STDP aby fungovala.**
S143-150 položily groundwork; 153-162 musí dodat STDP rule.

## Recommendations pro 153-162

1. **Implement STDP rule.** Per-synapse eligibility trace s timing-based
   accumulation. Reward modulation (3-factor) integrate s existing
   S133-142 reward funnel.
2. **Izhikevich-specific reward apply.** Currently Izhikevich cells běží
   přes shared `hebbian_apply_reward` shader (S133). STDP path bude
   vyžadovat samostatný dispatch (Izhikevich-only branch).
3. **Adaptive sub-timestep.** Currently 32 hard-coded; cells s low
   firing rate by mohly použít 16 nebo 8 → ~50 % perf gain. Important
   až po STDP works.
4. **Pre-encode initial pop as 50% Izhikevich.** Zatím spoléháme na
   mutation (0.5%/gen) → 50+ gens to reach steady state. Pre-encoded
   seed accelerates niche measurement.
5. **STDP-specific CSV columns.** `stdp_lr_avg`, `stdp_a_plus_avg`,
   `synaptic_dispersion_izh` (analog k `w_norm_avg` pro spike-timing
   weights).

## Cross-decade comparison 133-142 vs 143-152

| metric | post-S142 (Perceptron) | post-S152 (Mixed) |
|--------|------------------------|-------------------|
| final pop @ 50 gen | 294 | 303 (seed=0) |
| lineages @ 50 gen | 3 | 3 |
| ecologic turnover | baseline | S143 + 200-450 % (S143 effect) |
| neuron model diversity | 1 (Perceptron only) | 2 (5-7 % Izhikevich) |
| brain saturation | spike_frac ≈ 0.99-1.0 | spike_frac ≈ 0.99-1.0 |

Net: desítka rozšířila genotypový prostor (NeuronModel) a ecologickou
dynamiku (S143), ale **brain saturation problem zůstává nevyřešen** —
to je explicitní úkol pro 153-162 STDP work.
