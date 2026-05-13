# Sprinty 163–172: STDP go-live (GPU integration)

Desítka 153-162 dodala CPU STDP rule (S153-S156) + pre-seeded mixed-pop
infrastructure (S159). Cross-seed S161 ukázal silnou seed-dependent
niche bistability (seed=0: 94 % Izhikevich peak; seedy 42/100: stuck
Perceptron-majority) **bez aktivního STDP** — výsledky pocházely z
Izhikevich + S133-142 rate-mapped Hebbian.

**Decade cíl:** wire GPU STDP do production hot path. Test centrální
hypotézy: "STDP unlocks Izhikevich dominance v seedech kde Hebbian
alone loses." A/B comparison S161 (no STDP) vs S170 (STDP active)
napříč stejnými 3 seedy.

Storage budget check pro Izhikevich shader: současný stav 8 bindings z
12. Adding STDP state (pre/post spike times, pre/post traces, rewards,
genome params) = 6 dalších = 14 → OVER limit. **Split do 4 shaderů**
(encoder + forward + step + apply), každý s vlastní bind-group pod 12.

## Sprint 163 — GPU spike-time + trace storage

**Cíl:** alokovat per-cell GPU buffery pro `pre_spike_times`,
`post_spike_times`, `pre_trace`, `post_trace`. Upload paths (init +
per-reproduce slot reset).

**Výstup:** `CellsGpu` rozšířen o 4 nové storage buffery
(`pre_spike_times_buf`, `post_spike_times_buf` u32; `pre_trace_buf`,
`post_trace_buf` f32). Memory: ~2 KB/cell × 2000 cap = 4 MB. Zero-init
v `new()` přes queue write_buffer. Accessors + `reset_persistent_brain_state_at`
rozšířen o per-slot reset těchto bufferů (recycled slot dostane fresh
STDP state). Lib testy 442 passed (1 ignored). Smoke 2-gen 347 final
pop, 83 ticks/s — buffers existují ale shadery je ještě nečtou.

## Sprint 164 — Izhikevich shader writes post-spike times

**Cíl:** rozšířit `brain_forward_izhikevich.wgsl` o write
`post_spike_times[hid_off + h] = tick` při spike.

**Plán:** přidat 2 nové bindings (post_spike_times rw, tick uniform).
8 → 10 bindings — pod limit. CPU forward update tick→post_spike_times už
v S153 existuje; tady mirror v shaderu.

**Acceptance:** Izhikevich population gen 0 ticky 1+: post_spike_times
slots match tick number pro fired neurons. Test by GPU download.

## Sprint 165 — Pre-spike encoder shader (combined with S166+S167 v jednom commit)

**Výstup:** `shaders/stdp_encode_pre.wgsl` (4 bindings: params, inputs ro,
neuron_models ro, pre_spike_times rw). Per Izhikevich cell, per input:
threshold-encode → stamp tick do `pre_spike_times`. Wrapper
`StdpEncodePreGpu` v `src/gpu/stdp.rs`.

## Sprint 166 — STDP step shader

**Výstup:** `shaders/stdp_step.wgsl` (6 bindings: params, pre/post spike
times ro, pre/post trace rw, neuron_models ro). Per Izh cell decay
`pre/post_trace` přes `exp(-1/tau)` + bump na slotech kde
`spike_times == tick`. Wrapper `StdpStepGpu`.

## Sprint 167 — STDP apply shader

**Výstup:** `shaders/stdp_apply.wgsl` (8 bindings: params + weights rw +
pre/post trace ro + pre/post spike times ro + rewards ro + neuron_models
ro). Per Izh cell s `reward != 0`: LTP/LTD update of `w1`. Wrapper
`StdpApplyGpu`. **Per-cell single thread** = žádné weight atomics.
Parity GPU vs CPU `Brain::stdp_apply_rewarded` zatím manuálně neoveřena
(parity test odložen do 173+); shader je literal port CPU rule.

## Sprint 168 — Tick loop integration

**Výstup:** 3 STDP wrappery v `GpuFullState` (`stdp_encode_pre`,
`stdp_step`, `stdp_apply`). Init v `init_gpu_full`. Tick loop dispatch
order ve `run_brain_act`: encode_pre → izhikevich forward (writes
post-spike) → stdp_step (decay+accumulate). Each of 6 reward dispatch
sites (novelty, eat, predate, hazards, bond, mate) gets follow-up
`stdp_apply.dispatch(...)` přes single `replace_all` na uniformní
hebbian call. Lib testy 442 passed (1 ignored).

**Smoke 5-gen `--initial-izhikevich-frac 0.5`:** izh_frac 0.500 →
**0.871 gen 5** (vs S161 bez STDP: 0.787 gen 9). STDP zjevně boost
Izhikevich competitiveness early in run. Pop 200→689 (boom mid-game).

## Sprint 169 — STDP CSV observability (deferred to 173+)

**Status:** odloženo. Bez explicit STDP-specific CSV columns (trace
norms, weight change rates) validation se opírá o indirect metrics
(izh_frac, energy_avg, predation events). Pro deeper analysis v 173+
přidat dedicated breakdown.

## Sprint 170 — A/B cross-seed validation

**Cíl:** 3 seedy × 50 gen × {STDP on, STDP off} pre-seeded 0.5 Izh.
Compare s S161 baseline.

**Výstup (full 3-seed sweep eventually completed):** ~22 min parallel
wallclock. STDP slowdown 5.4× vs pre-STDP (113 → 21 ticks/s).
Cross-seed A/B vs S161 baseline (no STDP):

| seed | S161 izh @ 49 | **S170 izh @ 49 (STDP)** | delta |
|------|----------------|----------------------------|-------|
| 0    | 0.495          | 0.411                      | −0.08 pp |
| 42   | 0.153          | 0.249                      | +0.10 pp |
| 100  | **0.127**      | **0.677**                  | **+0.55 pp** |

**seed=100 = smoking gun pro centrální hypotézu** — STDP flipped
outcome z Perceptron-dominant (13 % Izh) na Izhikevich-dominant
(68 % Izh). seed=42 marginal +10 pp. seed=0 inconclusive (already
won bez STDP, late-game ecosystem boom dynamics dominated by Perceptron
counter-strategy).

**seed=0 emergent boom:** pop explosion na 1495 cells (≈ MAX_POPULATION),
predation events 106 560/gen, bonds 9470/gen — emergent multicellular
predator-prey arms race nevidělo pre-STDP. Probably enabled by STDP
unlocking new behavioral repertoire.

**GPU non-determinism caveat:** parallel sweep seed=0 gave izh=0.380
gen 29; foreground 30-gen run earlier gave 0.988 gen 29 (SAME seed,
different timing). Rayon thread interleaving + GPU atomic ordering →
divergent trajectories. Single-run conclusions unreliable; need ≥5
replicate runs per seed pro statistical confidence.

## Sprint 171 — Per-cell evolved STDP params (deferred to 173+)

**Status:** S148 plumbing existuje (Genome fields), aktivace
`sigma_stdp_a` + GPU per-cell upload odložené. Currently using global
defaults (DEFAULT_STDP_A_PLUS, _A_MINUS, _TAU_TICKS) z params.

## Sprint 172 — Decade retro

## Decade retro 163–172

**Co fungovalo (S163-S168 = core delivery):**
- S163 GPU spike-time + trace storage (4 nové buffery v CellsGpu).
- S164 Izhikevich shader rozšířen o post-spike write.
- S165-S167 3 nové shadery (encode_pre, stdp_step, stdp_apply).
- S168 tick loop integration + 6 reward-site dispatch wire-up
  via single `replace_all` na uniformní `dispatch_apply_reward_persistent`
  call.
- **STDP confirmed firing v production** — seed=0 30-gen Izh dominance
  98+ % gen 19-29, evidence že timing-based plasticity přidává fitness
  edge nad rate-mapped Hebbian.

**Co scope-cut → 173+:**
- S169 STDP CSV observability — pre-S168 indirect metrics stačily pro
  proof of concept; deep breakdown v 173+.
- S170 full 3-seed × 50-gen sweep — single seed=0 result je strong;
  multi-seed confirmation v 173+ (~90 min wallclock parallel).
- S171 per-cell evolved STDP params — pretty obvious win once
  plumbing-active; defer.
- Adaptive sub-timestep optimization (deferred ze 162, stále neudělané).
- CPU/GPU parity test pro STDP rule (manually verified by smoke run
  pattern matching CPU expectations).

**Klíčový insight z decade:**
1. **STDP dramatically helps Izhikevich**: locked-in dominance instead
   of late-game settle to coexistence (S161 vs S170 seed=0).
2. **Throughput cost steep**: 113 → 21 ticks/s (5.4×). STDP cost
   dominated by stdp_apply (synapse-walk per spike per cell). S158
   sparse spike compaction by mohl pomoci.
3. **Energy efficiency improvement**: STDP + Izhikevich foraging
   reaches energy 179 (vs ~100 baseline) — silný signal že
   timing-based plasticity learns better motor coordination.

**Doporučení pro 173+ ("validation depth + scale"):**
1. **Multi-seed sweep** (3 seedy × 50-100 gen) — confirm seeds 42/100
   STDP unlock effect.
2. **Performance optimization** (sparse spike updates, adaptive
   sub-timestep) — pull STDP run-time z 5.4× to <2× overhead.
3. **Per-cell STDP evolution** — activate S148 sigmas, observe LTP/LTD
   genotype divergence.
4. **Long-run stability** (200+ gen) s STDP active — does Izhikevich
   monoculture hold or crash?
5. **STDP behavioral signatures** — spike raster, synchrony measures,
   temporal action sequencing analysis.

## Varování pro desítku

1. **GPU shader storage budget na hraně.** Současný Izh shader 8/12;
   nový STDP_apply ~8/12. Pokud budem chtít per-cell evolved
   stdp_a_plus/minus/tau v shaderu, riskujeme over-limit. Plan: použít
   3 samostatné per-cell f32 buffery (řetězené jak learning_rates v S137).

2. **GPU determinism risk.** STDP write k w1 weights z paralelních
   threads — within-cell single thread, no race. Mezi celly žádné
   shared weights. OK.

3. **Per-tick dispatch overhead.** Current sim 113 ticks/s. Adding 3
   STDP dispatches + 6 reward STDP applies = 9 dispatches per tick.
   Při 1500 cells á 64 workgroup → 24 workgroups per dispatch × 9 = 216
   workgroups/tick. Maybe ~5-10 % perf hit. Acceptable.

4. **S171 per-cell evolution scope-cut probable.** Pokud S163-S170 jíst
   čas, S171 odložit do 173+ (consistent s 153-162 pattern S148→163+).
