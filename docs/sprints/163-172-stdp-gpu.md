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

## Sprint 169 — STDP CSV observability

**Cíl:** nové CSV sloupce pro STDP-specific metrics.

**Plán:** `stdp_trace_norm_avg_izh` (mean |pre_trace| + |post_trace|
across Izh cells), `stdp_w1_change_avg_izh` (avg |Δw1| / gen pro Izh
cells), `spike_rate_izh_avg` (mean spikes/tick across Izh population).
GPU readbacks: trace + brain_weights deltas at gen end.

**Acceptance:** 3+ nové sloupce. Perf regression < 10 %.

## Sprint 170 — A/B cross-seed validation

**Cíl:** 3 seedy × 50 gen × {STDP on, STDP off} pre-seeded 0.5 Izh.
Compare s S161 baseline.

**Plán:** flag `--stdp-enabled=true|false` (default true). Run 6 sims
(3 seedy × 2 conditions). Validation report compares izh_frac
trajectories, predator/forager differentiation, fitness metrics.

**Acceptance:** v 1+ seedu kde S161 Izh ztratil (42 nebo 100), s STDP
on Izh frakce > 30 % v steady state. Confirms STDP fitness edge.

## Sprint 171 — STDP per-cell evolved params

**Cíl:** S148 plumbing aktivovat. `sigma_stdp_a > 0` + GPU per-cell
upload `stdp_a_plus_buf`, `stdp_a_minus_buf`, `stdp_tau_buf`. Cells
evolve STDP signatures.

**Plán:** zapnout sigmy v `MUTATION_CONFIG`, plumb buffers do CellsGpu,
update stdp_step + stdp_apply shaders read from per-cell buffers.

**Acceptance:** 50-gen smoke ukazuje non-trivial drift `stdp_a_plus_avg`
od init default napříč seedy.

## Sprint 172 — Decade retro + 173+ outline

**Cíl:** retrospektiva, výhled.

**Plán:** decade retro v `docs/sprints/163-172-stdp-gpu.md` + validation
report `163-172-validation.md`. 173+ outline (dendritic compartments?
multi-channel STDP? long-term memory across generations? evolutionary
dynamics na 1000 gen?).

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
