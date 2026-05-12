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

## Sprint 165 — Pre-spike encoder shader

**Cíl:** new shader `stdp_encode_pre.wgsl`. Per cell, per input: if
`inputs[i] > SPIKE_ENCODE_THRESHOLD && neuron_models[cell] == Izh`,
write `pre_spike_times[cell × BRAIN_INPUTS + i] = tick`.

**Plán:** 4 bindings (params, inputs ro, neuron_models ro,
pre_spike_times rw). Dispatch per cell per tick before forward
(or after, ale before stdp_step).

**Acceptance:** Izh cells emit pre-spike events for inputs > 0.5.
Perceptron cells netknuty.

## Sprint 166 — STDP step shader

**Cíl:** GPU mirror `Brain::stdp_step` — decay + accumulate per-neuron
traces.

**Plán:** new shader `stdp_step.wgsl`. 7 bindings (params, pre_spike_times
ro, post_spike_times ro, pre_trace rw, post_trace rw, neuron_models ro,
genome_tau_buf ro). Genome `stdp_tau_ticks` přes new `stdp_taus_buf`
v CellsGpu, OR uniform if perf nepouští evolved tau (S148 plumbing).

**Acceptance:** GPU trace evolution matches CPU `stdp_step` výsledky
ε=1e-4 přes 10 ticks fixture.

## Sprint 167 — STDP apply shader

**Cíl:** GPU mirror `Brain::stdp_apply_rewarded` — LTP/LTD update of w1
weights, gated by per-cell reward + Izhikevich model.

**Plán:** new shader `stdp_apply.wgsl`. Bindings (params, brain_weights
rw, pre_trace ro, post_trace ro, pre_spike_times ro, post_spike_times
ro, rewards ro, neuron_models ro). Genome `a_plus`/`a_minus` přes nové
GPU buffery OR uniform constants.

**Per-synapse atomics:** LTP `Δw[h][i] += a_plus × pre_trace[i]` může
mít multi-cell-contention pokud paralelní threads píší stejné w1
slot — ale weights jsou per-cell, takže žádná inter-cell contention.
WITHIN cell: jeden thread per cell zatím (workgroup_size 64), žádné
intra-cell race.

**Acceptance:** parity test stdp_apply_gpu vs CPU ε=1e-4.

## Sprint 168 — Tick loop integration

**Cíl:** dispatch order v `World::run_brain_act`:
1. `stdp_encode_pre.dispatch()` — pre-spike from inputs
2. `gpu.izhikevich.dispatch()` — forward, writes post-spike
3. `stdp_step.dispatch()` — trace bookkeeping
4. STDP apply happens po reward dispatch (each existing reward dispatch
   site adds `stdp_apply.dispatch()` alongside)

**Plán:** modify world.rs to add 3 new dispatches per tick + STDP apply
at 6 reward sites (eat, novelty, predate, hazards, bond, mate).

**Acceptance:** seed=0 1-gen smoke byte-identical s S162 (all-Perceptron
default → STDP shaders early-exit). Pre-seeded smoke shows STDP
dispatches firing.

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
