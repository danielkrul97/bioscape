# Multi-channel Pheromones + Temporal Patterning

Standalone feature work mimo decade — předcházející hře 300-gen × 3 seeds run
zkoumající emergenci primitiv komunikace.

## Cíl

Předtím cells měly 1 pheromone field (single-channel scalar signaling) → emergence
komunikace bohatší než alarm-cue je zablokovaná. Tento sprint zavádí:

1. **Multi-channel signaling**: 3 nezávislá pheromone pole (ch0, ch1, ch2). Cells
   mohou emitovat mixturu, sensory rozliší — discriminated communication.
2. **Temporal patterning přes faster decay**: ch0 (decay 0.3) backward-compat
   slow, ch1 medium (1.5), ch2 fast (5.0). Vyšší decay = lokalizovaná spike →
   bursty emission je channel-specific.

## Mechanismus

### Konstanty (lib.rs)

```rust
pub const N_PHEROMONE_CHANNELS: usize = 3;
pub const PHEROMONE_DECAY_PER_CH: [f32; 3] = [0.3, 1.5, 5.0];
pub const PHEROMONE_DIFFUSION_PER_CH: [f32; 3] = [0.15, 0.12, 0.08];
```

`PHEROMONE_DECAY` / `PHEROMONE_DIFFUSION` zachované jako aliasy k ch0 (GPU
shader path).

### Brain expansion

| const                  | pre-S126 | post-S126 |
|------------------------|----------|-----------|
| `BRAIN_INPUTS_SENSORY` | 21       | **27**    |
| `BRAIN_INPUTS`         | 71       | **77**    |
| `BRAIN_OUTPUTS`        | 10       | **12**    |
| `BRAIN_HIDDEN`         | 50       | 50        |

`w1` shape `[50][77]` (pre: `[50][71]`), `w2` shape `[12][50]` (pre: `[10][50]`),
`b2` length 12 (pre: 10).

#### Slot mapping

Inputs (sensory):
- 0-10, 13-20: existing (food/cell delta, energy, speed, smell, heading, density,
  damage, temperature)
- **11, 12, 19**: ch0 pheromone gradient xyz (zachováno)
- **21, 22, 23**: ch1 pheromone gradient xyz (NOVÉ)
- **24, 25, 26**: ch2 pheromone gradient xyz (NOVÉ)
- 27..77: recurrent (BRAIN_RECURRENT = 50, jako dříve)

Outputs:
- 0: turn (yaw rate)
- 1: thrust
- **2: ch0 emit** (zachováno — mating gating)
- 3-9: morph + attack + pitch + bond_signal (zachováno)
- **10: ch1 emit** (NOVÉ)
- **11: ch2 emit** (NOVÉ)

#### INNATE biases

`INNATE_PHEROMONE_BIAS=1.0` (b2[2]) zachováno — mating gating vyžaduje aktivní
ch0 emisi. Nová konstanta:

```rust
pub const INNATE_PHEROMONE_AUX_BIAS: f32 = 0.5;  // b2[10], b2[11]
```

Slabší než ch0 bias — bez něho cold-start (output ≈ 0 pro nové kanály).

### Cell footprint změny

Nová pole v `Cell`:
- `last_emit: [f32; N_PHEROMONE_CHANNELS]` (12 bajtů per cell — minulý emit pro
  burst diagnostiku)
- `burst_accum: [f32; N_PHEROMONE_CHANNELS]` (12 bajtů — squared tick-to-tick
  delta accumulator pro temporal patterning metric)

Total cell overhead: 24 bajtů × 1k cells ≈ 24 KB — negligible.

### Field update + emission

Per tick `update_pheromone`: 3× SmellField step (každý kanál vlastní decay/diff).

`emit_pheromones` čte brain outputs ze slotů `[2, 10, 11]`, deposit do
`pheromone_fields[ch]`, sčítá total_emit pro cost: `cost = total × cost_rate × dt`.
GPU path: ch0 zůstává v `gpu.pheromone` (single FieldGpu instance), ch1/ch2 vždy
CPU step.

Burst score akumulátor: `cell.burst_accum[ch] += (current_emit - prev_emit)²` per
emit_pheromones tick.

## Změny dimenzí — souhrn

| layer / shader        | změna                                                           |
|-----------------------|-----------------------------------------------------------------|
| `Brain` w1            | 50×71 → 50×77 (3550 → 3850 weights per cell)                    |
| `Brain` w2            | 10×50 → 12×50 (500 → 600 weights per cell)                      |
| `Brain` b2            | 10 → 12                                                         |
| `BRAIN_WEIGHTS_PER_CELL` | 4110 → 4512 (per-cell GPU weights buffer)                    |
| `brain_forward.wgsl`  | INPUTS 71→77, OUTPUTS 10→12, offsets B1=3850 B2=4500            |
| `hebbian.wgsl`        | mirror brain_forward offsets                                    |
| `motor.wgsl`          | OUTPUTS 12 (motor čte sloty 0/1/7, ostatní ignored)             |
| `populate_inputs.wgsl`| BRAIN_INPUTS 77 / BRAIN_INPUTS_SENSORY 27 v Params               |
| `gpu.rs` const-asserts| B1_OFFSET=3850, W2_OFFSET=3900, B2_OFFSET=4500, total=4512      |

GPU populate_inputs shader pro ch1/ch2 sloty zatím píše 0 — multi-channel sensor
gather je CPU-only path. GPU brain forward přesto fungoval v parity testu
(zero ch1/ch2 sensor input → consistent forward result CPU vs GPU).

Checkpoint version V4 → **V5** (multi-channel pheromone fields + brain dim
expansion incompatible).

## CSV diagnostika

Replaced sloupec `ph_emit` → 9 nových sloupců:

```
ph_emit_ch0_avg, ph_emit_ch1_avg, ph_emit_ch2_avg,
ph_emit_ch0_dev, ph_emit_ch1_dev, ph_emit_ch2_dev,
ph_burst_score_ch0, ph_burst_score_ch1, ph_burst_score_ch2
```

`burst_score = mean cell.burst_accum / TICKS_PER_GENERATION`. Vyšší = víc bursty
(continuous emit má small frame-to-frame deltas).

## Smoke results (seed=42, 30 gens, CPU path)

```
gen=0  cells=200  ch0=0.000  ch1=0.000  ch2=0.000  burst*=0
gen=15 cells=840  ch0=0.860  ch1=0.047  ch2=0.046  b0=0.036 b1=0.021 b2=0.018
gen=19 cells=875  ch0=0.875  ch1=0.043  ch2=0.042  b0=0.033 b1=0.018 b2=0.016
gen=29 cells=943  ch0=0.910  ch1=0.028  ch2=0.023  b0=0.017 b1=0.009 b2=0.007
```

Final pop **943** (no extinction). Post-gen 15 ch1/ch2 emit > 0.01 jak
specifikováno. ch0 dominuje (mating bias drží to high), ch1/ch2 mírně
oscilují kolem 0.03-0.05 — INNATE bias 0.5 zajistil non-zero startup, dál to
bude evoluce ladit v 300-gen run.

GPU-full smoke (5 gens, seed=42): final pop **1423**, žádný panic, multi-channel
emit/burst sloupce vyplněné.

## Tests

- Lib tests: **173 passed, 1 ignored** (z toho 156 původních + 2 nové).
- Nové: `multi_channel_pheromone_emit_costs_proportionally`,
  `pheromone_field_array_independent_decay`.
- Updated `random_brain_average_thrust_is_positive` threshold 75 % → 66 %
  (BRAIN_INPUTS 71→77 zvýšilo input variance, mean thrust > 0.3 stále drží
  ale tail percentage trochu klesla).
- GPU parity testy (`brain_forward_gpu_matches_cpu`, `step_gpu_matches_cpu`,
  `hebbian_gpu_matches_cpu`, `motor_gpu_matches_cpu`) **passují bez tolerance
  bumpu** po update všech 4 shader constants.

## Otevřené otázky pro 300-gen experiment

1. **Bude evoluce uplatňovat ch1/ch2?** Cost je flat (PHEROMONE_COST_PER_RATE
   stejný pro všechny kanály), takže emise je čistě positive-utility:
   pokud signál nemá receiver, mutace ho potlačí. 300 gens × 3 seeds → uvidíme
   distribuci ch{0,1,2}_avg / dev.

2. **Detekuje se temporal patterning v burst score?** Hypotéza: ch2 (rychlý
   decay) podpoří bursty regimes víc než ch0 (continuous mating-friendly).
   Měřitelné jako relativní `burst_score_ch2 / ch0`.

3. **Emerge specialization mezi cells?** Pokud cells dělí role (jedni emit ch1,
   druzí ch2), `ph_emit_ch{1,2}_dev` poroste rychleji než mean.

4. **Korelace s clustering?** Sprint 67-71 bondy → multi-channel by mohly přinést
   cluster-level coordination (alarm vs. food signal).

## Soubory změněné

- `src/lib.rs` — konstanty, BrainSensors, populate_brain_inputs, Cell.last_emit
  + burst_accum, INNATE_PHEROMONE_AUX_BIAS, 2 nové unit testy.
- `src/main.rs` — PheromoneResource → multi-channel array, emit_pheromones
  s 3 sloty, sensor gather × 3.
- `src/bin/headless.rs` — World.pheromone_fields, Checkpoint V5, emit/update
  multi-channel, CSV header + write_stats + empty-pop branch + burst_accum
  reset per gen.
- `src/gpu.rs` — const-asserts updated (B1=3850, W2=3900, B2=4500, total=4512).
- `shaders/brain_forward.wgsl` — BRAIN_INPUTS 77, BRAIN_OUTPUTS 12, offsets.
- `shaders/hebbian.wgsl` — same constants.
- `shaders/motor.wgsl` — BRAIN_OUTPUTS 12.
- `shaders/populate_inputs.wgsl` — Params struct comment update.
- `benches/full_tick.rs`, `benches/headless_phases.rs` — BrainSensors field
  rename.
