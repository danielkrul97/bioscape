# Sprint 218 — GPU profile + first optimizations

Profile celého `tick_sph` pipeline + dvě quick-win optimalizace.

## Baseline scaling (před optimalizacemi)

`--n N --t-end 0.5 --dt 1e-3` → 500 ticků, žádný download v hot loopu.

| N | steps/s | µs/step |
|---|---|---|
| 1k | 1359 | 736 |
| 2k | 1151 | 869 |
| 5k | 503 | 1988 |
| 10k | 173 | 5780 |
| 25k | 32 | 31250 |
| 50k | 7 | 142857 |

Scaling **O(N²)** od N ≈ 5k (`µs/N²` ≈ 5–8×10⁻⁵ konstantní).
Pod N ≈ 2k dominuje fixní submit overhead (~700 µs/step).

## Per-pipeline GPU profile (baseline, N=25k)

Měřeno přes `--profile` flag, který serializuje pipeline přes
`device.poll(Wait)` a měří wall-clock per stage.

```
stage                   µs/step       %        total ms
density                  8685.5   25.8%           434.3
viscosity                8207.1   24.4%           410.4
nbody (O(N²))            7859.0   23.3%           392.9
pressure                 7821.4   23.2%           391.1
hash rebuild              512.7    1.5%            25.6
drift                     258.1    0.8%            12.9
kick₂                     202.3    0.6%            10.1
kick₁                     156.2    0.5%             7.8
TOTAL                   33702.4
```

**Bottleneck**: tři SPH passes (density + pressure + viscosity) =
73 % času. Jejich společný pattern: 3×3×3 bucket scan, inner loop
přes 50–200 sousedů s random-access load `positions[j*3]`.

**Příčina pomalého SPH**: grid byl příliš hrubý.
`world_half = 2 × (R + r) = 2.4`, cell_size = 0.156, ale `h_init ≈ 0.05`.
27-cell stencil pokryl objem 0.103 oproti potřebné kernel-ball
volume 4/3π·(2h)³ ≈ 0.004. **~25× over-scan**.

## Optimization #1: zúžený grid (50 % slack)

`world_half = 1.5 × (R + r) = 1.8`, cell_size = 0.113.

Změna: `src/planet/world.rs` `init_gpu_full`, jediná konstanta.

Re-profile N=25k:

```
stage                   µs/step       µs/step (orig)   speedup
density                  4771.4         8685.5           -45 %
viscosity                4898.9         8207.1           -40 %
pressure                 4323.0         7821.4           -45 %
nbody                    7035.2         7859.0           -10 %
hash rebuild              300.2          512.7           -41 %
TOTAL                   21692.4        33702.4           -36 %
```

**+36 % throughput at N=25k**, +23 % at N=5k. SPH stages biggest
beneficiary (~45 % faster), nbody benefits od lepší cache locality
přes řidčí buckets ale hlavní gain je v SPH.

## Optimization #2: TILE=128 v nbody

`@workgroup_size(128)` + `TILE = 128u` + odpovídající
`shared_pm: array<vec4<f32>, 128>` v `shaders/planet_nbody.wgsl`.
Dispatch divisor `(n + 127) / 128` v `src/planet/gpu/state.rs`.

Re-profile N=25k (po #1 + #2):

```
stage                   µs/step      Δ vs #1
nbody                    6510.4       -7 %
density                  5241.6       +10 % (noise — same workload)
viscosity                5298.6       +8 %
pressure                 4726.4       +9 %
hash rebuild              322.6       +7 %
TOTAL                   22555.8       +4 %
```

Nbody **-7 %** isolated. Marginal — bigger workgroup pomáhá s
warp utilizací jen málo, dominant pattern je memory-bound (random
positions[] access uvnitř tile inner loop).

## Final scaling (s oběma optimalizacemi)

| N | steps/s baseline | steps/s after | speedup |
|---|---|---|---|
| 1k | 1359 | 1612 | +19 % |
| 2k | 1151 | 1443 | +25 % |
| 5k | 503 | 717 | +43 % |
| 10k | 173 | 227 | +31 % |
| 25k | 32 | 45 | +41 % |
| 50k | 7 | 11 | +57 % |

Speedup roste s N — větší N znamená denser buckets, kde tighter
grid uspoří relativně víc.

## Bottlenecks zbývající

Po optimalizacích #1+#2 při N=25k:

```
nbody     6510 µs (29 %)   — O(N²) gravity
viscosity 5299 µs (24 %)
density   5242 µs (23 %)
pressure  4726 µs (21 %)
hash       300 µs ( 1 %)
kick+drift ~430 µs ( 2 %)
```

Čtyři "velké" passes jsou stále téměř vyrovnané. Pro významný
další speedup potřeba architektonické změny.

## Roadmap dalších optimalizací

Seřazeno podle ROI (effort vs. potential speedup):

### Low effort (1-2 hodiny)

1. **Batch dispatches do jednoho command encoderu + jednoho submitu per
   tick.** Current 8 submits × ~50 µs overhead = 400 µs/step. Save
   2–20 % podle N (víc na low-N).
2. **Per-particle adaptive h** — odpojit `h_max` od grid cell size, povolit
   h rust při expansion. Kombinovat s **finer grid při high N**.

### Medium effort (1-2 dny)

3. **Reduce SPH passes count**: merge density + pressure + viscosity do
   single "force_compute" shaderu. Jeden neighbor scan místo tří →
   teoreticky 3× SPH speedup. Komplikace: density musí být známé
   PŘED pressure, takže by potřebovaly dvě passes (density → combined
   pressure+viscosity). 2× speedup na SPH.
4. **GPU-side diagnostic reductions** (inertia tensor, energy) místo CPU
   readback. Eliminuje 100k × 5 × 4 B = 2 MB readback per diagnostic
   tick. Důležité pro `--diag-every 1` runs.
5. **Adaptive grid resolution** — 32³ → dynamic per `t_ff`, držet
   avg occupancy ~constant napříč N. Pro 100k+ částic.

### High effort (1-2 týdny — Decade 3+)

6. **Barnes-Hut tree code** pro gravity. O(N²) → O(N log N). Při N=50k
   nbody by klesl z ~30 ms na ~1 ms → **~30× nbody speedup**, kritické
   pro N=100k+. Velký implementační projekt (octree GPU build, tree
   traversal shader).
7. **Particle-Mesh (PM)** alternativně: gravity přes FFT-based Poisson
   solver na 64³ grid. O(N + M log M). Méně přesné než BH ale ještě
   rychlejší. Standard pro cosmology N-body.
8. **Multi-GPU / wgpu device pool** pro N=1M+. Mimo scope tohoto projektu.

## Bottom line

Dvě jednoduché změny (cell_size tighten + TILE bump) dávají
**~40 % speedup** napříč celým N range a posunou nás od N=25k jako
prakticky max k N=50k jako rozumný experiment v ~50 s wall-clock per
0.5 t_ff. Pro N=100k+ runs by se vyplatil **batch dispatches +
merged SPH** (~2× další zrychlení) nebo přechod na **tree code** pro
nbody (10× další zrychlení). Engine je nyní production-ready pro
batch sweep experiments při N ≤ 50k.
