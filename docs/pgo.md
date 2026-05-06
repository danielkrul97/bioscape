# Profile-Guided Optimization (PGO)

PGO je opt-in build pipeline, která dvouprůchodovou kompilací zlepšuje
inlining, branch prediction a code layout. Reálný workload se použije
jako profilovací zdroj — kompilátor vidí, které větve jsou hot a které
cold, a generuje binárku ušitou na daný workload. Typický win 5–15 %
na celé aplikaci, zero behavioral risk.

PGO **není default**. Default `cargo build --release` (`profile.release`
v `Cargo.toml`: `lto = "fat"`, `codegen-units = 1`, plus `target-cpu=native`
z `.cargo/config.toml`) zůstává nezměněn. PGO se zapíná explicitně přes
`cargo-pgo` subcommand nebo manuální `RUSTFLAGS`.

## Předpoklady

```bash
# llvm-profdata pro merge profilů — součást rustup komponenty
rustup component add llvm-tools-preview

# wrapper přes 3-pass build
cargo install cargo-pgo
```

`cargo pgo info` ověří, že rustc ≥ 1.39 a `llvm-profdata` v PATH.
`llvm-bolt` není potřeba (BOLT je orthogonální optimalizace, ne PGO).

## Recommended workflow (cargo-pgo)

```bash
# 1) Instrumented build + bench run → /tmp/pgo-profiles/*.profraw
cargo pgo bench

# 2) Merge profilů + optimized build + bench run
cargo pgo optimize bench

# Výstup: target/x86_64-unknown-linux-gnu/release/deps/full_tick-* je
# PGO-optimized binárka. cargo-pgo automaticky volá criterion s ní.
```

`cargo pgo bench` spouští `cargo bench --bench full_tick` s
`-Cprofile-generate=<dir>`. Bench tedy plní profil daty hot path
(SIMD brain forward, SIMD field diffuse, rayon par_iter spojené).
`cargo pgo optimize bench` mergne `*.profraw` přes `llvm-profdata`,
přebuilduje s `-Cprofile-use=<merged.profdata>` a znova bench.

`full_tick` je doporučený workload pro profilování — exercises ≥80 %
headless tick CPU cost (brain × N cells, kinematika, smell/pheromone
field diffuse). Driver runtime je dominated rep-iteration noise, ale
optimizer dostane reprezentativní hot-path profil.

## Manuální workflow (fallback)

Pokud `cargo-pgo` selže nebo není dostupný, ekvivalent:

```bash
# 1) Instrumented build (full_tick bench)
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data -C target-cpu=native" \
  cargo build --release --bench full_tick

# 2) Profile run — spusť bench tak, aby profil zachytil hot path
cargo bench --bench full_tick

# 3) Merge profile data
llvm-profdata=$(rustc --print sysroot)/lib/rustlib/x86_64-unknown-linux-gnu/bin/llvm-profdata
"$llvm-profdata" merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data

# 4) Optimized build
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata \
           -Cllvm-args=-pgo-warn-missing-function \
           -C target-cpu=native" \
  cargo build --release --bench full_tick

# 5) Re-run bench s optimized binary
cargo bench --bench full_tick
```

`-Cllvm-args=-pgo-warn-missing-function` je diagnostika — varuje,
když optimizer nenajde profile data pro funkci (např. kdyby workload
neexerciovat hot path). V běžném buildu se nepoužívá, jen při ladění
pokrytí profilu.

## Caveats

- **LTO=fat + PGO**: u některých LLVM verzí byly issues. Sprint 119
  na Rust 1.94.0 s LTO=fat prošel čistě; pokud build padne na novější
  toolchain, zkus přepnout LTO na `thin` *jen v PGO env*, ne globálně.
- **Cache invalidation**: `cargo bench` znovu-spuštěné po PGO buildu
  musí najít fresh artefakt. Pokud se PGO výhra "ztratí", `cargo clean
  --release` + retry. Criterion compare `--save-baseline` zachycuje
  inter-run trend, ale system noise může wina zamlžit.
- **Profile coverage**: PGO win závisí na tom, jak reprezentativní
  je profile-gen workload. `full_tick` bench pokrývá hot path, ale
  ne celé headless (mating, predation, food clustering). Pro headless
  binárku ideálně použít `cargo run --release --bin headless 42 30
  /tmp/profile_run.csv` jako profile-gen. PGO build za jednu workload
  *neoptimalizuje obě* — buď bench, nebo headless. Doporučení: bench
  pro micro-perf experimenty, headless pro shipping batch runy.
- **Reprodukovatelnost**: PGO profil je per-machine (sazba branchů
  závisí na CPU model + microarch). Profil z i5-12400F nemusí dát
  stejný win na Ryzen 5xxx. Pro CI je nutné gen + use ve stejném prostředí.

## Sprint 119 measurement — pending stable workspace

Bench `full_tick` měření **odložené** — v době, kdy Sprint 119 landl,
běžela paralelní decade `107-116-shocks.md` Sprint 121+ (multi-spike
biology) jejíž WIP změnila `BRAIN_HIDDEN: 50 → 40` a další konstanty;
projekt v tom mezistavu nekompiluje (`gpu.rs` const-asserts failnou).
PGO infrastructure (tooling + docs + workflow) je hotová — uživatel
spustí měření po stabilním post-S121+ state s pevnými konstantami.

Reprodukovatelný workflow:

```bash
# Save pre-PGO baseline
cargo bench --bench full_tick -- --save-baseline pre_pgo

# Run PGO pipeline
cargo pgo bench
cargo pgo optimize bench -- -- --save-baseline post_pgo

# Compare
cargo bench --bench full_tick -- --baseline pre_pgo
```
