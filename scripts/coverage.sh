#!/usr/bin/env bash
# Spustí cargo-llvm-cov se 80% hard gate (lib + binárky).
# Lokální použití:   ./scripts/coverage.sh
# Lokální HTML report: ./scripts/coverage.sh --html  → target/llvm-cov/html/index.html
set -euo pipefail

THRESHOLD="${COVERAGE_THRESHOLD:-80}"

cd "$(dirname "$0")/.."

# Standard Rust convention: main.rs entry-pointy se netestují (jsou to wrappery
# kolem `fn main()` bez logiky — `src/main.rs` jen volá `renderer::run()`,
# `bin/headless/main.rs` je argparse + setup). Lib + ostatní bin moduly OK.
IGNORE_REGEX='(src/main\.rs|src/bin/headless/main\.rs)'

ARGS=("--tests" "--lib" "--bins" "--ignore-filename-regex" "${IGNORE_REGEX}" "--fail-under-lines" "${THRESHOLD}")

if [[ "${1:-}" == "--html" ]]; then
    ARGS+=("--html")
fi

# Default features (gpu) — bin/headless/world.rs má BRAIN_INPUTS importy gated
# za feature `gpu`, takže bez něj build padne. Pokud GPU adapter chybí, paritní
# testy mají skip pattern.
cargo llvm-cov "${ARGS[@]}"
