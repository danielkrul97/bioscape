#!/usr/bin/env bash
# Spustí cargo-flamegraph na headless pro perf profile per-tick hot path.
# Defaultní smoke run je 5 generací (≤30 viz `feedback_perf_smoke_runs`).
#
# Build profile:  `release-debug` (thin LTO + 16 codegen units + debug symbols).
# Stack unwinding: frame pointers (RUSTFLAGS="-C force-frame-pointers=yes")
# místo DWARF — perf.data ~50× menší, post-processing v sekundách místo hodin.
#
# Lokální použití:
#   ./scripts/profile.sh                          # 5 gen, seed 0
#   ./scripts/profile.sh --gens 10                # 10 gen
#   ./scripts/profile.sh --compare HEAD~3         # before/after — 2 SVGs do data/profiling/
#   ./scripts/profile.sh --label perf-baseline    # custom output suffix
#
# Output:  data/profiling/<label>.svg
#          (compare mode: data/profiling/current_<sha>.svg + ref_<sha>.svg)
#
# Závislosti:
#   cargo install flamegraph
#   sudo apt install linux-tools-generic linux-tools-$(uname -r)
#   sudo sysctl kernel.perf_event_paranoid=1     # one-shot, dokud nerebootneš
set -euo pipefail

export RUSTFLAGS="${RUSTFLAGS:-} -C force-frame-pointers=yes"

cd "$(dirname "$0")/.."

GENS="${GENS:-5}"
SEED="${SEED:-0}"
COMPARE_REF=""
LABEL=""

while (( $# > 0 )); do
  case "$1" in
    --gens) GENS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --compare) COMPARE_REF="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ─── Pre-flight checks ────────────────────────────────────────────────────────
if ! command -v cargo-flamegraph &>/dev/null && ! cargo flamegraph --help &>/dev/null; then
  echo "error: cargo-flamegraph not installed" >&2
  echo "  install: cargo install flamegraph" >&2
  exit 1
fi
if ! command -v perf &>/dev/null; then
  echo "error: 'perf' not in PATH" >&2
  echo "  install: sudo apt install linux-tools-generic linux-tools-\$(uname -r)" >&2
  exit 1
fi
PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 4)"
if (( PARANOID > 1 )); then
  echo "warning: kernel.perf_event_paranoid=$PARANOID — perf may refuse to record samples" >&2
  echo "  fix (one-shot until reboot):  sudo sysctl kernel.perf_event_paranoid=1" >&2
  echo "  or run profile with sudo:     sudo -E $0 $*" >&2
  echo
fi

OUT_DIR="data/profiling"
mkdir -p "$OUT_DIR"

# Builds & runs cargo-flamegraph in the current directory (used both for the
# working tree and inside a temp worktree). `CARGO_PROFILE_RELEASE_DEBUG=true`
# preserves frame pointers + debug info on the release profile so `perf` can
# resolve Rust frames cleanly.
run_flamegraph() {
  local out="$1"
  local args=("$SEED" "$GENS")
  echo ">>> profile $out  (gens=$GENS seed=$SEED)"
  # Custom perf args: 99 Hz sampling + frame-pointer call-graph (kompaktní perf.data,
  # post-processing v sekundách). Nutné ve spojení s `force-frame-pointers=yes`
  # v RUSTFLAGS, jinak by stack walker neměl ramps.
  cargo flamegraph \
    --profile release-debug --bin headless \
    -c "record -F 99 --call-graph fp -g" \
    --output "$out" -- "${args[@]}"
  echo "<<< wrote $out"
}

git_short() { git rev-parse --short "${1:-HEAD}" 2>/dev/null || echo "unknown"; }

if [[ -n "$COMPARE_REF" ]]; then
  CURRENT_SHA="$(git_short HEAD)"
  REF_SHA="$(git_short "$COMPARE_REF")"
  CURRENT_OUT="$OUT_DIR/current_${LABEL:-$CURRENT_SHA}.svg"
  REF_OUT_REL="$OUT_DIR/ref_${LABEL:+${LABEL}_}${REF_SHA}.svg"
  REF_OUT_ABS="$(pwd)/${REF_OUT_REL}"

  # Profile current working tree first so any failure aborts before we touch the worktree.
  run_flamegraph "$CURRENT_OUT"

  WORKTREE="$(mktemp -d -t bioscape-profile-XXXX)"
  cleanup() { git worktree remove --force "$WORKTREE" >/dev/null 2>&1 || rm -rf "$WORKTREE"; }
  trap cleanup EXIT

  echo
  echo ">>> creating temp worktree at $WORKTREE for $COMPARE_REF (sha=$REF_SHA)"
  git worktree add --detach "$WORKTREE" "$COMPARE_REF"
  (
    cd "$WORKTREE"
    # Ref worktree was checked out at $COMPARE_REF, which may not yet contain
    # the `release-debug` profile. Append it locally (worktree is wiped by the
    # trap on exit).
    if ! grep -q '^\[profile.release-debug\]' Cargo.toml; then
      cat >> Cargo.toml <<'TOML'

[profile.release-debug]
inherits = "release"
lto = "thin"
codegen-units = 16
debug = true
TOML
    fi
    args=("$SEED" "$GENS")
    echo ">>> profile $REF_OUT_REL  (in worktree)"
    cargo flamegraph \
      --profile release-debug --bin headless \
      -c "record -F 99 --call-graph fp -g" \
      --output "$REF_OUT_ABS" -- "${args[@]}"
  )
  echo "<<< wrote $REF_OUT_REL"
  echo
  echo "compare: $CURRENT_OUT vs $REF_OUT_REL"
  echo "  (open both in a browser; flamegraph SVG has search + interactive zoom)"
else
  CURRENT_OUT="$OUT_DIR/${LABEL:-$(git_short HEAD)}.svg"
  run_flamegraph "$CURRENT_OUT"
fi
