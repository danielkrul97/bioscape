#!/usr/bin/env bash
# Sprint 219 — multi-shape parametric sweep over (shape, size, omega_frac).
# 3 shapes × 5 sizes × 5 omega = 75 CSV runs (~2 min wall-clock at N=2000),
# followed by a per-shape classification table.
#
# Each shape iterates a different "size" parameter:
#   torus   → r_minor      (tube radius; thinner ring = higher R/r)
#   cube    → cube_side    (edge length; varies total volume)
#   pancake → pancake_height (disc thickness)
#
# Usage: scripts/planet_sweep.sh [out_dir]
# Default out_dir: data/sweep_s219

set -euo pipefail
# Force C locale for awk so it treats `.` as decimal point regardless
# of system locale (cs_CZ uses comma → parses "1.234" as 1, truncates).
export LC_ALL=C
export LANG=C

OUT_DIR="${1:-data/sweep_s219}"
mkdir -p "$OUT_DIR"

N=2000
T_END=2.0
DT=1e-3
SEED=7

SHAPES=(torus cube pancake)
OMEGA_FRACS=(0.0 0.3 0.6 0.9 1.0)

# Size-parameter sets per shape. Index matches across shapes so we can
# label rows "thinness 1..5".
TORUS_RMINOR=(0.5 0.3333 0.2 0.125 0.0833)
CUBE_SIDE=(0.6 0.75 0.924 1.2 1.5)
PANCAKE_HEIGHT=(0.05 0.1 0.251 0.5 1.0)

n_runs=$(( ${#SHAPES[@]} * 5 * ${#OMEGA_FRACS[@]} ))
echo "Sweep: $n_runs configs (${#SHAPES[@]} shapes × 5 sizes × ${#OMEGA_FRACS[@]} omega), N=$N, t_end=$T_END t_ff"
echo "Output dir: $OUT_DIR"
START=$(date +%s)

for shape in "${SHAPES[@]}"; do
    case "$shape" in
        torus)   sizes=("${TORUS_RMINOR[@]}");   size_flag="--r-minor" ;;
        cube)    sizes=("${CUBE_SIDE[@]}");      size_flag="--cube-side" ;;
        pancake) sizes=("${PANCAKE_HEIGHT[@]}"); size_flag="--pancake-height" ;;
    esac

    for i in "${!sizes[@]}"; do
        size="${sizes[$i]}"
        for omega in "${OMEGA_FRACS[@]}"; do
            label="${shape}_s${i}_${size}_o${omega}"
            out_path="$OUT_DIR/run_${label}.csv"
            echo -n "  running $label ... "
            ./target/release/planet_headless \
                --shape "$shape" \
                --n "$N" \
                $size_flag "$size" \
                --omega-frac "$omega" \
                --seed "$SEED" \
                --t-end "$T_END" \
                --dt "$DT" \
                --diag-every 200 \
                --out "$out_path" 2>&1 | tail -1
        done
    done
done

ELAPSED=$(($(date +%s) - START))
echo ""
echo "Sweep done in ${ELAPSED}s. Summary:"
echo ""
printf "%-9s %-7s %-7s | %-9s %-10s %-10s %-10s %-12s\n" \
    "shape" "size" "omega" "axis_a/c" "I_a" "I_c" "max_r" "verdict"
echo "----------------------------------------------------------------------------------------"

for shape in "${SHAPES[@]}"; do
    case "$shape" in
        torus)   sizes=("${TORUS_RMINOR[@]}") ;;
        cube)    sizes=("${CUBE_SIDE[@]}") ;;
        pancake) sizes=("${PANCAKE_HEIGHT[@]}") ;;
    esac
    for i in "${!sizes[@]}"; do
        size="${sizes[$i]}"
        for omega in "${OMEGA_FRACS[@]}"; do
            label="${shape}_s${i}_${size}_o${omega}"
            f="$OUT_DIR/run_${label}.csv"
            [ -f "$f" ] || continue
            awk -F',' -v sh="$shape" -v sz="$size" -v om="$omega" '
                NR==2 { e0 = $7 }
                END {
                    axis_ac = $12
                    ia = $9
                    ic = $11
                    maxr = $14
                    if (axis_ac > 3.5) v = "torus"
                    else if (axis_ac > 1.8) v = "thick_torus"
                    else if (axis_ac > 1.3) v = "ellipsoid"
                    else v = "sphere"
                    printf "%-9s %-7s %-7s | %-9.3f %-10.3f %-10.3f %-10.3f %-12s\n", \
                        sh, sz, om, axis_ac, ia, ic, maxr, v
                }
            ' "$f"
        done
    done
done
