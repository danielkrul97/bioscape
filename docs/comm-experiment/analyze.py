#!/usr/bin/env python3
"""Communication metrics analysis for multi-channel pheromone experiment.

Reads per-generation CSV from headless runs, computes communication signal:
1. Per-channel emission trends across generations
2. Temporal patterning (burst_score growth)
3. Channel specialization (cross-cell stdev / mean)
4. Channel-environment correlations (predation, food, density, bonds)
5. Channel-channel anti-correlation (specialist signal)
"""

import csv
import statistics
import sys
from pathlib import Path

CHANNELS = [0, 1, 2]
SEEDS = [0, 1, 42]

def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (float(v) if v not in ('', None) else 0.0) for k, v in row.items()})
    return rows

def safe_corr(xs, ys):
    """Pearson correlation, robust to zero variance."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / (sx * sy) ** 0.5

def analyze_one(seed, rows):
    """Single-seed analysis."""
    if not rows:
        return None
    # Genmes 0-29, 30-99, 100-299 phase aggregates
    early = [r for r in rows if 0 <= r['gen'] <= 29]
    mid = [r for r in rows if 30 <= r['gen'] <= 99]
    late = [r for r in rows if 100 <= r['gen'] <= 299]

    def phase_stats(phase_rows):
        if not phase_rows:
            return {}
        return {
            'cells': statistics.mean(r['cells'] for r in phase_rows),
            'lineages': statistics.mean(r['lineages'] for r in phase_rows),
            'predation_events': statistics.mean(r['predation_events'] for r in phase_rows),
            'mean_bond_count': statistics.mean(r['mean_bond_count'] for r in phase_rows),
            'food': statistics.mean(r['food'] for r in phase_rows),
            'density_avg': statistics.mean(r['density_avg'] for r in phase_rows),
            'ph_emit_ch0_avg': statistics.mean(r['ph_emit_ch0_avg'] for r in phase_rows),
            'ph_emit_ch1_avg': statistics.mean(r['ph_emit_ch1_avg'] for r in phase_rows),
            'ph_emit_ch2_avg': statistics.mean(r['ph_emit_ch2_avg'] for r in phase_rows),
            'ph_emit_ch0_dev': statistics.mean(r['ph_emit_ch0_dev'] for r in phase_rows),
            'ph_emit_ch1_dev': statistics.mean(r['ph_emit_ch1_dev'] for r in phase_rows),
            'ph_emit_ch2_dev': statistics.mean(r['ph_emit_ch2_dev'] for r in phase_rows),
            'ph_burst_score_ch0': statistics.mean(r['ph_burst_score_ch0'] for r in phase_rows),
            'ph_burst_score_ch1': statistics.mean(r['ph_burst_score_ch1'] for r in phase_rows),
            'ph_burst_score_ch2': statistics.mean(r['ph_burst_score_ch2'] for r in phase_rows),
        }

    phases = {
        'early (0-29)': phase_stats(early),
        'mid (30-99)': phase_stats(mid),
        'late (100-299)': phase_stats(late),
    }

    # Channel-environment correlations
    env_metrics = ['predation_events', 'food', 'density_avg', 'mean_bond_count']
    ch_emits = ['ph_emit_ch0_avg', 'ph_emit_ch1_avg', 'ph_emit_ch2_avg']

    def corr_subset(subset):
        ct = {}
        for ch in ch_emits:
            ch_series = [r[ch] for r in subset]
            ct[ch] = {}
            for env in env_metrics:
                env_series = [r[env] for r in subset]
                ct[ch][env] = safe_corr(ch_series, env_series)
        return ct

    corr_table = corr_subset(rows) if len(rows) >= 30 else {}
    # Late-only correlation = test temporal artifact hypothesis. Pokud korelace
    # přežije v rámci late phase (po settled dynamics), je real. Pokud zmizí,
    # full-timeline correlation byla artifact monotonic decay.
    late_corr = corr_subset(late) if len(late) >= 20 else {}

    return {
        'seed': seed,
        'phases': phases,
        'corr': corr_table,
        'late_corr': late_corr,
        'n_gens': len(rows),
    }

def print_report(results):
    print("=" * 80)
    print("MULTI-CHANNEL PHEROMONE EXPERIMENT — COMMUNICATION SIGNAL ANALYSIS")
    print("=" * 80)
    for r in results:
        if r is None:
            continue
        print(f"\n## SEED {r['seed']} (n={r['n_gens']} generations)")
        for phase, st in r['phases'].items():
            if not st:
                continue
            print(f"\n### {phase}")
            print(f"  cells={st['cells']:.0f}, lineages={st['lineages']:.1f}, "
                  f"predation={st['predation_events']:.0f}, bonds={st['mean_bond_count']:.3f}")
            print(f"  Emit:    ch0={st['ph_emit_ch0_avg']:.3f}±{st['ph_emit_ch0_dev']:.3f}  "
                  f"ch1={st['ph_emit_ch1_avg']:.3f}±{st['ph_emit_ch1_dev']:.3f}  "
                  f"ch2={st['ph_emit_ch2_avg']:.3f}±{st['ph_emit_ch2_dev']:.3f}")
            print(f"  Burst:   ch0={st['ph_burst_score_ch0']:.4f}  "
                  f"ch1={st['ph_burst_score_ch1']:.4f}  "
                  f"ch2={st['ph_burst_score_ch2']:.4f}")

        if r['corr']:
            print(f"\n### Channel-environment correlations (Pearson, full timeline)")
            envs = list(next(iter(r['corr'].values())).keys())
            print(f"  {'channel':<22} | " + " | ".join(f"{e:<18}" for e in envs))
            for ch, corrs in r['corr'].items():
                vals = " | ".join(f"{corrs[e]:+.3f}              " for e in envs)
                print(f"  {ch:<22} | {vals}")
        if r['late_corr']:
            print(f"\n### Channel-environment correlations (LATE PHASE ONLY, gen 100+)")
            envs = list(next(iter(r['late_corr'].values())).keys())
            print(f"  {'channel':<22} | " + " | ".join(f"{e:<18}" for e in envs))
            for ch, corrs in r['late_corr'].items():
                vals = " | ".join(f"{corrs[e]:+.3f}              " for e in envs)
                print(f"  {ch:<22} | {vals}")

    # Cross-seed consistency
    print("\n" + "=" * 80)
    print("CROSS-SEED CONSISTENCY (late phase 100-299)")
    print("=" * 80)
    metrics = ['ph_emit_ch0_avg', 'ph_emit_ch1_avg', 'ph_emit_ch2_avg',
               'ph_burst_score_ch0', 'ph_burst_score_ch1', 'ph_burst_score_ch2']
    for m in metrics:
        values = []
        for r in results:
            if r and r['phases'].get('late (100-299)'):
                values.append(r['phases']['late (100-299)'][m])
        if values:
            mn = sum(values) / len(values)
            sd = (sum((v - mn) ** 2 for v in values) / len(values)) ** 0.5 if len(values) > 1 else 0
            cv = sd / mn if mn > 1e-6 else 0
            print(f"  {m:<28} mean={mn:.4f} sd={sd:.4f} CV={cv:.2f}")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT — did communication beyond ch0 emerge?")
    print("=" * 80)
    late_ch1 = [r['phases']['late (100-299)']['ph_emit_ch1_avg']
                for r in results if r and r['phases'].get('late (100-299)')]
    late_ch2 = [r['phases']['late (100-299)']['ph_emit_ch2_avg']
                for r in results if r and r['phases'].get('late (100-299)')]
    burst_ch1 = [r['phases']['late (100-299)']['ph_burst_score_ch1']
                 for r in results if r and r['phases'].get('late (100-299)')]
    burst_ch2 = [r['phases']['late (100-299)']['ph_burst_score_ch2']
                 for r in results if r and r['phases'].get('late (100-299)')]

    print(f"\n  Late ch1 emit (mean across seeds): {sum(late_ch1)/max(len(late_ch1),1):.3f}")
    print(f"  Late ch2 emit (mean across seeds): {sum(late_ch2)/max(len(late_ch2),1):.3f}")
    print(f"  Late ch1 burst_score:              {sum(burst_ch1)/max(len(burst_ch1),1):.4f}")
    print(f"  Late ch2 burst_score:              {sum(burst_ch2)/max(len(burst_ch2),1):.4f}")

    threshold = 0.10
    ch1_used = sum(late_ch1)/max(len(late_ch1),1) > threshold
    ch2_used = sum(late_ch2)/max(len(late_ch2),1) > threshold
    print(f"\n  ch1 above {threshold} threshold: {'YES' if ch1_used else 'no'}")
    print(f"  ch2 above {threshold} threshold: {'YES' if ch2_used else 'no'}")
    print(f"\n  Hot environment correlations |r| > 0.4 (= signal candidates):")
    for r in results:
        if not r or not r['corr']:
            continue
        for ch, corrs in r['corr'].items():
            for env, c in corrs.items():
                if abs(c) > 0.4 and ch != 'ph_emit_ch0_avg':
                    print(f"    seed={r['seed']}: {ch} ↔ {env}: r={c:+.3f}")

def main():
    base = Path('/tmp/comm_runs')
    results = []
    for seed in SEEDS:
        path = base / f'seed{seed}.csv'
        if not path.exists():
            print(f"WARN: {path} not found, skipping", file=sys.stderr)
            results.append(None)
            continue
        rows = load_csv(path)
        results.append(analyze_one(seed, rows))
    print_report(results)

if __name__ == '__main__':
    main()
