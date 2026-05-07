// Sprint 127: GPU predate dispatch benchmark. Měří overhead full multi-spike
// implementace (variable-length loop + 3D forward + complexity factor) vs
// zero-spike baseline (no inner spike loop, base predation gain only).
//
// Spuštění: `cargo bench --bench predate_gpu --features gpu`. Bez `gpu` feature
// bench je no-op (criterion harness se zaregistruje, ale žádné test cases).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

#[cfg(feature = "gpu")]
mod gpu_bench {
    use super::*;
    use bioscape::gpu::{GpuContext, PredateGpu, PredateParamsGpu, SpatialHashGpu};
    use bioscape::{
        ATTACK_THRESHOLD, CELL_RADIUS, DILUTION_K, HERD_RADIUS, PREDATION_DRAIN_PER_TICK,
        PREDATION_GAIN_PER_TICK, SIZE_RATIO_THRESHOLD, SPIKE_DOT_THRESHOLD, SPIKE_PREDATION_BONUS,
        SPIKE_SLOTS,
    };
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    struct Fixture {
        positions: Vec<[f32; 3]>,
        eff_radii: Vec<f32>,
        headings: Vec<f32>,
        pitches: Vec<f32>,
        spikes_packed: Vec<[f32; 4]>,
        spike_counts: Vec<u32>,
        attack_signals: Vec<f32>,
    }

    fn make_fixture(n: usize, seed: u64, all_zero_spike: bool) -> Fixture {
        let mut rng = StdRng::seed_from_u64(seed);
        let positions: Vec<[f32; 3]> = (0..n)
            .map(|_| {
                [
                    rng.random_range(-500.0_f32..500.0),
                    rng.random_range(-500.0_f32..500.0),
                    rng.random_range(-2.0_f32..2.0),
                ]
            })
            .collect();
        let eff_radii: Vec<f32> = (0..n).map(|_| rng.random_range(0.7_f32..1.3)).collect();
        let headings: Vec<f32> = (0..n)
            .map(|_| rng.random_range(0.0_f32..core::f32::consts::TAU))
            .collect();
        let pitches: Vec<f32> = (0..n)
            .map(|_| rng.random_range(-core::f32::consts::FRAC_PI_4..core::f32::consts::FRAC_PI_4))
            .collect();
        let spike_counts: Vec<u32> = (0..n)
            .map(|_| {
                if all_zero_spike {
                    0
                } else {
                    rng.random_range(1..=SPIKE_SLOTS as u32)
                }
            })
            .collect();
        let spikes_packed: Vec<[f32; 4]> = (0..n * SPIKE_SLOTS)
            .map(|_| {
                [
                    rng.random_range(0.0_f32..1.0),
                    rng.random_range(-core::f32::consts::PI..core::f32::consts::PI),
                    rng.random_range(
                        -core::f32::consts::FRAC_PI_2..core::f32::consts::FRAC_PI_2,
                    ),
                    rng.random_range(0.0_f32..1.0),
                ]
            })
            .collect();
        // Aby se attack pass spustil pro většinu cells.
        let attack_signals: Vec<f32> =
            (0..n).map(|_| rng.random_range(0.3_f32..1.5)).collect();
        Fixture {
            positions,
            eff_radii,
            headings,
            pitches,
            spikes_packed,
            spike_counts,
            attack_signals,
        }
    }

    pub fn bench_predate(c: &mut Criterion) {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("predate_gpu bench skipped: no GPU adapter ({e})");
                return;
            }
        };
        let cell_size = 64.0_f32;
        let world_half = [1000.0_f32, 1000.0];

        let mut group = c.benchmark_group("predate_gpu");
        for &n in &[1000_usize, 10000] {
            let fixture_full = make_fixture(n, 0xBEEF, false);
            let fixture_zero = make_fixture(n, 0xBEEF, true);
            let mut hash = SpatialHashGpu::with_context(&ctx, n, cell_size, world_half)
                .expect("hash init");
            let _ = hash.rebuild(&fixture_full.positions);
            let mut pred = PredateGpu::with_context(&ctx, n).expect("predate init");
            let params = PredateParamsGpu {
                cell_size,
                cell_radius_const: CELL_RADIUS,
                size_ratio_threshold: SIZE_RATIO_THRESHOLD,
                herd_radius_sq: HERD_RADIUS * HERD_RADIUS,
                attack_threshold: ATTACK_THRESHOLD,
                predation_gain: PREDATION_GAIN_PER_TICK,
                predation_drain: PREDATION_DRAIN_PER_TICK,
                spike_dot_threshold: SPIKE_DOT_THRESHOLD,
                spike_bonus: SPIKE_PREDATION_BONUS,
                dilution_k: DILUTION_K,
                world_half_x: world_half[0],
                world_half_y: world_half[1],
                ..PredateParamsGpu::default()
            };

            group.bench_with_input(
                BenchmarkId::new("full_multispike", n),
                &n,
                |b, &_n| {
                    b.iter(|| {
                        let res = pred.compute(
                            black_box(&fixture_full.positions),
                            black_box(&fixture_full.eff_radii),
                            black_box(&fixture_full.headings),
                            black_box(&fixture_full.pitches),
                            black_box(&fixture_full.spikes_packed),
                            black_box(&fixture_full.spike_counts),
                            black_box(&fixture_full.attack_signals),
                            &hash,
                            params,
                        );
                        black_box(res);
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("zero_spike_baseline", n),
                &n,
                |b, &_n| {
                    b.iter(|| {
                        let res = pred.compute(
                            black_box(&fixture_zero.positions),
                            black_box(&fixture_zero.eff_radii),
                            black_box(&fixture_zero.headings),
                            black_box(&fixture_zero.pitches),
                            black_box(&fixture_zero.spikes_packed),
                            black_box(&fixture_zero.spike_counts),
                            black_box(&fixture_zero.attack_signals),
                            &hash,
                            params,
                        );
                        black_box(res);
                    });
                },
            );
        }
        group.finish();
    }
}

#[cfg(feature = "gpu")]
fn predate_benches(c: &mut Criterion) {
    gpu_bench::bench_predate(c);
}

#[cfg(not(feature = "gpu"))]
fn predate_benches(_c: &mut Criterion) {
    eprintln!("predate_gpu bench skipped: build without --features gpu");
}

criterion_group!(benches, predate_benches);
criterion_main!(benches);
