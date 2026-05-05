// Sprint 57: per-phase micro-benches. Cílem je sledovat v čase hot lib funkce
// které tvoří backbone headless tick path. Není to plný end-to-end (ten je v
// `benches/full_tick.rs`); criterion na lib API umožňuje fine-grained signál
// pro paralelizaci a SIMD úvahy bez reaktor-level setup.
use bioscape::{
    populate_brain_inputs, Brain, BrainSensors, Cell, Genome, PhysicsConfig, SmellField, WorldMap,
    BRAIN_INPUTS, PHEROMONE_DECAY, PHEROMONE_DIFFUSION, PHYSICS_CONFIG, SMELL_DECAY, SMELL_DIFFUSION,
    SMELL_GRID_RES, SMELL_GRID_RES_Z, WORLD_MAP_BASE_RES, WORLD_MAP_BASE_RES_Z, WORLD_MAP_RES,
    WORLD_MAP_RES_Z,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::SeedableRng;

const WORLD_HALF: [f32; 3] = [960.0, 540.0, 20.0];

fn make_brain() -> Brain {
    let mut rng = StdRng::seed_from_u64(42);
    Brain::random(&mut rng)
}

fn make_inputs() -> [f32; BRAIN_INPUTS] {
    let mut x = [0.0_f32; BRAIN_INPUTS];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i as f32) * 0.137).sin();
    }
    x
}

fn make_cell(seed: u64) -> Cell {
    let mut rng = StdRng::seed_from_u64(seed);
    Cell::random(&mut rng, WORLD_HALF, seed, 0, seed)
}

fn make_smell_field(populated: bool) -> SmellField {
    let mut f = SmellField::new(
        [SMELL_GRID_RES, SMELL_GRID_RES, SMELL_GRID_RES_Z],
        WORLD_HALF,
    );
    if populated {
        // Sprinkle 200 sources to get a non-trivial gradient field.
        for k in 0..20 {
            for j in 0..10 {
                let x = -WORLD_HALF[0] + (k as f32) * 96.0;
                let y = -WORLD_HALF[1] + (j as f32) * 96.0;
                f.add_source([x, y, 0.0], 1.0);
            }
        }
        for _ in 0..6 {
            f.step(SMELL_DIFFUSION, SMELL_DECAY, 1.0 / 60.0);
        }
    }
    f
}

fn make_world_map() -> WorldMap {
    WorldMap::new(
        [WORLD_MAP_RES, WORLD_MAP_RES, WORLD_MAP_RES_Z],
        [WORLD_MAP_BASE_RES, WORLD_MAP_BASE_RES, WORLD_MAP_BASE_RES_Z],
        WORLD_HALF,
        12345,
    )
}

fn bench_brain_forward(c: &mut Criterion) {
    let brain = make_brain();
    let inputs = make_inputs();
    let mut group = c.benchmark_group("brain_forward");
    group.bench_function("single", |b| {
        b.iter(|| black_box(brain.forward_with_state(black_box(&inputs))))
    });
    // Batch — prozkoumat per-cell amortizaci (1000 cells × 1 forward).
    for n in [200usize, 500, 1000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("batch_seq", n), &n, |b, &n| {
            b.iter(|| {
                let mut acc = 0.0_f32;
                for _ in 0..n {
                    let (_h, o) = brain.forward_with_state(black_box(&inputs));
                    acc += o[0];
                }
                black_box(acc)
            })
        });
    }
    group.finish();
}

fn bench_smell_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("smell_field_step");
    group.throughput(Throughput::Elements(
        (SMELL_GRID_RES * SMELL_GRID_RES * SMELL_GRID_RES_Z) as u64,
    ));
    let mut field = make_smell_field(true);
    group.bench_function("populated", |b| {
        b.iter(|| field.step(SMELL_DIFFUSION, SMELL_DECAY, 1.0 / 60.0))
    });

    let mut field = make_smell_field(false);
    group.bench_function("empty", |b| {
        b.iter(|| field.step(SMELL_DIFFUSION, SMELL_DECAY, 1.0 / 60.0))
    });
    group.finish();
}

fn bench_pheromone_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("pheromone_field_step");
    group.throughput(Throughput::Elements(
        (SMELL_GRID_RES * SMELL_GRID_RES * SMELL_GRID_RES_Z) as u64,
    ));
    let mut field = make_smell_field(true);
    group.bench_function("populated", |b| {
        b.iter(|| field.step(PHEROMONE_DIFFUSION, PHEROMONE_DECAY, 1.0 / 60.0))
    });
    group.finish();
}

fn bench_world_map_sample(c: &mut Criterion) {
    let map = make_world_map();
    let mut group = c.benchmark_group("world_map_sample");
    group.throughput(Throughput::Elements(1000));
    group.bench_function("1000_random", |b| {
        b.iter(|| {
            let mut acc = 0.0_f32;
            for i in 0..1000 {
                let x = ((i as f32) * 1.7).sin() * WORLD_HALF[0];
                let y = ((i as f32) * 0.9).cos() * WORLD_HALF[1];
                let z = ((i as f32) * 0.3).sin() * WORLD_HALF[2];
                acc += map.sample([black_box(x), black_box(y), black_box(z)]);
            }
            black_box(acc)
        })
    });
    group.finish();
}

fn bench_smell_gradient(c: &mut Criterion) {
    let field = make_smell_field(true);
    let mut group = c.benchmark_group("smell_gradient_at");
    group.throughput(Throughput::Elements(1000));
    group.bench_function("1000_random", |b| {
        b.iter(|| {
            let mut acc = 0.0_f32;
            for i in 0..1000 {
                let x = ((i as f32) * 1.7).sin() * WORLD_HALF[0];
                let y = ((i as f32) * 0.9).cos() * WORLD_HALF[1];
                let z = ((i as f32) * 0.3).sin() * WORLD_HALF[2];
                let g = field.gradient_at([black_box(x), black_box(y), black_box(z)], 4.0);
                acc += g[0] + g[1] + g[2];
            }
            black_box(acc)
        })
    });
    group.finish();
}

fn bench_cell_step(c: &mut Criterion) {
    let physics: &PhysicsConfig = &PHYSICS_CONFIG;
    let mut group = c.benchmark_group("cell_step");
    for n in [200usize, 500, 1000] {
        group.throughput(Throughput::Elements(n as u64));
        let cells: Vec<Cell> = (0..n).map(|i| make_cell(i as u64)).collect();
        group.bench_with_input(BenchmarkId::new("seq", n), &n, |b, &_n| {
            let mut local = cells.clone();
            b.iter(|| {
                for c in local.iter_mut() {
                    c.step(1.0 / 60.0, WORLD_HALF, 0, 0, physics);
                }
            });
        });
    }
    group.finish();
}

fn bench_populate_brain_inputs(c: &mut Criterion) {
    let mut group = c.benchmark_group("populate_brain_inputs");
    let sensors = BrainSensors {
        nearest_food: Some([12.0, -7.0, 1.0]),
        nearest_cell: Some(([3.0, 4.0, -2.0], 6.0)),
        neighbors_in_vision: 5,
        smell_grad: [0.1, -0.2, 0.05],
        pheromone_grad: [0.0, 0.1, -0.05],
        temperature_local: 17.0,
    };
    for n in [1000usize] {
        group.throughput(Throughput::Elements(n as u64));
        let mut cells: Vec<Cell> = (0..n).map(|i| make_cell(i as u64)).collect();
        group.bench_with_input(BenchmarkId::new("seq", n), &n, |b, &_n| {
            b.iter(|| {
                let mut acc = 0.0_f32;
                for c in cells.iter_mut() {
                    let inp = populate_brain_inputs(c, &sensors, c.genome.vision_radius);
                    acc += inp[0];
                }
                black_box(acc)
            });
        });
    }
    group.finish();
}

fn bench_brain_random(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    c.bench_function("brain_random", |b| {
        b.iter(|| black_box(Brain::random(&mut rng)))
    });
}

fn bench_genome_random(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    c.bench_function("genome_random", |b| {
        b.iter(|| black_box(Genome::random(&mut rng)))
    });
}

criterion_group!(
    benches,
    bench_brain_forward,
    bench_smell_step,
    bench_pheromone_step,
    bench_world_map_sample,
    bench_smell_gradient,
    bench_cell_step,
    bench_populate_brain_inputs,
    bench_brain_random,
    bench_genome_random,
);
criterion_main!(benches);
