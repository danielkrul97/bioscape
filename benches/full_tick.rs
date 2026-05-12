// Sprint 111: end-to-end CPU-bound tick bench. World struct lives in
// `src/bin/headless.rs` (binary-private), so we compose the same hot-path
// phases via lib.rs public API: per-cell brain forward + populate inputs +
// kinematics step, plus smell/pheromone field diffusion. Approximates ≥80 %
// of headless tick CPU cost without dragging the GPU pipeline / collisions /
// reproduce logic into the bench harness.
use bioscape::{
    forward_vector, populate_brain_inputs, Brain, BrainSensors, Cell, Food, PhysicsConfig,
    SmellField, WorldMap, PHEROMONE_DECAY, PHEROMONE_DIFFUSION, PHEROMONE_GRID_RES,
    PHEROMONE_GRID_RES_Z, PHYSICS_CONFIG, SMELL_DECAY, SMELL_DIFFUSION, SMELL_GRID_RES,
    SMELL_GRID_RES_Z, SMELL_PER_FOOD, WORLD_MAP_BASE_RES, WORLD_MAP_BASE_RES_Z, WORLD_MAP_RES,
    WORLD_MAP_RES_Z,
};
use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::time::Duration;

const WORLD_HALF: [f32; 3] = [960.0, 540.0, 50.0];
const SEED: u64 = 42;
const TICKS_PER_MEASUREMENT: u64 = 1000;
const FOOD_PER_CELL: f32 = 0.4;

struct BenchWorld {
    cells: Vec<Cell>,
    brains: Vec<Brain>,
    foods: Vec<Food>,
    smell: SmellField,
    pheromone: SmellField,
    map: WorldMap,
    physics: PhysicsConfig,
    tick: u64,
}

impl BenchWorld {
    fn new(n_cells: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(SEED);
        let map = WorldMap::new(
            [WORLD_MAP_RES, WORLD_MAP_RES, WORLD_MAP_RES_Z],
            [WORLD_MAP_BASE_RES, WORLD_MAP_BASE_RES, WORLD_MAP_BASE_RES_Z],
            WORLD_HALF,
            0xC0FFEE,
        );
        let cells: Vec<Cell> = (0..n_cells)
            .map(|i| Cell::random(&mut rng, WORLD_HALF, i as u64, 0, i as u64))
            .collect();
        let brains: Vec<Brain> = (0..n_cells).map(|_| Brain::random(&mut rng)).collect();
        let n_food = ((n_cells as f32) * FOOD_PER_CELL) as usize;
        let foods: Vec<Food> = (0..n_food)
            .map(|_| Food::random(&mut rng, WORLD_HALF))
            .collect();
        let smell =
            SmellField::new([SMELL_GRID_RES, SMELL_GRID_RES, SMELL_GRID_RES_Z], WORLD_HALF);
        let pheromone = SmellField::new(
            [PHEROMONE_GRID_RES, PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z],
            WORLD_HALF,
        );
        Self {
            cells,
            brains,
            foods,
            smell,
            pheromone,
            map,
            physics: PHYSICS_CONFIG,
            tick: 0,
        }
    }

    fn tick(&mut self, dt: f32) -> f32 {
        for food in &self.foods {
            self.smell.add_source(food.position, SMELL_PER_FOOD * dt);
        }
        self.smell.step(SMELL_DIFFUSION, SMELL_DECAY, dt);
        self.pheromone
            .step(PHEROMONE_DIFFUSION, PHEROMONE_DECAY, dt);

        // Sprint 113: rayon par_iter přes (cell, brain). Closure capturuje
        // jen `&self.smell`, `&self.pheromone`, `&self.physics` (read-only) +
        // pre-extracted `tick` (Copy). `acc` je sum reduction — bez mutexu,
        // pořadí součtu je non-deterministic mezi worker threads.
        let tick = self.tick;
        let smell = &self.smell;
        let pheromone = &self.pheromone;
        let physics = &self.physics;
        let acc: f32 = self
            .cells
            .par_iter_mut()
            .zip(self.brains.par_iter_mut())
            .map(|(cell, brain)| {
                let smell_grad = smell.gradient_at(cell.position, 4.0);
                let pheromone_grad_ch0 = pheromone.gradient_at(cell.position, 4.0);
                let mut pheromone_grads = [[0.0_f32; 3]; bioscape::N_PHEROMONE_CHANNELS];
                pheromone_grads[0] = pheromone_grad_ch0;
                let sensors = BrainSensors {
                    nearest_food: None,
                    nearest_cell: None,
                    neighbors_in_vision: 0,
                    smell_grad,
                    pheromone_grads,
                    temperature_local: 17.0,
                    vibration_grad: [0.0, 0.0, 0.0],
                    vibration_amp: 0.0,
                    whisker_distances: [1.0; bioscape::WHISKER_COUNT],
                };
                let inputs = populate_brain_inputs(cell, &sensors, cell.genome.vision_radius);
                let (_h, outputs) = brain.forward_with_state(&inputs);
                cell.angular_velocity = outputs[0];
                cell.pitch_velocity = outputs[7];
                let speed = cell.genome.max_speed * outputs[1].clamp(-1.0, 1.0);
                let fwd = forward_vector(cell.heading, cell.pitch);
                cell.velocity[0] = fwd[0] * speed;
                cell.velocity[1] = fwd[1] * speed;
                cell.velocity[2] = fwd[2] * speed;
                cell.step(dt, WORLD_HALF, tick, 0, physics);
                outputs[0]
            })
            .sum();
        let sample = self.map.sample(self.cells[0].position);
        self.tick += 1;
        acc + sample
    }
}

fn bench_full_tick(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_tick");
    let dt = 1.0 / 60.0;
    for &n in &[1000usize, 2500, 5000] {
        group.throughput(Throughput::Elements(TICKS_PER_MEASUREMENT * n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            // `iter_batched` jasně odděluje setup (`BenchWorld::new`) a měřenou
            // smyčku — Criterion volá setup raz per iter, čas se měří jen
            // přes routine. Pre-fix `iter_custom` měl ekvivalentní bracketing
            // ale ručně; iter_batched je idiomatičtější + lépe spolupracuje
            // s Criterion warmup/sample logic.
            b.iter_batched(
                || BenchWorld::new(n),
                |mut world| {
                    let mut sink = 0.0_f32;
                    for _ in 0..TICKS_PER_MEASUREMENT {
                        sink += world.tick(dt);
                    }
                    black_box(sink);
                },
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

criterion_group! {
    name = full_tick;
    // Sample size 30 (3× původní 10) zlepší stabilitu odhadu bez zhoršení
    // wallclock — měření je pořád ohraničeno `measurement_time`, takže víc
    // samples = kratší per-sample, ale stabilnější mean/variance.
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(20));
    targets = bench_full_tick
}
criterion_main!(full_tick);
