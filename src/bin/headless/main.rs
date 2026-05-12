//! Headless harness — pure simulation loop, no Bevy renderer.
//!
//! Usage: `cargo run --release --bin headless -- [seed] [max_gens] [out_path]`
//! Defaults: seed=0, max_gens=500, out_path=run_seed{seed}.csv
//!
//! Logs per-generation stats (cell_count, mean/dev for max_speed/vision/body_size,
//! food count, density factor) to CSV. Reproducible: same seed → identical run.

use bioscape::{
    EventCalendar, MazeDifficulty,
    ShockScheduleConfig, CYCLE_AMPLITUDE,
    INITIAL_CELLS,
    MATING_RADIUS, MAX_POPULATION,
    N_PHEROMONE_CHANNELS,
    TICKS_PER_GENERATION, WORLD_HALF, WORLD_MAP_SEED,
};
use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

mod csv;
mod world;

use csv::*;
use world::*;

#[derive(Parser, Debug)]
#[command(
    name = "headless",
    about = "Bioscape headless simulator — deterministic, GPU compute mandatory.",
)]
struct Cli {
    /// RNG seed (deterministic; same seed → identical run).
    #[arg(default_value_t = 0)]
    seed: u64,

    /// Number of generations to simulate.
    #[arg(default_value_t = 500)]
    max_gens: u64,

    /// Output CSV path. Default: `run_seed{seed}.csv`.
    out_path: Option<String>,

    /// Map noise seed.
    map_seed: Option<u64>,

    /// Mating radius in world units.
    mating_radius: Option<f32>,

    /// Initial cell count.
    initial_cells: Option<usize>,

    /// Population cap.
    max_population: Option<usize>,

    /// Rayon thread count (default: std::thread::available_parallelism()).
    threads: Option<usize>,

    /// Save checkpoint to PATH at end of run.
    #[arg(long, value_name = "PATH")]
    save: Option<String>,

    /// Load checkpoint from PATH instead of starting fresh.
    #[arg(long, value_name = "PATH")]
    load: Option<String>,

    /// BOND_FOOD_SHARE_FRAC runtime override (Hamilton sweep).
    #[arg(long, value_name = "FRAC")]
    share_frac: Option<f32>,

    /// Kin filter: bond food sharing restricted to same lineage_id.
    #[arg(long)]
    kin: bool,

    /// Predation gain multiplier.
    #[arg(long, value_name = "X")]
    pred_gain: Option<f32>,

    /// Predation drain multiplier.
    #[arg(long, value_name = "X")]
    pred_drain: Option<f32>,

    /// Food spawn multiplier.
    #[arg(long, value_name = "X")]
    food: Option<f32>,

    /// Mean generations between environmental shocks. 0 = empty calendar (default).
    #[arg(long, default_value_t = 0, value_name = "N")]
    shocks_mean_gens: u32,

    /// Maze difficulty (easy|medium|hard). Bare `--maze` defaults to medium.
    #[arg(
        long,
        value_name = "DIFFICULTY",
        num_args = 0..=1,
        default_missing_value = "medium",
    )]
    maze: Option<String>,

    /// Curriculum: `difficulty:gens,difficulty:gens,…` (final segment may omit
    /// length = rest of run). Implies `--maze` from the first stage.
    #[arg(long, value_name = "SPEC")]
    maze_stages: Option<String>,
}

fn parse_maze_lenient(s: &str) -> MazeDifficulty {
    match MazeDifficulty::parse(s) {
        Some(d) => d,
        None => {
            eprintln!(
                "warning: unknown --maze value '{s}', using medium. Valid: easy|medium|hard"
            );
            MazeDifficulty::Medium
        }
    }
}

fn parse_maze_stages(spec: &str) -> Vec<(MazeDifficulty, u64)> {
    let mut out: Vec<(MazeDifficulty, u64)> = Vec::new();
    let mut cum: u64 = 0;
    let parts: Vec<&str> = spec.split(',').collect();
    let last_idx = parts.len().saturating_sub(1);
    for (i, p) in parts.iter().enumerate() {
        let mut it = p.splitn(2, ':');
        let diff_str = it.next().unwrap_or("");
        let gens_str = it.next();
        let diff = match MazeDifficulty::parse(diff_str) {
            Some(d) => d,
            None => {
                eprintln!("warning: unknown stage difficulty '{diff_str}', skipping");
                continue;
            }
        };
        let end_gen = if i == last_idx && gens_str.is_none() {
            u64::MAX
        } else {
            let n = gens_str.and_then(|s| s.parse::<u64>().ok()).unwrap_or(50);
            cum = cum.saturating_add(n);
            cum
        };
        out.push((diff, end_gen));
    }
    out
}

fn main() {
    let cli = Cli::parse();

    let seed = cli.seed;
    let max_gens = cli.max_gens;
    let out_path = cli
        .out_path
        .unwrap_or_else(|| format!("run_seed{}.csv", seed));
    let map_seed = cli.map_seed.unwrap_or(WORLD_MAP_SEED);
    let mating_radius = cli.mating_radius.unwrap_or(MATING_RADIUS);
    let initial_cells = cli.initial_cells.unwrap_or(INITIAL_CELLS);
    let max_population = cli.max_population.unwrap_or(MAX_POPULATION);
    let threads = cli.threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });

    let shocks_mean_gens = cli.shocks_mean_gens;
    let save_path = cli.save;
    let load_path = cli.load;
    let share_frac_override = cli.share_frac;
    let kin_filter = cli.kin;
    let pred_gain_override = cli.pred_gain;
    let pred_drain_override = cli.pred_drain;
    let food_mult_override = cli.food;
    let maze_difficulty: Option<MazeDifficulty> = cli.maze.as_deref().map(parse_maze_lenient);
    let maze_stages: Vec<(MazeDifficulty, u64)> = cli
        .maze_stages
        .as_deref()
        .map(parse_maze_stages)
        .unwrap_or_default();

    if maze_difficulty.is_some() {
        eprintln!(
            "info: --maze (Wave 7): full GPU parity — step wall collision, FieldGpu masked diffusion, sensor_gather LOS + whisker raycast, and hebbian eligibility traces (per-tick decay+accumulate + on-event trace-based reward apply) all run on-device."
        );
    }
    if !maze_stages.is_empty() {
        eprintln!(
            "info: --maze-stages — same caveat as --maze (see startup info)."
        );
    }
    let initial_maze_difficulty = if !maze_stages.is_empty() {
        Some(maze_stages[0].0)
    } else {
        maze_difficulty
    };

    if threads > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global();
    }

    let mut rng = StdRng::seed_from_u64(seed);
    // Sprint 109: kalendář environmentálních shocků. Když mean_gens_between == 0,
    // generate vrátí prázdný kalendář (no-op) — byte-identical s pre-S109 baseline.
    let shock_cfg = if shocks_mean_gens > 0 {
        ShockScheduleConfig {
            mean_gens_between: shocks_mean_gens,
            ..Default::default()
        }
    } else {
        ShockScheduleConfig::default()
    };
    let events = EventCalendar::generate(seed, &shock_cfg, max_gens);
    // Sprint 111: pokud kalendář není prázdný, dump sidecar `events_seed{seed}.csv`
    // vedle hlavního CSV. Žádný file pokud `events` empty (žádný error).
    if !events.events.is_empty() {
        if let Err(e) = write_events_sidecar(Path::new(&out_path), seed, &events) {
            eprintln!("events sidecar: write failed ({e})");
        }
    }
    let mut world = if let Some(path) = load_path.as_ref() {
        match World::load_checkpoint(Path::new(path)) {
            Ok(mut w) => {
                eprintln!(
                    "checkpoint: loaded {} (cells={}, foods={}, gen={}, tick={})",
                    path,
                    w.cells.len(),
                    w.foods.len(),
                    w.clock.generation,
                    w.clock.tick,
                );
                w.events = events.clone();
                w
            }
            Err(e) => {
                eprintln!("checkpoint: load failed ({e}); starting fresh");
                World::new_with_maze(
                    &mut rng,
                    map_seed,
                    mating_radius,
                    initial_cells,
                    max_population,
                    events.clone(),
                    initial_maze_difficulty,
                )
            }
        }
    } else {
        World::new_with_maze(
            &mut rng,
            map_seed,
            mating_radius,
            initial_cells,
            max_population,
            events,
            initial_maze_difficulty,
        )
    };
    if !maze_stages.is_empty() {
        world.maze_curriculum = maze_stages.clone();
        let stages_str: Vec<String> = maze_stages
            .iter()
            .map(|(d, end)| {
                if *end == u64::MAX {
                    format!("{}:rest", d.label())
                } else {
                    format!("{}→gen{}", d.label(), end)
                }
            })
            .collect();
        eprintln!("maze curriculum: {}", stages_str.join(", "));
    }
    if let Some(d) = initial_maze_difficulty {
        if let Some(field) = world.obstacles.as_ref() {
            eprintln!(
                "maze: {} ({}×{} voxels, goal at [{:.0}, {:.0}])",
                d.label(),
                field.resolution[0],
                field.resolution[1],
                field.goal_position[0],
                field.goal_position[1],
            );
        }
    }
    // Sprint 87 Hamilton sweep: aplikuj CLI overrides AFTER World::new (i po
    // checkpoint load) — nikdy se neserializují, vždy z aktuálního CLI.
    if let Some(sf) = share_frac_override {
        world.share_frac = sf;
    }
    world.kin_filter = kin_filter;
    if let Some(v) = pred_gain_override {
        world.predation_gain_mult = v;
    }
    if let Some(v) = pred_drain_override {
        world.predation_drain_mult = v;
    }
    if let Some(v) = food_mult_override {
        world.food_factor_mult = v;
    }

    {
        let cap = initial_cells.max(max_population).max(64);
        let field_sources_cap = (food_target(1.0 + CYCLE_AMPLITUDE) + max_population) * 2;
        match world.init_gpu_full() {
            Ok(()) => {
                eprintln!(
                    "gpu-full: brain + Hebbian + Brownian + Field + SensorGather + PopulateInputs + Motor + Step (cap {} cells, {} field sources)",
                    cap, field_sources_cap
                );
            }
            Err(e) => {
                // Wave N: no CPU fallback. GPU is the only compute path —
                // if init fails the run can't proceed.
                eprintln!("gpu-full: init failed: {e}");
                std::process::exit(1);
            }
        }
    }

    let file = std::fs::File::create(&out_path).expect("can't create output file");
    let mut log = BufWriter::new(file);
    writeln!(
        log,
        "gen,cells,spd_avg,spd_dev,vis_avg,vis_dev,len_avg,wid_avg,hgt_avg,asp_avg,asp_dev,spk_avg,spk_max,food,density,lineages,oldest,ph_emit_ch0_avg,ph_emit_ch1_avg,ph_emit_ch2_avg,ph_emit_ch0_dev,ph_emit_ch1_dev,ph_emit_ch2_dev,ph_burst_score_ch0,ph_burst_score_ch1,ph_burst_score_ch2,abs_x,abs_y,edge_frac,corner_frac,mean_x,mean_y,energy_avg,births,deaths,fertile_ticks,atk_emit,predation_events,recurrent_io,nn_dist_avg,density_avg,density_dev,dmg_avg,noise_avg,bonds_formed,bonds_broken,mean_bond_count,bond_active_frac,bond_signal_avg,adhesion_entropy,bond_stiff_avg,bond_damp_avg,state_avg,state_dev,altruist_frac,fov_avg,fov_dev,temp_avg,topt_avg,topt_dev,carnivore_avg,gain_vis_avg,gain_chem_avg,gain_def_avg,gain_vis_dev,gain_chem_dev,gain_def_dev,cppn_compat,shock_active_count,shock_hazard_intensity_max,shock_climate_offset,shock_food_factor,lineage_count,behavioral_entropy_attack,weight_diversity_w1_norm,spike_count_avg,spike_complexity_avg,spike_total_length_avg,ticks_per_sec,coop_food_solved,coop_food_failed,coop_food_arrivals_avg,bonded_attack_eff,swarm_attack_frac,pack_attack_frac,vib_emit_avg,vib_amp_avg,vib_grad_mag_avg,gain_mech_avg,gain_mech_dev,maze_active,maze_in_goal_frac,maze_unique_reach_frac,maze_first_reach_total,lr_avg,lr_std,decay_avg,decay_std,w_norm_avg,neural_spike_frac"
    )
    .unwrap();
    write_stats(&mut log, &world, 0.0).unwrap();

    let baseline_samples = 10_000;
    let mut bsum = 0.0_f64;
    let mut brng = StdRng::seed_from_u64(99);
    for _ in 0..baseline_samples {
        let p = [
            brng.random_range(-WORLD_HALF[0]..WORLD_HALF[0]),
            brng.random_range(-WORLD_HALF[1]..WORLD_HALF[1]),
            brng.random_range(-WORLD_HALF[2]..WORLD_HALF[2]),
        ];
        bsum += world.map.sample(p) as f64;
    }
    let noise_baseline = bsum / baseline_samples as f64;
    eprintln!("noise_baseline (uniform-position mean over map): {:.4}", noise_baseline);

    eprintln!(
        "headless: seed={} map_seed={} mating_radius={} max_gens={} out={} initial_cells={} initial_food={} max_pop={} threads={}",
        seed,
        map_seed,
        mating_radius,
        max_gens,
        out_path,
        world.cells.len(),
        world.foods.len(),
        max_population,
        rayon::current_num_threads()
    );
    eprintln!(
        "shocks: mean_gens_between={} scheduled={} (sim loop integration arrives in S110+)",
        shocks_mean_gens,
        world.events.events.len()
    );

    let start = Instant::now();
    let mut gen_start = Instant::now();
    let mut gen_ticks = 0_u64;
    while world.clock.generation < max_gens {
        let gen_ended = world.tick(&mut rng);
        gen_ticks += 1;
        if gen_ended.is_some() {
            let gen_elapsed = gen_start.elapsed().as_secs_f64();
            let tps = if gen_elapsed > 0.0 {
                gen_ticks as f64 / gen_elapsed
            } else {
                0.0
            };
            // GPU CPPN keeps child brains GPU-resident; sync to CPU before the
            // stats pass reads `Genome.brain` for diagnostic metrics.
            world.sync_brains_from_gpu();
            // V7: the vibration field lives on the GPU in --gpu-full mode
            // (sensor gather reads it inline). Pull it back to the CPU shadow
            // so the per-gen CSV write samples real values, not zeros.
            world.sync_vibration_from_gpu();
            write_stats(&mut log, &world, tps).unwrap();
            // Sprint 126: reset burst_accum aby každá generace měřila vlastní
            // tick-to-tick variance. Bez resetu by hodnoty monotonně rostly.
            for cell in &mut world.cells {
                cell.burst_accum = [0.0; N_PHEROMONE_CHANNELS];
            }
            gen_start = Instant::now();
            gen_ticks = 0;
            // Sprint 43: po první dokončené generaci vypiš per-fáze timing
            // (mikrosekundy total + průměr per tick). Reset accumulator.
            if world.clock.generation == 1 {
                let t = world.bench_timings;
                let ticks = TICKS_PER_GENERATION as f64;
                let dump = |name: &str, total_us: f64| {
                    eprintln!(
                        "phase={} n={} ticks={} us_total={:.1} us_avg={:.3}",
                        name,
                        world.cells.len(),
                        TICKS_PER_GENERATION,
                        total_us,
                        total_us / ticks
                    );
                };
                dump("update_smell", t.update_smell);
                dump("update_pheromone", t.update_pheromone);
                dump("brain_act", t.brain_act);
                dump("emit_pheromones", t.emit_pheromones);
                dump("apply_morph", t.apply_morph);
                dump("apply_brownian", t.apply_brownian);
                dump("step", t.step);
                dump("apply_food_gravity", t.apply_food_gravity);
                dump("apply_hazards", t.apply_hazards);
                dump("resolve_collisions", t.resolve_collisions);
                dump("predate", t.predate);
                dump("eat_food", t.eat_food);
                dump("spawn_food", t.spawn_food);
                dump("reproduce", t.reproduce);
                dump("die_and_drop_carrion", t.die_and_drop_carrion);
                world.bench_timings = PhaseTimings::default();
            }
            world.births_gen = 0;
            world.deaths_gen = 0;
            world.fertile_ticks_gen = 0;
            world.predation_events_gen = 0;
            // Sprint 66: bond formation/break per-gen counters.
            world.bonds_formed_gen = 0;
            world.bonds_broken_gen = 0;
            world.bonded_attacks_gen = 0;
            world.solo_attacks_gen = 0;
            world.bonded_attack_gain_sum_gen = 0.0;
            world.solo_attack_gain_sum_gen = 0.0;
            world.swarm_attacks_gen = 0;
            world.pack_attacks_gen = 0;
            world.attack_victims_gen = 0;
            // Sprint 128: coop food per-gen counters.
            world.coop_food_solved_gen = 0;
            world.coop_food_failed_gen = 0;
            world.coop_food_arrivals_sum_gen = 0;
            world.coop_food_events_gen = 0;
            world.goal_zone_ticks_gen = 0;
            world.goal_unique_reachers_gen.clear();
        }
        if world.cells.is_empty() {
            eprintln!("extinction at gen {}", world.clock.generation);
            break;
        }
    }
    log.flush().unwrap();

    if let Some(path) = save_path.as_ref() {
        match world.save_checkpoint(Path::new(path)) {
            Ok(()) => eprintln!(
                "checkpoint: saved to {} (cells={}, gen={}, tick={})",
                path,
                world.cells.len(),
                world.clock.generation,
                world.clock.tick,
            ),
            Err(e) => eprintln!("checkpoint: save failed ({e})"),
        }
    }

    let elapsed = start.elapsed();
    let ticks_per_sec = world.clock.tick as f32 / elapsed.as_secs_f32().max(1e-3);
    eprintln!(
        "done. {} gen, {} ticks in {:.1}s ({:.0} ticks/s). final pop: {}",
        world.clock.generation,
        world.clock.tick,
        elapsed.as_secs_f32(),
        ticks_per_sec,
        world.cells.len()
    );
}
