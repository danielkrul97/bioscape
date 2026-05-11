//! Headless harness — pure simulation loop, no Bevy renderer.
//!
//! Usage: `cargo run --release --bin headless -- [seed] [max_gens] [out_path]`
//! Defaults: seed=0, max_gens=500, out_path=run_seed{seed}.csv
//!
//! Logs per-generation stats (cell_count, mean/dev for max_speed/vision/body_size,
//! food count, density factor) to CSV. Reproducible: same seed → identical run.

use bioscape::{
    EventCalendar, MazeDifficulty,
    ShockScheduleConfig, CYCLE_AMPLITUDE, GRID_CELL_SIZE,
    INITIAL_CELLS,
    MATING_RADIUS, MAX_POPULATION,
    N_PHEROMONE_CHANNELS,
    PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z, SMELL_GRID_RES, SMELL_GRID_RES_Z, TICKS_PER_GENERATION, WORLD_HALF, WORLD_MAP_SEED,
};
#[cfg(feature = "gpu")]
use bioscape::gpu::{
        BrainGpu, BrownianGpu, CellsGpu, CppnGpu, FieldGpu, GpuContext, GpuFullScratch,
        HebbianGpu, MotorGpu, PopulateInputsGpu, SensorGatherGpu, SpatialHashGpu, StepGpu,
    };
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

mod csv;
mod world;

use csv::*;
use world::*;

fn main() {
    let raw_args: Vec<String> = env::args().collect();
    // Sprint 44: `--gpu` flag (filtered před positional parsingem). Bez
    // `--features gpu` se flag tiše ignoruje.
    // Sprint 51: `--gpu-full` flag — persistent brain weights + GPU Hebbian +
    // GPU Brownian. Implies --gpu (brain forward na GPU).
    let want_gpu_full = raw_args.iter().any(|a| a == "--gpu-full");
    let want_gpu = want_gpu_full || raw_args.iter().any(|a| a == "--gpu");
    // Sprint 48: `--save=PATH` / `--load=PATH` checkpoint flags. Form
    // `--key=value` aby se PATH ne-leakoval do positional indexingu.
    let save_path: Option<String> = raw_args
        .iter()
        .find_map(|a| a.strip_prefix("--save=").map(|s| s.to_string()));
    let load_path: Option<String> = raw_args
        .iter()
        .find_map(|a| a.strip_prefix("--load=").map(|s| s.to_string()));
    // Sprint 87 Hamilton sweep: `--share-frac=X` runtime override pro
    // BOND_FOOD_SHARE_FRAC, `--kin` zapne kin filter (food share jen na
    // partnery se stejným lineage_id).
    let share_frac_override: Option<f32> = raw_args
        .iter()
        .find_map(|a| a.strip_prefix("--share-frac=").and_then(|s| s.parse().ok()));
    let kin_filter = raw_args.iter().any(|a| a == "--kin");
    let pred_gain_override: Option<f32> = raw_args
        .iter()
        .find_map(|a| a.strip_prefix("--pred-gain=").and_then(|s| s.parse().ok()));
    let pred_drain_override: Option<f32> = raw_args
        .iter()
        .find_map(|a| a.strip_prefix("--pred-drain=").and_then(|s| s.parse().ok()));
    let food_mult_override: Option<f32> = raw_args
        .iter()
        .find_map(|a| a.strip_prefix("--food=").and_then(|s| s.parse().ok()));
    // Sprint 109: `--shocks-mean-gens N` (space-separated) nebo
    // `--shocks-mean-gens=N` (= form). Default 0 = no-op (empty kalendář).
    // `consumed_value_idx` drží pozici následujícího raw arg pokud je flag
    // space-separated; ten se musí vyfiltrovat z positional setu.
    let mut shocks_mean_gens: u32 = 0;
    let mut consumed_value_idx: Option<usize> = None;
    for (i, a) in raw_args.iter().enumerate() {
        if let Some(rest) = a.strip_prefix("--shocks-mean-gens") {
            if let Some(eq_val) = rest.strip_prefix('=') {
                if let Ok(v) = eq_val.parse::<u32>() {
                    shocks_mean_gens = v;
                }
            } else if rest.is_empty() {
                if let Some(next) = raw_args.get(i + 1) {
                    if let Ok(v) = next.parse::<u32>() {
                        shocks_mean_gens = v;
                        consumed_value_idx = Some(i + 1);
                    }
                }
            }
            break;
        }
    }
    // Maze toggle: `--maze` (default medium) or `--maze=easy|medium|hard`.
    // GPU paths don't yet honour walls/LOS/masks, so combining `--maze` with
    // `--gpu`/`--gpu-full` is rejected up-front to avoid misleading runs.
    let maze_difficulty: Option<MazeDifficulty> = raw_args.iter().find_map(|a| {
        if a == "--maze" {
            Some(MazeDifficulty::Medium)
        } else if let Some(val) = a.strip_prefix("--maze=") {
            match MazeDifficulty::parse(val) {
                Some(d) => Some(d),
                None => {
                    eprintln!(
                        "warning: unknown --maze value '{val}', using medium. Valid: easy|medium|hard"
                    );
                    Some(MazeDifficulty::Medium)
                }
            }
        } else {
            None
        }
    });
    if maze_difficulty.is_some() && want_gpu {
        eprintln!(
            "info: --maze + --gpu (Wave 5): step wall collision, FieldGpu masked diffusion, and sensor_gather LOS are GPU-aware. Whisker raycast still reads 0 in shader (CPU pre-pass populates last_whisker_distances but GPU populate_inputs zeroes the slot); hebbian eligibility traces run the CPU path against the persistent GPU brain weight buffer — patches reach GPU at next-gen sync. Wave 6 brings full parity."
        );
    }
    // Wave 3 curriculum ramp: --maze-stages=easy:50,medium:100,hard
    // Each segment is "difficulty:gens_in_segment". The final segment may omit
    // its length (interpreted as "rest of run"). Implies --maze automatically
    // (the first stage's difficulty seeds the initial obstacle field).
    let maze_stages: Vec<(MazeDifficulty, u64)> = raw_args
        .iter()
        .find_map(|a| a.strip_prefix("--maze-stages="))
        .map(|spec| {
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
        })
        .unwrap_or_default();
    if !maze_stages.is_empty() && want_gpu {
        eprintln!(
            "info: --maze-stages + --gpu — same caveat as --maze + --gpu (see startup info)."
        );
    }
    let initial_maze_difficulty = if !maze_stages.is_empty() {
        Some(maze_stages[0].0)
    } else {
        maze_difficulty
    };
    let args: Vec<String> = raw_args
        .iter()
        .enumerate()
        .filter(|(i, a)| !a.starts_with("--") && Some(*i) != consumed_value_idx)
        .map(|(_, a)| a.clone())
        .collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let max_gens: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(500);
    let out_path = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| format!("run_seed{}.csv", seed));
    let map_seed: u64 = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(WORLD_MAP_SEED);
    let mating_radius: f32 = args
        .get(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(MATING_RADIUS);
    // Sprint 43: positional override pro initial cells / max population /
    // rayon thread count. Default zachovává pre-Sprint-43 chování.
    let initial_cells: usize = args
        .get(6)
        .and_then(|s| s.parse().ok())
        .unwrap_or(INITIAL_CELLS);
    let max_population: usize = args
        .get(7)
        .and_then(|s| s.parse().ok())
        .unwrap_or(MAX_POPULATION);
    let threads: usize = args
        .get(8)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });

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

    #[cfg(feature = "gpu")]
    if want_gpu_full {
        let cap = initial_cells.max(max_population).max(64);
        // Sprint 59: FieldGpu sources capacity. Per-tick deposit count =
        // foods (smell) + cells (pheromone). food_target může bumpnout přes
        // density cycles (CYCLE_AMPLITUDE), s safety margin × 2.
        let field_sources_cap = (food_target(1.0 + CYCLE_AMPLITUDE) + max_population) * 2;
        let init = || -> Result<GpuFullState, String> {
            let ctx = GpuContext::new()?;
            let cells_gpu = CellsGpu::with_context(&ctx, cap);
            cells_gpu.upload_brains(world.cells.iter().map(|c| &c.genome.brain));
            // V7-unification: seed from `cell_id` so the GPU per-slot stream
            // matches the CPU `Cell.xoshiro_state` (also derived from cell_id
            // at spawn). Tests now expect CPU and GPU brownian outputs to be
            // byte-identical for any cell with a given cell_id.
            cells_gpu.upload_xoshiro_seeds(world.cells.iter().map(|c| c.cell_id));
            let brain = BrainGpu::with_context(&ctx, cap)?;
            let hebbian = HebbianGpu::with_context(&ctx, cap)?;
            let brownian = BrownianGpu::with_context(&ctx, cap)?;
            // Sprint 59: smell + pheromone FieldGpu instances, sdílí GpuContext.
            let smell = FieldGpu::with_context(
                &ctx,
                [SMELL_GRID_RES, SMELL_GRID_RES, SMELL_GRID_RES_Z],
                WORLD_HALF,
                field_sources_cap,
            )?;
            let pheromone = FieldGpu::with_context(
                &ctx,
                [PHEROMONE_GRID_RES, PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z],
                WORLD_HALF,
                field_sources_cap,
            )?;
            // V7: motion-driven vibration field on GPU. Source capacity is
            // `cap` (one deposit per cell per tick — far fewer sources than
            // smell which also takes food).
            let vibration = FieldGpu::with_context(
                &ctx,
                [
                    bioscape::VIBRATION_GRID_RES,
                    bioscape::VIBRATION_GRID_RES,
                    bioscape::VIBRATION_GRID_RES_Z,
                ],
                WORLD_HALF,
                cap,
            )?;
            // Sprint 60: spatial hashes pro sensor broad-phase. Sdílí
            // GRID_CELL_SIZE konstantu s CPU SpatialGrid; xy world bounds
            // pro toroidal bucket wrap.
            let cell_hash = SpatialHashGpu::with_context(
                &ctx,
                cap,
                GRID_CELL_SIZE,
                [WORLD_HALF[0], WORLD_HALF[1]],
            )?;
            let food_capacity = field_sources_cap;
            let food_hash = SpatialHashGpu::with_context(
                &ctx,
                food_capacity,
                GRID_CELL_SIZE,
                [WORLD_HALF[0], WORLD_HALF[1]],
            )?;
            let sensor = SensorGatherGpu::with_context(&ctx, cap, food_capacity)?;
            let populate = PopulateInputsGpu::with_context(&ctx)?;
            let motor = MotorGpu::with_context(&ctx, cap)?;
            let step = StepGpu::with_context(&ctx, cap)?;
            // GPU CPPN materialises child brain weights direct → cells.brain_weights_buf.
            // Capacity = cap (worst case: all cells reproduce in one tick after a
            // mass extinction; init upload is cap children too).
            let cppn = CppnGpu::with_context(&ctx, cap);
            // Sprint 62: turn_rate je per-cell genome konstanta. Upload na sim
            // init; reproduce volá `upload_turn_rates` znovu (per-event sparse).
            let turn_rates: Vec<f32> = world.cells.iter().map(|c| c.genome.turn_rate).collect();
            cells_gpu.upload_turn_rates(&turn_rates);
            Ok(GpuFullState {
                cells: cells_gpu,
                brain,
                hebbian,
                brownian,
                smell,
                pheromone,
                vibration,
                cell_hash,
                food_hash,
                sensor,
                populate,
                motor,
                step,
                cppn,
                scratch: GpuFullScratch::default(),
            })
        };
        match init() {
            Ok(state) => {
                eprintln!(
                    "gpu-full: brain + Hebbian + Brownian + Field + SensorGather + PopulateInputs + Motor + Step (cap {} cells, {} field sources)",
                    cap, field_sources_cap
                );
                world.gpu_full = Some(state);
                // Wave 4: upload maze masks once if obstacles already present
                // at init time. Curriculum rebuilds re-upload via tick path.
                if let Some(field) = world.obstacles.as_ref() {
                    let packed = field.packed_for_gpu();
                    let smell_mask =
                        field.mask_for_grid([SMELL_GRID_RES, SMELL_GRID_RES, SMELL_GRID_RES_Z]);
                    let phero_mask = field.mask_for_grid([
                        PHEROMONE_GRID_RES,
                        PHEROMONE_GRID_RES,
                        PHEROMONE_GRID_RES_Z,
                    ]);
                    let vib_mask = field.mask_for_grid([
                        bioscape::VIBRATION_GRID_RES,
                        bioscape::VIBRATION_GRID_RES,
                        bioscape::VIBRATION_GRID_RES_Z,
                    ]);
                    if let Some(gpu) = world.gpu_full.as_mut() {
                        gpu.step.upload_maze(&packed);
                        gpu.sensor.upload_maze(&packed);
                        gpu.smell.upload_obstacle_mask(&smell_mask);
                        gpu.pheromone.upload_obstacle_mask(&phero_mask);
                        gpu.vibration.upload_obstacle_mask(&vib_mask);
                    }
                }
            }
            Err(e) => {
                eprintln!("gpu-full: init failed ({e}); fallback to CPU");
            }
        }
    }
    #[cfg(feature = "gpu")]
    if want_gpu && !want_gpu_full && world.gpu_full.is_none() {
        match BrainGpu::new(initial_cells.max(64)) {
            Ok(g) => {
                eprintln!("gpu: BrainGpu initialized (capacity {})", initial_cells.max(64));
                world.gpu = Some(g);
            }
            Err(e) => {
                eprintln!("gpu: init failed ({e}); falling back to CPU");
            }
        }
    }
    #[cfg(not(feature = "gpu"))]
    if want_gpu {
        eprintln!("gpu: --gpu / --gpu-full requested but binary built without --features gpu");
    }

    let file = std::fs::File::create(&out_path).expect("can't create output file");
    let mut log = BufWriter::new(file);
    writeln!(
        log,
        "gen,cells,spd_avg,spd_dev,vis_avg,vis_dev,len_avg,wid_avg,hgt_avg,asp_avg,asp_dev,spk_avg,spk_max,food,density,lineages,oldest,ph_emit_ch0_avg,ph_emit_ch1_avg,ph_emit_ch2_avg,ph_emit_ch0_dev,ph_emit_ch1_dev,ph_emit_ch2_dev,ph_burst_score_ch0,ph_burst_score_ch1,ph_burst_score_ch2,abs_x,abs_y,edge_frac,corner_frac,mean_x,mean_y,energy_avg,births,deaths,fertile_ticks,atk_emit,predation_events,recurrent_io,nn_dist_avg,density_avg,density_dev,dmg_avg,noise_avg,bonds_formed,bonds_broken,mean_bond_count,bond_active_frac,bond_signal_avg,adhesion_entropy,bond_stiff_avg,bond_damp_avg,state_avg,state_dev,altruist_frac,fov_avg,fov_dev,temp_avg,topt_avg,topt_dev,carnivore_avg,gain_vis_avg,gain_chem_avg,gain_def_avg,gain_vis_dev,gain_chem_dev,gain_def_dev,cppn_compat,shock_active_count,shock_hazard_intensity_max,shock_climate_offset,shock_food_factor,lineage_count,behavioral_entropy_attack,weight_diversity_w1_norm,spike_count_avg,spike_complexity_avg,spike_total_length_avg,ticks_per_sec,coop_food_solved,coop_food_failed,coop_food_arrivals_avg,bonded_attack_eff,swarm_attack_frac,pack_attack_frac,vib_emit_avg,vib_amp_avg,vib_grad_mag_avg,gain_mech_avg,gain_mech_dev,maze_active,maze_in_goal_frac,maze_unique_reach_frac,maze_first_reach_total"
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
            #[cfg(feature = "gpu")]
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
