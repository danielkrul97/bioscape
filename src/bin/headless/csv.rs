//! CSV per-generation logging and analytics for headless harness.

use bioscape::sim::{World, EDGE_FRAC_THRESHOLD};

use bioscape::{
    Cell, EventCalendar, SpatialGrid, BRAIN_INPUTS_SENSORY, BRAIN_RECURRENT, GRID_CELL_SIZE,
    N_PHEROMONE_CHANNELS, TICKS_PER_GENERATION, WORLD_HALF,
};
use std::io::{BufWriter, Write};
use std::path::Path;

/// Sprint 48: versioned binary header pro checkpoint files.
/// Sprint 66 bump: Cell rozšířen o `cell_id` + `bonds`, BRAIN_OUTPUTS 9→10.
/// Sprint 68 bump: Genome rozšířen o `bond_stiffness` + `bond_damping`,
/// Bond rozšířen o `stiffness` + `damping`. Starší V1/V2 už nelze
/// deserializovat — load vrací error.
/// Sprint 103 bump: BRAIN_HIDDEN 32→50, BRAIN_INPUTS 53→71. Brain weight
/// arrays resize → V3 savefiles incompatible.
/// Sprint 126 bump: multi-channel pheromones (3 fields), BRAIN_INPUTS_SENSORY
/// 21→27, BRAIN_INPUTS 71→77, BRAIN_OUTPUTS 10→12. V4 savefiles incompatible.
pub fn write_stats<W: Write>(w: &mut W, world: &World, ticks_per_sec: f64) -> std::io::Result<()> {
    let n = world.cells.len();
    if n == 0 {
        // Empty-pop row: zero-padded cell metrics + actual world counters.
        // Sprint 111: shocks exist nezávisle na cell pop, takže reportujeme.
        // Sprint 128: coop events also independent.
        let (shock_active, shock_hazard_max) = shock_summary(world);
        let shock_food_factor =
            bioscape::food_density_shock_multiplier(&world.events.events, world.clock.generation);
        let coop_arrivals_avg = if world.coop_food_events_gen > 0 {
            world.coop_food_arrivals_sum_gen as f64 / world.coop_food_events_gen as f64
        } else {
            0.0
        };
        let maze_active: u8 = if world.obstacles.is_some() { 1 } else { 0 };
        return writeln!(
            w,
            // 86 columns total (82 base + 4 maze). Layout:
            //   gen, cells, 11×0(cell metrics), food, density,
            //   17×0(lineages..mean_y), 0(energy_avg), births, deaths, fertile_ticks,
            //   0(atk_emit), predation, 6×0(recurrent..noise),
            //   bonds_formed, bonds_broken, 21×0(bond..gain_def_dev), 0(cppn_compat),
            //   shock_active, shock_hazard_max, 0.000(climate), shock_food_factor,
            //   0(lineage_count), 0.000, 0.000, 3×0(spike),
            //   ticks_per_sec, coop_solved, coop_failed, coop_arrivals_avg,
            //   bonded_attack_eff, swarm_attack_frac, pack_attack_frac,
            //   maze_active, maze_in_goal_frac, maze_unique_reach_frac, maze_first_reach_total
            "{},0,0,0,0,0,0,0,0,0,0,0,0,{},{:.3},0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,{},{},{},0,{},0,0,0,0,0,0,{},{},0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,{},{:.3},0.000,{:.3},0,0.000,0.000,0,0,0,{:.1},{},{},{:.3},0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,{},0.000,0.000,{},0.000000,0.000000,0.0000,0.0000,0.000,0.0000,0.0000,0.00,0.00,0.00,0.00,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.0000,0.0000,0.000,0.0000,0.000,0.000,0.000,0.000,0.000,0,0.00,0,0.000,0.000",
            world.clock.generation,
            world.foods.len(),
            world.density_factor,
            world.births_gen,
            world.deaths_gen,
            world.fertile_ticks_gen,
            world.predation_events_gen,
            world.bonds_formed_gen,
            world.bonds_broken_gen,
            shock_active,
            shock_hazard_max,
            shock_food_factor,
            ticks_per_sec,
            world.coop_food_solved_gen,
            world.coop_food_failed_gen,
            coop_arrivals_avg,
            maze_active,
            world.goal_first_reach_tick.len() as u64,
        );
    }
    let mut spd_sum = 0.0_f64;
    let mut spd_sumsq = 0.0_f64;
    let mut vis_sum = 0.0_f64;
    let mut vis_sumsq = 0.0_f64;
    let mut len_sum = 0.0_f64;
    let mut wid_sum = 0.0_f64;
    let mut hgt_sum = 0.0_f64;
    let mut asp_sum = 0.0_f64;
    let mut asp_sumsq = 0.0_f64;
    let mut spk_sum = 0.0_f64;
    let mut spk_max = 0.0_f64;
    let mut spike_count_sum = 0_u64;
    let mut spike_complexity_sum = 0.0_f64;
    let mut spike_total_length_sum = 0.0_f64;
    // Sprint 126: per-channel emit statistics. ph_emit_sum nahrazen polem
    // (avg + sumsq pro stddev). burst_score = mean ((emit_now - emit_prev)²)
    // — vyšší = víc bursty (vs. continuous emission).
    let mut ph_emit_sums = [0.0_f64; N_PHEROMONE_CHANNELS];
    let mut ph_emit_sumsq = [0.0_f64; N_PHEROMONE_CHANNELS];
    let mut ph_burst_sums = [0.0_f64; N_PHEROMONE_CHANNELS];
    let mut atk_emit_sum = 0.0_f64;
    let mut recurrent_io_sum = 0.0_f64;
    let mut density_sum = 0.0_f64;
    let mut density_sumsq = 0.0_f64;
    let mut dmg_sum = 0.0_f64;
    let mut noise_sum = 0.0_f64;
    let mut energy_sum = 0.0_f64;
    let mut abs_x_sum = 0.0_f64;
    let mut abs_y_sum = 0.0_f64;
    let mut x_sum = 0.0_f64;
    let mut y_sum = 0.0_f64;
    let mut edge_count = 0_u64;
    let mut corner_count = 0_u64;
    let mut lineages: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut oldest_age: u64 = 0;
    // Sprint 67 bond/adhesion diagnostics: per-cell aggregates pro CSV.
    let mut bond_signal_sum = 0.0_f64;
    let mut total_bonds = 0_u64;
    let mut bonded_cells = 0_u64;
    let mut adhesion_hist = [0_u64; bioscape::ADHESION_TYPE_COUNT as usize];
    // Sprint 68: gene-level diagnostics — mean of bond_stiffness / bond_damping
    // přes celou populaci. Porovnáním s initial draw centerem (BOND_STIFFNESS=4.0,
    // BOND_DAMPING=0.6) se dá detekovat selekční drift.
    let mut bond_stiff_sum = 0.0_f64;
    let mut bond_damp_sum = 0.0_f64;
    // Sprint 80: bistable cell_state diagnostics. Mean + std + altruist_frac
    // (cells s state > 0.6 = committed k altruist attractoru). Bimodální
    // distribuce (mean okolo 0.5, std velký, altruist_frac mezi 0.3-0.7) =
    // očekávaný steady-state. Unimodální drift k 0 = celá populace selfish
    // = food share zcela vypnutý → tissue regime collapsuje.
    let mut state_sum = 0.0_f64;
    let mut state_sumsq = 0.0_f64;
    let mut altruist_count = 0_u64;
    // Sprint 83: per-gen FOV diagnostika. Init populace = π (full sphere);
    // pokles avg + růst dev = aktivní selekce na užší kužel.
    let mut fov_sum = 0.0_f64;
    let mut fov_sumsq = 0.0_f64;
    // Sprint 85: thermal niche metric. Mean temperature na cell pozicích —
    // posun směrem k THERMAL_BOTTOM (~4) = populace migrovala ke dnu (cold,
    // pomalý metabolism), posun k THERMAL_TOP (~30) = surface dwellers.
    let mut temp_sum = 0.0_f64;
    // Sprint 87: thermal_optimum gen statistics. Mean + std → speciation
    // ukazuje, jak se populace rozčlenila mezi cold-prefer / warm-prefer
    // fenotypy. Gen 0 mean ≈ (BOTTOM+TOP)/2 = 17 (uniform init), std ≈ 7.5
    // (uniform sigma).
    let mut topt_sum = 0.0_f64;
    let mut topt_sumsq = 0.0_f64;
    // Sprint 93: per-cell diet diagnostic.
    let mut carnivore_sum = 0.0_f64;
    // Sprint 97: per-category sensor_gain means + stddev. Specialization
    // signal — high stddev (bimodální distribuce) = role differentiation
    // mezi cells. Low stddev s drift v meanu = uniform shift, pokles bez
    // specializace. Mean alone nerozliší tyto dva režimy.
    let mut gain_sum = [0.0_f64; bioscape::N_SENSOR_CATEGORIES];
    let mut gain_sumsq = [0.0_f64; bioscape::N_SENSOR_CATEGORIES];
    for c in &world.cells {
        carnivore_sum += c.genome.carnivore_score as f64;
        for k in 0..bioscape::N_SENSOR_CATEGORIES {
            let g = c.genome.sensor_gains[k] as f64;
            gain_sum[k] += g;
            gain_sumsq[k] += g * g;
        }
    }
    let cell_count = world.cells.len();
    let carnivore_m = if cell_count > 0 { carnivore_sum / cell_count as f64 } else { 0.0 };
    let gain_mean_dev = |k: usize| -> (f64, f64) {
        if cell_count == 0 { return (0.0, 0.0); }
        let n = cell_count as f64;
        let m = gain_sum[k] / n;
        let var = (gain_sumsq[k] / n - m * m).max(0.0);
        (m, var.sqrt())
    };
    let (gain_vis_m, gain_vis_d) = gain_mean_dev(bioscape::SENSOR_CATEGORY_VISION);
    let (gain_chem_m, gain_chem_d) = gain_mean_dev(bioscape::SENSOR_CATEGORY_CHEMISTRY);
    let (gain_def_m, gain_def_d) = gain_mean_dev(bioscape::SENSOR_CATEGORY_DEFENSIVE);
    let (gain_mech_m, gain_mech_d) = gain_mean_dev(bioscape::SENSOR_CATEGORY_MECHANO);
    // Vibration field stats — per-cell aggregates so the row reports what
    // the brain actually saw this tick (sample + gradient at each cell's
    // position), plus the population's mean emission rate.
    let mut vib_emit_sum = 0.0_f64;
    let mut vib_amp_sum = 0.0_f64;
    let mut vib_grad_mag_sum = 0.0_f64;
    for c in &world.cells {
        let pos = c.position;
        vib_emit_sum += bioscape::vibration_emit_for_cell(c) as f64;
        vib_amp_sum += world.vibration.sample(pos) as f64;
        let g = world
            .vibration
            .gradient_at(pos, bioscape::VIBRATION_SAMPLE_EPSILON);
        vib_grad_mag_sum += ((g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) as f64).sqrt();
    }
    let vib_emit_m = if cell_count > 0 { vib_emit_sum / cell_count as f64 } else { 0.0 };
    let vib_amp_m = if cell_count > 0 { vib_amp_sum / cell_count as f64 } else { 0.0 };
    let vib_grad_mag_m = if cell_count > 0 { vib_grad_mag_sum / cell_count as f64 } else { 0.0 };
    // Sprint 107: speciation distance diagnostic. Sample N pairs random,
    // průměr compatibility_distance na CPPN. Indicator of population-wide
    // genetic divergence (vyšší = víc speciation).
    let cppn_compat_m: f64 = {
        let n_cells = world.cells.len();
        if n_cells < 2 {
            0.0
        } else {
            let pairs = (n_cells.min(50)).max(2);
            let mut sum = 0.0_f64;
            let mut count = 0u32;
            for k in 0..pairs {
                let i = k % n_cells;
                let j = (k + n_cells / 2) % n_cells;
                if i != j {
                    sum += bioscape::Cppn::compatibility_distance(
                        &world.cells[i].genome.cppn,
                        &world.cells[j].genome.cppn,
                    ) as f64;
                    count += 1;
                }
            }
            if count > 0 {
                sum / count as f64
            } else {
                0.0
            }
        }
    };
    let current_gen = world.clock.generation;
    for c in &world.cells {
        let s = c.genome.max_speed as f64;
        let v = c.genome.vision_radius as f64;
        let l = c.phenotype.body_length as f64;
        let wd = c.phenotype.body_width as f64;
        let hg = c.phenotype.body_height as f64;
        let aspect = if wd > 1e-6 { l / wd } else { 0.0 };
        let spk = c.phenotype.primary_spike_length() as f64;
        spd_sum += s;
        spd_sumsq += s * s;
        vis_sum += v;
        vis_sumsq += v * v;
        len_sum += l;
        wid_sum += wd;
        hgt_sum += hg;
        asp_sum += aspect;
        asp_sumsq += aspect * aspect;
        spk_sum += spk;
        let fov = c.genome.vision_fov as f64;
        fov_sum += fov;
        fov_sumsq += fov * fov;
        // Sprint 86: temperature includes diurnal + seasonal cycles. Snapshot
        // se bere v aktuálním clock state — temp_avg pak odráží environmental
        // teplotu cells **v okamžiku end-of-gen**, ne time-averaged.
        temp_sum += bioscape::temperature_at_z(
            c.position[2],
            WORLD_HALF,
            world.clock.tick,
            world.clock.generation,
        ) as f64;
        let topt = c.genome.thermal_optimum as f64;
        topt_sum += topt;
        topt_sumsq += topt * topt;
        if spk > spk_max {
            spk_max = spk;
        }
        // Sprint 123: per-cell aggregates pro multi-spike observability.
        spike_count_sum += c.phenotype.spike_count as u64;
        spike_total_length_sum += c.phenotype.total_spike_length() as f64;
        let active_n = c.phenotype.spike_count.min(bioscape::SPIKE_SLOTS as u8) as usize;
        if active_n > 0 {
            let mut cmplx_sum = 0.0_f64;
            for slot in 0..active_n {
                cmplx_sum += c.phenotype.spikes[slot].complexity as f64;
            }
            spike_complexity_sum += cmplx_sum / active_n as f64;
        }
        // Sprint 126: per-channel emit aggregates + burst score (累积 variance
        // ze squared tick-to-tick deltas, /TICKS_PER_GENERATION pro mean).
        // last_emit reflektuje právě emitovanou hodnotu (post emit_pheromones).
        for ch in 0..N_PHEROMONE_CHANNELS {
            let cur = c.last_emit[ch] as f64;
            ph_emit_sums[ch] += cur;
            ph_emit_sumsq[ch] += cur * cur;
            ph_burst_sums[ch] += c.burst_accum[ch] as f64;
        }
        atk_emit_sum += c.last_outputs[6].max(0.0) as f64;
        // Sprint 28 adoption metric: jak silně se používá recurrent state.
        // Mean |last_hidden| napříč 8 dimenzemi → ∈ [0, 1]. Pokud je ~0,
        // brain ignoruje paměť (recurrent weights se nepřipojily k hidden);
        // pokud roste, paměť je aktivně modulovaná.
        let mut h_abs = 0.0_f64;
        for &h in c.last_hidden.iter() {
            h_abs += h.abs() as f64;
        }
        recurrent_io_sum += h_abs / BRAIN_RECURRENT as f64;
        // Sprint 29 quorum sensing metric: index 13 = local_density input.
        let dens = c.last_inputs[13] as f64;
        density_sum += dens;
        density_sumsq += dens * dens;
        // Sprint 30 damage signal adoption: index 14 = damage input. Když roste
        // napříč generacemi, populace je pod selekčním tlakem (predace/hazard),
        // pokud zůstává ~0, žádná nedobrovolná energy loss neprobíhá.
        dmg_sum += c.last_inputs[14] as f64;
        noise_sum += world.map.sample([c.position[0], c.position[1], c.position[2]]) as f64;
        energy_sum += c.energy as f64;
        let nx = (c.position[0] / WORLD_HALF[0]).clamp(-1.0, 1.0);
        let ny = (c.position[1] / WORLD_HALF[1]).clamp(-1.0, 1.0);
        let ax = nx.abs();
        let ay = ny.abs();
        x_sum += nx as f64;
        y_sum += ny as f64;
        abs_x_sum += ax as f64;
        abs_y_sum += ay as f64;
        let near_x = ax >= EDGE_FRAC_THRESHOLD;
        let near_y = ay >= EDGE_FRAC_THRESHOLD;
        if near_x || near_y {
            edge_count += 1;
        }
        if near_x && near_y {
            corner_count += 1;
        }
        lineages.insert(c.lineage_id);
        let age = current_gen.saturating_sub(c.lineage_birth_gen);
        if age > oldest_age {
            oldest_age = age;
        }
        // Sprint 67 bond/adhesion diagnostics.
        bond_signal_sum += c.last_outputs[9].max(0.0) as f64;
        let cell_bonds = c.bonds.iter().filter(|b| b.is_some()).count() as u64;
        total_bonds += cell_bonds;
        if cell_bonds > 0 {
            bonded_cells += 1;
        }
        let t_idx = (c.genome.adhesion_type as usize) % adhesion_hist.len();
        adhesion_hist[t_idx] += 1;
        // Sprint 68 gene means.
        bond_stiff_sum += c.genome.bond_stiffness as f64;
        bond_damp_sum += c.genome.bond_damping as f64;
        // Sprint 80 cell_state.
        let cs = c.cell_state as f64;
        state_sum += cs;
        state_sumsq += cs * cs;
        if cs > 0.6 {
            altruist_count += 1;
        }
    }
    let nf = n as f64;
    let spd_m = spd_sum / nf;
    let vis_m = vis_sum / nf;
    let len_m = len_sum / nf;
    let wid_m = wid_sum / nf;
    let hgt_m = hgt_sum / nf;
    let asp_m = asp_sum / nf;
    let spk_m = spk_sum / nf;
    let spd_d = ((spd_sumsq / nf) - spd_m * spd_m).max(0.0).sqrt();
    let vis_d = ((vis_sumsq / nf) - vis_m * vis_m).max(0.0).sqrt();
    let asp_d = ((asp_sumsq / nf) - asp_m * asp_m).max(0.0).sqrt();
    // Sprint 126: per-channel emit/burst means + std. ph_burst is normalized
    // per-tick (suma squared deltas / ticks_per_gen) — comparable napříč gens.
    let ticks_per_gen = TICKS_PER_GENERATION as f64;
    let mut ph_emit_m = [0.0_f64; N_PHEROMONE_CHANNELS];
    let mut ph_emit_d = [0.0_f64; N_PHEROMONE_CHANNELS];
    let mut ph_burst_m = [0.0_f64; N_PHEROMONE_CHANNELS];
    for ch in 0..N_PHEROMONE_CHANNELS {
        let m = ph_emit_sums[ch] / nf;
        ph_emit_m[ch] = m;
        ph_emit_d[ch] = ((ph_emit_sumsq[ch] / nf) - m * m).max(0.0).sqrt();
        ph_burst_m[ch] = ph_burst_sums[ch] / (nf * ticks_per_gen);
    }
    let atk_emit_m = atk_emit_sum / nf;
    let recurrent_io_m = recurrent_io_sum / nf;
    let energy_m = energy_sum / nf;
    let abs_x_m = abs_x_sum / nf;
    let abs_y_m = abs_y_sum / nf;
    let x_m = x_sum / nf;
    let y_m = y_sum / nf;
    let edge_f = edge_count as f64 / nf;
    let corner_f = corner_count as f64 / nf;
    let density_m = density_sum / nf;
    let density_d = ((density_sumsq / nf) - density_m * density_m).max(0.0).sqrt();
    let dmg_m = dmg_sum / nf;
    let noise_m = noise_sum / nf;
    // Sprint 67 bond/adhesion diagnostics.
    let bond_signal_m = bond_signal_sum / nf;
    let mean_bond_count = total_bonds as f64 / nf;
    let bond_active_frac = bonded_cells as f64 / nf;
    // Shannon entropy of adhesion_type distribution, normalized by log2(K)
    // (K = ADHESION_TYPE_COUNT) — uniformní distribuce → 1.0, monokultura → 0.0.
    let adhesion_entropy = {
        let k = adhesion_hist.len() as f64;
        if k <= 1.0 {
            0.0
        } else {
            let mut h = 0.0_f64;
            for &count in adhesion_hist.iter() {
                if count == 0 {
                    continue;
                }
                let p = count as f64 / nf;
                h -= p * p.log2();
            }
            h / k.log2()
        }
    };
    // Sprint 68 gene means.
    let bond_stiff_m = bond_stiff_sum / nf;
    let bond_damp_m = bond_damp_sum / nf;
    // Sprint 80 cell_state stats.
    let state_m = state_sum / nf;
    let state_d = ((state_sumsq / nf) - state_m * state_m).max(0.0).sqrt();
    let altruist_frac = altruist_count as f64 / nf;
    // Sprint 83 vision_fov stats.
    let fov_m = fov_sum / nf;
    let fov_d = ((fov_sumsq / nf) - fov_m * fov_m).max(0.0).sqrt();
    // Sprint 85 thermal niche — mean teplota.
    let temp_m = temp_sum / nf;
    // Sprint 87 thermal_optimum gen.
    let topt_m = topt_sum / nf;
    let topt_d = ((topt_sumsq / nf) - topt_m * topt_m).max(0.0).sqrt();
    // Sprint 29 spatial clustering metric: mean nearest-neighbor distance.
    // Sprint 43: grid lookup s expanding radius. Začni na GRID_CELL_SIZE (=64),
    // pokud nikdo není, double až po WORLD diagonal — typický nn dist je < 50,
    // takže first try téměř vždy najde souseda.
    let nn_dist_m = if n >= 2 {
        let mut grid: SpatialGrid<usize, ()> = SpatialGrid::new(GRID_CELL_SIZE, WORLD_HALF);
        grid.rebuild(
            world
                .cells
                .iter()
                .enumerate()
                .map(|(i, c)| (i, c.position, ())),
        );
        let world_diag = (4.0 * (WORLD_HALF[0] * WORLD_HALF[0] + WORLD_HALF[1] * WORLD_HALF[1]))
            .sqrt();
        let mut sum = 0.0_f64;
        for i in 0..n {
            let pi = world.cells[i].position;
            let mut min_d2 = f32::MAX;
            let mut search_r = GRID_CELL_SIZE;
            while min_d2 == f32::MAX && search_r <= world_diag {
                grid.for_each_in_radius_toroidal(pi, search_r, WORLD_HALF, |j, pj, _| {
                    if i == j {
                        return;
                    }
                    let d = bioscape::min_image_delta(pi, pj, WORLD_HALF);
                    let d2 = d[0] * d[0] + d[1] * d[1];
                    if d2 < min_d2 {
                        min_d2 = d2;
                    }
                });
                search_r *= 2.0;
            }
            if min_d2 < f32::MAX {
                sum += min_d2.sqrt() as f64;
            }
        }
        sum / nf
    } else {
        0.0
    };
    // Sprint 111: shock observability + diversity metrics (Shannon attack
    // entropy, brain w1 frobenius std).
    let (shock_active, shock_hazard_max) = shock_summary(world);
    // Sprint 112: per-population mean climate offset (suma signed offsetů /
    // n). Identical 0.0 když events.empty nebo žádný ClimateShift aktivní.
    let climate_offset_avg: f64 = if n > 0 {
        let sum: f64 = world
            .cells
            .iter()
            .map(|c| world.climate_offset_at([c.position[0], c.position[1]]) as f64)
            .sum();
        sum / nf
    } else {
        0.0
    };
    let lineage_count = lineages.len();
    let behavioral_entropy_attack = attack_entropy(&world.cells);
    let weight_diversity_w1_norm = w1_frobenius_std(&world.cells);
    // Sprint 113: globální FoodCrash multiplikátor (1.0 default, < 1.0 při
    // aktivním FoodCrash). Single per-gen scalar — žádný spatial average.
    let shock_food_factor =
        bioscape::food_density_shock_multiplier(&world.events.events, world.clock.generation);
    let spike_count_avg = if n > 0 { spike_count_sum as f64 / nf } else { 0.0 };
    let spike_complexity_avg = if n > 0 { spike_complexity_sum / nf } else { 0.0 };
    let spike_total_length_avg = if n > 0 { spike_total_length_sum / nf } else { 0.0 };
    // Sprint 128: coop food per-gen events (solved/failed) + mean arrivals per
    // event (přes všechny zaniklé v gen, vč. expired-without-trigger).
    let coop_arrivals_avg = if world.coop_food_events_gen > 0 {
        world.coop_food_arrivals_sum_gen as f64 / world.coop_food_events_gen as f64
    } else {
        0.0
    };
    let bonded_attack_eff = if world.bonded_attacks_gen > 0 && world.solo_attacks_gen > 0 {
        let b_avg = world.bonded_attack_gain_sum_gen / world.bonded_attacks_gen as f64;
        let s_avg = world.solo_attack_gain_sum_gen / world.solo_attacks_gen as f64;
        if s_avg > 0.0 { b_avg / s_avg } else { 0.0 }
    } else {
        0.0
    };
    let swarm_attack_frac = if world.attack_victims_gen > 0 {
        world.swarm_attacks_gen as f64 / world.attack_victims_gen as f64
    } else {
        0.0
    };
    let pack_attack_frac = if world.swarm_attacks_gen > 0 {
        world.pack_attacks_gen as f64 / world.swarm_attacks_gen as f64
    } else {
        0.0
    };
    // Maze navigation metrics — all zero when `world.obstacles` is None,
    // i.e. no `--maze` flag. `maze_in_goal_frac` is the per-tick fraction of
    // the population inside the goal zone, averaged over the generation;
    // `maze_unique_reach_frac` is distinct cells that touched the goal this
    // gen / current pop; `maze_first_reach_total` is the cumulative count of
    // cell_ids that have ever first-touched the goal across all generations.
    let maze_active: u8 = if world.obstacles.is_some() { 1 } else { 0 };
    let maze_in_goal_frac = if maze_active == 1 && n > 0 {
        world.goal_zone_ticks_gen as f64 / (TICKS_PER_GENERATION as f64 * nf)
    } else {
        0.0
    };
    let maze_unique_reach_frac = if maze_active == 1 && n > 0 {
        world.goal_unique_reachers_gen.len() as f64 / nf
    } else {
        0.0
    };
    let maze_first_reach_total = world.goal_first_reach_tick.len() as u64;
    // Sprint 140: per-cell rate stats from Sprint 136/137 genome fields.
    let (lr_avg, lr_std) = mean_std(world.cells.iter().map(|c| c.genome.learning_rate as f64));
    let (decay_avg, decay_std) =
        mean_std(world.cells.iter().map(|c| c.genome.trace_decay_per_sec as f64));
    // Sprint 187: per-cell reproduction threshold + birth donation (r/K
    // niche tracking — two axes that should correlate negatively if pure
    // r and K strategies emerge as distinct attractors).
    let (repro_avg, repro_std) =
        mean_std(world.cells.iter().map(|c| c.genome.reproduce_at_energy as f64));
    let (birth_avg, birth_std) =
        mean_std(world.cells.iter().map(|c| c.genome.birth_energy as f64));
    // Sprint 187: per-cell altruism + cluster bonus. Polymorphism vs
    // convergence in long runs answers the kin-selection question.
    let (altruism_avg, altruism_std) =
        mean_std(world.cells.iter().map(|c| c.genome.altruism_share_frac as f64));
    let (cluster_bonus_avg, cluster_bonus_std) =
        mean_std(world.cells.iter().map(|c| c.genome.cluster_share_bonus as f64));
    // Sprint 187: per-cell aggression traits. attack_gate drift down = hawk
    // pressure, drift up = dove. defense_contribution drift up = altruistic
    // defense emerging (parallel to altruism_share_frac).
    let (attack_gate_avg, attack_gate_std) =
        mean_std(world.cells.iter().map(|c| c.genome.attack_gate as f64));
    let (size_ratio_avg, size_ratio_std) =
        mean_std(world.cells.iter().map(|c| c.genome.predation_size_ratio as f64));
    let (defense_avg, defense_std) =
        mean_std(world.cells.iter().map(|c| c.genome.defense_contribution as f64));
    // Sprint 187: per-cell reward weights (7 kinds). Means only — stds skipped
    // for CSV compactness (variance can be inferred from per-tick events).
    let mut rw_means = [0.0_f64; bioscape::N_REWARD_KINDS];
    if n > 0 {
        for c in world.cells.iter() {
            for k in 0..bioscape::N_REWARD_KINDS {
                rw_means[k] += c.genome.reward_weights[k] as f64;
            }
        }
        for k in 0..bioscape::N_REWARD_KINDS {
            rw_means[k] /= nf;
        }
    }
    // Sprint 140: mean L2 norm across w1 rows of all cells. Synaptic
    // scaling (S138) caps row norms at `W_NORM_CAP`; this metric tracks
    // how close the population sits to the cap on average.
    let w_norm_avg = w1_row_norm_avg(&world.cells);
    // Sprint 142 SNN bridge: end-of-gen snapshot fraction of hidden
    // activations close to tanh saturation (|h| > 0.8). Acts as a
    // passive proxy for "spike events" without the per-tick GPU readback
    // a true STDP path would need; informs whether the population is
    // running in a saturated regime (high frac) or staying in the
    // responsive zone (low frac).
    let neural_spike_frac = saturation_frac(&world.cells);
    // Sprint 149: fraction of population on the Izhikevich neuron model.
    // Activates with `MutationConfig::model_flip_rate > 0` — pre-S149 runs
    // produce 0.0 throughout. Tracking how this fraction evolves answers
    // whether SNN cells carve a niche worth investing in.
    let izhikevich_frac = if n > 0 {
        let count = world
            .cells
            .iter()
            .filter(|c| matches!(c.genome.neuron_model, bioscape::NeuronModel::Izhikevich))
            .count();
        count as f64 / nf
    } else {
        0.0
    };
    // Mean effective rank of the hidden layer's input weights. The
    // environment-pressure decade asks "does evolution use brain capacity";
    // every other column is a behavioural proxy — this is the only one that
    // measures the capacity directly. 1.0 = rank-1 collapse (every hidden
    // neuron the same receptive field), → hidden_n = fully decorrelated.
    let w1_eff_rank = w1_effective_rank_avg(&world.cells);
    // Brain topology size. `hidden_n` (active hidden-neuron count) is now an
    // inherited + mutated genome gene — was hardcoded `BRAIN_HIDDEN_DEFAULT`.
    // These columns show whether selection grows or shrinks the hidden layer.
    let hidden_n_sum: u64 = world
        .cells
        .iter()
        .map(|c| c.genome.brain.hidden_n as u64)
        .sum();
    let hidden_n_avg = hidden_n_sum as f64 / nf;
    let hidden_n_max = world
        .cells
        .iter()
        .map(|c| c.genome.brain.hidden_n)
        .max()
        .unwrap_or(0);
    writeln!(
        w,
        "{},{},{:.2},{:.3},{:.2},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{:.3},{},{},{:.3},{:.3},{:.3},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2},{},{},{},{:.3},{},{:.3},{:.2},{:.3},{:.3},{:.3},{:.3},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2},{:.2},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{:.3},{:.3},{:.3},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.1},{},{},{:.3},{:.3},{:.3},{:.3},{:.4},{:.4},{:.4},{:.3},{:.3},{},{:.4},{:.4},{},{:.6},{:.6},{:.4},{:.4},{:.3},{:.4},{:.4},{:.2},{:.2},{:.2},{:.2},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.4},{:.4},{:.3},{:.4},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2},{},{:.3},{:.3}",
        world.clock.generation,
        n,
        spd_m,
        spd_d,
        vis_m,
        vis_d,
        len_m,
        wid_m,
        hgt_m,
        asp_m,
        asp_d,
        spk_m,
        spk_max,
        world.foods.len(),
        world.density_factor,
        lineages.len(),
        oldest_age,
        ph_emit_m[0],
        ph_emit_m[1],
        ph_emit_m[2],
        ph_emit_d[0],
        ph_emit_d[1],
        ph_emit_d[2],
        ph_burst_m[0],
        ph_burst_m[1],
        ph_burst_m[2],
        abs_x_m,
        abs_y_m,
        edge_f,
        corner_f,
        x_m,
        y_m,
        energy_m,
        world.births_gen,
        world.deaths_gen,
        world.fertile_ticks_gen,
        atk_emit_m,
        world.predation_events_gen,
        recurrent_io_m,
        nn_dist_m,
        density_m,
        density_d,
        dmg_m,
        noise_m,
        // Sprint 67 bond/adhesion diagnostics.
        world.bonds_formed_gen,
        world.bonds_broken_gen,
        mean_bond_count,
        bond_active_frac,
        bond_signal_m,
        adhesion_entropy,
        // Sprint 68 gene means.
        bond_stiff_m,
        bond_damp_m,
        // Sprint 80 cell_state diagnostics.
        state_m,
        state_d,
        altruist_frac,
        // Sprint 83 vision_fov diagnostics.
        fov_m,
        fov_d,
        // Sprint 85 thermal niche.
        temp_m,
        // Sprint 87 thermal_optimum gen.
        topt_m,
        topt_d,
        // Sprint 93 diet diagnostic.
        carnivore_m,
        // Sprint 97 sensor_gain category means + stddev.
        gain_vis_m,
        gain_chem_m,
        gain_def_m,
        gain_vis_d,
        gain_chem_d,
        gain_def_d,
        // Sprint 107: CPPN speciation distance.
        cppn_compat_m,
        // Sprint 111: shock state + diversity metrics.
        // Sprint 112: shock_climate_offset = mean signed temperature offset
        // přes všechny cells z aktivních ClimateShift shocků (0.0 default).
        // Sprint 113: shock_food_factor = global FoodCrash multiplikátor
        // (1.0 default, clamp na FOOD_CRASH_MIN_FACTOR floor).
        shock_active,
        shock_hazard_max,
        climate_offset_avg,
        shock_food_factor,
        lineage_count,
        behavioral_entropy_attack,
        weight_diversity_w1_norm,
        // Sprint 123/125 multi-spike observability.
        spike_count_avg,
        spike_complexity_avg,
        spike_total_length_avg,
        ticks_per_sec,
        // Sprint 128: cooperative food packet events.
        world.coop_food_solved_gen,
        world.coop_food_failed_gen,
        coop_arrivals_avg,
        bonded_attack_eff,
        swarm_attack_frac,
        pack_attack_frac,
        // Vibration / mechanosensory diagnostics. Emit is the population's
        // mean motion-driven amplitude per tick; amp + grad_mag are sampled
        // at each cell's position so they reflect what the brain actually
        // read this tick.
        vib_emit_m,
        vib_amp_m,
        vib_grad_mag_m,
        gain_mech_m,
        gain_mech_d,
        maze_active,
        maze_in_goal_frac,
        maze_unique_reach_frac,
        maze_first_reach_total,
        // Sprint 140
        lr_avg,
        lr_std,
        decay_avg,
        decay_std,
        w_norm_avg,
        // Sprint 142
        neural_spike_frac,
        // Sprint 149
        izhikevich_frac,
        // Sprint 187
        repro_avg,
        repro_std,
        birth_avg,
        birth_std,
        altruism_avg,
        altruism_std,
        cluster_bonus_avg,
        cluster_bonus_std,
        attack_gate_avg,
        attack_gate_std,
        size_ratio_avg,
        size_ratio_std,
        defense_avg,
        defense_std,
        rw_means[0], // EatFood
        rw_means[1], // Novelty
        rw_means[2], // Predation
        rw_means[3], // EscapedAttack
        rw_means[4], // Damage
        rw_means[5], // BondFormed
        rw_means[6], // MateSignalAccepted
        w1_eff_rank,
        hidden_n_avg,
        hidden_n_max,
        rw_means[7], // HazardAvoided
        rw_means[8], // ThreatEscaped
    )
}

/// Sprint 140: shared mean/stddev reducer for the new per-cell genome
/// rate columns. Returns `(0.0, 0.0)` for an empty iterator so the empty-
/// population row stays well-formed.
fn mean_std<I: IntoIterator<Item = f64>>(it: I) -> (f64, f64) {
    let mut sum = 0.0_f64;
    let mut sumsq = 0.0_f64;
    let mut n = 0_u64;
    for v in it {
        sum += v;
        sumsq += v * v;
        n += 1;
    }
    if n == 0 {
        return (0.0, 0.0);
    }
    let nf = n as f64;
    let mean = sum / nf;
    let var = (sumsq / nf - mean * mean).max(0.0);
    (mean, var.sqrt())
}

/// Sprint 142 SNN bridge: fraction of (cell, hidden_neuron) pairs whose
/// final-tick `last_hidden` magnitude crossed the `0.8` saturation
/// threshold. A passive proxy for "spike events" — high values indicate
/// the population is running brains near tanh saturation, which is what a
/// future Izhikevich + STDP path would explicitly model.
fn saturation_frac(cells: &[Cell]) -> f64 {
    if cells.is_empty() {
        return 0.0;
    }
    const THRESHOLD: f32 = 0.8;
    let mut saturated = 0_u64;
    let mut total = 0_u64;
    for cell in cells {
        let h_n = cell.genome.brain.hidden_n as usize;
        for h in 0..h_n {
            if cell.last_hidden[h].abs() > THRESHOLD {
                saturated += 1;
            }
            total += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        saturated as f64 / total as f64
    }
}

/// Sprint 140: mean L2 norm of `w1` rows across the population. Each
/// cell contributes `BRAIN_HIDDEN` row norms; we average over `n_cells ×
/// BRAIN_HIDDEN` samples. With the S138 synaptic scaling cap at
/// `W_NORM_CAP = 8.0`, healthy populations sit at `1–4`; saturation toward
/// the cap signals chronic Hebbian overgrowth.
fn w1_row_norm_avg(cells: &[Cell]) -> f64 {
    if cells.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    let mut count = 0_u64;
    for cell in cells {
        let h_n = cell.genome.brain.hidden_n as usize;
        for row in cell.genome.brain.w1.iter().take(h_n) {
            let sq: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            sum += sq.sqrt();
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

/// Mean participation ratio (effective rank) of `w1` across the population.
/// Per cell `PR = (Σ‖rowᵢ‖²)² / Σᵢⱼ(rowᵢ·rowⱼ)²` over the active rows
/// `[0, hidden_n)` and active columns `[0, BRAIN_INPUTS_SENSORY + hidden_n)`
/// — dead-zone recurrent slots hold CPPN garbage on fresh cells and would
/// poison the dot products. The identity `PR = trace(G)² / ‖G‖_F²` for the
/// Gram matrix `G = W Wᵀ` yields the effective rank from pairwise dot
/// products alone, no SVD. `PR = 1` ⇒ rank-1 collapse (every hidden neuron
/// shares one receptive field); `PR → hidden_n` ⇒ fully decorrelated.
fn w1_effective_rank_avg(cells: &[Cell]) -> f64 {
    if cells.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for cell in cells {
        let h_n = cell.genome.brain.hidden_n as usize;
        if h_n == 0 {
            continue;
        }
        let active = BRAIN_INPUTS_SENSORY + h_n;
        let w1 = &cell.genome.brain.w1;
        let mut trace = 0.0_f64;
        let mut frob_sq = 0.0_f64;
        for i in 0..h_n {
            let row_i = &w1[i];
            // Diagonal: G_ii = ‖rowᵢ‖².
            let g_ii: f64 = row_i[..active]
                .iter()
                .map(|v| (*v as f64) * (*v as f64))
                .sum();
            trace += g_ii;
            frob_sq += g_ii * g_ii;
            // Off-diagonal pairs counted twice (G symmetric).
            for j in (i + 1)..h_n {
                let row_j = &w1[j];
                let g_ij: f64 = (0..active)
                    .map(|k| row_i[k] as f64 * row_j[k] as f64)
                    .sum();
                frob_sq += 2.0 * g_ij * g_ij;
            }
        }
        if frob_sq > 1e-12 {
            sum += (trace * trace) / frob_sq;
        }
    }
    sum / cells.len() as f64
}

/// Sprint 111: aktivní shock summary pro CSV. Vrací `(count, hazard_intensity_max)`,
/// kde max je `intensity * shock_ramp_factor(event, gen)` přes všechny aktivní
/// `HazardPulse` eventy. Bez aktivních eventů nebo žádný HazardPulse → `(0, 0.0)`.
pub fn shock_summary(world: &World) -> (u32, f64) {
    let gen = world.clock.generation;
    let tick = world.clock.tick;
    let mut count: u32 = 0;
    let mut hazard_max: f64 = 0.0;
    for event in world.events.active(gen, tick) {
        count += 1;
        if matches!(event.kind, bioscape::ShockKind::HazardPulse) {
            let factor = bioscape::shock_ramp_factor(event, gen);
            let intensity = (event.intensity * factor) as f64;
            if intensity > hazard_max {
                hazard_max = intensity;
            }
        }
    }
    (count, hazard_max)
}

/// Sprint 111: Shannon entropy histogramu `cell.last_outputs[6]` (attack signal).
/// 8 bins na intervalu [-1, 1], hodnoty mimo se clampují. Empty population → 0.0.
pub fn attack_entropy(cells: &[Cell]) -> f64 {
    if cells.is_empty() {
        return 0.0;
    }
    const BINS: usize = 8;
    let mut hist = [0_u64; BINS];
    for c in cells {
        let v = c.last_outputs[6].clamp(-1.0, 1.0);
        let mut idx = ((v + 1.0) * 0.5 * BINS as f32) as usize;
        if idx >= BINS {
            idx = BINS - 1;
        }
        hist[idx] += 1;
    }
    let total = cells.len() as f64;
    let mut h = 0.0_f64;
    for &count in hist.iter() {
        if count == 0 {
            continue;
        }
        let p = count as f64 / total;
        h -= p * p.log2();
    }
    h
}

/// Sprint 111: zapíše scheduled shock events do sidecar CSV vedle hlavního
/// out_path (`events_seed{seed}.csv`). Caller už ověřil, že `events` není prázdný.
pub fn write_events_sidecar(
    out_path: &Path,
    seed: u64,
    events: &EventCalendar,
) -> std::io::Result<()> {
    let dir = out_path.parent().unwrap_or_else(|| Path::new("."));
    let sidecar = dir.join(format!("events_seed{}.csv", seed));
    let file = std::fs::File::create(&sidecar)?;
    let mut w = BufWriter::new(file);
    writeln!(w, "start_gen,duration_gen,kind,intensity,center_x,center_y,radius")?;
    for e in &events.events {
        let kind = match e.kind {
            bioscape::ShockKind::HazardPulse => "hazard_pulse",
            bioscape::ShockKind::ClimateShift => "climate_shift",
            bioscape::ShockKind::FoodCrash => "food_crash",
        };
        let cx = e
            .center_xy
            .map(|c| format!("{:.2}", c[0]))
            .unwrap_or_default();
        let cy = e
            .center_xy
            .map(|c| format!("{:.2}", c[1]))
            .unwrap_or_default();
        let r = e
            .radius
            .map(|r| format!("{:.2}", r))
            .unwrap_or_default();
        writeln!(
            w,
            "{},{},{},{:.3},{},{},{}",
            e.start_gen, e.duration_gen, kind, e.intensity, cx, cy, r
        )?;
    }
    w.flush()?;
    Ok(())
}

/// Sprint 111: per-cell brain w1 Frobenius norm, pak std přes všechny cells.
/// `std = sqrt(mean((x_i - mean)²))`. Empty population → 0.0.
pub fn w1_frobenius_std(cells: &[Cell]) -> f64 {
    if cells.is_empty() {
        return 0.0;
    }
    let norms: Vec<f64> = cells
        .iter()
        .map(|c| {
            let mut sumsq = 0.0_f64;
            for row in c.genome.brain.w1.iter() {
                for &w in row.iter() {
                    sumsq += (w as f64) * (w as f64);
                }
            }
            sumsq.sqrt()
        })
        .collect();
    let n = norms.len() as f64;
    let mean = norms.iter().sum::<f64>() / n;
    let var = norms.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    var.sqrt()
}

#[cfg(test)]
#[path = "csv_tests.rs"]
mod tests;
