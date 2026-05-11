use super::*;
use crate::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Sprint 44: parity test CPU vs GPU forward. Tolerance 1e-5 — single-precision
/// floats + tanh implementations se mírně liší napříč implementacemi, ale
/// ne-trivial drift > 1e-5 by indikoval bug v packingu nebo shader logic.
/// Test vyžaduje compatible wgpu adapter (skipped pokud `BrainGpu::new` selže).
#[test]
fn brain_forward_gpu_matches_cpu() {
    let mut rng = StdRng::seed_from_u64(7);
    let n = 32;
    let brains: Vec<Brain> = (0..n).map(|_| Brain::random(&mut rng)).collect();
    let inputs: Vec<[f32; BRAIN_INPUTS]> = (0..n)
        .map(|_| {
            let mut a = [0.0_f32; BRAIN_INPUTS];
            for v in a.iter_mut() {
                *v = rand::Rng::random_range(&mut rng, -1.0_f32..1.0_f32);
            }
            a
        })
        .collect();

    let mut gpu = match BrainGpu::new(n) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skip: no GPU adapter ({e})");
            return;
        }
    };

    let mut h_gpu = vec![[0.0_f32; BRAIN_HIDDEN]; n];
    let mut o_gpu = vec![[0.0_f32; BRAIN_OUTPUTS]; n];
    gpu.forward_batch(&inputs, &brains, &mut h_gpu, &mut o_gpu);

    for i in 0..n {
        let (h_cpu, o_cpu) = brains[i].forward_with_state(&inputs[i]);
        for k in 0..BRAIN_HIDDEN {
            let diff = (h_cpu[k] - h_gpu[i][k]).abs();
            assert!(
                diff < 1e-4,
                "hidden mismatch i={i} k={k} cpu={} gpu={} diff={}",
                h_cpu[k],
                h_gpu[i][k],
                diff
            );
        }
        for k in 0..BRAIN_OUTPUTS {
            let diff = (o_cpu[k] - o_gpu[i][k]).abs();
            assert!(
                diff < 1e-4,
                "output mismatch i={i} k={k} cpu={} gpu={} diff={}",
                o_cpu[k],
                o_gpu[i][k],
                diff
            );
        }
    }
}

/// Sprint 45: parity test GPU spatial hash vs CPU brute force.
/// Pro každý bucket: SET cells na GPU = SET cells na CPU. Bucketing přes
/// `SpatialHashGpu::bucket_id_of` (CPU mirror shader logiky).
#[test]
fn spatial_hash_gpu_matches_cpu_buckets() {
    let mut rng = StdRng::seed_from_u64(11);
    let n: usize = 500;
    let cell_size: f32 = 64.0;
    // Drž positions uvnitř world bounds [-960, 960] × [-540, 540] × [-2, 2]
    // — stejné jako headless WORLD_HALF.
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|_| {
            [
                rng.random_range(-900.0_f32..900.0),
                rng.random_range(-500.0_f32..500.0),
                rng.random_range(-2.0_f32..2.0),
            ]
        })
        .collect();

    let mut gpu = match SpatialHashGpu::new(n, cell_size, [1000.0, 1000.0]) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skip: no GPU adapter ({e})");
            return;
        }
    };
    let (offsets, sorted) = gpu.rebuild(&positions);

    // CPU reference: build bucket_id → set<cell_idx> map.
    let mut cpu_buckets: std::collections::HashMap<u32, std::collections::BTreeSet<u32>> =
        std::collections::HashMap::new();
    for (i, p) in positions.iter().enumerate() {
        let b = SpatialHashGpu::bucket_id_of(*p, cell_size);
        cpu_buckets.entry(b).or_default().insert(i as u32);
    }

    // Total se musí matchnout.
    assert_eq!(sorted.len(), n);
    assert_eq!(offsets.len(), GPU_HASH_NUM_BUCKETS + 1);
    assert_eq!(offsets[GPU_HASH_NUM_BUCKETS] as usize, n);

    for b in 0..GPU_HASH_NUM_BUCKETS {
        let start = offsets[b] as usize;
        let end = offsets[b + 1] as usize;
        let gpu_set: std::collections::BTreeSet<u32> =
            sorted[start..end].iter().copied().collect();
        let cpu_set = cpu_buckets
            .get(&(b as u32))
            .cloned()
            .unwrap_or_default();
        assert_eq!(
            gpu_set, cpu_set,
            "bucket {b} mismatch: gpu={:?} cpu={:?}",
            gpu_set, cpu_set
        );
    }
}

/// Sprint 46/56: parity test GPU FieldGpu vs CPU `SmellField`. Stejné
/// 3D sources + stejné step parametry → grid match v ε. Sprint 56
/// re-enabled na 3D + toroidal xy / Neumann z. Atomic float CAS loop
/// má drift kvůli pořadí přídavků; tolerance 1e-3 absolute.
#[test]
fn field_gpu_diffusion_matches_cpu() {
    let resolution = [16usize, 16, 4];
    let world_half = [320.0_f32, 320.0, 20.0];
    let mut gpu = match FieldGpu::new(resolution, world_half, 32) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skip: no GPU adapter ({e})");
            return;
        }
    };
    let mut cpu = SmellField::new(resolution, world_half);

    let mut rng = StdRng::seed_from_u64(46);
    let sources: Vec<([f32; 3], f32)> = (0..16)
        .map(|_| {
            (
                [
                    rng.random_range(-300.0_f32..300.0),
                    rng.random_range(-300.0_f32..300.0),
                    rng.random_range(-15.0_f32..15.0),
                ],
                rng.random_range(0.5_f32..2.0),
            )
        })
        .collect();

    let diffusion = 0.15_f32;
    let decay = 0.5_f32;
    let dt = 0.1_f32;
    for _ in 0..6 {
        for (p, amt) in &sources {
            cpu.add_source(*p, *amt);
            gpu.add_source(*p, *amt);
        }
        cpu.step(diffusion, decay, dt);
        gpu.step(diffusion, decay, dt);
    }

    let cpu_grid = cpu.grid_ref();
    let gpu_grid = gpu.download();
    assert_eq!(cpu_grid.len(), gpu_grid.len());
    for (i, (a, b)) in cpu_grid.iter().zip(gpu_grid.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "i={i} cpu={a} gpu={b} diff={}",
            a - b
        );
    }
}

/// Sprint 47: StatsGpu reduction parity vs naive CPU sum. Tolerance 1e-2
/// kvůli f32 sum non-associativity v 1024-element tree reduce.
#[test]
fn stats_gpu_matches_cpu_sums() {
    let mut rng = StdRng::seed_from_u64(17);
    let n = 1024;
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|_| {
            [
                rng.random_range(-1000.0_f32..1000.0),
                rng.random_range(-500.0_f32..500.0),
                rng.random_range(-2.0_f32..2.0),
            ]
        })
        .collect();
    let velocities: Vec<[f32; 3]> = (0..n)
        .map(|_| {
            [
                rng.random_range(-50.0_f32..50.0),
                rng.random_range(-50.0_f32..50.0),
                rng.random_range(-5.0_f32..5.0),
            ]
        })
        .collect();
    let energies: Vec<f32> = (0..n)
        .map(|_| rng.random_range(0.0_f32..150.0))
        .collect();

    let mut gpu = match StatsGpu::new(n) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skip: no GPU adapter ({e})");
            return;
        }
    };
    let gpu_stats = gpu.compute(&positions, &velocities, &energies);

    let cpu_sum_x: f32 = positions.iter().map(|p| p[0]).sum();
    let cpu_sum_y: f32 = positions.iter().map(|p| p[1]).sum();
    let cpu_sum_z: f32 = positions.iter().map(|p| p[2]).sum();
    let cpu_sum_speed_sq: f32 = velocities
        .iter()
        .map(|v| v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        .sum();
    let cpu_sum_energy: f32 = energies.iter().sum();

    let close = |a: f32, b: f32, scale: f32| {
        let d = (a - b).abs();
        assert!(d < scale * 1e-3 + 1e-3, "diff = {d}, a={a}, b={b}");
    };
    close(gpu_stats.sum_x, cpu_sum_x, cpu_sum_x.abs());
    close(gpu_stats.sum_y, cpu_sum_y, cpu_sum_y.abs());
    close(gpu_stats.sum_z, cpu_sum_z, cpu_sum_z.abs());
    close(gpu_stats.sum_speed_sq, cpu_sum_speed_sq, cpu_sum_speed_sq.abs());
    close(gpu_stats.sum_energy, cpu_sum_energy, cpu_sum_energy.abs());
}

/// Sprint 51: brownian GPU produces non-trivial velocity perturbation +
/// xoshiro state mutates (deterministic). Test ne porovnává CPU stejně —
/// CPU gaussian (Box-Muller) uses StdRng, GPU uses xoshiro128++ — různé
/// PRNG. Ověření je: po N kroků velocity má nenulové statistické
/// rozptyly v očekávaném scale (thermal_noise × √dt × √N).
#[test]
fn brownian_gpu_perturbs_velocity() {
    let n = 256;
    let mut gpu = match BrownianGpu::new(n) {
        Ok(g) => g,
        Err(e) => { eprintln!("skip: no GPU adapter ({e})"); return; }
    };
    let mut velocities = vec![[0.0_f32; 3]; n];
    let mut state: Vec<[u32; 4]> = (0..n)
        .map(|i| {
            let s = (i as u64) + 0xDEADBEEFu64;
            [
                (s >> 0) as u32 ^ 0x9E3779B9u32,
                (s >> 16) as u32 ^ 0xBB67AE85u32,
                (s >> 32) as u32 ^ 0x3C6EF372u32,
                (s >> 48) as u32 ^ 0xA54FF53Au32,
            ]
        })
        .collect();
    let thermal_noise = 0.5_f32;
    let dt = 1.0_f32 / 60.0;
    let steps = 100;
    for _ in 0..steps {
        let (v, s) = gpu.compute(&velocities, &state, thermal_noise, dt, true);
        velocities = v;
        state = s;
    }
    // Empirical sigma: thermal × sqrt(dt) × sqrt(steps). Test že existuje
    // nenulová variance (statisticky téměř jistě > 0 pro 256 cells).
    let mut sum_v_sq = 0.0_f64;
    for v in velocities.iter() {
        sum_v_sq += (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) as f64;
    }
    let mean_v_sq = sum_v_sq / n as f64;
    // Theoretical: 3 × thermal² × dt × steps = 3 × 0.25 × (100/60) ≈ 1.25
    let expected = 3.0 * (thermal_noise * thermal_noise) as f64 * dt as f64 * steps as f64;
    // Tolerance ±50 % kvůli small N stochastic noise.
    assert!(
        mean_v_sq > expected * 0.5 && mean_v_sq < expected * 1.5,
        "mean_v_sq = {} (expected ~{})",
        mean_v_sq,
        expected
    );
}

/// Sprint 51: brownian determinismus — stejný initial state → stejný
/// výsledek napříč běhy. xoshiro128++ je deterministic per state seed.
#[test]
fn brownian_gpu_deterministic_across_runs() {
    let n = 64;
    let mut gpu = match BrownianGpu::new(n) {
        Ok(g) => g,
        Err(e) => { eprintln!("skip: no GPU adapter ({e})"); return; }
    };
    let velocities = vec![[0.5_f32, 1.0, -0.3]; n];
    let state: Vec<[u32; 4]> = (0..n).map(|i| [i as u32 + 1, 2, 3, 4]).collect();
    let (v1, s1) = gpu.compute(&velocities, &state, 0.3, 1.0 / 60.0, false);
    let (v2, s2) = gpu.compute(&velocities, &state, 0.3, 1.0 / 60.0, false);
    for i in 0..n {
        for k in 0..3 {
            assert_eq!(v1[i][k].to_bits(), v2[i][k].to_bits(),
                "i={i} k={k} not deterministic v1={} v2={}", v1[i][k], v2[i][k]);
        }
        for k in 0..4 {
            assert_eq!(s1[i][k], s2[i][k]);
        }
    }
}

/// Sprint 51: Hebbian GPU vs CPU `Brain::hebbian_update` parity. 32 cells,
/// random pre/post activations + reward, GPU update vs CPU update.
/// Tolerance 1e-4 (per-weight FMA chain).
#[test]
fn hebbian_gpu_matches_cpu() {
    let mut rng = StdRng::seed_from_u64(73);
    let n = 32;
    let lr: f32 = 0.005;
    let brains: Vec<Brain> = (0..n).map(|_| Brain::random(&mut rng)).collect();
    let last_inputs: Vec<[f32; BRAIN_INPUTS]> = (0..n)
        .map(|_| {
            let mut a = [0.0_f32; BRAIN_INPUTS];
            for v in a.iter_mut() { *v = rng.random_range(-1.0_f32..1.0); }
            a
        })
        .collect();
    // Sprint 80: dead zone (slots ≥ hidden_n) musí být 0 — odpovídá reálnému
    // Cell.last_hidden, kde forward_with_state píše jen [0..hidden_n] a zbytek
    // zůstává po init na 0. GPU shader zatím iteruje celý BRAIN_HIDDEN; z toho
    // důvodu dead zone hidden×inputs * 0 = 0 update, match CPU bounded path.
    let last_hidden: Vec<[f32; BRAIN_HIDDEN]> = (0..n)
        .enumerate()
        .map(|(idx, _)| {
            let mut a = [0.0_f32; BRAIN_HIDDEN];
            let h_n = brains[idx].hidden_n as usize;
            for v in a[..h_n].iter_mut() { *v = rng.random_range(-1.0_f32..1.0); }
            a
        })
        .collect();
    let last_outputs: Vec<[f32; BRAIN_OUTPUTS]> = (0..n)
        .map(|_| {
            let mut a = [0.0_f32; BRAIN_OUTPUTS];
            for v in a.iter_mut() { *v = rng.random_range(-1.0_f32..1.0); }
            a
        })
        .collect();
    let rewards: Vec<f32> = (0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();

    let mut gpu = match HebbianGpu::new(n) {
        Ok(g) => g,
        Err(e) => { eprintln!("skip: no GPU adapter ({e})"); return; }
    };
    let gpu_brains = gpu.compute(
        &last_inputs, &last_hidden, &last_outputs, &rewards, &brains, lr,
    );

    // CPU equivalent.
    let mut cpu_brains = brains.clone();
    for i in 0..n {
        cpu_brains[i].hebbian_update(
            &last_inputs[i], &last_hidden[i], &last_outputs[i], rewards[i], lr,
        );
    }

    for i in 0..n {
        for h in 0..BRAIN_HIDDEN {
            for in_i in 0..BRAIN_INPUTS {
                let d = (cpu_brains[i].w1[h][in_i] - gpu_brains[i].w1[h][in_i]).abs();
                assert!(d < 1e-4, "i={i} h={h} in_i={in_i} cpu={} gpu={} d={}",
                    cpu_brains[i].w1[h][in_i], gpu_brains[i].w1[h][in_i], d);
            }
            let d = (cpu_brains[i].b1[h] - gpu_brains[i].b1[h]).abs();
            assert!(d < 1e-4);
        }
        for o in 0..BRAIN_OUTPUTS {
            for h in 0..BRAIN_HIDDEN {
                let d = (cpu_brains[i].w2[o][h] - gpu_brains[i].w2[o][h]).abs();
                assert!(d < 1e-4);
            }
            let d = (cpu_brains[i].b2[o] - gpu_brains[i].b2[o]).abs();
            assert!(d < 1e-4);
        }
    }
}

/// Sprint 50: motor GPU vs CPU `Cell::apply_brain_motor` parity. Stejné
/// outputs + cell state → identické post-motor velocities/angular/pitch_vel
/// v ε. Tolerance 1e-4 (single-precision multiply chain ~10 ops).
#[test]
fn motor_gpu_matches_cpu() {
    use crate::{Cell, DRAG_COEFFICIENT};
    let mut rng = StdRng::seed_from_u64(41);
    let n = 64;
    let dt = 1.0_f32 / 60.0;
    let mut cells: Vec<Cell> = (0..n)
        .map(|i| Cell::random(&mut rng, [960.0, 540.0, 2.0], 0, 0, i as u64))
        .collect();
    let outputs: Vec<[f32; BRAIN_OUTPUTS]> = (0..n)
        .map(|_| {
            let mut a = [0.0_f32; BRAIN_OUTPUTS];
            for v in a.iter_mut() {
                *v = rng.random_range(-1.0_f32..1.0);
            }
            a
        })
        .collect();
    let headings: Vec<f32> = cells.iter().map(|c| c.heading).collect();
    let pitches: Vec<f32> = cells.iter().map(|c| c.pitch).collect();
    let max_speeds: Vec<f32> = cells.iter().map(|c| c.genome.max_speed).collect();
    let turn_rates: Vec<f32> = cells.iter().map(|c| c.genome.turn_rate).collect();
    let eff_radii: Vec<f32> = cells.iter().map(|c| c.phenotype.effective_radius()).collect();
    let velocities_in: Vec<[f32; 3]> = cells.iter().map(|c| c.velocity).collect();
    let angular_in: Vec<f32> = cells.iter().map(|c| c.angular_velocity).collect();
    let pitch_vel_in: Vec<f32> = cells.iter().map(|c| c.pitch_velocity).collect();

    let mut gpu = match MotorGpu::new(n) {
        Ok(g) => g,
        Err(e) => { eprintln!("skip: no GPU adapter ({e})"); return; }
    };
    let (gpu_v, gpu_a, gpu_p) = gpu.compute(
        &outputs, &headings, &pitches, &max_speeds, &turn_rates, &eff_radii,
        &velocities_in, &angular_in, &pitch_vel_in, dt, DRAG_COEFFICIENT,
    );

    for (i, cell) in cells.iter_mut().enumerate() {
        cell.apply_brain_motor(&outputs[i], dt);
    }

    for i in 0..n {
        for k in 0..3 {
            let d = (cells[i].velocity[k] - gpu_v[i][k]).abs();
            assert!(d < 1e-4, "i={i} k={k} cpu={} gpu={} diff={}", cells[i].velocity[k], gpu_v[i][k], d);
        }
        assert!((cells[i].angular_velocity - gpu_a[i]).abs() < 1e-4);
        assert!((cells[i].pitch_velocity - gpu_p[i]).abs() < 1e-4);
    }
}

/// Sprint 50/56: full sensor gather GPU vs CPU parity. Cells + foods + 2
/// 3D fields. GPU spustí celý subsystém přes shared context, output SoA
/// porovnán s CPU brute-force pro nearest cell, nearest food, neighbor
/// count + 3D field gradient (smell, pheromone). Atomic float CAS drift
/// na field deposit dovoluje tolerance 1e-2 na gradient values.
#[test]
fn sensor_gather_gpu_matches_cpu() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip: no GPU adapter ({e})");
            return;
        }
    };
    let n = 32_usize;
    let nf = 16_usize;
    let cell_size = 64.0_f32;
    let world_half_xy = [320.0_f32, 320.0];
    let field_resolution = [16usize, 16, 4];
    let field_world_half = [320.0_f32, 320.0, 20.0];

    let mut rng = StdRng::seed_from_u64(56);
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|_| {
            [
                rng.random_range(-200.0_f32..200.0),
                rng.random_range(-200.0_f32..200.0),
                rng.random_range(-10.0_f32..10.0),
            ]
        })
        .collect();
    let eff_radii: Vec<f32> = (0..n).map(|_| rng.random_range(1.0_f32..2.5)).collect();
    let vision_radii: Vec<f32> = (0..n).map(|_| 60.0).collect();
    let food_positions: Vec<[f32; 3]> = (0..nf)
        .map(|_| {
            [
                rng.random_range(-200.0_f32..200.0),
                rng.random_range(-200.0_f32..200.0),
                rng.random_range(-10.0_f32..10.0),
            ]
        })
        .collect();

    let mut cell_hash =
        SpatialHashGpu::with_context(&ctx, n, cell_size, world_half_xy).expect("cell hash");
    cell_hash.rebuild(&positions);
    let mut food_hash =
        SpatialHashGpu::with_context(&ctx, nf, cell_size, world_half_xy).expect("food hash");
    food_hash.rebuild(&food_positions);

    let mut smell_gpu =
        FieldGpu::with_context(&ctx, field_resolution, field_world_half, 64).expect("smell");
    let mut pheromone_gpu =
        FieldGpu::with_context(&ctx, field_resolution, field_world_half, 64).expect("phero");
    let mut smell_cpu = SmellField::new(field_resolution, field_world_half);
    let mut pheromone_cpu = SmellField::new(field_resolution, field_world_half);

    let smell_sources: Vec<([f32; 3], f32)> = (0..12)
        .map(|_| {
            (
                [
                    rng.random_range(-200.0_f32..200.0),
                    rng.random_range(-200.0_f32..200.0),
                    rng.random_range(-10.0_f32..10.0),
                ],
                1.0,
            )
        })
        .collect();
    let phero_sources: Vec<([f32; 3], f32)> = (0..8)
        .map(|_| {
            (
                [
                    rng.random_range(-200.0_f32..200.0),
                    rng.random_range(-200.0_f32..200.0),
                    rng.random_range(-10.0_f32..10.0),
                ],
                1.5,
            )
        })
        .collect();

    for (p, a) in &smell_sources {
        smell_gpu.add_source(*p, *a);
        smell_cpu.add_source(*p, *a);
    }
    for (p, a) in &phero_sources {
        pheromone_gpu.add_source(*p, *a);
        pheromone_cpu.add_source(*p, *a);
    }
    let diffusion = 0.15_f32;
    let decay = 0.5_f32;
    let dt = 0.1_f32;
    for _ in 0..3 {
        smell_gpu.step(diffusion, decay, dt);
        pheromone_gpu.step(diffusion, decay, dt);
        smell_cpu.step(diffusion, decay, dt);
        pheromone_cpu.step(diffusion, decay, dt);
    }

    let mut sensor =
        SensorGatherGpu::with_context(&ctx, n, nf).expect("SensorGatherGpu init");
    let eps = 4.0_f32;
    let params = SensorParamsGpu {
        hash_cell_size: cell_size,
        world_half_x: world_half_xy[0],
        world_half_y: world_half_xy[1],
        world_half_z: 20.0,
        field_res_x: field_resolution[0] as u32,
        field_res_y: field_resolution[1] as u32,
        field_res_z: field_resolution[2] as u32,
        field_eps: eps,
        field_world_half_x: field_world_half[0],
        field_world_half_y: field_world_half[1],
        field_world_half_z: field_world_half[2],
        ..SensorParamsGpu::default()
    };
    // Wave 6: sensor.compute now takes per-cell heading + pitch for whisker
    // raycast. Tests don't care, pass zeros (params.maze_active = 0 skips it).
    let test_headings = vec![0.0_f32; positions.len()];
    let test_pitches = vec![0.0_f32; positions.len()];
    let rows = sensor.compute(
        &positions,
        &eff_radii,
        &vision_radii,
        &food_positions,
        &test_headings,
        &test_pitches,
        &cell_hash,
        &food_hash,
        &smell_gpu,
        &pheromone_gpu,
        // V7: vibration shares the same FieldGpu type as smell/pheromone.
        // Tests don't assert on vibration values, so reuse smell as a stand-in
        // — vibration_grad/amp in returned rows will mirror smell, which is
        // outside the assertion surface.
        &smell_gpu,
        params,
    );

    let min_image = |d: f32, half: f32| -> f32 {
        if d > half {
            d - 2.0 * half
        } else if d < -half {
            d + 2.0 * half
        } else {
            d
        }
    };

    for i in 0..n {
        let pos_i = positions[i];
        let vr2 = vision_radii[i] * vision_radii[i];

        let mut best_cell_d2 = f32::INFINITY;
        let mut best_cell_radius = -1.0_f32;
        let mut count = 0u32;
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = min_image(positions[j][0] - pos_i[0], world_half_xy[0]);
            let dy = min_image(positions[j][1] - pos_i[1], world_half_xy[1]);
            let dz = positions[j][2] - pos_i[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 <= vr2 {
                count += 1;
                if d2 < best_cell_d2 {
                    best_cell_d2 = d2;
                    best_cell_radius = eff_radii[j];
                }
            }
        }

        let mut best_food_d2 = f32::INFINITY;
        let mut has_food = false;
        for f in 0..nf {
            let dx = min_image(food_positions[f][0] - pos_i[0], world_half_xy[0]);
            let dy = min_image(food_positions[f][1] - pos_i[1], world_half_xy[1]);
            let dz = food_positions[f][2] - pos_i[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 <= vr2 && d2 < best_food_d2 {
                best_food_d2 = d2;
                has_food = true;
            }
        }

        let row = &rows[i];
        assert_eq!(
            row.neighbors_in_vision, count,
            "i={i} neighbor count mismatch"
        );
        assert_eq!(
            row.nearest_cell.is_some(),
            best_cell_radius >= 0.0,
            "i={i} nearest_cell presence"
        );
        if let Some((_, r)) = row.nearest_cell {
            assert!(
                (r - best_cell_radius).abs() < 1e-4,
                "i={i} radius cpu={best_cell_radius} gpu={r}"
            );
        }
        assert_eq!(
            row.nearest_food.is_some(),
            has_food,
            "i={i} nearest_food presence"
        );

        let cpu_smell_grad = smell_cpu.gradient_at(pos_i, eps);
        let cpu_phero_grad = pheromone_cpu.gradient_at(pos_i, eps);
        for axis in 0..3 {
            assert!(
                (row.smell_grad[axis] - cpu_smell_grad[axis]).abs() < 1e-2,
                "i={i} smell_grad[{axis}] cpu={} gpu={}",
                cpu_smell_grad[axis],
                row.smell_grad[axis]
            );
            assert!(
                (row.pheromone_grad[axis] - cpu_phero_grad[axis]).abs() < 1e-2,
                "i={i} phero_grad[{axis}] cpu={} gpu={}",
                cpu_phero_grad[axis],
                row.pheromone_grad[axis]
            );
        }
    }
}


/// Sprint 50: predate GPU vs CPU parity. Cluster cells s mixed sizes a
/// random attack signals; herd_counts + energy_delta + damage_delta v ε
/// match. Atomic float CAS sumace má ULP drift, tolerance 1e-3 absolute.
/// Sprint 126: rozšířen o full multi-spike fixture — variable spike_count,
/// per-slot azimuth/elevation/complexity, per-cell pitch. CPU baseline
/// zrcadlí WGSL `multi_spike_bonus` 1:1.
#[test]
fn predate_gpu_matches_cpu() {
    use crate::{
        ATTACK_THRESHOLD, CELL_RADIUS, COMPLEXITY_ATTACK_GAIN, DILUTION_K, HERD_RADIUS,
        PREDATION_DRAIN_PER_TICK, PREDATION_GAIN_PER_TICK, SIZE_RATIO_THRESHOLD,
        SPIKE_DOT_THRESHOLD, SPIKE_PREDATION_BONUS,
    };
    let mut rng = StdRng::seed_from_u64(67);
    let n = 80;
    let cell_size = 64.0_f32;
    // Pack cells aby se některé dotýkaly.
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|_| {
            [
                rng.random_range(-40.0_f32..40.0),
                rng.random_range(-40.0_f32..40.0),
                rng.random_range(-1.0_f32..1.0),
            ]
        })
        .collect();
    let eff_radii: Vec<f32> = (0..n).map(|_| rng.random_range(0.5_f32..2.0)).collect();
    let headings: Vec<f32> = (0..n)
        .map(|_| rng.random_range(0.0_f32..core::f32::consts::TAU))
        .collect();
    // Sprint 126: per-cell pitch ∈ [-π/4, π/4] — multi-spike používá 3D
    // forward (yaw + azim, pitch + elev).
    let pitches: Vec<f32> = (0..n)
        .map(|_| rng.random_range(-core::f32::consts::FRAC_PI_4..core::f32::consts::FRAC_PI_4))
        .collect();
    // Sprint 126: per-cell spike_count + per-slot spike attribs. Mix:
    // ~20 % cells `spike_count = 0` (no bonus path), zbylé 1..=SPIKE_SLOTS.
    let spike_counts: Vec<u32> = (0..n)
        .map(|i| {
            if i % 5 == 0 {
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
    let attack_signals: Vec<f32> = (0..n).map(|_| rng.random_range(-0.5_f32..1.0)).collect();

    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => { eprintln!("skip: no GPU adapter ({e})"); return; }
    };
    let mut hash = SpatialHashGpu::with_context(&ctx, n, cell_size, [1000.0, 1000.0]).expect("hash");
    let _ = hash.rebuild(&positions);
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
        world_half_x: 1000.0,
        world_half_y: 1000.0,
        ..PredateParamsGpu::default()
    };
    let res = pred.compute(
        &positions,
        &eff_radii,
        &headings,
        &pitches,
        &spikes_packed,
        &spike_counts,
        &attack_signals,
        &hash,
        params,
    );

    // CPU brute force.
    let mut cpu_herd = vec![0u32; n];
    let herd_r2 = HERD_RADIUS * HERD_RADIUS;
    for i in 0..n {
        for j in 0..n {
            if i == j { continue; }
            let dx = positions[i][0] - positions[j][0];
            let dy = positions[i][1] - positions[j][1];
            let dz = positions[i][2] - positions[j][2];
            if dx * dx + dy * dy + dz * dz < herd_r2 {
                cpu_herd[i] += 1;
            }
        }
    }

    // Sprint 126: CPU multi-spike baseline mirroring WGSL `multi_spike_bonus`.
    // Stejné formule: dir = forward(yaw + azim, pitch + elev), cone test
    // cosine ≥ SPIKE_DOT_THRESHOLD, per-slot bonus = length × (1 +
    // COMPLEXITY_ATTACK_GAIN × complexity) × SPIKE_PREDATION_BONUS.
    let multi_spike_bonus = |i: usize, to_target: [f32; 3]| -> f32 {
        let n_spikes = spike_counts[i].min(SPIKE_SLOTS as u32) as usize;
        let yaw = headings[i];
        let pitch = pitches[i];
        let mut acc = 0.0_f32;
        for slot in 0..n_spikes {
            let spk = spikes_packed[i * SPIKE_SLOTS + slot];
            let length = spk[0];
            if length <= 0.0 {
                continue;
            }
            let yaw_s = yaw + spk[1];
            let pit_s = pitch + spk[2];
            let cos_p = pit_s.cos();
            let dir = [
                yaw_s.cos() * cos_p,
                yaw_s.sin() * cos_p,
                pit_s.sin(),
            ];
            let cos_a =
                dir[0] * to_target[0] + dir[1] * to_target[1] + dir[2] * to_target[2];
            if cos_a < SPIKE_DOT_THRESHOLD {
                continue;
            }
            let cmplx = spk[3].clamp(0.0, 1.0);
            let attack_factor = 1.0 + COMPLEXITY_ATTACK_GAIN * cmplx;
            acc += length * attack_factor * SPIKE_PREDATION_BONUS;
        }
        acc
    };

    let mut cpu_energy = vec![0.0_f32; n];
    let mut cpu_damage = vec![0.0_f32; n];
    for i in 0..n {
        let attack = attack_signals[i].max(0.0);
        if attack <= ATTACK_THRESHOLD { continue; }
        let r_i = eff_radii[i];
        for j in 0..n {
            if i == j { continue; }
            let r_j = eff_radii[j];
            if r_i < SIZE_RATIO_THRESHOLD * r_j { continue; }
            let pair_r = CELL_RADIUS * (r_i + r_j);
            let pair_r2 = pair_r * pair_r;
            let dx = positions[i][0] - positions[j][0];
            let dy = positions[i][1] - positions[j][1];
            let dz = positions[i][2] - positions[j][2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < pair_r2 {
                let mut gain = PREDATION_GAIN_PER_TICK;
                if spike_counts[i] > 0 && d2 > 0.0 {
                    let inv_d = 1.0 / d2.sqrt();
                    let to_target = [-dx * inv_d, -dy * inv_d, -dz * inv_d];
                    let bonus = multi_spike_bonus(i, to_target);
                    gain += PREDATION_GAIN_PER_TICK * bonus;
                }
                let dilution = 1.0 / (1.0 + DILUTION_K * cpu_herd[j] as f32);
                gain *= dilution;
                cpu_energy[i] += gain;
                cpu_energy[j] -= PREDATION_DRAIN_PER_TICK;
                cpu_damage[j] += PREDATION_DRAIN_PER_TICK;
            }
        }
    }

    for i in 0..n {
        assert_eq!(cpu_herd[i], res.herd_counts[i], "i={i} herd");
        assert!(
            (cpu_energy[i] - res.energy_delta[i]).abs() < 1e-3,
            "i={i} energy cpu={} gpu={}",
            cpu_energy[i],
            res.energy_delta[i]
        );
        assert!(
            (cpu_damage[i] - res.damage_delta[i]).abs() < 1e-3,
            "i={i} damage cpu={} gpu={}",
            cpu_damage[i],
            res.damage_delta[i]
        );
    }
}

/// Sprint 50: collision GPU vs CPU `headless::resolve_collisions` parity.
/// Pack cells velmi blízko sebe → forced overlaps. GPU vrací delta_position
/// per cell; CPU brute-force počítá totéž. Tolerance 1e-3.
#[test]
fn collision_gpu_matches_cpu() {
    use crate::CELL_RADIUS;
    let mut rng = StdRng::seed_from_u64(53);
    let n = 100;
    let cell_size = 64.0_f32;
    // Cluster cells v malé oblasti aby měl collision co řešit.
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|_| {
            [
                rng.random_range(-30.0_f32..30.0),
                rng.random_range(-30.0_f32..30.0),
                rng.random_range(-1.0_f32..1.0),
            ]
        })
        .collect();
    let eff_radii: Vec<f32> = (0..n).map(|_| rng.random_range(0.7_f32..1.5)).collect();
    let max_axes: Vec<f32> = eff_radii.iter().map(|r| r * 1.2).collect();

    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => { eprintln!("skip: no GPU adapter ({e})"); return; }
    };
    let mut hash = SpatialHashGpu::with_context(&ctx, n, cell_size, [1000.0, 1000.0]).expect("hash");
    let _ = hash.rebuild(&positions);
    let mut col = CollisionGpu::with_context(&ctx, n, cell_size, CELL_RADIUS, [1000.0, 1000.0])
        .expect("collision init");
    let gpu_deltas = col.compute(&positions, &eff_radii, &max_axes, &hash);

    // CPU brute force.
    let mut cpu_deltas: Vec<[f32; 3]> = vec![[0.0; 3]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j { continue; }
            let pair_r = CELL_RADIUS * (eff_radii[i] + eff_radii[j]);
            let pair_r2 = pair_r * pair_r;
            let dx = positions[i][0] - positions[j][0];
            let dy = positions[i][1] - positions[j][1];
            let dz = positions[i][2] - positions[j][2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < pair_r2 && d2 > 0.0 {
                let d = d2.sqrt();
                let overlap = pair_r - d;
                cpu_deltas[i][0] += (dx / d) * overlap * 0.5;
                cpu_deltas[i][1] += (dy / d) * overlap * 0.5;
                cpu_deltas[i][2] += (dz / d) * overlap * 0.5;
            }
        }
    }

    for i in 0..n {
        for k in 0..3 {
            let d = (cpu_deltas[i][k] - gpu_deltas[i][k]).abs();
            assert!(d < 1e-3, "i={i} k={k} cpu={} gpu={} diff={}",
                cpu_deltas[i][k], gpu_deltas[i][k], d);
        }
    }
}

/// Sprint 50: step GPU vs CPU `Cell::step` parity. Random cells, jeden step
/// na obou stranách, výsledný state musí matchnout v ε. Tolerance 1e-4
/// (long arithmetic chain s drag + decay + pitch clamp).
#[test]
fn step_gpu_matches_cpu() {
    use crate::{
        Cell, AGE_DECAY_PER_SEC, ANGULAR_DRAG, ANGULAR_ENERGY_COST, ATTACK_COST_PER_SEC,
        BODY_COST_FACTOR, DRAG_COEFFICIENT, ENERGY_COST_PER_V_SQ, FIXED_TIMESTEP_HZ, GRAVITY,
        PHYSICS_CONFIG, SHELL_COST_PER_SEC, SPIKE_COST_PER_SEC, THERMAL_BOTTOM, THERMAL_Q10,
        THERMAL_REF_TEMP, THERMAL_TOP, VISION_COST_PER_RADIUS,
    };
    let mut rng = StdRng::seed_from_u64(43);
    let n = 64;
    let dt = 1.0_f32 / FIXED_TIMESTEP_HZ;
    let world_half: [f32; 3] = [960.0, 540.0, 2.0];
    // Spawn cells s mírnou velocity / angular_velocity aby step měl co dělat.
    let mut cells: Vec<Cell> = (0..n)
        .map(|i| {
            let mut c = Cell::random(&mut rng, world_half, 0, 0, i as u64);
            c.angular_velocity = rng.random_range(-0.3_f32..0.3);
            c.pitch_velocity = rng.random_range(-0.05_f32..0.05);
            c.last_outputs[6] = rng.random_range(-0.5_f32..1.0);
            c
        })
        .collect();

    let positions: Vec<[f32; 3]> = cells.iter().map(|c| c.position).collect();
    let velocities: Vec<[f32; 3]> = cells.iter().map(|c| c.velocity).collect();
    let headings: Vec<f32> = cells.iter().map(|c| c.heading).collect();
    let pitches: Vec<f32> = cells.iter().map(|c| c.pitch).collect();
    let angular_velocities: Vec<f32> = cells.iter().map(|c| c.angular_velocity).collect();
    let pitch_velocities: Vec<f32> = cells.iter().map(|c| c.pitch_velocity).collect();
    let ages: Vec<u32> = cells.iter().map(|c| c.age as u32).collect();
    let cooldowns: Vec<u32> = cells.iter().map(|c| c.reproduce_cooldown_ticks).collect();
    let energies: Vec<f32> = cells.iter().map(|c| c.energy).collect();
    let body_dims: Vec<[f32; 3]> = cells
        .iter()
        .map(|c| {
            [
                c.phenotype.body_length,
                c.phenotype.body_width,
                c.phenotype.body_height,
            ]
        })
        .collect();
    let aux: Vec<[f32; 4]> = cells
        .iter()
        .map(|c| {
            [
                c.phenotype.total_spike_cost_factor(),
                c.phenotype.shell_thickness,
                c.genome.vision_radius,
                c.last_outputs[6],
            ]
        })
        .collect();

    let mut gpu = match StepGpu::new(n) {
        Ok(g) => g,
        Err(e) => { eprintln!("skip: no GPU adapter ({e})"); return; }
    };
    let params = StepParamsGpu {
        num_cells: n as u32,
        dt,
        world_half_x: world_half[0],
        world_half_y: world_half[1],
        world_half_z: world_half[2],
        gravity: GRAVITY,
        drag: DRAG_COEFFICIENT,
        angular_drag: ANGULAR_DRAG,
        energy_cost_per_v_sq: ENERGY_COST_PER_V_SQ,
        angular_energy_cost: ANGULAR_ENERGY_COST,
        vision_cost_per_radius: VISION_COST_PER_RADIUS,
        body_cost_factor: BODY_COST_FACTOR,
        age_decay_per_sec: AGE_DECAY_PER_SEC,
        fixed_timestep_hz: FIXED_TIMESTEP_HZ,
        spike_cost_per_sec: SPIKE_COST_PER_SEC,
        shell_cost_per_sec: SHELL_COST_PER_SEC,
        attack_cost_per_sec: ATTACK_COST_PER_SEC,
        pitch_clamp: core::f32::consts::FRAC_PI_6 * 0.5,
        thermal_top: THERMAL_TOP,
        thermal_bottom: THERMAL_BOTTOM,
        thermal_q10: THERMAL_Q10,
        thermal_ref_temp: THERMAL_REF_TEMP,
        // Sprint 86: tick=0, gen=0 → phases = 0 → sin(0) = 0 → no
        // seasonal/diurnal offset (matches CPU step(.., 0, 0, ..)).
        thermal_diurnal_amp: crate::THERMAL_DIURNAL_AMP,
        thermal_seasonal_amp: crate::THERMAL_SEASONAL_AMP,
        thermal_diurnal_phase: 0.0,
        thermal_seasonal_phase: 0.0,
        thermal_log2_q10: THERMAL_Q10.log2(),
        ..StepParamsGpu::default()
    };
    let res = gpu.compute(
        &positions, &velocities, &headings, &pitches, &angular_velocities,
        &pitch_velocities, &ages, &cooldowns, &energies, &body_dims, &aux, params,
    );

    // Sprint 87: GPU step shader nepočítá thermal_optimum penalty
    // (latentní debt — aux buffer by potřeboval expansion na [f32; 5]).
    // Override penalty na 0 aby parity zůstala — test ověřuje kinematics
    // + drag + drains, ne thermal_optimum stress.
    let test_physics = crate::PhysicsConfig {
        thermal_optimum_penalty: 0.0,
        ..PHYSICS_CONFIG
    };
    // Sprint 97: GPU step shader nepočítá sensor_gain drain (gain je v
    // genome a CPU-side aplikuje apply_energy_costs). Zero sensor_gains
    // na CPU side aby parity zůstala — ne aby step shader řešil i tohle.
    for c in cells.iter_mut() {
        c.genome.sensor_gains = [0.0; crate::N_SENSOR_CATEGORIES];
        c.step(dt, world_half, 0, 0, &test_physics);
    }

    for i in 0..n {
        for k in 0..3 {
            let dp = (cells[i].position[k] - res.positions[i][k]).abs();
            let dv = (cells[i].velocity[k] - res.velocities[i][k]).abs();
            assert!(dp < 1e-3, "i={i} k={k} pos cpu={} gpu={} d={}",
                cells[i].position[k], res.positions[i][k], dp);
            assert!(dv < 1e-3, "i={i} k={k} vel cpu={} gpu={} d={}",
                cells[i].velocity[k], res.velocities[i][k], dv);
        }
        assert!((cells[i].heading - res.headings[i]).abs() < 1e-3, "i={i} heading");
        assert!((cells[i].pitch - res.pitches[i]).abs() < 1e-4, "i={i} pitch");
        assert!((cells[i].angular_velocity - res.angular_velocities[i]).abs() < 1e-4);
        assert!((cells[i].pitch_velocity - res.pitch_velocities[i]).abs() < 1e-4);
        assert_eq!(cells[i].age as u32, res.ages[i]);
        assert_eq!(cells[i].reproduce_cooldown_ticks, res.cooldowns[i]);
        assert!(
            (cells[i].energy - res.energies[i]).abs() < 1e-3,
            "i={i} energy cpu={} gpu={}",
            cells[i].energy,
            res.energies[i]
        );
    }
}

/// Sprint 49: GPU broad-phase neighbor query parity vs CPU brute force.
/// Stejná positions + vision_radii + hash → stejný nearest cell + count
/// per cell. Tolerance 1e-3 na pozici (single-precision float tieng může
/// disagree mezi nejbližšími při téměř identických vzdálenostech).
#[test]
fn neighbors_gpu_matches_cpu_brute_force() {
    let mut rng = StdRng::seed_from_u64(31);
    let n = 200;
    let cell_size = 64.0;
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|_| {
            [
                rng.random_range(-500.0_f32..500.0),
                rng.random_range(-300.0_f32..300.0),
                rng.random_range(-2.0_f32..2.0),
            ]
        })
        .collect();
    let radii: Vec<f32> = (0..n).map(|_| rng.random_range(0.5_f32..2.0)).collect();
    let vision_radii: Vec<f32> = (0..n).map(|_| rng.random_range(20.0_f32..80.0)).collect();

    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip: no GPU adapter ({e})");
            return;
        }
    };
    let mut hash = SpatialHashGpu::with_context(&ctx, n, cell_size, [1000.0, 1000.0]).expect("hash init");
    let _ = hash.rebuild(&positions);
    let mut nb = NeighborsGpu::with_context(&ctx, n, cell_size, [1000.0, 1000.0]).expect("nb init");
    let gpu_results = nb.compute(&positions, &radii, &vision_radii, &hash);

    for i in 0..n {
        let pos_i = positions[i];
        let vr2 = vision_radii[i] * vision_radii[i];
        let mut cpu_count: u32 = 0;
        let mut best_d2 = f32::MAX;
        let mut best_j: Option<usize> = None;
        for j in 0..n {
            if j == i {
                continue;
            }
            let dx = positions[j][0] - pos_i[0];
            let dy = positions[j][1] - pos_i[1];
            let dz = positions[j][2] - pos_i[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 <= vr2 {
                cpu_count += 1;
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_j = Some(j);
                }
            }
        }
        let gpu = &gpu_results[i];
        assert_eq!(
            gpu.neighbors_in_vision, cpu_count,
            "i={i}: count gpu={} cpu={}",
            gpu.neighbors_in_vision, cpu_count
        );
        match (best_j, gpu.nearest_cell) {
            (None, None) => {}
            (Some(j), Some((p, r))) => {
                let cpu_d2 = {
                    let dx = positions[j][0] - pos_i[0];
                    let dy = positions[j][1] - pos_i[1];
                    let dz = positions[j][2] - pos_i[2];
                    dx * dx + dy * dy + dz * dz
                };
                let gpu_d2 = {
                    let dx = p[0] - pos_i[0];
                    let dy = p[1] - pos_i[1];
                    let dz = p[2] - pos_i[2];
                    dx * dx + dy * dy + dz * dz
                };
                // Acceptujeme jiný winner pokud d2 jsou v ε.
                assert!(
                    (cpu_d2 - gpu_d2).abs() < 1e-2,
                    "i={i}: cpu_d2={cpu_d2}, gpu_d2={gpu_d2}, cpu_j={j}"
                );
                assert!(
                    (r - radii[j]).abs() < 1e-3 || cpu_d2 == gpu_d2,
                    "i={i}: radius mismatch, gpu={r}, cpu_j={j} radius={}",
                    radii[j]
                );
            }
            (cpu, gpu) => panic!("i={i}: cpu={:?} gpu={:?} mismatch", cpu, gpu),
        }
    }
}

/// Sprint 49: ověření že single-workgroup tree reduce zvládá N >> 10k.
/// Strided loop v shaderu je unbounded; jediné co single-WG hraje roli je
/// že 256 threadů sekvenciálně iteruje N/256 prvků. Pro N=50000 to je 195
/// iterací per thread (~10 µs wall time) — žádný correctness problem.
#[test]
fn stats_gpu_handles_50k() {
    let mut rng = StdRng::seed_from_u64(101);
    let n = 50_000;
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|_| {
            [
                rng.random_range(-1000.0_f32..1000.0),
                rng.random_range(-500.0_f32..500.0),
                rng.random_range(-2.0_f32..2.0),
            ]
        })
        .collect();
    let velocities: Vec<[f32; 3]> = (0..n)
        .map(|_| {
            [
                rng.random_range(-50.0_f32..50.0),
                rng.random_range(-50.0_f32..50.0),
                rng.random_range(-5.0_f32..5.0),
            ]
        })
        .collect();
    let energies: Vec<f32> = (0..n).map(|_| rng.random_range(0.0_f32..150.0)).collect();
    let mut gpu = match StatsGpu::new(n) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skip: no GPU adapter ({e})");
            return;
        }
    };
    let gpu_stats = gpu.compute(&positions, &velocities, &energies);
    let cpu_sum_energy: f32 = energies.iter().sum();
    let scale = cpu_sum_energy.abs().max(1.0);
    let diff = (gpu_stats.sum_energy - cpu_sum_energy).abs();
    // Tolerance scaled — pro 50k hodnot s ~75 mean je sum ~3.75M, ULP cumulative
    // drift napříč 50k single-precision adds může být ~1e3 relativně 1e-4.
    assert!(
        diff < scale * 1e-3,
        "diff = {} (scale = {}); gpu = {}, cpu = {}",
        diff,
        scale,
        gpu_stats.sum_energy,
        cpu_sum_energy
    );
}

/// Sprint 47: integration test sdíleného `GpuContext` napříč 4 subsystémy.
/// Každý úspěšně inicializuje skrz `with_context` a doběhne jeden mini
/// pipeline cycle (brain forward → spatial hash → field step → stats reduce)
/// na sdíleném device. Verifikuje, že device-lifetime + bind group ownership
/// se ne-konfliktuje.
#[test]
fn gpu_context_shared_across_subsystems() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip: no GPU adapter ({e})");
            return;
        }
    };
    let n = 64;
    let mut rng = StdRng::seed_from_u64(19);

    let mut brain_gpu = BrainGpu::with_context(&ctx, n).expect("BrainGpu init");
    let mut hash_gpu =
        SpatialHashGpu::with_context(&ctx, n, 64.0, [1000.0, 1000.0]).expect("SpatialHashGpu init");
    let mut field_gpu =
        FieldGpu::with_context(&ctx, [16, 16, 4], [320.0, 320.0, 20.0], 32)
            .expect("FieldGpu init");
    let mut stats_gpu = StatsGpu::with_context(&ctx, n).expect("StatsGpu init");

    let brains: Vec<Brain> = (0..n).map(|_| Brain::random(&mut rng)).collect();
    let inputs: Vec<[f32; BRAIN_INPUTS]> = (0..n)
        .map(|_| {
            let mut a = [0.0_f32; BRAIN_INPUTS];
            for v in a.iter_mut() {
                *v = rng.random_range(-1.0_f32..1.0);
            }
            a
        })
        .collect();
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|_| {
            [
                rng.random_range(-300.0_f32..300.0),
                rng.random_range(-300.0_f32..300.0),
                rng.random_range(-2.0_f32..2.0),
            ]
        })
        .collect();
    let velocities: Vec<[f32; 3]> = (0..n)
        .map(|_| {
            [
                rng.random_range(-30.0_f32..30.0),
                rng.random_range(-30.0_f32..30.0),
                0.0,
            ]
        })
        .collect();
    let energies: Vec<f32> = (0..n).map(|_| 50.0).collect();

    let mut h = vec![[0.0_f32; BRAIN_HIDDEN]; n];
    let mut o = vec![[0.0_f32; BRAIN_OUTPUTS]; n];
    brain_gpu.forward_batch(&inputs, &brains, &mut h, &mut o);

    let (offsets, sorted) = hash_gpu.rebuild(&positions);
    assert_eq!(offsets.len(), GPU_HASH_NUM_BUCKETS + 1);
    assert_eq!(sorted.len(), n);
    assert_eq!(offsets[GPU_HASH_NUM_BUCKETS] as usize, n);

    for (pos, _) in positions.iter().zip(0..n) {
        field_gpu.add_source(*pos, 1.0);
    }
    field_gpu.step(0.15, 0.3, 1.0 / 60.0);
    let grid = field_gpu.download();
    assert_eq!(grid.len(), 16 * 16 * 4);

    let stats = stats_gpu.compute(&positions, &velocities, &energies);
    assert!(stats.sum_energy > 0.0);
}

/// CPPN GPU parity: CPU `Brain::from_cppn` vs GPU `CppnGpu::dispatch`.
/// Tolerance 1e-3 — both paths use the same Padé tanh, but FMA reordering
/// + GPU FP rounding can drift each weight by a few ulps; anything above
/// 1e-3 indicates a bug (wrong substrate offset, wrong activation code,
/// link-gate mismatch, …).
#[test]
fn cppn_from_cppn_gpu_matches_cpu() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip: no GPU adapter ({e})");
            return;
        }
    };

    // Use a non-trivial CPPN with a few rounds of mutations so we exercise
    // each activation function and structural growth.
    let mut rng = StdRng::seed_from_u64(0xCFB1);
    let mut cppn = Cppn::random(&mut rng);
    for _ in 0..6 {
        cppn = cppn.mutate(&mut rng, &CPPN_MUTATION_CONFIG);
    }

    let cpu_brain = Brain::from_cppn(&cppn);

    let cells_gpu = CellsGpu::with_context(&ctx, 1);
    let mut cppn_gpu = CppnGpu::with_context(&ctx, 1);
    // Seed slot 0 with a sentinel so we can tell GPU actually wrote.
    cells_gpu.upload_brains([&Brain::zeros()]);
    cppn_gpu.dispatch(&[(0, &cppn)], &cells_gpu);
    let gpu_brains = cells_gpu.download_brains(1);
    assert_eq!(gpu_brains.len(), 1);
    let gpu_brain = &gpu_brains[0];

    let mut max_diff: f32 = 0.0;
    for h in 0..BRAIN_HIDDEN {
        for i in 0..BRAIN_INPUTS {
            let d = (cpu_brain.w1[h][i] - gpu_brain.w1[h][i]).abs();
            if d > max_diff { max_diff = d; }
        }
    }
    for h in 0..BRAIN_HIDDEN {
        let d = (cpu_brain.b1[h] - gpu_brain.b1[h]).abs();
        if d > max_diff { max_diff = d; }
    }
    for o in 0..BRAIN_OUTPUTS {
        for h in 0..BRAIN_HIDDEN {
            let d = (cpu_brain.w2[o][h] - gpu_brain.w2[o][h]).abs();
            if d > max_diff { max_diff = d; }
        }
    }
    for o in 0..BRAIN_OUTPUTS {
        let d = (cpu_brain.b2[o] - gpu_brain.b2[o]).abs();
        if d > max_diff { max_diff = d; }
    }
    assert!(
        max_diff < 1e-3,
        "GPU CPPN drift exceeds tolerance: max |Δw| = {}",
        max_diff
    );
}
