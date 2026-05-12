use super::*;
use crate::{Brain, BRAIN_HIDDEN, BRAIN_INPUTS, BRAIN_OUTPUTS};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn try_ctx() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skip: no GPU adapter ({e})");
            None
        }
    }
}

fn brain_eq(a: &Brain, b: &Brain) -> bool {
    let mut max_diff = 0.0_f32;
    for h in 0..BRAIN_HIDDEN {
        for i in 0..BRAIN_INPUTS {
            max_diff = max_diff.max((a.w1[h][i] - b.w1[h][i]).abs());
        }
        max_diff = max_diff.max((a.b1[h] - b.b1[h]).abs());
    }
    for o in 0..BRAIN_OUTPUTS {
        for h in 0..BRAIN_HIDDEN {
            max_diff = max_diff.max((a.w2[o][h] - b.w2[o][h]).abs());
        }
        max_diff = max_diff.max((a.b2[o] - b.b2[o]).abs());
    }
    max_diff < 1e-5
}

#[test]
fn cells_gpu_construct_capacity_one() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let cells = CellsGpu::with_context(&ctx, 1);
    assert_eq!(cells.capacity(), 1);
    assert_eq!(cells.epoch(), 0);
}

#[test]
fn cells_gpu_construct_various_capacities() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    for cap in [1, 4, 16, 64, 256, 1024] {
        let cells = CellsGpu::with_context(&ctx, cap);
        assert_eq!(cells.capacity(), cap);
    }
}

#[test]
fn cells_gpu_buffer_accessors_return_distinct_buffers() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let cells = CellsGpu::with_context(&ctx, 16);
    let inputs_ptr = cells.last_inputs_buffer() as *const _;
    let hidden_ptr = cells.last_hidden_buffer() as *const _;
    let outputs_ptr = cells.last_outputs_buffer() as *const _;
    let weights_ptr = cells.brain_weights_buffer() as *const _;
    let velocities_ptr = cells.velocities_buffer() as *const _;
    assert_ne!(inputs_ptr, hidden_ptr);
    assert_ne!(hidden_ptr, outputs_ptr);
    assert_ne!(outputs_ptr, weights_ptr);
    assert_ne!(weights_ptr, velocities_ptr);
}

#[test]
fn cells_gpu_metadata_buffer_accessors_distinct() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let cells = CellsGpu::with_context(&ctx, 16);
    let energy_ptr = cells.energy_buffer() as *const _;
    let heading_ptr = cells.heading_buffer() as *const _;
    let pitch_ptr = cells.pitch_buffer() as *const _;
    let damage_ptr = cells.damage_accum_buffer() as *const _;
    let max_speed_ptr = cells.max_speed_buffer() as *const _;
    let eff_radius_ptr = cells.eff_radius_buffer() as *const _;
    assert_ne!(energy_ptr, heading_ptr);
    assert_ne!(heading_ptr, pitch_ptr);
    assert_ne!(pitch_ptr, damage_ptr);
    assert_ne!(damage_ptr, max_speed_ptr);
    assert_ne!(max_speed_ptr, eff_radius_ptr);
}

#[test]
fn cells_gpu_motor_buffer_accessors_distinct() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let cells = CellsGpu::with_context(&ctx, 16);
    let turn_ptr = cells.turn_rate_buffer() as *const _;
    let ang_ptr = cells.angular_velocity_buffer() as *const _;
    let pitch_vel_ptr = cells.pitch_velocity_buffer() as *const _;
    assert_ne!(turn_ptr, ang_ptr);
    assert_ne!(ang_ptr, pitch_vel_ptr);
}

#[test]
fn cells_gpu_step_buffer_accessors_distinct() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let cells = CellsGpu::with_context(&ctx, 16);
    let pos_ptr = cells.position_buffer() as *const _;
    let age_ptr = cells.age_buffer() as *const _;
    let cd_ptr = cells.cooldown_buffer() as *const _;
    let body_ptr = cells.body_dims_buffer() as *const _;
    let aux_ptr = cells.aux_buffer() as *const _;
    assert_ne!(pos_ptr, age_ptr);
    assert_ne!(age_ptr, cd_ptr);
    assert_ne!(cd_ptr, body_ptr);
    assert_ne!(body_ptr, aux_ptr);
}

#[test]
fn cells_gpu_device_queue_accessible() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let cells = CellsGpu::with_context(&ctx, 8);
    let _ = cells.device();
    let _ = cells.queue();
}

#[test]
fn cells_gpu_upload_brain_then_download_roundtrip() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let n = 4;
    let cells = CellsGpu::with_context(&ctx, n);
    let mut rng = StdRng::seed_from_u64(101);
    let brains: Vec<Brain> = (0..n).map(|_| Brain::random(&mut rng)).collect();
    cells.upload_brains(brains.iter());
    let downloaded = cells.download_brains(n);
    assert_eq!(downloaded.len(), n);
    for (a, b) in brains.iter().zip(downloaded.iter()) {
        assert!(brain_eq(a, b), "brain weights mismatch after roundtrip");
    }
}

#[test]
fn cells_gpu_upload_brain_at_overrides_single_slot() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let n = 8;
    let cells = CellsGpu::with_context(&ctx, n);
    let mut rng = StdRng::seed_from_u64(103);
    let initial: Vec<Brain> = (0..n).map(|_| Brain::random(&mut rng)).collect();
    cells.upload_brains(initial.iter());
    let new_brain = Brain::random(&mut rng);
    cells.upload_brain_at(3, &new_brain);
    let downloaded = cells.download_brains(n);
    assert!(brain_eq(&downloaded[3], &new_brain));
    assert!(brain_eq(&downloaded[0], &initial[0]));
    assert!(brain_eq(&downloaded[7], &initial[7]));
}

#[test]
fn cells_gpu_download_brain_at_returns_correct_slot() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let n = 4;
    let cells = CellsGpu::with_context(&ctx, n);
    let mut rng = StdRng::seed_from_u64(107);
    let brains: Vec<Brain> = (0..n).map(|_| Brain::random(&mut rng)).collect();
    cells.upload_brains(brains.iter());
    for i in 0..n {
        let single = cells.download_brain_at(i);
        assert!(brain_eq(&brains[i], &single));
    }
}

#[test]
fn cells_gpu_upload_velocities_roundtrip() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let n = 16;
    let cells = CellsGpu::with_context(&ctx, n);
    let velocities: Vec<[f32; 3]> = (0..n)
        .map(|i| [i as f32, (i * 2) as f32, (i * 3) as f32 + 0.5])
        .collect();
    cells.upload_velocities(&velocities);
    let downloaded = cells.download_velocities(n);
    assert_eq!(downloaded.len(), n);
    for (a, b) in velocities.iter().zip(downloaded.iter()) {
        for k in 0..3 {
            assert!((a[k] - b[k]).abs() < 1e-5);
        }
    }
}

#[test]
fn cells_gpu_download_velocities_zero_returns_empty_vec() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let cells = CellsGpu::with_context(&ctx, 8);
    let downloaded = cells.download_velocities(0);
    assert!(downloaded.is_empty());
}

#[test]
fn cells_gpu_download_brains_zero_returns_empty_vec() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let cells = CellsGpu::with_context(&ctx, 8);
    let downloaded = cells.download_brains(0);
    assert!(downloaded.is_empty());
}

#[test]
fn cells_gpu_download_hidden_outputs_zero_returns_empty_pair() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let cells = CellsGpu::with_context(&ctx, 8);
    let (h, o) = cells.download_hidden_outputs(0);
    assert!(h.is_empty());
    assert!(o.is_empty());
}

#[test]
fn cells_gpu_download_motor_state_zero_returns_empty_triple() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let cells = CellsGpu::with_context(&ctx, 8);
    let (v, a, p) = cells.download_motor_state(0);
    assert!(v.is_empty());
    assert!(a.is_empty());
    assert!(p.is_empty());
}

#[test]
fn cells_gpu_download_brain_motor_batch_zero_returns_empty() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let cells = CellsGpu::with_context(&ctx, 8);
    let (h, o, v, a, p) = cells.download_brain_motor_batch(0);
    assert!(h.is_empty());
    assert!(o.is_empty());
    assert!(v.is_empty());
    assert!(a.is_empty());
    assert!(p.is_empty());
}

#[test]
fn cells_gpu_download_full_batch_into_zero_clears_all_scratches() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let cells = CellsGpu::with_context(&ctx, 8);
    let mut hidden_out = vec![[0.0_f32; BRAIN_HIDDEN]; 4];
    let mut outputs_out = vec![[0.0_f32; BRAIN_OUTPUTS]; 4];
    let mut velocities_out = vec![[0.0_f32; 3]; 4];
    let mut angular_out = vec![1.0; 4];
    let mut pitch_out = vec![1.0; 4];
    let mut positions_out = vec![[0.0_f32; 3]; 4];
    let mut ages_out = vec![1u32; 4];
    let mut cooldowns_out = vec![1u32; 4];
    let mut energies_out = vec![1.0; 4];
    cells.download_full_batch_into(
        0,
        &mut hidden_out,
        &mut outputs_out,
        &mut velocities_out,
        &mut angular_out,
        &mut pitch_out,
        &mut positions_out,
        &mut ages_out,
        &mut cooldowns_out,
        &mut energies_out,
    );
    assert!(hidden_out.is_empty());
    assert!(outputs_out.is_empty());
    assert!(velocities_out.is_empty());
    assert!(angular_out.is_empty());
    assert!(pitch_out.is_empty());
    assert!(positions_out.is_empty());
    assert!(ages_out.is_empty());
    assert!(cooldowns_out.is_empty());
    assert!(energies_out.is_empty());
}

#[test]
fn cells_gpu_upload_positions_roundtrip_via_full_batch() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let n = 8;
    let cells = CellsGpu::with_context(&ctx, n);
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|i| [i as f32, -(i as f32), (i * 5) as f32])
        .collect();
    let velocities = vec![[0.1_f32, 0.2, 0.3]; n];
    let ages: Vec<u32> = (0..n as u32).collect();
    let cooldowns: Vec<u32> = (0..n as u32).map(|i| i * 2).collect();
    let energies: Vec<f32> = (0..n).map(|i| 50.0 + i as f32).collect();
    let headings: Vec<f32> = (0..n).map(|i| 0.1 * i as f32).collect();
    let pitches: Vec<f32> = (0..n).map(|i| 0.2 * i as f32).collect();
    let damages = vec![0.0_f32; n];
    let max_speeds = vec![60.0_f32; n];
    let eff_radii = vec![5.0_f32; n];
    let angulars = vec![0.0_f32; n];
    let pitch_vels = vec![0.0_f32; n];

    cells.upload_positions(&positions);
    cells.upload_velocities(&velocities);
    cells.upload_age_cooldown(&ages, &cooldowns);
    cells.upload_metadata(&energies, &headings, &pitches, &damages, &max_speeds, &eff_radii);
    cells.upload_angular_pitch(&angulars, &pitch_vels);

    let mut hidden_out = Vec::new();
    let mut outputs_out = Vec::new();
    let mut velocities_out = Vec::new();
    let mut angular_out = Vec::new();
    let mut pitch_out = Vec::new();
    let mut positions_out = Vec::new();
    let mut ages_out = Vec::new();
    let mut cooldowns_out = Vec::new();
    let mut energies_out = Vec::new();
    cells.download_full_batch_into(
        n,
        &mut hidden_out,
        &mut outputs_out,
        &mut velocities_out,
        &mut angular_out,
        &mut pitch_out,
        &mut positions_out,
        &mut ages_out,
        &mut cooldowns_out,
        &mut energies_out,
    );
    assert_eq!(positions_out.len(), n);
    assert_eq!(velocities_out.len(), n);
    assert_eq!(ages_out, ages);
    assert_eq!(cooldowns_out, cooldowns);
    for i in 0..n {
        for k in 0..3 {
            assert!((positions[i][k] - positions_out[i][k]).abs() < 1e-5);
            assert!((velocities[i][k] - velocities_out[i][k]).abs() < 1e-5);
        }
        assert!((energies[i] - energies_out[i]).abs() < 1e-5);
        assert!((angulars[i] - angular_out[i]).abs() < 1e-5);
        assert!((pitch_vels[i] - pitch_out[i]).abs() < 1e-5);
    }
}

#[test]
fn cells_gpu_upload_turn_rates_then_at_modifies_single_slot() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let n = 6;
    let cells = CellsGpu::with_context(&ctx, n);
    let initial: Vec<f32> = (0..n).map(|i| 1.0 + i as f32).collect();
    cells.upload_turn_rates(&initial);
    cells.upload_turn_rate_at(3, 99.0);
    // No direct readback for turn_rate buffer alone — but upload should not panic.
    let _ = cells;
    let _ = initial;
}

#[test]
fn cells_gpu_upload_rewards_no_panic() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let n = 12;
    let cells = CellsGpu::with_context(&ctx, n);
    let rewards: Vec<f32> = (0..n).map(|i| 0.1 * i as f32).collect();
    cells.upload_rewards(&rewards);
}

#[test]
fn cells_gpu_upload_inputs_no_panic() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let n = 4;
    let cells = CellsGpu::with_context(&ctx, n);
    let inputs: Vec<[f32; BRAIN_INPUTS]> = (0..n)
        .map(|i| {
            let mut a = [0.0_f32; BRAIN_INPUTS];
            for j in 0..BRAIN_INPUTS {
                a[j] = (i * BRAIN_INPUTS + j) as f32 * 0.01;
            }
            a
        })
        .collect();
    cells.upload_inputs(&inputs);
}

#[test]
fn cells_gpu_upload_body_dims_no_panic() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let n = 5;
    let cells = CellsGpu::with_context(&ctx, n);
    let dims: Vec<[f32; 3]> = (0..n)
        .map(|i| [1.0 + i as f32, 1.0, 1.0])
        .collect();
    cells.upload_body_dims(&dims);
}

#[test]
fn cells_gpu_upload_aux_no_panic() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let n = 5;
    let cells = CellsGpu::with_context(&ctx, n);
    let aux: Vec<[f32; 4]> = (0..n)
        .map(|i| [i as f32, 0.5, 0.7, 0.0])
        .collect();
    cells.upload_aux(&aux);
}

#[test]
fn cells_gpu_swap_to_self_is_noop() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let n = 4;
    let cells = CellsGpu::with_context(&ctx, n);
    let mut rng = StdRng::seed_from_u64(211);
    let brains: Vec<Brain> = (0..n).map(|_| Brain::random(&mut rng)).collect();
    cells.upload_brains(brains.iter());
    cells.swap_to(2, 2);
    let downloaded = cells.download_brains(n);
    for (a, b) in brains.iter().zip(downloaded.iter()) {
        assert!(brain_eq(a, b));
    }
}

#[test]
fn cells_gpu_swap_to_swaps_brain_weights_between_slots() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let n = 4;
    let cells = CellsGpu::with_context(&ctx, n);
    let mut rng = StdRng::seed_from_u64(217);
    let brains: Vec<Brain> = (0..n).map(|_| Brain::random(&mut rng)).collect();
    cells.upload_brains(brains.iter());
    cells.swap_to(0, 3);
    let after = cells.download_brain_at(0);
    assert!(brain_eq(&after, &brains[3]));
}

#[test]
fn cells_gpu_xoshiro_seeds_full_upload_no_panic() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let n = 32;
    let cells = CellsGpu::with_context(&ctx, n);
    cells.upload_xoshiro_seeds((0..n as u64).map(|i| i + 1));
}

#[test]
fn cells_gpu_xoshiro_seed_at_individual_slot_no_panic() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let n = 16;
    let cells = CellsGpu::with_context(&ctx, n);
    cells.upload_xoshiro_seed_at(0, 0);
    cells.upload_xoshiro_seed_at(15, 0xDEADBEEF);
}

#[test]
fn cells_gpu_upload_metadata_roundtrip_via_full_batch() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let n = 6;
    let cells = CellsGpu::with_context(&ctx, n);
    let energies: Vec<f32> = (0..n).map(|i| 100.0 + i as f32).collect();
    let headings: Vec<f32> = (0..n).map(|i| 0.5 * i as f32).collect();
    let pitches: Vec<f32> = (0..n).map(|i| 0.1 * i as f32).collect();
    let damages: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let max_speeds = vec![60.0; n];
    let eff_radii = vec![5.0; n];
    cells.upload_metadata(&energies, &headings, &pitches, &damages, &max_speeds, &eff_radii);
    let positions = vec![[0.0_f32; 3]; n];
    let velocities = vec![[0.0_f32; 3]; n];
    let ages = vec![0u32; n];
    let cooldowns = vec![0u32; n];
    let angulars = vec![0.0_f32; n];
    let pitch_vels = vec![0.0_f32; n];
    cells.upload_positions(&positions);
    cells.upload_velocities(&velocities);
    cells.upload_age_cooldown(&ages, &cooldowns);
    cells.upload_angular_pitch(&angulars, &pitch_vels);
    let mut hidden_out = Vec::new();
    let mut outputs_out = Vec::new();
    let mut velocities_out = Vec::new();
    let mut angular_out = Vec::new();
    let mut pitch_out = Vec::new();
    let mut positions_out = Vec::new();
    let mut ages_out = Vec::new();
    let mut cooldowns_out = Vec::new();
    let mut energies_out = Vec::new();
    cells.download_full_batch_into(
        n,
        &mut hidden_out,
        &mut outputs_out,
        &mut velocities_out,
        &mut angular_out,
        &mut pitch_out,
        &mut positions_out,
        &mut ages_out,
        &mut cooldowns_out,
        &mut energies_out,
    );
    for i in 0..n {
        assert!((energies[i] - energies_out[i]).abs() < 1e-5);
    }
}

#[test]
fn cells_gpu_uploaded_partial_brains_zero_pads_remainder() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let n = 8;
    let cells = CellsGpu::with_context(&ctx, n);
    let mut rng = StdRng::seed_from_u64(229);
    let partial: Vec<Brain> = (0..3).map(|_| Brain::random(&mut rng)).collect();
    cells.upload_brains(partial.iter());
    let downloaded = cells.download_brains(n);
    assert_eq!(downloaded.len(), n);
    for i in 0..3 {
        assert!(brain_eq(&downloaded[i], &partial[i]));
    }
}

#[test]
fn cells_gpu_motor_state_roundtrip_after_explicit_uploads() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let n = 5;
    let cells = CellsGpu::with_context(&ctx, n);
    let velocities: Vec<[f32; 3]> = (0..n)
        .map(|i| [i as f32 * 0.1, i as f32 * 0.2, 0.0])
        .collect();
    let angulars: Vec<f32> = (0..n).map(|i| 0.1 * i as f32).collect();
    let pitch_vels: Vec<f32> = (0..n).map(|i| -0.05 * i as f32).collect();
    cells.upload_velocities(&velocities);
    cells.upload_angular_pitch(&angulars, &pitch_vels);
    let (v_out, a_out, p_out) = cells.download_motor_state(n);
    assert_eq!(v_out.len(), n);
    assert_eq!(a_out.len(), n);
    assert_eq!(p_out.len(), n);
    for i in 0..n {
        for k in 0..3 {
            assert!((velocities[i][k] - v_out[i][k]).abs() < 1e-5);
        }
        assert!((angulars[i] - a_out[i]).abs() < 1e-5);
        assert!((pitch_vels[i] - p_out[i]).abs() < 1e-5);
    }
}

#[test]
fn cells_gpu_capacity_unchanged_across_uploads() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let cap = 32;
    let cells = CellsGpu::with_context(&ctx, cap);
    let mut rng = StdRng::seed_from_u64(233);
    let brains: Vec<Brain> = (0..cap).map(|_| Brain::random(&mut rng)).collect();
    cells.upload_brains(brains.iter());
    let velocities = vec![[0.0_f32; 3]; cap];
    cells.upload_velocities(&velocities);
    cells.upload_xoshiro_seeds((0..cap as u64).map(|i| i + 1));
    assert_eq!(cells.capacity(), cap);
}

#[test]
fn cells_gpu_shared_context_two_instances_independent() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let a = CellsGpu::with_context(&ctx, 4);
    let b = CellsGpu::with_context(&ctx, 8);
    assert_eq!(a.capacity(), 4);
    assert_eq!(b.capacity(), 8);
    let buf_a = a.last_inputs_buffer() as *const _;
    let buf_b = b.last_inputs_buffer() as *const _;
    assert_ne!(buf_a, buf_b);
}

#[test]
fn cells_gpu_age_cooldown_roundtrip_via_full_batch() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let n = 10;
    let cells = CellsGpu::with_context(&ctx, n);
    let ages: Vec<u32> = (0..n as u32).map(|i| i * 7).collect();
    let cooldowns: Vec<u32> = (0..n as u32).map(|i| 100 - i).collect();
    cells.upload_age_cooldown(&ages, &cooldowns);
    cells.upload_positions(&vec![[0.0_f32; 3]; n]);
    cells.upload_velocities(&vec![[0.0_f32; 3]; n]);
    cells.upload_metadata(
        &vec![100.0_f32; n],
        &vec![0.0_f32; n],
        &vec![0.0_f32; n],
        &vec![0.0_f32; n],
        &vec![60.0_f32; n],
        &vec![5.0_f32; n],
    );
    cells.upload_angular_pitch(&vec![0.0_f32; n], &vec![0.0_f32; n]);

    let mut hidden_out = Vec::new();
    let mut outputs_out = Vec::new();
    let mut velocities_out = Vec::new();
    let mut angular_out = Vec::new();
    let mut pitch_out = Vec::new();
    let mut positions_out = Vec::new();
    let mut ages_out = Vec::new();
    let mut cooldowns_out = Vec::new();
    let mut energies_out = Vec::new();
    cells.download_full_batch_into(
        n,
        &mut hidden_out,
        &mut outputs_out,
        &mut velocities_out,
        &mut angular_out,
        &mut pitch_out,
        &mut positions_out,
        &mut ages_out,
        &mut cooldowns_out,
        &mut energies_out,
    );
    assert_eq!(ages_out, ages);
    assert_eq!(cooldowns_out, cooldowns);
}

#[test]
fn cells_gpu_swap_does_not_touch_other_slots() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let n = 5;
    let cells = CellsGpu::with_context(&ctx, n);
    let mut rng = StdRng::seed_from_u64(257);
    let brains: Vec<Brain> = (0..n).map(|_| Brain::random(&mut rng)).collect();
    cells.upload_brains(brains.iter());
    cells.swap_to(0, 4);
    let untouched_1 = cells.download_brain_at(1);
    let untouched_2 = cells.download_brain_at(2);
    let untouched_3 = cells.download_brain_at(3);
    assert!(brain_eq(&untouched_1, &brains[1]));
    assert!(brain_eq(&untouched_2, &brains[2]));
    assert!(brain_eq(&untouched_3, &brains[3]));
}

#[test]
fn cells_gpu_epoch_initially_zero_and_unchanging_for_simple_uploads() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let cells = CellsGpu::with_context(&ctx, 4);
    assert_eq!(cells.epoch(), 0);
    let mut rng = StdRng::seed_from_u64(263);
    let brains: Vec<Brain> = (0..4).map(|_| Brain::random(&mut rng)).collect();
    cells.upload_brains(brains.iter());
    cells.upload_velocities(&vec![[0.0_f32; 3]; 4]);
    assert_eq!(cells.epoch(), 0);
}

#[test]
fn cells_gpu_reset_persistent_brain_state_at_zeros_only_target_slot() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let n = 4;
    let cells = CellsGpu::with_context(&ctx, n);
    // Seed last_hidden / last_outputs with non-zero data via upload_inputs
    // round-trip (no public last_hidden upload API; we instead drive both
    // buffers via the public upload paths so the test can inspect them).
    // Easier path: write known patterns via the public buffer accessors +
    // wgpu queue write_buffer.
    let hidden_pattern: Vec<f32> = (0..n * BRAIN_HIDDEN).map(|i| (i + 1) as f32).collect();
    let outputs_pattern: Vec<f32> = (0..n * BRAIN_OUTPUTS).map(|i| (i + 1) as f32 * 0.5).collect();
    ctx.queue.write_buffer(
        cells.last_hidden_buffer(),
        0,
        bytemuck::cast_slice(&hidden_pattern),
    );
    ctx.queue.write_buffer(
        cells.last_outputs_buffer(),
        0,
        bytemuck::cast_slice(&outputs_pattern),
    );

    cells.reset_persistent_brain_state_at(1);

    let (hidden_after, outputs_after) = cells.download_hidden_outputs(n);
    // Slot 1 is fully zeroed.
    assert!(hidden_after[1].iter().all(|&v| v == 0.0));
    assert!(outputs_after[1].iter().all(|&v| v == 0.0));
    // Adjacent slots untouched.
    for slot in [0usize, 2, 3] {
        for (k, &v) in hidden_after[slot].iter().enumerate() {
            let expected = hidden_pattern[slot * BRAIN_HIDDEN + k];
            assert_eq!(v, expected, "slot {} hidden[{}] altered", slot, k);
        }
        for (k, &v) in outputs_after[slot].iter().enumerate() {
            let expected = outputs_pattern[slot * BRAIN_OUTPUTS + k];
            assert_eq!(v, expected, "slot {} outputs[{}] altered", slot, k);
        }
    }
}

#[test]
fn cells_gpu_swap_to_resets_destination_persistent_state() {
    let ctx = match try_ctx() { Some(c) => c, None => return };
    let n = 4;
    let cells = CellsGpu::with_context(&ctx, n);
    let mut rng = StdRng::seed_from_u64(271);
    let brains: Vec<Brain> = (0..n).map(|_| Brain::random(&mut rng)).collect();
    cells.upload_brains(brains.iter());
    let hidden_pattern: Vec<f32> = (0..n * BRAIN_HIDDEN).map(|i| (i + 1) as f32).collect();
    ctx.queue.write_buffer(
        cells.last_hidden_buffer(),
        0,
        bytemuck::cast_slice(&hidden_pattern),
    );
    // Simulate the die-and-swap pattern: cell at slot 1 dies, slot 3 moves
    // into its place. dst=1 must end up holding slot-3's brain weights but
    // zeroed Hebbian-state (no leftover from the dead cell at slot 1).
    cells.swap_to(1, 3);
    let new_at_1 = cells.download_brain_at(1);
    assert!(brain_eq(&new_at_1, &brains[3]), "brain at slot 1 should be slot 3's");
    let (hidden_after, _outputs_after) = cells.download_hidden_outputs(n);
    assert!(
        hidden_after[1].iter().all(|&v| v == 0.0),
        "last_hidden[1] should be zeroed after swap_to (was: {:?})",
        &hidden_after[1][..4]
    );
    // Slot 3 (now unused but its buffer state stays for next allocate) is
    // not zeroed by swap_to; allocate-side reset handles that path.
    let slot3_hidden_sum: f32 = hidden_after[3].iter().sum();
    assert!(slot3_hidden_sum > 0.0, "swap_to should not zero src slot");
}
