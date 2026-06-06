use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use rustc_hash::FxHashSet;

use super::components::{EpochEnded, GenerationEnded, StatsRoot, StatsText};
use super::resources::{Clock, FoodDensityFactor, SimWorld};

/// Wall-clock cadence for the stats overlay refresh. The per-cell reduction
/// below — and especially `sync_vibration_from_gpu`, a full grid GPU readback
/// behind a `Wait` barrier — is far too heavy to run every frame for a debug
/// HUD. 10 Hz is plenty for numbers a human reads.
const STATS_REFRESH_INTERVAL: f32 = 0.1;

pub(super) fn log_clock_events(
    mut generation_ended: MessageReader<GenerationEnded>,
    mut epoch_ended: MessageReader<EpochEnded>,
) {
    for ev in generation_ended.read() {
        info!("generation {} ended", ev.generation);
    }
    for ev in epoch_ended.read() {
        info!("epoch {} ended", ev.epoch);
    }
}

pub(super) fn update_stats_overlay(
    clock: Res<Clock>,
    time: Res<Time<Virtual>>,
    real_time: Res<Time<Real>>,
    density: Res<FoodDensityFactor>,
    diagnostics: Res<DiagnosticsStore>,
    mut sim_world: ResMut<SimWorld>,
    text: Single<&mut Text, With<StatsText>>,
    stats_root: Single<&Node, With<StatsRoot>>,
    mut lineages_scratch: Local<FxHashSet<u64>>,
    mut refresh_accum: Local<f32>,
) {
    // Hidden overlay → nothing reads the reduction or the vibration shadow;
    // skip the whole system, GPU readback included.
    if matches!(stats_root.display, Display::None) {
        return;
    }
    *refresh_accum += real_time.delta_secs();
    if *refresh_accum < STATS_REFRESH_INTERVAL {
        return;
    }
    *refresh_accum = 0.0;

    // Pull the GPU vibration field into the CPU shadow here — this overlay is
    // its only consumer, so the readback rides the throttle instead of firing
    // every tick.
    sim_world.0.sync_vibration_from_gpu();

    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);
    let speed = if time.is_paused() {
        "paused".to_string()
    } else {
        format!("{}×", time.relative_speed())
    };

    let mut count = 0usize;
    let mut spd_sum = 0.0_f64;
    let mut spd_sumsq = 0.0_f64;
    let mut vis_sum = 0.0_f64;
    let mut vis_sumsq = 0.0_f64;
    let mut trn_sum = 0.0_f64;
    let mut len_sum = 0.0_f64;
    let mut wid_sum = 0.0_f64;
    let mut asp_sum = 0.0_f64;
    let mut asp_sumsq = 0.0_f64;
    let mut spk_sum = 0.0_f64;
    let mut spk_max = 0.0_f64;
    let mut e_sum = 0.0_f64;
    let mut vib_emit_sum = 0.0_f64;
    let mut vib_amp_sum = 0.0_f64;
    let mut vib_grad_sum = 0.0_f64;
    let mut vib_samples = 0usize;
    lineages_scratch.clear();
    let lineages = &mut *lineages_scratch;
    let mut oldest_age: u64 = 0;
    let current_gen = clock.0.generation;
    let world = &sim_world.0;
    let vibration_field = &world.vibration;
    // Stride-sample vibration (9 trilinear lookups per cell otherwise);
    // overlay only displays averages, so a fraction of the population suffices.
    const VIB_SAMPLE_STRIDE: usize = 16;
    for (i, cell) in world.cells.iter().enumerate() {
        count += 1;
        let s = cell.genome.max_speed as f64;
        let v = cell.genome.vision_radius as f64;
        let t = cell.genome.turn_rate as f64;
        let l = cell.phenotype.body_length as f64;
        let w = cell.phenotype.body_width as f64;
        let aspect = if w > 1e-6 { l / w } else { 0.0 };
        let spk = cell.phenotype.primary_spike_length() as f64;
        let e = cell.energy as f64;
        spd_sum += s;
        spd_sumsq += s * s;
        vis_sum += v;
        vis_sumsq += v * v;
        trn_sum += t;
        len_sum += l;
        wid_sum += w;
        asp_sum += aspect;
        asp_sumsq += aspect * aspect;
        spk_sum += spk;
        if spk > spk_max {
            spk_max = spk;
        }
        e_sum += e;
        vib_emit_sum += bioscape::vibration_emit_for_cell(cell) as f64;
        if i % VIB_SAMPLE_STRIDE == 0 {
            let pos = cell.position;
            vib_amp_sum += vibration_field.sample(pos) as f64;
            let g = vibration_field.gradient_at(pos, bioscape::VIBRATION_SAMPLE_EPSILON);
            vib_grad_sum += ((g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) as f64).sqrt();
            vib_samples += 1;
        }
        lineages.insert(cell.lineage_id);
        let age = current_gen.saturating_sub(cell.lineage_birth_gen);
        if age > oldest_age {
            oldest_age = age;
        }
    }
    let food_count = world.foods.len();
    let lineage_count = lineages.len();

    let (
        spd_avg,
        spd_dev,
        vis_avg,
        vis_dev,
        trn_avg,
        len_avg,
        wid_avg,
        asp_avg,
        asp_dev,
        spk_avg,
        e_avg,
    ) = if count == 0 {
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    } else {
        let n = count as f64;
        let spd_m = spd_sum / n;
        let vis_m = vis_sum / n;
        let asp_m = asp_sum / n;
        (
            spd_m,
            ((spd_sumsq / n) - spd_m * spd_m).max(0.0).sqrt(),
            vis_m,
            ((vis_sumsq / n) - vis_m * vis_m).max(0.0).sqrt(),
            trn_sum / n,
            len_sum / n,
            wid_sum / n,
            asp_m,
            ((asp_sumsq / n) - asp_m * asp_m).max(0.0).sqrt(),
            spk_sum / n,
            e_sum / n,
        )
    };
    let (vib_emit_avg, vib_amp_avg, vib_grad_avg) = if count == 0 {
        (0.0, 0.0, 0.0)
    } else {
        let n = count as f64;
        let vs = vib_samples.max(1) as f64;
        (vib_emit_sum / n, vib_amp_sum / vs, vib_grad_sum / vs)
    };

    // Effective compute throughput. `flops_per_tick` is a static estimate
    // from current pop (see `bioscape::sim::estimate_flops_per_tick`); the
    // effective tick rate equals `FIXED_TIMESTEP_HZ × relative_speed` when
    // the sim is keeping up, and 0 when paused.
    let flops_per_tick = bioscape::sim::estimate_flops_per_tick(count as u64);
    let effective_tps = if time.is_paused() {
        0.0
    } else {
        bioscape::FIXED_TIMESTEP_HZ as f64 * time.relative_speed() as f64
    };
    let flops_per_sec = flops_per_tick as f64 * effective_tps;
    let gflops = flops_per_sec / 1e9;

    let mut text = text.into_inner();
    text.0 = format!(
        "tick     {}\ngen      {}\nepoch    {}\nspeed    {}\ncells    {}\nfood     {}\ndensity  {:.2}\nfps      {:.0}\ngflops   {:.2}\nspd_avg  {:.1}\nspd_dev  {:.2}\nvis_avg  {:.1}\nvis_dev  {:.2}\ntrn_avg  {:.2}\nlen_avg  {:.2}\nwid_avg  {:.2}\nasp_avg  {:.2}\nasp_dev  {:.2}\nspk_avg  {:.2}\nspk_max  {:.2}\ne_avg    {:.1}\nlineages {}\noldest   {}\nvib_emit {:.3}\nvib_amp  {:.4}\nvib_grad {:.5}",
        clock.0.tick,
        clock.0.generation,
        clock.0.epoch,
        speed,
        count,
        food_count,
        density.0,
        fps,
        gflops,
        spd_avg,
        spd_dev,
        vis_avg,
        vis_dev,
        trn_avg,
        len_avg,
        wid_avg,
        asp_avg,
        asp_dev,
        spk_avg,
        spk_max,
        e_avg,
        lineage_count,
        oldest_age,
        vib_emit_avg,
        vib_amp_avg,
        vib_grad_avg,
    );
}
