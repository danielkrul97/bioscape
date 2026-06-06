use super::*;

use bioscape::test_helpers::base_cell;
use bioscape::{
    EventCalendar, ShockEvent, ShockKind, BRAIN_HIDDEN, BRAIN_INPUTS, MATING_RADIUS, WORLD_MAP_SEED,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::io::Cursor;
use std::path::PathBuf;

fn unique_temp_dir(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("bioscape_csv_test_{}_{}_{}", label, pid, nanos));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn fresh_world(seed: u64) -> World {
    let mut rng = StdRng::seed_from_u64(seed);
    World::new(
        &mut rng,
        WORLD_MAP_SEED,
        MATING_RADIUS,
        20,
        100,
        EventCalendar::default(),
    )
}

#[test]
fn write_stats_writes_a_line_with_generation_prefix() {
    let world = fresh_world(7);
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        write_stats(&mut cursor, &world, 60.0).expect("write_stats");
    }
    let output = String::from_utf8(buf).expect("utf8");
    assert!(output.ends_with('\n'));
    let first_field = output.split(',').next().expect("at least one column");
    assert_eq!(first_field.parse::<u64>().unwrap(), world.clock.generation);
}

#[test]
fn write_stats_writes_zero_padded_row_for_empty_population() {
    let mut world = fresh_world(7);
    world.cells.clear();
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        write_stats(&mut cursor, &world, 0.0).expect("write_stats empty");
    }
    let output = String::from_utf8(buf).unwrap();
    assert!(output.starts_with(&format!("{},0,0,0,0,0,", world.clock.generation)));
}

#[test]
fn shock_summary_zero_with_no_events() {
    let world = fresh_world(11);
    let (count, hazard) = shock_summary(&world);
    assert_eq!(count, 0);
    assert_eq!(hazard, 0.0);
}

#[test]
fn shock_summary_ignores_inactive_events() {
    let mut world = fresh_world(11);
    world.events.events.push(ShockEvent {
        kind: ShockKind::HazardPulse,
        start_gen: 100,
        duration_gen: 5,
        ramp_gens: 0,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    });
    world.clock.generation = 0;
    let (count, hazard) = shock_summary(&world);
    assert_eq!(count, 0);
    assert_eq!(hazard, 0.0);
}

#[test]
fn shock_summary_counts_active_hazard_pulse_with_intensity() {
    let mut world = fresh_world(11);
    world.events.events.push(ShockEvent {
        kind: ShockKind::HazardPulse,
        start_gen: 0,
        duration_gen: 10,
        ramp_gens: 0,
        intensity: 0.5,
        center_xy: None,
        radius: None,
    });
    world.clock.generation = 5;
    let (count, hazard) = shock_summary(&world);
    assert!(count >= 1);
    assert!(hazard > 0.0);
}

#[test]
fn shock_summary_climate_shift_has_zero_hazard_max() {
    let mut world = fresh_world(11);
    world.events.events.push(ShockEvent {
        kind: ShockKind::ClimateShift,
        start_gen: 0,
        duration_gen: 10,
        ramp_gens: 0,
        intensity: 1.0,
        center_xy: None,
        radius: None,
    });
    world.clock.generation = 5;
    let (count, hazard) = shock_summary(&world);
    assert_eq!(count, 1);
    assert_eq!(hazard, 0.0);
}

#[test]
fn attack_entropy_empty_population_is_zero() {
    let cells: Vec<bioscape::Cell> = Vec::new();
    assert_eq!(attack_entropy(&cells), 0.0);
}

#[test]
fn attack_entropy_single_cell_is_zero() {
    let cell = base_cell();
    let entropy = attack_entropy(std::slice::from_ref(&cell));
    assert_eq!(entropy, 0.0);
}

#[test]
fn attack_entropy_uniform_eight_bins_reaches_max_entropy() {
    let mut cells = Vec::new();
    let bin_centers = [
        -0.875_f32, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875,
    ];
    for &v in &bin_centers {
        let mut c = base_cell();
        c.last_outputs[6] = v;
        cells.push(c);
    }
    let entropy = attack_entropy(&cells);
    assert!((entropy - 3.0).abs() < 1e-9);
}

#[test]
fn attack_entropy_all_same_value_is_zero() {
    let mut cells = Vec::new();
    for _ in 0..32 {
        let mut c = base_cell();
        c.last_outputs[6] = 0.5;
        cells.push(c);
    }
    let entropy = attack_entropy(&cells);
    assert_eq!(entropy, 0.0);
}

#[test]
fn attack_entropy_clamps_out_of_range_values() {
    let mut cells = Vec::new();
    for _ in 0..8 {
        let mut c = base_cell();
        c.last_outputs[6] = 5.0;
        cells.push(c);
    }
    let entropy = attack_entropy(&cells);
    assert_eq!(entropy, 0.0);
}

#[test]
fn w1_frobenius_std_empty_is_zero() {
    let cells: Vec<bioscape::Cell> = Vec::new();
    assert_eq!(w1_frobenius_std(&cells), 0.0);
}

#[test]
fn w1_frobenius_std_identical_brains_is_zero() {
    let cells = vec![base_cell(), base_cell(), base_cell()];
    assert_eq!(w1_frobenius_std(&cells), 0.0);
}

#[test]
fn w1_frobenius_std_diverse_brains_is_positive() {
    let mut cells = Vec::new();
    for k in 0..4 {
        let mut c = base_cell();
        c.genome.brain.w1[0][0] = k as f32;
        cells.push(c);
    }
    let std = w1_frobenius_std(&cells);
    assert!(std > 0.0);
}

#[test]
fn w1_frobenius_std_scales_correctly_for_known_values() {
    let mut a = base_cell();
    let mut b = base_cell();
    for r in 0..BRAIN_HIDDEN {
        for c in 0..BRAIN_INPUTS {
            a.genome.brain.w1[r][c] = 0.0;
            b.genome.brain.w1[r][c] = 0.0;
        }
    }
    a.genome.brain.w1[0][0] = 1.0;
    b.genome.brain.w1[0][0] = 3.0;
    let cells = vec![a, b];
    let std = w1_frobenius_std(&cells);
    assert!((std - 1.0).abs() < 1e-9);
}

#[test]
fn normalized_shannon_uniform_is_one() {
    let hist = [3_u64; 8];
    assert!((normalized_shannon(&hist) - 1.0).abs() < 1e-9);
}

#[test]
fn normalized_shannon_single_bin_is_zero() {
    let hist = [0_u64, 5, 0, 0];
    assert_eq!(normalized_shannon(&hist), 0.0);
}

#[test]
fn normalized_shannon_empty_is_zero() {
    let hist = [0_u64; 8];
    assert_eq!(normalized_shannon(&hist), 0.0);
}

#[test]
fn behavioral_entropy_empty_is_zero() {
    let cells: Vec<bioscape::Cell> = Vec::new();
    assert_eq!(behavioral_entropy(&cells), 0.0);
}

#[test]
fn behavioral_entropy_monoculture_is_zero() {
    let mut cells = Vec::new();
    for _ in 0..32 {
        let mut c = base_cell();
        c.genome.carnivore_score = 0.5;
        c.genome.vision_fov = 1.0;
        cells.push(c);
    }
    assert_eq!(behavioral_entropy(&cells), 0.0);
}

#[test]
fn behavioral_entropy_spread_is_positive() {
    let mut cells = Vec::new();
    for k in 0..4 {
        let mut c = base_cell();
        c.genome.carnivore_score = (k as f32 + 0.5) / 4.0;
        cells.push(c);
    }
    assert!(behavioral_entropy(&cells) > 0.0);
}

#[test]
fn empty_and_populated_rows_have_same_column_count() {
    let world = fresh_world(7);
    let mut full = Vec::new();
    write_stats(&mut Cursor::new(&mut full), &world, 0.0).unwrap();
    let mut empty_world = fresh_world(7);
    empty_world.cells.clear();
    let mut empty = Vec::new();
    write_stats(&mut Cursor::new(&mut empty), &empty_world, 0.0).unwrap();
    let n_full = String::from_utf8(full)
        .unwrap()
        .trim_end()
        .split(',')
        .count();
    let n_empty = String::from_utf8(empty)
        .unwrap()
        .trim_end()
        .split(',')
        .count();
    assert_eq!(n_full, n_empty);
}

#[test]
fn write_events_sidecar_emits_csv_with_one_row_per_event() {
    let dir = unique_temp_dir("sidecar");
    let out_path = dir.join("run_seed7.csv");
    let mut events = EventCalendar::default();
    events.events.push(ShockEvent {
        kind: ShockKind::HazardPulse,
        start_gen: 5,
        duration_gen: 3,
        ramp_gens: 1,
        intensity: 0.42,
        center_xy: Some([10.0, 20.0]),
        radius: Some(50.0),
    });
    events.events.push(ShockEvent {
        kind: ShockKind::ClimateShift,
        start_gen: 12,
        duration_gen: 5,
        ramp_gens: 1,
        intensity: 0.7,
        center_xy: None,
        radius: None,
    });
    events.events.push(ShockEvent {
        kind: ShockKind::FoodCrash,
        start_gen: 20,
        duration_gen: 4,
        ramp_gens: 1,
        intensity: 0.9,
        center_xy: None,
        radius: None,
    });
    write_events_sidecar(&out_path, 7, &events).expect("write sidecar");

    let sidecar = dir.join("events_seed7.csv");
    let content = std::fs::read_to_string(&sidecar).expect("read sidecar");
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 4);
    assert_eq!(
        lines[0],
        "start_gen,duration_gen,kind,intensity,center_x,center_y,radius",
    );
    assert!(lines[1].starts_with("5,3,hazard_pulse,"));
    assert!(lines[2].starts_with("12,5,climate_shift,"));
    assert!(lines[3].starts_with("20,4,food_crash,"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn write_events_sidecar_handles_empty_calendar_with_only_header() {
    let dir = unique_temp_dir("empty");
    let out_path = dir.join("run_seed42.csv");
    let events = EventCalendar::default();
    write_events_sidecar(&out_path, 42, &events).expect("write empty sidecar");
    let sidecar = dir.join("events_seed42.csv");
    let content = std::fs::read_to_string(&sidecar).expect("read");
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 1);
    assert_eq!(
        lines[0],
        "start_gen,duration_gen,kind,intensity,center_x,center_y,radius",
    );
    let _ = std::fs::remove_dir_all(&dir);
}
