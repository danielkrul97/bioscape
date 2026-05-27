//! Bevy 3D viewer for the torus planet experiment. Runs the SPH +
//! self-gravity sim in real time and renders each particle as a small
//! sphere whose `Transform` updates per frame from the GPU state.
//!
//! Controls:
//!   F1            — pause / resume
//!   F2            — single step (when paused)
//!   F4 / F5       — halve / double steps-per-frame
//!   F8            — toggle Rock / Temperature colouring
//!   R             — restart simulation from initial state (same seed)
//!   LMB drag      — orbit camera (horizontal = yaw, vertical = pitch)
//!   Scroll wheel  — zoom in / out
//!
//! Default `--n 3000` keeps the per-entity render path responsive
//! on integrated GPUs. For larger N either drop the mesh
//! (--no-render) or run `planet_headless` instead.

use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bioscape::planet::diagnostics::{principal_moments, total_energy, ScalarDiagnostics};
use bioscape::planet::init::TemperatureProfile;
use bioscape::planet::world::primary_radius;
use bioscape::planet::{init, PlanetConfig, PlanetShape, PlanetWorld};
use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Parser, Debug, Resource, Clone)]
#[command(name = "planet_view", about = "Torus planet viewer + live SPH simulation.")]
struct Cli {
    #[arg(long, default_value_t = 3_000)]
    n: usize,
    /// Initial particle distribution shape.
    #[arg(long, value_enum, default_value_t = PlanetShape::Torus)]
    shape: PlanetShape,
    /// Torus major radius (torus only).
    #[arg(long, default_value_t = 1.0)]
    r_major: f32,
    /// Torus minor radius (torus only).
    #[arg(long, default_value_t = 0.2)]
    r_minor: f32,
    /// Cube edge length (cube only).
    #[arg(long, default_value_t = 0.924)]
    cube_side: f32,
    /// Pancake disc radius (pancake only).
    #[arg(long, default_value_t = 1.0)]
    pancake_radius: f32,
    /// Pancake disc thickness (pancake only).
    #[arg(long, default_value_t = 0.251)]
    pancake_height: f32,
    #[arg(long, default_value_t = 0.5)]
    omega_frac: f32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 1e-3)]
    dt: f32,
    #[arg(long, default_value_t = 0.1)]
    eos_k: f32,
    #[arg(long, default_value_t = 5.0 / 3.0)]
    eos_gamma: f32,
    #[arg(long, default_value_t = 0.01)]
    softening: f32,
    /// Monaghan artificial-viscosity α (linear term).
    #[arg(long, default_value_t = 1.0)]
    visc_alpha: f32,
    /// Monaghan artificial-viscosity β (quadratic term).
    #[arg(long, default_value_t = 2.0)]
    visc_beta: f32,
    /// Initial steps per render frame.
    #[arg(long, default_value_t = 4)]
    steps_per_frame: u32,
    /// Sprint 206 — initial temperature profile.
    #[arg(long, value_enum, default_value_t = TemperatureProfile::Uniform)]
    init_temp_profile: TemperatureProfile,
    #[arg(long, default_value_t = 0.01)]
    init_temp_core: f32,
    #[arg(long, default_value_t = 0.01)]
    init_temp_surface: f32,
}

#[derive(Resource)]
struct Sim {
    world: PlanetWorld,
    paused: bool,
    steps_per_frame: u32,
    t_ff: f32,
}

#[derive(Component)]
struct ParticleIdx(usize);

/// Per-particle override stored at spawn so the F8 Rock→Temperature
/// toggle can flip back without re-randomising the original palette.
#[derive(Component)]
struct RockMaterial(Handle<StandardMaterial>);

#[derive(Component)]
struct HudText;

#[derive(Resource)]
struct CameraOrbit {
    yaw: f32,
    pitch: f32,
    distance: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum ColorMode {
    Rock,
    #[default]
    Temperature,
}

#[derive(Resource, Default)]
struct ViewState {
    mode: ColorMode,
}

/// Pre-baked 16-step viridis-like ramp used when the viewer is in
/// `ColorMode::Temperature`. Per-frame work is a bucket lookup + handle
/// swap; Bevy then batches by handle so the draw call count stays low.
#[derive(Resource)]
struct ThermalPalette {
    handles: Vec<Handle<StandardMaterial>>,
}

fn main() {
    let cli = Cli::parse();
    App::new()
        .insert_resource(cli)
        .insert_resource(CameraOrbit {
            yaw: 0.6,
            pitch: 0.45,
            distance: 4.5,
        })
        .insert_resource(ViewState::default())
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Bioscape — Torus Planet".into(),
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                handle_input,
                tick_simulation,
                sync_particles,
                orbit_camera,
                update_hud,
            )
                .chain(),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    cli: Res<Cli>,
) {
    let mut config = PlanetConfig::default();
    config.shape = cli.shape;
    config.n_particles = cli.n;
    config.r_major = cli.r_major;
    config.r_minor = cli.r_minor;
    config.cube_side = cli.cube_side;
    config.pancake_radius = cli.pancake_radius;
    config.pancake_height = cli.pancake_height;
    config.seed = cli.seed;
    config.dt = cli.dt;
    config.eos_k = cli.eos_k;
    config.eos_gamma = cli.eos_gamma;
    config.softening = cli.softening;
    config.visc_alpha = cli.visc_alpha;
    config.visc_beta = cli.visc_beta;
    config.omega = init::omega_from_frac(&config, cli.omega_frac);

    let mut world = PlanetWorld::new(config.clone());
    world.particles = init::generate(&config);
    init::apply_temperature_profile(
        &mut world.particles,
        cli.init_temp_profile,
        cli.init_temp_core,
        cli.init_temp_surface,
        primary_radius(&config),
    );

    if let Err(e) = world.init_gpu_full() {
        eprintln!("gpu init failed: {e}");
        std::process::exit(1);
    }
    world.download_state();
    let t_ff = world.t_ff();
    let size_desc = match cli.shape {
        PlanetShape::Torus => format!("R={}, r={}", cli.r_major, cli.r_minor),
        PlanetShape::Cube => format!("side={}", cli.cube_side),
        PlanetShape::Pancake => format!(
            "radius={}, height={}",
            cli.pancake_radius, cli.pancake_height
        ),
    };
    eprintln!(
        "planet_view: shape={:?} spawned {} particles ({}, omega={}, t_ff={:.4})",
        cli.shape,
        world.particles.len(),
        size_desc,
        config.omega,
        t_ff,
    );

    // Per-particle visual variation — each entity is a sphere stretched
    // into an irregular ellipsoid (independent jitter on x/y/z) and
    // tilted with a random orientation. Combined with a tight earth-tone
    // palette, the swarm reads as a cloud of pebbles / rock fragments
    // rather than a uniform sea of perfect spheres. Single mesh handle
    // keeps Bevy batching tight; non-uniform `Transform.scale` is the
    // cheap mechanism for the irregular silhouette.
    let base_r = 0.5 * world.particles.smoothing_lengths.first().copied().unwrap_or(0.05);
    let sphere_mesh = meshes.add(Sphere::new(base_r));
    let material_pool: Vec<Handle<StandardMaterial>> = (0..6)
        .map(|i| {
            // Earth/stone palette mixing cool slate-grey with warm
            // ochre/terracotta. Low saturation, narrow lightness band.
            let (hue, sat, light) = match i {
                0 => (30.0, 0.18, 0.36), // dark umber
                1 => (25.0, 0.22, 0.48), // sienna
                2 => (40.0, 0.20, 0.52), // ochre
                3 => (20.0, 0.16, 0.42), // dusty terracotta
                4 => (45.0, 0.10, 0.55), // pale sandstone
                _ => (15.0, 0.08, 0.34), // slate
            };
            materials.add(StandardMaterial {
                base_color: Color::hsl(hue, sat, light),
                metallic: 0.0,
                perceptual_roughness: 0.95,
                ..default()
            })
        })
        .collect();

    let mut viz_rng = StdRng::seed_from_u64(cli.seed.wrapping_add(0xA5_A5_5A_5A));
    for (i, pos) in world.particles.positions.iter().enumerate() {
        let material = material_pool[viz_rng.random_range(0..material_pool.len())].clone();
        let sx = (viz_rng.random_range(-0.35..0.35) as f32).exp();
        let sy = (viz_rng.random_range(-0.35..0.35) as f32).exp();
        let sz = (viz_rng.random_range(-0.35..0.35) as f32).exp();
        let axis = Vec3::new(
            viz_rng.random_range(-1.0..1.0),
            viz_rng.random_range(-1.0..1.0),
            viz_rng.random_range(-1.0..1.0),
        )
        .try_normalize()
        .unwrap_or(Vec3::Z);
        let angle: f32 = viz_rng.random_range(0.0..std::f32::consts::TAU);
        commands.spawn((
            ParticleIdx(i),
            RockMaterial(material.clone()),
            Mesh3d(sphere_mesh.clone()),
            MeshMaterial3d(material),
            Transform {
                translation: Vec3::new(pos[0], pos[1], pos[2]),
                rotation: Quat::from_axis_angle(axis, angle),
                scale: Vec3::new(sx, sy, sz),
            },
        ));
    }

    // Pre-build the 16-step thermal ramp (blue → cyan → green → yellow → red).
    let thermal_handles: Vec<Handle<StandardMaterial>> = (0..16)
        .map(|k| {
            let f = k as f32 / 15.0;
            let rgb = viridis_like(f);
            materials.add(StandardMaterial {
                base_color: Color::srgb(rgb[0], rgb[1], rgb[2]),
                metallic: 0.0,
                perceptual_roughness: 0.6,
                emissive: LinearRgba::new(rgb[0] * 0.4, rgb[1] * 0.4, rgb[2] * 0.4, 1.0),
                ..default()
            })
        })
        .collect();
    commands.insert_resource(ThermalPalette {
        handles: thermal_handles,
    });

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(3.0, 2.0, 3.0).looking_at(Vec3::ZERO, Vec3::Z),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn(AmbientLight {
        color: Color::WHITE,
        brightness: 250.0,
        ..default()
    });

    commands.spawn((
        HudText,
        Text::new(""),
        TextFont {
            font_size: 14.0,
            ..default()
        },
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            left: Val::Px(8.0),
            ..default()
        },
    ));

    commands.insert_resource(Sim {
        world,
        paused: false,
        steps_per_frame: cli.steps_per_frame,
        t_ff,
    });
}

fn handle_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut sim: ResMut<Sim>,
    mut view: ResMut<ViewState>,
) {
    if keys.just_pressed(KeyCode::F1) {
        sim.paused = !sim.paused;
        eprintln!("paused = {}", sim.paused);
    }
    if keys.just_pressed(KeyCode::F2) && sim.paused {
        for _ in 0..sim.steps_per_frame {
            sim.world.tick_sph();
        }
        sim.world.download_state();
    }
    if keys.just_pressed(KeyCode::F4) {
        sim.steps_per_frame = (sim.steps_per_frame / 2).max(1);
        eprintln!("steps_per_frame = {}", sim.steps_per_frame);
    }
    if keys.just_pressed(KeyCode::F5) {
        sim.steps_per_frame = (sim.steps_per_frame * 2).min(512);
        eprintln!("steps_per_frame = {}", sim.steps_per_frame);
    }
    if keys.just_pressed(KeyCode::F8) {
        view.mode = match view.mode {
            ColorMode::Rock => ColorMode::Temperature,
            ColorMode::Temperature => ColorMode::Rock,
        };
        eprintln!(
            "color mode = {}",
            if view.mode == ColorMode::Rock { "Rock" } else { "Temperature" }
        );
    }
    if keys.just_pressed(KeyCode::KeyR) {
        sim.world.reset();
        eprintln!("simulation reset (tick = 0, time = 0)");
    }
}

/// Smooth viridis-like ramp via piecewise-linear segments on key
/// breakpoints. Avoids a 256-entry lookup table at the cost of a few
/// extra branches; visually indistinguishable for the 16-step palette.
fn viridis_like(f: f32) -> [f32; 3] {
    let f = f.clamp(0.0, 1.0);
    let stops: [(f32, [f32; 3]); 5] = [
        (0.00, [0.267, 0.005, 0.329]),
        (0.25, [0.231, 0.318, 0.546]),
        (0.50, [0.128, 0.567, 0.551]),
        (0.75, [0.478, 0.821, 0.318]),
        (1.00, [0.993, 0.906, 0.144]),
    ];
    for i in 0..stops.len() - 1 {
        let (a_t, a_c) = stops[i];
        let (b_t, b_c) = stops[i + 1];
        if f <= b_t {
            let span = (b_t - a_t).max(1e-30);
            let u = (f - a_t) / span;
            return [
                a_c[0] + (b_c[0] - a_c[0]) * u,
                a_c[1] + (b_c[1] - a_c[1]) * u,
                a_c[2] + (b_c[2] - a_c[2]) * u,
            ];
        }
    }
    stops.last().unwrap().1
}

fn tick_simulation(mut sim: ResMut<Sim>) {
    if sim.paused {
        return;
    }
    let steps = sim.steps_per_frame;
    for _ in 0..steps {
        sim.world.tick_sph();
    }
    sim.world.download_state();
}

fn sync_particles(
    sim: Res<Sim>,
    view: Res<ViewState>,
    palette: Option<Res<ThermalPalette>>,
    mut q: Query<(
        &ParticleIdx,
        &mut Transform,
        &mut MeshMaterial3d<StandardMaterial>,
        &RockMaterial,
    )>,
) {
    // For thermal mode, autoscale min/max of u per frame so the ramp
    // always shows full contrast. Cheap O(N) but only once per frame.
    let (t_lo, t_hi) = if view.mode == ColorMode::Temperature {
        let energies = &sim.world.particles.internal_energies;
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &u in energies {
            if u < lo { lo = u; }
            if u > hi { hi = u; }
        }
        if !lo.is_finite() || !hi.is_finite() || hi - lo < 1e-9 {
            lo = 0.0;
            hi = 1.0;
        }
        (lo, hi)
    } else {
        (0.0, 1.0)
    };

    for (idx, mut transform, mut mat, rock) in &mut q {
        let p = &sim.world.particles.positions[idx.0];
        transform.translation = Vec3::new(p[0], p[1], p[2]);
        match view.mode {
            ColorMode::Rock => {
                if mat.0 != rock.0 {
                    mat.0 = rock.0.clone();
                }
            }
            ColorMode::Temperature => {
                if let Some(pal) = &palette {
                    let u_i = sim.world.particles.internal_energies[idx.0];
                    let f = ((u_i - t_lo) / (t_hi - t_lo)).clamp(0.0, 1.0);
                    let bucket = ((f * (pal.handles.len() as f32 - 1.0)).round() as usize)
                        .min(pal.handles.len() - 1);
                    let new_handle = &pal.handles[bucket];
                    if mat.0 != *new_handle {
                        mat.0 = new_handle.clone();
                    }
                }
            }
        }
    }
}

fn orbit_camera(
    mouse_btn: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: MessageReader<MouseMotion>,
    mut mouse_wheel: MessageReader<MouseWheel>,
    mut orbit: ResMut<CameraOrbit>,
    mut q: Query<&mut Transform, With<Camera3d>>,
) {
    // Orbit only while LMB is held. When the button is released drain
    // the reader so motion accumulated outside the drag doesn't snap
    // the camera on the next press.
    let dragging = mouse_btn.pressed(MouseButton::Left);
    if dragging {
        const SENSITIVITY: f32 = 0.005;
        let mut delta = Vec2::ZERO;
        for ev in mouse_motion.read() {
            delta += ev.delta;
        }
        if delta != Vec2::ZERO {
            // Drag right → camera orbits CCW around target (yaw ↑).
            // Drag up    → camera tilts up        (pitch ↑).
            orbit.yaw += delta.x * SENSITIVITY;
            orbit.pitch =
                (orbit.pitch - delta.y * SENSITIVITY).clamp(-1.3, 1.3);
        }
    } else {
        mouse_motion.clear();
    }

    // Wheel: positive `y` is "scroll up" → zoom in → smaller distance.
    // Treat the two `MouseScrollUnit` variants on equal footing
    // (the trackpad pixel-delta case gets a ÷50 conversion to roughly
    // match a single click of a notched wheel).
    const ZOOM_STEP: f32 = 0.1;
    let mut scroll = 0.0_f32;
    for ev in mouse_wheel.read() {
        scroll += match ev.unit {
            MouseScrollUnit::Line => ev.y,
            MouseScrollUnit::Pixel => ev.y / 50.0,
        };
    }
    if scroll != 0.0 {
        let factor = (-scroll * ZOOM_STEP).exp();
        orbit.distance = (orbit.distance * factor).clamp(0.5, 50.0);
    }

    let cp = orbit.pitch.cos();
    let pos = Vec3::new(
        orbit.distance * orbit.yaw.cos() * cp,
        orbit.distance * orbit.yaw.sin() * cp,
        orbit.distance * orbit.pitch.sin(),
    );
    if let Ok(mut t) = q.single_mut() {
        *t = Transform::from_translation(pos).looking_at(Vec3::ZERO, Vec3::Z);
    }
}

fn update_hud(
    sim: Res<Sim>,
    view: Res<ViewState>,
    mut q: Query<&mut Text, With<HudText>>,
) {
    let particles = &sim.world.particles;
    let scalar = ScalarDiagnostics::compute(particles);
    let (ke, pe, e) = total_energy(particles, sim.world.config.g_const, sim.world.config.softening);
    let mom = principal_moments(particles);
    let axis_ac = mom[0] / mom[2].max(1e-30);
    let mode_name = if view.mode == ColorMode::Rock { "Rock" } else { "Temperature" };

    let text = format!(
        "tick: {}\ntime: {:.3} ({:.3} t_ff)\nsteps/frame: {}\npaused: {}\ncolor: {} (F8)\nKE: {:.4}\nPE: {:.4}\nU:  {:.4}\nE+U: {:.4}\nT̄: {:.4}\nT_min: {:.4}\nT_max: {:.4}\nLz: {:.4}\nI: [{:.3}, {:.3}, {:.3}]\na/c: {:.3}\nN: {}",
        sim.world.tick,
        sim.world.time,
        sim.world.time / sim.t_ff,
        sim.steps_per_frame,
        sim.paused,
        mode_name,
        ke,
        pe,
        scalar.internal_energy,
        e + scalar.internal_energy,
        scalar.mean_temperature,
        scalar.min_temperature,
        scalar.max_temperature,
        scalar.angular_momentum_z,
        mom[0],
        mom[1],
        mom[2],
        axis_ac,
        sim.world.particles.len(),
    );
    if let Ok(mut t) = q.single_mut() {
        *t = Text::new(text);
    }
}
