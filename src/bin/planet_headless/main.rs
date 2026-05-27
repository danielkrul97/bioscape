//! Headless harness for the planet-shape experiments — batch
//! parameter sweep entry point.
//!
//! Usage (torus, default):
//!   cargo run --release --bin planet_headless -- \
//!     --n 10000 --r-major 1.0 --r-minor 0.2 --omega-frac 0.5 \
//!     --t-end 10.0 --dt 1e-3 --diag-every 100 \
//!     --out run_torus.csv
//!
//! Same `t_end`, `omega_frac`, and CSV columns for `--shape cube` /
//! `--shape pancake`; the size flag set switches per shape.

use bioscape::planet::diagnostics::{
    inertia_tensor, principal_moments, total_energy, ScalarDiagnostics,
};
use bioscape::planet::init::TemperatureProfile;
use bioscape::planet::world::primary_radius;
use bioscape::planet::{init, PlanetConfig, PlanetShape, PlanetWorld};
use clap::Parser;
use std::io::{BufWriter, Write};
use std::time::Instant;
use wgpu::Maintain;

#[derive(Parser, Debug)]
#[command(
    name = "planet_headless",
    about = "Bioscape torus planet — headless batch experiment.",
)]
struct Cli {
    /// Number of SPH particles.
    #[arg(long, default_value_t = 10_000)]
    n: usize,

    /// Initial particle distribution shape.
    #[arg(long, value_enum, default_value_t = PlanetShape::Torus)]
    shape: PlanetShape,

    /// Torus major radius (only used when `--shape torus`).
    #[arg(long, default_value_t = 1.0)]
    r_major: f32,

    /// Torus minor radius (tube radius). Torus only.
    #[arg(long, default_value_t = 0.2)]
    r_minor: f32,

    /// Cube edge length, centred on origin. Cube only.
    #[arg(long, default_value_t = 0.924)]
    cube_side: f32,

    /// Pancake disc radius. Pancake only.
    #[arg(long, default_value_t = 1.0)]
    pancake_radius: f32,

    /// Pancake disc thickness along z. Pancake only.
    #[arg(long, default_value_t = 0.251)]
    pancake_height: f32,

    /// Rigid rotation rate, normalised to `Ω_circ = √(GM/R³)`.
    #[arg(long, default_value_t = 0.0)]
    omega_frac: f32,

    /// RNG seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Simulation length in free-fall times (`t_ff = √(R³/GM)`).
    #[arg(long, default_value_t = 5.0)]
    t_end: f32,

    /// Integration timestep (sim units).
    #[arg(long, default_value_t = 1e-3)]
    dt: f32,

    /// EOS coefficient K in `P = K · ρ^γ`.
    #[arg(long, default_value_t = 0.1)]
    eos_k: f32,

    /// EOS exponent γ.
    #[arg(long, default_value_t = 5.0 / 3.0)]
    eos_gamma: f32,

    /// Plummer softening ε.
    #[arg(long, default_value_t = 0.01)]
    softening: f32,

    /// Monaghan artificial-viscosity α (linear term).
    #[arg(long, default_value_t = 1.0)]
    visc_alpha: f32,

    /// Monaghan artificial-viscosity β (quadratic term).
    #[arg(long, default_value_t = 2.0)]
    visc_beta: f32,

    /// Ticks between diagnostic CSV rows.
    #[arg(long, default_value_t = 100)]
    diag_every: u64,

    /// Output CSV path. Defaults to `planet_seed{seed}.csv`.
    #[arg(long)]
    out: Option<String>,

    /// Profile mode — serialize each pipeline with `device.poll(Wait)`
    /// between dispatches and report µs/tick per stage at end of run.
    /// Run length is honoured (use `--t-end 0.1` for a quick profile).
    #[arg(long)]
    profile: bool,

    /// Sprint 206: initial temperature profile.
    #[arg(long, value_enum, default_value_t = TemperatureProfile::Uniform)]
    init_temp_profile: TemperatureProfile,

    /// Sprint 206: core/inner temperature for the chosen profile
    /// (in sim units; with `cv = 1`, equals internal energy per mass).
    #[arg(long, default_value_t = 0.01)]
    init_temp_core: f32,

    /// Sprint 206: surface/outer temperature for the chosen profile.
    /// Ignored when `--init-temp-profile uniform`.
    #[arg(long, default_value_t = 0.01)]
    init_temp_surface: f32,
}

fn main() {
    let cli = Cli::parse();
    let out_path = cli
        .out
        .clone()
        .unwrap_or_else(|| format!("planet_seed{}.csv", cli.seed));

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
    let t_ff = world.t_ff();
    let n_steps = ((cli.t_end * t_ff) / cli.dt).ceil() as u64;

    let size_desc = match cli.shape {
        PlanetShape::Torus => format!("R={} r={}", cli.r_major, cli.r_minor),
        PlanetShape::Cube => format!("side={}", cli.cube_side),
        PlanetShape::Pancake => format!(
            "radius={} height={}",
            cli.pancake_radius, cli.pancake_height
        ),
    };
    eprintln!(
        "planet_headless: shape={:?} n={} {} omega={} (frac={}) seed={} \
         t_ff={:.4} dt={} steps={} t_end={} t_ff",
        cli.shape,
        cli.n,
        size_desc,
        config.omega,
        cli.omega_frac,
        cli.seed,
        t_ff,
        cli.dt,
        n_steps,
        cli.t_end,
    );

    if let Err(e) = world.init_gpu_full() {
        eprintln!("gpu init failed: {e}");
        std::process::exit(1);
    }
    eprintln!("gpu init OK (pipelines: nbody, kick, drift, density, sph_force, thermal_integrate)");

    let file = std::fs::File::create(&out_path).expect("can't create CSV");
    let mut log = BufWriter::new(file);
    writeln!(
        log,
        "tick,time,t_over_t_ff,mass,ke,pe,e_total,u_total,e_full,mean_t,min_t,max_t,drift_pct,lz,i_a,i_b,i_c,axis_a_over_c,axis_b_over_c,max_radius"
    )
    .unwrap();

    let start = Instant::now();
    let mut last_progress = Instant::now();

    // Initial diagnostic row (t = 0).
    world.download_state();
    let scalar0 = ScalarDiagnostics::compute(&world.particles);
    let (ke0, pe0, _) = total_energy(&world.particles, world.config.g_const, cli.softening);
    let e_full_init = ke0 + pe0 + scalar0.internal_energy;
    write_diag(
        &mut log,
        world.tick,
        world.time,
        t_ff,
        &world,
        cli.softening,
        e_full_init,
    );
    let mut last_drift_warn = 0.0_f64;

    let mut profile = if cli.profile {
        Some(ProfileBuckets::default())
    } else {
        None
    };

    for step in 1..=n_steps {
        if let Some(p) = profile.as_mut() {
            tick_sph_profiled(&mut world, p);
        } else {
            world.tick_sph();
        }
        if step % cli.diag_every == 0 || step == n_steps {
            world.download_state();
            let drift = write_diag(
                &mut log,
                world.tick,
                world.time,
                t_ff,
                &world,
                cli.softening,
                e_full_init,
            );
            // Drift detector — only warn on each new 1 % milestone so
            // long runs don't spam the log.
            let milestone = (drift.abs() * 100.0).floor();
            if milestone >= 1.0 && milestone > last_drift_warn {
                eprintln!(
                    "  ⚠ energy drift {:+.2} % at step {} (radiation lowers e_full; \
                     adiabatic/leapfrog drift can lift it — interpret with care)",
                    drift * 100.0,
                    step
                );
                last_drift_warn = milestone;
            }
        }
        if last_progress.elapsed().as_secs() >= 5 {
            eprintln!(
                "  step {}/{} ({:.1}%, {:.2} steps/sec)",
                step,
                n_steps,
                100.0 * step as f32 / n_steps as f32,
                step as f32 / start.elapsed().as_secs_f32(),
            );
            last_progress = Instant::now();
        }
    }
    log.flush().unwrap();

    if let Some(p) = profile.as_ref() {
        p.print(n_steps, world.particles.len());
    }

    let elapsed = start.elapsed();
    eprintln!(
        "done. {} steps in {:.1}s ({:.0} steps/s). out: {}",
        n_steps,
        elapsed.as_secs_f32(),
        n_steps as f32 / elapsed.as_secs_f32().max(1e-3),
        out_path,
    );
}

#[derive(Default)]
struct ProfileBuckets {
    kick1: f64,
    drift: f64,
    hash: f64,
    density: f64,
    nbody: f64,
    sph_force: f64,
    kick2: f64,
}

impl ProfileBuckets {
    fn print(&self, n_steps: u64, n_particles: usize) {
        let steps = n_steps as f64;
        let entries = [
            ("nbody (O(N²))", self.nbody),
            ("hash rebuild", self.hash),
            ("density", self.density),
            ("sph_force (P+ν)", self.sph_force),
            ("kick₁", self.kick1),
            ("kick₂", self.kick2),
            ("drift", self.drift),
        ];
        let total: f64 = entries.iter().map(|(_, v)| v).sum();
        let mut rows: Vec<_> = entries.iter().collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        eprintln!();
        eprintln!(
            "=== GPU profile: {} steps, N = {} particles ===",
            n_steps, n_particles
        );
        eprintln!(
            "  {:<20} {:>11}  {:>6}  {:>14}",
            "stage", "µs/step", "%", "total ms"
        );
        for (name, v) in rows {
            eprintln!(
                "  {:<20} {:>11.1}  {:>5.1}%  {:>14.1}",
                name,
                v / steps,
                100.0 * v / total.max(1e-9),
                v / 1000.0
            );
        }
        eprintln!(
            "  {:<20} {:>11.1}  {:>5.1}%  {:>14.1}",
            "TOTAL (measured)",
            total / steps,
            100.0,
            total / 1000.0
        );
    }
}

fn tick_sph_profiled(world: &mut PlanetWorld, p: &mut ProfileBuckets) {
    let n = world.particles.len();
    if n == 0 {
        return;
    }
    let ctx = world.gpu_ctx.as_ref().unwrap().clone();
    let gpu = world.gpu_state.as_ref().unwrap();
    let hash = world.hash.as_ref().unwrap();
    let density = world.density.as_ref().unwrap();
    let sph_force = world.sph_force.as_ref().unwrap();
    let dt = world.config.dt;
    let g = world.config.g_const;
    let eps = world.config.softening;
    let k = world.config.eos_k;
    let gamma = world.config.eos_gamma;
    let alpha = world.config.visc_alpha;
    let beta = world.config.visc_beta;

    macro_rules! timed {
        ($bucket:expr, $body:block) => {{
            let t = Instant::now();
            $body
            ctx.device.poll(Maintain::Wait);
            $bucket += t.elapsed().as_secs_f64() * 1e6;
        }};
    }

    timed!(p.kick1, { gpu.kick(n, 0.5 * dt); });
    timed!(p.drift, { gpu.drift(n, dt); });
    timed!(p.hash, { hash.rebuild(n); });
    timed!(p.density, { density.dispatch(n); });
    timed!(p.nbody, { gpu.compute_accelerations(n, g, eps); });
    timed!(p.sph_force, { sph_force.dispatch(n, k, gamma, alpha, beta); });
    timed!(p.kick2, { gpu.kick(n, 0.5 * dt); });

    world.tick += 1;
    world.time += dt;
}

/// Writes one diagnostic row and returns the relative drift of e_full
/// vs. the run's initial value (radiation pulls negative, leapfrog
/// drift can push positive).
fn write_diag<W: Write>(
    log: &mut W,
    tick: u64,
    time: f32,
    t_ff: f32,
    world: &PlanetWorld,
    softening: f32,
    e_full_init: f64,
) -> f64 {
    let particles = &world.particles;
    let scalar = ScalarDiagnostics::compute(particles);
    let (ke, pe, e_total) = total_energy(particles, world.config.g_const, softening);
    let _it = inertia_tensor(particles);
    let mom = principal_moments(particles);
    let axis_ac = mom[0] / mom[2].max(1e-30);
    let axis_bc = mom[1] / mom[2].max(1e-30);

    let mut max_r2 = 0.0_f32;
    for pos in &particles.positions {
        let r2 = pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2];
        if r2 > max_r2 {
            max_r2 = r2;
        }
    }
    let max_r = max_r2.sqrt();

    let u_total = scalar.internal_energy;
    let mean_t = scalar.mean_temperature;
    let min_t = scalar.min_temperature;
    let max_t = scalar.max_temperature;
    let e_full = e_total + u_total;
    let drift = if e_full_init.abs() > 1e-30 {
        (e_full - e_full_init) / e_full_init.abs()
    } else {
        0.0
    };
    writeln!(
        log,
        "{tick},{time:.6},{:.6},{:.6},{ke:.6},{pe:.6},{e_total:.6},{u_total:.6},{e_full:.6},{mean_t:.6},{min_t:.6},{max_t:.6},{drift:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{max_r:.6}",
        time / t_ff,
        scalar.total_mass,
        scalar.angular_momentum_z,
        mom[0],
        mom[1],
        mom[2],
        axis_ac,
        axis_bc,
    )
    .unwrap();
    drift
}
