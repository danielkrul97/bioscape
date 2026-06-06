use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::image::Image;
use bevy::pbr::{DistanceFog, FogFalloff};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::view::Hdr;
use bioscape::gpu::{
    BrainGpu, BrownianGpu, CellsGpu, FieldGpu, GpuContext, HebbianGpu, MotorGpu, PopulateInputsGpu,
    SensorGatherGpu, SpatialHashGpu, StepGpu,
};
use bioscape::{
    reject_food_for_richness, Cell, EventCalendar, Food, ShockScheduleConfig, SmellField, WorldMap,
    CELL_RADIUS, CYCLE_AMPLITUDE, INITIAL_CELLS, MAX_POPULATION, MAX_SPAWN_ATTEMPTS,
    PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z, SMELL_GRID_RES, SMELL_GRID_RES_Z, SPIKE_SLOTS,
    VIBRATION_GRID_RES, VIBRATION_GRID_RES_Z, WORLD_HALF, WORLD_MAP_BASE_RES, WORLD_MAP_BASE_RES_Z,
    WORLD_MAP_RES, WORLD_MAP_RES_Z, WORLD_MAP_SEED,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Duration;

use super::components::{
    CellEntity, FoodEntity, SpikeEntity, StatsRoot, StatsText, WorldMapOverlay,
};
use super::config::{CAMERA_OFFSET_DISTANCE, FOOD_RADIUS};
use super::material::{adhesion_material, cell_rotation, cell_scale, BioMaterial};
use super::resources::{
    AdhesionMaterials, CellMesh, CellSlotMap, EventCalendarResource, FoodMaterial, FoodMesh,
    OrbitCamera, PheromoneResource, SimRng, SimWorld, SmellResource, SpikeMaterial, SpikeMesh,
    VibrationResource, WorldExtent, WorldMapResource,
};
use super::resources_gpu::GpuFullPipeline;
use super::sim_config::{SimConfig, CONFIG_FILENAME};
use super::world_map::{food_target, world_map_image};

pub(super) fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut bio_materials: ResMut<Assets<BioMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut adhesion_materials: ResMut<AdhesionMaterials>,
    mut window: Single<&mut Window>,
) {
    window.set_maximized(true);
    let half = WORLD_HALF;
    let extent = WorldExtent {
        half_x: half[0],
        half_y: half[1],
        half_z: half[2],
    };
    commands.insert_resource(extent);

    // Sprint 183 (post-S182 cleanup): load renderer overrides from
    // `bioscape.json` in CWD. Missing/unparseable file → library
    // defaults (identical to pre-S183 hardcoded behavior).
    let config = SimConfig::load_or_default(std::path::Path::new(CONFIG_FILENAME));

    // Sprint 184: build the event calendar from `config.seed` (not the
    // literal `WORLD_MAP_SEED` as pre-S184) and feed it into the shared
    // `World`. Without this the renderer's sim ran with an empty calendar
    // — shocks never fired regardless of `BIOSCAPE_SHOCKS_MEAN_GENS`.
    let shocks_mean_gens: u32 = std::env::var("BIOSCAPE_SHOCKS_MEAN_GENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let shock_cfg = if shocks_mean_gens > 0 {
        ShockScheduleConfig {
            mean_gens_between: shocks_mean_gens,
            ..Default::default()
        }
    } else {
        ShockScheduleConfig::default()
    };
    let event_calendar = EventCalendar::generate(config.seed, &shock_cfg, 1_000_000);
    if shocks_mean_gens > 0 {
        info!(
            "shocks: mean_gens_between={} scheduled={} (seed={})",
            shocks_mean_gens,
            event_calendar.events.len(),
            config.seed,
        );
    }

    // Sprint 175-176-183: instantiate shared `bioscape::sim::World` from
    // resolved config. GPU init runs alongside Bevy's RenderPlugin (2 wgpu
    // instances; consolidation in 184+).
    {
        let mut sim_rng = StdRng::seed_from_u64(config.seed);
        let mut world = bioscape::sim::World::new_with_maze(
            &mut sim_rng,
            config.resolved_map_seed(),
            config.resolved_mating_radius(),
            config.resolved_initial_cells(),
            config.resolved_max_population(),
            event_calendar.clone(),
            config.resolved_maze(),
        );
        // Sprint 183: pre-seed Izhikevich fraction (mirror headless
        // `--initial-izhikevich-frac` from S159).
        let izh_frac = config.initial_izhikevich_frac.clamp(0.0, 1.0);
        if izh_frac > 0.0 {
            let target = (izh_frac * world.cells.len() as f32).round() as usize;
            for cell in world.cells.iter_mut().take(target) {
                cell.genome.neuron_model = bioscape::NeuronModel::Izhikevich;
            }
            info!(
                "sim-world: pre-seeded {} of {} cells as Izhikevich (frac={:.2})",
                target,
                world.cells.len(),
                izh_frac
            );
        }
        match world.init_gpu_full() {
            Ok(()) => {
                info!(
                    "sim-world: shared sim driver initialised ({} initial cells, seed={})",
                    world.cells.len(),
                    config.seed
                );
            }
            Err(e) => {
                panic!("sim-world: init_gpu_full failed ({e}); GPU mandatory");
            }
        }
        commands.insert_resource(SimWorld(world));
        commands.insert_resource(SimRng(sim_rng));
    }

    // Sprint 36: Camera3d s orthographic projection — "scale" zoom feel bez
    // perspective void okolo scény. `IsDefaultUiCamera` marker říká
    // bevy_ui_render ať použije tuto kameru pro UI.
    // Near/far explicitně dimenzované na CAMERA_OFFSET_DISTANCE — default_3d()
    // má far ~1000, ale camera je 3000 od target, takže by scéna padla za far
    // plane a vše by bylo culled.
    //
    // Sprint 88: HDR + Bloom + Tonemapping + DistanceFog atmospheric pass.
    // HDR backbuffer dovolí emissive > 1.0 (cell glow), Bloom rozšíří
    // bright pixels na soft halos, Tonemapping namapuje HDR rozsah na sRGB.
    // DistanceFog přidá deep-ocean blue tint na vzdálené objekty (ortho má
    // limited depth differentiation, ale fade k floor overlay je signifikantní).
    let initial_orbit = OrbitCamera::default();
    commands.spawn((
        Camera3d::default(),
        Hdr,
        // Sprint 88.2: TonyMcMapface → AcesFitted (increased saturation), pak
        // S88.3: AcesFitted → Reinhard. ACES desaturuje brights („brights
        // desaturate across the spectrum"); Reinhard je jediný tonemapper, kde
        // „bright primaries and secondaries don't desaturate at all". Tradeoff:
        // lots of hue shifting v brights (nepatrné posuny barev), ale pro
        // 8 distinct adhesion hues je full saturation > hue purity.
        Tonemapping::Reinhard,
        Bloom::NATURAL,
        DistanceFog {
            color: Color::srgb(0.08, 0.18, 0.30),
            falloff: FogFalloff::ExponentialSquared { density: 0.0002 },
            ..default()
        },
        IsDefaultUiCamera,
        Projection::Orthographic(OrthographicProjection {
            scale: initial_orbit.scale,
            near: 0.1,
            far: CAMERA_OFFSET_DISTANCE * 3.0,
            ..OrthographicProjection::default_3d()
        }),
        initial_orbit.transform(),
    ));

    // Ambient + DirectionalLight pro 3D scénu. Sprint 88: tinted bluish ambient
    // (underwater feel) + DirectionalLight jako "sluneční" key light pronikající
    // od povrchu šikmo. Sprint 88.1: bumped up brightness — pre-fix illuminance
    // 6000 + ambient 600 produkovaly blackout scene. HDR + bloom kombinaci nutno
    // krmit dostatkem světla aby base scene byla viditelná, ne jen emissive
    // bloom highlights.
    // Sprint 88.2: ambient méně blue (0.6 → 0.85) — silně modré ambient
    // multiplikuje s cell hue a desaturuje warm colors (red/orange/yellow
    // adhesion types). Subtler tint zachová cell color identity.
    commands.spawn(AmbientLight {
        color: Color::srgb(0.85, 0.92, 1.0),
        brightness: 1500.0,
        ..default()
    });
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.95, 0.97, 1.0),
            illuminance: 10000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            -std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_6,
            0.0,
        )),
    ));

    // Sprint 53: WorldMap + SmellField/Pheromone plně 3D volumetric.
    let world_map = WorldMap::new(
        [WORLD_MAP_RES, WORLD_MAP_RES, WORLD_MAP_RES_Z],
        [WORLD_MAP_BASE_RES, WORLD_MAP_BASE_RES, WORLD_MAP_BASE_RES_Z],
        half,
        WORLD_MAP_SEED,
    );

    // Sprint 36: WorldMap overlay jako ground plane na z=-half_z-5 (pod cells).
    // Texture je grayscale richness; v 3D pohledu funguje jako "podlaha" světa.
    let overlay_image_handle = images.add(world_map_image(&world_map));
    let overlay_material = materials.add(StandardMaterial {
        base_color_texture: Some(overlay_image_handle),
        unlit: true,
        ..default()
    });
    let overlay_mesh = meshes.add(Plane3d::default().mesh().size(2.0 * half[0], 2.0 * half[1]));
    commands.spawn((
        Mesh3d(overlay_mesh),
        MeshMaterial3d(overlay_material),
        Transform::from_xyz(0.0, 0.0, -half[2] - 5.0)
            // Plane3d defaultně leží v xz; rotujem do xy aby normála ukazovala +z.
            .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
        WorldMapOverlay,
    ));

    // Sprint 36: cell mesh = unit-radius sphere, scale aplikuje ellipsoid
    // (length × width × height) per cell.
    let cell_mesh_handle = meshes.add(Sphere::new(CELL_RADIUS).mesh().ico(2).unwrap());
    // Spike mesh: unit cone (radius=1, height=1) v Bevy default orientation —
    // apex +Y, base v origin. Sync system škáluje na (thickness, length, thickness)
    // a rotuje tak, aby Y axis aligned se spike_direction.
    let spike_mesh_handle = meshes.add(
        Cone {
            radius: 1.0,
            height: 1.0,
        }
        .mesh(),
    );
    let spike_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.10, 0.10),
        perceptual_roughness: 0.4,
        metallic: 0.6,
        ..default()
    });
    // Sprint 53: jídlo decentnější — menší radius (10× větší food count po
    // 3D volume scaling jinak vytváří plný display) + ground-matching tint
    // (low-saturation green) místo skoro-černé proti bílému ClearColoru.
    let food_mesh_handle = meshes.add(Sphere::new(FOOD_RADIUS).mesh().ico(1).unwrap());
    let food_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.20, 0.30, 0.18),
        ..default()
    });

    let mut rng = rand::rng();
    let mut initial_cells: Vec<Cell> = Vec::with_capacity(INITIAL_CELLS);
    let mut slot_map = CellSlotMap::default();
    for i in 0..INITIAL_CELLS {
        // Sprint 66: cell_id == lineage_id pro initial pop (1:1 mapping). Po
        // mating se cell_id čerpá z `NextCellId` resource counteru.
        let cell = Cell::random(&mut rng, half, i as u64, 0, i as u64);
        let mat = adhesion_material(
            &mut adhesion_materials,
            &mut bio_materials,
            cell.genome.adhesion_type,
        );
        let entity = commands
            .spawn((
                CellEntity(cell),
                Mesh3d(cell_mesh_handle.clone()),
                MeshMaterial3d(mat),
                Transform::from_xyz(cell.position[0], cell.position[1], cell.position[2])
                    .with_rotation(cell_rotation(cell.heading, cell.pitch))
                    .with_scale(cell_scale(&cell.phenotype)),
            ))
            .id();
        for slot in 0..SPIKE_SLOTS as u8 {
            commands.spawn((
                SpikeEntity {
                    owner: entity,
                    slot,
                },
                Mesh3d(spike_mesh_handle.clone()),
                MeshMaterial3d(spike_material.clone()),
                Transform::default(),
                Visibility::Hidden,
            ));
        }
        slot_map.allocate(entity);
        initial_cells.push(cell);
    }

    // Wave N: GPU full pipeline is mandatory. Legacy `BIOSCAPE_GPU_BRAIN=1`
    // brain-only path is gone (it never matched gpu-full feature parity).
    // `BIOSCAPE_GPU_FULL=0` opt-out is also gone — init failure now panics
    // because there is no CPU compute fallback to fall through to.
    {
        // Full GPU pipeline (mirror headless --gpu-full): single-Wait
        // readback, sensor + populate + brain + motor + step + brownian
        // + collision + predate + food_spawn on GPU.
        let cap = MAX_POPULATION + 64;
        let initial_food_target = food_target(&extent, 1.0 + CYCLE_AMPLITUDE);
        let field_sources_cap = (initial_food_target + cap) * 2;
        let world_half = extent.as_array();
        let init_full = || -> Result<GpuFullPipeline, String> {
            let ctx = GpuContext::new()?;
            let cells = CellsGpu::with_context(&ctx, cap);
            cells.upload_brains(initial_cells.iter().map(|c| &c.genome.brain));
            // V7-unification: seed from `cell_id` (stable, unique per cell)
            // so CPU `Cell.xoshiro_state` and GPU per-slot state expand from
            // the same input and produce identical brownian streams.
            cells.upload_xoshiro_seeds(initial_cells.iter().map(|c| c.cell_id));
            let turn_rates: Vec<f32> = initial_cells.iter().map(|c| c.genome.turn_rate).collect();
            cells.upload_turn_rates(&turn_rates);
            let brain = BrainGpu::with_context(&ctx, cap)?;
            let hebbian = HebbianGpu::with_context(&ctx, cap)?;
            let brownian = BrownianGpu::with_context(&ctx, cap)?;
            let smell = FieldGpu::with_context(
                &ctx,
                [SMELL_GRID_RES, SMELL_GRID_RES, SMELL_GRID_RES_Z],
                world_half,
                field_sources_cap,
            )?;
            let pheromone = FieldGpu::with_context(
                &ctx,
                [PHEROMONE_GRID_RES, PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z],
                world_half,
                field_sources_cap,
            )?;
            let pheromone_ch1 = FieldGpu::with_context(
                &ctx,
                [PHEROMONE_GRID_RES, PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z],
                world_half,
                field_sources_cap,
            )?;
            let pheromone_ch2 = FieldGpu::with_context(
                &ctx,
                [PHEROMONE_GRID_RES, PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z],
                world_half,
                field_sources_cap,
            )?;
            let vibration = FieldGpu::with_context(
                &ctx,
                [
                    bioscape::VIBRATION_GRID_RES,
                    bioscape::VIBRATION_GRID_RES,
                    bioscape::VIBRATION_GRID_RES_Z,
                ],
                world_half,
                cap,
            )?;
            let cell_hash = SpatialHashGpu::with_context(
                &ctx,
                cap,
                bioscape::GRID_CELL_SIZE,
                [world_half[0], world_half[1]],
            )?;
            let food_hash = SpatialHashGpu::with_context(
                &ctx,
                field_sources_cap,
                bioscape::GRID_CELL_SIZE,
                [world_half[0], world_half[1]],
            )?;
            let sensor = SensorGatherGpu::with_context(&ctx, cap, field_sources_cap)?;
            let populate = PopulateInputsGpu::with_context(&ctx)?;
            let motor = MotorGpu::with_context(&ctx, cap)?;
            let step = StepGpu::with_context(&ctx, cap)?;
            let predate = bioscape::gpu::PredateGpu::with_context(&ctx, cap)?;
            // Wave J port: GPU food rejection sampling. K-attempts buffer sized
            // for the worst-case dispatch (FOOD_SPAWN_RATE × MAX_SPAWN_ATTEMPTS).
            // World map uploaded once at init; obstacle mask gets refreshed on
            // each maze toggle in `input::toggle_maze_world`.
            let food_spawn_cap = bioscape::FOOD_SPAWN_RATE * bioscape::MAX_SPAWN_ATTEMPTS;
            let world_map_size = (bioscape::WORLD_MAP_RES
                * bioscape::WORLD_MAP_RES
                * bioscape::WORLD_MAP_RES_Z) as u64;
            let obstacle_mask_cap: u64 = 256 * 256 * 4;
            let food_spawn = bioscape::gpu::FoodSpawnGpu::with_context(
                &ctx,
                food_spawn_cap,
                world_map_size,
                obstacle_mask_cap,
            )?;
            food_spawn.upload_world_map(world_map.field());
            // GPU eat_food candidate selection. Capacity mirrors food_hash
            // (field_sources_cap) so the on-device per-tick food array
            // upload always fits.
            let eat_food = bioscape::gpu::EatFoodGpu::with_context(
                &ctx,
                cap,
                field_sources_cap,
                world_map_size,
            )?;
            eat_food.upload_world_map(world_map.field());
            let collision = bioscape::gpu::CollisionGpu::with_context(
                &ctx,
                cap,
                bioscape::GRID_CELL_SIZE,
                bioscape::CELL_RADIUS,
                bioscape::COLLISION_RESTITUTION,
                bioscape::gpu::AdhesionParams {
                    strength: bioscape::ADHESION_STRENGTH,
                    cross_type: bioscape::ADHESION_CROSS_TYPE,
                    range_factor: bioscape::ADHESION_RANGE_FACTOR,
                },
                bioscape::gpu::BondParams {
                    bonds_per_cell: bioscape::MAX_BONDS_PER_CELL as u32,
                    break_factor: bioscape::BOND_BREAK_FACTOR,
                },
                bioscape::MAX_COLLISION_CONTACTS_PER_CELL,
                [world_half[0], world_half[1]],
            )?;
            let cppn = bioscape::gpu::CppnGpu::with_context(&ctx, cap);
            Ok(GpuFullPipeline {
                cells,
                brain,
                hebbian,
                brownian,
                smell,
                pheromone,
                pheromone_ch1,
                pheromone_ch2,
                vibration,
                cell_hash,
                food_hash,
                sensor,
                populate,
                motor,
                step,
                collision,
                predate,
                food_spawn,
                eat_food,
                cppn,
                scratch: bioscape::gpu::GpuFullScratch::default(),
            })
        };
        match init_full() {
            Ok(pipeline) => {
                info!(
                    "renderer-gpu-full: brain + Hebbian + Brownian + Field + SensorGather + PopulateInputs + Motor + Step + Collision + Predate + FoodSpawn (cap {} cells, {} field sources)",
                    cap, field_sources_cap
                );
                commands.insert_resource(pipeline);
            }
            Err(e) => {
                panic!("renderer-gpu-full: init failed ({e}); GPU is mandatory");
            }
        }
    }
    commands.insert_resource(slot_map);
    let _ = initial_cells;
    let initial_food = food_target(&extent, 1.0);
    for _ in 0..initial_food {
        let mut food = Food::random(&mut rng, half);
        for _ in 0..MAX_SPAWN_ATTEMPTS {
            let richness = world_map.sample([food.position[0], food.position[1], 0.0]);
            if !reject_food_for_richness(&mut rng, richness) {
                break;
            }
            food = Food::random(&mut rng, half);
        }
        commands.spawn((
            FoodEntity(food),
            Mesh3d(food_mesh_handle.clone()),
            MeshMaterial3d(food_material.clone()),
            Transform::from_xyz(food.position[0], food.position[1], food.position[2]),
            Visibility::Hidden,
        ));
    }

    commands.insert_resource(CellMesh(cell_mesh_handle));
    commands.insert_resource(FoodMesh(food_mesh_handle));
    commands.insert_resource(FoodMaterial(food_material));
    commands.insert_resource(SpikeMesh(spike_mesh_handle));
    commands.insert_resource(SpikeMaterial(spike_material));

    commands.insert_resource(SmellResource(SmellField::new(
        [SMELL_GRID_RES, SMELL_GRID_RES, SMELL_GRID_RES_Z],
        half,
    )));
    commands.insert_resource(PheromoneResource {
        fields: std::array::from_fn(|_| {
            SmellField::new(
                [PHEROMONE_GRID_RES, PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z],
                half,
            )
        }),
    });
    commands.insert_resource(VibrationResource(SmellField::new(
        [VIBRATION_GRID_RES, VIBRATION_GRID_RES, VIBRATION_GRID_RES_Z],
        half,
    )));
    commands.insert_resource(WorldMapResource(world_map));
    commands.insert_resource(EventCalendarResource(event_calendar));
    commands.insert_resource(config);
}

/// Cap virtual-time delta na 50 ms — limit catch-up FixedUpdate ticků (~4 při
/// 60 Hz) po lag spike. Default 250 ms (Bevy's `DEFAULT_MAX_DELTA`) by povolil
/// 15+ ticků a exponenciálně by dohánělo zpoždění (death spiral). Sim po lagu
/// poběží pomaleji než real time, ale zotaví se.
///
/// Musí běžet jako Startup systém přes ResMut, ne přes `insert_resource` v
/// `App` builderu — `DefaultPlugins.build()` přepíše Time<Virtual> až po
/// našem `insert_resource`, takže ten by se ztratil.
pub(super) fn setup_time_cap(mut virtual_time: ResMut<Time<Virtual>>) {
    virtual_time.set_max_delta(Duration::from_millis(50));
    info!(
        "Time<Virtual>::max_delta capped to {:?}",
        virtual_time.max_delta()
    );
}

pub(super) fn setup_stats_overlay(mut commands: Commands) {
    commands.spawn((
        StatsRoot,
        Node {
            position_type: PositionType::Absolute,
            right: Val::Px(8.0),
            bottom: Val::Px(8.0),
            padding: UiRect::all(Val::Px(8.0)),
            flex_direction: FlexDirection::Column,
            border_radius: BorderRadius::all(Val::Px(4.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)),
        children![(
            StatsText,
            Text::new(""),
            TextColor(Color::srgb(0.9, 0.9, 0.9)),
            TextFont {
                font_size: 13.0,
                ..default()
            },
        )],
    ));
}
