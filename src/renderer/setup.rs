use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::image::Image;
use bevy::pbr::{DistanceFog, FogFalloff};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::view::Hdr;
use bioscape::{
    Cell, Food, Hunter, SmellField, WorldMap, CELL_RADIUS, CYCLE_AMPLITUDE, HUNTER_TARGET_COUNT,
    INITIAL_CELLS, MAX_POPULATION, MAX_SPAWN_ATTEMPTS, PHEROMONE_GRID_RES, PHEROMONE_GRID_RES_Z,
    SMELL_GRID_RES, SMELL_GRID_RES_Z, WORLD_HALF, WORLD_MAP_BASE_RES, WORLD_MAP_BASE_RES_Z,
    WORLD_MAP_RES, WORLD_MAP_RES_Z, WORLD_MAP_SEED, reject_food_for_richness,
};
#[cfg(feature = "gpu")]
use bioscape::gpu::{
    BrainGpu, BrownianGpu, CellsGpu, FieldGpu, GpuContext, HebbianGpu, MotorGpu,
    PopulateInputsGpu, SensorGatherGpu, SpatialHashGpu, StepGpu,
};
use std::time::Duration;

use super::components::{CellEntity, FoodEntity, HunterEntity, StatsRoot, StatsText, WorldMapOverlay};
use super::config::{CAMERA_OFFSET_DISTANCE, FOOD_RADIUS};
use super::material::{adhesion_material, cell_rotation, cell_scale, BioMaterial, BioMaterialExt};
use super::resources::{
    AdhesionMaterials, CellMesh, CellSlotMap, FoodMaterial, FoodMesh, HunterMaterial, HunterMesh,
    OrbitCamera, PheromoneResource, SmellResource, WorldExtent, WorldMapResource,
};
#[cfg(feature = "gpu")]
use super::resources_gpu::{GpuBrainState, GpuFieldState, GpuFullPipeline};
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

    // Sprint 36: Camera3d s orthographic projection — "scale" zoom feel bez
    // perspective void okolo scény. `IsDefaultUiCamera` marker říká
    // bevy_ui_render ať použije tuto kameru pro UI.
    // Near/far explicitně dimenzované na CAMERA_OFFSET_DISTANCE — default_3d()
    // má far ~1000, ale camera je 3000 od target, takže by scéna padla za far
    // plane a vše by bylo culled.
    //
    // Sprint 88: HDR + Bloom + Tonemapping + DistanceFog atmospheric pass.
    // HDR backbuffer dovolí emissive > 1.0 (cells/hunter glow), Bloom rozšíří
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
    let overlay_mesh =
        meshes.add(Plane3d::default().mesh().size(2.0 * half[0], 2.0 * half[1]));
    commands.spawn((
        Mesh3d(overlay_mesh),
        MeshMaterial3d(overlay_material),
        Transform::from_xyz(0.0, 0.0, -half[2] - 5.0)
            // Plane3d defaultně leží v xz; rotujem do xy aby normála ukazovala +z.
            .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
        WorldMapOverlay,
    ));

    // Sprint 36: cell mesh = unit-radius sphere, scale aplikuje ellipsoid
    // (length × width × height) per cell. Spike rendering vynechán (visual
    // loss; predace mechanika beze změny).
    let cell_mesh_handle = meshes.add(Sphere::new(CELL_RADIUS).mesh().ico(2).unwrap());
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
        slot_map.allocate(entity);
        initial_cells.push(cell);
    }

    // GPU compute init. Default = full pipeline (`GpuFullPipeline`, single-Wait
    // readback). Init failure → CPU SIMD fallback (Resource None).
    //
    // Env var precedence:
    //   `BIOSCAPE_GPU_FULL=0`  → opt-out, čistý CPU SIMD path
    //   `BIOSCAPE_GPU_BRAIN=1` → legacy brain-only GPU (overrides GPU_FULL)
    //   default                → GPU_FULL on
    //
    // Sprint 132 verdikt (CPU SIMD 5–10× faster) byl měřen na FRAGMENTED GPU
    // path (`BIOSCAPE_GPU_BRAIN=1`) s per-system `Maintain::Wait` ~10 ms/tick.
    // Single-Wait `download_full_batch_into` v gpu-full agreguje 9 readbacků
    // do 1 polled barriera. Init fail je safety net pro adapters bez compute
    // support.
    #[cfg(feature = "gpu")]
    let want_gpu_full =
        !matches!(std::env::var("BIOSCAPE_GPU_FULL").as_deref(), Ok("0"));
    #[cfg(feature = "gpu")]
    let want_gpu_brain = std::env::var("BIOSCAPE_GPU_BRAIN").as_deref() == Ok("1");
    #[cfg(feature = "gpu")]
    if want_gpu_brain {
        let cap = MAX_POPULATION + 64;
        let initial_food_target = food_target(&extent, 1.0 + CYCLE_AMPLITUDE);
        let field_sources_cap = (initial_food_target + cap) * 2;
        let world_half = extent.as_array();
        let init = || -> Result<(GpuBrainState, GpuFieldState), String> {
            let ctx = GpuContext::new()?;
            let cells = CellsGpu::with_context(&ctx, cap);
            cells.upload_brains(initial_cells.iter().map(|c| &c.genome.brain));
            cells.upload_xoshiro_seeds(initial_cells.iter().enumerate().map(|(slot, c)| {
                c.lineage_id ^ (slot as u64).wrapping_mul(0x9E3779B97F4A7C15)
            }));
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
            Ok((
                GpuBrainState { cells, brain, hebbian, brownian },
                GpuFieldState { smell, pheromone },
            ))
        };
        match init() {
            Ok((brain_state, field_state)) => {
                info!(
                    "renderer-gpu: persistent brain weights + Hebbian + Field (opt-in via BIOSCAPE_GPU_BRAIN=1, cap {} cells, {} field sources)",
                    cap, field_sources_cap
                );
                commands.insert_resource(brain_state);
                commands.insert_resource(field_state);
            }
            Err(e) => {
                warn!("renderer-gpu: init failed ({}); CPU compute path active", e);
            }
        }
    } else if want_gpu_full {
        // Full GPU pipeline (mirror headless `--gpu-full`): single-Wait readback,
        // sensor + populate + brain + motor + step + brownian na GPU. Default
        // path. Disable přes `BIOSCAPE_GPU_FULL=0` (forced CPU SIMD).
        let cap = MAX_POPULATION + 64;
        let initial_food_target = food_target(&extent, 1.0 + CYCLE_AMPLITUDE);
        let field_sources_cap = (initial_food_target + cap) * 2;
        let world_half = extent.as_array();
        let init_full = || -> Result<GpuFullPipeline, String> {
            let ctx = GpuContext::new()?;
            let cells = CellsGpu::with_context(&ctx, cap);
            cells.upload_brains(initial_cells.iter().map(|c| &c.genome.brain));
            cells.upload_xoshiro_seeds(initial_cells.iter().enumerate().map(|(slot, c)| {
                c.lineage_id ^ (slot as u64).wrapping_mul(0x9E3779B97F4A7C15)
            }));
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
            let cppn = bioscape::gpu::CppnGpu::with_context(&ctx, cap);
            Ok(GpuFullPipeline {
                cells,
                brain,
                hebbian,
                brownian,
                smell,
                pheromone,
                cell_hash,
                food_hash,
                sensor,
                populate,
                motor,
                step,
                cppn,
                scratch: bioscape::gpu::GpuFullScratch::default(),
            })
        };
        match init_full() {
            Ok(pipeline) => {
                info!(
                    "renderer-gpu-full: brain + Hebbian + Brownian + Field + SensorGather + PopulateInputs + Motor + Step (default; disable s BIOSCAPE_GPU_FULL=0; cap {} cells, {} field sources)",
                    cap, field_sources_cap
                );
                commands.insert_resource(pipeline);
            }
            Err(e) => {
                warn!("renderer-gpu-full: init failed ({}); CPU compute path active", e);
            }
        }
    } else {
        info!("renderer: CPU compute path (BIOSCAPE_GPU_FULL=0; SIMD brain + field, no GPU sync stall)");
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

    // Sprint 71: macropredator setup. Hunter mesh = větší sphere (4× CELL_RADIUS),
    // tmavě červený material — visually distinct od cells. HUNTER_TARGET_COUNT
    // hunters spawnou na náhodné pozice; constant pop, žádný respawn.
    // Sprint 88: bumped emissive na red glow s HDR > 1.0 hodnoty — Bloom catches
    // hunter jako menacing red beacon viditelný z dálky.
    // Sprint 88.4: pure-red emissive (zero green/blue). Reinhard tonemapper
    // má dokumentované „lots of hue shifting" v brights — předchozí
    // LinearRgba(2.5, 0.2, 0.1) se posouvalo směrem k oranžové. Pure red
    // (3.5, 0.0, 0.0) zůstává nezpochybnitelně červené i pod tonemap +
    // bloom redistribution. Brighter base 0.4 → 0.85 aby hunter byl viditelně
    // červený i bez bloom kontribuce (např. v post-process toggle off).
    let hunter_mesh_handle = meshes.add(Sphere::new(CELL_RADIUS * 4.0).mesh().ico(2).unwrap());
    // Sprint 91: hunter ExtendedMaterial s chitinous-scales pattern (kind=1).
    // Scale 14 = denser scales than cells; intensity 1.0.
    let hunter_material = bio_materials.add(BioMaterial {
        base: StandardMaterial {
            base_color: Color::srgb(0.85, 0.05, 0.05),
            perceptual_roughness: 0.4,
            // Sprint 91: emissive.r >= 3.5 → shader detekuje jako HUNTER pattern.
            emissive: LinearRgba::new(3.5, 0.0, 0.0, 1.0),
            ..default()
        },
        extension: BioMaterialExt {},
    });
    // Sprint 89: každý hunter dostává random genome + lineage. Initial
    // population spawnuje se tady; Sprint 89+ lifecycle (death/reproduce)
    // mění populaci dynamicky v `step_hunters`.
    let mut hunter_rng = rand::rng();
    for i in 0..HUNTER_TARGET_COUNT {
        let h = Hunter::random(&mut hunter_rng, half, i as u64, i as u64, 0);
        commands.spawn((
            HunterEntity(h),
            Mesh3d(hunter_mesh_handle.clone()),
            MeshMaterial3d(hunter_material.clone()),
            Transform::from_xyz(h.position[0], h.position[1], h.position[2]),
        ));
    }
    commands.insert_resource(HunterMesh(hunter_mesh_handle));
    commands.insert_resource(HunterMaterial(hunter_material));
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
    commands.insert_resource(WorldMapResource(world_map));
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
