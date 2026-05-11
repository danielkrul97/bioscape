use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bioscape::{
    Cell, CoopFood, Food, FoodKind, ShockEvent, ShockKind, SmellField, SPIKE_SLOTS, WORLD_HALF,
};
use rand::Rng;

use super::components::{CellEntity, FoodEntity, SpikeEntity};
use super::material::{adhesion_material, cell_rotation, cell_scale, BioMaterial};
use super::resources::{
    AdhesionMaterials, CellMesh, CellSlotMap, Clock, CoopFoodResource, EventCalendarResource,
    FoodMaterial, FoodMesh, NextCellId, PheromoneResource, SpikeMaterial, SpikeMesh, WorldExtent,
};
#[cfg(feature = "gpu")]
use super::resources_gpu::{GpuBrainState, GpuFullPipeline};

/// Click-vs-drag threshold in pixels. Below this, RMB release opens the menu;
/// above it, the user is dragging the camera.
const CLICK_DRAG_THRESHOLD_PX: f32 = 4.0;

/// Approx pixel width per menu row — used to lay out the menu so it fits on
/// screen even when the cursor is near the edge.
const MENU_WIDTH_PX: f32 = 220.0;
const MENU_ROW_HEIGHT_PX: f32 = 24.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GodAction {
    BigFood,
    FoodCluster,
    Carrion,
    SpawnCell,
    CoopFood,
    PheromoneBurst,
    HazardPulse,
}

impl GodAction {
    fn label(self) -> &'static str {
        match self {
            GodAction::BigFood => "Velky chunk jidla (50)",
            GodAction::FoodCluster => "Maly food cluster (8)",
            GodAction::Carrion => "Mrsina (12 carrion)",
            GodAction::SpawnCell => "Bunka (random genom)",
            GodAction::CoopFood => "Coop food node",
            GodAction::PheromoneBurst => "Pheromone burst",
            GodAction::HazardPulse => "Lokalni hazard pulse",
        }
    }

    fn all() -> [GodAction; 7] {
        [
            GodAction::BigFood,
            GodAction::FoodCluster,
            GodAction::Carrion,
            GodAction::SpawnCell,
            GodAction::CoopFood,
            GodAction::PheromoneBurst,
            GodAction::HazardPulse,
        ]
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum GodModeState {
    Idle,
    /// RMB pressed — track if cursor moved past threshold. While in this state
    /// the camera orbit is suppressed; once movement passes threshold we
    /// transition back to `Idle` and let camera orbit take over.
    RmbPressed {
        press_screen_pos: Vec2,
    },
    /// Menu is open at a fixed world position. The screen anchor is recorded
    /// so the menu doesn't drift when the camera orbits.
    MenuOpen {
        world_pos: Vec3,
    },
}

#[derive(Resource)]
pub(super) struct GodMode {
    pub(super) state: GodModeState,
}

impl Default for GodMode {
    fn default() -> Self {
        Self {
            state: GodModeState::Idle,
        }
    }
}

impl GodMode {
    /// Camera orbit input must skip while we're tracking a potential click and
    /// while the menu is open. Otherwise a static RMB press would orbit (delta
    /// is zero, harmless) and clicks on menu buttons would also orbit.
    pub(super) fn orbit_suppressed(&self) -> bool {
        matches!(
            self.state,
            GodModeState::RmbPressed { .. } | GodModeState::MenuOpen { .. }
        )
    }
}

#[derive(Component)]
pub(super) struct GodMenuRoot;

#[derive(Component)]
pub(super) struct GodMenuButton(pub(super) GodAction);

/// Bundle of mesh/material handles consumed by the action handler. Bundling
/// keeps the `god_mode_handle_action` parameter count under Bevy's tuple cap.
#[derive(SystemParam)]
pub(super) struct SpawnAssets<'w> {
    pub(super) cell_mesh: Res<'w, CellMesh>,
    pub(super) food_mesh: Res<'w, FoodMesh>,
    pub(super) food_material: Res<'w, FoodMaterial>,
    pub(super) spike_mesh: Res<'w, SpikeMesh>,
    pub(super) spike_material: Res<'w, SpikeMaterial>,
}

/// Mutable resource bundle for the action handler. Same reasoning as
/// `SpawnAssets`.
#[derive(SystemParam)]
pub(super) struct SpawnRegistries<'w> {
    pub(super) adhesion_materials: ResMut<'w, AdhesionMaterials>,
    pub(super) bio_materials: ResMut<'w, Assets<BioMaterial>>,
    pub(super) slot_map: ResMut<'w, CellSlotMap>,
    pub(super) next_cell_id: ResMut<'w, NextCellId>,
}

/// World-state resource bundle written to by various god-mode actions.
#[derive(SystemParam)]
pub(super) struct WorldState<'w> {
    pub(super) events: ResMut<'w, EventCalendarResource>,
    pub(super) coop: ResMut<'w, CoopFoodResource>,
    pub(super) pheromone: ResMut<'w, PheromoneResource>,
}

/// Cast cursor screen position onto the z=0 world plane.
fn cursor_to_world(
    window: &Window,
    camera: &Camera,
    camera_transform: &GlobalTransform,
) -> Option<Vec3> {
    let cursor = window.cursor_position()?;
    let ray = camera.viewport_to_world(camera_transform, cursor).ok()?;
    let plane = InfinitePlane3d::new(Vec3::Z);
    ray.plane_intersection_point(Vec3::ZERO, plane)
}

pub(super) fn god_mode_input(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    menu_roots: Query<Entity, With<GodMenuRoot>>,
    mut god: ResMut<GodMode>,
    mut commands: Commands,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let Ok((camera, camera_transform)) = cameras.single() else {
        return;
    };
    let cursor = window.cursor_position();

    // Esc always closes the menu.
    if keys.just_pressed(KeyCode::Escape) {
        if matches!(god.state, GodModeState::MenuOpen { .. }) {
            despawn_menu(&menu_roots, &mut commands);
            god.state = GodModeState::Idle;
            return;
        }
    }

    match god.state {
        GodModeState::Idle => {
            if buttons.just_pressed(MouseButton::Right) {
                if let Some(c) = cursor {
                    god.state = GodModeState::RmbPressed { press_screen_pos: c };
                }
            }
        }
        GodModeState::RmbPressed { press_screen_pos } => {
            if buttons.just_released(MouseButton::Right) {
                // Released without enough movement → open menu at the cursor.
                let release = cursor.unwrap_or(press_screen_pos);
                let moved = release.distance(press_screen_pos);
                if moved <= CLICK_DRAG_THRESHOLD_PX {
                    if let Some(world_pos) =
                        cursor_to_world(window, camera, camera_transform)
                    {
                        spawn_menu(window.size(), release, &mut commands);
                        god.state = GodModeState::MenuOpen { world_pos };
                        return;
                    }
                }
                // Either moved too much (drag) or no world hit — fall back to Idle.
                god.state = GodModeState::Idle;
            } else if let Some(c) = cursor {
                if c.distance(press_screen_pos) > CLICK_DRAG_THRESHOLD_PX {
                    // User is dragging — release control to camera orbit.
                    god.state = GodModeState::Idle;
                }
            }
        }
        GodModeState::MenuOpen { .. } => {
            // Outside-click handling lives in `close_menu_on_outside_click`,
            // which runs after the action handler — that ordering lets a real
            // button press execute before we decide to close.
        }
    }
}

/// Handles button presses inside the menu. Runs *before* `god_mode_input` so
/// Interaction::Pressed is consumed first; when the menu is open and the user
/// clicks outside (no Pressed interaction), the next-tick `god_mode_input`
/// sees the LMB-just-pressed and closes via `close_menu_on_outside_click`.
pub(super) fn god_mode_handle_action(
    interactions: Query<(&Interaction, &GodMenuButton), Changed<Interaction>>,
    menu_roots: Query<Entity, With<GodMenuRoot>>,
    extent: Res<WorldExtent>,
    clock: Res<Clock>,
    assets: SpawnAssets,
    mut god: ResMut<GodMode>,
    mut registries: SpawnRegistries,
    mut world_state: WorldState,
    #[cfg(feature = "gpu")] gpu_state: Option<Res<GpuBrainState>>,
    #[cfg(feature = "gpu")] gpu_full: Option<ResMut<GpuFullPipeline>>,
    mut commands: Commands,
) {
    let GodModeState::MenuOpen { world_pos } = god.state else {
        return;
    };
    let mut chosen: Option<GodAction> = None;
    for (interaction, button) in &interactions {
        if matches!(interaction, Interaction::Pressed) {
            chosen = Some(button.0);
            break;
        }
    }
    let Some(action) = chosen else {
        return;
    };

    let half = extent.as_array();
    let mut rng = rand::rng();
    match action {
        GodAction::BigFood => {
            spawn_food_cluster(
                &mut commands,
                &assets.food_mesh,
                &assets.food_material,
                world_pos,
                50,
                30.0,
                FoodKind::Plant,
                half,
                &mut rng,
            );
        }
        GodAction::FoodCluster => {
            spawn_food_cluster(
                &mut commands,
                &assets.food_mesh,
                &assets.food_material,
                world_pos,
                8,
                12.0,
                FoodKind::Plant,
                half,
                &mut rng,
            );
        }
        GodAction::Carrion => {
            spawn_food_cluster(
                &mut commands,
                &assets.food_mesh,
                &assets.food_material,
                world_pos,
                12,
                15.0,
                FoodKind::Carrion,
                half,
                &mut rng,
            );
        }
        GodAction::SpawnCell => {
            let cell_id = registries.next_cell_id.0;
            registries.next_cell_id.0 += 1;
            // Brand new lineage — distinct from any existing one.
            let lineage_id = cell_id;
            let mut cell = Cell::random(
                &mut rng,
                half,
                lineage_id,
                clock.0.generation,
                cell_id,
            );
            cell.position = [world_pos.x, world_pos.y, world_pos.z];
            let mat = adhesion_material(
                &mut registries.adhesion_materials,
                &mut registries.bio_materials,
                cell.genome.adhesion_type,
            );
            let cell_id_for_seed = cell.cell_id;
            let turn_rate = cell.genome.turn_rate;
            #[cfg(feature = "gpu")]
            let cppn_copy = cell.genome.cppn;
            let entity = commands
                .spawn((
                    CellEntity(cell),
                    Mesh3d(assets.cell_mesh.0.clone()),
                    MeshMaterial3d(mat),
                    Transform::from_xyz(cell.position[0], cell.position[1], cell.position[2])
                        .with_rotation(cell_rotation(cell.heading, cell.pitch))
                        .with_scale(cell_scale(&cell.phenotype)),
                ))
                .id();
            for slot in 0..SPIKE_SLOTS as u8 {
                commands.spawn((
                    SpikeEntity { owner: entity, slot },
                    Mesh3d(assets.spike_mesh.0.clone()),
                    MeshMaterial3d(assets.spike_material.0.clone()),
                    Transform::default(),
                    Visibility::Hidden,
                ));
            }
            let slot = registries.slot_map.allocate(entity);
            #[cfg(feature = "gpu")]
            if let Some(gpu) = gpu_state.as_ref() {
                gpu.cells.upload_brain_at(slot, &cell.genome.brain);
                // V7-unification: seed from `cell_id` (matches CPU
                // `Cell.xoshiro_state`), not a slot-hashed lineage id.
                gpu.cells.upload_xoshiro_seed_at(slot, cell_id_for_seed);
            }
            #[cfg(feature = "gpu")]
            if let Some(mut gpu) = gpu_full {
                gpu.cells.upload_xoshiro_seed_at(slot, cell_id_for_seed);
                gpu.cells.upload_turn_rate_at(slot, turn_rate);
                let pipeline = &mut *gpu;
                pipeline
                    .cppn
                    .dispatch(&[(slot, &cppn_copy)], &pipeline.cells);
            }
            #[cfg(not(feature = "gpu"))]
            let _ = (cell_id_for_seed, turn_rate);
            let _ = slot;
        }
        GodAction::CoopFood => {
            let pos = [world_pos.x, world_pos.y, world_pos.z];
            world_state.coop.0.push(CoopFood::new(pos, clock.0.tick));
        }
        GodAction::PheromoneBurst => {
            // Channel 0 = mating-friendly slow channel. Strong injection so it
            // shows up in the field even after one decay tick.
            let pos = [world_pos.x, world_pos.y, world_pos.z];
            // Splash a 3×3×1 stencil for visibility; SmellField smooths it
            // out via diffusion next tick.
            for dx in -1..=1 {
                for dy in -1..=1 {
                    let p = [pos[0] + dx as f32 * 6.0, pos[1] + dy as f32 * 6.0, pos[2]];
                    SmellField::add_source(
                        &mut world_state.pheromone.fields[0],
                        p,
                        8.0,
                    );
                }
            }
        }
        GodAction::HazardPulse => {
            let radius = 80.0_f32.min(WORLD_HALF[0] * 0.5);
            world_state.events.0.events.push(ShockEvent {
                kind: ShockKind::HazardPulse,
                start_gen: clock.0.generation,
                duration_gen: 10,
                ramp_gens: 2,
                intensity: 1.0,
                center_xy: Some([world_pos.x, world_pos.y]),
                radius: Some(radius),
            });
            // Re-sort so `active()` order stays stable.
            world_state.events.0.events.sort_by_key(|e| e.start_gen);
        }
    }

    despawn_menu(&menu_roots, &mut commands);
    god.state = GodModeState::Idle;
}

/// Closes the menu on a click outside any menu button. Runs after
/// `god_mode_handle_action` — by then any clicked button has been consumed
/// and the menu is already despawned. So this only fires when the click
/// missed every button.
pub(super) fn close_menu_on_outside_click(
    buttons: Res<ButtonInput<MouseButton>>,
    interactions: Query<&Interaction, With<GodMenuButton>>,
    menu_roots: Query<Entity, With<GodMenuRoot>>,
    mut god: ResMut<GodMode>,
    mut commands: Commands,
) {
    if !matches!(god.state, GodModeState::MenuOpen { .. }) {
        return;
    }
    if !buttons.just_pressed(MouseButton::Left)
        && !buttons.just_pressed(MouseButton::Right)
    {
        return;
    }
    // If any button is currently in Pressed/Hovered state, the press is on
    // the menu — let `god_mode_handle_action` deal with it.
    let any_inside = interactions
        .iter()
        .any(|i| matches!(i, Interaction::Pressed | Interaction::Hovered));
    if any_inside {
        return;
    }
    despawn_menu(&menu_roots, &mut commands);
    god.state = GodModeState::Idle;
}

fn despawn_menu(menu_roots: &Query<Entity, With<GodMenuRoot>>, commands: &mut Commands) {
    for entity in menu_roots {
        commands.entity(entity).despawn();
    }
}

fn spawn_menu(window_size: Vec2, screen_pos: Vec2, commands: &mut Commands) {
    let actions = GodAction::all();
    let total_h = MENU_ROW_HEIGHT_PX * actions.len() as f32 + 8.0;
    // Clamp so the menu doesn't overflow the window.
    let left = screen_pos
        .x
        .min(window_size.x - MENU_WIDTH_PX - 4.0)
        .max(4.0);
    let top = screen_pos.y.min(window_size.y - total_h - 4.0).max(4.0);

    commands
        .spawn((
            GodMenuRoot,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(left),
                top: Val::Px(top),
                width: Val::Px(MENU_WIDTH_PX),
                padding: UiRect::all(Val::Px(4.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(2.0),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                ..default()
            },
            BorderColor::all(Color::srgba(0.8, 0.8, 0.85, 0.9)),
            BackgroundColor(Color::srgba(0.06, 0.07, 0.10, 0.92)),
            ZIndex(1000),
        ))
        .with_children(|parent| {
            for action in actions {
                parent
                    .spawn((
                        Button,
                        GodMenuButton(action),
                        Node {
                            width: Val::Percent(100.0),
                            height: Val::Px(MENU_ROW_HEIGHT_PX),
                            padding: UiRect::horizontal(Val::Px(6.0)),
                            justify_content: JustifyContent::FlexStart,
                            align_items: AlignItems::Center,
                            border_radius: BorderRadius::all(Val::Px(2.0)),
                            ..default()
                        },
                        BackgroundColor(Color::srgba(0.10, 0.12, 0.16, 1.0)),
                    ))
                    .with_children(|btn| {
                        btn.spawn((
                            Text::new(action.label()),
                            TextColor(Color::srgb(0.92, 0.92, 0.95)),
                            TextFont {
                                font_size: 12.0,
                                ..default()
                            },
                        ));
                    });
            }
        });
}

/// Highlights the hovered button. Cheap visual feedback.
pub(super) fn god_mode_button_hover(
    mut buttons: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<GodMenuButton>),
    >,
) {
    for (interaction, mut bg) in &mut buttons {
        *bg = match interaction {
            Interaction::Pressed => BackgroundColor(Color::srgba(0.30, 0.45, 0.60, 1.0)),
            Interaction::Hovered => BackgroundColor(Color::srgba(0.20, 0.28, 0.38, 1.0)),
            Interaction::None => BackgroundColor(Color::srgba(0.10, 0.12, 0.16, 1.0)),
        };
    }
}

fn spawn_food_cluster(
    commands: &mut Commands,
    food_mesh: &FoodMesh,
    food_material: &FoodMaterial,
    center: Vec3,
    count: usize,
    radius: f32,
    kind: FoodKind,
    half: [f32; 3],
    rng: &mut impl Rng,
) {
    for _ in 0..count {
        let dx = rng.random_range(-radius..radius);
        let dy = rng.random_range(-radius..radius);
        let dz = rng.random_range(-2.0..2.0);
        let pos = [
            (center.x + dx).clamp(-half[0], half[0]),
            (center.y + dy).clamp(-half[1], half[1]),
            (center.z + dz).clamp(-half[2], half[2]),
        ];
        let food = Food {
            position: pos,
            age_ticks: 0,
            kind,
        };
        commands.spawn((
            FoodEntity(food),
            Mesh3d(food_mesh.0.clone()),
            MeshMaterial3d(food_material.0.clone()),
            Transform::from_xyz(pos[0], pos[1], pos[2]),
            Visibility::Hidden,
        ));
    }
}
