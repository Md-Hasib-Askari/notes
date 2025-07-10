# Game Development with Bevy

## Overview
Learn game development using Bevy, a modern data-driven game engine built in Rust with Entity Component System (ECS) architecture.

## Bevy Fundamentals

### Basic Game Setup
```rust
use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, (movement_system, collision_system))
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 6.0, 12.0)
            .looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
        ..default()
    });
    
    // Light
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_euler(
            EulerRot::ZYX, 0.0, 1.0, -std::f32::consts::FRAC_PI_4,
        )),
        ..default()
    });
    
    // Ground plane
    commands.spawn(PbrBundle {
        mesh: meshes.add(Plane3d::default().mesh().size(20.0, 20.0)),
        material: materials.add(Color::rgb(0.3, 0.5, 0.3)),
        ..default()
    });
}
```

### Component System
```rust
// Components define data
#[derive(Component)]
struct Player {
    speed: f32,
    health: f32,
}

#[derive(Component)]
struct Velocity {
    linear: Vec3,
    angular: f32,
}

#[derive(Component)]
struct Enemy {
    damage: f32,
    detection_range: f32,
}

#[derive(Component)]
struct Health {
    current: f32,
    maximum: f32,
}

#[derive(Component)]
struct Weapon {
    damage: f32,
    range: f32,
    cooldown: Timer,
}

// Marker components
#[derive(Component)]
struct Bullet;

#[derive(Component)]
struct Collectible;

// Systems operate on components
fn movement_system(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut query: Query<(&mut Transform, &Player)>,
) {
    for (mut transform, player) in &mut query {
        let mut direction = Vec3::ZERO;
        
        if keyboard.pressed(KeyCode::KeyW) {
            direction.z -= 1.0;
        }
        if keyboard.pressed(KeyCode::KeyS) {
            direction.z += 1.0;
        }
        if keyboard.pressed(KeyCode::KeyA) {
            direction.x -= 1.0;
        }
        if keyboard.pressed(KeyCode::KeyD) {
            direction.x += 1.0;
        }
        
        if direction != Vec3::ZERO {
            direction = direction.normalize();
            transform.translation += direction * player.speed * time.delta_seconds();
        }
    }
}

fn velocity_system(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &Velocity)>,
) {
    for (mut transform, velocity) in &mut query {
        transform.translation += velocity.linear * time.delta_seconds();
        transform.rotate_y(velocity.angular * time.delta_seconds());
    }
}
```

### Game State Management
```rust
#[derive(States, Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
enum GameState {
    #[default]
    MainMenu,
    Playing,
    Paused,
    GameOver,
}

#[derive(Resource)]
struct GameData {
    score: u32,
    level: u32,
    lives: u32,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .init_state::<GameState>()
        .insert_resource(GameData {
            score: 0,
            level: 1,
            lives: 3,
        })
        .add_systems(Startup, setup)
        .add_systems(Update, menu_system.run_if(in_state(GameState::MainMenu)))
        .add_systems(Update, 
            (player_input, enemy_ai, collision_system)
                .run_if(in_state(GameState::Playing)))
        .add_systems(Update, pause_system.run_if(in_state(GameState::Paused)))
        .run();
}

fn menu_system(
    mut next_state: ResMut<NextState<GameState>>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        next_state.set(GameState::Playing);
    }
}

fn pause_system(
    mut next_state: ResMut<NextState<GameState>>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    if keyboard.just_pressed(KeyCode::Escape) {
        next_state.set(GameState::Playing);
    }
}

fn player_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    if keyboard.just_pressed(KeyCode::Escape) {
        next_state.set(GameState::Paused);
    }
}
```

### Physics and Collision Detection
```rust
use bevy_rapier3d::prelude::*;

fn setup_physics(mut commands: Commands) {
    // Ground
    commands.spawn((
        RigidBody::Fixed,
        Collider::cuboid(10.0, 0.1, 10.0),
        Transform::from_xyz(0.0, -0.1, 0.0),
    ));
    
    // Dynamic object
    commands.spawn((
        RigidBody::Dynamic,
        Collider::ball(0.5),
        Transform::from_xyz(0.0, 4.0, 0.0),
        Velocity::linear(Vec3::new(1.0, 0.0, 0.0)),
    ));
}

// Custom collision detection
fn collision_system(
    mut commands: Commands,
    bullet_query: Query<(Entity, &Transform), With<Bullet>>,
    enemy_query: Query<(Entity, &Transform, &mut Health), (With<Enemy>, Without<Bullet>)>,
    mut game_data: ResMut<GameData>,
) {
    for (bullet_entity, bullet_transform) in &bullet_query {
        for (enemy_entity, enemy_transform, mut health) in &enemy_query {
            let distance = bullet_transform.translation.distance(enemy_transform.translation);
            
            if distance < 1.0 { // Hit detected
                // Damage enemy
                health.current -= 20.0;
                
                // Remove bullet
                commands.entity(bullet_entity).despawn();
                
                // Check if enemy is dead
                if health.current <= 0.0 {
                    commands.entity(enemy_entity).despawn();
                    game_data.score += 100;
                }
            }
        }
    }
}
```

### Audio System
```rust
#[derive(Resource)]
struct GameAudio {
    background_music: Handle<AudioSource>,
    shoot_sound: Handle<AudioSource>,
    hit_sound: Handle<AudioSource>,
}

fn setup_audio(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    let game_audio = GameAudio {
        background_music: asset_server.load("sounds/background.ogg"),
        shoot_sound: asset_server.load("sounds/shoot.wav"),
        hit_sound: asset_server.load("sounds/hit.wav"),
    };
    
    commands.insert_resource(game_audio);
}

fn play_background_music(
    mut commands: Commands,
    game_audio: Res<GameAudio>,
) {
    commands.spawn(AudioBundle {
        source: game_audio.background_music.clone(),
        settings: PlaybackSettings::LOOP,
    });
}

fn shooting_system(
    mut commands: Commands,
    keyboard: Res<ButtonInput<KeyCode>>,
    game_audio: Res<GameAudio>,
    player_query: Query<&Transform, With<Player>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        for player_transform in &player_query {
            // Play sound
            commands.spawn(AudioBundle {
                source: game_audio.shoot_sound.clone(),
                ..default()
            });
            
            // Spawn bullet
            commands.spawn((
                PbrBundle {
                    mesh: meshes.add(Sphere::new(0.1)),
                    material: materials.add(Color::YELLOW),
                    transform: *player_transform,
                    ..default()
                },
                Bullet,
                Velocity {
                    linear: player_transform.forward() * 10.0,
                    angular: 0.0,
                },
            ));
        }
    }
}
```

### AI and Behavior Trees
```rust
#[derive(Component)]
struct EnemyAI {
    state: AIState,
    target: Option<Entity>,
    patrol_points: Vec<Vec3>,
    current_patrol: usize,
}

#[derive(Debug)]
enum AIState {
    Patrolling,
    Chasing,
    Attacking,
    Searching,
}

fn enemy_ai_system(
    time: Res<Time>,
    mut enemy_query: Query<(&mut Transform, &mut EnemyAI, &Enemy)>,
    player_query: Query<&Transform, (With<Player>, Without<Enemy>)>,
) {
    for (mut enemy_transform, mut ai, enemy) in &mut enemy_query {
        match ai.state {
            AIState::Patrolling => {
                // Move to patrol point
                let target_point = ai.patrol_points[ai.current_patrol];
                let direction = (target_point - enemy_transform.translation).normalize();
                enemy_transform.translation += direction * 2.0 * time.delta_seconds();
                
                // Check if reached patrol point
                if enemy_transform.translation.distance(target_point) < 1.0 {
                    ai.current_patrol = (ai.current_patrol + 1) % ai.patrol_points.len();
                }
                
                // Check for player
                if let Ok(player_transform) = player_query.get_single() {
                    let distance = enemy_transform.translation.distance(player_transform.translation);
                    if distance < enemy.detection_range {
                        ai.state = AIState::Chasing;
                    }
                }
            }
            
            AIState::Chasing => {
                if let Ok(player_transform) = player_query.get_single() {
                    let direction = (player_transform.translation - enemy_transform.translation).normalize();
                    enemy_transform.translation += direction * 4.0 * time.delta_seconds();
                    
                    let distance = enemy_transform.translation.distance(player_transform.translation);
                    if distance < 2.0 {
                        ai.state = AIState::Attacking;
                    } else if distance > enemy.detection_range * 1.5 {
                        ai.state = AIState::Searching;
                    }
                }
            }
            
            AIState::Attacking => {
                // Attack behavior
                // TODO: Implement attack logic
                ai.state = AIState::Chasing;
            }
            
            AIState::Searching => {
                // Search behavior - return to patrol after timeout
                ai.state = AIState::Patrolling;
            }
        }
    }
}
```

### UI System
```rust
fn setup_ui(mut commands: Commands) {
    // Root UI node
    commands.spawn(NodeBundle {
        style: Style {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            ..default()
        },
        ..default()
    }).with_children(|parent| {
        // HUD
        parent.spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.0),
                height: Val::Px(50.0),
                align_items: AlignItems::Center,
                padding: UiRect::all(Val::Px(10.0)),
                ..default()
            },
            background_color: Color::rgba(0.0, 0.0, 0.0, 0.8).into(),
            ..default()
        }).with_children(|parent| {
            // Score text
            parent.spawn((
                TextBundle::from_section(
                    "Score: 0",
                    TextStyle {
                        font_size: 24.0,
                        color: Color::WHITE,
                        ..default()
                    },
                ),
                ScoreText,
            ));
            
            // Health bar
            parent.spawn(NodeBundle {
                style: Style {
                    width: Val::Px(200.0),
                    height: Val::Px(20.0),
                    margin: UiRect::all(Val::Px(10.0)),
                    ..default()
                },
                background_color: Color::RED.into(),
                ..default()
            }).with_children(|parent| {
                parent.spawn((
                    NodeBundle {
                        style: Style {
                            width: Val::Percent(100.0),
                            height: Val::Percent(100.0),
                            ..default()
                        },
                        background_color: Color::GREEN.into(),
                        ..default()
                    },
                    HealthBar,
                ));
            });
        });
    });
}

#[derive(Component)]
struct ScoreText;

#[derive(Component)]
struct HealthBar;

fn update_ui_system(
    game_data: Res<GameData>,
    mut score_query: Query<&mut Text, With<ScoreText>>,
    mut health_query: Query<&mut Style, With<HealthBar>>,
    player_query: Query<&Health, With<Player>>,
) {
    // Update score
    for mut text in &mut score_query {
        text.sections[0].value = format!("Score: {}", game_data.score);
    }
    
    // Update health bar
    if let Ok(health) = player_query.get_single() {
        for mut style in &mut health_query {
            let health_percent = health.current / health.maximum;
            style.width = Val::Percent(health_percent * 100.0);
        }
    }
}
```

### Asset Management
```rust
#[derive(Resource)]
struct GameAssets {
    player_mesh: Handle<Mesh>,
    enemy_mesh: Handle<Mesh>,
    bullet_mesh: Handle<Mesh>,
    player_material: Handle<StandardMaterial>,
    enemy_material: Handle<StandardMaterial>,
}

fn load_assets(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let game_assets = GameAssets {
        player_mesh: meshes.add(Capsule3d::new(0.5, 1.0)),
        enemy_mesh: meshes.add(Cuboid::new(1.0, 1.0, 1.0)),
        bullet_mesh: meshes.add(Sphere::new(0.1)),
        player_material: materials.add(Color::BLUE),
        enemy_material: materials.add(Color::RED),
    };
    
    commands.insert_resource(game_assets);
}

// Async asset loading
#[derive(Resource)]
struct LoadingAssets {
    handles: Vec<UntypedHandle>,
}

fn check_loading_system(
    mut commands: Commands,
    loading: Option<Res<LoadingAssets>>,
    asset_server: Res<AssetServer>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    if let Some(loading) = loading {
        let all_loaded = loading.handles.iter()
            .all(|handle| asset_server.is_loaded_with_dependencies(handle));
        
        if all_loaded {
            commands.remove_resource::<LoadingAssets>();
            next_state.set(GameState::Playing);
        }
    }
}
```

## Performance Optimization

### System Scheduling
```rust
fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(
            Update,
            (
                // Run input systems first
                player_input_system,
                // Then physics
                (movement_system, physics_system).chain(),
                // Then game logic
                (enemy_ai_system, weapon_system).after(physics_system),
                // Finally rendering updates
                (animation_system, ui_update_system).after(enemy_ai_system),
            )
        )
        .run();
}

// Parallel system execution
fn parallel_systems() -> SystemSet {
    (
        enemy_ai_system,
        particle_system,
        audio_system,
    ).into()
}
```

## Key Learning Points

1. **ECS Architecture**: Entities, Components, and Systems design pattern
2. **Game Loop**: Update and render cycles with delta time
3. **State Management**: Game states and transitions
4. **Asset Pipeline**: Loading and managing game assets
5. **Physics Integration**: Collision detection and response
6. **Audio Systems**: Sound effects and music management
7. **UI Development**: In-game interfaces and HUD

## Popular Game Development Crates
- `bevy` - Modern game engine
- `ggez` - Lightweight 2D game framework  
- `piston` - Modular game engine
- `amethyst` - Data-driven game engine
- `macroquad` - Simple cross-platform game library
- `wgpu` - Low-level graphics API
- `winit` - Window creation and event handling

## Project Ideas
1. **Pong Clone**: Basic game mechanics and collision
2. **Platformer**: Character movement and level design
3. **Tower Defense**: AI, pathfinding, and resource management
4. **3D Adventure**: Camera control, physics, and exploration
5. **Multiplayer Game**: Networking and state synchronization
