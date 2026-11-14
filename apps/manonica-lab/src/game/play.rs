use super::{Action, Input, effects::SpellView, hud::Hud, lock, mesh::TerrainView, world};
use manonica_world::{
    gameplay::{Character, WalkInput, raycast},
    magic::SpellRegistry,
    runtime::{EntityId, EntityTransform, GameRuntime},
};
use meridian_sdk::{
    Device, FlyCamera, FlyControls, GraphicsSession, Scene3d, Vec3, Window, compose_frame,
};
use std::{
    io::{BufWriter, Write},
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
fn save_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../saves/game.nbt")
}
fn save(game: &GameRuntime) -> Result<(), Box<dyn std::error::Error>> {
    let path = save_path();
    std::fs::create_dir_all(path.parent().unwrap())?;
    let bytes = game.to_nbt()?;
    let temporary = path.with_extension("nbt.tmp");
    {
        use std::io::Write;
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
    }
    std::fs::rename(temporary, path)?;
    Ok(())
}
fn load() -> Result<(GameRuntime, EntityId), Box<dyn std::error::Error>> {
    let path = save_path();
    if std::fs::metadata(&path)?.len() > 64 * 1024 * 1024 {
        return Err("save exceeds limit".into());
    }
    let game = GameRuntime::from_nbt(world::content()?, &std::fs::read(path)?)?;
    let id = game
        .entities()
        .iter()
        .find_map(|(id, e)| {
            game.world
                .content()
                .entity(e.kind)
                .ok()
                .filter(|d| d.key.to_string() == "manonica:player")
                .map(|_| *id)
        })
        .ok_or("save has no player")?;
    Ok((game, id))
}
pub(super) fn run(
    window: Window,
    initial: (GameRuntime, EntityId),
    input: Arc<Mutex<Input>>,
    running: Arc<AtomicBool>,
    smoke_frames: Option<u64>,
    showcase: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut trace = std::env::var_os("MANONICA_FRAME_TRACE").map(|path| {
        let mut writer = BufWriter::new(std::fs::File::create_new(path)?);
        writeln!(writer,"frame,interval_ms,render_ms,input_ms,update_ms,terrain_ms,effects_ms,overlays_ms,draw_present_ms,bodies,programs")?;
        Ok::<_,std::io::Error>(writer)
    }).transpose()?;
    let runtime = tokio::runtime::Runtime::new()?;
    let (device, mut surface) =
        runtime.block_on(Device::new_windowed(window.surface_target(), 1280, 800))?;
    surface.enable_vsync(&device);
    let mut base = GraphicsSession::new(device.clone(), surface, 1280, 800);
    let (mut game, mut player) = initial;
    let spells = SpellRegistry::from_jsonl(
        include_str!("../../../../assets/demo/spells.jsonl"),
        game.world.content(),
    )?;
    let mut effects =
        SpellView::new(&device, &runtime, &mut base).map_err(std::io::Error::other)?;
    let mut terrain = TerrainView::new();
    terrain
        .rebuild(&mut base, &game.world)
        .map_err(std::io::Error::other)?;
    let mut hud = Hud::new(&mut base);
    let mut character = Character::new(world::SPAWN);
    let mut camera = FlyCamera::new(Vec3::new(0., 10., 13.));
    let mut previous = Instant::now();
    let mut accumulator = 0.0;
    let mut physics_accumulator = 0.0;
    let mut selected = 0;
    let mut status =
        "Explore the arch and crystal pillars. Scrolls are in slots 4, 5, 6.".to_string();
    let mut size = [1280, 800];
    let mut frame_count = 0;
    let mut hottest = 300_f64;
    let mut most_bodies = 0;
    let mut pending_jump = false;
    let mut diagnostics = false;
    let mut terrain_revision = game.world.geometry_revision();
    if showcase {
        for (slot, target) in [
            (3, [-11.0, 4.5, -3.0]),
            (4, [0.0, 4.5, -10.0]),
            (5, [11.0, 4.5, -3.0]),
        ] {
            game.cast_scroll(
                &spells,
                player,
                slot,
                [0.5, 5.5, 8.0],
                target,
                [0., 0., -1.],
            )?;
        }
        terrain
            .rebuild(&mut base, &game.world)
            .map_err(std::io::Error::other)?;
    }
    println!(
        "3D ready: {} chunks, generic manon-field observer on {}",
        game.world.statistics().chunks.total,
        device.adapter_name()
    );
    while running.load(Ordering::Acquire) {
        let now = Instant::now();
        let interval_ms = now.duration_since(previous).as_secs_f64() * 1000.;
        let dt = (interval_ms / 1000.).min(0.05);
        previous = now;
        let (controls, actions, viewport, locked) = {
            let mut state = lock(&input);
            let controls = state.controls;
            state.controls.look = (0., 0.);
            (
                controls,
                state.actions.drain(..).collect::<Vec<_>>(),
                state.size,
                state.locked,
            )
        };
        if viewport.contains(&0) {
            std::thread::sleep(Duration::from_millis(30));
            continue;
        }
        if viewport != size {
            base.resize(viewport[0], viewport[1]);
            size = viewport;
        }
        camera.update(
            FlyControls {
                look: controls.look,
                ..Default::default()
            },
            0.0,
        );
        let direction = [
            f64::from(camera.yaw.cos() * camera.pitch.cos()),
            f64::from(camera.pitch.sin()),
            f64::from(camera.yaw.sin() * camera.pitch.cos()),
        ];
        let mut dirty = false;

        for action in actions {
            match action {
                Action::Jump => pending_jump = true,
                Action::Diagnostics => {
                    diagnostics = !diagnostics;
                    effects.set_diagnostics(diagnostics);
                }
                Action::Select(slot) => selected = slot,
                Action::Respawn => {
                    character = Character::new(world::SPAWN);
                    status = "Respawned".into();
                }
                Action::Save => {
                    game.set_transform(
                        player,
                        EntityTransform {
                            position_metres: character.feet,
                            rotation_radians: [f64::from(camera.pitch), f64::from(camera.yaw), 0.0],
                        },
                    )?;
                    status = match save(&game) {
                        Ok(()) => "Saved: saves/game.nbt".into(),
                        Err(e) => format!("SAVE: {e}"),
                    };
                }
                Action::Load => match load() {
                    Ok((loaded, id)) => {
                        let transform = loaded.entity(id)?.transform;
                        let restored = Character::new(transform.position_metres);
                        if restored.blocked(&loaded.world) {
                            status = "LOAD: player position is blocked".into();
                        } else {
                            game = loaded;
                            effects.invalidate();
                            terrain.invalidate();
                            player = id;
                            character = restored;
                            camera.pitch = transform.rotation_radians[0] as f32;
                            camera.yaw = transform.rotation_radians[1] as f32;
                            dirty = true;
                            status = "World loaded".into();
                        }
                    }
                    Err(e) => status = format!("LOAD: {e}"),
                },
                Action::Break => {
                    if let Some(hit) = raycast(&game.world, character.eye(), direction, 6.0) {
                        let mut result = Err(manonica_world::WorldError::Invalid("inventory_full"));
                        for slot in (0..16).filter(|s| *s < 3 || *s >= 6) {
                            result = game.break_block(player, slot, hit.block);
                            if result.is_ok() {
                                break;
                            }
                        }
                        match result {
                            Ok(()) => {
                                dirty = true;
                                status = "Block collected".into();
                            }
                            Err(e) => status = e.to_string(),
                        }
                    }
                }
                Action::Use => {
                    let is_scroll =
                        game.entity(player)?
                            .inventory
                            .get(selected)?
                            .is_some_and(|s| {
                                game.world
                                    .content()
                                    .item(s.item())
                                    .is_ok_and(|d| spells.for_scroll(&d.key).is_ok())
                            });
                    if is_scroll {
                        let target = raycast(&game.world, character.eye(), direction, 18.0)
                            .map(|h| {
                                [
                                    h.block.x as f64 + 0.5,
                                    h.block.y as f64 + 1.02,
                                    h.block.z as f64 + 0.5,
                                ]
                            })
                            .unwrap_or_else(|| {
                                std::array::from_fn(|i| character.eye()[i] + direction[i] * 10.0)
                            });
                        match game.cast_scroll(
                            &spells,
                            player,
                            selected,
                            character.eye(),
                            target,
                            direction,
                        ) {
                            Ok(event) => {
                                status = format!(
                                    "{}: {} field cells, {:.0} manons",
                                    event.spell,
                                    event.field.densities.len(),
                                    event.field.manons()
                                );
                            }
                            Err(e) => status = e.to_string(),
                        }
                    } else if let Some(hit) = raycast(&game.world, character.eye(), direction, 6.0)
                    {
                        if character.intersects(hit.adjacent) {
                            status = "Cannot place inside player".into();
                        } else {
                            match game.place(player, selected, hit.adjacent) {
                                Ok(()) => {
                                    dirty = true;
                                    status = "Block placed".into();
                                }
                                Err(e) => status = e.to_string(),
                            }
                        }
                    }
                }
            }
        }
        game.set_transform(
            player,
            EntityTransform {
                position_metres: character.feet,
                rotation_radians: [f64::from(camera.pitch), f64::from(camera.yaw), 0.0],
            },
        )?;
        let input_ms = now.elapsed().as_secs_f64() * 1000.;
        let update_start = Instant::now();
        if locked || showcase {
            accumulator = (accumulator + dt).min(0.1);
            while accumulator >= 1.0 / 120.0 {
                let acceleration = [0.0; 3];
                character.step(
                    &game.world,
                    WalkInput {
                        forward: f64::from(controls.forward as u8)
                            - f64::from(controls.backward as u8),
                        strafe: f64::from(controls.right as u8) - f64::from(controls.left as u8),
                        yaw: f64::from(camera.yaw),
                        jump: pending_jump,
                        sprint: controls.boost,
                    },
                    1.0 / 120.0,
                    acceleration,
                );
                pending_jump = false;
                accumulator -= 1.0 / 120.0;
            }
            // The field already has a 40 Hz control clock. Couple material once
            // per control step; contact relaxation is analytic, motion uses CFL substeps.
            physics_accumulator = (physics_accumulator + dt).min(0.1);
            while physics_accumulator + 1e-12 >= 0.025 {
                if let Err(error) = game.advance_magic(0.025) {
                    if smoke_frames.is_some() {
                        return Err(error.into());
                    }
                    status = format!("FIELD PAUSED: {error}");
                }
                physics_accumulator = (physics_accumulator - 0.025).max(0.0);
            }
        } else {
            accumulator = 0.0;
            physics_accumulator = 0.0;
            pending_jump = false;
        }

        let update_ms = update_start.elapsed().as_secs_f64() * 1000.;
        let terrain_start = Instant::now();
        dirty |= terrain_revision != game.world.geometry_revision();
        if dirty {
            terrain_revision = game.world.geometry_revision();
            terrain
                .rebuild(&mut base, &game.world)
                .map_err(std::io::Error::other)?;
        }
        let terrain_ms = terrain_start.elapsed().as_secs_f64() * 1000.;
        let effects_start = Instant::now();
        let eye = character.eye();
        camera.position = Vec3::new(eye[0] as f32, eye[1] as f32, eye[2] as f32);
        let mut scene = Scene3d {
            camera: camera.camera(size[0] as f32 / size[1] as f32),
            renderables: terrain.renderables.clone(),
            ambient_ground: [0.25, 0.3, 0.35],
            ambient_sky: [0.65, 0.75, 0.85],
            ..Default::default()
        };
        effects
            .update(
                &mut base,
                game.magic(),
                &game.world,
                game.matter(),
                game.physics().thermal,
            )
            .map_err(std::io::Error::other)?;
        effects.append(&mut scene, game.magic());
        effects.append_matter(
            &mut base,
            &mut scene,
            game.matter(),
            &game.world,
            game.physics(),
            physics_accumulator,
        );
        let effects_ms = effects_start.elapsed().as_secs_f64() * 1000.;
        let overlays_start = Instant::now();
        let stack = game.entity(player)?.inventory.get(selected)?;
        let item = stack
            .map(|s| {
                format!(
                    "{} x{}",
                    game.world.content().item(s.item()).unwrap().display.name,
                    s.count()
                )
            })
            .unwrap_or_else(|| "EMPTY".into());
        let target = raycast(&game.world, eye, direction, 18.0)
            .and_then(|h| game.world.block(h.block).ok())
            .and_then(|s| game.world.content().block(s.block()).ok())
            .map(|b| b.display.name.as_str())
            .unwrap_or("sky");
        let peak_temperature = game.peak_temperature_kelvin()?;
        hottest = hottest.max(peak_temperature);
        most_bodies = most_bodies.max(game.matter().bodies.len());
        let overlay = hud.draw(
            &mut base,
            size,
            &[
                "MANONICA / ARCANE FRONTIER".into(),
                "WASD walk | SHIFT sprint | SPACE jump | ESC pause/cursor | R respawn".into(),
                "LMB collect | RMB place/cast | 1 stone 2 crystal 3 grass 4 compress/release 5 moving well 6 vortex"
                    .into(),
                format!("Selected {}: {} | target: {}", selected + 1, item, target),
                format!(
                    "XYZ {:.1} {:.1} {:.1} | {} | {:.1} ms/frame | K save / L load | F3 field diagnostics",
                    character.feet[0],
                    character.feet[1],
                    character.feet[2],
                    if locked {
                        "PLAY"
                    } else {
                        "PAUSED - ESC to resume"
                    },
                    dt * 1000.
                ),
                format!(
                    "Fields: {} active, {:.0} manons | {:.0} K peak | energy error {:.2e} J",
                    game.magic().active().len(),
                    game.magic().bound_manons(),peak_temperature,game.coupled_energy_residual_joules()?
                ),
                status.clone(),
            ],
        );
        let overlays_ms = overlays_start.elapsed().as_secs_f64() * 1000.;
        let draw_start = Instant::now();
        compose_frame(
            &mut base,
            &scene,
            &[overlay],
            [0.19, 0.34, 0.49, 1.0],
            &window,
        );
        let draw_present_ms = draw_start.elapsed().as_secs_f64() * 1000.;
        frame_count += 1;
        if let Some(writer) = &mut trace {
            writeln!(
                writer,
                "{frame_count},{interval_ms:.6},{:.6},{input_ms:.6},{update_ms:.6},{terrain_ms:.6},{effects_ms:.6},{overlays_ms:.6},{draw_present_ms:.6},{},{}",
                now.elapsed().as_secs_f64() * 1000.,
                game.matter().bodies.len(),
                game.magic().active().len()
            )?;
        }
        if smoke_frames.is_some_and(|limit| frame_count >= limit) {
            println!(
                "3D smoke complete: {frame_count} frames, {} active programs, peak {hottest:.0} K, peak {most_bodies} bodies, {} dispersed blocks, energy residual {:.3e} J",
                game.magic().active().len(),
                game.matter().dispersed_blocks,
                game.coupled_energy_residual_joules()?
            );
            break;
        }
    }
    if let Some(writer) = &mut trace {
        writer.flush()?;
    }
    Ok(())
}
