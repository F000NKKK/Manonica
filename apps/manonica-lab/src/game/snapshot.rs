//! Reproducible offscreen capture of the same game state and observer used by the client.
use super::{effects::SpellView, mesh::TerrainView, world};
use manonica_world::magic::SpellRegistry;
use meridian_sdk::{Device, FlyCamera, GraphicsSession, Scene3d, Surface, Vec3};
use std::{io::Write, path::Path};
pub fn capture(path: &Path, seconds: f64, slot: usize) -> Result<(), Box<dyn std::error::Error>> {
    if !seconds.is_finite() || !(0.0..=10.0).contains(&seconds) || !(3..=5).contains(&slot) {
        return Err("capture requires 0..10 seconds and scroll slot 4..6".into());
    }
    let (mut game, player) = world::generate()?;
    let spells = SpellRegistry::from_jsonl(
        include_str!("../../../../assets/demo/spells.jsonl"),
        game.world.content(),
    )?;
    let center = [0.5, 4.5, -3.5];
    game.cast_scroll(
        &spells,
        player,
        slot,
        [0.5, 5.5, 8.0],
        center,
        [0., 0., -1.],
    )?;
    let steps = (seconds / 0.025).round() as usize;
    for _ in 0..steps {
        game.advance_magic(0.025)?;
    }
    let runtime = tokio::runtime::Runtime::new()?;
    let device = runtime.block_on(Device::new())?;
    let surface = Surface::offscreen(&device, 1280, 800);
    let mut base = GraphicsSession::new(device.clone(), surface, 1280, 800);
    let mut terrain = TerrainView::new();
    terrain
        .rebuild(&mut base, &game.world)
        .map_err(std::io::Error::other)?;
    let mut effects =
        SpellView::new(&device, &runtime, &mut base).map_err(std::io::Error::other)?;
    let eye: [f64; 3] = [8.0, 10.0, 12.0];
    let target = [0.5, 5.0, -3.5];
    let d = std::array::from_fn::<_, 3, _>(|a| target[a] - eye[a]);
    let mut camera = FlyCamera::new(Vec3::new(eye[0] as f32, eye[1] as f32, eye[2] as f32));
    camera.yaw = d[2].atan2(d[0]) as f32;
    camera.pitch = (d[1] / d.iter().map(|v| v * v).sum::<f64>().sqrt()).asin() as f32;
    let mut scene = Scene3d {
        camera: camera.camera(1.6),
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
    );
    let pixels = base
        .render_to_pixels(&scene, [0.19, 0.34, 0.49, 1.0])
        .ok_or("offscreen render failed")?;
    let mut file = std::fs::File::create(path)?;
    file.write_all(b"P6\n1280 800\n255\n")?;
    for pixel in pixels.as_chunks::<4>().0 {
        file.write_all(&pixel[..3])?;
    }
    let peak = game.peak_temperature_kelvin()?;
    println!(
        "Captured {seconds:.3}s: peak {peak:.0} K, {} bodies, {} dispersed blocks, residual {:.3e} J",
        game.matter().bodies.len(),
        game.matter().dispersed_blocks,
        game.coupled_energy_residual_joules()?
    );
    Ok(())
}
