//! Headless registry / chunk / inventory / NBT smoke demo.
use manonica_world::{
    chunk::{CellPosition, ChunkCoord, ChunkLayout},
    registry::{ContentRegistry, ResourceKey},
    runtime::GameRuntime,
};
use std::{path::PathBuf, sync::Arc};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut chunks = 2_i32;
    let mut active = 2_i32;
    let mut steps = 30_u64;
    let mut save = None;
    let mut content = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/demo");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args.next().ok_or("missing argument value")?;
        match arg.as_str() {
            "--chunks" => chunks = value.parse()?,
            "--active" => active = value.parse()?,
            "--steps" => steps = value.parse()?,
            "--save" => save = Some(PathBuf::from(value)),
            "--content" => content = PathBuf::from(value),
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    if !(1..=1000).contains(&chunks) || active < 0 || active > chunks {
        return Err("require 1..1000 chunks and 0..chunks active".into());
    }
    let registry = Arc::new(ContentRegistry::from_jsonl(
        &std::fs::read_to_string(content.join("blocks.jsonl"))?,
        &std::fs::read_to_string(content.join("items.jsonl"))?,
        &std::fs::read_to_string(content.join("entities.jsonl"))?,
    )?);
    let mut game = GameRuntime::new(registry.clone(), ChunkLayout::column(400)?);
    for x in 0..chunks {
        game.world
            .create(ChunkCoord { x, z: 0 }, f64::from((x % 5) + 1), x < active)?;
    }
    let key = |v: &str| ResourceKey::parse(v);
    let player = game.spawn(&key("manonica:player")?)?;
    game.give(player, 0, &key("manonica:mana_crystal")?, 4)?;
    game.place(player, 0, CellPosition { x: 1, y: 1, z: 1 })?;
    let before = game.world.statistics().free_manons;
    for _ in 0..steps {
        game.tick(1.0, 0.1)?;
    }
    let bytes = game.to_nbt()?;
    let restored = GameRuntime::from_nbt(registry, &bytes)?;
    let after = restored.world.statistics().free_manons;
    if (after - before).abs() > before.max(1.0) * 1e-10 {
        return Err("resource audit failed".into());
    }
    if let Some(path) = save {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        use std::io::Write;
        file.write_all(&bytes)?;
        file.sync_all()?;
    }
    println!(
        "backend: CPU chunk reference; height: 400; ticks: {}",
        restored.clock().ticks
    );
    println!("world: {:?}", restored.world.statistics());
    println!(
        "NBT: {} bytes; restored entities: {}; mana error: {:.3e}",
        bytes.len(),
        restored.entities().len(),
        after - before
    );
    Ok(())
}
