use manonica_world::{
    WorldResult,
    chunk::{CellPosition, ChunkCoord, ChunkLayout},
    registry::{ContentRegistry, ResourceKey},
    runtime::{EntityId, EntityTransform, GameRuntime},
};
use std::{collections::BTreeMap, sync::Arc};
pub const SPAWN: [f64; 3] = [0.5, 10.0, 13.5];
pub fn content() -> WorldResult<Arc<ContentRegistry>> {
    Ok(Arc::new(ContentRegistry::from_jsonl(
        include_str!("../../../../assets/demo/blocks.jsonl"),
        include_str!("../../../../assets/demo/items.jsonl"),
        include_str!("../../../../assets/demo/entities.jsonl"),
    )?))
}
pub fn generate() -> WorldResult<(GameRuntime, EntityId)> {
    let mut game = GameRuntime::new(content()?, ChunkLayout::column(400)?);
    for z in -2..2 {
        for x in -2..2 {
            game.world.create(
                ChunkCoord { x, z },
                3.0 + f64::from((x - z).rem_euclid(3)),
                true,
            )?;
        }
    }
    let stone = ResourceKey::parse("manonica:stone")?;
    let dirt = ResourceKey::parse("manonica:dirt")?;
    let grass = ResourceKey::parse("manonica:grass")?;
    let crystal = ResourceKey::parse("manonica:mana_crystal")?;
    for z in -32_i64..32 {
        for x in -32_i64..32 {
            let h = if x.abs() < 8 && z.abs() < 18 {
                3
            } else {
                3 + ((x as f64 * 0.16).sin() * (z as f64 * 0.12).cos() * 2.0 + 2.0) as usize
            };
            for y in 0..=h {
                let key = if y == h {
                    &grass
                } else if y + 2 >= h {
                    &dirt
                } else {
                    &stone
                };
                game.world.set_block(
                    CellPosition { x, y, z },
                    key,
                    BTreeMap::new(),
                    Default::default(),
                )?;
            }
        }
    }
    for (x, z) in [(-7, -4), (7, -4), (-10, 7), (11, 8)] {
        for y in 4..9 {
            game.world.set_block(
                CellPosition { x, y, z },
                &stone,
                BTreeMap::new(),
                Default::default(),
            )?;
        }
        game.world.set_block(
            CellPosition { x, y: 9, z },
            &crystal,
            BTreeMap::from([("axis".into(), "y".into())]),
            Default::default(),
        )?;
    }
    // Walk-through stone arch and stepped platforms.
    for x in -3_i64..=3 {
        for y in 4..=8 {
            if x.abs() == 3 || y == 8 {
                game.world.set_block(
                    CellPosition { x, y, z: -12 },
                    &stone,
                    BTreeMap::new(),
                    Default::default(),
                )?;
            }
        }
    }
    for i in 0..5 {
        for x in 12..16 {
            game.world.set_block(
                CellPosition {
                    x,
                    y: 5 + i,
                    z: -12 + i as i64,
                },
                &stone,
                BTreeMap::new(),
                Default::default(),
            )?;
        }
    }
    let player = game.spawn_at(
        &ResourceKey::parse("manonica:player")?,
        EntityTransform {
            position_metres: SPAWN,
            rotation_radians: [0.0, -std::f64::consts::FRAC_PI_2, 0.0],
        },
    )?;
    for (slot, key, count) in [
        (0, "manonica:stone", 64),
        (1, "manonica:mana_crystal", 16),
        (2, "manonica:grass", 64),
        (3, "manonica:scroll_explosion", 16),
        (4, "manonica:scroll_rift", 16),
        (5, "manonica:scroll_tornado", 16),
    ] {
        game.give(player, slot, &ResourceKey::parse(key)?, count)?;
    }
    Ok((game, player))
}
