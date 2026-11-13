use manonica_world::{
    chunk::{CellPosition, ChunkCoord, ChunkLayout, ChunkWorld},
    metadata::{NbtCompound, NbtTag},
    registry::{ContentRegistry, ResourceKey},
    runtime::GameRuntime,
};
use std::{collections::BTreeMap, sync::Arc};
fn content() -> Arc<ContentRegistry> {
    Arc::new(
        ContentRegistry::from_jsonl(
            include_str!("../../../assets/demo/blocks.jsonl"),
            include_str!("../../../assets/demo/items.jsonl"),
            include_str!("../../../assets/demo/entities.jsonl"),
        )
        .unwrap(),
    )
}
fn key(v: &str) -> ResourceKey {
    ResourceKey::parse(v).unwrap()
}
fn pos(x: i64) -> CellPosition {
    CellPosition { x, y: 0, z: 0 }
}
#[test]
fn chunk_flux_and_lifecycle_conserve_mana() {
    let mut w = ChunkWorld::new(content(), ChunkLayout::new([2, 1, 1], 1.0).unwrap());
    let a = ChunkCoord { x: 0, z: 0 };
    let b = ChunkCoord { x: 1, z: 0 };
    w.create(a, 10.0, true).unwrap();
    w.create(b, 0.0, true).unwrap();
    w.diffuse(1.0, 0.1).unwrap();
    assert_eq!(w.density(pos(1)).unwrap(), 9.0);
    assert_eq!(w.density(pos(2)).unwrap(), 1.0);
    assert_eq!(w.statistics().free_manons, 20.0);
    w.unload(b).unwrap();
    let before = w.density(pos(1)).unwrap();
    w.diffuse(1.0, 0.1).unwrap();
    assert_eq!(w.density(pos(1)).unwrap(), before + 0.1);
    assert!(!w.active(b));
    assert_eq!(w.statistics().free_manons, 20.0);
    w.reload(b, true).unwrap();
    assert_eq!(w.density(pos(2)).unwrap(), 1.0);
    for _ in 0..50 {
        w.diffuse(1.0, 0.1).unwrap();
    }
    assert!((w.statistics().free_manons - 20.0).abs() < 1e-12);
    let saved = w.export(a).unwrap();
    assert!(w.import(&saved, true).is_err());
    assert!(w.diffuse(1.0, 1.0).is_err());
}
#[test]
fn negative_coordinates_and_impermeable_blocks() {
    let layout = ChunkLayout::new([2, 1, 1], 1.0).unwrap();
    assert_eq!(
        layout.locate(pos(-1)).unwrap(),
        (ChunkCoord { x: -1, z: 0 }, 1)
    );
    let mut w = ChunkWorld::new(content(), layout);
    w.create(ChunkCoord { x: -1, z: 0 }, 10.0, true).unwrap();
    w.create(ChunkCoord { x: 0, z: 0 }, 0.0, true).unwrap();
    w.set_block(
        pos(-1),
        &key("manonica:stone"),
        BTreeMap::new(),
        NbtCompound::default(),
    )
    .unwrap();
    w.diffuse(1.0, 0.1).unwrap();
    assert_eq!(w.density(pos(0)).unwrap(), 0.0);
    assert_eq!(w.statistics().free_manons, 20.0);
}
#[test]
fn gameplay_and_full_nbt_restore_are_transactional() {
    let c = content();
    let mut game = GameRuntime::new(c.clone(), ChunkLayout::column(400).unwrap());
    let a = ChunkCoord { x: 0, z: 0 };
    let b = ChunkCoord { x: 1, z: 0 };
    game.world.create(a, 3.0, true).unwrap();
    game.world.create(b, 7.0, false).unwrap();
    game.world.unload(b).unwrap();
    let player = game.spawn(&key("manonica:player")).unwrap();
    game.give(player, 0, &key("manonica:mana_crystal"), 2)
        .unwrap();
    game.place(player, 0, pos(0)).unwrap();
    assert!(game.place(player, 0, pos(0)).is_err());
    assert_eq!(
        game.entity(player)
            .unwrap()
            .inventory
            .get(0)
            .unwrap()
            .unwrap()
            .count(),
        1
    );
    game.break_block(player, 1, pos(0)).unwrap();
    game.place(player, 1, pos(1)).unwrap();
    game.tick(1.0, 0.1).unwrap();
    let bytes = game.to_nbt().unwrap();
    assert_eq!(bytes[0], 10);
    let restored = GameRuntime::from_nbt(c, &bytes).unwrap();
    assert_eq!(restored.clock(), game.clock());
    assert_eq!(
        restored.world.statistics().free_manons,
        game.world.statistics().free_manons
    );
    assert!(!restored.world.resident_status(b));
    assert_eq!(
        restored.entity(player).unwrap().inventory,
        game.entity(player).unwrap().inventory
    );
    assert_eq!(
        restored.world.block_data(pos(1)).unwrap(),
        game.world.block_data(pos(1)).unwrap()
    );
    let mut damaged = bytes.clone();
    damaged.pop();
    assert!(GameRuntime::from_nbt(content(), &damaged).is_err());
}
#[test]
fn nbt_all_tag_types_and_malformed_inputs() {
    let tags = vec![
        NbtTag::Byte(-1),
        NbtTag::Short(-2),
        NbtTag::Int(3),
        NbtTag::Long(4),
        NbtTag::Float(1.25),
        NbtTag::Double(2.5),
        NbtTag::String("мана\0🌍".into()),
        NbtTag::ByteArray(vec![-1, 0, 1]),
        NbtTag::IntArray(vec![i32::MIN]),
        NbtTag::LongArray(vec![i64::MAX]),
        NbtTag::List(vec![NbtTag::Int(1)]),
        NbtTag::Compound(BTreeMap::new()),
    ];
    let data = NbtCompound::new(
        tags.into_iter()
            .enumerate()
            .map(|(i, v)| (i.to_string(), v))
            .collect(),
    )
    .unwrap();
    let bytes = data.to_bytes().unwrap();
    assert_eq!(NbtCompound::from_bytes(&bytes).unwrap(), data);
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(NbtCompound::from_bytes(&trailing).is_err());
    assert!(NbtCompound::from_bytes(&bytes[..8]).is_err());
    assert!(
        NbtCompound::new(BTreeMap::from([(
            "bad".into(),
            NbtTag::List(vec![NbtTag::Int(1), NbtTag::Byte(1)])
        )]))
        .is_err()
    );
    assert!(NbtCompound::from_bytes(&[10, 0, 0, 7, 0, 1, b'a', 255, 255, 255, 255, 0]).is_err());
}
#[test]
fn registries_reject_duplicates_unknown_fields_and_unknown_references() {
    let blocks = include_str!("../../../assets/demo/blocks.jsonl");
    let items = include_str!("../../../assets/demo/items.jsonl");
    let entities = include_str!("../../../assets/demo/entities.jsonl");
    assert!(
        ContentRegistry::from_jsonl(
            &format!("{blocks}\n{}", blocks.lines().next().unwrap()),
            items,
            entities
        )
        .is_err()
    );
    assert!(
        ContentRegistry::from_jsonl(
            blocks,
            &items.replace("manonica:stone", "missing:stone"),
            entities
        )
        .is_err()
    );
    let reversed = blocks.lines().rev().collect::<Vec<_>>().join("\n");
    assert_eq!(
        ContentRegistry::from_jsonl(&reversed, items, entities)
            .unwrap()
            .fingerprint(),
        content().fingerprint()
    );
    assert!(
        content()
            .state(
                &key("manonica:mana_crystal"),
                BTreeMap::from([("axis".into(), "w".into())])
            )
            .is_err()
    );
}
#[test]
fn thousand_columns_allocate_uniform_payloads() {
    let mut w = ChunkWorld::new(content(), ChunkLayout::column(400).unwrap());
    for x in 0..1000 {
        w.create(ChunkCoord { x, z: 0 }, f64::from(x % 5), x < 500)
            .unwrap();
    }
    let stats = w.statistics();
    assert_eq!(stats.chunks.active, 500);
    assert_eq!(stats.payload_bytes, 10_000);
    assert_eq!(stats.free_manons, 204_800_000.0);
}
#[test]
fn chunked_transport_matches_monolithic_reference() {
    use manonica_core::{
        Evolution, Quantity, TimeStep,
        units::{Diffusivity, ManonDensity},
    };
    use manonica_physics::{FreeDiffusion, FreeManaField, GridShape};
    let mut w = ChunkWorld::new(content(), ChunkLayout::new([2, 2, 2], 0.5).unwrap());
    let mut field = FreeManaField::uniform(
        GridShape::new(4, 2, 4).unwrap(),
        0.5,
        Quantity::<ManonDensity>::nonnegative(0.0).unwrap(),
    )
    .unwrap();
    for z in 0..2 {
        for x in 0..2 {
            let density = f64::from(1 + x + 2 * z);
            w.create(ChunkCoord { x, z }, density, true).unwrap();
            for zz in 0..2 {
                for y in 0..2 {
                    for xx in 0..2 {
                        field
                            .set(
                                (2 * x) as usize + xx,
                                y,
                                (2 * z) as usize + zz,
                                Quantity::<ManonDensity>::nonnegative(density).unwrap(),
                            )
                            .unwrap();
                    }
                }
            }
        }
    }
    let diffusion =
        FreeDiffusion::new(Quantity::<Diffusivity>::nonnegative(0.25).unwrap()).unwrap();
    for _ in 0..20 {
        w.diffuse(0.25, 0.1).unwrap();
        diffusion
            .evolve(&mut field, TimeStep::from_seconds(0.1).unwrap())
            .unwrap();
    }
    for z in 0..4 {
        for y in 0..2 {
            for x in 0..4 {
                let actual = w
                    .density(CellPosition {
                        x: x as i64,
                        y,
                        z: z as i64,
                    })
                    .unwrap();
                assert!((actual - field.at(x, y, z).unwrap().value()).abs() < 1e-13);
            }
        }
    }
}
#[test]
fn failed_commands_keep_inventory_blocks_and_clock() {
    let mut game = GameRuntime::new(content(), ChunkLayout::new([2, 1, 1], 1.0).unwrap());
    game.world
        .create(ChunkCoord { x: 0, z: 0 }, 1.0, true)
        .unwrap();
    let player = game.spawn(&key("manonica:player")).unwrap();
    game.give(player, 0, &key("manonica:stone"), 64).unwrap();
    let before = game.entity(player).unwrap().inventory.clone();
    assert!(game.give(player, 0, &key("manonica:stone"), 1).is_err());
    assert_eq!(game.entity(player).unwrap().inventory, before);
    game.place(player, 0, pos(0)).unwrap();
    game.give(player, 1, &key("manonica:probe"), 1).unwrap();
    let block = game.world.block(pos(0)).unwrap().clone();
    let inventory = game.entity(player).unwrap().inventory.clone();
    assert!(game.break_block(player, 1, pos(0)).is_err());
    assert_eq!(game.world.block(pos(0)).unwrap(), &block);
    assert_eq!(game.entity(player).unwrap().inventory, inventory);
    let clock = game.clock();
    let total = game.world.statistics().free_manons;
    assert!(game.tick(1.0, 1.0).is_err());
    assert_eq!(game.clock(), clock);
    assert_eq!(game.world.statistics().free_manons, total);
}
#[test]
fn metadata_and_entity_transforms_survive_full_save() {
    use manonica_world::runtime::EntityTransform;
    let mut game = GameRuntime::new(content(), ChunkLayout::new([2, 1, 1], 1.0).unwrap());
    game.world
        .create(ChunkCoord { x: 0, z: 0 }, 0.0, true)
        .unwrap();
    let transform = EntityTransform {
        position_metres: [-2.0, 12.0, 0.5],
        rotation_radians: [0.0, 0.2, 0.0],
    };
    let player = game.spawn_at(&key("manonica:player"), transform).unwrap();
    let metadata = NbtCompound::new(BTreeMap::from([(
        "owner".into(),
        NbtTag::String("игрок".into()),
    )]))
    .unwrap();
    game.world
        .set_block(
            pos(0),
            &key("manonica:mana_crystal"),
            BTreeMap::from([("axis".into(), "z".into())]),
            metadata.clone(),
        )
        .unwrap();
    game.break_block(player, 0, pos(0)).unwrap();
    game.place(player, 0, pos(1)).unwrap();
    let restored = GameRuntime::from_nbt(content(), &game.to_nbt().unwrap()).unwrap();
    assert_eq!(restored.entity(player).unwrap().transform, transform);
    assert_eq!(
        restored.world.block_data(pos(1)).unwrap().get("owner"),
        metadata.get("owner")
    );
    assert!(
        game.spawn_at(
            &key("manonica:player"),
            EntityTransform {
                position_metres: [f64::NAN; 3],
                ..transform
            }
        )
        .is_err()
    );
}
#[test]
fn malformed_content_and_nbt_limits_fail_before_acceptance() {
    let blocks = include_str!("../../../assets/demo/blocks.jsonl");
    let items = include_str!("../../../assets/demo/items.jsonl");
    let entities = include_str!("../../../assets/demo/entities.jsonl");
    assert!(
        ContentRegistry::from_jsonl(
            &blocks.replace("\"display\"", "\"unknown\""),
            items,
            entities
        )
        .is_err()
    );
    assert!(
        ContentRegistry::from_jsonl(
            blocks,
            &items.replace("\"limit\":64", "\"limit\":0"),
            entities
        )
        .is_err()
    );
    let duplicate = [10, 0, 0, 1, 0, 1, b'a', 1, 1, 0, 1, b'a', 2, 0];
    assert!(NbtCompound::from_bytes(&duplicate).is_err());
    let mut tag = NbtTag::Int(1);
    for _ in 0..40 {
        tag = NbtTag::List(vec![tag]);
    }
    assert!(NbtCompound::new(BTreeMap::from([("deep".into(), tag)])).is_err());
    assert!(NbtCompound::new(BTreeMap::from([("nan".into(), NbtTag::Double(f64::NAN))])).is_err());
}
