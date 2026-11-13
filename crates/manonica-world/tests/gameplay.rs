use manonica_world::{
    chunk::{CellPosition, ChunkCoord, ChunkLayout, ChunkWorld},
    gameplay::{Character, WalkInput, raycast},
    magic::SpellRegistry,
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
fn world() -> ChunkWorld {
    let mut w = ChunkWorld::new(content(), ChunkLayout::new([8, 12, 8], 1.0).unwrap());
    w.create(ChunkCoord { x: 0, z: 0 }, 3.0, true).unwrap();
    for z in 0..8 {
        for x in 0..8 {
            w.set_block(
                CellPosition { x, y: 0, z },
                &key("manonica:stone"),
                BTreeMap::new(),
                Default::default(),
            )
            .unwrap();
        }
    }
    w
}
#[test]
fn walk_sprint_jump_and_wall_contacts() {
    let mut w = world();
    for y in 1..5 {
        for z in 0..8 {
            w.set_block(
                CellPosition { x: 5, y, z },
                &key("manonica:stone"),
                BTreeMap::new(),
                Default::default(),
            )
            .unwrap();
        }
    }
    let mut c = Character::new([2.5, 4.0, 3.5]);
    for _ in 0..120 {
        c.step(&w, WalkInput::default(), 1.0 / 120.0, [0.; 3]);
    }
    assert!(c.grounded);
    assert!((c.feet[1] - 1.0).abs() < 1e-5);
    c.step(
        &w,
        WalkInput {
            jump: true,
            ..Default::default()
        },
        1.0 / 120.0,
        [0.; 3],
    );
    assert!(c.velocity[1] > 0.0);
    assert!(c.feet[1] > 1.0);
    for _ in 0..240 {
        c.step(
            &w,
            WalkInput {
                forward: 1.0,
                sprint: true,
                ..Default::default()
            },
            1.0 / 120.0,
            [0.; 3],
        );
    }
    assert!(c.feet[0] <= 4.70001);
    assert!(!c.blocked(&w));
    assert!(c.grounded);
}
#[test]
fn voxel_raycast_hits_exact_faces_and_respects_range() {
    let w = world();
    let hit = raycast(&w, [2.5, 4.0, 2.5], [0.0, -1.0, 0.0], 10.0).unwrap();
    assert_eq!(hit.block, CellPosition { x: 2, y: 0, z: 2 });
    assert_eq!(hit.adjacent, CellPosition { x: 2, y: 1, z: 2 });
    assert_eq!(hit.distance, 3.0);
    assert!(raycast(&w, [2.5, 4.0, 2.5], [0.0, -1.0, 0.0], 2.9).is_none());
    assert!(raycast(&w, [f64::NAN; 3], [0.0, -1.0, 0.0], 10.0).is_none());
}
#[test]
fn spells_consume_scroll_reserve_mana_change_world_and_restore_resources() {
    let c = content();
    let registry =
        SpellRegistry::from_jsonl(include_str!("../../../assets/demo/spells.jsonl"), &c).unwrap();
    for (scroll, expected) in [
        ("manonica:scroll_explosion", 0),
        ("manonica:scroll_rift", 1),
        ("manonica:scroll_tornado", 2),
    ] {
        let mut game = GameRuntime::new(c.clone(), ChunkLayout::new([8, 12, 8], 1.0).unwrap());
        game.world = world();
        for z in 1..7 {
            for x in 1..7 {
                for y in 1..3 {
                    game.world
                        .set_block(
                            CellPosition { x, y, z },
                            &key("manonica:stone"),
                            BTreeMap::new(),
                            Default::default(),
                        )
                        .unwrap();
                }
            }
        }
        let material_before = game.world.occupied_blocks().count();
        let player = game.spawn(&key("manonica:player")).unwrap();
        game.give(player, 0, &key(scroll), 2).unwrap();
        let before = game.world.statistics().free_manons;
        let event = game
            .cast_scroll(
                &registry,
                player,
                0,
                [4.5, 5.5, 4.5],
                [4.5, 3.0, 4.5],
                [0., 0., -1.],
            )
            .unwrap();
        assert_eq!(event.effect.index(), expected);
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
        assert!(
            (game.world.statistics().free_manons + game.magic().bound_manons() - before).abs()
                < 1e-9
        );
        assert_eq!(event.changed_blocks, 0);
        assert_eq!(game.world.occupied_blocks().count(), material_before);
        game.advance_magic(0.1).unwrap();
        assert!(game.magic().debris().is_empty());
        for _ in 0..4 {
            game.advance_magic(0.2).unwrap();
        }
        assert!(!game.magic().debris().is_empty());
        assert_eq!(
            game.world.occupied_blocks().count() + game.magic().debris().len(),
            material_before
        );
        let moving = game
            .magic()
            .debris()
            .iter()
            .any(|b| b.motion.momentum.iter().any(|v| v.abs() > 0.1));
        assert!(moving, "spell {expected}: no moving bodies");
        if expected == 2 {
            assert!(game.magic_acceleration([4.5, 4.0, 4.5])[1] > 0.0);
        }
        let mut restored = GameRuntime::from_nbt(c.clone(), &game.to_nbt().unwrap()).unwrap();
        assert_eq!(restored.magic().active().len(), 1);
        for _ in 0..40 {
            restored.advance_magic(0.25).unwrap();
        }
        assert_eq!(
            restored.world.occupied_blocks().count() + restored.magic().debris().len(),
            material_before
        );
        assert_eq!(restored.magic().bound_manons(), 0.0);
        assert!((restored.world.statistics().free_manons - before).abs() < 1e-9);
    }
}
#[test]
fn failed_cast_is_atomic_and_spell_definitions_are_validated() {
    let c = content();
    let text = include_str!("../../../assets/demo/spells.jsonl");
    let registry = SpellRegistry::from_jsonl(text, &c).unwrap();
    let mut game = GameRuntime::new(c.clone(), ChunkLayout::new([8, 12, 8], 1.0).unwrap());
    game.world
        .create(ChunkCoord { x: 0, z: 0 }, 0.0, true)
        .unwrap();
    let player = game.spawn(&key("manonica:player")).unwrap();
    game.give(player, 0, &key("manonica:scroll_explosion"), 1)
        .unwrap();
    let before = game.to_nbt().unwrap();
    assert!(
        game.cast_scroll(
            &registry,
            player,
            0,
            [4.5, 5.5, 4.5],
            [4.5, 3.0, 4.5],
            [0., 0., -1.]
        )
        .is_err()
    );
    assert_eq!(game.magic().casts(), 0);
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
    assert_eq!(game.world.statistics().free_manons, 0.0);
    assert!(GameRuntime::from_nbt(c.clone(), &before).is_ok());
    assert!(
        SpellRegistry::from_jsonl(&text.replace("\"radius\":3.5", "\"radius\":-1.0"), &c).is_err()
    );
    assert!(
        SpellRegistry::from_jsonl(
            &text.replace("manonica:scroll_explosion", "missing:scroll"),
            &c
        )
        .is_err()
    );
}
