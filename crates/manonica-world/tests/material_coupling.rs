use manonica_world::{
    chunk::{CellPosition, ChunkCoord, ChunkLayout},
    magic::SpellRegistry,
    registry::{ContentRegistry, ResourceKey},
    runtime::GameRuntime,
};
use std::{collections::BTreeMap, sync::Arc};
fn key(v: &str) -> ResourceKey {
    ResourceKey::parse(v).unwrap()
}
#[test]
fn field_momentum_drives_ordinary_material_and_then_it_settles_without_losing_state() {
    let c = Arc::new(
        ContentRegistry::from_jsonl(
            include_str!("../../../assets/demo/blocks.jsonl"),
            include_str!("../../../assets/demo/items.jsonl"),
            include_str!("../../../assets/demo/entities.jsonl"),
        )
        .unwrap(),
    );
    let mut game = GameRuntime::new(c.clone(), ChunkLayout::new([12, 16, 12], 1.).unwrap());
    game.world
        .create(ChunkCoord { x: 0, z: 0 }, 30., true)
        .unwrap();
    for z in 0..12 {
        for x in 0..12 {
            game.world
                .set_block(
                    CellPosition { x, y: 0, z },
                    &key("manonica:stone"),
                    BTreeMap::new(),
                    Default::default(),
                )
                .unwrap();
        }
    }
    let p = CellPosition { x: 6, y: 1, z: 6 };
    game.world
        .set_block(
            p,
            &key("manonica:grass"),
            BTreeMap::new(),
            Default::default(),
        )
        .unwrap();
    let material_before = game.world.occupied_blocks().count();
    let manons = game.world.statistics().free_manons;
    let before_revision = game.world.geometry_revision();
    let player = game.spawn(&key("manonica:player")).unwrap();
    game.give(player, 0, &key("manonica:scroll_tornado"), 1)
        .unwrap();
    let mut recipe: serde_json::Value = serde_json::from_str(
        include_str!("../../../assets/demo/spells.jsonl")
            .lines()
            .nth(2)
            .unwrap(),
    )
    .unwrap();
    recipe["key"] = serde_json::json!("test:generic_momentum_reservoir");
    recipe["program"]["domain"] =
        serde_json::json!({"dimensions":[5,5,5],"anchor":[0.5,0.,0.5],"duration_seconds":0.8});
    recipe["program"]["controls"] = serde_json::json!({"potential_source":"0","manon_velocity":{"x":"0","y":"10","z":"0"},"heat_power":"0"});
    recipe["program"]["transport"]["diffusivity"] = serde_json::json!(0.);
    recipe["program"]["transport"]["mobility"] = serde_json::json!(0.);
    let registry = SpellRegistry::from_jsonl(&recipe.to_string(), &c).unwrap();
    game.cast_scroll(
        &registry,
        player,
        0,
        [6.5, 1.5, 6.5],
        [6.5, 1.5, 6.5],
        [0., 0., 1.],
    )
    .unwrap();
    let mut saw_entity = false;
    let mut highest = 1.5_f64;
    for _ in 0..320 {
        game.advance_magic(0.025).unwrap();
        saw_entity |= !game.matter().bodies.is_empty();
        for b in &game.matter().bodies {
            highest = highest.max(b.motion.position_metres[1]);
        }
        assert!(
            (game.world.statistics().free_manons + game.magic().bound_manons() - manons).abs()
                < 1e-8
        );
        assert!(game.coupled_energy_residual_joules().unwrap().abs() < 1e-6);
        assert_eq!(
            game.world.occupied_blocks().count() + game.matter().bodies.len(),
            material_before
        );
    }
    assert!(saw_entity);
    assert!(highest > 1.55);
    assert!(game.matter().bodies.is_empty());
    assert!(game.world.geometry_revision() > before_revision);
    let restored = GameRuntime::from_nbt(c, &game.to_nbt().unwrap()).unwrap();
    assert_eq!(restored.world.occupied_blocks().count(), material_before);
}
#[test]
fn jsonl_vortex_transfers_vertical_and_tangential_momentum_to_material() {
    let c = Arc::new(
        ContentRegistry::from_jsonl(
            include_str!("../../../assets/demo/blocks.jsonl"),
            include_str!("../../../assets/demo/items.jsonl"),
            include_str!("../../../assets/demo/entities.jsonl"),
        )
        .unwrap(),
    );
    let r =
        SpellRegistry::from_jsonl(include_str!("../../../assets/demo/spells.jsonl"), &c).unwrap();
    let mut game = GameRuntime::new(c, ChunkLayout::new([12, 20, 12], 1.).unwrap());
    game.world
        .create(ChunkCoord { x: 0, z: 0 }, 3., true)
        .unwrap();
    for z in 0..12 {
        for x in 0..12 {
            for y in 0..2 {
                game.world
                    .set_block(
                        CellPosition { x, y, z },
                        &key(if y == 0 {
                            "manonica:stone"
                        } else {
                            "manonica:grass"
                        }),
                        BTreeMap::new(),
                        Default::default(),
                    )
                    .unwrap();
            }
        }
    }
    let player = game.spawn(&key("manonica:player")).unwrap();
    game.give(player, 0, &key("manonica:scroll_tornado"), 1)
        .unwrap();
    let center = [6.5, 2.5, 6.5];
    game.cast_scroll(&r, player, 0, center, center, [0., 0., 1.])
        .unwrap();
    let mut highest = 1.5_f64;
    let mut angular = 0.0_f64;
    let mut bodies = 0;
    for _ in 0..240 {
        game.advance_magic(0.025).unwrap();
        bodies = bodies.max(game.matter().bodies.len());
        for b in &game.matter().bodies {
            let p = b.motion.position_metres;
            highest = highest.max(p[1]);
            let m = b.motion.momentum;
            angular = angular.max(((p[2] - center[2]) * m[0] - (p[0] - center[0]) * m[2]).abs());
        }
        assert!(game.coupled_energy_residual_joules().unwrap().abs() < 1e-5);
    }
    assert!(bodies > 0, "no detached bodies");
    assert!(highest > 3.0, "highest body {highest}");
    assert!(angular > 1., "angular momentum {angular}");
}
