use manonica_world::{
    chunk::{CellPosition, ChunkCoord, ChunkLayout, ChunkWorld},
    gameplay::{Character, WalkInput, raycast},
    registry::{ContentRegistry, ResourceKey},
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
