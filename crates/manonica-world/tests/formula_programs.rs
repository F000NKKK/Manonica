use manonica_world::{
    chunk::{ChunkCoord, ChunkLayout},
    magic::SpellRegistry,
    registry::{ContentRegistry, ResourceKey},
    runtime::GameRuntime,
};
use std::sync::Arc;
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
fn recipe(index: usize) -> serde_json::Value {
    let line = include_str!("../../../assets/demo/spells.jsonl")
        .lines()
        .nth(index)
        .unwrap();
    let mut v: serde_json::Value = serde_json::from_str(line).unwrap();
    v["program"]["domain"]["dimensions"] = serde_json::json!([5, 5, 5]);
    v["program"]["domain"]["anchor"] = serde_json::json!([0.5, 0.5, 0.5]);
    v
}
fn setup(
    v: &serde_json::Value,
    density: f64,
) -> (
    GameRuntime,
    SpellRegistry,
    manonica_world::runtime::EntityId,
) {
    let c = content();
    let registry = SpellRegistry::from_jsonl(&v.to_string(), &c).unwrap();
    let mut game = GameRuntime::new(c, ChunkLayout::new([12, 16, 12], 1.0).unwrap());
    game.world
        .create(ChunkCoord { x: 0, z: 0 }, density, true)
        .unwrap();
    let player = game.spawn(&key("manonica:player")).unwrap();
    game.give(player, 0, &key(v["scroll"].as_str().unwrap()), 2)
        .unwrap();
    (game, registry, player)
}
fn cast(
    game: &mut GameRuntime,
    registry: &SpellRegistry,
    player: manonica_world::runtime::EntityId,
) -> manonica_world::WorldResult<manonica_world::magic::SpellEvent> {
    game.cast_scroll(
        registry,
        player,
        0,
        [6.5, 6.5, 6.5],
        [6.5, 6.5, 6.5],
        [0., 0., -1.],
    )
}
fn advance(game: &mut GameRuntime, seconds: f64) {
    for _ in 0..(seconds / 0.025).round() as usize {
        game.advance_magic(0.025).unwrap();
    }
}
fn audit(game: &GameRuntime, before: f64) {
    assert!(
        (game.world.statistics().free_manons + game.magic().bound_manons() - before).abs() < 1e-8
    );
    assert!(game.magic().energy_residual_joules().abs() < 1e-7);
}
#[test]
fn compression_release_is_manon_transport_not_a_world_edit() {
    let v = recipe(0);
    let (mut game, registry, player) = setup(&v, 3.0);
    let before = game.world.statistics().free_manons;
    let blocks = game.world.occupied_blocks().count();
    cast(&mut game, &registry, player).unwrap();
    advance(&mut game, 0.8);
    let compressed = game.magic().active()[0]
        .field
        .densities
        .iter()
        .copied()
        .fold(0., f64::max);
    assert!(compressed > 3.1);
    audit(&game, before);
    advance(&mut game, 0.6);
    let released = game.magic().active()[0]
        .field
        .densities
        .iter()
        .copied()
        .fold(0., f64::max);
    assert!(released < compressed);
    assert!(
        game.magic().active()[0]
            .field
            .potential
            .iter()
            .all(|v| *v == 0.)
    );
    advance(&mut game, 1.8);
    assert!(game.magic().active().is_empty());
    audit(&game, before);
    assert_eq!(game.world.occupied_blocks().count(), blocks);
}
#[test]
fn recipes_drive_the_same_solver_without_spell_name_dispatch() {
    let a = recipe(2);
    let mut b = a.clone();
    b["key"] = serde_json::json!("custom:arbitrary_formula_name");
    let (mut ga, ra, pa) = setup(&a, 3.);
    let (mut gb, rb, pb) = setup(&b, 3.);
    cast(&mut ga, &ra, pa).unwrap();
    cast(&mut gb, &rb, pb).unwrap();
    advance(&mut ga, 0.3);
    advance(&mut gb, 0.3);
    assert_eq!(
        ga.magic().active()[0].field.densities,
        gb.magic().active()[0].field.densities
    );
    assert!(
        ga.magic().active()[0]
            .field
            .velocity
            .iter()
            .any(|v| v[0].abs() > 0.01 && v[2].abs() > 0.01)
    );
}
#[test]
fn no_op_has_no_transport_no_heat_and_no_work() {
    let mut v = recipe(0);
    v["program"]["controls"] = serde_json::json!({"potential_source":"0","manon_velocity":{"x":"0","y":"0","z":"0"},"heat_power":"0"});
    v["program"]["work"]["total_joules"] = serde_json::json!(0.);
    let (mut game, r, p) = setup(&v, 3.);
    let before = game.world.statistics().free_manons;
    cast(&mut game, &r, p).unwrap();
    advance(&mut game, 0.2);
    let e = &game.magic().active()[0];
    assert!(e.field.densities.iter().all(|n| *n == 3.));
    assert_eq!(e.field.energy.external_work_joules, 0.);
    assert_eq!(e.field.energy.heat_joules, 0.);
    audit(&game, before);
}
#[test]
fn invalid_or_unaffordable_formulas_do_not_consume_items_or_manons() {
    let mut v = recipe(0);
    v["program"]["work"]["total_joules"] = serde_json::json!(0.);
    let (mut game, r, p) = setup(&v, 3.);
    let before = game.world.statistics().free_manons;
    assert!(cast(&mut game, &r, p).is_err());
    assert!(game.magic().active().is_empty());
    assert_eq!(
        game.entity(p)
            .unwrap()
            .inventory
            .get(0)
            .unwrap()
            .unwrap()
            .count(),
        2
    );
    assert_eq!(game.world.statistics().free_manons, before);
    v["program"]["controls"]["potential_source"] = serde_json::json!("remove_block(x)");
    assert!(SpellRegistry::from_jsonl(&v.to_string(), &content()).is_err());
    v["program"]["controls"]["potential_source"] = serde_json::json!("block_id");
    assert!(SpellRegistry::from_jsonl(&v.to_string(), &content()).is_err());
}
#[test]
fn active_program_and_energy_restore_and_overlapping_domains_are_rejected() {
    let v = recipe(1);
    let (mut game, r, p) = setup(&v, 3.);
    let before = game.world.statistics().free_manons;
    cast(&mut game, &r, p).unwrap();
    assert!(cast(&mut game, &r, p).is_err());
    advance(&mut game, 0.15);
    let mut restored = GameRuntime::from_nbt(content(), &game.to_nbt().unwrap()).unwrap();
    advance(&mut game, 0.1);
    advance(&mut restored, 0.1);
    assert_eq!(
        game.magic().active()[0].field.densities,
        restored.magic().active()[0].field.densities
    );
    audit(&restored, before);
}
#[test]
fn formula_constructions_are_deterministic_and_respond_to_semantics() {
    use manonica_world::magic::{ManonProgram, construction::CircleGrammar};
    let grammar =
        CircleGrammar::from_jsonl(include_str!("../../../assets/demo/circles.jsonl")).unwrap();
    let v = recipe(0);
    let program: ManonProgram = serde_json::from_value(v["program"].clone()).unwrap();
    let a = grammar.compile(&program).unwrap();
    let b = grammar.compile(&program).unwrap();
    assert_eq!(a, b);
    assert!(!a.links.is_empty());
    assert!(a.nodes.iter().any(|n| n.symbol == "function:exp"));
    assert!(a.paths.iter().flatten().any(|p| p[1] != 0.0));
    let mut changed = program.clone();
    changed.controls.potential_source.0 = changed.controls.potential_source.0.replace("160", "80");
    let c = grammar.compile(&changed).unwrap();
    assert_ne!(a.fingerprint, c.fingerprint);
    changed = program.clone();
    changed.controls.potential_source.0 = format!(" ( {} ) ", program.controls.potential_source.0);
    assert_eq!(grammar.compile(&changed).unwrap(), a);
    let other: ManonProgram = serde_json::from_value(recipe(2)["program"].clone()).unwrap();
    assert_ne!(grammar.compile(&other).unwrap().fingerprint, a.fingerprint);
}
#[test]
fn visibility_queries_hide_background_and_reveal_actual_concentration() {
    use manonica_world::chunk::CellPosition;
    let (mut game, _, _) = setup(&recipe(0), 3.0);
    assert!(game.world.concentrations_above(7.5, 1024).is_empty());
    let p = CellPosition { x: 6, y: 6, z: 6 };
    game.world.set_density(p, 30.0).unwrap();
    assert_eq!(game.world.concentrations_above(7.5, 1024), vec![(p, 30.0)]);
}
#[test]
fn object_commands_and_nonfinite_evaluations_are_rejected() {
    let mut v = recipe(0);
    v["program"]["destroy_blocks"] = serde_json::json!(true);
    assert!(SpellRegistry::from_jsonl(&v.to_string(), &content()).is_err());
    v["program"]
        .as_object_mut()
        .unwrap()
        .remove("destroy_blocks");
    v["program"]["controls"]["potential_source"] = serde_json::json!("sqrt(-1)");
    let (mut game, r, p) = setup(&v, 3.);
    let before = game.world.statistics().free_manons;
    assert!(cast(&mut game, &r, p).is_err());
    assert_eq!(game.world.statistics().free_manons, before);
    assert_eq!(
        game.entity(p)
            .unwrap()
            .inventory
            .get(0)
            .unwrap()
            .unwrap()
            .count(),
        2
    );
}
