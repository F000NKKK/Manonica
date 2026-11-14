use manonica_world::{
    chunk::{CellPosition, ChunkCoord, ChunkLayout},
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
fn base() -> (GameRuntime, manonica_world::runtime::EntityId) {
    let mut g = GameRuntime::new(content(), ChunkLayout::new([12, 16, 12], 1.).unwrap());
    g.world.create(ChunkCoord { x: 0, z: 0 }, 3., true).unwrap();
    let p = g.spawn(&key("manonica:player")).unwrap();
    g.give(p, 0, &key("manonica:scroll_explosion"), 2).unwrap();
    (g, p)
}
fn heating(duration: f64) -> serde_json::Value {
    let mut recipe: serde_json::Value = serde_json::from_str(
        include_str!("../../../assets/demo/spells.jsonl")
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    recipe["key"] = serde_json::json!("test:spatial_heating");
    recipe["program"]["domain"] = serde_json::json!({"dimensions":[5,5,5],"anchor":[0.5,0.5,0.5],"duration_seconds":duration});
    recipe["program"]["controls"] = serde_json::json!({"potential_source":"0","manon_velocity":{"x":"0","y":"0","z":"0"},"heat_power":"150000*exp(-r*r/0.6)"});
    recipe["program"]["transport"]["diffusivity"] = serde_json::json!(0.);
    recipe["program"]["transport"]["mobility"] = serde_json::json!(0.);
    recipe
}
#[test]
fn local_heat_produces_temperature_pressure_radiation_and_survives_the_program() {
    let (mut g, p) = base();
    let recipe = heating(0.3);
    let r = SpellRegistry::from_jsonl(&recipe.to_string(), &content()).unwrap();
    let center = [6.5, 6.5, 6.5];
    let initial = g.world.statistics().free_manons;
    g.cast_scroll(&r, p, 0, center, center, [0., 0., 1.])
        .unwrap();
    for _ in 0..8 {
        g.advance_magic(0.025).unwrap();
    }
    let e = &g.magic().active()[0];
    let mid = (2 * 5 + 2) * 5 + 2;
    let t = e.field.thermal.temperature(mid, 1.);
    assert!(t > 1000., "temperature {t}");
    assert!(e.field.thermal.temperature(0, 1.) < t);
    assert!(e.field.energy.radiated_joules > 0.);
    assert!(
        e.field
            .velocity
            .iter()
            .any(|v| v.iter().any(|v| v.abs() > 0.1))
    );
    assert!(g.coupled_energy_residual_joules().unwrap().abs() < 1e-6);
    for _ in 0..8 {
        g.advance_magic(0.025).unwrap();
    }
    assert!(g.magic().active().is_empty());
    assert!(!g.matter().medium_heat.is_empty());
    assert!((g.world.statistics().free_manons - initial).abs() < 1e-8);
    let q: f64 = g.matter().medium_heat.iter().map(|c| c.heat_joules).sum();
    let radiated = g.matter().energy.radiated_joules;
    for _ in 0..20 {
        g.advance_magic(0.025).unwrap();
    }
    assert!(
        g.matter()
            .medium_heat
            .iter()
            .map(|c| c.heat_joules)
            .sum::<f64>()
            < q
    );
    assert!(g.matter().energy.radiated_joules > radiated);
    assert!(g.coupled_energy_residual_joules().unwrap().abs() < 1e-6);
    // Re-reserving a hot region transfers its existing heat; it does not create new energy.
    g.cast_scroll(&r, p, 0, center, center, [0., 0., 1.])
        .unwrap();
    assert!(g.magic().active()[0].field.energy.received_heat_joules > 0.);
    assert!(g.coupled_energy_residual_joules().unwrap().abs() < 1e-6);
    let mut restored = GameRuntime::from_nbt(content(), &g.to_nbt().unwrap()).unwrap();
    assert!((restored.coupled_energy_residual_joules().unwrap()).abs() < 1e-6);
    for _ in 0..16 {
        restored.advance_magic(0.025).unwrap();
    }
    assert!(restored.magic().active().is_empty());
    assert!(restored.coupled_energy_residual_joules().unwrap().abs() < 1e-6);
}
#[test]
fn heat_and_load_fragment_material_without_leaving_tiny_permanent_bodies() {
    let (mut g, p) = base();
    let target = CellPosition { x: 6, y: 6, z: 6 };
    g.world
        .set_block(
            target,
            &key("manonica:grass"),
            BTreeMap::new(),
            Default::default(),
        )
        .unwrap();
    let initial = g.world.occupied_blocks().count();
    let mut recipe = heating(0.8);
    recipe["program"]["controls"]["heat_power"] = serde_json::json!("500000*exp(-r*r/0.3)");
    let r = SpellRegistry::from_jsonl(&recipe.to_string(), &content()).unwrap();
    g.cast_scroll(&r, p, 0, [6.5, 6.5, 6.5], [6.5, 6.5, 6.5], [0., 0., 1.])
        .unwrap();
    let mut dust = false;
    for _ in 0..120 {
        g.advance_magic(0.025).unwrap();
        dust |= !g.matter().particles.is_empty();
        assert_eq!(
            g.world.occupied_blocks().count()
                + g.matter().bodies.len()
                + g.matter().dispersed_blocks as usize,
            initial
        );
        assert!(g.coupled_energy_residual_joules().unwrap().abs() < 1e-5);
    }
    assert!(
        g.matter().dispersed_blocks > 0,
        "no thermally fragmented material"
    );
    assert!(dust);
    for _ in 0..160 {
        g.advance_magic(0.025).unwrap();
    }
    assert!(g.matter().particles.is_empty());
}

#[test]
fn adaptive_prescribed_faces_agree_with_fine_density_dependent_path() {
    let run = |expression: &str| {
        let (mut g, p) = base();
        let mut recipe = heating(0.8);
        recipe["program"]["controls"]["heat_power"] = serde_json::json!("0");
        recipe["program"]["controls"]["manon_velocity"]["x"] = serde_json::json!(expression);
        let r = SpellRegistry::from_jsonl(&recipe.to_string(), &content()).unwrap();
        g.cast_scroll(&r, p, 0, [6.5, 6.5, 6.5], [6.5, 6.5, 6.5], [0., 0., 1.])
            .unwrap();
        for _ in 0..12 {
            g.advance_magic(0.025).unwrap();
        }
        g
    };
    let cached = run("2*sin(t)");
    let uncached = run("2*sin(t)+0*n");
    let a = &cached.magic().active()[0].field;
    let b = &uncached.magic().active()[0].field;
    let error = a
        .densities
        .iter()
        .zip(&b.densities)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    assert!(error < 0.02, "coarse/fine maximum density error {error}");
    assert!((a.manons() - b.manons()).abs() < 1e-9);
    assert!(a.densities.iter().all(|n| *n >= 0.));
    assert!(cached.coupled_energy_residual_joules().unwrap().abs() < 1e-6);
    assert!(uncached.coupled_energy_residual_joules().unwrap().abs() < 1e-6);
}
