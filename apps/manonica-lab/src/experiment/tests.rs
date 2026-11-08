use super::{Circle, Experiment, ExperimentBuilder, Visualization};
use crate::simulation::Simulation;

#[test]
fn builder_and_parser_share_same_rules() {
    let model = ExperimentBuilder::new("lab")
        .grid([10, 10, 10], 1.0)
        .diffusion(0.3, 0.1, 2)
        .gaussian([0.0, 0.0, 0.0], 100.0, 0.9)
        .circle(Circle {
            name: "test".into(), center: [0.0; 3], normal: [0.0, 1.0, 0.0],
            radius: 2.0, thickness: 0.04, spokes: 8, segments: 64,
            color: [0.3, 0.7, 1.0],
        })
        .build().unwrap();
    assert_eq!(model.circles.len(), 1);
    let text = include_str!("../../experiments/mana_flame.toml");
    let parsed = Experiment::parse(text).unwrap();
    assert!(!parsed.seeds.is_empty());
    assert!(!parsed.circles.is_empty());
}

#[test]
fn conserves_total_with_multiple_seeds() {
    let exp = ExperimentBuilder::new("balance")
        .grid([12, 12, 12], 0.5)
        .diffusion(0.25, 0.02, 1)
        .gaussian([0.0, 0.0, 0.0], 700.0, 0.7)
        .ring([0.0, 0.0, 0.0], 1.7, 0.25, 300.0)
        .build().unwrap();
    let mut sim = Simulation::new(exp).unwrap();
    let before = sim.total().unwrap().value();
    assert!((before - 1000.0).abs() < 1e-8, "{before}");
    for _ in 0..150 { sim.step().unwrap(); }
    assert!((sim.total().unwrap().value() - before).abs() < 1e-7);
    sim.reset().unwrap();
    assert_eq!(sim.steps(), 0);
    assert!((sim.total().unwrap().value() - before).abs() < 1e-8);
}

#[test]
fn validates_unstable_timestep_and_visualization() {
    let invalid = ExperimentBuilder::new("bad")
        .grid([16; 3], 1.0)
        .diffusion(1.0, 1.0, 1).build();
    assert!(invalid.is_err());
    let invalid = ExperimentBuilder::new("bad")
        .visualization(Visualization {
            relative_threshold: 2.0, ..Visualization::default()
        }).build();
    assert!(invalid.is_err());
}
