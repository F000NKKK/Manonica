use super::{AmbientFlow, Circle, Experiment, ExperimentBuilder, Visualization};
use crate::simulation::Simulation;

#[test]
fn builder_and_parser_share_same_rules() {
    let model = ExperimentBuilder::new("lab")
        .grid([10, 10, 10], 1.0)
        .diffusion(0.3, 0.1, 2)
        .gaussian([0.0, 0.0, 0.0], 100.0, 0.9)
        .circle(Circle {
            name: "test".into(),
            center: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            radius: 2.0,
            thickness: 0.04,
            spokes: 8,
            segments: 64,
            color: [0.3, 0.7, 1.0],
        })
        .build()
        .unwrap();
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
        .build()
        .unwrap();
    let mut sim = Simulation::new(exp).unwrap();
    let before = sim.total().unwrap().value();
    assert!((before - 1000.0).abs() < 1e-8, "{before}");
    for _ in 0..150 {
        sim.step().unwrap();
    }
    assert!((sim.total().unwrap().value() - before).abs() < 1e-7);
    sim.reset().unwrap();
    assert_eq!(sim.steps(), 0);
    assert!((sim.total().unwrap().value() - before).abs() < 1e-8);
}

#[test]
fn validates_unstable_timestep_and_visualization() {
    let invalid = ExperimentBuilder::new("bad")
        .grid([16; 3], 1.0)
        .diffusion(1.0, 1.0, 1)
        .build();
    assert!(invalid.is_err());
    let invalid = ExperimentBuilder::new("bad")
        .visualization(Visualization {
            relative_threshold: 2.0,
            ..Visualization::default()
        })
        .build();
    assert!(invalid.is_err());
}

#[test]
fn true_advection_displaces_mass_without_creating_manons() {
    let exp = ExperimentBuilder::new("advected")
        .grid([18, 12, 12], 0.6)
        .diffusion(0.0, 0.04, 1)
        .flow(AmbientFlow::Uniform {
            velocity: [0.7, 0.0, 0.0],
        })
        .gaussian([-2.0, 0.0, 0.0], 1500.0, 0.7)
        .build()
        .unwrap();
    let mut simulation = Simulation::new(exp).unwrap();
    let initial = simulation.total().unwrap().value();
    let centroid = |sim: &Simulation| {
        let shape = sim.field().shape();
        let dx = sim.field().cell_width_metres();
        let mut mass = 0.0;
        let mut weighted_x = 0.0;
        for z in 0..shape.nz() {
            for y in 0..shape.ny() {
                for x in 0..shape.nx() {
                    let n = sim.field().at(x, y, z).unwrap().value();
                    mass += n;
                    weighted_x += n * (x as f64 + 0.5 - shape.nx() as f64 * 0.5) * dx;
                }
            }
        }
        weighted_x / mass
    };
    let before = centroid(&simulation);
    for _ in 0..40 {
        simulation.step().unwrap();
    }
    assert!(
        centroid(&simulation) > before + 0.7,
        "cloud must move in +x"
    );
    assert!((simulation.total().unwrap().value() - initial).abs() < 1e-7);
}

#[test]
fn excessive_flow_is_rejected_before_rendering() {
    let invalid = ExperimentBuilder::new("unstable")
        .grid([4, 4, 4], 1.0)
        .diffusion(0.0, 1.0, 1)
        .flow(AmbientFlow::Uniform {
            velocity: [9.0, 0.0, 0.0],
        })
        .gaussian([0.0; 3], 50.0, 1.0)
        .build()
        .unwrap();
    assert!(Simulation::new(invalid).is_err());
}

#[test]
fn rejects_nonphysical_optical_settings() {
    let invalid = ExperimentBuilder::new("opacity")
        .visualization(Visualization {
            optical_density: f64::NAN,
            ..Visualization::default()
        })
        .build();
    assert!(invalid.is_err());

    let invalid = ExperimentBuilder::new("opacity")
        .visualization(Visualization {
            optical_density: -0.5,
            ..Visualization::default()
        })
        .build();
    assert!(invalid.is_err());
}

#[test]
fn crystal_trap_and_channel_conserve_total_inventory() {
    use super::{MaterialSite, ChannelSpec};
    let mut experiment = ExperimentBuilder::new("materials")
        .grid([2, 1, 1], 1.0)
        .diffusion(0.0, 0.1, 1)
        .gaussian([-0.5, 0.0, 0.0], 20.0, 0.3)
        .build()
        .unwrap();
    experiment.materials.push(MaterialSite {
        cell: [0, 0, 0], capacity: 8.0, capture_rate: 4.0,
        release_rate: 0.05, binding_energy: 1.5,
        heat_capacity: 10.0, reference_temperature: 290.0,
        activation_temperature: 100.0, max_flux: 3.0,
        conductivity: 1.0,
    });
    experiment.channels.push(ChannelSpec {
        from: [0, 0, 0], to: [1, 0, 0],
        length: 2.0, area: 1.0, conductance: 1.0, max_flux: 5.0,
    });
    let mut sim = Simulation::new(experiment).unwrap();
    let initial = sim.total().unwrap().value();
    let initial_energy = sim.material_balance().unwrap().energy();
    for _ in 0..50 { sim.step().unwrap(); }
    let balance = sim.material_balance().unwrap();
    assert!(balance.bound > 0.0 && balance.bound <= 8.0);
    assert!((balance.manons() - initial).abs() < 1e-7);
    assert!((balance.energy() - initial_energy).abs() < 1e-7);
    assert!(sim.field().at(1, 0, 0).unwrap().value() > 0.0);
    sim.reset().unwrap();
    assert_eq!(sim.material_balance().unwrap().bound, 0.0);
}
