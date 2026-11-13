use super::{AmbientFlow, Experiment, ExperimentBuilder, Formula, Visualization};
use crate::simulation::Simulation;

#[test]
fn builder_and_parser_share_same_rules() {
    let model = ExperimentBuilder::new("lab")
        .grid([10, 10, 10], 1.0)
        .diffusion(0.3, 0.1, 2)
        .gaussian([0.0, 0.0, 0.0], 100.0, 0.9)
        .formula(Formula {
            name: "test".into(),
            x: "2*cos(t)".into(),
            y: "0".into(),
            z: "2*sin(t)".into(),
            thickness: 0.04,
            segments: 64,
            color: [0.3, 0.7, 1.0],
            trap: None,
        })
        .build()
        .unwrap();
    assert_eq!(model.formulas.len(), 1);
    let text = include_str!("../../experiments/mana_flame.toml");
    let parsed = Experiment::parse(text).unwrap();
    assert!(!parsed.seeds.is_empty());
    assert!(!parsed.formulas.is_empty());
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
    use super::{ChannelSpec, MaterialSite};
    let mut experiment = ExperimentBuilder::new("materials")
        .grid([2, 1, 1], 1.0)
        .diffusion(0.0, 0.1, 1)
        .gaussian([-0.5, 0.0, 0.0], 20.0, 0.3)
        .build()
        .unwrap();
    experiment.materials.push(MaterialSite {
        cell: [0, 0, 0],
        capacity: 8.0,
        capture_rate: 4.0,
        release_rate: 0.05,
        binding_energy: 1.5,
        heat_capacity: 10.0,
        reference_temperature: 290.0,
        activation_temperature: 100.0,
        max_flux: 3.0,
        conductivity: 1.0,
    });
    experiment.channels.push(ChannelSpec {
        from: [0, 0, 0],
        to: [1, 0, 0],
        length: 2.0,
        area: 1.0,
        conductance: 1.0,
        max_flux: 5.0,
        anisotropy: None,
    });
    let mut sim = Simulation::new(experiment).unwrap();
    let initial = sim.total().unwrap().value();
    let initial_energy = sim.material_balance().unwrap().energy();
    for _ in 0..50 {
        sim.step().unwrap();
    }
    let balance = sim.material_balance().unwrap();
    assert!(balance.bound > 0.0 && balance.bound <= 8.0);
    assert!((balance.manons() - initial).abs() < 1e-7);
    assert!((balance.energy() - initial_energy).abs() < 1e-7);
    assert!(sim.field().at(1, 0, 0).unwrap().value() > 0.0);
    sim.reset().unwrap();
    assert_eq!(sim.material_balance().unwrap().bound, 0.0);
}

#[test]
fn formula_constructs_real_capture_sites_and_conserves_inventory() {
    let text = include_str!("../../experiments/formula_capture.toml");
    let exp = Experiment::parse(text).unwrap();
    assert_eq!(exp.formulas.len(), 1);
    let mut sim = Simulation::new(exp).unwrap();
    let initial = sim.material_balance().unwrap();
    let initial_energy = sim.tracked_energy().unwrap();
    for _ in 0..30 {
        sim.step().unwrap();
    }
    let final_balance = sim.material_balance().unwrap();
    assert!(
        final_balance.bound > 0.0,
        "parametric ring must capture mana"
    );
    assert!((initial.manons() - final_balance.manons()).abs() < 1e-6);
    let supplied_work = sim
        .field_ledger()
        .map_or(0.0, |ledger| ledger.operating_work_joules);
    assert!((sim.tracked_energy().unwrap() - initial_energy - supplied_work).abs() < 1e-6);
    sim.reset().unwrap();
    assert_eq!(sim.material_balance().unwrap().bound, 0.0);
}

#[test]
fn three_formula_demo_has_distinct_geometry_and_independent_capture() {
    use super::sample_curve;
    let experiment = Experiment::parse(include_str!("../../experiments/formula_comparison.toml"))
        .expect("comparison experiment parses");
    assert_eq!(experiment.formulas.len(), 3);
    let shapes: Vec<_> = experiment
        .formulas
        .iter()
        .map(|formula| sample_curve(formula).unwrap())
        .collect();

    let center_x =
        |points: &Vec<[f32; 3]>| points.iter().map(|p| p[0]).sum::<f32>() / points.len() as f32;
    assert!(center_x(&shapes[0]) < -3.0);
    assert!(center_x(&shapes[1]).abs() < 0.5);
    assert!(center_x(&shapes[2]) > 3.0);
    // Circle and star lie in the XY plane, while the third has
    // a substantial physical depth extent.
    let depth_range = |points: &Vec<[f32; 3]>| {
        let min = points.iter().map(|p| p[2]).fold(f32::INFINITY, f32::min);
        let max = points
            .iter()
            .map(|p| p[2])
            .fold(f32::NEG_INFINITY, f32::max);
        max - min
    };
    assert!(depth_range(&shapes[0]) < 0.01);
    assert!(depth_range(&shapes[1]) < 0.01);
    assert!(depth_range(&shapes[2]) > 1.0);

    let mut simulation = Simulation::new(experiment).unwrap();
    let initial = simulation.material_balance().unwrap();
    let initial_energy = simulation.tracked_energy().unwrap();
    for _ in 0..40 {
        simulation.step().unwrap();
    }
    let captures = simulation.formula_bound().unwrap();
    assert_eq!(captures.len(), 3);
    assert!(captures.iter().all(|&amount| amount > 0.0), "{captures:?}");
    let final_balance = simulation.material_balance().unwrap();
    assert!((final_balance.manons() - initial.manons()).abs() < 1e-6);
    let supplied_work = simulation
        .field_ledger()
        .map_or(0.0, |ledger| ledger.operating_work_joules);
    assert!(
        (simulation.tracked_energy().unwrap() - initial_energy - supplied_work).abs() < 1e-6,
        "full closed-system energy balance includes field, exported heat and external work"
    );
}

#[test]
fn powered_potential_concentrator_changes_the_real_free_mana_distribution() {
    let experiment = Experiment::parse(include_str!(
        "../../experiments/potential_concentrator.toml"
    ))
    .expect("powered field configuration must parse");
    let mut control = experiment.clone();
    control.physics.potential.as_mut().unwrap().mobility = 0.0;
    let mut active = Simulation::new(experiment).unwrap();
    let mut passive = Simulation::new(control).unwrap();
    let initial_manons = active.total().unwrap().value();
    let initial_energy = active.tracked_energy().unwrap();
    let initial_work = active.field_ledger().unwrap().operating_work_joules;
    let field = active.potential().unwrap();
    assert!(field.peak_absolute() > 0.0);
    assert!(field.field_energy_joules() > 0.0);
    for _ in 0..30 {
        active.step().unwrap();
        passive.step().unwrap();
    }
    let profile_difference: f64 = active
        .field()
        .densities()
        .iter()
        .zip(passive.field().densities())
        .map(|(a, b)| (a.value() - b.value()).abs())
        .sum();
    assert!(
        profile_difference > 0.1,
        "potential must change actual field, not mesh"
    );
    assert!((active.total().unwrap().value() - initial_manons).abs() < 1e-6);
    let work = active.field_ledger().unwrap().operating_work_joules - initial_work;
    assert!((active.tracked_energy().unwrap() - initial_energy - work).abs() < 1e-5);
}

#[test]
fn potential_source_requires_explicit_field_installation_budget() {
    let mut experiment = Experiment::parse(include_str!(
        "../../experiments/potential_concentrator.toml"
    ))
    .unwrap();
    experiment
        .physics
        .potential
        .as_mut()
        .unwrap()
        .initial_work_joules = 0.0;
    assert!(Simulation::new(experiment).is_err());
}

#[test]
fn changing_source_formula_changes_potential_distribution() {
    let mut experiment = Experiment::parse(include_str!(
        "../../experiments/potential_concentrator.toml"
    ))
    .unwrap();
    let first = Simulation::new(experiment.clone()).unwrap();
    let centre = first.potential().unwrap().at(16, 12, 16).unwrap();
    experiment.formulas[0].x = "0.8*cos(t)".into();
    experiment.formulas[0].z = "0.8*sin(t)".into();
    let second = Simulation::new(experiment).unwrap();
    let new_centre = second.potential().unwrap().at(16, 12, 16).unwrap();
    assert!(
        (centre - new_centre).abs() > 1e-5,
        "same formula geometry must drive both render and potential source"
    );
}

#[test]
fn saturation_triggers_physical_potential_shutdown_and_energy_audit() {
    let mut experiment = Experiment::parse(include_str!(
        "../../experiments/potential_concentrator.toml"
    ))
    .unwrap();
    let settings = experiment.physics.potential.as_mut().unwrap();
    // Trigger early for a short deterministic regression. The actual demo
    // switches off only once 99.5% of installed capacity is bound.
    settings.shutdown_at_bound_fraction = 0.001;
    settings.shutdown_decay_rate = 0.1;
    let mut simulation = Simulation::new(experiment).unwrap();
    let initial = simulation.tracked_energy().unwrap();
    let count = simulation.total().unwrap().value();
    let initial_peak = simulation.field_ledger().unwrap().peak_potential;
    let mut triggered = false;
    for _ in 0..120 {
        simulation.step().unwrap();
        triggered |= simulation.shutdown_active();
    }
    assert!(
        triggered,
        "material charge must trigger automatic power-down"
    );
    let ledger = simulation.field_ledger().unwrap();
    assert!(ledger.shutdown);
    assert!(ledger.peak_potential < initial_peak);
    assert!(ledger.exported_heat_joules > 0.0);
    assert!(
        simulation
            .peak_trap_temperature_kelvin()
            .unwrap()
            .is_finite()
    );
    assert!((simulation.total().unwrap().value() - count).abs() < 1e-6);
    assert!(
        (simulation.tracked_energy().unwrap() - initial - ledger.operating_work_joules).abs()
            < 1e-4,
        "closed ledger must account for exported heat and supplied work"
    );
}

#[test]
fn physical_equilibrium_is_observed_only_after_consecutive_stable_solver_steps() {
    use crate::simulation::EquilibriumPhase;
    let experiment = ExperimentBuilder::new("equilibrium")
        .grid([4, 4, 4], 0.5)
        .diffusion(0.12, 0.01, 1)
        .build()
        .unwrap();
    let mut sim = Simulation::new(experiment).unwrap();
    assert_ne!(sim.equilibrium_state().phase, EquilibriumPhase::Equilibrium);
    for _ in 0..40 {
        sim.step().unwrap();
        assert_ne!(sim.equilibrium_state().phase, EquilibriumPhase::Equilibrium);
    }
    sim.step().unwrap();
    assert_eq!(sim.equilibrium_state().phase, EquilibriumPhase::Equilibrium);
    assert!(sim.equilibrium_state().density_rate.abs() < 1e-12);
    sim.reset().unwrap();
    assert_ne!(sim.equilibrium_state().phase, EquilibriumPhase::Equilibrium);
}

#[test]
fn powered_concentrator_never_reports_equilibrium_while_field_source_is_active() {
    use crate::simulation::EquilibriumPhase;
    let experiment = Experiment::parse(include_str!(
        "../../experiments/potential_concentrator.toml"
    ))
    .unwrap();
    let mut sim = Simulation::new(experiment).unwrap();
    for _ in 0..30 {
        sim.step().unwrap();
    }
    assert_eq!(sim.equilibrium_state().phase, EquilibriumPhase::Charging);
    assert!(sim.equilibrium_state().potential_peak > 0.1);
}

#[test]
fn driven_uniform_flow_is_not_mistaken_for_equilibrium() {
    use crate::simulation::EquilibriumPhase;
    let experiment = ExperimentBuilder::new("driven-flow")
        .grid([4, 4, 4], 1.0)
        .diffusion(0.0, 0.02, 1)
        .flow(AmbientFlow::Uniform {
            velocity: [0.1, 0.0, 0.0],
        })
        .build()
        .unwrap();
    let mut sim = Simulation::new(experiment).unwrap();
    for _ in 0..50 {
        sim.step().unwrap();
    }
    assert_eq!(sim.equilibrium_state().phase, EquilibriumPhase::Charging);
    assert_eq!(sim.equilibrium_state().stable_steps, 0);
}

#[test]
fn formula_rasterization_installs_real_local_boundary_potential() {
    let constrained = Experiment::parse(include_str!(
        "../../experiments/local_boundary_conductor.toml"
    ))
    .unwrap();
    assert_eq!(constrained.local_boundaries.len(), 1);
    let mut baseline = constrained.clone();
    baseline.local_boundaries.clear();
    let constrained_sim = Simulation::new(constrained).unwrap();
    let baseline_sim = Simulation::new(baseline).unwrap();
    let a = constrained_sim.potential().unwrap();
    let b = baseline_sim.potential().unwrap();
    let shape = a.shape();
    let dx = constrained_sim.field().cell_width_metres();
    let formula = &constrained_sim.experiment.formulas[0];
    let cells = super::capture_cells(formula, [shape.nx(), shape.ny(), shape.nz()], dx).unwrap();
    assert!(!cells.is_empty());
    for [x, y, z] in cells {
        assert!(
            (a.at(x, y, z).unwrap() + 4.0).abs() < 1e-9,
            "fixed local potential is imposed at every sampled circuit cell"
        );
    }
    let deviation = a
        .values()
        .iter()
        .zip(b.values())
        .map(|(x, y)| (x - y).abs())
        .sum::<f64>();
    assert!(
        deviation > 1.0,
        "local condition must alter physics, not just rendered mesh"
    );
}

#[test]
fn formula_boundary_overlap_and_bad_parameters_are_rejected() {
    use super::LocalBoundarySpec;
    let mut experiment = Experiment::parse(include_str!(
        "../../experiments/local_boundary_conductor.toml"
    ))
    .unwrap();
    experiment
        .local_boundaries
        .push(experiment.local_boundaries[0].clone());
    assert!(Simulation::new(experiment.clone()).is_err());
    experiment.local_boundaries.truncate(1);
    experiment.local_boundaries[0] = LocalBoundarySpec::Robin {
        formula: 0,
        reference: 0.0,
        exchange: -1.0,
    };
    assert!(experiment.validate().is_err());
}

#[test]
fn resident_batch_matches_reference_diffusion_only_and_tracks_time() {
    use manonica_core::{Evolution, Quantity, TimeStep, units::Diffusivity};
    use manonica_error::BaseError;
    use manonica_physics::{FreeDiffusion, FreeManaDiffusionBackend, FreeManaField};

    struct CpuReference(FreeDiffusion);
    impl FreeManaDiffusionBackend for CpuReference {
        fn evolve_free_diffusion(
            &mut self,
            field: &mut FreeManaField,
            _diffusivity: f64,
            dt_seconds: f64,
        ) -> Result<(), BaseError> {
            self.0
                .evolve(field, TimeStep::from_seconds(dt_seconds)?)
                .map_err(|_| {
                    BaseError::new(
                        manonica_error::ErrorCode::NumericalInstability,
                        "cpu_reference_diffusion",
                    )
                })?;
            Ok(())
        }
    }

    let experiment = ExperimentBuilder::new("GPU-resident-validity")
        .grid([8, 8, 8], 0.5)
        .diffusion(0.12, 0.01, 4)
        .gaussian([0.0, 0.0, 0.0], 500.0, 0.7)
        .build()
        .unwrap();
    let mut expected = Simulation::new(experiment.clone()).unwrap();
    let mut actual = Simulation::new(experiment).unwrap();
    assert!(actual.supports_resident_diffusion());
    let mut backend = CpuReference(
        FreeDiffusion::new(Quantity::<Diffusivity>::nonnegative(0.12).unwrap()).unwrap(),
    );
    let original = actual.total().unwrap().value();
    for _ in 0..20 {
        expected.step().unwrap();
    }
    actual.step_diffusion_batch(&mut backend, 20).unwrap();
    assert_eq!(actual.steps(), expected.steps());
    assert!((actual.elapsed() - expected.elapsed()).abs() < 1e-10);
    for (i, (a, b)) in actual
        .field()
        .densities()
        .iter()
        .zip(expected.field().densities())
        .enumerate()
    {
        assert!(
            (a.value() - b.value()).abs() < 1e-10,
            "batch and reference mismatch at cell {i}"
        );
    }
    assert!((actual.total().unwrap().value() - original).abs() < 1e-7);
    // One batch = one physical observation, never 20 fake stable samples.
    assert!(actual.equilibrium_state().stable_steps <= 1);
}

#[test]
fn resident_batch_rejects_multiphysics_without_mutating_state() {
    use manonica_error::BaseError;
    use manonica_physics::{FreeManaDiffusionBackend, FreeManaField};
    struct Noop;
    impl FreeManaDiffusionBackend for Noop {
        fn evolve_free_diffusion(
            &mut self,
            _field: &mut FreeManaField,
            _diffusivity: f64,
            _dt: f64,
        ) -> Result<(), BaseError> {
            Ok(())
        }
    }
    let exp = Experiment::parse(include_str!(
        "../../experiments/potential_concentrator.toml"
    ))
    .unwrap();
    let mut simulation = Simulation::new(exp).unwrap();
    assert!(!simulation.supports_resident_diffusion());
    let before = simulation.total().unwrap().value();
    assert!(simulation.step_diffusion_batch(&mut Noop, 5).is_err());
    assert_eq!(simulation.steps(), 0);
    assert!((simulation.total().unwrap().value() - before).abs() < 1e-10);
}

/// LAW-FIELD-SETUP-001: a one-cell insulated ring source of integrated strength
/// S=2V and screening=1 gives phi=2, E_field=2V, U_manons=6.
/// The setup budget covers field energy; manon energy is initial state.
#[test]
fn initial_field_budget_matches_analytic_one_cell_source_and_volume() {
    for dx in [0.5_f64, 1.0] {
        let volume = dx.powi(3);
        let expected_field_energy = 2.0 * volume;
        let mut experiment = Experiment::parse(include_str!(
            "../../experiments/local_boundary_conductor.toml"
        ))
        .unwrap();
        experiment.grid.dimensions = [1, 1, 1];
        experiment.grid.cell_size = dx;
        experiment.physics.diffusivity = 0.0;
        experiment.physics.flow = AmbientFlow::Still;
        experiment.local_boundaries.clear();
        experiment.formulas[0].x = "0.1*cos(t)".into();
        experiment.formulas[0].y = "0*t".into();
        experiment.formulas[0].z = "0.1*sin(t)".into();
        experiment.formulas[0].trap = None;
        experiment.seeds[0].total_manons = 3.0;
        experiment.field_sources[0].strength = 2.0 * volume;
        let settings = experiment.physics.potential.as_mut().unwrap();
        settings.boundary = super::PotentialBoundaryKind::Insulated;
        settings.screening = 1.0;
        settings.initial_work_joules = expected_field_energy;
        let mut simulation = Simulation::new(experiment.clone()).unwrap();
        let ledger = simulation.field_ledger().unwrap();
        assert!((ledger.field_energy_joules - expected_field_energy).abs() < 1e-10);
        assert!((ledger.manon_potential_joules - 6.0).abs() < 1e-10);
        assert_eq!(ledger.operating_work_joules, 0.0);
        assert!((simulation.tracked_energy().unwrap() - expected_field_energy - 6.0).abs() < 1e-10);
        simulation.reset().unwrap();
        assert!((simulation.tracked_energy().unwrap() - expected_field_energy - 6.0).abs() < 1e-10);
        experiment
            .physics
            .potential
            .as_mut()
            .unwrap()
            .initial_work_joules = expected_field_energy - 0.001;
        assert!(Simulation::new(experiment).is_err());
    }
}

/// LAW-LAB-COUPLED-001: a ring crosses all four cells of a 2x1x2 grid.
/// Uniform source S=8 gives rho=2, phi=2 and E_field=8. Symmetric seeds
/// give N=8 per cell; total trap capacity 32 gives C=8 per cell.
/// Each cell follows the independently solved joint capture/release ODE.
#[test]
fn multicell_formula_source_and_traps_converge_to_joint_kinetics() {
    let total = 8.0;
    let capacity = 8.0;
    let alpha = 0.05;
    let release = 0.7;
    let sum: f64 = total + capacity + release / alpha;
    let hi = (sum + (sum * sum - 4.0 * total * capacity).sqrt()) / 2.0;
    let lo = total * capacity / hi;
    let q = lo / hi * (-alpha * (hi - lo)).exp();
    let exact_bound = (lo - q * hi) / (1.0 - q);
    for segments in [8, 64, 256] {
        let mut errors = Vec::new();
        for steps in [20, 40, 80, 160] {
            let mut experiment = Experiment::parse(include_str!(
                "../../experiments/local_boundary_conductor.toml"
            ))
            .unwrap();
            experiment.grid.dimensions = [2, 1, 2];
            experiment.grid.cell_size = 1.0;
            experiment.physics.diffusivity = 0.0;
            experiment.physics.dt = 1.0 / steps as f64;
            experiment.physics.flow = AmbientFlow::Still;
            experiment.physics.thermal_conductance = 0.0;
            experiment.physics.ambient_cooling_rate = 0.0;
            experiment.local_boundaries.clear();
            experiment.seeds[0].total_manons = 32.0;
            let formula = &mut experiment.formulas[0];
            formula.x = "0.4*cos(t)".into();
            formula.y = "0".into();
            formula.z = "0.4*sin(t)".into();
            formula.segments = segments;
            formula.trap = Some(super::FormulaTrap {
                capacity: 32.0,
                capture_rate: alpha,
                release_rate: release,
                binding_energy: 2.0,
                heat_capacity: 20.0,
                reference_temperature: 300.0,
                activation_temperature: 0.0,
                max_flux: 1000.0,
            });
            // Geometry reference is explicit, rather than generated by the rasterizer.
            assert_eq!(
                super::capture_cells(formula, [2, 1, 2], 1.0).unwrap(),
                vec![[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]]
            );
            experiment.field_sources[0].strength = 8.0;
            let settings = experiment.physics.potential.as_mut().unwrap();
            settings.screening = 1.0;
            settings.boundary = super::PotentialBoundaryKind::Insulated;
            settings.initial_work_joules = 8.0;
            settings.max_power_joules_per_second = 0.0;
            settings.shutdown_at_bound_fraction = 0.0;
            let mut simulation = Simulation::new(experiment).unwrap();
            for value in simulation.potential().unwrap().values() {
                assert!((*value - 2.0).abs() < 1e-10);
            }
            assert!((simulation.field_ledger().unwrap().field_energy_joules - 8.0).abs() < 1e-10);
            for _ in 0..steps {
                simulation.step().unwrap();
                let balance = simulation.material_balance().unwrap();
                assert!((balance.manons() - 32.0).abs() < 1e-10);
                assert!(balance.energy().abs() < 1e-10);
                assert!((simulation.tracked_energy().unwrap() - 72.0).abs() < 1e-9);
                assert!(
                    simulation
                        .field_ledger()
                        .unwrap()
                        .operating_work_joules
                        .abs()
                        < 1e-10
                );
                // Symmetry plus inventory specifies every free cell independently.
                for value in simulation.field().densities() {
                    assert!((value.value() - (8.0 - balance.bound / 4.0)).abs() < 1e-10);
                }
                assert!((balance.thermal_joules - 2.0 * balance.bound).abs() < 1e-10);
                assert!(
                    (simulation.peak_trap_temperature_kelvin().unwrap()
                        - (300.0 + balance.thermal_joules / 20.0))
                        .abs()
                        < 1e-10
                );
            }
            let bound = simulation.formula_bound().unwrap()[0];
            errors.push((bound / 4.0 - exact_bound).abs());
            assert_eq!(simulation.steps(), steps as u64);
            assert!((simulation.elapsed() - 1.0).abs() < 1e-12);
            simulation.reset().unwrap();
            assert_eq!(simulation.formula_bound().unwrap()[0], 0.0);
            assert!((simulation.tracked_energy().unwrap() - 72.0).abs() < 1e-9);
            assert_eq!(simulation.steps(), 0);
        }
        for pair in errors.windows(2) {
            assert!(
                (1.8..2.2).contains(&(pair[0] / pair[1])),
                "Lab joint errors: {errors:?}"
            );
        }
        assert!(errors[3] < 0.01, "Lab absolute bound error: {errors:?}");
    }
}

/// LAW-LAB-COUPLED-002: two disjoint formula traps at phi=[2,0].
/// Independent two-step reference, dx=1, dt=.1, initial free=[8,8]:
/// drift R=.05*n_left; capture=(4-b)*(1-exp(-.02*n_after_drift));
/// release=b*(1-exp(-.03)); binding heat=2*(capture-release);
/// heat exchange=.07*(Q_left/2-Q_right/5); cooling retains exp(-.03).
#[test]
fn heterogeneous_formula_traps_match_transport_reactions_and_heat_reference() {
    let mut experiment = Experiment::parse(include_str!(
        "../../experiments/local_boundary_conductor.toml"
    ))
    .unwrap();
    experiment.grid.dimensions = [2, 1, 1];
    experiment.grid.cell_size = 1.0;
    experiment.physics.dt = 0.1;
    experiment.physics.diffusivity = 0.0;
    experiment.physics.flow = AmbientFlow::Still;
    experiment.physics.thermal_conductance = 0.7;
    experiment.physics.ambient_cooling_rate = 0.3;
    experiment.field_sources.clear();
    experiment.seeds[0].total_manons = 16.0;
    let mut left = experiment.formulas[0].clone();
    left.x = "-0.5+0.1*cos(t)".into();
    left.y = "0.1*sin(t)".into();
    left.z = "0".into();
    left.trap = Some(super::FormulaTrap {
        capacity: 4.0,
        capture_rate: 0.2,
        release_rate: 0.3,
        binding_energy: 2.0,
        heat_capacity: 2.0,
        reference_temperature: 300.0,
        activation_temperature: 0.0,
        max_flux: 100.0,
    });
    let mut right = left.clone();
    right.name = "right".into();
    right.x = "0.5+0.1*cos(t)".into();
    right.trap.as_mut().unwrap().heat_capacity = 5.0;
    assert_eq!(
        super::capture_cells(&left, [2, 1, 1], 1.0).unwrap(),
        vec![[0, 0, 0]]
    );
    assert_eq!(
        super::capture_cells(&right, [2, 1, 1], 1.0).unwrap(),
        vec![[1, 0, 0]]
    );
    experiment.formulas = vec![left, right];
    experiment.local_boundaries = vec![
        super::LocalBoundarySpec::Dirichlet {
            formula: 0,
            value: 2.0,
        },
        super::LocalBoundarySpec::Dirichlet {
            formula: 1,
            value: 0.0,
        },
    ];
    let settings = experiment.physics.potential.as_mut().unwrap();
    settings.mobility = 0.25;
    settings.screening = 1.0;
    settings.permittivity = 1.0;
    settings.boundary = super::PotentialBoundaryKind::Insulated;
    settings.initial_work_joules = 4.0;
    settings.max_power_joules_per_second = 0.0;
    settings.shutdown_at_bound_fraction = 0.0;
    let mut simulation = Simulation::new(experiment).unwrap();
    assert!((simulation.tracked_energy().unwrap() - 20.0).abs() < 1e-10);
    // Reference constants come directly from the equations above, without
    // invoking the production operators or reading the simulation output.
    let reference = [
        (
            [7.035953122964493, 7.781415338738635],
            [0.5640468770355063, 0.618584661261365],
            [1.0732456471910026, 1.2221133411565976],
            0.7763564268388066,
            0.09354766140733563,
            0.4,
        ),
        (
            [6.27087533547864, 7.643864954969089],
            [0.9773270083731348, 1.1079327011791353],
            [1.8090308436635205, 2.1703929889685494],
            1.436212555111156,
            0.25847834365776323,
            0.35179765614822467,
        ),
    ];
    for (free, bound, heat, field_heat, exported, transported) in reference {
        simulation.step().unwrap();
        let state = simulation.coupled_state();
        for i in 0..2 {
            assert!((state.field.at(i, 0, 0).unwrap().value() - free[i]).abs() < 1e-10);
            assert!((state.materials.cells()[i].bound - bound[i]).abs() < 1e-10);
            assert!((state.materials.cells()[i].heat - heat[i]).abs() < 1e-10);
            assert!((simulation.formula_bound().unwrap()[i] - bound[i]).abs() < 1e-10);
        }
        let ledger = simulation.field_ledger().unwrap();
        assert!((ledger.manon_potential_joules - 2.0 * (free[0] + bound[0])).abs() < 1e-10);
        assert!((ledger.dissipated_heat_joules - field_heat).abs() < 1e-10);
        assert!((ledger.exported_heat_joules - exported).abs() < 1e-10);
        assert!((ledger.last_transport_manons - transported).abs() < 1e-10);
        assert!(ledger.operating_work_joules.abs() < 1e-10);
        assert!((simulation.material_balance().unwrap().manons() - 16.0).abs() < 1e-10);
        assert!((simulation.tracked_energy().unwrap() - 20.0).abs() < 1e-10);
    }
}
