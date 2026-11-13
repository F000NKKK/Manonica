//! Independent spatial reference for a formula-defined drift/reaction problem.
use super::*;

fn rhs(y: &[[f64; 2]], dx: f64) -> Vec<[f64; 2]> {
    let mut result: Vec<_> = y
        .iter()
        .map(|[n, b]| {
            let reaction = 0.2 * n * (4.0 - b) - 0.3 * b;
            [-reaction, reaction]
        })
        .collect();
    for face in 1..y.len() {
        // Exact continuum phi=cosh(x), velocity=-mobility*sinh(x).
        let flux = -0.2 * (face as f64 * dx).sinh() * y[face][0];
        result[face - 1][0] -= flux / dx;
        result[face][0] += flux / dx;
    }
    result
}

// Separate method-of-lines implementation, without Manonica's potential,
// rasterization, drift or reaction operators. RK4 controls the time error.
fn reference(cells: usize, steps: usize) -> Vec<[f64; 2]> {
    let dx = 1.0 / cells as f64;
    let mut y: Vec<_> = (0..cells)
        .map(|i| {
            let x = (i as f64 + 0.5) * dx - 0.5;
            [(-0.5 * (x / 0.15).powi(2)).exp(), 0.0]
        })
        .collect();
    let mean = y.iter().map(|v| v[0]).sum::<f64>() / cells as f64;
    for value in &mut y {
        value[0] *= 4.0 / mean;
    }
    let dt = 0.2 / steps as f64;
    for _ in 0..steps {
        let a = rhs(&y, dx);
        let stage = |derivative: &[[f64; 2]], fraction: f64| -> Vec<[f64; 2]> {
            y.iter()
                .zip(derivative)
                .map(|(v, d)| [v[0] + fraction * dt * d[0], v[1] + fraction * dt * d[1]])
                .collect()
        };
        let b = rhs(&stage(&a, 0.5), dx);
        let c = rhs(&stage(&b, 0.5), dx);
        let d = rhs(&stage(&c, 1.0), dx);
        for i in 0..cells {
            for j in 0..2 {
                y[i][j] += dt / 6.0 * (a[i][j] + 2.0 * b[i][j] + 2.0 * c[i][j] + d[i][j]);
            }
        }
    }
    assert!((y.iter().map(|v| v[0] + v[1]).sum::<f64>() * dx - 4.0).abs() < 1e-10);
    assert!(y.iter().all(|v| v[0] >= 0.0 && (0.0..=4.0).contains(&v[1])));
    y
}

fn restrict(reference: &[[f64; 2]], cells: usize) -> Vec<[f64; 2]> {
    let ratio = reference.len() / cells;
    assert_eq!(ratio * cells, reference.len());
    reference
        .chunks_exact(ratio)
        .map(|chunk| {
            std::array::from_fn(|j| chunk.iter().map(|v| v[j]).sum::<f64>() / ratio as f64)
        })
        .collect()
}

fn experiment(cells: usize) -> Experiment {
    let dx = 1.0 / cells as f64;
    let volume = dx.powi(3);
    let mut experiment = Experiment::parse(include_str!(
        "../../../experiments/local_boundary_conductor.toml"
    ))
    .unwrap();
    experiment.grid.dimensions = [cells, 1, 1];
    experiment.grid.cell_size = dx;
    experiment.physics.diffusivity = 0.0;
    experiment.physics.dt = 0.00005;
    experiment.physics.flow = AmbientFlow::Still;
    experiment.physics.thermal_conductance = 0.0;
    experiment.physics.ambient_cooling_rate = 0.0;
    experiment.field_sources.clear();
    experiment.seeds[0].sigma = 0.15;
    experiment.seeds[0].total_manons = 4.0 * cells as f64 * volume;
    let mut distributed = experiment.formulas[0].clone();
    distributed.x = format!("{}*cos(t)", 0.5 - 0.1 * dx);
    distributed.y = format!("{}*sin(t)", 0.1 * dx);
    distributed.z = "0".into();
    distributed.segments = 256;
    distributed.trap = Some(super::super::FormulaTrap {
        capacity: 4.0 * cells as f64 * volume,
        capture_rate: 0.2,
        release_rate: 0.3,
        binding_energy: 2.0,
        heat_capacity: cells as f64 * volume,
        reference_temperature: 300.0,
        activation_temperature: 0.0,
        max_flux: 1000.0,
    });
    let expected: Vec<_> = (0..cells).map(|i| [i, 0, 0]).collect();
    assert_eq!(
        super::super::capture_cells(&distributed, [cells, 1, 1], dx).unwrap(),
        expected
    );
    experiment.formulas = vec![distributed];
    experiment.local_boundaries.clear();
    experiment.physics.potential = None;
    experiment
}

fn measured(cells: usize, steps: usize) -> Vec<[f64; 2]> {
    let mut specification = experiment(cells);
    specification.physics.dt = 0.2 / steps as f64;
    use manonica_physics::{
        LocalBoundarySite, LocalPotentialBoundary, PotentialBoundary, PotentialField,
    };
    use manonica_simulation::{FieldControl, FieldLedger, MediumFlow, StepParameters};
    let simulation = Simulation::new(specification).unwrap();
    let mut state = simulation.coupled_state().clone();
    let dx = 1.0 / cells as f64;
    // Prescribed analytic profile, isolating nonlinear transport and geometry
    // from the independent elliptic convergence test LAW-POT-002.
    let constraints: Vec<_> = (0..cells)
        .map(|i| LocalBoundarySite {
            index: i,
            condition: LocalPotentialBoundary::Dirichlet {
                value: ((i as f64 + 0.5) * dx).cosh(),
            },
        })
        .collect();
    let potential = PotentialField::solve_with_local(
        state.field.shape(),
        dx,
        &vec![0.0; cells],
        1.0,
        1.0,
        PotentialBoundary::Insulated,
        1,
        &constraints,
    )
    .unwrap();
    state.field_ledger = Some(FieldLedger {
        field_energy_joules: potential.field_energy_joules(),
        manon_potential_joules: potential.manon_potential_energy(&state.field).unwrap(),
        peak_potential: potential.peak_absolute(),
        ..Default::default()
    });
    state.potential = Some(potential);
    let initial_energy = state.materials.balance(&state.field).unwrap().energy()
        + state.field_ledger.unwrap().tracked_energy();
    let initial_inventory = state.materials.balance(&state.field).unwrap().manons();
    let parameters = StepParameters {
        diffusivity: 0.0,
        dt: 0.2 / steps as f64,
        flow: MediumFlow::Still,
        thermal_conductance: 0.0,
        ambient_cooling_rate: 0.0,
        field: Some(FieldControl {
            mobility: 0.2,
            max_power_joules_per_second: 0.0,
            shutdown_at_bound_fraction: 0.0,
            shutdown_decay_rate: 0.0,
        }),
    };
    for _ in 0..steps {
        state.step(&parameters).unwrap();
    }
    let final_energy = state.materials.balance(&state.field).unwrap().energy()
        + state.field_ledger.unwrap().tracked_energy();
    assert!((final_energy - initial_energy).abs() < 1e-10);
    assert!(
        (state.materials.balance(&state.field).unwrap().manons() - initial_inventory).abs() < 1e-10
    );
    let volume = dx.powi(3);
    (0..cells)
        .map(|i| {
            [
                state.field.at(i, 0, 0).unwrap().value(),
                state.materials.cells()[i].bound / volume,
            ]
        })
        .collect()
}

/// LAW-LAB-SPATIAL-001: a formula trap fills a fixed-length domain with
/// a prescribed analytic profile phi=cosh(x). Refine geometry,
/// potential and nonlinear dynamics together against a separate fine solver.
#[test]
fn formula_drift_and_nonlinear_reactions_converge_in_space_together() {
    let fine = reference(2048, 400);
    let finer = reference(4096, 400);
    let reference_time_check = reference(4096, 200);
    let mut errors = Vec::new();
    for cells in [32, 64, 128] {
        let expected = restrict(&finer, cells);
        let reference_check = restrict(&fine, cells);
        let reference_time = restrict(&reference_time_check, cells);
        let actual = measured(cells, 800);
        let smaller_dt = measured(cells, 1600);
        let error: [f64; 2] = std::array::from_fn(|j| {
            actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a[j] - b[j]).abs())
                .sum::<f64>()
                / cells as f64
        });
        for j in 0..2 {
            let time_error = actual
                .iter()
                .zip(&smaller_dt)
                .map(|(a, b)| (a[j] - b[j]).abs())
                .sum::<f64>()
                / cells as f64;
            let reference_error = reference_check
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a[j] - b[j]).abs())
                .sum::<f64>()
                / cells as f64;
            let reference_time_error = reference_time
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a[j] - b[j]).abs())
                .sum::<f64>()
                / cells as f64;
            assert!(
                reference_time_error < 0.01 * error[j],
                "reference time error={reference_time_error}"
            );
            assert!(
                time_error < 0.1 * error[j],
                "time error={time_error}, spatial={error:?}"
            );
            assert!(
                reference_error < 0.1 * error[j],
                "reference error={reference_error}, spatial={error:?}"
            );
        }
        errors.push(error);
    }
    for j in 0..2 {
        for pair in errors.windows(2) {
            assert!(
                (1.5..3.0).contains(&(pair[0][j] / pair[1][j])),
                "joint spatial errors: {errors:?}"
            );
        }
        assert!(
            errors[2][j] < 0.02,
            "joint spatial absolute errors: {errors:?}"
        );
    }
}
