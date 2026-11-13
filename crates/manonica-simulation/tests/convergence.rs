//! Independent cosine-mode references for a coupled reaction/transport problem.
use manonica_core::Quantity;
use manonica_materials::{ManaChannel, MaterialLattice, TrapMaterial};
use manonica_physics::{FreeManaField, GridShape};
use manonica_simulation::{CoupledState, MediumFlow, StepParameters};

const TIME: f64 = 0.02;
const DIFFUSION: f64 = 0.05;
const CHANNEL: f64 = 0.1;
const THERMAL: f64 = 0.08;
const RELEASE: f64 = 0.7;
const COOLING: f64 = 0.3;

// Integral of exp(-decay*(t-s))*exp(-release*s) from 0 to t.
fn convolution(decay: f64) -> f64 {
    ((-RELEASE * TIME).exp() - (-decay * TIME).exp()) / (decay - RELEASE)
}

fn run(cells: usize, steps: usize, continuum: bool) -> [f64; 3] {
    let dx = 1.0 / cells as f64;
    let volume = dx.powi(3);
    let pi = std::f64::consts::PI;
    let shape = GridShape::new(cells, 1, 1).unwrap();
    let average = (pi * dx / 2.0).sin() / (pi * dx / 2.0);
    let mode = |x: usize| average * (pi * (x as f64 + 0.5) * dx).cos();
    let mut field = FreeManaField::uniform(shape, dx, Quantity::nonnegative(0.0).unwrap()).unwrap();
    let mut materials = MaterialLattice::new(shape);
    let mut storage = Vec::new();
    for x in 0..cells {
        field
            .set(x, 0, 0, Quantity::nonnegative(6.0 + mode(x)).unwrap())
            .unwrap();
        materials
            .install(
                [x, 0, 0],
                TrapMaterial {
                    capacity: 4.0 * volume,
                    capture_rate: 0.0,
                    release_rate: RELEASE,
                    binding_energy: 2.0,
                    heat_capacity: volume,
                    reference_temperature: 300.0,
                    activation_temperature: 0.0,
                    max_flux: 1000.0,
                    conductivity: 0.0,
                },
            )
            .unwrap();
        storage.push([
            (2.0 + 0.5 * mode(x)) * volume,
            (100.0 + 2.0 * mode(x)) * volume,
        ]);
        if x > 0 {
            materials
                .add_channel(ManaChannel {
                    from: [x - 1, 0, 0],
                    to: [x, 0, 0],
                    length: dx,
                    area: dx * dx,
                    conductance: CHANNEL,
                    max_flux: 1000.0,
                })
                .unwrap();
        }
    }
    materials.try_replace_storage(&storage).unwrap();
    let mut state = CoupledState {
        field,
        materials,
        potential: None,
        field_ledger: None,
        shutdown: false,
    };
    let initial = state.materials.balance(&state.field).unwrap();
    let parameters = StepParameters {
        diffusivity: DIFFUSION,
        dt: TIME / steps as f64,
        flow: MediumFlow::Still,
        thermal_conductance: THERMAL * dx,
        ambient_cooling_rate: COOLING,
        field: None,
    };
    for _ in 0..steps {
        state.step(&parameters).unwrap();
    }
    let final_balance = state.materials.balance(&state.field).unwrap();
    let exported = state.field_ledger.unwrap().exported_heat_joules;
    assert!((final_balance.manons() - initial.manons()).abs() < initial.manons() * 1e-10);
    assert!(
        (final_balance.energy() + exported - initial.energy()).abs() < initial.energy() * 1e-10
    );
    let eigenvalue = if continuum {
        pi * pi
    } else {
        4.0 * (pi * dx / 2.0).sin().powi(2) / (dx * dx)
    };
    let transport_decay = (DIFFUSION + CHANNEL) * eigenvalue;
    let heat_decay = THERMAL * eigenvalue + COOLING;
    let free_mean = 6.0 + 2.0 * (1.0 - (-RELEASE * TIME).exp());
    let free_amplitude =
        (-transport_decay * TIME).exp() + 0.5 * RELEASE * convolution(transport_decay);
    let heat_mean = 100.0 * (-COOLING * TIME).exp() - 4.0 * RELEASE * convolution(COOLING);
    let heat_amplitude = 2.0 * (-heat_decay * TIME).exp() - RELEASE * convolution(heat_decay);
    let mut errors = [0.0_f64; 3];
    for x in 0..cells {
        let cell = &state.materials.cells()[x];
        let expected_free = free_mean + free_amplitude * mode(x);
        let expected_bound = (2.0 + 0.5 * mode(x)) * (-RELEASE * TIME).exp();
        let expected_heat = heat_mean + heat_amplitude * mode(x);
        errors[0] = errors[0].max((state.field.at(x, 0, 0).unwrap().value() - expected_free).abs());
        errors[1] = errors[1].max((cell.bound / volume - expected_bound).abs());
        errors[2] = errors[2].max((cell.heat / volume - expected_heat).abs());
        assert!(cell.bound >= 0.0 && cell.heat >= 0.0);
    }
    assert!(
        errors[1] < 1e-10,
        "exact exponential bound inventory: {errors:?}"
    );
    errors
}

/// LAW-COUPLED-CONV-001: fixed grid, refine only time; compare against
/// exact semidiscrete reaction/diffusion/channel/thermal mode equations.
#[test]
fn coupled_network_first_order_time_convergence() {
    let errors: Vec<_> = [20, 40, 80, 160]
        .into_iter()
        .map(|steps| run(8, steps, false))
        .collect();
    for metric in [0, 2] {
        for pair in errors.windows(2) {
            assert!(
                (1.8..2.2).contains(&(pair[0][metric] / pair[1][metric])),
                "temporal errors: {errors:?}"
            );
        }
        assert!(
            errors[3][metric] < 0.001,
            "temporal absolute errors: {errors:?}"
        );
    }
}

/// LAW-COUPLED-CONV-002: refine only dx; fixed dt=5e-7 and exact
/// continuum reference. Bound the time error independently on every grid.
#[test]
fn coupled_network_second_order_space_convergence() {
    let mut errors = Vec::new();
    for cells in [6, 12, 24] {
        let spatial = run(cells, 40_000, true);
        let temporal = run(cells, 40_000, false);
        for metric in [0, 2] {
            assert!(
                temporal[metric] < spatial[metric] * 0.1,
                "time error must be subdominant: time={temporal:?}, space={spatial:?}"
            );
        }
        errors.push(spatial);
    }
    for metric in [0, 2] {
        for pair in errors.windows(2) {
            assert!(
                (3.5..4.5).contains(&(pair[0][metric] / pair[1][metric])),
                "spatial errors: {errors:?}"
            );
        }
        assert!(
            errors[2][metric] < 0.001,
            "spatial absolute errors: {errors:?}"
        );
    }
}

// Independent continuous-time two-cell equations. These do not invoke any
// production transport/reaction operator or reproduce its exponential split.
fn drift_capture_rhs(y: [f64; 4]) -> [f64; 4] {
    let drift = 0.2 * y[0];
    let left = 0.2 * y[0] * (4.0 - y[2]) - 0.3 * y[2];
    let right = 0.2 * y[1] * (4.0 - y[3]) - 0.3 * y[3];
    [-drift - left, drift - right, left, right]
}

fn independent_rk4(steps: usize) -> [f64; 4] {
    let mut y = [8.0, 8.0, 0.0, 0.0];
    let dt = 0.5 / steps as f64;
    for _ in 0..steps {
        let a = drift_capture_rhs(y);
        let b = drift_capture_rhs(std::array::from_fn(|i| y[i] + dt * a[i] / 2.0));
        let c = drift_capture_rhs(std::array::from_fn(|i| y[i] + dt * b[i] / 2.0));
        let d = drift_capture_rhs(std::array::from_fn(|i| y[i] + dt * c[i]));
        y = std::array::from_fn(|i| y[i] + dt / 6.0 * (a[i] + 2.0 * b[i] + 2.0 * c[i] + d[i]));
    }
    y
}

/// LAW-COUPLED-CONV-003: nonlinear capture/release and potential drift.
/// Fixed spatial grid, independent RK4 reference with refinement audit.
#[test]
fn nonlinear_capture_and_drift_first_order_time_convergence() {
    use manonica_physics::{
        LocalBoundarySite, LocalPotentialBoundary, PotentialBoundary, PotentialField,
    };
    use manonica_simulation::{FieldControl, FieldLedger};
    let reference = independent_rk4(2000);
    let refined = independent_rk4(4000);
    for i in 0..4 {
        assert!((reference[i] - refined[i]).abs() < 1e-11);
    }
    assert!((reference.iter().sum::<f64>() - 16.0).abs() < 1e-11);
    let shape = GridShape::new(2, 1, 1).unwrap();
    let potential = PotentialField::solve_with_local(
        shape,
        1.0,
        &[0.0; 2],
        1.0,
        1.0,
        PotentialBoundary::Insulated,
        10,
        &[
            LocalBoundarySite {
                index: 0,
                condition: LocalPotentialBoundary::Dirichlet { value: 1.0 },
            },
            LocalBoundarySite {
                index: 1,
                condition: LocalPotentialBoundary::Dirichlet { value: 0.0 },
            },
        ],
    )
    .unwrap();
    let mut errors = Vec::new();
    for steps in [20, 40, 80, 160] {
        let field =
            FreeManaField::uniform(shape, 1.0, Quantity::nonnegative(8.0).unwrap()).unwrap();
        let mut materials = MaterialLattice::new(shape);
        for x in 0..2 {
            materials
                .install(
                    [x, 0, 0],
                    TrapMaterial {
                        capacity: 4.0,
                        capture_rate: 0.2,
                        release_rate: 0.3,
                        binding_energy: 2.0,
                        heat_capacity: 5.0,
                        reference_temperature: 300.0,
                        activation_temperature: 0.0,
                        max_flux: 1000.0,
                        conductivity: 0.0,
                    },
                )
                .unwrap();
        }
        // Independent heat supply ensures the energy limiter never hides kinetics.
        materials
            .try_replace_storage(&[[0.0, 100.0], [0.0, 100.0]])
            .unwrap();
        let mut state = CoupledState {
            field,
            materials,
            potential: Some(potential.clone()),
            field_ledger: Some(FieldLedger {
                field_energy_joules: 1.0,
                manon_potential_joules: 8.0,
                peak_potential: 1.0,
                ..Default::default()
            }),
            shutdown: false,
        };
        let parameters = StepParameters {
            diffusivity: 0.0,
            dt: 0.5 / steps as f64,
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
            let balance = state.materials.balance(&state.field).unwrap();
            let ledger = state.field_ledger.unwrap();
            assert!((balance.manons() - 16.0).abs() < 1e-10);
            assert!((balance.energy() + ledger.tracked_energy() - 209.0).abs() < 1e-10);
            assert!(ledger.operating_work_joules.abs() < 1e-10);
            for cell in state.materials.cells() {
                assert!((0.0..=4.0).contains(&cell.bound) && cell.heat >= 0.0);
            }
        }
        let observed = [
            state.field.at(0, 0, 0).unwrap().value(),
            state.field.at(1, 0, 0).unwrap().value(),
            state.materials.cells()[0].bound,
            state.materials.cells()[1].bound,
        ];
        errors.push(std::array::from_fn::<_, 4, _>(|i| {
            (observed[i] - reference[i]).abs()
        }));
    }
    for metric in 0..4 {
        for pair in errors.windows(2) {
            assert!(
                (1.8..2.2).contains(&(pair[0][metric] / pair[1][metric])),
                "nonlinear first-order errors: {errors:?}"
            );
        }
        assert!(
            errors[3][metric] < 0.01,
            "nonlinear absolute errors: {errors:?}"
        );
    }
}
