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
