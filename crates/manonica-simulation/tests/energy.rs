use manonica_core::Quantity;
use manonica_materials::{MaterialLattice, TrapMaterial};
use manonica_physics::{FreeManaField, GridShape};
use manonica_simulation::{CoupledState, MediumFlow, StepParameters};
#[test]
fn cooling_without_potential_exports_energy_instead_of_destroying_it() {
    let shape = GridShape::new(1, 1, 1).unwrap();
    let field = FreeManaField::uniform(shape, 1.0, Quantity::nonnegative(10.0).unwrap()).unwrap();
    let mut materials = MaterialLattice::new(shape);
    materials
        .install(
            [0, 0, 0],
            TrapMaterial {
                capacity: 1.0,
                capture_rate: 0.0,
                release_rate: 0.0,
                binding_energy: 2.0,
                heat_capacity: 3.0,
                reference_temperature: 300.0,
                activation_temperature: 0.0,
                max_flux: 1.0,
                conductivity: 0.0,
            },
        )
        .unwrap();
    materials.try_replace_storage(&[[0.0, 10.0]]).unwrap();
    let mut state = CoupledState {
        field,
        materials,
        potential: None,
        field_ledger: None,
        shutdown: false,
    };
    let params = StepParameters {
        diffusivity: 0.0,
        dt: 0.1,
        flow: MediumFlow::Still,
        thermal_conductance: 0.0,
        ambient_cooling_rate: 0.5,
        field: None,
    };
    for _ in 0..100 {
        state.step(&params).unwrap();
    }
    let stored = state.materials.balance(&state.field).unwrap().energy();
    let exported = state.field_ledger.unwrap().exported_heat_joules;
    assert!(exported > 9.0);
    assert!((stored + exported - 10.0).abs() < 1e-10);
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-10,
        "actual={actual}, expected={expected}"
    );
}

/// LAW-FIELD-WORK-001: moving 0.25 manons from phi=0 to phi=2
/// costs 0.5 J. Pay from existing heat, then external work.
#[test]
fn uphill_transport_pays_exact_work_and_rejects_insufficient_power_atomically() {
    use manonica_physics::{
        LocalBoundarySite, LocalPotentialBoundary, PotentialBoundary, PotentialField,
    };
    use manonica_simulation::{FieldControl, FieldLedger};
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
                condition: LocalPotentialBoundary::Dirichlet { value: 0.0 },
            },
            LocalBoundarySite {
                index: 1,
                condition: LocalPotentialBoundary::Dirichlet { value: 2.0 },
            },
        ],
    )
    .unwrap();
    close(potential.field_energy_joules(), 4.0);
    for (heat, power, succeeds) in [
        (0.0, 5.0, true),
        (0.2, 3.0, true),
        (1.0, 0.0, true),
        (0.0, 2.0, false),
    ] {
        let mut field =
            FreeManaField::uniform(shape, 1.0, Quantity::nonnegative(0.0).unwrap()).unwrap();
        field
            .set(0, 0, 0, Quantity::nonnegative(10.0).unwrap())
            .unwrap();
        let mut state = CoupledState {
            field,
            materials: MaterialLattice::new(shape),
            potential: Some(potential.clone()),
            field_ledger: Some(FieldLedger {
                field_energy_joules: 4.0,
                peak_potential: 2.0,
                dissipated_heat_joules: heat,
                ..Default::default()
            }),
            shutdown: false,
        };
        let parameters = StepParameters {
            diffusivity: 0.0,
            dt: 0.1,
            flow: MediumFlow::Uniform {
                velocity: [0.25, 0.0, 0.0],
            },
            thermal_conductance: 0.0,
            ambient_cooling_rate: 0.0,
            field: Some(FieldControl {
                mobility: 0.0,
                max_power_joules_per_second: power,
                shutdown_at_bound_fraction: 0.0,
                shutdown_decay_rate: 0.0,
            }),
        };
        let result = state.step(&parameters);
        let ledger = state.field_ledger.unwrap();
        if succeeds {
            result.unwrap();
            close(state.field.at(0, 0, 0).unwrap().value(), 9.75);
            close(state.field.at(1, 0, 0).unwrap().value(), 0.25);
            close(ledger.manon_potential_joules, 0.5);
            close(ledger.operating_work_joules, (0.5_f64 - heat).max(0.0));
            close(ledger.dissipated_heat_joules, (heat - 0.5_f64).max(0.0));
            close(
                ledger.tracked_energy(),
                4.0 + heat + ledger.operating_work_joules,
            );
        } else {
            assert!(result.is_err());
            close(state.field.at(0, 0, 0).unwrap().value(), 10.0);
            close(state.field.at(1, 0, 0).unwrap().value(), 0.0);
            close(ledger.tracked_energy(), 4.0 + heat);
            close(ledger.operating_work_joules, 0.0);
            close(ledger.manon_potential_joules, 0.0);
            assert!(!state.shutdown && !ledger.shutdown);
        }
        assert_eq!(
            state.potential.as_ref().unwrap().values(),
            potential.values()
        );
        close(
            state.materials.balance(&state.field).unwrap().manons(),
            10.0,
        );
    }
}

/// LAW-FIELD-DECAY-001: phi=2*exp(-k*t), U_manons=6*exp(-k*t),
/// E_field=2*exp(-2*k*t). Lost energy becomes heat, then bath export.
#[test]
fn decaying_field_converts_linear_and_quadratic_energy_to_heat_and_export() {
    use manonica_physics::{PotentialBoundary, PotentialField};
    use manonica_simulation::{FieldControl, FieldLedger};
    let shape = GridShape::new(1, 1, 1).unwrap();
    let potential = PotentialField::solve(
        shape,
        1.0,
        &[2.0],
        1.0,
        1.0,
        PotentialBoundary::Insulated,
        10,
    )
    .unwrap();
    let field = FreeManaField::uniform(shape, 1.0, Quantity::nonnegative(3.0).unwrap()).unwrap();
    let mut state = CoupledState {
        field,
        materials: MaterialLattice::new(shape),
        potential: Some(potential),
        field_ledger: Some(FieldLedger {
            field_energy_joules: 2.0,
            manon_potential_joules: 6.0,
            peak_potential: 2.0,
            shutdown: true,
            ..Default::default()
        }),
        shutdown: true,
    };
    let parameters = StepParameters {
        diffusivity: 0.0,
        dt: 0.1,
        flow: MediumFlow::Still,
        thermal_conductance: 0.0,
        ambient_cooling_rate: 0.3,
        field: Some(FieldControl {
            mobility: 0.0,
            max_power_joules_per_second: 0.0,
            shutdown_at_bound_fraction: 0.0,
            shutdown_decay_rate: 0.7,
        }),
    };
    let retention = (-0.3_f64 * 0.1).exp();
    for step in 1..=40 {
        state.step(&parameters).unwrap();
        let time = step as f64 * 0.1;
        let decay = (-0.7 * time).exp();
        let ledger = state.field_ledger.unwrap();
        close(ledger.peak_potential, 2.0 * decay);
        close(ledger.manon_potential_joules, 6.0 * decay);
        close(ledger.field_energy_joules, 2.0 * decay * decay);
        // Each exact energy decrement is cooled for the remaining accepted steps.
        let expected_heat = (1..=step)
            .map(|j| {
                let before = (-0.7 * (j - 1) as f64 * 0.1).exp();
                let after = (-0.7 * j as f64 * 0.1).exp();
                (6.0 * (before - after) + 2.0 * (before * before - after * after))
                    * retention.powi(step - j + 1)
            })
            .sum::<f64>();
        close(ledger.dissipated_heat_joules, expected_heat);
        close(ledger.tracked_energy() + ledger.exported_heat_joules, 8.0);
        close(ledger.operating_work_joules, 0.0);
        close(state.materials.balance(&state.field).unwrap().manons(), 3.0);
        assert!(ledger.shutdown && state.shutdown);
    }
}

/// LAW-FIELD-BOUND-001: bound and free manons retain the same phi energy.
/// With total N=3 and phi=+/-2*d, U=+/-6*d and E_field=2*d².
/// Decaying a negative potential requires work rather than producing heat.
#[test]
fn bound_inventory_retains_potential_energy_during_positive_and_negative_decay() {
    use manonica_physics::{PotentialBoundary, PotentialField};
    use manonica_simulation::{FieldControl, FieldLedger};
    let shape = GridShape::new(1, 1, 1).unwrap();
    for sign in [-1.0, 1.0] {
        let field =
            FreeManaField::uniform(shape, 1.0, Quantity::nonnegative(1.0).unwrap()).unwrap();
        let mut materials = MaterialLattice::new(shape);
        materials
            .install(
                [0, 0, 0],
                TrapMaterial {
                    capacity: 2.0,
                    capture_rate: 0.0,
                    release_rate: 0.0,
                    binding_energy: 2.0,
                    heat_capacity: 3.0,
                    reference_temperature: 300.0,
                    activation_temperature: 0.0,
                    max_flux: 0.0,
                    conductivity: 0.0,
                },
            )
            .unwrap();
        materials.try_replace_storage(&[[2.0, 4.0]]).unwrap();
        let potential = PotentialField::solve(
            shape,
            1.0,
            &[sign * 2.0],
            1.0,
            1.0,
            PotentialBoundary::Insulated,
            10,
        )
        .unwrap();
        let mut state = CoupledState {
            field,
            materials,
            potential: Some(potential),
            field_ledger: Some(FieldLedger {
                field_energy_joules: 2.0,
                manon_potential_joules: sign * 6.0,
                peak_potential: 2.0,
                shutdown: true,
                ..Default::default()
            }),
            shutdown: true,
        };
        let parameters = StepParameters {
            diffusivity: 0.0,
            dt: 0.1,
            flow: MediumFlow::Still,
            thermal_conductance: 0.0,
            ambient_cooling_rate: 0.0,
            field: Some(FieldControl {
                mobility: 0.0,
                max_power_joules_per_second: 100.0,
                shutdown_at_bound_fraction: 0.0,
                shutdown_decay_rate: 0.7,
            }),
        };
        for step in 1..=40 {
            state.step(&parameters).unwrap();
            let d = (-0.7 * step as f64 * 0.1).exp();
            let ledger = state.field_ledger.unwrap();
            close(ledger.manon_potential_joules, sign * 6.0 * d);
            close(ledger.field_energy_joules, 2.0 * d * d);
            let change = sign * 6.0 * (d - 1.0) + 2.0 * (d * d - 1.0);
            close(ledger.operating_work_joules, change.max(0.0));
            close(ledger.dissipated_heat_joules, (-change).max(0.0));
            close(
                ledger.tracked_energy(),
                2.0 + sign * 6.0 + ledger.operating_work_joules,
            );
            let balance = state.materials.balance(&state.field).unwrap();
            close(balance.free, 1.0);
            close(balance.bound, 2.0);
            close(balance.thermal_joules, 4.0);
            close(balance.energy(), 0.0);
        }
    }
}
