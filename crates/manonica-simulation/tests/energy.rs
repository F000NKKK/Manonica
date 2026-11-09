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
