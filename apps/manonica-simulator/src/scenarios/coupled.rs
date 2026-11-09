//! Small coupled crystal experiment for headless CPU/GPU validation.
use manonica_core::Quantity;
use manonica_materials::{ManaChannel, MaterialLattice, TrapMaterial};
use manonica_physics::{
    FreeManaField, GridShape, LocalBoundarySite, LocalPotentialBoundary, PotentialBoundary,
    PotentialField,
};
use manonica_simulation::{CoupledState, FieldControl, FieldLedger, MediumFlow, StepParameters};

pub fn coupled_crystal() -> Result<(CoupledState, StepParameters), Box<dyn std::error::Error>> {
    coupled_crystal_grid([8, 6, 4])
}

pub fn coupled_crystal_grid(
    dimensions: [usize; 3],
) -> Result<(CoupledState, StepParameters), Box<dyn std::error::Error>> {
    if dimensions[0] < 8 || dimensions[1] < 6 || dimensions[2] < 4 {
        return Err(
            std::io::Error::other("crystal scenario requires at least 8 x 6 x 4 cells").into(),
        );
    }
    let shape = GridShape::new(dimensions[0], dimensions[1], dimensions[2])?;
    let mut field = FreeManaField::uniform(shape, 0.5, Quantity::nonnegative(2.0)?)?;
    field.set(2, 3, 2, Quantity::nonnegative(40.0)?)?;
    let mut materials = MaterialLattice::new(shape);
    for x in 2..6 {
        materials.install(
            [x, 3, 2],
            TrapMaterial {
                capacity: 1.5,
                capture_rate: 0.6,
                release_rate: 0.02,
                binding_energy: 0.4,
                heat_capacity: 2.0,
                reference_temperature: 300.0,
                activation_temperature: 100.0,
                max_flux: 1.0,
                conductivity: 1.0,
            },
        )?;
    }
    for (a, b) in [(2, 3), (3, 4), (4, 5), (2, 5)] {
        materials.add_channel(ManaChannel {
            from: [a, 3, 2],
            to: [b, 3, 2],
            length: ((b - a) as f64) * 0.5,
            area: 0.25,
            conductance: 0.1,
            max_flux: 1.0,
        })?;
    }
    let mut sources = vec![0.0; shape.len()];
    sources[shape.index(3, 3, 2).unwrap()] = -0.5;
    let potential = PotentialField::solve_with_local(
        shape,
        0.5,
        &sources,
        1.0,
        0.9,
        PotentialBoundary::Grounded,
        120,
        &[
            LocalBoundarySite {
                index: shape.index(5, 3, 2).unwrap(),
                condition: LocalPotentialBoundary::Robin {
                    reference: -0.1,
                    exchange: 0.3,
                },
            },
            LocalBoundarySite {
                index: shape.index(1, 1, 1).unwrap(),
                condition: LocalPotentialBoundary::Dirichlet { value: 0.1 },
            },
            LocalBoundarySite {
                index: shape.index(6, 4, 1).unwrap(),
                condition: LocalPotentialBoundary::Neumann,
            },
        ],
    )?;
    let ledger = FieldLedger {
        field_energy_joules: potential.field_energy_joules(),
        manon_potential_joules: potential.manon_potential_energy(&field)?,
        peak_potential: potential.peak_absolute(),
        ..Default::default()
    };
    let state = CoupledState {
        field,
        materials,
        potential: Some(potential),
        field_ledger: Some(ledger),
        shutdown: false,
    };
    let params = StepParameters {
        diffusivity: 0.08,
        dt: 0.02,
        flow: MediumFlow::Vortex {
            origin: [0.0; 3],
            axis: [0.0, 1.0, 0.0],
            angular_speed: 0.1,
            axial_speed: 0.02,
        },
        thermal_conductance: 0.2,
        ambient_cooling_rate: 0.1,
        field: Some(FieldControl {
            mobility: 0.1,
            max_power_joules_per_second: 1000.0,
            shutdown_at_bound_fraction: 0.4,
            shutdown_decay_rate: 0.3,
        }),
    };
    Ok((state, params))
}
