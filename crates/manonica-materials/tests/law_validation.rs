//! Independent numerical reference experiments for materials laws.
//! Each constitutive equation is checked against a quantitative reference.
use manonica_core::{Quantity, units::ManonDensity};
use manonica_materials::{DirectionalConductivity, ManaChannel, MaterialLattice, TrapMaterial};
use manonica_physics::{FreeManaField, GridShape};

fn close(label: &str, measured: f64, expected: f64, tolerance: f64) {
    assert!(
        (measured - expected).abs() <= tolerance,
        "{label}: measured={measured:.12e}, expected={expected:.12e}, tolerance={tolerance:.3e}",
    );
}
fn density(n: f64) -> Quantity<ManonDensity> {
    Quantity::nonnegative(n).unwrap()
}

/// LAW-HEAT-001: dQ/dt = -k Q for coupling to a fixed-temperature bath.
/// The exact solution after time t is Q(t) = Q(0) exp(-kt), and every
/// removed joule must be reported as exported heat.
#[test]
fn law_bath_cooling_matches_exponential_solution_and_energy_export() {
    let shape = GridShape::new(1, 1, 1).unwrap();
    let mut field = FreeManaField::uniform(shape, 1.0, density(50.0)).unwrap();
    let mut sites = MaterialLattice::new(shape);
    let material = TrapMaterial {
        capacity: 8.0,
        capture_rate: 4.0,
        release_rate: 0.0,
        binding_energy: 2.0,
        heat_capacity: 5.0,
        reference_temperature: 300.0,
        activation_temperature: 0.0,
        max_flux: 8.0,
        conductivity: 0.0,
    };
    sites.install([0, 0, 0], material).unwrap();
    sites.step(&mut field, 1.0).unwrap();
    let initial = sites.at(0, 0, 0).unwrap().heat;
    assert!(initial > 0.0);
    let count_before = sites.balance(&field).unwrap().manons();
    let k = 0.35;
    let dt = 0.2;
    let mut exported = 0.0;
    for _ in 0..20 {
        exported += sites.cool_into_bath(dt, k).unwrap();
    }
    let remaining = sites.at(0, 0, 0).unwrap().heat;
    let expected = initial * (-k * dt * 20.0).exp();
    close("exponential cooling", remaining, expected, 1e-11);
    close(
        "exported heat balance",
        remaining + exported,
        initial,
        1e-11,
    );
    close(
        "manons unaffected by cooling",
        sites.balance(&field).unwrap().manons(),
        count_before,
        1e-11,
    );
    close(
        "temperature from specific heat",
        sites.peak_temperature_kelvin().unwrap(),
        300.0 + remaining / material.heat_capacity,
        1e-11,
    );
}

/// LAW-MAT-ANISO-001: J=-K grad(n) projected onto a channel.
/// For principal K=[4,1,1], u_x^T K u_x=4 and u_y^T K u_y=1.
/// A single time step, away from saturation, must transfer in this ratio.
#[test]
fn law_anisotropic_crystal_channels_match_tensor_projection_and_flux() {
    let shape = GridShape::new(2, 2, 1).unwrap();
    let crystal = DirectionalConductivity {
        principal: [4.0, 1.0, 1.0],
        rotation_z_radians: 0.0,
    };
    close("Kxx", crystal.along([1.0, 0.0, 0.0]).unwrap(), 4.0, 1e-12);
    close("Kyy", crystal.along([0.0, 1.0, 0.0]).unwrap(), 1.0, 1e-12);
    let simulate = |target: [usize; 3]| {
        let mut lattice = MaterialLattice::new(shape);
        lattice
            .add_directional_channel(
                ManaChannel {
                    from: [0, 0, 0],
                    to: target,
                    length: 1.0,
                    area: 1.0,
                    conductance: 1.0,
                    max_flux: 1000.0,
                },
                crystal,
            )
            .unwrap();
        let mut field = FreeManaField::uniform(shape, 1.0, density(0.0)).unwrap();
        field.set(0, 0, 0, density(10.0)).unwrap();
        let before = field.total_manons().unwrap().value();
        lattice.step(&mut field, 0.1).unwrap();
        close(
            "inventory conserved",
            field.total_manons().unwrap().value(),
            before,
            1e-12,
        );
        field.at(target[0], target[1], target[2]).unwrap().value()
    };
    close("x directed flow", simulate([1, 0, 0]), 4.0, 1e-12);
    close("y directed flow", simulate([0, 1, 0]), 1.0, 1e-12);
}

/// LAW-MAT-ANISO-002/003: isotropy is rotation invariant;
/// rotating the crystal by pi/2 exchanges the X/Y conductivities.
#[test]
fn law_anisotropy_isotropic_limit_and_rotating_principal_axes() {
    let iso = DirectionalConductivity {
        principal: [2.0, 2.0, 2.0],
        rotation_z_radians: 0.71,
    };
    for direction in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.4, 0.5, 0.8]] {
        close(
            "isotropic directional multiplier",
            iso.along(direction).unwrap(),
            2.0,
            1e-12,
        );
    }
    let rotated = DirectionalConductivity {
        principal: [4.0, 1.0, 1.0],
        rotation_z_radians: std::f64::consts::FRAC_PI_2,
    };
    close(
        "rotated X",
        rotated.along([1.0, 0.0, 0.0]).unwrap(),
        1.0,
        1e-12,
    );
    close(
        "rotated Y",
        rotated.along([0.0, 1.0, 0.0]).unwrap(),
        4.0,
        1e-12,
    );
}

/// LAW-MAT-ANISO-004: positive semidefinite conductivity cannot create
/// negative transport. A zero eigenvalue blocks the associated flow;
/// invalid eigenvalues or orientation are rejected transactionally.
#[test]
fn law_anisotropic_zero_conductance_and_invalid_coefficients() {
    let shape = GridShape::new(2, 1, 1).unwrap();
    let blocked = DirectionalConductivity {
        principal: [0.0, 5.0, 1.0],
        rotation_z_radians: 0.0,
    };
    let channel = ManaChannel {
        from: [0, 0, 0],
        to: [1, 0, 0],
        length: 1.0,
        area: 1.0,
        conductance: 1.0,
        max_flux: 100.0,
    };
    let mut lattice = MaterialLattice::new(shape);
    lattice.add_directional_channel(channel, blocked).unwrap();
    let mut field = FreeManaField::uniform(shape, 1.0, density(0.0)).unwrap();
    field.set(0, 0, 0, density(10.0)).unwrap();
    lattice.step(&mut field, 0.1).unwrap();
    close(
        "zero eigenvalue blocks transport",
        field.at(0, 0, 0).unwrap().value(),
        10.0,
        1e-12,
    );
    let invalid = DirectionalConductivity {
        principal: [-1.0, 1.0, 1.0],
        rotation_z_radians: 0.0,
    };
    assert!(lattice.add_directional_channel(channel, invalid).is_err());
    assert!(
        DirectionalConductivity {
            principal: [1.0; 3],
            rotation_z_radians: f64::NAN,
        }
        .along([1.0, 0.0, 0.0])
        .is_err()
    );
    assert!(blocked.along([0.0; 3]).is_err());
}
