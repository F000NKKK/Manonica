//! Analytic reference experiments for Manonica's physical laws.
//!
//! Each test names the governing equation and compares a multi-cell
//! simulation against an independently derived finite-volume reference.
//! Tests of conservation alone cannot establish that a solver follows
//! the intended differential equation.

use manonica_core::{
    Evolution, Quantity, TimeStep,
    units::{Diffusivity, ManonDensity},
};
use manonica_physics::{
    FreeDiffusion, FreeManaField, GridShape, MaterialLattice, PotentialBoundary,
    PotentialField, TrapMaterial,
};

fn close(label: &str, measured: f64, expected: f64, tolerance: f64) {
    assert!(
        (measured - expected).abs() <= tolerance,
        "{label}: measured={measured:.12e}, expected={expected:.12e}, tolerance={tolerance:.3e}",
    );
}

fn density(n: f64) -> Quantity<ManonDensity> {
    Quantity::nonnegative(n).unwrap()
}

/// LAW-DIFF-001: J = -D ∇n.
/// On an insulated 1D cell-centered lattice the first cosine mode has
/// eigenvalue -4 sin²(pi/(2N))/dx² for the discrete Laplacian.
/// An explicit Euler step must multiply its amplitude by 1 + D dt λ.
#[test]
fn law_diffusion_discrete_cosine_eigenmode_matches_analytic_amplitude() {
    const N: usize = 12;
    let dx = 0.4;
    let d = 0.28;
    let dt = 0.05;
    let shape = GridShape::new(N, 1, 1).unwrap();
    let mut field = FreeManaField::uniform(shape, dx, density(0.0)).unwrap();
    let mode = |x: usize| ((x as f64 + 0.5) * std::f64::consts::PI / N as f64).cos();
    for x in 0..N {
        field.set(x, 0, 0, density(6.0 + 2.0 * mode(x))).unwrap();
    }
    let solver = FreeDiffusion::new(Quantity::<Diffusivity>::nonnegative(d).unwrap()).unwrap();
    let before = field.total_manons().unwrap().value();
    solver.evolve(&mut field, TimeStep::from_seconds(dt).unwrap()).unwrap();
    let eigenvalue = -4.0
        * (std::f64::consts::PI / (2.0 * N as f64)).sin().powi(2)
        / (dx * dx);
    let amplitude = 2.0 * (1.0 + d * dt * eigenvalue);
    for x in 0..N {
        close(
            &format!("cell {x}"),
            field.at(x, 0, 0).unwrap().value(),
            6.0 + amplitude * mode(x),
            1e-11,
        );
    }
    close("diffusion conservation", field.total_manons().unwrap().value(), before, 1e-11);
}

/// LAW-POT-001: -epsilon ∆phi + screening phi = rho.
/// For an insulated domain and spatially constant source rho,
/// phi = rho/screening is the exact discrete and continuum solution.
#[test]
fn law_screened_poisson_uniform_insulated_source_matches_constant_solution() {
    let shape = GridShape::new(4, 3, 2).unwrap();
    let field = PotentialField::solve(
        shape,
        1.0,
        &vec![6.0; shape.len()],
        1.0,
        2.0,
        PotentialBoundary::Insulated,
        180,
    )
    .unwrap();
    for (i, phi) in field.values().iter().enumerate() {
        close(&format!("phi[{i}]"), *phi, 3.0, 1e-9);
    }
    // With zero spatial gradient, field energy is just screening * phi² / 2.
    let expected_energy = 0.5 * 2.0 * 9.0 * shape.len() as f64;
    close("screened potential energy", field.field_energy_joules(), expected_energy, 1e-8);
}

/// LAW-DRIFT-001: J = -mu n grad(phi).
/// In a two-cell domain with one occupied upwind cell, the exact
/// finite-volume face transfer for one Euler step is |v| dt n_upwind.
#[test]
fn law_upwind_drift_face_flux_matches_gradient_and_preserves_manons() {
    let shape = GridShape::new(2, 1, 1).unwrap();
    let potential = PotentialField::solve(
        shape,
        1.0,
        &[-2.0, 0.0],
        1.0,
        1.0,
        PotentialBoundary::Insulated,
        160,
    )
    .unwrap();
    let phi0 = potential.at(0, 0, 0).unwrap();
    let phi1 = potential.at(1, 0, 0).unwrap();
    assert!(phi0 < phi1, "negative source attracts manons into first cell");
    let mobility = 0.25;
    let dt = 0.1;
    let initial = 10.0;
    let expected = mobility * (phi1 - phi0) * dt * initial;
    assert!(expected > 0.0 && expected < initial);
    let mut field = FreeManaField::uniform(shape, 1.0, density(0.0)).unwrap();
    field.set(1, 0, 0, density(initial)).unwrap();
    let report = potential.drift(&mut field, mobility, dt).unwrap();
    close("left receives flux", field.at(0, 0, 0).unwrap().value(), expected, 1e-11);
    close("right loses flux", field.at(1, 0, 0).unwrap().value(), initial - expected, 1e-11);
    close("transported manons", report.transported_manons, expected, 1e-11);
    close("drift conservation", report.after_manons, report.before_manons, 1e-11);
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
    close("exported heat balance", remaining + exported, initial, 1e-11);
    close("manons unaffected by cooling", sites.balance(&field).unwrap().manons(), count_before, 1e-11);
    close(
        "temperature from specific heat",
        sites.peak_temperature_kelvin().unwrap(),
        300.0 + remaining / material.heat_capacity,
        1e-11,
    );
}
