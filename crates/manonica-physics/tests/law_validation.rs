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
    FreeDiffusion, FreeManaField, GridShape, LocalBoundarySite, LocalPotentialBoundary,
    PotentialBoundary, PotentialField,
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
    solver
        .evolve(&mut field, TimeStep::from_seconds(dt).unwrap())
        .unwrap();
    let eigenvalue = -4.0 * (std::f64::consts::PI / (2.0 * N as f64)).sin().powi(2) / (dx * dx);
    let amplitude = 2.0 * (1.0 + d * dt * eigenvalue);
    for x in 0..N {
        close(
            &format!("cell {x}"),
            field.at(x, 0, 0).unwrap().value(),
            6.0 + amplitude * mode(x),
            1e-11,
        );
    }
    close(
        "diffusion conservation",
        field.total_manons().unwrap().value(),
        before,
        1e-11,
    );
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
    close(
        "screened potential energy",
        field.field_energy_joules(),
        expected_energy,
        1e-8,
    );
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
    assert!(
        phi0 < phi1,
        "negative source attracts manons into first cell"
    );
    let mobility = 0.25;
    let dt = 0.1;
    let initial = 10.0;
    let expected = mobility * (phi1 - phi0) * dt * initial;
    assert!(expected > 0.0 && expected < initial);
    let mut field = FreeManaField::uniform(shape, 1.0, density(0.0)).unwrap();
    field.set(1, 0, 0, density(initial)).unwrap();
    let report = potential.drift(&mut field, mobility, dt).unwrap();
    close(
        "left receives flux",
        field.at(0, 0, 0).unwrap().value(),
        expected,
        1e-11,
    );
    close(
        "right loses flux",
        field.at(1, 0, 0).unwrap().value(),
        initial - expected,
        1e-11,
    );
    close(
        "transported manons",
        report.transported_manons,
        expected,
        1e-11,
    );
    close(
        "drift conservation",
        report.after_manons,
        report.before_manons,
        1e-11,
    );
}

/// LAW-LOCAL-DIR-001: phi at a conducting cell is prescribed. On a
/// screened three-cell insulated line with kappa=screening=dx=1,
/// phi[0]=6 and zero volume source:
/// 3*phi[1]=6+phi[2], 2*phi[2]=phi[1] => [6, 2.4, 1.2].
#[test]
fn law_local_dirichlet_matches_exact_three_cell_profile() {
    let shape = GridShape::new(3, 1, 1).unwrap();
    let field = PotentialField::solve_with_local(
        shape,
        1.0,
        &[0.0; 3],
        1.0,
        1.0,
        PotentialBoundary::Insulated,
        160,
        &[LocalBoundarySite {
            index: shape.index(0, 0, 0).unwrap(),
            condition: LocalPotentialBoundary::Dirichlet { value: 6.0 },
        }],
    )
    .unwrap();
    for (i, expected) in [6.0, 2.4, 1.2].into_iter().enumerate() {
        close(
            &format!("dirichlet[{i}]"),
            field.at(i, 0, 0).unwrap(),
            expected,
            1e-10,
        );
    }
    assert!(field.field_energy_joules() > 0.0);
}

/// LAW-LOCAL-NEU-001: zero normal flux across a marked interface.
/// With a single insulated source cell (rho=6, screening=2),
/// phi[0]=rho/screening=3, while its disconnected neighbours remain 0.
#[test]
fn law_local_neumann_blocks_normal_potential_flux() {
    let shape = GridShape::new(3, 1, 1).unwrap();
    let field = PotentialField::solve_with_local(
        shape,
        1.0,
        &[6.0, 0.0, 0.0],
        1.0,
        2.0,
        PotentialBoundary::Insulated,
        120,
        &[LocalBoundarySite {
            index: shape.index(0, 0, 0).unwrap(),
            condition: LocalPotentialBoundary::Neumann,
        }],
    )
    .unwrap();
    for (i, expected) in [3.0, 0.0, 0.0].into_iter().enumerate() {
        close(
            &format!("neumann[{i}]"),
            field.at(i, 0, 0).unwrap(),
            expected,
            1e-10,
        );
    }
}

/// LAW-LOCAL-NEU-002: the potential-driven manon flux through the
/// discretized insulating interface must be zero even with distinct
/// cell-centre potentials. The diffusion operator remains independent.
#[test]
fn law_local_neumann_blocks_gradient_driven_manon_transfer() {
    let shape = GridShape::new(2, 1, 1).unwrap();
    let field = PotentialField::solve_with_local(
        shape,
        1.0,
        &[6.0, 0.0],
        1.0,
        1.0,
        PotentialBoundary::Insulated,
        100,
        &[LocalBoundarySite {
            index: shape.index(0, 0, 0).unwrap(),
            condition: LocalPotentialBoundary::Neumann,
        }],
    )
    .unwrap();
    let mut n = FreeManaField::uniform(shape, 1.0, density(0.0)).unwrap();
    n.set(1, 0, 0, density(10.0)).unwrap();
    let before = n.total_manons().unwrap().value();
    let report = field.drift(&mut n, 0.5, 0.1).unwrap();
    close(
        "no gradient transfer across insulating interface",
        report.transported_manons,
        0.0,
        1e-12,
    );
    close(
        "no drift density change",
        n.at(1, 0, 0).unwrap().value(),
        10.0,
        1e-12,
    );
    close(
        "conserved inventory",
        n.total_manons().unwrap().value(),
        before,
        1e-12,
    );
}

/// LAW-LOCAL-ROB-001: surface exchange satisfies the cell-centre
/// finite-volume balance (screening + h/dx)*phi = rho + h*phi_ref/dx.
/// Here (2+4)*phi = 6 + 4*(-1), so phi=1/3.
#[test]
fn law_local_robin_matches_exact_finite_volume_exchange() {
    let shape = GridShape::new(1, 1, 1).unwrap();
    let field = PotentialField::solve_with_local(
        shape,
        1.0,
        &[6.0],
        1.0,
        2.0,
        PotentialBoundary::Insulated,
        50,
        &[LocalBoundarySite {
            index: 0,
            condition: LocalPotentialBoundary::Robin {
                reference: -1.0,
                exchange: 4.0,
            },
        }],
    )
    .unwrap();
    close(
        "robin potential",
        field.at(0, 0, 0).unwrap(),
        1.0 / 3.0,
        1e-12,
    );
    let half = field.scaled(0.5).unwrap();
    close(
        "field decay with Robin reservoir",
        half.field_energy_joules(),
        field.field_energy_joules() * 0.25,
        1e-10,
    );
}

#[test]
fn overlapping_or_nonphysical_local_constraints_are_rejected() {
    let shape = GridShape::new(1, 1, 1).unwrap();
    let site = LocalBoundarySite {
        index: 0,
        condition: LocalPotentialBoundary::Dirichlet { value: 1.0 },
    };
    assert!(
        PotentialField::solve_with_local(
            shape,
            1.0,
            &[0.0],
            1.0,
            1.0,
            PotentialBoundary::Insulated,
            20,
            &[site, site]
        )
        .is_err()
    );
    assert!(
        PotentialField::solve_with_local(
            shape,
            1.0,
            &[0.0],
            1.0,
            1.0,
            PotentialBoundary::Insulated,
            20,
            &[LocalBoundarySite {
                index: 0,
                condition: LocalPotentialBoundary::Robin {
                    reference: 0.0,
                    exchange: -1.0,
                },
            }]
        )
        .is_err()
    );
}

