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

struct UniformVelocity([f64; 3]);

impl manonica_physics::VelocityField for UniformVelocity {
    fn at(
        &self,
        _: manonica_core::Vec3<manonica_core::units::Metres>,
    ) -> Result<
        manonica_core::Vec3<manonica_core::units::MetresPerSecond>,
        manonica_physics::PhysicsError,
    > {
        Ok(manonica_core::Vec3::try_values(
            self.0[0], self.0[1], self.0[2],
        )?)
    }
}

/// LAW-ADV-001: after m upwind steps an impulse has binomial weights
/// choose(m,k) C^k (1-C)^(m-k). At the impermeable downstream wall,
/// all weights reaching or passing the last cell accumulate there.
#[test]
fn law_advection_binomial_profile_and_wall_accumulation_on_all_axes() {
    const N: usize = 8;
    const STEPS: usize = 12;
    let c: f64 = 0.4;
    let dx: f64 = 0.5;
    for axis in 0..3 {
        for sign in [-1.0, 1.0] {
            let mut dimensions = [1; 3];
            dimensions[axis] = N;
            let shape = GridShape::new(dimensions[0], dimensions[1], dimensions[2]).unwrap();
            let mut field = FreeManaField::uniform(shape, dx, density(0.0)).unwrap();
            let mut start = [0; 3];
            if sign < 0.0 {
                start[axis] = N - 1;
            }
            field
                .set(start[0], start[1], start[2], density(10.0))
                .unwrap();
            let mut velocity = [0.0; 3];
            velocity[axis] = sign;
            let law = manonica_physics::FreeAdvection::new(UniformVelocity(velocity));
            for _ in 0..STEPS {
                law.evolve(&mut field, TimeStep::from_seconds(c * dx).unwrap())
                    .unwrap();
            }
            let mut expected = [0.0; N];
            let mut probability = (1.0 - c).powi(STEPS as i32);
            for k in 0..=STEPS {
                expected[k.min(N - 1)] += 10.0 * probability;
                probability *= (STEPS - k) as f64 / (k + 1) as f64 * c / (1.0 - c);
            }
            for (k, expected) in expected.into_iter().enumerate() {
                let mut position = [0; 3];
                position[axis] = if sign > 0.0 { k } else { N - 1 - k };
                let actual = field
                    .at(position[0], position[1], position[2])
                    .unwrap()
                    .value();
                assert!(actual >= 0.0);
                close("binomial transport profile", actual, expected, 1e-11);
            }
            close(
                "closed inventory",
                field.total_manons().unwrap().value(),
                10.0 * dx.powi(3),
                1e-11,
            );
            let before = field.densities().to_vec();
            assert!(
                law.evolve(&mut field, TimeStep::from_seconds(1.01 * dx).unwrap())
                    .is_err()
            );
            assert_eq!(field.densities(), before, "CFL rejection must be atomic");
        }
    }
}

/// LAW-ADV-002: n(x,t)=f(x-vt). Integrate the compact sin^4 pulse
/// over each cell independently; it stays away from both walls.
/// Fixed CFL refinement halves dx and dt together (first-order upwind).
#[test]
fn law_advection_converges_to_translated_continuum_profile() {
    fn primitive(x: f64) -> f64 {
        let u = (x - 0.5).clamp(0.0, 1.0);
        let pi = std::f64::consts::PI;
        3.0 * u / 8.0 - (2.0 * pi * u).sin() / (4.0 * pi) + (4.0 * pi * u).sin() / (32.0 * pi)
    }
    let mut errors = Vec::new();
    for n in [40, 80, 160] {
        let dx = 4.0 / n as f64;
        let dt = 0.4 * dx;
        let steps = n / 8;
        let time = steps as f64 * dt;
        let shape = GridShape::new(n, 1, 1).unwrap();
        let mut field = FreeManaField::uniform(shape, dx, density(0.0)).unwrap();
        for x in 0..n {
            let left = x as f64 * dx;
            field
                .set(
                    x,
                    0,
                    0,
                    density((primitive(left + dx) - primitive(left)) / dx),
                )
                .unwrap();
        }
        let before = field.total_manons().unwrap().value();
        let law = manonica_physics::FreeAdvection::new(UniformVelocity([1.0, 0.0, 0.0]));
        for _ in 0..steps {
            law.evolve(&mut field, TimeStep::from_seconds(dt).unwrap())
                .unwrap();
        }
        let error = (0..n)
            .map(|x| {
                let left = x as f64 * dx - time;
                let exact = (primitive(left + dx) - primitive(left)) / dx;
                (field.at(x, 0, 0).unwrap().value() - exact).abs() * dx
            })
            .sum::<f64>();
        errors.push(error);
        close(
            "continuum profile inventory",
            field.total_manons().unwrap().value(),
            before,
            1e-12,
        );
    }
    for pair in errors.windows(2) {
        let ratio = pair[0] / pair[1];
        assert!(
            (1.5..2.5).contains(&ratio),
            "expected first-order convergence, errors={errors:?}"
        );
    }
}

/// Run the same insulated cosine mode with a prescribed reference decay rate.
fn diffusion_mode_error(cells: usize, steps: usize, time: f64, reference_rate: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let dx = 1.0 / cells as f64;
    // Cell averages of the continuum cosine, rather than point samples.
    let average_factor = (pi * dx / 2.0).sin() / (pi * dx / 2.0);
    let mode = |x: usize| (pi * (x as f64 + 0.5) * dx).cos();
    let shape = GridShape::new(cells, 1, 1).unwrap();
    let mut field = FreeManaField::uniform(shape, dx, density(0.0)).unwrap();
    for x in 0..cells {
        field
            .set(x, 0, 0, density(6.0 + 2.0 * average_factor * mode(x)))
            .unwrap();
    }
    let inventory = field.total_manons().unwrap().value();
    let law = FreeDiffusion::new(Quantity::<Diffusivity>::nonnegative(0.1).unwrap()).unwrap();
    let dt = TimeStep::from_seconds(time / steps as f64).unwrap();
    for _ in 0..steps {
        law.evolve(&mut field, dt).unwrap();
    }
    close(
        "refined diffusion inventory",
        field.total_manons().unwrap().value(),
        inventory,
        1e-11,
    );
    assert!(field.densities().iter().all(|n| n.value() >= 0.0));
    (0..cells)
        .map(|x| {
            let exact = 6.0 + 2.0 * average_factor * (-reference_rate * time).exp() * mode(x);
            (field.at(x, 0, 0).unwrap().value() - exact).abs()
        })
        .fold(0.0, f64::max)
}

/// LAW-DIFF-002: fix dx and compare with the exact semidiscrete solution
/// exp(D*lambda_h*t), so spatial truncation cannot mask temporal convergence.
#[test]
fn law_diffusion_first_order_time_convergence_at_fixed_grid() {
    let cells = 12;
    let dx: f64 = 1.0 / cells as f64;
    let rate = 0.1 * 4.0 * (std::f64::consts::PI / (2.0 * cells as f64)).sin().powi(2) / dx.powi(2);
    let errors: Vec<_> = [10, 20, 40, 80]
        .into_iter()
        .map(|steps| diffusion_mode_error(cells, steps, 0.1, rate))
        .collect();
    for pair in errors.windows(2) {
        assert!(
            (1.9..2.1).contains(&(pair[0] / pair[1])),
            "temporal errors: {errors:?}"
        );
    }
    assert!(errors[3] < 0.0002, "temporal absolute error: {errors:?}");
}

/// LAW-DIFF-003: refine dx with dt fixed at 1e-5 s and compare against
/// the continuum mode exp(-D*pi²*t). The time error is bounded separately.
#[test]
fn law_diffusion_second_order_space_convergence_at_fixed_timestep() {
    let rate = 0.1 * std::f64::consts::PI.powi(2);
    let mut errors = Vec::new();
    for cells in [8, 16, 32] {
        errors.push(diffusion_mode_error(cells, 10_000, 0.1, rate));
        let dx = 1.0 / cells as f64;
        let discrete_rate =
            0.1 * 4.0 * (std::f64::consts::PI / (2.0 * cells as f64)).sin().powi(2) / (dx * dx);
        let temporal_error = diffusion_mode_error(cells, 10_000, 0.1, discrete_rate);
        assert!(
            temporal_error < 1e-6,
            "time error must be below spatial error: {temporal_error}"
        );
        assert!(temporal_error < errors.last().unwrap() * 0.01);
    }
    for pair in errors.windows(2) {
        assert!(
            (3.8..4.2).contains(&(pair[0] / pair[1])),
            "spatial errors: {errors:?}"
        );
    }
    assert!(errors[2] < 0.0002, "spatial absolute error: {errors:?}");
}

/// LAW-POT-002: manufacture phi=3+cos(pi*x) on [0,1] with insulated ends.
/// rho=6+(2+0.1*pi²)*cos(pi*x). Cell-average exact references separate
/// finite-volume truncation from the iterative solver's stopping error.
#[test]
fn law_screened_potential_second_order_manufactured_solution() {
    let pi = std::f64::consts::PI;
    let mut errors = Vec::new();
    for cells in [8, 16, 32] {
        let dx = 1.0 / cells as f64;
        let factor = (pi * dx / 2.0).sin() / (pi * dx / 2.0);
        let mode = |x: usize| factor * (pi * (x as f64 + 0.5) * dx).cos();
        let sources: Vec<_> = (0..cells)
            .map(|x| 6.0 + (2.0 + 0.1 * pi * pi) * mode(x))
            .collect();
        let potential = PotentialField::solve(
            GridShape::new(cells, 1, 1).unwrap(),
            dx,
            &sources,
            0.1,
            2.0,
            PotentialBoundary::Insulated,
            4096,
        )
        .unwrap();
        let eigenvalue = 4.0 * (pi / (2.0 * cells as f64)).sin().powi(2) / (dx * dx);
        let discrete_amplitude = (2.0 + 0.1 * pi * pi) / (2.0 + 0.1 * eigenvalue);
        let mut error: f64 = 0.0;
        for x in 0..cells {
            let observed = potential.at(x, 0, 0).unwrap();
            close(
                "converged discrete Poisson solution",
                observed,
                3.0 + discrete_amplitude * mode(x),
                1e-9,
            );
            error = error.max((observed - (3.0 + mode(x))).abs());
        }
        errors.push(error);
    }
    for pair in errors.windows(2) {
        assert!(
            (3.8..4.2).contains(&(pair[0] / pair[1])),
            "Poisson spatial errors: {errors:?}"
        );
    }
    assert!(errors[2] < 0.0003, "Poisson absolute error: {errors:?}");
}
