//! Periodic, mean-zero dynamic scalar field with Gaussian form factors.
//! Experimental stand, separate from the Lab solver; see mechanical-momentum.md.
use manonica_core::{Quantity, TimeStep, Vec3};
use manonica_physics::MechanicalBody;
use std::f64::consts::{PI, TAU};

use manonica_physics::{
    GaussianBody as Source, PeriodicSpectralDomain, SpectralMechanics as System,
};

fn source(x: f64, p: f64, charge: f64, width: f64) -> Source {
    Source {
        body: MechanicalBody::new(
            Quantity::new(2.0).unwrap(),
            Vec3::try_values(x, 0.0, 0.0).unwrap(),
            Vec3::try_values(p, 0.0, 0.0).unwrap(),
        )
        .unwrap(),
        charge,
        width,
    }
}
#[cfg(test)]
fn coordinates(body: MechanicalBody) -> [f64; 3] {
    let v = body.position();
    [v.x.value(), v.y.value(), v.z.value()]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}
// CLI adapters only; all physical stepping/forces are owned by manonica-physics.
fn equilibrium_default(sources: Vec<Source>, count: usize) -> System {
    equilibrium_line(sources, count, TAU).expect("valid fixed spectral model")
}
fn equilibrium_line(sources: Vec<Source>, count: usize, length: f64) -> Result<System, String> {
    PeriodicSpectralDomain::new(
        Vec3::try_values(length, 1.0, 1.0).map_err(|e| e.to_string())?,
        [count, 0, 0],
    )
    .and_then(|domain| domain.static_equilibrium(sources))
    .map_err(|e| e.to_string())
}
trait StandObservables {
    fn energy(&self) -> Result<f64, String>;
    fn momentum_vector(&self) -> [f64; 3];
    fn momentum(&self) -> f64 {
        self.momentum_vector()[0]
    }
    fn force_vector(&self, source: &Source) -> [f64; 3];
    fn force(&self, source: &Source) -> f64 {
        self.force_vector(source)[0]
    }
    fn step(&mut self, dt: TimeStep) -> Result<(), String>;
    #[cfg(test)]
    fn kick(&mut self, seconds: f64) -> Result<(), String>;
}
impl StandObservables for System {
    fn energy(&self) -> Result<f64, String> {
        self.total_energy()
            .map(|e| e.value())
            .map_err(|e| e.to_string())
    }
    fn momentum_vector(&self) -> [f64; 3] {
        let p = self.total_momentum().expect("validated state");
        [p.x.value(), p.y.value(), p.z.value()]
    }
    fn force_vector(&self, source: &Source) -> [f64; 3] {
        let index = self
            .sources()
            .iter()
            .position(|s| std::ptr::eq(s, source))
            .expect("owned source");
        let f = self.force_on(index).expect("finite force");
        [f.x.value(), f.y.value(), f.z.value()]
    }
    fn step(&mut self, dt: TimeStep) -> Result<(), String> {
        self.advance(dt).map_err(|e| e.to_string())
    }
    #[cfg(test)]
    fn kick(&mut self, seconds: f64) -> Result<(), String> {
        self.potential_kick(TimeStep::from_seconds(seconds).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())
    }
}
fn setup(count: usize) -> System {
    equilibrium_default(
        vec![source(0.4, 0.15, 0.4, 0.35), source(2.0, -0.1, -0.4, 0.35)],
        count,
    )
}
fn experiment(count: usize, steps: usize) -> Result<(System, f64, f64), String> {
    let mut system = setup(count);
    let energy = system.energy()?;
    let momentum = system.momentum();
    let mut maximum_energy: f64 = 0.0;
    let mut maximum_momentum: f64 = 0.0;
    for _ in 0..steps {
        system.step(TimeStep::from_seconds(1.0 / steps as f64).map_err(|e| e.to_string())?)?;
        maximum_energy = maximum_energy.max((system.energy()? - energy).abs());
        maximum_momentum = maximum_momentum.max((system.momentum() - momentum).abs());
    }
    if maximum_momentum > 1e-11 {
        return Err(format!("spectral momentum error {maximum_momentum}"));
    }
    Ok((system, maximum_energy, maximum_momentum))
}
fn validate() -> Result<(), String> {
    let mut previous: Option<f64> = None;
    for steps in [100, 200, 400] {
        let (_, energy, momentum) = experiment(16, steps)?;
        if let Some(error) = previous
            && !(3.8..4.2).contains(&(error / energy))
        {
            return Err(format!("spectral energy refinement {}", error / energy));
        }
        if steps == 400 && energy > 1e-5 {
            return Err(format!("spectral energy error {energy}"));
        }
        previous = Some(energy);
        println!(
            "modes=16, steps={steps}, energy error={energy:.6e} J, total momentum error={momentum:.6e} kg m/s"
        );
    }
    Ok(())
}
#[path = "spectral_field/three_dimensional.rs"]
mod three_dimensional;

fn main() -> Result<(), String> {
    validate()?;
    validate_domain()?;
    three_dimensional::validate_3d()?;
    three_dimensional::validate_coordinates()
}

// Independent real-space convolution with the screened Green function.
// G(r)=pi/2*exp(-|r|); Gaussian widths combine in quadrature.
fn real_space_force(separation: f64, width: f64, intervals: usize) -> f64 {
    let upper = separation.abs() + 12.0 * width;
    let h = upper / intervals as f64;
    let normal = |u: f64| (-0.5 * ((u - separation) / width).powi(2)).exp() / (TAU.sqrt() * width);
    let integrand = |u: f64| (-u).exp() * (normal(-u) - normal(u));
    let mut sum = integrand(0.0) + integrand(upper);
    for i in 1..intervals {
        sum += if i % 2 == 0 { 2.0 } else { 4.0 } * integrand(i as f64 * h);
    }
    -0.16 * PI / 2.0 * sum * h / 3.0
}
fn periodic_reference(length: f64, width: f64) -> f64 {
    (-4..=4)
        .map(|image| real_space_force(-1.6 + image as f64 * length, 2.0_f64.sqrt() * width, 8192))
        .sum()
}
fn validate_domain() -> Result<(), String> {
    for width in [0.2, 0.35, 0.6] {
        let infinite = real_space_force(-1.6, 2.0_f64.sqrt() * width, 8192);
        let quadrature = real_space_force(-1.6, 2.0_f64.sqrt() * width, 4096);
        if (infinite - quadrature).abs() > 1e-10 {
            return Err("Green-function quadrature not converged".into());
        }
        let mut previous = f64::INFINITY;
        for scale in [1, 2, 4] {
            let length = scale as f64 * TAU;
            let system = equilibrium_line(
                vec![source(0.4, 0.0, 0.4, width), source(2.0, 0.0, -0.4, width)],
                64 * scale,
                length,
            )?;
            let force = system.force(&system.sources()[0]);
            let periodic = periodic_reference(length, width);
            if (force - periodic).abs() > 1e-9 {
                return Err(format!(
                    "spectral/real-space force mismatch: {force} vs {periodic}"
                ));
            }
            let error = (force - infinite).abs();
            let image_bound =
                2.0 * PI * 0.16 * (width * width).exp() * 1.6_f64.cosh() * (-length).exp()
                    / (1.0 - (-length).exp());
            if error > image_bound + 1e-10 || error >= previous {
                return Err(format!("periodic image convergence: {error}"));
            }
            if scale == 4 && error > 1e-9 {
                return Err(format!("large-domain force error: {error}"));
            }
            previous = error;
            println!(
                "sigma={width:.2} m, L={length:.6} m, force={force:.9e} N, infinite-domain error={error:.3e} N"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn law_spectral_001_static_gaussian_has_no_self_force_at_any_phase() {
        for count in [1, 4, 16, 32] {
            for width in [0.2, 0.35, 0.6] {
                for x in [0.0, 0.13, 1.7, 4.2, 6.1] {
                    let system = equilibrium_default(vec![source(x, 0.0, 0.4, width)], count);
                    assert!(system.force(&system.sources()[0]).abs() < 1e-14);
                    assert!(system.momentum().abs() < 1e-14);
                }
            }
        }
    }
    #[test]
    fn law_spectral_002_pair_forces_have_equal_recoil_and_translation_symmetry() {
        let system = setup(16);
        let left = system.force(&system.sources()[0]);
        let right = system.force(&system.sources()[1]);
        assert!(left.abs() > 1e-4);
        assert!((left + right).abs() < 1e-14);
        let shifted = equilibrium_default(
            vec![
                source(1.13, 0.15, 0.4, 0.35),
                source(2.73, -0.1, -0.4, 0.35),
            ],
            16,
        );
        assert!((shifted.force(&shifted.sources()[0]) - left).abs() < 1e-14);
        assert!((shifted.energy().unwrap() - system.energy().unwrap()).abs() < 1e-14);
    }
    #[test]
    fn law_spectral_003_static_energy_tail_is_bounded_by_gaussian_kernel() {
        let charge: f64 = 0.4;
        let width: f64 = 0.35;
        let exact = equilibrium_default(vec![source(0.37, 0.0, charge, width)], 64)
            .energy()
            .unwrap();
        for count in [2, 4, 8, 16] {
            let energy = equilibrium_default(vec![source(0.37, 0.0, charge, width)], count)
                .energy()
                .unwrap();
            let first = (count + 1) as f64;
            let bound = charge * charge / 2.0 * (-(width * first).powi(2)).exp()
                / (first * first + 1.0)
                / (1.0 - (-width * width * (2.0 * first + 1.0)).exp());
            let error = (energy - exact).abs();
            assert!(
                error <= bound + 1e-15,
                "count={count}, error={error}, bound={bound}"
            );
            if count == 16 {
                assert!(error < 1e-12);
            }
        }
    }
    #[test]
    fn law_spectral_004_dynamic_momentum_energy_and_mode_cutoff_converge() {
        validate().unwrap();
        let (reference, _, _) = experiment(32, 400).unwrap();
        for (count, tolerance) in [(8, 1e-5), (16, 1e-11)] {
            let (observed, _, _) = experiment(count, 400).unwrap();
            for (a, b) in observed.sources().iter().zip(reference.sources()) {
                assert!(
                    (a.body.position().x.value() - b.body.position().x.value()).abs() < tolerance
                );
                assert!(
                    (a.body.momentum().x.value() - b.body.momentum().x.value()).abs() < tolerance
                );
            }
        }
    }
    #[test]
    fn spectral_failure_never_commits_partial_body_or_field_state() {
        let mut system = setup(16);
        let before = system.clone();
        assert!(
            system
                .step(TimeStep::from_seconds(f64::MAX).unwrap())
                .is_err()
        );
        assert_eq!(system, before);
    }
}

#[cfg(test)]
mod domain_tests {
    use super::*;
    #[test]
    fn law_spectral_005_domain_and_kernel_match_real_space_reference() {
        validate_domain().unwrap();
    }
    #[test]
    fn law_spectral_006_narrow_kernel_approaches_point_pair_force() {
        let length = 4.0 * TAU;
        let exact = -0.16 * PI / 2.0 * (-1.6_f64).exp();
        let mut errors = Vec::new();
        for width in [0.2, 0.1, 0.05] {
            let count = (32.0 / width) as usize;
            let system = equilibrium_line(
                vec![source(0.4, 0.0, 0.4, width), source(2.0, 0.0, -0.4, width)],
                count,
                length,
            )
            .unwrap();
            errors.push((system.force(&system.sources()[0]) - exact).abs());
        }
        for pair in errors.windows(2) {
            assert!(
                (3.8..4.2).contains(&(pair[0] / pair[1])),
                "point-limit errors: {errors:?}"
            );
        }
        assert!(errors[2] < 0.0002);
    }
    #[test]
    fn normalized_periods_preserve_dynamic_balances_and_reject_invalid_inputs() {
        for scale in [1, 2] {
            for width in [0.2, 0.6] {
                let mut system = equilibrium_line(
                    vec![
                        source(0.4, 0.15, 0.4, width),
                        source(2.0, -0.1, -0.4, width),
                    ],
                    32 * scale,
                    TAU * scale as f64,
                )
                .unwrap();
                let energy = system.energy().unwrap();
                let momentum = system.momentum();
                for _ in 0..400 {
                    system
                        .step(TimeStep::from_seconds(0.0025).unwrap())
                        .unwrap();
                    assert!((system.momentum() - momentum).abs() < 1e-11);
                    assert!((system.energy().unwrap() - energy).abs() < 1e-5);
                }
            }
        }
        for length in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(equilibrium_line(vec![], 16, length).is_err());
        }
        for count in [0, 4097] {
            assert!(equilibrium_line(vec![], count, TAU).is_err());
        }
        for width in [0.0, -1.0, f64::NAN] {
            assert!(equilibrium_line(vec![source(0.0, 0.0, 0.4, width)], 16, TAU).is_err());
        }
    }
}
