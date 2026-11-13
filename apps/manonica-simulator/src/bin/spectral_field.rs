//! Periodic, mean-zero dynamic scalar field with Gaussian form factors.
//! Experimental stand, separate from the Lab solver; see mechanical-momentum.md.
use manonica_core::{Quantity, TimeStep, Vec3};
use manonica_physics::MechanicalBody;
use std::f64::consts::{PI, TAU};

#[derive(Clone, Debug, PartialEq)]
struct Source {
    body: MechanicalBody,
    charge: f64,
    width: f64,
}
#[derive(Clone, Copy, Debug, PartialEq)]
struct Mode {
    k: f64,
    wave: [f64; 3],
    a: f64,
    b: f64,
    pa: f64,
    pb: f64,
}
#[derive(Clone, Debug, PartialEq)]
struct System {
    inertia: f64,
    sources: Vec<Source>,
    modes: Vec<Mode>,
}
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
fn factor(s: &Source, k: f64) -> f64 {
    (-0.5 * (k * s.width).powi(2)).exp()
}
fn coordinates(body: MechanicalBody) -> [f64; 3] {
    let v = body.position();
    [v.x.value(), v.y.value(), v.z.value()]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}
impl System {
    fn static_field(sources: Vec<Source>, count: usize) -> Self {
        Self::static_field_in_domain(sources, count, TAU).expect("valid fixed spectral model")
    }
    fn static_field_in_domain(
        sources: Vec<Source>,
        count: usize,
        length: f64,
    ) -> Result<Self, String> {
        if !length.is_finite()
            || length <= 0.0
            || !(1..=4096).contains(&count)
            || sources
                .iter()
                .any(|s| !s.charge.is_finite() || !s.width.is_finite() || s.width <= 0.0)
        {
            return Err("invalid spectral domain or source".into());
        }
        // a=epsilon=screening=1/pi fixed in physical units, mu=a*L/2.
        let inertia = length / TAU;
        if !inertia.is_finite() || inertia <= 0.0 {
            return Err("invalid modal inertia".into());
        }
        let waves = (1..=count)
            .map(|j| [j as f64 * TAU / length, 0.0, 0.0])
            .collect();
        Self::static_modes(sources, waves, inertia)
    }
    fn static_modes(
        sources: Vec<Source>,
        waves: Vec<[f64; 3]>,
        inertia: f64,
    ) -> Result<Self, String> {
        if !inertia.is_finite()
            || inertia <= 0.0
            || waves.is_empty()
            || sources
                .iter()
                .any(|s| !s.charge.is_finite() || !s.width.is_finite() || s.width <= 0.0)
        {
            return Err("invalid spectral mode parameters".into());
        }
        let mut modes = Vec::with_capacity(waves.len());
        for wave in waves {
            let k = dot(wave, wave).sqrt();
            if !k.is_finite() || k <= 0.0 {
                return Err("invalid wave vector".into());
            }
            let mut a = 0.0;
            let mut b = 0.0;
            for s in &sources {
                let (sin, cos) = dot(wave, coordinates(s.body)).sin_cos();
                a -= s.charge * factor(s, k) * cos / (inertia * (k * k + 1.0));
                b -= s.charge * factor(s, k) * sin / (inertia * (k * k + 1.0));
            }
            modes.push(Mode {
                k,
                wave,
                a,
                b,
                pa: 0.0,
                pb: 0.0,
            });
        }
        let system = Self {
            inertia,
            sources,
            modes,
        };
        system.energy()?;
        if system.momentum_vector().iter().any(|p| !p.is_finite()) {
            return Err("invalid spectral momentum".into());
        }
        Ok(system)
    }
    fn force_vector(&self, s: &Source) -> [f64; 3] {
        let mut force = [0.0; 3];
        for m in &self.modes {
            let (sin, cos) = dot(m.wave, coordinates(s.body)).sin_cos();
            let scalar = s.charge * factor(s, m.k) * (m.a * sin - m.b * cos);
            for (axis, value) in force.iter_mut().enumerate() {
                *value += m.wave[axis] * scalar;
            }
        }
        force
    }
    fn force(&self, s: &Source) -> f64 {
        self.force_vector(s)[0]
    }
    fn momentum_vector(&self) -> [f64; 3] {
        let mut result = [0.0; 3];
        for s in &self.sources {
            let p = s.body.momentum();
            for (a, b) in result
                .iter_mut()
                .zip([p.x.value(), p.y.value(), p.z.value()])
            {
                *a += b;
            }
        }
        for m in &self.modes {
            for (axis, value) in result.iter_mut().enumerate() {
                *value += m.wave[axis] * (m.a * m.pb - m.b * m.pa);
            }
        }
        result
    }
    fn momentum(&self) -> f64 {
        self.momentum_vector()[0]
    }
    fn energy(&self) -> Result<f64, String> {
        let mut energy = 0.0;
        for s in &self.sources {
            energy += s.body.kinetic_energy().map_err(|e| e.to_string())?.value();
        }
        for m in &self.modes {
            energy += 0.5
                * ((m.pa * m.pa + m.pb * m.pb) / self.inertia
                    + self.inertia * (m.k * m.k + 1.0) * (m.a * m.a + m.b * m.b));
            for s in &self.sources {
                let (sin, cos) = dot(m.wave, coordinates(s.body)).sin_cos();
                energy += s.charge * factor(s, m.k) * (m.a * cos + m.b * sin);
            }
        }
        if !energy.is_finite() {
            return Err("nonfinite spectral energy".into());
        }
        Ok(energy)
    }
    fn kick(&mut self, dt: f64) -> Result<(), String> {
        let forces: Vec<_> = self.sources.iter().map(|s| self.force_vector(s)).collect();
        for (s, f) in self.sources.iter_mut().zip(forces) {
            s.body
                .apply_impulse(
                    Vec3::try_values(dt * f[0], dt * f[1], dt * f[2]).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
        }
        for m in &mut self.modes {
            let mut ca = 0.0;
            let mut cb = 0.0;
            for s in &self.sources {
                let (sin, cos) = dot(m.wave, coordinates(s.body)).sin_cos();
                ca += s.charge * factor(s, m.k) * cos;
                cb += s.charge * factor(s, m.k) * sin;
            }
            m.pa -= dt * (self.inertia * (m.k * m.k + 1.0) * m.a + ca);
            m.pb -= dt * (self.inertia * (m.k * m.k + 1.0) * m.b + cb);
        }
        Ok(())
    }
    fn step(&mut self, dt: TimeStep) -> Result<(), String> {
        let mut next = self.clone();
        next.kick(dt.seconds() / 2.0)?;
        for s in &mut next.sources {
            s.body.advance_free(dt).map_err(|e| e.to_string())?;
        }
        for m in &mut next.modes {
            m.a += dt.seconds() * m.pa / self.inertia;
            m.b += dt.seconds() * m.pb / self.inertia;
        }
        next.kick(dt.seconds() / 2.0)?;
        next.energy()?;
        if next.momentum_vector().iter().any(|p| !p.is_finite()) {
            return Err("nonfinite spectral momentum".into());
        }
        *self = next;
        Ok(())
    }
}
fn setup(count: usize) -> System {
    System::static_field(
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
            let system = System::static_field_in_domain(
                vec![source(0.4, 0.0, 0.4, width), source(2.0, 0.0, -0.4, width)],
                64 * scale,
                length,
            )?;
            let force = system.force(&system.sources[0]);
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
                    let system = System::static_field(vec![source(x, 0.0, 0.4, width)], count);
                    assert!(system.force(&system.sources[0]).abs() < 1e-14);
                    assert!(system.momentum().abs() < 1e-14);
                }
            }
        }
    }
    #[test]
    fn law_spectral_002_pair_forces_have_equal_recoil_and_translation_symmetry() {
        let system = setup(16);
        let left = system.force(&system.sources[0]);
        let right = system.force(&system.sources[1]);
        assert!(left.abs() > 1e-4);
        assert!((left + right).abs() < 1e-14);
        let shifted = System::static_field(
            vec![
                source(1.13, 0.15, 0.4, 0.35),
                source(2.73, -0.1, -0.4, 0.35),
            ],
            16,
        );
        assert!((shifted.force(&shifted.sources[0]) - left).abs() < 1e-14);
        assert!((shifted.energy().unwrap() - system.energy().unwrap()).abs() < 1e-14);
    }
    #[test]
    fn law_spectral_003_static_energy_tail_is_bounded_by_gaussian_kernel() {
        let charge: f64 = 0.4;
        let width: f64 = 0.35;
        let exact = System::static_field(vec![source(0.37, 0.0, charge, width)], 64)
            .energy()
            .unwrap();
        for count in [2, 4, 8, 16] {
            let energy = System::static_field(vec![source(0.37, 0.0, charge, width)], count)
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
            for (a, b) in observed.sources.iter().zip(&reference.sources) {
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
            let system = System::static_field_in_domain(
                vec![source(0.4, 0.0, 0.4, width), source(2.0, 0.0, -0.4, width)],
                count,
                length,
            )
            .unwrap();
            errors.push((system.force(&system.sources[0]) - exact).abs());
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
                let mut system = System::static_field_in_domain(
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
            assert!(System::static_field_in_domain(vec![], 16, length).is_err());
        }
        for count in [0, 4097] {
            assert!(System::static_field_in_domain(vec![], count, TAU).is_err());
        }
        for width in [0.0, -1.0, f64::NAN] {
            assert!(
                System::static_field_in_domain(vec![source(0.0, 0.0, 0.4, width)], 16, TAU)
                    .is_err()
            );
        }
    }
}
