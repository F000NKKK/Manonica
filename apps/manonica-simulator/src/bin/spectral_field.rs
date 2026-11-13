//! Periodic, mean-zero dynamic scalar field with Gaussian form factors.
//! Experimental stand, separate from the Lab solver; see mechanical-momentum.md.
use manonica_core::{Quantity, TimeStep, Vec3};
use manonica_physics::MechanicalBody;

#[derive(Clone, Debug, PartialEq)]
struct Source {
    body: MechanicalBody,
    charge: f64,
    width: f64,
}
#[derive(Clone, Copy, Debug, PartialEq)]
struct Mode {
    k: f64,
    a: f64,
    b: f64,
    pa: f64,
    pb: f64,
}
#[derive(Clone, Debug, PartialEq)]
struct System {
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
impl System {
    fn static_field(sources: Vec<Source>, count: usize) -> Self {
        let modes = (1..=count)
            .map(|j| {
                let k = j as f64;
                let mut a = 0.0;
                let mut b = 0.0;
                for s in &sources {
                    let (sin, cos) = (k * s.body.position().x.value()).sin_cos();
                    a -= s.charge * factor(s, k) * cos / (k * k + 1.0);
                    b -= s.charge * factor(s, k) * sin / (k * k + 1.0);
                }
                Mode {
                    k,
                    a,
                    b,
                    pa: 0.0,
                    pb: 0.0,
                }
            })
            .collect();
        Self { sources, modes }
    }
    fn force(&self, s: &Source) -> f64 {
        self.modes
            .iter()
            .map(|m| {
                let (sin, cos) = (m.k * s.body.position().x.value()).sin_cos();
                s.charge * factor(s, m.k) * m.k * (m.a * sin - m.b * cos)
            })
            .sum()
    }
    fn momentum(&self) -> f64 {
        self.sources
            .iter()
            .map(|s| s.body.momentum().x.value())
            .sum::<f64>()
            + self
                .modes
                .iter()
                .map(|m| m.k * (m.a * m.pb - m.b * m.pa))
                .sum::<f64>()
    }
    fn energy(&self) -> Result<f64, String> {
        let mut energy = 0.0;
        for s in &self.sources {
            energy += s.body.kinetic_energy().map_err(|e| e.to_string())?.value();
        }
        for m in &self.modes {
            energy +=
                0.5 * (m.pa * m.pa + m.pb * m.pb + (m.k * m.k + 1.0) * (m.a * m.a + m.b * m.b));
            for s in &self.sources {
                let (sin, cos) = (m.k * s.body.position().x.value()).sin_cos();
                energy += s.charge * factor(s, m.k) * (m.a * cos + m.b * sin);
            }
        }
        if !energy.is_finite() {
            return Err("nonfinite spectral energy".into());
        }
        Ok(energy)
    }
    fn kick(&mut self, dt: f64) -> Result<(), String> {
        let forces: Vec<_> = self.sources.iter().map(|s| self.force(s)).collect();
        for (s, f) in self.sources.iter_mut().zip(forces) {
            s.body
                .apply_impulse(Vec3::try_values(dt * f, 0.0, 0.0).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        }
        for m in &mut self.modes {
            let mut ca = 0.0;
            let mut cb = 0.0;
            for s in &self.sources {
                let (sin, cos) = (m.k * s.body.position().x.value()).sin_cos();
                ca += s.charge * factor(s, m.k) * cos;
                cb += s.charge * factor(s, m.k) * sin;
            }
            m.pa -= dt * ((m.k * m.k + 1.0) * m.a + ca);
            m.pb -= dt * ((m.k * m.k + 1.0) * m.b + cb);
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
            m.a += dt.seconds() * m.pa;
            m.b += dt.seconds() * m.pb;
        }
        next.kick(dt.seconds() / 2.0)?;
        next.energy()?;
        if !next.momentum().is_finite() {
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
fn main() -> Result<(), String> {
    validate()
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
