//! Read-only local observables; sampling never advances or renormalizes physics.
use super::{SpectralMechanics, SpectralMode};
use manonica_core::{
    Quantity, Vec3,
    units::{JoulesPerCubicMetre, ManaPotential, Metres},
};
use manonica_error::BaseError;
#[derive(Clone, Debug)]
pub struct SpectralFieldSnapshot {
    modes: Vec<SpectralMode>,
    inertia: f64,
}
#[derive(Clone, Copy, Debug)]
pub struct SpectralFieldSample {
    pub potential: Quantity<ManaPotential>,
    /// Field kinetic + gradient + screened energy; excludes body/interaction energy.
    pub energy_density: Quantity<JoulesPerCubicMetre>,
}
impl SpectralMechanics {
    pub fn field_snapshot(&self) -> SpectralFieldSnapshot {
        SpectralFieldSnapshot {
            modes: self.modes.clone(),
            inertia: self.inertia,
        }
    }
}
impl SpectralFieldSnapshot {
    pub fn sample(&self, position: Vec3<Metres>) -> Result<SpectralFieldSample, BaseError> {
        let x = [position.x.value(), position.y.value(), position.z.value()];
        let mut potential = 0.0;
        let mut velocity = 0.0;
        let mut gradient = [0.0; 3];
        for m in &self.modes {
            let phase = (0..3).map(|i| m.wave[i] * x[i]).sum::<f64>();
            let (sin, cos) = phase.sin_cos();
            potential += m.a * cos + m.b * sin;
            velocity += (m.pa * cos + m.pb * sin) / self.inertia;
            for (axis, value) in gradient.iter_mut().enumerate() {
                *value += m.wave[axis] * (-m.a * sin + m.b * cos);
            }
        }
        let energy = (velocity * velocity
            + potential * potential
            + gradient.iter().map(|v| v * v).sum::<f64>())
            / (2.0 * std::f64::consts::PI);
        Ok(SpectralFieldSample {
            potential: Quantity::new(potential)?,
            energy_density: Quantity::nonnegative(energy)?,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_field_has_correct_sign_and_includes_all_energy_terms() {
        let mut system =
            SpectralMechanics::static_equilibrium(vec![], vec![[1.0, 0.0, 0.0]], 1.0).unwrap();
        system
            .replace_mode_storage(&[[2.0, 1.0, 3.0, 4.0]])
            .unwrap();
        let field = system.field_snapshot();
        let at_zero = field
            .sample(Vec3::try_values(0.0, 0.0, 0.0).unwrap())
            .unwrap();
        assert_eq!(at_zero.potential.value(), 2.0);
        assert!(
            (at_zero.energy_density.value() - 14.0 / (2.0 * std::f64::consts::PI)).abs() < 1e-12
        );
        let opposite = field
            .sample(Vec3::try_values(std::f64::consts::PI, 0.0, 0.0).unwrap())
            .unwrap();
        assert!((opposite.potential.value() + 2.0).abs() < 1e-12);
        assert!((opposite.energy_density.value() - at_zero.energy_density.value()).abs() < 1e-12);
    }
    #[test]
    fn integrated_local_energy_matches_modal_energy_and_snapshot_is_immutable() {
        let period = std::f64::consts::TAU;
        let mut system = super::super::PeriodicSpectralDomain::new(
            Vec3::try_values(period, period, period).unwrap(),
            [1; 3],
        )
        .unwrap()
        .static_equilibrium(vec![])
        .unwrap();
        let storage: Vec<_> = system
            .modes()
            .iter()
            .enumerate()
            .map(|(i, _)| [0.03 * (i + 1) as f64, -0.02, 0.01, 0.04])
            .collect();
        system.replace_mode_storage(&storage).unwrap();
        let snapshot = system.field_snapshot();
        let mut energy = 0.0;
        let n = 8;
        let dx = period / n as f64;
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let position = Vec3::try_values(
                        (x as f64 + 0.5) * dx,
                        (y as f64 + 0.5) * dx,
                        (z as f64 + 0.5) * dx,
                    )
                    .unwrap();
                    energy +=
                        snapshot.sample(position).unwrap().energy_density.value() * dx.powi(3);
                }
            }
        }
        assert!((energy - system.total_energy().unwrap().value()).abs() < 1e-10);
        let p = Vec3::try_values(0.3, 0.5, 0.7).unwrap();
        let before = snapshot.sample(p).unwrap().potential.value();
        system
            .replace_mode_storage(&vec![[0.0; 4]; storage.len()])
            .unwrap();
        assert_eq!(snapshot.sample(p).unwrap().potential.value(), before);
    }
}
