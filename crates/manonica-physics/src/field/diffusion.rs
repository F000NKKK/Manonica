use manonica_core::{
    Evolution, Quantity, TimeStep,
    units::{Diffusivity, ManonCount, ManonDensity},
};
use manonica_error::{BaseError, ErrorCode};

use crate::{PhysicsError, field::FreeManaField};

/// Только перенос СВОБОДНОЙ маны. Не содержит химических реакций или ловушек.
/// Явная 3D конечно-объёмная схема с нулевым нормальным потоком на границах.
#[derive(Debug, Clone, Copy)]
pub struct FreeDiffusion {
    diffusivity: Quantity<Diffusivity>,
}

#[derive(Debug, Clone, Copy)]
pub struct ConservationReport {
    pub before: Quantity<ManonCount>,
    pub after: Quantity<ManonCount>,
    pub absolute_error: Quantity<ManonCount>,
}

impl FreeDiffusion {
    pub fn new(diffusivity: Quantity<Diffusivity>) -> Result<Self, PhysicsError> {
        if diffusivity.value() < 0.0 {
            return Err(BaseError::new(ErrorCode::InvalidRange, "diffusivity").into());
        }
        Ok(Self { diffusivity })
    }

    pub fn diffusivity(&self) -> Quantity<Diffusivity> {
        self.diffusivity
    }
}

impl Evolution<FreeManaField> for FreeDiffusion {
    type Error = PhysicsError;
    type Report = ConservationReport;

    fn evolve(&self, field: &mut FreeManaField, dt: TimeStep) -> Result<Self::Report, Self::Error> {
        let dx = field.cell_width_metres();
        let lambda = self.diffusivity.value() * dt.seconds() / (dx * dx);
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(BaseError::new(ErrorCode::NonFinite, "diffusion_cfl").into());
        }
        // Простейшая достаточная оценка устойчивости для трёх измерений.
        if lambda > 1.0 / 6.0 {
            return Err(BaseError::new(ErrorCode::NumericalInstability, "diffusion_cfl").into());
        }

        let before = field.total_manons()?;
        let shape = field.shape();
        let old: Vec<f64> = field
            .densities()
            .iter()
            .map(|value| value.value())
            .collect();
        let mut next = old.clone();

        // Для каждой внутренней грани вычисляем один поток, добавляя его
        // в одну ячейку и вычитая из соседней: дискретная консервативность.
        for z in 0..shape.nz() {
            for y in 0..shape.ny() {
                for x in 0..shape.nx() {
                    let i = shape.index(x, y, z).expect("valid grid coordinate");
                    for (xx, yy, zz, exists) in [
                        (x + 1, y, z, x + 1 < shape.nx()),
                        (x, y + 1, z, y + 1 < shape.ny()),
                        (x, y, z + 1, z + 1 < shape.nz()),
                    ] {
                        if exists {
                            let j = shape.index(xx, yy, zz).expect("valid neighbour");
                            let transferred = lambda * (old[j] - old[i]);
                            next[i] += transferred;
                            next[j] -= transferred;
                        }
                    }
                }
            }
        }

        let mut converted = Vec::with_capacity(next.len());
        for density in next {
            let safe = if density < 0.0 && density > -1e-12 {
                0.0
            } else {
                density
            };
            if !safe.is_finite() || safe < 0.0 {
                return Err(BaseError::new(ErrorCode::NumericalInstability, "density").into());
            }
            converted.push(Quantity::<ManonDensity>::nonnegative(safe)?);
        }

        // Количественный баланс проверяем до принятия шага.
        let cell_volume = dx.powi(3);
        let after_value: f64 = converted.iter().map(|n| n.value()).sum::<f64>() * cell_volume;
        let absolute = (after_value - before.value()).abs();
        let tolerance = 1e-10 * before.value().max(1.0);
        if !after_value.is_finite() || absolute > tolerance {
            return Err(BaseError::new(ErrorCode::NumericalInstability, "conservation").into());
        }
        let after = Quantity::<ManonCount>::nonnegative(after_value)?;
        let absolute_error = Quantity::<ManonCount>::nonnegative(absolute)?;
        field.replace(converted);
        Ok(ConservationReport {
            before,
            after,
            absolute_error,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::GridShape;

    fn density(n: f64) -> Quantity<ManonDensity> {
        Quantity::nonnegative(n).unwrap()
    }

    fn step() -> TimeStep {
        TimeStep::from_seconds(0.1).unwrap()
    }

    #[test]
    fn closed_3d_diffusion_conserves_total_and_stays_nonnegative() {
        let mut grid =
            FreeManaField::uniform(GridShape::new(4, 3, 2).unwrap(), 1.0, density(0.0)).unwrap();
        grid.set(1, 1, 1, density(100.0)).unwrap();
        let solver = FreeDiffusion::new(Quantity::<Diffusivity>::new(0.5).unwrap()).unwrap();
        for _ in 0..200 {
            let report = solver.evolve(&mut grid, step()).unwrap();
            assert!(report.absolute_error.value() < 1e-8);
        }
        assert!((grid.total_manons().unwrap().value() - 100.0).abs() < 1e-8);
        assert!(grid.at(2, 1, 1).unwrap().value() > 0.0);
    }

    #[test]
    fn unstable_step_does_not_modify_field() {
        let mut grid =
            FreeManaField::uniform(GridShape::new(2, 2, 2).unwrap(), 1.0, density(0.0)).unwrap();
        grid.set(0, 0, 0, density(5.0)).unwrap();
        let initial = grid.at(0, 0, 0).unwrap();
        let solver = FreeDiffusion::new(Quantity::<Diffusivity>::new(2.0).unwrap()).unwrap();
        assert!(solver.evolve(&mut grid, step()).is_err());
        assert_eq!(grid.at(0, 0, 0), Some(initial));
    }

    #[test]
    fn nonnegative_total_is_volume_aware() {
        let grid =
            FreeManaField::uniform(GridShape::new(1, 1, 1).unwrap(), 3.0, density(2.0)).unwrap();
        assert_eq!(grid.total_manons().unwrap().value(), 54.0);
    }
}
