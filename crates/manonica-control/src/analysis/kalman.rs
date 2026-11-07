use manonica_core::{Matrix, units::Unit};

use crate::{ControlError, systems::LinearSystem};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnalysisReport {
    pub rank: usize,
    pub order: usize,
    pub full_rank: bool,
}

/// Критерий Калмана: rank[B AB ... A^(n-1) B] = n.
pub fn controllability<X: Unit, U: Unit, Y: Unit>(
    system: &LinearSystem<X, U, Y>,
    tolerance: f64,
) -> Result<AnalysisReport, ControlError> {
    let n = system.order();
    let mut power_times_b = system.b().clone();
    let mut terms = Vec::with_capacity(n);
    for _ in 0..n {
        terms.push(power_times_b.clone());
        power_times_b = system.a().multiply(&power_times_b)?;
    }
    report(Matrix::horizontal(&terms)?, n, tolerance)
}

/// Критерий Калмана: rank[C; CA; ...; CA^(n-1)] = n.
pub fn observability<X: Unit, U: Unit, Y: Unit>(
    system: &LinearSystem<X, U, Y>,
    tolerance: f64,
) -> Result<AnalysisReport, ControlError> {
    let n = system.order();
    let mut c_times_power = system.c().clone();
    let mut terms = Vec::with_capacity(n);
    for _ in 0..n {
        terms.push(c_times_power.clone());
        c_times_power = c_times_power.multiply(system.a())?;
    }
    report(Matrix::vertical(&terms)?, n, tolerance)
}

fn report(matrix: Matrix, order: usize, tolerance: f64) -> Result<AnalysisReport, ControlError> {
    let rank = matrix.rank(tolerance)?;
    Ok(AnalysisReport {
        rank,
        order,
        full_rank: rank == order,
    })
}

#[cfg(test)]
mod tests {
    use manonica_core::{
        GainMatrix, Matrix, Quantity,
        units::{Joules, Kelvin, Newtons, PerSecond},
    };

    use crate::{
        analysis::{controllability, observability},
        systems::LinearSystem,
    };

    fn system(b: Vec<f64>, c: Vec<f64>) -> LinearSystem<Joules, Newtons, Kelvin> {
        LinearSystem::new(
            GainMatrix::new(Matrix::new(2, 2, vec![0.0, 1.0, 0.0, 0.0]).unwrap()),
            GainMatrix::new(Matrix::new(2, 1, b).unwrap()),
            GainMatrix::new(Matrix::new(1, 2, c).unwrap()),
            GainMatrix::new(Matrix::new(1, 1, vec![0.0]).unwrap()),
            vec![Quantity::<Joules>::new(0.0).unwrap(); 2],
        )
        .unwrap()
    }

    #[test]
    fn canonical_double_integrator_is_controllable_and_observable() {
        let model = system(vec![0.0, 1.0], vec![1.0, 0.0]);
        assert!(controllability(&model, 1e-10).unwrap().full_rank);
        assert!(observability(&model, 1e-10).unwrap().full_rank);
    }

    #[test]
    fn unreachable_and_unobservable_modes_are_detected() {
        let model = system(vec![1.0, 0.0], vec![0.0, 1.0]);
        assert!(!controllability(&model, 1e-10).unwrap().full_rank);
        assert!(!observability(&model, 1e-10).unwrap().full_rank);
    }
}
