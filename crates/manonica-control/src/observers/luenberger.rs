use manonica_core::{
    GainMatrix, Quantity, TimeStep,
    units::{PerSecond, Unit},
};
use manonica_error::{BaseError, ErrorCode};

use crate::{ControlError, LinearSystem};

/// Результат шага наблюдателя: обновлённая оценка состояния и невязка измерения.
#[derive(Debug, Clone)]
pub struct ObserverReport<X: Unit, Y: Unit> {
    pub estimate: Vec<Quantity<X>>,
    pub innovation: Vec<Quantity<Y>>,
}

/// Наблюдатель Луенбергера для непрерывной системы:
/// x̂' = A x̂ + B u + L (y - C x̂ - D u).
/// На первом этапе явное интегрирование Эйлера.
#[derive(Debug, Clone)]
pub struct LuenbergerObserver<X: Unit, U: Unit, Y: Unit> {
    a: manonica_core::Matrix,
    b: manonica_core::Matrix,
    c: manonica_core::Matrix,
    d: manonica_core::Matrix,
    gain: GainMatrix<Y, PerSecond<X>>,
    estimate: Vec<Quantity<X>>,
    input_unit: std::marker::PhantomData<U>,
}

impl<X: Unit, U: Unit, Y: Unit> LuenbergerObserver<X, U, Y> {
    pub fn from_system(
        system: &LinearSystem<X, U, Y>,
        gain: GainMatrix<Y, PerSecond<X>>,
        initial_estimate: Vec<Quantity<X>>,
    ) -> Result<Self, ControlError> {
        if initial_estimate.len() != system.order()
            || gain.matrix().rows() != system.order()
            || gain.matrix().cols() != system.output_count()
        {
            return Err(BaseError::new(ErrorCode::InvalidShape, "observer").into());
        }
        Ok(Self {
            a: system.a().clone(),
            b: system.b().clone(),
            c: system.c().clone(),
            d: system.d().clone(),
            gain,
            estimate: initial_estimate,
            input_unit: std::marker::PhantomData,
        })
    }

    pub fn estimate(&self) -> &[Quantity<X>] {
        &self.estimate
    }

    pub fn update(
        &mut self,
        input: &[Quantity<U>],
        measurement: &[Quantity<Y>],
        dt: TimeStep,
    ) -> Result<ObserverReport<X, Y>, ControlError> {
        if input.len() != self.b.cols() || measurement.len() != self.c.rows() {
            return Err(BaseError::new(ErrorCode::InvalidShape, "observer_input").into());
        }
        let mut innovation = Vec::with_capacity(measurement.len());
        for (r, measured) in measurement.iter().enumerate() {
            let mut predicted = 0.0;
            for (i, state) in self.estimate.iter().enumerate() {
                predicted += self.c.at(r, i) * state.value();
            }
            for (i, command) in input.iter().enumerate() {
                predicted += self.d.at(r, i) * command.value();
            }
            innovation.push(Quantity::<Y>::new(measured.value() - predicted)?);
        }

        let mut updated = Vec::with_capacity(self.estimate.len());
        for row in 0..self.estimate.len() {
            let mut derivative = 0.0;
            for (i, state) in self.estimate.iter().enumerate() {
                derivative += self.a.at(row, i) * state.value();
            }
            for (i, command) in input.iter().enumerate() {
                derivative += self.b.at(row, i) * command.value();
            }
            for (i, residual) in innovation.iter().enumerate() {
                derivative += self.gain.matrix().at(row, i) * residual.value();
            }
            updated.push(Quantity::<X>::new(
                self.estimate[row].value() + dt.seconds() * derivative,
            )?);
        }

        self.estimate = updated.clone();
        Ok(ObserverReport {
            estimate: updated,
            innovation,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{LinearSystem, LuenbergerObserver};
    use manonica_core::{
        GainMatrix, Matrix, Quantity, TimeStep,
        units::{Dimensionless, PerSecond},
    };

    type U = Dimensionless;

    #[test]
    fn measured_output_corrects_unobserved_initial_state() {
        let system = LinearSystem::new(
            GainMatrix::<U, PerSecond<U>>::new(Matrix::new(1, 1, vec![0.0]).unwrap()),
            GainMatrix::<U, PerSecond<U>>::new(Matrix::new(1, 1, vec![0.0]).unwrap()),
            GainMatrix::<U, U>::new(Matrix::new(1, 1, vec![1.0]).unwrap()),
            GainMatrix::<U, U>::new(Matrix::new(1, 1, vec![0.0]).unwrap()),
            vec![Quantity::<U>::new(1.0).unwrap()],
        )
        .unwrap();
        let gain = GainMatrix::<U, PerSecond<U>>::new(Matrix::new(1, 1, vec![2.0]).unwrap());
        let mut observer =
            LuenbergerObserver::from_system(&system, gain, vec![Quantity::<U>::new(0.0).unwrap()])
                .unwrap();
        let input = [Quantity::<U>::new(0.0).unwrap()];
        let measurement = [Quantity::<U>::new(1.0).unwrap()];
        let result = observer
            .update(&input, &measurement, TimeStep::from_seconds(0.1).unwrap())
            .unwrap();
        assert!((result.estimate[0].value() - 0.2).abs() < 1e-12);
        assert!((result.innovation[0].value() - 1.0).abs() < 1e-12);
        let result = observer
            .update(&input, &measurement, TimeStep::from_seconds(0.1).unwrap())
            .unwrap();
        assert!((result.estimate[0].value() - 0.36).abs() < 1e-12);
    }

    #[test]
    fn malformed_measurement_does_not_change_estimate() {
        let system = LinearSystem::new(
            GainMatrix::<U, PerSecond<U>>::new(Matrix::new(1, 1, vec![-1.0]).unwrap()),
            GainMatrix::<U, PerSecond<U>>::new(Matrix::new(1, 1, vec![0.0]).unwrap()),
            GainMatrix::<U, U>::new(Matrix::new(1, 1, vec![1.0]).unwrap()),
            GainMatrix::<U, U>::new(Matrix::new(1, 1, vec![0.0]).unwrap()),
            vec![Quantity::<U>::new(0.0).unwrap()],
        )
        .unwrap();
        let gain = GainMatrix::<U, PerSecond<U>>::new(Matrix::new(1, 1, vec![1.0]).unwrap());
        let mut observer =
            LuenbergerObserver::from_system(&system, gain, vec![Quantity::<U>::new(0.0).unwrap()])
                .unwrap();
        assert!(
            observer
                .update(
                    &[Quantity::<U>::new(0.0).unwrap()],
                    &[],
                    TimeStep::from_seconds(0.1).unwrap()
                )
                .is_err()
        );
        assert_eq!(observer.estimate()[0].value(), 0.0);
    }
}
