use manonica_core::{
    GainMatrix, Quantity, TimeStep,
    units::{PerSecond, Unit},
};
use manonica_error::{BaseError, ErrorCode};

use crate::{ControlError, systems::DynamicSystem};

/// Линейная непрерывная система пространства состояний:
/// dx/dt = A*x + B*u, y = C*x + D*u.
///
/// Размерности коэффициентов зафиксированы на уровне типов GainMatrix.
/// Численное интегрирование сейчас явное Эйлерово (начальный эталон).
#[derive(Debug, Clone)]
pub struct LinearSystem<X: Unit, U: Unit, Y: Unit> {
    a: GainMatrix<X, PerSecond<X>>,
    b: GainMatrix<U, PerSecond<X>>,
    c: GainMatrix<X, Y>,
    d: GainMatrix<U, Y>,
    state: Vec<Quantity<X>>,
}

impl<X: Unit, U: Unit, Y: Unit> LinearSystem<X, U, Y> {
    pub fn new(
        a: GainMatrix<X, PerSecond<X>>,
        b: GainMatrix<U, PerSecond<X>>,
        c: GainMatrix<X, Y>,
        d: GainMatrix<U, Y>,
        state: Vec<Quantity<X>>,
    ) -> Result<Self, ControlError> {
        let n = state.len();
        let m = b.matrix().cols();
        let p = c.matrix().rows();
        if n == 0
            || a.matrix().rows() != n
            || a.matrix().cols() != n
            || b.matrix().rows() != n
            || c.matrix().cols() != n
            || d.matrix().rows() != p
            || d.matrix().cols() != m
        {
            return Err(BaseError::new(ErrorCode::InvalidShape, "state_space").into());
        }
        Ok(Self { a, b, c, d, state })
    }

    pub fn order(&self) -> usize {
        self.state.len()
    }
    pub fn input_count(&self) -> usize {
        self.b.matrix().cols()
    }
    pub fn output_count(&self) -> usize {
        self.c.matrix().rows()
    }
    pub fn state(&self) -> &[Quantity<X>] {
        &self.state
    }

    pub fn a(&self) -> &manonica_core::Matrix {
        self.a.matrix()
    }
    pub fn b(&self) -> &manonica_core::Matrix {
        self.b.matrix()
    }
    pub fn c(&self) -> &manonica_core::Matrix {
        self.c.matrix()
    }
    pub fn d(&self) -> &manonica_core::Matrix {
        self.d.matrix()
    }
}

impl<X: Unit, U: Unit, Y: Unit> DynamicSystem<U, Y> for LinearSystem<X, U, Y> {
    fn step(
        &mut self,
        input: &[Quantity<U>],
        dt: TimeStep,
    ) -> Result<Vec<Quantity<Y>>, ControlError> {
        if input.len() != self.input_count() {
            return Err(BaseError::new(ErrorCode::InvalidShape, "state_input").into());
        }
        // y(t) вычисляется до следующего шага x(t+dt).
        let mut output = Vec::with_capacity(self.output_count());
        for row in 0..self.output_count() {
            let mut value = 0.0;
            for (i, x) in self.state.iter().enumerate() {
                value += self.c().at(row, i) * x.value();
            }
            for (i, u) in input.iter().enumerate() {
                value += self.d().at(row, i) * u.value();
            }
            output.push(Quantity::new(value)?);
        }

        let mut updated = Vec::with_capacity(self.order());
        for row in 0..self.order() {
            let mut derivative = 0.0;
            for (i, x) in self.state.iter().enumerate() {
                derivative += self.a().at(row, i) * x.value();
            }
            for (i, u) in input.iter().enumerate() {
                derivative += self.b().at(row, i) * u.value();
            }
            updated.push(Quantity::new(
                self.state[row].value() + dt.seconds() * derivative,
            )?);
        }
        self.state = updated;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use manonica_core::{
        GainMatrix, Matrix, Quantity, TimeStep,
        units::{Joules, Kelvin, Newtons, PerSecond},
    };

    use super::{DynamicSystem, LinearSystem};

    #[test]
    fn simple_first_order_model() {
        let mut system = LinearSystem::new(
            GainMatrix::<Joules, PerSecond<Joules>>::new(Matrix::new(1, 1, vec![-1.0]).unwrap()),
            GainMatrix::<Newtons, PerSecond<Joules>>::new(Matrix::new(1, 1, vec![1.0]).unwrap()),
            GainMatrix::<Joules, Kelvin>::new(Matrix::new(1, 1, vec![1.0]).unwrap()),
            GainMatrix::<Newtons, Kelvin>::new(Matrix::new(1, 1, vec![0.0]).unwrap()),
            vec![Quantity::<Joules>::new(0.0).unwrap()],
        )
        .unwrap();
        let input = [Quantity::<Newtons>::new(1.0).unwrap()];
        let output = system
            .step(&input, TimeStep::from_seconds(0.1).unwrap())
            .unwrap();
        assert_eq!(output[0].value(), 0.0);
        assert!((system.state()[0].value() - 0.1).abs() < 1e-12);
    }
}
