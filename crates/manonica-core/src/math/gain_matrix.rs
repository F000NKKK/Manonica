use std::marker::PhantomData;

use crate::units::Unit;
use super::Matrix;

/// Матрица линейного преобразования типизированного входного сигнала в выходной.
/// Значения коэффициентов являются численными; единицы входа/выхода заданы типами.
#[derive(Debug, Clone)]
pub struct GainMatrix<In: Unit, Out: Unit> {
    matrix: Matrix,
    units: PhantomData<(In, Out)>,
}

impl<In: Unit, Out: Unit> GainMatrix<In, Out> {
    pub fn new(matrix: Matrix) -> Self {
        Self { matrix, units: PhantomData }
    }

    pub fn matrix(&self) -> &Matrix {
        &self.matrix
    }
}
