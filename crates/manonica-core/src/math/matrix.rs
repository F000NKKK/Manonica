use manonica_error::{BaseError, ErrorCode};

/// Плотная строко-ориентированная матрица для численного анализа ТАУ.
/// Коэффициенты матриц пока не являются типизированными размерностями.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, BaseError> {
        if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(data.len()) {
            return Err(BaseError::new(ErrorCode::InvalidShape, "matrix"));
        }
        if data.iter().any(|x| !x.is_finite()) {
            return Err(BaseError::new(ErrorCode::NonFinite, "matrix"));
        }
        Ok(Self { rows, cols, data })
    }

    pub fn identity(n: usize) -> Result<Self, BaseError> {
        let count = n
            .checked_mul(n)
            .ok_or(BaseError::new(ErrorCode::InvalidShape, "identity"))?;
        let mut data = vec![0.0; count];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self::new(n, n, data)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn at(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    pub fn multiply(&self, other: &Self) -> Result<Self, BaseError> {
        if self.cols != other.rows {
            return Err(BaseError::new(ErrorCode::InvalidShape, "matrix_multiply"));
        }
        let mut values = vec![0.0; self.rows * other.cols];
        for r in 0..self.rows {
            for k in 0..self.cols {
                for c in 0..other.cols {
                    values[r * other.cols + c] += self.at(r, k) * other.at(k, c);
                }
            }
        }
        Self::new(self.rows, other.cols, values)
    }

    pub fn horizontal(matrices: &[Self]) -> Result<Self, BaseError> {
        let rows = matrices
            .first()
            .ok_or(BaseError::new(ErrorCode::InvalidShape, "horizontal"))?
            .rows;
        if matrices.iter().any(|m| m.rows != rows) {
            return Err(BaseError::new(ErrorCode::InvalidShape, "horizontal"));
        }
        let cols: usize = matrices.iter().map(|m| m.cols).sum();
        let mut values = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for m in matrices {
                for c in 0..m.cols {
                    values.push(m.at(r, c));
                }
            }
        }
        Self::new(rows, cols, values)
    }

    pub fn vertical(matrices: &[Self]) -> Result<Self, BaseError> {
        let cols = matrices
            .first()
            .ok_or(BaseError::new(ErrorCode::InvalidShape, "vertical"))?
            .cols;
        if matrices.iter().any(|m| m.cols != cols) {
            return Err(BaseError::new(ErrorCode::InvalidShape, "vertical"));
        }
        let values = matrices
            .iter()
            .flat_map(|m| m.data.iter().copied())
            .collect();
        Self::new(matrices.iter().map(|m| m.rows).sum(), cols, values)
    }

    /// Ранг с частичным выбором главного элемента и заданным порогом.
    pub fn rank(&self, tolerance: f64) -> Result<usize, BaseError> {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(BaseError::new(ErrorCode::InvalidRange, "rank_tolerance"));
        }
        let mut a = self.data.clone();
        let mut pivot_row = 0;
        for column in 0..self.cols {
            if pivot_row == self.rows {
                break;
            }
            let best = (pivot_row..self.rows)
                .max_by(|&a_row, &b_row| {
                    a[a_row * self.cols + column]
                        .abs()
                        .total_cmp(&a[b_row * self.cols + column].abs())
                })
                .unwrap_or(pivot_row);
            if a[best * self.cols + column].abs() <= tolerance {
                continue;
            }
            for c in 0..self.cols {
                a.swap(pivot_row * self.cols + c, best * self.cols + c);
            }
            let pivot = a[pivot_row * self.cols + column];
            for r in (pivot_row + 1)..self.rows {
                let factor = a[r * self.cols + column] / pivot;
                for c in column..self.cols {
                    a[r * self.cols + c] -= factor * a[pivot_row * self.cols + c];
                }
            }
            pivot_row += 1;
        }
        Ok(pivot_row)
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;

    #[test]
    fn rank_identifies_singular_and_nonsingular() {
        let full = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let singular = Matrix::new(2, 2, vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        assert_eq!(full.rank(1e-10).unwrap(), 2);
        assert_eq!(singular.rank(1e-10).unwrap(), 1);
    }
}
