use manonica_core::{
    Quantity,
    units::{ManonCount, ManonDensity},
};
use manonica_error::{BaseError, ErrorCode};

/// Размерности регулярной расчётной области. Не является Minecraft chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridShape {
    nx: usize,
    ny: usize,
    nz: usize,
}

impl GridShape {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Result<Self, BaseError> {
        if nx == 0
            || ny == 0
            || nz == 0
            || nx.checked_mul(ny).and_then(|n| n.checked_mul(nz)).is_none()
        {
            return Err(BaseError::new(ErrorCode::InvalidShape, "grid_shape"));
        }
        Ok(Self { nx, ny, nz })
    }

    pub fn nx(self) -> usize {
        self.nx
    }
    pub fn ny(self) -> usize {
        self.ny
    }
    pub fn nz(self) -> usize {
        self.nz
    }
    pub fn is_empty(self) -> bool {
        false // Создание пустой сетки запрещено GridShape::new.
    }

    pub fn len(self) -> usize {
        self.nx * self.ny * self.nz
    }

    pub fn index(self, x: usize, y: usize, z: usize) -> Option<usize> {
        if x < self.nx && y < self.ny && z < self.nz {
            Some((z * self.ny + y) * self.nx + x)
        } else {
            None
        }
    }
}

/// Концентрация СВОБОДНЫХ манонов на регулярной 3D-сетке.
/// Связанные состояния, вещества и кристаллы здесь принципиально отсутствуют.
#[derive(Debug, Clone)]
pub struct FreeManaField {
    shape: GridShape,
    cell_width: f64,
    concentrations: Vec<Quantity<ManonDensity>>,
}

impl FreeManaField {
    pub fn uniform(
        shape: GridShape,
        cell_width_metres: f64,
        density: Quantity<ManonDensity>,
    ) -> Result<Self, BaseError> {
        if !cell_width_metres.is_finite()
            || cell_width_metres <= 0.0
            || !cell_width_metres.powi(3).is_finite()
        {
            return Err(BaseError::new(ErrorCode::InvalidRange, "cell_width"));
        }
        if density.value() < 0.0 {
            return Err(BaseError::new(ErrorCode::InvalidRange, "mana_density"));
        }
        Ok(Self {
            shape,
            cell_width: cell_width_metres,
            concentrations: vec![density; shape.len()],
        })
    }

    pub fn shape(&self) -> GridShape {
        self.shape
    }
    pub fn cell_width_metres(&self) -> f64 {
        self.cell_width
    }

    pub fn at(&self, x: usize, y: usize, z: usize) -> Option<Quantity<ManonDensity>> {
        self.shape
            .index(x, y, z)
            .map(|index| self.concentrations[index])
    }

    pub fn set(
        &mut self,
        x: usize,
        y: usize,
        z: usize,
        density: Quantity<ManonDensity>,
    ) -> Result<(), BaseError> {
        if density.value() < 0.0 {
            return Err(BaseError::new(ErrorCode::InvalidRange, "mana_density"));
        }
        let index = self
            .shape
            .index(x, y, z)
            .ok_or(BaseError::new(ErrorCode::InvalidShape, "cell_coordinate"))?;
        self.concentrations[index] = density;
        Ok(())
    }

    pub fn total_manons(&self) -> Result<Quantity<ManonCount>, BaseError> {
        let cell_volume = self.cell_width.powi(3);
        let density_sum: f64 = self.concentrations.iter().map(|v| v.value()).sum();
        Quantity::nonnegative(density_sum * cell_volume)
    }

    /// Immutable, X-major density grid (x varies fastest, then y, then z).
    /// Borrowing the slice avoids one indexed method call per interpolation
    /// sample while retaining validated, typed physical quantities.
    pub fn densities(&self) -> &[Quantity<ManonDensity>] {
        &self.concentrations
    }

    /// Validated direct index update for conservative coupled operators.
    pub(crate) fn replace_density_at(&mut self, index: usize, density: Quantity<ManonDensity>) {
        debug_assert!(index < self.concentrations.len());
        self.concentrations[index] = density;
    }

    /// Public validated replacement for external compute backends.
    /// Rejects mismatched grids or non-finite/negative densities without
    /// modifying the original state. X varies fastest, then Y, then Z.
    pub fn try_replace_densities(
        &mut self,
        next: Vec<Quantity<ManonDensity>>,
    ) -> Result<(), BaseError> {
        if next.len() != self.shape.len() {
            return Err(BaseError::new(ErrorCode::InvalidShape, "density_grid_length"));
        }
        if next.iter().any(|value| !value.value().is_finite() || value.value() < 0.0) {
            return Err(BaseError::new(ErrorCode::InvalidRange, "density_grid_values"));
        }
        self.concentrations = next;
        Ok(())
    }

    pub(crate) fn replace(&mut self, next: Vec<Quantity<ManonDensity>>) {
        debug_assert_eq!(next.len(), self.shape.len());
        self.concentrations = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_total_uses_volume_not_density_sum() {
        let shape = GridShape::new(2, 2, 1).unwrap();
        let density = Quantity::<ManonDensity>::nonnegative(3.0).unwrap();
        let grid = FreeManaField::uniform(shape, 2.0, density).unwrap();
        assert_eq!(grid.total_manons().unwrap().value(), 96.0);
        assert!(grid.at(2, 0, 0).is_none());
    }

    #[test]
    fn validated_external_backend_replacement_is_atomic() {
        let shape = GridShape::new(2, 1, 1).unwrap();
        let mut grid = FreeManaField::uniform(
            shape, 1.0, Quantity::<ManonDensity>::nonnegative(3.0).unwrap()
        ).unwrap();
        assert!(grid.try_replace_densities(vec![]).is_err());
        assert_eq!(grid.at(0,0,0).unwrap().value(),3.0);
        assert!(grid.try_replace_densities(vec![
            Quantity::<ManonDensity>::new(f64::INFINITY).unwrap(),
            Quantity::<ManonDensity>::nonnegative(1.0).unwrap()
        ]).is_err());
        assert_eq!(grid.at(1,0,0).unwrap().value(),3.0);
        grid.try_replace_densities(vec![
            Quantity::<ManonDensity>::nonnegative(2.0).unwrap(),
            Quantity::<ManonDensity>::nonnegative(4.0).unwrap()
        ]).unwrap();
        assert_eq!(grid.total_manons().unwrap().value(),6.0);
    }

    #[test]
    fn rejects_negative_density_and_zero_size() {
        assert!(GridShape::new(0, 1, 1).is_err());
        let shape = GridShape::new(1, 1, 1).unwrap();
        assert!(
            FreeManaField::uniform(shape, 1.0, Quantity::<ManonDensity>::new(-1.0).unwrap())
                .is_err()
        );
    }
}
