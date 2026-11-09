//! Validated storage exchange for accelerators and persistence.
use super::{MaterialLattice, invalid, nonnegative};
use manonica_error::BaseError;
use manonica_physics::GridShape;

impl MaterialLattice {
    pub fn shape(&self) -> GridShape {
        self.shape
    }
    pub fn cells(&self) -> &[super::TrapCell] {
        &self.cells
    }
    pub fn channels(&self) -> &[super::ManaChannel] {
        &self.channels
    }

    /// Replace bound/thermal storage atomically, preserving all material laws
    /// and geometry. The backend must separately audit global conservation.
    pub fn try_replace_storage(&mut self, storage: &[[f64; 2]]) -> Result<(), BaseError> {
        if storage.len() != self.cells.len() {
            return Err(invalid("material_snapshot_shape"));
        }
        for (cell, [bound, heat]) in self.cells.iter().zip(storage) {
            nonnegative(*bound, "material_snapshot_bound")?;
            nonnegative(*heat, "material_snapshot_heat")?;
            if let Some(m) = cell.material {
                m.validate()?;
                if *bound > m.capacity {
                    return Err(invalid("material_snapshot_saturation"));
                }
            } else if *bound != 0.0 || *heat != 0.0 {
                return Err(invalid("material_snapshot_empty_cell"));
            }
        }
        for (cell, [bound, heat]) in self.cells.iter_mut().zip(storage) {
            cell.bound = *bound;
            cell.heat = *heat;
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_snapshot_never_partially_updates_lattice() {
        let mut lattice = MaterialLattice::new(GridShape::new(2, 1, 1).unwrap());
        let material = super::super::TrapMaterial {
            capacity: 10.0,
            capture_rate: 1.0,
            release_rate: 0.0,
            binding_energy: 2.0,
            heat_capacity: 5.0,
            reference_temperature: 300.0,
            activation_temperature: 100.0,
            max_flux: 4.0,
            conductivity: 1.0,
        };
        lattice.install([0, 0, 0], material).unwrap();
        lattice.install([1, 0, 0], material).unwrap();
        assert!(
            lattice
                .try_replace_storage(&[[1.0, 2.0], [11.0, 3.0]])
                .is_err()
        );
        assert_eq!(lattice.at(0, 0, 0).unwrap().bound, 0.0);
        assert!(
            lattice
                .try_replace_storage(&[[1.0, 2.0], [f64::NAN, 3.0]])
                .is_err()
        );
        assert!(lattice.try_replace_storage(&[[1.0, 2.0]]).is_err());
        lattice
            .try_replace_storage(&[[1.0, 2.0], [3.0, 4.0]])
            .unwrap();
        assert_eq!(lattice.at(1, 0, 0).unwrap().heat, 4.0);
    }
}
