//! Conservative 3D transport of free manons by a prescribed medium velocity.
//!
//! This law knows neither magic circles nor materials. A caller supplies a
//! velocity field; internal face fluxes have equal and opposite contributions.
//! All exterior faces are impermeable (zero normal flux).

use manonica_core::{
    Evolution, Quantity, TimeStep, Vec3,
    units::{ManonCount, ManonDensity, Metres, MetresPerSecond},
};
use manonica_error::{BaseError, ErrorCode};

use crate::{PhysicsError, field::{ConservationReport, FreeManaField}};

/// Velocity of the surrounding medium at an arbitrary point in metres.
pub trait VelocityField: Send + Sync {
    fn at(&self, position: Vec3<Metres>) -> Result<Vec3<MetresPerSecond>, PhysicsError>;
}

/// The physically separate advection law J = n * v_medium.
#[derive(Debug, Clone)]
pub struct FreeAdvection<V> {
    velocity: V,
}

impl<V: VelocityField> FreeAdvection<V> {
    pub fn new(velocity: V) -> Self {
        Self { velocity }
    }

    pub fn velocity(&self) -> &V {
        &self.velocity
    }
}

impl<V: VelocityField> Evolution<FreeManaField> for FreeAdvection<V> {
    type Error = PhysicsError;
    type Report = ConservationReport;

    fn evolve(&self, field: &mut FreeManaField, dt: TimeStep) -> Result<Self::Report, Self::Error> {
        let shape = field.shape();
        let dx = field.cell_width_metres();
        let ratio = dt.seconds() / dx;
        let before = field.total_manons()?;
        let old: Vec<f64> = field.values().iter().map(|n| n.value()).collect();
        let mut next = old.clone();
        let mut outgoing = vec![0.0_f64; old.len()];

        for z in 0..shape.nz() {
            for y in 0..shape.ny() {
                for x in 0..shape.nx() {
                    let i = shape.index(x, y, z).expect("valid cell");
                    for axis in 0..3 {
                        let (xx, yy, zz) = match axis {
                            0 => (x + 1, y, z),
                            1 => (x, y + 1, z),
                            _ => (x, y, z + 1),
                        };
                        let Some(j) = shape.index(xx, yy, zz) else {
                            // The boundary is a wall, not a free manon sink.
                            continue;
                        };
                        let mut face = [
                            (x as f64 + 0.5 - shape.nx() as f64 * 0.5) * dx,
                            (y as f64 + 0.5 - shape.ny() as f64 * 0.5) * dx,
                            (z as f64 + 0.5 - shape.nz() as f64 * 0.5) * dx,
                        ];
                        face[axis] += 0.5 * dx;
                        let velocity = self.velocity.at(Vec3::try_values(
                            face[0], face[1], face[2],
                        )?)?;
                        let speed = match axis {
                            0 => velocity.x.value(),
                            1 => velocity.y.value(),
                            _ => velocity.z.value(),
                        };
                        let signed_fraction = ratio * speed;
                        if !signed_fraction.is_finite() {
                            return Err(BaseError::new(ErrorCode::NonFinite, "advection_speed").into());
                        }
                        let (donor, amount) = if signed_fraction >= 0.0 {
                            (i, signed_fraction * old[i])
                        } else {
                            (j, signed_fraction * old[j])
                        };
                        outgoing[donor] += signed_fraction.abs();
                        next[i] -= amount;
                        next[j] += amount;
                    }
                }
            }
        }

        // In an explicit upwind scheme each donor must retain at least 0%
        // of its concentration. Check the sum over all outgoing faces.
        if outgoing.iter().any(|fraction| !fraction.is_finite() || *fraction > 1.0) {
            return Err(BaseError::new(ErrorCode::NumericalInstability, "advection_cfl").into());
        }

        let mut converted = Vec::with_capacity(next.len());
        for density in next {
            if !density.is_finite() || density < -1e-12 {
                return Err(BaseError::new(ErrorCode::NumericalInstability, "advection_density").into());
            }
            converted.push(Quantity::<ManonDensity>::nonnegative(density.max(0.0))?);
        }

        let cell_volume = dx.powi(3);
        let after_value = converted.iter().map(|n| n.value()).sum::<f64>() * cell_volume;
        let absolute_error = (after_value - before.value()).abs();
        if !after_value.is_finite() || absolute_error > 1e-10 * before.value().max(1.0) {
            return Err(BaseError::new(ErrorCode::NumericalInstability, "advection_conservation").into());
        }

        let after = Quantity::<ManonCount>::nonnegative(after_value)?;
        let absolute_error = Quantity::<ManonCount>::nonnegative(absolute_error)?;
        field.replace(converted);
        Ok(ConservationReport { before, after, absolute_error })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GridShape;

    #[derive(Clone, Copy)]
    struct Constant([f64; 3]);

    impl VelocityField for Constant {
        fn at(&self, _: Vec3<Metres>) -> Result<Vec3<MetresPerSecond>, PhysicsError> {
            Ok(Vec3::try_values(self.0[0], self.0[1], self.0[2])?)
        }
    }

    fn test_field() -> FreeManaField {
        let mut f = FreeManaField::uniform(
            GridShape::new(3, 1, 1).unwrap(),
            1.0,
            Quantity::<ManonDensity>::nonnegative(0.0).unwrap(),
        ).unwrap();
        f.set(0, 0, 0, Quantity::nonnegative(100.0).unwrap()).unwrap();
        f
    }

    #[test]
    fn positive_x_flux_moves_mass_to_neighbour_without_leaking() {
        let mut field = test_field();
        let law = FreeAdvection::new(Constant([1.0, 0.0, 0.0]));
        let result = law.evolve(&mut field, TimeStep::from_seconds(0.25).unwrap()).unwrap();
        assert_eq!(field.at(0, 0, 0).unwrap().value(), 75.0);
        assert_eq!(field.at(1, 0, 0).unwrap().value(), 25.0);
        assert_eq!(field.total_manons().unwrap().value(), 100.0);
        assert_eq!(result.absolute_error.value(), 0.0);
    }

    #[test]
    fn negative_velocity_uses_the_other_face_donor() {
        let mut field = test_field();
        field.set(0, 0, 0, Quantity::nonnegative(0.0).unwrap()).unwrap();
        field.set(2, 0, 0, Quantity::nonnegative(40.0).unwrap()).unwrap();
        FreeAdvection::new(Constant([-1.0, 0.0, 0.0]))
            .evolve(&mut field, TimeStep::from_seconds(0.5).unwrap()).unwrap();
        assert_eq!(field.at(2, 0, 0).unwrap().value(), 20.0);
        assert_eq!(field.at(1, 0, 0).unwrap().value(), 20.0);
    }

    #[test]
    fn excessive_outgoing_fraction_rejects_without_modifying_field() {
        let mut field = test_field();
        let initial = field.at(0, 0, 0);
        let law = FreeAdvection::new(Constant([5.0, 0.0, 0.0]));
        assert!(law.evolve(&mut field, TimeStep::from_seconds(0.5).unwrap()).is_err());
        assert_eq!(field.at(0, 0, 0), initial);
    }

    #[test]
    fn closed_boundaries_do_not_leak_under_constant_transport() {
        let mut field = test_field();
        let law = FreeAdvection::new(Constant([0.75, 0.0, 0.0]));
        for _ in 0..400 {
            law.evolve(&mut field, TimeStep::from_seconds(0.1).unwrap()).unwrap();
        }
        assert!((field.total_manons().unwrap().value() - 100.0).abs() < 1e-8);
        assert!(field.at(2, 0, 0).unwrap().value() > 99.999);
    }
}
