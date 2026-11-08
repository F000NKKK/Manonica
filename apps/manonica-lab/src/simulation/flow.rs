//! Map a declarative medium velocity to the physics-level VelocityField
//! contract. No reference to circles, materials or rendering.

use manonica_core::{
    Vec3,
    units::{Metres, MetresPerSecond},
};
use manonica_physics::{PhysicsError, VelocityField};

use crate::experiment::AmbientFlow;

impl VelocityField for AmbientFlow {
    fn at(&self, position: Vec3<Metres>) -> Result<Vec3<MetresPerSecond>, PhysicsError> {
        match self {
            Self::Still => Ok(Vec3::try_values(0.0, 0.0, 0.0)?),
            Self::Uniform { velocity } => {
                Ok(Vec3::try_values(velocity[0], velocity[1], velocity[2])?)
            }
            Self::Vortex {
                origin,
                axis,
                angular_speed,
                axial_speed,
            } => {
                let length = (axis.iter().map(|v| v * v).sum::<f64>()).sqrt();
                let a = [axis[0] / length, axis[1] / length, axis[2] / length];
                let r = [
                    position.x.value() - origin[0],
                    position.y.value() - origin[1],
                    position.z.value() - origin[2],
                ];
                let cross = [
                    a[1] * r[2] - a[2] * r[1],
                    a[2] * r[0] - a[0] * r[2],
                    a[0] * r[1] - a[1] * r[0],
                ];
                Ok(Vec3::try_values(
                    angular_speed * cross[0] + axial_speed * a[0],
                    angular_speed * cross[1] + axial_speed * a[1],
                    angular_speed * cross[2] + axial_speed * a[2],
                )?)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertical_vortex_has_tangential_speed_and_updraft() {
        let f = AmbientFlow::Vortex {
            origin: [0.0, 0.0, 0.0],
            axis: [0.0, 1.0, 0.0],
            angular_speed: 2.0,
            axial_speed: 0.5,
        };
        let v = f.at(Vec3::try_values(2.0, 0.0, 0.0).unwrap()).unwrap();
        assert_eq!([v.x.value(), v.y.value(), v.z.value()], [0.0, 0.5, -4.0]);
    }
}
