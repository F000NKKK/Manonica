//! Map a declarative medium velocity to the physics-level VelocityField
//! contract. No reference to circles, materials or rendering.

use manonica_core::{
    Vec3,
    units::{Metres, MetresPerSecond},
};
use manonica_physics::{PhysicsError, VelocityField};

use crate::experiment::AmbientFlow;

impl AmbientFlow {
    pub(crate) fn physical(&self) -> manonica_simulation::MediumFlow {
        match self {
            Self::Still => manonica_simulation::MediumFlow::Still,
            Self::Uniform { velocity } => manonica_simulation::MediumFlow::Uniform {
                velocity: *velocity,
            },
            Self::Vortex {
                origin,
                axis,
                angular_speed,
                axial_speed,
            } => manonica_simulation::MediumFlow::Vortex {
                origin: *origin,
                axis: *axis,
                angular_speed: *angular_speed,
                axial_speed: *axial_speed,
            },
        }
    }
}
impl VelocityField for AmbientFlow {
    fn at(&self, position: Vec3<Metres>) -> Result<Vec3<MetresPerSecond>, PhysicsError> {
        self.physical().at(position)
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
