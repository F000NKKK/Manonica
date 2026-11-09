//! Prescribed medium velocity, sampled at finite-volume faces.
use manonica_core::{
    Vec3,
    units::{Metres, MetresPerSecond},
};
use manonica_physics::{PhysicsError, VelocityField};

#[derive(Clone, Debug, Default)]
pub enum MediumFlow {
    #[default]
    Still,
    Uniform {
        velocity: [f64; 3],
    },
    Vortex {
        origin: [f64; 3],
        axis: [f64; 3],
        angular_speed: f64,
        axial_speed: f64,
    },
}
impl VelocityField for MediumFlow {
    fn at(&self, position: Vec3<Metres>) -> Result<Vec3<MetresPerSecond>, PhysicsError> {
        let v = match self {
            Self::Still => [0.0; 3],
            Self::Uniform { velocity } => *velocity,
            Self::Vortex {
                origin,
                axis,
                angular_speed,
                axial_speed,
            } => {
                let length = axis.iter().map(|v| v * v).sum::<f64>().sqrt();
                let a = axis.map(|v| v / length);
                let r = [
                    position.x.value() - origin[0],
                    position.y.value() - origin[1],
                    position.z.value() - origin[2],
                ];
                [
                    angular_speed * (a[1] * r[2] - a[2] * r[1]) + axial_speed * a[0],
                    angular_speed * (a[2] * r[0] - a[0] * r[2]) + axial_speed * a[1],
                    angular_speed * (a[0] * r[1] - a[1] * r[0]) + axial_speed * a[2],
                ]
            }
        };
        Ok(Vec3::try_values(v[0], v[1], v[2])?)
    }
}
