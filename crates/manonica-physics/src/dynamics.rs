use manonica_core::{TimeStep, Vec3, units::Metres};

use crate::{Manon, PhysicsError};

/// Универсальный контракт микроскопического закона движения.
/// Физика манонов не зависит от конкретного интегратора.
pub trait ParticleLaw: Send + Sync {
    fn propagate(&self, manon: Manon, dt: TimeStep) -> Result<Manon, PhysicsError>;
}

/// Эталонный закон свободного движения: r(t+dt) = r(t) + v·dt.
/// Пока манон не взаимодействует с полями, средой и другими частицами.
pub struct FreeMotion;

impl FreeMotion {
    pub fn propagate(manon: Manon, dt: TimeStep) -> Result<Manon, PhysicsError> {
        <Self as ParticleLaw>::propagate(&Self, manon, dt)
    }
}

impl ParticleLaw for FreeMotion {
    fn propagate(&self, manon: Manon, dt: TimeStep) -> Result<Manon, PhysicsError> {
        let p = manon.position();
        let v = manon.velocity();
        let seconds = dt.seconds();
        let position: Vec3<Metres> = Vec3::try_values(
            p.x.value() + v.x.value() * seconds,
            p.y.value() + v.y.value() * seconds,
            p.z.value() + v.z.value() * seconds,
        )?;
        let mut next = manon;
        next.set_position(position);
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use crate::{FreeMotion, Manon, ManonId};
    use manonica_core::{
        Quantity, TimeStep, Vec3,
        units::{Joules, Metres, MetresPerSecond},
    };

    #[test]
    fn free_motion_keeps_energy_and_velocity() {
        let initial = Manon::free(
            ManonId(1),
            Vec3::<Metres>::try_values(1.0, 2.0, 3.0).unwrap(),
            Vec3::<MetresPerSecond>::try_values(2.0, -1.0, 0.5).unwrap(),
            Quantity::<Joules>::nonnegative(10.0).unwrap(),
        )
        .unwrap();
        let result = FreeMotion::propagate(initial, TimeStep::from_seconds(2.0).unwrap()).unwrap();
        assert_eq!(result.position().x.value(), 5.0);
        assert_eq!(result.position().y.value(), 0.0);
        assert_eq!(result.position().z.value(), 4.0);
        assert_eq!(result.energy(), initial.energy());
        assert_eq!(result.velocity(), initial.velocity());
    }
}
