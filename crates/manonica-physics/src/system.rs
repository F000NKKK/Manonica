use manonica_core::TimeStep;

use crate::{FreeMotion, Manon, ParticleLaw, PhysicsError};

/// Микроскопическая лаборатория: сам контейнер не знает конкретного закона.
/// Шаг транзакционный — либо обновятся все маноны, либо ни один.
#[derive(Debug, Default, Clone)]
pub struct ManonSystem {
    manons: Vec<Manon>,
}

impl ManonSystem {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, manon: Manon) -> Result<(), PhysicsError> {
        if self.manons.iter().any(|m| m.id() == manon.id()) {
            return Err(PhysicsError::DuplicateManonId(manon.id().0));
        }
        self.manons.push(manon);
        Ok(())
    }

    pub fn manons(&self) -> &[Manon] {
        &self.manons
    }

    pub fn advance(&mut self, dt: TimeStep) -> Result<(), PhysicsError> {
        self.advance_with(&FreeMotion, dt)
    }

    pub fn advance_with<L: ParticleLaw>(
        &mut self,
        law: &L,
        dt: TimeStep,
    ) -> Result<(), PhysicsError> {
        let next = self.manons.iter().copied()
            .map(|manon| law.propagate(manon, dt))
            .collect::<Result<Vec<_>, _>>()?;
        self.manons = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{Manon, ManonId, ManonSystem, ParticleLaw, PhysicsError};
    use manonica_core::{
        Quantity, TimeStep, Vec3,
        units::{Joules, Metres, MetresPerSecond},
    };

    fn particle(id: u64) -> Manon {
        Manon::free(
            ManonId(id),
            Vec3::<Metres>::try_values(0.0, 0.0, 0.0).unwrap(),
            Vec3::<MetresPerSecond>::try_values(1.0, 0.0, 0.0).unwrap(),
            Quantity::<Joules>::nonnegative(1.0).unwrap(),
        ).unwrap()
    }

    #[test]
    fn ids_are_unique() {
        let mut system = ManonSystem::new();
        system.insert(particle(7)).unwrap();
        assert!(system.insert(particle(7)).is_err());
    }

    struct FailOnSecond;
    impl ParticleLaw for FailOnSecond {
        fn propagate(&self, manon: Manon, _: TimeStep) -> Result<Manon, PhysicsError> {
            if manon.id().0 == 2 {
                Err(PhysicsError::DuplicateManonId(2))
            } else {
                Ok(manon)
            }
        }
    }

    #[test]
    fn external_law_failure_does_not_mutate_particle_system() {
        let mut system = ManonSystem::new();
        system.insert(particle(1)).unwrap();
        system.insert(particle(2)).unwrap();
        let original = system.manons().to_vec();
        assert!(system.advance_with(&FailOnSecond, TimeStep::from_seconds(0.1).unwrap()).is_err());
        assert_eq!(original, system.manons());
    }
}
