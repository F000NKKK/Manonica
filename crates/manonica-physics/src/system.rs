use manonica_core::TimeStep;

use crate::{FreeMotion, Manon, PhysicsError};

/// Первая микроскопическая лаборатория без взаимодействий.
/// Обновление транзакционное: не применяется частичный шаг при ошибке.
#[derive(Debug, Default, Clone)]
pub struct ManonSystem {
    manons: Vec<Manon>,
}

impl ManonSystem {
    pub fn new() -> Self { Self::default() }

    pub fn insert(&mut self, manon: Manon) -> Result<(), PhysicsError> {
        if self.manons.iter().any(|m| m.id() == manon.id()) {
            return Err(PhysicsError::DuplicateManonId(manon.id().0));
        }
        self.manons.push(manon);
        Ok(())
    }

    pub fn manons(&self) -> &[Manon] { &self.manons }

    pub fn advance(&mut self, dt: TimeStep) -> Result<(), PhysicsError> {
        let updated = self.manons.iter().copied()
            .map(|manon| FreeMotion::propagate(manon, dt))
            .collect::<Result<Vec<_>, _>>()?;
        self.manons = updated;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use manonica_core::{Quantity, Vec3, units::{Joules, Metres, MetresPerSecond}};
    use crate::{Manon, ManonId, ManonSystem};

    #[test]
    fn ids_are_unique() {
        let mut system = ManonSystem::new();
        let manon = Manon::free(
            ManonId(7),
            Vec3::<Metres>::try_values(0.0, 0.0, 0.0).unwrap(),
            Vec3::<MetresPerSecond>::try_values(0.0, 0.0, 0.0).unwrap(),
            Quantity::<Joules>::nonnegative(1.0).unwrap(),
        ).unwrap();
        system.insert(manon).unwrap();
        assert!(system.insert(manon).is_err());
    }
}
