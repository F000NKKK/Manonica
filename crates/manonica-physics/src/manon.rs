use manonica_core::{Quantity, Vec3, units::{Joules, Metres, MetresPerSecond}};

/// Постоянный идентификатор частицы.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ManonId(pub u64);

/// На первой стадии моделируем только свободные маноны.
/// Связывание и преобразование требуют отдельной физической модели.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManonState {
    Free,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Manon {
    id: ManonId,
    position: Vec3<Metres>,
    velocity: Vec3<MetresPerSecond>,
    energy: Quantity<Joules>,
    state: ManonState,
}

impl Manon {
    pub fn free(
        id: ManonId,
        position: Vec3<Metres>,
        velocity: Vec3<MetresPerSecond>,
        energy: Quantity<Joules>,
    ) -> Result<Self, manonica_error::BaseError> {
        if energy.value() < 0.0 {
            return Err(manonica_error::BaseError::new(
                manonica_error::ErrorCode::InvalidRange,
                "manon_energy",
            ));
        }
        Ok(Self { id, position, velocity, energy, state: ManonState::Free })
    }

    pub fn id(&self) -> ManonId { self.id }
    pub fn position(&self) -> Vec3<Metres> { self.position }
    pub fn velocity(&self) -> Vec3<MetresPerSecond> { self.velocity }
    pub fn energy(&self) -> Quantity<Joules> { self.energy }
    pub fn state(&self) -> ManonState { self.state }

    pub(crate) fn set_position(&mut self, position: Vec3<Metres>) {
        self.position = position;
    }
}
