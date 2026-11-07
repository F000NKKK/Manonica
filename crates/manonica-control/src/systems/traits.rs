use manonica_core::{Quantity, TimeStep, units::Unit};

use crate::ControlError;

/// Модель динамической системы с контролируемым состоянием.
pub trait DynamicSystem<Input: Unit, Output: Unit> {
    fn step(
        &mut self,
        input: &[Quantity<Input>],
        dt: TimeStep,
    ) -> Result<Vec<Quantity<Output>>, ControlError>;
}
