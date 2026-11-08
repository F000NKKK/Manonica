use manonica_core::{Quantity, TimeStep, units::Unit};

use crate::{
    ControlError,
    blocks::{ControllerTerm, Gain, ProportionalMode},
};

#[derive(Debug, Clone, Copy)]
pub struct Proportional<In: Unit, Out: Unit> {
    gain: Gain<In, Out, ProportionalMode>,
}

impl<In: Unit, Out: Unit> Proportional<In, Out> {
    pub fn new(gain: Gain<In, Out, ProportionalMode>) -> Self {
        Self { gain }
    }
}

impl<In: Unit, Out: Unit> ControllerTerm<In, Out> for Proportional<In, Out> {
    fn update(
        &mut self,
        error: Quantity<In>,
        _dt: TimeStep,
    ) -> Result<Quantity<Out>, ControlError> {
        Ok(Quantity::new(self.gain.coefficient() * error.value())?)
    }

    fn reset(&mut self) {}
}
