use std::marker::PhantomData;

use manonica_core::{Quantity, Signal, Time, TimeStep, units::Unit};
use manonica_error::BaseError;

/// Сумма совместимых по физической размерности сигналов.
pub struct Sum<U: Unit, A: Signal<U>, B: Signal<U>> {
    left: A,
    right: B,
    unit: PhantomData<U>,
}

impl<U: Unit, A: Signal<U>, B: Signal<U>> Sum<U, A, B> {
    pub fn new(left: A, right: B) -> Self {
        Self {
            left,
            right,
            unit: PhantomData,
        }
    }
}

impl<U: Unit, A: Signal<U>, B: Signal<U>> Signal<U> for Sum<U, A, B> {
    fn sample(&self, time: Time) -> Result<Quantity<U>, BaseError> {
        Quantity::new(self.left.sample(time)?.value() + self.right.sample(time)?.value())
    }
}

pub struct Scale<U: Unit, S: Signal<U>> {
    signal: S,
    factor: f64,
    unit: PhantomData<U>,
}

impl<U: Unit, S: Signal<U>> Scale<U, S> {
    pub fn new(signal: S, factor: f64) -> Result<Self, BaseError> {
        if !factor.is_finite() {
            return Err(BaseError::new(
                manonica_error::ErrorCode::NonFinite,
                "scale",
            ));
        }
        Ok(Self {
            signal,
            factor,
            unit: PhantomData,
        })
    }
}

impl<U: Unit, S: Signal<U>> Signal<U> for Scale<U, S> {
    fn sample(&self, time: Time) -> Result<Quantity<U>, BaseError> {
        Quantity::new(self.signal.sample(time)?.value() * self.factor)
    }
}

pub struct Delay<U: Unit, S: Signal<U>> {
    signal: S,
    amount: TimeStep,
    unit: PhantomData<U>,
}

impl<U: Unit, S: Signal<U>> Delay<U, S> {
    pub fn new(signal: S, amount: TimeStep) -> Self {
        Self {
            signal,
            amount,
            unit: PhantomData,
        }
    }
}

impl<U: Unit, S: Signal<U>> Signal<U> for Delay<U, S> {
    fn sample(&self, time: Time) -> Result<Quantity<U>, BaseError> {
        self.signal
            .sample(Time::from_seconds(time.seconds() - self.amount.seconds())?)
    }
}
