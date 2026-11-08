use std::marker::PhantomData;

use manonica_core::{Instrumented, Quantity, TimeStep, units::Unit};
use manonica_error::{BaseError, ErrorCode};

use crate::{
    ControlError,
    blocks::{ControllerSnapshot, ControllerTerm, IntegralTerm, Limits},
};

/// PID-композиция из независимых компонентных трейтов.
/// Все компоненты клонируются для транзакционного обновления.
#[derive(Debug, Clone)]
pub struct Pid<In, Out, P, I, D>
where
    In: Unit,
    Out: Unit,
    P: ControllerTerm<In, Out> + Clone,
    I: IntegralTerm<In, Out> + Clone,
    D: ControllerTerm<In, Out> + Clone,
{
    proportional: P,
    integral: I,
    derivative: D,
    limits: Option<Limits<Out>>,
    last: ControllerSnapshot,
    unit: PhantomData<In>,
}

impl<In, Out, P, I, D> Pid<In, Out, P, I, D>
where
    In: Unit,
    Out: Unit,
    P: ControllerTerm<In, Out> + Clone,
    I: IntegralTerm<In, Out> + Clone,
    D: ControllerTerm<In, Out> + Clone,
{
    pub fn new(proportional: P, integral: I, derivative: D) -> Self {
        Self {
            proportional,
            integral,
            derivative,
            limits: None,
            last: ControllerSnapshot::default(),
            unit: PhantomData,
        }
    }

    pub fn with_limits(mut self, limits: Limits<Out>) -> Self {
        self.limits = Some(limits);
        self
    }
}

impl<In, Out, P, I, D> ControllerTerm<In, Out> for Pid<In, Out, P, I, D>
where
    In: Unit,
    Out: Unit,
    P: ControllerTerm<In, Out> + Clone,
    I: IntegralTerm<In, Out> + Clone,
    D: ControllerTerm<In, Out> + Clone,
{
    fn update(&mut self, error: Quantity<In>, dt: TimeStep) -> Result<Quantity<Out>, ControlError> {
        let mut p = self.proportional.clone();
        let mut i = self.integral.clone();
        let mut d = self.derivative.clone();
        let p_value = p.update(error, dt)?.value();
        let mut i_value = i.update(error, dt)?.value();
        let d_value = d.update(error, dt)?.value();
        let old_i = self.integral.current_output()?.value();
        let mut raw = p_value + i_value + d_value;
        if !raw.is_finite() {
            return Err(BaseError::new(ErrorCode::NonFinite, "pid_output").into());
        }

        // Условная остановка интегрирования, если оно усиливает насыщение.
        if let Some(limits) = self.limits {
            let delta_i = i_value - old_i;
            let pushing_upper = raw > limits.max().value() && delta_i > 0.0;
            let pushing_lower = raw < limits.min().value() && delta_i < 0.0;
            if pushing_upper || pushing_lower {
                i = self.integral.clone();
                i_value = old_i;
                raw = p_value + i_value + d_value;
            }
        }

        let raw = Quantity::<Out>::new(raw)?;
        let (output, saturated) = if let Some(limits) = self.limits {
            limits.apply(raw)?
        } else {
            (raw, false)
        };
        self.proportional = p;
        self.integral = i;
        self.derivative = d;
        self.last = ControllerSnapshot {
            error: Some(error.value()),
            proportional: Some(p_value),
            integral: Some(i_value),
            derivative: Some(d_value),
            output: Some(output.value()),
            saturated,
        };
        Ok(output)
    }

    fn reset(&mut self) {
        self.proportional.reset();
        self.integral.reset();
        self.derivative.reset();
        self.last = ControllerSnapshot::default();
    }
}

impl<In, Out, P, I, D> Instrumented for Pid<In, Out, P, I, D>
where
    In: Unit,
    Out: Unit,
    P: ControllerTerm<In, Out> + Clone,
    I: IntegralTerm<In, Out> + Clone,
    D: ControllerTerm<In, Out> + Clone,
{
    type Snapshot = ControllerSnapshot;
    fn snapshot(&self) -> Self::Snapshot {
        self.last
    }
}
