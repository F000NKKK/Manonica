use super::{
    ControllerTerm, Derivative, DerivativeMode, Gain, Integral, IntegralMode, Limits, Pid,
    Proportional, ProportionalMode,
};
use manonica_core::{
    Instrumented, Quantity, TimeStep,
    units::{Joules, Newtons},
};

fn q(value: f64) -> Quantity<Joules> {
    Quantity::new(value).unwrap()
}
fn dt(value: f64) -> TimeStep {
    TimeStep::from_seconds(value).unwrap()
}

#[test]
fn independent_controller_terms() {
    let mut p = Proportional::new(Gain::<Joules, Newtons, ProportionalMode>::new(2.0).unwrap());
    let mut i = Integral::new(Gain::<Joules, Newtons, IntegralMode>::new(1.0).unwrap());
    let mut d = Derivative::new(
        Gain::<Joules, Newtons, DerivativeMode>::new(0.5).unwrap(),
        None,
    );
    assert_eq!(p.update(q(3.0), dt(1.0)).unwrap().value(), 6.0);
    assert_eq!(i.update(q(2.0), dt(1.0)).unwrap().value(), 2.0);
    assert_eq!(i.update(q(4.0), dt(1.0)).unwrap().value(), 5.0);
    assert_eq!(d.update(q(2.0), dt(1.0)).unwrap().value(), 0.0);
    assert_eq!(d.update(q(4.0), dt(1.0)).unwrap().value(), 1.0);
}

#[test]
fn pid_is_composed_with_typed_limits_and_diagnostics() {
    let p = Proportional::new(Gain::<Joules, Newtons, ProportionalMode>::new(2.0).unwrap());
    let i = Integral::new(Gain::<Joules, Newtons, IntegralMode>::new(3.0).unwrap());
    let d = Derivative::new(
        Gain::<Joules, Newtons, DerivativeMode>::new(0.0).unwrap(),
        None,
    );
    let limits = Limits::new(
        Quantity::<Newtons>::new(-5.0).unwrap(),
        Quantity::<Newtons>::new(5.0).unwrap(),
    )
    .unwrap();
    let mut pid = Pid::new(p, i, d).with_limits(limits);
    assert_eq!(pid.update(q(10.0), dt(1.0)).unwrap().value(), 5.0);
    assert!(pid.snapshot().saturated);
    assert_eq!(pid.snapshot().integral, Some(0.0));
}

#[test]
fn reset_restarts_integrator_and_derivative() {
    let mut i = Integral::new(Gain::<Joules, Newtons, IntegralMode>::new(1.0).unwrap());
    i.update(q(2.0), dt(1.0)).unwrap();
    i.reset();
    assert_eq!(i.update(q(3.0), dt(1.0)).unwrap().value(), 3.0);
}
