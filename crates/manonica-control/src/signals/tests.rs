use manonica_core::{
    Quantity, Signal, Time, TimeStep,
    units::{Dimensionless, Hertz, PerSecond, Radians},
};

use super::{Constant, Exponential, Pulse, Ramp, Sine, Square, Step, Sum};

fn q(v: f64) -> Quantity<Dimensionless> {
    Quantity::new(v).unwrap()
}
fn t(v: f64) -> Time {
    Time::from_seconds(v).unwrap()
}

#[test]
fn step_changes_exactly_at_threshold() {
    let step = Step::new(t(1.0), q(0.0), q(2.0));
    assert_eq!(step.sample(t(0.999)).unwrap().value(), 0.0);
    assert_eq!(step.sample(t(1.0)).unwrap().value(), 2.0);
}

#[test]
fn sinusoid_is_correct_at_quarter_period() {
    let sine = Sine::new(
        q(2.0),
        q(1.0),
        Quantity::<Hertz>::new(1.0).unwrap(),
        Quantity::<Radians>::new(0.0).unwrap(),
    )
    .unwrap();
    assert!((sine.sample(t(0.25)).unwrap().value() - 3.0).abs() < 1e-12);
}

#[test]
fn slope_and_composition_have_dimension() {
    let ramp = Ramp::new(
        q(1.0),
        Quantity::<PerSecond<Dimensionless>>::new(2.0).unwrap(),
        t(0.0),
    );
    let sum = Sum::new(ramp, Constant::new(q(2.0)));
    assert_eq!(sum.sample(t(2.0)).unwrap().value(), 7.0);
}

#[test]
fn periodic_and_transient_signals() {
    let square = Square::new(q(-1.0), q(1.0), Quantity::<Hertz>::new(2.0).unwrap(), 0.5).unwrap();
    assert_eq!(square.sample(t(0.1)).unwrap().value(), 1.0);
    assert_eq!(square.sample(t(0.3)).unwrap().value(), -1.0);
    let pulse = Pulse::new(t(1.0), TimeStep::from_seconds(0.5).unwrap(), q(0.0), q(3.0));
    assert_eq!(pulse.sample(t(1.3)).unwrap().value(), 3.0);
    assert_eq!(pulse.sample(t(1.5)).unwrap().value(), 0.0);
    let decay = Exponential::new(
        t(0.0),
        q(2.0),
        q(1.0),
        Quantity::<PerSecond<Dimensionless>>::new(1.0).unwrap(),
    )
    .unwrap();
    assert!(
        (decay.sample(t(1.0)).unwrap().value() - (1.0 + 2.0 / std::f64::consts::E)).abs() < 1e-12
    );
}
