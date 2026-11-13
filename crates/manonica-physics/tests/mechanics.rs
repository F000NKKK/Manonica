use manonica_core::{Quantity, TimeStep, Vec3};
use manonica_physics::{ExternalLoadLedger, MechanicalBody};
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-11, "{a} != {b}");
}
#[test]
fn law_load_001_constant_force_exact_trajectory_impulse_and_work_in_3d() {
    let mut body = MechanicalBody::new(
        Quantity::new(2.0).unwrap(),
        Vec3::try_values(1.0, 2.0, 3.0).unwrap(),
        Vec3::try_values(4.0, -2.0, 0.0).unwrap(),
    )
    .unwrap();
    let initial_energy = body.kinetic_energy().unwrap().value();
    let mut ledger = ExternalLoadLedger::default();
    for _ in 0..20 {
        ledger
            .advance_constant_force(
                &mut body,
                Vec3::try_values(2.0, -4.0, 6.0).unwrap(),
                TimeStep::from_seconds(0.1).unwrap(),
            )
            .unwrap();
    }
    close(body.position().x.value(), 7.0);
    close(body.position().y.value(), -4.0);
    close(body.position().z.value(), 9.0);
    close(body.momentum().x.value(), 8.0);
    close(body.momentum().y.value(), -10.0);
    close(body.momentum().z.value(), 12.0);
    close(ledger.impulse().x.value(), 4.0);
    close(ledger.impulse().y.value(), -8.0);
    close(ledger.impulse().z.value(), 12.0);
    close(ledger.work().value(), 72.0);
    close(
        body.kinetic_energy().unwrap().value() - initial_energy,
        ledger.work().value(),
    );
}
#[test]
fn law_load_002_braking_exports_energy_and_force_reversal_restores_momentum() {
    let mut body = MechanicalBody::new(
        Quantity::new(2.0).unwrap(),
        Vec3::try_values(0.0, 0.0, 0.0).unwrap(),
        Vec3::try_values(4.0, 0.0, 0.0).unwrap(),
    )
    .unwrap();
    let mut ledger = ExternalLoadLedger::default();
    ledger
        .advance_constant_force(
            &mut body,
            Vec3::try_values(-4.0, 0.0, 0.0).unwrap(),
            TimeStep::from_seconds(1.0).unwrap(),
        )
        .unwrap();
    close(body.position().x.value(), 1.0);
    close(body.momentum().x.value(), 0.0);
    close(ledger.work().value(), -4.0);
    ledger
        .advance_constant_force(
            &mut body,
            Vec3::try_values(4.0, 0.0, 0.0).unwrap(),
            TimeStep::from_seconds(1.0).unwrap(),
        )
        .unwrap();
    close(body.momentum().x.value(), 4.0);
    close(ledger.work().value(), 0.0);
    close(ledger.impulse().x.value(), 0.0);
}
#[test]
fn law_load_003_invalid_mass_and_overflow_never_partially_commit() {
    for mass in [0.0, -1.0] {
        assert!(
            MechanicalBody::new(
                Quantity::new(mass).unwrap(),
                Vec3::try_values(0.0, 0.0, 0.0).unwrap(),
                Vec3::try_values(0.0, 0.0, 0.0).unwrap()
            )
            .is_err()
        );
    }
    let mut body = MechanicalBody::new(
        Quantity::new(1.0).unwrap(),
        Vec3::try_values(f64::MAX, 0.0, 0.0).unwrap(),
        Vec3::try_values(1.0, 0.0, 0.0).unwrap(),
    )
    .unwrap();
    let mut ledger = ExternalLoadLedger::default();
    let before = (body, ledger);
    assert!(
        ledger
            .advance_constant_force(
                &mut body,
                Vec3::try_values(0.0, 0.0, 0.0).unwrap(),
                TimeStep::from_seconds(f64::MAX).unwrap()
            )
            .is_err()
    );
    assert_eq!((body, ledger), before);
    assert!(
        ledger
            .apply_impulse(&mut body, Vec3::try_values(f64::MAX, 0.0, 0.0).unwrap())
            .is_err()
    );
    assert_eq!((body, ledger), before);
}

#[test]
fn ledger_overflow_rolls_back_even_when_candidate_body_is_valid() {
    let initial = MechanicalBody::new(
        Quantity::new(1e308).unwrap(),
        Vec3::try_values(0.0, 0.0, 0.0).unwrap(),
        Vec3::try_values(0.0, 0.0, 0.0).unwrap(),
    )
    .unwrap();
    let mut ledger = ExternalLoadLedger::default();
    for sign in [1.0, -1.0, 1.0] {
        let mut body = initial;
        ledger
            .apply_impulse(&mut body, Vec3::try_values(sign * 1e308, 0.0, 0.0).unwrap())
            .unwrap();
    }
    let mut body = initial;
    let before = ledger;
    assert!(
        ledger
            .apply_impulse(&mut body, Vec3::try_values(-1e308, 0.0, 0.0).unwrap())
            .is_err()
    );
    assert_eq!(ledger, before);
    assert_eq!(body, initial);
}
