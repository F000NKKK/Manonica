//! API contracts complement the analytic simulator scenarios.
use manonica_core::{Quantity, TimeStep, Vec3};
use manonica_physics::{
    ExternalLoadLedger, GaussianBody, MechanicalBody, PeriodicSpectralDomain, SpectralMechanics,
};
use std::f64::consts::TAU;
fn source() -> GaussianBody {
    GaussianBody {
        body: MechanicalBody::new(
            Quantity::new(2.0).unwrap(),
            Vec3::try_values(0.4, 0.7, 0.2).unwrap(),
            Vec3::try_values(0.15, -0.1, 0.07).unwrap(),
        )
        .unwrap(),
        charge: 0.4,
        width: 0.6,
    }
}
fn state() -> SpectralMechanics {
    PeriodicSpectralDomain::new(Vec3::try_values(TAU, TAU, TAU).unwrap(), [2, 2, 2])
        .unwrap()
        .static_equilibrium(vec![source()])
        .unwrap()
}
#[test]
fn domain_and_wave_contracts_reject_ambiguous_or_unbounded_geometry() {
    let lengths = Vec3::try_values(TAU, TAU, TAU).unwrap();
    assert_eq!(
        PeriodicSpectralDomain::new(lengths, [2, 2, 2])
            .unwrap()
            .mode_count(),
        62
    );
    assert!(PeriodicSpectralDomain::new(lengths, [0, 0, 0]).is_err());
    assert!(PeriodicSpectralDomain::new(lengths, [65, 65, 65]).is_err());
    assert!(PeriodicSpectralDomain::new(lengths, [usize::MAX, 0, 0]).is_err());
    for length in [0.0, -1.0, f64::MAX] {
        assert!(
            PeriodicSpectralDomain::new(
                Vec3::try_values(length, length, length).unwrap(),
                [1, 1, 1]
            )
            .is_err()
        );
    }
    for waves in [
        vec![[0.0; 3]],
        vec![[-1.0, 0.0, 0.0]],
        vec![[1.0, 0.0, 0.0], [1.0, -0.0, 0.0]],
        vec![[f64::NAN, 0.0, 0.0]],
        vec![[f64::MAX, 0.0, 0.0]],
    ] {
        assert!(SpectralMechanics::static_equilibrium(vec![source()], waves, 1.0).is_err());
    }
    for inertia in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            SpectralMechanics::static_equilibrium(vec![source()], vec![[1.0, 0.0, 0.0]], inertia)
                .is_err()
        );
    }
}
#[test]
fn invalid_sources_and_storage_never_enter_or_partially_change_state() {
    for (charge, width) in [
        (f64::NAN, 0.6),
        (f64::INFINITY, 0.6),
        (0.4, 0.0),
        (0.4, -1.0),
        (0.4, f64::NAN),
    ] {
        let mut input = source();
        input.charge = charge;
        input.width = width;
        assert!(
            SpectralMechanics::static_equilibrium(vec![input], vec![[1.0, 0.0, 0.0]], 1.0).is_err()
        );
    }
    let mut system = state();
    let before = system.clone();
    assert!(system.replace_mode_storage(&[]).is_err());
    assert_eq!(system, before);
    for invalid in [f64::NAN, f64::INFINITY, f64::MAX] {
        let mut storage: Vec<_> = system
            .modes()
            .iter()
            .map(|m| [m.a, m.b, m.pa, m.pb])
            .collect();
        storage[0] = [0.2, 0.3, 0.4, 0.5];
        storage.last_mut().unwrap()[0] = invalid;
        assert!(system.replace_mode_storage(&storage).is_err());
        assert_eq!(system, before);
    }
    // A finite but linearly unstable Verlet step must also be rejected.
    assert!(
        system
            .advance(TimeStep::from_seconds(1.0).unwrap())
            .is_err()
    );
    assert_eq!(system, before);
    assert!(
        system
            .advance(TimeStep::from_seconds(f64::MAX).unwrap())
            .is_err()
    );
    assert_eq!(system, before);
    assert!(
        system
            .potential_kick(TimeStep::from_seconds(f64::MAX).unwrap())
            .is_err()
    );
    assert_eq!(system, before);
}
#[test]
fn external_loading_updates_typed_observables_and_ledger_atomically() {
    let mut system = state();
    let initial_energy = system.total_energy().unwrap().value();
    let initial = system.total_momentum().unwrap();
    let mut ledger = ExternalLoadLedger::default();
    system
        .apply_external_impulse(0, Vec3::try_values(0.2, -0.3, 0.4).unwrap(), &mut ledger)
        .unwrap();
    let final_momentum = system.total_momentum().unwrap();
    assert!((final_momentum.x.value() - initial.x.value() - 0.2).abs() < 1e-12);
    assert!((final_momentum.y.value() - initial.y.value() + 0.3).abs() < 1e-12);
    assert!((final_momentum.z.value() - initial.z.value() - 0.4).abs() < 1e-12);
    // J dot (p+J/2)/m for p=[.15,-.1,.07], m=2.
    let expected_work = (0.2 * (0.15 + 0.1) + (-0.3) * (-0.1 - 0.15) + 0.4 * (0.07 + 0.2)) / 2.0;
    assert!((ledger.work().value() - expected_work).abs() < 1e-12);
    assert!(
        (system.total_energy().unwrap().value() - initial_energy - ledger.work().value()).abs()
            < 1e-12
    );
    let before = system.clone();
    let ledger_before = ledger;
    assert!(system.force_on(1).is_err());
    assert!(
        system
            .apply_external_impulse(1, Vec3::try_values(0.1, 0.0, 0.0).unwrap(), &mut ledger)
            .is_err()
    );
    assert_eq!(system, before);
    assert_eq!(ledger, ledger_before);
    assert!(
        system
            .apply_external_impulse(
                0,
                Vec3::try_values(f64::MAX, 0.0, 0.0).unwrap(),
                &mut ledger
            )
            .is_err()
    );
    assert_eq!(system, before);
    assert_eq!(ledger, ledger_before);
}
