use manonica_core::{
    Quantity, TimeStep, Vec3,
    units::{Dimensionless, KilogramMetresPerSecond, Kilograms, Metres, PerSecond},
};
use manonica_physics::{DissipativeExchange, MechanicalBody};
fn body(m: f64, v: [f64; 3]) -> MechanicalBody {
    MechanicalBody::new(
        Quantity::<Kilograms>::new(m).unwrap(),
        Vec3::<Metres>::try_values(0., 0., 0.).unwrap(),
        Vec3::<KilogramMetresPerSecond>::try_values(m * v[0], m * v[1], m * v[2]).unwrap(),
    )
    .unwrap()
}
fn p(b: MechanicalBody) -> [f64; 3] {
    let p = b.momentum();
    [p.x.value(), p.y.value(), p.z.value()]
}
#[test]
fn exact_relaxation_has_opposite_impulses_and_converts_kinetic_energy_to_heat() {
    for ma in [0.1, 1., 10.] {
        for mb in [0.2, 2., 20.] {
            let mut a = body(ma, [3., -2., 1.]);
            let mut b = body(mb, [-1., 4., 2.]);
            let before = std::array::from_fn::<_, 3, _>(|i| p(a)[i] + p(b)[i]);
            let energy = a.kinetic_energy().unwrap().value() + b.kinetic_energy().unwrap().value();
            let law = DissipativeExchange::new(
                Quantity::<PerSecond<Dimensionless>>::nonnegative(2.).unwrap(),
            )
            .unwrap();
            let report = law
                .exchange(&mut a, &mut b, TimeStep::from_seconds(0.3).unwrap())
                .unwrap();
            for (i, initial) in before.iter().enumerate() {
                assert!((p(a)[i] + p(b)[i] - initial).abs() < 1e-12);
                let relative = p(a)[i] / ma - p(b)[i] / mb;
                let initial_relative = [4., -6., -1.][i];
                assert!((relative - initial_relative * (-0.6_f64).exp()).abs() < 1e-12);
            }
            let after = a.kinetic_energy().unwrap().value() + b.kinetic_energy().unwrap().value();
            assert!(report.heat.value() > 0.0);
            assert!((after + report.heat.value() - energy).abs() < 1e-12 * energy);
            assert_eq!(a.position(), b.position());
        }
    }
}
#[test]
fn equal_velocity_and_zero_rate_do_not_create_force_or_heat() {
    for rate in [0., 20.] {
        let mut a = body(1., [2., 3., 4.]);
        let mut b = body(10., [2., 3., 4.]);
        let initial = (a, b);
        let report = DissipativeExchange::new(Quantity::new(rate).unwrap())
            .unwrap()
            .exchange(&mut a, &mut b, TimeStep::from_seconds(0.1).unwrap())
            .unwrap();
        assert_eq!((a, b), initial);
        assert_eq!(report.heat.value(), 0.);
    }
}
#[test]
fn overflow_and_invalid_rate_leave_bodies_unchanged() {
    assert!(DissipativeExchange::new(Quantity::new(-1.).unwrap()).is_err());
    let mut a = body(1., [1., 0., 0.]);
    let mut b = body(1., [0., 0., 0.]);
    let before = (a, b);
    assert!(
        DissipativeExchange::new(Quantity::new(f64::MAX).unwrap())
            .unwrap()
            .exchange(&mut a, &mut b, TimeStep::from_seconds(2.).unwrap())
            .is_err()
    );
    assert_eq!((a, b), before);
}
