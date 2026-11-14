use manonica_materials::{ThermalReservoir, exchange_heat_exact, radiate_to_bath};
#[test]
fn exact_contact_conserves_heat_and_increases_entropy() {
    let mut a = ThermalReservoir {
        heat_joules: 10000.,
        capacity_joules_per_kelvin: 10.,
    };
    let mut b = ThermalReservoir {
        heat_joules: 0.,
        capacity_joules_per_kelvin: 100.,
    };
    let initial = a.heat_joules + b.heat_joules;
    let ta = a.temperature_kelvin(300.).unwrap();
    let tb = b.temperature_kelvin(300.).unwrap();
    exchange_heat_exact(&mut a, &mut b, 20., 0.5).unwrap();
    let fa = a.temperature_kelvin(300.).unwrap();
    let fb = b.temperature_kelvin(300.).unwrap();
    assert!((a.heat_joules + b.heat_joules - initial).abs() < 1e-10);
    assert!(fa >= fb);
    assert!((fa - fb - (ta - tb) * (-20_f64 * 0.5 * (0.1 + 0.01)).exp()).abs() < 1e-10);
    let entropy = 10. * (fa / ta).ln() + 100. * (fb / tb).ln();
    assert!(entropy > 0.);
}
#[test]
fn radiation_accounts_for_every_escaped_joule_and_rejects_invalid_input() {
    let mut r = ThermalReservoir {
        heat_joules: 10000.,
        capacity_joules_per_kelvin: 10.,
    };
    let before = r;
    let escaped = radiate_to_bath(&mut r, 300., 0.005, 6., 0.025).unwrap();
    assert!(escaped > 0.);
    assert_eq!(r.heat_joules + escaped, before.heat_joules);
    let copy = r;
    assert!(radiate_to_bath(&mut r, 300., 2., 6., 0.025).is_err());
    assert_eq!(r, copy);
}
