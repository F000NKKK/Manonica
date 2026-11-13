//! Cubic Fourier truncation for the same Gaussian scalar-field Hamiltonian.
use super::*;

fn cubic_sources() -> Vec<Source> {
    let make = |x: [f64; 3], p: [f64; 3], q: f64| Source {
        body: MechanicalBody::new(
            Quantity::new(2.0).unwrap(),
            Vec3::try_values(x[0], x[1], x[2]).unwrap(),
            Vec3::try_values(p[0], p[1], p[2]).unwrap(),
        )
        .unwrap(),
        charge: q,
        width: 0.6,
    };
    vec![
        make([0.4, 0.7, 0.2], [0.15, -0.1, 0.07], 0.4),
        make([2.0, 1.8, 2.3], [-0.1, 0.04, -0.03], -0.4),
    ]
}
fn cubic(sources: Vec<Source>, cutoff: i32) -> Result<System, String> {
    if !(1..=12).contains(&cutoff) {
        return Err("invalid cubic cutoff".into());
    }
    cubic_in_domain(sources, cutoff, TAU)
}
fn cubic_in_domain(sources: Vec<Source>, cutoff: i32, length: f64) -> Result<System, String> {
    if !(1..=24).contains(&cutoff) || !length.is_finite() || length <= 0.0 {
        return Err("invalid cubic domain".into());
    }
    let mut waves = Vec::new();
    for x in -cutoff..=cutoff {
        for y in -cutoff..=cutoff {
            for z in -cutoff..=cutoff {
                // One cosine/sine pair for each +/- wave-vector pair; no zero mode.
                if x > 0 || (x == 0 && y > 0) || (x == 0 && y == 0 && z > 0) {
                    waves.push([
                        x as f64 * TAU / length,
                        y as f64 * TAU / length,
                        z as f64 * TAU / length,
                    ]);
                }
            }
        }
    }
    // Fixed a=epsilon=screening=1/pi; mu=a*V/2 scales with volume.
    System::static_modes(sources, waves, length.powi(3) / TAU)
}
fn experiment_3d(cutoff: i32, steps: usize) -> Result<(System, f64, f64), String> {
    let mut system = cubic(cubic_sources(), cutoff)?;
    let e0 = system.energy()?;
    let p0 = system.momentum_vector();
    let mut max_energy: f64 = 0.0;
    let mut max_momentum: f64 = 0.0;
    for _ in 0..steps {
        system.step(TimeStep::from_seconds(1.0 / steps as f64).map_err(|e| e.to_string())?)?;
        max_energy = max_energy.max((system.energy()? - e0).abs());
        for (a, b) in system.momentum_vector().into_iter().zip(p0) {
            max_momentum = max_momentum.max((a - b).abs());
        }
    }
    if max_momentum > 1e-11 {
        return Err(format!("3D momentum error {max_momentum}"));
    }
    Ok((system, max_energy, max_momentum))
}
pub(super) fn validate_3d() -> Result<(), String> {
    let mut previous: Option<f64> = None;
    for steps in [100, 200, 400] {
        let (system, energy, momentum) = experiment_3d(3, steps)?;
        if let Some(error) = previous
            && !(3.8..4.2).contains(&(error / energy))
        {
            return Err(format!("3D energy refinement {}", error / energy));
        }
        if steps == 400 && energy > 1e-6 {
            return Err(format!("3D energy error {energy}"));
        }
        previous = Some(energy);
        println!(
            "3D modes={}, steps={steps}, energy error={energy:.6e} J, max component momentum error={momentum:.6e} kg m/s",
            system.modes.len()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn law_spectral_3d_001_half_lattice_and_static_self_force() {
        for cutoff in [1, 3, 6] {
            let mut sources = cubic_sources();
            sources.truncate(1);
            sources[0].body = MechanicalBody::new(
                sources[0].body.mass(),
                sources[0].body.position(),
                Vec3::try_values(0.0, 0.0, 0.0).unwrap(),
            )
            .unwrap();
            let system = cubic(sources, cutoff).unwrap();
            assert_eq!(
                system.modes.len(),
                (((2 * cutoff + 1).pow(3) - 1) / 2) as usize
            );
            for force in system.force_vector(&system.sources[0]) {
                assert!(force.abs() < 1e-14);
            }
            for p in system.momentum_vector() {
                assert!(p.abs() < 1e-14);
            }
            for (i, a) in system.modes.iter().enumerate() {
                for b in &system.modes[i + 1..] {
                    assert_ne!(a.wave, b.wave);
                    assert_ne!(a.wave, b.wave.map(|v| -v));
                }
            }
        }
    }
    #[test]
    fn single_oblique_mode_has_analytic_force_and_vector_recoil() {
        let mut sources = cubic_sources();
        sources.truncate(1);
        sources[0].body = MechanicalBody::new(
            sources[0].body.mass(),
            Vec3::try_values(0.0, 0.0, 0.0).unwrap(),
            Vec3::try_values(0.0, 0.0, 0.0).unwrap(),
        )
        .unwrap();
        let wave = [1.0, 2.0, -1.0];
        let mut system = System::static_modes(sources, vec![wave], 4.0 * PI * PI).unwrap();
        system.modes[0].a = 0.0;
        system.modes[0].b = 2.0;
        let g = (-0.5 * 6.0 * 0.6_f64.powi(2)).exp();
        system.kick(0.25).unwrap();
        let p = system.sources[0].body.momentum();
        for (actual, k) in [p.x.value(), p.y.value(), p.z.value()]
            .into_iter()
            .zip(wave)
        {
            assert!((actual + 0.2 * g * k).abs() < 1e-14);
        }
        for total in system.momentum_vector() {
            assert!(total.abs() < 1e-13);
        }
    }
    #[test]
    fn law_spectral_3d_002_pair_recoil_translation_and_axis_permutation() {
        let system = cubic(cubic_sources(), 4).unwrap();
        let a = system.force_vector(&system.sources[0]);
        let b = system.force_vector(&system.sources[1]);
        assert!(a.iter().all(|f| f.abs() > 1e-5));
        for axis in 0..3 {
            assert!((a[axis] + b[axis]).abs() < 1e-14);
        }
        let mut shifted = cubic_sources();
        for s in &mut shifted {
            let [x, y, z] = coordinates(s.body);
            let p = s.body.momentum();
            s.body = MechanicalBody::new(
                s.body.mass(),
                Vec3::try_values(x + 0.37, y - 0.51, z + 1.2).unwrap(),
                p,
            )
            .unwrap();
        }
        let shifted = cubic(shifted, 4).unwrap();
        for (a, b) in a.into_iter().zip(shifted.force_vector(&shifted.sources[0])) {
            assert!((a - b).abs() < 1e-14);
        }
        assert!((system.energy().unwrap() - shifted.energy().unwrap()).abs() < 1e-14);
        let mut permuted = cubic_sources();
        for s in &mut permuted {
            let [x, y, z] = coordinates(s.body);
            let p = s.body.momentum();
            s.body = MechanicalBody::new(
                s.body.mass(),
                Vec3::try_values(z, x, y).unwrap(),
                Vec3::try_values(p.z.value(), p.x.value(), p.y.value()).unwrap(),
            )
            .unwrap();
        }
        let rotated = cubic(permuted, 4).unwrap();
        let force = rotated.force_vector(&rotated.sources[0]);
        for (a, b) in [a[2], a[0], a[1]].into_iter().zip(force) {
            assert!((a - b).abs() < 1e-14);
        }
    }
    #[test]
    fn law_spectral_3d_003_force_is_negative_gradient_of_static_energy() {
        let system = cubic(cubic_sources(), 4).unwrap();
        let force = system.force_vector(&system.sources[0]);
        let h = 1e-5;
        for (axis, expected) in force.into_iter().enumerate() {
            let energy = |delta: f64| {
                let mut sources = cubic_sources();
                let s = &mut sources[0];
                let mut x = coordinates(s.body);
                x[axis] += delta;
                s.body = MechanicalBody::new(
                    s.body.mass(),
                    Vec3::try_values(x[0], x[1], x[2]).unwrap(),
                    s.body.momentum(),
                )
                .unwrap();
                cubic(sources, 4).unwrap().energy().unwrap()
            };
            let observed = -(energy(h) - energy(-h)) / (2.0 * h);
            assert!(
                (expected - observed).abs() < 1e-10,
                "force {expected}, energy gradient {observed}"
            );
        }
    }
    #[test]
    fn law_spectral_3d_004_dynamics_and_cutoff_converge() {
        validate_3d().unwrap();
        let reference = cubic(cubic_sources(), 10).unwrap();
        let expected = reference.force_vector(&reference.sources[0]);
        let mut errors = Vec::new();
        for cutoff in [2, 4, 6, 8] {
            let system = cubic(cubic_sources(), cutoff).unwrap();
            let actual = system.force_vector(&system.sources[0]);
            let error = actual
                .into_iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            errors.push(error);
        }
        for pair in errors.windows(2) {
            assert!(pair[1] < pair[0], "3D cutoff errors {errors:?}");
        }
        assert!(errors[3] < 1e-11, "3D cutoff errors {errors:?}");
    }
    #[test]
    fn invalid_cubic_cutoff_and_overflow_never_commit() {
        assert!(cubic(cubic_sources(), 0).is_err());
        assert!(cubic(cubic_sources(), 13).is_err());
        let mut system = cubic(cubic_sources(), 3).unwrap();
        let before = system.clone();
        assert!(
            system
                .step(TimeStep::from_seconds(f64::MAX).unwrap())
                .is_err()
        );
        assert_eq!(system, before);
    }
}

// Independent coordinate-space Gaussian convolution using the heat-kernel
// representation of the 3D screened Green function, epsilon=1/pi.
fn coordinate_force(displacement: [f64; 3], sigma: f64, intervals: usize) -> [f64; 3] {
    let squared = dot(displacement, displacement);
    let h = 64.0 / intervals as f64;
    let integrand = |t: f64| {
        let u = t + sigma * sigma; // Combined Gaussian variance is 2*sigma².
        (-t - squared / (4.0 * u)).exp() / (4.0 * PI * u).powf(1.5) / u
    };
    let mut sum = integrand(0.0) + integrand(64.0);
    for i in 1..intervals {
        sum += (if i % 2 == 0 { 2.0 } else { 4.0 }) * integrand(i as f64 * h);
    }
    // q1*q2=-0.16 and grad(G)=-pi*r/2 times the scalar integral.
    displacement.map(|r| 0.08 * PI * r * sum * h / 3.0)
}
fn image_force(length: f64, sigma: f64, images: i32, intervals: usize) -> [f64; 3] {
    let mut result = [0.0; 3];
    let separation = [-1.6, -1.1, -2.1];
    for x in -images..=images {
        for y in -images..=images {
            for z in -images..=images {
                let image = [x, y, z];
                let force = coordinate_force(
                    std::array::from_fn(|i| separation[i] + length * image[i] as f64),
                    sigma,
                    intervals,
                );
                for i in 0..3 {
                    result[i] += force[i];
                }
            }
        }
    }
    result
}
fn maximum_difference(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}
pub(super) fn validate_coordinates() -> Result<(), String> {
    for sigma in [0.6, 0.8, 1.0] {
        let infinite = coordinate_force([-1.6, -1.1, -2.1], sigma, 8192);
        let mut previous = f64::INFINITY;
        for (scale, cutoff) in [(1.0, 8), (1.5, 12), (2.0, 16)] {
            let length = scale * TAU;
            let mut sources = cubic_sources();
            for s in &mut sources {
                s.width = sigma;
            }
            let system = cubic_in_domain(sources.clone(), cutoff, length)?;
            let force = system.force_vector(&system.sources[0]);
            let refined = cubic_in_domain(sources, cutoff + 2, length)?;
            if maximum_difference(force, refined.force_vector(&refined.sources[0])) > 1e-9 {
                return Err("3D spectral truncation has not converged".into());
            }
            let coordinate = image_force(length, sigma, 3, 4096);
            let quadrature = image_force(length, sigma, 3, 8192);
            let image_check = image_force(length, sigma, 2, 4096);
            if maximum_difference(coordinate, quadrature) > 1e-10
                || maximum_difference(coordinate, image_check) > 1e-9
            {
                return Err(format!(
                    "3D coordinate reference not converged at sigma={sigma}, L={length}"
                ));
            }
            let difference = maximum_difference(force, quadrature);
            if difference > 1e-8 {
                return Err(format!("3D force reference mismatch {difference}"));
            }
            let domain_error = maximum_difference(force, infinite);
            if domain_error >= 0.2 * previous {
                return Err(format!("3D domain refinement failed {domain_error}"));
            }
            previous = domain_error;
            println!(
                "3D sigma={sigma:.1} m, L={length:.6} m, coordinate discrepancy={difference:.3e} N, infinite-domain error={domain_error:.3e} N"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod coordinate_tests {
    use super::*;
    #[test]
    fn law_spectral_3d_005_coordinate_green_reference_period_and_width() {
        validate_coordinates().unwrap();
    }
    #[test]
    fn coordinate_zero_width_limit_matches_analytic_point_yukawa_force() {
        let r: [f64; 3] = [-1.6, -1.1, -2.1];
        let radius = dot(r, r).sqrt();
        // G(r)=exp(-r)/(4r), force=q1*q2*grad(G), q1*q2=-0.16.
        let point =
            r.map(|x| 0.04 * (-radius).exp() * (1.0 / radius.powi(2) + 1.0 / radius.powi(3)) * x);
        let mut errors = Vec::new();
        for sigma in [0.2, 0.1, 0.05] {
            errors.push(maximum_difference(coordinate_force(r, sigma, 8192), point));
        }
        for pair in errors.windows(2) {
            assert!(
                (3.8..4.2).contains(&(pair[0] / pair[1])),
                "3D point errors {errors:?}"
            );
        }
        let point_scale = point.into_iter().map(f64::abs).fold(0.0, f64::max);
        // Away from r=0 Gaussian smoothing changes the screened kernel by
        // exp(sigma²); exp(sigma²)-1 < 2*sigma² on this width interval.
        assert!(errors[2] < point_scale * 2.0 * 0.05_f64.powi(2));
    }
    #[test]
    fn cubic_domain_rejects_invalid_periods_and_cutoffs() {
        for length in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MAX] {
            assert!(cubic_in_domain(cubic_sources(), 4, length).is_err());
        }
        assert!(cubic_in_domain(cubic_sources(), 25, TAU).is_err());
    }
}
