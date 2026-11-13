//! Independent numerical reference experiments for materials laws.
//! Each constitutive equation is checked against a quantitative reference.
use manonica_core::{Quantity, units::ManonDensity};
use manonica_materials::{DirectionalConductivity, ManaChannel, MaterialLattice, TrapMaterial};
use manonica_physics::{FreeManaField, GridShape};

fn close(label: &str, measured: f64, expected: f64, tolerance: f64) {
    assert!(
        (measured - expected).abs() <= tolerance,
        "{label}: measured={measured:.12e}, expected={expected:.12e}, tolerance={tolerance:.3e}",
    );
}
fn density(n: f64) -> Quantity<ManonDensity> {
    Quantity::nonnegative(n).unwrap()
}

/// LAW-HEAT-001: dQ/dt = -k Q for coupling to a fixed-temperature bath.
/// The exact solution after time t is Q(t) = Q(0) exp(-kt), and every
/// removed joule must be reported as exported heat.
#[test]
fn law_bath_cooling_matches_exponential_solution_and_energy_export() {
    let shape = GridShape::new(1, 1, 1).unwrap();
    let mut field = FreeManaField::uniform(shape, 1.0, density(50.0)).unwrap();
    let mut sites = MaterialLattice::new(shape);
    let material = TrapMaterial {
        capacity: 8.0,
        capture_rate: 4.0,
        release_rate: 0.0,
        binding_energy: 2.0,
        heat_capacity: 5.0,
        reference_temperature: 300.0,
        activation_temperature: 0.0,
        max_flux: 8.0,
        conductivity: 0.0,
    };
    sites.install([0, 0, 0], material).unwrap();
    sites.step(&mut field, 1.0).unwrap();
    let initial = sites.at(0, 0, 0).unwrap().heat;
    assert!(initial > 0.0);
    let count_before = sites.balance(&field).unwrap().manons();
    let k = 0.35;
    let dt = 0.2;
    let mut exported = 0.0;
    for _ in 0..20 {
        exported += sites.cool_into_bath(dt, k).unwrap();
    }
    let remaining = sites.at(0, 0, 0).unwrap().heat;
    let expected = initial * (-k * dt * 20.0).exp();
    close("exponential cooling", remaining, expected, 1e-11);
    close(
        "exported heat balance",
        remaining + exported,
        initial,
        1e-11,
    );
    close(
        "manons unaffected by cooling",
        sites.balance(&field).unwrap().manons(),
        count_before,
        1e-11,
    );
    close(
        "temperature from specific heat",
        sites.peak_temperature_kelvin().unwrap(),
        300.0 + remaining / material.heat_capacity,
        1e-11,
    );
}

/// LAW-MAT-ANISO-001: J=-K grad(n) projected onto a channel.
/// For principal K=[4,1,1], u_x^T K u_x=4 and u_y^T K u_y=1.
/// A single time step, away from saturation, must transfer in this ratio.
#[test]
fn law_anisotropic_crystal_channels_match_tensor_projection_and_flux() {
    let shape = GridShape::new(2, 2, 1).unwrap();
    let crystal = DirectionalConductivity {
        principal: [4.0, 1.0, 1.0],
        rotation_z_radians: 0.0,
    };
    close("Kxx", crystal.along([1.0, 0.0, 0.0]).unwrap(), 4.0, 1e-12);
    close("Kyy", crystal.along([0.0, 1.0, 0.0]).unwrap(), 1.0, 1e-12);
    let simulate = |target: [usize; 3]| {
        let mut lattice = MaterialLattice::new(shape);
        lattice
            .add_directional_channel(
                ManaChannel {
                    from: [0, 0, 0],
                    to: target,
                    length: 1.0,
                    area: 1.0,
                    conductance: 1.0,
                    max_flux: 1000.0,
                },
                crystal,
            )
            .unwrap();
        let mut field = FreeManaField::uniform(shape, 1.0, density(0.0)).unwrap();
        field.set(0, 0, 0, density(10.0)).unwrap();
        let before = field.total_manons().unwrap().value();
        lattice.step(&mut field, 0.1).unwrap();
        close(
            "inventory conserved",
            field.total_manons().unwrap().value(),
            before,
            1e-12,
        );
        field.at(target[0], target[1], target[2]).unwrap().value()
    };
    close("x directed flow", simulate([1, 0, 0]), 4.0, 1e-12);
    close("y directed flow", simulate([0, 1, 0]), 1.0, 1e-12);
}

/// LAW-MAT-ANISO-002/003: isotropy is rotation invariant;
/// rotating the crystal by pi/2 exchanges the X/Y conductivities.
#[test]
fn law_anisotropy_isotropic_limit_and_rotating_principal_axes() {
    let iso = DirectionalConductivity {
        principal: [2.0, 2.0, 2.0],
        rotation_z_radians: 0.71,
    };
    for direction in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.4, 0.5, 0.8]] {
        close(
            "isotropic directional multiplier",
            iso.along(direction).unwrap(),
            2.0,
            1e-12,
        );
    }
    let rotated = DirectionalConductivity {
        principal: [4.0, 1.0, 1.0],
        rotation_z_radians: std::f64::consts::FRAC_PI_2,
    };
    close(
        "rotated X",
        rotated.along([1.0, 0.0, 0.0]).unwrap(),
        1.0,
        1e-12,
    );
    close(
        "rotated Y",
        rotated.along([0.0, 1.0, 0.0]).unwrap(),
        4.0,
        1e-12,
    );
}

/// LAW-MAT-ANISO-004: positive semidefinite conductivity cannot create
/// negative transport. A zero eigenvalue blocks the associated flow;
/// invalid eigenvalues or orientation are rejected transactionally.
#[test]
fn law_anisotropic_zero_conductance_and_invalid_coefficients() {
    let shape = GridShape::new(2, 1, 1).unwrap();
    let blocked = DirectionalConductivity {
        principal: [0.0, 5.0, 1.0],
        rotation_z_radians: 0.0,
    };
    let channel = ManaChannel {
        from: [0, 0, 0],
        to: [1, 0, 0],
        length: 1.0,
        area: 1.0,
        conductance: 1.0,
        max_flux: 100.0,
    };
    let mut lattice = MaterialLattice::new(shape);
    lattice.add_directional_channel(channel, blocked).unwrap();
    let mut field = FreeManaField::uniform(shape, 1.0, density(0.0)).unwrap();
    field.set(0, 0, 0, density(10.0)).unwrap();
    lattice.step(&mut field, 0.1).unwrap();
    close(
        "zero eigenvalue blocks transport",
        field.at(0, 0, 0).unwrap().value(),
        10.0,
        1e-12,
    );
    let invalid = DirectionalConductivity {
        principal: [-1.0, 1.0, 1.0],
        rotation_z_radians: 0.0,
    };
    assert!(lattice.add_directional_channel(channel, invalid).is_err());
    assert!(
        DirectionalConductivity {
            principal: [1.0; 3],
            rotation_z_radians: f64::NAN,
        }
        .along([1.0, 0.0, 0.0])
        .is_err()
    );
    assert!(blocked.along([0.0; 3]).is_err());
}

fn kinetic_material() -> TrapMaterial {
    TrapMaterial {
        capacity: 8.0,
        capture_rate: 0.05,
        release_rate: 0.0,
        binding_energy: 2.0,
        heat_capacity: 5.0,
        reference_temperature: 300.0,
        activation_temperature: 0.0,
        max_flux: 1000.0,
        conductivity: 0.0,
    }
}

/// LAW-TRAP-001: isolated capture obeys db/dt=k(N-b)(C-b)/V.
/// With N=C and b(0)=0, b(t)=N*a*t/(1+a*t), a=k*N/V.
/// Finite free inventory changes the rate; no infinite bath is assumed.
#[test]
fn law_trap_capture_converges_to_finite_reservoir_kinetics() {
    let shape = GridShape::new(1, 1, 1).unwrap();
    let material = kinetic_material();
    let dx: f64 = 0.5;
    let volume = dx.powi(3);
    let time = 0.5;
    let a = material.capture_rate * material.capacity / volume;
    let exact = material.capacity * a * time / (1.0 + a * time);
    let mut errors = Vec::new();
    for steps in [20, 40, 80, 160] {
        let mut field =
            FreeManaField::uniform(shape, dx, density(material.capacity / volume)).unwrap();
        let mut sites = MaterialLattice::new(shape);
        sites.install([0, 0, 0], material).unwrap();
        for _ in 0..steps {
            let balance = sites.step(&mut field, time / steps as f64).unwrap();
            close(
                "closed trap count",
                balance.manons(),
                material.capacity,
                1e-11,
            );
            close("capture energy", balance.energy(), 0.0, 1e-11);
            let cell = sites.at(0, 0, 0).unwrap();
            assert!((0.0..=material.capacity).contains(&cell.bound));
            close(
                "capture heat",
                cell.heat,
                material.binding_energy * cell.bound,
                1e-11,
            );
        }
        errors.push((sites.total_bound() - exact).abs());
    }
    for pair in errors.windows(2) {
        assert!(
            (1.8..2.2).contains(&(pair[0] / pair[1])),
            "first-order capture errors: {errors:?}"
        );
    }
    assert!(errors[3] < 0.01, "absolute kinetic error: {errors:?}");
}

/// LAW-TRAP-002: without recapture and with sufficient stored heat,
/// db/dt=-r*b gives b(t)=b0 exp(-r*t), Q(t)=E_bind*b(t).
#[test]
fn law_trap_release_matches_exponential_and_pays_binding_energy() {
    let shape = GridShape::new(1, 1, 1).unwrap();
    let mut field = FreeManaField::uniform(shape, 0.5, density(800.0)).unwrap();
    let mut sites = MaterialLattice::new(shape);
    let mut material = kinetic_material();
    material.capture_rate = 100.0;
    sites.install([0, 0, 0], material).unwrap();
    sites.step(&mut field, 1.0).unwrap();
    close("prepared bound inventory", sites.total_bound(), 8.0, 1e-12);
    material.capture_rate = 0.0;
    material.release_rate = 0.7;
    sites.install([0, 0, 0], material).unwrap();
    for step in 1..=40 {
        let balance = sites.step(&mut field, 0.05).unwrap();
        let expected = 8.0 * (-0.7 * step as f64 * 0.05).exp();
        close("exponential release", balance.bound, expected, 1e-11);
        close("released free count", balance.free, 100.0 - expected, 1e-11);
        close(
            "remaining heat",
            balance.thermal_joules,
            2.0 * expected,
            1e-11,
        );
        close("release energy", balance.energy(), 0.0, 1e-11);
    }
    // Removing all heat prevents spontaneous endothermic release.
    let exported = sites.cool_into_bath(1.0, 1000.0).unwrap();
    let bound = sites.total_bound();
    let balance = sites.step(&mut field, 1.0).unwrap();
    close(
        "no heat means no passive release",
        balance.bound,
        bound,
        1e-12,
    );
    close(
        "exported energy ledger",
        balance.energy() + exported,
        0.0,
        1e-11,
    );
}

/// LAW-TRAP-003: finite supply, capacity and flux are distinct limits.
#[test]
fn law_trap_capture_limits_and_invalid_step_preserve_state() {
    let shape = GridShape::new(1, 1, 1).unwrap();
    for (free, max_flux, expected) in [
        (2.0, 1000.0, 2.0),
        (100.0, 1000.0, 8.0),
        (100.0, 3.0, 0.75),
        (100.0, 0.0, 0.0),
    ] {
        let mut field = FreeManaField::uniform(shape, 1.0, density(free)).unwrap();
        let mut sites = MaterialLattice::new(shape);
        let mut material = kinetic_material();
        material.capture_rate = 1000.0;
        material.max_flux = max_flux;
        sites.install([0, 0, 0], material).unwrap();
        let balance = sites.step(&mut field, 0.25).unwrap();
        close("limited capture", balance.bound, expected, 1e-12);
        close("limited inventory", balance.manons(), free, 1e-12);
        close("limited energy", balance.energy(), 0.0, 1e-12);
        let densities = field.densities().to_vec();
        let heat = sites.at(0, 0, 0).unwrap().heat;
        for invalid_dt in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(sites.step(&mut field, invalid_dt).is_err());
            assert_eq!(field.densities(), densities);
            assert_eq!(sites.total_bound(), expected);
            assert_eq!(sites.at(0, 0, 0).unwrap().heat, heat);
        }
    }
}

/// LAW-TRAP-004: db/dt = alpha*(N-b)*(C-b) - r*b.
/// Writing the RHS as alpha*(b-lo)*(b-hi) gives
/// (b-lo)/(b-hi) = ((b0-lo)/(b0-hi))*exp(-alpha*(hi-lo)*t).
/// Both empty and fully charged traps must approach the same stable root.
#[test]
fn law_joint_trap_kinetics_converges_from_both_sides_of_equilibrium() {
    let shape = GridShape::new(1, 1, 1).unwrap();
    let dx: f64 = 0.5;
    let volume = dx.powi(3);
    let total = 10.0;
    let mut material = kinetic_material();
    material.release_rate = 0.7;
    let alpha = material.capture_rate / volume;
    let sum = total + material.capacity + material.release_rate / alpha;
    let discriminant = (sum * sum - 4.0 * total * material.capacity).sqrt();
    let hi = (sum + discriminant) / 2.0;
    // Product of the two roots is N*C; avoid cancellation in the small root.
    let lo = total * material.capacity / hi;
    let exact = |initial: f64, time: f64| {
        let q = (initial - lo) / (initial - hi) * (-alpha * (hi - lo) * time).exp();
        (lo - q * hi) / (1.0 - q)
    };
    assert!(0.0 < lo && lo < material.capacity);
    for charged in [false, true] {
        let prepare = || {
            let mut field = FreeManaField::uniform(shape, dx, density(total / volume)).unwrap();
            let mut sites = MaterialLattice::new(shape);
            if charged {
                let mut preparation = material;
                preparation.capture_rate = 1000.0;
                preparation.release_rate = 0.0;
                sites.install([0, 0, 0], preparation).unwrap();
                sites.step(&mut field, 1.0).unwrap();
                close(
                    "charged initial state",
                    sites.total_bound(),
                    material.capacity,
                    1e-12,
                );
            }
            sites.install([0, 0, 0], material).unwrap();
            (field, sites)
        };
        let initial = if charged { material.capacity } else { 0.0 };
        let mut errors = Vec::new();
        for steps in [80, 160, 320] {
            let (mut field, mut sites) = prepare();
            let mut maximum_error: f64 = 0.0;
            for step in 1..=steps {
                let balance = sites.step(&mut field, 1.0 / steps as f64).unwrap();
                maximum_error = maximum_error
                    .max((balance.bound - exact(initial, step as f64 / steps as f64)).abs());
                close("joint count", balance.manons(), total, 1e-10);
                close("joint energy", balance.energy(), 0.0, 1e-10);
                close(
                    "joint binding heat",
                    balance.thermal_joules,
                    material.binding_energy * balance.bound,
                    1e-10,
                );
                assert!((0.0..=material.capacity).contains(&balance.bound));
                assert!(balance.free >= 0.0 && balance.thermal_joules >= 0.0);
            }
            errors.push(maximum_error);
        }
        for pair in errors.windows(2) {
            assert!(
                (1.8..2.2).contains(&(pair[0] / pair[1])),
                "joint first-order errors (charged={charged}): {errors:?}"
            );
        }
        assert!(errors[2] < 0.02, "joint trajectory error: {errors:?}");
        let (mut field, mut sites) = prepare();
        for _ in 0..10_000 {
            sites.step(&mut field, 0.002).unwrap();
        }
        close("joint stable equilibrium", sites.total_bound(), lo, 0.005);
    }
}

/// LAW-THERM-001: C1*dT1/dt=-G*(T1-T2), C2*dT2/dt=G*(T1-T2).
/// DeltaT(t)=DeltaT0*exp(-G*(1/C1+1/C2)*t); the capacity-weighted
/// mean temperature is constant. Check both the discrete mode and continuum.
#[test]
fn law_two_cell_heat_conduction_matches_mode_and_converges() {
    let capacities = [2.0, 5.0];
    let initial_heat = [20.0, 4.0];
    let initial_delta = initial_heat[0] / capacities[0] - initial_heat[1] / capacities[1];
    let equilibrium = 300.0 + 24.0 / 7.0;
    let conductance = 0.7;
    let rate = conductance * (1.0 / capacities[0] + 1.0 / capacities[1]);
    for axis in 0..3 {
        for reverse in [false, true] {
            let mut dimensions = [1; 3];
            dimensions[axis] = 2;
            let shape = GridShape::new(dimensions[0], dimensions[1], dimensions[2]).unwrap();
            let mut errors = Vec::new();
            for steps in [10, 20, 40, 80] {
                let mut sites = MaterialLattice::new(shape);
                let field = FreeManaField::uniform(shape, 0.5, density(3.0)).unwrap();
                let mut storage = [[0.0; 2]; 2];
                for i in 0..2 {
                    let index = if reverse { 1 - i } else { i };
                    let mut at = [0; 3];
                    at[axis] = index;
                    let mut material = kinetic_material();
                    material.heat_capacity = capacities[i];
                    sites.install(at, material).unwrap();
                    storage[index] = [0.0, initial_heat[i]];
                }
                sites.try_replace_storage(&storage).unwrap();
                let before = sites.balance(&field).unwrap();
                let dt = 1.0 / steps as f64;
                for step in 1..=steps {
                    sites.conduct_heat(dt, conductance).unwrap();
                    let delta = initial_delta * (1.0 - rate * dt).powi(step);
                    for i in 0..2 {
                        let index = if reverse { 1 - i } else { i };
                        let expected_temperature = if i == 0 {
                            equilibrium + capacities[1] / 7.0 * delta
                        } else {
                            equilibrium - capacities[0] / 7.0 * delta
                        };
                        let cell = &sites.cells()[index];
                        close(
                            "discrete thermal mode",
                            cell.temperature().unwrap(),
                            expected_temperature,
                            1e-10,
                        );
                        assert!(cell.heat >= 0.0);
                    }
                    let balance = sites.balance(&field).unwrap();
                    close("thermal energy", balance.energy(), before.energy(), 1e-11);
                    close("thermal count", balance.manons(), before.manons(), 1e-11);
                }
                let hot = if reverse { 1 } else { 0 };
                let cold = 1 - hot;
                let actual_delta = sites.cells()[hot].temperature().unwrap()
                    - sites.cells()[cold].temperature().unwrap();
                errors.push((actual_delta - initial_delta * (-rate).exp()).abs());
            }
            for pair in errors.windows(2) {
                assert!(
                    (1.9..2.1).contains(&(pair[0] / pair[1])),
                    "thermal first-order errors: {errors:?}"
                );
            }
            assert!(errors[3] < 0.01, "thermal absolute error: {errors:?}");
        }
    }
}

/// LAW-THERM-002: a large step reaches equilibrium without overshoot;
/// absent material is an insulating gap and zero conductance blocks transfer.
#[test]
fn law_heat_limits_insulation_and_invalid_parameters() {
    let shape = GridShape::new(3, 1, 1).unwrap();
    let mut sites = MaterialLattice::new(shape);
    let mut material = kinetic_material();
    material.heat_capacity = 2.0;
    sites.install([0, 0, 0], material).unwrap();
    material.heat_capacity = 5.0;
    sites.install([2, 0, 0], material).unwrap();
    sites
        .try_replace_storage(&[[0.0, 20.0], [0.0, 0.0], [0.0, 4.0]])
        .unwrap();
    sites.conduct_heat(100.0, 100.0).unwrap();
    close("insulating gap hot", sites.cells()[0].heat, 20.0, 1e-12);
    close("insulating gap cold", sites.cells()[2].heat, 4.0, 1e-12);
    sites.install([1, 0, 0], material).unwrap();
    let snapshot: Vec<_> = sites
        .cells()
        .iter()
        .map(|cell| [cell.bound, cell.heat])
        .collect();
    sites.conduct_heat(1.0, 0.0).unwrap();
    for (dt, g) in [
        (0.0, 1.0),
        (-1.0, 1.0),
        (f64::NAN, 1.0),
        (1.0, -1.0),
        (1.0, f64::INFINITY),
    ] {
        assert!(sites.conduct_heat(dt, g).is_err());
    }
    let after: Vec<_> = sites
        .cells()
        .iter()
        .map(|cell| [cell.bound, cell.heat])
        .collect();
    assert_eq!(after, snapshot);

    let mut pair = MaterialLattice::new(GridShape::new(2, 1, 1).unwrap());
    material.heat_capacity = 2.0;
    pair.install([0, 0, 0], material).unwrap();
    material.heat_capacity = 5.0;
    pair.install([1, 0, 0], material).unwrap();
    pair.try_replace_storage(&[[0.0, 20.0], [0.0, 4.0]])
        .unwrap();
    pair.conduct_heat(100.0, 100.0).unwrap();
    close(
        "bounded equilibrium hot",
        pair.cells()[0].heat,
        24.0 * 2.0 / 7.0,
        1e-11,
    );
    close(
        "bounded equilibrium cold",
        pair.cells()[1].heat,
        24.0 * 5.0 / 7.0,
        1e-11,
    );
    // Different reference baselines cannot supply untracked donor heat.
    material.reference_temperature = 400.0;
    pair.install([0, 0, 0], material).unwrap();
    pair.try_replace_storage(&[[0.0, 0.0], [0.0, 4.0]]).unwrap();
    pair.conduct_heat(1.0, 100.0).unwrap();
    close(
        "no baseline heat extraction",
        pair.cells()[1].heat,
        4.0,
        1e-12,
    );
}

/// LAW-WORK-001: releasing R manons requires E_bind*R joules.
/// Stored heat is spent first, then explicit external work. Insufficient
/// budget limits release; it never creates energy or manons.
#[test]
fn law_discharge_work_matches_independent_energy_budget() {
    let shape = GridShape::new(1, 1, 1).unwrap();
    for (budget, flux, released, heat_used, work_used) in [
        (0.0, 1000.0, 4.0, 8.0, 0.0),
        (3.0, 1000.0, 5.5, 8.0, 3.0),
        (10.0, 1000.0, 6.0, 8.0, 4.0),
        (10.0, 2.0, 0.5, 1.0, 0.0),
    ] {
        let mut material = kinetic_material();
        material.capture_rate = 1000.0;
        let mut sites = MaterialLattice::new(shape);
        let mut field = FreeManaField::uniform(shape, 0.5, density(800.0)).unwrap();
        sites.install([0, 0, 0], material).unwrap();
        sites.step(&mut field, 1.0).unwrap();
        close("work prepared charge", sites.total_bound(), 8.0, 1e-12);
        let exported = sites.cool_into_bath(1.0, 2.0_f64.ln()).unwrap();
        close("work prepared export", exported, 8.0, 1e-12);
        material.max_flux = flux;
        sites.install([0, 0, 0], material).unwrap();
        let before = sites.balance(&field).unwrap();
        let report = sites
            .release_with_work(&mut field, [0, 0, 0], 6.0, budget, 0.25)
            .unwrap();
        close("budgeted release", report.released, released, 1e-12);
        close("heat payment", report.heat_used_joules, heat_used, 1e-12);
        close(
            "external payment",
            report.work_used_joules,
            work_used,
            1e-12,
        );
        let after = sites.balance(&field).unwrap();
        close("discharge free count", after.free, 92.0 + released, 1e-12);
        close("discharge bound count", after.bound, 8.0 - released, 1e-12);
        close(
            "discharge remaining heat",
            after.thermal_joules,
            8.0 - heat_used,
            1e-12,
        );
        close(
            "discharge inventory",
            after.manons(),
            before.manons(),
            1e-12,
        );
        close(
            "global work and export",
            after.energy() + exported,
            work_used,
            1e-12,
        );
        let storage: Vec<_> = sites
            .cells()
            .iter()
            .map(|cell| [cell.bound, cell.heat])
            .collect();
        let free = field.densities().to_vec();
        for invalid_budget in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(
                sites
                    .release_with_work(&mut field, [0, 0, 0], 1.0, invalid_budget, 0.25)
                    .is_err()
            );
            assert_eq!(field.densities(), free);
            let actual: Vec<_> = sites
                .cells()
                .iter()
                .map(|cell| [cell.bound, cell.heat])
                .collect();
            assert_eq!(actual, storage);
        }
    }
}

/// LAW-CHANNEL-001: three equal-volume reservoirs in a closed chain.
/// dn/dt = (G/V)*[-n0+n1, n0-2*n1+n2, n1-n2].
/// For n(0)=[9,3,0], n(t)=4*[1,1,1]+4.5*exp(-a*t)*[1,0,-1]
/// +0.5*exp(-3*a*t)*[1,-2,1], a=G/V. Cell 3 is disconnected.
#[test]
fn law_channel_network_converges_to_exact_closed_chain_modes() {
    let shape = GridShape::new(4, 1, 1).unwrap();
    let dx: f64 = 0.5;
    let volume = dx.powi(3);
    let conductance = 0.4;
    let area = 2.0;
    let length = 4.0;
    let rate = conductance * area / length / volume;
    let time = 0.5;
    let exact = [
        4.0 + 4.5 * (-rate * time).exp() + 0.5 * (-3.0 * rate * time).exp(),
        4.0 - (-3.0 * rate * time).exp(),
        4.0 - 4.5 * (-rate * time).exp() + 0.5 * (-3.0 * rate * time).exp(),
    ];
    let mut errors = Vec::new();
    for steps in [20, 40, 80, 160] {
        let mut lattice = MaterialLattice::new(shape);
        for (from, to) in [(0, 1), (1, 2)] {
            lattice
                .add_channel(ManaChannel {
                    from: [from, 0, 0],
                    to: [to, 0, 0],
                    length,
                    area,
                    conductance,
                    max_flux: 1000.0,
                })
                .unwrap();
        }
        let mut field = FreeManaField::uniform(shape, dx, density(0.0)).unwrap();
        for (x, value) in [9.0, 3.0, 0.0, 7.0].into_iter().enumerate() {
            field.set(x, 0, 0, density(value)).unwrap();
        }
        let before = field.total_manons().unwrap().value();
        for _ in 0..steps {
            let balance = lattice.step(&mut field, time / steps as f64).unwrap();
            close("network inventory", balance.manons(), before, 1e-12);
            close("network binding energy", balance.energy(), 0.0, 1e-12);
            close(
                "disconnected reservoir",
                field.at(3, 0, 0).unwrap().value(),
                7.0,
                1e-12,
            );
            assert!(field.densities().iter().all(|n| n.value() >= 0.0));
        }
        errors.push(
            exact
                .iter()
                .enumerate()
                .map(|(x, expected)| (field.at(x, 0, 0).unwrap().value() - expected).abs())
                .sum::<f64>(),
        );
    }
    for pair in errors.windows(2) {
        assert!(
            (1.8..2.2).contains(&(pair[0] / pair[1])),
            "channel first-order errors: {errors:?}"
        );
    }
    assert!(errors[3] < 0.03, "channel absolute error: {errors:?}");
}

/// LAW-CHANNEL-002: shared endpoints use the updated donor inventory.
/// Saturating edges split 10 into 5+5, then split the remaining 5 into 2.5+2.5.
/// Insertion order is part of the operator split, rather than simultaneous flow.
#[test]
fn law_channel_shared_donor_limits_order_and_invalid_geometry() {
    let shape = GridShape::new(3, 1, 1).unwrap();
    for reverse in [false, true] {
        let mut lattice = MaterialLattice::new(shape);
        let channel = |to| ManaChannel {
            from: [0, 0, 0],
            to: [to, 0, 0],
            length: 1.0,
            area: 1.0,
            conductance: 1000.0,
            max_flux: 1000.0,
        };
        let targets = if reverse { [2, 1] } else { [1, 2] };
        for target in targets {
            lattice.add_channel(channel(target)).unwrap();
        }
        for invalid in [
            ManaChannel {
                length: 0.0,
                ..channel(1)
            },
            ManaChannel {
                area: -1.0,
                ..channel(1)
            },
            ManaChannel {
                conductance: f64::NAN,
                ..channel(1)
            },
            ManaChannel {
                max_flux: -1.0,
                ..channel(1)
            },
            channel(0),
            channel(3),
        ] {
            assert!(lattice.add_channel(invalid).is_err());
            assert_eq!(lattice.channels().len(), 2);
        }
        let mut field = FreeManaField::uniform(shape, 1.0, density(0.0)).unwrap();
        field.set(0, 0, 0, density(10.0)).unwrap();
        let balance = lattice.step(&mut field, 1.0).unwrap();
        close(
            "shared donor",
            field.at(0, 0, 0).unwrap().value(),
            2.5,
            1e-12,
        );
        close(
            "first receiver",
            field.at(targets[0], 0, 0).unwrap().value(),
            5.0,
            1e-12,
        );
        close(
            "second receiver",
            field.at(targets[1], 0, 0).unwrap().value(),
            2.5,
            1e-12,
        );
        close("shared inventory", balance.manons(), 10.0, 1e-12);
    }
}
