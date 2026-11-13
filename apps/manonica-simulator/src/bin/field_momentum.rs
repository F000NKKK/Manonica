//! Experimental two-mode dynamic scalar field; see docs/mechanical-momentum.md.
//! This stand is deliberately separate from the quasi-static Lab solver.
use std::f64::consts::TAU;

#[derive(Clone, Copy, Debug)]
struct State {
    x: f64,
    p: f64,
    a: f64,
    b: f64,
    pa: f64,
    pb: f64,
}
#[derive(Clone, Copy)]
struct Model {
    mass: f64,
    inertia: f64,
    omega: f64,
    wave_number: f64,
    charge: f64,
}
impl Model {
    fn energy(self, s: State) -> f64 {
        s.p * s.p / (2.0 * self.mass)
            + (s.pa * s.pa + s.pb * s.pb) / (2.0 * self.inertia)
            + 0.5 * self.inertia * self.omega * self.omega * (s.a * s.a + s.b * s.b)
            + self.charge
                * (s.a * (self.wave_number * s.x).cos() + s.b * (self.wave_number * s.x).sin())
    }
    fn momentum(self, s: State) -> f64 {
        s.p + self.wave_number * (s.a * s.pb - s.b * s.pa)
    }
    fn kick(self, s: &mut State, dt: f64) {
        let (sin, cos) = (self.wave_number * s.x).sin_cos();
        s.p += dt * self.charge * self.wave_number * (s.a * sin - s.b * cos);
        s.pa -= dt * (self.inertia * self.omega * self.omega * s.a + self.charge * cos);
        s.pb -= dt * (self.inertia * self.omega * self.omega * s.b + self.charge * sin);
    }
    fn step(self, s: &mut State, dt: f64) {
        self.kick(s, dt / 2.0);
        s.x += dt * s.p / self.mass;
        s.a += dt * s.pa / self.inertia;
        s.b += dt * s.pb / self.inertia;
        self.kick(s, dt / 2.0);
    }
}
fn setup() -> (Model, State) {
    (
        Model {
            mass: 2.0,
            inertia: 1.0,
            omega: 1.3,
            wave_number: 1.0,
            charge: 0.4,
        },
        State {
            x: 0.4,
            p: 0.2,
            a: 0.7,
            b: -0.2,
            pa: 0.1,
            pb: 0.3,
        },
    )
}

fn experiment(steps: usize) -> Result<(f64, f64), String> {
    let (model, mut state) = setup();
    let initial_energy = model.energy(state);
    let initial_momentum = model.momentum(state);
    let mut energy_error: f64 = 0.0;
    let mut momentum_error: f64 = 0.0;
    for _ in 0..steps {
        model.step(&mut state, 5.0 / steps as f64);
        let e = (model.energy(state) - initial_energy).abs();
        let p = (model.momentum(state) - initial_momentum).abs();
        if !e.is_finite() || !p.is_finite() {
            return Err("nonfinite state".into());
        }
        energy_error = energy_error.max(e);
        momentum_error = momentum_error.max(p);
    }
    if momentum_error > 1e-11 {
        return Err(format!("momentum error: {momentum_error}"));
    }
    Ok((energy_error, momentum_error))
}

fn validate() -> Result<(), String> {
    let mut previous: Option<f64> = None;
    for steps in [100, 200, 400] {
        let (energy, momentum) = experiment(steps)?;
        if let Some(error) = previous {
            let ratio = error / energy;
            if !(3.8..4.2).contains(&ratio) {
                return Err(format!("energy refinement ratio: {ratio}"));
            }
        }
        if steps == 400 && energy > 0.001 {
            return Err(format!("energy error: {energy}"));
        }
        previous = Some(energy);
        println!(
            "steps={steps}, max energy error={energy:.6e} J, max total momentum error={momentum:.6e} kg m/s"
        );
    }
    Ok(())
}
fn main() -> Result<(), String> {
    println!("Experimental dynamic scalar field: two modes, one body, periodic length {TAU:.6} m");
    validate()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual}, expected={expected}"
        );
    }
    #[test]
    fn law_mech_001_internal_force_has_equal_field_recoil() {
        let (model, mut s) = setup();
        // x=0, A=0, B=2 -> body force=-q*k*B=-0.8 N.
        s.x = 0.0;
        s.a = 0.0;
        s.b = 2.0;
        let before = s;
        model.kick(&mut s, 0.25);
        close(s.p - before.p, -0.2, 1e-14);
        let field_before = model.momentum(before) - before.p;
        let field_after = model.momentum(s) - s.p;
        close(field_after - field_before, 0.2, 1e-14);
        close(model.momentum(s), model.momentum(before), 1e-14);
    }
    #[test]
    fn law_mech_002_free_body_and_uncoupled_field_match_analytic_solution() {
        let (mut model, initial) = setup();
        model.charge = 0.0;
        let mut errors = Vec::new();
        for steps in [100, 200, 400] {
            let mut s = initial;
            for _ in 0..steps {
                model.step(&mut s, 1.0 / steps as f64);
            }
            close(s.x, initial.x + initial.p / model.mass, 1e-12);
            close(s.p, initial.p, 1e-14);
            let (sin, cos) = model.omega.sin_cos();
            let exact_a = initial.a * cos + initial.pa / (model.inertia * model.omega) * sin;
            let exact_b = initial.b * cos + initial.pb / (model.inertia * model.omega) * sin;
            let exact_pa = initial.pa * cos - model.inertia * model.omega * initial.a * sin;
            let exact_pb = initial.pb * cos - model.inertia * model.omega * initial.b * sin;
            errors.push(
                (s.a - exact_a).abs()
                    + (s.b - exact_b).abs()
                    + (s.pa - exact_pa).abs()
                    + (s.pb - exact_pb).abs(),
            );
        }
        for pair in errors.windows(2) {
            assert!((3.8..4.2).contains(&(pair[0] / pair[1])));
        }
        assert!(errors[2] < 1e-5);
    }
    #[test]
    fn law_mech_003_coupled_energy_converges_and_total_momentum_is_conserved() {
        validate().unwrap();
    }
    #[test]
    fn law_mech_004_translation_covariance_and_time_reversal() {
        let (model, initial) = setup();
        let delta = 0.73;
        let (sin, cos) = (model.wave_number * delta).sin_cos();
        let translate = |s: State| State {
            x: s.x + delta,
            a: s.a * cos - s.b * sin,
            b: s.a * sin + s.b * cos,
            pa: s.pa * cos - s.pb * sin,
            pb: s.pa * sin + s.pb * cos,
            ..s
        };
        let mut original = initial;
        let mut shifted = translate(initial);
        close(model.energy(original), model.energy(shifted), 1e-14);
        for _ in 0..100 {
            model.step(&mut original, 0.01);
            model.step(&mut shifted, 0.01);
        }
        let expected = translate(original);
        for (a, b) in [
            (shifted.x, expected.x),
            (shifted.p, expected.p),
            (shifted.a, expected.a),
            (shifted.b, expected.b),
            (shifted.pa, expected.pa),
            (shifted.pb, expected.pb),
        ] {
            close(a, b, 1e-12);
        }
        for _ in 0..100 {
            model.step(&mut original, -0.01);
        }
        for (a, b) in [
            (original.x, initial.x),
            (original.p, initial.p),
            (original.a, initial.a),
            (original.b, initial.b),
            (original.pa, initial.pa),
            (original.pb, initial.pb),
        ] {
            close(a, b, 1e-12);
        }
    }
}
