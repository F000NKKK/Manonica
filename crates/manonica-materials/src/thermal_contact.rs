//! Exact passive two-reservoir thermal exchange about a common reference bath.
use manonica_error::{BaseError, ErrorCode};
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThermalReservoir {
    pub heat_joules: f64,
    pub capacity_joules_per_kelvin: f64,
}
impl ThermalReservoir {
    pub fn temperature_kelvin(self, reference_kelvin: f64) -> Result<f64, BaseError> {
        self.validate()?;
        if !reference_kelvin.is_finite() || reference_kelvin <= 0.0 {
            return Err(invalid("thermal_reference"));
        }
        let t = reference_kelvin + self.heat_joules / self.capacity_joules_per_kelvin;
        if !t.is_finite() {
            return Err(invalid("thermal_temperature"));
        }
        Ok(t)
    }
    fn validate(self) -> Result<(), BaseError> {
        if !self.heat_joules.is_finite()
            || self.heat_joules < 0.0
            || !self.capacity_joules_per_kelvin.is_finite()
            || self.capacity_joules_per_kelvin <= 0.0
        {
            return Err(invalid("thermal_reservoir"));
        }
        Ok(())
    }
}
fn invalid(p: &'static str) -> BaseError {
    BaseError::new(ErrorCode::InvalidRange, p)
}
/// Returns signed heat delivered to the left. No overshoot or hidden thermostat.
pub fn exchange_heat_exact(
    left: &mut ThermalReservoir,
    right: &mut ThermalReservoir,
    conductance_watts_per_kelvin: f64,
    dt_seconds: f64,
) -> Result<f64, BaseError> {
    left.validate()?;
    right.validate()?;
    if !conductance_watts_per_kelvin.is_finite()
        || conductance_watts_per_kelvin < 0.0
        || !dt_seconds.is_finite()
        || dt_seconds <= 0.0
    {
        return Err(invalid("thermal_contact"));
    }
    let ca = left.capacity_joules_per_kelvin;
    let cb = right.capacity_joules_per_kelvin;
    let reduced = if ca <= cb {
        ca / (1.0 + ca / cb)
    } else {
        cb / (1.0 + cb / ca)
    };
    let exponent = conductance_watts_per_kelvin * dt_seconds / reduced;
    if !exponent.is_finite() {
        return Err(invalid("thermal_contact_exponent"));
    }
    let difference = right.heat_joules / cb - left.heat_joules / ca;
    let mut q = difference * reduced * (-(-exponent).exp_m1());
    q = q.clamp(-left.heat_joules, right.heat_joules);
    let a = ThermalReservoir {
        heat_joules: left.heat_joules + q,
        ..*left
    };
    let b = ThermalReservoir {
        heat_joules: right.heat_joules - q,
        ..*right
    };
    a.validate()?;
    b.validate()?;
    *left = a;
    *right = b;
    Ok(q)
}
/// Optically thin local exchange with a fixed radiative bath; escaped energy is returned.
pub fn radiate_to_bath(
    reservoir: &mut ThermalReservoir,
    reference_kelvin: f64,
    emissivity: f64,
    area_square_metres: f64,
    dt_seconds: f64,
) -> Result<f64, BaseError> {
    let t = reservoir.temperature_kelvin(reference_kelvin)?;
    if !emissivity.is_finite()
        || !(0.0..=1.0).contains(&emissivity)
        || !area_square_metres.is_finite()
        || area_square_metres < 0.0
        || !dt_seconds.is_finite()
        || dt_seconds <= 0.0
    {
        return Err(invalid("radiative_contact"));
    }
    let power =
        emissivity * 5.670374419e-8 * area_square_metres * (t.powi(4) - reference_kelvin.powi(4));
    if !power.is_finite() {
        return Err(invalid("radiative_power"));
    }
    let amount = (power * dt_seconds).min(reservoir.heat_joules);
    if !amount.is_finite() || amount < 0.0 {
        return Err(invalid("radiative_transfer"));
    }
    reservoir.heat_joules -= amount;
    Ok(amount)
}
