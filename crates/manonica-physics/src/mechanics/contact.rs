//! Exact two-body relaxation of relative velocity with an explicit thermal channel.
use super::{MechanicalBody, values, vector};
use manonica_core::{
    Quantity, TimeStep, Vec3,
    units::{Dimensionless, Joules, KilogramMetresPerSecond, PerSecond},
};
use manonica_error::{BaseError, ErrorCode};
#[derive(Clone, Copy, Debug)]
pub struct DissipativeExchange {
    rate: Quantity<PerSecond<Dimensionless>>,
}
#[derive(Clone, Copy, Debug)]
pub struct ContactExchange {
    pub impulse_to_left: Vec3<KilogramMetresPerSecond>,
    pub heat: Quantity<Joules>,
    pub retained_relative_velocity: f64,
}
impl DissipativeExchange {
    pub fn new(rate: Quantity<PerSecond<Dimensionless>>) -> Result<Self, BaseError> {
        if rate.value() < 0.0 {
            return Err(BaseError::new(ErrorCode::InvalidRange, "contact_rate"));
        }
        Ok(Self { rate })
    }
    /// Conserves pair momentum. Lost kinetic energy goes to `heat`, never vanishes.
    /// Both bodies remain unchanged if any validation or arithmetic fails.
    pub fn exchange(
        self,
        left: &mut MechanicalBody,
        right: &mut MechanicalBody,
        dt: TimeStep,
    ) -> Result<ContactExchange, BaseError> {
        let exponent = self.rate.value() * dt.seconds();
        if !exponent.is_finite() {
            return Err(BaseError::new(ErrorCode::NonFinite, "contact_exponent"));
        }
        if exponent == 0.0 {
            return Ok(ContactExchange {
                impulse_to_left: vector([0.; 3])?,
                heat: Quantity::nonnegative(0.0)?,
                retained_relative_velocity: 1.0,
            });
        }
        let alpha = -(-exponent).exp_m1();
        let ma = left.mass().value();
        let mb = right.mass().value();
        let reduced = if ma <= mb {
            ma / (1.0 + ma / mb)
        } else {
            mb / (1.0 + mb / ma)
        };
        let pa = values(left.momentum());
        let pb = values(right.momentum());
        let relative = std::array::from_fn::<_, 3, _>(|i| pa[i] / ma - pb[i] / mb);
        let j = relative.map(|v| -reduced * alpha * v);
        let impulse = vector(j)?;
        let heat = Quantity::<Joules>::nonnegative(
            relative.iter().map(|v| 0.5 * reduced * v * v).sum::<f64>() * alpha * (2.0 - alpha),
        )?;
        let mut a = *left;
        let mut b = *right;
        a.apply_impulse(impulse)?;
        b.apply_impulse(vector(j.map(|v| -v))?)?;
        *left = a;
        *right = b;
        Ok(ContactExchange {
            impulse_to_left: impulse,
            heat,
            retained_relative_velocity: (-exponent).exp(),
        })
    }
}
