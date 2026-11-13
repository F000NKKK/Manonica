//! Configuration of the separate experimental CPU mechanics scenario.
use super::ExperimentError;
use manonica_core::{Quantity, TimeStep, Vec3};
use manonica_physics::{GaussianBody, MechanicalBody, PeriodicSpectralDomain, SpectralMechanics};
use serde::Deserialize;
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MechanicsConfig {
    pub period: [f64; 3],
    pub cutoff: [usize; 3],
    pub bodies: Vec<MechanicalBodyConfig>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MechanicalBodyConfig {
    pub mass: f64,
    pub position: [f64; 3],
    pub momentum: [f64; 3],
    pub charge: f64,
    pub width: f64,
}
fn vector<U: manonica_core::units::Unit>(
    v: [f64; 3],
) -> Result<Vec3<U>, manonica_error::BaseError> {
    Vec3::try_values(v[0], v[1], v[2])
}
impl MechanicsConfig {
    pub fn build(&self, dt: f64) -> Result<SpectralMechanics, ExperimentError> {
        use manonica_error::{BaseError, ErrorCode};
        if self.bodies.is_empty() || self.bodies.len() > 8 || self.cutoff.iter().any(|n| *n > 6) {
            return Err(BaseError::new(ErrorCode::InvalidRange, "mechanics_lab_budget").into());
        }
        if self.period.iter().any(|p| !p.is_finite() || *p > 10000.0) {
            return Err(BaseError::new(ErrorCode::InvalidRange, "mechanics_display_period").into());
        }
        let domain = PeriodicSpectralDomain::new(vector(self.period)?, self.cutoff)?;
        let mut bodies = Vec::new();
        for b in &self.bodies {
            if b.position
                .iter()
                .any(|v| !v.is_finite() || v.abs() > 10000.0)
                || b.width > 1000.0
            {
                return Err(
                    BaseError::new(ErrorCode::InvalidRange, "mechanics_display_range").into(),
                );
            }
            bodies.push(GaussianBody {
                body: MechanicalBody::new(
                    Quantity::new(b.mass)?,
                    vector(b.position)?,
                    vector(b.momentum)?,
                )?,
                charge: b.charge,
                width: b.width,
            });
        }
        let system = domain.static_equilibrium(bodies)?;
        let mut probe = system.clone();
        probe.advance(TimeStep::from_seconds(dt)?)?;
        Ok(system)
    }
}
