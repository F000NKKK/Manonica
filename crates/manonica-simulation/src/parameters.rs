use crate::MediumFlow;
use manonica_error::{BaseError, ErrorCode};

#[derive(Clone, Copy, Debug)]
pub struct FieldControl {
    pub mobility: f64,
    pub max_power_joules_per_second: f64,
    pub shutdown_at_bound_fraction: f64,
    pub shutdown_decay_rate: f64,
}
#[derive(Clone, Debug)]
pub struct StepParameters {
    pub diffusivity: f64,
    pub dt: f64,
    pub flow: MediumFlow,
    pub thermal_conductance: f64,
    pub ambient_cooling_rate: f64,
    pub field: Option<FieldControl>,
}
impl StepParameters {
    pub fn validate(&self) -> Result<(), BaseError> {
        let mut values = vec![
            self.diffusivity,
            self.thermal_conductance,
            self.ambient_cooling_rate,
        ];
        if let Some(f) = self.field {
            values.extend([
                f.mobility,
                f.max_power_joules_per_second,
                f.shutdown_at_bound_fraction,
                f.shutdown_decay_rate,
            ]);
            if f.shutdown_at_bound_fraction > 1.0 {
                return Err(BaseError::new(ErrorCode::InvalidRange, "shutdown_fraction"));
            }
        }
        if !self.dt.is_finite()
            || self.dt <= 0.0
            || values.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(BaseError::new(
                ErrorCode::InvalidRange,
                "coupled_parameters",
            ));
        }
        Ok(())
    }
}
