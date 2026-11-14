//! Shared thermal parameters and sparse medium cells; independent of spell programs.
use crate::{WorldError, WorldResult};
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PressureResponse {
    pub mobility: f64,
    pub maximum_speed_metres_per_second: f64,
    pub adiabatic_index: f64,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediumThermodynamics {
    pub reference_temperature_kelvin: f64,
    pub heat_capacity_joules_per_cubic_metre_kelvin: f64,
    pub contact_watts_per_kelvin: f64,
    pub emissivity: f64,
    pub pressure: PressureResponse,
}
impl Default for MediumThermodynamics {
    fn default() -> Self {
        Self {
            reference_temperature_kelvin: 300.,
            heat_capacity_joules_per_cubic_metre_kelvin: 10.,
            contact_watts_per_kelvin: 100.,
            emissivity: 0.005,
            pressure: PressureResponse {
                mobility: 0.002,
                maximum_speed_metres_per_second: 20.,
                adiabatic_index: 5. / 3.,
            },
        }
    }
}
impl MediumThermodynamics {
    pub(crate) fn validate(self) -> WorldResult<()> {
        if [
            self.reference_temperature_kelvin,
            self.heat_capacity_joules_per_cubic_metre_kelvin,
            self.pressure.maximum_speed_metres_per_second,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.0)
            || [self.contact_watts_per_kelvin, self.pressure.mobility]
                .iter()
                .any(|v| !v.is_finite() || *v < 0.)
            || !self.emissivity.is_finite()
            || !(0.0..=1.).contains(&self.emissivity)
            || !self.pressure.adiabatic_index.is_finite()
            || !(1.0..=2.).contains(&self.pressure.adiabatic_index)
        {
            return Err(WorldError::Invalid("medium_thermodynamics"));
        }
        Ok(())
    }
    pub fn temperature(self, heat_joules: f64, volume: f64) -> f64 {
        self.reference_temperature_kelvin
            + heat_joules / (self.heat_capacity_joules_per_cubic_metre_kelvin * volume)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalCell {
    pub position: [i64; 3],
    pub heat_joules: f64,
}
