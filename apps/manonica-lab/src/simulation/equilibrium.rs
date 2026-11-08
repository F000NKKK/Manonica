//! Numerical equilibrium detector: a monitor, never a source of physical
//! forces. It cannot pause or change the physics solver.
use manonica_physics::FreeManaField;

const HOLD_STEPS: u32 = 40;
const DENSITY_RATE_LIMIT: f64 = 1e-4; // relative maximum change per second
const BOUND_RATE_LIMIT: f64 = 1e-4; // fraction of total manons per second
const FIELD_LIMIT: f64 = 0.01; // model potential units
const TEMPERATURE_EXCESS_LIMIT: f64 = 0.2; // kelvin

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EquilibriumPhase {
    Charging,
    Decaying,
    Relaxing,
    Equilibrium,
}

impl EquilibriumPhase {
    pub fn label(self) -> &'static str {
        match self {
            Self::Charging => "DRIVEN",
            Self::Decaying => "FIELD DECAY",
            Self::Relaxing => "RELAXING",
            Self::Equilibrium => "EQUILIBRIUM",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EquilibriumState {
    pub phase: EquilibriumPhase,
    pub density_rate: f64,
    pub bound_rate: f64,
    pub potential_peak: f64,
    pub temperature_excess_kelvin: f64,
    pub stable_steps: u32,
}
impl Default for EquilibriumState {
    fn default() -> Self {
        Self {
            phase: EquilibriumPhase::Relaxing,
            density_rate: 0.0,
            bound_rate: 0.0,
            potential_peak: 0.0,
            temperature_excess_kelvin: 0.0,
            stable_steps: 0,
        }
    }
}

#[derive(Default)]
pub struct EquilibriumMonitor {
    previous_density: Option<Vec<f64>>,
    previous_bound: Option<f64>,
    state: EquilibriumState,
}
impl EquilibriumMonitor {
    pub fn state(&self) -> EquilibriumState {
        self.state
    }

    /// Evaluate the actual latest physical state, comparing consecutive
    /// solver steps. No elapsed-time shortcut can force equilibrium.
    pub fn observe(
        &mut self,
        field: &FreeManaField,
        total_manons: f64,
        bound_manons: f64,
        peak_potential: f64,
        temperature_excess_kelvin: f64,
        powered_source: bool,
        dt: f64,
    ) {
        let sample: Vec<f64> = field
            .densities()
            .iter()
            .map(|value| value.value())
            .collect();
        let scale = sample.iter().copied().fold(1.0_f64, f64::max);
        let density_rate = self
            .previous_density
            .as_ref()
            .map_or(f64::INFINITY, |before| {
                if before.len() != sample.len() {
                    return f64::INFINITY;
                }
                before
                    .iter()
                    .zip(&sample)
                    .map(|(a, b)| (b - a).abs() / (scale * dt))
                    .fold(0.0_f64, f64::max)
            });
        let bound_rate = self.previous_bound.map_or(f64::INFINITY, |before| {
            (bound_manons - before).abs() / (total_manons.max(1.0) * dt)
        });
        self.previous_density = Some(sample);
        self.previous_bound = Some(bound_manons);
        let stable = !powered_source
            && density_rate.is_finite()
            && bound_rate.is_finite()
            && peak_potential.is_finite()
            && temperature_excess_kelvin.is_finite()
            && density_rate <= DENSITY_RATE_LIMIT
            && bound_rate <= BOUND_RATE_LIMIT
            && peak_potential <= FIELD_LIMIT
            && temperature_excess_kelvin <= TEMPERATURE_EXCESS_LIMIT;
        self.state.stable_steps = if stable {
            self.state.stable_steps.saturating_add(1)
        } else {
            0
        };
        let phase = if powered_source {
            EquilibriumPhase::Charging
        } else if peak_potential > FIELD_LIMIT {
            EquilibriumPhase::Decaying
        } else if self.state.stable_steps >= HOLD_STEPS {
            EquilibriumPhase::Equilibrium
        } else {
            EquilibriumPhase::Relaxing
        };
        self.state = EquilibriumState {
            phase,
            density_rate,
            bound_rate,
            potential_peak: peak_potential,
            temperature_excess_kelvin,
            stable_steps: self.state.stable_steps,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use manonica_core::Quantity;
    use manonica_core::units::ManonDensity;
    use manonica_physics::GridShape;

    fn uniform() -> FreeManaField {
        FreeManaField::uniform(
            GridShape::new(2, 2, 2).unwrap(),
            1.0,
            Quantity::<ManonDensity>::nonnegative(10.0).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn uniform_is_equilibrium_only_after_measured_stability_window() {
        let f = uniform();
        let mut monitor = EquilibriumMonitor::default();
        for _ in 0..HOLD_STEPS {
            monitor.observe(&f, 80.0, 0.0, 0.0, 0.0, false, 0.1);
            assert_ne!(monitor.state().phase, EquilibriumPhase::Charging);
        }
        assert_eq!(monitor.state().phase, EquilibriumPhase::Relaxing);
        monitor.observe(&f, 80.0, 0.0, 0.0, 0.0, false, 0.1);
        assert_eq!(monitor.state().phase, EquilibriumPhase::Equilibrium);
        assert_eq!(monitor.state().density_rate, 0.0);
    }

    #[test]
    fn nonzero_driving_field_prevents_false_equilibrium() {
        let f = uniform();
        let mut monitor = EquilibriumMonitor::default();
        for _ in 0..100 {
            monitor.observe(&f, 80.0, 0.0, 2.0, 0.0, false, 0.1);
        }
        assert_eq!(monitor.state().phase, EquilibriumPhase::Decaying);
        assert_eq!(monitor.state().stable_steps, 0);
        monitor.observe(&f, 80.0, 0.0, 0.0, 0.0, true, 0.1);
        assert_eq!(monitor.state().phase, EquilibriumPhase::Charging);
    }

    #[test]
    fn measurable_temperature_and_inventory_changes_reset_stability() {
        let f = uniform();
        let mut monitor = EquilibriumMonitor::default();
        for _ in 0..50 {
            monitor.observe(&f, 80.0, 0.0, 0.0, 0.0, false, 0.1);
        }
        assert_eq!(monitor.state().phase, EquilibriumPhase::Equilibrium);
        monitor.observe(&f, 80.0, 1.0, 0.0, 0.0, false, 0.1);
        assert_eq!(monitor.state().phase, EquilibriumPhase::Relaxing);
        monitor.observe(&f, 80.0, 1.0, 0.0, 0.3, false, 0.1);
        assert_eq!(monitor.state().stable_steps, 0);
    }
}
