//! Transactional resident intervals; never skip coupled physical operators.
use super::{EquilibriumObservation, Simulation};
use crate::experiment::{AmbientFlow, ExperimentError};
use manonica_core::{Quantity, TimeStep, units::ManonCount};
use manonica_physics::ConservationReport;

impl Simulation {
    /// Only a simulation with no other enabled physical interaction can
    /// safely execute multiple time steps without CPU-side material updates.
    /// This guard is about laws, not the visual presence of a formula.
    pub fn supports_resident_diffusion(&self) -> bool {
        !self.has_mechanics()
            && self.state.potential.is_none()
            && matches!(self.experiment.physics.flow, AmbientFlow::Still)
            && self.experiment.materials.is_empty()
            && self.experiment.channels.is_empty()
            && self.experiment.formulas.iter().all(|f| f.trap.is_none())
    }

    /// Resident transport is safe only without intermediate material,
    /// potential or thermal interactions. Vortex transport is not supported.
    pub fn supports_resident_transport(&self) -> bool {
        !self.has_mechanics()
            && self.state.potential.is_none()
            && matches!(
                self.experiment.physics.flow,
                AmbientFlow::Still | AmbientFlow::Uniform { .. }
            )
            && self.experiment.materials.is_empty()
            && self.experiment.channels.is_empty()
            && self.experiment.formulas.iter().all(|f| f.trap.is_none())
    }

    /// Advance a resident interval of diffusion and optional uniform flow.
    pub fn step_transport_batch(
        &mut self,
        backend: &mut dyn manonica_physics::FreeManaDiffusionBackend,
        steps: u32,
    ) -> Result<ConservationReport, ExperimentError> {
        self.step_resident_batch(backend, steps, true)
    }

    /// Runs a complete diffusion-only interval through a backend's resident
    /// batch interface; rejects scenarios requiring intermediate interactions.
    pub fn step_diffusion_batch(
        &mut self,
        backend: &mut dyn manonica_physics::FreeManaDiffusionBackend,
        steps: u32,
    ) -> Result<ConservationReport, ExperimentError> {
        self.step_resident_batch(backend, steps, false)
    }

    fn step_resident_batch(
        &mut self,
        backend: &mut dyn manonica_physics::FreeManaDiffusionBackend,
        steps: u32,
        allow_flow: bool,
    ) -> Result<ConservationReport, ExperimentError> {
        let supported = if allow_flow {
            self.supports_resident_transport()
        } else {
            self.supports_resident_diffusion()
        };
        if !supported || steps == 0 {
            return Err(manonica_error::BaseError::new(
                manonica_error::ErrorCode::InvalidRange,
                "resident_batch_unsupported_scenario",
            )
            .into());
        }
        let dt = TimeStep::from_seconds(self.experiment.physics.dt)?;
        let before = self.total()?;
        let mut candidate = self.state.field.clone();
        if let AmbientFlow::Uniform { velocity } = self.experiment.physics.flow {
            backend.evolve_uniform_flow_batch(
                &mut candidate,
                self.experiment.physics.diffusivity,
                dt.seconds(),
                velocity,
                steps,
            )?;
        } else {
            backend.evolve_diffusion_batch(
                &mut candidate,
                self.experiment.physics.diffusivity,
                dt.seconds(),
                steps,
            )?;
        }
        let after = candidate.total_manons()?;
        let error = (after.value() - before.value()).abs();
        if !error.is_finite() || error > 1e-8 * before.value().max(1.0) {
            return Err(manonica_error::BaseError::new(
                manonica_error::ErrorCode::NumericalInstability,
                "resident_diffusion_conservation",
            )
            .into());
        }
        self.state.field = candidate;
        self.time_seconds += dt.seconds() * steps as f64;
        self.steps += u64::from(steps);
        // A batch provides one real observation; it must not manufacture
        // the 40 distinct observations required by the equilibrium monitor.
        self.equilibrium.observe(&self.state.field, EquilibriumObservation {
            total_manons: after.value(),
            bound_manons: 0.0,
            peak_potential: 0.0,
            temperature_excess_kelvin: 0.0,
            powered_source: matches!(self.experiment.physics.flow, AmbientFlow::Uniform { velocity } if velocity.iter().any(|v| *v != 0.0)),
            dt: dt.seconds() * steps as f64,
        });
        Ok(ConservationReport {
            before,
            after,
            absolute_error: Quantity::<ManonCount>::nonnegative(error)?,
        })
    }
}
