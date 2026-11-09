//! Independent f64 CPU operator split used as the numerical reference.
use crate::{CoupledState, StepParameters};
use manonica_core::{
    Evolution, Quantity, TimeStep,
    units::{Diffusivity, ManonCount},
};
use manonica_error::BaseError;
use manonica_physics::{
    ConservationReport, FreeAdvection, FreeDiffusion, FreeManaDiffusionBackend, PhysicsError,
};

impl CoupledState {
    pub fn step(
        &mut self,
        parameters: &StepParameters,
    ) -> Result<ConservationReport, PhysicsError> {
        self.step_with_diffusion(parameters, None)
    }
    pub fn step_with_diffusion(
        &mut self,
        parameters: &StepParameters,
        backend: Option<&mut dyn FreeManaDiffusionBackend>,
    ) -> Result<ConservationReport, PhysicsError> {
        parameters.validate()?;
        if self.potential.is_some() != parameters.field.is_some()
            || self.potential.is_some() != self.field_ledger.is_some()
        {
            return Err(BaseError::new(
                manonica_error::ErrorCode::InvalidState,
                "coupled_potential_state",
            )
            .into());
        }
        let dt = TimeStep::from_seconds(parameters.dt)?;
        let before =
            Quantity::<ManonCount>::nonnegative(self.materials.balance(&self.field)?.manons())?;
        let mut candidate = self.field.clone();
        let mut fused = false;
        if let Some(backend) = backend {
            if let crate::MediumFlow::Uniform { velocity } = parameters.flow {
                fused = backend.evolve_diffusion_and_uniform_flow(
                    &mut candidate,
                    parameters.diffusivity,
                    parameters.dt,
                    velocity,
                )?;
            }
            if !fused {
                backend.evolve_free_diffusion(
                    &mut candidate,
                    parameters.diffusivity,
                    parameters.dt,
                )?;
            }
        } else {
            FreeDiffusion::new(Quantity::<Diffusivity>::nonnegative(
                parameters.diffusivity,
            )?)?
            .evolve(&mut candidate, dt)?;
        }
        if !fused {
            FreeAdvection::new(parameters.flow.clone()).evolve(&mut candidate, dt)?;
        }
        let mut ledger = self.field_ledger;
        let mut next_potential = self.potential.clone();
        let mut shutdown = self.shutdown;
        // Automatic power-down starts only when actual physical traps
        // reach the specified filling ratio, never at an arbitrary tick.
        if let Some(settings) = &parameters.field {
            let capacity = self.materials.total_capacity();
            if settings.shutdown_at_bound_fraction > 0.0
                && capacity > 0.0
                && self.materials.total_bound() / capacity >= settings.shutdown_at_bound_fraction
            {
                shutdown = true;
            }
            if shutdown && let Some(potential) = next_potential.as_mut() {
                let decay = (-settings.shutdown_decay_rate * dt.seconds()).exp();
                *potential = potential.scaled(decay)?;
            }
        }
        let old_energy = match &self.potential {
            Some(potential) => {
                potential.manon_potential_energy(&self.field)?
                    + self.materials.bound_potential_energy(potential.values())?
                    + potential.field_energy_joules()
            }
            None => 0.0,
        };
        if let (Some(potential), Some(state), Some(settings)) =
            (&next_potential, ledger.as_mut(), parameters.field.as_ref())
        {
            let drift = potential.drift(&mut candidate, settings.mobility, dt.seconds())?;
            state.last_transport_manons = drift.transported_manons;
        }
        let mut next_materials = self.materials.clone();
        next_materials.step(&mut candidate, dt.seconds())?;
        next_materials.conduct_heat(dt.seconds(), parameters.thermal_conductance)?;
        let exported =
            next_materials.cool_into_bath(dt.seconds(), parameters.ambient_cooling_rate)?;
        if let (Some(potential), Some(state), Some(settings)) =
            (&next_potential, ledger.as_mut(), parameters.field.as_ref())
        {
            let mobile_energy = potential.manon_potential_energy(&candidate)?;
            let bound_energy = next_materials.bound_potential_energy(potential.values())?;
            let stored_field = potential.field_energy_joules();
            let new_energy = mobile_energy + bound_energy + stored_field;
            let change = new_energy - old_energy;
            if !change.is_finite() {
                return Err(manonica_error::BaseError::new(
                    manonica_error::ErrorCode::InvalidRange,
                    "potential_energy_overflow",
                )
                .into());
            }
            if change < 0.0 {
                state.dissipated_heat_joules += -change;
            } else {
                let heat_used = change.min(state.dissipated_heat_joules);
                let external = (change - heat_used).max(0.0);
                if external > settings.max_power_joules_per_second * dt.seconds() + 1e-10 {
                    return Err(manonica_error::BaseError::new(
                        manonica_error::ErrorCode::InvalidRange,
                        "field_external_power",
                    )
                    .into());
                }
                state.dissipated_heat_joules -= heat_used;
                state.operating_work_joules += external;
            }
            state.manon_potential_joules = mobile_energy + bound_energy;
            state.field_energy_joules = stored_field;
            state.peak_potential = potential.peak_absolute();
            state.shutdown = shutdown;
            state.exported_heat_joules += exported;
            // Field heat may leave to the environment, but does not vanish
            // from the global closed-system energy audit.
            let heat_loss = state.dissipated_heat_joules
                * (1.0 - (-parameters.ambient_cooling_rate * dt.seconds()).exp());
            state.dissipated_heat_joules -= heat_loss;
            state.exported_heat_joules += heat_loss;
        }
        let after =
            Quantity::<ManonCount>::nonnegative(next_materials.balance(&candidate)?.manons())?;
        let absolute_error =
            Quantity::<ManonCount>::nonnegative((after.value() - before.value()).abs())?;
        // Diffusion + advection form one transactional operator-split step.
        self.field = candidate;
        self.materials = next_materials;
        self.field_ledger = ledger;
        self.potential = next_potential;
        self.shutdown = shutdown;
        Ok(ConservationReport {
            before,
            after,
            absolute_error,
        })
    }
}
