use manonica_core::{
    Evolution, Quantity, TimeStep,
    units::{Diffusivity, ManonCount},
};
use manonica_physics::{
    ConservationReport, FreeAdvection, FreeDiffusion, FreeManaField, MaterialLattice,
};

use super::field::seeded_field;
use crate::experiment::{AmbientFlow, Experiment, ExperimentError};

/// Расчётная сессия, ничего не знающая о рендеринге.
pub struct Simulation {
    pub experiment: Experiment,
    field: FreeManaField,
    materials: MaterialLattice,
    solver: FreeDiffusion,
    advection: FreeAdvection<AmbientFlow>,
    time_seconds: f64,
    steps: u64,
}

impl Simulation {
    pub fn new(experiment: Experiment) -> Result<Self, ExperimentError> {
        experiment.validate()?;
        let solver = FreeDiffusion::new(Quantity::<Diffusivity>::nonnegative(
            experiment.physics.diffusivity,
        )?)?;
        let field = seeded_field(&experiment)?;
        let advection = FreeAdvection::new(experiment.physics.flow.clone());
        let materials = build_materials(&experiment, &field)?;
        // Fail fast if the velocity is incompatible with the spatial or
        // temporal discretization. This validation must not advance the session.
        let mut probe = field.clone();
        advection.evolve(&mut probe, TimeStep::from_seconds(experiment.physics.dt)?)?;
        Ok(Self {
            experiment,
            field,
            materials,
            solver,
            advection,
            time_seconds: 0.0,
            steps: 0,
        })
    }

    pub fn field(&self) -> &FreeManaField {
        &self.field
    }
    /// Total free + bound inventory and the closed thermal/binding ledger.
    pub fn material_balance(&self) -> Result<manonica_physics::Balance, ExperimentError> {
        Ok(self.materials.balance(&self.field)?)
    }

    pub fn elapsed(&self) -> f64 {
        self.time_seconds
    }
    pub fn steps(&self) -> u64 {
        self.steps
    }
    pub fn total(&self) -> Result<Quantity<ManonCount>, ExperimentError> {
        Ok(Quantity::nonnegative(
            self.materials.balance(&self.field)?.manons(),
        )?)
    }

    pub fn step(&mut self) -> Result<ConservationReport, ExperimentError> {
        let dt = TimeStep::from_seconds(self.experiment.physics.dt)?;
        let before = self.total()?;
        let mut candidate = self.field.clone();
        self.solver.evolve(&mut candidate, dt)?;
        self.advection.evolve(&mut candidate, dt)?;
        let mut next_materials = self.materials.clone();
        next_materials.step(&mut candidate, dt.seconds())?;
        let after =
            Quantity::<ManonCount>::nonnegative(next_materials.balance(&candidate)?.manons())?;
        let absolute_error =
            Quantity::<ManonCount>::nonnegative((after.value() - before.value()).abs())?;
        // Diffusion + advection form one transactional operator-split step.
        self.field = candidate;
        self.materials = next_materials;
        self.time_seconds += dt.seconds();
        self.steps += 1;
        Ok(ConservationReport {
            before,
            after,
            absolute_error,
        })
    }

    pub fn reset(&mut self) -> Result<(), ExperimentError> {
        let field = seeded_field(&self.experiment)?;
        let materials = build_materials(&self.experiment, &field)?;
        self.field = field;
        self.materials = materials;
        self.steps = 0;
        self.time_seconds = 0.0;
        Ok(())
    }
}

fn build_materials(
    experiment: &Experiment,
    field: &FreeManaField,
) -> Result<MaterialLattice, ExperimentError> {
    let mut materials = MaterialLattice::new(field.shape());
    for site in &experiment.materials {
        materials.install(site.cell, site.physics())?;
    }
    for channel in &experiment.channels {
        materials.add_channel(channel.physics())?;
    }
    Ok(materials)
}
