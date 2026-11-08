use manonica_core::{Evolution, Quantity, TimeStep, units::{Diffusivity, ManonCount}};
use manonica_physics::{ConservationReport, FreeDiffusion, FreeManaField};

use crate::experiment::{Experiment, ExperimentError};
use super::field::seeded_field;

/// Расчётная сессия, ничего не знающая о рендеринге.
pub struct Simulation {
    pub experiment: Experiment,
    field: FreeManaField,
    solver: FreeDiffusion,
    time_seconds: f64,
    steps: u64,
}

impl Simulation {
    pub fn new(experiment: Experiment) -> Result<Self, ExperimentError> {
        experiment.validate()?;
        let solver = FreeDiffusion::new(
            Quantity::<Diffusivity>::nonnegative(experiment.physics.diffusivity)?
        )?;
        let field = seeded_field(&experiment)?;
        Ok(Self { experiment, field, solver, time_seconds: 0.0, steps: 0 })
    }

    pub fn field(&self) -> &FreeManaField { &self.field }
    pub fn elapsed(&self) -> f64 { self.time_seconds }
    pub fn steps(&self) -> u64 { self.steps }
    pub fn total(&self) -> Result<Quantity<ManonCount>, ExperimentError> {
        Ok(self.field.total_manons()?)
    }

    pub fn step(&mut self) -> Result<ConservationReport, ExperimentError> {
        let dt = TimeStep::from_seconds(self.experiment.physics.dt)?;
        let report = self.solver.evolve(&mut self.field, dt)?;
        self.time_seconds += dt.seconds();
        self.steps += 1;
        Ok(report)
    }

    pub fn reset(&mut self) -> Result<(), ExperimentError> {
        self.field = seeded_field(&self.experiment)?;
        self.steps = 0;
        self.time_seconds = 0.0;
        Ok(())
    }
}
