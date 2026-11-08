use manonica_core::{Evolution, Quantity, TimeStep, units::{Diffusivity, ManonCount}};
use manonica_physics::{ConservationReport, FreeAdvection, FreeDiffusion, FreeManaField};

use crate::experiment::{AmbientFlow, Experiment, ExperimentError};
use super::field::seeded_field;

/// Расчётная сессия, ничего не знающая о рендеринге.
pub struct Simulation {
    pub experiment: Experiment,
    field: FreeManaField,
    solver: FreeDiffusion,
    advection: FreeAdvection<AmbientFlow>,
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
        let advection = FreeAdvection::new(experiment.physics.flow.clone());
        // Fail fast if the velocity is incompatible with the spatial or
        // temporal discretization. This validation must not advance the session.
        let mut probe = field.clone();
        advection.evolve(&mut probe, TimeStep::from_seconds(experiment.physics.dt)?)?;
        Ok(Self { experiment, field, solver, advection, time_seconds: 0.0, steps: 0 })
    }

    pub fn field(&self) -> &FreeManaField { &self.field }
    pub fn elapsed(&self) -> f64 { self.time_seconds }
    pub fn steps(&self) -> u64 { self.steps }
    pub fn total(&self) -> Result<Quantity<ManonCount>, ExperimentError> {
        Ok(self.field.total_manons()?)
    }

    pub fn step(&mut self) -> Result<ConservationReport, ExperimentError> {
        let dt = TimeStep::from_seconds(self.experiment.physics.dt)?;
        let before = self.field.total_manons()?;
        let mut candidate = self.field.clone();
        self.solver.evolve(&mut candidate, dt)?;
        self.advection.evolve(&mut candidate, dt)?;
        let after = candidate.total_manons()?;
        let absolute_error = Quantity::<ManonCount>::nonnegative(
            (after.value() - before.value()).abs(),
        )?;
        // Diffusion + advection form one transactional operator-split step.
        self.field = candidate;
        self.time_seconds += dt.seconds();
        self.steps += 1;
        Ok(ConservationReport { before, after, absolute_error })
    }

    pub fn reset(&mut self) -> Result<(), ExperimentError> {
        self.field = seeded_field(&self.experiment)?;
        self.steps = 0;
        self.time_seconds = 0.0;
        Ok(())
    }
}
