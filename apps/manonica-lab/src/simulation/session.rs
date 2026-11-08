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

    /// Bound manon inventories for each geometric formula, in experiment order.
    /// The same discrete cells are used by the solver and scene geometry.
    pub fn formula_bound(&self) -> Result<Vec<f64>, ExperimentError> {
        let shape = self.field.shape();
        let dimensions = [shape.nx(), shape.ny(), shape.nz()];
        let dx = self.field.cell_width_metres();
        self.experiment
            .formulas
            .iter()
            .map(|formula| {
                if formula.trap.is_none() {
                    return Ok(0.0);
                }
                let cells = crate::experiment::capture_cells(formula, dimensions, dx)?;
                Ok(cells
                    .into_iter()
                    .filter_map(|c| self.materials.at(c[0], c[1], c[2]))
                    .map(|cell| cell.bound)
                    .sum())
            })
            .collect()
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
        next_materials.conduct_heat(dt.seconds(), self.experiment.physics.thermal_conductance)?;
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
    let mut occupied = std::collections::HashSet::new();
    for site in &experiment.materials {
        occupied.insert(site.cell);
        materials.install(site.cell, site.physics())?;
    }
    // The physical trap lies on the parametric geometry, not at a
    // decorative hard-coded centre. Capacity and maximum flux are divided
    // across distinct touched cells so sampling resolution cannot multiply
    // the total resource.
    let shape = field.shape();
    let dx = field.cell_width_metres();
    for formula in &experiment.formulas {
        let Some(trap) = &formula.trap else {
            continue;
        };
        let cells = crate::experiment::capture_cells(
            formula,
            [shape.nx(), shape.ny(), shape.nz()],
            dx,
        )?;
        if cells.is_empty() {
            return Err(manonica_error::BaseError::new(
                manonica_error::ErrorCode::InvalidRange, "empty_formula_trap"
            ).into());
        }
        for cell in &cells {
            if !occupied.insert(*cell) {
                return Err(manonica_error::BaseError::new(
                    manonica_error::ErrorCode::InvalidRange, "overlapping_formula_trap"
                ).into());
            }
        }
        let material = trap.material(cells.len());
        material.validate()?;
        for cell in cells {
            materials.install(cell, material)?;
        }
    }
    for channel in &experiment.channels {
        materials.add_channel(channel.physics())?;
    }
    Ok(materials)
}
