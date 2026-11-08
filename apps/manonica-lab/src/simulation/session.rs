use manonica_core::{
    Evolution, Quantity, TimeStep,
    units::{Diffusivity, ManonCount},
};
use manonica_physics::{
    ConservationReport, FreeAdvection, FreeDiffusion, FreeManaField, MaterialLattice,
    PotentialField,
};

use super::{
    equilibrium::{EquilibriumMonitor, EquilibriumState},
    field::seeded_field,
};
use crate::experiment::{AmbientFlow, Experiment, ExperimentError};

/// Energy state of the externally established, quasi-static field.
#[derive(Clone, Copy, Debug)]
pub struct FieldLedger {
    pub field_energy_joules: f64,
    pub manon_potential_joules: f64,
    pub dissipated_heat_joules: f64,
    /// Work supplied by external controller after field installation.
    pub operating_work_joules: f64,
    pub last_transport_manons: f64,
    pub peak_potential: f64,
    /// Heat carried into the ambient reservoir, part of global energy audit.
    pub exported_heat_joules: f64,
    pub shutdown: bool,
}
impl FieldLedger {
    pub fn tracked_energy(self) -> f64 {
        self.field_energy_joules + self.manon_potential_joules + self.dissipated_heat_joules
    }
}

/// Расчётная сессия, ничего не знающая о рендеринге.
pub struct Simulation {
    pub experiment: Experiment,
    field: FreeManaField,
    materials: MaterialLattice,
    potential: Option<PotentialField>,
    field_ledger: Option<FieldLedger>,
    shutdown: bool,
    equilibrium: EquilibriumMonitor,
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
        let (potential, field_ledger) = build_potential(&experiment, &field)?;
        // Fail fast if the velocity is incompatible with the spatial or
        // temporal discretization. This validation must not advance the session.
        let mut probe = field.clone();
        advection.evolve(&mut probe, TimeStep::from_seconds(experiment.physics.dt)?)?;
        Ok(Self {
            experiment,
            field,
            materials,
            potential,
            field_ledger,
            shutdown: false,
            equilibrium: EquilibriumMonitor::default(),
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

    pub fn potential(&self) -> Option<&PotentialField> {
        self.potential.as_ref()
    }
    pub fn peak_trap_temperature_kelvin(&self) -> Option<f64> {
        self.materials.peak_temperature_kelvin()
    }
    pub fn equilibrium_state(&self) -> EquilibriumState {
        self.equilibrium.state()
    }
    pub fn shutdown_active(&self) -> bool {
        self.shutdown
    }
    pub fn field_ledger(&self) -> Option<FieldLedger> {
        self.field_ledger
    }
    pub fn tracked_energy(&self) -> Result<f64, ExperimentError> {
        let material = self.material_balance()?.energy();
        Ok(material
            + self
                .field_ledger
                .map_or(0.0, |p| p.tracked_energy() + p.exported_heat_joules))
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
        let mut ledger = self.field_ledger;
        let mut next_potential = self.potential.clone();
        let mut shutdown = self.shutdown;
        // Automatic power-down starts only when actual physical traps
        // reach the specified filling ratio, never at an arbitrary tick.
        if let Some(settings) = &self.experiment.physics.potential {
            let capacity = self.materials.total_capacity();
            if settings.shutdown_at_bound_fraction > 0.0
                && capacity > 0.0
                && self.materials.total_bound() / capacity >= settings.shutdown_at_bound_fraction
            {
                shutdown = true;
            }
            if shutdown {
                if let Some(potential) = next_potential.as_mut() {
                    let decay = (-settings.shutdown_decay_rate * dt.seconds()).exp();
                    *potential = potential.scaled(decay)?;
                }
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
        if let (Some(potential), Some(state), Some(settings)) = (
            &next_potential,
            ledger.as_mut(),
            self.experiment.physics.potential.as_ref(),
        ) {
            let drift = potential.drift(&mut candidate, settings.mobility, dt.seconds())?;
            state.last_transport_manons = drift.transported_manons;
        }
        let mut next_materials = self.materials.clone();
        next_materials.step(&mut candidate, dt.seconds())?;
        next_materials.conduct_heat(dt.seconds(), self.experiment.physics.thermal_conductance)?;
        let exported = next_materials
            .cool_into_bath(dt.seconds(), self.experiment.physics.ambient_cooling_rate)?;
        if let (Some(potential), Some(state), Some(settings)) = (
            &next_potential,
            ledger.as_mut(),
            self.experiment.physics.potential.as_ref(),
        ) {
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
                * (1.0 - (-self.experiment.physics.ambient_cooling_rate * dt.seconds()).exp());
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
        self.time_seconds += dt.seconds();
        self.steps += 1;
        let source_active = self.potential.is_some() && !self.shutdown;
        let potential_peak = self
            .potential
            .as_ref()
            .map_or(0.0, PotentialField::peak_absolute);
        self.equilibrium.observe(
            &self.field,
            after.value(),
            self.materials.total_bound(),
            potential_peak,
            self.materials.peak_temperature_excess_kelvin(),
            source_active,
            dt.seconds(),
        );
        Ok(ConservationReport {
            before,
            after,
            absolute_error,
        })
    }

    pub fn reset(&mut self) -> Result<(), ExperimentError> {
        let field = seeded_field(&self.experiment)?;
        let materials = build_materials(&self.experiment, &field)?;
        let (potential, field_ledger) = build_potential(&self.experiment, &field)?;
        self.potential = potential;
        self.field_ledger = field_ledger;
        self.shutdown = false;
        self.equilibrium = EquilibriumMonitor::default();
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
        let cells =
            crate::experiment::capture_cells(formula, [shape.nx(), shape.ny(), shape.nz()], dx)?;
        if cells.is_empty() {
            return Err(manonica_error::BaseError::new(
                manonica_error::ErrorCode::InvalidRange,
                "empty_formula_trap",
            )
            .into());
        }
        for cell in &cells {
            if !occupied.insert(*cell) {
                return Err(manonica_error::BaseError::new(
                    manonica_error::ErrorCode::InvalidRange,
                    "overlapping_formula_trap",
                )
                .into());
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

fn build_potential(
    experiment: &Experiment,
    field: &FreeManaField,
) -> Result<(Option<PotentialField>, Option<FieldLedger>), ExperimentError> {
    let Some(settings) = &experiment.physics.potential else {
        return Ok((None, None));
    };
    let shape = field.shape();
    let dx = field.cell_width_metres();
    let dims = [shape.nx(), shape.ny(), shape.nz()];
    let mut sources = vec![0.0_f64; shape.len()];
    for source in &experiment.field_sources {
        let formula = &experiment.formulas[source.formula];
        let cells = crate::experiment::capture_cells(formula, dims, dx)?;
        if cells.is_empty() {
            return Err(manonica_error::BaseError::new(
                manonica_error::ErrorCode::InvalidRange,
                "empty_field_source",
            )
            .into());
        }
        // Source magnitude is integrated across the whole construction, not
        // multiplied by its tessellation or number of occupied cells.
        let density = source.strength / (cells.len() as f64 * dx.powi(3));
        for [x, y, z] in cells {
            let index = shape.index(x, y, z).unwrap();
            sources[index] += density;
        }
    }
    let potential = PotentialField::solve(
        shape,
        dx,
        &sources,
        settings.permittivity,
        settings.screening,
        settings.boundary.physics(),
        settings.iterations,
    )?;
    let setup_energy = potential.field_energy_joules();
    if !setup_energy.is_finite() || setup_energy > settings.initial_work_joules + 1e-9 {
        return Err(manonica_error::BaseError::new(
            manonica_error::ErrorCode::InvalidRange,
            "field_initial_work_budget",
        )
        .into());
    }
    let ledger = FieldLedger {
        field_energy_joules: setup_energy,
        manon_potential_joules: potential.manon_potential_energy(field)?,
        dissipated_heat_joules: 0.0,
        operating_work_joules: 0.0,
        last_transport_manons: 0.0,
        peak_potential: potential.peak_absolute(),
        exported_heat_joules: 0.0,
        shutdown: false,
    };
    Ok((Some(potential), Some(ledger)))
}
