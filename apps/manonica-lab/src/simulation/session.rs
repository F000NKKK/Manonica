mod resident;

use manonica_core::{Evolution, Quantity, TimeStep, units::ManonCount};
use manonica_physics::{
    ConservationReport, FreeAdvection, FreeManaField, LocalBoundarySite, PotentialField,
};

use super::{
    equilibrium::{EquilibriumMonitor, EquilibriumObservation, EquilibriumState},
    field::seeded_field,
};
use crate::experiment::{AmbientFlow, Experiment, ExperimentError};
use manonica_materials::MaterialLattice;

pub use manonica_simulation::FieldLedger;
use manonica_simulation::{CoupledState, FieldControl, StepParameters};

/// Расчётная сессия, ничего не знающая о рендеринге.
pub struct Simulation {
    pub experiment: Experiment,
    state: CoupledState,
    mechanics: Option<super::mechanics::MechanicsSession>,
    equilibrium: EquilibriumMonitor,
    time_seconds: f64,
    steps: u64,
    pub(crate) revision: u64,
}

impl Simulation {
    pub fn new(experiment: Experiment) -> Result<Self, ExperimentError> {
        Self::new_cancellable(experiment, &|| false)
    }
    pub(crate) fn new_cancellable(
        experiment: Experiment,
        should_cancel: &impl Fn() -> bool,
    ) -> Result<Self, ExperimentError> {
        if should_cancel() {
            return Err(manonica_error::BaseError::new(
                manonica_error::ErrorCode::InvalidState,
                "experiment_cancelled",
            )
            .into());
        }
        experiment.validate()?;
        let field = seeded_field(&experiment)?;
        let advection = FreeAdvection::new(experiment.physics.flow.clone());
        let materials = build_materials(&experiment, &field)?;
        let (potential, field_ledger) =
            build_potential_cancellable(&experiment, &field, should_cancel)?;
        // Fail fast if the velocity is incompatible with the spatial or
        // temporal discretization. This validation must not advance the session.
        let mut probe = field.clone();
        advection.evolve(&mut probe, TimeStep::from_seconds(experiment.physics.dt)?)?;
        let mechanics = experiment
            .mechanics
            .as_ref()
            .map(|config| super::mechanics::MechanicsSession::new(config, experiment.physics.dt))
            .transpose()?;
        if should_cancel() {
            return Err(manonica_error::BaseError::new(
                manonica_error::ErrorCode::InvalidState,
                "experiment_cancelled",
            )
            .into());
        }
        Ok(Self {
            mechanics,
            experiment,
            state: CoupledState {
                field,
                materials,
                potential,
                field_ledger,
                shutdown: false,
            },
            equilibrium: EquilibriumMonitor::default(),
            time_seconds: 0.0,
            steps: 0,
            revision: 0,
        })
    }

    pub fn has_mechanics(&self) -> bool {
        self.mechanics.is_some()
    }
    pub fn mechanics_observation(&self) -> Option<super::MechanicsObservation> {
        self.mechanics.as_ref().map(|m| m.observation())
    }

    pub fn field(&self) -> &FreeManaField {
        &self.state.field
    }
    /// Total free + bound inventory and the closed thermal/binding ledger.
    pub fn material_balance(&self) -> Result<manonica_materials::Balance, ExperimentError> {
        Ok(self.state.materials.balance(&self.state.field)?)
    }

    /// Bound manon inventories for each geometric formula, in experiment order.
    /// The same discrete cells are used by the solver and scene geometry.
    pub fn formula_bound(&self) -> Result<Vec<f64>, ExperimentError> {
        let shape = self.state.field.shape();
        let dimensions = [shape.nx(), shape.ny(), shape.nz()];
        let dx = self.state.field.cell_width_metres();
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
                    .filter_map(|c| self.state.materials.at(c[0], c[1], c[2]))
                    .map(|cell| cell.bound)
                    .sum())
            })
            .collect()
    }

    pub fn potential(&self) -> Option<&PotentialField> {
        self.state.potential.as_ref()
    }
    pub fn peak_trap_temperature_kelvin(&self) -> Option<f64> {
        self.state.materials.peak_temperature_kelvin()
    }
    pub fn equilibrium_state(&self) -> EquilibriumState {
        self.equilibrium.state()
    }
    pub fn shutdown_active(&self) -> bool {
        self.state.shutdown
    }
    pub fn field_ledger(&self) -> Option<FieldLedger> {
        self.state.field_ledger
    }
    pub fn tracked_energy(&self) -> Result<f64, ExperimentError> {
        let material = self.material_balance()?.energy();
        Ok(material
            + self
                .state
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
            self.state.materials.balance(&self.state.field)?.manons(),
        )?)
    }

    pub fn step(&mut self) -> Result<ConservationReport, ExperimentError> {
        self.step_internal(None)
    }

    /// Run with any physical diffusion backend (CPU reference, WGPU, or
    /// future Minecraft/server acceleration), independently of the Lab.
    pub fn step_with_backend(
        &mut self,
        backend: &mut dyn manonica_physics::FreeManaDiffusionBackend,
    ) -> Result<ConservationReport, ExperimentError> {
        self.step_internal(Some(backend))
    }

    fn step_internal(
        &mut self,
        backend: Option<&mut dyn manonica_physics::FreeManaDiffusionBackend>,
    ) -> Result<ConservationReport, ExperimentError> {
        let parameters = self.step_parameters();
        let dt = TimeStep::from_seconds(parameters.dt)?;
        let report = if let Some(mechanics) = &mut self.mechanics {
            if backend.is_some() {
                return Err(manonica_error::BaseError::new(
                    manonica_error::ErrorCode::InvalidState,
                    "mechanics_cpu_only",
                )
                .into());
            }
            let total = self.state.field.total_manons()?;
            mechanics.step(dt)?;
            ConservationReport {
                before: total,
                after: total,
                absolute_error: Quantity::nonnegative(0.0)?,
            }
        } else {
            self.state.step_with_diffusion(&parameters, backend)?
        };
        let after = report.after;
        self.time_seconds += dt.seconds();
        self.steps += 1;
        self.observe(dt.seconds(), after.value());
        Ok(report)
    }

    pub(crate) fn coupled_state(&self) -> &CoupledState {
        &self.state
    }

    /// Import one validated GPU snapshot after a complete resident interval.
    pub(crate) fn step_coupled_gpu(
        &mut self,
        gpu: &mut manonica_compute_gpu::GpuCoupled,
        steps: u32,
    ) -> Result<ConservationReport, ExperimentError> {
        if self.has_mechanics() {
            return Err(manonica_error::BaseError::new(
                manonica_error::ErrorCode::InvalidState,
                "mechanics_cpu_only",
            )
            .into());
        }
        let before = self.total()?;
        let state = gpu.advance(steps)?.clone();
        let after =
            Quantity::<ManonCount>::nonnegative(state.materials.balance(&state.field)?.manons())?;
        self.state = state;
        let elapsed = self.experiment.physics.dt * f64::from(steps);
        self.time_seconds += elapsed;
        self.steps += u64::from(steps);
        self.observe(elapsed, after.value());
        Ok(ConservationReport {
            before,
            after,
            absolute_error: Quantity::<ManonCount>::nonnegative(
                (after.value() - before.value()).abs(),
            )?,
        })
    }

    fn observe(&mut self, dt: f64, total: f64) {
        // A stationary density snapshot is not thermodynamic equilibrium if
        // the surrounding medium still drives a nonzero advective flux.
        let medium_driven = match &self.experiment.physics.flow {
            AmbientFlow::Still => false,
            AmbientFlow::Uniform { velocity } => velocity.iter().any(|v| *v != 0.0),
            AmbientFlow::Vortex {
                angular_speed,
                axial_speed,
                ..
            } => *angular_speed != 0.0 || *axial_speed != 0.0,
        };
        let source_active = (self.state.potential.is_some() && !self.state.shutdown)
            || medium_driven
            || self.has_mechanics();
        let potential_peak = self
            .state
            .potential
            .as_ref()
            .map_or(0.0, PotentialField::peak_absolute);
        self.equilibrium.observe(
            &self.state.field,
            EquilibriumObservation {
                total_manons: total,
                bound_manons: self.state.materials.total_bound(),
                peak_potential: potential_peak,
                temperature_excess_kelvin: self.state.materials.peak_temperature_excess_kelvin(),
                powered_source: source_active,
                dt,
            },
        );
    }

    pub(crate) fn step_parameters(&self) -> StepParameters {
        StepParameters {
            diffusivity: self.experiment.physics.diffusivity,
            dt: self.experiment.physics.dt,
            flow: self.experiment.physics.flow.physical(),
            thermal_conductance: self.experiment.physics.thermal_conductance,
            ambient_cooling_rate: self.experiment.physics.ambient_cooling_rate,
            field: self
                .experiment
                .physics
                .potential
                .as_ref()
                .map(|p| FieldControl {
                    mobility: p.mobility,
                    max_power_joules_per_second: p.max_power_joules_per_second,
                    shutdown_at_bound_fraction: p.shutdown_at_bound_fraction,
                    shutdown_decay_rate: p.shutdown_decay_rate,
                }),
        }
    }

    pub fn reset(&mut self) -> Result<(), ExperimentError> {
        let field = seeded_field(&self.experiment)?;
        let materials = build_materials(&self.experiment, &field)?;
        let (potential, field_ledger) = build_potential(&self.experiment, &field)?;
        let mechanics = self
            .experiment
            .mechanics
            .as_ref()
            .map(|config| {
                super::mechanics::MechanicsSession::new(config, self.experiment.physics.dt)
            })
            .transpose()?;
        self.mechanics = mechanics;
        self.state.potential = potential;
        self.state.field_ledger = field_ledger;
        self.state.shutdown = false;
        self.equilibrium = EquilibriumMonitor::default();
        self.state.field = field;
        self.state.materials = materials;
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
        channel.install(&mut materials)?;
    }
    Ok(materials)
}

fn build_potential(
    experiment: &Experiment,
    field: &FreeManaField,
) -> Result<(Option<PotentialField>, Option<FieldLedger>), ExperimentError> {
    build_potential_cancellable(experiment, field, &|| false)
}

fn build_potential_cancellable(
    experiment: &Experiment,
    field: &FreeManaField,
    should_cancel: &impl Fn() -> bool,
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
    let mut boundary_sites = Vec::<LocalBoundarySite>::new();
    let mut occupied = std::collections::HashSet::new();
    for boundary in &experiment.local_boundaries {
        let formula = &experiment.formulas[boundary.formula()];
        let cells = crate::experiment::capture_cells(formula, dims, dx)?;
        for [x, y, z] in cells {
            let index = shape.index(x, y, z).unwrap();
            if !occupied.insert(index) {
                return Err(manonica_error::BaseError::new(
                    manonica_error::ErrorCode::InvalidRange,
                    "local_boundary_overlap",
                )
                .into());
            }
            boundary_sites.push(LocalBoundarySite {
                index,
                condition: boundary.physics(),
            });
        }
    }
    let potential = PotentialField::solve_with_local_cancellable(
        shape,
        dx,
        &sources,
        settings.permittivity,
        settings.screening,
        settings.boundary.physics(),
        settings.iterations,
        &boundary_sites,
        should_cancel,
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
