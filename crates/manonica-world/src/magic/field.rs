//! Runs existing finite-volume laws; game objects are deliberately absent.
use super::*;
use manonica_core::{
    Evolution, Quantity, TimeStep, Vec3,
    units::{Diffusivity, ManonDensity, Metres, MetresPerSecond},
};
use manonica_physics::{
    FreeAdvection, FreeDiffusion, FreeManaField, GridShape, PhysicsError, PotentialBoundary,
    PotentialField, VelocityField,
};
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldEnergy {
    pub field_joules: f64,
    pub kinetic_joules: f64,
    pub exported_mechanical_work_joules: f64,
    pub potential_joules: f64,
    pub heat_joules: f64,
    pub external_work_joules: f64,
}
impl FieldEnergy {
    pub fn residual_joules(&self) -> f64 {
        self.field_joules
            + self.kinetic_joules
            + self.potential_joules
            + self.heat_joules
            + self.exported_mechanical_work_joules
            - self.external_work_joules
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManonField {
    pub origin: [i64; 3],
    pub dimensions: [usize; 3],
    pub cell_width: f64,
    pub carrier_mass_kg_per_manon: f64,
    pub densities: Vec<f64>,
    pub reference_density: Vec<f64>,
    pub potential: Vec<f64>,
    pub sources: Vec<f64>,
    pub velocity: Vec<[f64; 3]>,
    pub simulated_seconds: f64,
    pub energy: FieldEnergy,
}
impl ManonField {
    pub fn manons(&self) -> f64 {
        self.densities.iter().sum::<f64>() * self.cell_width.powi(3)
    }
    pub fn position(&self, index: usize) -> [f64; 3] {
        let [nx, ny, _] = self.dimensions;
        let v = [index % nx, (index / nx) % ny, index / (nx * ny)];
        std::array::from_fn(|a| (self.origin[a] as f64 + v[a] as f64 + 0.5) * self.cell_width)
    }
    pub(super) fn validate(&self) -> WorldResult<()> {
        let shape = GridShape::new(self.dimensions[0], self.dimensions[1], self.dimensions[2])?;
        let len = shape.len();
        if self.origin.iter().any(|v| v.unsigned_abs() > 100_000_000)
            || len > 4096
            || self.reference_density.len() != len
            || self
                .reference_density
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
            || self.densities.len() != len
            || self.potential.len() != len
            || self.sources.len() != len
            || self.velocity.len() != len
            || self.densities.iter().any(|v| !v.is_finite() || *v < 0.0)
            || self
                .potential
                .iter()
                .chain(self.sources.iter())
                .chain(self.velocity.iter().flatten())
                .any(|v| !v.is_finite())
            || !self.carrier_mass_kg_per_manon.is_finite()
            || self.carrier_mass_kg_per_manon <= 0.0
            || !self.cell_width.is_finite()
            || self.cell_width <= 0.0
            || !self.manons().is_finite()
            || !self.simulated_seconds.is_finite()
            || self.simulated_seconds < 0.0
        {
            return Err(WorldError::Invalid("manon_field"));
        }
        let actual_kinetic = self
            .densities
            .iter()
            .zip(&self.velocity)
            .map(|(n, v)| {
                0.5 * self.carrier_mass_kg_per_manon
                    * n
                    * self.cell_width.powi(3)
                    * v.iter().map(|v| v * v).sum::<f64>()
            })
            .sum::<f64>();
        if !actual_kinetic.is_finite()
            || (actual_kinetic - self.energy.kinetic_joules).abs() > 1e-8 * actual_kinetic.max(1.0)
        {
            return Err(WorldError::Invalid("carrier_kinetic_energy"));
        }
        let e = &self.energy;
        if [
            e.field_joules,
            e.kinetic_joules,
            e.exported_mechanical_work_joules,
            e.potential_joules,
            e.heat_joules,
            e.external_work_joules,
        ]
        .iter()
        .any(|v| !v.is_finite())
            || e.field_joules < 0.0
            || e.kinetic_joules < 0.0
            || e.heat_joules < 0.0
            || e.external_work_joules < 0.0
            || e.residual_joules().abs()
                > 1e-8 * (e.external_work_joules + e.potential_joules.abs() + 1.0)
        {
            return Err(WorldError::Invalid("manon_energy_ledger"));
        }
        Ok(())
    }
}
fn physical(e: PhysicsError) -> WorldError {
    WorldError::Codec(format!("manon solver: {e}"))
}
pub(super) fn local(event: &SpellEvent, p: [f64; 3]) -> [f64; 3] {
    let d = std::array::from_fn::<_, 3, _>(|a| p[a] - event.center[a]);
    let [x, _, z] = event.direction;
    [-d[0] * z + d[2] * x, d[1], d[0] * x + d[2] * z]
}
fn world_vector(event: &SpellEvent, v: [f64; 3]) -> [f64; 3] {
    let [x, _, z] = event.direction;
    [-z * v[0] + x * v[2], v[1], x * v[0] + z * v[2]]
}
struct Flow<'a> {
    event: &'a SpellEvent,
    compiled: &'a program::CompiledProgram,
    time: f64,
    field: &'a FreeManaField,
}
impl VelocityField for Flow<'_> {
    fn at(&self, p: Vec3<Metres>) -> Result<Vec3<MetresPerSecond>, PhysicsError> {
        let f = &self.event.field;
        let xyz = [p.x.value(), p.y.value(), p.z.value()];
        let physical = std::array::from_fn(|a| {
            xyz[a] + (f.origin[a] as f64 + f.dimensions[a] as f64 * 0.5) * f.cell_width
        });
        let mut nearest = [0; 3];
        for a in 0..3 {
            nearest[a] = ((xyz[a] / f.cell_width + f.dimensions[a] as f64 * 0.5).floor() as usize)
                .min(f.dimensions[a] - 1);
        }
        let n = self
            .field
            .at(nearest[0], nearest[1], nearest[2])
            .unwrap()
            .value();
        let point = program::FormulaPoint {
            xyz: local(self.event, physical),
            time: self.time,
            density: n,
        };
        let mut v = [0.; 3];
        for (a, expr) in self.compiled.velocity.iter().enumerate() {
            v[a] = self.event.program.evaluate(expr, point).map_err(|_| {
                manonica_error::BaseError::new(
                    manonica_error::ErrorCode::NonFinite,
                    "formula_velocity",
                )
            })?;
        }
        let v = world_vector(self.event, v);
        Ok(Vec3::try_values(v[0], v[1], v[2])?)
    }
}
fn pay(
    energy: &mut FieldEnergy,
    amount: f64,
    budget: WorkBudget,
    limit: &mut f64,
) -> WorldResult<()> {
    if amount <= 0.0 {
        energy.heat_joules -= amount;
        return Ok(());
    }
    let heat = energy.heat_joules.min(amount);
    energy.heat_joules -= heat;
    let work = amount - heat;
    if work > budget.total_joules - energy.external_work_joules + 1e-9 || work > *limit + 1e-9 {
        return Err(WorldError::Invalid("spell_work_budget_exhausted"));
    }
    energy.external_work_joules += work;
    *limit = (*limit - work).max(0.0);
    Ok(())
}
pub(super) fn step(event: &mut SpellEvent, dt: f64, finishing: bool) -> WorldResult<()> {
    let compiled = event.program.compile()?;
    let f = &event.field;
    let [nx, ny, nz] = f.dimensions;
    let shape = GridShape::new(nx, ny, nz)?;
    let mut density = FreeManaField::uniform(
        shape,
        f.cell_width,
        Quantity::<ManonDensity>::nonnegative(0.0)?,
    )?;
    density.try_replace_densities(
        f.densities
            .iter()
            .map(|v| Quantity::nonnegative(*v))
            .collect::<Result<Vec<_>, _>>()?,
    )?;
    let s = event.program.transport;
    let t = event.field.simulated_seconds + dt;
    let mut source = vec![0.; shape.len()];
    if !finishing {
        for (i, value) in source.iter_mut().enumerate() {
            *value = event.program.evaluate(
                &compiled.source,
                program::FormulaPoint {
                    xyz: local(event, f.position(i)),
                    time: t,
                    density: f.densities[i],
                },
            )?;
        }
    }
    let potential = PotentialField::solve(
        shape,
        f.cell_width,
        &source,
        s.permittivity,
        s.screening,
        PotentialBoundary::Insulated,
        s.iterations,
    )?;
    let mut energy = f.energy.clone();
    let mut limit = event.program.work.max_power_watts * dt;
    let field_energy = potential.field_energy_joules();
    let initial_u = potential.manon_potential_energy(&density)?;
    let installation = field_energy + initial_u - energy.field_joules - energy.potential_joules;
    pay(&mut energy, installation, event.program.work, &mut limit)?;
    energy.field_joules = field_energy;
    energy.potential_joules = initial_u;
    let mut velocity = vec![[0.; 3]; shape.len()];
    if !finishing {
        // Small physical substeps, including CFL-checked reference operators.
        let substeps = (dt / 0.005).ceil().max(1.0) as usize;
        let h = TimeStep::from_seconds(dt / substeps as f64)?;
        for _ in 0..substeps {
            FreeDiffusion::new(Quantity::<Diffusivity>::nonnegative(s.diffusivity)?)
                .map_err(physical)?
                .evolve(&mut density, h)
                .map_err(physical)?;
            potential.drift(&mut density, s.mobility, h.seconds())?;
            let flow = Flow {
                event,
                compiled: &compiled,
                time: t,
                field: &density,
            };
            let mut transported = density.clone();
            FreeAdvection::new(flow)
                .evolve(&mut transported, h)
                .map_err(physical)?;
            density = transported;
        }
        let flow = Flow {
            event,
            compiled: &compiled,
            time: t,
            field: &density,
        };
        for (i, v) in velocity.iter_mut().enumerate() {
            let p = std::array::from_fn::<_, 3, _>(|a| {
                f.position(i)[a]
                    - (f.origin[a] as f64 + f.dimensions[a] as f64 * 0.5) * f.cell_width
            });
            let out = flow
                .at(Vec3::try_values(p[0], p[1], p[2])?)
                .map_err(physical)?;
            *v = [out.x.value(), out.y.value(), out.z.value()];
        }
    }
    if !finishing {
        for (i, bulk) in velocity.iter_mut().enumerate() {
            let xyz = [i % nx, (i / nx) % ny, i / (nx * ny)];
            let n = density.densities()[i].value();
            for axis in 0..3 {
                let mut low = xyz;
                let mut high = xyz;
                low[axis] = low[axis].saturating_sub(1);
                high[axis] = (high[axis] + 1).min([nx, ny, nz][axis] - 1);
                let a = shape.index(low[0], low[1], low[2]).unwrap();
                let b = shape.index(high[0], high[1], high[2]).unwrap();
                let distance = (high[axis] - low[axis]) as f64 * f.cell_width;
                bulk[axis] -=
                    s.mobility * (potential.values()[b] - potential.values()[a]) / distance;
                bulk[axis] -= s.diffusivity
                    * (density.densities()[b].value() - density.densities()[a].value())
                    / distance
                    / n.max(1e-9);
            }
        }
    }
    let kinetic = velocity
        .iter()
        .zip(density.densities())
        .map(|(v, n)| {
            0.5 * f.carrier_mass_kg_per_manon
                * n.value()
                * f.cell_width.powi(3)
                * v.iter().map(|v| v * v).sum::<f64>()
        })
        .sum::<f64>();
    let kinetic_change = kinetic - energy.kinetic_joules;
    pay(&mut energy, kinetic_change, event.program.work, &mut limit)?;
    energy.kinetic_joules = kinetic;
    let after_u = potential.manon_potential_energy(&density)?;
    pay(
        &mut energy,
        after_u - initial_u,
        event.program.work,
        &mut limit,
    )?;
    energy.potential_joules = after_u;
    if !finishing {
        let heat = event.program.evaluate(
            &compiled.heat,
            program::FormulaPoint {
                xyz: [0.; 3],
                time: t,
                density: 0.0,
            },
        )?;
        if heat < 0.0 {
            return Err(WorldError::Invalid("negative_heat_power"));
        }
        let work = heat * dt;
        if work > limit + 1e-9
            || work > event.program.work.total_joules - energy.external_work_joules + 1e-9
        {
            return Err(WorldError::Invalid("spell_heat_work_budget"));
        }
        energy.heat_joules += work;
        energy.external_work_joules += work;
    }
    let next = ManonField {
        origin: f.origin,
        dimensions: f.dimensions,
        cell_width: f.cell_width,
        carrier_mass_kg_per_manon: f.carrier_mass_kg_per_manon,
        reference_density: f.reference_density.clone(),
        densities: density.densities().iter().map(|v| v.value()).collect(),
        potential: potential.values().to_vec(),
        sources: source,
        velocity,
        simulated_seconds: t,
        energy,
    };
    next.validate()?;
    if (next.manons() - f.manons()).abs() > 1e-10 * f.manons().max(1.0) {
        return Err(WorldError::Invalid("spell_manon_conservation"));
    }
    event.field = next;
    Ok(())
}

pub(super) fn exchange(
    event: &mut crate::magic::SpellEvent,
    index: usize,
    body: &mut manonica_physics::MechanicalBody,
    rate: f64,
    dt: f64,
) -> WorldResult<f64> {
    let f = &mut event.field;
    let mass = f.densities[index] * f.cell_width.powi(3) * f.carrier_mass_kg_per_manon;
    if mass <= 1e-12 {
        return Ok(0.0);
    }
    let position = f.position(index);
    let v = f.velocity[index];
    let mut packet = manonica_physics::MechanicalBody::new(
        Quantity::new(mass)?,
        Vec3::try_values(position[0], position[1], position[2])?,
        Vec3::try_values(v[0] * mass, v[1] * mass, v[2] * mass)?,
    )?;
    let old_packet = packet.kinetic_energy()?.value();
    let before = body.kinetic_energy()?.value();
    let report = manonica_physics::DissipativeExchange::new(Quantity::<
        manonica_core::units::PerSecond<manonica_core::units::Dimensionless>,
    >::nonnegative(rate)?)?
    .exchange(&mut packet, body, TimeStep::from_seconds(dt)?)?;
    let work = body.kinetic_energy()?.value() - before;
    let p = packet.momentum();
    f.velocity[index] = [p.x.value() / mass, p.y.value() / mass, p.z.value() / mass];
    f.energy.kinetic_joules += packet.kinetic_energy()?.value() - old_packet;
    f.energy.kinetic_joules = f.energy.kinetic_joules.max(0.0);
    f.energy.heat_joules += report.heat.value();
    f.energy.exported_mechanical_work_joules += work;
    Ok(work)
}
