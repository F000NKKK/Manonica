//! Experimental material response to manon carrier packets, independent of spell names.
use crate::{
    WorldError, WorldResult,
    chunk::{CellPosition, ChunkWorld},
    metadata::NbtCompound,
    registry::ResourceKey,
};
use manonica_core::{
    Quantity, TimeStep, Vec3,
    units::{KilogramMetresPerSecond, Kilograms, Metres, Newtons},
};
use manonica_physics::{ExternalLoadLedger, MechanicalBody};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MediumCellId {
    pub domain: usize,
    pub cell: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct MediumSample {
    pub id: MediumCellId,
    pub position: [f64; 3],
    pub velocity: [f64; 3],
}
/// Ordinary world mechanics depends on a physical momentum reservoir, not a spell API.
pub trait MomentumMedium {
    fn samples(&self) -> Vec<MediumSample>;
    fn transfer_heat(
        &mut self,
        _id: MediumCellId,
        _heat: &mut f64,
        _capacity: f64,
        _conductance: f64,
        _dt: f64,
    ) -> WorldResult<f64> {
        Ok(0.0)
    }
    fn transfer(
        &mut self,
        id: MediumCellId,
        body: &mut MechanicalBody,
        rate: f64,
        dt: f64,
    ) -> WorldResult<f64>;
}
#[derive(Clone, Debug)]
pub(crate) struct SettledMaterial {
    pub position: CellPosition,
    pub block: ResourceKey,
    pub properties: BTreeMap<String, String>,
    pub data: NbtCompound,
}
#[derive(Clone, Debug, Default)]
pub(crate) struct MaterialChanges {
    pub detached: Vec<CellPosition>,
    pub settled: Vec<SettledMaterial>,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldPhysics {
    pub thermal: crate::thermal::MediumThermodynamics,
    pub minimum_dynamic_height: usize,
    pub carrier_mass_kg_per_manon: f64,
    pub gravity_metres_per_second_squared: f64,
}
impl Default for WorldPhysics {
    fn default() -> Self {
        Self {
            minimum_dynamic_height: 1,
            thermal: crate::thermal::MediumThermodynamics::default(),
            carrier_mass_kg_per_manon: 0.25,
            gravity_metres_per_second_squared: 20.0,
        }
    }
}
impl WorldPhysics {
    pub(crate) fn validate(self) -> WorldResult<()> {
        self.thermal.validate()?;
        if !self.carrier_mass_kg_per_manon.is_finite()
            || self.carrier_mass_kg_per_manon <= 0.0
            || !self.gravity_metres_per_second_squared.is_finite()
            || self.gravity_metres_per_second_squared < 0.0
        {
            return Err(WorldError::Invalid("world_physics"));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodyMotion {
    pub position_metres: [f64; 3],
    pub momentum: [f64; 3],
    pub mass_kg: f64,
}
impl BodyMotion {
    fn body(self) -> WorldResult<MechanicalBody> {
        Ok(MechanicalBody::new(
            Quantity::<Kilograms>::new(self.mass_kg)?,
            Vec3::<Metres>::try_values(
                self.position_metres[0],
                self.position_metres[1],
                self.position_metres[2],
            )?,
            Vec3::<KilogramMetresPerSecond>::try_values(
                self.momentum[0],
                self.momentum[1],
                self.momentum[2],
            )?,
        )?)
    }
    fn update(&mut self, body: MechanicalBody) {
        let p = body.position();
        self.position_metres = [p.x.value(), p.y.value(), p.z.value()];
        let p = body.momentum();
        self.momentum = [p.x.value(), p.y.value(), p.z.value()];
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialBody {
    pub block: ResourceKey,
    pub properties: BTreeMap<String, String>,
    pub data: NbtCompound,
    pub motion: BodyMotion,
    pub heat_joules: f64,
    pub age_seconds: f64,
    pub drag_rate_per_second: f64,
    pub resting: bool,
    pub still_seconds: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialAnchor {
    pub position: [i64; 3],
    pub heat_joules: f64,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatterEnergy {
    pub heat_received_joules: f64,
    pub heat_exported_to_fields_joules: f64,
    pub radiated_joules: f64,
    pub dispersed_joules: f64,
    pub mechanical_work_received_joules: f64,
    pub gravity_work_joules: f64,
    pub surface_joules: f64,
    pub contact_heat_joules: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialParticles {
    pub emissivity: f64,
    pub heat_joules: f64,
    pub capacity_joules_per_kelvin: f64,
    pub velocity: [f64; 3],
    pub lifetime_seconds: f64,
    pub block: ResourceKey,
    pub position: [f64; 3],
    pub age_seconds: f64,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatterState {
    pub medium_heat: Vec<crate::thermal::ThermalCell>,
    pub dispersed_blocks: u64,
    pub dispersed_mass_kg: f64,
    pub particles: Vec<MaterialParticles>,
    pub bodies: Vec<MaterialBody>,
    pub anchors: Vec<MaterialAnchor>,
    pub energy: MatterEnergy,
    pub revision: u64,
}
impl MatterState {
    pub(crate) fn emit_break(&mut self, block: ResourceKey, position: [f64; 3]) {
        if self.particles.len() >= 128 {
            self.particles.remove(0);
        }
        self.particles.push(MaterialParticles {
            block,
            position,
            heat_joules: 0.,
            capacity_joules_per_kelvin: 1.,
            velocity: [0.; 3],
            lifetime_seconds: 1.25,
            emissivity: 0.8,
            age_seconds: 0.0,
        });
    }

    fn emit_thermal_break(
        &mut self,
        block: ResourceKey,
        position: [f64; 3],
        heat: f64,
        capacity: f64,
        velocity: [f64; 3],
        emissivity: f64,
    ) {
        if self.particles.len() >= 128 {
            self.particles.remove(0);
        }
        self.particles.push(MaterialParticles {
            block,
            position,
            heat_joules: heat,
            capacity_joules_per_kelvin: capacity,
            velocity,
            lifetime_seconds: 4.0,
            emissivity,
            age_seconds: 0.0,
        });
    }
    pub fn stored_energy_joules(&self) -> WorldResult<f64> {
        let mut stored = self.energy.dispersed_joules
            + self.medium_heat.iter().map(|c| c.heat_joules).sum::<f64>()
            + self.energy.surface_joules
            + self.energy.contact_heat_joules
            + self.anchors.iter().map(|a| a.heat_joules).sum::<f64>();
        for b in &self.bodies {
            stored += b.heat_joules + b.motion.body()?.kinetic_energy()?.value();
        }
        Ok(stored)
    }
    pub fn energy_residual_joules(&self) -> WorldResult<f64> {
        Ok(self.stored_energy_joules()? + self.energy.radiated_joules
            - self.energy.heat_received_joules
            + self.energy.heat_exported_to_fields_joules
            - self.energy.mechanical_work_received_joules
            - self.energy.gravity_work_joules)
    }
    pub(crate) fn validate(&self) -> WorldResult<()> {
        if self.medium_heat.len() > 32768
            || self
                .medium_heat
                .iter()
                .any(|c| !c.heat_joules.is_finite() || c.heat_joules < 0.0 || c.position[1] < 0)
            || self.particles.len() > 128
            || self.particles.iter().any(|p| {
                !p.emissivity.is_finite()
                    || !(0.0..=1.0).contains(&p.emissivity)
                    || !p.heat_joules.is_finite()
                    || p.heat_joules < 0.0
                    || !p.capacity_joules_per_kelvin.is_finite()
                    || p.capacity_joules_per_kelvin <= 0.0
                    || !p.lifetime_seconds.is_finite()
                    || p.lifetime_seconds <= 0.0
                    || p.age_seconds >= p.lifetime_seconds
                    || p.velocity.iter().any(|v| !v.is_finite())
                    || !p.age_seconds.is_finite()
                    || p.age_seconds < 0.0
                    || p.position.iter().any(|v| !v.is_finite())
            })
            || self.bodies.len() > 1024
            || self.anchors.len() > 8192
            || [self.energy.surface_joules, self.energy.contact_heat_joules]
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(WorldError::Invalid("matter_state"));
        }
        for b in &self.bodies {
            b.motion.body()?;
            b.data.validate()?;
            if !b.heat_joules.is_finite()
                || b.heat_joules < 0.0
                || !b.still_seconds.is_finite()
                || b.still_seconds < 0.0
                || !b.age_seconds.is_finite()
                || b.age_seconds < 0.0
                || !b.drag_rate_per_second.is_finite()
                || b.drag_rate_per_second < 0.0
                || b.motion.position_metres.iter().any(|v| v.abs() > 1e8)
            {
                return Err(WorldError::Invalid("material_body"));
            }
        }
        for a in &self.anchors {
            if !a.heat_joules.is_finite() || a.heat_joules < 0.0 || a.position[1] < 0 {
                return Err(WorldError::Invalid("anchor_heat"));
            }
        }
        if [
            self.energy.dispersed_joules,
            self.energy.radiated_joules,
            self.energy.heat_exported_to_fields_joules,
            self.dispersed_mass_kg,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(WorldError::Invalid("thermal_matter_ledger"));
        }
        if self.particles.iter().map(|p| p.heat_joules).sum::<f64>()
            > self.energy.dispersed_joules + 1e-7 * self.energy.dispersed_joules.max(1.0)
        {
            return Err(WorldError::Invalid("dispersed_heat_partition"));
        }
        let residual = self.energy_residual_joules()?;
        if !residual.is_finite()
            || residual.abs()
                > 1e-7
                    * (self.energy.mechanical_work_received_joules.abs()
                        + self.energy.gravity_work_joules.abs()
                        + 1.0)
        {
            return Err(WorldError::Invalid("matter_energy_balance"));
        }
        Ok(())
    }
}
fn mechanical(mass: f64, position: [f64; 3], velocity: [f64; 3]) -> WorldResult<MechanicalBody> {
    Ok(MechanicalBody::new(
        Quantity::new(mass)?,
        Vec3::try_values(position[0], position[1], position[2])?,
        Vec3::try_values(velocity[0] * mass, velocity[1] * mass, velocity[2] * mass)?,
    )?)
}
fn cell(point: [f64; 3], dx: f64) -> WorldResult<CellPosition> {
    let p = point.map(|v| (v / dx).floor() as i64);
    Ok(CellPosition {
        x: p[0],
        y: usize::try_from(p[1]).map_err(|_| WorldError::Invalid("matter_cell"))?,
        z: p[2],
    })
}
fn collision(world: &ChunkWorld, point: [f64; 3], ignored: &[CellPosition]) -> bool {
    let dx = world.layout().cell_width();
    let radius = 0.42 * dx;
    for z in ((point[2] - radius) / dx).floor() as i64..=((point[2] + radius) / dx).floor() as i64 {
        for y in
            ((point[1] - radius) / dx).floor() as i64..=((point[1] + radius) / dx).floor() as i64
        {
            for x in ((point[0] - radius) / dx).floor() as i64
                ..=((point[0] + radius) / dx).floor() as i64
            {
                if y < 0
                    || (!ignored.contains(&CellPosition {
                        x,
                        y: y as usize,
                        z,
                    }) && crate::gameplay::solid(
                        world,
                        CellPosition {
                            x,
                            y: y as usize,
                            z,
                        },
                        true,
                    ))
                {
                    return true;
                }
            }
        }
    }
    false
}
/// Grid reinsertion is a relocation policy, independent of the force source.
/// Search nearby shells first; if boxed in, search every resident column.
fn settlement_cell(
    world: &ChunkWorld,
    origin: CellPosition,
    occupied: &[([f64; 3], [f64; 3])],
    reserved: &std::collections::BTreeSet<[i64; 3]>,
) -> Option<CellPosition> {
    let dx = world.layout().cell_width();
    let free = |p: CellPosition| {
        let xyz = [p.x, p.y as i64, p.z];
        let lo = xyz.map(|v| v as f64 * dx);
        !reserved.contains(&xyz)
            && world.block(p).is_ok_and(|s| {
                world
                    .content()
                    .block(s.block())
                    .is_ok_and(|d| d.physics.replaceable)
            })
            && !occupied.iter().any(|(center, half)| {
                (0..3).all(|a| {
                    center[a] + half[a] > lo[a] + 1e-6 && center[a] - half[a] < lo[a] + dx - 1e-6
                })
            })
    };
    for radius in 0_i64..=8 {
        for y in -radius..=radius {
            for z in -radius..=radius {
                for x in -radius..=radius {
                    if x.abs().max(y.abs()).max(z.abs()) != radius {
                        continue;
                    }
                    let yy = origin.y as i64 + y;
                    if yy < 0 {
                        continue;
                    }
                    let p = CellPosition {
                        x: origin.x + x,
                        y: yy as usize,
                        z: origin.z + z,
                    };
                    if free(p) {
                        return Some(p);
                    }
                }
            }
        }
    }
    let dims = world.layout().dimensions();
    let mut nearest = None;
    let mut best = f64::INFINITY;
    for (coord, _) in world.geometry_chunks() {
        for z in 0..dims[2] {
            for x in 0..dims[0] {
                let xx = i64::from(coord.x) * dims[0] as i64 + x as i64;
                let zz = i64::from(coord.z) * dims[2] as i64 + z as i64;
                let horizontal = (xx - origin.x) as f64 * (xx - origin.x) as f64
                    + (zz - origin.z) as f64 * (zz - origin.z) as f64;
                if horizontal >= best {
                    continue;
                }
                for y in 0..dims[1] {
                    let distance = horizontal + (y as f64 - origin.y as f64).powi(2);
                    if distance >= best {
                        continue;
                    }
                    let p = CellPosition { x: xx, y, z: zz };
                    if free(p) {
                        nearest = Some(p);
                        best = distance;
                    }
                }
            }
        }
    }
    nearest
}
pub(crate) fn step(
    world: &ChunkWorld,
    medium: &mut impl MomentumMedium,
    previous: &MatterState,
    physics: WorldPhysics,
    dt: f64,
    occupied: &[([f64; 3], [f64; 3])],
    heat_deposits: &[(CellPosition, f64)],
) -> WorldResult<(MatterState, MaterialChanges)> {
    let mut next = previous.clone();
    let mut gas_heat = next
        .medium_heat
        .iter()
        .map(|c| (c.position, c.heat_joules))
        .collect::<BTreeMap<_, _>>();
    let mut particles = vec![];
    for mut event in next.particles.drain(..) {
        event.age_seconds += dt;
        if event.heat_joules > 0.0 {
            let mut reservoir = manonica_materials::ThermalReservoir {
                heat_joules: event.heat_joules,
                capacity_joules_per_kelvin: event.capacity_joules_per_kelvin,
            };
            let radiated = manonica_materials::radiate_to_bath(
                &mut reservoir,
                physics.thermal.reference_temperature_kelvin,
                event.emissivity,
                6. * world.layout().cell_width().powi(2) * 18_f64.cbrt(),
                dt,
            )?;
            event.heat_joules = reservoir.heat_joules;
            next.energy.dispersed_joules = (next.energy.dispersed_joules - radiated).max(0.0);
            next.energy.radiated_joules += radiated;
        }
        if event.age_seconds < event.lifetime_seconds {
            particles.push(event);
        } else if event.heat_joules > 0.0 {
            let p = cell(event.position, world.layout().cell_width())?;
            *gas_heat.entry([p.x, p.y as i64, p.z]).or_default() += event.heat_joules;
            next.energy.dispersed_joules =
                (next.energy.dispersed_joules - event.heat_joules).max(0.0);
        }
    }
    next.particles = particles;
    for (cell, q) in heat_deposits {
        *gas_heat.entry([cell.x, cell.y as i64, cell.z]).or_default() += q;
        next.energy.heat_received_joules += q;
    }
    let mut edits = vec![];
    let mut anchors = next
        .anchors
        .iter()
        .map(|a| (a.position, a.heat_joules))
        .collect::<BTreeMap<_, _>>();
    let samples = medium.samples();
    for sample in &samples {
        if next.bodies.len() >= 1024 {
            break;
        }
        let base = cell(sample.position, world.layout().cell_width())?;
        for offset in [
            [0_i64, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ] {
            let y = base.y as i64 + offset[1];
            if y < 0 {
                continue;
            }
            let p = CellPosition {
                x: base.x + offset[0],
                y: y as usize,
                z: base.z + offset[2],
            };
            if edits.contains(&p) {
                continue;
            }
            let Ok(block_state) = world.block(p) else {
                continue;
            };
            let weight = if offset == [0, 0, 0] { 1.0 } else { 0.5 };
            let definition = world.content().block(block_state.block())?;
            let Some(material) = definition.physics.mechanics else {
                continue;
            };
            let block = definition.key.clone();
            let properties = world.block(p)?.properties().clone();
            let dx = world.layout().cell_width();
            let position = [
                (p.x as f64 + 0.5) * dx,
                (p.y as f64 + 0.5) * dx,
                (p.z as f64 + 0.5) * dx,
            ];
            let anchor = anchors.entry([p.x, p.y as i64, p.z]).or_default();
            let heat = medium.transfer_heat(
                sample.id,
                anchor,
                material.heat_capacity_joules_per_kelvin,
                material.thermal_contact_watts_per_kelvin * weight,
                dt,
            )?;
            next.energy.heat_received_joules += heat;
            let mut body = mechanical(material.mass_kg, position, [0.; 3])?;
            let work = medium.transfer(
                sample.id,
                &mut body,
                material.drag_rate_per_second * weight,
                dt,
            )?;
            next.energy.mechanical_work_received_joules += work;
            let momentum = body.momentum();
            let stress = momentum
                .x
                .value()
                .hypot(momentum.y.value())
                .hypot(momentum.z.value())
                / dt;
            let anchor = anchors.entry([p.x, p.y as i64, p.z]).or_default();
            if p.y >= physics.minimum_dynamic_height
                && material
                    .crushing_energy_joules
                    .is_some_and(|threshold| *anchor + work >= threshold)
            {
                next.energy.surface_joules += material.fracture_energy_joules;
                let fragment_energy = *anchor + work - material.fracture_energy_joules;
                next.energy.dispersed_joules += fragment_energy;
                *anchor = 0.;
                next.dispersed_blocks += 1;
                next.dispersed_mass_kg += material.mass_kg;
                let fragment_momentum = body.momentum();
                next.emit_thermal_break(
                    block,
                    position,
                    fragment_energy,
                    material.heat_capacity_joules_per_kelvin,
                    [
                        fragment_momentum.x.value() / material.mass_kg,
                        fragment_momentum.y.value() / material.mass_kg,
                        fragment_momentum.z.value() / material.mass_kg,
                    ],
                    material.surface_emissivity,
                );
                edits.push(p);
                next.revision += 1;
                continue;
            }
            if p.y >= physics.minimum_dynamic_height
                && stress >= material.tensile_strength_newtons
                && *anchor >= material.fracture_energy_joules
            {
                let heat = *anchor - material.fracture_energy_joules;
                *anchor = 0.;
                next.energy.surface_joules += material.fracture_energy_joules;
                let mut motion = BodyMotion {
                    position_metres: position,
                    momentum: [0.; 3],
                    mass_kg: material.mass_kg,
                };
                motion.update(body);
                next.bodies.push(MaterialBody {
                    block,
                    properties,
                    data: world.block_data(p)?,
                    motion,
                    heat_joules: heat,
                    age_seconds: 0.,
                    drag_rate_per_second: material.drag_rate_per_second,
                    resting: false,
                    still_seconds: 0.0,
                });
                edits.push(p);
                next.revision += 1;
            } else {
                *anchor += work;
            }
        }
    }
    for (position, q) in &mut gas_heat {
        let p = CellPosition {
            x: position[0],
            y: position[1] as usize,
            z: position[2],
        };
        if let Ok(definition) = world
            .block(p)
            .and_then(|state| world.content().block(state.block()))
            && let Some(material) = definition.physics.mechanics
        {
            let solid = anchors.entry(*position).or_default();
            let mut gas = manonica_materials::ThermalReservoir {
                heat_joules: *q,
                capacity_joules_per_kelvin: physics
                    .thermal
                    .heat_capacity_joules_per_cubic_metre_kelvin
                    * world.layout().cell_width().powi(3),
            };
            let mut solid_reservoir = manonica_materials::ThermalReservoir {
                heat_joules: *solid,
                capacity_joules_per_kelvin: material.heat_capacity_joules_per_kelvin,
            };
            manonica_materials::exchange_heat_exact(
                &mut gas,
                &mut solid_reservoir,
                material.thermal_contact_watts_per_kelvin,
                dt,
            )?;
            *q = gas.heat_joules;
            *solid = solid_reservoir.heat_joules;
        }
        let mut reservoir = manonica_materials::ThermalReservoir {
            heat_joules: *q,
            capacity_joules_per_kelvin: physics.thermal.heat_capacity_joules_per_cubic_metre_kelvin
                * world.layout().cell_width().powi(3),
        };
        next.energy.radiated_joules += manonica_materials::radiate_to_bath(
            &mut reservoir,
            physics.thermal.reference_temperature_kelvin,
            physics.thermal.emissivity,
            6. * world.layout().cell_width().powi(2),
            dt,
        )?;
        *q = reservoir.heat_joules;
    }
    next.medium_heat = gas_heat
        .into_iter()
        .filter(|(_, q)| *q > 0.0)
        .map(|(position, heat_joules)| crate::thermal::ThermalCell {
            position,
            heat_joules,
        })
        .collect();
    for (position, q) in &mut anchors {
        let p = CellPosition {
            x: position[0],
            y: position[1] as usize,
            z: position[2],
        };
        let definition = world
            .block(p)
            .ok()
            .and_then(|s| world.content().block(s.block()).ok())
            .and_then(|d| d.physics.mechanics);
        let emissivity = definition
            .map(|m| m.surface_emissivity)
            .unwrap_or(physics.thermal.emissivity);
        let capacity = world
            .block(p)
            .ok()
            .and_then(|s| world.content().block(s.block()).ok())
            .and_then(|d| d.physics.mechanics)
            .map(|m| m.heat_capacity_joules_per_kelvin)
            .unwrap_or(
                physics.thermal.heat_capacity_joules_per_cubic_metre_kelvin
                    * world.layout().cell_width().powi(3),
            );
        let mut reservoir = manonica_materials::ThermalReservoir {
            heat_joules: *q,
            capacity_joules_per_kelvin: capacity,
        };
        next.energy.radiated_joules += manonica_materials::radiate_to_bath(
            &mut reservoir,
            physics.thermal.reference_temperature_kelvin,
            emissivity,
            6. * world.layout().cell_width().powi(2),
            dt,
        )?;
        *q = reservoir.heat_joules;
    }
    next.anchors = anchors
        .into_iter()
        .filter(|(_, heat)| *heat > 0.)
        .map(|(position, heat_joules)| MaterialAnchor {
            position,
            heat_joules,
        })
        .collect();
    // Broad phase: only the seven contact cells can exchange with a body.
    // Keep original sample order to preserve the sequential finite-reservoir law.
    let mut sample_cells: BTreeMap<[i64; 3], Vec<usize>> = BTreeMap::new();
    for (index, sample) in samples.iter().enumerate() {
        if let Ok(p) = cell(sample.position, world.layout().cell_width()) {
            sample_cells
                .entry([p.x, p.y as i64, p.z])
                .or_default()
                .push(index);
        }
    }
    for material in &mut next.bodies {
        let mut body = material.motion.body()?;
        if let Ok(target) = cell(material.motion.position_metres, world.layout().cell_width()) {
            let mut contacts = Vec::with_capacity(7);
            for offset in [
                [0, 0, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ] {
                let key = [
                    target.x + offset[0],
                    target.y as i64 + offset[1],
                    target.z + offset[2],
                ];
                if let Some(indices) = sample_cells.get(&key) {
                    contacts.extend(
                        indices
                            .iter()
                            .map(|index| (*index, if offset == [0, 0, 0] { 1.0 } else { 0.5 })),
                    );
                }
            }
            contacts.sort_unstable_by_key(|(index, _)| *index);
            for (index, weight) in contacts {
                let sample = &samples[index];
                let capacity = world
                    .content()
                    .block(world.content().block_id(&material.block)?)?
                    .physics
                    .mechanics
                    .map(|m| m.heat_capacity_joules_per_kelvin)
                    .unwrap_or(100.);
                let q = medium.transfer_heat(
                    sample.id,
                    &mut material.heat_joules,
                    capacity,
                    physics.thermal.contact_watts_per_kelvin * weight,
                    dt,
                )?;
                next.energy.heat_received_joules += q;
                let work = medium.transfer(
                    sample.id,
                    &mut body,
                    material.drag_rate_per_second * weight,
                    dt,
                )?;
                next.energy.mechanical_work_received_joules += work;
                if work != 0.0 {
                    material.resting = false;
                    material.still_seconds = 0.0;
                }
            }
        }
        let emissivity = world
            .content()
            .block(world.content().block_id(&material.block)?)?
            .physics
            .mechanics
            .map(|m| m.surface_emissivity)
            .unwrap_or(physics.thermal.emissivity);
        let capacity = world
            .content()
            .block(world.content().block_id(&material.block)?)?
            .physics
            .mechanics
            .map(|m| m.heat_capacity_joules_per_kelvin)
            .unwrap_or(100.);
        let mut reservoir = manonica_materials::ThermalReservoir {
            heat_joules: material.heat_joules,
            capacity_joules_per_kelvin: capacity,
        };
        next.energy.radiated_joules += manonica_materials::radiate_to_bath(
            &mut reservoir,
            physics.thermal.reference_temperature_kelvin,
            emissivity,
            6. * world.layout().cell_width().powi(2),
            dt,
        )?;
        material.heat_joules = reservoir.heat_joules;
        material.age_seconds += dt;
        if material.resting {
            material.still_seconds += dt;
            material.motion.update(body);
            continue;
        }
        let speed = body
            .momentum()
            .x
            .value()
            .hypot(body.momentum().y.value())
            .hypot(body.momentum().z.value())
            / body.mass().value();
        let substeps = ((speed * dt / (world.layout().cell_width() * 0.1))
            .ceil()
            .max(1.) as usize)
            .min(512);
        let h = dt / substeps as f64;
        for _ in 0..substeps {
            let old = body;
            let mut load = ExternalLoadLedger::default();
            let body_mass = body.mass().value();
            load.advance_constant_force(
                &mut body,
                Vec3::<Newtons>::try_values(
                    0.,
                    -physics.gravity_metres_per_second_squared * body_mass,
                    0.,
                )?,
                TimeStep::from_seconds(h)?,
            )?;
            next.energy.gravity_work_joules += load.work().value();
            material.motion.update(body);
            if collision(world, material.motion.position_metres, &edits) {
                let energy = body.kinetic_energy()?.value();
                let old_position = old.position();
                let momentum = body.momentum();
                let damped = [
                    momentum.x.value() * 0.15,
                    momentum.y.value() * -0.15,
                    momentum.z.value() * 0.15,
                ];
                body = MechanicalBody::new(
                    body.mass(),
                    old_position,
                    Vec3::try_values(damped[0], damped[1], damped[2])?,
                )?;
                let remaining = body.kinetic_energy()?.value();
                next.energy.contact_heat_joules += energy - remaining;
                if remaining
                    < body.mass().value()
                        * physics.gravity_metres_per_second_squared
                        * world.layout().cell_width()
                        * 0.001
                {
                    next.energy.contact_heat_joules += remaining;
                    body = MechanicalBody::new(
                        body.mass(),
                        old_position,
                        Vec3::try_values(0., 0., 0.)?,
                    )?;
                    material.resting = true;
                    break;
                }
            }
        }
        material.motion.update(body);
    }
    let mut settled = vec![];
    let mut retained = vec![];
    let mut places = std::collections::BTreeSet::new();
    let bodies = std::mem::take(&mut next.bodies);
    for body in bodies {
        let definition = world
            .content()
            .block(world.content().block_id(&body.block)?)?;
        let material = definition.physics.mechanics;
        if material.is_some_and(|m| {
            m.crushing_energy_joules
                .is_some_and(|threshold| body.heat_joules >= threshold)
        }) {
            let energy = body.heat_joules + body.motion.body()?.kinetic_energy()?.value();
            next.energy.dispersed_joules += energy;
            next.dispersed_blocks += 1;
            next.dispersed_mass_kg += body.motion.mass_kg;
            let velocity = body.motion.momentum.map(|v| v / body.motion.mass_kg);
            next.emit_thermal_break(
                body.block,
                body.motion.position_metres,
                energy,
                material.unwrap().heat_capacity_joules_per_kelvin,
                velocity,
                material.unwrap().surface_emissivity,
            );
            next.revision += 1;
            continue;
        }

        let origin = cell(body.motion.position_metres, world.layout().cell_width())?;
        let destination = if body.resting && body.still_seconds >= 1.0 {
            settlement_cell(world, origin, occupied, &places)
        } else {
            None
        };
        if let Some(p) = destination {
            let xyz = [p.x, p.y as i64, p.z];
            places.insert(xyz);
            if body.heat_joules > 0.0 {
                if let Some(anchor) = next.anchors.iter_mut().find(|a| a.position == xyz) {
                    anchor.heat_joules += body.heat_joules;
                } else {
                    next.anchors.push(MaterialAnchor {
                        position: xyz,
                        heat_joules: body.heat_joules,
                    });
                }
            }
            settled.push(SettledMaterial {
                position: p,
                block: body.block,
                properties: body.properties,
                data: body.data,
            });
            next.revision += 1;
        } else {
            retained.push(body);
        }
    }
    next.bodies = retained;
    next.validate()?;
    Ok((
        next,
        MaterialChanges {
            detached: edits,
            settled,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        chunk::{ChunkCoord, ChunkLayout},
        registry::ContentRegistry,
    };
    use std::sync::Arc;
    fn world() -> ChunkWorld {
        let c = Arc::new(
            ContentRegistry::from_jsonl(
                include_str!("../../../assets/demo/blocks.jsonl"),
                include_str!("../../../assets/demo/items.jsonl"),
                include_str!("../../../assets/demo/entities.jsonl"),
            )
            .unwrap(),
        );
        let mut w = ChunkWorld::new(c, ChunkLayout::new([8, 8, 8], 1.0).unwrap());
        w.create(ChunkCoord { x: 0, z: 0 }, 3., true).unwrap();
        for z in 0..8 {
            for x in 0..8 {
                w.set_block(
                    CellPosition { x, y: 0, z },
                    &ResourceKey::parse("manonica:stone").unwrap(),
                    BTreeMap::new(),
                    NbtCompound::default(),
                )
                .unwrap();
            }
        }
        w
    }
    struct Empty;
    impl MomentumMedium for Empty {
        fn samples(&self) -> Vec<MediumSample> {
            vec![]
        }
        fn transfer(
            &mut self,
            _: MediumCellId,
            _: &mut MechanicalBody,
            _: f64,
            _: f64,
        ) -> WorldResult<f64> {
            unreachable!()
        }
    }
    struct Driven {
        velocity: [f64; 3],
    }
    impl MomentumMedium for Driven {
        fn samples(&self) -> Vec<MediumSample> {
            vec![MediumSample {
                id: MediumCellId { domain: 0, cell: 0 },
                position: [2.5, 1.5, 2.5],
                velocity: self.velocity,
            }]
        }
        fn transfer(
            &mut self,
            _: MediumCellId,
            body: &mut MechanicalBody,
            rate: f64,
            dt: f64,
        ) -> WorldResult<f64> {
            let mut packet = mechanical(0.75, [2.5, 1.5, 2.5], self.velocity)?;
            let before = body.kinetic_energy()?.value();
            manonica_physics::DissipativeExchange::new(Quantity::<
                manonica_core::units::PerSecond<manonica_core::units::Dimensionless>,
            >::nonnegative(rate)?)?
            .exchange(&mut packet, body, TimeStep::from_seconds(dt)?)?;
            Ok(body.kinetic_energy()?.value() - before)
        }
    }
    fn apply(world: &mut ChunkWorld, changes: MaterialChanges) {
        let air = world
            .content()
            .block(world.content().air())
            .unwrap()
            .key
            .clone();
        for p in changes.detached {
            world
                .set_block(p, &air, BTreeMap::new(), NbtCompound::default())
                .unwrap();
        }
        for b in changes.settled {
            world
                .set_block(b.position, &b.block, b.properties, b.data)
                .unwrap();
        }
    }
    #[test]
    fn ordinary_medium_detaches_a_block_and_gravity_settles_it_with_its_state() {
        let mut w = world();
        let p = CellPosition { x: 2, y: 1, z: 2 };
        let key = ResourceKey::parse("manonica:mana_crystal").unwrap();
        let properties = BTreeMap::from([("axis".into(), "x".into())]);
        let metadata = NbtCompound::new(BTreeMap::from([(
            "owner".into(),
            crate::metadata::NbtTag::String("test".into()),
        )]))
        .unwrap();
        w.set_block(p, &key, properties.clone(), metadata.clone())
            .unwrap();
        let initial = w.occupied_blocks().count();
        let mut state = MatterState::default();
        for _ in 0..3 {
            let (next, changes) = step(
                &w,
                &mut Driven {
                    velocity: [0., 80., 0.],
                },
                &state,
                WorldPhysics::default(),
                0.025,
                &[],
                &[],
            )
            .unwrap();
            state = next;
            apply(&mut w, changes);
            if !state.bodies.is_empty() {
                break;
            }
        }
        assert_eq!(state.bodies.len(), 1);
        assert_eq!(w.occupied_blocks().count() + state.bodies.len(), initial);
        assert_eq!(state.bodies[0].data, metadata);
        assert_eq!(state.bodies[0].properties, properties);
        for _ in 0..240 {
            let (next, changes) = step(
                &w,
                &mut Empty,
                &state,
                WorldPhysics::default(),
                0.025,
                &[],
                &[],
            )
            .unwrap();
            state = next;
            apply(&mut w, changes);
            if state.bodies.is_empty() {
                break;
            }
        }
        assert!(state.bodies.is_empty());
        assert_eq!(w.occupied_blocks().count(), initial);
        assert!(state.energy_residual_joules().unwrap().abs() < 1e-8);
        let found = w
            .occupied_blocks()
            .find(|(_, s)| w.content().block(s.block()).unwrap().key == key)
            .unwrap()
            .0;
        assert_eq!(w.block_data(found).unwrap(), metadata);
        assert_eq!(w.block(found).unwrap().properties(), &properties);
    }
    #[test]
    fn a_stationary_body_waits_a_full_second_and_does_not_overwrite_occupants() {
        let w = world();
        let mut state = MatterState::default();
        state.bodies.push(MaterialBody {
            block: ResourceKey::parse("manonica:stone").unwrap(),
            properties: BTreeMap::new(),
            data: NbtCompound::default(),
            motion: BodyMotion {
                position_metres: [2.5, 1.5, 2.5],
                momentum: [0.; 3],
                mass_kg: 20.,
            },
            heat_joules: 0.,
            age_seconds: 0.,
            drag_rate_per_second: 40.,
            resting: true,
            still_seconds: 0.,
        });
        for _ in 0..19 {
            let (next, changes) = step(
                &w,
                &mut Empty,
                &state,
                WorldPhysics::default(),
                0.05,
                &[],
                &[],
            )
            .unwrap();
            assert!(changes.settled.is_empty());
            state = next;
        }
        let (next, changes) = step(
            &w,
            &mut Empty,
            &state,
            WorldPhysics::default(),
            0.05,
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(changes.settled.len(), 1);
        assert!(next.bodies.is_empty());
        let occupied = [([2.5, 1.5, 2.5], [0.3, 0.5, 0.3])];
        let (next, changes) = step(
            &w,
            &mut Empty,
            &state,
            WorldPhysics::default(),
            0.05,
            &occupied,
            &[],
        )
        .unwrap();
        assert!(next.bodies.is_empty());
        assert_eq!(changes.settled.len(), 1);
        assert_ne!(
            changes.settled[0].position,
            CellPosition { x: 2, y: 1, z: 2 }
        );
    }
    #[test]
    fn reinsertion_search_skips_solid_cells_and_reservations() {
        let mut w = world();
        let origin = CellPosition { x: 2, y: 1, z: 2 };
        w.set_block(
            origin,
            &ResourceKey::parse("manonica:stone").unwrap(),
            BTreeMap::new(),
            Default::default(),
        )
        .unwrap();
        let mut reserved = std::collections::BTreeSet::new();
        let first = settlement_cell(&w, origin, &[], &reserved).unwrap();
        assert_ne!(first, origin);
        reserved.insert([first.x, first.y as i64, first.z]);
        let second = settlement_cell(&w, origin, &[], &reserved).unwrap();
        assert_ne!(first, second);
        assert_ne!(second, origin);
    }
    #[test]
    fn boxed_in_reinsertion_searches_beyond_local_radius_and_retries_a_full_world() {
        let content = std::sync::Arc::new(world().content().clone());
        let mut w = ChunkWorld::new(
            content,
            crate::chunk::ChunkLayout::new([20, 1, 1], 1.).unwrap(),
        );
        w.create(crate::chunk::ChunkCoord { x: 0, z: 0 }, 3., true)
            .unwrap();
        let stone = ResourceKey::parse("manonica:stone").unwrap();
        for x in 0..20 {
            w.set_block(
                CellPosition { x, y: 0, z: 0 },
                &stone,
                BTreeMap::new(),
                Default::default(),
            )
            .unwrap();
        }
        let origin = CellPosition { x: 0, y: 0, z: 0 };
        let reserved = std::collections::BTreeSet::new();
        assert!(settlement_cell(&w, origin, &[], &reserved).is_none());
        let distant = CellPosition { x: 19, y: 0, z: 0 };
        w.set_block(
            distant,
            &ResourceKey::parse("manonica:air").unwrap(),
            BTreeMap::new(),
            Default::default(),
        )
        .unwrap();
        assert_eq!(settlement_cell(&w, origin, &[], &reserved), Some(distant));
    }
    #[test]
    fn weak_load_does_not_remove_static_material() {
        let mut w = world();
        let p = CellPosition { x: 2, y: 1, z: 2 };
        w.set_block(
            p,
            &ResourceKey::parse("manonica:stone").unwrap(),
            BTreeMap::new(),
            NbtCompound::default(),
        )
        .unwrap();
        let state = MatterState::default();
        let (state, changes) = step(
            &w,
            &mut Driven {
                velocity: [0., 0.01, 0.],
            },
            &state,
            WorldPhysics::default(),
            0.025,
            &[],
            &[],
        )
        .unwrap();
        assert!(changes.detached.is_empty());
        assert!(state.bodies.is_empty());
        assert!(state.energy_residual_joules().unwrap().abs() < 1e-10);
    }
}
