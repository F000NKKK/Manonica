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
    pub minimum_dynamic_height: usize,
    pub carrier_mass_kg_per_manon: f64,
    pub gravity_metres_per_second_squared: f64,
}
impl Default for WorldPhysics {
    fn default() -> Self {
        Self {
            minimum_dynamic_height: 1,
            carrier_mass_kg_per_manon: 0.25,
            gravity_metres_per_second_squared: 20.0,
        }
    }
}
impl WorldPhysics {
    pub(crate) fn validate(self) -> WorldResult<()> {
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
    pub mechanical_work_received_joules: f64,
    pub gravity_work_joules: f64,
    pub surface_joules: f64,
    pub contact_heat_joules: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialParticles {
    pub block: ResourceKey,
    pub position: [f64; 3],
    pub age_seconds: f64,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatterState {
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
            age_seconds: 0.0,
        });
    }

    pub fn stored_energy_joules(&self) -> WorldResult<f64> {
        let mut stored = self.energy.surface_joules
            + self.energy.contact_heat_joules
            + self.anchors.iter().map(|a| a.heat_joules).sum::<f64>();
        for b in &self.bodies {
            stored += b.heat_joules + b.motion.body()?.kinetic_energy()?.value();
        }
        Ok(stored)
    }
    pub fn energy_residual_joules(&self) -> WorldResult<f64> {
        Ok(self.stored_energy_joules()?
            - self.energy.mechanical_work_received_joules
            - self.energy.gravity_work_joules)
    }
    pub(crate) fn validate(&self) -> WorldResult<()> {
        if self.particles.len() > 128
            || self.particles.iter().any(|p| {
                !p.age_seconds.is_finite()
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
pub(crate) fn step(
    world: &ChunkWorld,
    medium: &mut impl MomentumMedium,
    previous: &MatterState,
    physics: WorldPhysics,
    dt: f64,
    occupied: &[([f64; 3], [f64; 3])],
) -> WorldResult<(MatterState, MaterialChanges)> {
    let mut next = previous.clone();
    for event in &mut next.particles {
        event.age_seconds += dt;
    }
    next.particles.retain(|p| p.age_seconds < 1.25);
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
        if sample.velocity.iter().map(|v| v * v).sum::<f64>() < 1e-8 {
            continue;
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
    next.anchors = anchors
        .into_iter()
        .filter(|(_, heat)| *heat > 0.)
        .map(|(position, heat_joules)| MaterialAnchor {
            position,
            heat_joules,
        })
        .collect();
    for material in &mut next.bodies {
        let mut body = material.motion.body()?;
        if let Ok(target) = cell(material.motion.position_metres, world.layout().cell_width()) {
            for sample in &samples {
                let Ok(p) = cell(sample.position, world.layout().cell_width()) else {
                    continue;
                };
                let distance = (p.x - target.x).abs()
                    + (p.y as i64 - target.y as i64).abs()
                    + (p.z - target.z).abs();
                if distance > 1 {
                    continue;
                }
                let weight = if distance == 0 { 1.0 } else { 0.5 };
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
    for body in next.bodies.drain(..) {
        let p = cell(body.motion.position_metres, world.layout().cell_width())?;
        let xyz = [p.x, p.y as i64, p.z];
        let lo = xyz.map(|v| v as f64 * world.layout().cell_width());
        let entity_overlap = occupied.iter().any(|(center, half)| {
            (0..3).all(|a| {
                center[a] + half[a] > lo[a] + 1e-6
                    && center[a] - half[a] < lo[a] + world.layout().cell_width() - 1e-6
            })
        });
        let can_place = body.resting
            && body.still_seconds >= 1.0
            && !entity_overlap
            && world.block(p).is_ok_and(|s| {
                world
                    .content()
                    .block(s.block())
                    .is_ok_and(|d| d.physics.replaceable)
            })
            && places.insert(xyz);
        if can_place {
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
            let (next, changes) =
                step(&w, &mut Empty, &state, WorldPhysics::default(), 0.025, &[]).unwrap();
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
            let (next, changes) =
                step(&w, &mut Empty, &state, WorldPhysics::default(), 0.05, &[]).unwrap();
            assert!(changes.settled.is_empty());
            state = next;
        }
        let (next, changes) =
            step(&w, &mut Empty, &state, WorldPhysics::default(), 0.05, &[]).unwrap();
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
        )
        .unwrap();
        assert_eq!(next.bodies.len(), 1);
        assert!(changes.settled.is_empty());
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
        )
        .unwrap();
        assert!(changes.detached.is_empty());
        assert!(state.bodies.is_empty());
        assert!(state.energy_residual_joules().unwrap().abs() < 1e-10);
    }
}
