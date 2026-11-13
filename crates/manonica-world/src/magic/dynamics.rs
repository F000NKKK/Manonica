//! Progressive voxel liberation and point-body dynamics driven by the reserved mana field.
use super::*;
use crate::{chunk::ChunkWorld, gameplay::solid, metadata::NbtCompound};
use manonica_core::{
    Quantity, TimeStep, Vec3,
    units::{KilogramMetresPerSecond, Kilograms, Metres, Newtons},
};
use manonica_physics::{ExternalLoadLedger, MechanicalBody};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct VoxelImpact {
    pub position: [i64; 3],
    pub at_seconds: f64,
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
            Quantity::<Kilograms>::nonnegative(self.mass_kg)?,
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
pub struct VoxelBody {
    pub block: ResourceKey,
    pub properties: BTreeMap<String, String>,
    pub data: NbtCompound,
    pub motion: BodyMotion,
    pub age_seconds: f64,
    pub resting: bool,
}
impl VoxelBody {
    pub(super) fn validate(&self) -> WorldResult<()> {
        self.motion.body()?;
        self.data.validate()?;
        if !self.age_seconds.is_finite()
            || self.age_seconds < 0.0
            || self.motion.position_metres.iter().any(|v| v.abs() > 1e9)
        {
            return Err(WorldError::Invalid("voxel_body"));
        }
        Ok(())
    }
}
pub(super) fn plan(
    world: &ChunkWorld,
    effect: SpellEffect,
    center: [f64; 3],
    direction: [f64; 3],
    charge: f64,
) -> WorldResult<Vec<VoxelImpact>> {
    let mut impacts = vec![];
    for (p, _) in world.occupied_blocks() {
        if p.y == 0 {
            continue;
        }
        let d = [
            p.x as f64 + 0.5 - center[0],
            p.y as f64 + 0.5 - center[1],
            p.z as f64 + 0.5 - center[2],
        ];
        let radial = d[0].hypot(d[2]);
        let at = match effect {
            SpellEffect::Explosion { radius } => {
                let r = d.iter().map(|v| v * v).sum::<f64>().sqrt();
                (r <= radius).then_some(0.15 + r / radius * 0.45)
            }
            SpellEffect::EarthRift {
                length,
                width,
                depth,
            } => {
                let along = d[0] * direction[0] + d[2] * direction[2];
                let across = d[0] * direction[2] - d[2] * direction[0];
                (along.abs() <= length * 0.5
                    && across.abs() <= width * 0.5
                    && d[1] <= 0.5
                    && d[1] >= -depth)
                    .then_some(0.25 + (along / length + 0.5) * 1.8 + (-d[1]).max(0.0) / depth * 0.4)
            }
            SpellEffect::Tornado { radius, .. } => {
                (radial <= radius && d[1] <= 1.0 && d[1] >= -3.5)
                    .then_some(0.3 + radial / radius * 0.5 + (-d[1]).max(0.0) * 0.45)
            }
        };
        if let Some(at_seconds) = at {
            impacts.push(VoxelImpact {
                position: [p.x, p.y as i64, p.z],
                at_seconds,
            });
        }
    }
    impacts.sort_by(|a, b| {
        a.at_seconds
            .total_cmp(&b.at_seconds)
            .then(a.position.cmp(&b.position))
    });
    impacts.truncate(((charge * 8.0) as usize).min(512));
    if let SpellEffect::Tornado { duration, .. } = effect {
        let count = impacts.len().max(1) as f64;
        for (i, impact) in impacts.iter_mut().enumerate() {
            impact.at_seconds = 0.3 + i as f64 / count * (duration - 0.7);
        }
    }
    Ok(impacts)
}
fn field_acceleration(events: &[SpellEvent], point: [f64; 3]) -> [f64; 3] {
    let mut a = [0.0; 3];
    for e in events {
        let d = std::array::from_fn::<_, 3, _>(|i| point[i] - e.center[i]);
        let radial = d[0].hypot(d[2]).max(0.3);
        let charge = e.reserved_manons / 16.0;
        if let SpellEffect::Tornado { radius, height, .. } = e.effect
            && radial < radius * 1.8
            && d[1] < height
            && d[1] > -4.0
        {
            let strength = (1.0 - radial / (radius * 1.8)).max(0.0) * charge;
            a[0] += (-d[2] / radial * 25.0 - d[0] * 8.0) * strength;
            a[2] += (d[0] / radial * 25.0 - d[2] * 8.0) * strength;
            a[1] += 45.0 * strength;
        }
    }
    a
}
fn collides(world: &ChunkWorld, point: [f64; 3]) -> bool {
    for z in (point[2] - 0.42).floor() as i64..=(point[2] + 0.42).floor() as i64 {
        for y in (point[1] - 0.42).floor() as i64..=(point[1] + 0.42).floor() as i64 {
            for x in (point[0] - 0.42).floor() as i64..=(point[0] + 0.42).floor() as i64 {
                if y < 0
                    || solid(
                        world,
                        CellPosition {
                            x,
                            y: y as usize,
                            z,
                        },
                        true,
                    )
                {
                    return true;
                }
            }
        }
    }
    false
}
pub(super) fn advance(world: &mut ChunkWorld, state: &mut MagicState, dt: f64) -> WorldResult<()> {
    if !dt.is_finite() || dt <= 0.0 || dt > 1.0 {
        return Err(WorldError::Invalid("magic_dt"));
    }
    let mut next = state.clone();
    let edits = next.active.iter().any(|e| {
        e.age_seconds + dt >= e.effect.duration()
            || e.impacts.iter().any(|i| i.at_seconds <= e.age_seconds + dt)
    });
    let mut candidate = edits.then(|| world.clone());
    if let Some(candidate) = candidate.as_mut() {
        let air = candidate
            .content()
            .block(candidate.content().air())?
            .key
            .clone();
        for event in &mut next.active {
            let mut pending = vec![];
            for impact in event.impacts.drain(..) {
                if impact.at_seconds > event.age_seconds + dt {
                    pending.push(impact);
                    continue;
                }
                if next.debris.len() >= 1024 {
                    continue;
                }
                let p = position(impact.position)?;
                let Ok(state) = candidate.block(p) else {
                    continue;
                };
                if state.block() == candidate.content().air() {
                    continue;
                }
                let definition = candidate.content().block(state.block())?;
                let block = definition.key.clone();
                let properties = state.properties().clone();
                let data = candidate.block_data(p)?;
                let center = [p.x as f64 + 0.5, p.y as f64 + 0.5, p.z as f64 + 0.5];
                let offset = std::array::from_fn::<_, 3, _>(|a| center[a] - event.center[a]);
                let distance = offset.iter().map(|v| v * v).sum::<f64>().sqrt().max(0.5);
                let strength = (event.reserved_manons / 12.0).sqrt().min(2.0);
                let velocity = match event.effect {
                    SpellEffect::Explosion { .. } => [
                        offset[0] / distance * 10.0 * strength,
                        (offset[1] / distance * 5.0 + 10.0) * strength,
                        offset[2] / distance * 10.0 * strength,
                    ],
                    SpellEffect::EarthRift { .. } => [
                        event.direction[2] * offset[0].signum() * 2.0 * strength,
                        7.0 * strength,
                        -event.direction[0] * offset[2].signum() * 2.0 * strength,
                    ],
                    SpellEffect::Tornado { .. } => [-offset[2] * 2.0, 8.0, offset[0] * 2.0],
                };
                let body = VoxelBody {
                    block,
                    properties,
                    data,
                    motion: BodyMotion {
                        position_metres: center,
                        momentum: velocity,
                        mass_kg: 1.0,
                    },
                    age_seconds: 0.0,
                    resting: false,
                };
                next.mechanical_work_joules += body.motion.body()?.kinetic_energy()?.value();
                candidate.set_block(p, &air, BTreeMap::new(), NbtCompound::default())?;
                next.debris.push(body);
                event.changed_blocks += 1;
                next.revision += 1;
            }
            event.impacts = pending;
            if event.age_seconds + dt >= event.effect.duration() {
                let p = position(event.source)?;
                candidate.set_density(
                    p,
                    candidate.density(p)?
                        + event.reserved_manons / candidate.layout().cell_width().powi(3),
                )?;
            }
        }
    }
    let reference = candidate.as_ref().unwrap_or(world);
    for debris in &mut next.debris {
        debris.age_seconds += dt;
        if debris.resting {
            continue;
        }
        let substeps = (dt / (1.0 / 120.0)).ceil() as usize;
        let subdt = dt / substeps as f64;
        let mut body = debris.motion.body()?;
        for _ in 0..substeps {
            let old = debris.motion.position_metres;
            let a = field_acceleration(&next.active, old);
            let mut load = ExternalLoadLedger::default();
            load.advance_constant_force(
                &mut body,
                Vec3::<Newtons>::try_values(a[0], a[1] - 20.0, a[2])?,
                TimeStep::from_seconds(subdt)?,
            )?;
            next.mechanical_work_joules += load.work().value();
            debris.motion.update(body);
            if collides(reference, debris.motion.position_metres) {
                let mut position = debris.motion.position_metres;
                let mut momentum = debris.motion.momentum;
                for axis in [0, 2, 1] {
                    let trial = position[axis];
                    position[axis] = old[axis];
                    if !collides(reference, position) {
                        momentum[axis] *= -0.2;
                    } else {
                        position[axis] = trial;
                    }
                }
                if collides(reference, position) {
                    position = old;
                    momentum = momentum.map(|v| v * 0.2);
                }
                debris.motion.position_metres = position;
                debris.motion.momentum = momentum;
                body = debris.motion.body()?;
                if momentum.iter().map(|v| v * v).sum::<f64>() < 0.04 && a[1] < 20.0 {
                    debris.resting = true;
                    break;
                }
            }
        }
    }
    for event in &mut next.active {
        event.age_seconds += dt;
    }
    next.active.retain(|e| e.age_seconds < e.effect.duration());
    next.bound_manons = next.active.iter().map(|e| e.reserved_manons).sum();
    next.validate()?;
    if let Some(candidate) = candidate {
        *world = candidate;
    }
    *state = next;
    Ok(())
}
