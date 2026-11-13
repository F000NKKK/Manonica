//! Data-driven game magic. Reserved manons remain accounted for until an effect expires.
//! Spell geometry is a demo game rule, not a validated continuum destruction model.
use crate::{
    WorldError, WorldResult,
    chunk::CellPosition,
    registry::{ContentRegistry, ResourceKey},
    runtime::{EntityId, GameRuntime},
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
mod dynamics;
pub use dynamics::{BodyMotion, VoxelBody};
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CastingRules {
    pub mana_cost: f64,
    pub range: f64,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SpellEffect {
    Explosion {
        radius: f64,
    },
    EarthRift {
        length: f64,
        width: f64,
        depth: f64,
    },
    Tornado {
        radius: f64,
        height: f64,
        duration: f64,
    },
}
impl SpellEffect {
    pub fn duration(self) -> f64 {
        match self {
            Self::Explosion { .. } => 1.5,
            Self::EarthRift { .. } => 3.0,
            Self::Tornado { duration, .. } => duration,
        }
    }
    pub fn index(self) -> usize {
        match self {
            Self::Explosion { .. } => 0,
            Self::EarthRift { .. } => 1,
            Self::Tornado { .. } => 2,
        }
    }
    fn valid(self) -> bool {
        let valid = |v: f64| v.is_finite() && v > 0.0 && v <= 32.0;
        match self {
            Self::Explosion { radius } => valid(radius),
            Self::EarthRift {
                length,
                width,
                depth,
            } => valid(length) && valid(width) && valid(depth),
            Self::Tornado {
                radius,
                height,
                duration,
            } => valid(radius) && valid(height) && valid(duration),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpellDefinition {
    pub key: ResourceKey,
    pub scroll: ResourceKey,
    pub casting: CastingRules,
    pub effect: SpellEffect,
}
#[derive(Clone, Debug)]
pub struct SpellRegistry {
    by_scroll: BTreeMap<ResourceKey, SpellDefinition>,
}
impl SpellRegistry {
    pub fn from_jsonl(text: &str, content: &ContentRegistry) -> WorldResult<Self> {
        if text.len() > 1024 * 1024 {
            return Err(WorldError::Invalid("spell_content_limit"));
        }
        let mut by_scroll = BTreeMap::new();
        let mut keys = BTreeSet::new();
        for (index, line) in text
            .lines()
            .enumerate()
            .filter(|(_, s)| !s.trim().is_empty())
        {
            if line.len() > 65536 {
                return Err(WorldError::Invalid("spell_definition_limit"));
            }
            let spell: SpellDefinition =
                serde_json::from_str(line).map_err(|e| WorldError::Content {
                    source: "spells.jsonl",
                    line: index + 1,
                    detail: e.to_string(),
                })?;
            content.item_id(&spell.scroll)?;
            if !spell.effect.valid()
                || !spell.casting.mana_cost.is_finite()
                || spell.casting.mana_cost <= 0.0
                || spell.casting.mana_cost > 10000.0
                || !spell.casting.range.is_finite()
                || spell.casting.range <= 0.0
                || spell.casting.range > 64.0
                || !keys.insert(spell.key.clone())
                || by_scroll.insert(spell.scroll.clone(), spell).is_some()
            {
                return Err(WorldError::Invalid("spell_definition"));
            }
        }
        Ok(Self { by_scroll })
    }
    pub fn for_scroll(&self, key: &ResourceKey) -> WorldResult<&SpellDefinition> {
        self.by_scroll
            .get(key)
            .ok_or(WorldError::Invalid("not_a_spell_scroll"))
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpellEvent {
    pub spell: ResourceKey,
    pub effect: SpellEffect,
    pub center: [f64; 3],
    pub direction: [f64; 3],
    pub age_seconds: f64,
    pub reserved_manons: f64,
    pub source: [i64; 3],
    pub changed_blocks: usize,
    pub(crate) impacts: Vec<dynamics::VoxelImpact>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagicState {
    pub(crate) bound_manons: f64,
    pub(crate) casts: u64,
    pub(crate) active: Vec<SpellEvent>,
    pub(crate) debris: Vec<dynamics::VoxelBody>,
    pub(crate) revision: u64,
    pub(crate) mechanical_work_joules: f64,
}
impl MagicState {
    pub fn debris(&self) -> &[dynamics::VoxelBody] {
        &self.debris
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn mechanical_work_joules(&self) -> f64 {
        self.mechanical_work_joules
    }
    pub fn bound_manons(&self) -> f64 {
        self.bound_manons
    }
    pub fn casts(&self) -> u64 {
        self.casts
    }
    pub fn active(&self) -> &[SpellEvent] {
        &self.active
    }
    pub(crate) fn validate(&self) -> WorldResult<()> {
        if self.debris.len() > 1024 || !self.mechanical_work_joules.is_finite() {
            return Err(WorldError::Invalid("magic_debris"));
        }
        for body in &self.debris {
            body.validate()?;
        }
        if self.active.len() > 16 || !self.bound_manons.is_finite() || self.bound_manons < 0.0 {
            return Err(WorldError::Invalid("magic_state"));
        }
        for e in &self.active {
            if e.impacts.len() > 512
                || e.impacts.iter().any(|i| {
                    !i.at_seconds.is_finite()
                        || i.at_seconds < e.age_seconds
                        || i.at_seconds >= e.effect.duration()
                        || i.position[1] < 0
                })
            {
                return Err(WorldError::Invalid("magic_plan"));
            }
            if !e.effect.valid()
                || e.center
                    .iter()
                    .chain(e.direction.iter())
                    .any(|v| !v.is_finite() || v.abs() > 1e9)
                || !e.age_seconds.is_finite()
                || e.age_seconds < 0.0
                || e.age_seconds >= e.effect.duration()
                || !e.reserved_manons.is_finite()
                || e.reserved_manons <= 0.0
                || e.source[1] < 0
            {
                return Err(WorldError::Invalid("spell_event"));
            }
        }
        let sum: f64 = self.active.iter().map(|e| e.reserved_manons).sum();
        if (sum - self.bound_manons).abs() > 1e-10 * sum.max(1.0) {
            return Err(WorldError::Invalid("magic_balance"));
        }
        Ok(())
    }
}
fn position(v: [i64; 3]) -> WorldResult<CellPosition> {
    Ok(CellPosition {
        x: v[0],
        y: usize::try_from(v[1]).map_err(|_| WorldError::Invalid("spell_position"))?,
        z: v[2],
    })
}
impl GameRuntime {
    pub fn magic(&self) -> &MagicState {
        &self.magic
    }
    pub fn cast_scroll(
        &mut self,
        registry: &SpellRegistry,
        caster: EntityId,
        slot: usize,
        origin: [f64; 3],
        target: [f64; 3],
        direction: [f64; 3],
    ) -> WorldResult<SpellEvent> {
        if self.magic.active.len() >= 16
            || origin
                .iter()
                .chain(target.iter())
                .chain(direction.iter())
                .any(|v| !v.is_finite() || v.abs() > 1e8)
        {
            return Err(WorldError::Invalid("spell_input"));
        }
        let entity = self.entity(caster)?;
        let stack = entity
            .inventory
            .get(slot)?
            .ok_or(WorldError::Invalid("empty_inventory_slot"))?;
        let key = &self.world.content().item(stack.item())?.key;
        let spell = registry.for_scroll(key)?;
        let distance = origin
            .iter()
            .zip(target)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        if distance > spell.casting.range {
            return Err(WorldError::Invalid("spell_out_of_range"));
        }
        let length = direction[0].hypot(direction[2]);
        if length < 1e-8 {
            return Err(WorldError::Invalid("spell_direction"));
        }
        let direction = [direction[0] / length, 0.0, direction[2] / length];
        let replacement = entity.inventory.after_consuming(slot, 1)?;
        let mut candidate = self.world.clone();
        let source = origin.map(|v| v.floor() as i64);
        candidate.density(position(source)?)?;
        let mut remaining = spell.casting.mana_cost;
        let volume = candidate.layout().cell_width().powi(3);
        for dz in -2..=2 {
            for dy in -2..=2 {
                for dx in -2..=2 {
                    if remaining <= 1e-12 {
                        continue;
                    }
                    let p = [source[0] + dx, source[1] + dy, source[2] + dz];
                    let Ok(p) = position(p) else { continue };
                    let Ok(density) = candidate.density(p) else {
                        continue;
                    };
                    let amount = (density * volume).min(remaining);
                    if amount > 0.0 {
                        candidate.set_density(p, (density - amount / volume).max(0.0))?;
                        remaining -= amount;
                    }
                }
            }
        }
        if remaining > 1e-9 {
            return Err(WorldError::Invalid("insufficient_ambient_mana"));
        }
        let impacts = dynamics::plan(
            &candidate,
            spell.effect,
            target,
            direction,
            spell.casting.mana_cost,
        )?;
        let event = SpellEvent {
            spell: spell.key.clone(),
            effect: spell.effect,
            center: target,
            direction,
            age_seconds: 0.0,
            reserved_manons: spell.casting.mana_cost,
            source,
            changed_blocks: 0,
            impacts,
        };
        let casts = self
            .magic
            .casts
            .checked_add(1)
            .ok_or(WorldError::Invalid("spell_counter"))?;
        self.world = candidate;
        self.entities
            .get_mut(&caster)
            .unwrap()
            .inventory
            .commit_slot(slot, replacement);
        self.magic.bound_manons += event.reserved_manons;
        self.magic.casts = casts;
        self.magic.active.push(event.clone());
        Ok(event)
    }
    pub fn advance_magic(&mut self, dt: f64) -> WorldResult<()> {
        dynamics::advance(&mut self.world, &mut self.magic, dt)
    }
    pub fn magic_acceleration(&self, point: [f64; 3]) -> [f64; 3] {
        let mut a = [0.0; 3];
        for e in &self.magic.active {
            let delta = std::array::from_fn::<_, 3, _>(|i| point[i] - e.center[i]);
            let radial = delta[0].hypot(delta[2]).max(0.3);
            match e.effect {
                SpellEffect::Tornado { radius, height, .. }
                    if radial < radius && delta[1] > -1.0 && delta[1] < height =>
                {
                    let charge = (e.reserved_manons / 16.0).min(2.0);
                    let strength = (1.0 - radial / radius) * 25.0 * charge;
                    a[0] += (-delta[2] / radial - delta[0] / radial * 0.5) * strength;
                    a[2] += (delta[0] / radial - delta[2] / radial * 0.5) * strength;
                    a[1] += 36.0 * (1.0 - radial / radius) * charge;
                }
                SpellEffect::Explosion { radius } if e.age_seconds < 0.25 => {
                    let r = delta.iter().map(|v| v * v).sum::<f64>().sqrt().max(0.3);
                    if r < radius * 2.0 {
                        for i in 0..3 {
                            a[i] += delta[i] / r
                                * 50.0
                                * (1.0 - r / (radius * 2.0))
                                * (e.reserved_manons / 12.0).min(2.0);
                        }
                    }
                }
                _ => {}
            }
        }
        a
    }
}
