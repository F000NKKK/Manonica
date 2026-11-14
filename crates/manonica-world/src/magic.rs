//! Formula-driven control of manon fields. No spell names or voxel mutation rules.
pub mod construction;
mod field;
mod program;
use crate::{
    WorldError, WorldResult,
    chunk::CellPosition,
    registry::{ContentRegistry, ResourceKey},
    runtime::{EntityId, GameRuntime},
};
pub use field::{FieldEnergy, ManonField};
pub use program::{
    FieldControls, FormulaDomain, ManonProgram, ScalarFormula, TransportSettings, VectorFormula,
    WorkBudget,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CastingRules {
    pub range: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpellDefinition {
    pub key: ResourceKey,
    pub scroll: ResourceKey,
    pub casting: CastingRules,
    pub program: ManonProgram,
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
        for (i, line) in text
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
                    line: i + 1,
                    detail: e.to_string(),
                })?;
            content.item_id(&spell.scroll)?;
            spell.program.compile()?;
            if !spell.casting.range.is_finite()
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
    pub id: u64,
    pub spell: ResourceKey,
    pub program: ManonProgram,
    pub center: [f64; 3],
    pub direction: [f64; 3],
    pub age_seconds: f64,
    pub field: ManonField,
}
impl SpellEvent {
    pub fn duration(&self) -> f64 {
        self.program.domain.duration_seconds
    }
    fn cell(&self, i: usize) -> WorldResult<CellPosition> {
        let [nx, ny, _] = self.field.dimensions;
        let y = self.field.origin[1] + ((i / nx) % ny) as i64;
        Ok(CellPosition {
            x: self.field.origin[0] + (i % nx) as i64,
            y: usize::try_from(y).map_err(|_| WorldError::Invalid("spell_cell"))?,
            z: self.field.origin[2] + (i / (nx * ny)) as i64,
        })
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReleasedEnergy {
    pub heat_joules: f64,
    pub external_work_joules: f64,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagicState {
    casts: u64,
    active: Vec<SpellEvent>,
    pub released: ReleasedEnergy,
}
impl MagicState {
    pub fn casts(&self) -> u64 {
        self.casts
    }
    pub fn active(&self) -> &[SpellEvent] {
        &self.active
    }
    /// Manons owned by active solver domains, not a separate magical source.
    pub fn bound_manons(&self) -> f64 {
        self.active.iter().map(|e| e.field.manons()).sum()
    }
    pub fn energy_residual_joules(&self) -> f64 {
        self.released.heat_joules - self.released.external_work_joules
            + self
                .active
                .iter()
                .map(|e| e.field.energy.residual_joules())
                .sum::<f64>()
    }
    pub(crate) fn validate(&self) -> WorldResult<()> {
        if self.active.len() > 8
            || !self.bound_manons().is_finite()
            || [
                self.released.heat_joules,
                self.released.external_work_joules,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(WorldError::Invalid("magic_state"));
        }
        let mut ids = BTreeSet::new();
        for e in &self.active {
            e.program.compile()?;
            e.field.validate()?;
            let norm = e.direction[0].hypot(e.direction[2]);
            if !ids.insert(e.id)
                || e.id == 0
                || e.id > self.casts
                || e.field.dimensions != e.program.domain.dimensions
                || e.field.origin[1] < 0
                || e.center
                    .iter()
                    .chain(e.direction.iter())
                    .any(|v| !v.is_finite() || v.abs() > 1e8)
                || (norm - 1.0).abs() > 1e-10
                || e.direction[1] != 0.0
                || !e.age_seconds.is_finite()
                || e.age_seconds < 0.0
                || e.age_seconds >= e.duration()
                || e.field.simulated_seconds > e.age_seconds + 1e-9
                || e.field.energy.external_work_joules > e.program.work.total_joules + 1e-8
            {
                return Err(WorldError::Invalid("spell_event"));
            }
        }
        for (i, a) in self.active.iter().enumerate() {
            for b in &self.active[i + 1..] {
                if overlap(&a.field, &b.field) {
                    return Err(WorldError::Invalid("overlapping_spell_domains"));
                }
            }
        }
        if self.energy_residual_joules().abs()
            > 1e-7
                * (self.released.external_work_joules
                    + self
                        .active
                        .iter()
                        .map(|e| {
                            e.field.energy.external_work_joules
                                + e.field.energy.potential_joules.abs()
                        })
                        .sum::<f64>()
                    + 1.0)
        {
            return Err(WorldError::Invalid("magic_energy_balance"));
        }
        Ok(())
    }
}
fn overlap(a: &ManonField, b: &ManonField) -> bool {
    (0..3).all(|i| {
        a.origin[i] < b.origin[i] + b.dimensions[i] as i64
            && b.origin[i] < a.origin[i] + a.dimensions[i] as i64
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
        if self.magic.active.len() >= 8
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
        let spell = registry.for_scroll(&self.world.content().item(stack.item())?.key)?;
        let distance = origin
            .iter()
            .zip(target)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        if distance > spell.casting.range {
            return Err(WorldError::Invalid("spell_out_of_range"));
        }
        let norm = direction[0].hypot(direction[2]);
        if norm < 1e-8 {
            return Err(WorldError::Invalid("spell_direction"));
        }
        let direction = [direction[0] / norm, 0., direction[2] / norm];
        let replacement = entity.inventory.after_consuming(slot, 1)?;
        let domain = spell.program.domain;
        let dims = domain.dimensions;
        let dx = self.world.layout().cell_width();
        if target
            .iter()
            .any(|v| !(*v / dx).is_finite() || (*v / dx).abs() > 99_000_000.0)
        {
            return Err(WorldError::Invalid("spell_domain_coordinate"));
        }
        let first = std::array::from_fn(|i| {
            (target[i] / dx).floor() as i64
                - ((dims[i] - 1) as f64 * domain.anchor[i]).floor() as i64
        });
        let len = dims.iter().product();
        let field = ManonField {
            origin: first,
            dimensions: dims,
            cell_width: dx,
            densities: vec![0.; len],
            reference_density: vec![0.; len],
            potential: vec![0.; len],
            sources: vec![0.; len],
            velocity: vec![[0.; 3]; len],
            simulated_seconds: 0.,
            energy: FieldEnergy::default(),
        };
        if self.magic.active.iter().any(|e| overlap(&field, &e.field)) {
            return Err(WorldError::Invalid("overlapping_spell_domains"));
        }
        let id = self
            .magic
            .casts
            .checked_add(1)
            .ok_or(WorldError::Invalid("spell_counter"))?;
        let mut event = SpellEvent {
            id,
            spell: spell.key.clone(),
            program: spell.program.clone(),
            center: target,
            direction,
            age_seconds: 0.,
            field,
        };
        let mut zero = Vec::with_capacity(len);
        for i in 0..len {
            let pos = event.cell(i)?;
            event.field.densities[i] = self.world.density(pos)?;
            zero.push((pos, 0.));
        }
        event.field.reference_density = event.field.densities.clone();
        if event.field.manons() <= 0.0 {
            return Err(WorldError::Invalid("insufficient_ambient_mana"));
        }
        // Evaluate the first physical step before consuming a scroll or reserving world state.
        let mut probe = event.clone();
        field::step(&mut probe, 0.025_f64.min(event.duration()), false)?;
        self.world.replace_cells(&zero)?;
        self.entities
            .get_mut(&caster)
            .unwrap()
            .inventory
            .commit_slot(slot, replacement);
        self.magic.casts = id;
        self.magic.active.push(event.clone());
        Ok(event)
    }
    pub fn advance_magic(&mut self, dt: f64) -> WorldResult<()> {
        if !dt.is_finite() || dt <= 0.0 || dt > 0.1 {
            return Err(WorldError::Invalid("magic_dt"));
        }
        let mut next = self.magic.clone();
        let mut restore = vec![];
        let mut active = vec![];
        for mut event in next.active.drain(..) {
            event.age_seconds = (event.age_seconds + dt).min(event.duration());
            let finishing = event.age_seconds >= event.duration();
            while event.field.simulated_seconds + 0.025 <= event.age_seconds + 1e-12 {
                field::step(&mut event, 0.025, false)?;
            }
            if finishing {
                let remaining = (event.age_seconds - event.field.simulated_seconds).max(0.0);
                field::step(&mut event, remaining.max(0.025), true)?;
                for (i, n) in event.field.densities.iter().enumerate() {
                    let cell = event.cell(i)?;
                    restore.push((cell, self.world.density(cell)? + n));
                }
                next.released.heat_joules += event.field.energy.heat_joules;
                next.released.external_work_joules += event.field.energy.external_work_joules;
            } else {
                active.push(event);
            }
        }
        next.active = active;
        next.validate()?;
        self.world.replace_cells(&restore)?;
        self.magic = next;
        Ok(())
    }
}
