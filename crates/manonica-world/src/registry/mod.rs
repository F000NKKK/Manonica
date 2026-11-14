//! Immutable, validated registries; persisted state uses namespaced keys, not IDs.
mod definitions;
mod key;
use crate::{WorldError, WorldResult};
pub use definitions::*;
pub use key::ResourceKey;
use serde::de::DeserializeOwned;
use std::collections::BTreeMap;
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct BlockId(pub(crate) u16);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ItemId(pub(crate) u16);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct EntityKindId(pub(crate) u16);
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockState {
    pub(crate) block: BlockId,
    pub(crate) properties: BTreeMap<String, String>,
}
impl BlockState {
    pub fn block(&self) -> BlockId {
        self.block
    }
    pub fn properties(&self) -> &BTreeMap<String, String> {
        &self.properties
    }
}
#[derive(Clone, Debug)]
pub struct ContentRegistry {
    blocks: Vec<BlockDefinition>,
    block_keys: BTreeMap<ResourceKey, BlockId>,
    items: Vec<ItemDefinition>,
    item_keys: BTreeMap<ResourceKey, ItemId>,
    entities: Vec<EntityDefinition>,
    entity_keys: BTreeMap<ResourceKey, EntityKindId>,
    block_items: BTreeMap<BlockId, ItemId>,
    air: BlockId,
    fingerprint: String,
}
fn lines<T: DeserializeOwned>(text: &str, source: &'static str) -> WorldResult<Vec<T>> {
    if text.len() > 4 * 1024 * 1024 {
        return Err(WorldError::Invalid("content_file_limit"));
    }
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(line, value)| {
            if value.len() > 65536 {
                return Err(WorldError::Content {
                    source,
                    line: line + 1,
                    detail: "definition exceeds line limit".into(),
                });
            }
            serde_json::from_str(value).map_err(|e| WorldError::Content {
                source,
                line: line + 1,
                detail: e.to_string(),
            })
        })
        .collect()
}
fn ordered<T>(values: Vec<T>, key: impl Fn(&T) -> &ResourceKey) -> WorldResult<Vec<T>> {
    if values.len() > 65535 {
        return Err(WorldError::Invalid("registry_capacity"));
    }
    let mut map = BTreeMap::new();
    for v in values {
        if map.insert(key(&v).clone(), v).is_some() {
            return Err(WorldError::Invalid("duplicate_resource_key"));
        }
    }
    Ok(map.into_values().collect())
}
impl ContentRegistry {
    pub fn from_jsonl(blocks: &str, items: &str, entities: &str) -> WorldResult<Self> {
        let blocks = ordered(lines::<BlockDefinition>(blocks, "blocks.jsonl")?, |b| {
            &b.key
        })?;
        let items = ordered(lines::<ItemDefinition>(items, "items.jsonl")?, |b| &b.key)?;
        let entities = ordered(
            lines::<EntityDefinition>(entities, "entities.jsonl")?,
            |b| &b.key,
        )?;
        let block_keys: BTreeMap<_, _> = blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (b.key.clone(), BlockId(i as u16)))
            .collect();
        let item_keys = items
            .iter()
            .enumerate()
            .map(|(i, b)| (b.key.clone(), ItemId(i as u16)))
            .collect();
        let entity_keys = entities
            .iter()
            .enumerate()
            .map(|(i, b)| (b.key.clone(), EntityKindId(i as u16)))
            .collect();
        for block in &blocks {
            if block.display.name.trim().is_empty()
                || !block.physics.mana.permeability.is_finite()
                || !(0.0..=1.0).contains(&block.physics.mana.permeability)
            {
                return Err(WorldError::Invalid("block_definition"));
            }
            block.metadata.validate()?;
            if block
                .appearance
                .base_color
                .iter()
                .chain(block.appearance.top_color.iter().flatten())
                .chain(block.appearance.emissive.iter())
                .any(|v| !v.is_finite() || *v < 0.0 || *v > 1.0)
            {
                return Err(WorldError::Invalid("block_appearance"));
            }
            if let Some(m) = block.physics.mechanics {
                validate_mechanics(m)?;
            }

            if block.state.allowed.len() != block.state.defaults.len() {
                return Err(WorldError::Invalid("block_state_defaults"));
            }
            for (key, allowed) in &block.state.allowed {
                if key.is_empty()
                    || allowed.is_empty()
                    || !block
                        .state
                        .defaults
                        .get(key)
                        .is_some_and(|v| allowed.contains(v))
                {
                    return Err(WorldError::Invalid("block_property_schema"));
                }
            }
        }
        let air = *block_keys
            .get(&ResourceKey::parse("manonica:air")?)
            .ok_or(WorldError::Invalid("missing_air_block"))?;
        let air_definition = &blocks[air.0 as usize];
        if air_definition.physics.collision != BlockCollision::Empty
            || !air_definition.physics.replaceable
            || air_definition.physics.mana.permeability != 1.0
        {
            return Err(WorldError::Invalid("air_properties"));
        }
        let mut block_items = BTreeMap::new();
        for (i, item) in items.iter().enumerate() {
            if item.display.name.trim().is_empty() || !(1..=1024).contains(&item.stack.limit) {
                return Err(WorldError::Invalid("item_definition"));
            }
            item.metadata.validate()?;
            if let Some(key) = &item.behavior.place_block {
                let id = *block_keys
                    .get(key)
                    .ok_or(WorldError::Invalid("unknown_placement_block"))?;
                if id == air || block_items.insert(id, ItemId(i as u16)).is_some() {
                    return Err(WorldError::Invalid("ambiguous_block_item"));
                }
            }
        }
        for entity in &entities {
            if entity.display.name.trim().is_empty()
                || !entity.physics.mass.is_finite()
                || entity.physics.mass <= 0.0
                || entity
                    .physics
                    .collider
                    .half_extents
                    .iter()
                    .any(|v| !v.is_finite() || *v <= 0.0 || *v > 16.0)
                || entity.inventory.slots > 256
            {
                return Err(WorldError::Invalid("entity_definition"));
            }
            entity.metadata.validate()?;
        }
        let canonical = serde_json::to_vec(&(&blocks, &items, &entities))
            .map_err(|e| WorldError::Codec(e.to_string()))?;
        let hash = canonical.iter().fold(0xcbf29ce484222325_u64, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
        });
        Ok(Self {
            blocks,
            block_keys,
            items,
            item_keys,
            entities,
            entity_keys,
            block_items,
            air,
            fingerprint: format!("{hash:016x}"),
        })
    }
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
    pub fn air(&self) -> BlockId {
        self.air
    }
    pub fn blocks(&self) -> &[BlockDefinition] {
        &self.blocks
    }
    pub fn items(&self) -> &[ItemDefinition] {
        &self.items
    }
    pub fn entities(&self) -> &[EntityDefinition] {
        &self.entities
    }
    pub fn block(&self, id: BlockId) -> WorldResult<&BlockDefinition> {
        self.blocks
            .get(id.0 as usize)
            .ok_or(WorldError::Invalid("block_id"))
    }
    pub fn item(&self, id: ItemId) -> WorldResult<&ItemDefinition> {
        self.items
            .get(id.0 as usize)
            .ok_or(WorldError::Invalid("item_id"))
    }
    pub fn entity(&self, id: EntityKindId) -> WorldResult<&EntityDefinition> {
        self.entities
            .get(id.0 as usize)
            .ok_or(WorldError::Invalid("entity_kind_id"))
    }
    pub fn block_id(&self, key: &ResourceKey) -> WorldResult<BlockId> {
        self.block_keys
            .get(key)
            .copied()
            .ok_or(WorldError::Invalid("unknown_block"))
    }
    pub fn item_id(&self, key: &ResourceKey) -> WorldResult<ItemId> {
        self.item_keys
            .get(key)
            .copied()
            .ok_or(WorldError::Invalid("unknown_item"))
    }
    pub fn entity_id(&self, key: &ResourceKey) -> WorldResult<EntityKindId> {
        self.entity_keys
            .get(key)
            .copied()
            .ok_or(WorldError::Invalid("unknown_entity"))
    }
    pub fn item_for_block(&self, id: BlockId) -> Option<ItemId> {
        self.block_items.get(&id).copied()
    }
    pub fn default_state(&self, key: &ResourceKey) -> WorldResult<BlockState> {
        let block = self.block_id(key)?;
        Ok(BlockState {
            block,
            properties: self.block(block)?.state.defaults.clone(),
        })
    }
    pub fn state(
        &self,
        key: &ResourceKey,
        properties: BTreeMap<String, String>,
    ) -> WorldResult<BlockState> {
        let block = self.block_id(key)?;
        let definition = self.block(block)?;
        if properties.len() != definition.state.allowed.len()
            || properties.iter().any(|(k, v)| {
                !definition
                    .state
                    .allowed
                    .get(k)
                    .is_some_and(|allowed| allowed.contains(v))
            })
        {
            return Err(WorldError::Invalid("block_state_properties"));
        }
        Ok(BlockState { block, properties })
    }
}

fn validate_mechanics(m: MaterialMechanics) -> WorldResult<()> {
    if !m.surface_emissivity.is_finite()
        || !(0.0..=1.0).contains(&m.surface_emissivity)
        || m.crushing_energy_joules
            .is_some_and(|v| v < m.fracture_energy_joules)
        || !m.thermal_contact_watts_per_kelvin.is_finite()
        || m.thermal_contact_watts_per_kelvin < 0.0
        || m.crushing_energy_joules
            .is_some_and(|v| !v.is_finite() || v < 0.0)
    {
        return Err(WorldError::Invalid("material_thermal_contact"));
    }
    if [m.mass_kg, m.heat_capacity_joules_per_kelvin]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.0)
        || [
            m.drag_rate_per_second,
            m.tensile_strength_newtons,
            m.fracture_energy_joules,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(WorldError::Invalid("material_mechanics"));
    }
    Ok(())
}
