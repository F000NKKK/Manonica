//! Headless demo game state. Transport currently uses the CPU reference solver.
use crate::{
    WorldError, WorldResult,
    chunk::{CellPosition, ChunkLayout, ChunkWorld},
    inventory::{Inventory, ItemStack, SavedInventory},
    metadata::{NbtCompound, decode_nbt, validate_wire},
    registry::{ContentRegistry, EntityKindId, ResourceKey},
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, sync::Arc};
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct EntityId(u64);
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameClock {
    pub ticks: u64,
    pub elapsed_seconds: f64,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntityTransform {
    pub position_metres: [f64; 3],
    pub rotation_radians: [f64; 3],
}
impl EntityTransform {
    fn validate(self) -> WorldResult<()> {
        if self
            .position_metres
            .iter()
            .chain(self.rotation_radians.iter())
            .any(|v| !v.is_finite())
        {
            return Err(WorldError::Invalid("entity_transform"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct EntityState {
    pub kind: EntityKindId,
    pub transform: EntityTransform,
    pub inventory: Inventory,
    pub data: NbtCompound,
}
#[derive(Clone, Debug)]
pub struct GameRuntime {
    pub world: ChunkWorld,
    pub(crate) entities: BTreeMap<EntityId, EntityState>,
    pub(crate) magic: crate::magic::MagicState,
    clock: GameClock,
    next_id: u64,
}
impl GameRuntime {
    pub fn new(content: Arc<ContentRegistry>, layout: ChunkLayout) -> Self {
        Self {
            world: ChunkWorld::new(content, layout),
            entities: BTreeMap::new(),
            magic: crate::magic::MagicState::default(),
            clock: GameClock::default(),
            next_id: 1,
        }
    }
    pub fn clock(&self) -> GameClock {
        self.clock
    }
    pub fn entities(&self) -> &BTreeMap<EntityId, EntityState> {
        &self.entities
    }
    pub fn entity(&self, id: EntityId) -> WorldResult<&EntityState> {
        self.entities
            .get(&id)
            .ok_or(WorldError::Invalid("unknown_entity"))
    }
    pub fn set_transform(&mut self, id: EntityId, transform: EntityTransform) -> WorldResult<()> {
        transform.validate()?;
        self.entities
            .get_mut(&id)
            .ok_or(WorldError::Invalid("unknown_entity"))?
            .transform = transform;
        Ok(())
    }
    pub fn spawn(&mut self, key: &ResourceKey) -> WorldResult<EntityId> {
        self.spawn_at(key, EntityTransform::default())
    }
    pub fn spawn_at(
        &mut self,
        key: &ResourceKey,
        transform: EntityTransform,
    ) -> WorldResult<EntityId> {
        transform.validate()?;
        let kind = self.world.content().entity_id(key)?;
        let definition = self.world.content().entity(kind)?;
        let state = EntityState {
            kind,
            transform,
            inventory: Inventory::new(definition.inventory.slots)?,
            data: definition.metadata.clone(),
        };
        let id = EntityId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(WorldError::Invalid("entity_id_exhausted"))?;
        self.entities.insert(id, state);
        Ok(id)
    }
    pub fn give(
        &mut self,
        id: EntityId,
        slot: usize,
        key: &ResourceKey,
        count: u16,
    ) -> WorldResult<()> {
        let content = self.world.content();
        let definition = content.item(content.item_id(key)?)?;
        let stack = ItemStack::new(content, key, count, definition.metadata.clone())?;
        self.entities
            .get_mut(&id)
            .ok_or(WorldError::Invalid("unknown_entity"))?
            .inventory
            .insert(content, slot, stack)
    }
    pub fn place(&mut self, id: EntityId, slot: usize, pos: CellPosition) -> WorldResult<()> {
        let entity = self.entity(id)?;
        let stack = entity
            .inventory
            .get(slot)?
            .ok_or(WorldError::Invalid("empty_inventory_slot"))?;
        let content = self.world.content();
        let key = content
            .item(stack.item())?
            .behavior
            .place_block
            .clone()
            .ok_or(WorldError::Invalid("item_cannot_place"))?;
        if !content
            .block(self.world.block(pos)?.block())?
            .physics
            .replaceable
        {
            return Err(WorldError::Invalid("block_not_replaceable"));
        }
        let definition = content.block(content.block_id(&key)?)?;
        let properties = definition.state.defaults.clone();
        let mut data = definition.metadata.clone();
        for (key, value) in stack.data().entries() {
            data.insert(key.clone(), value.clone())?;
        }
        let replacement = entity.inventory.after_consuming(slot, 1)?;
        self.world.set_block(pos, &key, properties, data)?;
        self.entities
            .get_mut(&id)
            .unwrap()
            .inventory
            .commit_slot(slot, replacement);
        Ok(())
    }
    pub fn break_block(&mut self, id: EntityId, slot: usize, pos: CellPosition) -> WorldResult<()> {
        let content = self.world.content();
        let state = self.world.block(pos)?;
        let item = content
            .item_for_block(state.block())
            .ok_or(WorldError::Invalid("block_has_no_drop"))?;
        let stack = ItemStack::new(
            content,
            &content.item(item)?.key,
            1,
            self.world.block_data(pos)?,
        )?;
        let mut inventory = self.entity(id)?.inventory.clone();
        inventory.insert(content, slot, stack)?;
        let air = content.block(content.air())?.key.clone();
        self.world
            .set_block(pos, &air, BTreeMap::new(), NbtCompound::default())?;
        self.entities.get_mut(&id).unwrap().inventory = inventory;
        Ok(())
    }
    pub fn tick(&mut self, diffusivity: f64, dt: f64) -> WorldResult<()> {
        let ticks = self
            .clock
            .ticks
            .checked_add(1)
            .ok_or(WorldError::Invalid("clock_overflow"))?;
        let elapsed = self.clock.elapsed_seconds + dt;
        if !elapsed.is_finite() {
            return Err(WorldError::Invalid("clock_overflow"));
        }
        self.world.diffuse(diffusivity, dt)?;
        self.clock = GameClock {
            ticks,
            elapsed_seconds: elapsed,
        };
        Ok(())
    }
    pub fn to_nbt(&self) -> WorldResult<Vec<u8>> {
        let chunks = self
            .world
            .coordinates()
            .map(|coord| {
                Ok(SavedChunk {
                    active: i8::from(self.world.active(coord)),
                    resident: i8::from(self.world.resident_status(coord)),
                    bytes: fastnbt::ByteArray::new(
                        self.world
                            .export(coord)?
                            .into_iter()
                            .map(|v| v as i8)
                            .collect(),
                    ),
                })
            })
            .collect::<WorldResult<Vec<_>>>()?;
        let entities = self
            .entities
            .iter()
            .map(|(id, e)| {
                Ok(SavedEntity {
                    id: i64::try_from(id.0)
                        .map_err(|_| WorldError::Invalid("entity_id_save_limit"))?,
                    kind: self.world.content().entity(e.kind)?.key.clone(),
                    transform: e.transform,
                    inventory: e.inventory.save(self.world.content())?,
                    data: e.data.to_value()?,
                })
            })
            .collect::<WorldResult<Vec<_>>>()?;
        let save = SavedGame {
            version: 2,
            magic: self.magic.clone(),
            registry: self.world.content().fingerprint().into(),
            dimensions: self.world.layout().dimensions().map(|v| v as i32),
            cell_width: self.world.layout().cell_width(),
            clock: self.clock,
            chunks,
            entities,
        };
        let bytes = fastnbt::to_bytes(&save).map_err(|e| WorldError::Codec(e.to_string()))?;
        validate_wire(&bytes)?;
        Ok(bytes)
    }
    pub fn from_nbt(content: Arc<ContentRegistry>, bytes: &[u8]) -> WorldResult<Self> {
        let save: SavedGame = decode_nbt(bytes)?;
        if save.version != 2
            || save.registry != content.fingerprint()
            || !save.clock.elapsed_seconds.is_finite()
            || save.clock.elapsed_seconds < 0.0
        {
            return Err(WorldError::Invalid("game_save_header"));
        }
        let dims = save.dimensions.map(|v| usize::try_from(v).unwrap_or(0));
        let mut game = Self::new(content, ChunkLayout::new(dims, save.cell_width)?);
        game.clock = save.clock;
        save.magic.validate()?;
        game.magic = save.magic;
        for chunk in save.chunks {
            if !matches!(chunk.active, 0 | 1) {
                return Err(WorldError::Invalid("chunk_activity"));
            }
            let bytes: Vec<_> = chunk.bytes.iter().map(|v| *v as u8).collect();
            let coord = game.world.import(&bytes, chunk.active == 1)?;
            match chunk.resident {
                0 if chunk.active == 0 => game.world.unload(coord)?,
                1 => {}
                _ => return Err(WorldError::Invalid("chunk_residency")),
            }
        }
        for entity in save.entities {
            let id = u64::try_from(entity.id).map_err(|_| WorldError::Invalid("entity_id"))?;
            if id == 0 {
                return Err(WorldError::Invalid("entity_id"));
            }
            let kind = game.world.content().entity_id(&entity.kind)?;
            let inventory = Inventory::restore(entity.inventory, game.world.content())?;
            if inventory.slots().len() != game.world.content().entity(kind)?.inventory.slots {
                return Err(WorldError::Invalid("entity_inventory_layout"));
            }
            entity.transform.validate()?;
            let state = EntityState {
                kind,
                transform: entity.transform,
                inventory,
                data: NbtCompound::from_value(entity.data)?,
            };
            if game.entities.insert(EntityId(id), state).is_some() {
                return Err(WorldError::Invalid("duplicate_entity_id"));
            }
            game.next_id = game
                .next_id
                .max(id.checked_add(1).ok_or(WorldError::Invalid("entity_id"))?);
        }
        Ok(game)
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedChunk {
    active: i8,
    resident: i8,
    bytes: fastnbt::ByteArray,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedEntity {
    id: i64,
    kind: ResourceKey,
    transform: EntityTransform,
    inventory: SavedInventory,
    data: fastnbt::Value,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedGame {
    version: i32,
    registry: String,
    magic: crate::magic::MagicState,
    dimensions: [i32; 3],
    cell_width: f64,
    clock: GameClock,
    chunks: Vec<SavedChunk>,
    entities: Vec<SavedEntity>,
}
