use crate::{
    WorldError, WorldResult,
    metadata::NbtCompound,
    registry::{ContentRegistry, ItemId, ResourceKey},
};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, PartialEq)]
pub struct ItemStack {
    item: ItemId,
    count: u16,
    data: NbtCompound,
}
impl ItemStack {
    pub fn new(
        content: &ContentRegistry,
        key: &ResourceKey,
        count: u16,
        data: NbtCompound,
    ) -> WorldResult<Self> {
        let item = content.item_id(key)?;
        if count == 0 || count > content.item(item)?.stack.limit {
            return Err(WorldError::Invalid("item_stack_count"));
        }
        data.validate()?;
        Ok(Self { item, count, data })
    }
    pub fn item(&self) -> ItemId {
        self.item
    }
    pub fn count(&self) -> u16 {
        self.count
    }
    pub fn data(&self) -> &NbtCompound {
        &self.data
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct Inventory {
    slots: Vec<Option<ItemStack>>,
}
impl Inventory {
    pub fn new(slots: usize) -> WorldResult<Self> {
        if slots > 256 {
            return Err(WorldError::Invalid("inventory_size"));
        }
        Ok(Self {
            slots: vec![None; slots],
        })
    }
    pub fn slots(&self) -> &[Option<ItemStack>] {
        &self.slots
    }
    pub fn get(&self, slot: usize) -> WorldResult<Option<&ItemStack>> {
        self.slots
            .get(slot)
            .map(|s| s.as_ref())
            .ok_or(WorldError::Invalid("inventory_slot"))
    }
    pub fn insert(
        &mut self,
        content: &ContentRegistry,
        slot: usize,
        stack: ItemStack,
    ) -> WorldResult<()> {
        let destination = self
            .slots
            .get(slot)
            .ok_or(WorldError::Invalid("inventory_slot"))?;
        let limit = content.item(stack.item)?.stack.limit;
        stack.data.validate()?;
        let result = if let Some(existing) = destination {
            if existing.item != stack.item || existing.data != stack.data {
                return Err(WorldError::Invalid("incompatible_stack"));
            }
            let count = existing
                .count
                .checked_add(stack.count)
                .ok_or(WorldError::Invalid("stack_overflow"))?;
            if count > limit {
                return Err(WorldError::Invalid("stack_limit"));
            }
            ItemStack {
                count,
                ..existing.clone()
            }
        } else {
            if stack.count == 0 || stack.count > limit {
                return Err(WorldError::Invalid("stack_limit"));
            }
            stack
        };
        self.slots[slot] = Some(result);
        Ok(())
    }
    pub(crate) fn after_consuming(
        &self,
        slot: usize,
        count: u16,
    ) -> WorldResult<Option<ItemStack>> {
        let stack = self
            .get(slot)?
            .ok_or(WorldError::Invalid("empty_inventory_slot"))?;
        if count == 0 || count > stack.count {
            return Err(WorldError::Invalid("consume_count"));
        }
        Ok(if count == stack.count {
            None
        } else {
            Some(ItemStack {
                count: stack.count - count,
                ..stack.clone()
            })
        })
    }
    pub(crate) fn commit_slot(&mut self, slot: usize, value: Option<ItemStack>) {
        self.slots[slot] = value;
    }
    pub(crate) fn save(&self, content: &ContentRegistry) -> WorldResult<SavedInventory> {
        let mut slots = Vec::new();
        for (index, stack) in self.slots.iter().enumerate() {
            if let Some(stack) = stack {
                slots.push(SavedSlot {
                    index: index as i32,
                    item: content.item(stack.item)?.key.clone(),
                    count: i32::from(stack.count),
                    data: stack.data.to_value()?,
                });
            }
        }
        Ok(SavedInventory {
            size: self.slots.len() as i32,
            slots,
        })
    }
    pub(crate) fn restore(saved: SavedInventory, content: &ContentRegistry) -> WorldResult<Self> {
        let size =
            usize::try_from(saved.size).map_err(|_| WorldError::Invalid("inventory_size"))?;
        let mut inventory = Self::new(size)?;
        for slot in saved.slots {
            let index =
                usize::try_from(slot.index).map_err(|_| WorldError::Invalid("inventory_slot"))?;
            if inventory.get(index)?.is_some() {
                return Err(WorldError::Invalid("duplicate_inventory_slot"));
            }
            let stack = ItemStack::new(
                content,
                &slot.item,
                u16::try_from(slot.count).map_err(|_| WorldError::Invalid("item_count"))?,
                NbtCompound::from_value(slot.data)?,
            )?;
            inventory.insert(content, index, stack)?;
        }
        Ok(inventory)
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SavedInventory {
    size: i32,
    slots: Vec<SavedSlot>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedSlot {
    index: i32,
    item: ResourceKey,
    count: i32,
    data: fastnbt::Value,
}
