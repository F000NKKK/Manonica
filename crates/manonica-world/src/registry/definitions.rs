use super::ResourceKey;
use crate::metadata::NbtCompound;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DisplayInfo {
    pub name: String,
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockCollision {
    Empty,
    FullCube,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockPhysics {
    pub collision: BlockCollision,
    pub replaceable: bool,
    pub mana: ManaTransport,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManaTransport {
    pub permeability: f64,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockStateSchema {
    #[serde(default)]
    pub allowed: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub defaults: BTreeMap<String, String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockDefinition {
    pub key: ResourceKey,
    pub display: DisplayInfo,
    pub physics: BlockPhysics,
    #[serde(default)]
    pub state: BlockStateSchema,
    #[serde(default)]
    pub metadata: NbtCompound,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StackRules {
    pub limit: u16,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ItemBehavior {
    #[serde(default)]
    pub place_block: Option<ResourceKey>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ItemDefinition {
    pub key: ResourceKey,
    pub display: DisplayInfo,
    pub stack: StackRules,
    #[serde(default)]
    pub behavior: ItemBehavior,
    #[serde(default)]
    pub metadata: NbtCompound,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntityCollider {
    pub half_extents: [f64; 3],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntityPhysics {
    pub mass: f64,
    pub collider: EntityCollider,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InventoryLayout {
    pub slots: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntityDefinition {
    pub key: ResourceKey,
    pub display: DisplayInfo,
    pub physics: EntityPhysics,
    pub inventory: InventoryLayout,
    #[serde(default)]
    pub metadata: NbtCompound,
}
