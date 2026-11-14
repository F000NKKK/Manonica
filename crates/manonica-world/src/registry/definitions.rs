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
    #[serde(default)]
    pub mechanics: Option<MaterialMechanics>,
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
    #[serde(default)]
    pub appearance: BlockAppearance,
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialMechanics {
    pub surface_emissivity: f64,
    pub thermal_contact_watts_per_kelvin: f64,
    pub crushing_energy_joules: Option<f64>,
    pub mass_kg: f64,
    pub drag_rate_per_second: f64,
    pub tensile_strength_newtons: f64,
    pub fracture_energy_joules: f64,
    pub heat_capacity_joules_per_kelvin: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockAppearance {
    pub base_color: [f32; 3],
    pub top_color: Option<[f32; 3]>,
    pub emissive: [f32; 3],
}
impl Default for BlockAppearance {
    fn default() -> Self {
        Self {
            base_color: [0.45, 0.49, 0.54],
            top_color: None,
            emissive: [0.; 3],
        }
    }
}
