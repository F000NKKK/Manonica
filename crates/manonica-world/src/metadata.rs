//! Explicitly typed JSON metadata and real uncompressed Java-NBT encoding.
use crate::{WorldError, WorldResult};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum NbtTag {
    Byte(i8),
    Short(i16),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    String(String),
    ByteArray(Vec<i8>),
    IntArray(Vec<i32>),
    LongArray(Vec<i64>),
    List(Vec<NbtTag>),
    Compound(BTreeMap<String, NbtTag>),
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NbtCompound(BTreeMap<String, NbtTag>);
impl NbtCompound {
    pub fn new(values: BTreeMap<String, NbtTag>) -> WorldResult<Self> {
        let data = Self(values);
        data.validate()?;
        Ok(data)
    }
    pub fn get(&self, key: &str) -> Option<&NbtTag> {
        self.0.get(key)
    }
    pub fn entries(&self) -> &BTreeMap<String, NbtTag> {
        &self.0
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn insert(&mut self, key: String, value: NbtTag) -> WorldResult<()> {
        let mut candidate = self.clone();
        candidate.0.insert(key, value);
        candidate.validate()?;
        *self = candidate;
        Ok(())
    }
    pub fn validate(&self) -> WorldResult<()> {
        let mut budget = 262144;
        validate(&NbtTag::Compound(self.0.clone()), 0, &mut budget)
    }
    pub fn to_value(&self) -> WorldResult<fastnbt::Value> {
        self.validate()?;
        Ok(to_value(&NbtTag::Compound(self.0.clone())))
    }
    pub fn from_value(value: fastnbt::Value) -> WorldResult<Self> {
        let tag = from_value(value);
        if let NbtTag::Compound(values) = tag {
            Self::new(values)
        } else {
            Err(WorldError::Invalid("nbt_root_compound"))
        }
    }
    pub fn to_bytes(&self) -> WorldResult<Vec<u8>> {
        fastnbt::to_bytes(&self.to_value()?).map_err(|e| WorldError::Codec(e.to_string()))
    }
    pub fn from_bytes(bytes: &[u8]) -> WorldResult<Self> {
        if bytes.len() > 64 * 1024 * 1024 {
            return Err(WorldError::Invalid("nbt_payload_limit"));
        }
        Self::from_value(decode_nbt(bytes)?)
    }
}
fn validate(tag: &NbtTag, depth: usize, budget: &mut usize) -> WorldResult<()> {
    if depth > 32 || *budget == 0 {
        return Err(WorldError::Invalid("nbt_depth_or_size"));
    }
    *budget -= 1;
    match tag {
        NbtTag::Float(v) if !v.is_finite() => return Err(WorldError::Invalid("nbt_nonfinite")),
        NbtTag::Double(v) if !v.is_finite() => return Err(WorldError::Invalid("nbt_nonfinite")),
        NbtTag::String(v) if v.len() > 16000 => {
            return Err(WorldError::Invalid("nbt_string_limit"));
        }
        NbtTag::ByteArray(v) => {
            *budget = budget
                .checked_sub(v.len())
                .ok_or(WorldError::Invalid("nbt_array_limit"))?;
        }
        NbtTag::IntArray(v) => {
            *budget = budget
                .checked_sub(v.len())
                .ok_or(WorldError::Invalid("nbt_array_limit"))?;
        }
        NbtTag::LongArray(v) => {
            *budget = budget
                .checked_sub(v.len())
                .ok_or(WorldError::Invalid("nbt_array_limit"))?;
        }
        NbtTag::List(v) => {
            if v.len() > 65536
                || v.first().is_some_and(|first| {
                    v.iter()
                        .any(|tag| std::mem::discriminant(tag) != std::mem::discriminant(first))
                })
            {
                return Err(WorldError::Invalid("nbt_list_type_or_size"));
            }
            for tag in v {
                validate(tag, depth + 1, budget)?;
            }
        }
        NbtTag::Compound(v) => {
            for (key, value) in v {
                if key.len() > 16000 {
                    return Err(WorldError::Invalid("nbt_key_limit"));
                }
                validate(value, depth + 1, budget)?;
            }
        }
        _ => {}
    }
    Ok(())
}
fn to_value(tag: &NbtTag) -> fastnbt::Value {
    use fastnbt::Value as V;
    match tag {
        NbtTag::Byte(v) => V::Byte(*v),
        NbtTag::Short(v) => V::Short(*v),
        NbtTag::Int(v) => V::Int(*v),
        NbtTag::Long(v) => V::Long(*v),
        NbtTag::Float(v) => V::Float(*v),
        NbtTag::Double(v) => V::Double(*v),
        NbtTag::String(v) => V::String(v.clone()),
        NbtTag::ByteArray(v) => V::ByteArray(fastnbt::ByteArray::new(v.clone())),
        NbtTag::IntArray(v) => V::IntArray(fastnbt::IntArray::new(v.clone())),
        NbtTag::LongArray(v) => V::LongArray(fastnbt::LongArray::new(v.clone())),
        NbtTag::List(v) => V::List(v.iter().map(to_value).collect()),
        NbtTag::Compound(v) => {
            V::Compound(v.iter().map(|(k, v)| (k.clone(), to_value(v))).collect())
        }
    }
}
fn from_value(value: fastnbt::Value) -> NbtTag {
    use fastnbt::Value as V;
    match value {
        V::Byte(v) => NbtTag::Byte(v),
        V::Short(v) => NbtTag::Short(v),
        V::Int(v) => NbtTag::Int(v),
        V::Long(v) => NbtTag::Long(v),
        V::Float(v) => NbtTag::Float(v),
        V::Double(v) => NbtTag::Double(v),
        V::String(v) => NbtTag::String(v),
        V::ByteArray(v) => NbtTag::ByteArray(v.to_vec()),
        V::IntArray(v) => NbtTag::IntArray(v.to_vec()),
        V::LongArray(v) => NbtTag::LongArray(v.to_vec()),
        V::List(v) => NbtTag::List(v.into_iter().map(from_value).collect()),
        V::Compound(v) => {
            NbtTag::Compound(v.into_iter().map(|(k, v)| (k, from_value(v))).collect())
        }
    }
}

mod wire;
pub(crate) use wire::{decode as decode_nbt, validate_wire};
