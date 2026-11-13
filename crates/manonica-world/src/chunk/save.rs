use super::*;
use crate::metadata::decode_nbt;
use serde::{Deserialize, Serialize};
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedPalette {
    key: crate::registry::ResourceKey,
    properties: BTreeMap<String, String>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedBlockData {
    index: i32,
    data: fastnbt::Value,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedBlocks {
    palette: Vec<SavedPalette>,
    #[serde(skip_serializing_if = "Option::is_none")]
    uniform: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dense: Option<fastnbt::IntArray>,
    data: Vec<SavedBlockData>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedMana {
    #[serde(skip_serializing_if = "Option::is_none")]
    uniform: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dense: Option<Vec<f64>>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedLayout {
    dimensions: fastnbt::IntArray,
    cell_width: f64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChunkSave {
    version: i32,
    coord: ChunkCoord,
    layout: SavedLayout,
    blocks: SavedBlocks,
    mana: SavedMana,
}
pub(super) fn encode(
    chunk: &Chunk,
    layout: ChunkLayout,
    content: &ContentRegistry,
) -> WorldResult<Vec<u8>> {
    let palette = chunk
        .blocks
        .palette
        .iter()
        .map(|s| {
            Ok(SavedPalette {
                key: content.block(s.block())?.key.clone(),
                properties: s.properties().clone(),
            })
        })
        .collect::<WorldResult<Vec<_>>>()?;
    let (uniform, dense) = match &chunk.blocks.indices {
        PaletteIndices::Uniform(v) => (Some(i32::from(*v)), None),
        PaletteIndices::Dense(v) => (
            None,
            Some(fastnbt::IntArray::new(
                v.iter().map(|v| i32::from(*v)).collect(),
            )),
        ),
    };
    let data = chunk
        .blocks
        .data
        .iter()
        .map(|(i, data)| {
            Ok(SavedBlockData {
                index: *i as i32,
                data: data.to_value()?,
            })
        })
        .collect::<WorldResult<Vec<_>>>()?;
    let mana = match &chunk.mana {
        DensityStorage::Uniform(v) => SavedMana {
            uniform: Some(*v),
            dense: None,
        },
        DensityStorage::Dense(v) => SavedMana {
            uniform: None,
            dense: Some(v.clone()),
        },
    };
    let save = ChunkSave {
        version: 1,
        coord: chunk.coord,
        layout: SavedLayout {
            dimensions: fastnbt::IntArray::new(layout.dimensions().map(|v| v as i32).to_vec()),
            cell_width: layout.cell_width,
        },
        blocks: SavedBlocks {
            palette,
            uniform,
            dense,
            data,
        },
        mana,
    };
    fastnbt::to_bytes(&save).map_err(|e| WorldError::Codec(e.to_string()))
}
pub(super) fn decode(
    bytes: &[u8],
    layout: ChunkLayout,
    content: &ContentRegistry,
) -> WorldResult<Chunk> {
    let save: ChunkSave = decode_nbt(bytes)?;
    if save.version != 1
        || save.layout.dimensions.as_ref() != layout.dimensions().map(|v| v as i32).as_slice()
        || save.layout.cell_width != layout.cell_width
    {
        return Err(WorldError::Invalid("chunk_save_layout_or_version"));
    }
    if save.blocks.palette.is_empty() || save.blocks.palette.len() > 65536 {
        return Err(WorldError::Invalid("saved_palette"));
    }
    let palette = save
        .blocks
        .palette
        .into_iter()
        .map(|s| content.state(&s.key, s.properties))
        .collect::<WorldResult<Vec<_>>>()?;
    let palette_index = |i: i32| -> WorldResult<u16> {
        let v = u16::try_from(i).map_err(|_| WorldError::Invalid("saved_palette_index"))?;
        if usize::from(v) >= palette.len() {
            return Err(WorldError::Invalid("saved_palette_index"));
        }
        Ok(v)
    };
    let indices = match (save.blocks.uniform, save.blocks.dense) {
        (Some(v), None) => PaletteIndices::Uniform(palette_index(v)?),
        (None, Some(v)) if v.len() == layout.cells() => PaletteIndices::Dense(
            v.iter()
                .map(|v| palette_index(*v))
                .collect::<WorldResult<Vec<_>>>()?,
        ),
        _ => return Err(WorldError::Invalid("saved_block_storage")),
    };
    let mut data = BTreeMap::new();
    for entry in save.blocks.data {
        let index =
            usize::try_from(entry.index).map_err(|_| WorldError::Invalid("block_data_index"))?;
        if index >= layout.cells()
            || data
                .insert(index, NbtCompound::from_value(entry.data)?)
                .is_some()
        {
            return Err(WorldError::Invalid("saved_block_data"));
        }
    }
    let mana = match (save.mana.uniform, save.mana.dense) {
        (Some(v), None) if v.is_finite() && v >= 0.0 => DensityStorage::Uniform(v),
        (None, Some(v))
            if v.len() == layout.cells() && v.iter().all(|v| v.is_finite() && *v >= 0.0) =>
        {
            DensityStorage::from_values(v)
        }
        _ => return Err(WorldError::Invalid("saved_mana_storage")),
    };
    let count = mana.sum(layout.cells()) * layout.cell_width.powi(3);
    if !count.is_finite() {
        return Err(WorldError::Invalid("saved_mana_balance"));
    }
    Ok(Chunk {
        coord: save.coord,
        blocks: ChunkBlocks {
            palette,
            indices,
            data,
        },
        mana,
    })
}
