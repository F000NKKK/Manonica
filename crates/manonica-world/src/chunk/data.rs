use crate::{WorldError, WorldResult, metadata::NbtCompound, registry::BlockState};
use manonica_physics::GridShape;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkCoord {
    pub x: i32,
    pub z: i32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellPosition {
    pub x: i64,
    pub y: usize,
    pub z: i64,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChunkLayout {
    pub(crate) shape: GridShape,
    pub(crate) cell_width: f64,
}
impl ChunkLayout {
    pub fn new(dimensions: [usize; 3], cell_width: f64) -> WorldResult<Self> {
        let shape = GridShape::new(dimensions[0], dimensions[1], dimensions[2])?;
        if shape.len() > 1_048_576
            || dimensions.iter().any(|v| *v > 4096)
            || !cell_width.is_finite()
            || cell_width <= 0.0
            || !cell_width.powi(3).is_finite()
            || cell_width.powi(3) <= 0.0
        {
            return Err(WorldError::Invalid("chunk_layout"));
        }
        Ok(Self { shape, cell_width })
    }
    pub fn column(height: usize) -> WorldResult<Self> {
        Self::new([16, height, 16], 1.0)
    }
    pub fn dimensions(self) -> [usize; 3] {
        [self.shape.nx(), self.shape.ny(), self.shape.nz()]
    }
    pub fn cell_width(self) -> f64 {
        self.cell_width
    }
    pub fn cells(self) -> usize {
        self.shape.len()
    }
    pub fn locate(self, position: CellPosition) -> WorldResult<(ChunkCoord, usize)> {
        let nx = self.shape.nx() as i64;
        let nz = self.shape.nz() as i64;
        let coord = ChunkCoord {
            x: i32::try_from(position.x.div_euclid(nx))
                .map_err(|_| WorldError::Invalid("chunk_coordinate"))?,
            z: i32::try_from(position.z.div_euclid(nz))
                .map_err(|_| WorldError::Invalid("chunk_coordinate"))?,
        };
        let index = self
            .shape
            .index(
                position.x.rem_euclid(nx) as usize,
                position.y,
                position.z.rem_euclid(nz) as usize,
            )
            .ok_or(WorldError::Invalid("cell_position"))?;
        Ok((coord, index))
    }
}
#[derive(Clone, Debug)]
pub(crate) enum DensityStorage {
    Uniform(f64),
    Dense(Vec<f64>),
}
impl DensityStorage {
    pub fn at(&self, index: usize) -> f64 {
        match self {
            Self::Uniform(v) => *v,
            Self::Dense(v) => v[index],
        }
    }
    pub fn values(&self, len: usize) -> Vec<f64> {
        match self {
            Self::Uniform(v) => vec![*v; len],
            Self::Dense(v) => v.clone(),
        }
    }
    pub fn sum(&self, len: usize) -> f64 {
        match self {
            Self::Uniform(v) => *v * len as f64,
            Self::Dense(v) => v.iter().sum(),
        }
    }
    pub fn bytes(&self) -> usize {
        match self {
            Self::Uniform(_) => 8,
            Self::Dense(v) => v.len() * 8,
        }
    }
    pub fn from_values(values: Vec<f64>) -> Self {
        if values.iter().all(|v| *v == values[0]) {
            Self::Uniform(values[0])
        } else {
            Self::Dense(values)
        }
    }
}
#[derive(Clone, Debug)]
pub(crate) enum PaletteIndices {
    Uniform(u16),
    Dense(Vec<u16>),
}
impl PaletteIndices {
    pub fn at(&self, index: usize) -> u16 {
        match self {
            Self::Uniform(v) => *v,
            Self::Dense(v) => v[index],
        }
    }
    pub fn set(&mut self, index: usize, palette: u16, len: usize) {
        if let Self::Uniform(old) = self {
            if *old == palette {
                return;
            }
            *self = Self::Dense(vec![*old; len]);
        }
        if let Self::Dense(values) = self {
            values[index] = palette;
        }
    }
    pub fn bytes(&self) -> usize {
        match self {
            Self::Uniform(_) => 2,
            Self::Dense(v) => v.len() * 2,
        }
    }
}
#[derive(Clone, Debug)]
pub(crate) struct ChunkBlocks {
    pub palette: Vec<BlockState>,
    pub indices: PaletteIndices,
    pub data: BTreeMap<usize, NbtCompound>,
}
#[derive(Clone, Debug)]
pub(crate) struct Chunk {
    pub coord: ChunkCoord,
    pub blocks: ChunkBlocks,
    pub mana: DensityStorage,
}
impl Chunk {
    pub fn state(&self, index: usize) -> &BlockState {
        &self.blocks.palette[self.blocks.indices.at(index) as usize]
    }
    pub fn set(
        &mut self,
        index: usize,
        state: BlockState,
        data: NbtCompound,
        len: usize,
    ) -> WorldResult<()> {
        let palette = if let Some(index) = self.blocks.palette.iter().position(|v| *v == state) {
            index as u16
        } else {
            let next = u16::try_from(self.blocks.palette.len())
                .map_err(|_| WorldError::Invalid("block_palette_limit"))?;
            self.blocks.palette.push(state);
            next
        };
        self.blocks.indices.set(index, palette, len);
        if data.is_empty() {
            self.blocks.data.remove(&index);
        } else {
            self.blocks.data.insert(index, data);
        }
        Ok(())
    }
}
