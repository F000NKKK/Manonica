//! Sparse chunk ownership and conservative reference transport across active borders.
mod data;
mod save;
use crate::{
    WorldError, WorldResult,
    metadata::NbtCompound,
    registry::{BlockState, ContentRegistry, ResourceKey},
};
use data::*;
pub use data::{CellPosition, ChunkCoord, ChunkLayout};
use std::{collections::BTreeMap, sync::Arc};
#[derive(Clone, Debug)]
enum Slot {
    Resident { chunk: Chunk, active: bool },
    Archived { bytes: Vec<u8>, total: f64 },
}
#[derive(Clone, Copy, Debug, Default)]
pub struct ChunkCounts {
    pub total: usize,
    pub resident: usize,
    pub active: usize,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct WorldStatistics {
    pub chunks: ChunkCounts,
    pub free_manons: f64,
    pub payload_bytes: usize,
}
#[derive(Clone, Debug)]
pub struct ChunkWorld {
    content: Arc<ContentRegistry>,
    layout: ChunkLayout,
    chunks: BTreeMap<ChunkCoord, Slot>,
}
impl ChunkWorld {
    pub fn new(content: Arc<ContentRegistry>, layout: ChunkLayout) -> Self {
        Self {
            content,
            layout,
            chunks: BTreeMap::new(),
        }
    }
    pub fn content(&self) -> &ContentRegistry {
        &self.content
    }
    pub fn layout(&self) -> ChunkLayout {
        self.layout
    }
    pub fn occupied_blocks(&self) -> impl Iterator<Item = (CellPosition, &BlockState)> + '_ {
        let dims = self.layout.dimensions();
        let air = self.content.air();
        self.chunks.iter().flat_map(move |(coord, slot)| {
            let chunk = match slot {
                Slot::Resident { chunk, .. } => Some(chunk),
                _ => None,
            };
            chunk.into_iter().flat_map(move |chunk| {
                (0..self.layout.cells()).filter_map(move |i| {
                    let state = chunk.state(i);
                    if state.block() == air {
                        return None;
                    }
                    let x = i % dims[0];
                    let y = (i / dims[0]) % dims[1];
                    let z = i / (dims[0] * dims[1]);
                    Some((
                        CellPosition {
                            x: i64::from(coord.x) * dims[0] as i64 + x as i64,
                            y,
                            z: i64::from(coord.z) * dims[2] as i64 + z as i64,
                        },
                        state,
                    ))
                })
            })
        })
    }
    pub fn coordinates(&self) -> impl Iterator<Item = ChunkCoord> + '_ {
        self.chunks.keys().copied()
    }
    pub fn resident_status(&self, coord: ChunkCoord) -> bool {
        matches!(self.chunks.get(&coord), Some(Slot::Resident { .. }))
    }
    pub fn active(&self, coord: ChunkCoord) -> bool {
        matches!(
            self.chunks.get(&coord),
            Some(Slot::Resident { active: true, .. })
        )
    }
    pub fn create(&mut self, coord: ChunkCoord, density: f64, active: bool) -> WorldResult<()> {
        if self.chunks.contains_key(&coord)
            || !density.is_finite()
            || density < 0.0
            || !(density * self.layout.cells() as f64 * self.layout.cell_width.powi(3)).is_finite()
        {
            return Err(WorldError::Invalid("chunk_creation"));
        }
        let air = self.content.block(self.content.air())?.key.clone();
        let chunk = Chunk {
            coord,
            blocks: ChunkBlocks {
                palette: vec![self.content.default_state(&air)?],
                indices: PaletteIndices::Uniform(0),
                data: BTreeMap::new(),
            },
            mana: DensityStorage::Uniform(density),
        };
        self.chunks.insert(coord, Slot::Resident { chunk, active });
        Ok(())
    }
    fn resident(&self, coord: ChunkCoord) -> WorldResult<&Chunk> {
        match self.chunks.get(&coord) {
            Some(Slot::Resident { chunk, .. }) => Ok(chunk),
            _ => Err(WorldError::Invalid("chunk_not_resident")),
        }
    }
    fn resident_mut(&mut self, coord: ChunkCoord) -> WorldResult<&mut Chunk> {
        match self.chunks.get_mut(&coord) {
            Some(Slot::Resident { chunk, .. }) => Ok(chunk),
            _ => Err(WorldError::Invalid("chunk_not_resident")),
        }
    }
    pub fn set_active(&mut self, coord: ChunkCoord, value: bool) -> WorldResult<()> {
        match self.chunks.get_mut(&coord) {
            Some(Slot::Resident { active, .. }) => {
                *active = value;
                Ok(())
            }
            _ => Err(WorldError::Invalid("chunk_not_resident")),
        }
    }
    pub fn density(&self, pos: CellPosition) -> WorldResult<f64> {
        let (c, i) = self.layout.locate(pos)?;
        Ok(self.resident(c)?.mana.at(i))
    }
    /// Explicit source/sink API; callers account for the corresponding resource transfer.
    pub fn set_density(&mut self, pos: CellPosition, density: f64) -> WorldResult<()> {
        if !density.is_finite() || density < 0.0 {
            return Err(WorldError::Invalid("density"));
        }
        let (coord, index) = self.layout.locate(pos)?;
        let len = self.layout.cells();
        let volume = self.layout.cell_width.powi(3);
        let chunk = self.resident_mut(coord)?;
        let mut values = chunk.mana.values(len);
        values[index] = density;
        if !(values.iter().sum::<f64>() * volume).is_finite() {
            return Err(WorldError::Invalid("density_balance"));
        }
        chunk.mana = DensityStorage::from_values(values);
        Ok(())
    }
    pub fn block(&self, pos: CellPosition) -> WorldResult<&BlockState> {
        let (c, i) = self.layout.locate(pos)?;
        Ok(self.resident(c)?.state(i))
    }
    pub fn block_data(&self, pos: CellPosition) -> WorldResult<NbtCompound> {
        let (c, i) = self.layout.locate(pos)?;
        Ok(self
            .resident(c)?
            .blocks
            .data
            .get(&i)
            .cloned()
            .unwrap_or_default())
    }
    pub fn set_block(
        &mut self,
        pos: CellPosition,
        key: &ResourceKey,
        properties: BTreeMap<String, String>,
        data: NbtCompound,
    ) -> WorldResult<()> {
        let state = self.content.state(key, properties)?;
        data.validate()?;
        let (c, i) = self.layout.locate(pos)?;
        let len = self.layout.cells();
        self.resident_mut(c)?.set(i, state, data, len)
    }
    pub fn export(&self, coord: ChunkCoord) -> WorldResult<Vec<u8>> {
        match self.chunks.get(&coord) {
            Some(Slot::Resident { chunk, .. }) => save::encode(chunk, self.layout, &self.content),
            Some(Slot::Archived { bytes, .. }) => Ok(bytes.clone()),
            None => Err(WorldError::Invalid("unknown_chunk")),
        }
    }
    pub fn import(&mut self, bytes: &[u8], active: bool) -> WorldResult<ChunkCoord> {
        let chunk = save::decode(bytes, self.layout, &self.content)?;
        let coord = chunk.coord;
        if self.chunks.contains_key(&coord) {
            return Err(WorldError::Invalid("duplicate_chunk"));
        }
        self.chunks.insert(coord, Slot::Resident { chunk, active });
        Ok(coord)
    }
    pub fn unload(&mut self, coord: ChunkCoord) -> WorldResult<()> {
        let chunk = self.resident(coord)?;
        let total = chunk.mana.sum(self.layout.cells()) * self.layout.cell_width.powi(3);
        let bytes = self.export(coord)?;
        self.chunks.insert(coord, Slot::Archived { bytes, total });
        Ok(())
    }
    pub fn reload(&mut self, coord: ChunkCoord, active: bool) -> WorldResult<()> {
        let bytes = match self.chunks.get(&coord) {
            Some(Slot::Archived { bytes, .. }) => bytes,
            _ => return Err(WorldError::Invalid("chunk_not_archived")),
        };
        let chunk = save::decode(bytes, self.layout, &self.content)?;
        self.chunks.insert(coord, Slot::Resident { chunk, active });
        Ok(())
    }
    pub fn statistics(&self) -> WorldStatistics {
        let mut s = WorldStatistics::default();
        for slot in self.chunks.values() {
            s.chunks.total += 1;
            match slot {
                Slot::Resident { chunk, active } => {
                    s.chunks.resident += 1;
                    s.chunks.active += usize::from(*active);
                    s.free_manons +=
                        chunk.mana.sum(self.layout.cells()) * self.layout.cell_width.powi(3);
                    s.payload_bytes += chunk.mana.bytes() + chunk.blocks.indices.bytes();
                }
                Slot::Archived { bytes, total } => {
                    s.free_manons += total;
                    s.payload_bytes += bytes.len();
                }
            }
        }
        s
    }
    /// Reflecting boundaries at missing, archived or inactive chunks. Explicit Euler, CFL <= 1/6.
    pub fn diffuse(&mut self, diffusivity: f64, dt: f64) -> WorldResult<()> {
        let alpha = diffusivity * dt / self.layout.cell_width.powi(2);
        if !diffusivity.is_finite()
            || diffusivity < 0.0
            || !dt.is_finite()
            || dt <= 0.0
            || !alpha.is_finite()
            || alpha > 1.0 / 6.0
        {
            return Err(WorldError::Invalid("diffusion_cfl"));
        }
        let mut updates: BTreeMap<ChunkCoord, Vec<f64>> = BTreeMap::new();
        let dims = self.layout.dimensions();
        for (&coord, slot) in &self.chunks {
            let Slot::Resident {
                chunk,
                active: true,
            } = slot
            else {
                continue;
            };
            for z in 0..dims[2] {
                for y in 0..dims[1] {
                    for x in 0..dims[0] {
                        let i = self.layout.shape.index(x, y, z).unwrap();
                        for axis in 0..3 {
                            let mut xyz = [x, y, z];
                            xyz[axis] += 1;
                            let mut neighbor = coord;
                            if xyz[axis] == dims[axis] {
                                xyz[axis] = 0;
                                match axis {
                                    0 => {
                                        let Some(v) = coord.x.checked_add(1) else {
                                            continue;
                                        };
                                        neighbor.x = v;
                                    }
                                    2 => {
                                        let Some(v) = coord.z.checked_add(1) else {
                                            continue;
                                        };
                                        neighbor.z = v;
                                    }
                                    _ => continue,
                                }
                            }
                            if !self.active(neighbor) {
                                continue;
                            }
                            let other = self.resident(neighbor)?;
                            let j = self.layout.shape.index(xyz[0], xyz[1], xyz[2]).unwrap();
                            let a = self
                                .content
                                .block(chunk.state(i).block())?
                                .physics
                                .mana
                                .permeability;
                            let b = self
                                .content
                                .block(other.state(j).block())?
                                .physics
                                .mana
                                .permeability;
                            if a + b == 0.0 {
                                continue;
                            }
                            let flux = alpha
                                * (2.0 * a * b / (a + b))
                                * (chunk.mana.at(i) - other.mana.at(j));
                            if flux == 0.0 {
                                continue;
                            }
                            updates
                                .entry(coord)
                                .or_insert_with(|| chunk.mana.values(self.layout.cells()))[i] -=
                                flux;
                            updates
                                .entry(neighbor)
                                .or_insert_with(|| other.mana.values(self.layout.cells()))[j] +=
                                flux;
                        }
                    }
                }
            }
        }
        for values in updates.values() {
            if values.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(WorldError::Invalid("diffusion_density"));
            }
        }
        let before: f64 = updates
            .keys()
            .map(|c| self.resident(*c).unwrap().mana.sum(self.layout.cells()))
            .sum();
        let after: f64 = updates.values().flatten().sum();
        if !after.is_finite() || (after - before).abs() > 1e-10 * before.max(1.0) {
            return Err(WorldError::Invalid("diffusion_balance"));
        }
        for (coord, values) in updates {
            self.resident_mut(coord)?.mana = DensityStorage::from_values(values);
        }
        Ok(())
    }
}
