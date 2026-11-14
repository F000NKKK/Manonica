use manonica_world::{
    chunk::{CellPosition, ChunkCoord, ChunkWorld},
    registry::BlockId,
};
use meridian_sdk::{
    GraphicsSession, Material, MaterialHandle, MeshHandle, MeshSource, Motor3, Renderable3d,
    Scale3, Vec3,
};
use std::collections::BTreeMap;
pub struct TerrainView {
    chunks: BTreeMap<ChunkCoord, TerrainChunk>,
    materials: BTreeMap<(BlockId, usize), MaterialHandle>,
    pub renderables: Vec<Renderable3d>,
}
struct TerrainChunk {
    signature: [Option<u64>; 5],
    meshes: Vec<MeshHandle>,
    renderables: Vec<Renderable3d>,
}
const NORMALS: [[i64; 3]; 6] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
];
const CORNERS: [[[f32; 3]; 4]; 6] = [
    [[1., 0., 0.], [1., 1., 0.], [1., 1., 1.], [1., 0., 1.]],
    [[0., 0., 1.], [0., 1., 1.], [0., 1., 0.], [0., 0., 0.]],
    [[0., 1., 1.], [1., 1., 1.], [1., 1., 0.], [0., 1., 0.]],
    [[0., 0., 0.], [1., 0., 0.], [1., 0., 1.], [0., 0., 1.]],
    [[1., 0., 1.], [1., 1., 1.], [0., 1., 1.], [0., 0., 1.]],
    [[0., 0., 0.], [0., 1., 0.], [1., 1., 0.], [1., 0., 0.]],
];
fn blank() -> MeshSource {
    MeshSource {
        positions: vec![],
        normals: vec![],
        uvs: vec![],
        indices: vec![],
        skin: None,
        morph: None,
    }
}
impl TerrainView {
    pub fn new() -> Self {
        Self {
            chunks: BTreeMap::new(),
            materials: BTreeMap::new(),
            renderables: vec![],
        }
    }
    pub fn invalidate(&mut self) {
        for chunk in self.chunks.values_mut() {
            chunk.signature = [None; 5];
        }
    }
    pub fn rebuild(
        &mut self,
        base: &mut GraphicsSession,
        world: &ChunkWorld,
    ) -> Result<(), String> {
        let versions = world.geometry_chunks().collect::<BTreeMap<_, _>>();
        let removed = self
            .chunks
            .keys()
            .filter(|c| !versions.contains_key(c))
            .copied()
            .collect::<Vec<_>>();
        for coord in removed {
            for handle in self.chunks.remove(&coord).unwrap().meshes {
                base.render.meshes.unregister(handle);
            }
        }
        for &coord in versions.keys() {
            let signature = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]].map(|offset| {
                versions
                    .get(&ChunkCoord {
                        x: coord.x + offset[0],
                        z: coord.z + offset[1],
                    })
                    .copied()
            });
            if self
                .chunks
                .get(&coord)
                .is_some_and(|c| c.signature == signature)
            {
                continue;
            }
            let mut groups: BTreeMap<(BlockId, usize), MeshSource> = BTreeMap::new();
            for (pos, state) in world.occupied_blocks_in(coord) {
                for (face, normal) in NORMALS.into_iter().enumerate() {
                    let y = pos.y as i64 + normal[1];
                    let neighbor = CellPosition {
                        x: pos.x + normal[0],
                        y: y.max(0) as usize,
                        z: pos.z + normal[2],
                    };
                    if y >= 0
                        && world
                            .block(neighbor)
                            .is_ok_and(|s| s.block() != world.content().air())
                    {
                        continue;
                    }
                    let mesh = groups.entry((state.block(), face)).or_insert_with(blank);
                    let start = mesh.positions.len() as u32;
                    for corner in CORNERS[face] {
                        mesh.positions.push([
                            pos.x as f32 + corner[0],
                            pos.y as f32 + corner[1],
                            pos.z as f32 + corner[2],
                        ]);
                        mesh.normals.push(normal.map(|v| v as f32));
                    }
                    mesh.uvs
                        .extend_from_slice(&[[0., 1.], [0., 0.], [1., 0.], [1., 1.]]);
                    mesh.indices.extend_from_slice(&[
                        start,
                        start + 1,
                        start + 2,
                        start,
                        start + 2,
                        start + 3,
                    ]);
                }
            }
            let mut renderables = vec![];
            let mut meshes = vec![];
            for ((block, face), mesh) in groups {
                let material = if let Some(handle) = self.materials.get(&(block, face)) {
                    *handle
                } else {
                    let appearance = world
                        .content()
                        .block(block)
                        .map_err(|e| e.to_string())?
                        .appearance;
                    let color = if face == 2 {
                        appearance.top_color.unwrap_or(appearance.base_color)
                    } else {
                        appearance.base_color
                    };
                    let shade = [0.75, 0.6, 1.0, 0.4, 0.82, 0.68][face];
                    let c = color.map(|v| v * shade);
                    let handle = base.render.materials.register(Material {
                        base_color_factor: [c[0], c[1], c[2], 1.0],
                        emissive: appearance.emissive,
                        unlit: false,
                        ..Default::default()
                    });
                    self.materials.insert((block, face), handle);
                    handle
                };
                let mesh = base
                    .render
                    .meshes
                    .register(mesh)
                    .map_err(|e| e.to_string())?;
                meshes.push(mesh);
                renderables.push(Renderable3d {
                    mesh,
                    material,
                    frame: Motor3::translation(Vec3::ZERO),
                    scale: Scale3::UNIT,
                    billboard: false,
                    casts_shadow: false,
                    palette: None,
                    morph_weights: vec![],
                });
            }
            if let Some(old) = self.chunks.insert(
                coord,
                TerrainChunk {
                    signature,
                    meshes,
                    renderables,
                },
            ) {
                for handle in old.meshes {
                    base.render.meshes.unregister(handle);
                }
            }
        }
        self.renderables = self
            .chunks
            .values()
            .flat_map(|c| c.renderables.iter().cloned())
            .collect();
        Ok(())
    }
}
