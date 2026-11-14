//! A generic field observer: particles, current vectors and isopotential contours.
//! No spell identifiers, particle presets or object commands enter this renderer.
use manonica_world::magic::{MagicState, ManonField, construction::CircleGrammar};
use meridian_sdk::{
    Device, GraphicsSession, Material, MaterialHandle, MeshHandle, MeshSource, Motor3,
    Renderable3d, Scale3, Vec3,
};
use std::collections::BTreeMap;
fn empty() -> MeshSource {
    MeshSource {
        positions: vec![],
        normals: vec![],
        uvs: vec![],
        indices: vec![],
        skin: None,
        morph: None,
    }
}
fn particle(mesh: &mut MeshSource, p: [f32; 3], size: f32) {
    let base = mesh.positions.len() as u32;
    for n in [
        [1., 0., 0.],
        [-1., 0., 0.],
        [0., 1., 0.],
        [0., -1., 0.],
        [0., 0., 1.],
        [0., 0., -1.],
    ] {
        mesh.positions
            .push(std::array::from_fn(|a| p[a] + n[a] * size));
        mesh.normals.push(n);
        mesh.uvs.push([0., 0.]);
    }
    for t in [
        [2, 0, 4],
        [2, 4, 1],
        [2, 1, 5],
        [2, 5, 0],
        [3, 4, 0],
        [3, 1, 4],
        [3, 5, 1],
        [3, 0, 5],
    ] {
        mesh.indices.extend(t.map(|i| base + i));
    }
}
fn beam(mesh: &mut MeshSource, a: [f32; 3], b: [f32; 3], width: f32) {
    let d = std::array::from_fn::<_, 3, _>(|i| b[i] - a[i]);
    let n = d.iter().map(|v| v * v).sum::<f32>().sqrt();
    if n < 1e-6 {
        return;
    }
    let u = if d[1].abs() / n < 0.9 {
        [d[2] / n, 0., -d[0] / n]
    } else {
        [1., 0., 0.]
    };
    let length = u.iter().map(|v| v * v).sum::<f32>().sqrt();
    let u = u.map(|v| v / length);
    let v = [
        d[1] * u[2] - d[2] * u[1],
        d[2] * u[0] - d[0] * u[2],
        d[0] * u[1] - d[1] * u[0],
    ]
    .map(|v| v / n);
    let base = mesh.positions.len() as u32;
    for p in [a, b] {
        for k in 0..4 {
            let angle = k as f32 * std::f32::consts::FRAC_PI_2;
            let normal =
                std::array::from_fn::<_, 3, _>(|i| u[i] * angle.cos() + v[i] * angle.sin());
            mesh.positions
                .push(std::array::from_fn(|i| p[i] + normal[i] * width));
            mesh.normals.push(normal);
            mesh.uvs.push([0., 0.]);
        }
    }
    for k in 0..4 {
        let j = (k + 1) % 4;
        mesh.indices.extend([
            base + k,
            base + j,
            base + k + 4,
            base + j,
            base + j + 4,
            base + k + 4,
        ]);
    }
}
fn draw(mesh: MeshHandle, material: MaterialHandle) -> Renderable3d {
    Renderable3d {
        mesh,
        material,
        frame: Motor3::translation(Vec3::ZERO),
        scale: Scale3::UNIT,
        billboard: false,
        casts_shadow: false,
        palette: None,
        morph_weights: vec![],
    }
}
fn index(f: &ManonField, p: [usize; 3]) -> usize {
    (p[2] * f.dimensions[1] + p[1]) * f.dimensions[0] + p[0]
}
fn contours(mesh: &mut MeshSource, f: &ManonField) {
    let peak = f
        .potential
        .iter()
        .copied()
        .max_by(|a, b| a.abs().total_cmp(&b.abs()))
        .unwrap_or(0.0);
    if peak.abs() < 1e-6 {
        return;
    }
    for axis in 0..3 {
        let a = (axis + 1) % 3;
        let b = (axis + 2) % 3;
        for slice in [0.25, 0.5, 0.75] {
            let layer = ((f.dimensions[axis] - 1) as f64 * slice) as usize;
            for fraction in [0.2, 0.5, 0.8] {
                let level = peak * fraction;
                for y in 0..f.dimensions[b] - 1 {
                    for x in 0..f.dimensions[a] - 1 {
                        let mut corners = [[0; 3]; 4];
                        for (i, [da, db]) in
                            [[0, 0], [1, 0], [1, 1], [0, 1]].into_iter().enumerate()
                        {
                            corners[i][axis] = layer;
                            corners[i][a] = x + da;
                            corners[i][b] = y + db;
                        }
                        let ids = corners.map(|p| index(f, p));
                        let mut crossings = vec![];
                        for k in 0..4 {
                            let j = (k + 1) % 4;
                            let va = f.potential[ids[k]];
                            let vb = f.potential[ids[j]];
                            if (va < level) != (vb < level) {
                                let t = (level - va) / (vb - va);
                                let p = f.position(ids[k]);
                                let q = f.position(ids[j]);
                                crossings.push(std::array::from_fn::<_, 3, _>(|a| {
                                    (p[a] + t * (q[a] - p[a])) as f32
                                }));
                            }
                        }
                        for pair in crossings.as_chunks::<2>().0 {
                            beam(mesh, pair[0], pair[1], 0.009);
                        }
                    }
                }
            }
        }
    }
}
pub struct SpellView {
    materials: [MaterialHandle; 3],
    meshes: Vec<MeshHandle>,
    renderables: Vec<Renderable3d>,
    signature: Vec<(u64, u64)>,
    grammar: CircleGrammar,
    constructions: BTreeMap<u64, (MeshHandle, Renderable3d)>,
    diagnostics: bool,
    body_mesh: MeshHandle,
    body_materials: BTreeMap<manonica_world::registry::ResourceKey, MaterialHandle>,
    particle_meshes: Vec<MeshHandle>,
}
impl SpellView {
    pub fn new(
        _device: &Device,
        _runtime: &tokio::runtime::Runtime,
        base: &mut GraphicsSession,
    ) -> Result<Self, String> {
        let materials = [[0.22, 0.9, 1.0], [0.75, 0.95, 1.0], [0.85, 0.5, 1.0]].map(|c| {
            base.render.materials.register(Material {
                base_color_factor: [c[0], c[1], c[2], 1.],
                emissive: c.map(|v| v * 0.6),
                unlit: true,
                ..Default::default()
            })
        });
        let body_mesh = base
            .render
            .meshes
            .register(meridian_sdk::cube_mesh_source(1.0))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            materials,
            meshes: vec![],
            renderables: vec![],
            signature: vec![(u64::MAX, u64::MAX)],
            grammar: CircleGrammar::from_jsonl(include_str!(
                "../../../../assets/demo/circles.jsonl"
            ))
            .map_err(|e| e.to_string())?,
            constructions: BTreeMap::new(),
            diagnostics: false,
            body_mesh,
            body_materials: BTreeMap::new(),
            particle_meshes: vec![],
        })
    }
    pub fn invalidate(&mut self) {
        self.signature = vec![(u64::MAX, u64::MAX)];
    }
    pub fn set_diagnostics(&mut self, enabled: bool) {
        if self.diagnostics != enabled {
            self.diagnostics = enabled;
            self.invalidate();
        }
    }
    pub fn update(
        &mut self,
        base: &mut GraphicsSession,
        state: &MagicState,
        world: &manonica_world::chunk::ChunkWorld,
    ) -> Result<(), String> {
        let signature = state
            .active()
            .iter()
            .map(|e| (e.id, e.field.simulated_seconds.to_bits()))
            .collect::<Vec<_>>();
        if signature == self.signature {
            return Ok(());
        }
        let mut meshes = [empty(), empty(), empty()];
        for event in state.active() {
            let f = &event.field;
            let average = f.manons() / (f.densities.len() as f64 * f.cell_width.powi(3));
            for (i, &n) in f.densities.iter().enumerate() {
                if !self.diagnostics
                    && n - f.reference_density[i] < 0.5_f64.max(f.reference_density[i] * 0.5)
                    || n <= 0.0
                {
                    continue;
                }
                let pos = f.position(i);
                let phase = i as f64 * 1.618;
                let jitter = [phase.sin(), (phase * 1.7).cos(), (phase * 0.7).sin()]
                    .map(|v| v * 0.28 * f.cell_width);
                let p = std::array::from_fn(|a| (pos[a] + jitter[a]) as f32);
                let size = (0.025 * (n / average.max(1e-6)).sqrt()).clamp(0.012, 0.13) as f32;
                particle(&mut meshes[0], p, size);
                if self.diagnostics && i % 11 == 0 {
                    let [nx, ny, nz] = f.dimensions;
                    let xyz = [i % nx, (i / nx) % ny, i / (nx * ny)];
                    let mut v = f.velocity[i];
                    for axis in 0..3 {
                        let mut lo = xyz;
                        lo[axis] = lo[axis].saturating_sub(1);
                        let mut hi = xyz;
                        hi[axis] = (hi[axis] + 1).min([nx, ny, nz][axis] - 1);
                        let distance = (hi[axis] - lo[axis]) as f64 * f.cell_width;
                        v[axis] -= event.program.transport.mobility
                            * (f.potential[index(f, hi)] - f.potential[index(f, lo)])
                            / distance;
                        v[axis] -= event.program.transport.diffusivity
                            * (f.densities[index(f, hi)] - f.densities[index(f, lo)])
                            / distance
                            / n.max(1e-9);
                    }
                    let speed = v.iter().map(|v| v * v).sum::<f64>().sqrt();
                    if speed > 1e-4 {
                        let q = std::array::from_fn(|a| p[a] + (v[a] / speed * 0.4) as f32);
                        beam(&mut meshes[1], p, q, 0.01);
                    }
                }
            }
            if self.diagnostics {
                contours(&mut meshes[2], f);
            }
            if !self.constructions.contains_key(&event.id) {
                let construction = self
                    .grammar
                    .compile(&event.program)
                    .map_err(|e| e.to_string())?;
                let mut mesh = empty();
                for path in &construction.paths {
                    for edge in path.windows(2) {
                        beam(
                            &mut mesh,
                            edge[0].map(|v| v as f32),
                            edge[1].map(|v| v as f32),
                            0.005,
                        );
                    }
                }
                for link in &construction.links {
                    beam(
                        &mut mesh,
                        link[0].map(|v| v as f32),
                        link[1].map(|v| v as f32),
                        0.003,
                    );
                }
                let radius = construction
                    .paths
                    .iter()
                    .flatten()
                    .map(|p| p[0].hypot(p[2]))
                    .fold(0.0, f64::max)
                    .max(1.0);
                let handle = base
                    .render
                    .meshes
                    .register(mesh)
                    .map_err(|e| e.to_string())?;
                let mut object = draw(handle, self.materials[2]);
                object.frame = Motor3::translation(Vec3::new(
                    event.center[0] as f32,
                    (event.center[1] + 0.04) as f32,
                    event.center[2] as f32,
                ));
                let scale = (2.8 / radius) as f32;
                object.scale =
                    Scale3::new(scale, scale, scale).expect("positive construction scale");
                self.constructions.insert(event.id, (handle, object));
            }
        }
        for (cell, density) in world.concentrations_above(7.5, 1024) {
            if manonica_world::gameplay::solid(world, cell, false) {
                continue;
            }
            let p = [
                cell.x as f32 + 0.5,
                cell.y as f32 + 0.5,
                cell.z as f32 + 0.5,
            ];
            particle(
                &mut meshes[0],
                p,
                ((density - 5.0).sqrt() * 0.012).clamp(0.012, 0.08) as f32,
            );
        }
        let mut handles = vec![];
        let mut renderables = vec![];
        for (i, mesh) in meshes.into_iter().enumerate() {
            if mesh.indices.is_empty() {
                continue;
            }
            let handle = base
                .render
                .meshes
                .register(mesh)
                .map_err(|e| e.to_string())?;
            handles.push(handle);
            renderables.push(draw(handle, self.materials[i]));
        }
        for handle in self.meshes.drain(..) {
            base.render.meshes.unregister(handle);
        }
        self.meshes = handles;
        self.renderables = renderables;
        let expired = self
            .constructions
            .keys()
            .filter(|id| !state.active().iter().any(|e| e.id == **id))
            .copied()
            .collect::<Vec<_>>();
        for id in expired {
            if let Some((mesh, _)) = self.constructions.remove(&id) {
                base.render.meshes.unregister(mesh);
            }
        }
        self.signature = signature;
        Ok(())
    }
    pub fn append_matter(
        &mut self,
        base: &mut GraphicsSession,
        scene: &mut meridian_sdk::Scene3d,
        state: &manonica_world::matter::MatterState,
        world: &manonica_world::chunk::ChunkWorld,
    ) {
        for mesh in self.particle_meshes.drain(..) {
            base.render.meshes.unregister(mesh);
        }
        let mut particles: BTreeMap<manonica_world::registry::ResourceKey, MeshSource> =
            BTreeMap::new();
        for event in &state.particles {
            let t = event.age_seconds;
            let mesh = particles.entry(event.block.clone()).or_insert_with(empty);
            for i in 0..18 {
                let phase = i as f64 * 2.399963;
                let speed = 1.0 + (i % 5) as f64 * 0.2;
                let p = [
                    event.position[0] + phase.cos() * speed * t,
                    event.position[1] + (1.2 + (i % 3) as f64 * 0.4) * t - 7.0 * t * t,
                    event.position[2] + phase.sin() * speed * t,
                ];
                particle(
                    mesh,
                    p.map(|v| v as f32),
                    (0.035 * (1.0 - t / 1.25).max(0.0)) as f32,
                );
            }
        }
        for body in &state.bodies {
            let material = if let Some(material) = self.body_materials.get(&body.block) {
                *material
            } else {
                let appearance = world
                    .content()
                    .block_id(&body.block)
                    .ok()
                    .and_then(|id| world.content().block(id).ok())
                    .map(|d| d.appearance)
                    .unwrap_or_default();
                let c = appearance.base_color;
                let material = base.render.materials.register(Material {
                    base_color_factor: [c[0], c[1], c[2], 1.0],
                    emissive: appearance.emissive,
                    unlit: true,
                    ..Default::default()
                });
                self.body_materials.insert(body.block.clone(), material);
                material
            };
            let p = body.motion.position_metres;
            let mut object = draw(self.body_mesh, material);
            object.frame = Motor3::translation(Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32));
            let extent = (world.layout().cell_width() * 0.5) as f32;
            object.scale = Scale3::new(extent, extent, extent).expect("positive cell extent");
            scene.renderables.push(object);
        }
        for (key, mesh) in particles {
            let material = if let Some(handle) = self.body_materials.get(&key) {
                *handle
            } else {
                let appearance = world
                    .content()
                    .block_id(&key)
                    .ok()
                    .and_then(|id| world.content().block(id).ok())
                    .map(|d| d.appearance)
                    .unwrap_or_default();
                let c = appearance.base_color;
                let handle = base.render.materials.register(Material {
                    base_color_factor: [c[0], c[1], c[2], 1.0],
                    unlit: true,
                    ..Default::default()
                });
                self.body_materials.insert(key, handle);
                handle
            };
            if let Ok(handle) = base.render.meshes.register(mesh) {
                scene.renderables.push(draw(handle, material));
                self.particle_meshes.push(handle);
            }
        }
    }
    pub fn append(&self, scene: &mut meridian_sdk::Scene3d, _state: &MagicState) {
        scene.renderables.extend(self.renderables.iter().cloned());
        scene.renderables.extend(
            self.constructions
                .values()
                .map(|(_, object)| object.clone()),
        );
    }
}
