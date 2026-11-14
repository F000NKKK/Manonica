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
fn thermal_emission(kelvin: f64) -> [f32; 3] {
    let normal = 1.0 / (650e-9_f64.powi(5) * (0.014387768775_f64 / (650e-9_f64 * 3000.)).exp_m1());
    [650e-9_f64, 550e-9, 450e-9].map(|wavelength| {
        ((1.0 / (wavelength.powi(5) * (0.014387768775 / (wavelength * kelvin.max(1.))).exp_m1()))
            / normal
            * 8.0)
            .clamp(0.0, 1000.) as f32
    })
}
fn thermal_bucket(kelvin: f64) -> usize {
    let temperatures = [800., 1100., 1500., 1900., 2400., 3000., 4000., 6000.];
    temperatures.iter().position(|t| kelvin <= *t).unwrap_or(7)
}
struct FieldTracers {
    time: f64,
    positions: Vec<[f64; 3]>,
}
fn field_index(field: &ManonField, position: [f64; 3]) -> usize {
    let c = std::array::from_fn::<_, 3, _>(|a| {
        ((position[a] / field.cell_width - field.origin[a] as f64).floor() as isize)
            .clamp(0, field.dimensions[a] as isize - 1) as usize
    });
    index(field, c)
}
fn make_tracers(field: &ManonField) -> FieldTracers {
    let sum = field.densities.iter().sum::<f64>();
    let mut cursor = 0;
    let mut accumulated = field.densities[0];
    let mut positions = vec![];
    for k in 0..512 {
        let target = sum * (k as f64 + 0.5) / 512.;
        while cursor + 1 < field.densities.len() && accumulated < target {
            cursor += 1;
            accumulated += field.densities[cursor];
        }
        let center = field.position(cursor);
        let phase = k as f64 * 2.399963;
        positions.push([
            center[0] + phase.sin() * 0.3 * field.cell_width,
            center[1] + (phase * 1.7).cos() * 0.3 * field.cell_width,
            center[2] + (phase * 0.7).sin() * 0.3 * field.cell_width,
        ]);
    }
    FieldTracers {
        time: field.simulated_seconds,
        positions,
    }
}
pub struct SpellView {
    materials: [MaterialHandle; 3],
    emission_materials: [MaterialHandle; 8],
    thermal_lights: Vec<meridian_sdk::Light>,
    tracers: BTreeMap<u64, FieldTracers>,
    meshes: Vec<MeshHandle>,
    renderables: Vec<Renderable3d>,
    signature: Vec<(u64, u64)>,
    grammar: CircleGrammar,
    constructions: BTreeMap<u64, (MeshHandle, Renderable3d)>,
    diagnostics: bool,
    body_mesh: MeshHandle,
    body_materials: BTreeMap<manonica_world::registry::ResourceKey, MaterialHandle>,
    particle_meshes: Vec<MeshHandle>,
    dust_materials: BTreeMap<(manonica_world::registry::ResourceKey, u8, u8), MaterialHandle>,
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
        let emission_materials = [800., 1100., 1500., 1900., 2400., 3000., 4000., 6000.].map(|t| {
            base.render.materials.register(Material {
                base_color_factor: [0.15, 0.15, 0.15, 1.],
                emissive: thermal_emission(t),
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
            emission_materials,
            thermal_lights: vec![],
            tracers: BTreeMap::new(),
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
            dust_materials: BTreeMap::new(),
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
        matter: &manonica_world::matter::MatterState,
        thermodynamics: manonica_world::thermal::MediumThermodynamics,
    ) -> Result<(), String> {
        let mut signature = state
            .active()
            .iter()
            .map(|e| (e.id, e.field.simulated_seconds.to_bits()))
            .collect::<Vec<_>>();
        let world_heat = matter
            .medium_heat
            .iter()
            .map(|c| c.heat_joules)
            .sum::<f64>();
        signature.push((u64::MAX, (world_heat.ln_1p() * 100.0) as u64));
        if signature == self.signature {
            return Ok(());
        }
        let mut meshes = std::array::from_fn::<_, 11, _>(|_| empty());
        for event in state.active() {
            let f = &event.field;
            let tracers = self
                .tracers
                .entry(event.id)
                .or_insert_with(|| make_tracers(f));
            let delta = (f.simulated_seconds - tracers.time).max(0.0);
            let substeps = (delta / 0.01).ceil().max(1.0) as usize;
            for p in &mut tracers.positions {
                for _ in 0..substeps {
                    let cell = field_index(f, *p);
                    let velocity = f.velocity[cell];
                    for a in 0..3 {
                        p[a] = (p[a] + velocity[a] * delta / substeps as f64).clamp(
                            f.origin[a] as f64 * f.cell_width + 0.02 * f.cell_width,
                            (f.origin[a] + f.dimensions[a] as i64) as f64 * f.cell_width
                                - 0.02 * f.cell_width,
                        );
                    }
                }
                let cell = field_index(f, *p);
                let n = f.densities[cell];
                if n - f.reference_density[cell] < 0.5_f64.max(f.reference_density[cell] * 0.5) {
                    continue;
                }
                let size = (0.023 * (n / f.reference_density[cell].max(0.1)).sqrt())
                    .clamp(0.015, 0.09) as f32;
                particle(&mut meshes[0], p.map(|v| v as f32), size);
            }
            tracers.time = f.simulated_seconds;
            let average = f.manons() / (f.densities.len() as f64 * f.cell_width.powi(3));
            for (i, &n) in f.densities.iter().enumerate() {
                let temperature = f.thermal.temperature(i, f.cell_width.powi(3));
                if !self.diagnostics && temperature < 700. {
                    continue;
                }
                let pos = f.position(i);
                let phase = i as f64 * 1.618;
                let jitter = [phase.sin(), (phase * 1.7).cos(), (phase * 0.7).sin()]
                    .map(|v| v * 0.28 * f.cell_width);
                let p = std::array::from_fn(|a| (pos[a] + jitter[a]) as f32);
                let size = (0.025 * (n / average.max(1e-6)).sqrt()).clamp(0.012, 0.13) as f32;
                let bucket = if temperature >= 700. {
                    3 + thermal_bucket(temperature)
                } else {
                    0
                };
                particle(
                    &mut meshes[bucket],
                    p,
                    if temperature >= 700. {
                        size.max(0.12)
                    } else {
                        size
                    },
                );
                if temperature > 1000. {
                    for phase in [0.0_f32, 2.1, 4.2] {
                        let q = [
                            p[0] + phase.cos() * 0.18,
                            p[1] + (phase + 1.0).sin() * 0.12,
                            p[2] + phase.sin() * 0.18,
                        ];
                        particle(&mut meshes[bucket], q, 0.08);
                    }
                }

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
        for cell in &matter.medium_heat {
            let temperature =
                thermodynamics.temperature(cell.heat_joules, world.layout().cell_width().powi(3));
            if temperature < 700. {
                continue;
            }
            let dx = world.layout().cell_width();
            let p = cell.position.map(|v| ((v as f64 + 0.5) * dx) as f32);
            particle(&mut meshes[3 + thermal_bucket(temperature)], p, 0.14);
        }
        let mut sources = vec![];
        for event in state.active() {
            let f = &event.field;
            for (i, q) in f.thermal.cell_heat_joules.iter().enumerate() {
                let t = f.thermal.temperature(i, f.cell_width.powi(3));
                if t < 1000. || *q <= 0.0 {
                    continue;
                }
                let power = f.thermal.settings.emissivity
                    * 5.670374419e-8
                    * 6.
                    * f.cell_width.powi(2)
                    * (t.powi(4) - f.thermal.settings.reference_temperature_kelvin.powi(4));
                sources.push((power, f.position(i), t));
            }
        }
        for cell in &matter.medium_heat {
            let t =
                thermodynamics.temperature(cell.heat_joules, world.layout().cell_width().powi(3));
            if t < 1000. {
                continue;
            }
            let p = cell
                .position
                .map(|v| (v as f64 + 0.5) * world.layout().cell_width());
            let power = thermodynamics.emissivity
                * 5.670374419e-8
                * 6.
                * world.layout().cell_width().powi(2)
                * (t.powi(4) - thermodynamics.reference_temperature_kelvin.powi(4));
            sources.push((power, p, t));
        }
        sources.sort_by(|a, b| b.0.total_cmp(&a.0));
        self.thermal_lights = sources
            .into_iter()
            .take(4)
            .map(|(power, p, t)| {
                let c = thermal_emission(t);
                let peak = c.iter().copied().fold(1e-6_f32, f32::max);
                meridian_sdk::Light::Point {
                    position: Motor3::translation(Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32)),
                    color: c.map(|v| v / peak),
                    intensity: (power / 10000.).min(100.) as f32,
                    range: 20.,
                }
            })
            .collect();
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
            renderables.push(draw(
                handle,
                if i < 3 {
                    self.materials[i]
                } else {
                    self.emission_materials[i - 3]
                },
            ));
        }
        for anchor in &matter.anchors {
            let p = manonica_world::chunk::CellPosition {
                x: anchor.position[0],
                y: anchor.position[1] as usize,
                z: anchor.position[2],
            };
            let Ok(state) = world.block(p) else { continue };
            let Some(mechanics) = world
                .content()
                .block(state.block())
                .ok()
                .and_then(|d| d.physics.mechanics)
            else {
                continue;
            };
            let temperature = thermodynamics.reference_temperature_kelvin
                + anchor.heat_joules / mechanics.heat_capacity_joules_per_kelvin;
            if temperature < 1500. {
                continue;
            }
            let dx = world.layout().cell_width();
            let mut object = draw(
                self.body_mesh,
                self.emission_materials[thermal_bucket(temperature)],
            );
            object.frame = Motor3::translation(Vec3::new(
                ((anchor.position[0] as f64 + 0.5) * dx) as f32,
                ((anchor.position[1] as f64 + 0.5) * dx) as f32,
                ((anchor.position[2] as f64 + 0.5) * dx) as f32,
            ));
            let extent = (dx * 0.501) as f32;
            object.scale =
                Scale3::new(extent, extent, extent).expect("positive thermal surface extent");
            renderables.push(object);
        }
        for handle in self.meshes.drain(..) {
            base.render.meshes.unregister(handle);
        }
        self.meshes = handles;
        self.renderables = renderables;
        self.tracers
            .retain(|id, _| state.active().iter().any(|e| e.id == *id));
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
        physics: manonica_world::matter::WorldPhysics,
    ) {
        for mesh in self.particle_meshes.drain(..) {
            base.render.meshes.unregister(mesh);
        }
        let mut particles: BTreeMap<(manonica_world::registry::ResourceKey, u8, u8), MeshSource> =
            BTreeMap::new();
        for event in &state.particles {
            let t = event.age_seconds;
            let fraction = (t / event.lifetime_seconds).clamp(0., 1.);
            let temperature = physics.thermal.reference_temperature_kelvin
                + event.heat_joules / event.capacity_joules_per_kelvin;
            let hot = event.lifetime_seconds > 1.25;
            let temp_bucket = if temperature >= 700. {
                1 + thermal_bucket(temperature) as u8
            } else {
                0
            };
            let age_bucket = (fraction * 8.) as u8;
            let mesh = particles
                .entry((event.block.clone(), age_bucket.min(7), temp_bucket))
                .or_insert_with(empty);
            let drag = 3.0;
            let travel = (1.0 - (-drag * t).exp()) / drag;
            let buoyancy = if hot {
                physics.gravity_metres_per_second_squared
                    * (1.0
                        - physics.thermal.reference_temperature_kelvin
                            / temperature.max(physics.thermal.reference_temperature_kelvin))
            } else {
                -physics.gravity_metres_per_second_squared
            };
            for i in 0..18 {
                let phase = (i / 2) as f64 * 2.399963;
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                let scatter = if hot { 2.5 } else { 1.0 };
                let mut p = [
                    event.position[0]
                        + event.velocity[0] * travel
                        + sign * phase.cos() * scatter * travel,
                    event.position[1]
                        + event.velocity[1] * travel
                        + buoyancy / drag * (t - travel)
                        + sign * (phase * 1.7).sin() * scatter * travel,
                    event.position[2]
                        + event.velocity[2] * travel
                        + sign * phase.sin() * scatter * travel,
                ];
                if !hot
                    && let Some(ground) = manonica_world::gameplay::raycast(
                        world,
                        [p[0], event.position[1], p[2]],
                        [0., -1., 0.],
                        6.0,
                    )
                {
                    p[1] = p[1].max(ground.block.y as f64 + 1.02);
                }
                particle(
                    mesh,
                    p.map(|v| v as f32),
                    if hot {
                        (0.08 + 0.12 * t).min(0.45) as f32
                    } else {
                        (0.035 * (1.0 - fraction)) as f32
                    },
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
                    unlit: false,
                    ..Default::default()
                });
                self.body_materials.insert(body.block.clone(), material);
                material
            };
            let capacity = world
                .content()
                .block_id(&body.block)
                .ok()
                .and_then(|id| world.content().block(id).ok())
                .and_then(|d| d.physics.mechanics)
                .map(|m| m.heat_capacity_joules_per_kelvin)
                .unwrap_or(100.);
            let temperature =
                physics.thermal.reference_temperature_kelvin + body.heat_joules / capacity;
            let material = if temperature > 1500. {
                self.emission_materials[thermal_bucket(temperature)]
            } else {
                material
            };
            let p = body.motion.position_metres;
            let mut object = draw(self.body_mesh, material);
            object.frame = Motor3::translation(Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32));
            let extent = (world.layout().cell_width() * 0.5) as f32;
            object.scale = Scale3::new(extent, extent, extent).expect("positive cell extent");
            scene.renderables.push(object);
        }
        for ((block, age, temp), mesh) in particles {
            let key = (block.clone(), age, temp);
            let material = if let Some(material) = self.dust_materials.get(&key) {
                *material
            } else {
                let appearance = world
                    .content()
                    .block_id(&block)
                    .ok()
                    .and_then(|id| world.content().block(id).ok())
                    .map(|d| d.appearance)
                    .unwrap_or_default();
                let c = appearance.base_color;
                let alpha = (1.0 - age as f32 / 8.0).powi(2) * 0.65;
                let emission = if temp > 0 {
                    thermal_emission(
                        [800., 1100., 1500., 1900., 2400., 3000., 4000., 6000.][temp as usize - 1],
                    )
                    .map(|v| v * alpha)
                } else {
                    [0.; 3]
                };
                let material = base.render.materials.register(Material {
                    base_color_factor: [c[0], c[1], c[2], alpha],
                    emissive: emission,
                    unlit: true,
                    blend: meridian_sdk::BlendMode::AlphaBlend,
                    ..Default::default()
                });
                self.dust_materials.insert(key, material);
                material
            };
            if let Ok(handle) = base.render.meshes.register(mesh) {
                scene.renderables.push(draw(handle, material));
                self.particle_meshes.push(handle);
            }
        }
    }
    pub fn append(&self, scene: &mut meridian_sdk::Scene3d, _state: &MagicState) {
        scene.renderables.extend(self.renderables.iter().cloned());
        scene.lights.extend(self.thermal_lights.iter().cloned());
        scene.renderables.extend(
            self.constructions
                .values()
                .map(|(_, object)| object.clone()),
        );
    }
}
