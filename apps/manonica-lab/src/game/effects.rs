//! GPU-authored 3D particles, volumetric casting circles and actual liberated block bodies.
use manonica_world::magic::{MagicState, SpellEffect};
use meridian_gpu_wgpu::BufferUsage;
use meridian_sdk::{
    Device, GraphicsSession, Material, MaterialHandle, MeshHandle, MeshSource, Motor3,
    Renderable3d, Rotor, Scale3, Vec3, cube_mesh_source,
};
const PARTICLES: usize = 384;
const FRAMES: usize = 64;
pub struct SpellView {
    materials: [[MaterialHandle; 4]; 3],
    frames: Vec<Vec<Vec<MeshHandle>>>,
    circle: MeshHandle,
    debris: MeshHandle,
    debris_material: MaterialHandle,
    mana: MeshHandle,
}
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
fn octahedron(mesh: &mut MeshSource, center: [f32; 3], size: f32) {
    let base = mesh.positions.len() as u32;
    for d in [
        [1., 0., 0.],
        [-1., 0., 0.],
        [0., 1., 0.],
        [0., -1., 0.],
        [0., 0., 1.],
        [0., 0., -1.],
    ] {
        mesh.positions
            .push(std::array::from_fn(|a| center[a] + d[a] * size));
        mesh.normals.push(d);
        mesh.uvs.push([0., 0.]);
    }
    for triangle in [
        [2, 0, 4],
        [2, 4, 1],
        [2, 1, 5],
        [2, 5, 0],
        [3, 4, 0],
        [3, 1, 4],
        [3, 5, 1],
        [3, 0, 5],
    ] {
        mesh.indices.extend(triangle.map(|i| base + i));
    }
}
fn beam(mesh: &mut MeshSource, a: [f32; 3], b: [f32; 3], radius: f32) {
    let center: [f32; 3] = std::array::from_fn(|i| (a[i] + b[i]) * 0.5);
    let direction = std::array::from_fn::<_, 3, _>(|i| b[i] - a[i]);
    let length = direction.iter().map(|v| v * v).sum::<f32>().sqrt();
    if length < 1e-6 {
        return;
    }
    let d = direction.map(|v| v / length);
    let up = if d[1].abs() < 0.9 {
        [0., 1., 0.]
    } else {
        [1., 0., 0.]
    };
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let u = cross(d, up);
    let n = u.iter().map(|v| v * v).sum::<f32>().sqrt();
    let u = u.map(|v| v / n);
    let v = cross(d, u);
    let base = mesh.positions.len() as u32;
    for sign in [-1., 1.] {
        for k in 0..4 {
            let angle = k as f32 * std::f32::consts::FRAC_PI_2;
            let normal =
                std::array::from_fn::<_, 3, _>(|i| u[i] * angle.cos() + v[i] * angle.sin());
            mesh.positions.push(std::array::from_fn(|i| {
                center[i] + sign * length * 0.5 * d[i] + normal[i] * radius
            }));
            mesh.normals.push(normal);
            mesh.uvs.push([0., 0.]);
        }
    }
    for k in 0..4 {
        let next = (k + 1) % 4;
        mesh.indices.extend([
            base + k,
            base + next,
            base + k + 4,
            base + next,
            base + next + 4,
            base + k + 4,
        ]);
    }
}
fn circle_mesh() -> MeshSource {
    let mut mesh = empty();
    for radius in [0.72_f32, 0.92, 1.0] {
        for i in 0..128 {
            let a = i as f32 / 128. * std::f32::consts::TAU;
            let b = (i + 1) as f32 / 128. * std::f32::consts::TAU;
            beam(
                &mut mesh,
                [radius * a.cos(), 0., radius * a.sin()],
                [radius * b.cos(), 0., radius * b.sin()],
                0.006,
            );
        }
    }
    for i in 0..12 {
        let a = i as f32 / 12. * std::f32::consts::TAU;
        let b = (i + 5) as f32 / 12. * std::f32::consts::TAU;
        beam(
            &mut mesh,
            [0.72 * a.cos(), 0., 0.72 * a.sin()],
            [0.72 * b.cos(), 0., 0.72 * b.sin()],
            0.004,
        );
        let c = a + 0.035;
        beam(
            &mut mesh,
            [0.94 * a.cos(), 0., 0.94 * a.sin()],
            [0.82 * c.cos(), 0., 0.82 * c.sin()],
            0.01,
        );
    }
    mesh
}
fn renderable(
    mesh: MeshHandle,
    material: MaterialHandle,
    center: [f64; 3],
    scale: [f64; 3],
    rotation: f32,
) -> Renderable3d {
    Renderable3d {
        mesh,
        material,
        frame: Motor3::from_rotation_translation(
            Rotor::from_axis_angle(Vec3::Y, rotation),
            Vec3::new(center[0] as f32, center[1] as f32, center[2] as f32),
        ),
        scale: Scale3::new(scale[0] as f32, scale[1] as f32, scale[2] as f32)
            .expect("positive effect scale"),
        billboard: false,
        casts_shadow: false,
        palette: None,
        morph_weights: vec![],
    }
}
impl SpellView {
    pub fn new(
        device: &Device,
        runtime: &tokio::runtime::Runtime,
        base: &mut GraphicsSession,
    ) -> Result<Self, String> {
        let buffer = device.create_buffer(3 * FRAMES * PARTICLES * 16, BufferUsage::Storage);
        let shader = device.create_shader("Manonica 3D particles", include_str!("effects.wgsl"));
        let pipeline = device.create_compute_pipeline(&shader, "bake");
        let group = device.create_compute_bind_group(&pipeline, &[&buffer]);
        let mut commands = device.create_command_buffer();
        commands.dispatch_compute(&pipeline, &group, (3 * FRAMES * PARTICLES) as u32 / 64);
        commands.submit();
        let data = runtime.block_on(device.read_buffer(&buffer));
        let colors = [
            [
                [1., 0.23, 0.015],
                [1., 0.65, 0.05],
                [1., 0.95, 0.5],
                [0.55, 0.25, 0.09],
            ],
            [
                [0.18, 0.6, 1.0],
                [0.4, 0.9, 1.0],
                [0.6, 0.4, 0.2],
                [0.9, 0.7, 0.3],
            ],
            [
                [0.25, 0.6, 0.75],
                [0.6, 0.9, 1.0],
                [0.17, 0.4, 0.45],
                [0.38, 0.3, 0.17],
            ],
        ];
        let materials = colors.map(|family| {
            family.map(|c| {
                base.render.materials.register(Material {
                    base_color_factor: [c[0], c[1], c[2], 1.0],
                    emissive: c.map(|v| v * 0.7),
                    unlit: true,
                    ..Default::default()
                })
            })
        });
        let mut frames = vec![];
        for family in 0..3 {
            let mut series = vec![];
            for frame in 0..FRAMES {
                let mut buckets: Vec<_> = (0..4).map(|_| empty()).collect();
                for particle in 0..PARTICLES {
                    let offset = ((family * FRAMES + frame) * PARTICLES + particle) * 16;
                    let v = std::array::from_fn::<_, 4, _>(|i| {
                        f32::from_le_bytes(
                            data[offset + i * 4..offset + i * 4 + 4].try_into().unwrap(),
                        )
                    });
                    octahedron(
                        &mut buckets[particle % 4],
                        [v[0], v[1], v[2]],
                        v[3].max(1e-5),
                    );
                }
                let mut handles = vec![];
                for mesh in buckets {
                    handles.push(
                        base.render
                            .meshes
                            .register(mesh)
                            .map_err(|e| e.to_string())?,
                    );
                }
                series.push(handles);
            }
            frames.push(series);
        }
        let circle = base
            .render
            .meshes
            .register(circle_mesh())
            .map_err(|e| e.to_string())?;
        let debris = base
            .render
            .meshes
            .register(cube_mesh_source(1.0))
            .map_err(|e| e.to_string())?;
        let debris_material = base.render.materials.register(Material {
            base_color_factor: [0.4, 0.3, 0.2, 1.],
            unlit: true,
            ..Default::default()
        });
        let mut mote = empty();
        octahedron(&mut mote, [0.; 3], 1.0);
        let mana = base
            .render
            .meshes
            .register(mote)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            mana,
            materials,
            frames,
            circle,
            debris,
            debris_material,
        })
    }
    pub fn append_mana(
        &self,
        scene: &mut meridian_sdk::Scene3d,
        world: &manonica_world::chunk::ChunkWorld,
        eye: [f64; 3],
        time: f64,
    ) {
        let origin = eye.map(|v| (v / 2.0).floor() as i64);
        for z in -3..=3 {
            for y in -2..=2 {
                for x in -3..=3 {
                    let cell = [
                        (origin[0] + x) * 2,
                        (origin[1] + y) * 2,
                        (origin[2] + z) * 2,
                    ];
                    if cell[1] < 0 {
                        continue;
                    }
                    let p = manonica_world::chunk::CellPosition {
                        x: cell[0],
                        y: cell[1] as usize,
                        z: cell[2],
                    };
                    let Ok(density) = world.density(p) else {
                        continue;
                    };
                    if density <= 0.0 || manonica_world::gameplay::solid(world, p, false) {
                        continue;
                    }
                    let seed = (cell[0] * 127 + cell[1] * 47 + cell[2] * 311) as f64;
                    let center = [
                        cell[0] as f64 + 1.0 + 0.6 * (seed + time * 0.3).sin(),
                        cell[1] as f64 + 0.8 + 0.5 * (seed * 1.7 + time * 0.6).cos(),
                        cell[2] as f64 + 1.0 + 0.6 * (seed * 0.7 + time * 0.25).sin(),
                    ];
                    let size = (density.sqrt() * 0.018).clamp(0.01, 0.08);
                    scene.renderables.push(renderable(
                        self.mana,
                        self.materials[2][1],
                        center,
                        [size; 3],
                        time as f32,
                    ));
                }
            }
        }
    }
    pub fn append(&self, scene: &mut meridian_sdk::Scene3d, state: &MagicState) {
        for event in state.active() {
            let index = event.effect.index();
            let fraction = event.age_seconds / event.effect.duration();
            let frame = if index == 2 {
                (event.age_seconds * 32.0) as usize % FRAMES
            } else {
                (fraction * FRAMES as f64) as usize
            };
            let (scale, center) = match event.effect {
                SpellEffect::Explosion { radius } => ([radius; 3], event.center),
                SpellEffect::EarthRift { length, width, .. } => {
                    ([width, 4.0, length * 0.5], event.center)
                }
                SpellEffect::Tornado { radius, height, .. } => {
                    ([radius, height, radius], event.center)
                }
            };
            let rotation = if index == 1 {
                -(event.direction[0].atan2(event.direction[2]) as f32)
            } else {
                0.
            };
            for bucket in 0..4 {
                scene.renderables.push(renderable(
                    self.frames[index][frame.min(FRAMES - 1)][bucket],
                    self.materials[index][bucket],
                    center,
                    scale,
                    rotation,
                ));
            }
            let size = match event.effect {
                SpellEffect::Explosion { radius } => radius * 1.1,
                SpellEffect::EarthRift { length, .. } => length * 0.4,
                SpellEffect::Tornado { radius, .. } => radius * 1.1,
            };
            let growth = (event.age_seconds / 0.2).clamp(0.02, 1.0);
            let fade = ((event.effect.duration() - event.age_seconds) / 0.3).clamp(0.02, 1.0);
            let size = size * growth * fade;
            let mut origin = event.center;
            origin[1] += 0.06 + (event.age_seconds * 3.0).sin() * 0.03;
            scene.renderables.push(renderable(
                self.circle,
                self.materials[index][1],
                origin,
                [size, 1., size],
                event.age_seconds as f32 * 0.3,
            ));
            let origin = [
                event.source[0] as f64 + 0.5,
                event.source[1] as f64 - 1.55,
                event.source[2] as f64 + 0.5,
            ];
            if event.age_seconds < 0.8 {
                scene.renderables.push(renderable(
                    self.circle,
                    self.materials[index][1],
                    origin,
                    [0.9, 0.9, 0.9],
                    -event.age_seconds as f32,
                ));
            }
        }
        for body in state.debris() {
            scene.renderables.push(renderable(
                self.debris,
                self.debris_material,
                body.motion.position_metres,
                [0.44; 3],
                if body.resting {
                    0.0
                } else {
                    body.age_seconds as f32 * 1.5
                },
            ));
        }
    }
}
