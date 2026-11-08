use meridian_sdk::{
    GraphicsSession, Light, Material, MaterialHandle, MeshHandle, Motor3,
    Renderable3d, Scale3, Scene3d, Vec3, icosphere_mesh_source,
};

use crate::experiment::Experiment;
use crate::simulation::{Simulation, field::cell_center};
use super::circle_mesh::circle_mesh;

/// Визуализация *концентрации* свободных манонов светящимися объёмными
/// пробами на сетке. Это пока не ray-marching / 3D-текстура.
/// Эффект wisps — только отображение, не изменение физического поля.
pub struct FieldView {
    pub base: GraphicsSession,
    pub scene: Scene3d,
    sample_mesh: MeshHandle,
    halo_mesh: MeshHandle,
    levels: Vec<MaterialHandle>,
    static_count: usize,
    frame: u64,
}

impl FieldView {
    pub fn new(mut base: GraphicsSession, experiment: &Experiment) -> Result<Self, String> {
        let cell = experiment.grid.cell_size as f32;
        let sample_mesh = base.render.meshes
            .register(icosphere_mesh_source(1, cell * 0.38))
            .map_err(|e| format!("mana sample mesh: {e}"))?;
        let halo_mesh = base.render.meshes
            .register(icosphere_mesh_source(0, cell * 0.16))
            .map_err(|e| format!("mana halo mesh: {e}"))?;

        let mut levels = Vec::with_capacity(16);
        for index in 0..16 {
            let t = index as f32 / 15.0;
            // Почти чёрные низкие плотности и сильно контрастные горячие зоны.
            let rgb = [
                (0.06 + t.powf(4.0) * 1.6).min(2.0),
                0.08 + t.powf(1.6) * 0.75,
                0.32 + t.powf(0.7) * 0.95,
            ];
            let energy = experiment.visualization.exposure * (0.15 + t.powf(2.5) * 2.2);
            levels.push(base.render.materials.register(Material {
                base_color_factor: [rgb[0], rgb[1], rgb[2], 1.0],
                emissive: rgb.map(|c| c * energy),
                unlit: true,
                ..Default::default()
            }));
        }

        let mut renderables = Vec::new();
        for circle in &experiment.circles {
            let mesh = base.render.meshes.register(circle_mesh(circle))
                .map_err(|e| format!("circle {}: {e}", circle.name))?;
            let material = base.render.materials.register(Material {
                base_color_factor: [circle.color[0], circle.color[1], circle.color[2], 1.0],
                emissive: circle.color.map(|v| v * experiment.visualization.exposure * 1.5),
                unlit: true,
                ..Default::default()
            });
            renderables.push(Renderable3d {
                mesh, material,
                frame: Motor3::translation(Vec3::new(
                    circle.center[0], circle.center[1], circle.center[2],
                )),
                scale: Scale3::UNIT,
                billboard: false,
                casts_shadow: false,
                palette: None,
                morph_weights: Vec::new(),
            });
        }
        let static_count = renderables.len();
        let scene = Scene3d {
            renderables,
            lights: vec![
                Light::Point {
                    position: Motor3::translation(Vec3::ZERO),
                    color: [0.28, 0.4, 1.0],
                    intensity: 2.0,
                    range: 11.0,
                },
            ],
            ambient_ground: [0.007, 0.008, 0.02],
            ambient_sky: [0.012, 0.015, 0.035],
            ..Scene3d::default()
        };
        Ok(Self {
            base, scene, sample_mesh, halo_mesh, levels,
            static_count, frame: 0,
        })
    }

    pub fn update(&mut self, session: &Simulation) {
        self.frame += 1;
        // При высокой частоте экрана обновляем объёмное поле ~30 раз/с,
        // однако камера и сам рендер обновляются каждый кадр.
        if self.frame % 2 != 0 { return; }
        let field = session.field();
        let spec = &session.experiment;
        let shape = field.shape();
        let mut cells: Vec<(usize, f64)> = Vec::new();
        let mut peak = 0.0_f64;
        for z in 0..shape.nz() {
            for y in 0..shape.ny() {
                for x in 0..shape.nx() {
                    let density = field.at(x, y, z).expect("valid cell").value();
                    peak = peak.max(density);
                    let id = shape.index(x, y, z).expect("valid index");
                    cells.push((id, density));
                }
            }
        }

        self.scene.renderables.truncate(self.static_count);
        if peak <= 0.0 { return; }
        let threshold = spec.visualization.relative_threshold;
        cells.retain(|(_, density)| *density >= peak * threshold);
        cells.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        cells.truncate(spec.visualization.max_points);

        for (rank, (id, density)) in cells.into_iter().enumerate() {
            let x = id % shape.nx();
            let y = (id / shape.nx()) % shape.ny();
            let z = id / (shape.nx() * shape.ny());
            let position = cell_center(shape, field.cell_width_metres(), x, y, z);
            let norm = (density / peak).clamp(0.0, 1.0);
            let intensity = norm.sqrt();
            let level = ((intensity * 15.0) as usize).min(15);
            let pos = Vec3::new(position[0] as f32, position[1] as f32, position[2] as f32);
            self.scene.renderables.push(self.sample(self.sample_mesh, level, pos));

            if spec.visualization.wisps && rank < 180 && norm > 0.22 {
                let t = session.elapsed() as f32;
                let id_phase = id as f32 * 0.618_033_9;
                let jitter = 0.065 * (0.2 + norm as f32);
                let displaced = Vec3::new(
                    pos.x + jitter * (t * 1.7 + id_phase).sin(),
                    pos.y + 0.12 + jitter * (t * 2.7 + id_phase * 0.5).sin(),
                    pos.z + jitter * (t * 2.1 + id_phase * 0.73).cos(),
                );
                self.scene.renderables.push(self.sample(self.halo_mesh, level, displaced));
            }
        }
    }

    fn sample(&self, mesh: MeshHandle, level: usize, position: Vec3) -> Renderable3d {
        Renderable3d {
            mesh,
            material: self.levels[level],
            frame: Motor3::translation(position),
            scale: Scale3::UNIT,
            billboard: false,
            casts_shadow: false,
            palette: None,
            morph_weights: Vec::new(),
        }
    }
}
