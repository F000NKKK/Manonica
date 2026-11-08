//! Static magical construction geometry and a continuously sampled mana volume.
use meridian_sdk::{
    GraphicsSession, Light, Material, Motor3, Renderable3d, Scale3, Scene3d, Vec3,
};

use crate::experiment::Experiment;
use crate::simulation::Simulation;

use super::circle_mesh::circle_mesh;
use super::volume::ManaVolume;

/// The magic circles are scene geometry, while the free-manon density is
/// rendered as a translucent volume. Neither display changes the solver.
pub struct FieldView {
    pub base: GraphicsSession,
    pub scene: Scene3d,
    volume: ManaVolume,
    static_count: usize,
}

impl FieldView {
    pub fn new(mut base: GraphicsSession, experiment: &Experiment) -> Result<Self, String> {
        let mut renderables = Vec::new();
        for circle in &experiment.circles {
            let mesh = base
                .render
                .meshes
                .register(circle_mesh(circle))
                .map_err(|e| format!("circle {}: {e}", circle.name))?;
            let material = base.render.materials.register(Material {
                base_color_factor: [circle.color[0], circle.color[1], circle.color[2], 1.0],
                emissive: circle.color.map(|v| v * experiment.visualization.exposure * 1.5),
                unlit: true,
                ..Default::default()
            });
            renderables.push(Renderable3d {
                mesh,
                material,
                frame: Motor3::translation(Vec3::new(
                    circle.center[0],
                    circle.center[1],
                    circle.center[2],
                )),
                scale: Scale3::UNIT,
                billboard: false,
                casts_shadow: false,
                palette: None,
                morph_weights: Vec::new(),
            });
        }

        let static_count = renderables.len();
        let volume = ManaVolume::new(&mut base, experiment)?;
        let scene = Scene3d {
            renderables,
            lights: vec![Light::Point {
                position: Motor3::translation(Vec3::ZERO),
                color: [0.28, 0.4, 1.0],
                intensity: 2.0,
                range: 11.0,
            }],
            ambient_ground: [0.007, 0.008, 0.02],
            ambient_sky: [0.012, 0.015, 0.035],
            ..Scene3d::default()
        };

        Ok(Self {
            base,
            scene,
            volume,
            static_count,
        })
    }

    pub fn update(&mut self, session: &Simulation) {
        self.volume.update(
            &mut self.base,
            &mut self.scene,
            self.static_count,
            session,
        );
    }

    pub fn visible_slices(&self) -> usize {
        self.volume.visible_slices()
    }

    pub fn peak_density(&self) -> f64 {
        self.volume.peak_density()
    }
}
