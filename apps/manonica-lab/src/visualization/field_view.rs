//! Static magical construction geometry and a continuously sampled mana volume.
use meridian_sdk::{GraphicsSession, Light, Material, Motor3, Renderable3d, Scale3, Scene3d, Vec3};

use crate::experiment::Experiment;
use crate::simulation::SimulationSnapshot;

use super::circle_mesh::formula_mesh;
use super::volume::ManaVolume;

/// The magic circles are scene geometry, while the free-manon density is
/// rendered as a translucent volume. Neither display changes the solver.
pub struct FieldView {
    pub base: GraphicsSession,
    pub scene: Scene3d,
    volume: ManaVolume,
    static_count: usize,
}

fn append_mechanical_markers(
    base: &mut GraphicsSession,
    renderables: &mut Vec<Renderable3d>,
    experiment: &Experiment,
) -> Result<(), String> {
    if let Some(mechanics) = &experiment.mechanics {
        for (i, body) in mechanics.bodies.iter().enumerate() {
            let color = if body.charge >= 0.0 {
                [0.2, 0.8, 1.0]
            } else {
                [1.0, 0.4, 0.2]
            };
            let formula = crate::experiment::Formula {
                name: format!("body {}", i + 1),
                x: "0.15*cos(t)".into(),
                y: "0".into(),
                z: "0.15*sin(t)".into(),
                segments: 32,
                thickness: 0.018,
                color,
                trap: None,
            };
            let mesh = base
                .render
                .meshes
                .register(formula_mesh(&formula)?)
                .map_err(|e| e.to_string())?;
            let material = base.render.materials.register(Material {
                base_color_factor: [color[0], color[1], color[2], 1.0],
                emissive: color,
                unlit: true,
                ..Default::default()
            });
            let x = body.position.map(|v| v as f32);
            renderables.push(Renderable3d {
                mesh,
                material,
                frame: Motor3::translation(Vec3::new(x[0], x[1], x[2])),
                scale: Scale3::UNIT,
                billboard: false,
                casts_shadow: false,
                palette: None,
                morph_weights: Vec::new(),
            });
        }
    }
    Ok(())
}
impl FieldView {
    pub fn new(mut base: GraphicsSession, experiment: &Experiment) -> Result<Self, String> {
        let mut renderables = Vec::new();
        for formula in &experiment.formulas {
            let mesh = base
                .render
                .meshes
                .register(formula_mesh(formula)?)
                .map_err(|e| format!("formula {}: {e}", formula.name))?;
            let material = base.render.materials.register(Material {
                base_color_factor: [formula.color[0], formula.color[1], formula.color[2], 1.0],
                emissive: formula
                    .color
                    .map(|v| v * experiment.visualization.exposure * 1.5),
                unlit: true,
                ..Default::default()
            });
            renderables.push(Renderable3d {
                mesh,
                material,
                frame: Motor3::translation(Vec3::ZERO),
                scale: Scale3::UNIT,
                billboard: false,
                casts_shadow: false,
                palette: None,
                morph_weights: Vec::new(),
            });
        }

        append_mechanical_markers(&mut base, &mut renderables, experiment)?;
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

    /// Atomically replace visible formula geometry while retaining the
    /// volume sampler, WGPU surface and rendering thread.
    pub fn replace_formulas(&mut self, experiment: &Experiment) -> Result<(), String> {
        let mut next = Vec::new();
        for formula in &experiment.formulas {
            let mesh = self
                .base
                .render
                .meshes
                .register(formula_mesh(formula)?)
                .map_err(|err| format!("formula {}: {err}", formula.name))?;
            let material = self.base.render.materials.register(Material {
                base_color_factor: [formula.color[0], formula.color[1], formula.color[2], 1.0],
                emissive: formula
                    .color
                    .map(|v| v * experiment.visualization.exposure * 1.5),
                unlit: true,
                ..Default::default()
            });
            next.push(Renderable3d {
                mesh,
                material,
                frame: Motor3::translation(Vec3::ZERO),
                scale: Scale3::UNIT,
                billboard: false,
                casts_shadow: false,
                palette: None,
                morph_weights: Vec::new(),
            });
        }
        append_mechanical_markers(&mut self.base, &mut next, experiment)?;
        for old in self.scene.renderables.drain(..self.static_count) {
            self.base.render.meshes.unregister(old.mesh);
            self.base.render.materials.unregister(old.material);
        }
        self.scene.renderables.clear();
        self.static_count = next.len();
        self.scene.renderables = next;
        self.volume.invalidate();
        Ok(())
    }

    pub fn update(&mut self, snapshot: &SimulationSnapshot, experiment: &Experiment) {
        if let (Some(observation), Some(config)) = (&snapshot.mechanics, &experiment.mechanics) {
            for (i, x) in observation.positions.iter().enumerate() {
                if let Some(marker) = self
                    .scene
                    .renderables
                    .get_mut(experiment.formulas.len() + i)
                {
                    let x: [f32; 3] = std::array::from_fn(|axis| {
                        ((x[axis] + config.period[axis] / 2.0).rem_euclid(config.period[axis])
                            - config.period[axis] / 2.0) as f32
                    });
                    marker.frame = Motor3::translation(Vec3::new(x[0], x[1], x[2]));
                }
            }
        }

        if experiment.mechanics.is_none() {
            self.volume.update(
                &mut self.base,
                &mut self.scene,
                self.static_count,
                snapshot,
                experiment,
            );
        }
    }

    pub fn sampling_cpu_ms(&self) -> f64 {
        self.volume.sampling_cpu_ms()
    }

    pub fn visible_slices(&self) -> usize {
        self.volume.visible_slices()
    }

    pub fn peak_density(&self) -> f64 {
        self.volume.peak_density()
    }
}
