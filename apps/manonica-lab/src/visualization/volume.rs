//! View-dependent volume slicing backed by *one RGBA atlas texture*.
//!
//! No sprites/icospheres per manon or per cell. A regular 3D density grid is
//! interpolated onto a modest number of translucent planes. The camera-facing
//! axis is chosen automatically; planes sort back-to-front through Meridian's
//! transparent scene pipeline. Physical density is never mutated here.

use meridian_sdk::{
    BlendMode, GraphicsSession, ImageAsset, Material, MaterialHandle, MeshHandle,
    MeshSource, Motor3, Renderable3d, Scale3, Scene3d, TextureHandle, TextureStore, Vec3,
};

use crate::experiment::Experiment;
use crate::simulation::Simulation;
use manonica_physics::FreeManaField;

const TEXTURE_SIZE: usize = 96;
const BORDER_ROWS: usize = 2;
const MAX_PLANES: usize = 24;
const REFRESH_INTERVAL: u64 = 6;

pub struct ManaVolume {
    /// For each of X/Y/Z normals, pre-register all plane geometries once.
    meshes: [Vec<MeshHandle>; 3],
    material: Option<MaterialHandle>,
    texture: Option<TextureHandle>,
    previous_axis: Option<usize>,
    previous_step: Option<u64>,
    previous_threshold: f64,
    frames: u64,
    count: usize,
    peak: f64,
}

impl ManaVolume {
    pub fn new(base: &mut GraphicsSession, exp: &Experiment) -> Result<Self, String> {
        let mut meshes: [Vec<MeshHandle>; 3] = std::array::from_fn(|_| Vec::new());
        for axis in 0..3 {
            let count = exp.grid.dimensions[axis].min(MAX_PLANES);
            for layer in 0..count {
                meshes[axis].push(
                    base.render
                        .meshes
                        .register(plane_mesh(exp.grid.dimensions, exp.grid.cell_size, axis, layer, count))
                        .map_err(|err| format!("volume plane {axis}/{layer}: {err}"))?,
                );
            }
        }
        Ok(Self {
            meshes,
            material: None,
            texture: None,
            previous_axis: None,
            previous_step: None,
            previous_threshold: f64::NAN,
            frames: 0,
            count: 0,
            peak: 0.0,
        })
    }

    pub fn visible_slices(&self) -> usize {
        self.count
    }

    pub fn peak_density(&self) -> f64 {
        self.peak
    }

    pub fn update(
        &mut self,
        base: &mut GraphicsSession,
        scene: &mut Scene3d,
        static_count: usize,
        simulation: &Simulation,
    ) {
        self.frames = self.frames.wrapping_add(1);
        let eye = scene.camera.frame.transform_point(Vec3::ZERO);
        let components = [eye.x.abs(), eye.y.abs(), eye.z.abs()];
        let axis = (0..3)
            .max_by(|&a, &b| components[a].total_cmp(&components[b]))
            .unwrap_or(2);
        let exp = &simulation.experiment;
        let threshold = exp.visualization.relative_threshold;
        let has_new_data = self.previous_step != Some(simulation.steps());
        let has_new_axis = self.previous_axis != Some(axis);
        let has_new_threshold = self.previous_threshold != threshold;

        if !has_new_axis && !has_new_threshold && !has_new_data {
            return;
        }
        if !has_new_axis && !has_new_threshold && self.frames % REFRESH_INTERVAL != 0 {
            return;
        }

        let field = simulation.field();
        self.peak = field.values().iter().map(|n| n.value()).fold(0.0_f64, f64::max);
        scene.renderables.truncate(static_count);
        self.count = 0;
        self.previous_step = Some(simulation.steps());
        self.previous_axis = Some(axis);
        self.previous_threshold = threshold;
        if self.peak <= 0.0 {
            return;
        }

        let count = self.meshes[axis].len();
        let image = build_density_atlas(field, axis, count, threshold, self.peak);
        let texture = base.render.recorder.upload(&image);
        let material = base.render.materials.register(Material {
            albedo: Some(texture),
            base_color_factor: [
                1.5 * exp.visualization.exposure.sqrt(),
                1.4 * exp.visualization.exposure.sqrt(),
                1.4 * exp.visualization.exposure.sqrt(),
                1.0,
            ],
            blend: BlendMode::AlphaBlend,
            unlit: true,
            ..Default::default()
        });

        if let Some(old_material) = self.material.replace(material) {
            base.render.materials.unregister(old_material);
        }
        if let Some(old_texture) = self.texture.replace(texture) {
            base.render.recorder.release(old_texture);
        }

        let lengths = exp.grid.dimensions.map(|n| n as f32 * exp.grid.cell_size as f32);
        for (layer, &mesh) in self.meshes[axis].iter().enumerate() {
            let coordinate = (layer as f32 + 0.5) * lengths[axis] / count as f32
                - lengths[axis] / 2.0;
            let mut position = [0.0; 3];
            position[axis] = coordinate;
            scene.renderables.push(Renderable3d {
                mesh,
                material,
                frame: Motor3::translation(Vec3::new(
                    position[0], position[1], position[2],
                )),
                scale: Scale3::UNIT,
                billboard: false,
                casts_shadow: false,
                palette: None,
                morph_weights: Vec::new(),
            });
        }
        self.count = count;
    }
}

fn other_axes(axis: usize) -> [usize; 2] {
    match axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    }
}

fn plane_mesh(
    dimensions: [usize; 3],
    cell_size: f64,
    axis: usize,
    layer: usize,
    count: usize,
) -> MeshSource {
    let [u, v] = other_axes(axis);
    let half_u = dimensions[u] as f32 * cell_size as f32 * 0.5;
    let half_v = dimensions[v] as f32 * cell_size as f32 * 0.5;
    let positions = [[-half_u, -half_v], [half_u, -half_v],
                     [half_u, half_v], [-half_u, half_v]].map(|uv| {
        let mut xyz = [0.0; 3];
        xyz[u] = uv[0];
        xyz[v] = uv[1];
        xyz
    });

    let height = (TEXTURE_SIZE + BORDER_ROWS) * count;
    let start = layer * (TEXTURE_SIZE + BORDER_ROWS);
    // Center of texels plus a replicated border row: no bleeding from the
    // adjacent slice when the GPU samples the atlas linearly.
    let uv_top = (start as f32 + 1.5) / height as f32;
    let uv_bottom = (start as f32 + TEXTURE_SIZE as f32 + 0.5) / height as f32;
    let uv_left = 0.5 / TEXTURE_SIZE as f32;
    let uv_right = 1.0 - uv_left;
    MeshSource {
        positions: positions.to_vec(),
        normals: vec![[0.0, 0.0, 1.0]; 4],
        uvs: vec![
            [uv_left, uv_top], [uv_right, uv_top],
            [uv_right, uv_bottom], [uv_left, uv_bottom],
        ],
        // Both windings: a volume must be visible when the camera crosses
        // the normal axis; back-face culling still draws each plane once.
        indices: vec![0, 1, 2, 0, 2, 3, 2, 1, 0, 3, 2, 0],
        skin: None,
        morph: None,
    }
}

/// Trilinear sampling at an arbitrary point, for *rendering only*.
fn sample_density(field: &FreeManaField, position: [f64; 3]) -> f64 {
    let shape = field.shape();
    let dims = [shape.nx(), shape.ny(), shape.nz()];
    let dx = field.cell_width_metres();
    let coordinates = std::array::from_fn::<_, 3, _>(|axis| {
        (position[axis] / dx + dims[axis] as f64 * 0.5 - 0.5)
            .clamp(0.0, (dims[axis] - 1) as f64)
    });
    let lower = coordinates.map(|x| x.floor() as usize);
    let upper = std::array::from_fn::<_, 3, _>(|axis| (lower[axis] + 1).min(dims[axis] - 1));
    let weights = std::array::from_fn::<_, 3, _>(|axis| coordinates[axis] - lower[axis] as f64);
    let mut result = 0.0;
    for corners in 0..8 {
        let mut id = [0; 3];
        let mut weight = 1.0;
        for axis in 0..3 {
            if corners & (1 << axis) != 0 {
                id[axis] = upper[axis];
                weight *= weights[axis];
            } else {
                id[axis] = lower[axis];
                weight *= 1.0 - weights[axis];
            }
        }
        let index = shape.index(id[0], id[1], id[2]).expect("sample clamped to grid");
        result += field.values()[index].value() * weight;
    }
    result
}

fn build_density_atlas(
    field: &FreeManaField,
    axis: usize,
    planes: usize,
    threshold: f64,
    peak: f64,
) -> ImageAsset {
    let shape = field.shape();
    let dimensions = [shape.nx(), shape.ny(), shape.nz()];
    let width = TEXTURE_SIZE;
    let stride = TEXTURE_SIZE + BORDER_ROWS;
    let height = stride * planes;
    let mut pixels = vec![0_u8; width * height * 4];
    let dx = field.cell_width_metres();
    let [u, v] = other_axes(axis);
    let slice_thickness = dimensions[axis] as f64 / planes as f64;

    for layer in 0..planes {
        let center_index = (layer as f64 + 0.5) * slice_thickness;
        for yy in 0..TEXTURE_SIZE {
            for xx in 0..TEXTURE_SIZE {
                let mut p = [0.0_f64; 3];
                p[axis] = (center_index - dimensions[axis] as f64 * 0.5) * dx;
                p[u] = ((xx as f64 + 0.5) / width as f64 - 0.5) * dimensions[u] as f64 * dx;
                p[v] = ((yy as f64 + 0.5) / TEXTURE_SIZE as f64 - 0.5)
                    * dimensions[v] as f64 * dx;

                let relative = (sample_density(field, p) / peak).clamp(0.0, 1.0);
                let visible = ((relative - threshold) / (1.0 - threshold)).max(0.0);
                let brightness = visible.sqrt();
                // Beer-Lambert optical depth: the thickness is compensated
                // when the number of view slices changes.
                let alpha = 1.0 - (-visible.powf(0.9) * 0.23 * slice_thickness).exp();
                let i = ((layer * stride + yy + 1) * width + xx) * 4;
                pixels[i] = (0.30 + 0.70 * brightness).mul_add(255.0, 0.0) as u8;
                pixels[i + 1] = (0.52 + 0.48 * brightness).mul_add(255.0, 0.0) as u8;
                pixels[i + 2] = 255;
                pixels[i + 3] = (alpha * 255.0).round() as u8;
            }
        }
        // Avoid filtering an unrelated atlas tile at the top/bottom edge.
        let first = (layer * stride + 1) * width * 4;
        let last = (layer * stride + TEXTURE_SIZE) * width * 4;
        pixels.copy_within(first..first + width * 4, first - width * 4);
        pixels.copy_within(last..last + width * 4, last + width * 4);
    }

    ImageAsset {
        width: width as u32,
        height: height as u32,
        pixels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use manonica_core::{Quantity, units::ManonDensity};
    use manonica_physics::GridShape;

    #[test]
    fn trilinear_sampling_preserves_uniform_density() {
        let f = FreeManaField::uniform(
            GridShape::new(3, 3, 3).unwrap(), 0.5,
            Quantity::<ManonDensity>::nonnegative(15.0).unwrap(),
        ).unwrap();
        let v = sample_density(&f, [0.1, -0.3, 0.15]);
        assert!((v - 15.0).abs() < 1e-10);
    }

    #[test]
    fn one_atlas_renders_many_slices_and_has_valid_size() {
        let f = FreeManaField::uniform(
            GridShape::new(6, 4, 8).unwrap(), 0.3,
            Quantity::<ManonDensity>::nonnegative(5.0).unwrap(),
        ).unwrap();
        let atlas = build_density_atlas(&f, 2, 8, 0.04, 5.0);
        assert_eq!(atlas.height, (8 * (TEXTURE_SIZE + BORDER_ROWS)) as u32);
        assert_eq!(atlas.pixels.len(), atlas.width as usize * atlas.height as usize * 4);
        assert!(atlas.pixels.chunks_exact(4).any(|px| px[3] > 0));
        assert!(atlas.pixels.chunks_exact(4).any(|px| px[2] == 255));
    }

    #[test]
    fn plane_mesh_is_valid_for_all_three_axes() {
        for axis in 0..3 {
            let mesh = plane_mesh([12, 6, 8], 0.25, axis, 3, 8);
            assert_eq!(mesh.positions.len(), 4);
            assert!(mesh.indices.iter().all(|&i| (i as usize) < mesh.positions.len()));
            assert!(mesh.uvs.iter().flatten().all(|n| (0.0..=1.0).contains(n)));
        }
    }
}
