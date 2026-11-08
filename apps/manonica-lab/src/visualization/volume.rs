//! GPU-facing side of the manon volume. The costly trilinear resampling
//! runs in a separate CPU worker, never inside the window/redraw callback.

use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use meridian_sdk::{
    BlendMode, GraphicsSession, ImageAsset, Material, MaterialHandle, MeshHandle,
    MeshSource, Motor3, Renderable3d, Scale3, Scene3d, TextureHandle, TextureStore, Vec3,
};
use manonica_physics::FreeManaField;

use crate::experiment::Experiment;
use crate::simulation::SimulationSnapshot;

const TEXTURE_SIZE: usize = 64;
const BORDER_ROWS: usize = 2;
const MAX_PLANES: usize = 20;
const ATLAS_PERIOD: Duration = Duration::from_millis(100);

struct AtlasRequest {
    field: Arc<FreeManaField>,
    axis: usize,
    planes: usize,
    threshold: f64,
    step: u64,
}

struct AtlasResult {
    image: ImageAsset,
    axis: usize,
    threshold: f64,
    step: u64,
    peak: f64,
    cpu_ms: f64,
}

/// Bounded one-slot queues: if rendering is behind, work is dropped instead
/// of accumulating stale frames or blocking the main thread.
struct AtlasWorker {
    sender: Option<SyncSender<AtlasRequest>>,
    receiver: Receiver<AtlasResult>,
    thread: Option<JoinHandle<()>>,
}

impl AtlasWorker {
    fn spawn() -> Self {
        let (request_tx, request_rx) = mpsc::sync_channel::<AtlasRequest>(1);
        let (result_tx, result_rx) = mpsc::sync_channel::<AtlasResult>(1);
        let thread = thread::Builder::new()
            .name("manonica-volume".into())
            .spawn(move || {
                while let Ok(request) = request_rx.recv() {
                    let now = Instant::now();
                    let peak = request
                        .field
                        .densities()
                        .iter()
                        .map(|n| n.value())
                        .fold(0.0_f64, f64::max);
                    let image = build_density_atlas(
                        &request.field,
                        request.axis,
                        request.planes,
                        request.threshold,
                        peak,
                    );
                    let result = AtlasResult {
                        image,
                        axis: request.axis,
                        threshold: request.threshold,
                        step: request.step,
                        peak,
                        cpu_ms: now.elapsed().as_secs_f64() * 1000.0,
                    };
                    if result_tx.try_send(result).is_err() {
                        // Renderer has not yet consumed the previous texture.
                        // The most recent state will be requested again.
                    }
                }
            })
            .expect("failed to spawn volume sampling thread");
        Self {
            sender: Some(request_tx),
            receiver: result_rx,
            thread: Some(thread),
        }
    }

    fn offer(&self, request: AtlasRequest) -> bool {
        self.sender
            .as_ref()
            .is_some_and(|sender| sender.try_send(request).is_ok())
    }

    fn latest(&self) -> Option<AtlasResult> {
        let mut newest = None;
        while let Ok(result) = self.receiver.try_recv() {
            newest = Some(result);
        }
        newest
    }
}

impl Drop for AtlasWorker {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

pub struct ManaVolume {
    meshes: [Vec<MeshHandle>; 3],
    material: Option<MaterialHandle>,
    texture: Option<TextureHandle>,
    worker: AtlasWorker,
    last_requested: Option<(usize, u64, f64)>,
    last_submitted: Option<Instant>,
    count: usize,
    peak: f64,
    cpu_ms: f64,
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
                        .register(plane_mesh(
                            exp.grid.dimensions,
                            exp.grid.cell_size,
                            axis,
                            layer,
                            count,
                        ))
                        .map_err(|err| format!("volume plane {axis}/{layer}: {err}"))?,
                );
            }
        }
        Ok(Self {
            meshes,
            material: None,
            texture: None,
            worker: AtlasWorker::spawn(),
            last_requested: None,
            last_submitted: None,
            count: 0,
            peak: 0.0,
            cpu_ms: 0.0,
        })
    }

    pub fn visible_slices(&self) -> usize {
        self.count
    }

    pub fn peak_density(&self) -> f64 {
        self.peak
    }

    pub fn sampling_cpu_ms(&self) -> f64 {
        self.cpu_ms
    }

    /// Nonblocking: uploads a ready image at most once and queues a new
    /// sampling job only when the field, camera direction or threshold changes.
    pub fn update(
        &mut self,
        base: &mut GraphicsSession,
        scene: &mut Scene3d,
        static_count: usize,
        snapshot: &SimulationSnapshot,
        experiment: &Experiment,
    ) {
        let eye = scene.camera.frame.transform_point(Vec3::ZERO);
        let components = [eye.x.abs(), eye.y.abs(), eye.z.abs()];
        let axis = (0..3)
            .max_by(|&a, &b| components[a].total_cmp(&components[b]))
            .unwrap_or(2);
        let threshold = experiment.visualization.relative_threshold;

        if let Some(result) = self.worker.latest() {
            // A moving camera may make a completed job obsolete. Never
            // display an X slice after the camera has switched to Y slices.
            if result.axis == axis && result.threshold == threshold {
                self.apply_atlas(base, scene, static_count, result, experiment);
            }
        }

        let requested = (axis, snapshot.steps, threshold);
        if self.last_requested == Some(requested) {
            return;
        }
        let urgent = self
            .last_requested
            .is_none_or(|(old_axis, _, old_threshold)| {
                axis != old_axis || old_threshold != threshold
            });
        if !urgent && self.last_submitted.is_some_and(|time| time.elapsed() < ATLAS_PERIOD) {
            return;
        }

        let sent = self.worker.offer(AtlasRequest {
            field: Arc::clone(&snapshot.field),
            axis,
            planes: self.meshes[axis].len(),
            threshold,
            step: snapshot.steps,
        });
        if sent {
            self.last_requested = Some(requested);
            self.last_submitted = Some(Instant::now());
        }
    }

    fn apply_atlas(
        &mut self,
        base: &mut GraphicsSession,
        scene: &mut Scene3d,
        static_count: usize,
        result: AtlasResult,
        experiment: &Experiment,
    ) {
        let texture = base.render.recorder.upload(&result.image);
        let material = base.render.materials.register(Material {
            albedo: Some(texture),
            base_color_factor: [
                1.5 * experiment.visualization.exposure.sqrt(),
                1.4 * experiment.visualization.exposure.sqrt(),
                1.4 * experiment.visualization.exposure.sqrt(),
                1.0,
            ],
            blend: BlendMode::AlphaBlend,
            unlit: true,
            ..Default::default()
        });
        if let Some(old) = self.material.replace(material) {
            base.render.materials.unregister(old);
        }
        if let Some(old) = self.texture.replace(texture) {
            base.render.recorder.release(old);
        }

        scene.renderables.truncate(static_count);
        let axis = result.axis;
        let count = self.meshes[axis].len();
        let lengths = experiment
            .grid
            .dimensions
            .map(|n| n as f32 * experiment.grid.cell_size as f32);
        for (layer, &mesh) in self.meshes[axis].iter().enumerate() {
            let coordinate = (layer as f32 + 0.5) * lengths[axis] / count as f32
                - lengths[axis] / 2.0;
            let mut position = [0.0; 3];
            position[axis] = coordinate;
            scene.renderables.push(Renderable3d {
                mesh,
                material,
                frame: Motor3::translation(Vec3::new(
                    position[0],
                    position[1],
                    position[2],
                )),
                scale: Scale3::UNIT,
                billboard: false,
                casts_shadow: false,
                palette: None,
                morph_weights: Vec::new(),
            });
        }
        self.count = count;
        self.peak = result.peak;
        self.cpu_ms = result.cpu_ms;
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
        result += field.densities()[index].value() * weight;
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
