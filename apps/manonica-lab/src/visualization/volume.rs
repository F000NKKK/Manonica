//! GPU-facing side of the manon volume. The costly trilinear resampling
//! runs in a separate CPU worker, never inside the window/redraw callback.

use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use manonica_physics::FreeManaField;
use meridian_sdk::{
    BlendMode, GraphicsSession, ImageAsset, Material, MaterialHandle, MeshHandle, MeshSource,
    Motor3, Renderable3d, Scale3, Scene3d, TextureHandle, TextureStore, Vec3,
};

use crate::experiment::Experiment;
use crate::simulation::SimulationSnapshot;

// Soft, low-frequency appearance intentionally chosen for Manonica Lab.
const TEXTURE_SIZE: usize = 64;
const BORDER_ROWS: usize = 2;
// Rendering resolution along the depth axis is independent of physics cells.
// Each plane samples the physical field with trilinear interpolation.
const MAX_PLANES: usize = 64;
const ATLAS_PERIOD: Duration = Duration::from_millis(100);
// Meridian's perspective near clip plane is 0.1 metres.
const NEAR_CLIP_METRES: f64 = 0.1;

/// Volumetric slice orientation is driven by the camera's forward ray,
/// never by its distance from the domain origin. That distinction matters
/// when the camera is *inside* the volume.
fn slice_axis(forward: Vec3) -> usize {
    let components = [forward.x.abs(), forward.y.abs(), forward.z.abs()];
    (0..3)
        .max_by(|&a, &b| components[a].total_cmp(&components[b]))
        .unwrap_or(2)
}

/// Quantized full camera pose used to fade individual samples, not whole
/// axis-aligned planes. A diagonal view must use the true ray depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SliceView {
    camera_bins: [i32; 3],
    forward_bins: [i16; 3],
    sign: i8,
}

impl SliceView {
    fn new(eye: Vec3, forward: Vec3, axis: usize, dx: f64) -> Self {
        let eye_components = [eye.x, eye.y, eye.z];
        let forward_components = [forward.x, forward.y, forward.z];
        Self {
            camera_bins: eye_components.map(|value| ((value as f64 / dx) * 2.0).round() as i32),
            forward_bins: forward_components.map(|value| (value as f64 * 128.0).round() as i16),
            sign: if forward_components[axis] >= 0.0 {
                1
            } else {
                -1
            },
        }
    }

    fn camera(self, dx: f64) -> [f64; 3] {
        self.camera_bins.map(|value| value as f64 * dx * 0.5)
    }

    fn forward(self) -> [f64; 3] {
        self.forward_bins.map(|value| value as f64 / 128.0)
    }
}

/// Attenuate only the rendering of layers that cross the camera's near
/// plane. The physical density remains unchanged.
fn near_slice_visibility(depth: f64, cell_width: f64) -> f64 {
    let t = ((depth - NEAR_CLIP_METRES) / (cell_width * 1.5)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

struct AtlasRequest {
    field: Arc<FreeManaField>,
    axis: usize,
    planes: usize,
    threshold: f64,
    optical_density: f64,
    view: SliceView,
}

struct AtlasResult {
    image: ImageAsset,
    axis: usize,
    threshold: f64,
    optical_density: f64,
    view: SliceView,
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
                        request.optical_density,
                        request.view,
                    );
                    let result = AtlasResult {
                        image,
                        axis: request.axis,
                        threshold: request.threshold,
                        optical_density: request.optical_density,
                        view: request.view,
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
    last_requested: Option<(usize, u64, f64, f64, SliceView)>,
    last_submitted: Option<Instant>,
    count: usize,
    peak: f64,
    cpu_ms: f64,
}

impl ManaVolume {
    pub fn new(base: &mut GraphicsSession, exp: &Experiment) -> Result<Self, String> {
        let mut meshes: [Vec<MeshHandle>; 3] = std::array::from_fn(|_| Vec::new());
        for (axis, axis_meshes) in meshes.iter_mut().enumerate() {
            let count = MAX_PLANES;
            for layer in 0..count {
                axis_meshes.push(
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

    /// Discard cached atlas metadata after a live experiment reconfiguration.
    pub fn invalidate(&mut self) {
        self.last_requested = None;
        self.last_submitted = None;
        self.count = 0;
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
        // Meridian's fly camera uses local +X as its forward axis.
        let forward = scene.camera.frame.transform_vector(Vec3::X);
        let axis = slice_axis(forward);
        let view = SliceView::new(eye, forward, axis, snapshot.field.cell_width_metres());
        let threshold = experiment.visualization.relative_threshold;
        let optical_density = experiment.visualization.optical_density;

        if let Some(result) = self.worker.latest() {
            // A moving camera may make a completed job obsolete. Never
            // display an X slice after the camera has switched to Y slices.
            if result.axis == axis
                && result.threshold == threshold
                && result.optical_density == optical_density
                && result.view.sign == view.sign
            {
                self.apply_atlas(base, scene, static_count, result, experiment);
            }
        }

        let requested = (axis, snapshot.steps, threshold, optical_density, view);
        if self.last_requested == Some(requested) {
            return;
        }
        let urgent = self.last_requested.is_none_or(
            |(old_axis, _, old_threshold, old_density, old_view)| {
                axis != old_axis
                    || old_threshold != threshold
                    || old_density != optical_density
                    || old_view.sign != view.sign
            },
        );
        if !urgent
            && self
                .last_submitted
                .is_some_and(|time| time.elapsed() < ATLAS_PERIOD)
        {
            return;
        }

        let sent = self.worker.offer(AtlasRequest {
            field: Arc::clone(&snapshot.field),
            axis,
            planes: self.meshes[axis].len(),
            threshold,
            optical_density,
            view,
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
        // Keep the high-concentration core icy blue instead of driving all
        // three HDR channels far above 1.0 (which washed out the volume).
        let gain = (experiment.visualization.exposure / 3.0).sqrt().min(2.0);
        let material = base.render.materials.register(Material {
            albedo: Some(texture),
            base_color_factor: [0.70 * gain, 0.90 * gain, 1.05 * gain, 1.0],
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
        // Alpha blending requires back-to-front order relative to the
        // dominant viewing axis. The previous ascending order was wrong
        // whenever the camera looked toward positive X/Y/Z.
        for draw_index in 0..count {
            let layer = if result.view.sign > 0 {
                count - 1 - draw_index
            } else {
                draw_index
            };
            let mesh = self.meshes[axis][layer];
            let coordinate =
                (layer as f32 + 0.5) * lengths[axis] / count as f32 - lengths[axis] / 2.0;
            let mut position = [0.0; 3];
            position[axis] = coordinate;
            scene.renderables.push(Renderable3d {
                mesh,
                material,
                frame: Motor3::translation(Vec3::new(position[0], position[1], position[2])),
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
    let positions = [
        [-half_u, -half_v],
        [half_u, -half_v],
        [half_u, half_v],
        [-half_u, half_v],
    ]
    .map(|uv| {
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
            [uv_left, uv_top],
            [uv_right, uv_top],
            [uv_right, uv_bottom],
            [uv_left, uv_bottom],
        ],
        // Both windings: a volume must be visible when the camera crosses
        // the normal axis; back-face culling still draws each plane once.
        indices: vec![0, 1, 2, 0, 2, 3, 2, 1, 0, 3, 2, 0],
        skin: None,
        morph: None,
    }
}

/// One-dimensional interpolation coordinates for a regular Cartesian grid.
/// All weights, source indices and boundary fading are evaluated once per
/// atlas axis, rather than independently for every pixel of every slice.
#[derive(Debug, Clone, Copy)]
struct AxisSample {
    first: usize,
    second: usize,
    t: f64,
    fade: f64,
}

impl AxisSample {
    fn from_position(position: f64, dimension: usize, dx: f64) -> Self {
        let coordinate =
            (position / dx + dimension as f64 * 0.5 - 0.5).clamp(0.0, (dimension - 1) as f64);
        let first = coordinate.floor() as usize;
        let second = (first + 1).min(dimension - 1);
        let half_extent = dimension as f64 * dx * 0.5;
        let edge = ((half_extent - position.abs()) / (dx * 2.5)).clamp(0.0, 1.0);
        Self {
            first,
            second,
            t: coordinate - first as f64,
            fade: edge * edge * (3.0 - 2.0 * edge),
        }
    }

    fn texel(index: usize, total: usize, dimension: usize, dx: f64) -> Self {
        let position = ((index as f64 + 0.5) / total as f64 - 0.5) * dimension as f64 * dx;
        Self::from_position(position, dimension, dx)
    }
}

/// Exact trilinear interpolation of the same solver grid. No new particles
/// or physics cells are produced; this is just smooth visualization.
#[inline]
fn interpolated_density(field: &FreeManaField, coords: [AxisSample; 3]) -> f64 {
    let shape = field.shape();
    let [x, y, z] = coords;
    let row = shape.nx();
    let plane = row * shape.ny();
    let values = field.densities();
    let at = |ix: usize, iy: usize, iz: usize| values[iz * plane + iy * row + ix].value();
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let z0y0 = lerp(
        at(x.first, y.first, z.first),
        at(x.second, y.first, z.first),
        x.t,
    );
    let z0y1 = lerp(
        at(x.first, y.second, z.first),
        at(x.second, y.second, z.first),
        x.t,
    );
    let z1y0 = lerp(
        at(x.first, y.first, z.second),
        at(x.second, y.first, z.second),
        x.t,
    );
    let z1y1 = lerp(
        at(x.first, y.second, z.second),
        at(x.second, y.second, z.second),
        x.t,
    );
    lerp(lerp(z0y0, z0y1, y.t), lerp(z1y0, z1y1, y.t), z.t)
}

#[cfg(test)]
fn sample_density(field: &FreeManaField, position: [f64; 3]) -> f64 {
    let shape = field.shape();
    let dimensions = [shape.nx(), shape.ny(), shape.nz()];
    let dx = field.cell_width_metres();
    interpolated_density(
        field,
        std::array::from_fn(|axis| AxisSample::from_position(position[axis], dimensions[axis], dx)),
    )
}

/// Convert normalized density into ice-blue radiance and Beer–Lambert
/// opacity. `optical_density` has units of inverse metres.
fn rgba_from_density(normalized: f64, layer_thickness: f64, optical_density: f64) -> [u8; 4] {
    let core = normalized.clamp(0.0, 1.0).powf(0.72);
    // Cyan edges, pale high-density center, retaining a blue channel > red.
    let red = 0.12 + 0.78 * core * core;
    let green = 0.40 + 0.60 * core;
    let blue = 0.90 + 0.10 * core;
    let tau = optical_density * normalized.clamp(0.0, 1.0).powf(1.15) * layer_thickness;
    let alpha = 1.0 - (-tau).exp();
    [
        (red * 255.0).round() as u8,
        (green * 255.0).round() as u8,
        (blue * 255.0).round() as u8,
        (alpha * 255.0).round() as u8,
    ]
}

fn build_density_atlas(
    field: &FreeManaField,
    axis: usize,
    planes: usize,
    threshold: f64,
    peak: f64,
    optical_density: f64,
    view: SliceView,
) -> ImageAsset {
    let shape = field.shape();
    let dimensions = [shape.nx(), shape.ny(), shape.nz()];
    let width = TEXTURE_SIZE;
    let stride = TEXTURE_SIZE + BORDER_ROWS;
    let height = stride * planes;
    let mut pixels = vec![0_u8; width * height * 4];
    let dx = field.cell_width_metres();
    let [u, v] = other_axes(axis);
    let layer_thickness = dimensions[axis] as f64 * dx / planes as f64;
    let axis_lookup = (0..planes)
        .map(|layer| AxisSample::texel(layer, planes, dimensions[axis], dx))
        .collect::<Vec<_>>();
    let column_lookup = (0..width)
        .map(|x| AxisSample::texel(x, width, dimensions[u], dx))
        .collect::<Vec<_>>();
    let row_lookup = (0..TEXTURE_SIZE)
        .map(|y| AxisSample::texel(y, TEXTURE_SIZE, dimensions[v], dx))
        .collect::<Vec<_>>();

    let camera = view.camera(dx);
    let forward = view.forward();
    for (layer, &normal) in axis_lookup.iter().enumerate() {
        let layer_position =
            ((layer as f64 + 0.5) / planes as f64 - 0.5) * dimensions[axis] as f64 * dx;
        let mut coords = [normal; 3];
        coords[axis] = normal;
        for (yy, &row_sample) in row_lookup.iter().enumerate() {
            coords[v] = row_sample;
            let i = ((layer * stride + yy + 1) * width) * 4;
            for (xx, &column_sample) in column_lookup.iter().enumerate() {
                coords[u] = column_sample;
                let relative = if peak > 0.0 {
                    (interpolated_density(field, coords) / peak).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                // Use per-texel view-space depth: an oblique plane crosses
                // the near clip at different positions across its surface.
                let mut world = [0.0_f64; 3];
                world[axis] = layer_position;
                world[u] = ((xx as f64 + 0.5) / width as f64 - 0.5) * dimensions[u] as f64 * dx;
                world[v] =
                    ((yy as f64 + 0.5) / TEXTURE_SIZE as f64 - 0.5) * dimensions[v] as f64 * dx;
                let depth = (0..3).map(|i| (world[i] - camera[i]) * forward[i]).sum();
                let near_fade = near_slice_visibility(depth, dx);
                let visible = ((relative - threshold) / (1.0 - threshold)).max(0.0)
                    * normal.fade
                    * row_sample.fade
                    * column_sample.fade
                    * near_fade;
                let rgba = rgba_from_density(visible, layer_thickness, optical_density);
                pixels[i + xx * 4..i + xx * 4 + 4].copy_from_slice(&rgba);
            }
        }
        // Border texels repeat their nearest data row, so bilinear
        // filtering never pulls color from the adjacent slice.
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
            GridShape::new(3, 3, 3).unwrap(),
            0.5,
            Quantity::<ManonDensity>::nonnegative(15.0).unwrap(),
        )
        .unwrap();
        let v = sample_density(&f, [0.1, -0.3, 0.15]);
        assert!((v - 15.0).abs() < 1e-10);
    }

    #[test]
    fn render_depth_resolution_exceeds_coarse_physics_grid() {
        // A 28-cell grid must render 64 translucent layers, not 28.
        // The physical field and its conserved inventory are unaffected.
        let shape = GridShape::new(28, 28, 28).unwrap();
        let field = FreeManaField::uniform(
            shape,
            0.5,
            Quantity::<ManonDensity>::nonnegative(2.0).unwrap(),
        )
        .unwrap();
        let before = field.total_manons().unwrap().value();
        let atlas = build_density_atlas(
            &field,
            2,
            MAX_PLANES,
            0.04,
            2.0,
            0.18,
            SliceView::new(Vec3::new(0.0, 0.0, -10.0), Vec3::new(0.0, 0.0, 1.0), 2, 0.5),
        );
        assert_eq!(atlas.height, ((TEXTURE_SIZE + BORDER_ROWS) * 64) as u32);
        assert_eq!(field.total_manons().unwrap().value(), before);
    }

    #[test]
    fn one_atlas_renders_many_slices_and_has_valid_size() {
        let f = FreeManaField::uniform(
            GridShape::new(6, 4, 8).unwrap(),
            0.3,
            Quantity::<ManonDensity>::nonnegative(5.0).unwrap(),
        )
        .unwrap();
        let atlas = build_density_atlas(
            &f,
            2,
            8,
            0.04,
            5.0,
            0.18,
            SliceView::new(Vec3::new(0.0, 0.0, -2.4), Vec3::new(0.0, 0.0, 1.0), 2, 0.3),
        );
        assert_eq!(atlas.height, (8 * (TEXTURE_SIZE + BORDER_ROWS)) as u32);
        assert_eq!(
            atlas.pixels.len(),
            atlas.width as usize * atlas.height as usize * 4
        );
        assert!(atlas.pixels.as_chunks::<4>().0.iter().any(|px| px[3] > 0));
        assert!(
            atlas
                .pixels
                .as_chunks::<4>()
                .0
                .iter()
                .any(|px| px[3] > 0 && px[2] >= 230 && px[2] > px[0])
        );
    }

    #[test]
    fn visual_boundary_fades_to_zero_without_affecting_density() {
        let dims = [12, 12, 12];
        let center = AxisSample::from_position(0.0, dims[0], 0.5).fade;
        let near_edge = AxisSample::from_position(2.6, dims[0], 0.5).fade;
        let edge = AxisSample::from_position(3.0, dims[0], 0.5).fade;
        assert!((center - 1.0).abs() < 1e-12);
        assert!(near_edge > 0.0 && near_edge < center);
        assert_eq!(edge, 0.0);
    }

    #[test]
    fn plane_mesh_is_valid_for_all_three_axes() {
        for axis in 0..3 {
            let mesh = plane_mesh([12, 6, 8], 0.25, axis, 3, 8);
            assert_eq!(mesh.positions.len(), 4);
            assert!(
                mesh.indices
                    .iter()
                    .all(|&i| (i as usize) < mesh.positions.len())
            );
            assert!(mesh.uvs.iter().flatten().all(|n| (0.0..=1.0).contains(n)));
        }
    }
}

#[cfg(test)]
mod interpolation_regressions {
    use super::*;
    use manonica_core::{Quantity, units::ManonDensity};
    use manonica_physics::GridShape;

    #[test]
    fn cached_interpolation_is_equivalent_to_corner_sum() {
        let mut grid = FreeManaField::uniform(
            GridShape::new(4, 3, 5).unwrap(),
            0.5,
            Quantity::<ManonDensity>::nonnegative(0.0).unwrap(),
        )
        .unwrap();
        for z in 0..5 {
            for y in 0..3 {
                for x in 0..4 {
                    grid.set(
                        x,
                        y,
                        z,
                        Quantity::nonnegative((x + 10 * y + 100 * z) as f64).unwrap(),
                    )
                    .unwrap();
                }
            }
        }
        let shape = grid.shape();
        for point in [[-0.9, 0.3, 1.1], [0.1, -0.1, 0.4], [0.8, 0.8, -0.3]] {
            let coordinates = std::array::from_fn::<_, 3, _>(|axis| {
                let size = [shape.nx(), shape.ny(), shape.nz()][axis];
                (point[axis] / 0.5 + size as f64 * 0.5 - 0.5).clamp(0.0, (size - 1) as f64)
            });
            let mut reference = 0.0;
            for corners in 0..8 {
                let mut weight = 1.0;
                let mut index = [0_usize; 3];
                for axis in 0..3 {
                    let low = coordinates[axis].floor() as usize;
                    let high = (low + 1).min([shape.nx(), shape.ny(), shape.nz()][axis] - 1);
                    let t = coordinates[axis] - low as f64;
                    if corners & (1 << axis) == 0 {
                        index[axis] = low;
                        weight *= 1.0 - t;
                    } else {
                        index[axis] = high;
                        weight *= t;
                    }
                }
                reference += grid.at(index[0], index[1], index[2]).unwrap().value() * weight;
            }
            let optimized = sample_density(&grid, point);
            assert!(
                (optimized - reference).abs() < 1e-10,
                "optimized {optimized}, reference {reference}"
            );
        }
    }

    #[test]
    fn optical_density_monotonically_controls_alpha() {
        let thin = rgba_from_density(0.5, 0.5, 0.1);
        let dense = rgba_from_density(0.5, 0.5, 0.5);
        assert!(dense[3] > thin[3]);
        let empty = rgba_from_density(0.0, 0.5, 0.5);
        assert_eq!(empty[3], 0);
    }

    #[test]
    fn empty_field_is_an_invisible_atlas() {
        let field = FreeManaField::uniform(
            GridShape::new(4, 4, 4).unwrap(),
            0.5,
            Quantity::<ManonDensity>::nonnegative(0.0).unwrap(),
        )
        .unwrap();
        let atlas = build_density_atlas(
            &field,
            2,
            4,
            0.06,
            0.0,
            0.18,
            SliceView::new(Vec3::new(0.0, 0.0, -2.4), Vec3::new(0.0, 0.0, 1.0), 2, 0.3),
        );
        assert!(atlas.pixels.as_chunks::<4>().0.iter().all(|px| px[3] == 0));
    }
}

#[cfg(test)]
mod interior_view_tests {
    use super::*;
    use manonica_core::{Quantity, units::ManonDensity};
    use manonica_physics::GridShape;

    #[test]
    fn slice_axis_uses_view_direction_not_camera_position() {
        assert_eq!(slice_axis(Vec3::new(0.0, 0.0, -1.0)), 2);
        assert_eq!(slice_axis(Vec3::new(0.1, 0.95, 0.2)), 1);
        assert_eq!(slice_axis(Vec3::new(-0.9, 0.0, 0.1)), 0);
    }

    #[test]
    fn diagonal_camera_depth_varies_across_one_plane() {
        let view = SliceView::new(Vec3::ZERO, Vec3::new(0.70710677, 0.0, 0.70710677), 2, 0.5);
        let camera = view.camera(0.5);
        let forward = view.forward();
        let left = [-1.0_f64, 0.0, 0.5];
        let right = [1.0_f64, 0.0, 0.5];
        let depth = |p: [f64; 3]| (0..3).map(|i| (p[i] - camera[i]) * forward[i]).sum::<f64>();
        assert!(depth(left) < NEAR_CLIP_METRES);
        assert!(depth(right) > NEAR_CLIP_METRES);
        assert_eq!(near_slice_visibility(depth(left), 0.5), 0.0);
        assert!(near_slice_visibility(depth(right), 0.5) > 0.0);
    }

    #[test]
    fn camera_near_clip_fades_instead_of_popping() {
        assert_eq!(near_slice_visibility(-1.0, 0.5), 0.0);
        assert_eq!(near_slice_visibility(NEAR_CLIP_METRES, 0.5), 0.0);
        let middle = near_slice_visibility(NEAR_CLIP_METRES + 0.375, 0.5);
        assert!((middle - 0.5).abs() < 1e-12);
        assert_eq!(near_slice_visibility(2.0, 0.5), 1.0);
    }

    #[test]
    fn uniform_field_seen_from_inside_does_not_render_behind_eye() {
        let f = FreeManaField::uniform(
            GridShape::new(12, 12, 12).unwrap(),
            0.5,
            Quantity::<ManonDensity>::nonnegative(10.0).unwrap(),
        )
        .unwrap();
        let atlas = build_density_atlas(
            &f,
            2,
            12,
            0.01,
            10.0,
            0.3,
            SliceView::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), 2, 0.5),
        );
        let alpha = |layer: usize| {
            let yy = TEXTURE_SIZE / 2;
            let xx = TEXTURE_SIZE / 2;
            let y = layer * (TEXTURE_SIZE + BORDER_ROWS) + yy + 1;
            atlas.pixels[(y * TEXTURE_SIZE + xx) * 4 + 3]
        };
        assert_eq!(
            alpha(4),
            0,
            "behind-camera slice must not overlay the scene"
        );
        assert!(alpha(6) > 0, "forward slice is visible");
        assert!(
            alpha(8) >= alpha(6),
            "more distant layers should not be near-clipped"
        );
        // Physical grid is immutable to the rendering operation.
        assert_eq!(
            f.total_manons().unwrap().value(),
            10.0 * (12.0_f64 * 0.5).powi(3)
        );
    }
}
