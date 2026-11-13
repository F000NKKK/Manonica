//! F3 diagnostics: raw samples, background rasterization and persistent textures.
mod canvas;
mod plot;
mod worker;
use meridian_sdk::{
    Camera2d, GraphicsSession, Motor3, Overlay2d, Scene2d, Sprite, TextureHandle, TextureStore,
    Vec3,
};
use plot::Plot;
use std::time::{Duration, Instant};
use worker::HudWorker;
const WIDTH: usize = 520;
const HEIGHT: usize = 500;
const HISTORY: usize = 180;
const UPDATE_PERIOD: Duration = Duration::from_millis(200);
#[derive(Debug, Clone, Copy, Default)]
pub struct MechanicsMetrics {
    pub energy: f64,
    pub energy_error: f64,
    pub momentum: [f64; 3],
    pub momentum_error: f64,
    pub bodies: usize,
    pub modes: usize,
    pub positions: [[f64; 3]; 2],
}
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameMetrics {
    pub timing: FrameTimingMetrics,
    pub simulation: SimulationMetrics,
    pub mana: ManaMetrics,
    pub materials: MaterialMetrics,
    pub field: FieldMetrics,
    pub volume: VolumeMetrics,
    pub mechanics: Option<MechanicsMetrics>,
}
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameTimingMetrics {
    pub frame_ms: f64,
    pub sim_cpu_ms: f64,
    pub draw_wall_ms: f64,
    pub field_cpu_ms: f64,
}
#[derive(Debug, Clone, Copy, Default)]
pub struct SimulationMetrics {
    pub time_seconds: f64,
    pub steps: u64,
    pub compute: crate::simulation::ComputeTelemetry,
    pub paused: bool,
    pub equilibrium: crate::simulation::EquilibriumState,
}
#[derive(Debug, Clone, Copy, Default)]
pub struct ManaMetrics {
    pub total: f64,
    pub initial_total: f64,
}
#[derive(Debug, Clone, Copy, Default)]
pub struct MaterialMetrics {
    pub bound: f64,
    pub formula_bound: [f64; 3],
    pub formula_count: usize,
    pub energy_joules: f64,
    pub temperature_kelvin: f64,
}
#[derive(Debug, Clone, Copy, Default)]
pub struct FieldMetrics {
    pub peak: f64,
    pub energy_joules: f64,
    pub heat_joules: f64,
    pub work_joules: f64,
    pub transported_manons: f64,
    pub exported_heat_joules: f64,
    pub shutdown: bool,
}
#[derive(Debug, Clone, Copy, Default)]
pub struct VolumeMetrics {
    pub peak_density: f64,
    pub shown_slices: usize,
    pub grid_cells: usize,
}

pub struct DebugHud {
    image: Option<TextureHandle>,
    last_request: Option<Instant>,
    plot: Plot,
    worker: HudWorker,
}
impl Default for DebugHud {
    fn default() -> Self {
        Self {
            image: None,
            last_request: None,
            plot: Plot::default(),
            worker: HudWorker::new(),
        }
    }
}
impl DebugHud {
    pub fn record(&mut self, metrics: FrameMetrics) {
        self.plot.record(metrics);
    }
    pub fn fps(&self) -> f64 {
        self.plot.fps()
    }
    pub fn overlay(
        &mut self,
        base: &mut GraphicsSession,
        viewport: [u32; 2],
        editor_open: bool,
    ) -> Option<Overlay2d> {
        self.plot.latest?;
        if let Some(image) = self.worker.latest()
            && !self
                .image
                .is_some_and(|handle| base.render.recorder.update(handle, &image))
        {
            let next = base.render.recorder.upload(&image);
            if let Some(previous) = self.image.replace(next) {
                base.render.recorder.release(previous);
            }
        }
        let now = Instant::now();
        if self
            .last_request
            .is_none_or(|last| now.duration_since(last) >= UPDATE_PERIOD)
            && self.worker.offer(&self.plot)
        {
            self.last_request = Some(now);
        }
        let handle = self.image?;
        let scene = Scene2d {
            camera: Camera2d::from_surface_size(viewport[0] as f32, viewport[1] as f32),
            sprites: vec![Sprite {
                texture: handle,
                source_rect: None,
                size: [WIDTH as f32, HEIGHT as f32],
                tint: [1.0; 4],
                frame: Motor3::translation(Vec3::new(
                    12.0 + WIDTH as f32 * 0.5,
                    if editor_open {
                        (viewport[1] as f32 - HEIGHT as f32 * 0.5 - 12.0).max(HEIGHT as f32 * 0.5)
                    } else {
                        12.0 + HEIGHT as f32 * 0.5
                    },
                    0.0,
                )),
                layer: 0,
            }],
        };
        Some(Overlay2d::new(scene))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_is_bounded_and_fps_is_finite() {
        let mut hud = DebugHud::default();
        for _ in 0..400 {
            hud.record(FrameMetrics {
                mechanics: None,
                timing: crate::diagnostics::FrameTimingMetrics {
                    frame_ms: 16.0,
                    sim_cpu_ms: 2.0,
                    draw_wall_ms: 3.0,
                    field_cpu_ms: 1.0,
                },
                simulation: crate::diagnostics::SimulationMetrics {
                    time_seconds: 0.0,
                    steps: 0,
                    compute: crate::simulation::ComputeTelemetry::default(),
                    paused: false,
                    equilibrium: crate::simulation::EquilibriumState::default(),
                },
                mana: crate::diagnostics::ManaMetrics {
                    total: 10.0,
                    initial_total: 10.0,
                },
                materials: crate::diagnostics::MaterialMetrics {
                    bound: 2.0,
                    formula_bound: [2.0, 0.0, 0.0],
                    formula_count: 1,
                    energy_joules: 4.0,
                    temperature_kelvin: 300.0,
                },
                field: crate::diagnostics::FieldMetrics {
                    peak: 0.0,
                    energy_joules: 0.0,
                    heat_joules: 0.0,
                    work_joules: 0.0,
                    transported_manons: 0.0,
                    exported_heat_joules: 0.0,
                    shutdown: false,
                },
                volume: crate::diagnostics::VolumeMetrics {
                    peak_density: 3.0,
                    shown_slices: 20,
                    grid_cells: 1000,
                },
            });
        }
        assert_eq!(hud.plot.fps_samples.len(), HISTORY);
        assert!((hud.fps() - 62.5).abs() < 0.01);
        let image = hud.plot.render_image(hud.plot.latest.unwrap());
        assert_eq!(image.pixels.len(), WIDTH * HEIGHT * 4);
    }
    #[test]
    fn background_rasterizer_delivers_an_image_and_joins() {
        let worker = HudWorker::new();
        let mut plot = Plot::default();
        plot.record(FrameMetrics {
            timing: crate::diagnostics::FrameTimingMetrics {
                frame_ms: 6.0,
                ..Default::default()
            },
            volume: crate::diagnostics::VolumeMetrics {
                peak_density: 1.0,
                ..Default::default()
            },
            ..Default::default()
        });
        assert!(worker.offer(&plot));
        let deadline = Instant::now() + Duration::from_secs(2);
        let image = loop {
            if let Some(image) = worker.latest() {
                break image;
            }
            assert!(
                Instant::now() < deadline,
                "HUD rasterizer failed to produce an image"
            );
            std::thread::sleep(Duration::from_millis(5));
        };
        assert_eq!([image.width, image.height], [WIDTH as u32, HEIGHT as u32]);
        assert_eq!(image.pixels.len(), WIDTH * HEIGHT * 4);
        drop(worker);
    }
}
