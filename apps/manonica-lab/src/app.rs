//! Жизненный цикл Meridian SDK и управление экспериментом.
use std::time::Instant;

use meridian_sdk::{
    AppHandler, Device, FlyCamera, GraphicsSession, InputState, KeyCode,
    Vec3, Window, compose_frame, fly_controls,
};

use crate::{simulation::Simulation, visualization::FieldView};

pub struct App {
    runtime: tokio::runtime::Runtime,
    simulation: Simulation,
    camera: FlyCamera,
    graphics: Option<FieldView>,
    paused: bool,
    mouse_locked: bool,
    last_frame: Instant,
    accumulator: f64,
}

impl App {
    pub fn new(simulation: Simulation) -> Self {
        Self {
            runtime: tokio::runtime::Runtime::new().expect("Tokio runtime"),
            simulation,
            camera: FlyCamera::new(Vec3::new(0.0, 3.0, 18.0)),
            graphics: None,
            paused: false,
            mouse_locked: true,
            last_frame: Instant::now(),
            accumulator: 0.0,
        }
    }

    fn advance_simulation(&mut self, elapsed: f64) {
        if self.paused { return; }
        let initial_steps = self.simulation.steps();
        self.accumulator += elapsed.min(0.1);
        let dt = self.simulation.experiment.physics.dt;
        let max_steps = self.simulation.experiment.physics.steps_per_frame;
        for _ in 0..max_steps {
            if self.accumulator < dt { break; }
            self.accumulator -= dt;
            if let Err(error) = self.simulation.step() {
                eprintln!("Simulation paused after numerical error: {error}");
                self.paused = true;
                break;
            }
        }
        // Не накапливать бесконечное отставание при торможении GPU.
        self.accumulator = self.accumulator.min(dt * max_steps as f64 * 2.0);
        if self.simulation.steps() != initial_steps
            && self.simulation.steps() % 250 == 0
        {
            if let Ok(count) = self.simulation.total() {
                println!("t={:.2}s, steps={}, total_manons={:.9}",
                    self.simulation.elapsed(), self.simulation.steps(), count.value());
            }
        }
    }
}

impl AppHandler for App {
    fn on_ready(&mut self, window: &Window) {
        window.set_cursor_grabbed(true);
        let target = window.surface_target();
        let (width, height) = (window.width(), window.height());
        let (device, surface) = self
            .runtime.block_on(Device::new_windowed(target, width, height))
            .expect("failed to initialize Meridian WGPU device");
        let base = GraphicsSession::new(device, surface, width, height);
        self.graphics = Some(FieldView::new(base, &self.simulation.experiment)
            .expect("failed to initialize manon field meshes"));
    }

    fn on_redraw(&mut self, window: &Window, input: &InputState) {
        if input.was_key_pressed(KeyCode::Escape) {
            self.mouse_locked = !self.mouse_locked;
            window.set_cursor_grabbed(self.mouse_locked);
        }
        if input.was_key_pressed(KeyCode::Space) {
            self.paused = !self.paused;
            println!("{}", if self.paused { "Paused" } else { "Running" });
        }
        if input.was_key_pressed(KeyCode::R) {
            if let Err(err) = self.simulation.reset() {
                eprintln!("Unable to reset experiment: {err}");
                self.paused = true;
            } else {
                self.accumulator = 0.0;
                println!("Experiment reset");
            }
        }
        if input.was_key_pressed(KeyCode::F) {
            let visible = &mut self.simulation.experiment.visualization.wisps;
            *visible = !*visible;
        }
        if input.was_key_pressed(KeyCode::ArrowUp) {
            let visible = &mut self.simulation.experiment.visualization.relative_threshold;
            *visible = (*visible * 1.2).min(0.95);
        }
        if input.was_key_pressed(KeyCode::ArrowDown) {
            let visible = &mut self.simulation.experiment.visualization.relative_threshold;
            *visible = (*visible / 1.2).max(0.001);
        }

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f64();
        self.last_frame = now;
        self.advance_simulation(dt);

        if self.mouse_locked {
            self.camera.update(fly_controls(input), dt.min(0.1) as f32);
        }
        let Some(gfx) = &mut self.graphics else { return; };
        let aspect = window.width() as f32 / window.height().max(1) as f32;
        gfx.scene.camera = self.camera.camera(aspect);
        gfx.update(&self.simulation);
        compose_frame(
            &mut gfx.base,
            &gfx.scene,
            &[],
            [0.003, 0.003, 0.012, 1.0],
            window,
        );
    }

    fn on_resized(&mut self, width: u32, height: u32) {
        if let Some(gfx) = &mut self.graphics {
            gfx.base.resize(width, height);
        }
    }
}
