//! Three independent execution loops:
//! - winit main thread: OS events, input capture and command dispatch
//! - graphics thread: Meridian SDK / WGPU, camera, overlays and presentation
//! - physics thread: fixed-step manon simulation and immutable snapshots
//! Additional atlas worker resamples the field for visualization.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use meridian_sdk::{
    AppHandler, Device, FlyCamera, FlyControls, GraphicsSession, InputState, KeyCode, Vec3, Window,
    compose_frame, fly_controls,
};

use crate::{
    diagnostics::{DebugHud, FrameMetrics},
    editor::{EditorOverlay, FormulaEditor, FormulaLegend},
    experiment::Experiment,
    simulation::{BackendPreference, Simulation, SimulationSnapshot, SimulationWorker},
    visualization::FieldView,
};

const INPUT_POLL_PERIOD: Duration = Duration::from_millis(2);
// FIFO presentation and CPU frame pacing are distinct controls.

/// Only short-lived CPU state crosses the main/render boundary. Neither
/// GraphicsSession nor a mutable physics field is shared between threads.
struct InputBridge {
    controls: FlyControls,
    look_delta: (f32, f32),
    mouse_locked: bool,
    debug: bool,
    paused: bool,
    threshold: f64,
    wisps: bool,
    size: [u32; 2],
    snapshot: Arc<SimulationSnapshot>,
    editor: FormulaEditor,
    experiment: Arc<Experiment>,
    revision: u64,
}

impl InputBridge {
    fn new(snapshot: SimulationSnapshot, experiment: &Experiment) -> Self {
        Self {
            controls: FlyControls::default(),
            look_delta: (0.0, 0.0),
            mouse_locked: true,
            debug: false,
            paused: false,
            threshold: experiment.visualization.relative_threshold,
            wisps: experiment.visualization.wisps,
            size: [1280, 800],
            snapshot: Arc::new(snapshot),
            editor: FormulaEditor::default(),
            experiment: Arc::new(experiment.clone()),
            revision: 0,
        }
    }

    fn take_render_input(&mut self) -> RenderInput {
        let look_delta = std::mem::replace(&mut self.look_delta, (0.0, 0.0));
        let mut controls = self.controls;
        controls.look = if self.mouse_locked {
            look_delta
        } else {
            (0.0, 0.0)
        };
        RenderInput {
            controls,
            mouse_locked: self.mouse_locked,
            debug: self.debug,
            paused: self.paused,
            threshold: self.threshold,
            wisps: self.wisps,
            size: self.size,
            snapshot: Arc::clone(&self.snapshot),
            editor: self.editor.clone(),
            experiment: Arc::clone(&self.experiment),
            revision: self.revision,
        }
    }
}

struct RenderInput {
    controls: FlyControls,
    mouse_locked: bool,
    debug: bool,
    paused: bool,
    threshold: f64,
    wisps: bool,
    size: [u32; 2],
    snapshot: Arc<SimulationSnapshot>,
    editor: FormulaEditor,
    experiment: Arc<Experiment>,
    revision: u64,
}

fn lock_bridge(bridge: &Arc<Mutex<InputBridge>>) -> std::sync::MutexGuard<'_, InputBridge> {
    bridge.lock().unwrap_or_else(PoisonError::into_inner)
}

pub struct App {
    simulation: SimulationWorker,
    experiment: Experiment,
    bridge: Arc<Mutex<InputBridge>>,
    running: Arc<AtomicBool>,
    render_thread: Option<JoinHandle<()>>,
}

impl App {
    pub fn new(simulation: Simulation) -> Self {
        let backend = BackendPreference::resolve(
            None, std::env::var("MANONICA_SIM_BACKEND").ok().as_deref()
        ).expect("Invalid MANONICA_SIM_BACKEND");
        Self::with_backend(simulation, backend)
    }

    pub fn with_backend(simulation: Simulation, backend: BackendPreference) -> Self {
        let experiment = simulation.experiment.clone();
        let (simulation, initial) = SimulationWorker::spawn_with_backend(simulation, backend);
        Self {
            simulation,
            experiment: experiment.clone(),
            bridge: Arc::new(Mutex::new(InputBridge::new(initial, &experiment))),
            running: Arc::new(AtomicBool::new(true)),
            render_thread: None,
        }
    }

    fn process_input(&mut self, window: &Window, input: &InputState) {
        let mut pause = None;
        let mut reset = false;
        let mut apply_formula = None;
        let mut load_comparison = false;

        {
            let mut state = lock_bridge(&self.bridge);
            if input.was_key_pressed(KeyCode::E) && !state.editor.open {
                state.editor.open = true;
                window.set_ime_allowed(true);
                state.mouse_locked = false;
                window.set_cursor_grabbed(false);
                state.editor.load(&self.experiment.formulas);
            } else if state.editor.open {
                if input.was_key_pressed(KeyCode::Escape) {
                    state.editor.open = false;
                    window.set_ime_allowed(false);
                    state.mouse_locked = true;
                    window.set_cursor_grabbed(true);
                } else if input.was_key_pressed(KeyCode::Tab) {
                    state.editor.next();
                } else if input.was_key_pressed(KeyCode::ArrowLeft) {
                    state.editor.switch(&self.experiment.formulas, -1);
                } else if input.was_key_pressed(KeyCode::ArrowRight) {
                    state.editor.switch(&self.experiment.formulas, 1);
                } else if input.was_key_pressed(KeyCode::Digit1) {
                    state.editor.preset(1);
                } else if input.was_key_pressed(KeyCode::Digit2) {
                    state.editor.preset(2);
                } else if input.was_key_pressed(KeyCode::Digit3) {
                    state.editor.preset(3);
                } else if input.was_key_pressed(KeyCode::Backspace) {
                    state.editor.backspace();
                } else if input.was_key_pressed(KeyCode::Enter) {
                    apply_formula = Some((
                        state.editor.circuit,
                        state
                            .editor
                            .update_formula(self.experiment.formulas.get(state.editor.circuit)),
                    ));
                } else {
                    // Text is committed by Meridian/Winit; don't treat navigation
                    // keys as text or move the camera while editing.
                    state.editor.input(input.text());
                }
            } else if input.was_key_pressed(KeyCode::Escape) {
                state.mouse_locked = !state.mouse_locked;
                window.set_cursor_grabbed(state.mouse_locked);
            }
            if !state.editor.open {
                if input.was_key_pressed(KeyCode::C) {
                    load_comparison = true;
                }
                if input.was_key_pressed(KeyCode::Space) {
                    state.paused = !state.paused;
                    pause = Some(state.paused);
                }
                if input.was_key_pressed(KeyCode::R) {
                    reset = true;
                }
                if input.was_key_pressed(KeyCode::F3) {
                    state.debug = !state.debug;
                }
                if input.was_key_pressed(KeyCode::F) {
                    state.wisps = !state.wisps;
                }
                if input.was_key_pressed(KeyCode::ArrowUp) {
                    state.threshold = (state.threshold * 1.2).min(0.95);
                }
                if input.was_key_pressed(KeyCode::ArrowDown) {
                    state.threshold = (state.threshold / 1.2).max(0.001);
                }

                let mut controls = fly_controls(input);
                let delta = controls.look;
                controls.look = (0.0, 0.0);
                state.controls = controls;
                if state.mouse_locked {
                    state.look_delta.0 += delta.0;
                    state.look_delta.1 += delta.1;
                }
            }
            if let Some(snapshot) = self.simulation.poll_latest() {
                state.paused = snapshot.paused;
                state.snapshot = Arc::new(snapshot);
            }
        }

        if let Some((formula_index, formula)) = apply_formula {
            let mut experiment = self.experiment.clone();
            if experiment.formulas.is_empty() {
                experiment.formulas.push(formula);
            } else {
                // Editing the first circuit must not silently destroy the
                // other independently defined physical constructions.
                experiment.formulas[formula_index] = formula;
            }
            // Validate geometry and cell occupancy before modifying either
            // simulation or renderer; invalid input leaves the old state live.
            let validation = Simulation::new(experiment.clone());
            let mut state = lock_bridge(&self.bridge);
            match validation {
                Ok(_) => {
                    state.editor.message = "APPLIED - physics reset; mana capture is active".into();
                    state.editor.open = false;
                    window.set_ime_allowed(false);
                    state.mouse_locked = true;
                    window.set_cursor_grabbed(true);
                    state.experiment = Arc::new(experiment.clone());
                    state.revision += 1;
                    self.experiment = experiment.clone();
                    self.simulation.replace_experiment(experiment);
                }
                Err(err) => {
                    state.editor.message = format!("ERROR: {err}");
                }
            }
        }

        if load_comparison {
            let candidate =
                Experiment::parse(include_str!("../experiments/formula_comparison.toml"));
            match candidate.and_then(|exp| Simulation::new(exp.clone()).map(|_| exp)) {
                Ok(experiment) => {
                    let mut state = lock_bridge(&self.bridge);
                    state.experiment = Arc::new(experiment.clone());
                    state.revision += 1;
                    state.editor.circuit = 0;
                    state.editor.load(&experiment.formulas);
                    state.editor.open = false;
                    state.mouse_locked = true;
                    window.set_cursor_grabbed(true);
                    state.paused = false;
                    state.threshold = experiment.visualization.relative_threshold;
                    state.wisps = experiment.visualization.wisps;
                    self.experiment = experiment.clone();
                    self.simulation.replace_experiment(experiment);
                    window
                        .surface_target()
                        .set_title("Manonica Lab — Сравнение трёх формул");
                    eprintln!("Manonica: switched to 3-formula comparison (key C)");
                }
                Err(err) => eprintln!("Cannot load comparison experiment: {err}"),
            }
        }

        // These messages are asynchronous and cannot block winit or the GPU.
        if let Some(paused) = pause {
            self.simulation.pause(paused);
        }
        if reset {
            self.simulation.reset();
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Release);
        if let Some(handle) = self.render_thread.take() {
            let _ = handle.join();
        }
    }
}

impl AppHandler for App {
    fn on_ready(&mut self, window: &Window) {
        window.set_cursor_grabbed(true);
        let cloned_window = window.clone();
        let bridge = Arc::clone(&self.bridge);
        let running = Arc::clone(&self.running);
        let experiment = self.experiment.clone();

        self.render_thread = Some(
            thread::Builder::new()
                .name("manonica-render".into())
                .spawn(move || render_loop(cloned_window, experiment, bridge, running))
                .expect("failed to start graphics thread"),
        );
    }

    fn on_redraw(&mut self, window: &Window, input: &InputState) {
        // OS messages/input only: never call a physics solver, GPU resource
        // upload, or Present from the window's event loop.
        self.process_input(window, input);
        // Meridian 0.8.0 currently schedules redraws continuously under
        // ControlFlow::Poll; avoid burning a CPU core spinning the event loop.
        thread::sleep(INPUT_POLL_PERIOD);
    }

    fn on_resized(&mut self, width: u32, height: u32) {
        lock_bridge(&self.bridge).size = [width, height];
    }
}

fn target_render_rate(window: &Window) -> (f64, &'static str) {
    if let Some(fps) = std::env::var("MANONICA_RENDER_FPS")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|fps| fps.is_finite() && (30.0..=360.0).contains(fps))
    {
        return (fps, "configured");
    }
    let refresh = window
        .surface_target()
        .current_monitor()
        .and_then(|monitor| monitor.refresh_rate_millihertz())
        .filter(|&mhz| (30_000..=500_000).contains(&mhz))
        .map(|mhz| f64::from(mhz) / 1000.0);
    (
        refresh.unwrap_or(60.0),
        if refresh.is_some() {
            "monitor"
        } else {
            "fallback"
        },
    )
}

/// Paces frame submission, retaining FIFO presentation as a separate guarantee.
fn pace_frame(deadline: &mut Instant, interval: Duration) {
    let now = Instant::now();
    if now < *deadline {
        std::thread::sleep(*deadline - now);
    }
    *deadline += interval;
    if *deadline <= Instant::now() {
        *deadline = Instant::now() + interval;
    }
}

fn render_loop(
    window: Window,
    mut experiment: Experiment,
    bridge: Arc<Mutex<InputBridge>>,
    running: Arc<AtomicBool>,
) {
    let runtime = tokio::runtime::Runtime::new().expect("Tokio runtime");
    let (frame_cap, rate_source) = target_render_rate(&window);
    let interval = Duration::from_secs_f64(1.0 / frame_cap);
    let size = [window.width().max(1), window.height().max(1)];
    let target = window.surface_target();
    let (device, mut surface) = runtime
        .block_on(Device::new_windowed(target, size[0], size[1]))
        .expect("failed to initialize Meridian WGPU device on rendering thread");
    // The SDK 0.8 source revision exposes explicit FIFO selection.
    // Do this *before* building GraphicsSession so its swapchain is paced
    // by vertical blanking rather than a second 120 Hz software timer.
    surface.enable_vsync(&device);
    let base = GraphicsSession::new(device, surface, size[0], size[1]);
    let mut graphics =
        FieldView::new(base, &experiment).expect("failed to initialize manon rendering resources");
    let mut debug = DebugHud::default();
    let mut editor_overlay = EditorOverlay::default();
    let mut formula_legend = FormulaLegend::default();
    let mut revision = 0_u64;
    let mut camera = FlyCamera::new(Vec3::new(0.0, 3.0, 18.0));
    let mut last_frame = Instant::now();
    let mut deadline = last_frame;
    let mut last_draw_cpu_ms = 0.0;
    let mut initial_manons = lock_bridge(&bridge).snapshot.total;
    let mut current_size = size;
    eprintln!(
        "Manonica: FIFO VSync + paced render target {:.3} FPS ({rate_source})",
        frame_cap
    );

    while running.load(Ordering::Acquire) {
        pace_frame(&mut deadline, interval);
        let frame_begin = Instant::now();
        let dt = frame_begin.duration_since(last_frame).as_secs_f64();
        last_frame = frame_begin;

        let render_input = lock_bridge(&bridge).take_render_input();
        if render_input.revision != revision {
            experiment = (*render_input.experiment).clone();
            initial_manons = experiment.seeds.iter().map(|seed| seed.total_manons).sum();
            if let Err(err) = graphics.replace_formulas(&experiment) {
                eprintln!("failed to replace formula geometry: {err}");
            }
            revision = render_input.revision;
        }
        if render_input.size[0] == 0 || render_input.size[1] == 0 {
            continue;
        }
        if render_input.size != current_size {
            graphics
                .base
                .resize(render_input.size[0], render_input.size[1]);
            current_size = render_input.size;
        }
        if render_input.mouse_locked {
            camera.update(render_input.controls, dt.min(0.1) as f32);
        }

        experiment.visualization.relative_threshold = render_input.threshold;
        experiment.visualization.wisps = render_input.wisps;
        let aspect = current_size[0] as f32 / current_size[1].max(1) as f32;
        graphics.scene.camera = camera.camera(aspect);

        let cpu_start = Instant::now();
        graphics.update(&render_input.snapshot, &experiment);
        let graphics_update_cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;
        let field = &render_input.snapshot.field;
        let metrics = FrameMetrics {
            frame_ms: dt * 1000.0,
            sim_cpu_ms: render_input.snapshot.sim_cpu_ms,
            compute: render_input.snapshot.compute,
            draw_cpu_ms: last_draw_cpu_ms,
            field_cpu_ms: graphics.sampling_cpu_ms(),
            sim_time: render_input.snapshot.elapsed,
            steps: render_input.snapshot.steps,
            total: render_input.snapshot.total,
            bound: render_input.snapshot.bound,
            formula_bound: std::array::from_fn(|i| {
                render_input
                    .snapshot
                    .formula_bound
                    .get(i)
                    .copied()
                    .unwrap_or(0.0)
            }),
            formula_count: render_input.snapshot.formula_bound.len(),
            material_energy: render_input.snapshot.material_energy,
            potential_peak: render_input.snapshot.potential_peak,
            field_energy_joules: render_input.snapshot.field_energy_joules,
            field_heat_joules: render_input.snapshot.field_heat_joules,
            field_work_joules: render_input.snapshot.field_work_joules,
            potential_transport: render_input.snapshot.potential_transport,
            trap_temperature_kelvin: render_input.snapshot.trap_temperature_kelvin,
            exported_heat_joules: render_input.snapshot.exported_heat_joules,
            field_shutdown: render_input.snapshot.field_shutdown,
            equilibrium: render_input.snapshot.equilibrium,
            initial_total: initial_manons,
            peak_density: graphics.peak_density(),
            shown_slices: graphics.visible_slices(),
            grid_cells: field.shape().len(),
            paused: render_input.paused || render_input.snapshot.paused,
        };
        debug.record(metrics);
        if let Some(error) = &render_input.snapshot.error {
            eprintln!("Physics suspended: {error}");
            running.store(false, Ordering::Release);
        }
        let mut overlays = Vec::new();
        if render_input.debug {
            if let Some(overlay) =
                debug.overlay(&mut graphics.base, current_size, render_input.editor.open)
            {
                overlays.push(overlay);
            }
        }
        if let Some(overlay) =
            editor_overlay.draw(&mut graphics.base, &render_input.editor, current_size)
        {
            overlays.push(overlay);
        }
        if let Some(overlay) = formula_legend.draw(
            &mut graphics.base,
            &experiment,
            &render_input.snapshot.formula_bound,
            current_size,
            render_input.editor.open,
        ) {
            overlays.push(overlay);
        }
        let draw_start = Instant::now();
        compose_frame(
            &mut graphics.base,
            &graphics.scene,
            &overlays,
            [0.003, 0.003, 0.012, 1.0],
            &window,
        );
        last_draw_cpu_ms = draw_start.elapsed().as_secs_f64() * 1000.0 + graphics_update_cpu_ms;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::ExperimentBuilder;

    #[test]
    fn input_bridge_coalesces_mouse_motion_and_consumes_it_once() {
        let experiment = ExperimentBuilder::new("input")
            .grid([4, 4, 4], 1.0)
            .build()
            .unwrap();
        let sim = Simulation::new(experiment.clone()).unwrap();
        let (_, snapshot) = SimulationWorker::spawn(sim);
        let mut bridge = InputBridge::new(snapshot, &experiment);
        bridge.look_delta = (13.0, -4.0);
        let a = bridge.take_render_input();
        assert_eq!(a.controls.look, (13.0, -4.0));
        let b = bridge.take_render_input();
        assert_eq!(b.controls.look, (0.0, 0.0));
    }
}
