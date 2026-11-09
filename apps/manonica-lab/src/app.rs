//! Three independent execution loops:
//! - winit main thread: OS events, input capture and command dispatch
//! - graphics thread: Meridian SDK / WGPU, camera, overlays and presentation
//! - physics thread: fixed-step manon simulation and immutable snapshots
//!
//! Additional atlas worker resamples the field for visualization.

mod bridge;
mod configuration;
mod frame_trace;
mod input;
mod render;
use bridge::{InputBridge, lock_bridge};
use configuration::PendingConfiguration;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use meridian_sdk::{AppHandler, InputState, Window};

use crate::{
    experiment::Experiment,
    simulation::{BackendPreference, Simulation, SimulationWorker},
};

const INPUT_POLL_PERIOD: Duration = Duration::from_millis(2);
// FIFO presentation and CPU frame pacing are distinct controls.

pub struct App {
    simulation: SimulationWorker,
    experiment: Experiment,
    bridge: Arc<Mutex<InputBridge>>,
    running: Arc<AtomicBool>,
    render_thread: Option<JoinHandle<()>>,
    /// Clipboard ownership must persist on Linux after Ctrl+C.
    clipboard: Option<arboard::Clipboard>,
    pending_experiment: Option<PendingConfiguration>,
}

impl App {
    pub fn new(simulation: Simulation) -> Self {
        let backend =
            BackendPreference::resolve(None, std::env::var("MANONICA_SIM_BACKEND").ok().as_deref())
                .expect("Invalid MANONICA_SIM_BACKEND");
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
            clipboard: None,
            pending_experiment: None,
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Release);
        self.simulation.request_shutdown();
        if let Some(handle) = self.render_thread.take() {
            let _ = handle.join();
        }
    }
}

impl AppHandler for App {
    fn on_close_requested(&mut self) -> bool {
        self.running.store(false, Ordering::Release);
        self.simulation.request_shutdown();
        true
    }

    fn ready_to_close(&self) -> bool {
        self.render_thread
            .as_ref()
            .is_none_or(JoinHandle::is_finished)
            && self.simulation.is_finished()
    }

    fn on_ready(&mut self, window: &Window) {
        window.set_cursor_grabbed(true);
        let cloned_window = window.clone();
        let bridge = Arc::clone(&self.bridge);
        let running = Arc::clone(&self.running);
        let experiment = self.experiment.clone();

        self.render_thread = Some(
            thread::Builder::new()
                .name("manonica-render".into())
                .spawn(move || render::render_loop(cloned_window, experiment, bridge, running))
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
