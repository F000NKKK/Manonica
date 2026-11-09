//! CPU-only event/render handoff. No mutable solver or GPU device is shared.
use crate::{editor::FormulaEditor, experiment::Experiment, simulation::SimulationSnapshot};
use meridian_sdk::FlyControls;
use std::sync::{Arc, Mutex, PoisonError};

/// Only short-lived CPU state crosses the main/render boundary. Neither
/// GraphicsSession nor a mutable physics field is shared between threads.
pub(super) struct InputBridge {
    pub(super) controls: FlyControls,
    pub(super) look_delta: (f32, f32),
    pub(super) mouse_locked: bool,
    pub(super) debug: bool,
    pub(super) paused: bool,
    pub(super) threshold: f64,
    pub(super) wisps: bool,
    pub(super) size: [u32; 2],
    pub(super) snapshot: Arc<SimulationSnapshot>,
    pub(super) editor: FormulaEditor,
    pub(super) experiment: Arc<Experiment>,
    pub(super) revision: u64,
}

impl InputBridge {
    pub(super) fn new(snapshot: SimulationSnapshot, experiment: &Experiment) -> Self {
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

    pub(super) fn take_render_input(&mut self) -> RenderInput {
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

pub(super) struct RenderInput {
    pub(super) controls: FlyControls,
    pub(super) mouse_locked: bool,
    pub(super) debug: bool,
    pub(super) paused: bool,
    pub(super) threshold: f64,
    pub(super) wisps: bool,
    pub(super) size: [u32; 2],
    pub(super) snapshot: Arc<SimulationSnapshot>,
    pub(super) editor: FormulaEditor,
    pub(super) experiment: Arc<Experiment>,
    pub(super) revision: u64,
}

pub(super) fn lock_bridge(
    bridge: &Arc<Mutex<InputBridge>>,
) -> std::sync::MutexGuard<'_, InputBridge> {
    bridge.lock().unwrap_or_else(PoisonError::into_inner)
}
