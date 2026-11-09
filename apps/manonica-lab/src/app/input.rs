//! OS event routing; physical reconfiguration is asynchronous.
use super::{App, configuration::ConfigurationIntent, lock_bridge};
use crate::experiment::Experiment;
use meridian_sdk::{InputState, KeyCode, Window, fly_controls};
use std::sync::Arc;

impl App {
    pub(super) fn process_input(&mut self, window: &Window, input: &InputState) {
        self.poll_configuration(window);
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
            } else if state.editor.open && self.pending_experiment.is_none() {
                let ctrl = input.is_key_down(KeyCode::ControlLeft)
                    || input.is_key_down(KeyCode::ControlRight);
                if ctrl && input.was_key_pressed(KeyCode::A) {
                    state.editor.select_all();
                } else if ctrl && input.was_key_pressed(KeyCode::C) {
                    if let Some(text) = state.editor.selected_text() {
                        if self.clipboard.is_none() {
                            self.clipboard = arboard::Clipboard::new().ok();
                        }
                        match self.clipboard.as_mut() {
                            Some(clipboard) => match clipboard.set_text(text.to_owned()) {
                                Ok(()) => state.editor.message = "COPIED SELECTION".into(),
                                Err(err) => {
                                    state.editor.message = format!("CLIPBOARD COPY ERROR: {err}")
                                }
                            },
                            None => state.editor.message = "OS CLIPBOARD UNAVAILABLE".into(),
                        }
                    }
                } else if ctrl && input.was_key_pressed(KeyCode::V) {
                    if self.clipboard.is_none() {
                        self.clipboard = arboard::Clipboard::new().ok();
                    }
                    match self.clipboard.as_mut() {
                        Some(clipboard) => match clipboard.get_text() {
                            Ok(value) => {
                                state.editor.paste(&value);
                                state.editor.message = "PASTED INTO SELECTED FORMULA".into();
                            }
                            Err(err) => {
                                state.editor.message = format!("CLIPBOARD PASTE ERROR: {err}")
                            }
                        },
                        None => state.editor.message = "OS CLIPBOARD UNAVAILABLE".into(),
                    }
                } else if input.was_key_pressed(KeyCode::Escape) {
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
                } else if ctrl && input.was_key_pressed(KeyCode::Digit1) {
                    state.editor.preset(1);
                } else if ctrl && input.was_key_pressed(KeyCode::Digit2) {
                    state.editor.preset(2);
                } else if ctrl && input.was_key_pressed(KeyCode::Digit3) {
                    state.editor.preset(3);
                } else if input.was_key_pressed(KeyCode::Backspace) {
                    state.editor.backspace();
                } else if input.was_key_pressed(KeyCode::Enter)
                    || input.text().contains(['\r', '\n'])
                {
                    state.editor.input(input.text());
                    apply_formula = Some((
                        state.editor.circuit,
                        state
                            .editor
                            .update_formula(self.experiment.formulas.get(state.editor.circuit)),
                    ));
                } else if !ctrl {
                    // Ignore committed shortcut characters while Control is held.
                    state.editor.input(input.text());
                }
            } else if state.editor.open {
                // A submitted configuration is immutable until acknowledged.
                // Keep the window responsive without losing edits to a race.
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
            if state.editor.open && self.pending_experiment.is_none() {
                state.editor.validate_physical_preview(&self.experiment);
            }
            if let Some(snapshot) = self.simulation.poll_latest()
                && snapshot.revision == state.revision
            {
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
            self.submit_configuration(experiment, ConfigurationIntent::Formula);
        }

        if load_comparison && self.pending_experiment.is_none() {
            match Experiment::parse(include_str!("../../experiments/formula_comparison.toml")) {
                Ok(experiment) => {
                    self.submit_configuration(experiment, ConfigurationIntent::Experiment)
                }
                Err(error) => eprintln!("Cannot load comparison experiment: {error}"),
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
