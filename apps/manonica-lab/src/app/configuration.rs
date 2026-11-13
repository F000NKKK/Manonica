//! Apply experiment edits only after worker acknowledgement; no solver runs
//! in the OS event callback and stale snapshots cannot overwrite accepted state.
use super::{App, lock_bridge};
use crate::{experiment::Experiment, simulation::ConfigurationResult};
use meridian_sdk::Window;
use std::sync::Arc;

pub(super) enum ConfigurationIntent {
    Formula,
    Experiment,
}
pub(super) struct PendingConfiguration {
    revision: u64,
    experiment: Experiment,
    intent: ConfigurationIntent,
}
impl App {
    pub(super) fn submit_configuration(
        &mut self,
        experiment: Experiment,
        intent: ConfigurationIntent,
    ) {
        if self.pending_experiment.is_some() {
            return;
        }
        match self.simulation.replace_experiment(experiment.clone()) {
            Ok(revision) => {
                eprintln!("Manonica: preparing experiment revision {revision}");
                self.pending_experiment = Some(PendingConfiguration {
                    revision,
                    experiment,
                    intent,
                });
                lock_bridge(&self.bridge).editor.message =
                    "APPLYING - preparing physics on worker...".into();
            }
            Err(error) => {
                lock_bridge(&self.bridge).editor.message = format!("ERROR: {error}");
            }
        }
    }
    pub(super) fn poll_configuration(&mut self, window: &Window) {
        while let Some(result) = self.simulation.poll_configuration() {
            let revision = match &result {
                ConfigurationResult::Applied { revision, .. }
                | ConfigurationResult::Rejected { revision, .. } => *revision,
            };
            if self
                .pending_experiment
                .as_ref()
                .is_none_or(|pending| pending.revision != revision)
            {
                continue;
            }
            let Some(pending) = self.pending_experiment.take() else {
                continue;
            };
            let mut state = lock_bridge(&self.bridge);
            match result {
                ConfigurationResult::Applied { snapshot, .. } => {
                    if matches!(pending.intent, ConfigurationIntent::Experiment) {
                        state.threshold = pending.experiment.visualization.relative_threshold;
                        state.field_display = if pending.experiment.mechanics.is_some() {
                            crate::visualization::FieldDisplay::Potential
                        } else {
                            crate::visualization::FieldDisplay::Density
                        };
                        state.wisps = pending.experiment.visualization.wisps;
                        state.editor.circuit = 0;
                        state.editor.load(&pending.experiment.formulas);
                    }
                    state.experiment = Arc::new(pending.experiment.clone());
                    state.revision = revision;
                    state.paused = snapshot.paused;
                    state.snapshot = Arc::new(*snapshot);
                    state.editor.message = "APPLIED - physics restarted".into();
                    state.editor.open = false;
                    state.mouse_locked = true;
                    window.set_ime_allowed(false);
                    window.set_cursor_grabbed(true);
                    window
                        .surface_target()
                        .set_title(&format!("Manonica Lab — {}", pending.experiment.name));
                    self.experiment = pending.experiment;
                    eprintln!("Manonica: applied experiment revision {revision}");
                }
                ConfigurationResult::Rejected { message, .. } => {
                    state.editor.message = format!("ERROR: {message}");
                    eprintln!("Manonica: rejected experiment revision {revision}: {message}");
                }
            }
        }
    }
}
