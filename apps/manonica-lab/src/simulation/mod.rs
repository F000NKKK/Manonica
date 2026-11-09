mod backend_policy;
mod equilibrium;
mod field;
mod flow;
#[cfg(test)]
mod resident_tests;
mod session;
mod worker;

pub use backend_policy::BackendPreference;
pub use field::{cell_center, seeded_field};
pub use session::Simulation;

pub use worker::{ComputeBackend, ComputeTelemetry, SimulationSnapshot, SimulationWorker};

pub use equilibrium::{EquilibriumPhase, EquilibriumState};

#[cfg(test)]
mod gpu_tests;
