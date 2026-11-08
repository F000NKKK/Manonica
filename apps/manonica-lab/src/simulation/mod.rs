mod backend_policy;
mod equilibrium;
mod field;
mod flow;
mod session;
mod worker;

pub use field::{cell_center, seeded_field};
pub use session::Simulation;
pub use backend_policy::BackendPreference;

pub use worker::{ComputeBackend, ComputeTelemetry, SimulationSnapshot, SimulationWorker};

pub use equilibrium::{EquilibriumPhase, EquilibriumState};
