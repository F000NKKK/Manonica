mod backend_policy;
mod equilibrium;
mod field;
mod flow;
mod mechanics;
pub use mechanics::MechanicsObservation;
#[cfg(test)]
mod resident_tests;
mod session;
mod worker;

pub use backend_policy::BackendPreference;
pub use field::{cell_center, seeded_field};
pub use session::Simulation;

pub use worker::{
    ComputeBackend, ComputeTelemetry, ConfigurationResult, SimulationSnapshot, SimulationWorker,
};

pub use equilibrium::{EquilibriumPhase, EquilibriumState};

#[cfg(test)]
mod gpu_tests;
