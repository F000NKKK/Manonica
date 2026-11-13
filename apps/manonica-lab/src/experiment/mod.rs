//! Декларативные эксперименты. Не зависят от SDK визуализации.

mod builder;
mod error;
mod formula;
mod mechanics;
pub use mechanics::{MechanicalBodyConfig, MechanicsConfig};
mod specification;

pub use builder::ExperimentBuilder;
pub use error::ExperimentError;
pub use formula::{capture_cells, sample_curve};
pub use specification::{
    AmbientFlow, ChannelSpec, Diffusion, Experiment, FieldSeed, FieldSource, Formula, FormulaTrap,
    Grid, LocalBoundarySpec, MaterialSite, PotentialBoundaryKind, PotentialConfig, SeedShape,
    Visualization,
};

#[cfg(test)]
mod tests;
