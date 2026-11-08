//! Декларативные эксперименты. Не зависят от SDK визуализации.

mod builder;
mod error;
mod formula;
mod specification;

pub use builder::ExperimentBuilder;
pub use error::ExperimentError;
pub use formula::{capture_cells, sample_curve};
pub use specification::{
    AmbientFlow, ChannelSpec, Diffusion, Experiment, FieldSeed, Grid, MaterialSite,
    FieldSource, Formula, FormulaTrap, PotentialBoundaryKind, PotentialConfig, SeedShape, Visualization,
};

#[cfg(test)]
mod tests;
