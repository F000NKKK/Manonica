//! Декларативные эксперименты. Не зависят от SDK визуализации.

mod builder;
mod error;
mod formula;
mod specification;

pub use builder::ExperimentBuilder;
pub use error::ExperimentError;
pub use formula::sample_curve;
pub use specification::{
    AmbientFlow, ChannelSpec, Circle, Diffusion, Experiment, FieldSeed, Grid, MaterialSite,
    Formula, FormulaTrap, SeedShape, Visualization,
};

#[cfg(test)]
mod tests;
