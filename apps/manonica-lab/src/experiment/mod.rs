//! Декларативные эксперименты. Не зависят от SDK визуализации.

mod builder;
mod error;
mod specification;

pub use builder::ExperimentBuilder;
pub use error::ExperimentError;
pub use specification::{
    AmbientFlow, ChannelSpec, Circle, Diffusion, Experiment, FieldSeed, Grid, MaterialSite,
    SeedShape, Visualization,
};

#[cfg(test)]
mod tests;
