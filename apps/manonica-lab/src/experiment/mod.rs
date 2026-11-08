//! Декларативные эксперименты. Не зависят от SDK визуализации.

mod builder;
mod error;
mod specification;

pub use builder::ExperimentBuilder;
pub use error::ExperimentError;
pub use specification::{
    Circle, Diffusion, Experiment, FieldSeed, Grid, SeedShape, Visualization,
};

#[cfg(test)]
mod tests;
