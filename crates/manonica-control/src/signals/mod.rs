mod chirp;
mod combinators;
mod constant;
mod damped_sine;
mod exponential;
mod gaussian;
mod noise;
mod pulse;
mod ramp;
mod sawtooth;
mod sine;
mod square;
mod step;
mod triangle;

pub use chirp::Chirp;
pub use combinators::{Delay, Scale, Sum};
pub use constant::Constant;
pub use damped_sine::DampedSine;
pub use exponential::Exponential;
pub use gaussian::Gaussian;
pub use noise::SampledNoise;
pub use pulse::Pulse;
pub use ramp::Ramp;
pub use sawtooth::Sawtooth;
pub use sine::Sine;
pub use square::Square;
pub use step::Step;
pub use triangle::Triangle;

#[cfg(test)]
mod tests;
