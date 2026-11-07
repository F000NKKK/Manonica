use manonica_core::{Quantity, Signal, Time, TimeStep, units::Unit};
use manonica_error::BaseError;

/// Воспроизводимый кусочно-постоянный псевдослучайный сигнал.
/// Это не идеальный белый шум; значения равномерны в диапазоне [-1, 1].
pub struct SampledNoise<U: Unit> {
    offset: Quantity<U>,
    amplitude: Quantity<U>,
    period: TimeStep,
    seed: u64,
}

impl<U: Unit> SampledNoise<U> {
    pub fn new(offset: Quantity<U>, amplitude: Quantity<U>, period: TimeStep, seed: u64) -> Self {
        Self { offset, amplitude, period, seed }
    }
}

impl<U: Unit> Signal<U> for SampledNoise<U> {
    fn sample(&self, time: Time) -> Result<Quantity<U>, BaseError> {
        let index = (time.seconds() / self.period.seconds()).floor() as i64 as u64;
        let mut bits = index ^ self.seed;
        bits = bits.wrapping_add(0x9E3779B97F4A7C15);
        bits = (bits ^ (bits >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        bits = (bits ^ (bits >> 27)).wrapping_mul(0x94D049BB133111EB);
        bits ^= bits >> 31;
        let uniform = ((bits >> 11) as f64) / ((1u64 << 53) as f64);
        Quantity::new(self.offset.value() + self.amplitude.value() * (uniform * 2.0 - 1.0))
    }
}
