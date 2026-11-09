//! Shared byte-level layout for resident state. Wire floats are always checked.
use crate::error::invalid;
use manonica_error::BaseError;
pub(super) const LEDGER_WORDS: usize = 16;
pub(super) const STATE_WORDS: usize = 4;
pub(super) fn floats(values: impl IntoIterator<Item = f64>) -> Result<Vec<u8>, BaseError> {
    let mut out = Vec::new();
    for value in values {
        let narrow = value as f32;
        if !narrow.is_finite() || (value != 0.0 && narrow == 0.0) {
            return Err(invalid("gpu_coupled_precision"));
        }
        out.extend_from_slice(&narrow.to_le_bytes());
    }
    Ok(out)
}
pub(super) fn decode(bytes: &[u8]) -> Vec<f64> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b) as f64)
        .collect()
}
