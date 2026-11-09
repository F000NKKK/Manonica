//! Immutable problem metadata and validated accelerator readback.
use super::{LocalBoundarySite, PotentialBoundary, PotentialField, bad};
use manonica_error::BaseError;
impl PotentialField {
    pub fn cell_width_metres(&self) -> f64 {
        self.dx
    }
    pub fn permittivity(&self) -> f64 {
        self.permittivity
    }
    pub fn screening(&self) -> f64 {
        self.screening
    }
    pub fn boundary(&self) -> PotentialBoundary {
        self.boundary
    }
    pub fn sources(&self) -> &[f64] {
        &self.sources
    }
    pub fn iterations(&self) -> usize {
        self.iterations
    }
    pub fn local_boundaries(&self) -> impl Iterator<Item = LocalBoundarySite> + '_ {
        self.local
            .iter()
            .enumerate()
            .filter_map(|(index, condition)| {
                condition.map(|condition| LocalBoundarySite { index, condition })
            })
    }
    /// Import a solver snapshot atomically. Equation parity and residuals are
    /// the caller's responsibility; malformed storage is always rejected.
    pub fn try_replace_values(&mut self, values: Vec<f64>) -> Result<(), BaseError> {
        if values.len() != self.shape.len() || values.iter().any(|v| !v.is_finite()) {
            return Err(bad("potential_snapshot_values"));
        }
        self.values = values;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn snapshot_validation_and_scaled_problem_remain_consistent() {
        let shape = crate::GridShape::new(2, 1, 1).unwrap();
        let mut p = PotentialField::solve(
            shape,
            1.0,
            &[2.0, 0.0],
            1.0,
            1.0,
            PotentialBoundary::Insulated,
            20,
        )
        .unwrap();
        let old = p.values().to_vec();
        assert!(p.try_replace_values(vec![f64::NAN; 2]).is_err());
        assert_eq!(p.values(), old);
        assert!(p.try_replace_values(vec![1.0]).is_err());
        assert_eq!(p.scaled(0.5).unwrap().sources(), &[1.0, 0.0]);
        assert_eq!(p.iterations(), 20);
    }
}
