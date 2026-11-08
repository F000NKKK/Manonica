//! Parameterized spatial circuits. The same samples drive the mesh and
//! physical capture sites; no hard-coded spell or energy source is involved.
use std::str::FromStr;

use manonica_error::{BaseError, ErrorCode};

use super::{ExperimentError, Formula};

fn invalid() -> ExperimentError {
    BaseError::new(ErrorCode::InvalidRange, "formula_expression").into()
}

/// Evaluate a bounded, deterministic parametric curve for t in [0, 2π].
/// Expressions may use t, pi and meval mathematical functions.
pub fn sample_curve(formula: &Formula) -> Result<Vec<[f32; 3]>, ExperimentError> {
    if !(8..=256).contains(&formula.segments) {
        return Err(invalid());
    }
    let x = meval::Expr::from_str(&formula.x).map_err(|_| invalid())?;
    let y = meval::Expr::from_str(&formula.y).map_err(|_| invalid())?;
    let z = meval::Expr::from_str(&formula.z).map_err(|_| invalid())?;
    let x = x.bind("t").map_err(|_| invalid())?;
    let y = y.bind("t").map_err(|_| invalid())?;
    let z = z.bind("t").map_err(|_| invalid())?;
    let mut samples = Vec::with_capacity(formula.segments);
    for i in 0..formula.segments {
        let t = std::f64::consts::TAU * i as f64 / formula.segments as f64;
        let p = [x(t), y(t), z(t)];
        if p.iter().any(|v| !v.is_finite() || v.abs() > 100_000.0) {
            return Err(invalid());
        }
        samples.push([p[0] as f32, p[1] as f32, p[2] as f32]);
    }
    if samples.windows(2).all(|p| p[0] == p[1]) {
        return Err(invalid());
    }
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parametrized_circle_is_closed_with_finite_coordinates() {
        let formula = Formula {
            name: "test".into(),
            x: "2*cos(t)".into(),
            y: "0".into(),
            z: "2*sin(t)".into(),
            segments: 64,
            thickness: 0.03,
            color: [0.5, 0.8, 1.0],
            trap: None,
        };
        let samples = sample_curve(&formula).unwrap();
        assert_eq!(samples.len(), 64);
        assert!((samples[0][0] - 2.0).abs() < 1e-6);
        assert!(samples.iter().all(|p| p.iter().all(|x| x.is_finite())));
    }
}
