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
    // The renderer closes the last edge to the first point. Refuse a
    // non-periodic parameterization rather than inventing a long edge.
    let end = [
        x(std::f64::consts::TAU),
        y(std::f64::consts::TAU),
        z(std::f64::consts::TAU),
    ];
    let start = [x(0.0), y(0.0), z(0.0)];
    if end.iter().chain(start.iter()).any(|v| !v.is_finite()) {
        return Err(invalid());
    }
    let max_coord = samples
        .iter()
        .flatten()
        .map(|v| f64::from(v.abs()))
        .fold(1.0_f64, f64::max);
    let closing_error = (0..3)
        .map(|i| (end[i] - start[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    if closing_error > max_coord * 1e-3 {
        return Err(invalid());
    }
    if samples.windows(2).all(|p| p[0] == p[1]) {
        return Err(invalid());
    }
    Ok(samples)
}

/// Rasterise the same line segments drawn by the renderer into physical
/// trap cells. Sampling *only the vertices* made capture depend on segment
/// count and ignored most of a long visible segment.
pub fn capture_cells(
    formula: &Formula,
    dimensions: [usize; 3],
    dx: f64,
) -> Result<Vec<[usize; 3]>, ExperimentError> {
    if !dx.is_finite() || dx <= 0.0 || dimensions.contains(&0) {
        return Err(invalid());
    }
    let points = sample_curve(formula)?;
    let mut cells = std::collections::HashSet::new();
    for i in 0..points.len() {
        let from = points[i];
        let to = points[(i + 1) % points.len()];
        let length = (0..3)
            .map(|axis| f64::from(to[axis] - from[axis]).powi(2))
            .sum::<f64>()
            .sqrt();
        let steps = (length / (dx * 0.5)).ceil().max(1.0);
        if !steps.is_finite() || steps > 512.0 {
            return Err(invalid());
        }
        let steps = steps as usize;
        for j in 0..=steps {
            let t = j as f64 / steps as f64;
            let mut cell = [0_usize; 3];
            for axis in 0..3 {
                let coordinate = (f64::from(from[axis]) * (1.0 - t) + f64::from(to[axis]) * t) / dx
                    + dimensions[axis] as f64 * 0.5;
                if !coordinate.is_finite()
                    || coordinate < 0.0
                    || coordinate >= dimensions[axis] as f64
                {
                    return Err(invalid());
                }
                cell[axis] = coordinate.floor() as usize;
            }
            cells.insert(cell);
        }
    }
    let mut sorted: Vec<_> = cells.into_iter().collect();
    sorted.sort_unstable();
    Ok(sorted)
}

#[cfg(test)]
mod tests {
    #[test]
    fn physical_cells_follow_segments_not_only_vertices() {
        let formula = Formula {
            name: "circle".into(),
            x: "2*cos(t)".into(),
            y: "2*sin(t)".into(),
            z: "0".into(),
            segments: 8,
            thickness: 0.06,
            color: [1.0, 1.0, 1.0],
            trap: None,
        };
        let cells = capture_cells(&formula, [24, 24, 24], 0.25).unwrap();
        assert!(
            cells.len() > 8,
            "segments must populate intermediate physical cells"
        );
        let unique = cells.iter().collect::<std::collections::HashSet<_>>();
        assert_eq!(unique.len(), cells.len());
    }

    use super::*;
    #[test]
    fn nonperiodic_formula_does_not_invent_closing_segment() {
        let formula = Formula {
            name: "open".into(),
            x: "t".into(),
            y: "0".into(),
            z: "0".into(),
            segments: 64,
            thickness: 0.03,
            color: [0.5, 0.8, 1.0],
            trap: None,
        };
        assert!(sample_curve(&formula).is_err());
    }

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
