//! Quasi-static screened potential and conservative finite-volume manon drift.
//! Model: (-kappa Laplacian + screening) phi = source,
//!         J_potential = -mobility * n * grad(phi).
//! Source strength, mobility, screening and dielectric response are model
//! parameters, NOT real-world constants. No external boundary manon flux.
use super::{FreeManaField, GridShape};
use manonica_core::{Quantity, units::ManonDensity};
use manonica_error::{BaseError, ErrorCode};

fn bad(parameter: &'static str) -> BaseError {
    BaseError::new(ErrorCode::InvalidRange, parameter)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PotentialBoundary {
    /// Potential on the outside of the computational box is fixed at zero.
    Grounded,
    /// Normal derivative is zero at the outside computational faces.
    Insulated,
}

/// Embedded boundary imposed on a rasterized mathematical construction.
/// Dirichlet pins cell-center potential; Neumann isolates normal face flux
/// across the marked/unmarked interface; Robin couples a marked cell to an
/// external potential reservoir through a finite surface conductance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LocalPotentialBoundary {
    Dirichlet { value: f64 },
    Neumann,
    Robin { reference: f64, exchange: f64 },
}
#[derive(Debug, Clone, Copy)]
pub struct LocalBoundarySite {
    pub index: usize,
    pub condition: LocalPotentialBoundary,
}
#[derive(Debug, Clone)]
pub struct PotentialField {
    shape: GridShape,
    dx: f64,
    values: Vec<f64>,
    boundary: PotentialBoundary,
    permittivity: f64,
    screening: f64,
    local: Vec<Option<LocalPotentialBoundary>>,
}
#[derive(Debug, Clone, Copy)]
pub struct PotentialStep {
    /// Integral count, transferred across cell faces in this step.
    pub transported_manons: f64,
    pub before_manons: f64,
    pub after_manons: f64,
    /// Change in sum(n * phi * cell volume); this is NOT heat on its own.
    pub potential_energy_change_joules: f64,
}
impl PotentialField {
    pub fn solve(
        shape: GridShape,
        dx: f64,
        sources: &[f64],
        permittivity: f64,
        screening: f64,
        boundary: PotentialBoundary,
        iterations: usize,
    ) -> Result<Self, BaseError> {
        if !dx.is_finite()
            || dx <= 0.0
            || !permittivity.is_finite()
            || permittivity <= 0.0
            || !screening.is_finite()
            || screening <= 0.0
            || !(1..=4096).contains(&iterations)
            || sources.len() != shape.len()
            || sources.iter().any(|v| !v.is_finite())
        {
            return Err(bad("potential_parameters"));
        }
        Self::solve_with_local(
            shape, dx, sources, permittivity, screening, boundary, iterations, &[],
        )
    }
    pub fn solve_with_local(
        shape: GridShape,
        dx: f64,
        sources: &[f64],
        permittivity: f64,
        screening: f64,
        boundary: PotentialBoundary,
        iterations: usize,
        sites: &[LocalBoundarySite],
    ) -> Result<Self, BaseError> {
        if !dx.is_finite() || dx <= 0.0 || !permittivity.is_finite()
            || permittivity <= 0.0 || !screening.is_finite() || screening <= 0.0
            || !(1..=4096).contains(&iterations) || sources.len() != shape.len()
            || sources.iter().any(|v| !v.is_finite())
        { return Err(bad("potential_parameters")); }
        let mut local = vec![None; shape.len()];
        for site in sites {
            if site.index >= shape.len() || local[site.index].is_some() {
                return Err(bad("local_boundary_overlap"));
            }
            let valid = match site.condition {
                LocalPotentialBoundary::Dirichlet { value } => value.is_finite(),
                LocalPotentialBoundary::Neumann => true,
                LocalPotentialBoundary::Robin { reference, exchange } =>
                    reference.is_finite() && exchange.is_finite() && exchange > 0.0,
            };
            if !valid { return Err(bad("local_boundary_parameters")); }
            local[site.index] = Some(site.condition);
        }
        let alpha = permittivity / (dx * dx);
        let mut old = vec![0.0; shape.len()];
        let mut next = old.clone();
        for _ in 0..iterations {
            for z in 0..shape.nz() {
                for y in 0..shape.ny() {
                    for x in 0..shape.nx() {
                        let index = shape.index(x, y, z).unwrap();
                        if let Some(LocalPotentialBoundary::Dirichlet { value }) = local[index] {
                            next[index] = value;
                            continue;
                        }
                        let mut neighbours = 0.0;
                        let mut count = 0.0;
                        // Missing neighbours become ghost cells: phi=0 for
                        // grounded, phi=current for insulated, yielding
                        // a zero exterior normal gradient.
                        for (axis, (coordinate, limit)) in
                            [(x, shape.nx()), (y, shape.ny()), (z, shape.nz())]
                                .into_iter()
                                .enumerate()
                        {
                            for offset in [-1_i8, 1_i8] {
                                let at = if offset < 0 {
                                    coordinate.checked_sub(1)
                                } else {
                                    coordinate.checked_add(1).filter(|v| *v < limit)
                                };
                                if let Some(v) = at {
                                    let coords = match axis_index(axis, x, y, z, v) {
                                        Some(c) => c,
                                        None => continue,
                                    };
                                    let neighbour =
                                        shape.index(coords[0], coords[1], coords[2]).unwrap();
                                    // An embedded Neumann surface carries no
                                    // normal coupling across its discrete interface.
                                    if matches!(local[index], Some(LocalPotentialBoundary::Neumann))
                                        || matches!(local[neighbour], Some(LocalPotentialBoundary::Neumann))
                                    {
                                        continue;
                                    }
                                    neighbours += old[neighbour];
                                    count += 1.0;
                                } else if boundary == PotentialBoundary::Grounded {
                                    count += 1.0;
                                }
                            }
                        }
                        let (exchange, reference) = match local[index] {
                            Some(LocalPotentialBoundary::Robin { reference, exchange }) =>
                                (exchange / dx, reference),
                            _ => (0.0, 0.0),
                        };
                        next[index] = (sources[index] + alpha * neighbours + exchange * reference)
                            / (screening + alpha * count + exchange);
                        if !next[index].is_finite() {
                            return Err(bad("potential_overflow"));
                        }
                    }
                }
            }
            std::mem::swap(&mut next, &mut old);
        }
        Ok(Self {
            shape,
            dx,
            values: old,
            boundary,
            permittivity,
            screening,
            local,
        })
    }
    /// Scale an already solved potential without rerunning the Poisson solve.
    /// Used for gradual power-down of a quasi-static source.
    pub fn scaled(&self, factor: f64) -> Result<Self, BaseError> {
        if !factor.is_finite() || !(0.0..=1.0).contains(&factor) {
            return Err(bad("potential_scale"));
        }
        let mut next = self.clone();
        for value in &mut next.values {
            *value *= factor;
        }
        Ok(next)
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }
    pub fn at(&self, x: usize, y: usize, z: usize) -> Option<f64> {
        self.shape.index(x, y, z).map(|i| self.values[i])
    }
    pub fn shape(&self) -> GridShape {
        self.shape
    }
    pub fn peak_absolute(&self) -> f64 {
        self.values.iter().map(|x| x.abs()).fold(0.0, f64::max)
    }
    /// Positive quadratic energy of the computed static field.
    /// Includes screened mode and internal gradients; grounded boundary
    /// faces contribute extra gradients to the zero-potential exterior.
    pub fn field_energy_joules(&self) -> f64 {
        let volume = self.dx.powi(3);
        let mut energy = self
            .values
            .iter()
            .map(|p| 0.5 * self.screening * p * p * volume)
            .sum::<f64>();
        for z in 0..self.shape.nz() {
            for y in 0..self.shape.ny() {
                for x in 0..self.shape.nx() {
                    let i = self.shape.index(x, y, z).unwrap();
                    for (a, b, c) in [(x + 1, y, z), (x, y + 1, z), (x, y, z + 1)] {
                        if let Some(j) = self.shape.index(a, b, c) {
                            if !matches!(self.local[i], Some(LocalPotentialBoundary::Neumann))
                                && !matches!(self.local[j], Some(LocalPotentialBoundary::Neumann))
                            {
                                energy += 0.5 * self.permittivity
                                    * (self.values[j] - self.values[i]).powi(2) * self.dx;
                            }
                        } else if self.boundary == PotentialBoundary::Grounded {
                            energy += 0.5 * self.permittivity * self.values[i].powi(2) * self.dx;
                        }
                    }
                    if self.boundary == PotentialBoundary::Grounded {
                        if x == 0 {
                            energy += 0.5 * self.permittivity * self.values[i].powi(2) * self.dx;
                        }
                        if y == 0 {
                            energy += 0.5 * self.permittivity * self.values[i].powi(2) * self.dx;
                        }
                        if z == 0 {
                            energy += 0.5 * self.permittivity * self.values[i].powi(2) * self.dx;
                        }
                    }
                }
            }
        }
        // Robin coupling contributes a nonnegative reservoir-interface
        // energy; the reference potential is externally imposed.
        for (i, boundary) in self.local.iter().enumerate() {
            if let Some(LocalPotentialBoundary::Robin { exchange, reference }) = boundary {
                energy += 0.5 * exchange / self.dx
                    * (self.values[i] - reference).powi(2) * volume;
            }
        }
        energy
    }
    pub fn manon_potential_energy(&self, field: &FreeManaField) -> Result<f64, BaseError> {
        self.validate_field(field)?;
        let energy = self
            .values
            .iter()
            .zip(field.densities())
            .map(|(p, n)| p * n.value())
            .sum::<f64>()
            * self.dx.powi(3);
        if !energy.is_finite() {
            return Err(bad("potential_energy"));
        }
        Ok(energy)
    }
    fn validate_field(&self, field: &FreeManaField) -> Result<(), BaseError> {
        if self.shape != field.shape() || self.dx != field.cell_width_metres() {
            Err(bad("potential_grid_mismatch"))
        } else {
            Ok(())
        }
    }
    /// Explicit upwind drift, subdividing only when the potential CFL needs
    /// it. At each internal face exactly the same amount is removed from one
    /// cell and added to its neighbour. Exterior manon faces are impermeable.
    pub fn drift(
        &self,
        field: &mut FreeManaField,
        mobility: f64,
        dt: f64,
    ) -> Result<PotentialStep, BaseError> {
        self.validate_field(field)?;
        if !mobility.is_finite() || mobility < 0.0 || !dt.is_finite() || dt <= 0.0 {
            return Err(bad("potential_drift_parameters"));
        }
        let volume = self.dx.powi(3);
        let mut old: Vec<f64> = field.densities().iter().map(|d| d.value()).collect();
        let before = old.iter().sum::<f64>() * volume;
        let mut faces = Vec::<(usize, usize, f64)>::new();
        let mut outgoing = vec![0.0_f64; old.len()];
        for z in 0..self.shape.nz() {
            for y in 0..self.shape.ny() {
                for x in 0..self.shape.nx() {
                    let i = self.shape.index(x, y, z).unwrap();
                    for (a, b, c) in [(x + 1, y, z), (x, y + 1, z), (x, y, z + 1)] {
                        if let Some(j) = self.shape.index(a, b, c) {
                            let velocity = -mobility * (self.values[j] - self.values[i]) / self.dx;
                            let fraction = velocity * dt / self.dx;
                            if !fraction.is_finite() {
                                return Err(bad("potential_cfl"));
                            }
                            if fraction > 0.0 {
                                outgoing[i] += fraction;
                            }
                            if fraction < 0.0 {
                                outgoing[j] -= fraction;
                            }
                            faces.push((i, j, fraction));
                        }
                    }
                }
            }
        }
        let max_fraction = outgoing.into_iter().fold(0.0_f64, f64::max);
        let substeps = (max_fraction / 0.9).ceil().max(1.0);
        if !substeps.is_finite() || substeps > 256.0 {
            return Err(BaseError::new(
                ErrorCode::NumericalInstability,
                "potential_cfl",
            ));
        }
        let substeps = substeps as usize;
        let before_energy = old
            .iter()
            .zip(&self.values)
            .map(|(n, p)| n * p)
            .sum::<f64>()
            * volume;
        let mut transported = 0.0;
        for _ in 0..substeps {
            let mut next = old.clone();
            for &(i, j, fraction) in &faces {
                let fraction = fraction / substeps as f64;
                let amount = if fraction >= 0.0 {
                    fraction * old[i]
                } else {
                    fraction * old[j]
                };
                next[i] -= amount;
                next[j] += amount;
                transported += amount.abs() * volume;
            }
            if next.iter().any(|v| !v.is_finite() || *v < -1e-10) {
                return Err(BaseError::new(
                    ErrorCode::NumericalInstability,
                    "potential_density",
                ));
            }
            old = next.into_iter().map(|v| v.max(0.0)).collect();
        }
        let after = old.iter().sum::<f64>() * volume;
        if !after.is_finite() || (after - before).abs() > 1e-9 * before.max(1.0) {
            return Err(BaseError::new(
                ErrorCode::NumericalInstability,
                "potential_conservation",
            ));
        }
        let energy_after = old
            .iter()
            .zip(&self.values)
            .map(|(n, p)| n * p)
            .sum::<f64>()
            * volume;
        let next = old
            .into_iter()
            .map(Quantity::<ManonDensity>::nonnegative)
            .collect::<Result<Vec<_>, _>>()?;
        field.replace(next);
        Ok(PotentialStep {
            transported_manons: transported,
            before_manons: before,
            after_manons: after,
            potential_energy_change_joules: energy_after - before_energy,
        })
    }
}

fn axis_index(axis: usize, x: usize, y: usize, z: usize, value: usize) -> Option<[usize; 3]> {
    match axis {
        0 => Some([value, y, z]),
        1 => Some([x, value, z]),
        2 => Some([x, y, value]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GridShape;
    #[test]
    fn negative_source_attracts_manons_and_conserves_count() {
        let shape = GridShape::new(9, 9, 9).unwrap();
        let mut sources = vec![0.0; shape.len()];
        sources[shape.index(4, 4, 4).unwrap()] = -20.0;
        let potential = PotentialField::solve(
            shape,
            1.0,
            &sources,
            1.0,
            1.0,
            PotentialBoundary::Insulated,
            100,
        )
        .unwrap();
        assert!(potential.at(4, 4, 4).unwrap() < potential.at(3, 4, 4).unwrap());
        let mut field =
            FreeManaField::uniform(shape, 1.0, Quantity::nonnegative(10.0).unwrap()).unwrap();
        let before = field.total_manons().unwrap().value();
        for _ in 0..20 {
            potential.drift(&mut field, 0.025, 0.1).unwrap();
        }
        assert!(field.at(4, 4, 4).unwrap().value() > 10.0);
        assert!((field.total_manons().unwrap().value() - before).abs() < 1e-7);
    }
    #[test]
    fn uniform_zero_source_is_motionless() {
        let shape = GridShape::new(5, 5, 5).unwrap();
        let potential = PotentialField::solve(
            shape,
            0.5,
            &vec![0.0; shape.len()],
            1.0,
            1.0,
            PotentialBoundary::Grounded,
            30,
        )
        .unwrap();
        let mut field =
            FreeManaField::uniform(shape, 0.5, Quantity::nonnegative(2.0).unwrap()).unwrap();
        let original = field.total_manons().unwrap().value();
        let step = potential.drift(&mut field, 1.0, 0.1).unwrap();
        assert_eq!(step.transported_manons, 0.0);
        assert_eq!(field.total_manons().unwrap().value(), original);
        assert_eq!(potential.field_energy_joules(), 0.0);
    }
    #[test]
    fn decay_reduces_field_energy_quadratically() {
        let shape = GridShape::new(3, 3, 3).unwrap();
        let mut source = vec![0.0; shape.len()];
        source[shape.index(1, 1, 1).unwrap()] = -4.0;
        let field = PotentialField::solve(
            shape,
            1.0,
            &source,
            1.0,
            1.0,
            PotentialBoundary::Grounded,
            80,
        )
        .unwrap();
        let energy = field.field_energy_joules();
        assert!(energy > 0.0);
        let half = field.scaled(0.5).unwrap();
        assert!((half.field_energy_joules() - energy * 0.25).abs() < 1e-9);
        assert!(field.scaled(-1.0).is_err());
    }

    #[test]
    fn invalid_parameters_rejected_without_mutation() {
        let shape = GridShape::new(3, 3, 3).unwrap();
        assert!(
            PotentialField::solve(shape, 1.0, &[], 1.0, 1.0, PotentialBoundary::Grounded, 20)
                .is_err()
        );
        let p = PotentialField::solve(
            shape,
            1.0,
            &vec![0.0; shape.len()],
            1.0,
            1.0,
            PotentialBoundary::Grounded,
            20,
        )
        .unwrap();
        let mut field =
            FreeManaField::uniform(shape, 1.0, Quantity::nonnegative(1.0).unwrap()).unwrap();
        assert!(p.drift(&mut field, f64::NAN, 0.01).is_err());
        assert_eq!(field.at(1, 1, 1).unwrap().value(), 1.0);
    }
}
