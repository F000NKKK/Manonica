//! Семена задают только начальные условия, а не постоянные источники.
use manonica_core::{Quantity, units::ManonDensity};
use manonica_physics::{FreeManaField, GridShape};
use super::super::experiment::{Experiment, ExperimentError, FieldSeed, SeedShape};

pub fn cell_center(shape: GridShape, dx: f64, x: usize, y: usize, z: usize) -> [f64; 3] {
    [
        (x as f64 + 0.5 - shape.nx() as f64 / 2.0) * dx,
        (y as f64 + 0.5 - shape.ny() as f64 / 2.0) * dx,
        (z as f64 + 0.5 - shape.nz() as f64 / 2.0) * dx,
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn weight(seed: &FieldSeed, p: [f64; 3]) -> f64 {
    let q = [
        p[0] - seed.center[0],
        p[1] - seed.center[1],
        p[2] - seed.center[2],
    ];
    let r2 = dot(q, q);
    let squared_distance = match seed.shape {
        SeedShape::Gaussian => r2,
        SeedShape::Ring => {
            let normal2 = dot(seed.normal, seed.normal);
            let parallel = dot(q, seed.normal) / normal2.sqrt();
            let radial = (r2 - parallel * parallel).max(0.0).sqrt();
            (radial - seed.radius).powi(2) + parallel * parallel
        }
    };
    (-0.5 * squared_distance / (seed.sigma * seed.sigma)).exp()
}

pub fn seeded_field(experiment: &Experiment) -> Result<FreeManaField, ExperimentError> {
    let [nx, ny, nz] = experiment.grid.dimensions;
    let shape = GridShape::new(nx, ny, nz)?;
    let dx = experiment.grid.cell_size;
    let mut accum = vec![0.0; shape.len()];
    let mut weights = vec![0.0; shape.len()];
    for seed in &experiment.seeds {
        let mut sum = 0.0;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let index = shape.index(x, y, z).expect("coordinates within grid");
                    let w = weight(seed, cell_center(shape, dx, x, y, z));
                    weights[index] = w;
                    sum += w;
                }
            }
        }
        if !sum.is_finite() || sum <= 0.0 {
            return Err(manonica_error::BaseError::new(
                manonica_error::ErrorCode::InvalidRange, "seed_outside_grid"
            ).into());
        }
        // В каждой ячейке хранится концентрация; интеграл по объёму
        // должен точно соответствовать заданному total_manons.
        let factor = seed.total_manons / (sum * dx.powi(3));
        for (destination, source) in accum.iter_mut().zip(&weights) {
            *destination += factor * source;
        }
    }
    let zero = Quantity::<ManonDensity>::nonnegative(0.0)?;
    let mut field = FreeManaField::uniform(shape, dx, zero)?;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let index = shape.index(x, y, z).expect("coordinates within grid");
                field.set(x, y, z, Quantity::nonnegative(accum[index])?)?;
            }
        }
    }
    Ok(field)
}
