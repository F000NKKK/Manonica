use super::{
    AmbientFlow, Circle, Diffusion, Experiment, ExperimentError, FieldSeed, Grid, SeedShape,
    Visualization,
};

/// Rust-строитель воспроизводимых экспериментов; эквивалент TOML-схемы.
pub struct ExperimentBuilder {
    experiment: Experiment,
}

impl ExperimentBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            experiment: Experiment {
                name: name.into(),
                grid: Grid {
                    dimensions: [24, 24, 24],
                    cell_size: 0.5,
                },
                physics: Diffusion {
                    diffusivity: 0.25,
                    dt: 0.02,
                    flow: AmbientFlow::Still,
                    steps_per_frame: 1,
                },
                seeds: Vec::new(),
                circles: Vec::new(),
                visualization: Visualization::default(),
            },
        }
    }

    pub fn grid(mut self, dimensions: [usize; 3], cell_size: f64) -> Self {
        self.experiment.grid = Grid {
            dimensions,
            cell_size,
        };
        self
    }

    pub fn diffusion(mut self, diffusivity: f64, dt: f64, steps_per_frame: usize) -> Self {
        self.experiment.physics = Diffusion {
            diffusivity,
            dt,
            flow: self.experiment.physics.flow.clone(),
            steps_per_frame,
        };
        self
    }

    pub fn flow(mut self, flow: AmbientFlow) -> Self {
        self.experiment.physics.flow = flow;
        self
    }

    pub fn seed(mut self, seed: FieldSeed) -> Self {
        self.experiment.seeds.push(seed);
        self
    }

    pub fn gaussian(self, center: [f64; 3], total_manons: f64, sigma: f64) -> Self {
        self.seed(FieldSeed {
            shape: SeedShape::Gaussian,
            center,
            total_manons,
            sigma,
            radius: 0.0,
            normal: [0.0, 1.0, 0.0],
        })
    }

    pub fn ring(self, center: [f64; 3], radius: f64, sigma: f64, total_manons: f64) -> Self {
        self.seed(FieldSeed {
            shape: SeedShape::Ring,
            center,
            radius,
            sigma,
            total_manons,
            normal: [0.0, 1.0, 0.0],
        })
    }

    pub fn circle(mut self, circle: Circle) -> Self {
        self.experiment.circles.push(circle);
        self
    }

    pub fn visualization(mut self, visualization: Visualization) -> Self {
        self.experiment.visualization = visualization;
        self
    }

    pub fn build(self) -> Result<Experiment, ExperimentError> {
        self.experiment.validate()?;
        Ok(self.experiment)
    }
}
