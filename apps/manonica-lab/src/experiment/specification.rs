use std::path::Path;
use serde::Deserialize;
use manonica_error::{BaseError, ErrorCode};

use super::ExperimentError;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Experiment {
    pub name: String,
    pub grid: Grid,
    pub physics: Diffusion,
    #[serde(default)]
    pub seeds: Vec<FieldSeed>,
    #[serde(default)]
    pub circles: Vec<Circle>,
    #[serde(default)]
    pub visualization: Visualization,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Grid {
    pub dimensions: [usize; 3],
    /// Сторона кубической ячейки в метрах.
    pub cell_size: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Diffusion {
    /// Эталонный начальный закон J = -D grad(n).
    pub diffusivity: f64,
    pub dt: f64,
    #[serde(default)]
    pub flow: AmbientFlow,
    #[serde(default = "one")]
    pub steps_per_frame: usize,
}

/// Prescribed movement of the surrounding medium (not a magic-circle law).
/// A rigid vortex has v = angular_speed * (axis × (p - origin)) + axial_speed * axis.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AmbientFlow {
    Still,
    Uniform {
        /// Metres per second in world X/Y/Z.
        velocity: [f64; 3],
    },
    Vortex {
        /// Axis anchor, in metres.
        origin: [f64; 3],
        /// Arbitrary nonzero axis; normalized when sampled.
        axis: [f64; 3],
        /// Radians per second.
        angular_speed: f64,
        /// Constant motion along the normalized axis (m/s).
        #[serde(default)]
        axial_speed: f64,
    },
}

impl Default for AmbientFlow {
    fn default() -> Self {
        Self::Still
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldSeed {
    pub shape: SeedShape,
    /// Центр в метрах относительно центра расчётной области.
    pub center: [f64; 3],
    /// Интегральный запас манонов: распределяется по ячейкам с нормировкой.
    pub total_manons: f64,
    /// Ширина Гаусса или толщина кольца (метры).
    pub sigma: f64,
    /// Радиус для shape = ring (метры).
    #[serde(default)]
    pub radius: f64,
    #[serde(default = "default_normal")]
    pub normal: [f64; 3],
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeedShape {
    Gaussian,
    Ring,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Circle {
    pub name: String,
    pub center: [f32; 3],
    pub normal: [f32; 3],
    pub radius: f32,
    pub thickness: f32,
    #[serde(default)]
    pub spokes: usize,
    #[serde(default = "default_segments")]
    pub segments: usize,
    #[serde(default = "default_color")]
    pub color: [f32; 3],
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Visualization {
    #[serde(default = "default_threshold")]
    pub relative_threshold: f64,
    #[serde(default = "default_max_points")]
    pub max_points: usize,
    #[serde(default = "default_exposure")]
    pub exposure: f32,
    #[serde(default = "default_wisps")]
    pub wisps: bool,
}

impl Default for Visualization {
    fn default() -> Self {
        Self {
            relative_threshold: default_threshold(),
            max_points: default_max_points(),
            exposure: default_exposure(),
            wisps: default_wisps(),
        }
    }
}

fn one() -> usize { 1 }
fn default_threshold() -> f64 { 0.06 }
fn default_max_points() -> usize { 800 }
fn default_exposure() -> f32 { 2.0 }
fn default_wisps() -> bool { true }
fn default_segments() -> usize { 96 }
fn default_color() -> [f32; 3] { [0.35, 0.7, 1.0] }
fn default_normal() -> [f64; 3] { [0.0, 1.0, 0.0] }

fn check(test: bool, parameter: &'static str) -> Result<(), ExperimentError> {
    if test { Ok(()) } else {
        Err(BaseError::new(ErrorCode::InvalidRange, parameter).into())
    }
}

fn good_position(v: [f64; 3]) -> bool {
    v.iter().all(|x| x.is_finite())
}

impl Experiment {
    pub fn parse(source: &str) -> Result<Self, ExperimentError> {
        let model: Self = toml::from_str(source)?;
        model.validate()?;
        Ok(model)
    }

    pub fn from_file(path: &Path) -> Result<Self, ExperimentError> {
        Self::parse(&std::fs::read_to_string(path)?)
    }

    pub fn validate(&self) -> Result<(), ExperimentError> {
        check(!self.name.trim().is_empty(), "experiment_name")?;
        let [nx, ny, nz] = self.grid.dimensions;
        let volume = nx.checked_mul(ny).and_then(|v| v.checked_mul(nz));
        check(volume.is_some_and(|n| n > 0 && n <= 128 * 128 * 128), "grid_dimensions")?;
        let dx = self.grid.cell_size;
        check(dx.is_finite() && dx > 0.0 && dx.powi(3).is_finite(), "cell_size")?;
        let p = &self.physics;
        check(p.diffusivity.is_finite() && p.diffusivity >= 0.0, "diffusivity")?;
        check(p.dt.is_finite() && p.dt > 0.0, "dt")?;
        check((1..=16).contains(&p.steps_per_frame), "steps_per_frame")?;
        match &p.flow {
            AmbientFlow::Still => {}
            AmbientFlow::Uniform { velocity } => {
                check(good_position(*velocity), "flow_velocity")?;
            }
            AmbientFlow::Vortex { origin, axis, angular_speed, axial_speed } => {
                check(good_position(*origin), "flow_origin")?;
                check(good_position(*axis), "flow_axis")?;
                let norm_squared: f64 = axis.iter().map(|v| v * v).sum();
                check(norm_squared.is_finite() && norm_squared > 1e-24, "flow_axis")?;
                check(angular_speed.is_finite(), "flow_angular_speed")?;
                check(axial_speed.is_finite(), "flow_axial_speed")?;
            }
        }
        check(
            (p.diffusivity * p.dt / dx.powi(2)).is_finite()
                && p.diffusivity * p.dt / dx.powi(2) <= 1.0 / 6.0,
            "diffusion_stability",
        )?;
        check(self.seeds.len() <= 128, "seeds_count")?;
        for seed in &self.seeds {
            check(good_position(seed.center), "seed_center")?;
            check(seed.total_manons.is_finite() && seed.total_manons > 0.0, "seed_total")?;
            check(seed.sigma.is_finite() && seed.sigma > 0.0, "seed_sigma")?;
            check(seed.radius.is_finite() && seed.radius >= 0.0, "seed_radius")?;
            if matches!(seed.shape, SeedShape::Ring) {
                check(seed.radius > 0.0, "ring_radius")?;
                let sq: f64 = seed.normal.iter().map(|v| v * v).sum();
                check(good_position(seed.normal) && sq > 1e-12, "seed_normal")?;
            }
        }
        check(self.circles.len() <= 64, "circles_count")?;
        for circle in &self.circles {
            check(!circle.name.trim().is_empty(), "circle_name")?;
            check(circle.center.iter().all(|n| n.is_finite()), "circle_center")?;
            let n2: f32 = circle.normal.iter().map(|n| n * n).sum();
            check(n2.is_finite() && n2 > 1e-12, "circle_normal")?;
            check(circle.radius.is_finite() && circle.radius > 0.0, "circle_radius")?;
            check(
                circle.thickness.is_finite() && circle.thickness > 0.0 && circle.thickness < circle.radius,
                "circle_thickness",
            )?;
            check((8..=256).contains(&circle.segments), "circle_segments")?;
            check(circle.spokes <= 64, "circle_spokes")?;
            check(circle.color.iter().all(|n| n.is_finite() && *n >= 0.0), "circle_color")?;
        }
        let v = &self.visualization;
        check(v.relative_threshold.is_finite() && (0.0..1.0).contains(&v.relative_threshold), "threshold")?;
        check((1..=4096).contains(&v.max_points), "max_points")?;
        check(v.exposure.is_finite() && v.exposure > 0.0 && v.exposure <= 30.0, "exposure")?;
        Ok(())
    }
}
