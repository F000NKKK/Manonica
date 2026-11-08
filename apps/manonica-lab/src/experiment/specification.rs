use manonica_error::{BaseError, ErrorCode};
use serde::Deserialize;
use std::path::Path;

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
    pub formulas: Vec<Formula>,
    #[serde(default)]
    pub field_sources: Vec<FieldSource>,
    #[serde(default)]
    pub materials: Vec<MaterialSite>,
    #[serde(default)]
    pub channels: Vec<ChannelSpec>,
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
    /// Nearest-neighbour thermal conductance (J / (s · K)).
    #[serde(default)]
    pub thermal_conductance: f64,
    /// None keeps legacy experiments physically unchanged.
    #[serde(default)]
    pub potential: Option<PotentialConfig>,
    pub dt: f64,
    #[serde(default)]
    pub flow: AmbientFlow,
    #[serde(default = "one")]
    pub steps_per_frame: usize,
}

/// Screened quasi-static potential: (-permittivity Laplacian + screening) phi = source.
/// Sources are generated from the same parametric geometry as the visible meshes.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PotentialConfig {
    /// Velocity response in m²/(J · s) when phi is J/manon.
    pub mobility: f64,
    pub permittivity: f64,
    pub screening: f64,
    #[serde(default = "default_potential_iterations")]
    pub iterations: usize,
    #[serde(default)]
    pub boundary: PotentialBoundaryKind,
    /// External energy supplied to establish the initial field (J).
    pub initial_work_joules: f64,
    /// Maximum external work rate available to maintain the model (J/s).
    pub max_power_joules_per_second: f64,
}
#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PotentialBoundaryKind {
    #[default]
    Grounded,
    Insulated,
}
impl PotentialBoundaryKind {
    pub fn physics(self) -> manonica_physics::PotentialBoundary {
        match self {
            Self::Grounded => manonica_physics::PotentialBoundary::Grounded,
            Self::Insulated => manonica_physics::PotentialBoundary::Insulated,
        }
    }
}
/// Integral source strength distributed along the evaluated geometry.
/// Negative strengths attract free manons; positive strengths repel them.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldSource {
    pub formula: usize,
    pub strength: f64,
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

/// Parametric circuit defined by three mathematical functions of t.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Formula {
    pub name: String,
    pub x: String,
    pub y: String,
    pub z: String,
    #[serde(default = "default_segments")]
    pub segments: usize,
    pub thickness: f32,
    #[serde(default = "default_color")]
    pub color: [f32; 3],
    #[serde(default)]
    pub trap: Option<FormulaTrap>,
}

/// Each unique grid cell touched by a circuit receives an equal share of
/// its total trap capacity and maximum capture flux.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FormulaTrap {
    pub capacity: f64,
    pub capture_rate: f64,
    pub release_rate: f64,
    pub binding_energy: f64,
    pub heat_capacity: f64,
    pub reference_temperature: f64,
    pub activation_temperature: f64,
    pub max_flux: f64,
}
impl FormulaTrap {
    pub fn material(&self, cells: usize) -> manonica_physics::TrapMaterial {
        let share = 1.0 / cells as f64;
        manonica_physics::TrapMaterial {
            capacity: self.capacity * share,
            capture_rate: self.capture_rate,
            release_rate: self.release_rate,
            binding_energy: self.binding_energy,
            heat_capacity: self.heat_capacity * share,
            reference_temperature: self.reference_temperature,
            activation_temperature: self.activation_temperature,
            max_flux: self.max_flux * share,
            conductivity: 0.0,
        }
    }
}

/// Physical trap placed in a numerical grid cell; no decorative effects.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialSite {
    pub cell: [usize; 3],
    pub capacity: f64,
    pub capture_rate: f64,
    pub release_rate: f64,
    pub binding_energy: f64,
    pub heat_capacity: f64,
    pub reference_temperature: f64,
    pub activation_temperature: f64,
    pub max_flux: f64,
    pub conductivity: f64,
}
impl MaterialSite {
    pub fn physics(&self) -> manonica_physics::TrapMaterial {
        manonica_physics::TrapMaterial {
            capacity: self.capacity,
            capture_rate: self.capture_rate,
            release_rate: self.release_rate,
            binding_energy: self.binding_energy,
            heat_capacity: self.heat_capacity,
            reference_temperature: self.reference_temperature,
            activation_temperature: self.activation_temperature,
            max_flux: self.max_flux,
            conductivity: self.conductivity,
        }
    }
}
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChannelSpec {
    pub from: [usize; 3],
    pub to: [usize; 3],
    pub length: f64,
    pub area: f64,
    pub conductance: f64,
    pub max_flux: f64,
}
impl ChannelSpec {
    pub fn physics(&self) -> manonica_physics::ManaChannel {
        manonica_physics::ManaChannel {
            from: self.from,
            to: self.to,
            length: self.length,
            area: self.area,
            conductance: self.conductance,
            max_flux: self.max_flux,
        }
    }
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
    /// Optical extinction coefficient (1/m) used only for visualization.
    /// Affects rendered opacity, not the number of manons or transport.
    #[serde(default = "default_optical_density")]
    pub optical_density: f64,
    #[serde(default = "default_wisps")]
    pub wisps: bool,
}

impl Default for Visualization {
    fn default() -> Self {
        Self {
            relative_threshold: default_threshold(),
            max_points: default_max_points(),
            exposure: default_exposure(),
            optical_density: default_optical_density(),
            wisps: default_wisps(),
        }
    }
}

fn default_potential_iterations() -> usize {
    120
}
fn one() -> usize {
    1
}
fn default_threshold() -> f64 {
    0.06
}
fn default_max_points() -> usize {
    800
}
fn default_exposure() -> f32 {
    2.0
}
fn default_optical_density() -> f64 {
    0.18
}
fn default_wisps() -> bool {
    true
}
fn default_segments() -> usize {
    96
}
fn default_color() -> [f32; 3] {
    [0.35, 0.7, 1.0]
}
fn default_normal() -> [f64; 3] {
    [0.0, 1.0, 0.0]
}

fn check(test: bool, parameter: &'static str) -> Result<(), ExperimentError> {
    if test {
        Ok(())
    } else {
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
        check(
            volume.is_some_and(|n| n > 0 && n <= 128 * 128 * 128),
            "grid_dimensions",
        )?;
        let dx = self.grid.cell_size;
        check(
            dx.is_finite() && dx > 0.0 && dx.powi(3).is_finite(),
            "cell_size",
        )?;
        let p = &self.physics;
        check(
            p.diffusivity.is_finite() && p.diffusivity >= 0.0,
            "diffusivity",
        )?;
        check(
            p.thermal_conductance.is_finite() && p.thermal_conductance >= 0.0,
            "thermal_conductance",
        )?;
        if let Some(potential) = &p.potential {
            check(
                potential.mobility.is_finite() && potential.mobility >= 0.0
                    && potential.permittivity.is_finite() && potential.permittivity > 0.0
                    && potential.screening.is_finite() && potential.screening > 0.0,
                "potential_parameters",
            )?;
            check((1..=4096).contains(&potential.iterations), "potential_iterations")?;
            check(
                potential.initial_work_joules.is_finite()
                    && potential.initial_work_joules >= 0.0
                    && potential.max_power_joules_per_second.is_finite()
                    && potential.max_power_joules_per_second >= 0.0,
                "potential_work_budget",
            )?;
        }
        check(p.dt.is_finite() && p.dt > 0.0, "dt")?;
        check((1..=16).contains(&p.steps_per_frame), "steps_per_frame")?;
        match &p.flow {
            AmbientFlow::Still => {}
            AmbientFlow::Uniform { velocity } => {
                check(good_position(*velocity), "flow_velocity")?;
            }
            AmbientFlow::Vortex {
                origin,
                axis,
                angular_speed,
                axial_speed,
            } => {
                check(good_position(*origin), "flow_origin")?;
                check(good_position(*axis), "flow_axis")?;
                let norm_squared: f64 = axis.iter().map(|v| v * v).sum();
                check(
                    norm_squared.is_finite() && norm_squared > 1e-24,
                    "flow_axis",
                )?;
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
            check(
                seed.total_manons.is_finite() && seed.total_manons > 0.0,
                "seed_total",
            )?;
            check(seed.sigma.is_finite() && seed.sigma > 0.0, "seed_sigma")?;
            check(seed.radius.is_finite() && seed.radius >= 0.0, "seed_radius")?;
            if matches!(seed.shape, SeedShape::Ring) {
                check(seed.radius > 0.0, "ring_radius")?;
                let sq: f64 = seed.normal.iter().map(|v| v * v).sum();
                check(good_position(seed.normal) && sq > 1e-12, "seed_normal")?;
            }
        }
        check(self.formulas.len() <= 32, "formulas_count")?;
        check(self.field_sources.len() <= 64, "field_sources_count")?;
        if !self.field_sources.is_empty() {
            check(p.potential.is_some(), "potential_required_for_sources")?;
        }
        for source in &self.field_sources {
            check(source.formula < self.formulas.len(), "field_source_formula")?;
            check(source.strength.is_finite(), "field_source_strength")?;
        }
        for formula in &self.formulas {
            check(!formula.name.trim().is_empty(), "formula_name")?;
            check(
                formula.thickness.is_finite() && formula.thickness > 0.0,
                "formula_thickness",
            )?;
            check(
                formula.color.iter().all(|c| c.is_finite() && *c >= 0.0),
                "formula_color",
            )?;
            super::sample_curve(formula)?;
            if let Some(trap) = &formula.trap {
                trap.material(1).validate()?;
            }
        }
        check(
            self.materials.len() <= volume.unwrap_or(0),
            "materials_count",
        )?;
        check(self.channels.len() <= 4096, "channels_count")?;
        let mut occupied = std::collections::HashSet::new();
        for site in &self.materials {
            check(
                site.cell[0] < nx && site.cell[1] < ny && site.cell[2] < nz,
                "material_cell",
            )?;
            check(occupied.insert(site.cell), "duplicate_material_cell")?;
            site.physics().validate()?;
        }
        let shape = manonica_physics::GridShape::new(nx, ny, nz)?;
        let mut lattice = manonica_physics::MaterialLattice::new(shape);
        for channel in &self.channels {
            lattice.add_channel(channel.physics())?;
        }
        let v = &self.visualization;
        check(
            v.relative_threshold.is_finite() && (0.0..1.0).contains(&v.relative_threshold),
            "threshold",
        )?;
        check((1..=4096).contains(&v.max_points), "max_points")?;
        check(
            v.exposure.is_finite() && v.exposure > 0.0 && v.exposure <= 30.0,
            "exposure",
        )?;
        check(
            v.optical_density.is_finite() && (0.01..=4.0).contains(&v.optical_density),
            "optical_density",
        )?;
        Ok(())
    }
}
