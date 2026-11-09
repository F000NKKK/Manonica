//! Checked f64→f32 encoding of immutable material/geometry configuration.
use crate::{error::invalid, layout::floats};
use manonica_error::BaseError;
use manonica_physics::{LocalPotentialBoundary, PotentialBoundary};
use manonica_simulation::{CoupledState, MediumFlow, StepParameters};

pub(super) struct Upload {
    pub cells: Vec<u8>,
    pub materials: Vec<u8>,
    pub channels: Vec<u8>,
    pub parameters: Vec<u8>,
    pub local: Vec<u8>,
    pub ledger: Vec<u8>,
}
impl Upload {
    pub fn new(state: &CoupledState, parameters: &StepParameters) -> Result<Self, BaseError> {
        parameters.validate()?;
        let shape = state.field.shape();
        if state.materials.shape() != shape
            || state.potential.is_some() != parameters.field.is_some()
        {
            return Err(invalid("gpu_coupled_shape"));
        }
        let n = shape.len();
        let phi = state.potential.as_ref().map(|p| p.values());
        let mut cells = Vec::with_capacity(n * 16);
        let mut materials = Vec::with_capacity(n * 40);
        for (i, (density, cell)) in state
            .field
            .densities()
            .iter()
            .zip(state.materials.cells())
            .enumerate()
        {
            cells.extend(floats([
                density.value(),
                cell.bound,
                cell.heat,
                phi.map_or(0.0, |p| p[i]),
            ])?);
            let data = if let Some(m) = cell.material {
                m.validate()?;
                [
                    1.0,
                    m.capacity,
                    m.capture_rate,
                    m.release_rate,
                    m.binding_energy,
                    m.heat_capacity,
                    m.reference_temperature,
                    m.activation_temperature,
                    m.max_flux,
                    0.0,
                ]
            } else {
                [0.0; 10]
            };
            materials.extend(floats(data)?);
        }
        let mut channels = Vec::new();
        for c in state.materials.channels() {
            let a = shape
                .index(c.from[0], c.from[1], c.from[2])
                .ok_or(invalid("gpu_channel_from"))?;
            let b = shape
                .index(c.to[0], c.to[1], c.to[2])
                .ok_or(invalid("gpu_channel_to"))?;
            channels.extend_from_slice(&(a as u32).to_le_bytes());
            channels.extend_from_slice(&(b as u32).to_le_bytes());
            channels.extend(floats([c.length, c.area, c.conductance, c.max_flux])?);
        }
        if channels.is_empty() {
            channels.resize(24, 0);
        }
        let mut local = vec![[0.0; 4]; n];
        if let Some(p) = &state.potential {
            if p.shape() != shape || p.cell_width_metres() != state.field.cell_width_metres() {
                return Err(invalid("gpu_potential_shape"));
            }
            for site in p.local_boundaries() {
                local[site.index] = match site.condition {
                    LocalPotentialBoundary::Dirichlet { value } => [1.0, value, 0.0, 0.0],
                    LocalPotentialBoundary::Neumann => [2.0, 0.0, 0.0, 0.0],
                    LocalPotentialBoundary::Robin {
                        reference,
                        exchange,
                    } => [3.0, reference, exchange, 0.0],
                };
            }
        }
        let mut params = vec![
            shape.nx() as f64,
            shape.ny() as f64,
            shape.nz() as f64,
            state.field.cell_width_metres(),
            parameters.diffusivity,
            parameters.dt,
            parameters.thermal_conductance,
            parameters.ambient_cooling_rate,
        ];
        if let Some(f) = parameters.field {
            params.extend([
                1.0,
                f.mobility,
                f.max_power_joules_per_second,
                f.shutdown_at_bound_fraction,
                f.shutdown_decay_rate,
            ]);
        } else {
            params.extend([0.0; 5]);
        }
        match parameters.flow {
            MediumFlow::Still => params.extend([0.0; 9]),
            MediumFlow::Uniform { velocity } => {
                params.push(1.0);
                params.extend(velocity);
                params.extend([0.0; 5]);
            }
            MediumFlow::Vortex {
                origin,
                axis,
                angular_speed,
                axial_speed,
            } => {
                let length = axis.iter().map(|v| v * v).sum::<f64>().sqrt();
                if !length.is_finite() || length <= 0.0 {
                    return Err(invalid("gpu_vortex_axis"));
                }
                params.push(2.0);
                params.extend(origin);
                params.extend(axis);
                params.extend([angular_speed, axial_speed]);
            }
        }
        if let Some(p) = &state.potential {
            params.extend([
                p.permittivity(),
                p.screening(),
                f64::from(p.boundary() == PotentialBoundary::Grounded),
            ]);
        } else {
            params.extend([0.0; 3]);
        }
        params.push(state.materials.channels().len() as f64);
        let l = state.field_ledger.unwrap_or_default();
        let ledger = floats([
            l.field_energy_joules,
            l.manon_potential_joules,
            l.dissipated_heat_joules,
            l.operating_work_joules,
            l.last_transport_manons,
            l.peak_potential,
            l.exported_heat_joules,
            f64::from(state.shutdown),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        ])?;
        Ok(Self {
            cells,
            materials,
            channels,
            parameters: floats(params)?,
            local: floats(local.into_iter().flatten())?,
            ledger,
        })
    }
}
