//! Atomic snapshot import and independent host-side validation of device output.
use crate::{
    error::invalid,
    layout::{LEDGER_WORDS, STATE_WORDS, decode},
};
use manonica_core::{Quantity, units::ManonDensity};
use manonica_error::BaseError;
use manonica_simulation::{CoupledState, FieldLedger};

pub(super) fn snapshot(
    previous: &CoupledState,
    bytes: &[u8],
    previous_scale: f64,
) -> Result<(CoupledState, f64), BaseError> {
    let n = previous.field.shape().len();
    let state_size = n * STATE_WORDS * 4;
    if bytes.len() != state_size + LEDGER_WORDS * 4 + 4 {
        return Err(invalid("gpu_coupled_readback_shape"));
    }
    let flag = u32::from_le_bytes(
        bytes[bytes.len() - 4..]
            .try_into()
            .map_err(|_| invalid("gpu_coupled_status"))?,
    );
    if flag != 0 {
        return Err(invalid(match flag {
            1 => "gpu_coupled_diffusion_cfl",
            2 => "gpu_coupled_advection_cfl",
            4 => "gpu_coupled_drift_cfl",
            8 => "gpu_coupled_material",
            16 => "gpu_coupled_thermal",
            32 => "gpu_coupled_power_budget",
            64 => "gpu_coupled_nonfinite",
            _ => "gpu_coupled_conservation",
        }));
    }
    let data = decode(&bytes[..state_size]);
    let ledger = decode(&bytes[state_size..bytes.len() - 4]);
    if data.iter().chain(&ledger).any(|v| !v.is_finite()) {
        return Err(invalid("gpu_coupled_nonfinite"));
    }
    let mut next = previous.clone();
    let volume = next.field.cell_width_metres().powi(3);
    let mut density = Vec::with_capacity(n);
    let mut storage = Vec::with_capacity(n);
    let mut potential = Vec::with_capacity(n);
    for (cell, values) in next
        .materials
        .cells()
        .iter()
        .zip(data.as_chunks::<4>().0.iter())
    {
        if values[..3].iter().any(|v| *v < -1e-5) {
            return Err(invalid("gpu_coupled_negative"));
        }
        let capacity = cell.material.map_or(0.0, |m| m.capacity);
        if values[1] > capacity + capacity.max(1.0) * 1e-5 {
            return Err(invalid("gpu_coupled_saturation"));
        }
        // Correct only representation-level capacity overshoot, transferring
        // the corresponding count and binding energy back to free storage.
        let bound = values[1].max(0.0).min(capacity);
        let excess = (values[1] - bound).max(0.0);
        density.push(Quantity::<ManonDensity>::nonnegative(
            values[0].max(0.0) + excess / volume,
        )?);
        storage.push([
            bound,
            (values[2] - excess * cell.material.map_or(0.0, |m| m.binding_energy)).max(0.0),
        ]);
        potential.push(values[3]);
    }
    next.field.try_replace_densities(density)?;
    next.materials.try_replace_storage(&storage)?;
    if let Some(p) = next.potential.as_mut() {
        let ratio = if previous_scale > 0.0 {
            (ledger[13] / previous_scale).clamp(0.0, 1.0)
        } else {
            1.0
        };
        *p = p.scaled(ratio)?;
        p.try_replace_values(potential)?;
    }
    next.field_ledger = Some(FieldLedger {
        field_energy_joules: ledger[0],
        manon_potential_joules: ledger[1],
        dissipated_heat_joules: ledger[2],
        operating_work_joules: ledger[3],
        last_transport_manons: ledger[4],
        peak_potential: ledger[5],
        exported_heat_joules: ledger[6],
        shutdown: ledger[7] != 0.0,
    });
    next.shutdown = ledger[7] != 0.0;
    let balance = next.materials.balance(&next.field)?;
    let count = balance.manons();
    let energy = balance.energy() + ledger[0] + ledger[1] + ledger[2] + ledger[6] - ledger[3];
    let scale = (balance.energy().abs()
        + ledger[..4].iter().map(|v| v.abs()).sum::<f64>()
        + ledger[6].abs())
    .max(1.0);
    if (count - ledger[9]).abs() > ledger[9].max(1.0) * 5e-4
        || (energy - ledger[10]).abs() > scale * 0.002
    {
        return Err(invalid("gpu_coupled_host_audit"));
    }
    Ok((next, ledger[13]))
}
