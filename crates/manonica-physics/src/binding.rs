//! Conservative free/bound manon exchange in material traps.
//! A minimal discrete model for specification v0.1 §§3, 6-8, 10.
//! Units: count (manons), energy (J), temperature (K), time (s).
use crate::{FreeManaField, GridShape};
use manonica_core::{Quantity, units::ManonDensity};
use manonica_error::{BaseError, ErrorCode};

fn invalid(name: &'static str) -> BaseError {
    BaseError::new(ErrorCode::InvalidRange, name)
}
fn nonnegative(value: f64, name: &'static str) -> Result<f64, BaseError> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(invalid(name))
    }
}
fn positive(value: f64, name: &'static str) -> Result<f64, BaseError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(invalid(name))
    }
}

/// Material parameters, not universal physical constants.
#[derive(Debug, Clone, Copy)]
pub struct TrapMaterial {
    pub capacity: f64,               // manons per cell
    pub capture_rate: f64,           // m³ / (manon·s), multiplied by density
    pub release_rate: f64,           // 1/s at reference temperature
    pub binding_energy: f64,         // J/manon, released as heat on capture
    pub heat_capacity: f64,          // J/K per cell
    pub reference_temperature: f64,  // K
    pub activation_temperature: f64, // K, Arrhenius-type release suppression
    pub max_flux: f64,               // manons/s, independent of capacity
    pub conductivity: f64,           // manons/(s·potential), passive channel
}
impl TrapMaterial {
    pub fn validate(&self) -> Result<(), BaseError> {
        nonnegative(self.capacity, "trap_capacity")?;
        nonnegative(self.capture_rate, "capture_rate")?;
        nonnegative(self.release_rate, "release_rate")?;
        nonnegative(self.binding_energy, "binding_energy")?;
        positive(self.heat_capacity, "heat_capacity")?;
        positive(self.reference_temperature, "reference_temperature")?;
        nonnegative(self.activation_temperature, "activation_temperature")?;
        nonnegative(self.max_flux, "max_flux")?;
        nonnegative(self.conductivity, "conductivity")?;
        Ok(())
    }
}

/// The empty reservoir has no active traps; free diffusion stays unchanged.
#[derive(Debug, Clone)]
pub struct TrapCell {
    pub material: Option<TrapMaterial>,
    pub bound: f64,
    /// Thermal reservoir above a fixed reference baseline in joules.
    pub heat: f64,
}
impl Default for TrapCell {
    fn default() -> Self {
        Self {
            material: None,
            bound: 0.0,
            heat: 0.0,
        }
    }
}
impl TrapCell {
    pub fn temperature(&self) -> Option<f64> {
        self.material
            .map(|m| m.reference_temperature + self.heat / m.heat_capacity)
    }
}

/// An explicit physical channel between two distinct grid cells.
#[derive(Debug, Clone, Copy)]
pub struct ManaChannel {
    pub from: [usize; 3],
    pub to: [usize; 3],
    pub length: f64,      // m
    pub area: f64,        // m²
    pub conductance: f64, // manons/(s·density) per metre of area
    pub max_flux: f64,    // manons/s
}
impl ManaChannel {
    fn validate(&self, shape: GridShape) -> Result<(), BaseError> {
        if self.from == self.to
            || shape
                .index(self.from[0], self.from[1], self.from[2])
                .is_none()
            || shape.index(self.to[0], self.to[1], self.to[2]).is_none()
        {
            return Err(invalid("channel_coordinates"));
        }
        positive(self.length, "channel_length")?;
        positive(self.area, "channel_area")?;
        nonnegative(self.conductance, "channel_conductance")?;
        nonnegative(self.max_flux, "channel_max_flux")?;
        Ok(())
    }
}

/// External work and heat spent to release bound manons.
#[derive(Debug, Clone, Copy)]
pub struct DischargeReport {
    pub released: f64,
    pub work_used_joules: f64,
    pub heat_used_joules: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Balance {
    pub free: f64,
    pub bound: f64,
    pub thermal_joules: f64,
    /// Energy of all manons relative to free unbound state (negative for bound).
    pub binding_joules: f64,
}
impl Balance {
    pub fn manons(self) -> f64 {
        self.free + self.bound
    }
    pub fn energy(self) -> f64 {
        self.thermal_joules + self.binding_joules
    }
}

/// Explicitly stores only material sites; free mana still belongs to FreeManaField.
#[derive(Debug, Clone)]
pub struct MaterialLattice {
    shape: GridShape,
    cells: Vec<TrapCell>,
    channels: Vec<ManaChannel>,
}
impl MaterialLattice {
    pub fn new(shape: GridShape) -> Self {
        Self {
            shape,
            cells: vec![TrapCell::default(); shape.len()],
            channels: Vec::new(),
        }
    }
    pub fn at(&self, x: usize, y: usize, z: usize) -> Option<&TrapCell> {
        self.shape.index(x, y, z).map(|i| &self.cells[i])
    }
    pub fn install(&mut self, at: [usize; 3], material: TrapMaterial) -> Result<(), BaseError> {
        material.validate()?;
        let index = self
            .shape
            .index(at[0], at[1], at[2])
            .ok_or(invalid("trap_coordinates"))?;
        if self.cells[index].bound > material.capacity {
            return Err(invalid("capacity_below_charge"));
        }
        if self.cells[index].bound > 0.0 {
            let old = self.cells[index]
                .material
                .ok_or(invalid("charged_trap_without_material"))?;
            if old.binding_energy != material.binding_energy {
                return Err(invalid("charged_trap_binding_energy_change"));
            }
        }
        self.cells[index].material = Some(material);
        Ok(())
    }
    /// Controlled crystal discharge, supplying explicit external work.
    /// Work is consumed only when the cell's heat cannot pay the binding
    /// energy. No manons are created; the operation is transactional.
    pub fn release_with_work(
        &mut self,
        field: &mut FreeManaField,
        at: [usize; 3],
        requested: f64,
        work_budget_joules: f64,
        dt: f64,
    ) -> Result<DischargeReport, BaseError> {
        nonnegative(requested, "requested_release")?;
        nonnegative(work_budget_joules, "discharge_work_budget")?;
        positive(dt, "discharge_dt")?;
        if field.shape() != self.shape {
            return Err(invalid("material_grid_shape"));
        }
        let i = self
            .shape
            .index(at[0], at[1], at[2])
            .ok_or(invalid("discharge_coordinates"))?;
        let cell = &self.cells[i];
        let material = cell.material.ok_or(invalid("missing_discharge_trap"))?;
        material.validate()?;
        nonnegative(cell.bound, "bound_manons")?;
        nonnegative(cell.heat, "stored_heat")?;
        if cell.bound > material.capacity {
            return Err(invalid("trap_saturation"));
        }
        let budget = cell.heat + work_budget_joules;
        if !budget.is_finite() {
            return Err(invalid("discharge_energy_overflow"));
        }
        let affordable = if material.binding_energy > 0.0 {
            budget / material.binding_energy
        } else {
            cell.bound
        };
        let released = requested
            .min(cell.bound)
            .min(material.max_flux * dt)
            .min(affordable);
        let needed = released * material.binding_energy;
        let heat_used = needed.min(cell.heat);
        let work_used = needed - heat_used;
        let volume = field.cell_width_metres().powi(3);
        let free = field.densities()[i].value() * volume;
        let new_density = Quantity::<ManonDensity>::nonnegative((free + released) / volume)?;
        // Validate all prospective scalars before committing either state.
        nonnegative(cell.bound - released, "discharge_bound")?;
        nonnegative(cell.heat - heat_used, "discharge_heat")?;
        if !work_used.is_finite() || work_used > work_budget_joules {
            return Err(invalid("discharge_work_budget"));
        }
        field.replace_density_at(i, new_density);
        self.cells[i].bound -= released;
        self.cells[i].heat -= heat_used;
        Ok(DischargeReport {
            released,
            work_used_joules: work_used,
            heat_used_joules: heat_used,
        })
    }

    pub fn add_channel(&mut self, channel: ManaChannel) -> Result<(), BaseError> {
        channel.validate(self.shape)?;
        self.channels.push(channel);
        Ok(())
    }
    pub fn balance(&self, field: &FreeManaField) -> Result<Balance, BaseError> {
        if field.shape() != self.shape {
            return Err(invalid("material_grid_shape"));
        }
        Ok(Balance {
            free: field.total_manons()?.value(),
            bound: self.cells.iter().map(|c| c.bound).sum(),
            thermal_joules: self.cells.iter().map(|c| c.heat).sum(),
            binding_joules: self
                .cells
                .iter()
                .map(|c| -c.bound * c.material.map_or(0.0, |m| m.binding_energy))
                .sum(),
        })
    }
    /// Transactional step: all validation and transfers run on copies.
    /// No externally supplied source of mana or energy is assumed.
    pub fn step(&mut self, field: &mut FreeManaField, dt: f64) -> Result<Balance, BaseError> {
        positive(dt, "material_dt")?;
        if field.shape() != self.shape {
            return Err(invalid("material_grid_shape"));
        }
        let before = self.balance(field)?;
        let mut next = field.clone();
        let mut cells = self.cells.clone();
        let dx = field.cell_width_metres();
        let volume = dx.powi(3);
        for (i, cell) in cells.iter_mut().enumerate() {
            let Some(m) = cell.material else { continue };
            m.validate()?;
            nonnegative(cell.heat, "trap_heat")?;
            nonnegative(cell.bound, "trap_bound")?;
            if cell.bound > m.capacity {
                return Err(invalid("trap_saturation"));
            }
            let free = next.densities()[i].value() * volume;
            let temperature = m.reference_temperature + cell.heat / m.heat_capacity;
            let activation = (-m.activation_temperature
                * (1.0 / temperature - 1.0 / m.reference_temperature).max(0.0))
            .exp();
            let capture_fraction = 1.0 - (-m.capture_rate * free / volume * dt).exp();
            let capture = ((m.capacity - cell.bound) * capture_fraction)
                .min(free)
                .min(m.max_flux * dt);
            // Releasing a bound manon raises its reference energy by E_bind.
            // Without enough heat, no release is possible in this isolated cell.
            let release_fraction = 1.0 - (-m.release_rate * activation * dt).exp();
            let release = (cell.bound * release_fraction).min(m.max_flux * dt).min(
                if m.binding_energy > 0.0 {
                    (cell.heat + capture * m.binding_energy) / m.binding_energy
                } else {
                    cell.bound
                },
            );
            let final_free = free - capture + release;
            cell.bound += capture - release;
            cell.heat += (capture - release) * m.binding_energy;
            cell.heat = cell.heat.max(0.0);
            next.replace_density_at(
                i,
                Quantity::<ManonDensity>::nonnegative(final_free / volume)?,
            );
        }
        // Passive geometry-dependent transport. Working on the candidate
        // field means channels cannot transport more than actually available.
        for channel in &self.channels {
            channel.validate(self.shape)?;
            let a = self
                .shape
                .index(channel.from[0], channel.from[1], channel.from[2])
                .unwrap();
            let b = self
                .shape
                .index(channel.to[0], channel.to[1], channel.to[2])
                .unwrap();
            let fa = next.densities()[a].value() * volume;
            let fb = next.densities()[b].value() * volume;
            let difference = (fa - fb) / volume;
            let proposed = channel.conductance * channel.area / channel.length * difference * dt;
            let amount = proposed
                .abs()
                .min(channel.max_flux * dt)
                .min(if proposed >= 0.0 { fa } else { fb })
                .min((fa - fb).abs() * 0.5);
            let signed = proposed.signum() * amount;
            next.replace_density_at(
                a,
                Quantity::<ManonDensity>::nonnegative((fa - signed) / volume)?,
            );
            next.replace_density_at(
                b,
                Quantity::<ManonDensity>::nonnegative((fb + signed) / volume)?,
            );
        }
        let after = Balance {
            free: next.total_manons()?.value(),
            bound: cells.iter().map(|c| c.bound).sum(),
            thermal_joules: cells.iter().map(|c| c.heat).sum(),
            binding_joules: cells
                .iter()
                .map(|c| -c.bound * c.material.map_or(0.0, |m| m.binding_energy))
                .sum(),
        };
        let scale = before.manons().abs().max(1.0);
        if (after.manons() - before.manons()).abs() > scale * 1e-9
            || (after.energy() - before.energy()).abs() > before.energy().abs().max(1.0) * 1e-9
        {
            return Err(invalid("material_conservation"));
        }
        *field = next;
        self.cells = cells;
        Ok(after)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn material() -> TrapMaterial {
        TrapMaterial {
            capacity: 10.0,
            capture_rate: 3.0,
            release_rate: 0.2,
            binding_energy: 2.0,
            heat_capacity: 5.0,
            reference_temperature: 300.0,
            activation_temperature: 100.0,
            max_flux: 4.0,
            conductivity: 1.0,
        }
    }
    #[test]
    fn capture_saturates_and_conserves_count_and_energy() {
        let shape = GridShape::new(1, 1, 1).unwrap();
        let mut field =
            FreeManaField::uniform(shape, 1.0, Quantity::nonnegative(20.0).unwrap()).unwrap();
        let mut sites = MaterialLattice::new(shape);
        sites.install([0, 0, 0], material()).unwrap();
        let before = sites.balance(&field).unwrap();
        for _ in 0..100 {
            sites.step(&mut field, 0.1).unwrap();
        }
        let after = sites.balance(&field).unwrap();
        assert!((before.manons() - after.manons()).abs() < 1e-7);
        assert!((before.energy() - after.energy()).abs() < 1e-7);
        assert!(after.bound > 0.0 && after.bound <= 10.0);
        assert!(sites.at(0, 0, 0).unwrap().temperature().unwrap() >= 300.0);
    }
    #[test]
    fn channel_length_and_area_affect_flux() {
        let shape = GridShape::new(2, 1, 1).unwrap();
        let mut field =
            FreeManaField::uniform(shape, 1.0, Quantity::nonnegative(0.0).unwrap()).unwrap();
        field
            .set(0, 0, 0, Quantity::nonnegative(10.0).unwrap())
            .unwrap();
        let mut sites = MaterialLattice::new(shape);
        sites
            .add_channel(ManaChannel {
                from: [0, 0, 0],
                to: [1, 0, 0],
                length: 2.0,
                area: 1.0,
                conductance: 1.0,
                max_flux: 100.0,
            })
            .unwrap();
        let before = sites.balance(&field).unwrap();
        sites.step(&mut field, 0.2).unwrap();
        assert!((field.at(1, 0, 0).unwrap().value() - 1.0).abs() < 1e-8);
        assert!((sites.balance(&field).unwrap().manons() - before.manons()).abs() < 1e-8);
    }
    #[test]
    fn controlled_discharge_consumes_work_and_preserves_energy() {
        let shape = GridShape::new(1, 1, 1).unwrap();
        let mut field =
            FreeManaField::uniform(shape, 1.0, Quantity::nonnegative(20.0).unwrap()).unwrap();
        let mut sites = MaterialLattice::new(shape);
        let mut m = material();
        m.release_rate = 0.0;
        sites.install([0, 0, 0], m).unwrap();
        sites.step(&mut field, 1.0).unwrap();
        let before = sites.balance(&field).unwrap();
        assert!(before.bound > 0.0);
        let requested = before.bound.min(2.0);
        let report = sites
            .release_with_work(&mut field, [0, 0, 0], requested, 10.0, 1.0)
            .unwrap();
        let after = sites.balance(&field).unwrap();
        assert!((before.manons() - after.manons()).abs() < 1e-8);
        assert!((before.energy() + report.work_used_joules - after.energy()).abs() < 1e-8);
        assert!((report.released - requested).abs() < 1e-8);
        let snapshot = sites.balance(&field).unwrap();
        assert!(
            sites
                .release_with_work(&mut field, [0, 0, 0], 1.0, -1.0, 1.0)
                .is_err()
        );
        assert_eq!(sites.balance(&field).unwrap().manons(), snapshot.manons());
    }

    #[test]
    fn failed_step_does_not_mutate_state() {
        let shape = GridShape::new(1, 1, 1).unwrap();
        let mut field =
            FreeManaField::uniform(shape, 1.0, Quantity::nonnegative(3.0).unwrap()).unwrap();
        let mut sites = MaterialLattice::new(shape);
        let before = sites.balance(&field).unwrap();
        assert!(sites.step(&mut field, f64::NAN).is_err());
        assert_eq!(sites.balance(&field).unwrap().manons(), before.manons());
    }
}
