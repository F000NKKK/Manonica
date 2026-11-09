use manonica_materials::MaterialLattice;
use manonica_physics::{FreeManaField, PotentialField};

/// Energy reservoirs of the externally established quasi-static field.
#[derive(Clone, Copy, Debug, Default)]
pub struct FieldLedger {
    pub field_energy_joules: f64,
    pub manon_potential_joules: f64,
    pub dissipated_heat_joules: f64,
    pub operating_work_joules: f64,
    pub last_transport_manons: f64,
    pub peak_potential: f64,
    pub exported_heat_joules: f64,
    pub shutdown: bool,
}
impl FieldLedger {
    pub fn tracked_energy(self) -> f64 {
        self.field_energy_joules + self.manon_potential_joules + self.dissipated_heat_joules
    }
}

/// A complete transactional snapshot, shared by CPU/GPU adapters and game integration.
#[derive(Clone, Debug)]
pub struct CoupledState {
    pub field: FreeManaField,
    pub materials: MaterialLattice,
    pub potential: Option<PotentialField>,
    pub field_ledger: Option<FieldLedger>,
    pub shutdown: bool,
}
