//! Session adapter; physical equations are owned by manonica-physics.
use crate::experiment::{ExperimentError, MechanicsConfig};
use manonica_core::TimeStep;
use manonica_physics::SpectralMechanics;
#[derive(Clone, Debug)]
pub struct MechanicsObservation {
    pub positions: Vec<[f64; 3]>,
    pub energy: f64,
    pub momentum: [f64; 3],
    pub energy_error: f64,
    pub momentum_error: f64,
    pub mode_count: usize,
}
pub(super) struct MechanicsSession {
    system: SpectralMechanics,
    initial_energy: f64,
    initial_momentum: [f64; 3],
}
impl MechanicsSession {
    pub fn new(config: &MechanicsConfig, dt: f64) -> Result<Self, ExperimentError> {
        let system = config.build(dt)?;
        let p = system.total_momentum()?;
        Ok(Self {
            initial_energy: system.total_energy()?.value(),
            initial_momentum: [p.x.value(), p.y.value(), p.z.value()],
            system,
        })
    }
    pub fn step(&mut self, dt: TimeStep) -> Result<(), ExperimentError> {
        self.system.advance(dt)?;
        Ok(())
    }
    pub fn observation(&self) -> MechanicsObservation {
        let energy = self
            .system
            .total_energy()
            .expect("validated mechanics state")
            .value();
        let p = self
            .system
            .total_momentum()
            .expect("validated mechanics state");
        let momentum = [p.x.value(), p.y.value(), p.z.value()];
        MechanicsObservation {
            positions: self
                .system
                .sources()
                .iter()
                .map(|s| {
                    let x = s.body.position();
                    [x.x.value(), x.y.value(), x.z.value()]
                })
                .collect(),
            energy,
            momentum,
            energy_error: energy - self.initial_energy,
            momentum_error: momentum
                .into_iter()
                .zip(self.initial_momentum)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max),
            mode_count: self.system.modes().len(),
        }
    }
}
