//! Session adapter; physical equations are owned by manonica-physics.
use crate::experiment::{ExperimentError, MechanicsConfig};
use manonica_core::TimeStep;
use manonica_physics::SpectralMechanics;
#[derive(Clone, Debug)]
pub struct MechanicsObservation {
    pub positions: Vec<[f64; 3]>,
    pub field: std::sync::Arc<manonica_physics::SpectralFieldSnapshot>,
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
    pub fn state(&self) -> &SpectralMechanics {
        &self.system
    }
    pub fn import(&mut self, state: &SpectralMechanics) {
        self.system = state.clone();
    }
    pub fn step(&mut self, dt: TimeStep) -> Result<(), ExperimentError> {
        self.system.advance(dt)?;
        Ok(())
    }
    pub fn energy(&self) -> f64 {
        self.system
            .total_energy()
            .expect("validated mechanics state")
            .value()
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
            field: std::sync::Arc::new(self.system.field_snapshot()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{experiment::Experiment, simulation::Simulation};
    fn experiment() -> Experiment {
        Experiment::parse(include_str!("../../experiments/dynamic_mechanics.toml")).unwrap()
    }
    #[test]
    fn mechanical_experiment_steps_and_reset_restore_all_observations() {
        let mut simulation = Simulation::new(experiment()).unwrap();
        let initial = simulation.mechanics_observation().unwrap();
        for _ in 0..100 {
            simulation.step().unwrap();
        }
        let observed = simulation.mechanics_observation().unwrap();
        assert_ne!(observed.positions, initial.positions);
        assert!(observed.energy_error.abs() < 1e-6);
        assert!(observed.momentum_error < 1e-11);
        assert_eq!(simulation.steps(), 100);
        assert!(!simulation.supports_resident_diffusion());
        simulation.reset().unwrap();
        let reset = simulation.mechanics_observation().unwrap();
        assert_eq!(reset.positions, initial.positions);
        assert_eq!(reset.energy, initial.energy);
        assert_eq!(reset.energy_error, 0.0);
        assert_eq!(simulation.steps(), 0);
    }
    #[test]
    fn mechanical_config_rejects_mixed_laws_and_invalid_bodies() {
        let mut mixed = experiment();
        mixed.physics.diffusivity = 0.1;
        assert!(Simulation::new(mixed).is_err());
        let mut bad = experiment();
        bad.mechanics.as_mut().unwrap().bodies[0].mass = 0.0;
        assert!(Simulation::new(bad).is_err());
        let mut bad = experiment();
        bad.physics.dt = 1.0;
        assert!(Simulation::new(bad).is_err());
    }
    #[test]
    #[ignore = "requires a Vulkan compute adapter"]
    fn mechanical_gpu_matches_cpu_across_resident_batches() {
        let mut cpu = Simulation::new(experiment()).unwrap();
        let mut observed = Simulation::new(experiment()).unwrap();
        let mut gpu = manonica_compute_gpu::GpuSpectral::new(
            observed.mechanics_state().unwrap(),
            TimeStep::from_seconds(observed.experiment.physics.dt).unwrap(),
        )
        .unwrap();
        eprintln!("mechanics adapter: {}", gpu.adapter_name());
        for _ in 0..5 {
            for _ in 0..20 {
                cpu.step().unwrap();
            }
            observed.step_mechanics_gpu(&mut gpu, 20).unwrap();
            let a = cpu.mechanics_state().unwrap();
            let b = observed.mechanics_state().unwrap();
            for (a, b) in a.sources().iter().zip(b.sources()) {
                let x = a.body.position();
                let y = b.body.position();
                let p = a.body.momentum();
                let q = b.body.momentum();
                for (a, b) in [
                    x.x.value(),
                    x.y.value(),
                    x.z.value(),
                    p.x.value(),
                    p.y.value(),
                    p.z.value(),
                ]
                .into_iter()
                .zip([
                    y.x.value(),
                    y.y.value(),
                    y.z.value(),
                    q.x.value(),
                    q.y.value(),
                    q.z.value(),
                ]) {
                    assert!((a - b).abs() < 1e-4, "body parity {a} vs {b}");
                }
            }
            for (a, b) in a.modes().iter().zip(b.modes()) {
                for (a, b) in [a.a, a.b, a.pa, a.pb]
                    .into_iter()
                    .zip([b.a, b.b, b.pa, b.pb])
                {
                    assert!((a - b).abs() < 1e-5, "mode parity {a} vs {b}");
                }
            }
            let snapshot = observed.mechanics_observation().unwrap();
            assert!(snapshot.energy_error.abs() < 1e-4);
            assert!(snapshot.momentum_error < 1e-4);
        }
        assert_eq!(cpu.steps(), observed.steps());
        assert!((cpu.elapsed() - observed.elapsed()).abs() < 1e-12);
    }
}
