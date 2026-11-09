//! Session-level contracts for resident uniform transport.
use super::Simulation;
use crate::experiment::{AmbientFlow, ExperimentBuilder};
use manonica_core::{Evolution, Quantity, TimeStep, units::Diffusivity};
use manonica_error::{BaseError, ErrorCode};
use manonica_physics::{FreeAdvection, FreeDiffusion, FreeManaDiffusionBackend, FreeManaField};

struct ReferenceBackend;
impl FreeManaDiffusionBackend for ReferenceBackend {
    fn evolve_free_diffusion(
        &mut self,
        field: &mut FreeManaField,
        diffusivity: f64,
        dt: f64,
    ) -> Result<(), BaseError> {
        FreeDiffusion::new(Quantity::<Diffusivity>::nonnegative(diffusivity)?)
            .map_err(|_| BaseError::new(ErrorCode::InvalidRange, "reference_diffusivity"))?
            .evolve(field, TimeStep::from_seconds(dt)?)
            .map_err(|_| BaseError::new(ErrorCode::NumericalInstability, "reference_diffusion"))?;
        Ok(())
    }
    fn evolve_diffusion_and_uniform_flow(
        &mut self,
        field: &mut FreeManaField,
        diffusivity: f64,
        dt: f64,
        velocity: [f64; 3],
    ) -> Result<bool, BaseError> {
        self.evolve_free_diffusion(field, diffusivity, dt)?;
        FreeAdvection::new(AmbientFlow::Uniform { velocity })
            .evolve(field, TimeStep::from_seconds(dt)?)
            .map_err(|_| BaseError::new(ErrorCode::NumericalInstability, "reference_flow"))?;
        Ok(true)
    }
}

fn session() -> Simulation {
    Simulation::new(
        ExperimentBuilder::new("uniform transport")
            .grid([8, 6, 4], 0.5)
            .diffusion(0.12, 0.02, 4)
            .flow(AmbientFlow::Uniform {
                velocity: [0.3, -0.2, 0.1],
            })
            .gaussian([0.0, 0.0, 0.0], 500.0, 0.7)
            .build()
            .unwrap(),
    )
    .unwrap()
}

#[test]
fn batch_preserves_operator_order_time_and_observation_count() {
    let mut actual = session();
    let mut expected = session();
    assert!(actual.supports_resident_transport());
    assert!(!actual.supports_resident_diffusion());
    for _ in 0..25 {
        expected.step().unwrap();
    }
    actual
        .step_transport_batch(&mut ReferenceBackend, 25)
        .unwrap();
    assert_eq!(actual.steps(), expected.steps());
    assert!((actual.elapsed() - expected.elapsed()).abs() < 1e-10);
    for (a, b) in actual
        .field()
        .densities()
        .iter()
        .zip(expected.field().densities())
    {
        assert!((a.value() - b.value()).abs() < 1e-10);
    }
    assert!(actual.equilibrium_state().stable_steps <= 1);
}

#[test]
fn failed_batch_does_not_commit_partial_state_or_time() {
    struct FailsAfterOneStep(u32);
    impl FreeManaDiffusionBackend for FailsAfterOneStep {
        fn evolve_free_diffusion(
            &mut self,
            _: &mut FreeManaField,
            _: f64,
            _: f64,
        ) -> Result<(), BaseError> {
            unreachable!()
        }
        fn evolve_diffusion_and_uniform_flow(
            &mut self,
            field: &mut FreeManaField,
            d: f64,
            dt: f64,
            v: [f64; 3],
        ) -> Result<bool, BaseError> {
            self.0 += 1;
            if self.0 == 2 {
                return Err(BaseError::new(
                    ErrorCode::NumericalInstability,
                    "injected_failure",
                ));
            }
            ReferenceBackend.evolve_diffusion_and_uniform_flow(field, d, dt, v)
        }
    }
    let mut sim = session();
    let initial = sim.field().densities().to_vec();
    assert!(
        sim.step_transport_batch(&mut FailsAfterOneStep(0), 4)
            .is_err()
    );
    assert_eq!(sim.steps(), 0);
    assert_eq!(sim.elapsed(), 0.0);
    assert_eq!(sim.field().densities(), initial);
    assert!(sim.step_transport_batch(&mut ReferenceBackend, 0).is_err());
    sim.experiment.physics.flow = AmbientFlow::Vortex {
        origin: [0.0; 3],
        axis: [0.0, 1.0, 0.0],
        angular_speed: 0.1,
        axial_speed: 0.0,
    };
    assert!(!sim.supports_resident_transport());
    assert!(sim.step_transport_batch(&mut ReferenceBackend, 4).is_err());
}
