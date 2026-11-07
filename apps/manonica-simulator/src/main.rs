use manonica_core::{Quantity, TimeStep, Vec3, units::{Joules, Metres, MetresPerSecond}};
use manonica_physics::{Manon, ManonId, ManonSystem};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut simulation = ManonSystem::new();
    simulation.insert(Manon::free(
        ManonId(1),
        Vec3::<Metres>::try_values(0.0, 0.0, 0.0)?,
        Vec3::<MetresPerSecond>::try_values(1.0, 0.0, 0.0)?,
        Quantity::<Joules>::nonnegative(5.0)?,
    )?)?;

    for _ in 0..100 {
        simulation.advance(TimeStep::from_seconds(0.01)?)?;
    }

    for manon in simulation.manons() {
        println!(
            "manon={} x={:.4}m energy={:.4}J",
            manon.id().0,
            manon.position().x.value(),
            manon.energy().value()
        );
    }
    Ok(())
}
