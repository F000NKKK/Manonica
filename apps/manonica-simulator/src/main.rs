use manonica_core::{
    Evolution, Quantity, TimeStep, Vec3,
    units::{Diffusivity, Joules, ManonDensity, Metres, MetresPerSecond},
};
use manonica_physics::{FreeDiffusion, FreeManaField, GridShape, Manon, ManonId, ManonSystem};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Микроэксперимент: один свободный манон без внешних воздействий.
    let mut particles = ManonSystem::new();
    particles.insert(Manon::free(
        ManonId(1),
        Vec3::<Metres>::try_values(0.0, 0.0, 0.0)?,
        Vec3::<MetresPerSecond>::try_values(1.0, 0.0, 0.0)?,
        Quantity::<Joules>::nonnegative(5.0)?,
    )?)?;
    for _ in 0..100 {
        particles.advance(TimeStep::from_seconds(0.01)?)?;
    }
    for manon in particles.manons() {
        println!(
            "free manon id={} x={:.4}m",
            manon.id().0,
            manon.position().x.value()
        );
    }

    // Макроэксперимент: 3D-поле БЕЗ моделирования каждого манона.
    let shape = GridShape::new(8, 8, 8)?;
    let mut field =
        FreeManaField::uniform(shape, 1.0, Quantity::<ManonDensity>::nonnegative(0.0)?)?;
    field.set(4, 4, 4, Quantity::<ManonDensity>::nonnegative(1_000.0)?)?;
    let diffusion = FreeDiffusion::new(Quantity::<Diffusivity>::nonnegative(0.25)?)?;
    for _ in 0..100 {
        diffusion.evolve(&mut field, TimeStep::from_seconds(0.05)?)?;
    }
    println!(
        "free mana field total={:.6} manons",
        field.total_manons()?.value()
    );
    Ok(())
}
