#![forbid(unsafe_code)]

//! Manonica Lab: декларативные эксперименты + Meridian SDK / WGPU.

use std::path::PathBuf;
use manonica_lab::{app, experiment::Experiment, simulation::Simulation};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("experiments/mana_flame.toml")
    });
    let experiment = Experiment::from_file(&path)?;
    println!("Experiment: {} ({})", experiment.name, path.display());
    println!(
        "Grid: {:?}, dt: {}s, structures: {}, seeds: {}",
        experiment.grid.dimensions,
        experiment.physics.dt,
        experiment.circles.len(),
        experiment.seeds.len(),
    );
    println!("Controls: WASD + mouse, Escape cursor, Space pause, R reset, F wisps, arrows threshold");
    let title = format!("Manonica Lab — {}", experiment.name);
    let sim = Simulation::new(experiment)?;
    println!("Initial free manons: {:.6}", sim.total()?.value());
    meridian_sdk::run_windowed_app(&title, 1280, 800, app::App::new(sim))?;
    Ok(())
}
