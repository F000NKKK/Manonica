#![forbid(unsafe_code)]

//! Manonica Lab: декларативные эксперименты + Meridian SDK / WGPU.

use manonica_lab::{
    app,
    experiment::{Experiment, Formula, FormulaTrap},
    simulation::Simulation,
};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("experiments/formula_comparison.toml");
    let mut path_given = false;
    let mut formula = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(argument) = args.next() {
        if argument == "--formula" {
            let mut expressions = Vec::new();
            for _ in 0..3 {
                let expression = args.next().ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "--formula requires exactly three expressions: x(t) y(t) z(t)",
                    )
                })?;
                expressions.push(expression.to_string_lossy().to_string());
            }
            formula = Some(Formula {
                name: "Interactive circuit".into(),
                x: expressions[0].clone(),
                y: expressions[1].clone(),
                z: expressions[2].clone(),
                segments: 96,
                thickness: 0.035,
                color: [0.2, 0.9, 0.6],
                trap: Some(FormulaTrap {
                    capacity: 2500.0,
                    capture_rate: 0.18,
                    release_rate: 0.002,
                    binding_energy: 0.08,
                    heat_capacity: 100.0,
                    reference_temperature: 300.0,
                    activation_temperature: 300.0,
                    max_flux: 75.0,
                }),
            });
        } else if !path_given {
            path = PathBuf::from(argument);
            path_given = true;
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "unknown argument; usage: manonica-lab [experiment.toml] [--formula x(t) y(t) z(t)]",
            ).into());
        }
    }
    let mut experiment = Experiment::from_file(&path)?;
    if let Some(formula) = formula {
        experiment.formulas.push(formula);
        experiment.validate()?;
    }
    println!("Experiment: {} ({})", experiment.name, path.display());
    println!(
        "Grid: {:?}, dt: {}s, formulas: {}, seeds: {}",
        experiment.grid.dimensions,
        experiment.physics.dt,
        experiment.formulas.len(),
        experiment.seeds.len(),
    );
    println!(
        "Controls: WASD + mouse, E formula editor, TAB field, ENTER apply, ESC cancel, F3 diagnostics, Space pause, R reset, F wisps, arrows threshold"
    );
    let title = format!("Manonica Lab — {}", experiment.name);
    let sim = Simulation::new(experiment)?;
    println!("Initial free manons: {:.6}", sim.total()?.value());
    meridian_sdk::run_windowed_app(&title, 1280, 800, app::App::new(sim))?;
    Ok(())
}
