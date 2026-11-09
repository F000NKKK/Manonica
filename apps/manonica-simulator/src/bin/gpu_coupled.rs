//! End-to-end coupled physics parity and timing without Lab or a window.
use manonica_compute_gpu::GpuCoupled;
use manonica_simulator::scenarios::coupled_crystal;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (mut cpu, params) = coupled_crystal()?;
    let initial = cpu.materials.balance(&cpu.field)?;
    let mut gpu = GpuCoupled::new(&cpu, &params).map_err(std::io::Error::other)?;
    let started = Instant::now();
    for _ in 0..200 {
        cpu.step(&params)?;
    }
    let cpu_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    for _ in 0..4 {
        gpu.advance(50)?;
    }
    let gpu_ms = started.elapsed().as_secs_f64() * 1000.0;
    let actual = gpu.snapshot();
    let error = cpu
        .field
        .densities()
        .iter()
        .zip(actual.field.densities())
        .map(|(a, b)| (a.value() - b.value()).abs())
        .fold(0.0_f64, f64::max);
    let material_error = cpu
        .materials
        .cells()
        .iter()
        .zip(actual.materials.cells())
        .flat_map(|(a, b)| [(a.bound - b.bound).abs(), (a.heat - b.heat).abs()])
        .fold(0.0_f64, f64::max);
    let mass_error = (actual.materials.balance(&actual.field)?.manons() - initial.manons()).abs();
    println!("adapter: {}", gpu.adapter_name());
    println!(
        "{} cells, 200 coupled steps, 4 snapshots: CPU {cpu_ms:.3} ms, GPU end-to-end {gpu_ms:.3} ms",
        cpu.field.shape().len()
    );
    println!(
        "maximum free-density error {error:.3e}; maximum bound/heat error {material_error:.3e}; inventory error {mass_error:.3e}"
    );
    let expected = cpu.field_ledger.unwrap();
    let observed = actual.field_ledger.unwrap();
    let ledger_error = [
        (expected.field_energy_joules - observed.field_energy_joules).abs(),
        (expected.dissipated_heat_joules - observed.dissipated_heat_joules).abs(),
        (expected.exported_heat_joules - observed.exported_heat_joules).abs(),
        (expected.operating_work_joules - observed.operating_work_joules).abs(),
    ]
    .into_iter()
    .fold(0.0_f64, f64::max);
    println!(
        "maximum energy-ledger error {ledger_error:.3e}; shutdown {}",
        actual.shutdown
    );
    if error > 1e-3
        || material_error > 1e-3
        || ledger_error > 3e-3
        || mass_error > initial.manons() * 5e-4
        || actual.shutdown != cpu.shutdown
    {
        return Err(std::io::Error::other("coupled GPU solver differs from CPU reference").into());
    }
    Ok(())
}
