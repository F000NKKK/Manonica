//! End-to-end coupled physics parity and timing without Lab or a window.
use manonica_compute_gpu::GpuCoupled;
use manonica_simulator::scenarios::coupled_crystal_grid;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dimensions = [8, 6, 4];
    let mut steps = 200u32;
    let mut batch = 50u32;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--grid" => {
                for axis in &mut dimensions {
                    *axis = args.next().ok_or("--grid requires nx ny nz")?.parse()?;
                }
            }
            "--steps" => steps = args.next().ok_or("--steps requires a count")?.parse()?,
            "--batch" => batch = args.next().ok_or("--batch requires a count")?.parse()?,
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    if steps == 0 || batch == 0 || batch > 4096 {
        return Err("steps and batch must be positive; batch <=4096".into());
    }
    let (mut cpu, params) = coupled_crystal_grid(dimensions)?;
    let initial = cpu.materials.balance(&cpu.field)?;
    let mut gpu = GpuCoupled::new(&cpu, &params).map_err(std::io::Error::other)?;
    // Warm shader compilation and driver resources before timing either path.
    gpu.advance(1)?;
    cpu.step(&params)?;
    let started = Instant::now();
    for _ in 0..steps {
        cpu.step(&params)?;
    }
    let cpu_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let mut remaining = steps;
    let mut samples = Vec::new();
    while remaining > 0 {
        let count = remaining.min(batch);
        let stamp = Instant::now();
        gpu.advance(count)?;
        samples.push(stamp.elapsed().as_secs_f64() * 1000.0);
        remaining -= count;
    }
    let gpu_ms = started.elapsed().as_secs_f64() * 1000.0;
    samples.sort_by(f64::total_cmp);
    let p50 = samples[samples.len() / 2];
    let p95 = samples[((samples.len() - 1) as f64 * 0.95).ceil() as usize];
    println!(
        "batch <= {batch} steps: p50 {p50:.3} ms, p95 {p95:.3} ms; mean {:.3} ms/step",
        gpu_ms / f64::from(steps)
    );
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
        "{} cells, {steps} timed coupled steps (+1 warmup), {} snapshots: CPU {cpu_ms:.3} ms, GPU end-to-end {gpu_ms:.3} ms",
        cpu.field.shape().len(),
        samples.len()
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
