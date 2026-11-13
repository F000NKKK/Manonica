//! Headless full-packet spectral mechanics benchmark and CPU parity audit.
use manonica_compute_gpu::GpuSpectral;
use manonica_core::{Quantity, TimeStep, Vec3};
use manonica_physics::{GaussianBody, MechanicalBody, PeriodicSpectralDomain};
use std::{
    error::Error,
    fs::OpenOptions,
    io::{BufWriter, Write},
    time::Instant,
};
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let at = (sorted.len() - 1) as f64 * q;
    let low = at.floor() as usize;
    let high = at.ceil() as usize;
    sorted[low] + (sorted[high] - sorted[low]) * (at - low as f64)
}
fn main() -> Result<(), Box<dyn Error>> {
    let mut steps = 3000_u32;
    let mut batch = 10_u32;
    let mut warmup = 100_u32;
    let mut cutoff = 3_usize;
    let mut csv = None;
    let mut label = "unspecified".to_owned();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args.next().ok_or("each option requires a value")?;
        match arg.as_str() {
            "--steps" => steps = value.parse()?,
            "--batch" => batch = value.parse()?,
            "--warmup" => warmup = value.parse()?,
            "--cutoff" => cutoff = value.parse()?,
            "--csv" => csv = Some(value),
            "--label" => label = value,
            _ => return Err(format!("unknown option {arg}").into()),
        }
    }
    if steps == 0
        || steps > 100000
        || batch == 0
        || batch > 256
        || !steps.is_multiple_of(batch)
        || warmup > 10000
        || !(1..=6).contains(&cutoff)
    {
        return Err("invalid steps/batch/warmup/cutoff".into());
    }
    let mut output = csv
        .map(|path| {
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map(BufWriter::new)
        })
        .transpose()?;
    if let Some(output) = &mut output {
        writeln!(output, "batch,steps,wall_ms,simulated_seconds")?;
    }
    let make = |x: [f64; 3], p: [f64; 3], charge: f64| -> Result<GaussianBody, Box<dyn Error>> {
        Ok(GaussianBody {
            body: MechanicalBody::new(
                Quantity::new(2.0)?,
                Vec3::try_values(x[0], x[1], x[2])?,
                Vec3::try_values(p[0], p[1], p[2])?,
            )?,
            charge,
            width: 0.6,
        })
    };
    let domain = PeriodicSpectralDomain::new(
        Vec3::try_values(
            std::f64::consts::TAU,
            std::f64::consts::TAU,
            std::f64::consts::TAU,
        )?,
        [cutoff; 3],
    )?;
    let mut cpu = domain.static_equilibrium(vec![
        make([-0.8, -0.55, -1.05], [0.15, -0.1, 0.07], 0.4)?,
        make([0.8, 0.55, 1.05], [-0.1, 0.04, -0.03], -0.4)?,
    ])?;
    let dt = TimeStep::from_seconds(0.005)?;
    let mut gpu = GpuSpectral::new(&cpu, dt).map_err(std::io::Error::other)?;
    let mut remaining = warmup;
    while remaining > 0 {
        let count = remaining.min(batch);
        gpu.advance(count).map_err(std::io::Error::other)?;
        remaining -= count;
    }
    for _ in 0..warmup {
        cpu.advance(dt)?;
    }
    let started = Instant::now();
    for _ in 0..steps {
        cpu.advance(dt)?;
    }
    let cpu_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut samples = Vec::new();
    remaining = steps;
    let mut accepted = 0_u32;
    let started = Instant::now();
    while remaining > 0 {
        let count = remaining.min(batch);
        let stamp = Instant::now();
        gpu.advance(count).map_err(std::io::Error::other)?;
        let wall = stamp.elapsed().as_secs_f64() * 1000.0;
        samples.push(wall);
        accepted += count;
        if let Some(output) = &mut output {
            writeln!(
                output,
                "{},{count},{wall:.9},{:.9}",
                samples.len(),
                f64::from(accepted) * dt.seconds()
            )?;
        }
        remaining -= count;
    }
    let gpu_ms = started.elapsed().as_secs_f64() * 1000.0;
    if let Some(output) = &mut output {
        output.flush()?;
    }
    let observed = gpu.snapshot();
    let mut body_error = 0.0_f64;
    let mut mode_error = 0.0_f64;
    for (a, b) in cpu.sources().iter().zip(observed.sources()) {
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
            body_error = body_error.max((a - b).abs());
        }
    }
    for (a, b) in cpu.modes().iter().zip(observed.modes()) {
        for (a, b) in [a.a, a.b, a.pa, a.pb]
            .into_iter()
            .zip([b.a, b.b, b.pa, b.pb])
        {
            mode_error = mode_error.max((a - b).abs());
        }
    }
    let energy_error = (cpu.total_energy()?.value() - observed.total_energy()?.value()).abs();
    let p = cpu.total_momentum()?;
    let q = observed.total_momentum()?;
    let momentum_error = [
        (p.x.value() - q.x.value()).abs(),
        (p.y.value() - q.y.value()).abs(),
        (p.z.value() - q.z.value()).abs(),
    ]
    .into_iter()
    .fold(0.0, f64::max);
    samples.sort_by(f64::total_cmp);
    println!("label: {label}; adapter: {}", gpu.adapter_name());
    println!(
        "{} bodies, {} mode pairs; dt {} s; {steps} timed steps + {warmup} warmup; {} packets of <= {batch} steps",
        cpu.sources().len(),
        cpu.modes().len(),
        dt.seconds(),
        samples.len()
    );
    println!(
        "packet p50 {:.6} ms, p95 {:.6} ms, p99 {:.6} ms, max {:.6} ms",
        percentile(&samples, 0.5),
        percentile(&samples, 0.95),
        percentile(&samples, 0.99),
        samples.last().unwrap()
    );
    println!(
        "mean {:.6} ms/step; CPU {cpu_ms:.6} ms; GPU end-to-end {gpu_ms:.6} ms",
        gpu_ms / f64::from(steps)
    );
    println!(
        "max CPU/GPU body error {body_error:.6e}; mode error {mode_error:.6e}; energy error {energy_error:.6e} J; momentum error {momentum_error:.6e}"
    );
    if body_error > 1e-3 + f64::from(steps) * 1e-7
        || mode_error > 3e-4
        || energy_error > 1e-3
        || momentum_error > 1e-4
    {
        return Err("GPU mechanics parity audit failed".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn percentile_uses_linear_interpolation_between_sorted_packets() {
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.5), 3.0);
        assert!((percentile(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.95) - 4.8).abs() < 1e-12);
        assert_eq!(percentile(&[2.5], 0.99), 2.5);
    }
}
