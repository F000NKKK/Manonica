//! Resident Gaussian-body/spectral-field Hamiltonian steps, validated snapshots.
use crate::kernel::{Kernel, copy, dispatch};
use manonica_core::{TimeStep, Vec3};
use manonica_physics::{MechanicalBody, SpectralMechanics};
use meridian_gpu_wgpu::{Buffer, BufferUsage};
use meridian_sdk::GpuComputeDevice;
use std::time::Instant;

pub struct GpuSpectral {
    runtime: tokio::runtime::Runtime,
    device: GpuComputeDevice,
    bodies: Buffer,
    modes: Buffer,
    backup_bodies: Buffer,
    backup_modes: Buffer,
    kick_bodies: Kernel,
    kick_modes: Kernel,
    drift_bodies: Kernel,
    drift_modes: Kernel,
    snapshot: SpectralMechanics,
    last_wall_ms: f64,
    initial_energy: f64,
    initial_momentum: [f64; 3],
}
fn encode(values: impl IntoIterator<Item = f64>) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    for value in values {
        let f = value as f32;
        if !f.is_finite() {
            return Err("spectral input exceeds f32".into());
        }
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    Ok(bytes)
}
fn decode(bytes: &[u8]) -> Result<Vec<f64>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err("invalid spectral readback size".into());
    }
    let values: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b) as f64)
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite spectral readback".into());
    }
    Ok(values)
}
impl GpuSpectral {
    pub fn new(state: &SpectralMechanics, dt: TimeStep) -> Result<Self, String> {
        if state.sources().is_empty()
            || state.sources().len() > 2048
            || state.modes().len() > 65535
            || state.modes().iter().any(|m| {
                dt.seconds() * (m.wave.iter().map(|v| v * v).sum::<f64>() + 1.0).sqrt() >= 2.0
            })
        {
            return Err("invalid GPU spectral dimensions or timestep".into());
        }
        let body_data = encode(state.sources().iter().flat_map(|s| {
            let x = s.body.position();
            let p = s.body.momentum();
            [
                x.x.value(),
                x.y.value(),
                x.z.value(),
                s.body.mass().value(),
                p.x.value(),
                p.y.value(),
                p.z.value(),
                0.0,
                s.charge,
                s.width,
                0.0,
                0.0,
            ]
        }))?;
        let mode_data = encode(state.modes().iter().flat_map(|m| {
            [
                m.wave[0],
                m.wave[1],
                m.wave[2],
                m.wave.iter().map(|v| v * v).sum::<f64>() + 1.0,
                m.a,
                m.b,
                m.pa,
                m.pb,
            ]
        }))?;
        let mut params = encode([dt.seconds(), 1.0 / state.modal_inertia()])?;
        if (dt.seconds() as f32) <= 0.0
            || (1.0 / state.modal_inertia()) as f32 <= 0.0
            || state
                .sources()
                .iter()
                .any(|s| (s.body.mass().value() as f32) <= 0.0 || (s.width as f32) <= 0.0)
        {
            return Err("GPU spectral parameter underflow".into());
        }
        params.extend_from_slice(&(state.sources().len() as u32).to_le_bytes());
        params.extend_from_slice(&(state.modes().len() as u32).to_le_bytes());
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;
        let device = runtime
            .block_on(GpuComputeDevice::new())
            .map_err(|e| e.to_string())?;
        let raw = device.gpu_driver_device();
        let limits = raw.wgpu_device().limits();
        if body_data.len().max(mode_data.len()) as u64 > limits.max_storage_buffer_binding_size
            || limits.max_storage_buffers_per_shader_stage < 3
        {
            return Err("spectral state exceeds GPU limits".into());
        }
        let bodies = raw.create_buffer(body_data.len(), BufferUsage::Storage);
        raw.write_buffer(&bodies, &body_data);
        let modes = raw.create_buffer(mode_data.len(), BufferUsage::Storage);
        raw.write_buffer(&modes, &mode_data);
        let parameters = raw.create_buffer(params.len(), BufferUsage::Storage);
        raw.write_buffer(&parameters, &params);
        let backup_bodies = raw.create_buffer(body_data.len(), BufferUsage::Storage);
        let backup_modes = raw.create_buffer(mode_data.len(), BufferUsage::Storage);
        let shader = raw.create_shader("manonica-spectral", include_str!("spectral.wgsl"));
        let kernel = |entry: &str| {
            let pipeline = raw.create_compute_pipeline(&shader, entry);
            let layout = pipeline.wgpu_pipeline().get_bind_group_layout(0);
            let bind = raw.create_bind_group(&layout, &[&bodies, &modes, &parameters]);
            Kernel { pipeline, bind }
        };
        let kick_bodies = kernel("kick_bodies");
        let kick_modes = kernel("kick_modes");
        let drift_bodies = kernel("drift_bodies");
        let drift_modes = kernel("drift_modes");
        let p = state.total_momentum().map_err(|e| e.to_string())?;
        let initial_energy = state.total_energy().map_err(|e| e.to_string())?.value();
        Ok(Self {
            initial_energy,
            initial_momentum: [p.x.value(), p.y.value(), p.z.value()],
            runtime,
            device,
            bodies,
            modes,
            backup_bodies,
            backup_modes,
            kick_bodies,
            kick_modes,
            drift_bodies,
            drift_modes,
            snapshot: state.clone(),
            last_wall_ms: 0.0,
        })
    }
    pub fn adapter_name(&self) -> &str {
        self.device.gpu_driver_device().adapter_name()
    }
    pub fn last_wall_ms(&self) -> f64 {
        self.last_wall_ms
    }
    pub fn snapshot(&self) -> &SpectralMechanics {
        &self.snapshot
    }
    pub fn advance(&mut self, steps: u32) -> Result<&SpectralMechanics, String> {
        if steps == 0 || steps > 256 {
            return Err("invalid spectral batch length".into());
        }
        let started = Instant::now();
        let raw = self.device.gpu_driver_device();
        let mut commands = raw.create_command_buffer();
        copy(&mut commands, &self.bodies, &self.backup_bodies, 0);
        copy(&mut commands, &self.modes, &self.backup_modes, 0);
        let bg = (self.snapshot.sources().len() as u32).div_ceil(64);
        let mg = (self.snapshot.modes().len() as u32).div_ceil(64);
        for _ in 0..steps {
            dispatch(&mut commands, &self.kick_bodies, bg);
            dispatch(&mut commands, &self.kick_modes, mg);
            dispatch(&mut commands, &self.drift_bodies, bg);
            dispatch(&mut commands, &self.drift_modes, mg);
            dispatch(&mut commands, &self.kick_bodies, bg);
            dispatch(&mut commands, &self.kick_modes, mg);
        }
        commands.submit();
        let result = (|| {
            let body_data = decode(&self.runtime.block_on(self.device.read_buffer(&self.bodies)))?;
            let mode_data = decode(&self.runtime.block_on(self.device.read_buffer(&self.modes)))?;
            if body_data.len() != self.snapshot.sources().len() * 12
                || mode_data.len() != self.snapshot.modes().len() * 8
            {
                return Err("spectral readback length".into());
            }
            let mut bodies = Vec::new();
            for (s, values) in self
                .snapshot
                .sources()
                .iter()
                .zip(body_data.as_chunks::<12>().0.iter())
            {
                if values[3] != (s.body.mass().value() as f32) as f64
                    || values[8] != (s.charge as f32) as f64
                    || values[9] != (s.width as f32) as f64
                {
                    return Err("spectral immutable body data changed".into());
                }
                let position =
                    Vec3::try_values(values[0], values[1], values[2]).map_err(|e| e.to_string())?;
                let momentum =
                    Vec3::try_values(values[4], values[5], values[6]).map_err(|e| e.to_string())?;
                bodies.push(
                    MechanicalBody::new(s.body.mass(), position, momentum)
                        .map_err(|e| e.to_string())?,
                );
            }
            for (mode, values) in self
                .snapshot
                .modes()
                .iter()
                .zip(mode_data.as_chunks::<8>().0.iter())
            {
                let expected = [
                    mode.wave[0],
                    mode.wave[1],
                    mode.wave[2],
                    mode.wave.iter().map(|v| v * v).sum::<f64>() + 1.0,
                ];
                if expected
                    .into_iter()
                    .zip(&values[..4])
                    .any(|(a, b)| (a as f32) as f64 != *b)
                {
                    return Err("spectral immutable mode data changed".into());
                }
            }
            let storage: Vec<_> = mode_data
                .as_chunks::<8>()
                .0
                .iter()
                .map(|v| [v[4], v[5], v[6], v[7]])
                .collect();
            let mut next = self.snapshot.clone();
            next.replace_snapshot(&bodies, &storage)
                .map_err(|e| e.to_string())?;
            let after = next.total_momentum().map_err(|e| e.to_string())?;
            let before = self.initial_momentum;
            let after = [after.x.value(), after.y.value(), after.z.value()];
            if before
                .into_iter()
                .zip(after)
                .any(|(a, b)| (a - b).abs() > 1e-4 * (1.0 + a.abs()))
            {
                return Err("GPU spectral momentum audit failed".into());
            }
            let e0 = self.initial_energy;
            let e1 = next.total_energy().map_err(|e| e.to_string())?.value();
            if (e1 - e0).abs() > 1e-3 * (1.0 + e0.abs()) {
                return Err("GPU spectral energy audit failed".into());
            }
            Ok(next)
        })();
        match result {
            Ok(next) => self.snapshot = next,
            Err(error) => {
                let mut commands = raw.create_command_buffer();
                copy(&mut commands, &self.backup_bodies, &self.bodies, 0);
                copy(&mut commands, &self.backup_modes, &self.modes, 0);
                commands.submit();
                return Err(error);
            }
        }
        self.last_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(&self.snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use manonica_core::Quantity;
    use manonica_physics::{GaussianBody, PeriodicSpectralDomain};
    fn state(charge: f64) -> SpectralMechanics {
        let bodies = [
            ([-0.8, -0.55, -1.05], [0.15, -0.1, 0.07], charge),
            ([0.8, 0.55, 1.05], [-0.1, 0.04, -0.03], -charge),
        ]
        .into_iter()
        .map(|(x, p, q)| GaussianBody {
            body: MechanicalBody::new(
                Quantity::new(2.0).unwrap(),
                Vec3::try_values(x[0], x[1], x[2]).unwrap(),
                Vec3::try_values(p[0], p[1], p[2]).unwrap(),
            )
            .unwrap(),
            charge: q,
            width: 0.6,
        })
        .collect();
        PeriodicSpectralDomain::new(
            Vec3::try_values(
                std::f64::consts::TAU,
                std::f64::consts::TAU,
                std::f64::consts::TAU,
            )
            .unwrap(),
            [3; 3],
        )
        .unwrap()
        .static_equilibrium(bodies)
        .unwrap()
    }
    #[test]
    fn spectral_gpu_matches_cpu_in_resident_batches() {
        let mut cpu = state(0.4);
        let dt = TimeStep::from_seconds(0.005).unwrap();
        let mut gpu = GpuSpectral::new(&cpu, dt).unwrap();
        for _ in 0..5 {
            for _ in 0..20 {
                cpu.advance(dt).unwrap();
            }
            let observed = gpu.advance(20).unwrap();
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
                    assert!((a - b).abs() < 1e-4);
                }
            }
            for (a, b) in cpu.modes().iter().zip(observed.modes()) {
                for (a, b) in [a.a, a.b, a.pa, a.pb]
                    .into_iter()
                    .zip([b.a, b.b, b.pa, b.pb])
                {
                    assert!((a - b).abs() < 1e-5);
                }
            }
        }
    }
    #[test]
    fn spectral_failed_batch_restores_device_buffers_and_host_snapshot() {
        let initial = state(1e25);
        let mut gpu = GpuSpectral::new(&initial, TimeStep::from_seconds(0.005).unwrap()).unwrap();
        let before_bodies = gpu.runtime.block_on(gpu.device.read_buffer(&gpu.bodies));
        let before_modes = gpu.runtime.block_on(gpu.device.read_buffer(&gpu.modes));
        assert!(gpu.advance(1).is_err());
        assert_eq!(gpu.snapshot(), &initial);
        assert_eq!(
            gpu.runtime.block_on(gpu.device.read_buffer(&gpu.bodies)),
            before_bodies
        );
        assert_eq!(
            gpu.runtime.block_on(gpu.device.read_buffer(&gpu.modes)),
            before_modes
        );
        assert!(gpu.advance(0).is_err());
        assert_eq!(gpu.snapshot(), &initial);
    }
}
