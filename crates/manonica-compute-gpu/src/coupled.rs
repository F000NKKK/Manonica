//! Resident coupled solver. Only explicit snapshots cross the device boundary.
mod packing;
mod readback;
use crate::{
    GpuPotential, channels, coupled_transport, energy,
    error::invalid,
    kernel::Kernel,
    layout::{LEDGER_WORDS, STATE_WORDS},
    material_exchange, potential_drift, thermal,
};
use manonica_error::BaseError;
use manonica_simulation::{CoupledState, StepParameters};
use meridian_gpu_wgpu::{Buffer, BufferUsage, CommandBuffer};
use meridian_sdk::GpuComputeDevice;
use packing::Upload;
use std::time::Instant;

/// Complete device-side operator split with persistent state and rollback.
/// The host retains a validated snapshot for rendering, persistence and fallback.
pub struct GpuCoupled {
    runtime: tokio::runtime::Runtime,
    device: GpuComputeDevice,
    cells: Buffer,
    scratch: Buffer,
    ledger: Buffer,
    status: Buffer,
    backup_cells: Buffer,
    backup_ledger: Buffer,
    readback: Buffer,
    initialize: Kernel,
    begin: Kernel,
    diffuse: Kernel,
    advect: Kernel,
    drift: Kernel,
    exchange: Kernel,
    channels: Kernel,
    thermal: Kernel,
    finish: Kernel,
    snapshot: CoupledState,
    scale: f64,
    last_wall_ms: f64,
}
impl GpuCoupled {
    pub fn new(state: &CoupledState, parameters: &StepParameters) -> Result<Self, String> {
        let mut state = state.clone();
        // Potential is solved on the compute adapter at setup; no CPU Jacobi
        // solve or potential readback occurs in the resident stepping loop.
        if let Some(p) = state.potential.as_mut() {
            let mut solver = GpuPotential::new(p.shape())?;
            let sites = p.local_boundaries().collect::<Vec<_>>();
            let values = solver
                .solve(
                    p.cell_width_metres(),
                    p.sources(),
                    p.permittivity(),
                    p.screening(),
                    p.boundary(),
                    p.iterations(),
                    &sites,
                )
                .map_err(|e| e.to_string())?;
            p.try_replace_values(values).map_err(|e| e.to_string())?;
        }
        let upload = Upload::new(&state, parameters).map_err(|e| e.to_string())?;
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
        if [
            &upload.cells,
            &upload.materials,
            &upload.channels,
            &upload.parameters,
            &upload.local,
            &upload.ledger,
        ]
        .iter()
        .any(|data| data.len() as u64 > limits.max_storage_buffer_binding_size)
            || limits.max_storage_buffers_per_shader_stage < 6
            || upload.cells.len() as u64 + LEDGER_WORDS as u64 * 4 + 4 > limits.max_buffer_size
            || (state.field.shape().len() as u64).div_ceil(64)
                > u64::from(limits.max_compute_workgroups_per_dimension)
        {
            return Err("coupled grid exceeds device limits".into());
        }
        let make = |data: &[u8]| {
            let b = raw.create_buffer(data.len(), BufferUsage::Storage);
            raw.write_buffer(&b, data);
            b
        };
        let cells = make(&upload.cells);
        let scratch = raw.create_buffer(upload.cells.len(), BufferUsage::Storage);
        let materials = make(&upload.materials);
        let links = make(&upload.channels);
        let params = make(&upload.parameters);
        let local = make(&upload.local);
        let ledger = make(&upload.ledger);
        let status = make(&[0u8; 4]);
        let backup_cells = raw.create_buffer(upload.cells.len(), BufferUsage::Storage);
        let backup_ledger = raw.create_buffer(LEDGER_WORDS * 4, BufferUsage::Storage);
        let readback = raw.create_buffer(
            upload.cells.len() + LEDGER_WORDS * 4 + 4,
            BufferUsage::Storage,
        );
        let energy_buffers = [&cells, &materials, &params, &local, &ledger, &status];
        let initialize = energy::kernel(raw, "initialize", &energy_buffers);
        let begin = energy::kernel(raw, "begin", &energy_buffers);
        let finish = energy::kernel(raw, "finish", &energy_buffers);
        let transport_buffers = [&cells, &scratch, &params, &status];
        let diffuse = coupled_transport::kernel(raw, "diffuse", &transport_buffers);
        let advect = coupled_transport::kernel(raw, "advect", &transport_buffers);
        let drift = potential_drift::kernel(
            raw,
            "drift",
            &[&cells, &scratch, &params, &local, &ledger, &status],
        );
        let exchange = material_exchange::kernel(raw, &[&cells, &materials, &params, &status]);
        let channels = channels::kernel(raw, &[&cells, &links, &params, &status]);
        let thermal = thermal::kernel(raw, &[&cells, &materials, &params, &ledger, &status]);
        let mut result = Self {
            runtime,
            device,
            cells,
            scratch,
            ledger,
            status,
            backup_cells,
            backup_ledger,
            readback,
            initialize,
            begin,
            diffuse,
            advect,
            drift,
            exchange,
            channels,
            thermal,
            finish,
            snapshot: state,
            scale: 1.0,
            last_wall_ms: 0.0,
        };
        let mut commands = result.device.gpu_driver_device().create_command_buffer();
        dispatch(&mut commands, &result.initialize, 1);
        result.copy_readback(&mut commands);
        commands.submit();
        result.receive().map_err(|e| e.to_string())?;
        Ok(result)
    }
    pub fn snapshot(&self) -> &CoupledState {
        &self.snapshot
    }
    pub fn last_wall_ms(&self) -> f64 {
        self.last_wall_ms
    }
    pub fn adapter_name(&self) -> &str {
        self.device.gpu_driver_device().adapter_name()
    }

    /// Advance all physical operators without intermediate host transfers.
    /// On any kernel or validation failure both host and device roll back.
    pub fn advance(&mut self, steps: u32) -> Result<&CoupledState, BaseError> {
        if steps == 0 || steps > 4096 {
            return Err(invalid("gpu_coupled_batch_steps"));
        }
        let started = Instant::now();
        let mut commands = self.device.gpu_driver_device().create_command_buffer();
        copy(&mut commands, &self.cells, &self.backup_cells, 0);
        copy(&mut commands, &self.ledger, &self.backup_ledger, 0);
        let groups = (self.snapshot.field.shape().len() as u32).div_ceil(64);
        for _ in 0..steps {
            dispatch(&mut commands, &self.begin, 1);
            dispatch(&mut commands, &self.diffuse, groups);
            copy(&mut commands, &self.scratch, &self.cells, 0);
            dispatch(&mut commands, &self.advect, groups);
            copy(&mut commands, &self.scratch, &self.cells, 0);
            dispatch(&mut commands, &self.drift, 1);
            dispatch(&mut commands, &self.exchange, groups);
            dispatch(&mut commands, &self.channels, 1);
            dispatch(&mut commands, &self.thermal, 1);
            dispatch(&mut commands, &self.finish, 1);
        }
        self.copy_readback(&mut commands);
        commands.submit();
        if let Err(error) = self.receive() {
            let raw = self.device.gpu_driver_device();
            let mut rollback = raw.create_command_buffer();
            copy(&mut rollback, &self.backup_cells, &self.cells, 0);
            copy(&mut rollback, &self.backup_ledger, &self.ledger, 0);
            rollback.submit();
            raw.write_buffer(&self.status, &[0u8; 4]);
            return Err(error);
        }
        self.last_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(&self.snapshot)
    }
    fn copy_readback(&self, commands: &mut CommandBuffer<'_>) {
        let bytes = self.snapshot.field.shape().len() * STATE_WORDS * 4;
        copy(commands, &self.cells, &self.readback, 0);
        copy(commands, &self.ledger, &self.readback, bytes as u64);
        copy(
            commands,
            &self.status,
            &self.readback,
            (bytes + LEDGER_WORDS * 4) as u64,
        );
    }
    fn receive(&mut self) -> Result<(), BaseError> {
        let bytes = self
            .runtime
            .block_on(self.device.read_buffer(&self.readback));
        let (next, scale) = readback::snapshot(&self.snapshot, &bytes, self.scale)?;
        self.snapshot = next;
        self.scale = scale;
        Ok(())
    }
}
fn dispatch(commands: &mut CommandBuffer<'_>, kernel: &Kernel, groups: u32) {
    commands.dispatch_compute_with_bind_group(&kernel.pipeline, &kernel.bind, groups);
}
fn copy(commands: &mut CommandBuffer<'_>, source: &Buffer, target: &Buffer, offset: u64) {
    commands.encoder_mut().copy_buffer_to_buffer(
        source.wgpu_buffer(),
        0,
        target.wgpu_buffer(),
        offset,
        source.byte_len as u64,
    );
}
