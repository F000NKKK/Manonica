//! GPU screened potential solve with local geometric constraints.
//! The GPU solves the finite-volume equations; CPU remains reference.
//! No rendering or experiment-specific shapes are encoded here.
use manonica_error::{BaseError, ErrorCode};
use manonica_physics::{GridShape, LocalBoundarySite, LocalPotentialBoundary, PotentialBoundary};
use meridian_gpu_wgpu::{BindGroup, Buffer, BufferUsage, ComputePipeline};
use meridian_sdk::GpuComputeDevice;

fn invalid(name: &'static str) -> BaseError {
    BaseError::new(ErrorCode::InvalidRange, name)
}

pub struct GpuPotential {
    runtime: tokio::runtime::Runtime,
    device: GpuComputeDevice,
    pipeline: ComputePipeline,
    previous: Buffer,
    next: Buffer,
    sources: Buffer,
    constraints: Buffer,
    parameters: Buffer,
    forward: BindGroup,
    reverse: BindGroup,
    shape: GridShape,
}

impl GpuPotential {
    pub fn new(shape: GridShape) -> Result<Self, String> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|error| error.to_string())?;
        let device = runtime
            .block_on(GpuComputeDevice::new())
            .map_err(|error| error.to_string())?;
        let raw = device.gpu_driver_device();
        let shader = raw.create_shader(
            "manonica-screened-potential",
            include_str!("gpu_potential.wgsl"),
        );
        let pipeline = raw.create_compute_pipeline(&shader, "potential_jacobi");
        let previous = raw.create_buffer(shape.len() * 4, BufferUsage::Storage);
        let next = raw.create_buffer(shape.len() * 4, BufferUsage::Storage);
        let sources = raw.create_buffer(shape.len() * 4, BufferUsage::Storage);
        let constraints = raw.create_buffer(shape.len() * 16, BufferUsage::Storage);
        let parameters = raw.create_buffer(7 * 4, BufferUsage::Storage);
        let layout = pipeline.wgpu_pipeline().get_bind_group_layout(0);
        let forward = raw.create_bind_group(
            &layout,
            &[&previous, &next, &sources, &constraints, &parameters],
        );
        let reverse = raw.create_bind_group(
            &layout,
            &[&next, &previous, &sources, &constraints, &parameters],
        );
        Ok(Self {
            runtime,
            device,
            pipeline,
            previous,
            next,
            sources,
            constraints,
            parameters,
            forward,
            reverse,
            shape,
        })
    }

    /// Solves a screened positive-definite field with exactly the Jacobi
    /// iteration count of the CPU reference. Only snapshots leave the GPU.
    #[allow(
        clippy::too_many_arguments,
        reason = "numeric solver coefficients and boundary policy are independent inputs"
    )]
    pub fn solve(
        &mut self,
        dx: f64,
        source: &[f64],
        permittivity: f64,
        screening: f64,
        boundary: PotentialBoundary,
        iterations: usize,
        sites: &[LocalBoundarySite],
    ) -> Result<Vec<f64>, BaseError> {
        if !dx.is_finite()
            || dx <= 0.0
            || !permittivity.is_finite()
            || permittivity <= 0.0
            || !screening.is_finite()
            || screening <= 0.0
            || !(1..=4096).contains(&iterations)
            || source.len() != self.shape.len()
            || source.iter().any(|value| !value.is_finite())
        {
            return Err(invalid("gpu_potential_parameters"));
        }
        let alpha = permittivity / dx.powi(2);
        if !alpha.is_finite()
            || alpha > f32::MAX as f64
            || screening > f32::MAX as f64
            || dx > f32::MAX as f64
        {
            return Err(invalid("gpu_potential_precision"));
        }
        let mut constraints = vec![[0.0_f32; 4]; self.shape.len()];
        let mut defined = vec![false; self.shape.len()];
        for site in sites {
            if site.index >= self.shape.len() || defined[site.index] {
                return Err(invalid("gpu_local_boundary_overlap"));
            }
            defined[site.index] = true;
            constraints[site.index] = match site.condition {
                LocalPotentialBoundary::Dirichlet { value } if value.is_finite() => {
                    [1.0, value as f32, 0.0, 0.0]
                }
                LocalPotentialBoundary::Neumann => [2.0, 0.0, 0.0, 0.0],
                LocalPotentialBoundary::Robin {
                    reference,
                    exchange,
                } if reference.is_finite() && exchange.is_finite() && exchange > 0.0 => {
                    [3.0, reference as f32, exchange as f32, 0.0]
                }
                _ => return Err(invalid("gpu_local_boundary_parameters")),
            };
        }
        let mut source_bytes = Vec::with_capacity(source.len() * 4);
        for value in source {
            let narrowed = *value as f32;
            if !narrowed.is_finite() {
                return Err(invalid("gpu_potential_source_precision"));
            }
            source_bytes.extend_from_slice(&narrowed.to_le_bytes());
        }
        let mut constraint_bytes = Vec::with_capacity(constraints.len() * 16);
        for site in constraints {
            for value in site {
                if !value.is_finite() {
                    return Err(invalid("gpu_local_boundary_precision"));
                }
                constraint_bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        let mut parameters = Vec::with_capacity(28);
        for word in [
            self.shape.nx() as u32,
            self.shape.ny() as u32,
            self.shape.nz() as u32,
            (alpha as f32).to_bits(),
            (screening as f32).to_bits(),
            u32::from(boundary == PotentialBoundary::Grounded),
            (dx as f32).to_bits(),
        ] {
            parameters.extend_from_slice(&word.to_le_bytes());
        }
        let raw = self.device.gpu_driver_device();
        raw.write_buffer(&self.previous, &vec![0_u8; self.shape.len() * 4]);
        raw.write_buffer(&self.next, &vec![0_u8; self.shape.len() * 4]);
        raw.write_buffer(&self.sources, &source_bytes);
        raw.write_buffer(&self.constraints, &constraint_bytes);
        raw.write_buffer(&self.parameters, &parameters);
        let mut commands = raw.create_command_buffer();
        for iteration in 0..iterations {
            commands.dispatch_compute_with_bind_group(
                &self.pipeline,
                if iteration % 2 == 0 { &self.forward } else { &self.reverse },
                (self.shape.len() as u32).div_ceil(64),
            );
        }
        commands.submit();
        let last = if iterations % 2 == 0 {
            &self.previous
        } else {
            &self.next
        };
        let result = self.runtime.block_on(self.device.read_buffer(last));
        if result.len() != self.shape.len() * 4 {
            return Err(invalid("gpu_potential_readback_shape"));
        }
        let output: Vec<f64> = result
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64)
            .collect();
        if output.iter().any(|value| !value.is_finite()) {
            return Err(invalid("gpu_potential_nonfinite"));
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use manonica_physics::PotentialField;

    /// LAW-GPU-POT-001: GPU Jacobi must reproduce screened Poisson
    /// reference including fixed, isolated and reservoir constraints.
    #[test]
    fn screened_potential_matches_cpu_with_all_boundary_laws() {
        if std::env::var_os("CI").is_some()
            && std::env::var("MANONICA_GPU_TEST").as_deref() != Ok("1")
        {
            return;
        }
        let shape = GridShape::new(8, 6, 4).unwrap();
        let mut source = vec![0.0; shape.len()];
        source[shape.index(2, 3, 1).unwrap()] = -7.0;
        source[shape.index(4, 2, 2).unwrap()] = 3.0;
        let sites = [
            LocalBoundarySite {
                index: shape.index(1, 1, 1).unwrap(),
                condition: LocalPotentialBoundary::Dirichlet { value: -0.75 },
            },
            LocalBoundarySite {
                index: shape.index(6, 3, 2).unwrap(),
                condition: LocalPotentialBoundary::Neumann,
            },
            LocalBoundarySite {
                index: shape.index(2, 4, 1).unwrap(),
                condition: LocalPotentialBoundary::Robin {
                    reference: 0.5,
                    exchange: 0.35,
                },
            },
        ];
        let cpu = PotentialField::solve_with_local(
            shape, 0.5, &source, 1.2, 0.9,
            PotentialBoundary::Grounded, 140, &sites,
        ).unwrap();
        let mut gpu = GpuPotential::new(shape).expect("GPU device required for physics parity");
        let values = gpu.solve(
            0.5, &source, 1.2, 0.9, PotentialBoundary::Grounded, 140, &sites,
        ).unwrap();
        let max_error = values.iter().zip(cpu.values())
            .map(|(a,b)| (a-b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_error < 3e-4, "GPU screened potential reference error: {max_error}");
    }
}
