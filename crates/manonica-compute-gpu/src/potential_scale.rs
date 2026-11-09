//! Parallel potential amplitude decay after source shutdown.
use crate::kernel::Kernel;
use meridian_gpu_wgpu::{Buffer, Device};
pub(super) fn kernel(device: &Device, buffers: &[&Buffer]) -> Kernel {
    Kernel::new(
        device,
        "manonica-potential-scale",
        include_str!("potential_scale.wgsl"),
        "scale_potential",
        buffers,
    )
}
