//! Scalar reservoir decisions after parallel physical reductions.
use crate::kernel::Kernel;
use meridian_gpu_wgpu::{Buffer, Device};
pub(super) fn kernel(device: &Device, entry: &str, buffers: &[&Buffer]) -> Kernel {
    Kernel::new(
        device,
        "manonica-energy-control",
        include_str!("energy_control.wgsl"),
        entry,
        buffers,
    )
}
