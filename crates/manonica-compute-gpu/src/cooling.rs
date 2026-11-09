//! Parallel cooling, with per-cell exported heat reduced on the device.
use crate::kernel::Kernel;
use meridian_gpu_wgpu::{Buffer, Device};
pub(super) fn kernel(device: &Device, buffers: &[&Buffer]) -> Kernel {
    Kernel::new(
        device,
        "manonica-cooling",
        include_str!("cooling.wgsl"),
        "cooling",
        buffers,
    )
}
