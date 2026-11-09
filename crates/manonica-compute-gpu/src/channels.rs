//! Ordered conservative channel transfers, preserving the reference semantics.
use crate::kernel::Kernel;
use meridian_gpu_wgpu::{Buffer, Device};
pub(super) fn kernel(device: &Device, buffers: &[&Buffer]) -> Kernel {
    Kernel::new(
        device,
        "manonica-material-channels",
        include_str!("channels.wgsl"),
        "channels",
        buffers,
    )
}
