//! Device operator for potential drift.
use crate::kernel::Kernel;
use meridian_gpu_wgpu::{Buffer, Device};
pub(super) fn kernel(device: &Device, entry: &str, buffers: &[&Buffer]) -> Kernel {
    Kernel::new(
        device,
        "manonica-potential-drift",
        include_str!("potential_drift.wgsl"),
        entry,
        buffers,
    )
}
