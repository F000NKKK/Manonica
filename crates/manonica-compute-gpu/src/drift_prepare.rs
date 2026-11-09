//! Parallel adaptive drift scheduling without host readbacks.
use crate::kernel::Kernel;
use meridian_gpu_wgpu::{Buffer, Device};
pub(super) fn kernel(device: &Device, buffers: &[&Buffer]) -> Kernel {
    Kernel::new(
        device,
        "manonica-drift_prepare",
        include_str!("drift_prepare.wgsl"),
        "prepare",
        buffers,
    )
}
