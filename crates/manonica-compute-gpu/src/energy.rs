//! Device operator for energy ledger.
use crate::kernel::Kernel;
use meridian_gpu_wgpu::{Buffer, Device};
pub(super) fn kernel(device: &Device, entry: &str, buffers: &[&Buffer]) -> Kernel {
    Kernel::new(
        device,
        "manonica-energy-ledger",
        include_str!("energy.wgsl"),
        entry,
        buffers,
    )
}
