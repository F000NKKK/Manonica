//! Device operator for resident transport.
use crate::kernel::Kernel;
use meridian_gpu_wgpu::{Buffer, Device};
pub(super) fn kernel(device: &Device, entry: &str, buffers: &[&Buffer]) -> Kernel {
    Kernel::new(
        device,
        "manonica-resident-transport",
        include_str!("coupled_transport.wgsl"),
        entry,
        buffers,
    )
}
