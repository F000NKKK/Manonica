//! Local conservative trap kinetics: capture, release, saturation and heat.
use crate::kernel::Kernel;
use meridian_gpu_wgpu::{Buffer, Device};
pub(super) fn kernel(device: &Device, buffers: &[&Buffer]) -> Kernel {
    Kernel::new(
        device,
        "manonica-material-exchange",
        include_str!("material_exchange.wgsl"),
        "exchange",
        buffers,
    )
}
