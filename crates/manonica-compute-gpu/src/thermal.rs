//! Bounded thermal conduction and cooling into an explicit ambient reservoir.
use crate::kernel::Kernel;
use meridian_gpu_wgpu::{Buffer, Device};
pub(super) fn kernel(device: &Device, buffers: &[&Buffer]) -> Kernel {
    Kernel::new(
        device,
        "manonica-thermal",
        include_str!("thermal.wgsl"),
        "thermal",
        buffers,
    )
}
