//! Small pipeline resource owner shared by the coupled operators.
use meridian_gpu_wgpu::{BindGroup, Buffer, ComputePipeline, Device};
pub(super) struct Kernel {
    pub(super) pipeline: ComputePipeline,
    pub(super) bind: BindGroup,
}
impl Kernel {
    pub(super) fn new(
        device: &Device,
        label: &str,
        source: &str,
        entry: &str,
        buffers: &[&Buffer],
    ) -> Self {
        let source = format!("{}\n{source}", include_str!("layout.wgsl"));
        let shader = device.create_shader(label, &source);
        let pipeline = device.create_compute_pipeline(&shader, entry);
        let layout = pipeline.wgpu_pipeline().get_bind_group_layout(0);
        let bind = device.create_bind_group(&layout, buffers);
        Self { pipeline, bind }
    }
}
