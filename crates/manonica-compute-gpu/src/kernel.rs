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

pub(super) fn dispatch(
    commands: &mut meridian_gpu_wgpu::CommandBuffer<'_>,
    kernel: &Kernel,
    groups: u32,
) {
    commands.dispatch_compute_with_bind_group(&kernel.pipeline, &kernel.bind, groups);
}
pub(super) fn copy(
    commands: &mut meridian_gpu_wgpu::CommandBuffer<'_>,
    source: &Buffer,
    target: &Buffer,
    offset: u64,
) {
    commands.encoder_mut().copy_buffer_to_buffer(
        source.wgpu_buffer(),
        0,
        target.wgpu_buffer(),
        offset,
        source.byte_len as u64,
    );
}
