//! Hierarchical workgroup reductions; no invocation scans the entire grid.
use crate::kernel::Kernel;
use meridian_gpu_wgpu::{Buffer, BufferUsage, CommandBuffer, Device};
pub(super) struct Reduction {
    pub buffers: Vec<Buffer>,
    folds: Vec<(Kernel, u32)>,
}
impl Reduction {
    pub fn new(raw: &Device, count: u32) -> Self {
        let mut buffers = vec![raw.create_buffer(count as usize * 48, BufferUsage::Storage)];
        let mut folds = Vec::new();
        let mut remaining = count;
        while remaining > 1 {
            let groups = remaining.div_ceil(64);
            let output = raw.create_buffer(groups as usize * 48, BufferUsage::Storage);
            let range = raw.create_buffer(4, BufferUsage::Storage);
            raw.write_buffer(&range, &remaining.to_le_bytes());
            let fold = Kernel::new(
                raw,
                "manonica-physical-reduction",
                include_str!("reduction.wgsl"),
                "fold",
                &[buffers.last().unwrap(), &output, &range],
            );
            folds.push((fold, groups));
            buffers.push(output);
            remaining = groups;
        }
        Self { buffers, folds }
    }
    pub fn first(&self) -> &Buffer {
        &self.buffers[0]
    }
    pub fn last(&self) -> &Buffer {
        self.buffers.last().unwrap()
    }
    pub fn record(&self, commands: &mut CommandBuffer<'_>) {
        for (kernel, groups) in &self.folds {
            commands.dispatch_compute_with_bind_group(&kernel.pipeline, &kernel.bind, *groups);
        }
    }
}
