//! Parallel conservative potential drift and device-side CFL subdivision.
use crate::{
    drift_plan, drift_prepare,
    kernel::{Kernel, copy, dispatch},
};
use manonica_simulation::{CoupledState, StepParameters};
use meridian_gpu_wgpu::{BindGroup, Buffer, BufferUsage, CommandBuffer, ComputePipeline, Device};
pub(super) struct Drift {
    prepare: Kernel,
    plan: Kernel,
    pipeline: ComputePipeline,
    steps: Vec<BindGroup>,
    groups: u32,
}
impl Drift {
    pub fn new(
        raw: &Device,
        buffers: [&Buffer; 7],
        state: &CoupledState,
        parameters: &StepParameters,
    ) -> Option<Self> {
        let potential = state.potential.as_ref()?;
        let mobility = parameters.field?.mobility;
        if mobility == 0.0 || potential.peak_absolute() == 0.0 {
            return None;
        }
        let [cells, scratch, params, local, ledger, metrics, status] = buffers;
        let upper = 12.0 * potential.peak_absolute() * mobility * parameters.dt
            / state.field.cell_width_metres().powi(2)
            / 0.9;
        let max_steps = (upper.ceil() + 1.0).clamp(1.0, 256.0) as u32;
        let groups = (state.field.shape().len() as u32).div_ceil(64);
        let partial = raw.create_buffer(groups as usize * 4, BufferUsage::Storage);
        let prepare =
            drift_prepare::kernel(raw, &[cells, params, local, metrics, &partial, status]);
        let limit = raw.create_buffer(4, BufferUsage::Storage);
        raw.write_buffer(&limit, &max_steps.to_le_bytes());
        let plan = drift_plan::kernel(raw, &[&partial, ledger, &limit, status]);
        let shader = raw.create_shader(
            "manonica-parallel-drift",
            &format!(
                "{}\n{}",
                include_str!("layout.wgsl"),
                include_str!("potential_drift.wgsl")
            ),
        );
        let pipeline = raw.create_compute_pipeline(&shader, "drift");
        let layout = pipeline.wgpu_pipeline().get_bind_group_layout(0);
        let steps = (0..max_steps)
            .map(|index| {
                let range = raw.create_buffer(4, BufferUsage::Storage);
                raw.write_buffer(&range, &index.to_le_bytes());
                raw.create_bind_group(
                    &layout,
                    &[
                        cells, scratch, params, local, ledger, metrics, &range, status,
                    ],
                )
            })
            .collect();
        Some(Self {
            prepare,
            plan,
            pipeline,
            steps,
            groups,
        })
    }
    pub fn record(&self, commands: &mut CommandBuffer<'_>, scratch: &Buffer, cells: &Buffer) {
        dispatch(commands, &self.prepare, self.groups);
        dispatch(commands, &self.plan, 1);
        for bind in &self.steps {
            commands.dispatch_compute_with_bind_group(&self.pipeline, bind, self.groups);
            copy(commands, scratch, cells, 0);
        }
    }
}
