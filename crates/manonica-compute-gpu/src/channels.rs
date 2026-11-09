//! Parallel layers of channels, preserving dependent transfer order.
use crate::schedule::OrderedPass;
use manonica_simulation::CoupledState;
use meridian_gpu_wgpu::{Buffer, Device};
pub(super) fn kernel(
    device: &Device,
    cells: &Buffer,
    params: &Buffer,
    status: &Buffer,
    state: &CoupledState,
    payload: &[u8],
) -> Option<OrderedPass> {
    let shape = state.field.shape();
    let edges = state
        .materials
        .channels()
        .iter()
        .map(|c| {
            (
                shape.index(c.from[0], c.from[1], c.from[2]).unwrap() as u32,
                shape.index(c.to[0], c.to[1], c.to[2]).unwrap() as u32,
            )
        })
        .collect::<Vec<_>>();
    OrderedPass::new(
        device,
        "manonica-channels",
        include_str!("channels.wgsl"),
        "channels",
        &[cells],
        &[params],
        status,
        shape.len(),
        &edges,
        payload,
        24,
    )
}
