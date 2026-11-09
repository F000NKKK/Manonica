//! Sparse parallel thermal edges with reference-equivalent ordering.
use crate::schedule::OrderedPass;
use manonica_simulation::CoupledState;
use meridian_gpu_wgpu::{Buffer, Device};
pub(super) fn kernel(
    device: &Device,
    buffers: [&Buffer; 4],
    state: &CoupledState,
    conductance: f64,
) -> Option<OrderedPass> {
    if conductance == 0.0 {
        return None;
    }
    let [cells, materials, params, status] = buffers;
    let shape = state.field.shape();
    let mut edges = Vec::new();
    let mut payload = Vec::new();
    for z in 0..shape.nz() {
        for y in 0..shape.ny() {
            for x in 0..shape.nx() {
                let a = shape.index(x, y, z).unwrap();
                if state.materials.cells()[a].material.is_none() {
                    continue;
                }
                for (xx, yy, zz) in [(x + 1, y, z), (x, y + 1, z), (x, y, z + 1)] {
                    if let Some(b) = shape.index(xx, yy, zz)
                        && state.materials.cells()[b].material.is_some()
                    {
                        edges.push((a as u32, b as u32));
                        payload.extend_from_slice(&(a as u32).to_le_bytes());
                        payload.extend_from_slice(&(b as u32).to_le_bytes());
                    }
                }
            }
        }
    }
    OrderedPass::new(
        device,
        "manonica-thermal",
        include_str!("thermal.wgsl"),
        "thermal",
        &[cells, materials, params],
        &[],
        status,
        shape.len(),
        &edges,
        &payload,
        8,
    )
}
