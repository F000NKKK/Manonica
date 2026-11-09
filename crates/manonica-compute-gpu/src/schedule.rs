//! Dependency-preserving layers for ordered pairwise physical exchanges.
//! Independent edges run together; exchanges sharing a cell retain CPU order.
use meridian_gpu_wgpu::{BindGroup, Buffer, BufferUsage, CommandBuffer, ComputePipeline, Device};

pub(super) struct Schedule {
    pub order: Vec<usize>,
    pub ranges: Vec<[u32; 2]>,
}
impl Schedule {
    pub fn new(vertices: usize, edges: &[(u32, u32)]) -> Self {
        let mut next = vec![0usize; vertices];
        let mut layers: Vec<Vec<usize>> = Vec::new();
        for (i, &(a, b)) in edges.iter().enumerate() {
            let layer = next[a as usize].max(next[b as usize]);
            layers.resize_with(layers.len().max(layer + 1), Vec::new);
            layers[layer].push(i);
            next[a as usize] = layer + 1;
            next[b as usize] = layer + 1;
        }
        let mut order = Vec::with_capacity(edges.len());
        let mut ranges = Vec::new();
        for layer in layers {
            ranges.push([order.len() as u32, layer.len() as u32]);
            order.extend(layer);
        }
        Self { order, ranges }
    }
}
pub(super) struct OrderedPass {
    pipeline: ComputePipeline,
    binds: Vec<(BindGroup, u32)>,
}
impl OrderedPass {
    #[allow(
        clippy::too_many_arguments,
        reason = "static pipeline resources and dependency schedule are independent inputs"
    )]
    pub fn new(
        raw: &Device,
        label: &str,
        shader: &str,
        entry: &str,
        prefix: &[&Buffer],
        suffix: &[&Buffer],
        status: &Buffer,
        vertices: usize,
        edges: &[(u32, u32)],
        payload: &[u8],
        stride: usize,
    ) -> Option<Self> {
        if edges.is_empty() {
            return None;
        }
        let schedule = Schedule::new(vertices, edges);
        let mut ordered = Vec::with_capacity(edges.len() * stride);
        for i in schedule.order {
            ordered.extend_from_slice(&payload[i * stride..(i + 1) * stride]);
        }
        let edge_buffer = raw.create_buffer(ordered.len(), BufferUsage::Storage);
        raw.write_buffer(&edge_buffer, &ordered);
        let shader =
            raw.create_shader(label, &format!("{}\n{shader}", include_str!("layout.wgsl")));
        let pipeline = raw.create_compute_pipeline(&shader, entry);
        let layout = pipeline.wgpu_pipeline().get_bind_group_layout(0);
        let binds = schedule
            .ranges
            .iter()
            .map(|range| {
                let buffer = raw.create_buffer(8, BufferUsage::Storage);
                let bytes = range
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>();
                raw.write_buffer(&buffer, &bytes);
                let mut buffers = prefix.to_vec();
                buffers.push(&edge_buffer);
                buffers.extend_from_slice(suffix);
                buffers.extend([&buffer, status]);
                (
                    raw.create_bind_group(&layout, &buffers),
                    range[1].div_ceil(64),
                )
            })
            .collect();
        Some(Self { pipeline, binds })
    }
    pub fn record(&self, commands: &mut CommandBuffer<'_>) {
        for (bind, groups) in &self.binds {
            commands.dispatch_compute_with_bind_group(&self.pipeline, bind, *groups);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn exchange(values: &mut [f64], a: usize, b: usize) {
        let difference = values[a] - values[b];
        let amount = difference * 0.2;
        values[a] -= amount;
        values[b] += amount;
    }
    #[test]
    fn parallel_layers_preserve_overlapping_exchange_semantics() {
        let edges = [(0, 1), (2, 3), (1, 2), (0, 3), (4, 5), (3, 4), (2, 5)];
        let schedule = Schedule::new(6, &edges);
        let mut serial = [10.0, 0.0, 2.0, 8.0, 1.0, 3.0];
        let mut parallel = serial;
        for &(a, b) in &edges {
            exchange(&mut serial, a as usize, b as usize);
        }
        for [start, count] in schedule.ranges {
            let mut used = [false; 6];
            // Reverse independent operations to test commutativity rather
            // than relying on the original insertion order inside a layer.
            for &i in schedule.order[start as usize..(start + count) as usize]
                .iter()
                .rev()
            {
                let (a, b) = edges[i];
                assert!(!used[a as usize] && !used[b as usize]);
                used[a as usize] = true;
                used[b as usize] = true;
                exchange(&mut parallel, a as usize, b as usize);
            }
        }
        assert_eq!(serial, parallel);
    }
}
