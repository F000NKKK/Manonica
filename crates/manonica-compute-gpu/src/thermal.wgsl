@group(0) @binding(0) var<storage,read_write> cells: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> materials: array<Material>;
@group(0) @binding(2) var<storage,read> params: array<f32>;
@group(0) @binding(3) var<storage,read> edges: array<vec2<u32>>;
@group(0) @binding(4) var<storage,read> range: array<u32>;
@group(0) @binding(5) var<storage,read_write> status: atomic<u32>;
@compute @workgroup_size(64)
fn thermal(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x>=range[1] || atomicLoad(&status)!=0u) {return;}
    let edge=edges[range[0]+gid.x];let a=edge.x;let b=edge.y;
    let ma=materials[a];let mb=materials[b];
    // Subtract reference baselines first to preserve small thermal gradients.
    let delta=(ma.reference-mb.reference)+(cells[a].z/ma.heat_capacity-cells[b].z/mb.heat_capacity);
    let equilibrium=abs(delta)/(1.0/ma.heat_capacity+1.0/mb.heat_capacity);
    var donor=cells[b].z;if (delta>0.0) {donor=cells[a].z;}
    let amount=min(min(params[6]*abs(delta)*params[5],equilibrium),donor);
    if (!finite(amount) || amount<0.0) {atomicOr(&status,16u);return;}
    let transfer=sign(delta)*amount;
    cells[a].z-=transfer;cells[b].z+=transfer;
}
