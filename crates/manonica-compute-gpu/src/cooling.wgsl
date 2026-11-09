@group(0) @binding(0) var<storage,read_write> cells: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> materials: array<Material>;
@group(0) @binding(2) var<storage,read> params: array<f32>;
@group(0) @binding(3) var<storage,read_write> metrics: array<vec2<f32>>;
@group(0) @binding(4) var<storage,read_write> status: atomic<u32>;
@compute @workgroup_size(64)
fn cooling(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i=gid.x;if (i>=arrayLength(&cells) || atomicLoad(&status)!=0u) {return;}
    var amount=0.0;
    if (materials[i].enabled!=0.0) {amount=cells[i].z*(1.0-exp(-params[7]*params[5]));cells[i].z-=amount;}
    metrics[i].x=amount;
    if (!finite(amount)) {atomicOr(&status,16u);}
}
