@group(0) @binding(0) var<storage,read_write> cells: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> ledger: array<f32>;
@group(0) @binding(2) var<storage,read_write> status: atomic<u32>;
@compute @workgroup_size(64)
fn scale_potential(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x>=arrayLength(&cells) || atomicLoad(&status)!=0u) {return;}
    cells[gid.x].w*=ledger[14];
}
