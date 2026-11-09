@group(0) @binding(0) var<storage,read> cells: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> params: array<f32>;
@group(0) @binding(2) var<storage,read> local: array<vec4<f32>>;
@group(0) @binding(3) var<storage,read_write> metrics: array<vec2<f32>>;
@group(0) @binding(4) var<storage,read_write> partial: array<f32>;
@group(0) @binding(5) var<storage,read_write> status: atomic<u32>;
var<workgroup> maxima: array<f32,64>;
@compute @workgroup_size(64)
fn prepare(@builtin(global_invocation_id) gid: vec3<u32>,@builtin(local_invocation_index) lane: u32,@builtin(workgroup_id) group: vec3<u32>) {
    let i=gid.x;var outgoing=0.0;
    if (i<arrayLength(&cells)) {
        metrics[i].y=0.0;
        if (local[i].x!=2.0) {
            let dims=vec3<u32>(u32(params[0]),u32(params[1]),u32(params[2]));
            for (var axis=0u;axis<3u;axis++) {
                for (var side=-1;side<=1;side+=2) {
                    let j=neighbour(i,axis,side,dims);if (j==0xffffffffu || local[j].x==2.0) {continue;}
                    let fraction=-params[9]*(cells[j].w-cells[i].w)*params[5]/(params[3]*params[3]);
                    if (!finite(fraction)) {atomicOr(&status,4u);}
                    outgoing+=max(0.0,fraction);
                }
            }
        }
    }
    maxima[lane]=outgoing;workgroupBarrier();
    for (var stride=32u;stride>0u;stride/=2u) {if (lane<stride) {maxima[lane]=max(maxima[lane],maxima[lane+stride]);}workgroupBarrier();}
    if (lane==0u) {partial[group.x]=maxima[0];}
}
