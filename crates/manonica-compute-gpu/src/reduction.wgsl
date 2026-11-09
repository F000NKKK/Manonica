@group(0) @binding(0) var<storage,read> input: array<Summary>;
@group(0) @binding(1) var<storage,read_write> output: array<Summary>;
@group(0) @binding(2) var<storage,read> range: array<u32>;
var<workgroup> sum_a: array<vec4<f32>,64>;
var<workgroup> sum_b: array<vec4<f32>,64>;
var<workgroup> sum_c: array<vec4<f32>,64>;
@compute @workgroup_size(64)
fn fold(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    var value=Summary(vec4(0.0),vec4(0.0),vec4(0.0));
    if (gid.x<range[0]) {value=input[gid.x];}
    sum_a[lane]=value.a;sum_b[lane]=value.b;sum_c[lane]=value.c;
    workgroupBarrier();
    for (var stride=32u;stride>0u;stride/=2u) {
        if (lane<stride) {
            sum_a[lane]+=sum_a[lane+stride];
            let peak=max(sum_b[lane].z,sum_b[lane+stride].z);
            sum_b[lane]+=sum_b[lane+stride];sum_b[lane].z=peak;
            sum_c[lane]+=sum_c[lane+stride];
        }
        workgroupBarrier();
    }
    if (lane==0u) {output[group.x]=Summary(sum_a[0],sum_b[0],sum_c[0]);}
}
