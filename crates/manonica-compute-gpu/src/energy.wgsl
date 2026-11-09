@group(0) @binding(0) var<storage,read> cells: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> materials: array<Material>;
@group(0) @binding(2) var<storage,read> params: array<f32>;
@group(0) @binding(3) var<storage,read> local: array<vec4<f32>>;
@group(0) @binding(4) var<storage,read> ledger: array<f32>;
@group(0) @binding(5) var<storage,read> metrics: array<vec2<f32>>;
@group(0) @binding(6) var<storage,read_write> partial: array<Summary>;
@group(0) @binding(7) var<storage,read_write> status: atomic<u32>;
var<workgroup> sum_a: array<vec4<f32>,64>;
var<workgroup> sum_b: array<vec4<f32>,64>;
var<workgroup> sum_c: array<vec4<f32>,64>;
fn contribution(i: u32) -> Summary {
    let c=cells[i];let m=materials[i];let dx=params[3];let volume=dx*dx*dx;
    let dims=vec3<u32>(u32(params[0]),u32(params[1]),u32(params[2]));
    let p=xyz(i,dims.x,dims.y);var field=0.0;
    if (params[8]!=0.0) {
        field=0.5*params[23]*c.w*c.w*volume;
        for (var axis=0u;axis<3u;axis++) {
            let j=neighbour(i,axis,1,dims);
            if (j!=0xffffffffu) {
                if (local[i].x!=2.0 && local[j].x!=2.0) {let delta=cells[j].w-c.w;field+=0.5*params[22]*delta*delta*dx;}
            } else if (params[24]!=0.0) {field+=0.5*params[22]*c.w*c.w*dx;}
            if (params[24]!=0.0 && p[axis]==0u) {field+=0.5*params[22]*c.w*c.w*dx;}
        }
        if (local[i].x==3.0) {let delta=c.w-local[i].y*ledger[13];field+=0.5*local[i].z/dx*delta*delta*volume;}
    }
    if (!finite(c.x) || !finite(c.y) || !finite(c.z) || !finite(c.w) || c.x < -1e-5 || c.y < -1e-5 || c.z < -1e-5) {atomicOr(&status,64u);}
    return Summary(vec4(field,(c.x*volume+c.y)*c.w,c.z-c.y*m.binding,c.x*volume+c.y),vec4(c.y,m.capacity,abs(c.w),metrics[i].x),vec4(metrics[i].y,0.0,0.0,0.0));
}
@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    var value=Summary(vec4(0.0),vec4(0.0),vec4(0.0));
    if (gid.x<arrayLength(&cells)) {value=contribution(gid.x);}
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
    if (lane==0u) {partial[group.x]=Summary(sum_a[0],sum_b[0],sum_c[0]);}
}
