@group(0) @binding(0) var<storage,read> cells: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> next: array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> params: array<f32>;
@group(0) @binding(3) var<storage,read_write> status: atomic<u32>;
fn velocity(position: vec3<f32>) -> vec3<f32> {
    if (params[13]==0.0) { return vec3(0.0); }
    let v=vec3(params[14],params[15],params[16]);
    if (params[13]==1.0) { return v; }
    let axis=normalize(vec3(params[17],params[18],params[19]));
    return params[20]*cross(axis,position-v)+params[21]*axis;
}
@compute @workgroup_size(64)
fn diffuse(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if (i>=arrayLength(&cells) || atomicLoad(&status)!=0u) { return; }
    let dims=vec3<u32>(u32(params[0]),u32(params[1]),u32(params[2]));
    let alpha=params[4]*params[5]/(params[3]*params[3]);
    if (alpha>1.0/6.0 || alpha<0.0 || !finite(alpha)) { atomicOr(&status,1u); return; }
    var c=cells[i]; var sum=0.0;
    for (var axis=0u;axis<3u;axis++) {
        for (var side=-1;side<=1;side+=2) {
            let j=neighbour(i,axis,side,dims); if (j!=0xffffffffu) { sum+=cells[j].x-c.x; }
        }
    }
    c.x+=alpha*sum;
    if (!finite(c.x) || c.x < -1e-5) { atomicOr(&status,1u); return; }
    next[i]=c;
}
@compute @workgroup_size(64)
fn advect(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if (i>=arrayLength(&cells) || atomicLoad(&status)!=0u) { return; }
    let dims=vec3<u32>(u32(params[0]),u32(params[1]),u32(params[2]));
    let center=(vec3<f32>(xyz(i,dims.x,dims.y))+vec3(0.5)-vec3<f32>(dims)*0.5)*params[3];
    var c=cells[i]; var density=c.x; var outgoing=0.0;
    for (var axis=0u;axis<3u;axis++) {
        for (var side=-1;side<=1;side+=2) {
            let j=neighbour(i,axis,side,dims); if (j==0xffffffffu) { continue; }
            var face=center; face[axis]+=f32(side)*0.5*params[3];
            let fraction=velocity(face)[axis]*params[5]/params[3]*f32(side);
            var donor=cells[j].x; if (fraction>=0.0) { donor=c.x; outgoing+=fraction; }
            density-=fraction*donor;
        }
    }
    c.x=density;
    if (!finite(density) || density < -1e-5 || !finite(outgoing) || outgoing>1.0) { atomicOr(&status,2u); return; }
    next[i]=c;
}
