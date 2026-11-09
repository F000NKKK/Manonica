@group(0) @binding(0) var<storage,read_write> cells: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> scratch: array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> params: array<f32>;
@group(0) @binding(3) var<storage,read> local: array<vec4<f32>>;
@group(0) @binding(4) var<storage,read_write> ledger: array<f32>;
@group(0) @binding(5) var<storage,read_write> status: atomic<u32>;
fn fraction(a: u32,b: u32) -> f32 {
    return -params[9]*(cells[b].w-cells[a].w)*params[5]/(params[3]*params[3]);
}
@compute @workgroup_size(1)
fn drift() {
    if (atomicLoad(&status)!=0u || params[8]==0.0) { return; }
    let dims=vec3<u32>(u32(params[0]),u32(params[1]),u32(params[2]));
    let n=arrayLength(&cells); let volume=params[3]*params[3]*params[3];
    for (var i=0u;i<n;i++) { scratch[i].z=0.0; }
    for (var a=0u;a<n;a++) {
        if (local[a].x==2.0) { continue; }
        for (var axis=0u;axis<3u;axis++) {
            let b=neighbour(a,axis,1,dims); if (b==0xffffffffu || local[b].x==2.0) { continue; }
            let f=fraction(a,b);
            if (!finite(f)) { atomicOr(&status,4u); return; }
            if (f>0.0) { scratch[a].z+=f; } else { scratch[b].z-=f; }
        }
    }
    var maximum=0.0; for (var i=0u;i<n;i++) { maximum=max(maximum,scratch[i].z); }
    let steps=max(1.0,ceil(maximum/0.9));
    if (!finite(steps) || steps>256.0) { atomicOr(&status,4u); return; }
    var transported=0.0;
    for (var step=0u;step<u32(steps);step++) {
        for (var i=0u;i<n;i++) { scratch[i].x=cells[i].x; scratch[i].y=cells[i].x; }
        for (var a=0u;a<n;a++) {
            if (local[a].x==2.0) { continue; }
            for (var axis=0u;axis<3u;axis++) {
                let b=neighbour(a,axis,1,dims); if (b==0xffffffffu || local[b].x==2.0) { continue; }
                let f=fraction(a,b)/steps;
                var donor=scratch[b].x; if (f>=0.0) { donor=scratch[a].x; }
                let amount=f*donor;
                scratch[a].y-=amount; scratch[b].y+=amount;
                transported+=abs(amount)*volume;
            }
        }
        for (var i=0u;i<n;i++) {
            if (!finite(scratch[i].y) || scratch[i].y < -1e-5) { atomicOr(&status,4u); return; }
            cells[i].x=max(0.0,scratch[i].y);
        }
    }
    ledger[4]=transported;
}
