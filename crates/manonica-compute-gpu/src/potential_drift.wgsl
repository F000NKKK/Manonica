@group(0) @binding(0) var<storage,read> cells: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> next: array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> params: array<f32>;
@group(0) @binding(3) var<storage,read> local: array<vec4<f32>>;
@group(0) @binding(4) var<storage,read> ledger: array<f32>;
@group(0) @binding(5) var<storage,read_write> metrics: array<vec2<f32>>;
@group(0) @binding(6) var<storage,read> range: array<u32>;
@group(0) @binding(7) var<storage,read_write> status: atomic<u32>;
@compute @workgroup_size(64)
fn drift(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i=gid.x;if (i>=arrayLength(&cells)) {return;}
    var c=cells[i];next[i]=c;
    if (atomicLoad(&status)!=0u || f32(range[0])>=ledger[12] || local[i].x==2.0) {return;}
    let dims=vec3<u32>(u32(params[0]),u32(params[1]),u32(params[2]));
    let volume=params[3]*params[3]*params[3];
    var change=0.0;var transported=0.0;
    for (var axis=0u;axis<3u;axis++) {
        for (var side=-1;side<=1;side+=2) {
            let j=neighbour(i,axis,side,dims);if (j==0xffffffffu || local[j].x==2.0) {continue;}
            let fraction=-params[9]*(cells[j].w-c.w)*params[5]/(params[3]*params[3]*ledger[12]);
            var donor=cells[j].x;if (fraction>=0.0) {donor=c.x;}
            let amount=fraction*donor;change-=amount;
            if (side>0) {transported+=abs(amount)*volume;}
        }
    }
    c.x=max(0.0,c.x+change);
    if (!finite(c.x) || !finite(change) || cells[i].x+change < -1e-5) {atomicOr(&status,4u);return;}
    next[i]=c;metrics[i].y+=transported;
}
