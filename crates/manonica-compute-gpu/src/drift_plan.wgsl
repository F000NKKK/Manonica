@group(0) @binding(0) var<storage,read> partial: array<f32>;
@group(0) @binding(1) var<storage,read_write> ledger: array<f32>;
@group(0) @binding(2) var<storage,read> range: array<u32>;
@group(0) @binding(3) var<storage,read_write> status: atomic<u32>;
var<workgroup> maxima: array<f32,64>;
@compute @workgroup_size(64)
fn plan(@builtin(local_invocation_index) lane: u32) {
    var value=0.0;
    for (var i=lane;i<arrayLength(&partial);i+=64u) {value=max(value,partial[i]);}
    maxima[lane]=value;workgroupBarrier();
    for (var stride=32u;stride>0u;stride/=2u) {if (lane<stride) {maxima[lane]=max(maxima[lane],maxima[lane+stride]);}workgroupBarrier();}
    if (lane==0u) {
        let steps=max(1.0,ceil(maxima[0]/0.9));
        if (!finite(steps) || steps>256.0 || steps>f32(range[0])) {atomicOr(&status,4u);}
        ledger[12]=steps;
    }
}
