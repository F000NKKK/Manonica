@group(0) @binding(0) var<storage,read_write> cells: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> links: array<Channel>;
@group(0) @binding(2) var<storage,read> params: array<f32>;
@group(0) @binding(3) var<storage,read_write> status: atomic<u32>;
// Channels may share endpoints. Ordered execution is intentional: the CPU
// reference bounds every transfer using the result of preceding channels.
@compute @workgroup_size(1)
fn channels() {
    if (atomicLoad(&status)!=0u) { return; }
    let volume=params[3]*params[3]*params[3];
    for (var k=0u;k<u32(params[25]);k++) {
        let ch=links[k]; let fa=cells[ch.a].x*volume; let fb=cells[ch.b].x*volume;
        let proposed=ch.conductance*ch.area/ch.length*(fa-fb)/volume*params[5];
        var donor=fb; if (proposed>=0.0) { donor=fa; }
        let amount=min(min(min(abs(proposed),ch.max_flux*params[5]),donor),abs(fa-fb)*0.5);
        if (!finite(amount)) { atomicOr(&status,8u); return; }
        let transfer=sign(proposed)*amount;
        cells[ch.a].x=(fa-transfer)/volume; cells[ch.b].x=(fb+transfer)/volume;
    }
}
