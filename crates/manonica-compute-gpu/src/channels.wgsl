@group(0) @binding(0) var<storage,read_write> cells: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> links: array<Channel>;
@group(0) @binding(2) var<storage,read> params: array<f32>;
@group(0) @binding(3) var<storage,read> range: array<u32>;
@group(0) @binding(4) var<storage,read_write> status: atomic<u32>;
@compute @workgroup_size(64)
fn channels(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x>=range[1] || atomicLoad(&status)!=0u) {return;}
    let ch=links[range[0]+gid.x];let volume=params[3]*params[3]*params[3];
    let fa=cells[ch.a].x*volume;let fb=cells[ch.b].x*volume;
    let proposed=ch.conductance*ch.area/ch.length*(fa-fb)/volume*params[5];
    var donor=fb;if (proposed>=0.0) {donor=fa;}
    let amount=min(min(min(abs(proposed),ch.max_flux*params[5]),donor),abs(fa-fb)*0.5);
    if (!finite(amount)) {atomicOr(&status,8u);return;}
    let transfer=sign(proposed)*amount;
    cells[ch.a].x=(fa-transfer)/volume;cells[ch.b].x=(fb+transfer)/volume;
}
