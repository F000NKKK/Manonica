@group(0) @binding(0) var<storage,read_write> cells: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> materials: array<Material>;
@group(0) @binding(2) var<storage,read> params: array<f32>;
@group(0) @binding(3) var<storage,read_write> status: atomic<u32>;
@compute @workgroup_size(64)
fn exchange(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if (i>=arrayLength(&cells) || atomicLoad(&status)!=0u) { return; }
    let m=materials[i]; if (m.enabled==0.0) { return; }
    var c=cells[i]; let volume=params[3]*params[3]*params[3]; let dt=params[5];
    let free=c.x*volume;
    let temperature=m.reference+c.z/m.heat_capacity;
    let activation=exp(-m.activation*max(1.0/temperature-1.0/m.reference,0.0));
    let capture=min(min((m.capacity-c.y)*(1.0-exp(-m.capture*free/volume*dt)),free),m.max_flux*dt);
    var affordable=c.y;
    if (m.binding>0.0) { affordable=(c.z+capture*m.binding)/m.binding; }
    let release=min(min(c.y*(1.0-exp(-m.release*activation*dt)),m.max_flux*dt),affordable);
    c.x=(free-capture+release)/volume;
    c.y+=capture-release; c.z=max(0.0,c.z+(capture-release)*m.binding);
    if (!finite(c.x) || !finite(c.y) || !finite(c.z) || c.x < -1e-5 || c.y < -1e-5 || c.y > m.capacity+1e-4) {
        atomicOr(&status,8u); return;
    }
    cells[i]=c;
}
