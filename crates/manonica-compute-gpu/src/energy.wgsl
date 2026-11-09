@group(0) @binding(0) var<storage,read_write> cells: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> materials: array<Material>;
@group(0) @binding(2) var<storage,read> params: array<f32>;
@group(0) @binding(3) var<storage,read> local: array<vec4<f32>>;
@group(0) @binding(4) var<storage,read_write> ledger: array<f32>;
@group(0) @binding(5) var<storage,read_write> status: atomic<u32>;
fn field_energy() -> f32 {
    if (params[8]==0.0) { return 0.0; }
    let dx=params[3]; let volume=dx*dx*dx;
    let dims=vec3<u32>(u32(params[0]),u32(params[1]),u32(params[2]));
    var total=0.0;
    for (var i=0u;i<arrayLength(&cells);i++) {
        let phi=cells[i].w; let p=xyz(i,dims.x,dims.y);
        total+=0.5*params[23]*phi*phi*volume;
        for (var axis=0u;axis<3u;axis++) {
            let j=neighbour(i,axis,1,dims);
            if (j!=0xffffffffu) {
                if (local[i].x!=2.0 && local[j].x!=2.0) { let delta=cells[j].w-phi; total+=0.5*params[22]*delta*delta*dx; }
            } else if (params[24]!=0.0) { total+=0.5*params[22]*phi*phi*dx; }
            if (params[24]!=0.0 && p[axis]==0u) { total+=0.5*params[22]*phi*phi*dx; }
        }
        if (local[i].x==3.0) {
            let delta=phi-local[i].y*ledger[13];
            total+=0.5*local[i].z/dx*delta*delta*volume;
        }
    }
    return total;
}
fn manon_energy() -> f32 {
    let volume=params[3]*params[3]*params[3]; var total=0.0;
    for (var i=0u;i<arrayLength(&cells);i++) { total+=(cells[i].x*volume+cells[i].y)*cells[i].w; }
    return total;
}
fn inventory() -> f32 {
    let volume=params[3]*params[3]*params[3]; var total=0.0;
    for (var i=0u;i<arrayLength(&cells);i++) { total+=cells[i].x*volume+cells[i].y; }
    return total;
}
fn material_energy() -> f32 {
    var total=0.0;
    for (var i=0u;i<arrayLength(&cells);i++) { total+=cells[i].z-cells[i].y*materials[i].binding; }
    return total;
}
@compute @workgroup_size(1)
fn initialize() {
    if (atomicLoad(&status)!=0u) { return; }
    ledger[0]=field_energy(); ledger[1]=manon_energy();
    ledger[9]=inventory();
    ledger[10]=material_energy()+ledger[0]+ledger[1]+ledger[2]+ledger[6]-ledger[3];
    if (!finite(ledger[10]) || !finite(ledger[9])) { atomicOr(&status,64u); }
}
@compute @workgroup_size(1)
fn begin() {
    if (atomicLoad(&status)!=0u) { return; }
    ledger[8]=field_energy()+manon_energy(); ledger[4]=0.0;
    var capacity=0.0; var bound=0.0;
    for (var i=0u;i<arrayLength(&cells);i++) { capacity+=materials[i].capacity; bound+=cells[i].y; }
    if (params[8]!=0.0 && params[11]>0.0 && capacity>0.0 && bound/capacity>=params[11]) { ledger[7]=1.0; }
    if (ledger[7]!=0.0 && params[8]!=0.0) {
        let decay=exp(-params[12]*params[5]);
        for (var i=0u;i<arrayLength(&cells);i++) { cells[i].w*=decay; }
        ledger[13]*=decay;
    }
}
@compute @workgroup_size(1)
fn finish() {
    if (atomicLoad(&status)!=0u) { return; }
    ledger[0]=field_energy(); ledger[1]=manon_energy();
    if (params[8]!=0.0) {
        let change=ledger[0]+ledger[1]-ledger[8];
        if (change<0.0) { ledger[2]-=change; }
        else {
            let used=min(change,ledger[2]); let supplied_work=max(0.0,change-used);
            if (supplied_work>params[10]*params[5]+1e-5) { atomicOr(&status,32u); return; }
            ledger[2]-=used; ledger[3]+=supplied_work;
        }
        let loss=ledger[2]*(1.0-exp(-params[7]*params[5])); ledger[2]-=loss; ledger[6]+=loss;
    }
    var peak=0.0;
    for (var i=0u;i<arrayLength(&cells);i++) {
        let c=cells[i]; peak=max(peak,abs(c.w));
        if (!finite(c.x) || !finite(c.y) || !finite(c.z) || !finite(c.w) || c.x < -1e-5 || c.y < -1e-5 || c.z < -1e-5) { atomicOr(&status,64u); return; }
    }
    ledger[5]=peak;
    let count=inventory();
    let energy=material_energy()+ledger[0]+ledger[1]+ledger[2]+ledger[6]-ledger[3];
    let scale=max(1.0,abs(material_energy())+abs(ledger[0])+abs(ledger[1])+abs(ledger[2])+abs(ledger[6])+abs(ledger[3]));
    if (!finite(count) || !finite(energy) || abs(count-ledger[9])>max(1.0,ledger[9])*0.0005 || abs(energy-ledger[10])>scale*0.002) { atomicOr(&status,128u); }
    ledger[11]+=1.0;
}
