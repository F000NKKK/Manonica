@group(0) @binding(0) var<storage,read_write> cells: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> materials: array<Material>;
@group(0) @binding(2) var<storage,read> params: array<f32>;
@group(0) @binding(3) var<storage,read_write> ledger: array<f32>;
@group(0) @binding(4) var<storage,read_write> status: atomic<u32>;
@compute @workgroup_size(1)
fn thermal() {
    if (atomicLoad(&status)!=0u) { return; }
    let dims=vec3<u32>(u32(params[0]),u32(params[1]),u32(params[2]));
    // Match the reference lexicographic face order; overlapping exchanges
    // are sequential to preserve bounded temperature equilibration.
    for (var a=0u;a<arrayLength(&cells);a++) {
        let ma=materials[a]; if (ma.enabled==0.0) { continue; }
        for (var axis=0u;axis<3u;axis++) {
            let b=neighbour(a,axis,1,dims); if (b==0xffffffffu || materials[b].enabled==0.0) { continue; }
            let mb=materials[b];
            let ta=ma.reference+cells[a].z/ma.heat_capacity;
            let tb=mb.reference+cells[b].z/mb.heat_capacity;
            let delta=ta-tb;
            let equilibrium=abs(delta)/(1.0/ma.heat_capacity+1.0/mb.heat_capacity);
            var donor=cells[b].z; if (delta>0.0) { donor=cells[a].z; }
            let amount=min(min(params[6]*abs(delta)*params[5],equilibrium),donor);
            if (!finite(amount) || amount<0.0) { atomicOr(&status,16u); return; }
            let transfer=sign(delta)*amount;
            cells[a].z-=transfer; cells[b].z+=transfer;
        }
    }
    let fraction=1.0-exp(-params[7]*params[5]);
    var exported=0.0;
    for (var i=0u;i<arrayLength(&cells);i++) {
        if (materials[i].enabled!=0.0) {
            let amount=cells[i].z*fraction; cells[i].z-=amount; exported+=amount;
        }
    }
    ledger[6]+=exported;
    if (!finite(ledger[6])) { atomicOr(&status,16u); }
}
