@group(0) @binding(0) var<storage,read> summary: array<Summary>;
@group(0) @binding(1) var<storage,read> params: array<f32>;
@group(0) @binding(2) var<storage,read_write> ledger: array<f32>;
@group(0) @binding(3) var<storage,read_write> status: atomic<u32>;
@compute @workgroup_size(1)
fn initialize() {
    if (atomicLoad(&status)!=0u) {return;}
    let s=summary[0];ledger[0]=s.a.x;ledger[1]=s.a.y;ledger[9]=s.a.w;
    ledger[10]=s.a.z+ledger[0]+ledger[1]+ledger[2]+ledger[6]-ledger[3];
    if (!finite(ledger[10]) || !finite(ledger[9])) {atomicOr(&status,64u);}
    if (params[5]<=0.0) {atomicOr(&status,64u);}
    ledger[14]=1.0;
}
@compute @workgroup_size(1)
fn begin() {
    if (atomicLoad(&status)!=0u) {return;}
    let s=summary[0];ledger[8]=s.a.x+s.a.y;ledger[4]=0.0;ledger[14]=1.0;
    if (params[8]!=0.0 && params[11]>0.0 && s.b.y>0.0 && s.b.x/s.b.y>=params[11]) {ledger[7]=1.0;}
    if (ledger[7]!=0.0 && params[8]!=0.0) {ledger[14]=exp(-params[12]*params[5]);ledger[13]*=ledger[14];}
}
@compute @workgroup_size(1)
fn finish() {
    if (atomicLoad(&status)!=0u) {return;}
    let s=summary[0];ledger[0]=s.a.x;ledger[1]=s.a.y;ledger[4]=s.c.x;ledger[5]=s.b.z;ledger[6]+=s.b.w;
    if (params[8]!=0.0) {
        let change=ledger[0]+ledger[1]-ledger[8];
        if (change<0.0) {ledger[2]-=change;}
        else {
            let used=min(change,ledger[2]);let supplied_work=max(0.0,change-used);
            if (supplied_work>params[10]*params[5]+1e-5) {atomicOr(&status,32u);return;}
            ledger[2]-=used;ledger[3]+=supplied_work;
        }
        let loss=ledger[2]*(1.0-exp(-params[7]*params[5]));ledger[2]-=loss;ledger[6]+=loss;
    }
    let energy=s.a.z+ledger[0]+ledger[1]+ledger[2]+ledger[6]-ledger[3];
    let scale=max(1.0,abs(s.a.z)+abs(ledger[0])+abs(ledger[1])+abs(ledger[2])+abs(ledger[6])+abs(ledger[3]));
    if (!finite(s.a.w) || !finite(energy) || abs(s.a.w-ledger[9])>max(1.0,ledger[9])*0.0005 || abs(energy-ledger[10])>scale*0.002) {atomicOr(&status,128u);}
    ledger[11]+=1.0;
}
