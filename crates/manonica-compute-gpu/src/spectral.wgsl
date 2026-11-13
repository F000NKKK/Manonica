struct Body { position:vec4<f32>, momentum:vec4<f32>, properties:vec4<f32> }
struct Mode { wave:vec4<f32>, state:vec4<f32> }
struct Config { dt:f32, inverse_inertia:f32, bodies:u32, modes:u32 }
@group(0) @binding(0) var<storage,read_write> bodies:array<Body>;
@group(0) @binding(1) var<storage,read_write> modes:array<Mode>;
@group(0) @binding(2) var<storage,read> config:Config;
fn valid()->bool { return arrayLength(&bodies)>0u && arrayLength(&modes)>0u && config.bodies>0u && config.modes>0u; }
@compute @workgroup_size(64)
fn kick_bodies(@builtin(global_invocation_id) id:vec3<u32>) {
 if !valid() || id.x>=config.bodies {return;}
 let b=bodies[id.x]; var force=vec3<f32>(0.0);
 for(var j=0u;j<config.modes;j++) {
  let m=modes[j]; let phase=dot(m.wave.xyz,b.position.xyz);
  let factor=exp(-0.5*b.properties.y*b.properties.y*dot(m.wave.xyz,m.wave.xyz));
  force+=b.properties.x*factor*m.wave.xyz*(m.state.x*sin(phase)-m.state.y*cos(phase));
 }
 bodies[id.x].momentum=vec4<f32>(b.momentum.xyz+0.5*config.dt*force,b.momentum.w);
}
@compute @workgroup_size(64)
fn kick_modes(@builtin(global_invocation_id) id:vec3<u32>) {
 if !valid() || id.x>=config.modes {return;}
 let m=modes[id.x]; var ca=0.0;var cb=0.0;
 for(var i=0u;i<config.bodies;i++) {
  let b=bodies[i];let phase=dot(m.wave.xyz,b.position.xyz);
  let factor=exp(-0.5*b.properties.y*b.properties.y*dot(m.wave.xyz,m.wave.xyz));
  ca+=b.properties.x*factor*cos(phase);cb+=b.properties.x*factor*sin(phase);
 }
 modes[id.x].state=vec4<f32>(m.state.xy,m.state.zw-0.5*config.dt*(m.wave.w*m.state.xy/config.inverse_inertia+vec2<f32>(ca,cb)));
}
@compute @workgroup_size(64)
fn drift_bodies(@builtin(global_invocation_id) id:vec3<u32>) {
 if !valid() || id.x>=config.bodies {return;}
 let b=bodies[id.x];
 bodies[id.x].position=vec4<f32>(b.position.xyz+config.dt*b.momentum.xyz/b.position.w,b.position.w);
}
@compute @workgroup_size(64)
fn drift_modes(@builtin(global_invocation_id) id:vec3<u32>) {
 if !valid() || id.x>=config.modes {return;}
 let s=modes[id.x].state;
 modes[id.x].state=vec4<f32>(s.xy+config.dt*config.inverse_inertia*s.zw,s.zw);
}
