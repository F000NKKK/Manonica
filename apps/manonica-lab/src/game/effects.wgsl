// 3D particle trajectories. Every output is a volumetric shard, not a billboard.
struct Particle {position_size:vec4<f32>};
@group(0) @binding(0) var<storage,read_write> particles:array<Particle>;
fn random(n:f32)->f32 {return fract(sin(n*127.1+311.7)*43758.5453);}
@compute @workgroup_size(64)
fn bake(@builtin(global_invocation_id) id:vec3<u32>) {
 let i=id.x;if i>=3u*64u*384u{return;}let family=i/(64u*384u);let frame=(i/384u)%64u;let index=i%384u;let seed=f32(index)+0.5;let t=(f32(frame)+0.5)/64.0;
 let a=random(seed)*6.2831853;let b=random(seed+35.0)*2.0-1.0;let c=random(seed+73.0);var p=vec3(0.0);var size=0.02;
 if family==0u {
  let radial=sqrt(max(0.0,1.0-b*b));let velocity=vec3(cos(a)*radial,b,sin(a)*radial);let expansion=t*(0.4+c*1.6);
  p=velocity*expansion+vec3(0.0,0.2+t*0.6-t*t*0.7,0.0);size=(0.008+c*0.026)*(1.0-t);
 } else if family==1u {
  let delay=c*0.65;let age=clamp((t-delay)*3.0,0.0,1.0);p=vec3((random(seed+2.0)-0.5)*(0.3+age),sin(age*3.14159)*(0.12+random(seed+9.0)*0.4),c*2.0-1.0);
  size=(0.015+random(seed+21.0)*0.04)*sin(age*3.14159);
 } else {
  let h=fract(c+t*2.0);let radius=(0.12+h*0.75)*(0.7+random(seed+11.0)*0.3);let angle=a+t*12.56637+h*10.0;
  p=vec3(cos(angle)*radius,h,sin(angle)*radius);size=(0.008+random(seed+9.0)*0.018)*(1.0-h*0.5);
 }
 particles[i].position_size=vec4(p,size);
}
