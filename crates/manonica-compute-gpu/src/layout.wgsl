// Resident Cell = (free density, bound count, heat J, potential J/manon).
// Parameters: nx,ny,nz,dx,D,dt,thermal,cooling,has_field,mobility,
// max_power,shutdown_fraction,decay,flow_kind,vx/ox,vy/oy,vz/oz,
// axis_x,axis_y,axis_z,angular,axial,kappa,screening,grounded,channels.
struct Material {
    enabled: f32, capacity: f32, capture: f32, release: f32, binding: f32,
    heat_capacity: f32, reference: f32, activation: f32, max_flux: f32, padding: f32,
}
struct Channel { a: u32, b: u32, length: f32, area: f32, conductance: f32, max_flux: f32 }
fn finite(v: f32) -> bool { return v == v && abs(v) <= 3.402823e38; }
fn xyz(i: u32, nx: u32, ny: u32) -> vec3<u32> { return vec3(i % nx, (i / nx) % ny, i / (nx * ny)); }
fn neighbour(i: u32, axis: u32, side: i32, dims: vec3<u32>) -> u32 {
    let p=xyz(i,dims.x,dims.y);
    if ((side < 0 && p[axis] == 0u) || (side > 0 && p[axis]+1u >= dims[axis])) { return 0xffffffffu; }
    let stride=array<u32,3>(1u,dims.x,dims.x*dims.y);
    if (side < 0) { return i-stride[axis]; }
    return i+stride[axis];
}

// Hierarchical physical reductions. a=(field,manon,material,count),
// b=(bound,capacity,peak,exported), c=(transported,0,0,0).
struct Summary { a: vec4<f32>, b: vec4<f32>, c: vec4<f32> }
