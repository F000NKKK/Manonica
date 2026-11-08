// Conservative first-order upwind advection by a uniform medium velocity.
// Every interface contributes opposite terms to its two adjacent cells.
// All external faces are impermeable; no atomic writes are required.
@group(0) @binding(0) var<storage, read> input_density: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_density: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
// nx, ny, nz, bitcast(vx*dt/dx), bitcast(vy*dt/dx), bitcast(vz*dt/dx)
@compute @workgroup_size(64)
fn advect_uniform(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let nx = params[0];
    let ny = params[1];
    let nz = params[2];
    let i = invocation.x;
    if (i >= nx * ny * nz) { return; }
    let plane = nx * ny;
    let x = i % nx;
    let y = (i / nx) % ny;
    let z = i / plane;
    let cx = bitcast<f32>(params[3]);
    let cy = bitcast<f32>(params[4]);
    let cz = bitcast<f32>(params[5]);
    let middle = input_density[i];
    var result = middle;

    if (cx >= 0.0) {
        if (x + 1u < nx) { result -= cx * middle; }
        if (x > 0u) { result += cx * input_density[i - 1u]; }
    } else {
        if (x > 0u) { result += cx * middle; }
        if (x + 1u < nx) { result -= cx * input_density[i + 1u]; }
    }
    if (cy >= 0.0) {
        if (y + 1u < ny) { result -= cy * middle; }
        if (y > 0u) { result += cy * input_density[i - nx]; }
    } else {
        if (y > 0u) { result += cy * middle; }
        if (y + 1u < ny) { result -= cy * input_density[i + nx]; }
    }
    if (cz >= 0.0) {
        if (z + 1u < nz) { result -= cz * middle; }
        if (z > 0u) { result += cz * input_density[i - plane]; }
    } else {
        if (z > 0u) { result += cz * middle; }
        if (z + 1u < nz) { result -= cz * input_density[i + plane]; }
    }
    output_density[i] = max(0.0, result);
}
