// Explicit 3D six-face diffusion with impermeable external boundaries.
// The same finite-volume stencil as FreeDiffusion CPU reference:
// n_next = n + (D dt / dx^2) * sum_faces(n_neighbour - n).
@group(0) @binding(0) var<storage, read> old_density: array<f32>;
@group(0) @binding(1) var<storage, read_write> new_density: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

@compute @workgroup_size(64)
fn diffusion(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let x_size = params[0];
    let y_size = params[1];
    let z_size = params[2];
    let alpha = bitcast<f32>(params[3]);
    let length = x_size * y_size * z_size;
    let i = invocation.x;
    if (i >= length) { return; }
    let slice = x_size * y_size;
    let x = i % x_size;
    let y = (i / x_size) % y_size;
    let z = i / slice;
    let middle = old_density[i];
    var laplace = 0.0;
    if (x > 0u) { laplace += old_density[i - 1u] - middle; }
    if (x + 1u < x_size) { laplace += old_density[i + 1u] - middle; }
    if (y > 0u) { laplace += old_density[i - x_size] - middle; }
    if (y + 1u < y_size) { laplace += old_density[i + x_size] - middle; }
    if (z > 0u) { laplace += old_density[i - slice] - middle; }
    if (z + 1u < z_size) { laplace += old_density[i + slice] - middle; }
    new_density[i] = max(0.0, middle + alpha * laplace);
}
