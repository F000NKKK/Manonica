// 3D screened Poisson/Helmholtz Jacobi iteration.
// (-permittivity * Laplacian + screening) phi = source.
// Physically imposed local Dirichlet, Neumann and Robin constraints.
// One invocation per cell; ping-pong avoids read/write races.
@group(0) @binding(0) var<storage, read> previous: array<f32>;
@group(0) @binding(1) var<storage, read_write> next: array<f32>;
@group(0) @binding(2) var<storage, read> sources: array<f32>;
@group(0) @binding(3) var<storage, read> local: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> params: array<u32>;

// local[i]: (kind, value_or_reference, exchange, unused)
// kind 0=none, 1=Dirichlet, 2=Neumann, 3=Robin.
// params: nx, ny, nz, bitcast(alpha), bitcast(screening),
//         grounded(1/0), bitcast(dx)
@compute @workgroup_size(64)
fn potential_jacobi(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let nx = params[0];
    let ny = params[1];
    let nz = params[2];
    let i = invocation.x;
    if (i >= nx * ny * nz) { return; }

    let condition = local[i];
    if (condition.x == 1.0) {
        next[i] = condition.y;
        return;
    }

    let alpha = bitcast<f32>(params[3]);
    let screening = bitcast<f32>(params[4]);
    let grounded = params[5] == 1u;
    let dx = bitcast<f32>(params[6]);
    let plane = nx * ny;
    let x = i % nx;
    let y = (i / nx) % ny;
    let z = i / plane;
    var sum = 0.0;
    var count = 0.0;
    // A Neumann cell blocks both sides of each embedded interface.
    let isolated = condition.x == 2.0;

    if (x > 0u) {
        let j = i - 1u;
        if (!isolated && local[j].x != 2.0) {
            sum += previous[j];
            count += 1.0;
        }
    } else if (grounded) { count += 1.0; }
    if (x + 1u < nx) {
        let j = i + 1u;
        if (!isolated && local[j].x != 2.0) {
            sum += previous[j];
            count += 1.0;
        }
    } else if (grounded) { count += 1.0; }
    if (y > 0u) {
        let j = i - nx;
        if (!isolated && local[j].x != 2.0) {
            sum += previous[j];
            count += 1.0;
        }
    } else if (grounded) { count += 1.0; }
    if (y + 1u < ny) {
        let j = i + nx;
        if (!isolated && local[j].x != 2.0) {
            sum += previous[j];
            count += 1.0;
        }
    } else if (grounded) { count += 1.0; }
    if (z > 0u) {
        let j = i - plane;
        if (!isolated && local[j].x != 2.0) {
            sum += previous[j];
            count += 1.0;
        }
    } else if (grounded) { count += 1.0; }
    if (z + 1u < nz) {
        let j = i + plane;
        if (!isolated && local[j].x != 2.0) {
            sum += previous[j];
            count += 1.0;
        }
    } else if (grounded) { count += 1.0; }

    var exchange = 0.0;
    var reference = 0.0;
    if (condition.x == 3.0) {
        exchange = condition.z / dx;
        reference = condition.y;
    }
    next[i] = (sources[i] + alpha * sum + exchange * reference)
        / (screening + alpha * count + exchange);
}
