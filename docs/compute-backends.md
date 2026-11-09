# Manonica compute architecture and GPU acceleration

The physical model is deliberately independent of a graphics API.
Rendering never determines the numerical outcome of physical experiments.

## Crates and ownership

| Component | Responsibilities |
| --- | --- |
| `manonica-physics` | Scalar typed physical quantities, conservation, diffusion, advection, potential, material kinetics and reference numerical laws |
| `manonica-compute-gpu` | Optional Meridian/WGPU compute adapter, validated field readback, GPU/CPU parity checks; no window |
| `manonica-lab` | Assemble TOML experiments, own the simulation loop and F3 display; no physical laws encoded into the renderer |
| Fabric mod (future) | Use the same physical core and choose a CPU or GPU backend at runtime; no dependency on the Lab UI |

The core exports `FreeManaDiffusionBackend`. The GPU crate implements
this neutral contract without exposing `wgpu` types in core.
A future server-side/headless Minecraft implementation can use CPU-only
physics without a GPU, or install an accelerator when suitable.

There are exactly **two physical compute backends**: CPU and GPU.
`auto` selects between them; `FALLBACK` is telemetry, not an
additional backend. Forcing `--backend gpu` prevents fallback.

## Current computational coverage

GPU-accelerated operators include 3D free-manon
diffusion, `J=-D grad(n)`, with an explicit finite-volume six-face
stencil and zero flux through domain boundaries, and conservative upwind
advection by a uniform medium. A standalone GPU screened-potential solver
supports local Dirichlet, Neumann and Robin constraints; Lab still uses the
CPU potential and energy cycle.

Material capture/release, heat flow, potential solve, potential drift,
conduction channels and energy auditing **remain on CPU**. The
simulation continues to enforce material and manon conservation and
transactional updates; GPU numerical results are checked, mass-adjusted
for f32 roundoff and committed to a validated physics-core field.

`GpuDiffusion::evolve_resident_batch` executes diffusion-only intervals.
`GpuDiffusion::evolve_uniform_flow_batch` executes diffusion followed by
uniform advection at every step. Both paths use persistent ping-pong buffers,
one initial upload, one command submission, and one final readback. Numerical
validation and inventory correction happen before committing the final field.
Lab coalesces overdue steps for these scenarios only when materials, traps,
channels and potential interactions are absent. Vortex flow and coupled
multi-physics scenarios still require individual transactional steps.

Run a headless, checked 100-step transport comparison in the simulator:

```bash
cargo run -p manonica-simulator --features gpu --bin gpu-transport
```

The command requires a compute adapter, compares every cell against the
independent f64 CPU laws and exits unsuccessfully if accuracy or conservation
checks fail. Software Vulkan can validate shader execution, but its timing
does not measure performance on a physical GPU. The default simulator and
core tests do not build the optional graphics dependencies.

## Selecting the backend

Backend selection is **automatic by default**: Manonica attempts GPU first,
then uses the CPU reference when the GPU adapter cannot initialize or the
GPU diffusion operator fails. A forced selection never silently switches
devices.

Automatic selection:

```bash
cargo run --release --manifest-path apps/manonica-lab/Cargo.toml -- \
  apps/manonica-lab/experiments/potential_concentrator.toml
```

Force GPU or fail with a nonzero exit code:

```bash
cargo run --release --manifest-path apps/manonica-lab/Cargo.toml -- \
  apps/manonica-lab/experiments/potential_concentrator.toml --backend gpu
```

Force CPU, even if a GPU is available:

```bash
cargo run --release --manifest-path apps/manonica-lab/Cargo.toml -- \
  apps/manonica-lab/experiments/potential_concentrator.toml --backend cpu
```

Supported command line modes: `--backend auto|gpu|cpu`,
`--backend=gpu` and environment variable `MANONICA_SIM_BACKEND`.
An explicit CLI flag overrides the environment. Invalid values abort
before creating the application window.

The default command below also uses auto-selection (GPU-first):


```bash
cargo run --release --manifest-path apps/manonica-lab/Cargo.toml -- \
  apps/manonica-lab/experiments/potential_concentrator.toml
```

Force GPU diffusion through the environment:

```bash
MANONICA_SIM_BACKEND=gpu cargo run --release -- \
  --manifest-path apps/manonica-lab/Cargo.toml -- \
  apps/manonica-lab/experiments/potential_concentrator.toml
```

In `auto` mode, failure to initialize the GPU or to advance
diffusion switches to the CPU reference, retrying the same transactional
physics step. In forced `gpu` mode, either failure terminates the entire
process with a nonzero exit code. In forced `cpu` mode, no GPU compute
adapter is initialized.

### F3 telemetry

- `PHYSICS CPU`: CPU physical solver (GPU not requested).
- `PHYSICS GPU`: selected GPU computation (currently only the diffusion operator runs on GPU; the complete solver remains hybrid).
- `PHYSICS CPU (FALLBACK)`: the automatic policy selected the CPU after GPU unavailability/failure. This is a status, **not a third backend**.
- `DIFF WALL`: total wall-clock duration of the GPU diffusion
  operation, **including** upload, submission, readback and validation.
  It is **not** GPU-only execution time.
- `SIM STEP`: full simulation wall time for one step (the previous
  `SIM CPU` label was misleading when GPU compute blocks on readback).
- `RENDER CPU` and `FIELD CPU`: graphic-side work, **not** the
  time spent computing physics on the GPU.

Do not estimate speedup from FPS. Verify CPU and GPU with equal
initial conditions, grid size and timestep using `SIM STEP`
and the dedicated benchmark.

## Verification and benchmarking

```bash
cargo test --manifest-path crates/manonica-compute-gpu/Cargo.toml
cargo test --manifest-path apps/manonica-lab/Cargo.toml
cargo test --workspace
```

A hardware-dependent parity test compares GPU results cell by cell to
the independent f64 CPU solver, checking total count and output shape.
GitHub-hosted CI intentionally skips device parity unless
`MANONICA_GPU_TEST=1`; a green CI result alone is **not proof** of
correctness on a specific NVIDIA/AMD/Intel device.

On a physical GPU, explicitly run the parity test:

```bash
MANONICA_GPU_TEST=1 cargo test \
  --manifest-path crates/manonica-compute-gpu/Cargo.toml \
  gpu_diffusion_matches_cpu_when_an_adapter_exists -- --nocapture
```

To profile a local adapter, run the ignored 32³ x 100-step benchmark:

```bash
cargo test --manifest-path crates/manonica-compute-gpu/Cargo.toml \
  --release benchmark_gpu_vs_cpu_including_transfer_overhead -- --ignored --nocapture
```

Record measured CPU time, GPU end-to-end time and maximum cell error.
Prefer CPU at sizes where transfer overhead dominates.

## Roadmap, in order

1. Persistent device-side free field, fused diffusion/gradient/advection
   kernels with physical invariants and CPU/GPU parity tests.
2. Material/trap state on device with bounded conservative transfers,
   heat, energy and saturation tests.
3. Device-side source deposition and numerical potential solver with
   independently checked residuals and boundary conditions.
4. Coalesced multi-step execution and snapshots only when needed by
   simulation observability / rendering. This is required to eliminate
   routine GPU→CPU roundtrips.
5. Adaptive load balancing, CPU fallback and reproducibility
   requirements appropriate for a Minecraft dedicated server.

No new physical law may be added without a reference simulation test
and conservation/energy audit.
