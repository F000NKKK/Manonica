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

## Current computational coverage

The **only** GPU-accelerated physical law so far is 3D free-manon
diffusion, `J=-D grad(n)`, with an explicit finite-volume six-face
stencil and zero flux through domain boundaries.

Material capture/release, heat flow, potential solve, potential drift,
conduction channels and energy auditing **remain on CPU**. The
simulation continues to enforce material and manon conservation and
transactional updates; GPU numerical results are checked, mass-adjusted
for f32 roundoff and committed to a validated physics-core field.

A new `GpuDiffusion::evolve_resident_batch` provides a ping-pong
device-resident path for **diffusion-only** numerical experiments:
many explicit diffusion steps are dispatched in one submission with one
initial upload and one final readback. Its output is checked against
the CPU reference in `resident_diffusion_batch_matches_cpu_reference`.
The main Lab still uses the full coupled operator-split simulation with
materials and heat, which requires a CPU exchange after each step until
those operators are ported and fused. Thus the Lab still performs one
upload/dispatch/readback per physics step. On **small grids** this can
be slower than the CPU. A PhysX-like performance target requires
resident buffers, fused kernels, sparse updates, asynchronous transfers,
material locality and representative profiling, not merely a GPU switch.

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
- `PHYSICS GPU HYBRID`: diffusion on GPU, remaining physics on CPU.
- `PHYSICS CPU FALLBACK`: GPU was requested but could not initialize.
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
