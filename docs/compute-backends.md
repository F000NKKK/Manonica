# Compute backends

Manonica has two physical backends: the f64 CPU reference and the resident
Meridian/WGPU solver. `auto` tries GPU first and allows CPU fallback;
`--backend gpu` requires the complete GPU solver and never silently falls back.
Rendering and experiment configuration do not define physical laws.

## Ownership and reuse

| Component | Responsibility |
| --- | --- |
| `manonica-physics` | Typed fields, reference diffusion/advection and screened potential laws |
| `manonica-materials` | Traps, finite capacity, channels, thermal storage and material laws |
| `manonica-simulation` | Coupled state, step parameters, CPU operator split and energy ledger |
| `manonica-compute-gpu` | Resident state, compute kernels, device reductions, readback validation and rollback |
| `manonica-simulator` | Numerical experiments and comparisons without a window |
| `manonica-lab` | Rasterize mathematical constructions, configure the shared solver and display snapshots |

The simulation and GPU APIs do not depend on Lab, TOML, Minecraft or windowing.
A game adapter can supply the same `CoupledState` and `StepParameters`.
`lib.rs` in the GPU crate is a barrel. Physical operators have matching Rust
and WGSL modules; device ownership and wire validation have separate modules.

## Resident operator split

`GpuCoupled` runs every enabled operator of the coupled Lab solver:

| Operation | Execution |
| --- | --- |
| Screened potential and local Dirichlet/Neumann/Robin conditions | GPU Jacobi at setup |
| Diffusion | One invocation per cell |
| Uniform or vortex medium advection | One invocation per cell, conservative upwind faces |
| Potential-driven drift | Parallel outgoing-flux bounds and device-side CFL subdivision; parallel cell updates |
| Trap capture/release, saturation and binding heat | One invocation per material cell |
| Material channels | Independent edges run concurrently; shared endpoints preserve reference order |
| Heat conduction | Sparse dependency-preserving edge layers |
| Cooling | Parallel cells, explicit exported-heat contributions |
| Energy, inventory, peak and capacity audits | Hierarchical workgroup reductions |
| Source shutdown and work/heat budgets | Constant-size scalar decisions after reductions |

No invocation scans the entire grid to integrate drift, heat or energy.
Thermal/channel layers are constructed once from immutable geometry. Within
one layer no two exchanges write the same cell. Layer dependencies preserve
the sequential reference's bounded, conservative exchange semantics.

At setup, Lab prepares geometry and validates the CPU reference problem; the
GPU adapter solves the initial potential on its compute device. Buffers,
pipelines, dependency schedules and bind groups are allocated during setup,
not rebuilt for each physics step. Free density, bound storage, heat, potential
and the ledger remain on the device across batches. Initialization may transfer
potential data; stepping has no intermediate operator readbacks.

Lab submits bounded intervals at its observation cadence (50 ms). It reads a
single packed snapshot after the interval, rather than after each physical law
or timestep. Catch-up is capped so an overdue simulation cannot submit an
unbounded GPU queue that starves rendering. Pause keeps resident state; reset
and experiment replacement reconstruct it.

The public `GpuDiffusion` adapter also remains available for independent
single-law diffusion/uniform-flow experiments. Lab's GPU mode uses
`GpuCoupled`, including materials and energy, rather than that partial adapter.

## Correctness and failure

CPU arithmetic is f64; WGSL storage and arithmetic are f32. Parity is numerical,
not bitwise. Device audits reject nonfinite output, unstable transport, negative
storage, excess capacity, inadequate external power and conservation failures.
Readback undergoes an independent host check before replacing the snapshot.
Representation-level capacity overshoot is corrected conservatively when
constructing the typed host snapshot.

Each batch retains a device backup. A rejected batch restores device storage
and leaves the host snapshot and simulation time unchanged. Automatic fallback
replays every timestep of that interval on the CPU reference. Forced GPU mode
terminates on initialization or execution failure. CPU mode initializes no
compute device.

## Running and measuring

```bash
cargo run --release --manifest-path apps/manonica-lab/Cargo.toml -- \
  apps/manonica-lab/experiments/local_boundary_conductor.toml --backend gpu

cargo run --release -p manonica-simulator --features gpu --bin gpu-coupled -- \
  --grid 32 24 32 --steps 120 --batch 3
```

The headless command executes one warm-up step before timing, compares every
cell and material store against the independent CPU reference, checks the
ledger and reports batch p50/p95 and mean time per physical step. Validation
failure produces a nonzero exit status. Smaller diffusion/transport comparisons
are available through `--bin gpu-transport`.

Use release builds and identical grids, timesteps and enabled laws when
comparing performance. Software Vulkan validates real shader execution but
its timings do not predict performance on an RTX/AMD/Intel GPU. Changing the
frame rate or skipping physical operators is not a valid solver speedup.

F3 reports:

- `SIM STEP`: wall time per simulated step, including synchronization.
- `GPU BATCH`: complete batch wall time and its actual timestep count.
- `RENDER WALL`: rendering wall time, including driver/presentation waits.
- `FIELD CPU`: background volume-atlas sampling time.

The graphics surface uses FIFO VSync. No extra software timer runs by default;
`MANONICA_RENDER_FPS` enables an optional submission limit only when it is
below the detected monitor refresh limit. Redundant timers are omitted.

## Validation

```bash
cargo test --workspace
MANONICA_GPU_TEST=1 cargo test --manifest-path crates/manonica-compute-gpu/Cargo.toml
cargo test --manifest-path apps/manonica-lab/Cargo.toml
cargo test --manifest-path apps/manonica-lab/Cargo.toml gpu_tests -- --ignored --nocapture
cargo craftcheck check
cargo craftcheck deps
```

GPU integration covers coupled CPU/GPU parity, thermal and binding storage,
shared channels, local boundaries, saturation/shutdown, multiple drift substeps,
partial workgroups, multi-level reductions and failed-batch rollback. Explicit
Lab integration checks the physical construction example and worker reset.
CI installs software Vulkan and runs these checks; target hardware performance
still requires measurement on that device.

Native input/shutdown smoke validation uses a real X11 window:

```bash
cargo build --manifest-path apps/manonica-lab/Cargo.toml
xvfb-run -a python3 scripts/smoke_lab_x11.py apps/manonica-lab/target/debug/manonica-lab apps/manonica-lab/experiments/local_boundary_conductor.toml
```

When using `CARGO_TARGET_DIR`, pass that directory's `debug/manonica-lab` instead.
The smoke test checks main/keypad Enter, rejected edits and closing during
configuration setup. CI installs Xvfb, xauth and xdotool for it. Raw frame traces
are enabled through `MANONICA_FRAME_TRACE`; see the Lab README for profiling.

## Dynamic spectral mechanics

The separate Lab `dynamic_mechanics.toml` scenario uses `GpuSpectral` or
`SpectralMechanics`, with the same `gpu`/`auto`/`cpu` selection policy.
`GpuSpectral` keeps body positions/momenta and field amplitudes/canonical
momenta resident. Each kick has independent body and mode invocations;
drift kernels update bodies and modes. All six dispatches execute for each
accepted timestep. A batch performs no intermediate operator readbacks.

CPU prepares validated initial spectral geometry/equilibrium and uploads once.
Packed snapshots are read for observation, not to execute the next physical
operator. Snapshot validation checks finite values, immutable metadata,
and total momentum/energy against the initial closed-system baseline.
The f32 audit budgets are `1e-4*(1+abs(P_initial_component))` and
`1e-3*(1+abs(E_initial))`; they include numerical integration error and
are rejection budgets, not corrections or parity guarantees. Dedicated
CPU/GPU tests enforce 1e-4 body and 1e-5 modal agreement over 100 steps.

Every batch backs up both buffers. Rejection restores device buffers and
retains the host snapshot/time. Auto fallback replays all rejected steps
on CPU; forced GPU terminates. Tests verify actual byte-identical device
rollback after nonfinite shader output, alongside parity and Lab device
reset/replacement. Software Vulkan executes these shaders in CI; it does
not establish RTX performance. The current GPU scenario is an autonomous
closed body/field system, without a configured external load.
