//! Graphics-thread lifecycle, presentation and overlay composition.
use super::frame_trace::{FrameSample, FrameTrace};
use super::{InputBridge, lock_bridge};
use crate::{
    diagnostics::{DebugHud, FrameMetrics},
    editor::{EditorOverlay, FormulaLegend},
    experiment::Experiment,
    visualization::FieldView,
};
use meridian_sdk::{Device, FlyCamera, GraphicsSession, Vec3, Window, compose_frame};
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

/// FIFO already paces presentation. A software limit is opt-in only.
fn configured_render_rate(window: &Window) -> Option<f64> {
    let requested = std::env::var("MANONICA_RENDER_FPS")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|fps| fps.is_finite() && (30.0..=360.0).contains(fps))?;
    let refresh = window
        .surface_target()
        .current_monitor()
        .and_then(|monitor| monitor.refresh_rate_millihertz())
        .filter(|rate| *rate > 0)
        .map(|rate| f64::from(rate) / 1000.0);
    // A requested upper bound at or above FIFO's existing limit needs no
    // second timer, even when the environment variable is explicitly set.
    if refresh.is_some_and(|rate| requested >= rate) {
        None
    } else {
        Some(requested)
    }
}

/// Paces frame submission, retaining FIFO presentation as a separate guarantee.
fn pace_frame(deadline: &mut Instant, interval: Duration) {
    let now = Instant::now();
    if now < *deadline {
        std::thread::sleep(*deadline - now);
    }
    *deadline += interval;
    if *deadline <= Instant::now() {
        *deadline = Instant::now() + interval;
    }
}

pub(super) fn render_loop(
    window: Window,
    mut experiment: Experiment,
    bridge: Arc<Mutex<InputBridge>>,
    running: Arc<AtomicBool>,
) {
    let runtime = tokio::runtime::Runtime::new().expect("Tokio runtime");
    let mut trace = FrameTrace::from_environment().unwrap_or_else(|error| {
        eprintln!("Cannot start frame trace: {error}");
        None
    });
    let frame_cap = configured_render_rate(&window);
    let interval = frame_cap.map(|fps| Duration::from_secs_f64(1.0 / fps));
    let size = [window.width().max(1), window.height().max(1)];
    let target = window.surface_target();
    let (device, mut surface) = runtime
        .block_on(Device::new_windowed(target, size[0], size[1]))
        .expect("failed to initialize Meridian WGPU device on rendering thread");
    // The SDK 0.8 source revision exposes explicit FIFO selection.
    // Do this *before* building GraphicsSession so its swapchain is paced
    // by vertical blanking rather than a second 120 Hz software timer.
    surface.enable_vsync(&device);
    let base = GraphicsSession::new(device, surface, size[0], size[1]);
    let mut graphics =
        FieldView::new(base, &experiment).expect("failed to initialize manon rendering resources");
    let mut debug = DebugHud::default();
    let mut editor_overlay = EditorOverlay::default();
    let mut formula_legend = FormulaLegend::default();
    let mut revision = 0_u64;
    let mut camera = FlyCamera::new(Vec3::new(0.0, 3.0, 18.0));
    let mut last_frame = Instant::now();
    let mut deadline = last_frame;
    let mut last_draw_wall_ms = 0.0;
    let mut initial_manons = lock_bridge(&bridge).snapshot.total;
    let mut current_size = size;
    if let Some(fps) = frame_cap {
        eprintln!("Manonica: FIFO VSync with configured limit {fps:.3} FPS");
    } else {
        eprintln!("Manonica: FIFO VSync; no additional software frame limiter");
    }

    while running.load(Ordering::Acquire) {
        if let Some(interval) = interval {
            pace_frame(&mut deadline, interval);
        }
        let frame_begin = Instant::now();
        let dt = frame_begin.duration_since(last_frame).as_secs_f64();
        last_frame = frame_begin;

        let input_start = Instant::now();
        let render_input = lock_bridge(&bridge).take_render_input();
        let input_ms = input_start.elapsed().as_secs_f64() * 1000.0;
        if render_input.revision != revision {
            experiment = (*render_input.experiment).clone();
            initial_manons = experiment.seeds.iter().map(|seed| seed.total_manons).sum();
            if let Err(err) = graphics.replace_formulas(&experiment) {
                eprintln!("failed to replace formula geometry: {err}");
            }
            revision = render_input.revision;
        }
        if render_input.size[0] == 0 || render_input.size[1] == 0 {
            thread::sleep(Duration::from_millis(50));
            last_frame = Instant::now();
            continue;
        }
        if render_input.size != current_size {
            graphics
                .base
                .resize(render_input.size[0], render_input.size[1]);
            current_size = render_input.size;
        }
        if render_input.mouse_locked {
            camera.update(render_input.controls, dt.min(0.1) as f32);
        }

        experiment.visualization.relative_threshold = render_input.threshold;
        experiment.visualization.wisps = render_input.wisps;
        let aspect = current_size[0] as f32 / current_size[1].max(1) as f32;
        graphics.scene.camera = camera.camera(aspect);

        let cpu_start = Instant::now();
        graphics.update(&render_input.snapshot, &experiment);
        let graphics_update_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;
        let field = &render_input.snapshot.field;
        let metrics = FrameMetrics {
            frame_ms: dt * 1000.0,
            sim_cpu_ms: render_input.snapshot.sim_cpu_ms,
            compute: render_input.snapshot.compute,
            draw_wall_ms: last_draw_wall_ms,
            field_cpu_ms: graphics.sampling_cpu_ms(),
            sim_time: render_input.snapshot.elapsed,
            steps: render_input.snapshot.steps,
            total: render_input.snapshot.total,
            bound: render_input.snapshot.bound,
            formula_bound: std::array::from_fn(|i| {
                render_input
                    .snapshot
                    .formula_bound
                    .get(i)
                    .copied()
                    .unwrap_or(0.0)
            }),
            formula_count: render_input.snapshot.formula_bound.len(),
            material_energy: render_input.snapshot.material_energy,
            potential_peak: render_input.snapshot.potential_peak,
            field_energy_joules: render_input.snapshot.field_energy_joules,
            field_heat_joules: render_input.snapshot.field_heat_joules,
            field_work_joules: render_input.snapshot.field_work_joules,
            potential_transport: render_input.snapshot.potential_transport,
            trap_temperature_kelvin: render_input.snapshot.trap_temperature_kelvin,
            exported_heat_joules: render_input.snapshot.exported_heat_joules,
            field_shutdown: render_input.snapshot.field_shutdown,
            equilibrium: render_input.snapshot.equilibrium,
            initial_total: initial_manons,
            peak_density: graphics.peak_density(),
            shown_slices: graphics.visible_slices(),
            grid_cells: field.shape().len(),
            paused: render_input.paused || render_input.snapshot.paused,
        };
        debug.record(metrics);
        if let Some(error) = &render_input.snapshot.error {
            eprintln!("Physics suspended: {error}");
            running.store(false, Ordering::Release);
        }
        let overlay_start = Instant::now();
        let mut overlays = Vec::new();
        if render_input.debug
            && let Some(overlay) =
                debug.overlay(&mut graphics.base, current_size, render_input.editor.open)
        {
            overlays.push(overlay);
        }
        if let Some(overlay) =
            editor_overlay.draw(&mut graphics.base, &render_input.editor, current_size)
        {
            overlays.push(overlay);
        }
        if let Some(overlay) = formula_legend.draw(
            &mut graphics.base,
            &experiment,
            &render_input.snapshot.formula_bound,
            current_size,
            render_input.editor.open,
        ) {
            overlays.push(overlay);
        }
        let overlays_ms = overlay_start.elapsed().as_secs_f64() * 1000.0;
        let draw_start = Instant::now();
        compose_frame(
            &mut graphics.base,
            &graphics.scene,
            &overlays,
            [0.003, 0.003, 0.012, 1.0],
            &window,
        );
        let draw_present_ms = draw_start.elapsed().as_secs_f64() * 1000.0;
        last_draw_wall_ms = frame_begin.elapsed().as_secs_f64() * 1000.0;
        if let Some(writer) = trace.as_mut()
            && let Err(error) = writer.record(FrameSample {
                interval_ms: dt * 1000.0,
                render_ms: last_draw_wall_ms,
                input_ms,
                update_ms: graphics_update_ms,
                overlays_ms,
                draw_present_ms,
                gpu_batch_ms: render_input.snapshot.compute.batch_wall_ms,
                batch_steps: render_input.snapshot.compute.batch_steps,
                simulation_steps: render_input.snapshot.steps,
            })
        {
            eprintln!("Frame trace stopped: {error}");
            trace = None;
        }
    }
    if let Some(writer) = trace.as_mut()
        && let Err(error) = writer.flush()
    {
        eprintln!("Cannot flush frame trace: {error}");
    }
}
