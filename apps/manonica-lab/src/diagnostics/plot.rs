//! Pure diagnostic sampling and CPU rasterization, independent of GPU resources.
use super::{FrameMetrics, HEIGHT, HISTORY, WIDTH, canvas::Canvas};
use meridian_sdk::ImageAsset;
use std::collections::VecDeque;
#[derive(Clone)]
pub(super) struct Plot {
    pub(super) fps_samples: VecDeque<f64>,
    pub(super) frame_samples: VecDeque<f64>,
    pub(super) peak_samples: VecDeque<f64>,
    pub(super) latest: Option<FrameMetrics>,
}
impl Default for Plot {
    fn default() -> Self {
        Self {
            fps_samples: VecDeque::with_capacity(HISTORY),
            frame_samples: VecDeque::with_capacity(HISTORY),
            peak_samples: VecDeque::with_capacity(HISTORY),
            latest: None,
        }
    }
}
fn record(history: &mut VecDeque<f64>, sample: f64) {
    if history.len() == HISTORY {
        history.pop_front();
    }
    history.push_back(sample);
}

impl Plot {
    pub(super) fn record(&mut self, metrics: FrameMetrics) {
        let fps = if metrics.timing.frame_ms > 0.000_01 {
            1000.0 / metrics.timing.frame_ms
        } else {
            0.0
        };
        record(&mut self.fps_samples, fps.clamp(0.0, 1000.0));
        record(
            &mut self.frame_samples,
            metrics.timing.frame_ms.clamp(0.0, 1000.0),
        );
        record(&mut self.peak_samples, metrics.volume.peak_density);
        self.latest = Some(metrics);
    }

    pub(super) fn fps(&self) -> f64 {
        if self.fps_samples.is_empty() {
            return 0.0;
        }
        // Mean throughput is frames / elapsed time, not mean(1 / frame_time).
        // The latter is biased high by short frames and exaggerates jitter.
        let sampled = self.frame_samples.iter().rev().take(60);
        let (milliseconds, count) =
            sampled.fold((0.0, 0_usize), |(sum, count), ms| (sum + ms, count + 1));
        if milliseconds > 0.0 {
            count as f64 * 1000.0 / milliseconds
        } else {
            0.0
        }
    }

    pub(super) fn render_image(&self, m: FrameMetrics) -> ImageAsset {
        let mut canvas = Canvas::new(WIDTH, HEIGHT);
        canvas.rect(0, 0, WIDTH, HEIGHT, [9, 16, 30, 226]);
        canvas.rect(0, 0, WIDTH, 29, [19, 40, 67, 247]);
        canvas.text(
            12,
            7,
            "MANONICA / LAB - DEBUG [F3]",
            1,
            [220, 243, 255, 255],
        );

        let state = if m.simulation.paused {
            "PAUSED"
        } else {
            "RUNNING"
        };
        canvas.text(
            12,
            39,
            &format!(
                "FPS {:>6.1}    FRAME {:>6.2} ms    {state}",
                self.fps(),
                m.timing.frame_ms
            ),
            1,
            [233, 245, 255, 255],
        );
        canvas.text(
            12,
            55,
            &format!(
                "SIM STEP {:>6.2} ms    RENDER WALL {:>6.2} ms",
                m.timing.sim_cpu_ms, m.timing.draw_wall_ms
            ),
            1,
            [170, 214, 245, 255],
        );
        canvas.text(
            12,
            71,
            &format!(
                "FIELD CPU {:>6.2} ms   SIM TIME {:>8.2} s  STEP {}",
                m.timing.field_cpu_ms, m.simulation.time_seconds, m.simulation.steps
            ),
            1,
            [170, 214, 245, 255],
        );
        canvas.text(
            12,
            87,
            &format!(
                "DENSITY MAX {:>11.3} /m3    LAYERS {:>2}   GRID {}",
                m.volume.peak_density, m.volume.shown_slices, m.volume.grid_cells
            ),
            1,
            [185, 232, 255, 255],
        );
        let delta = m.mana.total - m.mana.initial_total;
        canvas.text(
            12,
            103,
            &format!(
                "FREE {:>11.3}  BOUND {:>11.3}",
                m.mana.total - m.materials.bound,
                m.materials.bound
            ),
            1,
            [190, 236, 236, 255],
        );

        canvas.text(
            12,
            115,
            &format!(
                "TOTAL {:.3}  DELTA {:+.4}  ENERGY {:.3} J",
                m.mana.total, delta, m.materials.energy_joules
            ),
            1,
            [190, 236, 236, 255],
        );

        canvas.text(13, 129, "FPS / 180 FRAMES", 1, [135, 224, 255, 255]);
        canvas.text(268, 129, "FRAME TIME / MS", 1, [135, 224, 255, 255]);
        canvas.graph(12, 145, 240, 88, &self.fps_samples, [78, 220, 255, 255]);
        canvas.graph(266, 145, 240, 88, &self.frame_samples, [255, 186, 104, 255]);
        canvas.text(13, 246, "MAX MANON DENSITY / TIME", 1, [160, 223, 255, 255]);
        canvas.graph(12, 262, 494, 65, &self.peak_samples, [156, 195, 255, 255]);
        canvas.text(
            12,
            341,
            &format!(
                "FIELD |phi| {:.3}   J {:.3} manons/step",
                m.field.peak, m.field.transported_manons
            ),
            1,
            [160, 223, 255, 255],
        );
        if let Some(mechanics) = m.mechanics {
            for i in 0..mechanics.bodies.min(2) {
                let x = mechanics.positions[i];
                canvas.text(
                    12,
                    357 + i as i32 * 13,
                    &format!(
                        "BODY {}  X [{:.3}, {:.3}, {:.3}] m",
                        i + 1,
                        x[0],
                        x[1],
                        x[2]
                    ),
                    1,
                    [116, 238, 169, 255],
                );
            }
        }
        for i in 0..if m.mechanics.is_some() {
            0
        } else {
            m.materials.formula_count.min(3)
        } {
            canvas.text(
                12,
                357 + i as i32 * 13,
                &format!(
                    "FORMULA {} / BOUND {:>14.3} manons",
                    i + 1,
                    m.materials.formula_bound[i]
                ),
                1,
                [116, 238, 169, 255],
            );
        }
        canvas.text(
            12,
            397,
            &if let Some(mechanics) = m.mechanics {
                format!(
                    "MECH E {:.6} J  dE {:.2e} J",
                    mechanics.energy, mechanics.energy_error
                )
            } else {
                format!(
                    "FIELD E {:.2} J  HEAT {:.2}  WORK {:.2}",
                    m.field.energy_joules, m.field.heat_joules, m.field.work_joules
                )
            },
            1,
            [160, 223, 255, 255],
        );
        canvas.text(
            12,
            412,
            &if let Some(mechanics) = m.mechanics {
                let p = mechanics.momentum;
                format!(
                    "P [{:.4}, {:.4}, {:.4}]  dP {:.1e}",
                    p[0], p[1], p[2], mechanics.momentum_error
                )
            } else {
                format!(
                    "TRAP T {:.1} K   EXPORTED {:.2} J   {}",
                    m.materials.temperature_kelvin,
                    m.field.exported_heat_joules,
                    if m.field.shutdown {
                        "SOURCE DECAYING"
                    } else {
                        "SOURCE ACTIVE"
                    }
                )
            },
            1,
            [160, 223, 255, 255],
        );
        canvas.text(
            12,
            433,
            &if let Some(mechanics) = m.mechanics {
                format!(
                    "PERIODIC MECHANICS / {} bodies / {} modes",
                    mechanics.bodies, mechanics.modes
                )
            } else {
                format!(
                    "PHASE {}  DENSITY RATE {:.2e}",
                    m.simulation.equilibrium.phase.label(),
                    m.simulation.equilibrium.density_rate
                )
            },
            1,
            [144, 229, 255, 255],
        );
        canvas.text(
            12,
            449,
            &if m.mechanics.is_some() {
                "E/P INCLUDE BODIES AND DYNAMIC FIELD".to_owned()
            } else {
                format!(
                    "BOUND RATE {:.2e}/s  STABLE {} steps",
                    m.simulation.equilibrium.bound_rate, m.simulation.equilibrium.stable_steps
                )
            },
            1,
            [144, 229, 255, 255],
        );
        canvas.text(
            12,
            465,
            &format!(
                "PHYSICS {}{}   GPU BATCH {:>6.2} ms / {} steps",
                m.simulation.compute.backend.label(),
                if m.simulation.compute.cpu_fallback {
                    " (FALLBACK)"
                } else {
                    ""
                },
                m.simulation.compute.batch_wall_ms,
                m.simulation.compute.batch_steps
            ),
            1,
            [118, 147, 177, 255],
        );
        canvas.text(
            12,
            484,
            "GPU BATCH includes transfers; not GPU kernel time",
            1,
            [118, 147, 177, 255],
        );
        ImageAsset {
            width: WIDTH as u32,
            height: HEIGHT as u32,
            pixels: canvas.data,
        }
    }
}
