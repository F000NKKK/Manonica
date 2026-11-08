//! F3 diagnostics, composed as a single texture and one Meridian Overlay2d.
//! The HUD is not part of physical simulation. No direct wgpu dependencies.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use font8x8::{BASIC_FONTS, UnicodeFonts};
use meridian_sdk::{
    Camera2d, GraphicsSession, ImageAsset, Motor3, Overlay2d, Scene2d, Sprite, TextureHandle,
    TextureStore, Vec3,
};

const WIDTH: usize = 520;
const HEIGHT: usize = 418;
const HISTORY: usize = 180;
const UPDATE_PERIOD: Duration = Duration::from_millis(200);

#[derive(Debug, Clone, Copy)]
pub struct FrameMetrics {
    pub frame_ms: f64,
    pub sim_cpu_ms: f64,
    pub draw_cpu_ms: f64,
    pub field_cpu_ms: f64,
    pub sim_time: f64,
    pub steps: u64,
    pub total: f64,
    pub bound: f64,
    pub formula_bound: [f64; 3],
    pub formula_count: usize,
    pub material_energy: f64,
    pub initial_total: f64,
    pub peak_density: f64,
    pub shown_slices: usize,
    pub grid_cells: usize,
    pub paused: bool,
}

pub struct DebugHud {
    image: Option<TextureHandle>,
    last_image: Option<Instant>,
    fps_samples: VecDeque<f64>,
    frame_samples: VecDeque<f64>,
    peak_samples: VecDeque<f64>,
    latest: Option<FrameMetrics>,
}

impl Default for DebugHud {
    fn default() -> Self {
        Self {
            image: None,
            last_image: None,
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

impl DebugHud {
    pub fn record(&mut self, metrics: FrameMetrics) {
        let fps = if metrics.frame_ms > 0.000_01 {
            1000.0 / metrics.frame_ms
        } else {
            0.0
        };
        record(&mut self.fps_samples, fps.clamp(0.0, 1000.0));
        record(&mut self.frame_samples, metrics.frame_ms.clamp(0.0, 1000.0));
        record(&mut self.peak_samples, metrics.peak_density);
        self.latest = Some(metrics);
    }

    pub fn fps(&self) -> f64 {
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

    pub fn overlay(&mut self, base: &mut GraphicsSession, viewport: [u32; 2]) -> Option<Overlay2d> {
        let latest = self.latest?;
        let now = Instant::now();
        if self.image.is_none()
            || self
                .last_image
                .is_none_or(|last| now.duration_since(last) >= UPDATE_PERIOD)
        {
            let image = self.render_image(latest);
            let next = base.render.recorder.upload(&image);
            if let Some(previous) = self.image.replace(next) {
                base.render.recorder.release(previous);
            }
            self.last_image = Some(now);
        }
        let handle = self.image?;
        let scene = Scene2d {
            camera: Camera2d::from_surface_size(viewport[0] as f32, viewport[1] as f32),
            sprites: vec![Sprite {
                texture: handle,
                source_rect: None,
                size: [WIDTH as f32, HEIGHT as f32],
                tint: [1.0; 4],
                frame: Motor3::translation(Vec3::new(
                    12.0 + WIDTH as f32 * 0.5,
                    12.0 + HEIGHT as f32 * 0.5,
                    0.0,
                )),
                layer: 0,
            }],
        };
        Some(Overlay2d::new(scene))
    }

    fn render_image(&self, m: FrameMetrics) -> ImageAsset {
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

        let state = if m.paused { "PAUSED" } else { "RUNNING" };
        canvas.text(
            12,
            39,
            &format!(
                "FPS {:>6.1}    FRAME {:>6.2} ms    {state}",
                self.fps(),
                m.frame_ms
            ),
            1,
            [233, 245, 255, 255],
        );
        canvas.text(
            12,
            55,
            &format!(
                "SIM CPU {:>6.2} ms    RENDER CPU {:>6.2} ms",
                m.sim_cpu_ms, m.draw_cpu_ms
            ),
            1,
            [170, 214, 245, 255],
        );
        canvas.text(
            12,
            71,
            &format!(
                "FIELD CPU {:>6.2} ms   SIM TIME {:>8.2} s  STEP {}",
                m.field_cpu_ms, m.sim_time, m.steps
            ),
            1,
            [170, 214, 245, 255],
        );
        canvas.text(
            12,
            87,
            &format!(
                "DENSITY MAX {:>11.3} /m3    LAYERS {:>2}   GRID {}",
                m.peak_density, m.shown_slices, m.grid_cells
            ),
            1,
            [185, 232, 255, 255],
        );
        let delta = m.total - m.initial_total;
        canvas.text(
            12,
            103,
            &format!("FREE {:>11.3}  BOUND {:>11.3}", m.total - m.bound, m.bound),
            1,
            [190, 236, 236, 255],
        );

        canvas.text(
            12,
            115,
            &format!("TOTAL {:.3}  DELTA {:+.4}  ENERGY {:.3} J", m.total, delta, m.material_energy),
            1,
            [190, 236, 236, 255],
        );

        canvas.text(13, 129, "FPS / 180 FRAMES", 1, [135, 224, 255, 255]);
        canvas.text(268, 129, "FRAME TIME / MS", 1, [135, 224, 255, 255]);
        canvas.graph(12, 145, 240, 88, &self.fps_samples, [78, 220, 255, 255]);
        canvas.graph(266, 145, 240, 88, &self.frame_samples, [255, 186, 104, 255]);
        canvas.text(13, 246, "MAX MANON DENSITY / TIME", 1, [160, 223, 255, 255]);
        canvas.graph(12, 262, 494, 65, &self.peak_samples, [156, 195, 255, 255]);
        canvas.text(12, 341, "PHYSICAL CAPTURE BY FORMULA", 1, [160, 223, 255, 255]);
        for i in 0..m.formula_count.min(3) {
            canvas.text(
                12,
                355 + i as i32 * 16,
                &format!("FORMULA {} / BOUND {:>14.3} manons", i + 1, m.formula_bound[i]),
                1,
                [116, 238, 169, 255],
            );
        }
        canvas.text(
            12,
            409,
            "CPU frame interval and solver timings; not GPU execution",
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

struct Canvas {
    width: usize,
    height: usize,
    data: Vec<u8>,
}

impl Canvas {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0; width * height * 4],
        }
    }

    fn pixel(&mut self, x: i32, y: i32, color: [u8; 4]) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let i = (y as usize * self.width + x as usize) * 4;
        self.data[i..i + 4].copy_from_slice(&color);
    }

    fn rect(&mut self, x: i32, y: i32, width: usize, height: usize, rgba: [u8; 4]) {
        for iy in 0..height {
            for ix in 0..width {
                self.pixel(x + ix as i32, y + iy as i32, rgba);
            }
        }
    }

    fn text(&mut self, x: i32, y: i32, content: &str, scale: i32, color: [u8; 4]) {
        let mut cursor = x;
        for ch in content.chars() {
            if let Some(glyph) = BASIC_FONTS.get(ch) {
                for (row, bits) in glyph.into_iter().enumerate() {
                    for col in 0..8 {
                        if (bits >> col) & 1 != 0 {
                            self.rect(
                                cursor + col * scale,
                                y + row as i32 * scale,
                                scale as usize,
                                scale as usize,
                                color,
                            );
                        }
                    }
                }
            }
            cursor += 8 * scale;
            if cursor >= self.width as i32 {
                break;
            }
        }
    }

    fn line(&mut self, a: (i32, i32), b: (i32, i32), color: [u8; 4]) {
        let (mut x, mut y) = a;
        let dx = (b.0 - x).abs();
        let sx = if x < b.0 { 1 } else { -1 };
        let dy = -(b.1 - y).abs();
        let sy = if y < b.1 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            self.pixel(x, y, color);
            if x == b.0 && y == b.1 {
                break;
            }
            let twice = 2 * err;
            if twice >= dy {
                err += dy;
                x += sx;
            }
            if twice <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    fn graph(
        &mut self,
        x: i32,
        y: i32,
        width: usize,
        height: usize,
        samples: &VecDeque<f64>,
        color: [u8; 4],
    ) {
        self.rect(x, y, width, height, [16, 28, 44, 233]);
        for row in 1..4 {
            let gy = y + (height * row / 4) as i32;
            self.line((x, gy), (x + width as i32 - 1, gy), [35, 58, 78, 255]);
        }
        if samples.len() < 2 {
            return;
        }
        let low = samples.iter().copied().fold(f64::INFINITY, f64::min);
        let high = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let span = (high - low).max(high.abs() * 0.05).max(0.01);
        let start_x = x + 1;
        let plot_w = width.saturating_sub(3) as f64;
        let plot_h = height.saturating_sub(3) as f64;
        let mut last = None;
        for (index, sample) in samples.iter().enumerate() {
            let px = start_x + (index as f64 / (HISTORY - 1) as f64 * plot_w) as i32;
            let normalized = ((sample - low) / span).clamp(0.0, 1.0);
            let py = y + height as i32 - 2 - (normalized * plot_h) as i32;
            if let Some(previous) = last {
                self.line(previous, (px, py), color);
            }
            last = Some((px, py));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_is_bounded_and_fps_is_finite() {
        let mut hud = DebugHud::default();
        for _ in 0..400 {
            hud.record(FrameMetrics {
                frame_ms: 16.0,
                sim_cpu_ms: 2.0,
                draw_cpu_ms: 3.0,
                field_cpu_ms: 1.0,
                sim_time: 0.0,
                steps: 0,
                total: 10.0,
                bound: 2.0,
                formula_bound: [2.0, 0.0, 0.0],
                formula_count: 1,
                material_energy: 4.0,
                initial_total: 10.0,
                peak_density: 3.0,
                shown_slices: 20,
                grid_cells: 1000,
                paused: false,
            });
        }
        assert_eq!(hud.fps_samples.len(), HISTORY);
        assert!((hud.fps() - 62.5).abs() < 0.01);
        let image = hud.render_image(hud.latest.unwrap());
        assert_eq!(image.pixels.len(), WIDTH * HEIGHT * 4);
    }
}
