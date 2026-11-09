//! Bounded bitmap plotting primitives for the background HUD rasterizer.
use super::HISTORY;
use font8x8::{BASIC_FONTS, UnicodeFonts};
use std::collections::VecDeque;
pub(super) struct Canvas {
    width: usize,
    height: usize,
    pub(super) data: Vec<u8>,
}

impl Canvas {
    pub(super) fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0; width * height * 4],
        }
    }

    pub(super) fn pixel(&mut self, x: i32, y: i32, color: [u8; 4]) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let i = (y as usize * self.width + x as usize) * 4;
        self.data[i..i + 4].copy_from_slice(&color);
    }

    pub(super) fn rect(&mut self, x: i32, y: i32, width: usize, height: usize, rgba: [u8; 4]) {
        for iy in 0..height {
            for ix in 0..width {
                self.pixel(x + ix as i32, y + iy as i32, rgba);
            }
        }
    }

    pub(super) fn text(&mut self, x: i32, y: i32, content: &str, scale: i32, color: [u8; 4]) {
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

    pub(super) fn line(&mut self, a: (i32, i32), b: (i32, i32), color: [u8; 4]) {
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

    pub(super) fn graph(
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
