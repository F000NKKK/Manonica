//! Lightweight in-window editor for a live parametric mana circuit.
use crate::experiment::{Experiment, Formula, FormulaTrap, sample_curve};
use font8x8::{BASIC_FONTS, UnicodeFonts};
use meridian_sdk::{
    Camera2d, GraphicsSession, ImageAsset, Motor3, Overlay2d, Scene2d, Sprite, TextureHandle,
    TextureStore, Vec3,
};
use std::time::{Duration, Instant};

const WIDTH: usize = 820;
const HEIGHT: usize = 264;
const MAX_EXPRESSION: usize = 128;

#[derive(Clone)]
pub struct FormulaEditor {
    pub open: bool,
    pub expressions: [String; 3],
    pub selected: usize,
    /// Select-all is confined to the active formula coordinate.
    pub selection_all: bool,
    pub circuit: usize,
    pub circuits: usize,
    /// Geometric centre of the selected circuit; presets keep its position.
    pub center: [f64; 3],
    pub message: String,
}
impl Default for FormulaEditor {
    fn default() -> Self {
        Self {
            open: false,
            expressions: ["2*cos(t)".into(), "-0.8".into(), "2*sin(t)".into()],
            selected: 0,
            selection_all: false,
            circuit: 0,
            circuits: 1,
            center: [0.0; 3],
            message: "TAB: next field   ENTER: apply   ESC: close".into(),
        }
    }
}
impl FormulaEditor {
    /// Replace active selection with keyboard or clipboard text. All
    /// insertions share the same ASCII and maximum-length constraints.
    pub fn input(&mut self, value: &str) {
        let filtered: String = value.chars()
            .filter(|ch| ch.is_ascii() && !ch.is_control())
            .take(MAX_EXPRESSION)
            .collect();
        let field = &mut self.expressions[self.selected];
        if self.selection_all {
            field.clear();
            self.selection_all = false;
        }
        let remaining = MAX_EXPRESSION.saturating_sub(field.len());
        field.extend(filtered.chars().take(remaining));
    }
    pub fn select_all(&mut self) {
        self.selection_all = true;
    }
    /// Copy text only when an explicit selection exists, matching usual
    /// single-line text-field behavior.
    pub fn selected_text(&self) -> Option<&str> {
        self.selection_all.then_some(&self.expressions[self.selected])
    }
    pub fn paste(&mut self, text: &str) {
        self.input(text);
    }
    pub fn backspace(&mut self) {
        if self.selection_all {
            self.expressions[self.selected].clear();
            self.selection_all = false;
        } else {
            self.expressions[self.selected].pop();
        }
    }
    pub fn next(&mut self) {
        self.selected = (self.selected + 1) % 3;
        self.selection_all = false;
    }
    pub fn load(&mut self, formulas: &[Formula]) {
        self.selection_all = false;
        self.circuits = formulas.len().max(1);
        self.circuit = self.circuit.min(self.circuits - 1);
        if let Some(formula) = formulas.get(self.circuit) {
            self.expressions = [formula.x.clone(), formula.y.clone(), formula.z.clone()];
            if let Ok(points) = sample_curve(formula) {
                let mut center = [0.0_f64; 3];
                for point in &points {
                    for axis in 0..3 {
                        center[axis] += f64::from(point[axis]) / points.len() as f64;
                    }
                }
                self.center = center;
            }
        }
        self.message = if formulas.len() == 1 {
            "ONE CIRCUIT LOADED. ESC, THEN C: SHOW ALL THREE".into()
        } else {
            "TAB: field  LEFT/RIGHT: circuit  ENTER: apply".into()
        };
    }
    pub fn switch(&mut self, formulas: &[Formula], delta: isize) {
        if formulas.is_empty() {
            return;
        }
        self.circuit = self
            .circuit
            .saturating_add_signed(delta)
            .min(formulas.len() - 1);
        self.load(formulas);
    }
    /// Three mathematically distinct user-editable presets. All positions
    /// derive from the expressions, never from fixed render meshes.
    pub fn preset(&mut self, preset: u8) {
        let expressions = match preset {
            1 => ["2*cos(t)", "1.6*sin(t)", "0"],
            2 => [
                "(1.6+0.4*cos(5*t))*cos(t)",
                "(1.6+0.4*cos(5*t))*sin(t)",
                "0",
            ],
            3 => [
                "(1.45+0.4*cos(3*t))*cos(2*t)",
                "(1.45+0.4*cos(3*t))*sin(2*t)",
                "0.8*sin(3*t)",
            ],
            _ => return,
        };
        self.selection_all = false;
        self.expressions = std::array::from_fn(|axis| {
            format!("({})+({:.4})", expressions[axis], self.center[axis])
        });
        self.message = format!("PRESET {preset}: preview updated. ENTER to apply");
    }

    /// Changing the curve must not silently replace an existing crystal's
    /// physical capacity, binding energy, colour or material parameters.
    pub fn update_formula(&self, existing: Option<&Formula>) -> Formula {
        let mut updated = existing.cloned().unwrap_or_else(|| self.formula());
        updated.x.clone_from(&self.expressions[0]);
        updated.y.clone_from(&self.expressions[1]);
        updated.z.clone_from(&self.expressions[2]);
        updated
    }

    pub fn formula(&self) -> Formula {
        Formula {
            name: format!("Live circuit {}", self.circuit + 1),
            x: self.expressions[0].clone(),
            y: self.expressions[1].clone(),
            z: self.expressions[2].clone(),
            segments: 96,
            thickness: 0.035,
            color: [0.2, 0.9, 0.6],
            trap: Some(FormulaTrap {
                capacity: 2500.0,
                capture_rate: 0.18,
                release_rate: 0.002,
                binding_energy: 0.08,
                heat_capacity: 100.0,
                reference_temperature: 300.0,
                activation_temperature: 300.0,
                max_flux: 75.0,
            }),
        }
    }
}

/// One texture update per changed editor state, not per physics tick.
#[derive(Default)]
pub struct EditorOverlay {
    texture: Option<TextureHandle>,
    signature: String,
}
impl EditorOverlay {
    pub fn draw(
        &mut self,
        base: &mut GraphicsSession,
        editor: &FormulaEditor,
        viewport: [u32; 2],
    ) -> Option<Overlay2d> {
        if !editor.open {
            return None;
        }
        let signature = format!(
            "{:?}|{}|{}|{}",
            editor.expressions, editor.selected, editor.selection_all, editor.message
        );
        if self.texture.is_none() || self.signature != signature {
            let next = base.render.recorder.upload(&editor.image());
            if let Some(old) = self.texture.replace(next) {
                base.render.recorder.release(old);
            }
            self.signature = signature;
        }
        let texture = self.texture?;
        Some(Overlay2d::new(Scene2d {
            camera: Camera2d::from_surface_size(viewport[0] as f32, viewport[1] as f32),
            sprites: vec![Sprite {
                texture,
                source_rect: None,
                size: [WIDTH as f32, HEIGHT as f32],
                tint: [1.0; 4],
                frame: Motor3::translation(Vec3::new(
                    20.0 + WIDTH as f32 * 0.5,
                    20.0 + HEIGHT as f32 * 0.5,
                    0.0,
                )),
                layer: 100,
            }],
        }))
    }
}
impl FormulaEditor {
    fn image(&self) -> ImageAsset {
        let mut pixels = vec![0u8; WIDTH * HEIGHT * 4];
        fn rect(p: &mut [u8], x: usize, y: usize, w: usize, h: usize, rgba: [u8; 4]) {
            for yy in y..(y + h).min(HEIGHT) {
                for xx in x..(x + w).min(WIDTH) {
                    let at = (yy * WIDTH + xx) * 4;
                    p[at..at + 4].copy_from_slice(&rgba);
                }
            }
        }
        fn text(p: &mut [u8], x: usize, y: usize, value: &str, color: [u8; 4]) {
            for (i, ch) in value.chars().enumerate() {
                if x + i * 8 + 8 >= WIDTH {
                    break;
                }
                if let Some(glyph) = BASIC_FONTS.get(ch) {
                    for (gy, row) in glyph.iter().enumerate() {
                        for gx in 0..8 {
                            if row & (1 << gx) != 0 {
                                rect(p, x + i * 8 + gx, y + gy, 1, 1, color);
                            }
                        }
                    }
                }
            }
        }
        fn line(p: &mut [u8], a: (i32, i32), b: (i32, i32), rgba: [u8; 4]) {
            let (mut x, mut y) = a;
            let dx = (b.0 - x).abs();
            let sx = if x < b.0 { 1 } else { -1 };
            let dy = -(b.1 - y).abs();
            let sy = if y < b.1 { 1 } else { -1 };
            let mut err = dx + dy;
            loop {
                if x >= 0 && y >= 0 && (x as usize) < WIDTH && (y as usize) < HEIGHT {
                    let at = (y as usize * WIDTH + x as usize) * 4;
                    p[at..at + 4].copy_from_slice(&rgba);
                }
                if x == b.0 && y == b.1 {
                    break;
                }
                let e2 = err * 2;
                if e2 >= dy {
                    err += dy;
                    x += sx;
                }
                if e2 <= dx {
                    err += dx;
                    y += sy;
                }
            }
        }
        rect(&mut pixels, 0, 0, WIDTH, HEIGHT, [9, 17, 31, 245]);
        rect(&mut pixels, 0, 0, WIDTH, 28, [18, 52, 77, 255]);
        text(
            &mut pixels,
            12,
            10,
            &format!(
                "MANONICA / CIRCUIT {}/{} - LIVE FORMULA",
                self.circuit + 1,
                self.circuits
            ),
            [225, 245, 255, 255],
        );
        for (i, expr) in self.expressions.iter().enumerate() {
            let y = 40 + i * 43;
            rect(
                &mut pixels,
                12,
                y,
                425,
                35,
                if i == self.selected {
                    [41, 78, 102, 255]
                } else {
                    [20, 37, 57, 255]
                },
            );
            let displayed: String = expr.chars().take(40).collect();
            text(
                &mut pixels,
                21,
                y + 13,
                &format!("{}(t) = {}", ["x", "y", "z"][i], displayed),
                [220, 240, 255, 255],
            );
            if i == self.selected && self.selection_all {
                rect(
                    &mut pixels,
                    19,
                    y + 8,
                    (expr.chars().take(40).count() * 8 + 60).min(408),
                    19,
                    [56, 102, 152, 255],
                );
                text(
                    &mut pixels,
                    21,
                    y + 13,
                    &format!("{}(t) = {}", ["x", "y", "z"][i], displayed),
                    [246, 250, 255, 255],
                );
            }
            if i == self.selected && !self.selection_all {
                let cursor_x = 21 + (7 + displayed.len()) * 8;
                rect(
                    &mut pixels,
                    cursor_x.min(427),
                    y + 11,
                    2,
                    12,
                    [255, 255, 255, 255],
                );
            }
        }
        // Immediate mathematical preview of both independent projections.
        // Users can distinguish a planar circle, a star, and a 3D knot
        // before committing the formula to the physical simulation.
        match sample_curve(&self.formula()) {
            Ok(samples) => {
                text(&mut pixels, 462, 37, "X / Y (FRONT)", [165, 228, 244, 255]);
                text(&mut pixels, 641, 37, "X / Z (TOP)", [165, 228, 244, 255]);
                for (view, axes) in [[0, 1], [0, 2]].into_iter().enumerate() {
                    let x0 = 453 + view * 181;
                    let y0 = 54;
                    let w = 173_usize;
                    let h = 146_usize;
                    rect(&mut pixels, x0, y0, w, h, [17, 34, 53, 255]);
                    let mut min = [f32::INFINITY; 2];
                    let mut max = [f32::NEG_INFINITY; 2];
                    for p in &samples {
                        for j in 0..2 {
                            min[j] = min[j].min(p[axes[j]]);
                            max[j] = max[j].max(p[axes[j]]);
                        }
                    }
                    let extent = (max[0] - min[0]).max(max[1] - min[1]).max(0.05);
                    let cx = (min[0] + max[0]) * 0.5;
                    let cy = (min[1] + max[1]) * 0.5;
                    let map = |p: &[f32; 3]| {
                        let px = (x0 as f32 + w as f32 * 0.5 + (p[axes[0]] - cx) / extent * 134.0)
                            .round() as i32;
                        let py = (y0 as f32 + h as f32 * 0.5 - (p[axes[1]] - cy) / extent * 114.0)
                            .round() as i32;
                        (px, py)
                    };
                    for i in 0..samples.len() {
                        line(
                            &mut pixels,
                            map(&samples[i]),
                            map(&samples[(i + 1) % samples.len()]),
                            [81, 240, 164, 255],
                        );
                    }
                }
                text(
                    &mut pixels,
                    462,
                    212,
                    "PREVIEW: VALID",
                    [104, 247, 166, 255],
                );
            }
            Err(_) => text(
                &mut pixels,
                462,
                90,
                "INVALID FORMULA",
                [255, 120, 120, 255],
            ),
        }
        let message: String = self.message.chars().take(93).collect();
        text(&mut pixels, 13, 227, &message, [153, 224, 190, 255]);
        text(
            &mut pixels,
            13,
            246,
            "CTRL+A SELECT  CTRL+C COPY  CTRL+V PASTE   TAB FIELD  ENTER APPLY",
            [167, 191, 213, 255],
        );
        ImageAsset {
            width: WIDTH as u32,
            height: HEIGHT as u32,
            pixels,
        }
    }
}
/// A compact legend, independent from F3, that makes the physical
/// difference between active mathematical contours visible at all times.
#[derive(Default)]
pub struct FormulaLegend {
    texture: Option<TextureHandle>,
    last_upload: Option<Instant>,
    last_revision: usize,
}
impl FormulaLegend {
    pub fn draw(
        &mut self,
        base: &mut GraphicsSession,
        experiment: &Experiment,
        bound: &[f64],
        viewport: [u32; 2],
        editor_open: bool,
    ) -> Option<Overlay2d> {
        const W: usize = 500;
        const H: usize = 116;
        if editor_open || experiment.formulas.len() < 2 {
            return None;
        }
        let now = Instant::now();
        let revision = experiment.formulas.iter().fold(0_usize, |state, formula| {
            state
                .wrapping_mul(31)
                .wrapping_add(formula.x.len())
                .wrapping_add(formula.y.len())
                .wrapping_add(formula.z.len())
        });
        if self.texture.is_none()
            || self.last_revision != revision
            || self
                .last_upload
                .is_none_or(|last| now.duration_since(last) >= Duration::from_millis(250))
        {
            let mut pixels = vec![0_u8; W * H * 4];
            let mut rect = |x: usize, y: usize, w: usize, h: usize, color: [u8; 4]| {
                for yy in y..(y + h).min(H) {
                    for xx in x..(x + w).min(W) {
                        let i = (yy * W + xx) * 4;
                        pixels[i..i + 4].copy_from_slice(&color);
                    }
                }
            };
            rect(0, 0, W, H, [9, 19, 32, 234]);
            rect(0, 0, W, 23, [24, 49, 71, 248]);
            // End rect's mutable borrow before drawing individual glyphs.
            fn write(p: &mut [u8], x: usize, y: usize, label: &str, color: [u8; 4]) {
                for (i, ch) in label.chars().enumerate() {
                    if x + 8 * (i + 1) >= W {
                        break;
                    }
                    if let Some(glyph) = BASIC_FONTS.get(ch) {
                        for (gy, bits) in glyph.iter().enumerate() {
                            for gx in 0..8 {
                                if bits & (1 << gx) != 0 {
                                    let index = ((y + gy) * W + x + i * 8 + gx) * 4;
                                    p[index..index + 4].copy_from_slice(&color);
                                }
                            }
                        }
                    }
                }
            }
            write(
                &mut pixels,
                10,
                8,
                "ACTIVE MATHEMATICAL CIRCUITS",
                [212, 240, 255, 255],
            );
            for (index, formula) in experiment.formulas.iter().take(3).enumerate() {
                let y = 31 + 27 * index;
                let rgb = formula.color.map(|c| (c.clamp(0.0, 1.0) * 255.0) as u8);
                for yy in y..y + 12 {
                    for xx in 10..24 {
                        let i = (yy * W + xx) * 4;
                        pixels[i..i + 4].copy_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
                    }
                }
                let expression: String = formula.x.chars().take(33).collect();
                write(
                    &mut pixels,
                    34,
                    y + 2,
                    &format!("{}: x(t)={}", index + 1, expression),
                    [216, 235, 245, 255],
                );
                write(
                    &mut pixels,
                    369,
                    y + 2,
                    &format!("B:{:.0}", bound.get(index).copied().unwrap_or(0.0)),
                    [121, 246, 172, 255],
                );
            }
            let image = ImageAsset {
                width: W as u32,
                height: H as u32,
                pixels,
            };
            let next = base.render.recorder.upload(&image);
            if let Some(old) = self.texture.replace(next) {
                base.render.recorder.release(old);
            }
            self.last_upload = Some(now);
            self.last_revision = revision;
        }
        Some(Overlay2d::new(Scene2d {
            camera: Camera2d::from_surface_size(viewport[0] as f32, viewport[1] as f32),
            sprites: vec![Sprite {
                texture: self.texture?,
                source_rect: None,
                size: [W as f32, H as f32],
                tint: [1.0; 4],
                frame: Motor3::translation(Vec3::new(
                    viewport[0] as f32 - W as f32 * 0.5 - 12.0,
                    H as f32 * 0.5 + 12.0,
                    0.0,
                )),
                layer: 20,
            }],
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn presets_keep_the_selected_circuit_position() {
        let mut editor = FormulaEditor::default();
        let mut left = editor.formula();
        left.x = "-4.2+1.35*cos(t)".into();
        left.y = "1.35*sin(t)".into();
        left.z = "0".into();
        editor.load(&[left]);
        editor.preset(3);
        let points = sample_curve(&editor.formula()).unwrap();
        let mean_x = points.iter().map(|p| p[0]).sum::<f32>() / points.len() as f32;
        assert!((mean_x + 4.2).abs() < 0.01, "{mean_x}");
        assert!(points.iter().any(|p| p[2].abs() > 0.7));
    }

    #[test]
    fn select_copy_paste_replaces_only_active_axis_and_limits_length() {
        let mut editor = FormulaEditor::default();
        let original_y = editor.expressions[1].clone();
        editor.select_all();
        assert_eq!(editor.selected_text(), Some("2*cos(t)"));
        editor.paste("sin(t)+2*sin(2*t)\n");
        assert_eq!(editor.expressions[0], "sin(t)+2*sin(2*t)");
        assert_eq!(editor.expressions[1], original_y);
        assert_eq!(editor.selected_text(), None);
        editor.select_all();
        editor.backspace();
        assert!(editor.expressions[0].is_empty());
        editor.input(&"a".repeat(MAX_EXPRESSION + 30));
        assert_eq!(editor.expressions[0].len(), MAX_EXPRESSION);
        editor.next();
        assert!(!editor.selection_all);
    }

    #[test]
    fn editor_retains_formula_and_limits_text() {
        let mut editor = FormulaEditor::default();
        editor.expressions[0].clear();
        editor.input("2*cos(t)");
        assert_eq!(editor.formula().x, "2*cos(t)");
        editor.backspace();
        assert_eq!(editor.expressions[0], "2*cos(t");
        editor.next();
        assert_eq!(editor.selected, 1);
        assert_eq!(editor.image().pixels.len(), WIDTH * HEIGHT * 4);
    }
}
