//! Lightweight in-window editor for a live parametric mana circuit.
use font8x8::{BASIC_FONTS, UnicodeFonts};
use meridian_sdk::{
    Camera2d, GraphicsSession, ImageAsset, Motor3, Overlay2d, Scene2d, Sprite, TextureHandle,
    TextureStore, Vec3,
};
use crate::experiment::{Formula, FormulaTrap};

const WIDTH: usize = 630;
const HEIGHT: usize = 196;
const MAX_EXPRESSION: usize = 128;

#[derive(Clone)]
pub struct FormulaEditor {
    pub open: bool,
    pub expressions: [String; 3],
    pub selected: usize,
    pub message: String,
}
impl Default for FormulaEditor {
    fn default() -> Self {
        Self {
            open: false,
            expressions: ["2*cos(t)".into(), "-0.8".into(), "2*sin(t)".into()],
            selected: 0,
            message: "TAB: next field   ENTER: apply   ESC: close".into(),
        }
    }
}
impl FormulaEditor {
    pub fn input(&mut self, value: &str) {
        for ch in value.chars() {
            if !ch.is_control()
                && ch.is_ascii()
                && self.expressions[self.selected].len() < MAX_EXPRESSION
            {
                self.expressions[self.selected].push(ch);
            }
        }
    }
    pub fn backspace(&mut self) {
        self.expressions[self.selected].pop();
    }
    pub fn next(&mut self) {
        self.selected = (self.selected + 1) % 3;
    }
    pub fn formula(&self) -> Formula {
        Formula {
            name: "Live formula circuit".into(),
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
    pub fn draw(&mut self, base: &mut GraphicsSession, editor: &FormulaEditor, viewport: [u32; 2]) -> Option<Overlay2d> {
        if !editor.open {
            return None;
        }
        let signature = format!("{:?}|{}|{}", editor.expressions, editor.selected, editor.message);
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
            for yy in y..(y+h).min(HEIGHT) {
                for xx in x..(x+w).min(WIDTH) {
                    let at = (yy*WIDTH+xx)*4;
                    p[at..at+4].copy_from_slice(&rgba);
                }
            }
        }
        fn text(p: &mut [u8], x: usize, y: usize, value: &str, color: [u8; 4]) {
            for (i, ch) in value.chars().enumerate() {
                if x+i*8+8 >= WIDTH { break; }
                if let Some(glyph) = BASIC_FONTS.get(ch) {
                    for (gy, row) in glyph.iter().enumerate() {
                        for gx in 0..8 {
                            if row & (1 << gx) != 0 {
                                rect(p, x+i*8+gx, y+gy, 1, 1, color);
                            }
                        }
                    }
                }
            }
        }
        rect(&mut pixels, 0, 0, WIDTH, HEIGHT, [9, 17, 31, 245]);
        rect(&mut pixels, 0, 0, WIDTH, 27, [18, 52, 77, 255]);
        text(&mut pixels, 12, 9, "MANONICA / LIVE FORMULA EDITOR [E]", [225,245,255,255]);
        for (i, expr) in self.expressions.iter().enumerate() {
            let y = 37+i*35;
            rect(&mut pixels, 12, y, WIDTH-24, 30, if i==self.selected {
                [41,78,102,255]
            } else {
                [20,37,57,255]
            });
            text(&mut pixels, 21, y+10, &format!("{}(t) = {}", ["x","y","z"][i],expr), [220,240,255,255]);
            if i == self.selected {
                let cursor_x = 21 + (7+expr.len())*8;
                rect(&mut pixels, cursor_x.min(WIDTH-16), y+9, 2, 10, [255,255,255,255]);
            }
        }
        text(&mut pixels, 13, 149, &self.message, [153,224,190,255]);
        text(&mut pixels, 13, 172, "E: OPEN  TAB: NEXT  ENTER: APPLY  ESC: CLOSE", [167,191,213,255]);
        ImageAsset { width: WIDTH as u32, height: HEIGHT as u32, pixels }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn editor_retains_formula_and_limits_text() {
        let mut editor=FormulaEditor::default();
        editor.expressions[0].clear();
        editor.input("2*cos(t)");
        assert_eq!(editor.formula().x,"2*cos(t)");
        editor.backspace();
        assert_eq!(editor.expressions[0],"2*cos(t");
        editor.next();
        assert_eq!(editor.selected,1);
        assert_eq!(editor.image().pixels.len(),WIDTH*HEIGHT*4);
    }
}
