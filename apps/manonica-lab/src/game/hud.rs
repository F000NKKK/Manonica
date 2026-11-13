use font8x8::{BASIC_FONTS, UnicodeFonts};
use meridian_sdk::{
    Camera2d, GraphicsSession, ImageAsset, Motor3, Overlay2d, Scene2d, Sprite, TextureHandle,
    TextureStore, Vec3,
};
use std::time::{Duration, Instant};
pub struct Hud {
    texture: Option<TextureHandle>,
    cross: TextureHandle,
    last: Instant,
}
impl Hud {
    pub fn new(base: &mut GraphicsSession) -> Self {
        let mut pixels = vec![0; 16 * 16 * 4];
        for y in 0..16 {
            for x in 0..16 {
                if (x == 7 && (3..13).contains(&y)) || (y == 7 && (3..13).contains(&x)) {
                    let i = (y * 16 + x) * 4;
                    pixels[i..i + 4].copy_from_slice(&[255, 255, 255, 240]);
                }
            }
        }
        let cross = base.render.recorder.upload(&ImageAsset {
            width: 16,
            height: 16,
            pixels,
        });
        Self {
            texture: None,
            cross,
            last: Instant::now() - Duration::from_secs(1),
        }
    }
    pub fn draw(
        &mut self,
        base: &mut GraphicsSession,
        size: [u32; 2],
        lines: &[String],
    ) -> Overlay2d {
        if self.last.elapsed() > Duration::from_millis(100) {
            let mut pixels = vec![0; 800 * 144 * 4];
            for pixel in pixels.as_chunks_mut::<4>().0 {
                pixel.copy_from_slice(&[7, 14, 24, 190]);
            }
            for (row, line) in lines.iter().take(8).enumerate() {
                for (column, ch) in line.chars().take(98).enumerate() {
                    if let Some(glyph) = BASIC_FONTS.get(ch) {
                        for (y, bits) in glyph.iter().enumerate() {
                            for x in 0..8 {
                                if bits & (1 << x) != 0 {
                                    let i = ((8 + row * 16 + y) * 800 + 8 + column * 8 + x) * 4;
                                    if i + 4 <= pixels.len() {
                                        pixels[i..i + 4].copy_from_slice(&[213, 235, 242, 255]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let image = ImageAsset {
                width: 800,
                height: 144,
                pixels,
            };
            if let Some(texture) = self.texture {
                base.render.recorder.update(texture, &image);
            } else {
                self.texture = Some(base.render.recorder.upload(&image));
            }
            self.last = Instant::now();
        }
        let mut sprites = vec![Sprite {
            texture: self.cross,
            source_rect: None,
            size: [16., 16.],
            tint: [1.; 4],
            frame: Motor3::translation(Vec3::new(size[0] as f32 * 0.5, size[1] as f32 * 0.5, 0.)),
            layer: 1,
        }];
        if let Some(texture) = self.texture {
            sprites.push(Sprite {
                texture,
                source_rect: None,
                size: [800., 144.],
                tint: [1.; 4],
                frame: Motor3::translation(Vec3::new(408., 80., 0.)),
                layer: 0,
            });
        }
        Overlay2d::new(Scene2d {
            camera: Camera2d::from_surface_size(size[0] as f32, size[1] as f32),
            sprites,
        })
    }
}
