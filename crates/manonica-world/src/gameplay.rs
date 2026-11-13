//! Engine-independent first-person movement and voxel queries.
use crate::{
    chunk::{CellPosition, ChunkWorld},
    registry::BlockCollision,
};
#[derive(Clone, Copy, Debug, Default)]
pub struct WalkInput {
    pub forward: f64,
    pub strafe: f64,
    pub yaw: f64,
    pub jump: bool,
    pub sprint: bool,
}
#[derive(Clone, Copy, Debug)]
pub struct Character {
    pub feet: [f64; 3],
    pub velocity: [f64; 3],
    pub grounded: bool,
}
impl Character {
    pub const HALF_WIDTH: f64 = 0.3;
    pub const HEIGHT: f64 = 1.8;
    pub fn new(feet: [f64; 3]) -> Self {
        Self {
            feet,
            velocity: [0.0; 3],
            grounded: false,
        }
    }
    pub fn eye(&self) -> [f64; 3] {
        [self.feet[0], self.feet[1] + 1.62, self.feet[2]]
    }
    pub fn intersects(&self, pos: CellPosition) -> bool {
        let b = [pos.x as f64, pos.y as f64, pos.z as f64];
        (0..3).all(|a| {
            let (lo, hi) = if a == 1 {
                (self.feet[a], self.feet[a] + Self::HEIGHT)
            } else {
                (
                    self.feet[a] - Self::HALF_WIDTH,
                    self.feet[a] + Self::HALF_WIDTH,
                )
            };
            hi > b[a] + 1e-7 && lo < b[a] + 1.0 - 1e-7
        })
    }
    pub fn blocked(&self, world: &ChunkWorld) -> bool {
        let min = [
            (self.feet[0] - Self::HALF_WIDTH + 1e-7).floor() as i64,
            (self.feet[1] + 1e-7).floor() as i64,
            (self.feet[2] - Self::HALF_WIDTH + 1e-7).floor() as i64,
        ];
        let max = [
            (self.feet[0] + Self::HALF_WIDTH - 1e-7).floor() as i64,
            (self.feet[1] + Self::HEIGHT - 1e-7).floor() as i64,
            (self.feet[2] + Self::HALF_WIDTH - 1e-7).floor() as i64,
        ];
        for z in min[2]..=max[2] {
            for y in min[1]..=max[1] {
                for x in min[0]..=max[0] {
                    if y < 0
                        || solid(
                            world,
                            CellPosition {
                                x,
                                y: y as usize,
                                z,
                            },
                            true,
                        )
                    {
                        return true;
                    }
                }
            }
        }
        false
    }
    /// Small collision substeps prevent tunnelling at sprint/jump speed.
    pub fn step(&mut self, world: &ChunkWorld, input: WalkInput, dt: f64, acceleration: [f64; 3]) {
        if !dt.is_finite()
            || dt <= 0.0
            || dt > 0.1
            || !input.yaw.is_finite()
            || !input.forward.is_finite()
            || !input.strafe.is_finite()
            || acceleration.iter().any(|v| !v.is_finite())
        {
            return;
        }
        let length = input.forward.hypot(input.strafe).max(1.0);
        let speed = if input.sprint { 7.0 } else { 4.2 };
        let (sy, cy) = input.yaw.sin_cos();
        let response = if self.grounded { 24.0 } else { 4.0 };
        let weight = (response * dt).min(1.0);
        self.velocity[0] +=
            ((cy * input.forward - sy * input.strafe) * speed / length - self.velocity[0]) * weight;
        self.velocity[2] +=
            ((sy * input.forward + cy * input.strafe) * speed / length - self.velocity[2]) * weight;
        if input.jump && self.grounded {
            self.velocity[1] = 7.5;
            self.grounded = false;
        }
        for (axis, a) in acceleration.into_iter().enumerate() {
            self.velocity[axis] += a.clamp(-60.0, 60.0) * dt;
        }
        self.velocity[1] = (self.velocity[1] - 20.0 * dt).clamp(-30.0, 30.0);
        let count = ((self.velocity.iter().map(|v| v.abs()).fold(0.0, f64::max) * dt) / 0.1)
            .ceil()
            .max(1.0) as usize;
        self.grounded = false;
        for _ in 0..count {
            for axis in [0, 2, 1] {
                let old = self.feet[axis];
                let delta = self.velocity[axis] * dt / count as f64;
                self.feet[axis] += delta;
                if self.blocked(world) {
                    self.feet[axis] = old;
                    let direction = delta.signum();
                    let mut low = 0.0;
                    let mut high = delta.abs();
                    for _ in 0..18 {
                        let middle = (low + high) * 0.5;
                        self.feet[axis] = old + direction * middle;
                        if self.blocked(world) {
                            high = middle;
                        } else {
                            low = middle;
                        }
                    }
                    self.feet[axis] = old + direction * low;
                    if axis == 1 && delta < 0.0 {
                        self.grounded = true;
                    }
                    self.velocity[axis] = 0.0;
                }
            }
        }
    }
}
pub fn solid(world: &ChunkWorld, pos: CellPosition, missing: bool) -> bool {
    world
        .block(pos)
        .ok()
        .and_then(|s| world.content().block(s.block()).ok())
        .map(|b| b.physics.collision == BlockCollision::FullCube)
        .unwrap_or(missing)
}
#[derive(Clone, Copy, Debug)]
pub struct BlockHit {
    pub block: CellPosition,
    pub adjacent: CellPosition,
    pub distance: f64,
}
/// Amanatides/Woo traversal: exact voxel faces, including negative coordinates.
pub fn raycast(
    world: &ChunkWorld,
    origin: [f64; 3],
    direction: [f64; 3],
    range: f64,
) -> Option<BlockHit> {
    if origin
        .iter()
        .chain(direction.iter())
        .any(|v| !v.is_finite())
        || !range.is_finite()
        || range <= 0.0
    {
        return None;
    }
    let length = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
    if length < 1e-12 {
        return None;
    }
    let d = direction.map(|v| v / length);
    let mut cell = origin.map(|v| v.floor() as i64);
    let mut previous = cell;
    let mut distance = 0.0;
    let step = d.map(|v| if v > 0.0 { 1 } else { -1 });
    let delta = d.map(|v| {
        if v == 0.0 {
            f64::INFINITY
        } else {
            v.abs().recip()
        }
    });
    let mut next = std::array::from_fn::<_, 3, _>(|a| {
        if d[a] == 0.0 {
            f64::INFINITY
        } else {
            ((cell[a] + i64::from(step[a] > 0)) as f64 - origin[a]) / d[a]
        }
    });
    for _ in 0..2048 {
        if distance > range {
            return None;
        }
        if cell[1] >= 0 && previous[1] >= 0 {
            let block = CellPosition {
                x: cell[0],
                y: cell[1] as usize,
                z: cell[2],
            };
            if solid(world, block, false) {
                return Some(BlockHit {
                    block,
                    adjacent: CellPosition {
                        x: previous[0],
                        y: previous[1] as usize,
                        z: previous[2],
                    },
                    distance,
                });
            }
        }
        let axis = (0..3).min_by(|a, b| next[*a].total_cmp(&next[*b])).unwrap();
        previous = cell;
        cell[axis] = cell[axis].checked_add(step[axis])?;
        distance = next[axis];
        next[axis] += delta[axis];
    }
    None
}
