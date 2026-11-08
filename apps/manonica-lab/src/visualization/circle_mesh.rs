//! Геометрические окружности как тонкие трубки; не физический закон.
use crate::experiment::Circle;
use meridian_sdk::MeshSource;

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn mul(a: [f32; 3], k: f32) -> [f32; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn unit(a: [f32; 3]) -> [f32; 3] {
    mul(a, dot(a, a).sqrt().recip())
}

fn append_tube(mesh: &mut MeshSource, start: [f32; 3], end: [f32; 3], thickness: f32) {
    const SIDES: usize = 8;
    let axis = unit(sub(end, start));
    let up = if axis[1].abs() < 0.85 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let tangent = unit(cross(axis, up));
    let bitangent = cross(axis, tangent);
    let offset =
        u32::try_from(mesh.positions.len()).expect("circle mesh cannot exceed u32 indices");
    for (end_index, center) in [start, end].into_iter().enumerate() {
        for i in 0..SIDES {
            let angle = std::f32::consts::TAU * i as f32 / SIDES as f32;
            let normal = add(mul(tangent, angle.cos()), mul(bitangent, angle.sin()));
            mesh.positions.push(add(center, mul(normal, thickness)));
            mesh.normals.push(normal);
            mesh.uvs.push([i as f32 / SIDES as f32, end_index as f32]);
        }
    }
    for i in 0..SIDES {
        let j = (i + 1) % SIDES;
        let a = offset + i as u32;
        let b = offset + j as u32;
        let c = a + SIDES as u32;
        let d = b + SIDES as u32;
        mesh.indices.extend_from_slice(&[a, b, c, b, d, c]);
    }
}

pub fn circle_mesh(circle: &Circle) -> MeshSource {
    let mut mesh = MeshSource {
        positions: Vec::new(),
        normals: Vec::new(),
        uvs: Vec::new(),
        indices: Vec::new(),
        skin: None,
        morph: None,
    };
    let normal = unit(circle.normal);
    let up = if normal[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let tangent = unit(cross(up, normal));
    let bitangent = cross(normal, tangent);
    let point = |radius: f32, angle: f32| {
        add(
            mul(tangent, radius * angle.cos()),
            mul(bitangent, radius * angle.sin()),
        )
    };
    for i in 0..circle.segments {
        let a = std::f32::consts::TAU * i as f32 / circle.segments as f32;
        let b = std::f32::consts::TAU * (i + 1) as f32 / circle.segments as f32;
        append_tube(
            &mut mesh,
            point(circle.radius, a),
            point(circle.radius, b),
            circle.thickness,
        );
    }
    for i in 0..circle.spokes {
        let angle = std::f32::consts::TAU * i as f32 / circle.spokes as f32;
        append_tube(
            &mut mesh,
            point(circle.radius * 0.15, angle),
            point(circle.radius * 0.96, angle),
            circle.thickness * 0.55,
        );
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arbitrary_plane_circle_produces_triangle_mesh() {
        let circle = Circle {
            name: "slanted".into(),
            center: [0.0; 3],
            normal: [1.0, 1.0, 0.5],
            radius: 2.0,
            thickness: 0.03,
            segments: 32,
            spokes: 8,
            color: [1.0; 3],
        };
        let mesh = circle_mesh(&circle);
        assert!(!mesh.indices.is_empty());
        assert!(
            mesh.indices
                .iter()
                .all(|i| (*i as usize) < mesh.positions.len())
        );
        assert!(mesh.positions.iter().flatten().all(|v| v.is_finite()));
        assert_eq!(mesh.positions.len(), mesh.normals.len());
    }
}
