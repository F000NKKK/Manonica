//! Физические геометрические контуры из параметрических формул.
use crate::experiment::{Formula, sample_curve};
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

/// Geometry is sampled from the same equation used for installing physical
/// sites. Coincident points are skipped to avoid zero-length tube normals.
pub fn formula_mesh(formula: &Formula) -> Result<MeshSource, String> {
    let positions = sample_curve(formula).map_err(|e| e.to_string())?;
    let mut mesh = MeshSource {
        positions: Vec::new(),
        normals: Vec::new(),
        uvs: Vec::new(),
        indices: Vec::new(),
        skin: None,
        morph: None,
    };
    for i in 0..positions.len() {
        let a = positions[i];
        let b = positions[(i + 1) % positions.len()];
        let d = sub(b, a);
        if dot(d, d) > 1e-10 {
            append_tube(&mut mesh, a, b, formula.thickness);
        }
    }
    if mesh.indices.is_empty() {
        return Err("formula has no non-degenerate segments".into());
    }
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arbitrary_spatial_formula_produces_valid_triangle_mesh() {
        let formula = Formula {
            name: "spatial".into(),
            x: "2*cos(t)".into(),
            y: "0.8*sin(3*t)".into(),
            z: "2*sin(t)".into(),
            segments: 96,
            thickness: 0.03,
            color: [0.2, 0.9, 0.6],
            trap: None,
        };
        let mesh = formula_mesh(&formula).unwrap();
        assert!(!mesh.indices.is_empty());
        assert!(
            mesh.indices
                .iter()
                .all(|i| (*i as usize) < mesh.positions.len())
        );
        assert!(mesh.positions.iter().flatten().all(|v| v.is_finite()));
    }
}
