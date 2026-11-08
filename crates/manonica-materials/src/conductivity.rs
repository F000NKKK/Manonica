//! Passive anisotropic manon conductance along geometric channels.
//!
//! Stage B1 represents a crystal by a positive semidefinite conductivity
//! tensor K = R diag(k_parallel_1, k_parallel_2, k_parallel_3) R^T.
//! The effective multiplier of a channel in direction u is u^T K u.
//! The current model supports rotation around Z only; later versions can
//! generalize orientation without changing the constitutive law.
use manonica_error::{BaseError, ErrorCode};

fn invalid() -> BaseError {
    BaseError::new(ErrorCode::InvalidRange, "anisotropic_conductivity")
}

#[derive(Debug, Clone, Copy)]
pub struct DirectionalConductivity {
    /// Dimensionless nonnegative factors relative to a channel's baseline
    /// physical conductance, along the three principal crystal axes.
    pub principal: [f64; 3],
    /// Rotation of the first two principal axes around world Z, in radians.
    pub rotation_z_radians: f64,
}
impl DirectionalConductivity {
    pub fn validate(self) -> Result<(), BaseError> {
        if self.principal.iter().any(|x| !x.is_finite() || *x < 0.0)
            || !self.rotation_z_radians.is_finite()
        {
            return Err(invalid());
        }
        Ok(())
    }

    /// u^T K u for an arbitrary nonzero world direction.
    pub fn along(self, direction: [f64; 3]) -> Result<f64, BaseError> {
        self.validate()?;
        if direction.iter().any(|x| !x.is_finite()) {
            return Err(invalid());
        }
        let squared_norm: f64 = direction.iter().map(|v| v * v).sum();
        if !squared_norm.is_finite() || squared_norm <= 0.0 {
            return Err(invalid());
        }
        let norm = squared_norm.sqrt();
        let [x, y, z] = direction.map(|v| v / norm);
        let (sin, cos) = self.rotation_z_radians.sin_cos();
        let local_x = cos * x + sin * y;
        let local_y = -sin * x + cos * y;
        let value = self.principal[0] * local_x.powi(2)
            + self.principal[1] * local_y.powi(2)
            + self.principal[2] * z.powi(2);
        if !value.is_finite() || value < 0.0 {
            return Err(invalid());
        }
        Ok(value)
    }
}
