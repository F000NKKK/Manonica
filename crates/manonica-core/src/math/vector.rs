use manonica_error::BaseError;

use crate::{Quantity, units::Unit};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3<U: Unit> {
    pub x: Quantity<U>,
    pub y: Quantity<U>,
    pub z: Quantity<U>,
}

impl<U: Unit> Vec3<U> {
    pub fn new(x: Quantity<U>, y: Quantity<U>, z: Quantity<U>) -> Self {
        Self { x, y, z }
    }

    pub fn try_values(x: f64, y: f64, z: f64) -> Result<Self, BaseError> {
        Ok(Self {
            x: Quantity::new(x)?,
            y: Quantity::new(y)?,
            z: Quantity::new(z)?,
        })
    }
}
