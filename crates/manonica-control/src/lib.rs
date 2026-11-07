#![forbid(unsafe_code)]

//! Начальный регулятор для магического ТАУ.

#[derive(Debug, Clone, Copy)]
pub struct Pid {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    pub min: f64,
    pub max: f64,
    integral: f64,
    previous_error: Option<f64>,
}

impl Pid {
    pub fn new(kp: f64, ki: f64, kd: f64, min: f64, max: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            min,
            max,
            integral: 0.0,
            previous_error: None,
        }
    }

    pub fn update(&mut self, setpoint: f64, measurement: f64, dt: f64) -> Option<f64> {
        if !setpoint.is_finite()
            || !measurement.is_finite()
            || !dt.is_finite()
            || dt <= 0.0
            || !self.kp.is_finite()
            || !self.ki.is_finite()
            || !self.kd.is_finite()
            || !self.min.is_finite()
            || !self.max.is_finite()
            || self.min > self.max
        {
            return None;
        }
        let error = setpoint - measurement;
        let derivative = self.previous_error.map_or(0.0, |e| (error - e) / dt);
        let next_integral = self.integral + error * dt;
        let unbounded = self.kp * error + self.ki * next_integral + self.kd * derivative;
        if !unbounded.is_finite() {
            return None;
        }
        if !((unbounded > self.max && error > 0.0) || (unbounded < self.min && error < 0.0)) {
            self.integral = next_integral;
        }
        self.previous_error = Some(error);
        Some(
            (self.kp * error + self.ki * self.integral + self.kd * derivative)
                .clamp(self.min, self.max),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_is_limited() {
        let mut pid = Pid::new(2.0, 0.0, 0.0, -5.0, 5.0);
        assert_eq!(pid.update(10.0, 0.0, 0.1), Some(5.0));
        assert_eq!(pid.update(0.0, 10.0, 0.1), Some(-5.0));
    }
}
