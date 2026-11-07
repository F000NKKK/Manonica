#![forbid(unsafe_code)]

//! Базовые законы обмена свободных и связанных манонов.
//! Коэффициенты являются параметрами калибровки, а не установленными константами мира.

#[derive(Debug, Clone, Copy)]
pub struct Reservoir {
    pub free: f64,
    pub bound: f64,
}

impl Reservoir {
    pub fn total(self) -> f64 {
        self.free + self.bound
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Crystal {
    pub capacity: f64,
    pub binding_rate: f64,
    pub release_rate: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsError {
    InvalidValue,
    UnstableStep,
}

pub fn step_crystal(
    reservoir: &mut Reservoir,
    crystal: Crystal,
    dt: f64,
) -> Result<(), PhysicsError> {
    if !dt.is_finite()
        || dt <= 0.0
        || !crystal.capacity.is_finite()
        || crystal.capacity < 0.0
        || !crystal.binding_rate.is_finite()
        || crystal.binding_rate < 0.0
        || !crystal.release_rate.is_finite()
        || crystal.release_rate < 0.0
        || !reservoir.free.is_finite()
        || reservoir.free < 0.0
        || !reservoir.bound.is_finite()
        || reservoir.bound < 0.0
        || reservoir.bound > crystal.capacity
    {
        return Err(PhysicsError::InvalidValue);
    }

    let total = reservoir.total();
    let vacancy = if crystal.capacity > 0.0 {
        1.0 - reservoir.bound / crystal.capacity
    } else {
        0.0
    };
    let absorbed = reservoir.free * (1.0 - (-crystal.binding_rate * dt).exp()) * vacancy;
    let released = reservoir.bound * (1.0 - (-crystal.release_rate * dt).exp());
    reservoir.bound = (reservoir.bound + absorbed - released).clamp(0.0, crystal.capacity);
    reservoir.free = total - reservoir.bound;
    Ok(())
}

/// Явный конечно-разностный шаг диффузии в одном измерении.
/// Отражающие границы: общее количество манонов сохраняется.
/// Условие устойчивости: diffusivity * dt / dx² <= 1/2.
pub fn diffuse_1d(
    concentrations: &mut [f64],
    diffusivity: f64,
    dx: f64,
    dt: f64,
) -> Result<(), PhysicsError> {
    if !diffusivity.is_finite()
        || diffusivity < 0.0
        || !dx.is_finite()
        || dx <= 0.0
        || !dt.is_finite()
        || dt <= 0.0
        || concentrations.iter().any(|x| !x.is_finite() || *x < 0.0)
    {
        return Err(PhysicsError::InvalidValue);
    }
    let lambda = diffusivity * dt / (dx * dx);
    if !lambda.is_finite() || lambda > 0.5 {
        return Err(PhysicsError::UnstableStep);
    }
    let old = concentrations.to_vec();
    for (i, value) in concentrations.iter_mut().enumerate() {
        let middle = old[i];
        let left = if i == 0 { middle } else { old[i - 1] };
        let right = old.get(i + 1).copied().unwrap_or(middle);
        *value = middle + lambda * (left - 2.0 * middle + right);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crystal_exchange_conserves_mana() {
        let mut state = Reservoir {
            free: 100.0,
            bound: 5.0,
        };
        let crystal = Crystal {
            capacity: 70.0,
            binding_rate: 0.7,
            release_rate: 0.03,
        };
        let initial = state.total();
        for _ in 0..1000 {
            step_crystal(&mut state, crystal, 0.01).unwrap();
        }
        assert!((state.total() - initial).abs() < 1e-9);
        assert!(state.free >= 0.0 && state.bound <= crystal.capacity);
    }

    #[test]
    fn diffusion_conserves_mana() {
        let mut field = vec![100.0, 0.0, 0.0, 0.0];
        let initial: f64 = field.iter().sum();
        for _ in 0..1000 {
            diffuse_1d(&mut field, 0.25, 1.0, 0.1).unwrap();
        }
        let final_total: f64 = field.iter().sum();
        assert!((initial - final_total).abs() < 1e-8);
        assert!(field.iter().all(|x| *x >= 0.0));
    }

    #[test]
    fn unstable_timestep_is_rejected() {
        let mut field = vec![10.0, 0.0];
        assert_eq!(
            diffuse_1d(&mut field, 1.0, 1.0, 1.0),
            Err(PhysicsError::UnstableStep)
        );
    }
}
