//! Experimental periodic Gaussian scalar-field mechanics; independent of Lab/GPU.
use crate::MechanicalBody;
use manonica_core::{
    Quantity, TimeStep, Vec3,
    units::{Joules, KilogramMetresPerSecond, Newtons},
};
use manonica_error::{BaseError, ErrorCode};
fn invalid(name: &'static str) -> BaseError {
    BaseError::new(ErrorCode::InvalidRange, name)
}

#[derive(Clone, Debug, PartialEq)]
/// Mechanical body with a signed coupling charge in manons and Gaussian width
/// in metres. Fields are inputs; the solver validates them at construction.
pub struct GaussianBody {
    pub body: MechanicalBody,
    pub charge: f64,
    pub width: f64,
}
#[derive(Clone, Copy, Debug, PartialEq)]
/// Read-only mode snapshot: wave vector in inverse metres, amplitudes A/B and
/// their canonical momenta. Solver geometry cannot be mutated through snapshots.
pub struct SpectralMode {
    k: f64,
    pub wave: [f64; 3],
    pub a: f64,
    pub b: f64,
    pub pa: f64,
    pub pb: f64,
}
#[derive(Clone, Debug, PartialEq)]
/// Experimental CPU Hamiltonian solver, separate from quasi-static density laws.
/// State is private; advancing, replacing storage and loading are transactional.
pub struct SpectralMechanics {
    inertia: f64,
    sources: Vec<GaussianBody>,
    modes: Vec<SpectralMode>,
}
fn factor(s: &GaussianBody, k: f64) -> f64 {
    (-0.5 * (k * s.width).powi(2)).exp()
}
fn coordinates(body: MechanicalBody) -> [f64; 3] {
    let v = body.position();
    [v.x.value(), v.y.value(), v.z.value()]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}
impl SpectralMechanics {
    /// Initialize the stationary field for fixed source positions.
    /// The supplied waves must contain one canonical representative of each
    /// +/- pair (first nonzero component positive), with no duplicates.
    /// Modal inertia is in J*s²/(J/manon)², and omega²=|k|²+1.
    /// Prefer PeriodicSpectralDomain for consistent domain normalization.
    pub fn static_equilibrium(
        sources: Vec<GaussianBody>,
        waves: Vec<[f64; 3]>,
        inertia: f64,
    ) -> Result<Self, BaseError> {
        if !inertia.is_finite()
            || inertia <= 0.0
            || waves.is_empty()
            || sources
                .iter()
                .any(|s| !s.charge.is_finite() || !s.width.is_finite() || s.width <= 0.0)
        {
            return Err(invalid("spectral_parameters"));
        }
        if waves.len() > 65_535 {
            return Err(invalid("spectral_mode_count"));
        }
        let mut unique = std::collections::HashSet::new();
        let mut modes = Vec::with_capacity(waves.len());
        for wave in waves {
            let first = wave.iter().find(|v| **v != 0.0).copied().unwrap_or(0.0);
            let key = wave.map(|v| if v == 0.0 { 0 } else { v.to_bits() });
            if first <= 0.0 || !unique.insert(key) {
                return Err(invalid("spectral_wave_representative"));
            }
            let k = dot(wave, wave).sqrt();
            if !k.is_finite() || k <= 0.0 {
                return Err(invalid("spectral_wave"));
            }
            let mut a = 0.0;
            let mut b = 0.0;
            for s in &sources {
                let (sin, cos) = dot(wave, coordinates(s.body)).sin_cos();
                a -= s.charge * factor(s, k) * cos / (inertia * (k * k + 1.0));
                b -= s.charge * factor(s, k) * sin / (inertia * (k * k + 1.0));
            }
            modes.push(SpectralMode {
                k,
                wave,
                a,
                b,
                pa: 0.0,
                pb: 0.0,
            });
        }
        let system = Self {
            inertia,
            sources,
            modes,
        };
        system.validate_state()?;
        if system.momentum_vector().iter().any(|p| !p.is_finite()) {
            return Err(invalid("spectral_momentum"));
        }
        Ok(system)
    }
    fn force_vector(&self, s: &GaussianBody) -> [f64; 3] {
        let mut force = [0.0; 3];
        for m in &self.modes {
            let (sin, cos) = dot(m.wave, coordinates(s.body)).sin_cos();
            let scalar = s.charge * factor(s, m.k) * (m.a * sin - m.b * cos);
            for (axis, value) in force.iter_mut().enumerate() {
                *value += m.wave[axis] * scalar;
            }
        }
        force
    }
    fn momentum_vector(&self) -> [f64; 3] {
        let mut result = [0.0; 3];
        for s in &self.sources {
            let p = s.body.momentum();
            for (a, b) in result
                .iter_mut()
                .zip([p.x.value(), p.y.value(), p.z.value()])
            {
                *a += b;
            }
        }
        for m in &self.modes {
            for (axis, value) in result.iter_mut().enumerate() {
                *value += m.wave[axis] * (m.a * m.pb - m.b * m.pa);
            }
        }
        result
    }
    fn energy(&self) -> Result<f64, BaseError> {
        let mut energy = 0.0;
        for s in &self.sources {
            energy += s.body.kinetic_energy()?.value();
        }
        for m in &self.modes {
            energy += 0.5
                * ((m.pa * m.pa + m.pb * m.pb) / self.inertia
                    + self.inertia * (m.k * m.k + 1.0) * (m.a * m.a + m.b * m.b));
            for s in &self.sources {
                let (sin, cos) = dot(m.wave, coordinates(s.body)).sin_cos();
                energy += s.charge * factor(s, m.k) * (m.a * cos + m.b * sin);
            }
        }
        if !energy.is_finite() {
            return Err(invalid("spectral_energy"));
        }
        Ok(energy)
    }
    fn kick_update(&mut self, dt: f64) -> Result<(), BaseError> {
        let forces: Vec<_> = self.sources.iter().map(|s| self.force_vector(s)).collect();
        for (s, f) in self.sources.iter_mut().zip(forces) {
            s.body
                .apply_impulse(Vec3::try_values(dt * f[0], dt * f[1], dt * f[2])?)?;
        }
        for m in &mut self.modes {
            let mut ca = 0.0;
            let mut cb = 0.0;
            for s in &self.sources {
                let (sin, cos) = dot(m.wave, coordinates(s.body)).sin_cos();
                ca += s.charge * factor(s, m.k) * cos;
                cb += s.charge * factor(s, m.k) * sin;
            }
            m.pa -= dt * (self.inertia * (m.k * m.k + 1.0) * m.a + ca);
            m.pb -= dt * (self.inertia * (m.k * m.k + 1.0) * m.b + cb);
        }
        Ok(())
    }
    /// Full second-order kick-drift-kick step of the closed body/field system.
    /// Momentum is conserved to rounding; energy has integration error.
    pub fn advance(&mut self, dt: TimeStep) -> Result<(), BaseError> {
        // Necessary stability bound for the highest free field oscillator.
        // Coupling/accuracy can require a smaller step; this is not an error estimator.
        if self
            .modes
            .iter()
            .any(|m| dt.seconds() * (m.k * m.k + 1.0).sqrt() >= 2.0)
        {
            return Err(BaseError::new(
                ErrorCode::NumericalInstability,
                "spectral_field_timestep",
            ));
        }
        let mut next = self.clone();
        next.kick_update(dt.seconds() / 2.0)?;
        for s in &mut next.sources {
            s.body.advance_free(dt)?;
        }
        for m in &mut next.modes {
            m.a += dt.seconds() * m.pa / self.inertia;
            m.b += dt.seconds() * m.pb / self.inertia;
        }
        next.kick_update(dt.seconds() / 2.0)?;
        next.validate_state()?;
        if next.momentum_vector().iter().any(|p| !p.is_finite()) {
            return Err(invalid("spectral_momentum"));
        }
        *self = next;
        Ok(())
    }
    pub fn sources(&self) -> &[GaussianBody] {
        &self.sources
    }
    pub fn modes(&self) -> &[SpectralMode] {
        &self.modes
    }
    pub fn total_energy(&self) -> Result<Quantity<Joules>, BaseError> {
        Quantity::new(self.energy()?)
    }
    pub fn total_momentum(&self) -> Result<Vec3<KilogramMetresPerSecond>, BaseError> {
        let p = self.momentum_vector();
        Vec3::try_values(p[0], p[1], p[2])
    }
    pub fn force_on(&self, index: usize) -> Result<Vec3<Newtons>, BaseError> {
        let source = self
            .sources
            .get(index)
            .ok_or_else(|| invalid("spectral_source_index"))?;
        let f = self.force_vector(source);
        Vec3::try_values(f[0], f[1], f[2])
    }
    fn validate_state(&self) -> Result<(), BaseError> {
        for m in &self.modes {
            if [m.a, m.b, m.pa, m.pb].iter().any(|v| !v.is_finite()) {
                return Err(invalid("spectral_storage"));
            }
        }
        self.total_energy()?;
        self.total_momentum()?;
        Ok(())
    }
    /// Apply an external instantaneous impulse at a fixed source position,
    /// booking its signed kinetic work. Body, field state and ledger commit
    /// together. Field amplitudes and source charge/width remain fixed.
    pub fn apply_external_impulse(
        &mut self,
        index: usize,
        impulse: Vec3<KilogramMetresPerSecond>,
        ledger: &mut super::ExternalLoadLedger,
    ) -> Result<(), BaseError> {
        if index >= self.sources.len() {
            return Err(invalid("spectral_source_index"));
        }
        let mut next = self.clone();
        let mut next_ledger = *ledger;
        next_ledger.apply_impulse(&mut next.sources[index].body, impulse)?;
        next.validate_state()?;
        *self = next;
        *ledger = next_ledger;
        Ok(())
    }
    /// Replace [A,B,PA,PB] for each mode atomically, preserving geometry.
    /// This validates finite state, not conservation with the previous snapshot;
    /// importing accelerators must audit their own energy/momentum balances.
    pub fn replace_mode_storage(&mut self, storage: &[[f64; 4]]) -> Result<(), BaseError> {
        if storage.len() != self.modes.len() {
            return Err(invalid("spectral_storage_shape"));
        }
        let mut next = self.clone();
        for (mode, values) in next.modes.iter_mut().zip(storage) {
            [mode.a, mode.b, mode.pa, mode.pb] = *values;
        }
        next.validate_state()?;
        *self = next;
        Ok(())
    }
    /// Potential-Hamiltonian substep at fixed positions and amplitudes.
    /// Use advance() for a complete second-order physical step.
    pub fn potential_kick(&mut self, dt: TimeStep) -> Result<(), BaseError> {
        let mut next = self.clone();
        next.kick_update(dt.seconds())?;
        next.validate_state()?;
        *self = next;
        Ok(())
    }
}

/// Periodic rectangular domain in metres, with paired +/- Fourier vectors.
/// Fixed physical normalization a=epsilon=screening=1/pi; zero mode excluded.
#[derive(Clone, Debug, PartialEq)]
pub struct PeriodicSpectralDomain {
    inertia: f64,
    waves: Vec<[f64; 3]>,
}
impl PeriodicSpectralDomain {
    pub fn new(
        lengths: Vec3<manonica_core::units::Metres>,
        cutoff: [usize; 3],
    ) -> Result<Self, BaseError> {
        let lengths = [lengths.x.value(), lengths.y.value(), lengths.z.value()];
        if lengths.iter().any(|v| *v <= 0.0) || cutoff.iter().any(|v| *v > 4096) {
            return Err(invalid("spectral_domain"));
        }
        let count = cutoff
            .iter()
            .try_fold(1_usize, |product, n| product.checked_mul(2 * n + 1))
            .ok_or_else(|| invalid("spectral_mode_count"))?;
        if count <= 1 || (count - 1) / 2 > 65_535 {
            return Err(invalid("spectral_mode_count"));
        }
        let inertia = lengths.iter().product::<f64>() / std::f64::consts::TAU;
        if !inertia.is_finite() || inertia <= 0.0 {
            return Err(invalid("spectral_inertia"));
        }
        let limits = cutoff.map(|n| n as i32);
        let mut waves = Vec::with_capacity((count - 1) / 2);
        for x in -limits[0]..=limits[0] {
            for y in -limits[1]..=limits[1] {
                for z in -limits[2]..=limits[2] {
                    if x > 0 || (x == 0 && y > 0) || (x == 0 && y == 0 && z > 0) {
                        let indices = [x, y, z];
                        let wave = std::array::from_fn(|i| {
                            indices[i] as f64 * std::f64::consts::TAU / lengths[i]
                        });
                        let squared = dot(wave, wave);
                        if !squared.is_finite() || squared <= 0.0 {
                            return Err(invalid("spectral_wave"));
                        }
                        waves.push(wave);
                    }
                }
            }
        }
        Ok(Self { inertia, waves })
    }
    pub fn mode_count(&self) -> usize {
        self.waves.len()
    }
    pub fn static_equilibrium(
        &self,
        sources: Vec<GaussianBody>,
    ) -> Result<SpectralMechanics, BaseError> {
        SpectralMechanics::static_equilibrium(sources, self.waves.clone(), self.inertia)
    }
}
