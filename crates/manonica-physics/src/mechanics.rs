//! Typed translational point mechanics, without field or Minecraft ownership.
use manonica_core::{
    Quantity, TimeStep, Vec3,
    units::{Joules, KilogramMetresPerSecond, Kilograms, Metres, Newtons, Unit},
};
use manonica_error::{BaseError, ErrorCode};

fn values<U: Unit>(v: Vec3<U>) -> [f64; 3] {
    [v.x.value(), v.y.value(), v.z.value()]
}
fn vector<U: Unit>(v: [f64; 3]) -> Result<Vec3<U>, BaseError> {
    Vec3::try_values(v[0], v[1], v[2])
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MechanicalBody {
    mass: Quantity<Kilograms>,
    position: Vec3<Metres>,
    momentum: Vec3<KilogramMetresPerSecond>,
}
impl MechanicalBody {
    pub fn new(
        mass: Quantity<Kilograms>,
        position: Vec3<Metres>,
        momentum: Vec3<KilogramMetresPerSecond>,
    ) -> Result<Self, BaseError> {
        if mass.value() <= 0.0 {
            return Err(BaseError::new(ErrorCode::InvalidRange, "body_mass"));
        }
        let body = Self {
            mass,
            position,
            momentum,
        };
        body.kinetic_energy()?;
        Ok(body)
    }
    pub fn mass(self) -> Quantity<Kilograms> {
        self.mass
    }
    pub fn position(self) -> Vec3<Metres> {
        self.position
    }
    pub fn momentum(self) -> Vec3<KilogramMetresPerSecond> {
        self.momentum
    }
    pub fn kinetic_energy(self) -> Result<Quantity<Joules>, BaseError> {
        // Multiply by velocity/2 to avoid squaring a large momentum first.
        Quantity::nonnegative(
            values(self.momentum)
                .iter()
                .map(|p| p * (p / self.mass.value() / 2.0))
                .sum(),
        )
    }
    /// Exact momentum kick at fixed position. Return signed kinetic work.
    /// Internal interactions must book the opposite impulse in their partner.
    pub fn apply_impulse(
        &mut self,
        impulse: Vec3<KilogramMetresPerSecond>,
    ) -> Result<Quantity<Joules>, BaseError> {
        let p = values(self.momentum);
        let j = values(impulse);
        let momentum = vector(std::array::from_fn(|i| p[i] + j[i]))?;
        let candidate = Self { momentum, ..*self };
        // J dot (p_before+p_after)/(2m): exact impulse work with less cancellation.
        let work = Quantity::new(
            (0..3)
                .map(|i| j[i] * (p[i] / self.mass.value() + j[i] / self.mass.value() / 2.0))
                .sum(),
        )?;
        candidate.kinetic_energy()?;
        *self = candidate;
        Ok(work)
    }
}

/// Signed impulse/work delivered by external agents to the tracked system.
/// The external support receives the opposite impulse and loses this work.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExternalLoadLedger {
    impulse: Vec3<KilogramMetresPerSecond>,
    work: Quantity<Joules>,
}
impl Default for ExternalLoadLedger {
    fn default() -> Self {
        Self {
            impulse: Vec3::try_values(0.0, 0.0, 0.0).expect("finite zero"),
            work: Quantity::new(0.0).expect("finite zero"),
        }
    }
}
impl ExternalLoadLedger {
    pub fn impulse(self) -> Vec3<KilogramMetresPerSecond> {
        self.impulse
    }
    pub fn work(self) -> Quantity<Joules> {
        self.work
    }
    /// Commit both body and ledger, or neither, including ledger overflow.
    pub fn apply_impulse(
        &mut self,
        body: &mut MechanicalBody,
        impulse: Vec3<KilogramMetresPerSecond>,
    ) -> Result<(), BaseError> {
        let mut candidate = *body;
        let work = candidate.apply_impulse(impulse)?;
        let before = values(self.impulse);
        let added = values(impulse);
        let ledger = Self {
            impulse: vector(std::array::from_fn(|i| before[i] + added[i]))?,
            work: Quantity::new(self.work.value() + work.value())?,
        };
        *body = candidate;
        *self = ledger;
        Ok(())
    }
    /// Exact free-body trajectory under a constant prescribed force over dt.
    /// Do not use as a substitute for coupled field dynamics.
    pub fn advance_constant_force(
        &mut self,
        body: &mut MechanicalBody,
        force: Vec3<Newtons>,
        dt: TimeStep,
    ) -> Result<(), BaseError> {
        let impulse = vector(values(force).map(|f| f * dt.seconds()))?;
        let mut candidate = *body;
        let mut ledger = *self;
        ledger.apply_impulse(&mut candidate, impulse)?;
        let position = values(body.position);
        let p = values(body.momentum);
        let j = values(impulse);
        candidate.position = vector(std::array::from_fn(|i| {
            position[i] + dt.seconds() * (p[i] / body.mass.value() + j[i] / body.mass.value() / 2.0)
        }))?;
        *body = candidate;
        *self = ledger;
        Ok(())
    }
}
