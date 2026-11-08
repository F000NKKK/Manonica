use std::fmt::Debug;

/// Тип-маркер физической размерности, а не название визуального эффекта.
pub trait Unit: Copy + Clone + Debug + Send + Sync + 'static {}

macro_rules! units {
    ($($name:ident),+ $(,)?) => {
        $(
            #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
            pub struct $name;
            impl Unit for $name {}
        )+
    };
}

units!(
    Dimensionless,
    Metres,
    MetresPerSecond,
    Seconds,
    Hertz,
    Radians,
    Joules,
    ManonCount,
    ManonDensity,
    ManonFlux,
    ManaPotential,
    Diffusivity,
    Newtons,
    Kelvin,
);

/// Типизированная скорость изменения величины единиц U за секунду.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PerSecond<U: Unit>(std::marker::PhantomData<U>);
impl<U: Unit> Unit for PerSecond<U> {}
