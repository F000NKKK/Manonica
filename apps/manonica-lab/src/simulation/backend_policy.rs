//! Explicit selection policy; the numerical physics backend is independent
//! of the renderer. Neither automatic fallback nor forced GPU is implicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendPreference {
    #[default]
    Auto,
    Cpu,
    Gpu,
}
impl BackendPreference {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            other => Err(format!(
                "unknown physics backend '{other}'; valid values: auto, gpu, cpu"
            )),
        }
    }
    /// The CLI wins over the environment variable; without either, try GPU.
    pub fn resolve(cli: Option<&str>, env: Option<&str>) -> Result<Self, String> {
        Self::parse(cli.or(env).unwrap_or("auto"))
    }
    pub fn attempts_gpu(self) -> bool { self != Self::Cpu }
    pub fn allows_cpu_fallback(self) -> bool { self == Self::Auto }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_is_gpu_first_and_cpu_fallback() {
        let mode = BackendPreference::resolve(None, None).unwrap();
        assert_eq!(mode, BackendPreference::Auto);
        assert!(mode.attempts_gpu());
        assert!(mode.allows_cpu_fallback());
    }
    #[test]
    fn forced_device_never_silently_falls_back() {
        assert!(!BackendPreference::Cpu.attempts_gpu());
        assert!(!BackendPreference::Cpu.allows_cpu_fallback());
        assert!(BackendPreference::Gpu.attempts_gpu());
        assert!(!BackendPreference::Gpu.allows_cpu_fallback());
    }
    #[test]
    fn cli_overrides_environment_and_invalid_modes_fail() {
        assert_eq!(BackendPreference::resolve(Some("CPU"), Some("gpu")).unwrap(), BackendPreference::Cpu);
        assert_eq!(BackendPreference::resolve(None, Some("GPU")).unwrap(), BackendPreference::Gpu);
        assert!(BackendPreference::resolve(Some("cuda"), None).is_err());
    }
}
