//! Bounded, pure formula programs. No object IDs, block commands or effect-name dispatch.
use super::*;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ScalarFormula(pub String);
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorFormula {
    pub x: ScalarFormula,
    pub y: ScalarFormula,
    pub z: ScalarFormula,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldControls {
    pub potential_source: ScalarFormula,
    pub manon_velocity: VectorFormula,
    pub heat_power: ScalarFormula,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FormulaDomain {
    pub dimensions: [usize; 3],
    pub anchor: [f64; 3],
    pub duration_seconds: f64,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportSettings {
    pub diffusivity: f64,
    pub mobility: f64,
    pub permittivity: f64,
    pub screening: f64,
    pub iterations: usize,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkBudget {
    pub total_joules: f64,
    pub max_power_watts: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManonProgram {
    pub domain: FormulaDomain,
    pub transport: TransportSettings,
    pub work: WorkBudget,
    pub parameters: BTreeMap<String, f64>,
    pub controls: FieldControls,
}
#[derive(Clone, Debug)]
pub(crate) struct CompiledProgram {
    pub source: meval::Expr,
    pub velocity: [meval::Expr; 3],
    pub heat: meval::Expr,
}
thread_local! {static FUNCTIONS:meval::Context<'static>=meval::Context::new();}
#[derive(Clone, Copy)]
pub(crate) struct FormulaPoint {
    pub xyz: [f64; 3],
    pub time: f64,
    pub density: f64,
}
impl ManonProgram {
    pub(crate) fn compile(&self) -> WorldResult<CompiledProgram> {
        let d = self.domain;
        if d.dimensions.iter().any(|v| *v < 3 || *v > 24)
            || d.dimensions.iter().product::<usize>() > 4096
            || d.anchor
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            || !d.duration_seconds.is_finite()
            || d.duration_seconds <= 0.0
            || d.duration_seconds > 60.0
        {
            return Err(WorldError::Invalid("program_domain"));
        }
        let s = self.transport;
        if [s.diffusivity, s.mobility]
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=2.0).contains(v))
            || !s.permittivity.is_finite()
            || s.permittivity <= 0.0
            || !s.screening.is_finite()
            || s.screening <= 0.0
            || !(1..=128).contains(&s.iterations)
        {
            return Err(WorldError::Invalid("program_transport"));
        }
        if [self.work.total_joules, self.work.max_power_watts]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0 || *v > 1e8)
            || self.parameters.len() > 32
        {
            return Err(WorldError::Invalid("program_work"));
        }
        for (key, value) in &self.parameters {
            if key.is_empty()
                || key.len() > 32
                || !key
                    .bytes()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_')
                || ["x", "y", "z", "t", "r", "n", "pi", "e"].contains(&key.as_str())
                || !value.is_finite()
            {
                return Err(WorldError::Invalid("program_parameter"));
            }
        }
        let parse = |formula: &ScalarFormula| -> WorldResult<meval::Expr> {
            if formula.0.len() > 512 {
                return Err(WorldError::Invalid("formula_length"));
            }
            let expr = formula
                .0
                .parse::<meval::Expr>()
                .map_err(|e| WorldError::Codec(format!("formula: {e}")))?;
            if expr.len() > 256 {
                return Err(WorldError::Invalid("formula_complexity"));
            }
            let mut variables = vec!["x", "y", "z", "t", "r", "n"];
            variables.extend(self.parameters.keys().map(String::as_str));
            let _bound = expr
                .clone()
                .bindn(&variables)
                .map_err(|e| WorldError::Codec(format!("formula variables: {e}")))?;
            Ok(expr)
        };
        let v = &self.controls.manon_velocity;
        Ok(CompiledProgram {
            source: parse(&self.controls.potential_source)?,
            velocity: [parse(&v.x)?, parse(&v.y)?, parse(&v.z)?],
            heat: parse(&self.controls.heat_power)?,
        })
    }
    pub(crate) fn evaluate(&self, expr: &meval::Expr, p: FormulaPoint) -> WorldResult<f64> {
        let [x, y, z] = p.xyz;
        let vars = [
            ("x", x),
            ("y", y),
            ("z", z),
            ("t", p.time),
            ("r", x.hypot(y).hypot(z)),
            ("n", p.density),
        ];
        let result = FUNCTIONS
            .with(|functions| expr.eval_with_context((vars, (&self.parameters, functions))))
            .map_err(|e| WorldError::Codec(format!("formula evaluation: {e}")))?;
        if !result.is_finite() || result.abs() > 1e6 {
            return Err(WorldError::Invalid("formula_nonfinite_or_range"));
        }
        Ok(result)
    }
}
