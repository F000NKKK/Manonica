//! A deterministic graphical notation of compiled field-control expressions.
//! Glyph curves are data; no spell name, field mutation or physical effect is encoded here.
use super::*;
use meval::tokenizer::{Operation, Token};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurveFormula {
    pub x: String,
    pub y: String,
    pub z: String,
    pub parameter_range: [f64; 2],
    pub samples: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlyphRule {
    pub symbol: String,
    pub curve: CurveFormula,
}
#[derive(Clone, Debug)]
pub struct CircleGrammar {
    rules: BTreeMap<String, GlyphRule>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ConstructionNode {
    pub symbol: String,
    pub value: Option<f64>,
    pub operands: Vec<usize>,
    pub depth: usize,
    pub center: [f64; 3],
}
#[derive(Clone, Debug, PartialEq)]
pub struct MagicConstruction {
    pub nodes: Vec<ConstructionNode>,
    pub paths: Vec<Vec<[f64; 3]>>,
    pub links: Vec<[[f64; 3]; 2]>,
    pub fingerprint: u64,
}
impl CircleGrammar {
    pub fn from_jsonl(text: &str) -> WorldResult<Self> {
        if text.len() > 1024 * 1024 {
            return Err(WorldError::Invalid("circle_grammar_limit"));
        }
        let mut rules = BTreeMap::new();
        for (i, line) in text
            .lines()
            .enumerate()
            .filter(|(_, s)| !s.trim().is_empty())
        {
            if line.len() > 8192 {
                return Err(WorldError::Invalid("glyph_line_limit"));
            }
            let rule: GlyphRule = serde_json::from_str(line).map_err(|e| WorldError::Content {
                source: "circles.jsonl",
                line: i + 1,
                detail: e.to_string(),
            })?;
            if rule.symbol.len() > 64
                || rule.symbol.is_empty()
                || rule.curve.samples < 8
                || rule.curve.samples > 256
                || rule.curve.parameter_range.iter().any(|v| !v.is_finite())
                || rule.curve.parameter_range[0] >= rule.curve.parameter_range[1]
            {
                return Err(WorldError::Invalid("glyph_rule"));
            }
            sample_curve(&rule.curve, 1.0, 0.0)?;
            if rules.insert(rule.symbol.clone(), rule).is_some() {
                return Err(WorldError::Invalid("duplicate_glyph_rule"));
            }
        }
        for required in [
            "constant",
            "variable",
            "function",
            "add",
            "subtract",
            "multiply",
            "divide",
            "power",
            "remainder",
            "negate",
            "root",
        ] {
            if !rules.contains_key(required) {
                return Err(WorldError::Invalid("missing_glyph_rule"));
            }
        }
        Ok(Self { rules })
    }
    pub fn compile(&self, program: &ManonProgram) -> WorldResult<MagicConstruction> {
        let compiled = program.compile()?;
        let expressions = [
            &compiled.source,
            &compiled.velocity[0],
            &compiled.velocity[1],
            &compiled.velocity[2],
            &compiled.heat,
        ];
        let mut nodes = vec![];
        let mut roots = vec![];
        for expr in expressions {
            let mut stack = vec![];
            for token in expr.iter() {
                let (symbol, value, arity) = match token {
                    Token::Number(value) => ("constant".to_string(), Some(*value), 0),
                    Token::Var(name) => (
                        format!("variable:{name}"),
                        program.parameters.get(name).copied(),
                        0,
                    ),
                    Token::Unary(_) => ("negate".into(), None, 1),
                    Token::Binary(op) => (
                        match op {
                            Operation::Plus => "add",
                            Operation::Minus => "subtract",
                            Operation::Times => "multiply",
                            Operation::Div => "divide",
                            Operation::Pow => "power",
                            Operation::Rem => "remainder",
                        }
                        .into(),
                        None,
                        2,
                    ),
                    Token::Func(name, Some(arity)) => (format!("function:{name}"), None, *arity),
                    _ => return Err(WorldError::Invalid("construction_token")),
                };
                if stack.len() < arity {
                    return Err(WorldError::Invalid("construction_stack"));
                }
                let operands = stack.split_off(stack.len() - arity);
                let depth = operands
                    .iter()
                    .map(|i: &usize| {
                        nodes
                            .get(*i)
                            .map(|n: &ConstructionNode| n.depth + 1)
                            .unwrap_or(0)
                    })
                    .max()
                    .unwrap_or(0);
                let index = nodes.len();
                nodes.push(ConstructionNode {
                    symbol,
                    value,
                    operands,
                    depth,
                    center: [0.; 3],
                });
                stack.push(index);
            }
            if stack.len() != 1 {
                return Err(WorldError::Invalid("construction_root"));
            }
            roots.push(stack[0]);
        }
        if nodes.len() > 1280 {
            return Err(WorldError::Invalid("construction_node_limit"));
        }
        // Five physical channels occupy five planes. Dependency depth is radial;
        // operand order is angular. Coefficients affect scale and chirality below.
        let mut begin = 0;
        for (channel, &root) in roots.iter().enumerate() {
            let end = root + 1;
            let max_depth = nodes[begin..end].iter().map(|n| n.depth).max().unwrap_or(0);
            for depth in 0..=max_depth {
                let ids = (begin..end)
                    .filter(|i| nodes[*i].depth == depth)
                    .collect::<Vec<_>>();
                let count = ids.len();
                for (ordinal, id) in ids.into_iter().enumerate() {
                    let angle = std::f64::consts::TAU * ordinal as f64 / count.max(1) as f64
                        + channel as f64 * 0.37;
                    let radius = 0.35 + (max_depth - depth) as f64 * 0.3;
                    nodes[id].center = [
                        angle.cos() * radius,
                        channel as f64 * 0.18,
                        angle.sin() * radius,
                    ];
                }
            }
            begin = end;
        }
        let mut paths = vec![];
        let mut links = vec![];
        for node in &nodes {
            let short = node.symbol.split(':').next().unwrap();
            let rule = self
                .rules
                .get(&node.symbol)
                .or_else(|| self.rules.get(short))
                .ok_or(WorldError::Invalid("missing_glyph"))?;
            let value = node.value.unwrap_or(1.0);
            let amplitude = 0.09 + 0.025 * value.abs().ln_1p().min(6.0);
            let phase = if value < 0.0 { -1.0 } else { 1.0 };
            let raw = sample_curve(&rule.curve, amplitude, phase)?;
            let path = raw
                .into_iter()
                .map(|p| std::array::from_fn(|a| p[a] + node.center[a]))
                .collect();
            paths.push(path);
            for operand in &node.operands {
                links.push([node.center, nodes[*operand].center]);
            }
        }
        // Concentric support rings encode expression depth; they are notation,
        // not the isopotential debug contours of the simulated scalar field.
        let outer = nodes
            .iter()
            .map(|n| n.center[0].hypot(n.center[2]))
            .fold(0., f64::max)
            + 0.25;
        let ring = &self.rules["root"].curve;
        for radius in [outer, outer + 0.08] {
            paths.push(sample_curve(ring, radius, 1.0)?);
        }
        let mut hash = 0xcbf29ce484222325_u64;
        for path in &paths {
            for p in path {
                for value in p {
                    for b in value.to_bits().to_le_bytes() {
                        hash = (hash ^ u64::from(b)).wrapping_mul(0x100000001b3);
                    }
                }
            }
        }
        for link in &links {
            for p in link {
                for v in p {
                    for b in v.to_bits().to_le_bytes() {
                        hash = (hash ^ u64::from(b)).wrapping_mul(0x100000001b3);
                    }
                }
            }
        }
        Ok(MagicConstruction {
            nodes,
            paths,
            links,
            fingerprint: hash,
        })
    }
}
fn sample_curve(curve: &CurveFormula, a: f64, phase: f64) -> WorldResult<Vec<[f64; 3]>> {
    let parse = |s: &str| -> WorldResult<meval::Expr> {
        if s.len() > 256 {
            return Err(WorldError::Invalid("glyph_expression_length"));
        }
        s.parse()
            .map_err(|e| WorldError::Codec(format!("glyph formula: {e}")))
    };
    let expr = [parse(&curve.x)?, parse(&curve.y)?, parse(&curve.z)?];
    let mut points = Vec::with_capacity(curve.samples + 1);
    let functions = meval::Context::new();
    for i in 0..=curve.samples {
        let u = curve.parameter_range[0]
            + (curve.parameter_range[1] - curve.parameter_range[0]) * i as f64
                / curve.samples as f64;
        let vars = [("u", u), ("a", a), ("s", phase)];
        let mut p = [0.; 3];
        for axis in 0..3 {
            p[axis] = expr[axis]
                .eval_with_context((vars, &functions))
                .map_err(|e| WorldError::Codec(format!("glyph evaluation: {e}")))?;
            if !p[axis].is_finite() || p[axis].abs() > 100.0 {
                return Err(WorldError::Invalid("glyph_coordinate"));
            }
        }
        points.push(p);
    }
    Ok(points)
}
