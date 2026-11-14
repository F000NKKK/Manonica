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
    pub symmetry: SymmetryGroup,
    pub symbol: String,
    pub curve: CurveFormula,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymmetryAxis {
    X,
    Y,
    Z,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SymmetryGroup {
    pub order: usize,
    pub reflections: bool,
    pub axis: SymmetryAxis,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ConstructionLayer {
    pub channel: usize,
    pub symmetry: SymmetryGroup,
    pub paths: [usize; 2],
    pub links: [usize; 2],
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
    pub layers: Vec<ConstructionLayer>,
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
            if rule.symmetry.order == 0
                || rule.symmetry.order > 12
                || rule.symbol.len() > 64
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
        let mut paths: Vec<Vec<[f64; 3]>> = vec![];
        let mut links: Vec<[[f64; 3]; 2]> = vec![];
        let mut layers = vec![];
        let mut begin = 0;
        for (channel, &root) in roots.iter().enumerate() {
            let end = root + 1;
            if end - begin == 1
                && nodes[root].symbol == "constant"
                && nodes[root].value == Some(0.0)
            {
                begin = end;
                continue;
            }
            let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
            for (id, node) in nodes.iter().enumerate().take(end).skip(begin) {
                groups.entry(node.symbol.clone()).or_default().push(id);
            }
            let axis = match channel {
                1 => SymmetryAxis::X,
                3 => SymmetryAxis::Z,
                _ => SymmetryAxis::Y,
            };
            let root_group = self.rules["root"].symmetry;
            let symmetry = SymmetryGroup { axis, ..root_group };
            let start_paths = paths.len();
            let start_links = links.len();
            let mut centers = BTreeMap::new();
            let mut base_paths = vec![];
            for (ordinal, (symbol, ids)) in groups.iter().enumerate() {
                let rule = self
                    .rules
                    .get(symbol)
                    .or_else(|| self.rules.get(symbol.split(':').next().unwrap()))
                    .ok_or(WorldError::Invalid("missing_glyph"))?;
                let ring = ordinal / 3;
                let sector = ordinal % 3;
                let angle =
                    (sector as f64 - 1.0) * std::f64::consts::PI / (symmetry.order as f64 * 3.0);
                let average_depth =
                    ids.iter().map(|i| nodes[*i].depth as f64).sum::<f64>() / ids.len() as f64;
                let radius = 0.65 + ring as f64 * 0.34 + average_depth * 0.015;
                let center = [radius * angle.cos(), 0.0, radius * angle.sin()];
                centers.insert(symbol.clone(), center);
                let magnitude = ids
                    .iter()
                    .filter_map(|i| nodes[*i].value)
                    .map(|v| v.abs().ln_1p())
                    .sum::<f64>();
                let negative = ids
                    .iter()
                    .filter_map(|i| nodes[*i].value)
                    .filter(|v| *v < 0.0)
                    .count();
                let amplitude =
                    0.055 + 0.008 * magnitude.min(8.0) + 0.004 * (ids.len() as f64).ln_1p();
                let phase = if negative % 2 == 1 { -1.0 } else { 1.0 };
                let raw = sample_curve(&rule.curve, amplitude, phase)?;
                for reflected in 0..=usize::from(rule.symmetry.reflections) {
                    for rotation in 0..rule.symmetry.order {
                        let path = raw
                            .iter()
                            .map(|p| {
                                let q = group_action(
                                    *p,
                                    rule.symmetry.axis,
                                    rotation,
                                    rule.symmetry.order,
                                    reflected == 1,
                                );
                                std::array::from_fn(|i| q[i] + center[i])
                            })
                            .collect::<Vec<_>>();
                        base_paths.push(path);
                    }
                }
            }
            let mut base_links = vec![];
            let mut seen = BTreeSet::new();
            for id in begin..end {
                nodes[id].center = orient(centers[&nodes[id].symbol], axis);
                for operand in &nodes[id].operands {
                    let pair = (nodes[id].symbol.clone(), nodes[*operand].symbol.clone());
                    if pair.0 != pair.1 && seen.insert(pair.clone()) {
                        base_links.push([centers[&pair.0], centers[&pair.1]]);
                    }
                }
            }
            let outer = centers
                .values()
                .map(|p| p[0].hypot(p[2]))
                .fold(0.0, f64::max)
                + 0.22;
            for radius in [outer, outer + 0.08] {
                base_paths.push(sample_curve(&self.rules["root"].curve, radius, 1.0)?);
            }
            for reflected in 0..=usize::from(symmetry.reflections) {
                for rotation in 0..symmetry.order {
                    for path in &base_paths {
                        paths.push(
                            path.iter()
                                .map(|p| {
                                    orient(
                                        group_action(
                                            *p,
                                            SymmetryAxis::Y,
                                            rotation,
                                            symmetry.order,
                                            reflected == 1,
                                        ),
                                        axis,
                                    )
                                })
                                .collect(),
                        );
                    }
                    for link in &base_links {
                        links.push(link.map(|p| {
                            orient(
                                group_action(
                                    p,
                                    SymmetryAxis::Y,
                                    rotation,
                                    symmetry.order,
                                    reflected == 1,
                                ),
                                axis,
                            )
                        }));
                    }
                }
            }
            layers.push(ConstructionLayer {
                channel,
                symmetry,
                paths: [start_paths, paths.len()],
                links: [start_links, links.len()],
            });
            begin = end;
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
            layers,
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

fn orient(p: [f64; 3], axis: SymmetryAxis) -> [f64; 3] {
    match axis {
        SymmetryAxis::X => [p[1], p[0], p[2]],
        SymmetryAxis::Y => p,
        SymmetryAxis::Z => [p[0], p[2], p[1]],
    }
}
/// Conjugate the standard action about Y onto a principal axis.
pub fn group_action(
    p: [f64; 3],
    axis: SymmetryAxis,
    rotation: usize,
    order: usize,
    reflected: bool,
) -> [f64; 3] {
    let mut p = orient(p, axis);
    if reflected {
        p[2] = -p[2];
    }
    let angle = std::f64::consts::TAU * rotation as f64 / order as f64;
    let (s, c) = angle.sin_cos();
    orient([p[0] * c - p[2] * s, p[1], p[0] * s + p[2] * c], axis)
}
