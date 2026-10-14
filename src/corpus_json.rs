//! Strict corpus JSON: preserve numeric lexemes and reject duplicate keys.

use anyhow::{Result, anyhow, bail};
use serde_json::value::RawValue;

#[derive(Clone)]
pub(crate) enum Node {
    Object(Vec<(String, Self)>),
    Array(Vec<Self>),
    String(String),
    Scalar(String),
}

impl Node {
    pub(crate) fn parse(text: &str) -> Result<Self> {
        let raw: &RawValue =
            serde_json::from_str(text).map_err(|_| anyhow!("invalid corpus JSON"))?;
        Self::from_raw(raw, 0)
    }

    fn from_raw(raw: &RawValue, depth: usize) -> Result<Self> {
        if depth > 64 {
            bail!("corpus JSON depth limit exceeded");
        }
        let text = raw.get();
        Ok(match text.as_bytes().first() {
            Some(b'{') => {
                let mut parser = serde_json::Deserializer::from_str(text);
                let members: Vec<(String, &RawValue)> = crate::unique_json::members(&mut parser)
                    .map_err(|_| anyhow!("invalid or duplicate corpus JSON keys"))?;
                Self::Object(
                    members
                        .into_iter()
                        .map(|(key, value)| Ok((key, Self::from_raw(value, depth + 1)?)))
                        .collect::<Result<_>>()?,
                )
            }
            Some(b'[') => {
                let items: Vec<&RawValue> =
                    serde_json::from_str(text).map_err(|_| anyhow!("invalid corpus JSON array"))?;
                Self::Array(
                    items
                        .into_iter()
                        .map(|value| Self::from_raw(value, depth + 1))
                        .collect::<Result<_>>()?,
                )
            }
            Some(b'"') => Self::String(
                serde_json::from_str(text).map_err(|_| anyhow!("invalid corpus JSON string"))?,
            ),
            _ => Self::Scalar(text.to_string()),
        })
    }

    pub(crate) fn scalar_kind(text: &str) -> &'static str {
        match text {
            "null" => "null",
            "true" | "false" => "boolean",
            _ => "number",
        }
    }

    pub(crate) fn visit(&self, path: &str, out: &mut Vec<(String, String, String)>) {
        match self {
            Self::Object(members) => {
                if members.is_empty() {
                    out.push((path.into(), "empty_object".into(), "{}".into()));
                }
                let media = members.iter().any(|(key, value)| {
                    key == "mimeType"
                        || (key == "type" && matches!(value, Self::String(s) if s == "image"))
                });
                for (key, value) in members {
                    let next = format!("{path}/{}", key.replace('~', "~0").replace('/', "~1"));
                    out.push((format!("{next}#key"), "key".into(), key.clone()));
                    if media
                        && key == "data"
                        && let Self::String(data) = value
                    {
                        out.push((next, "media".into(), data.clone()));
                    } else {
                        value.visit(&next, out);
                    }
                }
            }
            Self::Array(items) => {
                if items.is_empty() {
                    out.push((path.into(), "empty_array".into(), "[]".into()));
                }
                for (index, value) in items.iter().enumerate() {
                    value.visit(&format!("{path}/{index}"), out);
                }
            }
            Self::String(text) => out.push((path.into(), "string".into(), text.clone())),
            Self::Scalar(text) => {
                out.push((path.into(), Self::scalar_kind(text).into(), text.clone()))
            }
        }
    }

    pub(crate) fn render(&self, values: &mut impl Iterator<Item = String>) -> Result<String> {
        Ok(match self {
            Self::Object(members) => {
                if members.is_empty() && values.next().as_deref() != Some("{}") {
                    bail!("empty corpus object changed");
                }
                let mut seen = std::collections::HashSet::new();
                let mut output = Vec::new();
                for (_, value) in members {
                    let key = values.next().ok_or_else(|| anyhow!("missing corpus key"))?;
                    if !seen.insert(key.clone()) {
                        bail!("transformed corpus key collision");
                    }
                    output.push(format!(
                        "{}:{}",
                        serde_json::to_string(&key)?,
                        value.render(values)?
                    ));
                }
                format!("{{{}}}", output.join(","))
            }
            Self::Array(items) => {
                if items.is_empty() && values.next().as_deref() != Some("[]") {
                    bail!("empty corpus array changed");
                }
                format!(
                    "[{}]",
                    items
                        .iter()
                        .map(|value| value.render(values))
                        .collect::<Result<Vec<_>>>()?
                        .join(",")
                )
            }
            Self::String(_) => serde_json::to_string(
                &values
                    .next()
                    .ok_or_else(|| anyhow!("missing corpus string"))?,
            )?,
            Self::Scalar(original) => {
                let text = values
                    .next()
                    .ok_or_else(|| anyhow!("missing corpus scalar"))?;
                let parsed = Self::parse(&text)?;
                if !matches!(parsed, Self::Scalar(ref s) if Self::scalar_kind(s) == Self::scalar_kind(original))
                {
                    bail!("corpus scalar type changed");
                }
                text
            }
        })
    }
}
