//! Explicit discovery-only scalar projection. Strict scanning/redaction are unaffected.
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Cow;

pub const POLICY: &str = "replace_unpaired_utf16_v1";

#[derive(Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Projection {
    pub policy: String,
    pub records: u64,
    pub replacements: u64,
    /// All selected units from repaired records; not necessarily individually changed strings.
    pub record_units: u64,
}
impl Default for Projection {
    fn default() -> Self {
        Self {
            policy: POLICY.into(),
            records: 0,
            replacements: 0,
            record_units: 0,
        }
    }
}
impl Projection {
    pub fn validate(&self, units: u64) -> Result<()> {
        if self.policy != POLICY
            || self.records > 10_000_000
            || self.replacements < self.records
            || (self.records == 0) != (self.replacements == 0)
            || self.record_units > units
            || (self.records == 0 && self.record_units != 0)
        {
            bail!("invalid vocabulary Unicode projection metadata");
        }
        Ok(())
    }
}

pub struct Parsed {
    pub value: Value,
    pub replacements: u64,
}

pub fn parse(raw: &str, number: usize, repair: bool) -> Result<Parsed> {
    let input = if number == 1 {
        raw.strip_prefix('\u{feff}').unwrap_or(raw)
    } else {
        raw
    };
    let (text, replacements) = if repair {
        replace_unpaired(input)?
    } else {
        (Cow::Borrowed(input), 0)
    };
    let value = serde_json::from_str(&text)
        .map_err(|_| anyhow::anyhow!("invalid JSON vocabulary input at line {number}"))?;
    Ok(Parsed {
        value,
        replacements,
    })
}

fn escape(bytes: &[u8], start: usize) -> Option<u16> {
    let raw = bytes.get(start..start + 6)?;
    if &raw[..2] != br"\u" || !raw[2..].iter().all(u8::is_ascii_hexdigit) {
        return None;
    }
    u16::from_str_radix(std::str::from_utf8(&raw[2..]).ok()?, 16).ok()
}

fn escape_step(bytes: &[u8], start: usize) -> (usize, bool) {
    let Some(code) = escape(bytes, start) else {
        return (2, false);
    };
    if (0xd800..=0xdbff).contains(&code)
        && escape(bytes, start + 6).is_some_and(|low| (0xdc00..=0xdfff).contains(&low))
    {
        return (12, false);
    }
    (6, (0xd800..=0xdfff).contains(&code))
}

fn string_end(bytes: &[u8], mut start: usize, positions: &mut Vec<usize>) -> usize {
    while start < bytes.len() && bytes[start] != b'"' {
        if bytes[start] != b'\\' {
            start += 1;
            continue;
        }
        let (width, unpaired) = escape_step(bytes, start);
        if unpaired {
            positions.push(start);
        }
        start += width;
    }
    start
}

fn replace_unpaired(input: &str) -> Result<(Cow<'_, str>, u64)> {
    let bytes = input.as_bytes();
    let mut positions = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'"' {
            i += 1;
            continue;
        }
        i += 1;
        let prior = positions.len();
        i = string_end(bytes, i, &mut positions);
        // Never change selectors or collapse distinct object keys into one scalar key.
        if positions.len() != prior && bytes.get(i) == Some(&b'"') {
            let next = bytes[i + 1..].iter().find(|b| !b.is_ascii_whitespace());
            if next == Some(&b':') {
                bail!("Unicode projection cannot repair JSON object keys; input unchanged");
            }
        }
        i += 1;
    }
    if positions.is_empty() {
        return Ok((Cow::Borrowed(input), 0));
    }
    let count = positions.len() as u64;
    let mut output = bytes.to_vec();
    for start in positions {
        output[start..start + 6].copy_from_slice(br"\ufffd");
    }
    let output =
        String::from_utf8(output).map_err(|_| anyhow::anyhow!("Unicode projection failed"))?;
    Ok((Cow::Owned(output), count))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repair_is_explicit_and_preserves_pairs_and_literal_escape_text() {
        let raw = r#"{"text":"Jane \uD800 Doë \udfff \ud801\udc00 \\ud800"}"#;
        assert!(parse(raw, 1, false).is_err());
        let result = parse(raw, 1, true).unwrap();
        assert_eq!(
            (result.value, result.replacements),
            (
                serde_json::json!({"text":"Jane � Doë � \u{10400} \\ud800"}),
                2
            )
        );
    }
    #[test]
    fn malformed_grammar_keys_and_parser_depth_still_fail() {
        for raw in [
            r#"{"\ud800":"Jane"}"#,
            r#"{"text":"Jane \u12xx"}"#,
            r#"{"text":"Jane \ud800""#,
        ] {
            assert!(parse(raw, 1, true).is_err());
        }
        let raw = format!("{}\"Jane\"{}", "[".repeat(140), "]".repeat(140));
        assert!(parse(&raw, 1, true).is_err());
    }
    #[test]
    fn no_repairs_borrows_and_is_identical_to_strict_parser() {
        let raw = r#"{"text":"� \uD801\uDC00 \ud801\udc00 \\ud800"}"#;
        let (projected, count) = replace_unpaired(raw).unwrap();
        assert!(matches!(projected, Cow::Borrowed(_)));
        assert_eq!(count, 0);
        assert_eq!(
            parse(raw, 1, true).unwrap().value,
            parse(raw, 1, false).unwrap().value
        );
    }
}
