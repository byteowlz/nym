//! Format-aware PII processing.
//!
//! This module provides handlers for processing structured formats (JSON, YAML, TOML)
//! while preserving structure and only modifying string values containing PII.

use serde_json::Value as JsonValue;

use super::detector::{Detector, PiiMatch};
use super::replacer::{Replacement, Replacer};

/// Process JSON, walking all string values and applying PII detection/replacement.
///
/// Returns the processed JSON and all replacements made.
pub fn process_json(
    input: &str,
    detector: &Detector,
    replacer: &mut Replacer,
) -> Result<(String, Vec<Replacement>), serde_json::Error> {
    let mut value: JsonValue = serde_json::from_str(input)?;
    let mut all_replacements = Vec::new();

    process_json_value(&mut value, detector, replacer, &mut all_replacements);

    let output = serde_json::to_string_pretty(&value)?;
    Ok((output, all_replacements))
}

/// Process a JSON value recursively, replacing PII in string values.
fn process_json_value(
    value: &mut JsonValue,
    detector: &Detector,
    replacer: &mut Replacer,
    replacements: &mut Vec<Replacement>,
) {
    match value {
        JsonValue::String(s) => {
            // Detect and replace PII in this string
            let matches = detector.detect(s);
            if !matches.is_empty() {
                let (replaced, new_replacements) = replacer.replace_all(s, &matches);
                *s = replaced;
                replacements.extend(new_replacements);
            }
        }
        JsonValue::Array(arr) => {
            for item in arr {
                process_json_value(item, detector, replacer, replacements);
            }
        }
        JsonValue::Object(obj) => {
            for (_, v) in obj.iter_mut() {
                process_json_value(v, detector, replacer, replacements);
            }
        }
        // Numbers, booleans, null - no PII possible
        _ => {}
    }
}

/// Detect PII in JSON, returning all matches with their JSON paths.
#[derive(Debug, Clone)]
pub struct JsonPiiMatch {
    /// JSON path to the value (e.g., "users[0].email")
    pub path: String,
    /// The PII match details
    pub pii_match: PiiMatch,
}

/// Detect all PII in a JSON document, returning matches with their paths.
pub fn detect_json(
    input: &str,
    detector: &Detector,
) -> Result<Vec<JsonPiiMatch>, serde_json::Error> {
    let value: JsonValue = serde_json::from_str(input)?;
    let mut matches = Vec::new();

    detect_json_value(&value, detector, String::new(), &mut matches);

    Ok(matches)
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "Recursive function builds path strings"
)]
fn detect_json_value(
    value: &JsonValue,
    detector: &Detector,
    path: String,
    matches: &mut Vec<JsonPiiMatch>,
) {
    match value {
        JsonValue::String(s) => {
            for pii_match in detector.detect(s) {
                matches.push(JsonPiiMatch {
                    path: if path.is_empty() {
                        "(root)".to_string()
                    } else {
                        path.clone()
                    },
                    pii_match,
                });
            }
        }
        JsonValue::Array(arr) => {
            for (i, item) in arr.iter().enumerate() {
                let item_path = if path.is_empty() {
                    format!("[{i}]")
                } else {
                    format!("{path}[{i}]")
                };
                detect_json_value(item, detector, item_path, matches);
            }
        }
        JsonValue::Object(obj) => {
            for (key, v) in obj {
                let key_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                detect_json_value(v, detector, key_path, matches);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{DetectorConfig, ReplacementStrategy, ReplacerConfig};

    #[test]
    fn test_process_json_simple() {
        let detector = Detector::new(&DetectorConfig::default());
        let mut replacer = Replacer::new(ReplacerConfig {
            strategy: ReplacementStrategy::Placeholder,
            ..Default::default()
        });

        let input = r#"{"email": "test@example.com", "name": "John"}"#;
        let (output, replacements) = process_json(input, &detector, &mut replacer).unwrap();

        assert!(output.contains("<EMAIL>"));
        assert!(!output.contains("test@example.com"));
        assert_eq!(replacements.len(), 1);
    }

    #[test]
    fn test_process_json_nested() {
        let detector = Detector::new(&DetectorConfig::default());
        let mut replacer = Replacer::new(ReplacerConfig {
            strategy: ReplacementStrategy::Placeholder,
            ..Default::default()
        });

        let input = r#"{
            "user": {
                "contact": {
                    "email": "alice@company.org"
                }
            },
            "logs": ["Visit from 192.168.1.100"]
        }"#;

        let (output, replacements) = process_json(input, &detector, &mut replacer).unwrap();

        assert!(output.contains("<EMAIL>"));
        assert!(output.contains("<IPV4>"));
        assert_eq!(replacements.len(), 2);
    }

    #[test]
    fn test_process_json_array() {
        let detector = Detector::new(&DetectorConfig::default());
        let mut replacer = Replacer::new(ReplacerConfig {
            strategy: ReplacementStrategy::Placeholder,
            ..Default::default()
        });

        let input = r#"["user1@test.com", "user2@test.com", "not-an-email"]"#;
        let (output, replacements) = process_json(input, &detector, &mut replacer).unwrap();

        // Should have 2 email replacements
        assert_eq!(replacements.len(), 2);
        assert!(output.contains("<EMAIL>"));
    }

    #[test]
    fn test_detect_json_paths() {
        let detector = Detector::new(&DetectorConfig::default());

        let input = r#"{
            "user": {
                "email": "test@example.com"
            },
            "ips": ["192.168.1.1", "10.0.0.1"]
        }"#;

        let matches = detect_json(input, &detector).unwrap();

        assert_eq!(matches.len(), 3);

        // Check paths are correctly generated
        let paths: Vec<&str> = matches.iter().map(|m| m.path.as_str()).collect();
        assert!(paths.contains(&"user.email"));
        assert!(paths.contains(&"ips[0]"));
        assert!(paths.contains(&"ips[1]"));
    }

    #[test]
    fn test_json_preserves_structure() {
        let detector = Detector::new(&DetectorConfig::default());
        let mut replacer = Replacer::new(ReplacerConfig {
            strategy: ReplacementStrategy::Placeholder,
            ..Default::default()
        });

        let input = r#"{"count": 42, "active": true, "email": "x@y.com", "data": null}"#;
        let (output, _) = process_json(input, &detector, &mut replacer).unwrap();

        // Parse the output to verify it's valid JSON with preserved types
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();

        assert_eq!(parsed["count"], 42);
        assert_eq!(parsed["active"], true);
        assert_eq!(parsed["data"], serde_json::Value::Null);
        assert_eq!(parsed["email"], "<EMAIL>");
    }
}
