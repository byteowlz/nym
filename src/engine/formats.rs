//! Format-aware PII processing.
//!
//! This module provides handlers for processing structured formats (JSON, YAML, TOML)
//! while preserving structure and only modifying string values containing PII.

use serde_json::Value as JsonValue;

use super::detector::{Detector, PiiMatch};
use super::replacer::{Replacement, Replacer};
use super::selector::{CoverageReport, PathSelector};

/// Process JSON, walking all string values and applying PII detection/replacement.
///
/// Returns the processed JSON and all replacements made.
#[cfg_attr(not(test), allow(dead_code))]
pub fn process_json(
    input: &str,
    detector: &Detector,
    replacer: &mut Replacer,
) -> Result<(String, Vec<Replacement>), serde_json::Error> {
    process_json_with_selector(input, detector, replacer, &PathSelector::default())
        .map(|(out, reps, _)| (out, reps))
}

/// Like [`process_json`], but only scans string values whose path is selected
/// and returns a coverage report of which paths were scanned vs skipped.
pub fn process_json_with_selector(
    input: &str,
    detector: &Detector,
    replacer: &mut Replacer,
    selector: &PathSelector,
) -> Result<(String, Vec<Replacement>, CoverageReport), serde_json::Error> {
    let mut value: JsonValue = serde_json::from_str(input)?;
    let mut all_replacements = Vec::new();
    let mut report = CoverageReport::default();

    walk_json_mut(
        &mut value,
        "",
        detector,
        replacer,
        &mut all_replacements,
        selector,
        &mut report,
    );

    let output = serde_json::to_string_pretty(&value)?;
    Ok((output, all_replacements, report))
}

/// Walk a JSON value, applying selection at each string leaf.
fn walk_json_mut(
    value: &mut JsonValue,
    path: &str,
    detector: &Detector,
    replacer: &mut Replacer,
    replacements: &mut Vec<Replacement>,
    selector: &PathSelector,
    report: &mut CoverageReport,
) {
    match value {
        JsonValue::String(s) => {
            let display = if path.is_empty() { "(root)" } else { path };
            if selector.should_scan(path) {
                report.scanned.insert(display.to_string());
                let matches = detector.detect(s);
                if !matches.is_empty() {
                    let (replaced, new_replacements) = replacer.replace_all(s, &matches);
                    *s = replaced;
                    replacements.extend(new_replacements);
                }
            } else {
                report.skipped.insert(display.to_string());
            }
        }
        JsonValue::Array(arr) => {
            for (i, item) in arr.iter_mut().enumerate() {
                let item_path = if path.is_empty() {
                    format!("[{i}]")
                } else {
                    format!("{path}[{i}]")
                };
                walk_json_mut(
                    item,
                    &item_path,
                    detector,
                    replacer,
                    replacements,
                    selector,
                    report,
                );
            }
        }
        JsonValue::Object(obj) => {
            for (key, v) in obj.iter_mut() {
                let key_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                walk_json_mut(
                    v,
                    &key_path,
                    detector,
                    replacer,
                    replacements,
                    selector,
                    report,
                );
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
#[cfg_attr(not(test), allow(dead_code))]
pub fn detect_json(
    input: &str,
    detector: &Detector,
) -> Result<Vec<JsonPiiMatch>, serde_json::Error> {
    detect_json_with_selector(input, detector, &PathSelector::default()).map(|(m, _)| m)
}

/// Like [`detect_json`] but only scans selected paths and returns coverage.
pub fn detect_json_with_selector(
    input: &str,
    detector: &Detector,
    selector: &PathSelector,
) -> Result<(Vec<JsonPiiMatch>, CoverageReport), serde_json::Error> {
    let value: JsonValue = serde_json::from_str(input)?;
    let mut matches = Vec::new();
    let mut report = CoverageReport::default();

    detect_json_value(
        &value,
        detector,
        String::new(),
        selector,
        &mut matches,
        &mut report,
    );

    Ok((matches, report))
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "Recursive function builds path strings"
)]
fn detect_json_value(
    value: &JsonValue,
    detector: &Detector,
    path: String,
    selector: &PathSelector,
    matches: &mut Vec<JsonPiiMatch>,
    report: &mut CoverageReport,
) {
    match value {
        JsonValue::String(s) => {
            let display = if path.is_empty() {
                "(root)".to_string()
            } else {
                path.clone()
            };
            if selector.should_scan(&path) {
                report.scanned.insert(display.clone());
                for pii_match in detector.detect(s) {
                    matches.push(JsonPiiMatch {
                        path: display.clone(),
                        pii_match,
                    });
                }
            } else {
                report.skipped.insert(display);
            }
        }
        JsonValue::Array(arr) => {
            for (i, item) in arr.iter().enumerate() {
                let item_path = if path.is_empty() {
                    format!("[{i}]")
                } else {
                    format!("{path}[{i}]")
                };
                detect_json_value(item, detector, item_path, selector, matches, report);
            }
        }
        JsonValue::Object(obj) => {
            for (key, v) in obj {
                let key_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                detect_json_value(v, detector, key_path, selector, matches, report);
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

    #[test]
    fn test_selector_skips_structural_ids() {
        let detector = Detector::new(&DetectorConfig::default());
        let mut replacer = Replacer::new(ReplacerConfig {
            strategy: ReplacementStrategy::Placeholder,
            ..Default::default()
        });
        let selector = PathSelector::new(
            &["session.user.email".to_string()],
            &["session.id".to_string(), "session.parentId".to_string()],
        )
        .unwrap();

        let input = r#"{"session": {"id": "11111111-2222-3333-4444-555555555555", "parentId": "99999999-8888-7777-6666-555555555555", "user": {"email": "alice@example.com"}}}"#;
        let (output, _reps, report) =
            process_json_with_selector(input, &detector, &mut replacer, &selector).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        // email is selected and anonymized
        assert!(
            parsed["session"]["user"]["email"]
                .as_str()
                .unwrap()
                .contains("<EMAIL>")
        );
        // structural ids are untouched
        assert_eq!(
            parsed["session"]["id"].as_str().unwrap(),
            "11111111-2222-3333-4444-555555555555"
        );
        assert_eq!(
            parsed["session"]["parentId"].as_str().unwrap(),
            "99999999-8888-7777-6666-555555555555"
        );
        // coverage: email scanned; id, parentId skipped
        assert!(report.scanned.contains("session.user.email"));
        assert!(report.skipped.contains("session.id"));
        assert!(report.skipped.contains("session.parentId"));
    }

    #[test]
    fn test_selector_array_wildcard() {
        let detector = Detector::new(&DetectorConfig::default());
        let mut replacer = Replacer::new(ReplacerConfig {
            strategy: ReplacementStrategy::Placeholder,
            ..Default::default()
        });
        let selector = PathSelector::new(&["users[*].email".to_string()], &[]).unwrap();

        let input =
            r#"{"users": [{"email": "a@b.com", "id": "u1"}, {"email": "c@d.com", "id": "u2"}]}"#;
        let (output, reps, report) =
            process_json_with_selector(input, &detector, &mut replacer, &selector).unwrap();

        assert_eq!(reps.len(), 2, "both emails redacted");
        assert!(report.scanned.contains("users[0].email"));
        assert!(report.scanned.contains("users[1].email"));
        // id fields skipped
        assert!(report.skipped.contains("users[0].id"));
        assert!(report.skipped.contains("users[1].id"));

        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed["users"][0]["id"].as_str().unwrap(), "u1");
        assert!(
            parsed["users"][0]["email"]
                .as_str()
                .unwrap()
                .contains("<EMAIL>")
        );
    }
}
