//! Format-aware PII processing.
//!
//! JSON document handlers preserve structure and scan decoded string values.
//! The CLI processes JSONL records through these same handlers.

use serde_json::Value as JsonValue;

use super::detector::{DetectionError, Detector, PiiMatch};

/// Structured processing errors never retain parser payloads or private values.
#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    #[error("invalid JSON input at line {line}, column {column}")]
    InvalidJson { line: usize, column: usize },
    #[error("JSON serialization failed")]
    Serialization,
    #[error(transparent)]
    Detection(#[from] DetectionError),
}
use super::replacer::{Replacement, Replacer};
use super::selector::{CoverageReport, PathSelector};

fn parse_json(input: &str) -> Result<JsonValue, FormatError> {
    // A single optional BOM is allowed before the document (after whitespace),
    // never inside records or string values. Keep ordinary JSON whitespace.
    let trimmed = input.trim_start();
    let input = trimmed.strip_prefix('\u{feff}').unwrap_or(input);
    serde_json::from_str(input).map_err(|error| FormatError::InvalidJson {
        line: error.line(),
        column: error.column(),
    })
}

/// Process JSON, walking all string values and applying PII detection/replacement.
///
/// Returns the processed JSON and all replacements made.
#[cfg_attr(not(test), allow(dead_code))]
pub fn process_json(
    input: &str,
    detector: &Detector,
    replacer: &mut Replacer,
) -> Result<(String, Vec<Replacement>), FormatError> {
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
) -> Result<(String, Vec<Replacement>, CoverageReport), FormatError> {
    let mut value = parse_json(input)?;
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
    )?;

    let output = serde_json::to_string_pretty(&value).map_err(|_| FormatError::Serialization)?;
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
) -> Result<(), FormatError> {
    match value {
        JsonValue::String(s) => {
            let display = if path.is_empty() { "(root)" } else { path };
            if selector.should_scan(path) {
                report.scanned.insert(display.to_string());
                let matches = detector.detect(s)?;
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
                )?;
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
                )?;
            }
        }
        // Non-string scalars are outside this text-only handler's scan scope.
        _ => {}
    }
    Ok(())
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
pub fn detect_json(input: &str, detector: &Detector) -> Result<Vec<JsonPiiMatch>, FormatError> {
    detect_json_with_selector(input, detector, &PathSelector::default()).map(|(m, _)| m)
}

/// Like [`detect_json`] but only scans selected paths and returns coverage.
pub fn detect_json_with_selector(
    input: &str,
    detector: &Detector,
    selector: &PathSelector,
) -> Result<(Vec<JsonPiiMatch>, CoverageReport), FormatError> {
    let value = parse_json(input)?;
    let mut matches = Vec::new();
    let mut report = CoverageReport::default();

    detect_json_value(
        &value,
        detector,
        String::new(),
        selector,
        &mut matches,
        &mut report,
    )?;

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
) -> Result<(), FormatError> {
    match value {
        JsonValue::String(s) => {
            let display = if path.is_empty() {
                "(root)".to_string()
            } else {
                path.clone()
            };
            if selector.should_scan(&path) {
                report.scanned.insert(display.clone());
                for pii_match in detector.detect(s)? {
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
                detect_json_value(item, detector, item_path, selector, matches, report)?;
            }
        }
        JsonValue::Object(obj) => {
            for (key, v) in obj {
                let key_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                detect_json_value(v, detector, key_path, selector, matches, report)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{ReplacementStrategy, ReplacerConfig};

    #[test]
    fn single_json_allows_one_initial_bom_but_not_embedded_boms() {
        let detector = Detector::with_defaults();
        let mut replacer = Replacer::with_defaults();
        let input = "  \n\u{feff} {\"safe\":\"ordinary text\"}";
        assert!(detect_json(input, &detector).unwrap().is_empty());
        assert!(process_json(input, &detector, &mut replacer).is_ok());
        assert!(detect_json("[\u{feff}{}]", &detector).is_err());
        assert!(detect_json("\u{feff}\u{feff}{}", &detector).is_err());
    }

    #[test]
    fn malformed_json_diagnostics_never_echo_private_payload() {
        let detector = Detector::with_defaults();
        let mut replacer = Replacer::with_defaults();
        for input in [
            "{\"secret\": confidentialCredential}",
            "[1, private@example.invalid]",
            "{\"confidentialName\": NaN}",
        ] {
            let errors = [
                detect_json(input, &detector).unwrap_err(),
                process_json(input, &detector, &mut replacer).unwrap_err(),
            ];
            for error in errors {
                assert!(matches!(
                    error,
                    FormatError::InvalidJson {
                        line: 1,
                        column: 1..
                    }
                ));
                for rendered in [error.to_string(), format!("{error:?}")] {
                    assert!(!rendered.contains("confidential"));
                    assert!(!rendered.contains("private@example.invalid"));
                }
                assert!(std::error::Error::source(&error).is_none());
            }
        }
    }

    #[test]
    fn test_process_json_simple() {
        let detector = Detector::with_defaults();
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
        let detector = Detector::with_defaults();
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
        let detector = Detector::with_defaults();
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
        let detector = Detector::with_defaults();

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
        let detector = Detector::with_defaults();
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
        let detector = Detector::with_defaults();
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
        let detector = Detector::with_defaults();
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
