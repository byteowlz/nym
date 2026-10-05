//! Audit failure policy and value-free machine-readable summaries.
//!
//! `nym detect` is an inspection command: it exits 0 when findings are present.
//! That is deliberate, but it is not a release gate. This module adds an
//! opt-in fail-on policy so a training/CI pipeline can fail when sensitive
//! classes are found, plus a value-free aggregate summary suitable for a
//! public run manifest (no matched values, no source paths, no reversible
//! mappings).

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use serde::Serialize;

use super::patterns::PiiCategory;

/// Convert a pattern name or category label (as typed by the user) into a
/// predicate used to decide whether a finding is a "blocker".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailOnPolicy {
    /// Pattern names to block on.
    pattern_names: Vec<String>,
    /// Category labels to block on (lowercased).
    categories: Vec<String>,
}

impl FailOnPolicy {
    /// Build a policy from CLI `--fail-on` values. Each value may be a pattern
    /// name (e.g. `email`, `ssn`) or a category label (e.g. `financial`,
    /// `authentication`). Additional loaded names can come from configured
    /// custom/NER labels. Reject every unknown value before producing a manifest.
    pub fn new<'a>(
        values: &[String],
        loaded_pattern_names: impl IntoIterator<Item = &'a str>,
    ) -> Result<Self> {
        let known_names: Vec<&str> = loaded_pattern_names.into_iter().collect();
        let mut pattern_names = Vec::new();
        let mut categories = Vec::new();
        for v in values {
            let lower = v.trim().to_lowercase();
            if is_category_label(&lower) {
                categories.push(lower);
            } else if super::get_pattern(v.trim()).is_some()
                || NER_PATTERN_NAMES.contains(&v.trim())
                || known_names.contains(&v.trim())
            {
                pattern_names.push(v.trim().to_string());
            } else {
                bail!(
                    "unknown audit policy {v:?}: expected a supported category or loaded pattern name"
                );
            }
        }
        Ok(Self {
            pattern_names,
            categories,
        })
    }

    fn blocks_finding(&self, name: &str, category: PiiCategory) -> bool {
        self.pattern_names.iter().any(|n| n == name)
            || self
                .categories
                .iter()
                .any(|c| c == &category_label(category))
    }
}

// Canonical output names supported by the GLiNER and token-classification
// backends, in addition to regex patterns. Configured custom labels are supplied
// by the caller; they are not limited to this built-in schema.
const NER_PATTERN_NAMES: &[&str] = &[
    "person",
    "first_name",
    "last_name",
    "organization",
    "street_address",
    "city",
    "county",
    "state",
    "country",
    "location",
    "phone_ner",
    "date_of_birth",
    "age",
    "gender",
    "tax_id",
    "medical_record_number",
    "health_plan_beneficiary_number",
    "certificate_license_number",
    "government_id",
    "account_number",
    "unique_id",
    "biometric_identifier",
    "fax_number",
    "postcode",
    "coordinate",
    "cvv",
    "pin",
    "bank_routing_number",
    "swift_bic",
    "url",
    "http_cookie",
    "password",
    "license_plate",
    "vehicle_identifier",
    "ner_entity",
];

/// Match a field (pattern or category) for reporting; never leaks the value.
fn category_label(cat: PiiCategory) -> String {
    format!("{:?}", cat).to_lowercase()
}

fn is_category_label(s: &str) -> bool {
    const LABELS: &[&str] = &[
        "contact",
        "identity",
        "financial",
        "network",
        "authentication",
        "social",
        "other",
    ];
    LABELS.contains(&s)
}

/// A value-free aggregate summary suitable for a public run manifest.
#[derive(Debug, Clone, Serialize)]
pub struct AuditSummary {
    /// Total number of findings, regardless of class.
    pub total: usize,
    /// Findings count per pattern name.
    pub by_pattern: BTreeMap<String, usize>,
    /// Findings count per category label.
    pub by_category: BTreeMap<String, usize>,
    /// Names of patterns that triggered the fail-on policy (if any).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub blockers: Vec<String>,
}

impl AuditSummary {
    /// Build a value-free summary from an iterable of (pattern, category);
    /// never includes matched text or source paths.
    pub fn from_findings<'a>(
        findings: impl IntoIterator<Item = (&'a str, PiiCategory)>,
        policy: &FailOnPolicy,
    ) -> Self {
        let mut by_pattern = BTreeMap::new();
        let mut by_category = BTreeMap::new();
        let mut total = 0usize;
        let mut blockers = Vec::new();
        for (pattern_name, category) in findings {
            total += 1;
            *by_pattern.entry(pattern_name.to_string()).or_insert(0) += 1;
            *by_category.entry(category_label(category)).or_insert(0) += 1;
            if policy.blocks_finding(pattern_name, category)
                && !blockers.iter().any(|b| b == pattern_name)
            {
                blockers.push(pattern_name.to_string());
            }
        }
        Self {
            total,
            by_pattern,
            by_category,
            blockers,
        }
    }

    /// Whether any finding triggered the fail-on policy.
    pub fn blocked(&self) -> bool {
        !self.blockers.is_empty()
    }
}

/// Serialize a value-free summary to JSON (compact, for manifests).
pub fn to_summary_json(summary: &AuditSummary) -> Result<String> {
    Ok(serde_json::to_string(summary)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_blocks_pattern_name() {
        let policy = FailOnPolicy::new(&["email".to_string(), "ssn".to_string()], []).unwrap();
        let email = super::super::get_pattern("email").unwrap();
        let ssn = super::super::get_pattern("ssn").unwrap();
        let ipv4 = super::super::get_pattern("ipv4").unwrap();
        assert!(policy.blocks_finding(email.name, email.category));
        assert!(policy.blocks_finding(ssn.name, ssn.category));
        assert!(!policy.blocks_finding(ipv4.name, ipv4.category));
    }

    #[test]
    fn test_policy_blocks_category() {
        let policy = FailOnPolicy::new(&["financial".to_string()], []).unwrap();
        let card = super::super::get_pattern("credit_card").unwrap();
        assert!(policy.blocks_finding(card.name, card.category));
    }

    #[test]
    fn test_policy_empty() {
        let policy = FailOnPolicy::new(&[], []).unwrap();
        assert_eq!(
            policy,
            FailOnPolicy {
                pattern_names: Vec::new(),
                categories: Vec::new()
            }
        );
    }

    #[test]
    fn test_summary_is_value_free() {
        let policy = FailOnPolicy::new(&["email".to_string()], []).unwrap();
        let summary = AuditSummary::from_findings(
            vec![
                ("email", PiiCategory::Contact),
                ("email", PiiCategory::Contact),
                ("ipv4", PiiCategory::Network),
            ],
            &policy,
        );
        assert_eq!(summary.total, 3);
        assert_eq!(summary.by_pattern.get("email"), Some(&2));
        assert_eq!(summary.by_category.get("contact"), Some(&2));
        assert!(summary.blocked());
        assert_eq!(summary.blockers, vec!["email".to_string()]);

        let json = to_summary_json(&summary).unwrap();
        assert!(!json.contains("matched"), "no matched values in summary");
        assert!(!json.contains("@"), "no value content in summary");
    }

    #[test]
    fn loaded_custom_and_ner_findings_block_without_builtin_pattern_metadata() {
        let findings = [
            ("client_tag", PiiCategory::Other),
            ("person", PiiCategory::Identity),
            ("person", PiiCategory::Identity),
        ];
        for values in [vec!["client_tag", "person"], vec!["other", "identity"]] {
            let values: Vec<String> = values.into_iter().map(str::to_string).collect();
            let policy = FailOnPolicy::new(&values, ["client_tag"]).unwrap();
            let summary = AuditSummary::from_findings(findings, &policy);
            assert_eq!(
                serde_json::to_value(summary).unwrap(),
                serde_json::json!({
                    "total": 3, "by_pattern": {"client_tag": 1, "person": 2},
                    "by_category": {"other": 1, "identity": 2}, "blockers": ["client_tag", "person"]
                })
            );
        }
        assert!(FailOnPolicy::new(&["client_tag".into()], []).is_err());
        assert!(FailOnPolicy::new(&["person".into(), "unknown".into()], ["client_tag"]).is_err());
        assert!(FailOnPolicy::new(&["client_tag".into()], ["client_tag"]).is_ok());
    }

    #[test]
    fn test_summary_not_blocked_when_no_match() {
        let policy = FailOnPolicy::new(&["ssn".to_string()], []).unwrap();
        let summary = AuditSummary::from_findings(vec![("ipv4", PiiCategory::Network)], &policy);
        assert!(!summary.blocked());
    }
}
