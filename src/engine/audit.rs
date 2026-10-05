//! Audit failure policy and value-free machine-readable summaries.
//!
//! `nym detect` is an inspection command: it exits 0 when findings are present.
//! That is deliberate, but it is not a release gate. This module adds an
//! opt-in fail-on policy so a training/CI pipeline can fail when sensitive
//! classes are found, plus a value-free aggregate summary suitable for a
//! public run manifest (no matched values, no source paths, no reversible
//! mappings).

use std::collections::BTreeMap;

use anyhow::Result;
use serde::Serialize;

use super::patterns::{PiiCategory, PiiPattern};

/// Convert a pattern name or category label (as typed by the user) into a
/// predicate used to decide whether a finding is a "blocker".
#[derive(Debug, Clone)]
pub struct FailOnPolicy {
    /// Pattern names to block on.
    pattern_names: Vec<String>,
    /// Category labels to block on (lowercased).
    categories: Vec<String>,
}

impl FailOnPolicy {
    /// Build a policy from CLI `--fail-on` values. Each value may be a pattern
    /// name (e.g. `email`, `ssn`) or a category label (e.g. `financial`,
    /// `authentication`). Unknown values are accepted (a finding simply won't
    /// match) rather than failing at parse time, so a pipeline can list future
    /// classes without breaking.
    pub fn new(values: &[String]) -> Self {
        let mut pattern_names = Vec::new();
        let mut categories = Vec::new();
        for v in values {
            let lower = v.trim().to_lowercase();
            if lower.is_empty() {
                continue;
            }
            // Category labels are matched against the lowercased variant name;
            // everything else is treated as a pattern name.
            if is_category_label(&lower) {
                categories.push(lower);
            } else {
                pattern_names.push(v.clone());
            }
        }
        Self {
            pattern_names,
            categories,
        }
    }

    /// Whether this policy blocks anything at all.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_empty(&self) -> bool {
        self.pattern_names.is_empty() && self.categories.is_empty()
    }

    /// Whether a finding with the given pattern should block the gate.
    pub fn blocks(&self, pattern: &PiiPattern) -> bool {
        if self.pattern_names.iter().any(|n| n == pattern.name) {
            return true;
        }
        if self
            .categories
            .iter()
            .any(|c| c == &category_label(pattern.category))
        {
            return true;
        }
        false
    }
}

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
            if let Some(p) = super::get_pattern(pattern_name)
                && policy.blocks(p)
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
        let policy = FailOnPolicy::new(&["email".to_string(), "ssn".to_string()]);
        let email = super::super::get_pattern("email").unwrap();
        let ssn = super::super::get_pattern("ssn").unwrap();
        let ipv4 = super::super::get_pattern("ipv4").unwrap();
        assert!(policy.blocks(email));
        assert!(policy.blocks(ssn));
        assert!(!policy.blocks(ipv4));
    }

    #[test]
    fn test_policy_blocks_category() {
        let policy = FailOnPolicy::new(&["financial".to_string()]);
        let card = super::super::get_pattern("credit_card").unwrap();
        assert!(policy.blocks(card));
    }

    #[test]
    fn test_policy_empty() {
        let policy = FailOnPolicy::new(&[]);
        assert!(policy.is_empty());
    }

    #[test]
    fn test_summary_is_value_free() {
        let policy = FailOnPolicy::new(&["email".to_string()]);
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
    fn test_summary_not_blocked_when_no_match() {
        let policy = FailOnPolicy::new(&["ssn".to_string()]);
        let summary = AuditSummary::from_findings(vec![("ipv4", PiiCategory::Network)], &policy);
        assert!(!summary.blocked());
    }
}
