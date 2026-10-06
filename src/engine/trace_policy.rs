//! Opt-in agent-trace heuristics, not a claim about model correctness.
//!
//! Only NER candidates cross the filtering boundary. Regex findings never do.
//! Literal lists work independently of the profile; benign-list suppression
//! requires the profile. Unknown contexts are retained, not treated as public.

#![cfg_attr(
    not(any(feature = "ner", test)),
    allow(
        dead_code,
        reason = "NER filtering boundary is unused in regex-only builds"
    )
)]

use std::fmt;
use std::sync::LazyLock;

use regex::{Regex, RegexBuilder};
use serde::{Deserialize, Serialize};

use super::{Confidence, PiiCategory, PiiMatch};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceProfile {
    AgentTrace,
}

/// Literal matching never interprets regex syntax. Word boundaries use Unicode
/// regex word characters (letters, numbers, combining marks and underscore).
/// Substring mode also matches inside paths/identifiers. No Unicode normalization
/// is performed. Case-insensitive mode uses regex's Unicode simple case folding.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TermBoundary {
    #[default]
    Word,
    Substring,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TracePolicyConfig {
    pub profile: Option<TraceProfile>,
    /// Exact ASCII DNS hosts, not suffixes/wildcards. Empty trusts no URLs.
    pub public_hosts: Vec<String>,
    pub sensitive_terms: Vec<String>,
    pub benign_terms: Vec<String>,
    pub case_sensitive: bool,
    pub term_boundary: TermBoundary,
}

impl Default for TracePolicyConfig {
    fn default() -> Self {
        Self {
            profile: None,
            public_hosts: Vec::new(),
            sensitive_terms: Vec::new(),
            benign_terms: Vec::new(),
            case_sensitive: true,
            term_boundary: TermBoundary::Word,
        }
    }
}

// Terms (including invalid ones) must not escape through configuration debug logs.
impl fmt::Debug for TracePolicyConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TracePolicyConfig")
            .field("profile", &self.profile)
            .field("public_host_count", &self.public_hosts.len())
            .field("sensitive_term_count", &self.sensitive_terms.len())
            .field("benign_term_count", &self.benign_terms.len())
            .field("case_sensitive", &self.case_sensitive)
            .field("term_boundary", &self.term_boundary)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("invalid trace policy configuration")]
pub struct TracePolicyError;

/// Aggregate counts only: no values, paths, labels supplied by users or snippets.
/// Counts precede overlap merging and count each backend's candidates separately.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TracePolicyStats {
    pub ner_candidates: usize,
    pub suppressed_public_urls: usize,
    pub suppressed_technical_values: usize,
    pub suppressed_benign_terms: usize,
    /// Distinct literal spans, before overlap coverage expansion.
    pub sensitive_term_matches: usize,
}

pub struct TracePolicy {
    profile: Option<TraceProfile>,
    public_hosts: Vec<String>,
    sensitive: Vec<Regex>,
    benign: Vec<Regex>,
    boundary: TermBoundary,
}

impl TracePolicyConfig {
    /// Validate without retaining user values in errors or error source chains.
    #[allow(
        dead_code,
        reason = "Public configuration validation boundary for parent integration"
    )]
    pub fn validate(&self) -> Result<(), TracePolicyError> {
        TracePolicy::new(self).map(|_| ())
    }
}

impl TracePolicy {
    pub fn new(config: &TracePolicyConfig) -> Result<Self, TracePolicyError> {
        if config.public_hosts.iter().any(|host| !valid_host(host)) {
            return Err(TracePolicyError);
        }
        Ok(Self {
            profile: config.profile,
            public_hosts: config
                .public_hosts
                .iter()
                .map(|host| host.to_ascii_lowercase())
                .collect(),
            sensitive: compile_terms(&config.sensitive_terms, config.case_sensitive)?,
            benign: compile_terms(&config.benign_terms, config.case_sensitive)?,
            boundary: config.term_boundary,
        })
    }

    /// Deterministic high-confidence findings, even when NER is disabled. Terms
    /// are matched against the original text so Unicode byte offsets stay valid.
    /// All overlapping occurrences are found, including self-overlapping terms.
    pub fn sensitive_matches(&self, text: &str) -> Vec<PiiMatch> {
        let mut spans = Vec::new();
        for term in &self.sensitive {
            let mut offset = 0;
            while let Some(found) = term.find_at(text, offset) {
                if self.boundary == TermBoundary::Substring
                    || word_bounded(text, found.start(), found.end())
                {
                    spans.push((found.start(), found.end()));
                }
                let Some(next) = text[found.start()..].chars().next() else {
                    break;
                };
                offset = found.start() + next.len_utf8();
            }
        }
        spans.sort_unstable();
        spans.dedup();
        spans
            .into_iter()
            .map(|(start, end)| sensitive_match(text, start, end))
            .collect()
    }

    /// Accept only candidates from an NER backend. Do not pass regex findings.
    /// Invalid spans are retained rather than guessed at or silently suppressed.
    #[allow(
        dead_code,
        reason = "Public context-free filtering boundary retained for consumers"
    )]
    pub fn filter_ner(
        &self,
        text: &str,
        candidates: Vec<PiiMatch>,
        sensitive: &[PiiMatch],
        stats: &mut TracePolicyStats,
    ) -> Vec<PiiMatch> {
        self.filter_ner_with_private_context(text, candidates, sensitive, false, stats)
    }

    /// Context is a veto only. It must never establish technical/public evidence
    /// for a leaf or participate in inference, literal matching or byte offsets.
    pub(crate) fn has_private_context(context: &str) -> bool {
        PRIVATE_CONTEXT.is_match(context)
    }

    pub(crate) fn filter_ner_with_private_context(
        &self,
        text: &str,
        candidates: Vec<PiiMatch>,
        sensitive: &[PiiMatch],
        private_context: bool,
        stats: &mut TracePolicyStats,
    ) -> Vec<PiiMatch> {
        candidates
            .into_iter()
            .filter(|candidate| {
                stats.ner_candidates += 1;
                if self.profile != Some(TraceProfile::AgentTrace)
                    || private_context
                    || candidate.start >= candidate.end
                    || text.get(candidate.start..candidate.end)
                        != Some(candidate.matched_text.as_str())
                    || sensitive.iter().any(|term| overlaps(candidate, term))
                {
                    return true;
                }
                let context = context(text, candidate);
                // Negation never cancels a private cue: "not a PIN" is still a cue.
                if PRIVATE_CONTEXT.is_match(&context)
                    || PRIVATE_CONTEXT.is_match(&candidate.matched_text)
                    || candidate.category == PiiCategory::Authentication
                {
                    return true;
                }
                if candidate.pattern_name == "url"
                    && url_complete(text, candidate)
                    && self.public_reference(&candidate.matched_text, &context)
                {
                    stats.suppressed_public_urls += 1;
                    return false;
                }
                if technical_value(text, candidate, &context) {
                    stats.suppressed_technical_values += 1;
                    return false;
                }
                // A benign term must cover the entire candidate, never a substring.
                // Credential/financial/network/document findings cannot be allowlisted.
                if benign_eligible(candidate)
                    && self.benign.iter().any(|term| {
                        term.find(&candidate.matched_text)
                            .is_some_and(|m| m.start() == 0 && m.end() == candidate.len())
                            && (self.boundary == TermBoundary::Substring
                                || word_bounded(text, candidate.start, candidate.end))
                    })
                {
                    stats.suppressed_benign_terms += 1;
                    return false;
                }
                true
            })
            .collect()
    }

    fn public_reference(&self, value: &str, context: &str) -> bool {
        let Some(rest) = value
            .strip_prefix("https://")
            .or_else(|| value.strip_prefix("http://"))
        else {
            return false;
        };
        let (host, path) = rest.split_once('/').unwrap_or((rest, ""));
        // No userinfo, ports, query strings, fragments, escapes or Unicode-host
        // ambiguity. A public host alone is not permission to expose credentials.
        if !self
            .public_hosts
            .iter()
            .any(|allowed| host.eq_ignore_ascii_case(allowed))
            || !path
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"/._-".contains(&b))
            || PRIVATE_CONTEXT.is_match(path)
            || path.split('/').any(|part| part.len() >= 16)
        {
            return false;
        }
        PUBLIC_REFERENCE.is_match(context)
    }

    /// Ensure whole sensitive coverage survives an earlier/shorter overlapping
    /// finding. Original regex/NER findings are retained for detection consumers;
    /// a union finding sorts first for the current non-overlapping replacer.
    /// This may redact extra bytes in a connected overlap, never less sensitive text.
    pub(crate) fn add_sensitive_coverage(
        text: &str,
        matches: &mut Vec<PiiMatch>,
        sensitive: Vec<PiiMatch>,
    ) {
        if sensitive.is_empty() {
            return;
        }
        // Sweep connected intervals once; a component touching any sensitive
        // span must be redacted in full regardless of original merge precedence.
        let mut spans: Vec<_> = matches
            .iter()
            .map(|m| (m, false))
            .chain(sensitive.iter().map(|m| (m, true)))
            .filter(|(m, _)| text.get(m.start..m.end) == Some(m.matched_text.as_str()))
            .map(|(m, sensitive)| (m.start, m.end, sensitive))
            .collect();
        spans.sort_unstable();
        let mut coverage: Vec<(usize, usize, bool)> = Vec::new();
        for (start, end, sensitive) in spans {
            if let Some(last) = coverage.last_mut()
                && start < last.1
            {
                last.1 = last.1.max(end);
                last.2 |= sensitive;
            } else {
                coverage.push((start, end, sensitive));
            }
        }
        matches.extend(
            coverage
                .into_iter()
                .filter(|(_, _, sensitive)| *sensitive)
                .map(|(start, end, _)| sensitive_match(text, start, end)),
        );
    }
}

fn compile_terms(terms: &[String], case_sensitive: bool) -> Result<Vec<Regex>, TracePolicyError> {
    terms
        .iter()
        .map(|term| {
            if term.trim().is_empty() || term.chars().any(char::is_control) {
                return Err(TracePolicyError);
            }
            RegexBuilder::new(&regex::escape(term))
                .case_insensitive(!case_sensitive)
                .build()
                .map_err(|_| TracePolicyError)
        })
        .collect()
}

fn valid_host(host: &str) -> bool {
    host.contains('.')
        && host.len() <= 253
        && host.split('.').all(|label| {
            !label.is_empty()
                && label.len() <= 63
                && !label.starts_with('-')
                && !label.ends_with('-')
                && label
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-')
        })
        && !host.bytes().all(|b| b.is_ascii_digit() || b == b'.')
}

fn sensitive_match(text: &str, start: usize, end: usize) -> PiiMatch {
    PiiMatch {
        pattern_name: "sensitive_term".into(),
        matched_text: text[start..end].into(),
        start,
        end,
        confidence: Confidence::High,
        category: PiiCategory::Other,
    }
}

fn overlaps(a: &PiiMatch, b: &PiiMatch) -> bool {
    a.start < b.end && b.start < a.end
}

fn word_bounded(text: &str, start: usize, end: usize) -> bool {
    !text[..start].chars().next_back().is_some_and(word_char)
        && !text[end..].chars().next().is_some_and(word_char)
}

fn word_char(c: char) -> bool {
    WORD_CHAR.is_match(c.encode_utf8(&mut [0; 4]))
}

/// Bounded same-line surroundings, excluding the candidate itself. Deliberately
/// conservative: any nearby private cue wins, even across a technical clause.
fn context(text: &str, candidate: &PiiMatch) -> String {
    let before: String = text[..candidate.start]
        .chars()
        .rev()
        .take_while(|c| *c != '\n' && *c != '\r')
        .take(96)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    let after: String = text[candidate.end..]
        .chars()
        .take_while(|c| *c != '\n' && *c != '\r')
        .take(96)
        .collect();
    format!("{before} {after}")
}

// A truncated URL prediction must not certify the remaining URL as public.
fn url_complete(text: &str, candidate: &PiiMatch) -> bool {
    let terminator =
        |c: char| c.is_whitespace() || matches!(c, '\"' | '\'' | ')' | ']' | '}' | ',' | ';');
    text[..candidate.start]
        .chars()
        .next_back()
        .is_none_or(|c| terminator(c) || matches!(c, '(' | '[' | '{' | '=' | ':'))
        && text[candidate.end..].chars().next().is_none_or(terminator)
}

fn benign_eligible(candidate: &PiiMatch) -> bool {
    matches!(
        candidate.pattern_name.as_str(),
        "person" | "first_name" | "last_name" | "organization" | "ner_entity"
    )
}

fn technical_value(text: &str, candidate: &PiiMatch, context: &str) -> bool {
    let value = candidate.matched_text.as_str();
    match candidate.pattern_name.as_str() {
        "date" | "time" | "date_of_birth" => {
            TIMESTAMP.is_match(value) && BUILD_CONTEXT.is_match(context)
        }
        "pin" | "cvv" | "age" | "government_id" | "unique_id" | "ssn" | "mac" => {
            value.len() <= 8
                && !value.is_empty()
                && value.bytes().all(|b| b.is_ascii_hexdigit())
                && word_bounded(text, candidate.start, candidate.end)
                && COUNTER_CONTEXT.is_match(context)
        }
        "license_plate" | "certificate_license_number" => {
            GPU_MODEL.is_match(value)
                && word_bounded(text, candidate.start, candidate.end)
                && GPU_CONTEXT.is_match(context)
        }
        "person" | "first_name" | "last_name" | "ner_entity" => {
            matches!(
                value,
                "bash" | "cargo" | "rustc" | "bun" | "npm" | "null" | "true" | "false"
            ) && TOOL_FIELD.is_match(
                text[..candidate.start]
                    .rsplit(['\n', '\r'])
                    .next()
                    .unwrap_or(""),
            )
        }
        _ => false,
    }
}

#[expect(
    clippy::unwrap_used,
    reason = "Static regex literals validated by unit tests"
)]
fn static_regex(pattern: &str) -> Regex {
    Regex::new(pattern).unwrap()
}

static WORD_CHAR: LazyLock<Regex> = LazyLock::new(|| static_regex(r"^\w$"));
static PRIVATE_CONTEXT: LazyLock<Regex> = LazyLock::new(|| {
    static_regex(
        r"(?i)(?:^|[^a-z0-9])(?:pin|cvv|password|passwd|secret|token|credential|auth(?:orization)?|api[_ -]?key|private|internal|customer|client|employee|patient|user(?:name)?|account|birth(?:day|date)?|date[_ -]?of[_ -]?birth|born|dob|ssn|passport|license(?:[_ -]?(?:plate|number))?|plate|driver|email|contact|person|(?:first[_ -]?|last[_ -]?)?name|author|owner|bank|medical)(?:$|[^a-z0-9])",
    )
});
static PUBLIC_REFERENCE: LazyLock<Regex> = LazyLock::new(|| {
    static_regex(r"(?i)\b(?:docs|documentation|reference|dependency|homepage|repository)\b")
});
static BUILD_CONTEXT: LazyLock<Regex> =
    LazyLock::new(|| static_regex(r"(?i)\b(?:build|compiled|timestamp|logged|log[_ -]?time)\b"));
static COUNTER_CONTEXT: LazyLock<Regex> = LazyLock::new(|| {
    static_regex(
        r"(?i)\b(?:buffer|bytes|port|counter|tokens|timeout|batch[_ -]?size|usage|memory|iterations)\b",
    )
});
static TIMESTAMP: LazyLock<Regex> = LazyLock::new(|| {
    static_regex(
        r"^(?:\d{4}-\d{2}-\d{2}(?:T(?:\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?)?)?|\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)$",
    )
});
static GPU_MODEL: LazyLock<Regex> = LazyLock::new(|| static_regex(r"(?i)^(?:rtx|gtx|a|h)\d{2,4}$"));
static GPU_CONTEXT: LazyLock<Regex> =
    LazyLock::new(|| static_regex(r"(?i)\b(?:gpu|cuda|graphics)\b"));
static TOOL_FIELD: LazyLock<Regex> =
    LazyLock::new(|| static_regex(r#"(?i)\b(?:tool|command|language|type)["']?\s*[:=]\s*["']?$"#));

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(text: &str, value: &str, pattern: &str, category: PiiCategory) -> PiiMatch {
        let start = text.find(value).unwrap();
        PiiMatch {
            pattern_name: pattern.into(),
            matched_text: value.into(),
            start,
            end: start + value.len(),
            confidence: Confidence::High,
            category,
        }
    }

    fn profile() -> TracePolicyConfig {
        TracePolicyConfig {
            profile: Some(TraceProfile::AgentTrace),
            public_hosts: vec!["docs.example".into()],
            ..Default::default()
        }
    }

    fn filtered(
        config: &TracePolicyConfig,
        text: &str,
        found: PiiMatch,
    ) -> (Vec<PiiMatch>, TracePolicyStats) {
        let policy = TracePolicy::new(config).unwrap();
        let sensitive = policy.sensitive_matches(text);
        let mut stats = TracePolicyStats {
            sensitive_term_matches: sensitive.len(),
            ..Default::default()
        };
        (
            policy.filter_ner(text, vec![found], &sensitive, &mut stats),
            stats,
        )
    }

    #[test]
    fn default_and_profileless_lists_do_not_apply_semantic_suppression() {
        let mut config = profile();
        config.profile = None;
        config.benign_terms = vec!["bash".into(), "4096".into()];
        for (text, value, name, category) in [
            (
                "buffer is 4096 bytes",
                "4096",
                "pin",
                PiiCategory::Financial,
            ),
            (
                "build timestamp: 2026-10-06T09:00:00Z",
                "2026-10-06T09:00:00Z",
                "date_of_birth",
                PiiCategory::Identity,
            ),
            ("tool: bash", "bash", "ner_entity", PiiCategory::Other),
            (
                "docs https://docs.example/crate",
                "https://docs.example/crate",
                "url",
                PiiCategory::Network,
            ),
        ] {
            let found = candidate(text, value, name, category);
            for policy in [&config, &TracePolicyConfig::default()] {
                assert_eq!(
                    filtered(policy, text, found.clone()),
                    (
                        vec![found.clone()],
                        TracePolicyStats {
                            ner_candidates: 1,
                            ..Default::default()
                        }
                    )
                );
            }
        }
    }

    #[test]
    fn contrastive_technical_values_need_evidence_and_private_cues_always_win() {
        for (text, value, name, category, suppress) in [
            (
                "buffer size is 4096 bytes",
                "4096",
                "pin",
                PiiCategory::Financial,
                true,
            ),
            (
                "The PIN is 4096",
                "4096",
                "pin",
                PiiCategory::Financial,
                false,
            ),
            (
                "buffer=4096; bank PIN=4096",
                "4096",
                "pin",
                PiiCategory::Financial,
                false,
            ),
            (
                "not a PIN: 4096; authorization for bank account",
                "4096",
                "pin",
                PiiCategory::Financial,
                false,
            ),
            ("4096", "4096", "pin", PiiCategory::Financial, false),
            (
                "buffer bytes=7a",
                "7a",
                "government_id",
                PiiCategory::Identity,
                true,
            ),
            (
                "employee_id=7a; buffer bytes",
                "7a",
                "government_id",
                PiiCategory::Identity,
                false,
            ),
            (
                "buffer bytes=7a",
                "7a",
                "password",
                PiiCategory::Authentication,
                false,
            ),
            (
                "build timestamp: 2026-10-06T09:00:00Z",
                "2026-10-06T09:00:00Z",
                "date_of_birth",
                PiiCategory::Identity,
                true,
            ),
            (
                "birth date: 2026-10-06",
                "2026-10-06",
                "date_of_birth",
                PiiCategory::Identity,
                false,
            ),
            (
                "date_of_birth=2026-10-06; build timestamp",
                "2026-10-06",
                "date_of_birth",
                PiiCategory::Identity,
                false,
            ),
            (
                "compiled: 2026-10-06",
                "2026-10-06",
                "date",
                PiiCategory::Other,
                true,
            ),
            (
                "born 2026-10-06, compiled record",
                "2026-10-06",
                "date",
                PiiCategory::Other,
                false,
            ),
            (
                "2026-10-06",
                "2026-10-06",
                "date_of_birth",
                PiiCategory::Identity,
                false,
            ),
            (
                "log time 09:00:00Z",
                "09:00:00Z",
                "time",
                PiiCategory::Other,
                true,
            ),
            (
                "GPU: rtx6000",
                "rtx6000",
                "license_plate",
                PiiCategory::Other,
                true,
            ),
            (
                "license plate rtx6000",
                "rtx6000",
                "license_plate",
                PiiCategory::Other,
                false,
            ),
            (
                "GPU rtx6000; license_plate",
                "rtx6000",
                "license_plate",
                PiiCategory::Other,
                false,
            ),
            (
                "rtx6000",
                "rtx6000",
                "license_plate",
                PiiCategory::Other,
                false,
            ),
            (
                "tool: bash",
                "bash",
                "first_name",
                PiiCategory::Identity,
                true,
            ),
            ("tool: bash", "bash", "ner_entity", PiiCategory::Other, true),
            (
                "name: bash",
                "bash",
                "first_name",
                PiiCategory::Identity,
                false,
            ),
            (
                "tool: Jörg",
                "Jörg",
                "first_name",
                PiiCategory::Identity,
                false,
            ),
            (
                "Alice Example wrote code with buffer=4096",
                "Alice Example",
                "person",
                PiiCategory::Identity,
                false,
            ),
            (
                "```\napi_key=7a\n```",
                "7a",
                "api_key",
                PiiCategory::Authentication,
                false,
            ),
            (
                "metadata: {password: 32}",
                "32",
                "password",
                PiiCategory::Authentication,
                false,
            ),
        ] {
            let found = candidate(text, value, name, category);
            assert_eq!(
                filtered(&profile(), text, found.clone()),
                (
                    if suppress { vec![] } else { vec![found] },
                    TracePolicyStats {
                        ner_candidates: 1,
                        suppressed_technical_values: usize::from(suppress),
                        ..Default::default()
                    }
                ),
                "synthetic case: {text}"
            );
        }
        // Nearby contexts do not leak across record lines.
        let text = "buffer=4096\nPIN=4096";
        let first = candidate(text, "4096", "pin", PiiCategory::Financial);
        let mut second = first.clone();
        second.start = text.rfind("4096").unwrap();
        second.end = second.start + 4;
        let policy = TracePolicy::new(&profile()).unwrap();
        let mut stats = TracePolicyStats::default();
        assert_eq!(
            policy.filter_ner(text, vec![first, second.clone()], &[], &mut stats),
            vec![second]
        );
        assert_eq!(
            stats,
            TracePolicyStats {
                ner_candidates: 2,
                suppressed_technical_values: 1,
                ..Default::default()
            }
        );
    }

    #[test]
    fn only_complete_static_public_reference_urls_are_filtered() {
        for (text, value, suppress) in [
            (
                "dependency https://docs.example/crate",
                "https://docs.example/crate",
                true,
            ),
            (
                "documentation https://DOCS.EXAMPLE/crate",
                "https://DOCS.EXAMPLE/crate",
                true,
            ),
            (
                "reference http://docs.example/path",
                "http://docs.example/path",
                true,
            ),
            (
                "https://docs.example/crate",
                "https://docs.example/crate",
                false,
            ),
            (
                "docs https://internal.example/crate",
                "https://internal.example/crate",
                false,
            ),
            (
                "docs https://docs.example.attacker.example/crate",
                "https://docs.example.attacker.example/crate",
                false,
            ),
            (
                "docs https://sub.docs.example/crate",
                "https://sub.docs.example/crate",
                false,
            ),
            (
                "docs https://someone@docs.example/crate",
                "https://someone@docs.example/crate",
                false,
            ),
            (
                "docs https://docs.example/crate?token=synthetic",
                "https://docs.example/crate?token=synthetic",
                false,
            ),
            (
                "docs https://docs.example/crate?x=synthetic",
                "https://docs.example/crate?x=synthetic",
                false,
            ),
            (
                "docs https://docs.example/crate#synthetic",
                "https://docs.example/crate#synthetic",
                false,
            ),
            (
                "docs https://docs.example/private_key",
                "https://docs.example/private_key",
                false,
            ),
            (
                "docs https://docs.example/a123456789abcdef0",
                "https://docs.example/a123456789abcdef0",
                false,
            ),
            (
                "docs https://docs.example/%70rivate",
                "https://docs.example/%70rivate",
                false,
            ),
            (
                "docs https://docs.example:443/crate",
                "https://docs.example:443/crate",
                false,
            ),
            (
                "private reference https://docs.example/crate",
                "https://docs.example/crate",
                false,
            ),
            // Truncated NER predictions cannot authorize the whole URL.
            (
                "docs https://docs.example/crate?x=synthetic",
                "https://docs.example/crate",
                false,
            ),
            (
                "docs https://docs.example/crate/hidden",
                "https://docs.example/crate",
                false,
            ),
        ] {
            let found = candidate(text, value, "url", PiiCategory::Network);
            assert_eq!(
                filtered(&profile(), text, found.clone()),
                (
                    if suppress { vec![] } else { vec![found] },
                    TracePolicyStats {
                        ner_candidates: 1,
                        suppressed_public_urls: usize::from(suppress),
                        ..Default::default()
                    }
                ),
                "synthetic case: {text}"
            );
        }
    }

    #[test]
    fn benign_lists_only_cover_whole_eligible_candidates_and_sensitive_wins() {
        let mut config = profile();
        config.benign_terms = vec!["bash".into(), "4096".into(), "a.b".into()];
        for (text, value, pattern, category, suppress) in [
            ("bash", "bash", "ner_entity", PiiCategory::Other, true),
            ("bash", "bash", "first_name", PiiCategory::Identity, true),
            (
                "contact: bash",
                "bash",
                "person",
                PiiCategory::Identity,
                false,
            ),
            (
                "name: bash",
                "bash",
                "first_name",
                PiiCategory::Identity,
                false,
            ),
            (
                "bash.example",
                "bash.example",
                "organization",
                PiiCategory::Other,
                false,
            ),
            ("4096", "4096", "pin", PiiCategory::Financial, false),
            (
                "4096",
                "4096",
                "password",
                PiiCategory::Authentication,
                false,
            ),
            (
                "4096",
                "4096",
                "government_id",
                PiiCategory::Identity,
                false,
            ),
            ("a.b", "a.b", "ner_entity", PiiCategory::Other, true),
            ("axb", "axb", "ner_entity", PiiCategory::Other, false),
        ] {
            let found = candidate(text, value, pattern, category);
            assert_eq!(
                filtered(&config, text, found.clone()),
                (
                    if suppress { vec![] } else { vec![found] },
                    TracePolicyStats {
                        ner_candidates: 1,
                        suppressed_benign_terms: usize::from(suppress),
                        ..Default::default()
                    }
                )
            );
        }
        config.sensitive_terms = vec!["bash".into()];
        let found = candidate("tool: bash", "bash", "ner_entity", PiiCategory::Other);
        assert_eq!(
            filtered(&config, "tool: bash", found.clone()),
            (
                vec![found],
                TracePolicyStats {
                    ner_candidates: 1,
                    sensitive_term_matches: 1,
                    ..Default::default()
                }
            )
        );
        config.sensitive_terms = vec!["docs.example".into()];
        let text = "docs https://docs.example/crate";
        let found = candidate(
            text,
            "https://docs.example/crate",
            "url",
            PiiCategory::Network,
        );
        assert_eq!(
            filtered(&config, text, found.clone()),
            (
                vec![found],
                TracePolicyStats {
                    ner_candidates: 1,
                    sensitive_term_matches: 1,
                    ..Default::default()
                }
            )
        );
    }

    #[test]
    fn unicode_literal_case_boundaries_and_no_normalization_have_original_offsets() {
        let text =
            "é 東京 /東京/ x東京 東京x 東京_ 東京\u{301} a.b axb ÉLAN élan élanx Straße STRASSE K";
        let mut config = TracePolicyConfig {
            sensitive_terms: vec![
                "東京".into(),
                "a.b".into(),
                "élan".into(),
                "Straße".into(),
                "k".into(),
            ],
            ..Default::default()
        };
        let expected = |values: &[&str]| {
            values
                .iter()
                .map(|value| {
                    let start = text.find(value).unwrap();
                    sensitive_match(text, start, start + value.len())
                })
                .collect::<Vec<_>>()
        };
        let mut exact = expected(&["東京", "a.b", "élan", "Straße"]);
        let second = text.find("/東京/").unwrap() + 1;
        exact.insert(1, sensitive_match(text, second, second + "東京".len()));
        assert_eq!(
            TracePolicy::new(&config).unwrap().sensitive_matches(text),
            exact
        );
        config.case_sensitive = false;
        let mut folded = exact;
        let start = text.find("ÉLAN").unwrap();
        folded.push(sensitive_match(text, start, start + "ÉLAN".len()));
        let start = text.find('K').unwrap();
        folded.push(sensitive_match(text, start, start + 'K'.len_utf8()));
        folded.sort_by_key(|m| m.start);
        assert_eq!(
            TracePolicy::new(&config).unwrap().sensitive_matches(text),
            folded
        );
        config.term_boundary = TermBoundary::Substring;
        let matches = TracePolicy::new(&config).unwrap().sensitive_matches(text);
        assert_eq!(
            matches.iter().filter(|m| m.matched_text == "東京").count(),
            6
        );
        for m in matches {
            assert_eq!(text.get(m.start..m.end), Some(m.matched_text.as_str()));
        }
        config.sensitive_terms = vec!["é".into()];
        assert!(
            TracePolicy::new(&config)
                .unwrap()
                .sensitive_matches("e\u{301}")
                .is_empty()
        );
    }

    #[test]
    fn self_overlap_and_cross_term_overlap_cover_the_whole_sensitive_union() {
        let config = TracePolicyConfig {
            sensitive_terms: vec!["aba".into(), "bab".into(), "aba".into()],
            term_boundary: TermBoundary::Substring,
            ..Default::default()
        };
        let policy = TracePolicy::new(&config).unwrap();
        let text = "ababa";
        let terms = policy.sensitive_matches(text);
        assert_eq!(
            terms,
            vec![
                sensitive_match(text, 0, 3),
                sensitive_match(text, 1, 4),
                sensitive_match(text, 2, 5)
            ]
        );
        let mut matches = Vec::new();
        TracePolicy::add_sensitive_coverage(text, &mut matches, terms);
        assert_eq!(matches, vec![sensitive_match(text, 0, 5)]);
    }

    #[test]
    fn connected_overlap_coverage_is_order_independent_and_does_not_join_adjacent_spans() {
        let text = "abcdefghijklmn";
        let first = candidate(text, "abcde", "person", PiiCategory::Identity);
        let second = candidate(text, "defgh", "ner_entity", PiiCategory::Other);
        let adjacent = candidate(text, "lmn", "ner_entity", PiiCategory::Other);
        let term = sensitive_match(text, 7, 11);
        for mut matches in [
            vec![first.clone(), second.clone(), adjacent.clone()],
            vec![adjacent.clone(), second.clone(), first.clone()],
        ] {
            let original = matches.clone();
            TracePolicy::add_sensitive_coverage(text, &mut matches, vec![term.clone()]);
            let mut expected = original;
            expected.push(sensitive_match(text, 0, 11));
            assert_eq!(matches, expected);
        }
    }

    #[test]
    fn fragmentary_short_ids_and_private_license_or_name_fields_are_retained() {
        for (text, value, pattern, category) in [
            ("buffer: ab4096cd", "4096", "pin", PiiCategory::Financial),
            (
                "GPU: rtx6000_private",
                "rtx6000",
                "license_plate",
                PiiCategory::Other,
            ),
            (
                "GPU license: rtx6000",
                "rtx6000",
                "license_plate",
                PiiCategory::Other,
            ),
            (
                "firstName: bash; tool: bash",
                "bash",
                "first_name",
                PiiCategory::Identity,
            ),
        ] {
            let found = candidate(text, value, pattern, category);
            assert_eq!(
                filtered(&profile(), text, found.clone()),
                (
                    vec![found],
                    TracePolicyStats {
                        ner_candidates: 1,
                        ..Default::default()
                    }
                )
            );
        }
    }

    #[test]
    fn external_context_is_only_a_private_veto_and_never_suppression_evidence() {
        let config = profile();
        let policy = TracePolicy::new(&config).unwrap();
        let text = "build 1990-04-06";
        let found = candidate(text, "1990-04-06", "date_of_birth", PiiCategory::Identity);
        for path in [
            "$.birth",
            "$.date_of_birth",
            "$.dob",
            "$.client",
            "$.name",
            "$.firstName",
            "$.credentials.token",
        ] {
            assert!(TracePolicy::has_private_context(path));
            let mut stats = TracePolicyStats::default();
            assert_eq!(
                policy.filter_ner_with_private_context(
                    text,
                    vec![found.clone()],
                    &[],
                    TracePolicy::has_private_context(path),
                    &mut stats
                ),
                vec![found.clone()]
            );
            assert_eq!(
                stats,
                TracePolicyStats {
                    ner_candidates: 1,
                    ..Default::default()
                }
            );
        }
        for hint in [
            "$.build",
            "$.buffer",
            "$.tool",
            "$.docs",
            "unknown public reference timestamp",
        ] {
            assert!(!TracePolicy::has_private_context(hint));
            let leaf = "1990-04-06";
            let found = candidate(leaf, leaf, "date_of_birth", PiiCategory::Identity);
            let mut stats = TracePolicyStats::default();
            assert_eq!(
                policy.filter_ner_with_private_context(
                    leaf,
                    vec![found.clone()],
                    &[],
                    TracePolicy::has_private_context(hint),
                    &mut stats
                ),
                vec![found]
            );
            assert_eq!(
                stats,
                TracePolicyStats {
                    ner_candidates: 1,
                    ..Default::default()
                }
            );
        }
    }

    #[test]
    fn aggregate_stats_serialize_only_fixed_keys_and_counts() {
        let text = "buffer: 4096";
        let found = candidate(text, "4096", "pin", PiiCategory::Financial);
        let (_, stats) = filtered(&profile(), text, found);
        assert_eq!(
            serde_json::to_value(stats).unwrap(),
            serde_json::json!({
                "ner_candidates": 1, "suppressed_public_urls": 0,
                "suppressed_technical_values": 1, "suppressed_benign_terms": 0,
                "sensitive_term_matches": 0
            })
        );
    }

    #[test]
    fn malformed_candidates_are_retained_and_validation_errors_are_value_free() {
        let mut found = candidate("bash", "bash", "ner_entity", PiiCategory::Other);
        found.end = 900;
        assert_eq!(filtered(&profile(), "bash", found.clone()).0, vec![found]);
        let mut config = profile();
        config.benign_terms = vec!["東京".into()];
        let found = PiiMatch {
            pattern_name: "ner_entity".into(),
            matched_text: "東京".into(),
            start: 1,
            end: 2,
            confidence: Confidence::High,
            category: PiiCategory::Other,
        };
        assert_eq!(filtered(&config, "東京", found.clone()).0, vec![found]);
        for bad in ["", "  ", "sensitive\ncanary"] {
            let config = TracePolicyConfig {
                sensitive_terms: vec![bad.into()],
                ..Default::default()
            };
            let error = config.validate().unwrap_err();
            assert_eq!(error.to_string(), "invalid trace policy configuration");
            assert_eq!(format!("{error:?}"), "TracePolicyError");
            assert!(std::error::Error::source(&error).is_none());
        }
        for host in [
            "internal",
            "*.private.example",
            "https://private.example",
            "private.example:443",
            "127.0.0.1",
            "private.example.",
            "-private.example",
        ] {
            let config = TracePolicyConfig {
                public_hosts: vec![host.into()],
                ..Default::default()
            };
            assert_eq!(config.validate(), Err(TracePolicyError));
        }
        let config = TracePolicyConfig {
            sensitive_terms: vec!["synthetic-sensitive-canary".into()],
            benign_terms: vec!["synthetic-benign-canary".into()],
            public_hosts: vec!["synthetic-host.example".into()],
            ..Default::default()
        };
        let debug = format!("{config:?}");
        assert!(!debug.contains("synthetic-"));
        assert!(config.validate().is_ok());
    }
}
