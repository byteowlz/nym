//! PII detection engine.
//!
//! This module provides the core detection logic using compiled regex patterns
//! and optional NER (Named Entity Recognition) for detecting names and addresses.

use std::sync::LazyLock;

use regex::{Regex, RegexSet};
use serde::{Deserialize, Serialize};

use super::patterns::{BUILTIN_PATTERNS, Confidence, PiiCategory, PiiPattern};

/// Static regex for TLD detection in social handle validation.
#[expect(clippy::unwrap_used, reason = "Static regex literal is infallible")]
static TLD_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\.[a-z]{2,}(\s|$|[^a-z])").unwrap());

#[cfg(feature = "ner")]
use super::ner::{NerDetector, NerModelConfig};
#[cfg(feature = "ner")]
use super::ner_token::TokenClassDetector;

/// Which NER backend(s) to run.
///
/// nym supports two NER backends in parallel:
/// - [`NerBackend::Gliner`] — zero-shot span model via `gline-rs`.
/// - [`NerBackend::TokenClass`] — BERT/DeBERTa token classification via ONNX
///   Runtime (OpenMed, Rampart, or any HF token-classification PII model).
/// - [`NerBackend::Both`] — run both and merge results (the default; best recall).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NerBackend {
    /// GLiNER zero-shot span model.
    Gliner,
    /// Token-classification model (BERT/DeBERTa). Config value `"tokens"`;
    /// `"openmed"` is accepted as a back-compat alias.
    #[serde(rename = "tokens", alias = "openmed")]
    TokenClass,
    /// Run both backends and merge their matches (default).
    #[default]
    Both,
}

/// A detected PII match.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiiMatch {
    /// The pattern name that matched
    pub pattern_name: String,
    /// The matched text
    pub matched_text: String,
    /// Start byte offset in the input
    pub start: usize,
    /// End byte offset in the input
    pub end: usize,
    /// Confidence level
    pub confidence: Confidence,
    /// Category
    pub category: PiiCategory,
}

impl PiiMatch {
    /// Get the length of the match in bytes.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Check if the match is empty (required by clippy when `len` is defined).
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Public API - used by consumers")
    )]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Configuration for the PII detector.
#[derive(Debug, Clone)]
pub struct DetectorConfig {
    /// Pattern names to include (empty = all)
    pub include_patterns: Vec<String>,
    /// Pattern names to exclude
    pub exclude_patterns: Vec<String>,
    /// Minimum confidence level
    pub min_confidence: Confidence,
    /// Enable NER-based detection (requires 'ner' feature)
    pub ner_enabled: bool,
    /// NER model repository (e.g., "onnx-community/gliner_multi-v2.1")
    pub ner_model: Option<String>,
    /// NER confidence threshold (0.0-1.0)
    pub ner_threshold: Option<f32>,
    /// NER entity labels to detect
    pub ner_labels: Option<Vec<String>>,
    /// NER cache directory for model files
    pub ner_cache_dir: Option<std::path::PathBuf>,
    /// Which NER backend(s) to run.
    pub ner_backend: NerBackend,
    /// Path to the token-classification model: a local dir (model.onnx + tokenizer.json
    /// + config.json) or a HuggingFace repo id. Used for the TokenClass/Both backends.
    pub ner_token_model: Option<std::path::PathBuf>,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            include_patterns: Vec::new(),
            exclude_patterns: Vec::new(),
            min_confidence: Confidence::High,
            ner_enabled: false,
            ner_model: None,
            ner_threshold: None,
            ner_labels: None,
            ner_cache_dir: None,
            ner_backend: NerBackend::default(),
            ner_token_model: None,
        }
    }
}

impl DetectorConfig {
    /// Create a new config with all patterns enabled.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Public API - used by consumers")
    )]
    pub fn all_patterns() -> Self {
        Self::default()
    }

    /// Create a config with only high-confidence patterns.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Public API - used by consumers")
    )]
    pub fn high_confidence_only() -> Self {
        Self {
            min_confidence: Confidence::High,
            ..Default::default()
        }
    }

    /// Include specific patterns by name.
    pub fn with_patterns(mut self, patterns: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.include_patterns = patterns.into_iter().map(Into::into).collect();
        self
    }

    /// Exclude specific patterns by name.
    pub fn without_patterns(
        mut self,
        patterns: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.exclude_patterns = patterns.into_iter().map(Into::into).collect();
        self
    }

    /// Set minimum confidence level.
    pub fn with_min_confidence(mut self, confidence: Confidence) -> Self {
        self.min_confidence = confidence;
        self
    }

    /// Enable NER-based detection.
    pub fn with_ner(mut self, enabled: bool) -> Self {
        self.ner_enabled = enabled;
        self
    }

    /// Set NER model repository.
    pub fn with_ner_model(mut self, model: impl Into<String>) -> Self {
        self.ner_model = Some(model.into());
        self
    }

    /// Set NER confidence threshold.
    pub fn with_ner_threshold(mut self, threshold: f32) -> Self {
        self.ner_threshold = Some(threshold);
        self
    }

    /// Set NER entity labels.
    pub fn with_ner_labels(mut self, labels: Vec<String>) -> Self {
        self.ner_labels = Some(labels);
        self
    }

    /// Set NER cache directory.
    pub fn with_ner_cache_dir(mut self, cache_dir: impl Into<std::path::PathBuf>) -> Self {
        self.ner_cache_dir = Some(cache_dir.into());
        self
    }

    /// Select which NER backend(s) to run.
    pub fn with_ner_backend(mut self, backend: NerBackend) -> Self {
        self.ner_backend = backend;
        self
    }

    /// Set the token-classification model (local dir or HuggingFace repo id).
    pub fn with_ner_token_model(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.ner_token_model = Some(dir.into());
        self
    }
}

/// The PII detector engine.
///
/// Uses a `RegexSet` for efficient parallel matching of all patterns,
/// then extracts individual matches. Optionally includes NER-based
/// detection for names and addresses.
pub struct Detector {
    /// The compiled regex set for fast matching
    regex_set: RegexSet,
    /// Ordered list of patterns corresponding to regex set indices
    patterns: Vec<&'static PiiPattern>,
    /// Optional GLiNER detector for name/address detection
    #[cfg(feature = "ner")]
    ner_detector: Option<NerDetector>,
    /// Optional token-classification detector
    #[cfg(feature = "ner")]
    token_detector: Option<TokenClassDetector>,
}

impl Detector {
    /// Create a new detector with the given configuration.
    pub fn new(config: &DetectorConfig) -> Self {
        let patterns: Vec<&'static PiiPattern> = BUILTIN_PATTERNS
            .iter()
            .filter(|p| {
                // Check confidence level (pattern must meet minimum)
                let confidence_ok = p.confidence >= config.min_confidence;

                if !confidence_ok {
                    return false;
                }

                // Check include list (empty = include all)
                let include_ok = config.include_patterns.is_empty()
                    || config.include_patterns.iter().any(|n| n == p.name);

                if !include_ok {
                    return false;
                }

                // Check exclude list

                !config.exclude_patterns.iter().any(|n| n == p.name)
            })
            .collect();

        let regex_patterns: Vec<String> = patterns
            .iter()
            .map(|p| p.regex.as_str().to_string())
            .collect();

        // All built-in patterns are compile-time string literals validated by tests.
        #[expect(clippy::expect_used, reason = "Built-in regex patterns are infallible")]
        let regex_set =
            RegexSet::new(&regex_patterns).expect("built-in patterns must be valid regexes");

        // Initialize NER backend(s) if enabled
        #[cfg(feature = "ner")]
        let (ner_detector, token_detector) = if config.ner_enabled {
            match config.ner_backend {
                NerBackend::Gliner => (Self::init_ner(config), None),
                NerBackend::TokenClass => (None, Self::init_token(config)),
                NerBackend::Both => (Self::init_ner(config), Self::init_token(config)),
            }
        } else {
            (None, None)
        };

        Self {
            regex_set,
            patterns,
            #[cfg(feature = "ner")]
            ner_detector,
            #[cfg(feature = "ner")]
            token_detector,
        }
    }

    /// Initialize NER detector from config.
    #[cfg(feature = "ner")]
    fn init_ner(config: &DetectorConfig) -> Option<NerDetector> {
        use log::{info, warn};

        // Build model config
        let mut model_config = if let Some(ref model) = config.ner_model {
            NerModelConfig::with_model(model)
        } else {
            NerModelConfig::default()
        };

        if let Some(ref cache_dir) = config.ner_cache_dir {
            model_config = model_config.with_cache_dir(cache_dir);
        }

        // Download or get cached model paths
        let paths = match model_config.download() {
            Ok(p) => p,
            Err(e) => {
                warn!(
                    "Failed to download NER model: {}. NER detection disabled.",
                    e
                );
                return None;
            }
        };

        info!("NER model loaded from: {:?}", paths.model);

        // Create detector
        match NerDetector::new(
            &paths.tokenizer,
            &paths.model,
            config.ner_labels.clone(),
            config.ner_threshold,
        ) {
            Ok(detector) => Some(detector),
            Err(e) => {
                warn!(
                    "Failed to initialize NER detector: {}. NER detection disabled.",
                    e
                );
                None
            }
        }
    }

    /// Initialize the token-classification detector from config.
    #[cfg(feature = "ner")]
    fn init_token(config: &DetectorConfig) -> Option<TokenClassDetector> {
        use log::{info, warn};

        // A local directory is loaded directly; anything else is treated as a
        // HuggingFace repo id and downloaded/cached (like the GLiNER backend).
        // When unset, fall back to the default published model repo.
        let result = match config.ner_token_model {
            Some(ref model) if model.is_dir() => {
                TokenClassDetector::from_dir(model, config.ner_threshold)
            }
            Some(ref model) => TokenClassDetector::from_repo(
                &model.to_string_lossy(),
                config.ner_cache_dir.as_deref(),
                config.ner_threshold,
            ),
            None => {
                info!(
                    "No ner.token_model set; using default {}",
                    super::ner_token::DEFAULT_TOKEN_MODEL
                );
                TokenClassDetector::from_repo(
                    super::ner_token::DEFAULT_TOKEN_MODEL,
                    config.ner_cache_dir.as_deref(),
                    config.ner_threshold,
                )
            }
        };

        match result {
            Ok(detector) => {
                info!("Token-classification NER detector ready");
                Some(detector)
            }
            Err(e) => {
                warn!(
                    "Failed to initialize token-classification detector: {}. Disabled.",
                    e
                );
                None
            }
        }
    }

    /// Create a detector with default configuration (all high-confidence patterns).
    pub fn with_defaults() -> Self {
        Self::new(&DetectorConfig::default())
    }

    /// Detect all PII in the given text.
    ///
    /// Returns matches sorted by start position. Combines regex-based
    /// detection with NER-based detection if enabled.
    pub fn detect(&self, text: &str) -> Vec<PiiMatch> {
        let mut matches = Vec::new();

        // Regex-based detection
        let matching_indices: Vec<usize> = self.regex_set.matches(text).into_iter().collect();

        for idx in matching_indices {
            let pattern = self.patterns[idx];

            for m in pattern.regex.find_iter(text) {
                // Filter out false positives for social handles
                // (e.g., @domain in emails should not match as a handle)
                if Self::is_social_handle_pattern(pattern.name)
                    && !Self::is_valid_social_handle(text, m.start(), m.end())
                {
                    continue;
                }

                matches.push(PiiMatch {
                    pattern_name: pattern.name.to_string(),
                    matched_text: m.as_str().to_string(),
                    start: m.start(),
                    end: m.end(),
                    confidence: pattern.confidence,
                    category: pattern.category,
                });
            }
        }

        // NER-based detection
        #[cfg(feature = "ner")]
        if let Some(ref ner) = self.ner_detector {
            match ner.detect(text) {
                Ok(ner_matches) => {
                    // Add NER matches, avoiding duplicates with regex matches
                    for nm in ner_matches {
                        // Check if this span overlaps with an existing match
                        let overlaps = matches.iter().any(|m| {
                            // Check for overlap: ranges overlap if start < other_end && end > other_start
                            nm.start < m.end && nm.end > m.start
                        });

                        if !overlaps {
                            matches.push(nm);
                        }
                    }
                }
                Err(e) => {
                    log::warn!("NER detection failed: {}", e);
                }
            }
        }

        // Token-classification detection (runs alongside GLiNER)
        #[cfg(feature = "ner")]
        if let Some(ref token) = self.token_detector {
            match token.detect(text) {
                Ok(token_matches) => {
                    for nm in token_matches {
                        let overlaps = matches
                            .iter()
                            .any(|m| nm.start < m.end && nm.end > m.start);
                        if !overlaps {
                            matches.push(nm);
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Token-classification NER detection failed: {}", e);
                }
            }
        }

        // Sort by start position, then by length (longer first for overlaps)
        matches.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| b.len().cmp(&a.len())));

        matches
    }

    /// Check if the text contains any PII (fast check without extracting matches).
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Public API - used by consumers")
    )]
    pub fn contains_pii(&self, text: &str) -> bool {
        self.regex_set.is_match(text)
    }

    /// Get the list of active pattern names.
    pub fn active_patterns(&self) -> Vec<&'static str> {
        self.patterns.iter().map(|p| p.name).collect()
    }

    /// Get pattern info by name.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Public API - used by consumers")
    )]
    pub fn get_pattern(&self, name: &str) -> Option<&'static PiiPattern> {
        self.patterns.iter().find(|p| p.name == name).copied()
    }

    /// Check if NER detection is enabled and initialized.
    #[cfg(feature = "ner")]
    pub fn ner_enabled(&self) -> bool {
        self.ner_detector.is_some()
    }

    /// Check if NER detection is enabled (always false without feature).
    #[cfg(not(feature = "ner"))]
    #[expect(
        clippy::unused_self,
        reason = "Matches the ner-feature variant signature"
    )]
    pub fn ner_enabled(&self) -> bool {
        false
    }

    /// Check if a pattern is a social handle pattern.
    fn is_social_handle_pattern(pattern_name: &str) -> bool {
        matches!(
            pattern_name,
            "social_handle" | "twitter_handle" | "instagram_handle"
        )
    }

    /// Validate that a social handle match is not part of an email address.
    ///
    /// Social handles like @username should not match @domain.com in emails.
    fn is_valid_social_handle(text: &str, start: usize, end: usize) -> bool {
        // Check if this @ is preceded by alphanumeric (likely email local part)
        if start > 0 {
            let prev_char = text[..start].chars().last();
            if let Some(c) = prev_char
                && (c.is_alphanumeric() || c == '.' || c == '_' || c == '+' || c == '-')
            {
                // Preceded by email-like characters, likely part of email
                return false;
            }
        }

        // Check if followed by a TLD-like pattern (e.g., .com, .org)
        // This would indicate it's part of an email domain
        let after = &text[end..];
        if after.starts_with('.') && TLD_REGEX.is_match(&after.to_lowercase()) {
            return false;
        }

        true
    }
}

impl Default for Detector {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_email() {
        let detector = Detector::with_defaults();
        let matches = detector.detect("Contact me at john.doe@example.com for more info.");

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern_name, "email");
        assert_eq!(matches[0].matched_text, "john.doe@example.com");
    }

    #[test]
    fn test_detect_multiple() {
        let detector =
            Detector::new(&DetectorConfig::default().with_min_confidence(Confidence::High));
        // Use valid US phone format (exchange must start with 2-9)
        let text = "Email: test@example.com, Phone: (555) 234-5678, SSN: 123-45-6789";
        let matches = detector.detect(text);

        assert!(
            matches.len() >= 3,
            "Expected at least 3 matches, got {}: {:?}",
            matches.len(),
            matches
        );

        let pattern_names: Vec<&str> = matches.iter().map(|m| m.pattern_name.as_str()).collect();
        assert!(pattern_names.contains(&"email"));
        assert!(pattern_names.contains(&"phone_us"));
        assert!(pattern_names.contains(&"ssn"));
    }

    #[test]
    fn test_detect_none() {
        let detector = Detector::with_defaults();
        let matches = detector.detect("This text contains no PII.");

        assert!(matches.is_empty());
    }

    #[test]
    fn test_contains_pii() {
        let detector = Detector::with_defaults();

        assert!(detector.contains_pii("My email is test@example.com"));
        assert!(!detector.contains_pii("No PII here"));
    }

    #[test]
    fn test_pattern_filtering() {
        let config = DetectorConfig::default().with_patterns(["email", "ssn"]);
        let detector = Detector::new(&config);

        let active = detector.active_patterns();
        assert!(active.contains(&"email"));
        assert!(active.contains(&"ssn"));
        assert!(!active.contains(&"phone_us"));
    }

    #[test]
    fn test_pattern_exclusion() {
        let config = DetectorConfig::default().without_patterns(["email"]);
        let detector = Detector::new(&config);

        let active = detector.active_patterns();
        assert!(!active.contains(&"email"));
        assert!(active.contains(&"ssn"));
    }

    #[test]
    fn test_confidence_filtering() {
        let high_only = Detector::new(&DetectorConfig::high_confidence_only());
        let all =
            Detector::new(&DetectorConfig::all_patterns().with_min_confidence(Confidence::Low));

        // High confidence detector should have fewer patterns
        assert!(high_only.active_patterns().len() <= all.active_patterns().len());
    }
}
