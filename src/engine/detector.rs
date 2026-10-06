//! PII detection engine.
//!
//! This module provides the core detection logic using compiled regex patterns
//! and optional NER (Named Entity Recognition) for detecting names and addresses.

use std::sync::LazyLock;

use regex::{Regex, RegexSet};
use serde::{Deserialize, Serialize};

use crate::config::NerProvider;

use super::patterns::{BUILTIN_PATTERNS, Confidence, PiiCategory, PiiPattern, iban_span};
use super::trace_policy::{TracePolicy, TracePolicyConfig, TracePolicyStats};

/// Static regex for TLD detection in social handle validation.
#[expect(clippy::unwrap_used, reason = "Static regex literal is infallible")]
static TLD_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\.[a-z]{2,}(\s|$|[^a-z])").unwrap());

#[cfg(feature = "ner")]
use super::ner::{NerDetector, NerModelConfig, NerModelPaths};
#[cfg(feature = "ner")]
use super::ner_token::TokenClassDetector;

/// Which NER backend(s) to run.
///
/// nym supports two NER backends in parallel:
/// - [`NerBackend::Gliner`] — zero-shot span model via `gline-rs`.
/// - [`NerBackend::TokenClass`] — BERT/DeBERTa token classification via ONNX
///   Runtime (OpenMed, Rampart, or any HF token-classification PII model).
/// - [`NerBackend::Both`] — run both and merge results (best recall, but also
///   loads the ~1.1 GB GLiNER model).
///
/// The default is [`NerBackend::TokenClass`]: a single, fast token model. Set
/// `[ner] backend = "both"` to additionally run GLiNER and merge results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NerBackend {
    /// GLiNER zero-shot span model.
    Gliner,
    /// Token-classification model (BERT/DeBERTa). Config value `"tokens"`;
    /// `"openmed"` is accepted as a back-compat alias. The default backend.
    #[serde(rename = "tokens", alias = "openmed")]
    #[default]
    TokenClass,
    /// Run both backends and merge their matches.
    Both,
}

/// Privacy-safe detection failures. Backend errors are deliberately not retained:
/// their messages and source chains may contain input text or model credentials.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DetectionError {
    #[cfg(not(feature = "ner"))]
    #[error("NER is requested but unavailable in this build")]
    NerUnavailable,
    #[error("invalid NER configuration")]
    Configuration,
    #[error("invalid trace policy configuration")]
    TracePolicyConfiguration,
    #[cfg(feature = "ner")]
    #[error("{backend} NER initialization failed")]
    Initialization { backend: &'static str },
    #[cfg(feature = "ner")]
    #[error("{backend} NER inference failed; scan incomplete")]
    Inference { backend: &'static str },
}

/// The external inference boundary. Production and injected runtimes follow
/// the same merging/error path; no test-only bypass of detection is needed.
#[cfg(feature = "ner")]
pub(crate) trait NerRuntime {
    fn detect(&self, text: &str)
    -> Result<Vec<PiiMatch>, Box<dyn std::error::Error + Send + Sync>>;

    #[cfg(any(feature = "decision", test))]
    fn detect_batch(
        &self,
        texts: &[&str],
    ) -> Result<Vec<Vec<PiiMatch>>, Box<dyn std::error::Error + Send + Sync>> {
        texts.iter().map(|text| self.detect(text)).collect()
    }
}

#[cfg(feature = "ner")]
impl NerRuntime for NerDetector {
    fn detect(
        &self,
        text: &str,
    ) -> Result<Vec<PiiMatch>, Box<dyn std::error::Error + Send + Sync>> {
        NerDetector::detect(self, text)
    }
}

#[cfg(feature = "ner")]
impl NerRuntime for TokenClassDetector {
    fn detect(
        &self,
        text: &str,
    ) -> Result<Vec<PiiMatch>, Box<dyn std::error::Error + Send + Sync>> {
        TokenClassDetector::detect(self, text)
    }

    #[cfg(feature = "decision")]
    fn detect_batch(
        &self,
        texts: &[&str],
    ) -> Result<Vec<Vec<PiiMatch>>, Box<dyn std::error::Error + Send + Sync>> {
        TokenClassDetector::detect_batch(self, texts)
    }
}

/// A detected PII match.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

/// Complete detection plus value-free policy counters. Errors return neither
/// partial findings nor success statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectionResult {
    pub matches: Vec<PiiMatch>,
    pub stats: TracePolicyStats,
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
    /// Recall-first decoding for the token backend: flag on total entity mass
    /// (1 - P(O)) instead of argmax. See `TokenClassDetector::set_recall_first`.
    pub ner_recall_first: bool,
    /// NER entity labels to detect
    pub ner_labels: Option<Vec<String>>,
    /// NER cache directory for model files
    pub ner_cache_dir: Option<std::path::PathBuf>,
    /// Which NER backend(s) to run.
    pub ner_backend: NerBackend,
    /// Execution provider for token-classification NER.
    pub ner_provider: NerProvider,
    /// Path to the token-classification model: a local dir (model.onnx + tokenizer.json
    /// + config.json) or a HuggingFace repo id. Used for the TokenClass/Both backends.
    pub ner_token_model: Option<std::path::PathBuf>,
    /// Opt-in trace filtering and literal lists. None preserves existing behavior.
    pub trace_policy: Option<TracePolicyConfig>,
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
            ner_recall_first: false,
            ner_labels: None,
            ner_cache_dir: None,
            ner_backend: NerBackend::default(),
            ner_provider: NerProvider::default(),
            ner_token_model: None,
            trace_policy: None,
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

    /// Enable recall-first decoding for the token backend.
    pub fn with_ner_recall_first(mut self, on: bool) -> Self {
        self.ner_recall_first = on;
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

    /// Select the token-classification NER execution provider.
    pub fn with_ner_provider(mut self, provider: NerProvider) -> Self {
        self.ner_provider = provider;
        self
    }

    /// Configure literal lists and optional agent-trace filtering.
    #[allow(
        dead_code,
        reason = "Public policy builder for parent configuration integration"
    )]
    pub fn with_trace_policy(mut self, policy: TracePolicyConfig) -> Self {
        self.trace_policy = Some(policy);
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
    trace_policy: Option<TracePolicy>,
    /// Optional GLiNER detector for name/address detection
    #[cfg(feature = "ner")]
    ner: Option<Box<dyn NerRuntime>>,
    /// Optional token-classification detector
    #[cfg(feature = "ner")]
    token: Option<Box<dyn NerRuntime>>,
}

impl Detector {
    /// Create a new detector with the given configuration.
    pub fn new(config: &DetectorConfig) -> Result<Self, DetectionError> {
        let trace_policy = config
            .trace_policy
            .as_ref()
            .map(TracePolicy::new)
            .transpose()
            .map_err(|_| DetectionError::TracePolicyConfiguration)?;
        #[cfg(not(feature = "ner"))]
        if config.ner_enabled {
            return Err(DetectionError::NerUnavailable);
        }
        if config.ner_enabled
            && config.ner_labels.is_some()
            && config.ner_backend != NerBackend::Gliner
        {
            return Err(DetectionError::Configuration);
        }
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
                NerBackend::Gliner => (
                    Some(Self::guard_ner(
                        DetectionError::Initialization { backend: "GLiNER" },
                        || Self::init_ner(config),
                    )?),
                    None,
                ),
                NerBackend::TokenClass => (
                    None,
                    Some(Self::guard_ner(
                        DetectionError::Initialization {
                            backend: "token-classification",
                        },
                        || Self::init_token(config),
                    )?),
                ),
                NerBackend::Both => (
                    Some(Self::guard_ner(
                        DetectionError::Initialization { backend: "GLiNER" },
                        || Self::init_ner(config),
                    )?),
                    Some(Self::guard_ner(
                        DetectionError::Initialization {
                            backend: "token-classification",
                        },
                        || Self::init_token(config),
                    )?),
                ),
            }
        } else {
            (None, None)
        };

        Ok(Self {
            regex_set,
            patterns,
            trace_policy,
            #[cfg(feature = "ner")]
            ner: ner_detector,
            #[cfg(feature = "ner")]
            token: token_detector,
        })
    }

    /// Third-party native wrappers may panic instead of returning an error.
    /// Any such failure is terminal; a caller must not reuse partial results.
    #[cfg(feature = "ner")]
    fn guard_ner<T>(
        error: DetectionError,
        operation: impl FnOnce() -> Result<T, DetectionError>,
    ) -> Result<T, DetectionError> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)).unwrap_or(Err(error))
    }

    /// Initialize NER detector from config.
    #[cfg(feature = "ner")]
    fn init_ner(config: &DetectorConfig) -> Result<Box<dyn NerRuntime>, DetectionError> {
        let labels = match &config.ner_labels {
            Some(labels) => crate::config::normalize_gliner_labels(labels)
                .map_err(|_| DetectionError::Configuration)?,
            None => crate::config::DEFAULT_GLINER_LABELS
                .iter()
                .map(|label| (*label).to_string())
                .collect(),
        };
        let error = || DetectionError::Initialization { backend: "GLiNER" };
        let paths = match Self::local_gliner_paths(config.ner_model.as_deref())? {
            Some(paths) => paths,
            None => {
                let mut model_config = config
                    .ner_model
                    .as_ref()
                    .map_or_else(NerModelConfig::default, NerModelConfig::with_model);
                if let Some(cache_dir) = &config.ner_cache_dir {
                    model_config = model_config.with_cache_dir(cache_dir);
                }
                model_config.download().map_err(|_| error())?
            }
        };
        NerDetector::new(
            &paths.tokenizer,
            &paths.model,
            Some(labels),
            config.ner_threshold,
        )
        .map(|runtime| Box::new(runtime) as Box<dyn NerRuntime>)
        .map_err(|_| error())
    }

    /// Explicit local paths never become remote repository requests. Preflight
    /// missing files before registering native exit hooks (notably on macOS).
    #[cfg(feature = "ner")]
    fn local_gliner_paths(model: Option<&str>) -> Result<Option<NerModelPaths>, DetectionError> {
        let Some(model) = model else {
            return Ok(None);
        };
        let path = std::path::Path::new(model);
        if !path.is_absolute() && !path.exists() {
            return Ok(None);
        }
        let onnx = if path.join("model.onnx").is_file() {
            path.join("model.onnx")
        } else {
            path.join("onnx/model.onnx")
        };
        let tokenizer = path.join("tokenizer.json");
        if !tokenizer.is_file() || !onnx.is_file() {
            return Err(DetectionError::Initialization { backend: "GLiNER" });
        }
        Ok(Some(NerModelPaths::new(tokenizer, onnx)))
    }

    /// Initialize the token-classification detector from config.
    #[cfg(feature = "ner")]
    fn init_token(config: &DetectorConfig) -> Result<Box<dyn NerRuntime>, DetectionError> {
        use log::info;

        // A local directory is loaded directly; anything else is treated as a
        // HuggingFace repo id and downloaded/cached (like the GLiNER backend).
        // When unset, fall back to the default published model repo.
        let result = match config.ner_token_model {
            Some(ref model) if model.is_absolute() || model.exists() => {
                TokenClassDetector::from_dir(model, config.ner_threshold, config.ner_provider)
            }
            Some(ref model) => TokenClassDetector::from_repo(
                &model.to_string_lossy(),
                config.ner_cache_dir.as_deref(),
                config.ner_threshold,
                config.ner_provider,
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
                    config.ner_provider,
                )
            }
        };

        match result {
            Ok(mut detector) => {
                if config.ner_recall_first {
                    detector.set_recall_first(true);
                    info!("Token-classification NER: recall-first decoding enabled");
                }
                info!("Token-classification NER detector ready");
                Ok(Box::new(detector))
            }
            Err(_) => Err(DetectionError::Initialization {
                backend: "token-classification",
            }),
        }
    }

    /// Create a detector with default configuration (all high-confidence patterns).
    pub fn with_defaults() -> Self {
        #[expect(
            clippy::expect_used,
            reason = "Regex-only built-in defaults are infallible"
        )]
        Self::new(&DetectorConfig::default()).expect("built-in regex-only detector must initialize")
    }

    /// Detect all PII in the given text.
    ///
    /// Returns matches sorted by start position. Combines regex-based
    /// detection with NER-based detection if enabled.
    pub fn detect(&self, text: &str) -> Result<Vec<PiiMatch>, DetectionError> {
        self.detect_with_stats(text).map(|result| result.matches)
    }

    /// Same fail-closed scan as `detect`, with aggregate policy counters.
    pub fn detect_with_stats(&self, text: &str) -> Result<DetectionResult, DetectionError> {
        self.detect_with_stats_in_context(text, "")
    }

    /// Scan the original leaf text with an optional structured-field privacy cue.
    /// Context only vetoes NER suppressions when it contains a private cue; it
    /// never authorizes a suppression. Inference inputs, regex/literal matching
    /// and returned byte offsets refer exclusively to `text`, not `context`.
    pub fn detect_with_stats_in_context(
        &self,
        text: &str,
        context: &str,
    ) -> Result<DetectionResult, DetectionError> {
        #[cfg(not(feature = "ner"))]
        let _ = context;
        #[cfg(feature = "ner")]
        let private_context = TracePolicy::has_private_context(context);
        let mut matches = self.regex_matches(text);
        let sensitive = self.sensitive_matches(text);
        #[allow(unused_mut, reason = "NER builds update runtime counters")]
        let mut stats = TracePolicyStats {
            sensitive_term_matches: sensitive.len(),
            ..Default::default()
        };
        #[cfg(feature = "ner")]
        for (backend, runtime) in self.runtimes() {
            let detected = Self::guard_ner(DetectionError::Inference { backend }, || {
                runtime
                    .detect(text)
                    .map_err(|_| DetectionError::Inference { backend })
            })?;
            let detected = self.filter_ner(text, detected, &sensitive, private_context, &mut stats);
            Self::merge_matches(&mut matches, detected);
        }
        Self::finish_matches(text, &mut matches, sensitive);
        Ok(DetectionResult { matches, stats })
    }

    fn sensitive_matches(&self, text: &str) -> Vec<PiiMatch> {
        self.trace_policy
            .as_ref()
            .map_or_else(Vec::new, |policy| policy.sensitive_matches(text))
    }

    #[cfg(feature = "ner")]
    fn filter_ner(
        &self,
        text: &str,
        detected: Vec<PiiMatch>,
        sensitive: &[PiiMatch],
        private_context: bool,
        stats: &mut TracePolicyStats,
    ) -> Vec<PiiMatch> {
        match &self.trace_policy {
            Some(policy) => policy.filter_ner_with_private_context(
                text,
                detected,
                sensitive,
                private_context,
                stats,
            ),
            None => {
                stats.ner_candidates += detected.len();
                detected
            }
        }
    }

    fn finish_matches(text: &str, matches: &mut Vec<PiiMatch>, sensitive: Vec<PiiMatch>) {
        TracePolicy::add_sensitive_coverage(text, matches, sensitive);
        Self::sort_matches(matches);
    }

    #[cfg(feature = "ner")]
    fn runtimes(&self) -> impl Iterator<Item = (&'static str, &dyn NerRuntime)> {
        [
            ("GLiNER", self.ner.as_deref()),
            ("token-classification", self.token.as_deref()),
        ]
        .into_iter()
        .filter_map(|(backend, runtime)| runtime.map(|runtime| (backend, runtime)))
    }

    #[cfg(feature = "ner")]
    fn merge_matches(matches: &mut Vec<PiiMatch>, detected: Vec<PiiMatch>) {
        for candidate in detected {
            let overlaps = matches
                .iter()
                .any(|existing| candidate.start < existing.end && candidate.end > existing.start);
            if !overlaps {
                matches.push(candidate);
            }
        }
    }

    fn sort_matches(matches: &mut [PiiMatch]) {
        matches.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| b.len().cmp(&a.len())));
    }

    /// Detect across records. Tokens use a padded forward pass; GLiNER scans
    /// each record. Every requested runtime must succeed before any result is
    /// returned, including when regex has already found PII.
    #[cfg(any(feature = "decision", test))]
    pub fn detect_batch(&self, texts: &[&str]) -> Result<Vec<Vec<PiiMatch>>, DetectionError> {
        self.detect_batch_with_stats(texts)
            .map(|results| results.into_iter().map(|result| result.matches).collect())
    }

    /// Batch counterpart of `detect_with_stats`; all engines must succeed.
    #[cfg(any(feature = "decision", test))]
    pub fn detect_batch_with_stats(
        &self,
        texts: &[&str],
    ) -> Result<Vec<DetectionResult>, DetectionError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let sensitive: Vec<_> = texts
            .iter()
            .map(|text| self.sensitive_matches(text))
            .collect();
        let mut per_text: Vec<_> = texts
            .iter()
            .zip(&sensitive)
            .map(|(text, terms)| DetectionResult {
                matches: self.regex_matches(text),
                stats: TracePolicyStats {
                    sensitive_term_matches: terms.len(),
                    ..Default::default()
                },
            })
            .collect();
        #[cfg(feature = "ner")]
        for (backend, runtime) in self.runtimes() {
            let batch = Self::guard_ner(DetectionError::Inference { backend }, || {
                runtime
                    .detect_batch(texts)
                    .map_err(|_| DetectionError::Inference { backend })
            })?;
            // A malformed runtime response cannot certify complete coverage.
            if batch.len() != texts.len() {
                return Err(DetectionError::Inference { backend });
            }
            for (((result, detected), text), terms) in
                per_text.iter_mut().zip(batch).zip(texts).zip(&sensitive)
            {
                let detected = self.filter_ner(text, detected, terms, false, &mut result.stats);
                Self::merge_matches(&mut result.matches, detected);
            }
        }
        for ((result, text), terms) in per_text.iter_mut().zip(texts).zip(sensitive) {
            Self::finish_matches(text, &mut result.matches, terms);
        }
        Ok(per_text)
    }

    /// Run the regex pass over a single text and return its matches.
    fn regex_matches(&self, text: &str) -> Vec<PiiMatch> {
        let mut matches = Vec::new();
        let matching_indices: Vec<usize> = self.regex_set.matches(text).into_iter().collect();
        for idx in matching_indices {
            let pattern = self.patterns[idx];
            for captures in pattern.regex.captures_iter(text) {
                // Contextual patterns may include benign syntax to establish
                // sensitivity. Only their named `pii` span is a finding; other
                // alternatives and ordinary patterns retain the full match.
                let Some(m) = captures.name("pii").or_else(|| captures.get(0)) else {
                    continue;
                };
                if Self::is_social_handle_pattern(pattern.name)
                    && !Self::is_valid_social_handle(text, m.start(), m.end())
                {
                    continue;
                }
                let range = if pattern.name == "iban" {
                    let Some(range) = iban_span(text, m.start(), m.end()) else {
                        continue;
                    };
                    range
                } else {
                    m.range()
                };
                matches.push(PiiMatch {
                    pattern_name: pattern.name.to_string(),
                    matched_text: text[range.clone()].to_string(),
                    start: range.start,
                    end: range.end,
                    confidence: pattern.confidence,
                    category: pattern.category,
                });
            }
        }
        matches
    }

    /// Check for PII using every active engine; errors never imply absence.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Public API - used by consumers")
    )]
    pub fn contains_pii(&self, text: &str) -> Result<bool, DetectionError> {
        self.detect(text).map(|matches| !matches.is_empty())
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
        self.ner.is_some() || self.token.is_some()
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
pub(crate) mod tests {
    use super::*;

    #[cfg(feature = "ner")]
    struct InjectedRuntime {
        fail_on: Option<&'static str>,
        batch_fails: bool,
    }

    #[cfg(feature = "ner")]
    impl NerRuntime for InjectedRuntime {
        fn detect(
            &self,
            text: &str,
        ) -> Result<Vec<PiiMatch>, Box<dyn std::error::Error + Send + Sync>> {
            if self.fail_on.is_some_and(|needle| text.contains(needle)) {
                return Err(format!(
                    "private runtime error: {text}; credential=secret-provider-key"
                )
                .into());
            }
            Ok(vec![PiiMatch {
                pattern_name: "person".into(),
                matched_text: text.into(),
                start: 0,
                end: text.len(),
                confidence: Confidence::High,
                category: PiiCategory::Identity,
            }])
        }

        fn detect_batch(
            &self,
            texts: &[&str],
        ) -> Result<Vec<Vec<PiiMatch>>, Box<dyn std::error::Error + Send + Sync>> {
            if self.batch_fails {
                return Err(format!("private batch error: {texts:?}; secret-provider-key").into());
            }
            texts.iter().map(|text| self.detect(text)).collect()
        }
    }

    #[cfg(feature = "ner")]
    pub(crate) fn injected(
        backend: NerBackend,
        fail_on: Option<&'static str>,
        batch_fails: bool,
    ) -> Detector {
        let mut detector = Detector::with_defaults();
        let failing = || {
            Box::new(InjectedRuntime {
                fail_on,
                batch_fails,
            }) as Box<dyn NerRuntime>
        };
        let healthy = || {
            Box::new(InjectedRuntime {
                fail_on: None,
                batch_fails: false,
            }) as Box<dyn NerRuntime>
        };
        match backend {
            NerBackend::Gliner => detector.ner = Some(failing()),
            NerBackend::TokenClass => detector.token = Some(failing()),
            NerBackend::Both => {
                detector.ner = Some(healthy());
                detector.token = Some(failing());
            }
        }
        detector
    }

    #[cfg(feature = "ner")]
    #[test]
    fn runtime_panics_are_terminal_sanitized_initialization_and_inference_errors() {
        struct PanickingRuntime;
        impl NerRuntime for PanickingRuntime {
            #[expect(clippy::panic, reason = "Inject a third-party native-wrapper panic")]
            fn detect(
                &self,
                _text: &str,
            ) -> Result<Vec<PiiMatch>, Box<dyn std::error::Error + Send + Sync>> {
                panic!("synthetic-private-provider-credential");
            }
        }
        let expected = DetectionError::Initialization { backend: "test" };
        #[expect(clippy::panic, reason = "Inject a third-party initialization panic")]
        let error =
            Detector::guard_ner::<()>(expected, || panic!("synthetic-private-config")).unwrap_err();
        assert_eq!(error, expected);
        for backend in [NerBackend::Gliner, NerBackend::TokenClass, NerBackend::Both] {
            let mut detector = injected(backend, None, false);
            let name = if backend == NerBackend::Gliner {
                detector.ner = Some(Box::new(PanickingRuntime));
                "GLiNER"
            } else {
                detector.token = Some(Box::new(PanickingRuntime));
                "token-classification"
            };
            let expected = DetectionError::Inference { backend: name };
            assert_eq!(
                detector
                    .detect("private-name secret@example.invalid")
                    .unwrap_err(),
                expected
            );
            assert_eq!(
                detector
                    .detect_batch(&["first", "secret@example.invalid"])
                    .unwrap_err(),
                expected
            );
            assert!(!format!("{expected:?}").contains("synthetic-private"));
        }
    }

    #[test]
    fn default_detection_redacts_whole_valid_ibans_not_digit_fragments() {
        let detector = Detector::with_defaults();
        for (text, iban) in [
            ("IBAN: DE89370400440532013000\n", "DE89370400440532013000"),
            (
                "pay DE89 3704 0044 0532 0130 00 today",
                "DE89 3704 0044 0532 0130 00",
            ),
        ] {
            let matches = detector.detect(text).unwrap();
            let start = text.find(iban).unwrap();
            assert!(
                matches.iter().any(|m| m.pattern_name == "iban"
                    && m.start == start
                    && m.end == start + iban.len()
                    && m.confidence == Confidence::High),
                "{matches:?}"
            );
        }
        assert!(
            detector
                .detect("IBAN: DE88370400440532013000")
                .unwrap()
                .iter()
                .all(|m| m.pattern_name != "iban")
        );
    }

    #[cfg(feature = "ner")]
    #[test]
    fn runtime_errors_are_not_partial_regex_success_and_never_retain_private_context() {
        for backend in [NerBackend::Gliner, NerBackend::TokenClass, NerBackend::Both] {
            let detector = injected(backend, Some("Alice"), false);
            let error = detector
                .detect("Alice Smith secret@example.invalid")
                .unwrap_err();
            assert!(error.to_string().contains("scan incomplete"));
            for rendered in [error.to_string(), format!("{error:?}")] {
                for private in ["Alice", "secret@example.invalid", "secret-provider-key"] {
                    assert!(!rendered.contains(private));
                }
            }
            assert!(std::error::Error::source(&error).is_none());
        }
    }

    #[cfg(feature = "ner")]
    #[test]
    fn both_fails_when_gliner_fails_even_with_healthy_token_runtime() {
        let mut detector = injected(NerBackend::Both, None, false);
        detector.ner = Some(Box::new(InjectedRuntime {
            fail_on: Some("Alice"),
            batch_fails: false,
        }));
        assert_eq!(
            detector.detect("Alice Smith").unwrap_err(),
            DetectionError::Inference { backend: "GLiNER" }
        );
        assert_eq!(
            detector
                .detect_batch(&["Completed record", "Alice Smith"])
                .unwrap_err(),
            DetectionError::Inference { backend: "GLiNER" }
        );
        assert_eq!(
            detector
                .contains_pii("Alice Smith secret@example.invalid")
                .unwrap_err(),
            DetectionError::Inference { backend: "GLiNER" }
        );
    }

    #[cfg(feature = "ner")]
    #[test]
    fn incomplete_batch_cardinality_cannot_be_reported_as_complete() {
        struct IncompleteBatch(usize);
        impl NerRuntime for IncompleteBatch {
            fn detect(
                &self,
                _text: &str,
            ) -> Result<Vec<PiiMatch>, Box<dyn std::error::Error + Send + Sync>> {
                Ok(Vec::new())
            }
            fn detect_batch(
                &self,
                _texts: &[&str],
            ) -> Result<Vec<Vec<PiiMatch>>, Box<dyn std::error::Error + Send + Sync>> {
                Ok(vec![Vec::new(); self.0])
            }
        }
        for count in [0, 1, 3] {
            let mut detector = Detector::with_defaults();
            detector.token = Some(Box::new(IncompleteBatch(count)));
            assert_eq!(
                detector
                    .detect_batch(&["one record", "second record"])
                    .unwrap_err(),
                DetectionError::Inference {
                    backend: "token-classification"
                }
            );
        }
    }

    #[cfg(feature = "ner")]
    #[test]
    fn batch_failure_or_failed_record_is_an_error_for_every_requested_backend() {
        for backend in [NerBackend::Gliner, NerBackend::TokenClass, NerBackend::Both] {
            for batch_fails in [false, true] {
                let detector = injected(backend, Some("Alice"), batch_fails);
                assert!(
                    detector
                        .detect_batch(&["Completed record", "Alice Smith"])
                        .is_err()
                );
            }
        }
    }

    #[cfg(feature = "ner")]
    #[test]
    fn structured_detection_and_sanitization_propagate_failed_leaf() {
        for backend in [NerBackend::Gliner, NerBackend::TokenClass, NerBackend::Both] {
            let detector = injected(backend, Some("Alice"), false);
            let input = r#"{"a":"Completed record", "b":["Alice Smith"]}"#;
            let mut replacer = crate::engine::Replacer::with_defaults();
            assert!(crate::engine::detect_json(input, &detector).is_err());
            assert!(crate::engine::process_json(input, &detector, &mut replacer).is_err());
        }
    }

    #[cfg(feature = "ner")]
    #[test]
    fn gliner_is_not_skipped_in_batch_and_token_only_reports_ready() {
        let detector = injected(NerBackend::Gliner, None, false);
        let batch = detector
            .detect_batch(&["Alice Smith", "Jörg Beispiel"])
            .unwrap();
        assert_eq!(
            batch
                .iter()
                .map(|matches| matches[0].matched_text.as_str())
                .collect::<Vec<_>>(),
            ["Alice Smith", "Jörg Beispiel"]
        );
        assert!(injected(NerBackend::TokenClass, None, false).ner_enabled());
    }

    #[cfg(feature = "ner")]
    #[test]
    fn invalid_local_models_fail_without_network_or_path_disclosure() {
        for backend in [NerBackend::Gliner, NerBackend::TokenClass, NerBackend::Both] {
            for fixture in ["empty", "corrupt", "missing"] {
                let dir = tempfile::tempdir().unwrap();
                let model = dir.path().join("private-model-credential");
                std::fs::create_dir(&model).unwrap();
                if fixture == "corrupt" {
                    for file in ["model.onnx", "tokenizer.json", "config.json"] {
                        std::fs::write(model.join(file), "confidential-corrupt-model").unwrap();
                    }
                }
                let path = if fixture == "missing" {
                    model.join("missing")
                } else {
                    model.clone()
                };
                let config = DetectorConfig::default()
                    .with_ner(true)
                    .with_ner_backend(backend)
                    .with_ner_model(path.to_string_lossy())
                    .with_ner_token_model(path);
                let error = match Detector::new(&config) {
                    Err(error) => error,
                    Ok(_) => panic!("unavailable NER must fail closed"),
                };
                for rendered in [error.to_string(), format!("{error:?}")] {
                    assert!(!rendered.contains("private-model-credential"));
                    assert!(!rendered.contains("confidential-corrupt-model"));
                }
                assert!(std::error::Error::source(&error).is_none());
            }
        }
    }

    #[cfg(feature = "ner")]
    #[test]
    fn missing_or_corrupt_individual_model_artifacts_are_operational_errors() {
        for backend in [NerBackend::Gliner, NerBackend::TokenClass, NerBackend::Both] {
            let files: &[&str] = if backend == NerBackend::Gliner {
                &["model.onnx", "tokenizer.json"]
            } else {
                &["model.onnx", "tokenizer.json", "config.json"]
            };
            for file in files {
                for corrupt in [false, true] {
                    let dir = tempfile::tempdir().unwrap();
                    let path = dir.path().join("private-model-credential");
                    std::fs::create_dir(&path).unwrap();
                    tokenizers::Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default())
                        .save(path.join("tokenizer.json"), false)
                        .unwrap();
                    std::fs::write(
                        path.join("config.json"),
                        r#"{"id2label":{"0":"O","1":"B-person"}}"#,
                    )
                    .unwrap();
                    std::fs::write(path.join("model.onnx"), "not an ONNX model").unwrap();
                    if corrupt {
                        std::fs::write(path.join(file), "confidential-corrupt-artifact").unwrap();
                    } else {
                        std::fs::remove_file(path.join(file)).unwrap();
                    }
                    let config = DetectorConfig::default()
                        .with_ner(true)
                        .with_ner_backend(backend)
                        .with_ner_provider(NerProvider::Cpu)
                        .with_ner_model(path.to_string_lossy())
                        .with_ner_token_model(path);
                    let error = Detector::new(&config).err().unwrap();
                    assert!(matches!(error, DetectionError::Initialization { .. }));
                    for rendered in [error.to_string(), format!("{error:?}")] {
                        assert!(!rendered.contains("private-model-credential"));
                        assert!(!rendered.contains("confidential-corrupt-artifact"));
                    }
                }
            }
        }
    }

    #[cfg(feature = "ner")]
    #[test]
    fn public_detector_rejects_incompatible_or_unknown_labels_before_loading_models() {
        for backend in [NerBackend::TokenClass, NerBackend::Both] {
            assert!(matches!(
                Detector::new(
                    &DetectorConfig::default()
                        .with_ner(true)
                        .with_ner_backend(backend)
                        .with_ner_labels(vec!["person".into()])
                ),
                Err(DetectionError::Configuration)
            ));
        }
        for labels in [vec![], vec!["private-custom-label".into()]] {
            let error = Detector::new(
                &DetectorConfig::default()
                    .with_ner(true)
                    .with_ner_backend(NerBackend::Gliner)
                    .with_ner_labels(labels),
            )
            .err()
            .unwrap();
            assert_eq!(error, DetectionError::Configuration);
            assert!(!error.to_string().contains("private-custom-label"));
        }
    }

    #[cfg(not(feature = "ner"))]
    #[test]
    fn ner_requested_in_regex_only_build_is_an_error() {
        assert!(matches!(
            Detector::new(&DetectorConfig::default().with_ner(true)),
            Err(DetectionError::NerUnavailable)
        ));
        assert!(Detector::new(&DetectorConfig::default().with_ner(false)).is_ok());
    }

    #[test]
    fn disabled_policy_is_identical_and_sensitive_lists_work_without_ner_or_profile() {
        let text = "東京 secret@example.invalid buffer 4096";
        let baseline = Detector::with_defaults().detect(text).unwrap();
        let disabled = Detector::new(
            &DetectorConfig::default().with_trace_policy(TracePolicyConfig::default()),
        )
        .unwrap();
        assert_eq!(
            disabled.detect_with_stats(text).unwrap(),
            DetectionResult {
                matches: baseline.clone(),
                stats: TracePolicyStats::default()
            }
        );
        let policy = TracePolicyConfig {
            sensitive_terms: vec!["東京".into()],
            benign_terms: vec!["東京".into(), "secret@example.invalid".into()],
            ..Default::default()
        };
        let detector = Detector::new(&DetectorConfig::default().with_trace_policy(policy)).unwrap();
        let result = detector.detect_with_stats(text).unwrap();
        let mut expected = vec![PiiMatch {
            pattern_name: "sensitive_term".into(),
            matched_text: "東京".into(),
            start: 0,
            end: "東京".len(),
            confidence: Confidence::High,
            category: PiiCategory::Other,
        }];
        expected.extend(baseline);
        assert_eq!(
            result,
            DetectionResult {
                matches: expected,
                stats: TracePolicyStats {
                    sensitive_term_matches: 1,
                    ..Default::default()
                }
            }
        );
        assert_eq!(detector.detect(text).unwrap(), result.matches);
        assert_eq!(
            detector.detect_batch_with_stats(&[text, "clean"]).unwrap(),
            vec![
                result,
                DetectionResult {
                    matches: vec![],
                    stats: TracePolicyStats::default()
                }
            ]
        );
    }

    #[test]
    fn whole_sensitive_overlap_is_redacted_without_removing_regex_findings() {
        let text = "person@example.invalid-extra";
        let baseline = Detector::with_defaults().detect(text).unwrap();
        let config = DetectorConfig::default().with_trace_policy(TracePolicyConfig {
            sensitive_terms: vec!["invalid-extra".into()],
            term_boundary: super::super::trace_policy::TermBoundary::Substring,
            ..Default::default()
        });
        let detector = Detector::new(&config).unwrap();
        let result = detector.detect_with_stats(text).unwrap();
        let mut expected = vec![PiiMatch {
            pattern_name: "sensitive_term".into(),
            matched_text: text.into(),
            start: 0,
            end: text.len(),
            confidence: Confidence::High,
            category: PiiCategory::Other,
        }];
        expected.extend(baseline.clone());
        assert_eq!(
            result,
            DetectionResult {
                matches: expected,
                stats: TracePolicyStats {
                    sensitive_term_matches: 1,
                    ..Default::default()
                }
            }
        );
        for regex in baseline {
            assert!(result.matches.contains(&regex));
        }
        let (redacted, replacements) =
            crate::engine::Replacer::with_defaults().replace_all(text, &result.matches);
        assert!(!redacted.contains("invalid-extra"));
        assert_eq!(replacements.len(), 1);
        assert_eq!(replacements[0].original, text);
    }

    #[test]
    fn trace_lists_preserve_every_active_high_confidence_regex_credential() {
        use super::super::trace_policy::TraceProfile;
        let baseline = Detector::with_defaults();
        let credentials: Vec<_> = baseline
            .patterns
            .iter()
            .filter(|p| {
                p.category == PiiCategory::Authentication && p.confidence == Confidence::High
            })
            .collect();
        assert!(!credentials.is_empty());
        for pattern in credentials {
            let text = format!("code metadata buffer: {}", pattern.example);
            let expected = baseline.detect(&text).unwrap();
            assert!(
                expected.iter().any(|m| m.pattern_name == pattern.name),
                "credential fixture {}",
                pattern.name
            );
            let detector = Detector::new(&DetectorConfig::default().with_trace_policy(
                TracePolicyConfig {
                    profile: Some(TraceProfile::AgentTrace),
                    benign_terms: expected.iter().map(|m| m.matched_text.clone()).collect(),
                    ..Default::default()
                },
            ))
            .unwrap();
            assert_eq!(detector.detect(&text).unwrap(), expected);
            assert_eq!(
                detector.detect_batch(&[&text]).unwrap(),
                vec![expected.clone()]
            );
            let detector = Detector::new(&DetectorConfig::default().with_trace_policy(
                TracePolicyConfig {
                    profile: Some(TraceProfile::AgentTrace),
                    sensitive_terms: expected.iter().map(|m| m.matched_text.clone()).collect(),
                    benign_terms: expected.iter().map(|m| m.matched_text.clone()).collect(),
                    ..Default::default()
                },
            ))
            .unwrap();
            let result = detector.detect_with_stats(&text).unwrap();
            for original in &expected {
                assert!(result.matches.contains(original));
            }
            assert!(result.stats.sensitive_term_matches > 0);
        }
    }

    #[test]
    fn invalid_trace_policy_configuration_is_a_sanitized_operational_error() {
        let config = DetectorConfig::default().with_trace_policy(TracePolicyConfig {
            sensitive_terms: vec!["synthetic-secret\ninvalid".into()],
            ..Default::default()
        });
        let error = Detector::new(&config).err().unwrap();
        assert_eq!(error, DetectionError::TracePolicyConfiguration);
        assert_eq!(error.to_string(), "invalid trace policy configuration");
        assert!(std::error::Error::source(&error).is_none());
        assert!(!format!("{config:?}").contains("synthetic-secret"));
    }

    #[cfg(feature = "ner")]
    #[test]
    fn trace_policy_is_applied_before_ner_merge_and_both_paths_agree() {
        use super::super::trace_policy::TraceProfile;
        struct CounterRuntime;
        impl NerRuntime for CounterRuntime {
            fn detect(
                &self,
                text: &str,
            ) -> Result<Vec<PiiMatch>, Box<dyn std::error::Error + Send + Sync>> {
                let start = text.find("4096").unwrap();
                Ok(vec![PiiMatch {
                    pattern_name: "pin".into(),
                    matched_text: "4096".into(),
                    start,
                    end: start + 4,
                    confidence: Confidence::High,
                    category: PiiCategory::Financial,
                }])
            }
        }
        let mut detector = Detector::new(&DetectorConfig::default().with_trace_policy(
            TracePolicyConfig {
                profile: Some(TraceProfile::AgentTrace),
                ..Default::default()
            },
        ))
        .unwrap();
        detector.token = Some(Box::new(CounterRuntime));
        let texts = [
            "buffer 4096 bytes",
            "PIN: 4096",
            "not a PIN 4096; account credential",
        ];
        let results: Vec<_> = texts
            .iter()
            .map(|text| detector.detect_with_stats(text).unwrap())
            .collect();
        assert_eq!(
            results[0],
            DetectionResult {
                matches: vec![],
                stats: TracePolicyStats {
                    ner_candidates: 1,
                    suppressed_technical_values: 1,
                    ..Default::default()
                }
            }
        );
        for (result, text) in results.iter().zip(texts).skip(1) {
            let start = text.find("4096").unwrap();
            assert_eq!(
                *result,
                DetectionResult {
                    matches: vec![PiiMatch {
                        pattern_name: "pin".into(),
                        matched_text: "4096".into(),
                        start,
                        end: start + 4,
                        confidence: Confidence::High,
                        category: PiiCategory::Financial
                    }],
                    stats: TracePolicyStats {
                        ner_candidates: 1,
                        ..Default::default()
                    }
                }
            );
        }
        assert_eq!(detector.detect_batch_with_stats(&texts).unwrap(), results);
        assert_eq!(
            detector.detect_batch(&texts).unwrap(),
            results
                .into_iter()
                .map(|result| result.matches)
                .collect::<Vec<_>>()
        );
    }

    #[cfg(feature = "ner")]
    #[test]
    fn trace_policy_and_term_success_never_hide_engine_failure() {
        use super::super::trace_policy::TraceProfile;
        for backend in [NerBackend::Gliner, NerBackend::TokenClass, NerBackend::Both] {
            for batch_fails in [false, true] {
                let mut detector = injected(backend, Some("Alice"), batch_fails);
                detector.trace_policy = Some(
                    TracePolicy::new(&TracePolicyConfig {
                        profile: Some(TraceProfile::AgentTrace),
                        sensitive_terms: vec!["Alice".into()],
                        benign_terms: vec!["Alice".into()],
                        ..Default::default()
                    })
                    .unwrap(),
                );
                assert!(
                    detector
                        .detect_with_stats("Alice secret@example.invalid")
                        .is_err()
                );
                assert!(
                    detector
                        .detect_batch_with_stats(&["buffer 4096", "Alice secret@example.invalid"])
                        .is_err()
                );
                assert!(
                    detector
                        .contains_pii("Alice secret@example.invalid")
                        .is_err()
                );
            }
        }
    }

    #[test]
    fn test_detect_email() {
        let detector = Detector::with_defaults();
        let matches = detector
            .detect("Contact me at john.doe@example.com for more info.")
            .unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern_name, "email");
        assert_eq!(matches[0].matched_text, "john.doe@example.com");
    }

    #[test]
    fn test_detect_multiple() {
        let detector =
            Detector::new(&DetectorConfig::default().with_min_confidence(Confidence::High))
                .unwrap();
        // Use valid US phone format (exchange must start with 2-9)
        let text = "Email: test@example.com, Phone: (555) 234-5678, SSN: 123-45-6789";
        let matches = detector.detect(text).unwrap();

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
        let matches = detector.detect("This text contains no PII.").unwrap();

        assert!(matches.is_empty());
    }

    #[test]
    fn test_contains_pii() {
        let detector = Detector::with_defaults();

        assert!(
            detector
                .contains_pii("My email is test@example.com")
                .unwrap()
        );
        assert!(!detector.contains_pii("No PII here").unwrap());
    }

    #[test]
    fn test_pattern_filtering() {
        let config = DetectorConfig::default().with_patterns(["email", "ssn"]);
        let detector = Detector::new(&config).unwrap();

        let active = detector.active_patterns();
        assert!(active.contains(&"email"));
        assert!(active.contains(&"ssn"));
        assert!(!active.contains(&"phone_us"));
    }

    #[test]
    fn test_pattern_exclusion() {
        let config = DetectorConfig::default().without_patterns(["email"]);
        let detector = Detector::new(&config).unwrap();

        let active = detector.active_patterns();
        assert!(!active.contains(&"email"));
        assert!(active.contains(&"ssn"));
    }

    #[test]
    fn test_confidence_filtering() {
        let high_only = Detector::new(&DetectorConfig::high_confidence_only()).unwrap();
        let all =
            Detector::new(&DetectorConfig::all_patterns().with_min_confidence(Confidence::Low))
                .unwrap();

        // High confidence detector should have fewer patterns
        assert!(high_only.active_patterns().len() <= all.active_patterns().len());
    }
}

#[cfg(test)]
mod structured_context_tests {
    use super::*;
    use crate::engine::trace_policy::TraceProfile;

    fn profile() -> TracePolicyConfig {
        TracePolicyConfig {
            profile: Some(TraceProfile::AgentTrace),
            public_hosts: vec!["docs.example".into()],
            ..Default::default()
        }
    }

    #[cfg(feature = "ner")]
    struct ExactLeafRuntime {
        text: String,
        finding: PiiMatch,
    }

    #[cfg(feature = "ner")]
    impl NerRuntime for ExactLeafRuntime {
        fn detect(
            &self,
            text: &str,
        ) -> Result<Vec<PiiMatch>, Box<dyn std::error::Error + Send + Sync>> {
            // Every call proves context was not prepended to the model input.
            assert_eq!(text, self.text);
            Ok(vec![self.finding.clone()])
        }
    }

    #[cfg(feature = "ner")]
    fn leaf_detector(
        config: TracePolicyConfig,
        text: &str,
        value: &str,
        name: &str,
        category: PiiCategory,
    ) -> (Detector, PiiMatch) {
        let start = text.find(value).unwrap();
        let finding = PiiMatch {
            pattern_name: name.into(),
            matched_text: value.into(),
            start,
            end: start + value.len(),
            confidence: Confidence::High,
            category,
        };
        let mut detector =
            Detector::new(&DetectorConfig::default().with_trace_policy(config)).unwrap();
        detector.token = Some(Box::new(ExactLeafRuntime {
            text: text.into(),
            finding: finding.clone(),
        }));
        (detector, finding)
    }

    #[cfg(feature = "ner")]
    #[test]
    fn structured_private_field_vetoes_dob_filtering_with_original_utf8_offsets() {
        let record: serde_json::Value =
            serde_json::from_str(r#"{"dob":"東京 build 1990-04-06"}"#).unwrap();
        let text = record["dob"].as_str().unwrap();
        let (detector, finding) = leaf_detector(
            profile(),
            text,
            "1990-04-06",
            "date_of_birth",
            PiiCategory::Identity,
        );
        assert_eq!(
            detector.detect_with_stats(text).unwrap(),
            DetectionResult {
                matches: vec![],
                stats: TracePolicyStats {
                    ner_candidates: 1,
                    suppressed_technical_values: 1,
                    ..Default::default()
                },
            }
        );
        for context in [
            "$.dob",
            "$.birth",
            "$.date_of_birth",
            "$.client",
            "$.name",
            "$.firstName",
            "$.license",
        ] {
            let result = detector
                .detect_with_stats_in_context(text, context)
                .unwrap();
            assert_eq!(
                result,
                DetectionResult {
                    matches: vec![finding.clone()],
                    stats: TracePolicyStats {
                        ner_candidates: 1,
                        ..Default::default()
                    },
                }
            );
            assert_eq!(result.matches[0].start, "東京 build ".len());
            assert_eq!(
                &text[result.matches[0].start..result.matches[0].end],
                "1990-04-06"
            );
        }
    }

    #[cfg(feature = "ner")]
    #[test]
    fn private_context_vetoes_public_technical_and_benign_suppressions() {
        for (text, value, pattern, category, baseline_stats) in [
            (
                "buffer 4096",
                "4096",
                "pin",
                PiiCategory::Financial,
                TracePolicyStats {
                    ner_candidates: 1,
                    suppressed_technical_values: 1,
                    ..Default::default()
                },
            ),
            (
                "docs https://docs.example/crate",
                "https://docs.example/crate",
                "url",
                PiiCategory::Network,
                TracePolicyStats {
                    ner_candidates: 1,
                    suppressed_public_urls: 1,
                    ..Default::default()
                },
            ),
            (
                "Cargo",
                "Cargo",
                "person",
                PiiCategory::Identity,
                TracePolicyStats {
                    ner_candidates: 1,
                    suppressed_benign_terms: 1,
                    ..Default::default()
                },
            ),
        ] {
            let mut config = profile();
            config.benign_terms = vec!["Cargo".into()];
            let (detector, finding) = leaf_detector(config, text, value, pattern, category);
            assert_eq!(
                detector.detect_with_stats(text).unwrap(),
                DetectionResult {
                    matches: vec![],
                    stats: baseline_stats
                }
            );
            assert_eq!(
                detector
                    .detect_with_stats_in_context(text, "$.client")
                    .unwrap(),
                DetectionResult {
                    matches: vec![finding],
                    stats: TracePolicyStats {
                        ner_candidates: 1,
                        ..Default::default()
                    }
                }
            );
        }
    }

    #[cfg(feature = "ner")]
    #[test]
    fn unknown_fields_cannot_authorize_any_suppression_or_change_batch_semantics() {
        for (text, pattern, category) in [
            ("4096", "pin", PiiCategory::Financial),
            ("1990-04-06", "date_of_birth", PiiCategory::Identity),
            ("https://docs.example/crate", "url", PiiCategory::Network),
            ("bash", "person", PiiCategory::Identity),
        ] {
            let (detector, finding) = leaf_detector(profile(), text, text, pattern, category);
            let expected = DetectionResult {
                matches: vec![finding],
                stats: TracePolicyStats {
                    ner_candidates: 1,
                    ..Default::default()
                },
            };
            for context in [
                "",
                "$.build",
                "$.buffer",
                "$.tool",
                "$.docs",
                "$.metadata",
                "unknown public reference timestamp",
            ] {
                assert_eq!(
                    detector
                        .detect_with_stats_in_context(text, context)
                        .unwrap(),
                    expected
                );
            }
            assert_eq!(detector.detect_with_stats(text).unwrap(), expected);
            assert_eq!(detector.detect(text).unwrap(), expected.matches);
            assert_eq!(
                detector.detect_batch_with_stats(&[text]).unwrap(),
                vec![expected]
            );
        }
    }

    #[test]
    fn regex_credentials_and_literals_ignore_context_and_keep_leaf_offsets() {
        let text = "é 東京 AKIAIOSFODNN7EXAMPLE";
        let mut config = profile();
        config.sensitive_terms = vec!["東京".into()];
        config.benign_terms = vec!["東京".into(), "AKIAIOSFODNN7EXAMPLE".into()];
        let detector = Detector::new(&DetectorConfig::default().with_trace_policy(config)).unwrap();
        let expected = DetectionResult {
            matches: vec![
                PiiMatch {
                    pattern_name: "sensitive_term".into(),
                    matched_text: "東京".into(),
                    start: "é ".len(),
                    end: "é 東京".len(),
                    confidence: Confidence::High,
                    category: PiiCategory::Other,
                },
                PiiMatch {
                    pattern_name: "aws_key".into(),
                    matched_text: "AKIAIOSFODNN7EXAMPLE".into(),
                    start: "é 東京 ".len(),
                    end: text.len(),
                    confidence: Confidence::High,
                    category: PiiCategory::Authentication,
                },
            ],
            stats: TracePolicyStats {
                sensitive_term_matches: 1,
                ..Default::default()
            },
        };
        for context in [
            "",
            "$.dob",
            "$.docs",
            "東京 AKIAJQ7S6CFTMN2P5HUX",
            "a very long arbitrary field context unrelated to the leaf",
        ] {
            assert_eq!(
                detector
                    .detect_with_stats_in_context(text, context)
                    .unwrap(),
                expected
            );
            assert_eq!(
                detector
                    .detect_with_stats_in_context("clean", context)
                    .unwrap(),
                DetectionResult {
                    matches: vec![],
                    stats: TracePolicyStats::default()
                }
            );
        }
    }

    #[cfg(feature = "ner")]
    #[test]
    fn private_field_context_does_not_skip_inference_or_hide_errors() {
        for backend in [NerBackend::Gliner, NerBackend::TokenClass, NerBackend::Both] {
            let mut detector = super::tests::injected(backend, Some("Alice"), false);
            detector.trace_policy = Some(TracePolicy::new(&profile()).unwrap());
            assert!(
                detector
                    .detect_with_stats_in_context("Alice secret@example.invalid", "$.name")
                    .is_err()
            );
        }
    }
}
