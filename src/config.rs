//! Configuration loading and management.
//!
//! Configuration is loaded from the following sources in order of priority:
//! 1. Command line arguments (highest)
//! 2. Environment variables (NYM_*)
//! 3. Local config file (./config.toml if exists)
//! 4. Global config file ($XDG_CONFIG_HOME/nym/config.toml)
//! 5. Built-in defaults (lowest)

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::engine::detector::NerBackend;

#[cfg(feature = "decision")]
use crate::engine::DecisionConfig;
use crate::engine::{Confidence, ReplacementStrategy};

/// Application configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct Config {
    pub detection: DetectionConfig,
    pub replacement: ReplacementConfig,
    pub keys: KeysConfig,
    pub logging: LoggingConfig,
    pub runtime: RuntimeConfig,
    pub paths: PathsConfig,
    pub ner: NerConfig,
    pub ocr: OcrConfig,
    pub trace_policy: TracePolicySettings,
    /// Bounded, model-independent vocabulary discovery defaults.
    pub terms: crate::terms::Settings,
    /// Decision-model adjudication layer (System-One gate over candidate spans).
    #[cfg(feature = "decision")]
    #[serde(default)]
    pub decision: DecisionConfig,
    /// Named rulesets for common use cases
    #[serde(default)]
    pub rulesets: HashMap<String, RulesetConfig>,
    /// Per-pattern configuration overrides
    #[serde(default)]
    pub patterns: HashMap<String, PatternConfig>,
}

/// File-backed trace policy settings. Lists contain one literal per nonblank line.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TracePolicySettings {
    pub profile: Option<crate::engine::TraceProfile>,
    pub public_hosts: Vec<String>,
    pub sensitive_terms_files: Vec<String>,
    pub benign_terms_files: Vec<String>,
    pub term_boundary: crate::engine::TermBoundary,
    #[serde(default = "default_case_sensitive")]
    pub case_sensitive: bool,
}

const fn default_case_sensitive() -> bool {
    true
}

impl Default for TracePolicySettings {
    fn default() -> Self {
        let policy = crate::engine::TracePolicyConfig::default();
        Self {
            profile: policy.profile,
            public_hosts: policy.public_hosts,
            sensitive_terms_files: Vec::default(),
            benign_terms_files: Vec::default(),
            term_boundary: policy.term_boundary,
            case_sensitive: policy.case_sensitive,
        }
    }
}

impl std::fmt::Debug for TracePolicySettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TracePolicySettings")
            .field("profile", &self.profile)
            .field("public_host_count", &self.public_hosts.len())
            .field("sensitive_file_count", &self.sensitive_terms_files.len())
            .field("benign_file_count", &self.benign_terms_files.len())
            .field("term_boundary", &self.term_boundary)
            .field("case_sensitive", &self.case_sensitive)
            .finish()
    }
}

/// Load UTF-8 literal lists in declared file/line order; never expose values or paths.
pub fn load_term_files(paths: &[String]) -> Result<Vec<String>> {
    let mut terms = Vec::new();
    for path in paths {
        let expanded = shellexpand::full(path)
            .map_err(|_| anyhow::anyhow!("failed to expand trace term list path"))?;
        let content = std::fs::read_to_string(expanded.as_ref())
            .map_err(|_| anyhow::anyhow!("failed to read UTF-8 trace term list"))?;
        for line in content.lines().filter(|line| !line.trim().is_empty()) {
            if !terms.iter().any(|term| term == line) {
                terms.push(line.to_string());
            }
        }
    }
    Ok(terms)
}

/// Detection configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DetectionConfig {
    /// Patterns to enable (empty = all)
    pub enabled_patterns: Vec<String>,
    /// Patterns to disable
    pub disabled_patterns: Vec<String>,
    /// Minimum confidence level
    pub min_confidence: ConfidenceConfig,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            enabled_patterns: Vec::new(),
            disabled_patterns: Vec::new(),
            min_confidence: ConfidenceConfig::High,
        }
    }
}

/// Confidence level for config.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ConfidenceConfig {
    #[default]
    High,
    Medium,
    Low,
}

impl From<ConfidenceConfig> for Confidence {
    fn from(c: ConfidenceConfig) -> Self {
        match c {
            ConfidenceConfig::High => Confidence::High,
            ConfidenceConfig::Medium => Confidence::Medium,
            ConfidenceConfig::Low => Confidence::Low,
        }
    }
}

/// Replacement configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ReplacementConfig {
    /// Replacement strategy
    pub strategy: StrategyConfig,
    /// Seed for deterministic replacements
    pub seed: Option<u64>,
    /// Domain for email replacements
    pub email_domain: String,
}

impl Default for ReplacementConfig {
    fn default() -> Self {
        Self {
            strategy: StrategyConfig::Placeholder,
            seed: None,
            email_domain: "example.com".to_string(),
        }
    }
}

/// Strategy for config.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum StrategyConfig {
    #[default]
    Placeholder,
    Mask,
    Hash,
    Random,
    Consistent,
}

impl From<StrategyConfig> for ReplacementStrategy {
    fn from(s: StrategyConfig) -> Self {
        match s {
            StrategyConfig::Placeholder => ReplacementStrategy::Placeholder,
            StrategyConfig::Mask => ReplacementStrategy::Mask,
            StrategyConfig::Hash => ReplacementStrategy::Hash,
            StrategyConfig::Random => ReplacementStrategy::Random,
            StrategyConfig::Consistent => ReplacementStrategy::Consistent,
        }
    }
}

/// Key file configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct KeysConfig {
    /// Default key file location
    pub default_key_file: Option<String>,
    /// Auto-generate key file
    pub auto_generate: bool,
    /// Key file retention in days (0 = forever)
    pub retention_days: u32,
}

impl Default for KeysConfig {
    fn default() -> Self {
        Self {
            default_key_file: None,
            auto_generate: false,
            retention_days: 30,
        }
    }
}

/// Logging configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    /// Log level
    pub level: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "warn".to_string(),
        }
    }
}

/// Runtime configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RuntimeConfig {
    /// Parallel processing threads (0 = auto)
    pub parallelism: usize,
    /// Buffer size for file reading
    pub buffer_size: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            parallelism: 0,
            buffer_size: 65536,
        }
    }
}

/// Path configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct PathsConfig {
    /// Data directory
    pub data_dir: Option<String>,
    /// State directory
    pub state_dir: Option<String>,
}

/// OCR configuration for raster redaction (images, scanned PDF pages).
/// The engine runs as an external process — see docs/document-redaction.md.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OcrConfig {
    /// Scan raster images inside PDFs during detect/anon (`--ocr` overrides).
    pub enabled: bool,
    /// `auto` (nym-ocr, then tesseract), `nym-ocr`, `tesseract`,
    /// or a custom command template containing `{input}` that prints nym's
    /// OCR JSON contract.
    pub engine: String,
    /// Words below this recognition confidence are ignored (0.0-1.0).
    pub min_confidence: f32,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            engine: "auto".to_string(),
            min_confidence: 0.3,
        }
    }
}

/// Token NER execution-provider policy. CPU bypasses accelerator registration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum NerProvider {
    /// Use compiled-in accelerators, retrying on CPU if session setup fails.
    #[default]
    Auto,
    /// Use ONNX Runtime's CPU provider even in accelerator-enabled builds.
    Cpu,
}

/// NER (Named Entity Recognition) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NerConfig {
    /// Enable NER-based detection (requires 'ner' feature)
    pub enabled: bool,
    /// HuggingFace model repository
    /// Default: "onnx-community/gliner_multi-v2.1"
    pub model: String,
    /// Custom cache directory for model files
    /// If not set, uses HF_HOME env var or ~/.cache/huggingface/
    pub cache_dir: Option<String>,
    /// Confidence threshold for NER detections (0.0-1.0)
    pub threshold: f32,
    /// GLiNER-only entity restrictions. Omit to use the default GLiNER labels.
    /// Explicit labels require backend `gliner`; `tokens` and `both` reject them
    /// because token model classes are not filtered by this setting. Regex
    /// findings are unaffected. Empty and unknown explicit labels are errors.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<Vec<String>>,
    /// Recall-first decoding for the token backend: flag a token when its total
    /// entity probability (1 - P(O)) clears `threshold`, instead of requiring a
    /// single entity class to win the argmax. Raises recall (measured +5-9
    /// char-recall on OOD text) at moderate precision cost -- for redaction, a
    /// miss is a leak while an over-flag only over-redacts. Pair with a lower
    /// `threshold` (e.g. 0.2) for maximum-recall operation.
    #[serde(default)]
    pub recall_first: bool,
    /// Which NER backend(s) to run: `tokens` (default), `gliner`, or `both`.
    pub backend: NerBackend,
    /// Token NER only: `auto` (default) or explicit `cpu`; GLiNER is unaffected.
    /// For `both`, this controls only the token-classification session.
    /// Environment override: NYM_NER_PROVIDER when loading layered config.
    pub provider: NerProvider,
    /// Token-classification model for the `tokens`/`both` backends: a local dir
    /// (`model.onnx` + `tokenizer.json` + `config.json`) or a HuggingFace repo id
    /// (e.g. `Wismut/openmed-onnx/small`, `nationaldesignstudio/rampart`).
    /// Defaults to `Wismut/nym-pii-multilingual-small/int8` (144 MB) when unset.
    /// `openmed_model` is a legacy alias.
    #[serde(alias = "openmed_model")]
    pub token_model: Option<String>,
}

impl Default for NerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "onnx-community/gliner_multi-v2.1".to_string(),
            cache_dir: None,
            threshold: 0.5,
            labels: None,
            recall_first: false,
            backend: NerBackend::default(),
            provider: NerProvider::default(),
            token_model: None,
        }
    }
}

/// Shareable effective configuration, not proof of model/runtime readiness.
/// Model revisions remain `null` until runtime resolution; never invent a pin.
#[derive(Debug, Clone, Serialize)]
pub struct NerStatus {
    pub configured_enabled: bool,
    pub backend: NerBackend,
    pub threshold: f32,
    pub regex_scope: &'static str,
    pub models: Vec<NerModelStatus>,
}

/// Only built-in public catalog identifiers may appear; all other identifiers
/// (including local paths and private repositories) are replaced with `[custom]`.
#[derive(Debug, Clone, Serialize)]
pub struct NerModelStatus {
    pub backend: NerBackend,
    pub identifier: String,
    pub revision: Option<String>,
    pub label_scope: &'static str,
    pub labels: Option<Vec<String>>,
    pub decoding: &'static str,
    pub provider: Option<NerProvider>,
}

fn safe_model_identifier(model: &str) -> String {
    // Do not use the refreshed user catalog: it may contain private identifiers.
    let public = serde_json::from_str::<serde_json::Value>(include_str!("engine/catalog.json"))
        .ok()
        .is_some_and(|catalog| {
            catalog["models"].as_array().is_some_and(|models| {
                models
                    .iter()
                    .any(|entry| entry["slug"].as_str() == Some(model))
            })
        });
    if public {
        model.to_string()
    } else {
        "[custom]".to_string()
    }
}

/// Default GLiNER labels; token classes are defined by the selected token model.
pub const DEFAULT_GLINER_LABELS: &[&str] = &[
    "person",
    "organization",
    "street_address",
    "city",
    "country",
];

/// Validate and canonicalize supported GLiNER label aliases without echoing input.
/// Canonicalization ensures the engine maps aliases to the same PII category.
pub fn normalize_gliner_labels(labels: &[String]) -> Result<Vec<String>> {
    if labels.is_empty() {
        anyhow::bail!("ner.labels must not be empty; omit it to use GLiNER defaults");
    }
    let mut normalized = Vec::new();
    for label in labels {
        let canonical = match label.trim().to_ascii_lowercase().as_str() {
            "person" => "person",
            "first_name" | "firstname" | "given_name" | "givenname" => "first_name",
            "last_name" | "lastname" | "surname" | "family_name" => "last_name",
            "organization" | "company" | "org" => "organization",
            "street_address" | "address" => "street_address",
            "city" => "city",
            "state" | "province" | "region" => "state",
            "country" => "country",
            "location" => "location",
            "phone_number" | "phone" => "phone_number",
            "date" => "date",
            "date_of_birth" | "dob" | "birthday" | "birthdate" => "date_of_birth",
            "time" => "time",
            _ => anyhow::bail!(
                "ner.labels contains an unsupported GLiNER label; see the config schema for supported labels"
            ),
        };
        if !normalized.iter().any(|existing| existing == canonical) {
            normalized.push(canonical.to_string());
        }
    }
    Ok(normalized)
}

impl NerConfig {
    /// Resolve a CLI model override against the selected backend, before scanning.
    /// `both` requires the backend-specific config fields rather than an ambiguous
    /// generic override. This does not load or download models.
    pub fn resolved(&self, model_override: Option<&str>) -> Result<Self> {
        let mut effective = self.clone();
        if let Some(model) = model_override {
            match effective.backend {
                NerBackend::TokenClass => effective.token_model = Some(model.to_string()),
                NerBackend::Gliner => effective.model = model.to_string(),
                NerBackend::Both => anyhow::bail!(
                    "--ner-model is ambiguous with backend=both; set ner.model and ner.token_model separately"
                ),
            }
        }
        if !effective.threshold.is_finite() || !(0.0..=1.0).contains(&effective.threshold) {
            anyhow::bail!("ner.threshold must be finite and between 0 and 1");
        }
        let active_models = match effective.backend {
            NerBackend::TokenClass => vec![effective.token_model.as_deref()],
            NerBackend::Gliner => vec![Some(effective.model.as_str())],
            NerBackend::Both => vec![
                Some(effective.model.as_str()),
                effective.token_model.as_deref(),
            ],
        };
        for model in active_models.into_iter().flatten() {
            if model.trim().is_empty() || model.chars().any(char::is_control) {
                anyhow::bail!(
                    "NER model identifier must be nonempty and contain no control characters"
                );
            }
        }
        if let Some(ref labels) = effective.labels {
            if !matches!(effective.backend, NerBackend::Gliner) {
                anyhow::bail!(
                    "ner.labels is GLiNER-only; omit it for backend=tokens or backend=both (token model classes are not filtered)"
                );
            }
            effective.labels = Some(normalize_gliner_labels(labels)?);
        }
        Ok(effective)
    }

    /// Share effective settings without input text, private paths, cache
    /// locations, or arbitrary model/repository identifiers. Resolve CLI
    /// overrides first, and report runtime readiness separately.
    pub fn safe_status(&self) -> Result<NerStatus> {
        let effective = self.resolved(None)?;
        let mut models = Vec::new();
        if matches!(effective.backend, NerBackend::TokenClass | NerBackend::Both) {
            models.push(NerModelStatus {
                backend: NerBackend::TokenClass,
                identifier: safe_model_identifier(
                    effective
                        .token_model
                        .as_deref()
                        .unwrap_or("Wismut/nym-pii-multilingual-small/int8"),
                ),
                revision: None,
                label_scope: "all-model-classes",
                labels: None,
                decoding: if effective.recall_first {
                    "recall-first"
                } else {
                    "argmax"
                },
                provider: Some(effective.provider),
            });
        }
        if matches!(effective.backend, NerBackend::Gliner | NerBackend::Both) {
            models.push(NerModelStatus {
                backend: NerBackend::Gliner,
                identifier: safe_model_identifier(&effective.model),
                revision: None,
                label_scope: "gliner-only",
                labels: Some(effective.effective_gliner_labels()),
                decoding: "span",
                provider: None,
            });
        }
        Ok(NerStatus {
            configured_enabled: effective.enabled,
            backend: effective.backend,
            threshold: effective.threshold,
            regex_scope: "independent",
            models,
        })
    }

    /// Effective GLiNER labels after resolution. Does not filter token or regex
    /// findings; callers must only send explicit labels to the GLiNER backend.
    pub fn effective_gliner_labels(&self) -> Vec<String> {
        self.labels.clone().unwrap_or_else(|| {
            DEFAULT_GLINER_LABELS
                .iter()
                .map(|label| (*label).to_string())
                .collect()
        })
    }
}

/// NER mode for rulesets.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum NerMode {
    /// Automatically decide based on enabled patterns.
    /// Uses NER when patterns require it (e.g., person, organization).
    #[default]
    Auto,
    /// Always use NER (if available).
    On,
    /// Never use NER.
    Off,
}

impl NerMode {
    /// Check if NER should be enabled for the given patterns.
    pub fn should_enable_ner(self, patterns: &[String]) -> bool {
        match self {
            NerMode::On => true,
            NerMode::Off => false,
            NerMode::Auto => {
                // NER is needed for patterns that can't be detected by regex alone
                const NER_REQUIRED_PATTERNS: &[&str] = &[
                    "person",
                    "first_name",
                    "last_name",
                    "organization",
                    "street_address",
                    "city",
                    "state",
                    "country",
                    "location",
                ];
                patterns
                    .iter()
                    .any(|p| NER_REQUIRED_PATTERNS.contains(&p.as_str()))
            }
        }
    }
}

/// A ruleset is a named collection of detection and replacement settings.
/// Use rulesets for common scenarios like "programming", "gdpr", "hipaa".
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RulesetConfig {
    /// Human-readable description
    #[serde(default)]
    pub description: String,
    /// Patterns to enable (empty = use detection.enabled_patterns)
    #[serde(default)]
    pub enabled_patterns: Vec<String>,
    /// Patterns to disable
    #[serde(default)]
    pub disabled_patterns: Vec<String>,
    /// Minimum confidence level
    #[serde(default)]
    pub min_confidence: Option<ConfidenceConfig>,
    /// NER mode: auto, on, off
    #[serde(default)]
    pub ner: NerMode,
    /// Replacement strategy override
    #[serde(default)]
    pub strategy: Option<StrategyConfig>,
    /// Per-pattern replacement strategy overrides
    #[serde(default)]
    pub pattern_strategies: HashMap<String, StrategyConfig>,
}

impl Default for RulesetConfig {
    fn default() -> Self {
        Self {
            description: String::new(),
            enabled_patterns: Vec::new(),
            disabled_patterns: Vec::new(),
            min_confidence: None,
            ner: NerMode::Auto,
            strategy: None,
            pattern_strategies: HashMap::new(),
        }
    }
}

/// Per-pattern configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct PatternConfig {
    /// Whether this pattern is enabled by default
    #[serde(default)]
    pub enabled: Option<bool>,
    /// Replacement strategy for this pattern
    #[serde(default)]
    pub strategy: Option<StrategyConfig>,
    /// Custom replacement template (for placeholder strategy)
    #[serde(default)]
    pub template: Option<String>,
    /// Minimum confidence override
    #[serde(default)]
    pub min_confidence: Option<ConfidenceConfig>,
}

/// Built-in rulesets.
impl Config {
    /// Get a specific ruleset by name.
    pub fn get_ruleset(&self, name: &str) -> Option<RulesetConfig> {
        self.rulesets
            .get(name)
            .cloned()
            .or_else(|| Self::builtin_rulesets().get(name).cloned())
    }

    /// Built-in rulesets for common use cases.
    pub fn builtin_rulesets() -> HashMap<String, RulesetConfig> {
        let mut rulesets = HashMap::new();

        // Programming: API keys, tokens, secrets - no NER needed
        rulesets.insert(
            "programming".to_string(),
            RulesetConfig {
                description: "API keys, tokens, and secrets for code anonymization".to_string(),
                enabled_patterns: vec![
                    "api_key".to_string(),
                    "aws_key".to_string(),
                    "jwt".to_string(),
                    "uuid".to_string(),
                    "ipv4".to_string(),
                    "ipv6".to_string(),
                    "mac".to_string(),
                ],
                disabled_patterns: vec![],
                min_confidence: Some(ConfidenceConfig::Medium),
                ner: NerMode::Off,
                strategy: Some(StrategyConfig::Placeholder),
                pattern_strategies: HashMap::new(),
            },
        );

        // Identity: Names and personal identifiers - NER recommended
        rulesets.insert(
            "identity".to_string(),
            RulesetConfig {
                description: "Names and personal identifiers".to_string(),
                enabled_patterns: vec![
                    "person".to_string(),
                    "email".to_string(),
                    "phone_us".to_string(),
                    "phone_intl".to_string(),
                    "ssn".to_string(),
                    "ssn_nodash".to_string(),
                    "passport_us".to_string(),
                    "drivers_license".to_string(),
                    "date".to_string(),
                ],
                disabled_patterns: vec![],
                min_confidence: Some(ConfidenceConfig::Medium),
                ner: NerMode::On,
                strategy: Some(StrategyConfig::Consistent),
                pattern_strategies: HashMap::new(),
            },
        );

        // Contact: Contact information - NER for addresses
        rulesets.insert(
            "contact".to_string(),
            RulesetConfig {
                description: "Contact information: email, phone, address".to_string(),
                enabled_patterns: vec![
                    "email".to_string(),
                    "phone_us".to_string(),
                    "phone_intl".to_string(),
                    "street_address".to_string(),
                    "city".to_string(),
                    "state".to_string(),
                    "country".to_string(),
                    "zip_code".to_string(),
                    "social_handle".to_string(),
                    "twitter_handle".to_string(),
                    "social_url".to_string(),
                ],
                disabled_patterns: vec![],
                min_confidence: Some(ConfidenceConfig::Medium),
                ner: NerMode::Auto,
                strategy: Some(StrategyConfig::Consistent),
                pattern_strategies: HashMap::new(),
            },
        );

        // Financial: Credit cards, bank accounts
        rulesets.insert(
            "financial".to_string(),
            RulesetConfig {
                description: "Financial information: credit cards, bank accounts".to_string(),
                enabled_patterns: vec![
                    "credit_card".to_string(),
                    "credit_card_nodash".to_string(),
                    "iban".to_string(),
                ],
                disabled_patterns: vec![],
                min_confidence: Some(ConfidenceConfig::High),
                ner: NerMode::Off,
                strategy: Some(StrategyConfig::Mask),
                pattern_strategies: HashMap::new(),
            },
        );

        // GDPR: European privacy regulation compliance
        rulesets.insert(
            "gdpr".to_string(),
            RulesetConfig {
                description: "GDPR compliance: all personal data".to_string(),
                enabled_patterns: vec![
                    "person".to_string(),
                    "email".to_string(),
                    "phone_us".to_string(),
                    "phone_intl".to_string(),
                    "street_address".to_string(),
                    "city".to_string(),
                    "country".to_string(),
                    "ipv4".to_string(),
                    "ipv6".to_string(),
                    "date".to_string(),
                    "iban".to_string(),
                    "uk_nino".to_string(),
                    "fr_nir".to_string(),
                    "it_cf".to_string(),
                    "es_dni".to_string(),
                    "eu_id".to_string(),
                    "geocoord".to_string(),
                ],
                disabled_patterns: vec![],
                min_confidence: Some(ConfidenceConfig::Medium),
                ner: NerMode::On,
                strategy: Some(StrategyConfig::Consistent),
                pattern_strategies: HashMap::new(),
            },
        );

        // HIPAA: US healthcare privacy compliance
        rulesets.insert(
            "hipaa".to_string(),
            RulesetConfig {
                description: "HIPAA compliance: protected health information".to_string(),
                enabled_patterns: vec![
                    "person".to_string(),
                    "email".to_string(),
                    "phone_us".to_string(),
                    "phone_intl".to_string(),
                    "street_address".to_string(),
                    "city".to_string(),
                    "state".to_string(),
                    "zip_code".to_string(),
                    "ssn".to_string(),
                    "ssn_nodash".to_string(),
                    "date".to_string(),
                    "ipv4".to_string(),
                    "ipv6".to_string(),
                    "mac".to_string(),
                ],
                disabled_patterns: vec![],
                min_confidence: Some(ConfidenceConfig::Medium),
                ner: NerMode::On,
                strategy: Some(StrategyConfig::Consistent),
                pattern_strategies: HashMap::new(),
            },
        );

        // All: Everything detected - useful for maximum coverage
        rulesets.insert(
            "all".to_string(),
            RulesetConfig {
                description: "All patterns enabled for maximum coverage".to_string(),
                enabled_patterns: vec![], // Empty means all
                disabled_patterns: vec![],
                min_confidence: Some(ConfidenceConfig::Low),
                ner: NerMode::On,
                strategy: None, // Use default
                pattern_strategies: HashMap::new(),
            },
        );

        // Minimal: High-confidence patterns only - lowest false positives
        rulesets.insert(
            "minimal".to_string(),
            RulesetConfig {
                description: "High-confidence patterns only, minimal false positives".to_string(),
                enabled_patterns: vec![
                    "email".to_string(),
                    "phone_us".to_string(),
                    "ssn".to_string(),
                    "credit_card".to_string(),
                    "ipv4".to_string(),
                    "uuid".to_string(),
                ],
                disabled_patterns: vec![],
                min_confidence: Some(ConfidenceConfig::High),
                ner: NerMode::Off,
                strategy: None,
                pattern_strategies: HashMap::new(),
            },
        );

        rulesets
    }
}

impl Config {
    /// Load configuration from all sources.
    pub fn load() -> Result<Self> {
        // Start with defaults
        let defaults = Config::default();
        let defaults_str = toml::to_string(&defaults).context("Failed to serialize defaults")?;

        let mut builder = config::Config::builder().add_source(config::File::from_str(
            &defaults_str,
            config::FileFormat::Toml,
        ));

        // Add global config file if it exists
        if let Some(global_path) = global_config_path()
            && global_path.exists()
        {
            builder = builder.add_source(config::File::from(global_path).required(false));
        }

        // Add local config file if it exists
        let local_path = PathBuf::from("config.toml");
        if local_path.exists() {
            builder = builder.add_source(config::File::from(local_path).required(false));
        }

        // Add environment variables with NYM_ prefix
        // e.g., NYM_REPLACEMENT_EMAIL_DOMAIN=example.com (single underscore separator)
        builder = builder.add_source(
            config::Environment::with_prefix("NYM")
                .prefix_separator("_")
                .separator("_")
                .try_parsing(true),
        );

        // Parser/source errors can include paths, environment values and raw
        // TOML. Return a value-free error, never silently fall back to defaults.
        let settings = builder
            .build()
            .map_err(|_| anyhow::anyhow!("Failed to build configuration"))?;

        settings
            .try_deserialize()
            .map_err(|_| anyhow::anyhow!("Failed to parse configuration"))
    }
}

/// Get the global config file path.
pub fn global_config_path() -> Option<PathBuf> {
    // Try XDG_CONFIG_HOME first
    if let Ok(xdg_config) = std::env::var("XDG_CONFIG_HOME") {
        let path = PathBuf::from(xdg_config).join("nym").join("config.toml");
        return Some(path);
    }

    // Fall back to dirs crate
    dirs::config_dir().map(|p| p.join("nym").join("config.toml"))
}

/// Get the data directory path.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "Public API - used by consumers")
)]
pub fn data_dir() -> PathBuf {
    // Try XDG_DATA_HOME first
    if let Ok(xdg_data) = std::env::var("XDG_DATA_HOME") {
        return PathBuf::from(xdg_data).join("nym");
    }

    // Fall back to dirs crate or default
    dirs::data_dir().map_or_else(
        || {
            dirs::home_dir().map_or_else(
                || PathBuf::from(".nym"),
                |h| h.join(".local").join("share").join("nym"),
            )
        },
        |p| p.join("nym"),
    )
}

/// Get the state directory path.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "Public API - used by consumers")
)]
pub fn state_dir() -> PathBuf {
    // Try XDG_STATE_HOME first
    if let Ok(xdg_state) = std::env::var("XDG_STATE_HOME") {
        return PathBuf::from(xdg_state).join("nym");
    }

    // Fall back to dirs crate or default
    dirs::state_dir().map_or_else(
        || {
            dirs::home_dir().map_or_else(
                || PathBuf::from(".nym"),
                |h| h.join(".local").join("state").join("nym"),
            )
        },
        |p| p.join("nym"),
    )
}

/// Expand environment variables and ~ in a path string.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "Public API - used by consumers")
)]
pub fn expand_path(path: &str) -> PathBuf {
    let expanded = shellexpand::full(path).unwrap_or_else(|_| path.into());
    PathBuf::from(expanded.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ner_model_override_targets_selected_backend() {
        let configured = NerConfig {
            token_model: Some("configured/token".into()),
            ..NerConfig::default()
        };
        let tokens = configured.resolved(Some("override/token")).unwrap();
        assert_eq!(
            (tokens.model.as_str(), tokens.token_model.as_deref()),
            ("onnx-community/gliner_multi-v2.1", Some("override/token"))
        );
        let gliner = NerConfig {
            backend: NerBackend::Gliner,
            ..configured.clone()
        }
        .resolved(Some("override/gliner"))
        .unwrap();
        assert_eq!(
            (gliner.model.as_str(), gliner.token_model.as_deref()),
            ("override/gliner", Some("configured/token"))
        );
        assert_eq!(configured.token_model.as_deref(), Some("configured/token"));
    }

    #[test]
    fn ner_rejects_ambiguous_overrides_and_invalid_active_settings() {
        let both = NerConfig {
            backend: NerBackend::Both,
            token_model: Some("configured/token".into()),
            model: "configured/gliner".into(),
            ..NerConfig::default()
        };
        assert!(
            both.resolved(Some("override/model"))
                .unwrap_err()
                .to_string()
                .contains("ambiguous")
        );
        let effective = both.resolved(None).unwrap();
        assert_eq!(
            (effective.model.as_str(), effective.token_model.as_deref()),
            ("configured/gliner", Some("configured/token"))
        );
        for backend in [NerBackend::TokenClass, NerBackend::Gliner] {
            let ner = NerConfig {
                backend,
                ..NerConfig::default()
            };
            for invalid in ["", "   ", "secret\nidentifier"] {
                let error = ner.resolved(Some(invalid)).unwrap_err().to_string();
                assert!(error.contains("model"));
                assert!(!error.contains("secret"));
            }
        }
        for threshold in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
            assert!(
                NerConfig {
                    threshold,
                    ..NerConfig::default()
                }
                .resolved(None)
                .is_err()
            );
        }
    }

    #[test]
    fn ner_labels_are_explicit_gliner_only_restrictions() {
        for backend in [NerBackend::TokenClass, NerBackend::Both] {
            assert!(
                NerConfig {
                    backend,
                    ..NerConfig::default()
                }
                .resolved(None)
                .is_ok()
            );
            for labels in [vec![], vec!["person".to_string()]] {
                let error = NerConfig {
                    backend,
                    labels: Some(labels),
                    ..NerConfig::default()
                }
                .resolved(None)
                .unwrap_err()
                .to_string();
                assert!(error.contains("GLiNER-only"));
            }
        }
        for labels in [
            vec![],
            vec!["email".to_string()],
            vec!["private_unknown_label".to_string()],
        ] {
            let error = NerConfig {
                backend: NerBackend::Gliner,
                labels: Some(labels),
                ..NerConfig::default()
            }
            .resolved(None)
            .unwrap_err()
            .to_string();
            assert!(!error.contains("private_unknown_label"));
        }
        let ner = NerConfig {
            backend: NerBackend::Gliner,
            labels: Some(vec![
                "PERSON".into(),
                "GIVEN_NAME".into(),
                "surname".into(),
                "person".into(),
                " Country ".into(),
            ]),
            ..NerConfig::default()
        }
        .resolved(None)
        .unwrap();
        assert_eq!(
            ner.labels,
            Some(vec![
                "person".into(),
                "first_name".into(),
                "last_name".into(),
                "country".into()
            ])
        );
    }

    #[test]
    fn ner_omitted_labels_survive_layered_defaults_but_explicit_empty_does_not() {
        let defaults = toml::to_string(&Config::default()).unwrap();
        assert!(!defaults.contains("labels"));
        for backend in ["tokens", "both", "gliner"] {
            let settings = config::Config::builder()
                .add_source(config::File::from_str(&defaults, config::FileFormat::Toml))
                .add_source(config::File::from_str(
                    &format!("[ner]\nbackend = '{backend}'\n"),
                    config::FileFormat::Toml,
                ))
                .build()
                .unwrap()
                .try_deserialize::<Config>()
                .unwrap();
            assert!(settings.ner.resolved(None).is_ok());
            let explicit: Config =
                toml::from_str(&format!("[ner]\nbackend = '{backend}'\nlabels = []\n")).unwrap();
            assert!(explicit.ner.resolved(None).is_err());
        }
    }

    #[test]
    fn ner_status_reports_effective_settings_without_private_identifiers() {
        let status = NerConfig {
            enabled: true,
            backend: NerBackend::Both,
            ..NerConfig::default()
        }
        .safe_status()
        .unwrap();
        assert_eq!(
            serde_json::to_value(status).unwrap(),
            serde_json::json!({
                "configured_enabled": true,
                "backend": "both",
                "threshold": 0.5,
                "regex_scope": "independent",
                "models": [
                    {"backend": "tokens", "identifier": "Wismut/nym-pii-multilingual-small/int8", "revision": null,
                     "label_scope": "all-model-classes", "labels": null, "decoding": "argmax", "provider": "auto"},
                    {"backend": "gliner", "identifier": "onnx-community/gliner_multi-v2.1", "revision": null,
                     "label_scope": "gliner-only", "labels": ["person", "organization", "street_address", "city", "country"],
                     "decoding": "span", "provider": null}
                ]
            })
        );
        for private_model in [
            "/private/client/model",
            "~/private/client/model",
            "internal-tenant/private-model",
            "C:\\private\\client\\model",
        ] {
            let private = NerConfig {
                model: private_model.to_string(),
                token_model: Some(private_model.to_string()),
                cache_dir: Some("/private/cache-secret".into()),
                backend: NerBackend::Both,
                ..NerConfig::default()
            };
            let json = serde_json::to_string(&private.safe_status().unwrap()).unwrap();
            assert!(
                !json.contains("private") && !json.contains("client") && !json.contains("secret")
            );
            assert!(json.contains("[custom]"));
        }
        let tokens = NerConfig {
            recall_first: true,
            provider: NerProvider::Cpu,
            ..NerConfig::default()
        }
        .resolved(Some("nationaldesignstudio/rampart"))
        .unwrap();
        assert_eq!(
            serde_json::to_value(tokens.safe_status().unwrap()).unwrap(),
            serde_json::json!({
                "configured_enabled": false,
                "backend": "tokens",
                "threshold": 0.5,
                "regex_scope": "independent",
                "models": [{"backend": "tokens", "identifier": "nationaldesignstudio/rampart", "revision": null,
                    "label_scope": "all-model-classes", "labels": null, "decoding": "recall-first", "provider": "cpu"}]
            })
        );
    }

    #[test]
    fn ner_example_and_schema_agree_with_backend_scoped_defaults() {
        let example: Config = toml::from_str(include_str!("../examples/config.toml")).unwrap();
        assert!(example.ner.resolved(None).is_ok());
        assert!(example.ner.labels.is_none());
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../examples/config.schema.json")).unwrap();
        let ner = &schema["properties"]["ner"];
        assert_eq!(ner["properties"]["backend"]["default"], "tokens");
        assert_eq!(ner["properties"]["recall_first"]["default"], false);
        assert_eq!(ner["properties"]["token_model"]["type"], "string");
        assert_eq!(ner["properties"]["labels"]["minItems"], 1);
        assert_eq!(ner["then"]["properties"]["backend"]["const"], "gliner");
        let labels = ner["properties"]["labels"]["items"]["enum"]
            .as_array()
            .unwrap();
        for label in labels {
            assert!(normalize_gliner_labels(&[label.as_str().unwrap().to_string()]).is_ok());
        }
        // A schema-inserted default would become an explicit restriction and
        // conflict with the default token backend; document GL defaults instead.
        assert!(ner["properties"]["labels"].get("default").is_none());
    }

    #[test]
    fn trace_defaults_and_literal_loading_are_deterministic_and_value_free() {
        let config = Config::default();
        assert!(config.trace_policy.profile.is_none());
        assert!(config.trace_policy.case_sensitive);
        assert_eq!(
            config.trace_policy.term_boundary,
            crate::engine::TermBoundary::Word
        );
        let decoded: Config = toml::from_str(&toml::to_string(&config).unwrap()).unwrap();
        assert!(decoded.trace_policy.case_sensitive);
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("first.txt");
        let second = dir.path().join("second.txt");
        std::fs::write(&first, "alpha\n\r\n literal spaces \r\n東京\n").unwrap();
        std::fs::write(&second, "東京\nalpha\nβeta\n").unwrap();
        let paths = [
            first.to_string_lossy().into_owned(),
            second.to_string_lossy().into_owned(),
        ];
        assert_eq!(
            load_term_files(&paths).unwrap(),
            vec!["alpha", " literal spaces ", "東京", "βeta"]
        );
        let error = load_term_files(&["${NYM_UNSET_LITERAL_PATH_FOR_TEST}".into()]).unwrap_err();
        assert!(!format!("{error:?}").contains("NYM_UNSET_LITERAL_PATH_FOR_TEST"));
        let debug = format!("{:?}", config.trace_policy);
        assert!(debug.contains("sensitive_file_count"));
    }

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.replacement.email_domain, "example.com");
        assert!(matches!(
            config.detection.min_confidence,
            ConfidenceConfig::High
        ));
    }

    #[test]
    fn test_expand_path_tilde() {
        let expanded = expand_path("~/test");
        assert!(!expanded.to_string_lossy().contains('~'));
    }

    #[test]
    fn test_global_config_path() {
        // Should return Some path
        let path = global_config_path();
        assert!(path.is_some());
        assert!(path.unwrap().ends_with("config.toml"));
    }
}
