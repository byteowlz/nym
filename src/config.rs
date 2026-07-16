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
    /// Named rulesets for common use cases
    #[serde(default)]
    pub rulesets: HashMap<String, RulesetConfig>,
    /// Per-pattern configuration overrides
    #[serde(default)]
    pub patterns: HashMap<String, PatternConfig>,
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
    /// Entity labels to detect.
    /// Default: `person`, `organization`, `street_address`, `city`, `country`
    pub labels: Vec<String>,
    /// Recall-first decoding for the token backend: flag a token when its total
    /// entity probability (1 - P(O)) clears `threshold`, instead of requiring a
    /// single entity class to win the argmax. Raises recall (measured +5-9
    /// char-recall on OOD text) at moderate precision cost -- for redaction, a
    /// miss is a leak while an over-flag only over-redacts. Pair with a lower
    /// `threshold` (e.g. 0.2) for maximum-recall operation.
    #[serde(default)]
    pub recall_first: bool,
    /// Which NER backend(s) to run: `both` (default), `gliner`, or `tokens`.
    pub backend: NerBackend,
    /// Token-classification model for the `tokens`/`both` backends: a local dir
    /// (`model.onnx` + `tokenizer.json` + `config.json`) or a HuggingFace repo id
    /// (e.g. `Wismut/openmed-onnx/small`, `nationaldesignstudio/rampart`).
    /// Defaults to OpenMed-small when unset. `openmed_model` is a legacy alias.
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
            labels: vec![
                "person".to_string(),
                "organization".to_string(),
                "street_address".to_string(),
                "city".to_string(),
                "country".to_string(),
            ],
            recall_first: false,
            backend: NerBackend::default(),
            token_model: None,
        }
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

        let settings = builder.build().context("Failed to build configuration")?;

        settings
            .try_deserialize()
            .context("Failed to parse configuration")
    }

    /// Load configuration or return defaults on error.
    pub fn load_or_default() -> Self {
        Self::load().unwrap_or_default()
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
