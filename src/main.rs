//! nym - Fast, reversible PII anonymization CLI.
//!
//! A command-line tool for detecting and anonymizing personally identifiable
//! information (PII) in text files with optional reversibility.

#![expect(clippy::print_stdout, reason = "CLI binary communicates via stdout")]
#![expect(
    clippy::print_stderr,
    reason = "CLI binary communicates errors via stderr"
)]

use std::env;
use std::fs;
use std::io::{self, BufRead, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::Shell;
use log::{debug, info};
use serde::{Deserialize, Serialize};

#[cfg(feature = "bench")]
mod bench;
mod config;
mod engine;
mod session;
#[cfg(feature = "streaming")]
mod streaming;
#[cfg(all(feature = "streaming", feature = "ner"))]
mod streaming_ner;

use config::Config;
use engine::{
    BUILTIN_PATTERNS, Confidence, Detector, DetectorConfig, JsonPiiMatch, PiiMatch,
    ReplacementStrategy, Replacer, ReplacerConfig, detect_json, process_json,
};
use session::Session;

const APP_NAME: &str = env!("CARGO_PKG_NAME");

/// Apply the configured NER backend selection (`gliner`/`tokens`/`both`) and
/// the token-classification model onto a detector config.
fn apply_ner_backend(mut dc: DetectorConfig, ner: &config::NerConfig) -> DetectorConfig {
    dc = dc.with_ner_backend(ner.backend);
    if let Some(ref model) = ner.token_model {
        dc = dc.with_ner_token_model(shellexpand::tilde(model).into_owned());
    }
    dc
}

fn main() {
    if let Err(err) = try_main() {
        let _ = writeln!(io::stderr(), "error: {err:?}");
        std::process::exit(1);
    }
}

fn try_main() -> Result<()> {
    let cli = Cli::parse();

    init_logging(&cli.common)?;

    // Load configuration
    let config = load_config(&cli.common)?;

    match cli.command {
        Command::Anon(cmd) => handle_anon(&cli.common, &config, cmd),
        Command::Deanon(cmd) => handle_deanon(&cli.common, cmd),
        Command::Detect(cmd) => handle_detect(&cli.common, &config, cmd),
        Command::Patterns(cmd) => handle_patterns(&cli.common, cmd),
        Command::Config(cmd) => handle_config(&cli.common, &config, cmd),
        Command::Sessions(cmd) => handle_sessions(&cli.common, cmd),
        #[cfg(feature = "ner")]
        Command::Models(cmd) => handle_models(&cli.common, &config, cmd),
        #[cfg(feature = "bench")]
        Command::Bench(cmd) => handle_bench(&cli.common, &config, cmd),
        Command::Completions { shell } => handle_completions(shell),
    }
}

fn load_config(common: &CommonOpts) -> Result<Config> {
    if let Some(ref config_path) = common.config {
        // Load from specific config file
        let content = fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;
        toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", config_path.display()))
    } else {
        // Load from default locations
        Ok(Config::load_or_default())
    }
}

/// Resolved ruleset: (enabled_patterns, disabled_patterns, min_confidence, ner_mode).
type ResolvedRuleset = (
    Vec<String>,
    Vec<String>,
    Option<Confidence>,
    Option<config::NerMode>,
);

/// Resolve ruleset name or quick flags to a list of patterns and NER mode.
#[expect(
    clippy::fn_params_excessive_bools,
    reason = "Quick-select flags from CLI"
)]
fn resolve_ruleset(
    config: &Config,
    ruleset_name: Option<&str>,
    only_names: bool,
    only_keys: bool,
    only_contact: bool,
    only_financial: bool,
) -> ResolvedRuleset {
    use config::NerMode;

    // Quick flags take precedence
    if only_names {
        return (
            vec![
                "person".to_string(),
                "email".to_string(),
                "ssn".to_string(),
                "ssn_nodash".to_string(),
                "passport_us".to_string(),
                "drivers_license".to_string(),
            ],
            vec![],
            Some(Confidence::Medium),
            Some(NerMode::On),
        );
    }

    if only_keys {
        return (
            vec![
                "api_key".to_string(),
                "aws_key".to_string(),
                "jwt".to_string(),
            ],
            vec![],
            Some(Confidence::Medium),
            Some(NerMode::Off),
        );
    }

    if only_contact {
        return (
            vec![
                "email".to_string(),
                "phone_us".to_string(),
                "phone_intl".to_string(),
                "social_handle".to_string(),
                "twitter_handle".to_string(),
                "social_url".to_string(),
            ],
            vec![],
            Some(Confidence::Medium),
            Some(NerMode::Off),
        );
    }

    if only_financial {
        return (
            vec![
                "credit_card".to_string(),
                "credit_card_nodash".to_string(),
                "iban".to_string(),
            ],
            vec![],
            Some(Confidence::High),
            Some(NerMode::Off),
        );
    }

    // Check for named ruleset
    if let Some(name) = ruleset_name
        && let Some(ruleset) = config.get_ruleset(name)
    {
        let confidence = ruleset.min_confidence.map(std::convert::Into::into);
        return (
            ruleset.enabled_patterns,
            ruleset.disabled_patterns,
            confidence,
            Some(ruleset.ner),
        );
    }

    // No ruleset specified, return empty (will use defaults)
    (vec![], vec![], None, None)
}

// =============================================================================
// CLI Definition
// =============================================================================

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Fast, reversible PII anonymization CLI",
    long_about = "nym detects and anonymizes personally identifiable information (PII) in text files.\n\n\
                  It supports multiple formats (text, JSON, TOML, YAML, CSV) and can optionally \
                  store a key file for reversing the anonymization later.",
    propagate_version = true
)]
struct Cli {
    #[command(flatten)]
    common: CommonOpts,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, Args)]
#[expect(clippy::struct_excessive_bools, reason = "CLI flags from clap")]
struct CommonOpts {
    /// Path to config file
    #[arg(short, long, global = true, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Reduce output to only errors
    #[arg(short, long, global = true)]
    quiet: bool,

    /// Increase logging verbosity (stackable: -v, -vv, -vvv)
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    /// Output as JSON
    #[arg(long, global = true, conflicts_with = "yaml")]
    json: bool,

    /// Output as YAML
    #[arg(long, global = true)]
    yaml: bool,

    /// Disable ANSI colors in output
    #[arg(long = "no-color", global = true)]
    no_color: bool,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Anonymize PII in input text
    Anon(AnonCommand),

    /// Restore original PII from a key file
    Deanon(DeanonCommand),

    /// Detect PII without modifying the text
    Detect(DetectCommand),

    /// List and inspect available PII patterns
    Patterns(PatternsCommand),

    /// Show or manage configuration
    Config(ConfigCommand),

    /// Search for sessions by ID
    Sessions(SessionsCommand),

    /// List, download, and select NER models
    #[cfg(feature = "ner")]
    Models(ModelsCommand),

    /// Benchmark PII detection accuracy
    #[cfg(feature = "bench")]
    Bench(BenchCommand),

    /// Generate shell completions
    Completions {
        #[arg(value_enum)]
        shell: Shell,
    },
}

// -----------------------------------------------------------------------------
// Sessions Command
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Args)]
struct SessionsCommand {
    #[command(subcommand)]
    action: Option<SessionsAction>,
}

#[derive(Debug, Clone, Subcommand)]
enum SessionsAction {
    /// List all key files with their session IDs
    List {
        /// Directory to search (default: current directory and common locations)
        #[arg(short, long, value_name = "DIR")]
        directory: Option<PathBuf>,
    },
    /// Search for sessions by ID
    Search {
        /// Session ID to search for
        #[arg(value_name = "ID")]
        id: String,
        /// Directory to search (default: current directory and common locations)
        #[arg(short, long, value_name = "DIR")]
        directory: Option<PathBuf>,
    },
}

// -----------------------------------------------------------------------------
// Models Command
// -----------------------------------------------------------------------------

#[cfg(feature = "ner")]
#[derive(Debug, Clone, Args)]
struct ModelsCommand {
    #[command(subcommand)]
    action: Option<ModelsAction>,
}

#[cfg(feature = "ner")]
#[derive(Debug, Clone, Subcommand)]
enum ModelsAction {
    /// List catalog models, marking downloaded (✓) and default (*)
    List,
    /// Fuzzy-pick a model and download it (exact slug downloads directly)
    Pull {
        /// Model slug (or search query to seed the fuzzy picker)
        #[arg(value_name = "QUERY")]
        query: Option<String>,
    },
    /// Set the default model (fuzzy-pick among downloaded models)
    Use {
        /// Model slug (or search query to seed the fuzzy picker)
        #[arg(value_name = "QUERY")]
        query: Option<String>,
    },
    /// Refresh the model catalog from the byteowlz/nym repository
    Refresh,
}

// -----------------------------------------------------------------------------
// Anon Command
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Args)]
#[expect(clippy::struct_excessive_bools, reason = "CLI flags from clap")]
struct AnonCommand {
    /// Input file (reads from stdin if not specified)
    #[arg(value_name = "INPUT")]
    input: Option<PathBuf>,

    /// Output file (writes to stdout if not specified)
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Store replacement mappings for later reversal
    #[arg(short = 'k', long = "key-file", value_name = "FILE")]
    key_file: Option<PathBuf>,

    /// PDF only: proceed even when some fonts cannot be decoded (their text
    /// cannot be inspected for PII)
    #[arg(long = "no-strict-pdf")]
    no_strict_pdf: bool,

    /// OCR raster images inside PDFs and redact recognized PII (requires an
    /// external OCR engine; see docs/document-redaction.md)
    #[arg(long)]
    ocr: bool,

    /// Replacement strategy
    #[arg(long, value_enum, default_value_t = StrategyArg::Fake)]
    strategy: StrategyArg,

    /// Input format (auto-detected from extension if not specified)
    #[arg(short = 'f', long, value_enum)]
    format: Option<FormatArg>,

    /// Use a predefined ruleset (programming, identity, contact, financial, gdpr, hipaa, all, minimal)
    #[arg(short = 'r', long, value_name = "NAME")]
    ruleset: Option<String>,

    /// Patterns to use (comma-separated, default: all high-confidence)
    #[arg(long, value_delimiter = ',')]
    patterns: Option<Vec<String>>,

    /// Patterns to exclude (comma-separated)
    #[arg(long, value_delimiter = ',')]
    exclude: Option<Vec<String>>,

    // Quick pattern selection shortcuts
    /// Only anonymize names and personal identifiers
    #[arg(long, conflicts_with = "ruleset")]
    only_names: bool,

    /// Only anonymize API keys, tokens, and secrets
    #[arg(long, conflicts_with = "ruleset")]
    only_keys: bool,

    /// Only anonymize contact information (email, phone)
    #[arg(long, conflicts_with = "ruleset")]
    only_contact: bool,

    /// Only anonymize financial data (credit cards, IBAN)
    #[arg(long, conflicts_with = "ruleset")]
    only_financial: bool,

    /// Minimum confidence level
    #[arg(long, value_enum, default_value_t = ConfidenceArg::High)]
    min_confidence: ConfidenceArg,

    /// Seed for deterministic replacements
    #[arg(long)]
    seed: Option<u64>,

    /// Session tag for tracking (auto-generated if not specified)
    #[arg(long)]
    tag: Option<String>,

    /// Enable NER-based detection for names and addresses (requires 'ner' feature)
    #[arg(long)]
    ner: bool,

    /// Disable NER-based detection (overrides config)
    #[arg(long, conflicts_with = "ner")]
    no_ner: bool,

    /// NER model repository (default: onnx-community/gliner_multi-v2.1)
    #[arg(long, value_name = "REPO")]
    ner_model: Option<String>,

    /// NER confidence threshold (0.0-1.0, default: 0.5)
    #[arg(long, value_name = "THRESHOLD")]
    ner_threshold: Option<f32>,

    /// Stream mode: process stdin line-by-line with immediate output
    /// (requires 'streaming' feature)
    #[arg(long)]
    stream: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum FormatArg {
    /// Plain text (default)
    Text,
    /// JSON - preserves structure, only anonymizes string values
    Json,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum StrategyArg {
    Placeholder,
    Mask,
    Hash,
    Random,
    Consistent,
    #[default]
    Fake,
}

impl From<StrategyArg> for ReplacementStrategy {
    fn from(arg: StrategyArg) -> Self {
        match arg {
            StrategyArg::Placeholder => ReplacementStrategy::Placeholder,
            StrategyArg::Mask => ReplacementStrategy::Mask,
            StrategyArg::Hash => ReplacementStrategy::Hash,
            StrategyArg::Random => ReplacementStrategy::Random,
            StrategyArg::Consistent => ReplacementStrategy::Consistent,
            StrategyArg::Fake => ReplacementStrategy::Fake,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum ConfidenceArg {
    High,
    Medium,
    Low,
}

impl From<ConfidenceArg> for Confidence {
    fn from(arg: ConfidenceArg) -> Self {
        match arg {
            ConfidenceArg::High => Confidence::High,
            ConfidenceArg::Medium => Confidence::Medium,
            ConfidenceArg::Low => Confidence::Low,
        }
    }
}

// -----------------------------------------------------------------------------
// Deanon Command
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Args)]
struct DeanonCommand {
    /// Input file containing anonymized text (reads from stdin if not specified)
    #[arg(value_name = "INPUT")]
    input: Option<PathBuf>,

    /// Output file (writes to stdout if not specified)
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Key file containing replacement mappings (required)
    #[arg(short = 'k', long = "key-file", value_name = "FILE", required = true)]
    key_file: PathBuf,
}

// -----------------------------------------------------------------------------
// Detect Command
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Args)]
#[expect(clippy::struct_excessive_bools, reason = "CLI flags from clap")]
struct DetectCommand {
    /// Input file (reads from stdin if not specified)
    #[arg(value_name = "INPUT")]
    input: Option<PathBuf>,

    /// OCR raster images inside PDFs and scan the recognized text (requires
    /// an external OCR engine; see docs/document-redaction.md)
    #[arg(long)]
    ocr: bool,

    /// Input format (auto-detected from extension if not specified)
    #[arg(short = 'f', long, value_enum)]
    format: Option<FormatArg>,

    /// Use a predefined ruleset (programming, identity, contact, financial, gdpr, hipaa, all, minimal)
    #[arg(short = 'r', long, value_name = "NAME")]
    ruleset: Option<String>,

    /// Patterns to use (comma-separated, default: all high-confidence)
    #[arg(long, value_delimiter = ',')]
    patterns: Option<Vec<String>>,

    /// Patterns to exclude (comma-separated)
    #[arg(long, value_delimiter = ',')]
    exclude: Option<Vec<String>>,

    // Quick pattern selection shortcuts
    /// Only detect names and personal identifiers
    #[arg(long, conflicts_with = "ruleset")]
    only_names: bool,

    /// Only detect API keys, tokens, and secrets
    #[arg(long, conflicts_with = "ruleset")]
    only_keys: bool,

    /// Only detect contact information (email, phone)
    #[arg(long, conflicts_with = "ruleset")]
    only_contact: bool,

    /// Only detect financial data (credit cards, IBAN)
    #[arg(long, conflicts_with = "ruleset")]
    only_financial: bool,

    /// Minimum confidence level
    #[arg(long, value_enum, default_value_t = ConfidenceArg::High)]
    min_confidence: ConfidenceArg,

    /// Show only summary counts
    #[arg(long)]
    summary: bool,

    /// Enable NER-based detection for names and addresses (requires 'ner' feature)
    #[arg(long)]
    ner: bool,

    /// Disable NER-based detection (overrides config)
    #[arg(long, conflicts_with = "ner")]
    no_ner: bool,

    /// NER model repository (default: onnx-community/gliner_multi-v2.1)
    #[arg(long, value_name = "REPO")]
    ner_model: Option<String>,

    /// NER confidence threshold (0.0-1.0, default: 0.5)
    #[arg(long, value_name = "THRESHOLD")]
    ner_threshold: Option<f32>,

    /// Stream mode: process stdin line-by-line with immediate output
    /// (requires 'streaming' feature)
    #[arg(long)]
    stream: bool,
}

// -----------------------------------------------------------------------------
// Patterns Command
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Args)]
struct PatternsCommand {
    #[command(subcommand)]
    action: Option<PatternsAction>,
}

#[derive(Debug, Clone, Subcommand)]
enum PatternsAction {
    /// List all available patterns
    List,
    /// Show details for a specific pattern
    Show {
        /// Pattern name
        name: String,
    },
    /// Test a pattern against sample text
    Test {
        /// Pattern name
        name: String,
        /// Text to test against
        text: String,
    },
    /// List available rulesets
    Rulesets,
    /// Show details for a specific ruleset
    Ruleset {
        /// Ruleset name
        name: String,
    },
}

// -----------------------------------------------------------------------------
// Config Command
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Args)]
struct ConfigCommand {
    #[command(subcommand)]
    action: Option<ConfigAction>,
}

#[derive(Debug, Clone, Subcommand)]
enum ConfigAction {
    /// Show current configuration
    Show,
    /// Show config file path
    Path,
    /// Initialize config file with defaults
    Init {
        /// Force overwrite existing config
        #[arg(short, long)]
        force: bool,
    },
}

// -----------------------------------------------------------------------------
// Bench Command
// -----------------------------------------------------------------------------

#[cfg(feature = "bench")]
#[derive(Debug, Clone, Args)]
struct BenchCommand {
    /// Dataset source: HuggingFace dataset name or path to JSONL file
    #[arg(value_name = "SOURCE")]
    source: String,

    /// Dataset split for HuggingFace datasets (train, validation, test)
    #[arg(long, default_value = "train")]
    split: String,

    /// Number of examples to process (e.g., 100, 1000)
    #[arg(long, short = 'n')]
    num_examples: Option<usize>,

    /// Percentage of dataset to sample (0.0-100.0)
    #[arg(long, short = 'p', value_name = "PERCENT")]
    sample: Option<f64>,

    /// Use strict matching (exact span) instead of overlap
    #[arg(long)]
    strict: bool,

    /// Ignore entity labels when matching (count any span overlap as a hit).
    /// Fair when comparing NER backends with different label taxonomies.
    #[arg(long)]
    ignore_labels: bool,

    /// Enable NER-based detection
    #[arg(long)]
    ner: bool,

    /// Minimum confidence level
    #[arg(long, value_enum, default_value_t = ConfidenceArg::Medium)]
    min_confidence: ConfidenceArg,

    /// Cache directory for downloaded datasets
    #[arg(long, value_name = "DIR")]
    cache_dir: Option<PathBuf>,

    /// Number of rows to fetch from HuggingFace (default: 1000)
    #[arg(long, default_value = "1000")]
    fetch_limit: usize,

    /// Show false negatives (missed detections) for a specific label
    #[arg(long, value_name = "LABEL")]
    show_misses: Option<String>,

    /// Show false positives (incorrect detections) for a specific label
    #[arg(long, value_name = "LABEL")]
    show_false_positives: Option<String>,
}

// -----------------------------------------------------------------------------
// Command Handlers
// =============================================================================

fn handle_anon(common: &CommonOpts, config: &Config, cmd: AnonCommand) -> Result<()> {
    // Check for streaming mode
    #[cfg(feature = "streaming")]
    if cmd.stream {
        return handle_anon_streaming(common, config, cmd);
    }

    #[cfg(not(feature = "streaming"))]
    if cmd.stream {
        return Err(anyhow!(
            "Streaming mode requires the 'streaming' feature. \
             Rebuild with: cargo build --features streaming"
        ));
    }

    // Read input. Office documents (docx/xlsx/pptx/odt/...) and PDFs are
    // binary and are redacted in place further down.
    let office_fmt = cmd
        .input
        .as_ref()
        .and_then(|p| engine::office::sniff_path(p));
    let is_pdf = cmd
        .input
        .as_ref()
        .is_some_and(|p| engine::pdf::is_pdf_path(p));
    #[cfg(feature = "ocr")]
    let img_fmt = sniff_image(cmd.input.as_ref());
    #[cfg(not(feature = "ocr"))]
    let img_fmt: Option<()> = None;
    let input_text = if office_fmt.is_some() || is_pdf || img_fmt.is_some() {
        String::new()
    } else {
        read_input(cmd.input.as_ref())?
    };

    // Determine format (explicit or auto-detect from file extension)
    let format = cmd
        .format
        .unwrap_or_else(|| detect_format(cmd.input.as_ref()));

    // Generate session ID
    let source_filename = cmd
        .input
        .as_ref()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str());
    let session = if let Some(ref tag) = cmd.tag {
        Session::with_tag(tag, source_filename)
    } else {
        Session::new(source_filename)
    };

    // Resolve ruleset or quick flags first
    let (ruleset_patterns, ruleset_excluded, ruleset_confidence, ruleset_ner_mode) =
        resolve_ruleset(
            config,
            cmd.ruleset.as_deref(),
            cmd.only_names,
            cmd.only_keys,
            cmd.only_contact,
            cmd.only_financial,
        );

    // Configure detector: CLI args > ruleset > config
    let min_confidence: Confidence = if cmd.min_confidence != ConfidenceArg::High {
        // Explicit CLI override (non-default value)
        cmd.min_confidence.into()
    } else if let Some(c) = ruleset_confidence {
        c
    } else {
        config.detection.min_confidence.into()
    };

    let mut detector_config = DetectorConfig::default().with_min_confidence(min_confidence);

    // Use patterns from CLI, or ruleset, or fall back to config
    if let Some(ref patterns) = cmd.patterns {
        detector_config = detector_config.with_patterns(patterns.iter().cloned());
    } else if !ruleset_patterns.is_empty() {
        detector_config = detector_config.with_patterns(ruleset_patterns.iter().cloned());
    } else if !config.detection.enabled_patterns.is_empty() {
        detector_config =
            detector_config.with_patterns(config.detection.enabled_patterns.iter().cloned());
    }

    // Exclude patterns from CLI, ruleset, and config
    let mut excluded: Vec<String> = config.detection.disabled_patterns.clone();
    excluded.extend(ruleset_excluded);
    if let Some(ref exclude) = cmd.exclude {
        excluded.extend(exclude.iter().cloned());
    }
    if !excluded.is_empty() {
        detector_config = detector_config.without_patterns(excluded);
    }

    // Configure NER: CLI args > ruleset > config
    let ner_enabled = if cmd.no_ner {
        false
    } else if cmd.ner {
        true
    } else if let Some(ner_mode) = ruleset_ner_mode {
        // Get the patterns that will be used
        let active_patterns: Vec<String> = if cmd.patterns.is_some() {
            cmd.patterns.clone().unwrap_or_default()
        } else if !ruleset_patterns.is_empty() {
            ruleset_patterns.clone()
        } else {
            config.detection.enabled_patterns.clone()
        };
        ner_mode.should_enable_ner(&active_patterns)
    } else {
        config.ner.enabled
    };

    detector_config = detector_config.with_ner(ner_enabled);

    if let Some(ref model) = cmd.ner_model {
        detector_config = detector_config.with_ner_model(model);
    } else if !config.ner.model.is_empty() {
        detector_config = detector_config.with_ner_model(&config.ner.model);
    }

    if let Some(threshold) = cmd.ner_threshold {
        detector_config = detector_config.with_ner_threshold(threshold);
    } else if config.ner.threshold > 0.0 {
        detector_config = detector_config.with_ner_threshold(config.ner.threshold);
    }
    detector_config = detector_config.with_ner_recall_first(config.ner.recall_first);

    if !config.ner.labels.is_empty() {
        detector_config = detector_config.with_ner_labels(config.ner.labels.clone());
    }

    if let Some(ref cache_dir) = config.ner.cache_dir {
        detector_config = detector_config.with_ner_cache_dir(cache_dir);
    }

    detector_config = apply_ner_backend(detector_config, &config.ner);

    let detector = Detector::new(&detector_config);
    debug!("Active patterns: {:?}", detector.active_patterns());

    if detector.ner_enabled() {
        debug!("NER detection enabled");
    }

    // Configure replacer (CLI args override config)
    let strategy: ReplacementStrategy = cmd.strategy.into();
    let seed = cmd.seed.or(config.replacement.seed);

    let replacer_config = ReplacerConfig {
        strategy,
        seed,
        email_domain: config.replacement.email_domain.clone(),
    };

    let mut replacer = Replacer::new(replacer_config).with_session_id(session.id.clone());

    // Office documents: in-place redaction of the archive's text nodes.
    if let Some(fmt) = office_fmt {
        let path = cmd
            .input
            .as_ref()
            .ok_or_else(|| anyhow!("office document input requires a file path"))?;
        let bytes = fs::read(path)
            .with_context(|| format!("Failed to read input file: {}", path.display()))?;
        let (out_bytes, replacements) =
            engine::office::anonymize(&bytes, fmt, &detector, &mut replacer)
                .map_err(|e| anyhow!("Failed to redact office document: {e}"))?;

        let out_path = cmd
            .output
            .clone()
            .unwrap_or_else(|| derive_document_output(path, "anon"));
        fs::write(&out_path, &out_bytes)
            .with_context(|| format!("Failed to write output: {}", out_path.display()))?;

        if replacements.is_empty() {
            if !common.quiet {
                eprintln!("No PII detected; unmodified copy written to: {}", out_path.display());
            }
            return Ok(());
        }
        if let Some(ref key_path) = cmd.key_file {
            write_key_file(key_path, &replacements, &session)?;
            if !common.quiet {
                eprintln!("Key file written to: {}", key_path.display());
                eprintln!("Session: {}", session.full_reference());
            }
        } else if !common.quiet {
            eprintln!("Session: {}", session.full_reference());
        }
        if !common.quiet {
            eprintln!(
                "Anonymized {} PII occurrences -> {}",
                replacements.len(),
                out_path.display()
            );
        }
        return Ok(());
    }

    // Raster images: OCR-based redaction — paint, re-encode, re-OCR verify.
    #[cfg(feature = "ocr")]
    if let Some(fmt) = img_fmt {
        let path = cmd
            .input
            .as_ref()
            .ok_or_else(|| anyhow!("image input requires a file path"))?;
        let bytes = fs::read(path)
            .with_context(|| format!("Failed to read input file: {}", path.display()))?;
        let ocr_engine = build_ocr_engine(config)?;
        let out_path = cmd
            .output
            .clone()
            .unwrap_or_else(|| derive_document_output(path, "anon"));
        match engine::ocr::redact_image(&bytes, fmt, &ocr_engine, &detector, &mut replacer)
            .map_err(|e| anyhow!("image redaction failed: {e}"))?
        {
            None => {
                fs::write(&out_path, &bytes)
                    .with_context(|| format!("Failed to write output: {}", out_path.display()))?;
                if !common.quiet {
                    eprintln!("No PII recognized; unmodified copy written to: {}", out_path.display());
                }
            }
            Some(red) => {
                fs::write(&out_path, &red.bytes)
                    .with_context(|| format!("Failed to write output: {}", out_path.display()))?;
                if let Some(ref key_path) = cmd.key_file {
                    write_key_file(key_path, &red.replacements, &session)?;
                    if !common.quiet {
                        eprintln!("Key file written to: {}", key_path.display());
                    }
                }
                if !common.quiet {
                    if red.escalated {
                        eprintln!("Note: verification escalated painting to full text regions.");
                    }
                    eprintln!(
                        "Redacted {} PII occurrences (re-OCR verified) -> {}",
                        red.replacements.len(),
                        out_path.display()
                    );
                }
            }
        }
        return Ok(());
    }

    // PDFs: true redaction — text removed from content streams and verified gone.
    if is_pdf {
        let path = cmd
            .input
            .as_ref()
            .ok_or_else(|| anyhow!("PDF input requires a file path"))?;
        let bytes = fs::read(path)
            .with_context(|| format!("Failed to read input file: {}", path.display()))?;
        #[expect(unused_mut, reason = "mutated only with the ocr feature")]
        let (out_bytes, mut replacements, report) =
            engine::pdf::redact(&bytes, &detector, &mut replacer, !cmd.no_strict_pdf)
                .map_err(|e| anyhow!("PDF redaction failed: {e}"))?;

        // Optional OCR pass over raster images in the (already text-redacted) PDF.
        #[expect(unused_mut, reason = "reassigned only with the ocr feature")]
        let mut out_bytes = out_bytes;
        #[cfg(feature = "ocr")]
        if cmd.ocr || config.ocr.enabled {
            let ocr_engine = build_ocr_engine(config)?;
            let (b, ocr_log, ocr_report) = engine::ocr::redact_pdf_images(
                &out_bytes,
                &ocr_engine,
                &detector,
                &mut replacer,
                !cmd.no_strict_pdf,
            )
            .map_err(|e| anyhow!("PDF image OCR redaction failed: {e}"))?;
            out_bytes = b;
            replacements.extend(ocr_log);
            if !common.quiet {
                eprintln!(
                    "OCR: scanned {} image(s), redacted {}, {} unsupported codec(s).",
                    ocr_report.images_scanned,
                    ocr_report.images_redacted,
                    ocr_report.images_unsupported
                );
            }
        }
        #[cfg(not(feature = "ocr"))]
        if cmd.ocr {
            return Err(anyhow!("--ocr requires nym to be built with the `ocr` feature"));
        }

        let out_path = cmd
            .output
            .clone()
            .unwrap_or_else(|| derive_document_output(path, "anon"));
        fs::write(&out_path, &out_bytes)
            .with_context(|| format!("Failed to write output: {}", out_path.display()))?;

        #[cfg(feature = "ocr")]
        let ocr_active = cmd.ocr || config.ocr.enabled;
        #[cfg(not(feature = "ocr"))]
        let ocr_active = false;
        if !common.quiet {
            if report.images_seen > 0 && !ocr_active {
                eprintln!(
                    "Note: {} image(s) present — pixels are not scanned (pass --ocr).",
                    report.images_seen
                );
            }
            if report.unmapped_text_ops > 0 {
                eprintln!(
                    "Warning: {} text operator(s) had no usable font mapping and were not inspected.",
                    report.unmapped_text_ops
                );
            }
            if report.metadata_scrubbed > 0 {
                eprintln!("Scrubbed {} metadata/annotation field(s).", report.metadata_scrubbed);
            }
        }
        if replacements.is_empty() {
            if !common.quiet {
                eprintln!("No PII detected; document rewritten to: {}", out_path.display());
            }
            return Ok(());
        }
        if let Some(ref key_path) = cmd.key_file {
            write_key_file(key_path, &replacements, &session)?;
            if !common.quiet {
                eprintln!("Key file written to: {}", key_path.display());
                eprintln!(
                    "Note: PDF redaction is destructive — the key file documents what was \
                     removed, but only the original file can fully restore it."
                );
            }
        }
        if !common.quiet {
            eprintln!(
                "Redacted {} PII occurrences (verified absent from output) -> {}",
                replacements.len(),
                out_path.display()
            );
        }
        return Ok(());
    }

    // Process based on format
    let (anonymized, replacements) = match format {
        FormatArg::Json => process_json(&input_text, &detector, &mut replacer)
            .with_context(|| "Failed to parse input as JSON")?,
        FormatArg::Text => {
            let matches = detector.detect(&input_text);
            info!("Found {} PII matches", matches.len());

            if matches.is_empty() {
                // No PII found, output unchanged
                write_output(cmd.output.as_ref(), &input_text)?;
                if !common.quiet {
                    eprintln!("No PII detected in input");
                }
                return Ok(());
            }

            replacer.replace_all(&input_text, &matches)
        }
    };

    if replacements.is_empty() {
        write_output(cmd.output.as_ref(), &input_text)?;
        if !common.quiet {
            eprintln!("No PII detected in input");
        }
        return Ok(());
    }

    // Write key file if requested
    if let Some(ref key_path) = cmd.key_file {
        write_key_file(key_path, &replacements, &session)?;
        if !common.quiet {
            eprintln!("Key file written to: {}", key_path.display());
            eprintln!("Session: {}", session.full_reference());
        }
    } else if !common.quiet {
        eprintln!("Session: {}", session.full_reference());
    }

    // Write output
    write_output(cmd.output.as_ref(), &anonymized)?;

    if !common.quiet {
        eprintln!("Anonymized {} PII occurrences", replacements.len());
    }

    Ok(())
}

/// Handle anonymization in streaming mode.
#[cfg(feature = "streaming")]
#[expect(
    clippy::needless_pass_by_value,
    reason = "CLI command struct consumed by handler"
)]
fn handle_anon_streaming(common: &CommonOpts, config: &Config, cmd: AnonCommand) -> Result<()> {
    use streaming::StreamConfig;
    use tokio::fs::File;
    use tokio::io::BufWriter;

    // Build detector config
    let min_confidence = cmd.min_confidence.into();
    let mut detector_config = DetectorConfig::default().with_min_confidence(min_confidence);

    if let Some(ref patterns) = cmd.patterns {
        detector_config = detector_config.with_patterns(patterns.iter().cloned());
    } else if !config.detection.enabled_patterns.is_empty() {
        detector_config =
            detector_config.with_patterns(config.detection.enabled_patterns.iter().cloned());
    }

    let mut excluded: Vec<String> = config.detection.disabled_patterns.clone();
    if let Some(ref exclude) = cmd.exclude {
        excluded.extend(exclude.iter().cloned());
    }
    if !excluded.is_empty() {
        detector_config = detector_config.without_patterns(excluded);
    }

    // NER config
    let ner_enabled = if cmd.no_ner {
        false
    } else if cmd.ner {
        true
    } else {
        config.ner.enabled
    };
    detector_config = detector_config.with_ner(ner_enabled);

    if let Some(ref model) = cmd.ner_model {
        detector_config = detector_config.with_ner_model(model);
    }
    if let Some(threshold) = cmd.ner_threshold {
        detector_config = detector_config.with_ner_threshold(threshold);
    }
    detector_config = detector_config.with_ner_recall_first(config.ner.recall_first);

    // Build replacer config
    let strategy: ReplacementStrategy = cmd.strategy.into();
    let seed = cmd.seed.or(config.replacement.seed);

    let replacer_config = ReplacerConfig {
        strategy,
        seed,
        email_domain: config.replacement.email_domain.clone(),
    };

    // Generate session
    let session = if let Some(ref tag) = cmd.tag {
        Session::with_tag(tag, None)
    } else {
        Session::new(None)
    };

    let stream_config = StreamConfig {
        detector_config: detector_config.clone(),
        replacer_config,
        session_id: Some(session.id.clone()),
    };

    // Create tokio runtime and run
    let rt = tokio::runtime::Runtime::new().with_context(|| "Failed to create async runtime")?;

    let stats = rt.block_on(async {
        // Open key file if specified
        let key_writer: Option<BufWriter<File>> = if let Some(ref key_path) = cmd.key_file {
            // Write header first (synchronously to avoid complexity)
            let header = session.to_key_file_header();
            std::fs::write(key_path, format!("{header}\n"))
                .with_context(|| format!("Failed to create key file: {}", key_path.display()))?;

            // Open for appending
            let file = tokio::fs::OpenOptions::new()
                .append(true)
                .open(key_path)
                .await
                .with_context(|| format!("Failed to open key file: {}", key_path.display()))?;
            Some(BufWriter::new(file))
        } else {
            None
        };

        // Use smart NER streaming if NER is enabled, otherwise use basic line-by-line
        #[cfg(feature = "ner")]
        if ner_enabled {
            return streaming_ner::stream_anon_ner(stream_config, key_writer)
                .await
                .map_err(|e| anyhow!("Streaming error: {e}"));
        }

        streaming::stream_anon(stream_config, key_writer)
            .await
            .map_err(|e| anyhow!("Streaming error: {e}"))
    })?;

    if !common.quiet {
        if let Some(ref key_path) = cmd.key_file {
            eprintln!("Key file written to: {}", key_path.display());
        }
        eprintln!("Session: {}", session.full_reference());
        eprintln!(
            "Processed {} lines, anonymized {} PII occurrences",
            stats.lines_processed, stats.pii_found
        );
    }

    Ok(())
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "CLI command struct consumed by handler"
)]
fn handle_deanon(common: &CommonOpts, cmd: DeanonCommand) -> Result<()> {
    use std::io::BufRead;

    // PDF and raster-image redaction are destructive by design.
    if cmd.input.as_ref().is_some_and(|p| engine::pdf::is_pdf_path(p)) {
        return Err(anyhow!(
            "PDF redaction is destructive (text is removed from the file); \
             deanonymization is not possible. Keep the original PDF instead."
        ));
    }
    #[cfg(feature = "ocr")]
    if sniff_image(cmd.input.as_ref()).is_some() {
        return Err(anyhow!(
            "image redaction is destructive (pixels are painted over); \
             deanonymization is not possible. Keep the original image instead."
        ));
    }

    // Office documents are restored in place further down.
    let office_fmt = cmd
        .input
        .as_ref()
        .and_then(|p| engine::office::sniff_path(p));

    // Read input
    let mut input_text = if office_fmt.is_some() {
        String::new()
    } else {
        read_input(cmd.input.as_ref())?
    };

    // Read and parse key file
    let key_file = fs::File::open(&cmd.key_file)
        .with_context(|| format!("Failed to open key file: {}", cmd.key_file.display()))?;
    let reader = io::BufReader::new(key_file);

    let mut replacements: Vec<engine::Replacement> = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line_num = line_num + 1;
        let line = line.with_context(|| format!("Failed to read line {line_num} from key file"))?;

        if line.trim().is_empty() {
            continue;
        }

        // Try to parse as replacement entry
        if let Ok(replacement) = serde_json::from_str::<engine::Replacement>(&line) {
            replacements.push(replacement);
        }
        // Skip header lines (version info, etc.)
    }

    if replacements.is_empty() {
        return Err(anyhow!("No replacement mappings found in key file"));
    }

    info!("Loaded {} replacement mappings", replacements.len());

    // Build a list of all replacements including component mappings
    // Sort by replacement length (longest first) to avoid partial replacement issues
    let mut all_mappings: Vec<(&str, &str)> = Vec::new();

    for r in &replacements {
        // Add the full replacement
        all_mappings.push((&r.replacement, &r.original));

        // Add component mappings (e.g., first name, last name)
        for c in &r.components {
            all_mappings.push((&c.replacement, &c.original));
        }
    }

    // Sort by replacement length descending (replace longer strings first)
    // This prevents "Donald" from being replaced before "Donald Duck"
    all_mappings.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    // Deduplicate mappings (same replacement -> same original)
    all_mappings.dedup_by(|a, b| a.0 == b.0);

    debug!(
        "Total mappings (including components): {}",
        all_mappings.len()
    );

    // Office documents: restore text nodes inside the archive.
    if let Some(fmt) = office_fmt {
        let path = cmd
            .input
            .as_ref()
            .ok_or_else(|| anyhow!("office document input requires a file path"))?;
        let bytes = fs::read(path)
            .with_context(|| format!("Failed to read input file: {}", path.display()))?;
        let owned: Vec<(String, String)> = all_mappings
            .iter()
            .map(|(r, o)| ((*r).to_string(), (*o).to_string()))
            .collect();
        let (out_bytes, restored) = engine::office::deanonymize(&bytes, fmt, &owned)
            .map_err(|e| anyhow!("Failed to restore office document: {e}"))?;
        let out_path = cmd
            .output
            .clone()
            .unwrap_or_else(|| derive_document_output(path, "restored"));
        fs::write(&out_path, &out_bytes)
            .with_context(|| format!("Failed to write output: {}", out_path.display()))?;
        if !common.quiet {
            eprintln!("Restored {restored} PII values -> {}", out_path.display());
        }
        return Ok(());
    }

    // Apply replacements in reverse (replace anonymized values with originals)
    let mut restored_count = 0;
    for (replacement, original) in &all_mappings {
        if input_text.contains(*replacement) {
            // Use word-boundary aware replacement for short strings to avoid false positives
            if replacement.len() <= 3 {
                // For very short replacements (initials like "D."), use word boundary matching
                let pattern = format!(r"\b{}\b", regex::escape(replacement));
                if let Ok(re) = regex::Regex::new(&pattern) {
                    let before_len = input_text.len();
                    input_text = re.replace_all(&input_text, *original).to_string();
                    if input_text.len() != before_len {
                        restored_count += 1;
                    }
                }
            } else {
                input_text = input_text.replace(*replacement, original);
                restored_count += 1;
            }
        }
    }

    // Write output
    write_output(cmd.output.as_ref(), &input_text)?;

    if !common.quiet {
        eprintln!("Restored {restored_count} PII values");
    }

    Ok(())
}

fn handle_detect(common: &CommonOpts, config: &Config, cmd: DetectCommand) -> Result<()> {
    // Check for streaming mode
    #[cfg(feature = "streaming")]
    if cmd.stream {
        return handle_detect_streaming(common, config, cmd);
    }

    #[cfg(not(feature = "streaming"))]
    if cmd.stream {
        return Err(anyhow!(
            "Streaming mode requires the 'streaming' feature. \
             Rebuild with: cargo build --features streaming"
        ));
    }

    // Read input. Office documents (docx/xlsx/pptx/odt/...) get their text
    // extracted (body, headers, notes, comments, metadata) and run through the
    // normal text pipeline.
    let office_fmt = cmd
        .input
        .as_ref()
        .and_then(|p| engine::office::sniff_path(p));
    let is_pdf = cmd
        .input
        .as_ref()
        .is_some_and(|p| engine::pdf::is_pdf_path(p));
    let input_text = if let Some(fmt) = office_fmt {
        let path = cmd
            .input
            .as_ref()
            .ok_or_else(|| anyhow!("office document input requires a file path"))?;
        let bytes = fs::read(path)
            .with_context(|| format!("Failed to read input file: {}", path.display()))?;
        engine::office::extract_text(&bytes, fmt)
            .map_err(|e| anyhow!("Failed to parse office document: {e}"))?
    } else if is_pdf {
        let path = cmd
            .input
            .as_ref()
            .ok_or_else(|| anyhow!("PDF input requires a file path"))?;
        let bytes = fs::read(path)
            .with_context(|| format!("Failed to read input file: {}", path.display()))?;
        #[expect(unused_mut, reason = "mutated only with the ocr feature")]
        let mut text =
            engine::pdf::extract_text(&bytes).map_err(|e| anyhow!("Failed to parse PDF: {e}"))?;
        #[cfg(feature = "ocr")]
        if cmd.ocr || config.ocr.enabled {
            let ocr_engine = build_ocr_engine(config)?;
            let img_text = engine::ocr::extract_pdf_image_text(&bytes, &ocr_engine)
                .map_err(|e| anyhow!("PDF image OCR failed: {e}"))?;
            if !img_text.is_empty() {
                text.push('\n');
                text.push_str(&img_text);
            }
        }
        #[cfg(not(feature = "ocr"))]
        if cmd.ocr {
            return Err(anyhow!("--ocr requires nym to be built with the `ocr` feature"));
        }
        text
    } else {
        #[cfg(feature = "ocr")]
        if sniff_image(cmd.input.as_ref()).is_some() {
            // Raster image input: recognize its text and scan that.
            let path = cmd
                .input
                .as_ref()
                .ok_or_else(|| anyhow!("image input requires a file path"))?;
            let ocr_engine = build_ocr_engine(config)?;
            let recognized = ocr_engine
                .recognize_file(path)
                .map_err(|e| anyhow!("OCR failed: {e}"))?;
            engine::ocr::assemble_text(&recognized).text
        } else {
            read_input(cmd.input.as_ref())?
        }
        #[cfg(not(feature = "ocr"))]
        read_input(cmd.input.as_ref())?
    };

    #[cfg(feature = "ocr")]
    let is_img_input = sniff_image(cmd.input.as_ref()).is_some();
    #[cfg(not(feature = "ocr"))]
    let is_img_input = false;

    // Determine format (explicit or auto-detect from file extension)
    let format = if office_fmt.is_some() || is_pdf || is_img_input {
        FormatArg::Text
    } else {
        cmd.format
            .unwrap_or_else(|| detect_format(cmd.input.as_ref()))
    };

    // Resolve ruleset or quick flags first
    let (ruleset_patterns, ruleset_excluded, ruleset_confidence, ruleset_ner_mode) =
        resolve_ruleset(
            config,
            cmd.ruleset.as_deref(),
            cmd.only_names,
            cmd.only_keys,
            cmd.only_contact,
            cmd.only_financial,
        );

    // Configure detector: CLI args > ruleset > config
    let min_confidence: Confidence = if cmd.min_confidence != ConfidenceArg::High {
        cmd.min_confidence.into()
    } else if let Some(c) = ruleset_confidence {
        c
    } else {
        config.detection.min_confidence.into()
    };

    let mut detector_config = DetectorConfig::default().with_min_confidence(min_confidence);

    // Use patterns from CLI, or ruleset, or fall back to config
    if let Some(ref patterns) = cmd.patterns {
        detector_config = detector_config.with_patterns(patterns.iter().cloned());
    } else if !ruleset_patterns.is_empty() {
        detector_config = detector_config.with_patterns(ruleset_patterns.iter().cloned());
    } else if !config.detection.enabled_patterns.is_empty() {
        detector_config =
            detector_config.with_patterns(config.detection.enabled_patterns.iter().cloned());
    }

    // Exclude patterns from CLI, ruleset, and config
    let mut excluded: Vec<String> = config.detection.disabled_patterns.clone();
    excluded.extend(ruleset_excluded);
    if let Some(ref exclude) = cmd.exclude {
        excluded.extend(exclude.iter().cloned());
    }
    if !excluded.is_empty() {
        detector_config = detector_config.without_patterns(excluded);
    }

    // Configure NER: CLI args > ruleset > config
    let ner_enabled = if cmd.no_ner {
        false
    } else if cmd.ner {
        true
    } else if let Some(ner_mode) = ruleset_ner_mode {
        let active_patterns: Vec<String> = if cmd.patterns.is_some() {
            cmd.patterns.clone().unwrap_or_default()
        } else if !ruleset_patterns.is_empty() {
            ruleset_patterns.clone()
        } else {
            config.detection.enabled_patterns.clone()
        };
        ner_mode.should_enable_ner(&active_patterns)
    } else {
        config.ner.enabled
    };

    detector_config = detector_config.with_ner(ner_enabled);

    if let Some(ref model) = cmd.ner_model {
        detector_config = detector_config.with_ner_model(model);
    } else if !config.ner.model.is_empty() {
        detector_config = detector_config.with_ner_model(&config.ner.model);
    }

    if let Some(threshold) = cmd.ner_threshold {
        detector_config = detector_config.with_ner_threshold(threshold);
    } else if config.ner.threshold > 0.0 {
        detector_config = detector_config.with_ner_threshold(config.ner.threshold);
    }
    detector_config = detector_config.with_ner_recall_first(config.ner.recall_first);

    if !config.ner.labels.is_empty() {
        detector_config = detector_config.with_ner_labels(config.ner.labels.clone());
    }

    if let Some(ref cache_dir) = config.ner.cache_dir {
        detector_config = detector_config.with_ner_cache_dir(cache_dir);
    }

    detector_config = apply_ner_backend(detector_config, &config.ner);

    let detector = Detector::new(&detector_config);

    // Detect based on format
    match format {
        FormatArg::Json => {
            let json_matches = detect_json(&input_text, &detector)
                .with_context(|| "Failed to parse input as JSON")?;

            if cmd.summary {
                let summary = create_json_summary(&json_matches);
                if common.json {
                    println!("{}", serde_json::to_string_pretty(&summary)?);
                } else if common.yaml {
                    println!("{}", serde_yaml::to_string(&summary)?);
                } else {
                    println!("PII Detection Summary (JSON):");
                    println!("  Total matches: {}", summary.total);
                    for (pattern, count) in &summary.by_pattern {
                        println!("  {pattern}: {count}");
                    }
                }
            } else {
                output_json_matches(common, &json_matches)?;
            }
        }
        FormatArg::Text => {
            let matches = detector.detect(&input_text);

            if cmd.summary {
                let summary = create_summary(&matches);
                if common.json {
                    println!("{}", serde_json::to_string_pretty(&summary)?);
                } else if common.yaml {
                    println!("{}", serde_yaml::to_string(&summary)?);
                } else {
                    println!("PII Detection Summary:");
                    println!("  Total matches: {}", summary.total);
                    for (pattern, count) in &summary.by_pattern {
                        println!("  {pattern}: {count}");
                    }
                }
            } else {
                output_text_matches(common, &matches)?;
            }
        }
    }

    Ok(())
}

/// Handle detection in streaming mode.
#[cfg(feature = "streaming")]
#[expect(
    clippy::needless_pass_by_value,
    reason = "CLI command struct consumed by handler"
)]
fn handle_detect_streaming(common: &CommonOpts, config: &Config, cmd: DetectCommand) -> Result<()> {
    use streaming::StreamConfig;

    // Build detector config
    let min_confidence = cmd.min_confidence.into();
    let mut detector_config = DetectorConfig::default().with_min_confidence(min_confidence);

    if let Some(ref patterns) = cmd.patterns {
        detector_config = detector_config.with_patterns(patterns.iter().cloned());
    } else if !config.detection.enabled_patterns.is_empty() {
        detector_config =
            detector_config.with_patterns(config.detection.enabled_patterns.iter().cloned());
    }

    let mut excluded: Vec<String> = config.detection.disabled_patterns.clone();
    if let Some(ref exclude) = cmd.exclude {
        excluded.extend(exclude.iter().cloned());
    }
    if !excluded.is_empty() {
        detector_config = detector_config.without_patterns(excluded);
    }

    // NER config
    let ner_enabled = if cmd.no_ner {
        false
    } else if cmd.ner {
        true
    } else {
        config.ner.enabled
    };
    detector_config = detector_config.with_ner(ner_enabled);

    if let Some(ref model) = cmd.ner_model {
        detector_config = detector_config.with_ner_model(model);
    }
    if let Some(threshold) = cmd.ner_threshold {
        detector_config = detector_config.with_ner_threshold(threshold);
    }

    let stream_config = StreamConfig {
        detector_config,
        replacer_config: ReplacerConfig::default(),
        session_id: None,
    };

    // Create tokio runtime and run
    let rt = tokio::runtime::Runtime::new().with_context(|| "Failed to create async runtime")?;

    let stats = rt.block_on(async {
        streaming::stream_detect(stream_config)
            .await
            .map_err(|e| anyhow!("Streaming error: {e}"))
    })?;

    if !common.quiet {
        eprintln!(
            "Processed {} lines, found {} PII occurrences",
            stats.lines_processed, stats.pii_found
        );
    }

    Ok(())
}

fn output_text_matches(common: &CommonOpts, matches: &[PiiMatch]) -> Result<()> {
    if common.json {
        println!("{}", serde_json::to_string_pretty(&matches)?);
    } else if common.yaml {
        println!("{}", serde_yaml::to_string(&matches)?);
    } else if matches.is_empty() {
        println!("No PII detected");
    } else {
        for m in matches {
            println!(
                "[{}] '{}' at {}..{} ({})",
                m.pattern_name,
                m.matched_text,
                m.start,
                m.end,
                format!("{:?}", m.confidence).to_lowercase()
            );
        }
        println!("\nTotal: {} matches", matches.len());
    }
    Ok(())
}

fn output_json_matches(common: &CommonOpts, matches: &[JsonPiiMatch]) -> Result<()> {
    if common.json {
        // Create serializable output
        let output: Vec<_> = matches
            .iter()
            .map(|m| {
                serde_json::json!({
                    "path": m.path,
                    "pattern_name": m.pii_match.pattern_name,
                    "matched_text": m.pii_match.matched_text,
                    "confidence": m.pii_match.confidence,
                    "category": m.pii_match.category,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else if common.yaml {
        let output: Vec<_> = matches
            .iter()
            .map(|m| {
                serde_json::json!({
                    "path": m.path,
                    "pattern_name": m.pii_match.pattern_name,
                    "matched_text": m.pii_match.matched_text,
                    "confidence": m.pii_match.confidence,
                    "category": m.pii_match.category,
                })
            })
            .collect();
        println!("{}", serde_yaml::to_string(&output)?);
    } else if matches.is_empty() {
        println!("No PII detected");
    } else {
        for m in matches {
            println!(
                "[{}] '{}' at {} ({})",
                m.pii_match.pattern_name,
                m.pii_match.matched_text,
                m.path,
                format!("{:?}", m.pii_match.confidence).to_lowercase()
            );
        }
        println!("\nTotal: {} matches", matches.len());
    }
    Ok(())
}

fn handle_patterns(common: &CommonOpts, cmd: PatternsCommand) -> Result<()> {
    match cmd.action.unwrap_or(PatternsAction::List) {
        PatternsAction::List => {
            if common.json {
                let patterns: Vec<PatternInfo> =
                    BUILTIN_PATTERNS.iter().map(PatternInfo::from).collect();
                println!("{}", serde_json::to_string_pretty(&patterns)?);
            } else if common.yaml {
                let patterns: Vec<PatternInfo> =
                    BUILTIN_PATTERNS.iter().map(PatternInfo::from).collect();
                println!("{}", serde_yaml::to_string(&patterns)?);
            } else {
                println!("Available PII patterns:\n");
                for pattern in BUILTIN_PATTERNS {
                    println!(
                        "  {:<20} {:?} confidence, {:?}",
                        pattern.name, pattern.confidence, pattern.category
                    );
                    println!("    {}", pattern.description);
                    println!("    Example: {}", pattern.example);
                    println!();
                }
            }
        }
        PatternsAction::Show { name } => {
            let pattern =
                engine::get_pattern(&name).ok_or_else(|| anyhow!("Unknown pattern: {name}"))?;

            if common.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&PatternInfo::from(pattern))?
                );
            } else if common.yaml {
                println!("{}", serde_yaml::to_string(&PatternInfo::from(pattern))?);
            } else {
                println!("Pattern: {}", pattern.name);
                println!("Description: {}", pattern.description);
                println!("Confidence: {:?}", pattern.confidence);
                println!("Category: {:?}", pattern.category);
                println!("Example: {}", pattern.example);
                println!("Regex: {}", pattern.regex.as_str());
            }
        }
        PatternsAction::Test { name, text } => {
            let pattern =
                engine::get_pattern(&name).ok_or_else(|| anyhow!("Unknown pattern: {name}"))?;

            let matches: Vec<_> = pattern.regex.find_iter(&text).collect();

            if common.json {
                let results: Vec<_> = matches
                    .iter()
                    .map(|m| {
                        serde_json::json!({
                            "matched": m.as_str(),
                            "start": m.start(),
                            "end": m.end(),
                        })
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&results)?);
            } else if matches.is_empty() {
                println!("No matches found");
            } else {
                for m in matches {
                    println!("Match: '{}' at {}..{}", m.as_str(), m.start(), m.end());
                }
            }
        }
        PatternsAction::Rulesets => {
            let rulesets = Config::builtin_rulesets();

            if common.json {
                let rulesets_info: Vec<_> = rulesets
                    .iter()
                    .map(|(name, rs)| {
                        serde_json::json!({
                            "name": name,
                            "description": rs.description,
                            "patterns": rs.enabled_patterns,
                            "ner": format!("{:?}", rs.ner),
                        })
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&rulesets_info)?);
            } else if common.yaml {
                let rulesets_info: Vec<_> = rulesets
                    .iter()
                    .map(|(name, rs)| {
                        serde_json::json!({
                            "name": name,
                            "description": rs.description,
                            "patterns": rs.enabled_patterns,
                            "ner": format!("{:?}", rs.ner),
                        })
                    })
                    .collect();
                println!("{}", serde_yaml::to_string(&rulesets_info)?);
            } else {
                println!("Available rulesets:\n");
                let mut names: Vec<_> = rulesets.keys().collect();
                names.sort();
                for name in names {
                    let rs = &rulesets[name];
                    println!("  {name}");
                    println!("    {}", rs.description);
                    println!("    Patterns: {}", rs.enabled_patterns.join(", "));
                    println!("    NER: {:?}", rs.ner);
                    println!();
                }
            }
        }
        PatternsAction::Ruleset { name } => {
            let rulesets = Config::builtin_rulesets();
            let ruleset = rulesets
                .get(&name)
                .ok_or_else(|| anyhow!("Unknown ruleset: {name}"))?;

            if common.json {
                let info = serde_json::json!({
                    "name": name,
                    "description": ruleset.description,
                    "enabled_patterns": ruleset.enabled_patterns,
                    "disabled_patterns": ruleset.disabled_patterns,
                    "min_confidence": ruleset.min_confidence.map(|c| format!("{c:?}")),
                    "ner": format!("{:?}", ruleset.ner),
                    "strategy": ruleset.strategy.map(|s| format!("{s:?}")),
                });
                println!("{}", serde_json::to_string_pretty(&info)?);
            } else if common.yaml {
                let info = serde_json::json!({
                    "name": name,
                    "description": ruleset.description,
                    "enabled_patterns": ruleset.enabled_patterns,
                    "disabled_patterns": ruleset.disabled_patterns,
                    "min_confidence": ruleset.min_confidence.map(|c| format!("{c:?}")),
                    "ner": format!("{:?}", ruleset.ner),
                    "strategy": ruleset.strategy.map(|s| format!("{s:?}")),
                });
                println!("{}", serde_yaml::to_string(&info)?);
            } else {
                println!("Ruleset: {name}");
                println!("Description: {}", ruleset.description);
                println!("Enabled patterns: {}", ruleset.enabled_patterns.join(", "));
                if !ruleset.disabled_patterns.is_empty() {
                    println!(
                        "Disabled patterns: {}",
                        ruleset.disabled_patterns.join(", ")
                    );
                }
                if let Some(c) = ruleset.min_confidence {
                    println!("Min confidence: {c:?}");
                }
                println!("NER mode: {:?}", ruleset.ner);
                if let Some(s) = ruleset.strategy {
                    println!("Strategy: {s:?}");
                }
            }
        }
    }

    Ok(())
}

fn handle_config(common: &CommonOpts, config: &Config, cmd: ConfigCommand) -> Result<()> {
    match cmd.action.unwrap_or(ConfigAction::Show) {
        ConfigAction::Show => {
            if common.json {
                println!("{}", serde_json::to_string_pretty(&config)?);
            } else if common.yaml {
                println!("{}", serde_yaml::to_string(&config)?);
            } else {
                println!("{}", toml::to_string_pretty(&config)?);
            }
        }
        ConfigAction::Path => {
            if let Some(path) = config::global_config_path() {
                println!("{}", path.display());
                if path.exists() {
                    eprintln!("(file exists)");
                } else {
                    eprintln!("(file does not exist)");
                }
            } else {
                eprintln!("Could not determine config path");
            }
        }
        ConfigAction::Init { force } => {
            let path = config::global_config_path()
                .ok_or_else(|| anyhow!("Could not determine config directory"))?;

            if path.exists() && !force {
                return Err(anyhow!(
                    "Config file already exists: {}\nUse --force to overwrite",
                    path.display()
                ));
            }

            // Create parent directories
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).with_context(|| {
                    format!("Failed to create config directory: {}", parent.display())
                })?;
            }

            // Read the example config and write it
            let default_config = include_str!("../examples/config.toml");
            fs::write(&path, default_config)
                .with_context(|| format!("Failed to write config file: {}", path.display()))?;

            println!("Config file created: {}", path.display());
        }
    }

    Ok(())
}

// -----------------------------------------------------------------------------
// Models command
// -----------------------------------------------------------------------------

#[cfg(feature = "ner")]
use engine::model_catalog::{self, CatalogModel};

/// Configured NER model cache directory, if any.
#[cfg(feature = "ner")]
fn ner_cache_dir(config: &Config) -> Option<PathBuf> {
    config.ner.cache_dir.as_ref().map(PathBuf::from)
}

/// The currently-configured default slug for each backend: (tokens, gliner).
#[cfg(feature = "ner")]
fn current_defaults(config: &Config) -> (String, String) {
    let tokens = config
        .ner
        .token_model
        .clone()
        .unwrap_or_else(|| engine::ner_token::DEFAULT_TOKEN_MODEL.to_string());
    (tokens, config.ner.model.clone())
}

#[cfg(feature = "ner")]
fn handle_models(common: &CommonOpts, config: &Config, cmd: ModelsCommand) -> Result<()> {
    match cmd.action.unwrap_or(ModelsAction::List) {
        ModelsAction::List => models_list(common, config),
        ModelsAction::Pull { query } => models_pull(config, query.as_deref()),
        ModelsAction::Use { query } => models_use(config, query.as_deref()),
        ModelsAction::Refresh => {
            let (url, count) = model_catalog::refresh()?;
            eprintln!("fetched {url}");
            println!(
                "updated catalog: {count} models -> {}",
                model_catalog::cache_path().display()
            );
            Ok(())
        }
    }
}

#[cfg(feature = "ner")]
fn model_row(m: &CatalogModel) -> String {
    format!(
        "{:<42} {:<32} {:<7} {:<19} {:>5}M  {}",
        m.slug, m.name, m.backend, m.languages, m.size_mb, m.description
    )
}

#[cfg(feature = "ner")]
fn models_list(common: &CommonOpts, config: &Config) -> Result<()> {
    let models = model_catalog::load();
    let cache_dir = ner_cache_dir(config);
    let (tok_default, gli_default) = current_defaults(config);

    if common.json {
        println!("{}", serde_json::to_string_pretty(&models)?);
        return Ok(());
    }

    println!(
        "   {:<42} {:<32} {:<7} {:<19} {:>6}  {}",
        "SLUG", "NAME", "BACKEND", "LANGUAGES", "SIZE", "DESCRIPTION"
    );
    for m in &models {
        let is_default =
            (m.is_tokens() && m.slug == tok_default) || (!m.is_tokens() && m.slug == gli_default);
        let cached = m.is_cached(cache_dir.as_deref());
        let mark_default = if is_default { '*' } else { ' ' };
        let mark_cached = if cached { '✓' } else { ' ' };
        println!("{mark_default}{mark_cached} {}", model_row(m));
    }
    eprintln!();
    eprintln!("  * = current default   ✓ = downloaded");
    eprintln!("Download with `nym models pull [query]`, switch with `nym models use [query]`.");
    Ok(())
}

#[cfg(feature = "ner")]
fn download_catalog_model(config: &Config, m: &CatalogModel) -> Result<()> {
    let cache_dir = ner_cache_dir(config);
    eprintln!("downloading {} ({}M)...", m.slug, m.size_mb);
    if m.is_tokens() {
        engine::ner_token::TokenClassDetector::download(&m.slug, cache_dir.as_deref())
            .map_err(|e| anyhow!("failed to download {}: {e}", m.slug))?;
    } else {
        let mut cfg = engine::ner::NerModelConfig::with_model(m.slug.clone());
        if let Some(dir) = cache_dir {
            cfg = cfg.with_cache_dir(dir);
        }
        cfg.download()
            .map_err(|e| anyhow!("failed to download {}: {e}", m.slug))?;
    }
    Ok(())
}

#[cfg(feature = "ner")]
fn models_pull(config: &Config, query: Option<&str>) -> Result<()> {
    let models = model_catalog::load();

    // An exact slug downloads directly; otherwise fuzzy-pick from the catalog.
    let selected = match query {
        Some(q) if models.iter().any(|m| m.slug == q) => {
            models.iter().find(|m| m.slug == q).cloned()
        }
        // Unknown slug that still looks like an HF repo id: pull it directly.
        Some(q) if q.contains('/') && !models.iter().any(|m| m.slug == q) => Some(CatalogModel {
            slug: q.to_string(),
            name: q.to_string(),
            backend: "tokens".to_string(),
            languages: "-".to_string(),
            size_mb: 0,
            description: "(not in catalog)".to_string(),
            recommended: false,
            default: false,
        }),
        other => pick_catalog(&models, other)?,
    };

    let Some(m) = selected else {
        eprintln!("nothing selected");
        return Ok(());
    };

    download_catalog_model(config, &m)?;
    println!("{}", m.slug);
    eprintln!("cached {}", m.slug);
    eprintln!("activate with: nym models use {}", m.slug);
    Ok(())
}

#[cfg(feature = "ner")]
fn models_use(config: &Config, query: Option<&str>) -> Result<()> {
    let models = model_catalog::load();
    let cache_dir = ner_cache_dir(config);

    // Exact slug (catalog or raw HF id): set it directly.
    if let Some(q) = query {
        if let Some(m) = models.iter().find(|m| m.slug == q) {
            return apply_default_model(config, m);
        }
        if q.contains('/') {
            let m = CatalogModel {
                slug: q.to_string(),
                name: q.to_string(),
                backend: "tokens".to_string(),
                languages: "-".to_string(),
                size_mb: 0,
                description: String::new(),
                recommended: false,
                default: false,
            };
            return apply_default_model(config, &m);
        }
    }

    // Otherwise pick among downloaded models.
    let downloaded: Vec<CatalogModel> = models
        .iter()
        .filter(|m| m.is_cached(cache_dir.as_deref()))
        .cloned()
        .collect();

    match downloaded.len() {
        0 => {
            eprintln!("No models downloaded yet.");
            eprintln!("Run `nym models pull` to download one, then `nym models use`.");
            Ok(())
        }
        1 => {
            eprintln!("Only one model downloaded; selecting it.");
            apply_default_model(config, &downloaded[0])
        }
        _ => match pick_catalog(&downloaded, query)? {
            Some(m) => apply_default_model(config, &m),
            None => {
                eprintln!("nothing selected");
                Ok(())
            }
        },
    }
}

/// Persist a model as the default for its backend, enabling NER.
#[cfg(feature = "ner")]
fn apply_default_model(config: &Config, m: &CatalogModel) -> Result<()> {
    let cache_dir = ner_cache_dir(config);
    let path = set_default_model(m)?;
    let field = if m.is_tokens() { "token_model" } else { "model" };
    println!("[ner] {field} = \"{}\"", m.slug);
    eprintln!("[ner] enabled = true");
    eprintln!("wrote {}", path.display());
    if !m.is_cached(cache_dir.as_deref()) {
        eprintln!("(not downloaded yet — it will fetch on first use, or run `nym models pull {}`)", m.slug);
    }
    Ok(())
}

/// Write the model into the global config via a surgical toml edit (comments and
/// formatting preserved). Seeds from the example config when none exists yet.
#[cfg(feature = "ner")]
fn set_default_model(m: &CatalogModel) -> Result<PathBuf> {
    use toml_edit::{DocumentMut, value};

    let path = config::global_config_path()
        .ok_or_else(|| anyhow!("Could not determine config directory"))?;

    let mut doc: DocumentMut = if path.exists() {
        fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?
            .parse()
            .with_context(|| format!("parsing {}", path.display()))?
    } else {
        include_str!("../examples/config.toml")
            .parse()
            .context("parsing bundled example config")?
    };

    if !doc.contains_key("ner") {
        doc["ner"] = toml_edit::table();
    }
    let ner = doc["ner"]
        .as_table_mut()
        .ok_or_else(|| anyhow!("[ner] is not a table in config"))?;
    ner["enabled"] = value(true);
    let field = if m.is_tokens() { "token_model" } else { "model" };
    ner[field] = value(m.slug.clone());

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    fs::write(&path, doc.to_string()).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

/// Fuzzy-pick a catalog model via fzf, falling back to a numbered prompt.
/// Returns the chosen model, or `None` if the user cancelled.
#[cfg(feature = "ner")]
fn pick_catalog(models: &[CatalogModel], query: Option<&str>) -> Result<Option<CatalogModel>> {
    use std::io::stdin;
    use std::process::{Command, Stdio};

    if models.is_empty() {
        return Ok(None);
    }
    let lines: Vec<String> = models.iter().map(model_row).collect();

    let mut cmd = Command::new("fzf");
    cmd.arg("--prompt=model> ")
        .arg("--height=40%")
        .arg("--reverse")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped());
    if let Some(q) = query {
        cmd.arg(format!("--query={q}"));
    }
    match cmd.spawn() {
        Ok(mut child) => {
            if let Some(mut child_stdin) = child.stdin.take() {
                child_stdin.write_all(lines.join("\n").as_bytes()).ok();
            }
            let output = child.wait_with_output().context("running fzf")?;
            if !output.status.success() {
                return Ok(None); // cancelled
            }
            let selection = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            Ok(models
                .iter()
                .zip(lines.iter())
                .find(|(_, line)| **line == selection)
                .map(|(m, _)| m.clone()))
        }
        Err(_) => {
            eprintln!("(fzf not found - pick a number)");
            for (i, line) in lines.iter().enumerate() {
                eprintln!("{:>3}  {}", i + 1, line);
            }
            eprint!("model number: ");
            io::stderr().flush().ok();
            let mut buf = String::new();
            stdin().read_line(&mut buf).context("reading selection")?;
            let Ok(n) = buf.trim().parse::<usize>() else {
                return Ok(None);
            };
            Ok(n.checked_sub(1).and_then(|i| models.get(i)).cloned())
        }
    }
}

fn handle_sessions(common: &CommonOpts, cmd: SessionsCommand) -> Result<()> {
    match cmd
        .action
        .unwrap_or(SessionsAction::List { directory: None })
    {
        SessionsAction::List { directory } => list_sessions(common, directory),
        SessionsAction::Search { id, directory } => search_sessions(common, &id, directory),
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "CLI command struct consumed by handler"
)]
fn list_sessions(common: &CommonOpts, search_dir: Option<PathBuf>) -> Result<()> {
    let mut sessions = Vec::new();

    // Search directories: current, then user config dir
    let default_path = PathBuf::from(".");
    let keys_dir = config::global_config_path()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| default_path.clone())
        .join("keys");

    let mut search_dirs: Vec<PathBuf> = vec![std::env::current_dir().unwrap_or_default(), keys_dir];

    // Add user-specified directory
    if let Some(ref dir) = search_dir {
        search_dirs.insert(0, dir.clone());
    }

    for dir in &search_dirs {
        if dir.exists()
            && let Ok(entries) = fs::read_dir(dir)
        {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("jsonl")
                    && let Ok(session_info) = parse_key_file_header(&path)
                {
                    sessions.push((path, session_info));
                }
            }
        }
    }

    // Sort by creation time
    sessions.sort_by(|a, b| b.1.created.cmp(&a.1.created));

    if common.json {
        let output: Vec<_> = sessions
            .into_iter()
            .map(|(path, info)| {
                serde_json::json!({
                    "file": path.to_string_lossy(),
                    "session": info.session,
                    "source": info.source,
                    "created": info.created,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("Found {} session key files:", sessions.len());
        for (path, info) in sessions {
            println!("  {}  {}", path.display(), info.full_reference());
        }
    }

    Ok(())
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "CLI command struct consumed by handler"
)]
fn search_sessions(
    common: &CommonOpts,
    search_id: &str,
    search_dir: Option<PathBuf>,
) -> Result<()> {
    let mut matches = Vec::new();

    // Search directories
    let default_path = PathBuf::from(".");
    let keys_dir = config::global_config_path()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| default_path.clone())
        .join("keys");

    let mut search_dirs: Vec<PathBuf> = vec![std::env::current_dir().unwrap_or_default(), keys_dir];

    // Add user-specified directory
    if let Some(ref dir) = search_dir {
        search_dirs.insert(0, dir.clone());
    }

    for dir in &search_dirs {
        if dir.exists()
            && let Ok(entries) = fs::read_dir(dir)
        {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("jsonl")
                    && let Ok(session_info) = parse_key_file_header(&path)
                    && session_info.session.contains(search_id)
                {
                    matches.push((path, session_info));
                }
            }
        }
    }

    if matches.is_empty() {
        eprintln!("No sessions found with ID: {search_id}");
        eprintln!("Available session IDs:");
        for info in list_session_ids_only(&search_dirs) {
            println!("  {}", info.session);
        }
        return Ok(());
    }

    if common.json {
        let output: Vec<_> = matches
            .into_iter()
            .map(|(path, info)| {
                serde_json::json!({
                    "file": path.to_string_lossy(),
                    "session": info.session,
                    "source": info.source,
                    "created": info.created,
                    "replacements": info.replacement_count,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("Found {} matching sessions:", matches.len());
        for (path, info) in matches {
            println!(
                "  {}  {}  ({} replacements)",
                path.display(),
                info.full_reference(),
                info.replacement_count
            );
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Serialize)]
struct SessionInfo {
    session: String,
    source: Option<String>,
    created: String,
    replacement_count: usize,
}

impl SessionInfo {
    fn full_reference(&self) -> String {
        match &self.source {
            Some(src) => format!("{} ({})", self.session, src),
            None => self.session.clone(),
        }
    }
}

fn parse_key_file_header(path: &PathBuf) -> Result<SessionInfo> {
    let file = fs::File::open(path)?;
    let mut reader = io::BufReader::new(file);

    // Read first line for header
    let mut header_line = String::new();
    reader.read_line(&mut header_line)?;

    if let Ok(header) = serde_json::from_str::<serde_json::Value>(&header_line) {
        Ok(SessionInfo {
            session: header["session"].as_str().unwrap_or("unknown").to_string(),
            source: header["source"].as_str().map(String::from),
            created: header["created"].as_str().unwrap_or("unknown").to_string(),
            replacement_count: count_replacements(path)?,
        })
    } else {
        Err(anyhow!("Invalid header format"))
    }
}

fn count_replacements(path: &PathBuf) -> Result<usize> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);

    let mut count = 0;
    let mut line_num = 0;

    for line in reader.lines() {
        line_num += 1;
        if line_num > 1 {
            // Skip header line
            if let Ok(line_str) = line
                && serde_json::from_str::<engine::Replacement>(&line_str).is_ok()
            {
                count += 1;
            }
            // Skip malformed lines
        }
    }

    Ok(count)
}

fn list_session_ids_only(search_dirs: &[PathBuf]) -> Vec<SessionInfo> {
    let mut sessions = Vec::new();

    for dir in search_dirs {
        if dir.exists()
            && let Ok(entries) = fs::read_dir(dir)
        {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("jsonl")
                    && let Ok(info) = parse_key_file_header(&path)
                {
                    sessions.push(info);
                }
            }
        }
    }

    // Sort by creation time
    sessions.sort_by(|a, b| b.created.cmp(&a.created));
    sessions
}

#[expect(clippy::unnecessary_wraps, reason = "Consistent handler return type")]
fn handle_completions(shell: Shell) -> Result<()> {
    let mut cmd = Cli::command();
    clap_complete::generate(shell, &mut cmd, APP_NAME, &mut io::stdout());
    Ok(())
}

/// Handle the benchmark command.
#[cfg(feature = "bench")]
fn handle_bench(common: &CommonOpts, config: &Config, cmd: BenchCommand) -> Result<()> {
    use bench::{BenchConfig, load_jsonl, run_benchmark};
    use std::path::Path;

    // Build detector config
    let min_confidence = cmd.min_confidence.into();
    let mut detector_config = DetectorConfig::default().with_min_confidence(min_confidence);

    // Configure NER
    let ner_enabled = cmd.ner || config.ner.enabled;
    detector_config = detector_config.with_ner(ner_enabled);

    if !config.ner.model.is_empty() {
        detector_config = detector_config.with_ner_model(&config.ner.model);
    }
    if config.ner.threshold > 0.0 {
        detector_config = detector_config.with_ner_threshold(config.ner.threshold);
    }
    if !config.ner.labels.is_empty() {
        detector_config = detector_config.with_ner_labels(config.ner.labels.clone());
    }

    detector_config = apply_ner_backend(detector_config, &config.ner);

    // Load examples
    let source_path = Path::new(&cmd.source);
    let examples = if source_path.exists()
        && source_path
            .extension()
            .map(|e| e == "jsonl")
            .unwrap_or(false)
    {
        // Load from local JSONL file
        if !common.quiet {
            eprintln!("Loading dataset from {}...", source_path.display());
        }
        load_jsonl(source_path)?
    } else {
        // Try to download from HuggingFace
        #[cfg(feature = "bench")]
        {
            if !common.quiet {
                eprintln!(
                    "Fetching {} examples from {} (split: {})...",
                    cmd.fetch_limit, cmd.source, cmd.split
                );
            }
            bench::download_huggingface_dataset(
                &cmd.source,
                &cmd.split,
                cmd.cache_dir.as_deref(),
                cmd.fetch_limit,
            )?
        }
        #[cfg(not(feature = "bench"))]
        {
            return Err(anyhow!(
                "HuggingFace dataset download requires the 'bench' feature"
            ));
        }
    };

    // Apply sampling
    let total_loaded = examples.len();
    let max_examples = if let Some(pct) = cmd.sample {
        // Percentage-based sampling
        let pct = pct.clamp(0.0, 100.0);
        Some((total_loaded as f64 * pct / 100.0).ceil() as usize)
    } else {
        cmd.num_examples
    };

    let bench_config = BenchConfig {
        detector_config,
        strict_matching: cmd.strict,
        ignore_labels: cmd.ignore_labels,
        max_examples,
        track_misses_for: cmd.show_misses.clone(),
        track_false_positives_for: cmd.show_false_positives.clone(),
        ..Default::default()
    };

    let examples_to_run = max_examples.unwrap_or(total_loaded).min(total_loaded);

    if !common.quiet {
        eprintln!("Loaded {} examples", total_loaded);
        if max_examples.is_some() {
            eprintln!("Running benchmark on {} examples...", examples_to_run);
        } else {
            eprintln!("Running benchmark on all examples...");
        }
        if ner_enabled {
            eprintln!("NER detection enabled");
        }
    }

    // Run benchmark
    let results = run_benchmark(&examples, &bench_config)?;

    // Output results
    if common.json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else if common.yaml {
        println!("{}", serde_yaml::to_string(&results)?);
    } else {
        println!("Benchmark Results");
        println!("=================");
        println!();
        println!("Examples:      {}", results.total_examples);
        println!("Ground Truth:  {}", results.total_ground_truth);
        println!("Detected:      {}", results.total_detected);
        println!();
        println!("True Positives:  {}", results.true_positives);
        println!("False Positives: {}", results.false_positives);
        println!("False Negatives: {}", results.false_negatives);
        println!();
        println!("Precision: {:.2}%", results.precision * 100.0);
        println!("Recall:    {:.2}%", results.recall * 100.0);
        println!("F1 Score:  {:.2}%", results.f1 * 100.0);

        if !results.by_label.is_empty() {
            println!();
            println!("By Label:");
            println!("---------");
            let mut labels: Vec<_> = results.by_label.iter().collect();
            labels.sort_by(|a, b| a.0.cmp(b.0));
            for (label, lr) in labels {
                println!(
                    "  {:<15} P: {:.1}%  R: {:.1}%  F1: {:.1}%  (TP:{} FP:{} FN:{})",
                    label,
                    lr.precision * 100.0,
                    lr.recall * 100.0,
                    lr.f1 * 100.0,
                    lr.true_positives,
                    lr.false_positives,
                    lr.false_negatives
                );
            }
        }

        // Show missed detections if requested
        if !results.missed_detections.is_empty() {
            println!();
            println!(
                "Missed Detections ({}):",
                cmd.show_misses.as_deref().unwrap_or("")
            );
            println!("-----------------------");
            for miss in &results.missed_detections {
                println!("  Text: {:?}", miss.text);
                println!("  Context: ...{}...", miss.context.replace('\n', " "));
                println!();
            }
        }

        // Show false positives if requested
        if !results.false_positive_detections.is_empty() {
            println!();
            println!(
                "False Positives ({}):",
                cmd.show_false_positives.as_deref().unwrap_or("")
            );
            println!("---------------------");
            for fp in &results.false_positive_detections {
                println!("  Text: {:?}", fp.text);
                println!("  Context: ...{}...", fp.context.replace('\n', " "));
                println!();
            }
        }
    }

    Ok(())
}

// =============================================================================
// Helper Functions
// =============================================================================

#[expect(
    clippy::unnecessary_wraps,
    reason = "May return errors in future logging backends"
)]
fn init_logging(common: &CommonOpts) -> Result<()> {
    if common.quiet {
        return Ok(());
    }

    let level = match common.verbose {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        2 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };

    env_logger::Builder::from_env(env_logger::Env::default())
        .filter_level(level)
        .try_init()
        .ok();

    Ok(())
}

/// Detect a raster-image input by extension (OCR feature).
#[cfg(feature = "ocr")]
fn sniff_image(path: Option<&PathBuf>) -> Option<engine::ocr::RasterFormat> {
    path.and_then(|p| p.extension())
        .and_then(|e| e.to_str())
        .and_then(engine::ocr::RasterFormat::from_ext)
}

/// Resolve the configured external OCR engine.
#[cfg(feature = "ocr")]
fn build_ocr_engine(config: &Config) -> Result<engine::ocr::OcrEngine> {
    engine::ocr::OcrEngine::resolve(&config.ocr.engine, config.ocr.min_confidence)
        .map_err(|e| anyhow!("{e}"))
}

/// Default output path for in-place document redaction:
/// `report.docx` -> `report.anon.docx` / `report.restored.docx`.
fn derive_document_output(path: &std::path::Path, tag: &str) -> PathBuf {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("bin");
    path.with_file_name(format!("{stem}.{tag}.{ext}"))
}

fn read_input(path: Option<&PathBuf>) -> Result<String> {
    if let Some(p) = path {
        fs::read_to_string(p).with_context(|| format!("Failed to read input file: {}", p.display()))
    } else {
        let stdin = io::stdin();
        if stdin.is_terminal() {
            eprintln!("Reading from stdin (Ctrl+D to finish)...");
        }
        let mut buffer = String::new();
        stdin.lock().read_to_string(&mut buffer)?;
        Ok(buffer)
    }
}

fn write_output(path: Option<&PathBuf>, content: &str) -> Result<()> {
    if let Some(p) = path {
        fs::write(p, content)
            .with_context(|| format!("Failed to write output file: {}", p.display()))
    } else {
        print!("{content}");
        io::stdout().flush()?;
        Ok(())
    }
}

fn write_key_file(
    path: &PathBuf,
    replacements: &[engine::Replacement],
    session: &Session,
) -> Result<()> {
    use std::io::BufWriter;

    let file = fs::File::create(path)
        .with_context(|| format!("Failed to create key file: {}", path.display()))?;
    let mut writer = BufWriter::new(file);

    // Write header with session info
    let header = serde_json::json!({
        "version": "1",
        "created": chrono::Utc::now().to_rfc3339(),
        "session": session.id,
        "source": session.source,
    });
    serde_json::to_writer(&mut writer, &header)?;
    writeln!(writer)?;

    // Write each replacement
    for r in replacements {
        serde_json::to_writer(&mut writer, r)?;
        writeln!(writer)?;
    }

    Ok(())
}

#[derive(Debug, Serialize)]
struct DetectionSummary {
    total: usize,
    by_pattern: std::collections::HashMap<String, usize>,
}

fn create_summary(matches: &[PiiMatch]) -> DetectionSummary {
    let mut by_pattern = std::collections::HashMap::new();
    for m in matches {
        *by_pattern.entry(m.pattern_name.clone()).or_insert(0) += 1;
    }
    DetectionSummary {
        total: matches.len(),
        by_pattern,
    }
}

fn create_json_summary(matches: &[JsonPiiMatch]) -> DetectionSummary {
    let mut by_pattern = std::collections::HashMap::new();
    for m in matches {
        *by_pattern
            .entry(m.pii_match.pattern_name.clone())
            .or_insert(0) += 1;
    }
    DetectionSummary {
        total: matches.len(),
        by_pattern,
    }
}

/// Detect format from file extension.
fn detect_format(path: Option<&PathBuf>) -> FormatArg {
    match path {
        Some(p) => match p.extension().and_then(|e| e.to_str()) {
            Some("json") => FormatArg::Json,
            _ => FormatArg::Text,
        },
        None => FormatArg::Text, // Default to text for stdin
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct PatternInfo {
    name: String,
    description: String,
    confidence: Confidence,
    category: engine::PiiCategory,
    example: String,
}

impl From<&engine::PiiPattern> for PatternInfo {
    fn from(p: &engine::PiiPattern) -> Self {
        Self {
            name: p.name.to_string(),
            description: p.description.to_string(),
            confidence: p.confidence,
            category: p.category,
            example: p.example.to_string(),
        }
    }
}
