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
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::Shell;
use log::{debug, info};
use serde::{Deserialize, Serialize};

mod config;
mod engine;
mod input;
mod session;
mod terms;
mod terms_cli;
use input::FormatArg;
#[cfg(feature = "streaming")]
mod streaming;
#[cfg(all(feature = "streaming", feature = "ner"))]
mod streaming_ner;

use config::Config;
use engine::{
    BUILTIN_PATTERNS, Confidence, Detector, DetectorConfig, JsonPiiMatch, PiiCategory, PiiMatch,
    ReplacementStrategy, Replacer, ReplacerConfig, process_json_with_selector,
};
use session::Session;

const APP_NAME: &str = env!("CARGO_PKG_NAME");

/// One backend-aware NER resolution path for every command, including streams.
fn configure_ner(
    dc: DetectorConfig,
    ner: &config::NerConfig,
    model: Option<&str>,
    threshold: Option<f32>,
) -> Result<DetectorConfig> {
    if !dc.ner_enabled && (model.is_some() || threshold.is_some()) {
        anyhow::bail!(
            "explicit NER controls require enabled NER; use --ner or remove the controls"
        );
    }
    if !dc.ner_enabled {
        return Ok(dc);
    }
    let mut ner = ner.clone();
    if let Some(threshold) = threshold {
        ner.threshold = threshold;
    }
    let mut ner = ner.resolved(model)?;
    ner.token_model = ner
        .token_model
        .as_deref()
        .map(|model| shellexpand::full(model).map(|path| path.into_owned()))
        .transpose()
        .map_err(|_| anyhow!("failed to expand NER model path"))?;
    let mut dc = apply_ner_backend(dc, &ner)
        .with_ner_model(
            shellexpand::full(&ner.model)
                .map_err(|_| anyhow!("failed to expand NER model path"))?
                .into_owned(),
        )
        .with_ner_threshold(threshold.unwrap_or(ner.threshold))
        .with_ner_recall_first(ner.recall_first);
    if let Some(ref labels) = ner.labels {
        dc = dc.with_ner_labels(labels.clone());
    }
    if let Some(ref cache) = ner.cache_dir {
        dc = dc.with_ner_cache_dir(
            shellexpand::full(cache)
                .map_err(|_| anyhow!("failed to expand NER cache path"))?
                .into_owned(),
        );
    }
    Ok(dc)
}

/// Apply the configured NER backend selection (`gliner`/`tokens`/`both`) and
/// the token-classification model/provider onto a detector config.
fn apply_ner_backend(mut dc: DetectorConfig, ner: &config::NerConfig) -> DetectorConfig {
    dc = dc
        .with_ner_backend(ner.backend)
        .with_ner_provider(ner.provider);
    if let Some(ref model) = ner.token_model {
        dc = dc.with_ner_token_model(shellexpand::tilde(model).into_owned());
    }
    dc
}

#[cfg(test)]
mod provider_wiring_tests {
    use super::*;

    #[test]
    fn detector_provider_defaults_to_auto_and_can_be_overridden() {
        assert_eq!(
            (
                DetectorConfig::default().ner_provider,
                DetectorConfig::default()
                    .with_ner_provider(config::NerProvider::Cpu)
                    .ner_provider,
            ),
            (config::NerProvider::Auto, config::NerProvider::Cpu)
        );
    }

    #[test]
    fn configured_provider_reaches_detector_without_overwriting_other_ner_options() {
        for provider in [config::NerProvider::Auto, config::NerProvider::Cpu] {
            let ner = config::NerConfig {
                provider,
                backend: engine::detector::NerBackend::Both,
                token_model: Some("synthetic-model".into()),
                ..Default::default()
            };
            let detector = apply_ner_backend(DetectorConfig::default(), &ner);
            assert_eq!(
                (
                    detector.ner_provider,
                    detector.ner_backend,
                    detector.ner_token_model
                ),
                (
                    provider,
                    ner.backend,
                    Some(PathBuf::from("synthetic-model"))
                )
            );
        }
    }
}

fn main() {
    // If a native teardown hook runs after an unexpected failure, fail closed.
    // Only a fully successful CLI run may publish a successful native exit.
    engine::ner::set_exit_code(1);
    // A native wrapper's panic payload can contain private paths/configuration.
    // Guarded detector operations still propagate a typed error and exit 1.
    std::panic::set_hook(Box::new(|_| {
        let _ = writeln!(
            io::stderr().lock(),
            "error: internal runtime failure; scan incomplete"
        );
    }));
    if let Err(err) = try_main() {
        let _ = writeln!(io::stderr(), "error: {err:?}");
        let code = error_exit_code(&err);
        // GLiNER may have registered the macOS ONNX teardown hook even when
        // initialization failed. Its fast exit must preserve this CLI status.
        engine::ner::set_exit_code(code);
        std::process::exit(code);
    }
    engine::ner::set_exit_code(0);
}

fn error_exit_code(err: &anyhow::Error) -> i32 {
    err.downcast_ref::<ExitError>()
        .map(|error| error.0)
        .unwrap_or(1)
}

/// An error carrying an explicit process exit code, used to signal distinct
/// statuses (e.g. an audit gate that found blockers) without conflating them
/// with a generic failure.
#[derive(Debug)]
struct ExitError(i32);

impl std::fmt::Display for ExitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "exit code {}", self.0)
    }
}

impl std::error::Error for ExitError {}

fn try_main() -> Result<()> {
    let cli = Cli::parse();

    init_logging(&cli.common)?;

    // Load configuration
    let config = load_config(&cli.common)?;

    match cli.command {
        Command::Anon(cmd) => handle_anon(&cli.common, &config, cmd),
        Command::Deanon(cmd) => handle_deanon(&cli.common, cmd),
        Command::Detect(cmd) => handle_detect(&cli.common, &config, cmd),
        #[cfg(feature = "decision")]
        Command::Decide(cmd) => handle_decide(&cli.common, &config, &cmd),
        Command::Patterns(cmd) => handle_patterns(&cli.common, cmd),
        Command::Terms(cmd) => terms_cli::run(&cli.common, &config.terms, cmd),
        Command::Config(cmd) => handle_config(&cli.common, &config, cmd),
        Command::Sessions(cmd) => handle_sessions(&cli.common, cmd),
        #[cfg(feature = "ner")]
        Command::Models(cmd) => handle_models(&cli.common, &config, cmd),
        Command::Completions { shell } => handle_completions(shell),
    }
}

fn load_config(common: &CommonOpts) -> Result<Config> {
    if let Some(ref config_path) = common.config {
        // Load from specific config file
        let content = fs::read_to_string(config_path)
            .map_err(|_| anyhow!("failed to read configuration file"))?;
        toml::from_str(&content).map_err(|_| anyhow!("invalid configuration file"))
    } else {
        // Load from default locations
        Config::load().map_err(|_| anyhow!("invalid configuration"))
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
    version = env!("NYM_BUILD_VERSION"),
    about = "Fast, reversible PII anonymization CLI",
    long_about = "nym detects and anonymizes personally identifiable information (PII) in text files.\n\n\
                  It supports text, JSON, JSONL and dedicated document handlers, and can optionally \
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

    /// Decide whether detected spans are real secrets using a decision model.
    /// Adjudicates regex + NER + high-entropy candidates against an
    /// OpenAI-compatible endpoint, returning a keep/redact/flag verdict per
    /// span so a downstream scrubber can veto over-redactions and catch
    /// unlabeled secrets. Requires the `decision` feature.
    #[cfg(feature = "decision")]
    Decide(DecideCommand),

    /// List and inspect available PII patterns
    Patterns(PatternsCommand),

    /// Discover, review and export corpus-specific sensitive vocabulary
    Terms(terms_cli::TermsCommand),

    /// Show or manage configuration
    Config(ConfigCommand),

    /// Search for sessions by ID
    Sessions(SessionsCommand),

    /// List, download, and select NER models
    #[cfg(feature = "ner")]
    Models(ModelsCommand),

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

/// Shared CLI overrides; a supplied list replaces the corresponding config list.
#[derive(Debug, Clone, Default, Args)]
struct TraceOpts {
    /// Opt-in trace heuristics, or disable configured heuristics
    #[arg(long, value_enum)]
    profile: Option<ProfileArg>,
    /// UTF-8 file of literal sensitive terms (repeatable; replaces configured files)
    #[arg(long = "sensitive-terms-file", value_name = "FILE")]
    sensitive_terms_files: Vec<String>,
    /// UTF-8 file of benign literals (repeatable; never vetoes sensitive/regex findings)
    #[arg(long = "benign-terms-file", value_name = "FILE")]
    benign_terms_files: Vec<String>,
    /// Exact public DNS host (repeatable; replaces configured hosts)
    #[arg(long = "public-host", value_name = "HOST")]
    public_hosts: Vec<String>,
    /// Literal matching boundary (Unicode word or substring)
    #[arg(long, value_enum)]
    term_boundary: Option<BoundaryArg>,
    /// Literal matching case sensitivity (true or false)
    #[arg(long, action = clap::ArgAction::Set)]
    term_case_sensitive: Option<bool>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ProfileArg {
    Default,
    AgentTrace,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BoundaryArg {
    Word,
    Substring,
}

fn configure_trace(
    mut detector: DetectorConfig,
    config: &Config,
    opts: &TraceOpts,
) -> Result<DetectorConfig> {
    let settings = &config.trace_policy;
    let files = |cli: &[String], configured: &[String]| {
        config::load_term_files(if cli.is_empty() { configured } else { cli })
    };
    let policy = engine::TracePolicyConfig {
        profile: match opts.profile {
            Some(ProfileArg::Default) => None,
            Some(ProfileArg::AgentTrace) => Some(engine::TraceProfile::AgentTrace),
            None => settings.profile,
        },
        public_hosts: if opts.public_hosts.is_empty() {
            settings.public_hosts.clone()
        } else {
            opts.public_hosts.clone()
        },
        sensitive_terms: files(&opts.sensitive_terms_files, &settings.sensitive_terms_files)?,
        benign_terms: files(&opts.benign_terms_files, &settings.benign_terms_files)?,
        case_sensitive: opts.term_case_sensitive.unwrap_or(settings.case_sensitive),
        term_boundary: match opts.term_boundary {
            Some(BoundaryArg::Word) => engine::TermBoundary::Word,
            Some(BoundaryArg::Substring) => engine::TermBoundary::Substring,
            None => settings.term_boundary,
        },
    };
    policy.validate()?;
    if policy.profile.is_some()
        || !policy.sensitive_terms.is_empty()
        || !policy.benign_terms.is_empty()
        || !policy.public_hosts.is_empty()
    {
        detector = detector.with_trace_policy(policy);
    }
    Ok(detector)
}

// -----------------------------------------------------------------------------
// Anon Command
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Args)]
#[expect(clippy::struct_excessive_bools, reason = "CLI flags from clap")]
struct AnonCommand {
    #[command(flatten)]
    trace: TraceOpts,
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

    /// Pseudonym context tag recorded in the key-file header, so a seeded
    /// dataset build can be reproduced and verified later (e.g. a release
    /// identifier or dataset label).
    #[arg(long)]
    context: Option<String>,

    /// Enable the configured NER backend alongside regex (requires 'ner' feature)
    #[arg(long)]
    ner: bool,

    /// Disable NER-based detection (overrides config)
    #[arg(long, conflicts_with = "ner")]
    no_ner: bool,

    /// Override the selected backend's model: tokens -> token_model, gliner -> model;
    /// rejected for both (configure ner.model and ner.token_model separately)
    #[arg(long, value_name = "MODEL")]
    ner_model: Option<String>,

    /// NER confidence threshold (0.0-1.0, default: 0.5)
    #[arg(long, value_name = "THRESHOLD")]
    ner_threshold: Option<f32>,

    /// Stream text line-by-line; JSONL remains structured and output files are staged
    /// (requires 'streaming' feature; single JSON documents are buffered)
    #[arg(long)]
    stream: bool,

    /// JSON/JSONL: only anonymize string values at these JSON paths
    /// (repeatable; dot and array-index syntax, e.g. `session.user.email`,
    /// `users[*].email`)
    #[arg(long = "include-path", value_name = "PATH")]
    include_paths: Vec<String>,

    /// JSON only: skip string values at these JSON paths (repeatable)
    #[arg(long = "exclude-path", value_name = "PATH")]
    exclude_paths: Vec<String>,

    /// JSON only: print a coverage report of scanned/skipped paths
    #[arg(long)]
    json_coverage: bool,
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
    /// Disable fail-closed checks for recognizable residual pseudonyms
    #[arg(long)]
    no_verify_restore: bool,

    /// Text/JSON/JSONL input format (otherwise resolved like anon/detect)
    #[arg(short = 'f', long, value_enum)]
    format: Option<FormatArg>,

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
    #[command(flatten)]
    trace: TraceOpts,
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

    /// Enable the configured NER backend alongside regex (requires 'ner' feature)
    #[arg(long)]
    ner: bool,

    /// Disable NER-based detection (overrides config)
    #[arg(long, conflicts_with = "ner")]
    no_ner: bool,

    /// Override the selected backend's model: tokens -> token_model, gliner -> model;
    /// rejected for both (configure ner.model and ner.token_model separately)
    #[arg(long, value_name = "MODEL")]
    ner_model: Option<String>,

    /// NER confidence threshold (0.0-1.0, default: 0.5)
    #[arg(long, value_name = "THRESHOLD")]
    ner_threshold: Option<f32>,

    /// Stream text/JSONL records with possible partial stdout on later failure;
    /// audit policies and summaries require omitting --stream
    #[arg(long)]
    stream: bool,

    /// JSON/JSONL: only detect string values at these JSON paths (repeatable)
    #[arg(long = "include-path", value_name = "PATH")]
    include_paths: Vec<String>,

    /// JSON only: skip string values at these JSON paths (repeatable)
    #[arg(long = "exclude-path", value_name = "PATH")]
    exclude_paths: Vec<String>,

    /// JSON only: print a coverage report of scanned/skipped paths
    #[arg(long)]
    json_coverage: bool,

    /// Fail (nonzero exit) when any finding matches a pattern name or PII
    /// category (repeatable, e.g. `--fail-on email --fail-on ssn` or
    /// `--fail-on financial`). Default inspection behavior is unchanged when
    /// this is not supplied.
    #[arg(long = "fail-on", value_name = "PATTERN_OR_CATEGORY")]
    fail_on: Vec<String>,

    /// Emit a value-free machine-readable summary (aggregate counts only, no
    /// matched values, no source paths) suitable for a public run manifest.
    #[arg(long)]
    summary_json: bool,
}

// -----------------------------------------------------------------------------
// Decide Command
// -----------------------------------------------------------------------------

/// Adjudicate detected spans with a decision model (keep/redact/flag).
#[expect(
    clippy::struct_excessive_bools,
    reason = "clap CLI struct; splitting into enums would complicate flag parsing"
)]
#[derive(Debug, Clone, Args)]
struct DecideCommand {
    #[command(flatten)]
    trace: TraceOpts,
    /// Input file (reads from stdin if not specified)
    #[arg(value_name = "INPUT")]
    input: Option<PathBuf>,

    /// Input format (auto-detected from extension if not specified)
    #[arg(short = 'f', long, value_enum)]
    format: Option<FormatArg>,

    /// Decision endpoint (overrides [decision] endpoint)
    #[arg(long, value_name = "URL")]
    endpoint: Option<String>,

    /// Decision model id (overrides [decision] model)
    #[arg(long, value_name = "MODEL")]
    model: Option<String>,

    /// Decision backend. `chat` = OpenAI-compatible /v1/chat/completions
    /// (label-only); `systemone` = TypeSafe/Jev /v1/systemone (calibrated
    /// Choice/Noul/Score readout).
    #[arg(long, value_name = "BACKEND")]
    backend: Option<String>,

    /// p(secret) at or above which a candidate is adjudicated `redact`
    #[arg(long, value_name = "THRESHOLD")]
    threshold: Option<f32>,

    /// Max number of candidates to adjudicate (0 = unlimited)
    #[arg(long, value_name = "N")]
    max_candidates: Option<usize>,

    /// Disable NER-based detection (defaults to config; `decide` always runs
    /// the deterministic detector, and enables NER only if configured on)
    #[arg(long)]
    no_ner: bool,

    /// Batch mode: read line-delimited `{"text": ...}` chunks from stdin (or
    /// the input file) and emit one JSON object per line, reusing a single
    /// process so the startup cost is amortized across all chunks.
    #[arg(long)]
    jsonl: bool,

    /// Output as JSON
    #[arg(long)]
    output_json: bool,
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
    /// Show safe effective NER settings (does not load models or prove readiness)
    NerStatus {
        /// Backend-aware model override, as on detect/anon
        #[arg(long)]
        ner_model: Option<String>,
        /// Threshold override, as on detect/anon
        #[arg(long)]
        ner_threshold: Option<f32>,
    },
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
// Command Handlers
// =============================================================================

fn handle_anon(common: &CommonOpts, config: &Config, cmd: AnonCommand) -> Result<()> {
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
    let binary_input = office_fmt.is_some() || is_pdf || img_fmt.is_some();
    let (format, mut reader) = if binary_input {
        if cmd.stream {
            anyhow::bail!("document inputs do not support --stream");
        }
        (FormatArg::Text, None)
    } else {
        let (format, reader) = input::open(cmd.input.as_deref(), cmd.format)?;
        (format, Some(reader))
    };
    if binary_input && cmd.format.is_some() {
        anyhow::bail!(
            "--format overrides are only supported for textual inputs, not document handlers"
        );
    }
    if format == FormatArg::Text
        && (!cmd.include_paths.is_empty() || !cmd.exclude_paths.is_empty() || cmd.json_coverage)
    {
        anyhow::bail!("JSON path selection/coverage requires JSON or JSONL input");
    }
    let mut input_text = String::new();
    if !(cmd.stream && format == FormatArg::Text)
        && format != FormatArg::Jsonl
        && let Some(reader) = reader.as_mut()
    {
        reader.read_to_string(&mut input_text)?;
    }

    // Generate session ID
    let source_filename = cmd
        .input
        .as_ref()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str());
    let session = if let Some(ref tag) = cmd.tag {
        Session::with_tag(tag, source_filename)
    } else if let Some(seed_val) = cmd.seed.or(config.replacement.seed) {
        // A seeded (deterministic) build derives a stable session id so the
        // pseudonym context is reproducible across runs and processes.
        Session::with_seed(seed_val, source_filename)
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

    detector_config = configure_ner(
        detector_config.with_ner(ner_enabled),
        &config.ner,
        cmd.ner_model.as_deref(),
        cmd.ner_threshold,
    )?;
    detector_config = configure_trace(detector_config, config, &cmd.trace)?;
    #[cfg(feature = "streaming")]
    if cmd.stream && format == FormatArg::Text {
        return handle_anon_streaming(
            common,
            config,
            cmd,
            reader.ok_or_else(|| anyhow!("text reader unavailable"))?,
            detector_config,
        );
    }

    let detector = Detector::new(&detector_config)?;
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

    let mut replacer = Replacer::new(replacer_config.clone()).with_session_id(session.id.clone());

    // Load any existing replacement map so consistent/fake strategies reuse the
    // recorded aliases across runs (safe, non-destructive key reuse).
    if let Some(ref key_path) = cmd.key_file {
        let loaded = seed_replacer_from_key_file(key_path, &mut replacer)?;
        if loaded > 0 {
            debug!("Loaded {loaded} existing mapping(s) into replacer");
        }
    }

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
                eprintln!(
                    "No PII detected; unmodified copy written to: {}",
                    out_path.display()
                );
            }
            return Ok(());
        }
        if let Some(ref key_path) = cmd.key_file {
            write_key_file(
                key_path,
                &replacements,
                &session,
                &replacer_config,
                cmd.context.as_deref(),
            )?;
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
                    eprintln!(
                        "No PII recognized; unmodified copy written to: {}",
                        out_path.display()
                    );
                }
            }
            Some(red) => {
                fs::write(&out_path, &red.bytes)
                    .with_context(|| format!("Failed to write output: {}", out_path.display()))?;
                if let Some(ref key_path) = cmd.key_file {
                    write_key_file(
                        key_path,
                        &red.replacements,
                        &session,
                        &replacer_config,
                        cmd.context.as_deref(),
                    )?;
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
        #[cfg_attr(
            not(feature = "ocr"),
            expect(unused_mut, reason = "mutated only with the ocr feature")
        )]
        let (out_bytes, mut replacements, report) =
            engine::pdf::redact(&bytes, &detector, &mut replacer, !cmd.no_strict_pdf)
                .map_err(|e| anyhow!("PDF redaction failed: {e}"))?;

        // Optional OCR pass over raster images in the (already text-redacted) PDF.
        #[cfg_attr(
            not(feature = "ocr"),
            expect(unused_mut, reason = "reassigned only with the ocr feature")
        )]
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
                    ocr_report.scanned, ocr_report.redacted, ocr_report.unsupported
                );
            }
        }
        #[cfg(not(feature = "ocr"))]
        if cmd.ocr {
            return Err(anyhow!(
                "--ocr requires nym to be built with the `ocr` feature"
            ));
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
                eprintln!(
                    "Scrubbed {} metadata/annotation field(s).",
                    report.metadata_scrubbed
                );
            }
        }
        if replacements.is_empty() {
            if !common.quiet {
                eprintln!(
                    "No PII detected; document rewritten to: {}",
                    out_path.display()
                );
            }
            return Ok(());
        }
        if let Some(ref key_path) = cmd.key_file {
            write_key_file(
                key_path,
                &replacements,
                &session,
                &replacer_config,
                cmd.context.as_deref(),
            )?;
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
        FormatArg::Json => {
            let selector = build_path_selector(&cmd.include_paths, &cmd.exclude_paths)?;
            let (out, reps, coverage) =
                process_json_with_selector(&input_text, &detector, &mut replacer, &selector)
                    .with_context(|| "Failed to parse input as JSON")?;
            if cmd.json_coverage {
                print_coverage(&coverage);
            }
            (out, reps)
        }
        FormatArg::Jsonl => {
            let (mut staged, replacements, replacement_count) = stage_jsonl_anon(
                reader
                    .as_mut()
                    .ok_or_else(|| anyhow!("text reader unavailable"))?,
                &detector,
                &mut replacer,
                &cmd,
            )?;
            if let Some(ref key_path) = cmd.key_file {
                write_key_file(
                    key_path,
                    &replacements,
                    &session,
                    &replacer_config,
                    cmd.context.as_deref(),
                )?;
            }
            publish_staged(cmd.output.as_deref(), &mut staged)?;
            if !common.quiet {
                eprintln!("Anonymized {replacement_count} PII occurrences");
            }
            return Ok(());
        }
        FormatArg::Text => {
            let matches = detector.detect(&input_text)?;
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
        write_key_file(
            key_path,
            &replacements,
            &session,
            &replacer_config,
            cmd.context.as_deref(),
        )?;
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
fn handle_anon_streaming(
    common: &CommonOpts,
    config: &Config,
    cmd: AnonCommand,
    reader: input::TextReader,
    detector_config: DetectorConfig,
) -> Result<()> {
    use streaming::{StreamConfig, StreamFormat};
    let ner_enabled = detector_config.ner_enabled;

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
    } else if let Some(seed_val) = cmd.seed.or(config.replacement.seed) {
        Session::with_seed(seed_val, None)
    } else {
        Session::new(None)
    };

    // Structured input uses the complete-scan staging path; this is text only.
    let stream_config = StreamConfig {
        detector_config: detector_config.clone(),
        replacer_config: replacer_config.clone(),
        session_id: Some(session.id.clone()),
        format: StreamFormat::Text,
        seed_mappings: load_seed_mappings(cmd.key_file.as_ref()),
        path_selector: build_path_selector(&cmd.include_paths, &cmd.exclude_paths)?,
        json_coverage: cmd.json_coverage,
    };

    // Create tokio runtime and run
    let rt = tokio::runtime::Runtime::new().with_context(|| "Failed to create async runtime")?;

    let stats = rt.block_on(async {
        let result = run_anon_stream(
            ner_enabled,
            stream_config,
            &replacer_config,
            &session,
            common.quiet,
            input::BlockingReader(reader),
            &cmd,
        )
        .await;
        result.map_err(|e| anyhow!("{e}"))
    })?;

    if !common.quiet {
        eprintln!("Session: {}", session.full_reference());
        eprintln!(
            "Processed {} lines, anonymized {} PII occurrences",
            stats.lines_processed, stats.pii_found
        );
    }

    Ok(())
}

/// Run the anon stream for a given async reader, writing to the output file
/// (or stdout), and persisting any key file safely via the keyfile module.
#[cfg(feature = "streaming")]
#[allow(clippy::too_many_arguments)]
async fn run_anon_stream<R>(
    ner_enabled: bool,
    stream_config: streaming::StreamConfig,
    replacer_config: &ReplacerConfig,
    session: &Session,
    quiet: bool,
    reader: R,
    cmd: &AnonCommand,
) -> Result<streaming::StreamStats>
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    // Do not truncate an existing destination until the complete scan succeeds.
    let staged = cmd
        .output
        .as_ref()
        .map(|path| tempfile::NamedTempFile::new_in(output_parent(path)))
        .transpose()?;
    let writer: Box<dyn tokio::io::AsyncWrite + Unpin + Send> = match staged.as_ref() {
        Some(staged) => Box::new(tokio::fs::File::from_std(staged.reopen()?)),
        None => Box::new(tokio::io::stdout()),
    };

    let stats = if ner_enabled && stream_config.format == streaming::StreamFormat::Text {
        #[cfg(feature = "ner")]
        {
            streaming_ner::stream_anon_ner(stream_config, reader, writer)
                .await
                .map_err(|e| anyhow!("Streaming error: {e}"))?
        }
        #[cfg(not(feature = "ner"))]
        {
            let _ = ner_enabled;
            streaming::stream_anon(stream_config, reader, writer)
                .await
                .map_err(|e| anyhow!("Streaming error: {e}"))?
        }
    } else {
        streaming::stream_anon(stream_config, reader, writer)
            .await
            .map_err(|e| anyhow!("Streaming error: {e}"))?
    };

    // Persist any collected replacements as a key file, non-destructively.
    if let Some(ref key_path) = cmd.key_file {
        if !stats.replacements.is_empty() {
            let header = engine::KeyHeader::new(
                &session.id,
                session.source.as_deref(),
                Some(&format!("{:?}", replacer_config.strategy).to_lowercase()),
                replacer_config.seed,
                cmd.context.as_deref(),
            );
            let existing = engine::load_key_file(key_path)?;
            engine::save_key_file(key_path, existing.as_ref(), &stats.replacements, &header)?;
            if !quiet {
                eprintln!("Key file written to: {}", key_path.display());
            }
        }
    }

    if let (Some(staged), Some(path)) = (staged, cmd.output.as_ref()) {
        staged.as_file().sync_all()?;
        staged
            .persist(path)
            .map_err(|_| anyhow!("failed to publish completed output"))?;
    }
    Ok(stats)
}

/// Full aliases override components; conflicting component originals are omitted.
/// The same filtered dictionary is used for text and office restoration.
fn restoration_mappings(replacements: &[engine::Replacement]) -> Result<Vec<(&str, &str)>> {
    if replacements
        .iter()
        .any(|entry| entry.replacement.is_empty())
    {
        return Err(anyhow!("key file contains an empty restoration alias"));
    }
    let full: std::collections::HashMap<&str, &str> = replacements
        .iter()
        .map(|entry| (entry.replacement.as_str(), entry.original.as_str()))
        .collect();
    let mut components: std::collections::HashMap<&str, Option<&str>> =
        std::collections::HashMap::new();
    for component in replacements.iter().flat_map(|entry| &entry.components) {
        let alias = component.replacement.as_str();
        if alias.is_empty() || full.contains_key(alias) {
            continue;
        }
        let original = component.original.as_str();
        components
            .entry(alias)
            .and_modify(|previous| {
                if *previous != Some(original) {
                    *previous = None;
                }
            })
            .or_insert(Some(original));
    }
    let mut mappings: Vec<_> = full
        .into_iter()
        .chain(
            components
                .into_iter()
                .filter_map(|(alias, original)| original.map(|original| (alias, original))),
        )
        .collect();
    mappings.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(b.0)));
    Ok(mappings)
}

/// Compile once for every decoded leaf/record; inserted originals are never scanned.
struct TextRestorer<'a> {
    originals: std::collections::HashMap<&'a str, &'a str>,
    pattern: regex::Regex,
    verifier: Option<&'a engine::keyfile::restore_verification::RestoreVerifier<'a>>,
}

impl<'a> TextRestorer<'a> {
    fn new(mappings: &[(&'a str, &'a str)]) -> Result<Self> {
        if mappings.is_empty() {
            anyhow::bail!("no restoration aliases");
        }
        let originals = mappings.iter().copied().collect();
        let alternatives: Vec<String> = mappings
            .iter()
            .map(|(alias, _)| {
                let escaped = regex::escape(alias);
                if alias.chars().count() <= 3 {
                    format!(r"\b{escaped}\b")
                } else {
                    escaped
                }
            })
            .collect();
        let pattern = regex::Regex::new(&alternatives.join("|"))
            .map_err(|_| anyhow!("failed to prepare restoration aliases"))?;
        Ok(Self {
            originals,
            pattern,
            verifier: None,
        })
    }

    fn restore(&self, text: &str) -> (String, usize) {
        let mut count = 0;
        let restored = self
            .pattern
            .replace_all(text, |matched: &regex::Captures<'_>| {
                count += 1;
                self.originals[&matched[0]].to_string()
            });
        (restored.into_owned(), count)
    }

    fn restore_verified(&self, text: &str) -> Result<(String, usize)> {
        if let Some(verifier) = self.verifier {
            let spans: Vec<_> = self
                .pattern
                .find_iter(text)
                .map(|matched| matched.range())
                .collect();
            verifier.verify_text(text, &spans)?.ensure_clear()?;
        }
        Ok(self.restore(text))
    }
}

fn restore_json_once(
    input: &str,
    restorer: &TextRestorer<'_>,
) -> Result<(serde_json::Value, usize)> {
    let mut value = serde_json::from_str(input::structured_text(input)).map_err(|error| {
        anyhow!(
            "invalid JSON at line {}, column {}",
            error.line(),
            error.column()
        )
    })?;
    if let Some(verifier) = restorer.verifier {
        verifier.verify_json_keys(&value)?.ensure_clear()?;
    }
    let count = restore_json_value(&mut value, restorer)?;
    Ok((value, count))
}

fn restore_json_value(value: &mut serde_json::Value, restorer: &TextRestorer<'_>) -> Result<usize> {
    match value {
        serde_json::Value::String(text) => {
            let (restored, count) = restorer.restore_verified(text)?;
            *text = restored;
            Ok(count)
        }
        serde_json::Value::Array(values) => values
            .iter_mut()
            .map(|value| restore_json_value(value, restorer))
            .sum(),
        serde_json::Value::Object(values) => values
            .values_mut()
            .map(|value| restore_json_value(value, restorer))
            .sum(),
        _ => Ok(0),
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "CLI command struct consumed by handler"
)]
fn handle_deanon(common: &CommonOpts, cmd: DeanonCommand) -> Result<()> {
    // PDF and raster-image redaction are destructive by design.
    if cmd
        .input
        .as_ref()
        .is_some_and(|p| engine::pdf::is_pdf_path(p))
    {
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

    let (format, mut reader) = if office_fmt.is_some() {
        if cmd.format.is_some() {
            anyhow::bail!(
                "--format overrides are only supported for textual inputs, not document handlers"
            );
        }
        (FormatArg::Text, None)
    } else {
        let (format, reader) = input::open(cmd.input.as_deref(), cmd.format)?;
        (format, Some(reader))
    };
    let input_text = read_non_jsonl(&mut reader, format, false)?;

    // Use the shared loader: malformed or ambiguous mappings must fail before
    // any payload is restored, rather than silently skipping invalid entries.
    let (_, replacements) = engine::keyfile::read_replacements(&cmd.key_file)?;

    if replacements.is_empty() {
        return Err(anyhow!("No replacement mappings found in key file"));
    }

    info!("Loaded {} replacement mappings", replacements.len());

    let verifier = if cmd.no_verify_restore {
        None
    } else {
        Some(engine::keyfile::restore_verification::RestoreVerifier::new(
            &replacements,
        )?)
    };
    let all_mappings = if verifier.is_some() {
        let mut mappings: Vec<_> = replacements
            .iter()
            .map(|entry| (entry.replacement.as_str(), entry.original.as_str()))
            .collect();
        mappings.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(b.0)));
        mappings
    } else {
        restoration_mappings(&replacements)?
    };

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
        let (out_bytes, restored) = if let Some(verifier) = &verifier {
            engine::office::deanonymize_verified(&bytes, fmt, &owned, verifier)
        } else {
            engine::office::deanonymize(&bytes, fmt, &owned)
        }
        .map_err(|_| anyhow!("office restore failed; output not published"))?;
        let out_path = cmd
            .output
            .clone()
            .unwrap_or_else(|| derive_document_output(path, "restored"));
        let mut staged = tempfile::tempfile()?;
        staged.write_all(&out_bytes)?;
        publish_staged(Some(&out_path), &mut staged)?;
        if !common.quiet {
            eprintln!("Restored {restored} PII values -> {}", out_path.display());
        }
        return Ok(());
    }

    let mut restorer = TextRestorer::new(&all_mappings)?;
    restorer.verifier = verifier.as_ref();
    let restored_count = match format {
        FormatArg::Jsonl => {
            let mut staged = tempfile::tempfile()?;
            let mut count = 0;
            for_each_jsonl(
                reader
                    .as_mut()
                    .ok_or_else(|| anyhow!("text reader unavailable"))?,
                |_, text| {
                    let (restored, found) = restore_json_once(text, &restorer)?;
                    serde_json::to_writer(&mut staged, &restored)?;
                    writeln!(staged)?;
                    count += found;
                    Ok(())
                },
            )?;
            publish_staged(cmd.output.as_deref(), &mut staged)?;
            count
        }
        FormatArg::Json => {
            let (restored, count) = restore_json_once(&input_text, &restorer)?;
            write_output(
                cmd.output.as_ref(),
                &serde_json::to_string_pretty(&restored)?,
            )?;
            count
        }
        FormatArg::Text => {
            let (restored, count) = restorer.restore_verified(&input_text)?;
            write_output(cmd.output.as_ref(), &restored)?;
            count
        }
    };

    if !common.quiet {
        eprintln!("Restored {restored_count} PII values");
    }

    Ok(())
}

/// Adjudicate detected spans against a decision model.
///
/// `decide` runs the deterministic detector (regex + optional NER) plus the
/// high-entropy unlabeled-secret backstop, then asks an OpenAI-compatible
/// decision endpoint whether each candidate is a real secret (`redact`),
/// benign (`keep`), or ambiguous (`flag`). It is the residual gate for the
/// secrets and PII that carry no name and no recognisable structure.
///
/// The original input is never rewritten -- `decide` only reports decisions.
#[cfg(feature = "decision")]
fn handle_decide(common: &CommonOpts, config: &Config, cmd: &DecideCommand) -> Result<()> {
    use engine::DecisionGate;

    let (format, mut reader) = input::open(cmd.input.as_deref(), cmd.format)?;
    if !cmd.jsonl && format != FormatArg::Text {
        anyhow::bail!(
            "decide requires --format text or --jsonl text-envelope batch input; structured documents are not supported"
        );
    }
    let mut input_text = String::new();
    reader.read_to_string(&mut input_text)?;

    // Resolve the decision configuration: CLI args override config.
    let mut dc = config.decision.clone();
    if let Some(ref ep) = cmd.endpoint {
        dc.endpoint.clone_from(ep);
    }
    if let Some(ref model) = cmd.model {
        dc.model.clone_from(model);
    }
    if let Some(t) = cmd.threshold {
        dc.threshold = t;
    }
    if let Some(n) = cmd.max_candidates {
        dc.max_candidates = n;
    }
    dc.enabled = true;
    dc.api_key_env.clone_from(&config.decision.api_key_env);
    if let Some(ref b) = cmd.backend {
        dc.backend.clone_from(b);
    }

    // Batch mode: process each line of `{"text": ...}` input in one process,
    // emitting one JSON object per line. Amortizes startup + detector init
    // across every chunk instead of spawning a process per chunk.
    if cmd.jsonl {
        let gate = DecisionGate::new(dc);
        // Build the Detector (and its NER model) ONCE and reuse it across every
        // chunk. Previously each chunk re-created the Detector, re-loading the
        // ONNX NER model (~2s) per line -- the real per-call cost.
        let detector = build_decide_detector(config, cmd)?;
        // Collect all chunks, run the NER/token detector in one batched padded
        // forward pass, then adjudicate each span per chunk. The batched NER
        // amortizes ONNX per-call overhead across the whole input.
        let mut chunks: Vec<String> = Vec::new();
        for (line_number, line) in input::structured_text(&input_text).lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let invalid = || {
                anyhow!(
                    "invalid text-envelope JSONL record at line {}",
                    line_number + 1
                )
            };
            let value: serde_json::Value = serde_json::from_str(line).map_err(|_| invalid())?;
            let text = value
                .get("text")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(invalid)?;
            chunks.push(text.to_owned());
        }
        let chunk_refs: Vec<&str> = chunks.iter().map(std::string::String::as_str).collect();
        let all_matches = detector.detect_batch(&chunk_refs)?;
        for (i, chunk) in chunks.iter().enumerate() {
            let matches = all_matches.get(i).cloned().unwrap_or_default();
            let candidates = gate.candidates(chunk, &matches);
            let decisions = gate.adjudicate(chunk, candidates)?;
            println!("{}", serde_json::to_string(&decisions)?);
        }
        return Ok(());
    }

    let gate = DecisionGate::new(dc);
    let detector = build_decide_detector(config, cmd)?;
    let decisions = decide_one(&gate, &detector, &input_text)?;

    if cmd.output_json || common.json {
        println!("{}", serde_json::to_string_pretty(&decisions)?);
    } else {
        for d in &decisions {
            let verdict = match d.verdict {
                engine::Verdict::Redact => "redact",
                engine::Verdict::Keep => "keep",
                engine::Verdict::Flag => "flag",
            };
            let span = match (d.start, d.end) {
                (Some(s), Some(e)) => format!("{s}..{e}"),
                _ => "?".to_string(),
            };
            println!(
                "[{verdict:6}] {span:>12}  {:<12}  conf={:.2}  {}",
                d.class, d.confidence, d.text
            );
        }
    }
    Ok(())
}

/// Run detection + adjudication for one chunk and return its decisions.
#[cfg(feature = "decision")]
fn decide_one(
    gate: &engine::DecisionGate,
    detector: &Detector,
    input_text: &str,
) -> Result<Vec<engine::Decision>> {
    let matches = detector.detect(input_text)?;
    let candidates = gate.candidates(input_text, &matches);
    gate.adjudicate(input_text, candidates)
}

/// Build the detector configuration used by `decide`/`detect` from the config
/// and CLI flags (enabled/disabled patterns, min confidence, NER backend).
#[cfg(feature = "decision")]
fn build_decide_detector(config: &Config, cmd: &DecideCommand) -> Result<Detector> {
    let min_confidence = config.detection.min_confidence.into();
    let mut detector_config = DetectorConfig::default().with_min_confidence(min_confidence);
    if !config.detection.enabled_patterns.is_empty() {
        detector_config =
            detector_config.with_patterns(config.detection.enabled_patterns.iter().cloned());
    }
    if !config.detection.disabled_patterns.is_empty() {
        detector_config =
            detector_config.without_patterns(config.detection.disabled_patterns.iter().cloned());
    }
    let ner_enabled = !cmd.no_ner && config.ner.enabled;
    detector_config = detector_config.with_ner(ner_enabled);
    detector_config = configure_ner(detector_config, &config.ner, None, None)?;
    detector_config = configure_trace(detector_config, config, &cmd.trace)?;
    Ok(Detector::new(&detector_config)?)
}

fn handle_detect(common: &CommonOpts, config: &Config, cmd: DetectCommand) -> Result<()> {
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
    #[cfg(feature = "ocr")]
    let is_img_input = sniff_image(cmd.input.as_ref()).is_some();
    #[cfg(not(feature = "ocr"))]
    let is_img_input = false;
    let binary_input = office_fmt.is_some() || is_pdf || is_img_input;
    let (format, mut reader) = if binary_input {
        if cmd.stream {
            anyhow::bail!("document inputs do not support --stream");
        }
        (FormatArg::Text, None)
    } else {
        let (format, reader) = input::open(cmd.input.as_deref(), cmd.format)?;
        (format, Some(reader))
    };
    if binary_input && cmd.format.is_some() {
        anyhow::bail!(
            "--format overrides are only supported for textual inputs, not document handlers"
        );
    }
    if format == FormatArg::Text && (!cmd.include_paths.is_empty() || !cmd.exclude_paths.is_empty())
    {
        anyhow::bail!("JSON path selection requires JSON or JSONL input");
    }
    #[cfg(feature = "streaming")]
    if cmd.stream {
        // Explicit streaming audit combinations are rejected, never silently ignored.
        if cmd.json_coverage || !cmd.fail_on.is_empty() || cmd.summary_json || cmd.summary {
            anyhow::bail!(
                "--json-coverage, --fail-on and summaries are not supported with detect --stream; omit --stream to use them"
            );
        }
        if common.json || common.yaml {
            anyhow::bail!(
                "machine-readable output is not supported with detect --stream; omit --stream"
            );
        }
    }
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
        #[cfg_attr(
            not(feature = "ocr"),
            expect(unused_mut, reason = "mutated only with the ocr feature")
        )]
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
            return Err(anyhow!(
                "--ocr requires nym to be built with the `ocr` feature"
            ));
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
            read_non_jsonl(&mut reader, format, cmd.stream)?
        }
        #[cfg(not(feature = "ocr"))]
        read_non_jsonl(&mut reader, format, cmd.stream)?
    };
    if format == FormatArg::Text
        && (!cmd.include_paths.is_empty() || !cmd.exclude_paths.is_empty() || cmd.json_coverage)
    {
        anyhow::bail!("JSON path selection/coverage requires JSON or JSONL input");
    }

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

    detector_config = configure_ner(
        detector_config.with_ner(ner_enabled),
        &config.ner,
        cmd.ner_model.as_deref(),
        cmd.ner_threshold,
    )?;
    detector_config = configure_trace(detector_config, config, &cmd.trace)?;
    #[cfg(feature = "streaming")]
    if cmd.stream && format != FormatArg::Json {
        return handle_detect_streaming(
            common,
            cmd,
            reader.ok_or_else(|| anyhow!("text reader unavailable"))?,
            format,
            detector_config,
        );
    }

    let detector = Detector::new(&detector_config)?;

    // Validate the complete policy before emitting findings or a clean manifest.
    let fail_policy = engine::FailOnPolicy::new(
        &cmd.fail_on,
        config
            .ner
            .effective_gliner_labels()
            .iter()
            .filter(|_| ner_enabled)
            .map(String::as_str),
    )?;

    // Detect based on format
    match format {
        FormatArg::Json => {
            let selector = build_path_selector(&cmd.include_paths, &cmd.exclude_paths)?;
            let (json_matches, coverage, trace_stats) =
                engine::formats::detect_json_with_stats(&input_text, &detector, &selector)
                    .with_context(|| "Failed to parse input as JSON")?;
            if cmd.json_coverage {
                print_coverage(&coverage);
            }

            let findings: Vec<(&str, PiiCategory)> = json_matches
                .iter()
                .map(|m| (m.pii_match.pattern_name.as_str(), m.pii_match.category))
                .collect();
            let mut summary = engine::AuditSummary::from_findings(findings, &fail_policy);
            summary.trace_policy = detector_config.trace_policy.as_ref().map(|_| trace_stats);

            if cmd.summary_json {
                println!("{}", engine::to_summary_json(&summary)?);
            } else if cmd.summary {
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

            if summary.blocked() {
                return Err(anyhow::Error::new(ExitError(2)).context(
                    "audit gate: sensitive findings present (use --summary-json for a value-free manifest)",
                ));
            }
        }
        FormatArg::Jsonl => {
            detect_jsonl(
                common,
                &cmd,
                reader
                    .as_mut()
                    .ok_or_else(|| anyhow!("text reader unavailable"))?,
                &detector,
                &fail_policy,
                detector_config.trace_policy.is_some(),
            )?;
        }
        FormatArg::Text => {
            let detected = detector.detect_with_stats(&input_text)?;
            let trace_stats = detected.stats;
            let matches = detected.matches;
            let findings: Vec<(&str, PiiCategory)> = matches
                .iter()
                .map(|m| (m.pattern_name.as_str(), m.category))
                .collect();
            let mut summary = engine::AuditSummary::from_findings(findings, &fail_policy);
            summary.trace_policy = detector_config.trace_policy.as_ref().map(|_| trace_stats);

            if cmd.summary_json {
                println!("{}", engine::to_summary_json(&summary)?);
            } else if cmd.summary {
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

            if summary.blocked() {
                return Err(anyhow::Error::new(ExitError(2)).context(
                    "audit gate: sensitive findings present (use --summary-json for a value-free manifest)",
                ));
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
fn handle_detect_streaming(
    common: &CommonOpts,
    cmd: DetectCommand,
    reader: input::TextReader,
    format: FormatArg,
    detector_config: DetectorConfig,
) -> Result<()> {
    use streaming::StreamConfig;
    let trace_enabled = detector_config.trace_policy.is_some();

    let stream_config = StreamConfig {
        detector_config,
        replacer_config: ReplacerConfig::default(),
        session_id: None,
        format: if format == FormatArg::Text {
            streaming::StreamFormat::Text
        } else {
            streaming::StreamFormat::Json
        },
        seed_mappings: Vec::new(),
        path_selector: build_path_selector(&cmd.include_paths, &cmd.exclude_paths)?,
        json_coverage: false,
    };

    // Create tokio runtime and run
    let rt = tokio::runtime::Runtime::new().with_context(|| "Failed to create async runtime")?;

    let stats = rt.block_on(async {
        streaming::stream_detect(
            stream_config,
            input::BlockingReader(reader),
            tokio::io::stdout(),
        )
        .await
        .map_err(|e| anyhow!("Streaming error: {e}"))
    })?;

    if !common.quiet {
        eprintln!(
            "Processed {} lines, found {} PII occurrences",
            stats.lines_processed, stats.pii_found
        );
        if trace_enabled {
            eprintln!(
                "Trace policy counts: {}",
                serde_json::to_string(&stats.trace_stats)?
            );
        }
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
        let output: Vec<_> = matches.iter().map(json_match_value).collect();
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else if common.yaml {
        let output: Vec<_> = matches.iter().map(json_match_value).collect();
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

fn print_ner_status(
    common: &CommonOpts,
    ner: &config::NerConfig,
    model: Option<&str>,
    threshold: Option<f32>,
) -> Result<()> {
    let mut ner = ner.clone();
    if let Some(threshold) = threshold {
        ner.threshold = threshold;
    }
    let status = ner.resolved(model)?.safe_status()?;
    if common.yaml {
        println!("{}", serde_yaml::to_string(&status)?);
    } else {
        println!("{}", serde_json::to_string_pretty(&status)?);
    }
    Ok(())
}

fn handle_config(common: &CommonOpts, config: &Config, cmd: ConfigCommand) -> Result<()> {
    match cmd.action.unwrap_or(ConfigAction::Show) {
        ConfigAction::NerStatus {
            ner_model,
            ner_threshold,
        } => {
            print_ner_status(common, &config.ner, ner_model.as_deref(), ner_threshold)?;
        }
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

/// Human-readable download size (GB above 1024 MB, else MB) — the on-disk
/// footprint, not a parameter count.
#[cfg(feature = "ner")]
fn fmt_size(mb: u32) -> String {
    if mb >= 1024 {
        format!("{:.1} GB", f64::from(mb) / 1024.0)
    } else {
        format!("{mb} MB")
    }
}

#[cfg(feature = "ner")]
fn model_row(m: &CatalogModel) -> String {
    format!(
        "{:<42} {:<32} {:<7} {:<19} {:>8}  {}",
        m.slug,
        m.name,
        m.backend,
        m.languages,
        fmt_size(m.size_mb),
        m.description
    )
}

#[cfg(feature = "ner")]
fn models_list(common: &CommonOpts, config: &Config) -> Result<()> {
    use engine::detector::NerBackend;
    let models = model_catalog::load();
    let cache_dir = ner_cache_dir(config);
    let (tok_default, gli_default) = current_defaults(config);
    let tokens_active = matches!(
        config.ner.backend,
        NerBackend::TokenClass | NerBackend::Both
    );
    let gliner_active = matches!(config.ner.backend, NerBackend::Gliner | NerBackend::Both);

    if common.json {
        println!("{}", serde_json::to_string_pretty(&models)?);
        return Ok(());
    }

    println!(
        "   {:<42} {:<32} {:<7} {:<19} {:>8}  DESCRIPTION",
        "SLUG", "NAME", "BACKEND", "LANGUAGES", "DISK"
    );
    for m in &models {
        let is_default = (m.is_tokens() && tokens_active && m.slug == tok_default)
            || (!m.is_tokens() && gliner_active && m.slug == gli_default);
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
        _ => {
            if let Some(m) = pick_catalog(&downloaded, query)? {
                apply_default_model(config, &m)
            } else {
                eprintln!("nothing selected");
                Ok(())
            }
        }
    }
}

/// Persist a model as the default, switching to its backend and enabling NER.
#[cfg(feature = "ner")]
fn apply_default_model(config: &Config, m: &CatalogModel) -> Result<()> {
    let cache_dir = ner_cache_dir(config);
    let backend = if m.is_tokens() { "tokens" } else { "gliner" };
    let path = set_default_model(m)?;
    let field = if m.is_tokens() {
        "token_model"
    } else {
        "model"
    };
    println!("[ner] {field} = \"{}\"", m.slug);
    eprintln!("[ner] backend = \"{backend}\"");
    eprintln!("[ner] enabled = true");
    eprintln!("wrote {}", path.display());
    eprintln!(
        "(now running only the {backend} backend; set [ner] backend = \"both\" to also run the other)"
    );
    if !m.is_cached(cache_dir.as_deref()) {
        eprintln!(
            "(not downloaded yet — it will fetch on first use, or run `nym models pull {}`)",
            m.slug
        );
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
    // Switch to the selected model's backend so `use X` runs X alone, rather
    // than also paying for the other backend (e.g. the 1.1 GB GLiNER model).
    ner["backend"] = value(if m.is_tokens() { "tokens" } else { "gliner" });
    let field = if m.is_tokens() {
        "token_model"
    } else {
        "model"
    };
    ner[field] = value(m.slug.clone());

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
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
    if let Ok(mut child) = cmd.spawn() {
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
    } else {
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
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("bin");
    path.with_file_name(format!("{stem}.{tag}.{ext}"))
}

fn write_output(path: Option<&PathBuf>, content: &str) -> Result<()> {
    let mut staged = tempfile::tempfile()?;
    staged.write_all(content.as_bytes())?;
    publish_staged(path.map(PathBuf::as_path), &mut staged)
}

/// Persist a key file non-destructively.
///
/// Loads and preserves any existing replacement map, validates compatibility
/// with the current strategy/seed/context, and writes atomically under a lock
/// so a second invocation extends (never truncates) the file.
fn write_key_file(
    path: &PathBuf,
    replacements: &[engine::Replacement],
    session: &Session,
    replacer_config: &ReplacerConfig,
    context: Option<&str>,
) -> Result<()> {
    let strategy = format!("{:?}", replacer_config.strategy).to_lowercase();
    let header = engine::KeyHeader::new(
        &session.id,
        session.source.as_deref(),
        Some(&strategy),
        replacer_config.seed,
        context,
    );
    let existing = engine::load_key_file(path)?;
    engine::save_key_file(path, existing.as_ref(), replacements, &header).map(|_| ())
}

/// Load an existing replacement map and seed the replacer so consistent/fake
/// strategies reuse recorded aliases across runs and processes. Returns the
/// number of mappings loaded (0 when the file is absent or empty).
fn seed_replacer_from_key_file(key_path: &PathBuf, replacer: &mut Replacer) -> Result<usize> {
    let mappings = load_seed_mappings(Some(key_path));
    replacer.seed_mappings(&mappings);
    Ok(mappings.len())
}

/// Load the replacement mappings recorded in a key file (empty when absent).
fn load_seed_mappings(key_path: Option<&PathBuf>) -> Vec<engine::Replacement> {
    let Some(key_path) = key_path else {
        return Vec::new();
    };
    engine::load_key_file(key_path)
        .ok()
        .flatten()
        .map(|kf| kf.replacements)
        .unwrap_or_default()
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

fn read_non_jsonl(
    reader: &mut Option<input::TextReader>,
    format: FormatArg,
    streaming: bool,
) -> Result<String> {
    let mut text = String::new();
    if !(format == FormatArg::Jsonl || streaming && format == FormatArg::Text)
        && let Some(reader) = reader.as_mut()
    {
        reader.read_to_string(&mut text)?;
    }
    Ok(text)
}

/// Per-record memory is capped; prior bytes are never discarded during sniffing.
fn for_each_jsonl(
    reader: &mut input::TextReader,
    mut visit: impl FnMut(usize, &str) -> Result<()>,
) -> Result<()> {
    const MAX_RECORD_BYTES: u64 = 16 * 1024 * 1024;
    let mut line = String::new();
    let mut line_number = 0;
    loop {
        line.clear();
        let count = reader
            .by_ref()
            .take(MAX_RECORD_BYTES + 1)
            .read_line(&mut line)
            .map_err(|_| anyhow!("failed to read JSONL line {}", line_number + 1))?;
        if count == 0 {
            break;
        }
        line_number += 1;
        if count as u64 > MAX_RECORD_BYTES {
            anyhow::bail!("JSONL line {line_number} exceeds the 16 MiB record limit");
        }
        let text = if line_number == 1 {
            input::structured_text(&line)
        } else {
            line.trim()
        };
        if text.trim().is_empty() {
            continue;
        }
        serde_json::from_str::<serde_json::Value>(text).map_err(|err| {
            anyhow!(
                "invalid JSONL record at line {line_number}, column {}",
                err.column()
            )
        })?;
        visit(line_number, text)
            .with_context(|| format!("JSONL processing failed at line {line_number}"))?;
    }
    Ok(())
}

fn output_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

/// Publish only a completed result. Spooling is owner-only and outside git.
fn publish_staged(path: Option<&Path>, staged: &mut fs::File) -> Result<()> {
    use std::io::{Seek, SeekFrom};
    staged.seek(SeekFrom::Start(0))?;
    if let Some(path) = path {
        let mut output = tempfile::NamedTempFile::new_in(output_parent(path))?;
        io::copy(staged, output.as_file_mut())?;
        output.as_file().sync_all()?;
        output
            .persist(path)
            .map_err(|_| anyhow!("failed to publish completed output"))?;
    } else {
        io::copy(staged, &mut io::stdout().lock())?;
        io::stdout().flush()?;
    }
    Ok(())
}

fn print_record_coverage(line: usize, coverage: &engine::CoverageReport) {
    let coverage = engine::CoverageReport {
        scanned: coverage
            .scanned
            .iter()
            .map(|path| format!("record[{line}].{path}"))
            .collect(),
        skipped: coverage
            .skipped
            .iter()
            .map(|path| format!("record[{line}].{path}"))
            .collect(),
    };
    print_coverage(&coverage);
}

fn stage_jsonl_anon(
    reader: &mut input::TextReader,
    detector: &Detector,
    replacer: &mut Replacer,
    cmd: &AnonCommand,
) -> Result<(fs::File, Vec<engine::Replacement>, usize)> {
    let selector = build_path_selector(&cmd.include_paths, &cmd.exclude_paths)?;
    let mut staged = tempfile::tempfile()?;
    let mut replacements = Vec::new();
    let mut count = 0;
    for_each_jsonl(reader, |line, text| {
        let (out, reps, coverage) =
            process_json_with_selector(text, detector, replacer, &selector)?;
        let value: serde_json::Value = serde_json::from_str(&out)?;
        serde_json::to_writer(&mut staged, &value)?;
        writeln!(staged)?;
        count += reps.len();
        if cmd.key_file.is_some() {
            replacements.extend(reps);
        }
        if cmd.json_coverage {
            print_record_coverage(line, &coverage);
        }
        Ok(())
    })?;
    Ok((staged, replacements, count))
}

fn detect_jsonl(
    common: &CommonOpts,
    cmd: &DetectCommand,
    reader: &mut input::TextReader,
    detector: &Detector,
    policy: &engine::FailOnPolicy,
    trace_enabled: bool,
) -> Result<()> {
    let selector = build_path_selector(&cmd.include_paths, &cmd.exclude_paths)?;
    let mut summary = engine::AuditSummary::from_findings(std::iter::empty(), policy);
    summary.trace_policy = trace_enabled.then(engine::TracePolicyStats::default);
    let mut staged = tempfile::tempfile()?;
    let mut first = true;
    if common.json {
        write!(staged, "[")?;
    }
    for_each_jsonl(reader, |line, text| {
        let (matches, coverage, trace_stats) =
            engine::formats::detect_json_with_stats(text, detector, &selector)?;
        if let Some(total) = &mut summary.trace_policy {
            engine::audit::merge_trace_stats(total, trace_stats);
        }
        let partial = engine::AuditSummary::from_findings(
            matches
                .iter()
                .map(|m| (m.pii_match.pattern_name.as_str(), m.pii_match.category)),
            policy,
        );
        merge_audit_summary(&mut summary, partial);
        if cmd.json_coverage {
            print_record_coverage(line, &coverage);
        }
        if !cmd.summary && !cmd.summary_json {
            write_record_findings(&mut staged, common, matches, line, &mut first)?;
        }
        Ok(())
    })?;
    if cmd.summary_json {
        println!("{}", engine::to_summary_json(&summary)?);
    } else if cmd.summary {
        let counts = serde_json::json!({"total":summary.total,"by_pattern":summary.by_pattern});
        if common.json {
            println!("{}", serde_json::to_string_pretty(&counts)?);
        } else if common.yaml {
            println!("{}", serde_yaml::to_string(&counts)?);
        } else {
            println!(
                "PII Detection Summary (JSONL):\n  Total matches: {}",
                summary.total
            );
            for (name, count) in &summary.by_pattern {
                println!("  {name}: {count}");
            }
        }
    } else {
        if common.json {
            writeln!(staged, "]")?;
        } else if common.yaml && first {
            writeln!(staged, "[]")?;
        } else if !common.yaml {
            writeln!(staged, "Total: {} matches", summary.total)?;
        }
        publish_staged(None, &mut staged)?;
    }
    if summary.blocked() {
        return Err(
            anyhow::Error::new(ExitError(2)).context("audit gate: sensitive findings present")
        );
    }
    Ok(())
}

fn merge_audit_summary(summary: &mut engine::AuditSummary, partial: engine::AuditSummary) {
    summary.total += partial.total;
    for (name, count) in partial.by_pattern {
        *summary.by_pattern.entry(name).or_default() += count;
    }
    for (name, count) in partial.by_category {
        *summary.by_category.entry(name).or_default() += count;
    }
    summary.blockers.extend(partial.blockers);
    summary.blockers.sort();
    summary.blockers.dedup();
}

fn json_match_value(m: &JsonPiiMatch) -> serde_json::Value {
    serde_json::json!({"path":m.path,"pattern_name":m.pii_match.pattern_name,"matched_text":m.pii_match.matched_text,
        "start":m.pii_match.start,"end":m.pii_match.end,"confidence":m.pii_match.confidence,"category":m.pii_match.category})
}

fn write_record_findings(
    staged: &mut fs::File,
    common: &CommonOpts,
    matches: Vec<JsonPiiMatch>,
    line: usize,
    first: &mut bool,
) -> Result<()> {
    for mut m in matches {
        m.path = format!("record[{line}].{}", m.path);
        if common.json {
            if !*first {
                write!(staged, ",")?;
            }
            serde_json::to_writer(&mut *staged, &json_match_value(&m))?;
        } else if common.yaml {
            write!(
                staged,
                "{}",
                serde_yaml::to_string(&vec![json_match_value(&m)])?
            )?;
        } else {
            writeln!(
                staged,
                "[{}] '{}' at {} ({:?})",
                m.pii_match.pattern_name, m.pii_match.matched_text, m.path, m.pii_match.confidence
            )?;
        }
        *first = false;
    }
    Ok(())
}

/// Build a JSON path selector from CLI include/exclude flags. An empty
/// selector scans everything (the historical default).
fn build_path_selector(include: &[String], exclude: &[String]) -> Result<engine::PathSelector> {
    engine::PathSelector::new(include, exclude).map_err(|e| anyhow!("invalid path selector: {e}"))
}

/// Report private scanned/skipped paths on stderr, never the payload channel.
fn print_coverage(coverage: &engine::CoverageReport) {
    eprintln!("JSON scan coverage:");
    eprintln!("  scanned: {} path(s)", coverage.scanned_count());
    for p in &coverage.scanned {
        eprintln!("    + {p}");
    }
    eprintln!("  skipped: {} path(s)", coverage.skipped_count());
    for p in &coverage.skipped {
        eprintln!("    - {p}");
    }
}

#[cfg(all(test, feature = "ner"))]
mod jsonl_runtime_tests {
    use super::*;
    use engine::detector::{NerBackend, tests::injected};

    fn reader() -> input::TextReader {
        Box::new(io::Cursor::new(
            b"{\"text\":\"completed@example.com\"}\n{\"text\":\"Alice confidential\"}\n".to_vec(),
        ))
    }

    #[test]
    fn failed_later_record_is_exit_one_not_a_manifest_or_published_sanitization() {
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join("destination.jsonl");
        fs::write(&destination, "existing destination").unwrap();
        let anon = Cli::try_parse_from([
            "nym",
            "anon",
            "--no-ner",
            "-o",
            destination.to_str().unwrap(),
        ])
        .unwrap();
        let Command::Anon(anon) = anon.command else {
            unreachable!()
        };
        let cli = Cli::try_parse_from([
            "nym",
            "detect",
            "--no-ner",
            "--summary-json",
            "--fail-on",
            "identity",
        ])
        .unwrap();
        let Command::Detect(detect) = cli.command else {
            unreachable!()
        };
        let policy = engine::FailOnPolicy::new(&detect.fail_on, std::iter::empty()).unwrap();
        for backend in [NerBackend::Gliner, NerBackend::TokenClass, NerBackend::Both] {
            let detector = injected(backend, Some("Alice"), false);
            let error = detect_jsonl(
                &cli.common,
                &detect,
                &mut reader(),
                &detector,
                &policy,
                false,
            )
            .unwrap_err();
            assert_eq!(error_exit_code(&error), 1);
            assert!(format!("{error:?}").contains("line 2"));
            assert!(!format!("{error:?}").contains("Alice"));
            let mut replacer = Replacer::new(ReplacerConfig::default());
            let error =
                stage_jsonl_anon(&mut reader(), &detector, &mut replacer, &anon).unwrap_err();
            assert_eq!(error_exit_code(&error), 1);
            assert!(!format!("{error:?}").contains("secret-provider-key"));
            assert_eq!(
                fs::read_to_string(&destination).unwrap(),
                "existing destination"
            );
            let detector = injected(backend, None, true);
            let error: anyhow::Error = detector
                .detect_batch(&["first record", "second confidential record"])
                .unwrap_err()
                .into();
            assert_eq!(error_exit_code(&error), 1);
            assert!(!format!("{error:?}").contains("confidential"));
        }
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
