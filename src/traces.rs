//! Per-session agent-trace anonymization.
//!
//! Every session file gets its own replacer, so pseudonyms are consistent
//! within a session and independent across sessions. Detection is tiered for
//! speed: opaque fields (images, signatures, identifiers) are never scanned,
//! regex and identity literals run on every other string, and NER runs on
//! prose only (unless `--ner-scope all`). Every value found anywhere in a
//! session is then replaced at all of its occurrences in that session.
use crate::{
    config::Config,
    engine::{Detector, DetectorConfig, PiiMatch, Replacer, ReplacerConfig},
    session::Session,
};
use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Subcommand, ValueEnum};
use rayon::prelude::*;
use serde::Serialize;
use serde_json::Value;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    io::{IsTerminal, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

#[derive(Args, Debug)]
pub struct TracesCommand {
    #[command(subcommand)]
    action: Action,
}

#[derive(Subcommand, Debug)]
enum Action {
    /// Anonymize agent session JSONL files, one consistent pseudonym map per session
    Anon(AnonArgs),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum NerScope {
    /// Prose, plus tool-output lines with name, address or account signals
    Auto,
    /// User/assistant text, thinking and summaries only
    Prose,
    /// Fastest NER: only lines with name signals, in prose and tool output alike
    Lines,
    /// Every scanned string, including tool output (slower)
    All,
}

#[derive(Args, Debug)]
struct AnonArgs {
    /// Session files or directories (directories are searched recursively for *.jsonl)
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    /// Output directory; mirrors the input layout. Must not already contain outputs unless --force
    #[arg(short, long, required_unless_present = "plan")]
    output: Option<PathBuf>,
    /// Write the values that would be replaced (per session, with class, source,
    /// count and a short context) to this private JSON file instead of writing
    /// outputs. Contains originals: keep it private.
    #[arg(long, conflicts_with_all = ["output", "keys"])]
    plan: Option<PathBuf>,
    /// UTF-8 file of exact values never to replace, one per line (repeatable).
    /// Explicit user decisions: unlike allow lists, these also apply to
    /// credentials and identity terms.
    #[arg(long)]
    keep_values_file: Vec<PathBuf>,
    /// Replace exactly the values in this plan file (the `--plan` format, e.g.
    /// after review), everywhere in each session, without running detectors.
    /// Sessions the plan does not cover fail.
    #[arg(long, conflicts_with = "plan")]
    apply_plan: Option<PathBuf>,
    /// Do not read or write the NER result cache ($XDG_CACHE_HOME/nym/ner)
    #[arg(long)]
    no_cache: bool,
    /// Write one key file per session into this directory, for later `nym deanon`
    #[arg(long)]
    keys: Option<PathBuf>,
    /// Replacement strategy; per-session consistent strategies only
    #[arg(long, value_enum, default_value = "fake")]
    strategy: TraceStrategy,
    /// Disable NER (regex, identity literals and propagation only; fastest)
    #[arg(long)]
    no_ner: bool,
    /// Which strings NER runs on
    #[arg(long, value_enum, default_value = "auto")]
    ner_scope: NerScope,
    /// Remove embedded base64 images (screenshots may show private content NYM
    /// cannot inspect). Images are kept unchanged by default.
    #[arg(long)]
    strip_images: bool,
    /// Do not add the local account/git/host identity to the sensitive literals
    #[arg(long)]
    no_local_identity: bool,
    /// UTF-8 file of additional sensitive literals, one per line (repeatable)
    #[arg(long)]
    sensitive_terms_file: Vec<PathBuf>,
    /// UTF-8 file of literals never to replace, one per line (repeatable). Never
    /// vetoes credentials, local identity or sensitive terms.
    #[arg(long)]
    allow_terms_file: Vec<PathBuf>,
    /// Add a redaction profile's sensitive and allow lists (repeatable); the
    /// `default` profile always applies when present. See `nym profiles`.
    #[arg(long = "redaction-profile", value_name = "NAME")]
    redaction_profiles: Vec<String>,
    /// Base seed for reproducible pseudonyms (each session derives its own seed)
    #[arg(long)]
    seed: Option<u64>,
    /// Parallel sessions (default: available CPUs, at most 8 with NER)
    #[arg(short, long)]
    jobs: Option<usize>,
    /// Replace existing output and key files
    #[arg(long)]
    force: bool,
    /// Disable stderr progress
    #[arg(long)]
    no_progress: bool,
    /// Emit machine-readable progress on stderr, one JSON object per line:
    /// `{"nym_progress":{"phase":"detect|ner|replace|done","done":N,"total":M}}`
    /// (`detect`/`ner` count text bytes, `replace` string fields). For tools
    /// such as pi-nym; independent of --no-progress.
    #[arg(long)]
    progress_json: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum TraceStrategy {
    /// Realistic fake values (names stay names, emails stay emails)
    Fake,
    /// Hash-derived values
    Hash,
}

/// String-leaf roles, decided from the key name and record context.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Skip,
    Image,
    Prose,
    Other,
}

/// Keys whose values are identifiers, enums or opaque encodings.
const STRUCTURAL: &[&str] = &[
    "id",
    "parentId",
    "timestamp",
    "type",
    "role",
    "toolCallId",
    "toolName",
    "model",
    "modelId",
    "provider",
    "api",
    "stopReason",
    "mimeType",
    "version",
    "customType",
    "responseId",
    "thinkingLevel",
    "thinkingSignature",
    "signature",
    "encrypted_content",
    "textSignature",
];
const PROSE: &[&str] = &["text", "thinking", "summary", "message"];
/// Shared service/system accounts are not personal identities.
const GENERIC_ACCOUNTS: &[&str] = &[
    "root",
    "admin",
    "administrator",
    "user",
    "guest",
    "nobody",
    "daemon",
    "www-data",
    "ubuntu",
    "debian",
    "ec2-user",
    "runner",
    "vscode",
    "node",
    "app",
    "codespace",
    "jovyan",
    "postgres",
    "mysql",
    "git",
    "localhost",
    "raspberrypi",
    "sudo",
    "system",
];
/// Finding classes that are not identities; never spread to other occurrences.
const NO_PROPAGATE: &[&str] = &[
    "date",
    "time",
    "url",
    "unix_path",
    "uuid",
    "number",
    "age",
    "gender",
];
/// Noise for agent traces: system paths, generated IDs and timestamps.
const TRACE_EXCLUDED: &[&str] = &["unix_path", "uuid", "date", "time"];

/// Lines in tool output worth a NER pass: capitalized word pairs (names,
/// places), postcode + capitalized word (addresses) and account labels.
static CANDIDATE_LINE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(
        r"\b\p{Lu}[\p{Ll}\p{M}]+[ \t]+\p{Lu}[\p{Ll}\p{M}]+|\b\d{4,5}[ \t]+\p{Lu}[\p{Ll}\p{M}]+|(?i)\b(?:user(?:name)?|login|handle|account|author|owner|contact)\s*[:=]",
    )
    .expect("candidate line pattern is valid")
});
/// Prose lines also qualify on a mid-sentence capitalized word ("talked to Anna").
static PROSE_CANDIDATE_LINE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"[\p{Ll}\p{N},;:)][ \t]+\p{Lu}[\p{Ll}\p{M}]+")
        .expect("prose candidate pattern is valid")
});
/// Longer single lines are data (minified JSON, base64), not natural text.
const MAX_CANDIDATE_LINE: usize = 2000;

#[derive(Default, Serialize, Clone)]
struct SessionStats {
    records: usize,
    strings: usize,
    scanned_bytes: usize,
    ner_bytes: usize,
    images_removed: usize,
    replacements: usize,
    propagated: usize,
    identity: usize,
    by_class: BTreeMap<String, usize>,
}

#[derive(Serialize)]
struct Summary {
    sessions: usize,
    failed: usize,
    input_bytes: u64,
    seconds: f64,
    megabytes_per_second: f64,
    #[serde(flatten)]
    totals: SessionStats,
}

/// Per-session `--progress-json` reporter, throttled to about ten lines a second.
struct Progress {
    enabled: bool,
    last: Instant,
}

impl Progress {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            last: Instant::now(),
        }
    }

    fn report(&mut self, phase: &str, done: usize, total: usize, force: bool) {
        if !self.enabled || (!force && self.last.elapsed().as_millis() < 100) {
            return;
        }
        self.last = Instant::now();
        // One locked write per line: parallel sessions never interleave a line.
        eprintln!(
            "{}",
            serde_json::json!({ "nym_progress": { "phase": phase, "done": done, "total": total } })
        );
    }
}

/// One value that would be replaced, for `--plan` review.
#[derive(Serialize, Clone)]
struct Candidate {
    value: String,
    class: String,
    /// `detected`, `listed` (identity/sensitive terms) or `propagated`
    source: &'static str,
    credential: bool,
    count: usize,
    context: String,
}

#[derive(Serialize)]
struct SessionPlan {
    session: String,
    candidates: Vec<Candidate>,
}

/// Short single-occurrence context around a match, on char boundaries.
fn context(text: &str, start: usize, end: usize) -> String {
    let mut from = start.saturating_sub(60);
    while !text.is_char_boundary(from) {
        from -= 1;
    }
    let mut to = (end + 60).min(text.len());
    while !text.is_char_boundary(to) {
        to += 1;
    }
    format!(
        "{}{}{}",
        if from > 0 { "…" } else { "" },
        &text[from..to],
        if to < text.len() { "…" } else { "" }
    )
}

#[derive(serde::Deserialize)]
struct PlanFile {
    sessions: Vec<PlanSession>,
}

#[derive(serde::Deserialize)]
struct PlanSession {
    session: String,
    candidates: Vec<PlanValue>,
}

#[derive(serde::Deserialize)]
struct PlanValue {
    value: String,
    class: String,
}

/// Approved values per session, from `--apply-plan`.
fn load_plan(path: &Path) -> Result<HashMap<String, Vec<Literal>>> {
    let text = fs::read_to_string(path).context("failed to read the plan file")?;
    let plan: PlanFile = serde_json::from_str(&text).context("invalid plan file")?;
    Ok(plan
        .sessions
        .into_iter()
        .map(|s| {
            let literals = s
                .candidates
                .into_iter()
                .filter(|c| !c.value.is_empty())
                .map(|c| Literal {
                    text: c.value,
                    class: c.class,
                })
                .collect();
            (s.session, literals)
        })
        .collect())
}

/// Everything `process` needs besides the session itself.
struct Shared<'a> {
    regex: &'a Detector,
    ner: Option<&'a Detector>,
    identity: &'a Identity,
    allow: &'a HashSet<String>,
    keep: &'a HashSet<String>,
    /// Matcher over kept values: approved literals inside a kept value stay.
    keep_matcher: Option<AhoCorasick>,
    approved: Option<HashMap<String, Vec<Literal>>>,
    cache: Option<crate::traces_cache::NerCache>,
}

/// Executable identity, so any rebuild invalidates cached NER results.
fn executable_stamp() -> String {
    std::env::current_exe()
        .and_then(fs::metadata)
        .map(|m| {
            let modified = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos());
            format!("{}:{modified}", m.len())
        })
        .unwrap_or_default()
}

/// A literal that must be replaced wherever it occurs in a session.
#[derive(Clone)]
struct Literal {
    text: String,
    class: String,
}

pub fn run(common: &crate::CommonOpts, config: &Config, cmd: TracesCommand) -> Result<()> {
    match cmd.action {
        Action::Anon(args) => anon(common, config, args),
    }
}

fn detector(config: &Config, ner: bool) -> Result<Detector> {
    let mut excluded = config.detection.disabled_patterns.clone();
    excluded.extend(TRACE_EXCLUDED.iter().map(|s| (*s).to_string()));
    let dc = DetectorConfig::default()
        .with_min_confidence(config.detection.min_confidence.into())
        .without_patterns(excluded)
        .with_ner(ner);
    let dc = crate::configure_ner(dc, &config.ner, None, None)?;
    Ok(Detector::new(&dc)?)
}

fn anon(common: &crate::CommonOpts, config: &Config, args: AnonArgs) -> Result<()> {
    let started = Instant::now();
    let files = resolve(&args.inputs, args.output.as_deref())?;
    if files.is_empty() {
        bail!("no session files (*.jsonl) found");
    }
    if let (false, Some(output)) = (args.force, &args.output) {
        for (_, rel) in &files {
            if output.join(rel).exists() {
                bail!("output already exists; use --force to replace it");
            }
        }
    }
    let mut literals = Vec::new();
    if !args.no_local_identity {
        literals.extend(local_identity());
    }
    for path in &args.sensitive_terms_file {
        let text = fs::read_to_string(path).context("failed to read sensitive terms file")?;
        literals.extend(
            text.lines()
                .map(str::trim)
                .filter(|l| l.chars().count() >= 3 && !l.starts_with('#'))
                .map(|l| Literal {
                    text: l.to_string(),
                    class: sensitive_class(l).into(),
                }),
        );
    }
    let profile_lists = crate::profiles::load(&args.redaction_profiles)?;
    literals.extend(profile_lists.sensitive.iter().map(|term| Literal {
        text: term.clone(),
        class: sensitive_class(term).into(),
    }));
    let identity = Identity::new(&literals)?;
    let mut keep = HashSet::new();
    for path in &args.keep_values_file {
        let text = fs::read_to_string(path).context("failed to read keep values file")?;
        keep.extend(
            text.lines()
                .map(|l| l.trim_end_matches('\r'))
                .filter(|l| !l.is_empty())
                .map(str::to_string),
        );
    }
    let mut allow = allowlist(&args.allow_terms_file)?;
    allow.extend(profile_lists.allow.iter().map(|t| t.to_lowercase()));
    let regex = detector(config, false)?;
    let approved = args.apply_plan.as_deref().map(load_plan).transpose()?;
    let keep_matcher = if keep.is_empty() {
        None
    } else {
        Some(
            AhoCorasickBuilder::new()
                .match_kind(MatchKind::LeftmostLongest)
                .build(keep.iter())
                .map_err(|_| anyhow!("failed to build keep matcher"))?,
        )
    };
    let cpus = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let jobs = args
        .jobs
        .unwrap_or(if args.no_ner { cpus } else { cpus.min(8) })
        .max(1);
    // One shared model session: NER is compute-bound, and measured per-worker
    // sessions with split cores were slower than one session using all cores.
    // Applying a reviewed plan needs no detectors, so no model and no cache.
    let ner = if args.no_ner || approved.is_some() {
        None
    } else {
        Some(detector(config, true)?)
    };
    let cache = if args.no_cache || ner.is_none() {
        None
    } else {
        let fingerprint = format!(
            "{}|{}|{}|{:?}",
            env!("NYM_BUILD_VERSION"),
            executable_stamp(),
            serde_json::to_string(&config.ner)?,
            args.ner_scope
        );
        crate::traces_cache::NerCache::open(&crate::config::cache_dir().join("ner"), &fingerprint)
            .ok()
    };
    let shared = Shared {
        regex: &regex,
        ner: ner.as_ref(),
        identity: &identity,
        allow: &allow,
        keep: &keep,
        keep_matcher,
        approved,
        cache,
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .map_err(|_| anyhow!("failed to start worker threads"))?;
    let progress = !args.no_progress && !common.quiet && std::io::stderr().is_terminal();
    let done = AtomicUsize::new(0);
    let total = files.len();
    let input_bytes: u64 = files
        .iter()
        .filter_map(|(p, _)| fs::metadata(p).ok())
        .map(|m| m.len())
        .sum();
    let results: Vec<Result<(SessionStats, SessionPlan)>> = pool.install(|| {
        files
            .par_iter()
            .map(|(path, rel)| {
                let result = process(path, rel, &args, &shared);
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if progress {
                    eprint!(
                        "\r[{n}/{total}] sessions, {:.0}s elapsed",
                        started.elapsed().as_secs_f64()
                    );
                }
                result
            })
            .collect()
    });
    if progress {
        eprintln!();
    }
    let mut totals = SessionStats::default();
    let mut failed = 0;
    let mut plans = Vec::new();
    for result in &results {
        match result {
            Ok((stats, plan)) => {
                merge(&mut totals, stats);
                plans.push(plan);
            }
            Err(error) => {
                failed += 1;
                // Errors are generic by construction; never echo session content.
                if !common.quiet {
                    eprintln!("error: {error}");
                }
            }
        }
    }
    if let Some(cache) = &shared.cache
        && let Err(error) = cache.flush()
        && !common.quiet
    {
        // Losing cache entries only costs recomputation next time.
        eprintln!("warning: {error}");
    }
    if let (Some(path), 0) = (&args.plan, failed) {
        let body = serde_json::to_vec_pretty(&serde_json::json!({ "sessions": plans }))?;
        publish(path, &body).context("failed to write the plan file")?;
    }
    let seconds = started.elapsed().as_secs_f64();
    let summary = Summary {
        sessions: total - failed,
        failed,
        input_bytes,
        seconds,
        megabytes_per_second: input_bytes as f64 / 1e6 / seconds.max(1e-9),
        totals,
    };
    if common.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else if !common.quiet {
        eprintln!(
            "Anonymized {} session(s) ({:.1} MB) in {:.1}s: {} replacements ({} propagated, {} identity), {} images removed{}",
            summary.sessions,
            input_bytes as f64 / 1e6,
            seconds,
            summary.totals.replacements,
            summary.totals.propagated,
            summary.totals.identity,
            summary.totals.images_removed,
            if failed > 0 {
                format!("; {failed} FAILED (no output written for those)")
            } else {
                String::new()
            }
        );
    }
    if failed > 0 {
        bail!("{failed} session(s) could not be anonymized");
    }
    Ok(())
}

fn merge(totals: &mut SessionStats, stats: &SessionStats) {
    totals.records += stats.records;
    totals.strings += stats.strings;
    totals.scanned_bytes += stats.scanned_bytes;
    totals.ner_bytes += stats.ner_bytes;
    totals.images_removed += stats.images_removed;
    totals.replacements += stats.replacements;
    totals.propagated += stats.propagated;
    totals.identity += stats.identity;
    for (k, v) in &stats.by_class {
        *totals.by_class.entry(k.clone()).or_default() += v;
    }
}

/// Resolve inputs into (source, relative output path). Symlinks are not followed.
fn resolve(inputs: &[PathBuf], output: Option<&Path>) -> Result<Vec<(PathBuf, PathBuf)>> {
    let out = output.and_then(|o| fs::canonicalize(o).ok());
    let mut files = Vec::new();
    let multiple = inputs.len() > 1;
    for input in inputs {
        let meta = fs::symlink_metadata(input).context("input is not readable")?;
        let canonical = fs::canonicalize(input).context("input is not readable")?;
        if out
            .as_ref()
            .is_some_and(|o| o.starts_with(&canonical) || canonical.starts_with(o))
        {
            bail!("output directory must not overlap an input");
        }
        let prefix = if multiple {
            PathBuf::from(canonical.file_name().unwrap_or_default())
        } else {
            PathBuf::new()
        };
        if meta.is_file() {
            files.push((
                canonical.clone(),
                PathBuf::from(canonical.file_name().unwrap_or_default()),
            ));
        } else if meta.is_dir() {
            walk(&canonical, &canonical, &prefix, &mut files, 0)?;
        } else {
            bail!("inputs must be regular files or directories");
        }
    }
    files.sort();
    let mut seen = HashSet::new();
    for (_, rel) in &files {
        if !seen.insert(rel.clone()) {
            bail!("two inputs map to the same output path; pass a single parent directory");
        }
    }
    Ok(files)
}

fn walk(
    root: &Path,
    dir: &Path,
    prefix: &Path,
    files: &mut Vec<(PathBuf, PathBuf)>,
    depth: usize,
) -> Result<()> {
    if depth > 64 {
        bail!("directory nesting is too deep");
    }
    for entry in fs::read_dir(dir).context("directory is not readable")? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let path = entry.path();
        if kind.is_dir() {
            walk(root, &path, prefix, files, depth + 1)?;
        } else if kind.is_file()
            && path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("jsonl"))
        {
            let rel = path
                .strip_prefix(root)
                .map_err(|_| anyhow!("invalid path"))?;
            files.push((path.clone(), prefix.join(rel)));
        }
    }
    Ok(())
}

/// Identity of the person running NYM: the most frequent PII in their traces.
fn local_identity() -> Vec<Literal> {
    let mut out = Vec::new();
    let mut add = |text: &str, class: &str| {
        let text = text.trim();
        if text.chars().count() >= 3 && !GENERIC_ACCOUNTS.contains(&text.to_lowercase().as_str()) {
            out.push(Literal {
                text: text.to_string(),
                class: class.to_string(),
            });
        }
    };
    for var in ["USER", "USERNAME", "LOGNAME"] {
        if let Ok(v) = std::env::var(var) {
            add(&v, "username");
        }
    }
    if let Some(home) =
        dirs::home_dir().and_then(|h| h.file_name().map(|n| n.to_string_lossy().into_owned()))
    {
        add(&home, "username");
    }
    let run = |cmd: &str, args: &[&str]| {
        std::process::Command::new(cmd)
            .args(args)
            .stderr(std::process::Stdio::null())
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .filter(|s| !s.is_empty())
    };
    let mut names = Vec::new();
    if let Some(n) = run("id", &["-F"]) {
        names.push(n);
    }
    if let Some(n) = run("git", &["config", "--global", "user.name"]) {
        names.push(n);
    }
    if let Some(e) = run("git", &["config", "--global", "user.email"]) {
        add(&e, "email");
        if let Some(local) = e.split('@').next() {
            add(local, "username");
        }
    }
    if let Some(h) = run("hostname", &["-s"]) {
        add(&h, "hostname");
    }
    for name in names {
        add(&name, "person");
        let parts: Vec<_> = name.split_whitespace().collect();
        for (i, part) in parts.iter().enumerate() {
            if part.chars().count() >= 4 {
                add(part, if i == 0 { "first_name" } else { "last_name" });
            }
        }
        // Joined forms used in handles and paths: tommyfalkowski, tommy.falkowski
        if parts.len() >= 2 {
            let lower: Vec<_> = parts.iter().map(|p| p.to_lowercase()).collect();
            add(&lower.concat(), "username");
            add(&lower.join("."), "username");
            add(&lower.join("_"), "username");
        }
    }
    out.sort_by(|a, b| a.text.cmp(&b.text));
    out.dedup_by(|a, b| a.text.eq_ignore_ascii_case(&b.text));
    out
}

/// Case-insensitive literal matcher with word-boundary checks.
struct Identity {
    matcher: Option<AhoCorasick>,
    literals: Vec<Literal>,
}

impl Identity {
    fn new(literals: &[Literal]) -> Result<Self> {
        if literals.is_empty() {
            return Ok(Self {
                matcher: None,
                literals: Vec::new(),
            });
        }
        let matcher = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .match_kind(MatchKind::LeftmostLongest)
            .build(literals.iter().map(|l| &l.text))
            .map_err(|_| anyhow!("failed to build identity matcher"))?;
        Ok(Self {
            matcher: Some(matcher),
            literals: literals.to_vec(),
        })
    }
}

fn bounded(text: &str, start: usize, end: usize) -> bool {
    let prefix = &text[..start];
    // Double-escaped tool output contains literal `\n`, `\t`, `\r` sequences.
    let escaped = ["\\n", "\\t", "\\r"].iter().any(|e| prefix.ends_with(e));
    let before = prefix.chars().next_back();
    let after = text[end..].chars().next();
    (escaped || !before.is_some_and(char::is_alphanumeric))
        && !after.is_some_and(char::is_alphanumeric)
}

/// Reviewed values match wherever they occur, except inside a longer word:
/// boundaries are only required on alphanumeric value edges.
fn approved_matches(matcher: &AhoCorasick, literals: &[Literal], text: &str) -> Vec<PiiMatch> {
    matcher
        .find_iter(text)
        .filter(|m| {
            let value = &text[m.start()..m.end()];
            let first = value.chars().next().is_some_and(char::is_alphanumeric);
            let last = value.chars().next_back().is_some_and(char::is_alphanumeric);
            let prefix = &text[..m.start()];
            // Double-escaped tool output: a literal `\n` before a value is a line break.
            let escaped = ["\\n", "\\t", "\\r"].iter().any(|e| prefix.ends_with(e));
            let before = !escaped
                && prefix
                    .chars()
                    .next_back()
                    .is_some_and(char::is_alphanumeric);
            let after = text[m.end()..]
                .chars()
                .next()
                .is_some_and(char::is_alphanumeric);
            !(first && before) && !(last && after)
        })
        .map(|m| PiiMatch {
            pattern_name: literals[m.pattern().as_usize()].class.clone(),
            matched_text: text[m.start()..m.end()].to_string(),
            start: m.start(),
            end: m.end(),
            confidence: crate::engine::patterns::Confidence::High,
            category: crate::engine::patterns::PiiCategory::Identity,
        })
        .collect()
}

fn literal_matches(matcher: &AhoCorasick, literals: &[Literal], text: &str) -> Vec<PiiMatch> {
    matcher
        .find_iter(text)
        .filter(|m| bounded(text, m.start(), m.end()))
        .map(|m| PiiMatch {
            pattern_name: literals[m.pattern().as_usize()].class.clone(),
            matched_text: text[m.start()..m.end()].to_string(),
            start: m.start(),
            end: m.end(),
            confidence: crate::engine::patterns::Confidence::High,
            category: crate::engine::patterns::PiiCategory::Identity,
        })
        .collect()
}

/// A JSON pointer to a string leaf plus its classification.
struct Leaf {
    record: usize,
    pointer: String,
    kind: Kind,
}

fn escape(key: &str) -> String {
    key.replace('~', "~0").replace('/', "~1")
}

fn classify(key: &str, parent: &serde_json::Map<String, Value>, role: &str) -> Kind {
    if key == "data"
        && (parent.get("type").and_then(Value::as_str) == Some("image")
            || parent.contains_key("mimeType"))
    {
        return Kind::Image;
    }
    if STRUCTURAL.contains(&key) {
        return Kind::Skip;
    }
    // Chat-style `{"role": ..., "content": "..."}` is prose; tool-call
    // argument `content` (file bodies) is not.
    let prose = PROSE.contains(&key) || (key == "content" && parent.contains_key("role"));
    if prose && role != "toolResult" {
        return Kind::Prose;
    }
    Kind::Other
}

fn collect(
    value: &Value,
    record: usize,
    pointer: &str,
    key: &str,
    parent: Option<&serde_json::Map<String, Value>>,
    role: &str,
    out: &mut Vec<Leaf>,
) {
    match value {
        Value::String(_) => {
            let kind = parent.map_or(Kind::Other, |p| classify(key, p, role));
            out.push(Leaf {
                record,
                pointer: pointer.to_string(),
                kind,
            });
        }
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                // Array elements inherit their parent key for classification.
                collect(
                    item,
                    record,
                    &format!("{pointer}/{i}"),
                    key,
                    parent,
                    role,
                    out,
                );
            }
        }
        Value::Object(map) => {
            for (k, v) in map {
                collect(
                    v,
                    record,
                    &format!("{pointer}/{}", escape(k)),
                    k,
                    Some(map),
                    role,
                    out,
                );
            }
        }
        _ => {}
    }
}

fn role(record: &Value) -> String {
    record
        .pointer("/message/role")
        .or_else(|| record.get("role"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// Keep the structural `/Users/` prefix; only the account name is private.
fn narrow_home(m: &mut PiiMatch) {
    if m.pattern_name != "home_dir" {
        return;
    }
    if let Some(i) = m.matched_text.rfind(['/', '\\']) {
        m.start += i + 1;
        m.matched_text = m.matched_text[i + 1..].to_string();
        m.pattern_name = "username".into();
    }
}

/// Merge overlapping spans into their union: a partial overlap must never leave
/// part of a sensitive value behind. The longer span's class names the union.
fn resolve_overlaps(text: &str, mut matches: Vec<PiiMatch>) -> Vec<PiiMatch> {
    matches.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));
    let mut out: Vec<PiiMatch> = Vec::with_capacity(matches.len());
    for m in matches {
        if m.start >= m.end {
            continue;
        }
        match out.last_mut() {
            Some(last) if m.start < last.end => {
                if m.end > last.end {
                    if m.end - m.start > last.end - last.start {
                        last.pattern_name = m.pattern_name;
                        last.category = m.category;
                    }
                    last.end = m.end;
                    last.matched_text = text[last.start..last.end].to_string();
                }
            }
            _ => out.push(m),
        }
    }
    out
}

/// Same value, same pseudonym: strip delimiters a detector attached to a
/// non-credential span (credentials keep their exact token).
fn trim_edges(m: &mut PiiMatch) {
    if m.category == crate::engine::patterns::PiiCategory::Authentication {
        return;
    }
    let edge = |c: char| c.is_whitespace() || ";,:!?.\"'`|()[]{}<>=".contains(c);
    let start = m.matched_text.len() - m.matched_text.trim_start_matches(edge).len();
    let end = m.matched_text.trim_end_matches(edge).len().max(start);
    m.start += start;
    m.end = m.start + (end - start);
    m.matched_text = m.matched_text[start..end].to_string();
}

/// Shape checks for numeric classes: agent traces are full of timestamps,
/// ports, line numbers and IDs that regex/NER mislabel as phones or cards.
/// PIN, CVV and birth dates are kept regardless (privacy-critical, rare).
fn plausible(m: &PiiMatch, leaf: &str, prose: bool, allow: &HashSet<String>) -> bool {
    let text = m.matched_text.as_str();
    let lower = text.to_lowercase();
    if m.category != crate::engine::patterns::PiiCategory::Authentication && allow.contains(&lower)
    {
        return false;
    }
    // Shared system accounts (`root`, `www-data`) identify no person.
    if GENERIC_ACCOUNTS.contains(&lower.as_str()) {
        return false;
    }
    let digits = text.chars().filter(char::is_ascii_digit).count();
    let numeric = text
        .chars()
        .all(|c| c.is_ascii_digit() || c.is_whitespace() || "+-.:/()".contains(c));
    let labelled = |keywords: &[&str]| {
        let mut from = m.start.saturating_sub(48);
        while !leaf.is_char_boundary(from) {
            from -= 1;
        }
        let before = leaf[from..m.start.min(leaf.len())].to_lowercase();
        keywords.iter().any(|k| before.contains(k))
    };
    match m.pattern_name.as_str() {
        "date" | "time" => false,
        // Vendors, products and languages dominate; list employers and clients
        // explicitly with --sensitive-terms-file instead.
        "organization" | "company" => false,
        "url" | "social_url" => url_like(&lower) && !public_url(&lower),
        // Decorators, npm scopes and doc tags in code; mentions only in prose.
        "twitter_handle" | "social_handle" | "instagram_handle" => {
            prose && !CODE_HANDLES.contains(&lower.trim_start_matches('@'))
        }
        "ipv4" => !(text.starts_with("127.") || text == "0.0.0.0" || text == "255.255.255.255"),
        "first_name" | "last_name" | "person" | "name" if !person_like(m, leaf) => false,
        "first_name" | "last_name" | "person" | "name" | "username" => {
            !text
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| !w.is_empty())
                .all(|w| allow.contains(&w.to_lowercase()))
                && (m.pattern_name != "username" || text.chars().count() >= 4)
                && !text.contains('&')
        }
        "date_of_birth" => {
            // Recent dates are log timestamps, not birthdays.
            let year = text
                .split(|c: char| !c.is_ascii_digit())
                .find(|p| p.len() == 4)
                .and_then(|y| y.parse::<u32>().ok());
            digits >= 6 && year.is_none_or(|y| y < 2015)
        }
        "pin" | "cvv" => labelled(&[
            "pin",
            "cvv",
            "cvc",
            "passcode",
            "security code",
            "geheimzahl",
        ]),
        "account_number" | "bank_routing_number" => labelled(&[
            "account",
            "konto",
            "acct",
            "routing",
            "aba",
            "blz",
            "sort code",
            "bank",
        ]),
        "certificate_license_number" | "government_id" | "tax_id" | "passport_us" => labelled(&[
            "licen",
            "lizenz",
            "führerschein",
            "passport",
            "pass",
            "tax",
            "steuer",
            "id",
            "ausweis",
            "nummer",
            "number",
            "no.",
        ]),
        "ner_entity" => prose && !numeric,
        // Bare one- or two-digit "ages" are almost always parts of times or counts.
        "age" => labelled(&[
            "age", "aged", "years", "year-old", "yrs", "jahre", "alter", " alt",
        ]),
        "postcode" => true,
        "phone_us" | "phone_intl" | "phone_ner" => {
            digits >= 7 && (text.contains('+') || text.chars().any(|c| " -.()".contains(c)))
        }
        "credit_card" | "credit_card_nodash" => (13..=19).contains(&digits) && luhn(text),
        "ssn" | "ssn_nodash" => digits == 9,
        "mac" => {
            let parts: Vec<_> = text.split([':', '-']).collect();
            parts.len() == 6 && parts.iter().all(|p| p.len() == 2)
        }
        _ => !numeric || digits >= 6,
    }
}

/// Names are capitalized and not code: `greet`, `Type.Object`, `x.Type` and
/// `Type(` are identifiers, not people.
fn person_like(m: &PiiMatch, leaf: &str) -> bool {
    let text = m.matched_text.as_str();
    if !text.chars().any(char::is_uppercase) {
        return false;
    }
    let before = leaf.get(..m.start).and_then(|s| s.chars().next_back());
    let mut after = leaf.get(m.end..).map(str::chars).into_iter().flatten();
    let next = after.next();
    let member = next == Some('.') && after.next().is_some_and(char::is_alphanumeric);
    !(member || next == Some('(') || before == Some('.'))
}

/// Hosts whose URLs are public references, not personal data. Identities,
/// emails and tokens inside such URLs are still caught by other detectors.
const PUBLIC_HOSTS: &[&str] = &[
    "github.com",
    "raw.githubusercontent.com",
    "gist.github.com",
    "gitlab.com",
    "docs.rs",
    "crates.io",
    "npmjs.com",
    "www.npmjs.com",
    "pypi.org",
    "huggingface.co",
    "stackoverflow.com",
    "en.wikipedia.org",
    "de.wikipedia.org",
    "developer.mozilla.org",
    "doc.rust-lang.org",
    "rust-lang.org",
    "www.rust-lang.org",
    "python.org",
    "docs.python.org",
    "go.dev",
    "pkg.go.dev",
    "developer.apple.com",
    "learn.microsoft.com",
    "docs.anthropic.com",
    "docs.claude.com",
    "platform.openai.com",
    "openai.com",
    "anthropic.com",
    "arxiv.org",
    "youtube.com",
    "www.youtube.com",
    "google.com",
    "www.google.com",
    "localhost",
    "127.0.0.1",
    "example.com",
    "schemas.openxmlformats.org",
    "www.w3.org",
    "json-schema.org",
    "registry.npmjs.org",
];

/// NER labels file names (`README.md`), permission strings and word fragments
/// as URLs. Require a scheme or a dotted host that does not end in a file type.
fn url_like(lower: &str) -> bool {
    if lower.contains("://") || lower.starts_with("www.") {
        return true;
    }
    let host = lower.split(['/', '?', '#', ':']).next().unwrap_or_default();
    let labels: Vec<_> = host.split('.').collect();
    let tld = labels.last().copied().unwrap_or_default();
    labels.len() >= 2
        && labels
            .iter()
            .all(|l| !l.is_empty() && l.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'))
        && tld.len() >= 2
        && tld.chars().all(|c| c.is_ascii_alphabetic())
        && !FILE_TYPES.contains(&tld)
}

const FILE_TYPES: &[&str] = &[
    "md", "rs", "toml", "json", "jsonl", "yaml", "yml", "sh", "py", "js", "ts", "tsx", "jsx",
    "lock", "txt", "log", "html", "css", "go", "c", "h", "cpp", "hpp", "swift", "kt", "java", "rb",
    "sql", "csv", "pdf", "png", "jpg", "jpeg", "svg", "gif", "zip", "tar", "gz", "env", "ini",
    "cfg", "conf", "xml", "mjs", "cjs", "pyi", "ipynb", "wasm", "so", "dylib", "dll", "exe", "bin",
    "onnx", "pt",
];

fn public_url(lower: &str) -> bool {
    let rest = lower.split_once("://").map_or(lower, |(_, r)| r);
    let host = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default()
        .rsplit('@')
        .next()
        .unwrap_or_default();
    let host = host.split(':').next().unwrap_or_default();
    PUBLIC_HOSTS.contains(&host)
}

/// Code decorators, npm scopes and doc-comment tags that look like handles.
const CODE_HANDLES: &[&str] = &[
    "types",
    "param",
    "returns",
    "return",
    "dataclass",
    "property",
    "staticmethod",
    "classmethod",
    "schema",
    "file",
    "path",
    "mention",
    "radix",
    "just",
    "angular",
    "babel",
    "vue",
    "tanstack",
    "testing",
    "override",
    "deprecated",
    "example",
    "see",
    "throws",
    "author",
    "since",
    "type",
    "typedef",
    "import",
    "media",
    "keyframes",
    "apply",
    "tailwind",
    "layer",
    "ts",
    "app",
    "vercel",
];

/// Built-in allowlist: models, products, tools and type names that detectors
/// often label as people, organizations or accounts. Matched case-insensitively
/// against whole values (and, for names, against every word of the value).
const BUILTIN_ALLOW: &[&str] = &[
    "claude",
    "code",
    "claude code",
    "sonnet",
    "opus",
    "haiku",
    "gemini",
    "gpt",
    "chatgpt",
    "codex",
    "copilot",
    "cursor",
    "llama",
    "mistral",
    "qwen",
    "kimi",
    "grok",
    "deepseek",
    "ollama",
    "openai",
    "anthropic",
    "huggingface",
    "hugging face",
    "string",
    "main",
    "json",
    "yaml",
    "toml",
    "true",
    "false",
    "null",
    "none",
    "self",
    "master",
    "test",
    "rust",
    "python",
    "node",
    "nodejs",
    "deno",
    "bun",
    "docker",
    "podman",
    "kubernetes",
    "k8s",
    "linux",
    "macos",
    "ios",
    "android",
    "windows",
    "ubuntu",
    "debian",
    "arch",
    "archlinux",
    "homebrew",
    "apple",
    "apple silicon",
    "silicon",
    "nvidia",
    "amd",
    "intel",
    "cuda",
    "metal",
    "coreml",
    "onnx",
    "pytorch",
    "torch",
    "tensorflow",
    "pi",
    "github",
    "gitlab",
    "bitbucket",
    "git",
    "npm",
    "pnpm",
    "yarn",
    "cargo",
    "uv",
    "pip",
    "react",
    "vue",
    "svelte",
    "tailwind",
    "typescript",
    "javascript",
    "golang",
    "go",
    "swift",
    "kotlin",
    "java",
    "postgres",
    "postgresql",
    "sqlite",
    "redis",
    "nginx",
    "caddy",
    "chrome",
    "chromium",
    "safari",
    "firefox",
    "vscode",
    "neovim",
    "vim",
    "emacs",
    "zed",
    "tmux",
    "ghostty",
    "iterm",
    "warp",
    "slack",
    "discord",
    "telegram",
    "notion",
    "obsidian",
    "google",
    "microsoft",
    "amazon",
    "aws",
    "azure",
    "gcp",
    "cloudflare",
    "vercel",
    "netlify",
    "tailscale",
    "comfyui",
    "playwright",
    "jupyter",
];

/// Pseudonym class for a user-listed sensitive literal.
fn sensitive_class(term: &str) -> &'static str {
    if term.contains('@') {
        "email"
    } else if term.contains(char::is_whitespace) {
        "organization"
    } else {
        "username"
    }
}

fn allowlist(files: &[PathBuf]) -> Result<HashSet<String>> {
    let mut allow: HashSet<String> = BUILTIN_ALLOW.iter().map(|s| (*s).to_string()).collect();
    for path in files {
        let text = fs::read_to_string(&path).context("failed to read allow terms file")?;
        allow.extend(
            text.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(str::to_lowercase),
        );
    }
    Ok(allow)
}

fn luhn(text: &str) -> bool {
    let digits: Vec<u32> = text.chars().filter_map(|c| c.to_digit(10)).collect();
    let sum: u32 = digits
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &d)| {
            if i % 2 == 1 {
                if d * 2 > 9 { d * 2 - 9 } else { d * 2 }
            } else {
                d
            }
        })
        .sum();
    sum % 10 == 0
}

fn propagatable(m: &PiiMatch) -> bool {
    let chars = m.matched_text.chars().count();
    !NO_PROPAGATE.contains(&m.pattern_name.as_str())
        && chars >= 4
        && (m.matched_text.chars().any(char::is_alphabetic) || chars >= 7)
}

fn session_seed(base: Option<u64>, rel: &Path) -> u64 {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    match base {
        Some(seed) => hasher.update(seed.to_le_bytes()),
        // Randomly keyed per process: unseeded runs get unrelated pseudonyms.
        None => hasher.update(
            std::hash::BuildHasher::hash_one(&std::collections::hash_map::RandomState::new(), rel)
                .to_le_bytes(),
        ),
    }
    hasher.update(rel.to_string_lossy().as_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("digest has 32 bytes"))
}

fn process(
    path: &Path,
    rel: &Path,
    args: &AnonArgs,
    shared: &Shared,
) -> Result<(SessionStats, SessionPlan)> {
    let Shared {
        regex,
        ner,
        identity,
        allow,
        keep,
        ..
    } = *shared;
    let failure = |what: &str| anyhow!("{what} for session #{}", short_id(rel));
    let approved = match &shared.approved {
        Some(plan) => Some(
            plan.get(rel.to_string_lossy().as_ref())
                .ok_or_else(|| failure("plan does not cover this session"))?,
        ),
        None => None,
    };
    let raw = fs::read(path).map_err(|_| failure("read failed"))?;
    let text = std::str::from_utf8(&raw).map_err(|_| failure("invalid UTF-8"))?;
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let mut records: Vec<Option<Value>> = Vec::with_capacity(lines.len());
    let mut leaves = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let body = line.trim_end_matches(['\n', '\r']);
        if body.trim().is_empty() {
            records.push(None);
            continue;
        }
        let value: Value = serde_json::from_str(body)
            .map_err(|_| failure(&format!("invalid JSON on line {}", i + 1)))?;
        let role = role(&value);
        collect(&value, i, "", "", None, &role, &mut leaves);
        records.push(Some(value));
    }
    let mut stats = SessionStats {
        records: records.iter().flatten().count(),
        strings: leaves.len(),
        ..SessionStats::default()
    };

    // Unique strings per tier; repeated tool output is scanned once.
    let get = |leaf: &Leaf| -> &str {
        records[leaf.record]
            .as_ref()
            .and_then(|v| v.pointer(&leaf.pointer))
            .and_then(Value::as_str)
            .unwrap_or_default()
    };
    let ner_all = args.ner_scope == NerScope::All;
    let mut ner_texts: Vec<&str> = Vec::new();
    let mut regex_texts: Vec<&str> = Vec::new();
    let mut seen = HashSet::new();
    let mut prose_texts: HashSet<&str> = HashSet::new();
    for leaf in &leaves {
        let s = get(leaf);
        if leaf.kind == Kind::Prose {
            prose_texts.insert(s);
        }
        let wants_ner = ner.is_some()
            && args.ner_scope != NerScope::Lines
            && (leaf.kind == Kind::Prose || (ner_all && leaf.kind == Kind::Other));
        if matches!(leaf.kind, Kind::Skip | Kind::Image)
            || s.is_empty()
            || !seen.insert((s, wants_ner))
        {
            continue;
        }
        if wants_ner {
            ner_texts.push(s);
        } else {
            regex_texts.push(s);
        }
    }
    let mut found: HashMap<&str, Vec<PiiMatch>> = HashMap::new();
    let mut ner_total = 0;
    let mut ner_done = 0;
    // (parent string, line offset, line) for auto-scope NER on tool output.
    let mut lines_for_ner: Vec<(&str, usize, &str)> = Vec::new();
    let line_ner = ner.is_some() && matches!(args.ner_scope, NerScope::Auto | NerScope::Lines);
    let mut progress = Progress::new(args.progress_json);
    let detect_total: usize = regex_texts.iter().map(|s| s.len()).sum();
    let mut detect_done = 0;
    progress.report("detect", 0, detect_total, true);
    for s in &regex_texts {
        detect_done += s.len();
        progress.report("detect", detect_done, detect_total, false);
        if approved.is_some() {
            continue;
        }
        stats.scanned_bytes += s.len();
        found.insert(s, regex.detect(s).map_err(|_| failure("detection failed"))?);
        if line_ner {
            let prose = prose_texts.contains(s);
            let mut offset = 0;
            for line in s.split_inclusive('\n') {
                if line.len() <= MAX_CANDIDATE_LINE
                    && (CANDIDATE_LINE.is_match(line)
                        || (prose && PROSE_CANDIDATE_LINE.is_match(line)))
                {
                    lines_for_ner.push((s, offset, line));
                }
                offset += line.len();
            }
        }
    }
    if let Some(ner) = ner {
        let unique: Vec<&str> = {
            let mut seen = HashSet::new();
            lines_for_ner
                .iter()
                .map(|(_, _, line)| *line)
                .filter(|line| seen.insert(*line))
                .collect()
        };
        let mut by_line: HashMap<&str, Vec<PiiMatch>> = HashMap::new();
        ner_total = unique.iter().map(|l| l.len()).sum::<usize>()
            + ner_texts.iter().map(|t| t.len()).sum::<usize>();
        progress.report("ner", 0, ner_total, true);
        let results =
            crate::traces_cache::detect_cached(ner, shared.cache.as_ref(), &unique, 64, |bytes| {
                ner_done += bytes;
                progress.report("ner", ner_done, ner_total, false);
            })
            .map_err(|_| failure("NER inference failed"))?;
        for (line, matches) in unique.iter().zip(results) {
            stats.ner_bytes += line.len();
            by_line.insert(line, matches);
        }
        for (parent, offset, line) in &lines_for_ner {
            let Some(matches) = by_line.get(line) else {
                continue;
            };
            let entry = found.entry(parent).or_default();
            for m in matches {
                let mut m = m.clone();
                m.start += offset;
                m.end += offset;
                entry.push(m);
            }
        }
    }
    if let Some(ner) = ner {
        let results = crate::traces_cache::detect_cached(
            ner,
            shared.cache.as_ref(),
            &ner_texts,
            32,
            |bytes| {
                ner_done += bytes;
                progress.report("ner", ner_done, ner_total, false);
            },
        )
        .map_err(|_| failure("NER inference failed"))?;
        for (s, matches) in ner_texts.iter().zip(results) {
            stats.scanned_bytes += s.len();
            stats.ner_bytes += s.len();
            // A string may be both prose and other; keep the richer result.
            found.insert(s, matches);
        }
    }

    for (leaf, matches) in &mut found {
        let prose = prose_texts.contains(leaf);
        matches.iter_mut().for_each(trim_edges);
        matches.retain(|m| {
            m.start < m.end && !keep.contains(&m.matched_text) && plausible(m, leaf, prose, allow)
        });
    }

    // Session-wide propagation set: identity literals plus every identity-like finding.
    let mut session_literals: Vec<Literal> = approved.cloned().unwrap_or_default();
    let mut known = HashSet::new();
    // Session order, not hash order: a value found with different classes
    // gets the class of its first occurrence, deterministically.
    let mut first_seen: HashMap<&str, usize> = HashMap::new();
    for (index, leaf) in leaves.iter().enumerate() {
        first_seen.entry(get(leaf)).or_insert(index);
    }
    let mut ordered: Vec<(&&str, &Vec<PiiMatch>)> = found.iter().collect();
    ordered.sort_by_key(|(text, _)| first_seen.get(**text).copied().unwrap_or(usize::MAX));
    for (_, matches) in ordered {
        for m in matches {
            let mut m = m.clone();
            narrow_home(&mut m);
            if propagatable(&m) && known.insert(m.matched_text.clone()) {
                session_literals.push(Literal {
                    text: m.matched_text,
                    class: m.pattern_name,
                });
            }
        }
    }
    let propagation = if session_literals.is_empty() {
        None
    } else {
        Some(
            AhoCorasickBuilder::new()
                .match_kind(MatchKind::LeftmostLongest)
                .build(session_literals.iter().map(|l| &l.text))
                .map_err(|_| failure("propagation matcher failed"))?,
        )
    };

    let seed = session_seed(args.seed, rel);
    let mut replacer = Replacer::new(ReplacerConfig {
        strategy: match args.strategy {
            TraceStrategy::Fake => crate::engine::ReplacementStrategy::Fake,
            TraceStrategy::Hash => crate::engine::ReplacementStrategy::Hash,
        },
        seed: Some(seed),
        ..ReplacerConfig::default()
    });
    let name = rel.file_name().and_then(|n| n.to_str());
    // Seeded runs must be reproducible, including session-derived aliases.
    let session = if args.seed.is_some() {
        Session::with_seed(seed, name)
    } else {
        Session::new(name)
    };
    replacer = replacer.with_session_id(session.id.clone());
    let mut multiplicity: HashMap<&str, usize> = HashMap::new();
    for leaf in &leaves {
        if matches!(leaf.kind, Kind::Prose | Kind::Other) {
            *multiplicity.entry(get(leaf)).or_default() += 1;
        }
    }
    let mut candidates: HashMap<String, Candidate> = HashMap::new();
    let mut all_replacements = Vec::new();
    let mut rewritten: HashMap<String, String> = HashMap::new();
    let mut edits: Vec<(usize, String, Option<String>)> = Vec::new();
    progress.report("replace", 0, leaves.len(), true);
    for (index, leaf) in leaves.iter().enumerate() {
        progress.report("replace", index, leaves.len(), false);
        let s = get(leaf);
        match leaf.kind {
            Kind::Skip => continue,
            Kind::Image => {
                if args.strip_images && !s.is_empty() {
                    stats.images_removed += 1;
                    edits.push((leaf.record, leaf.pointer.clone(), Some(String::new())));
                }
                continue;
            }
            Kind::Prose | Kind::Other => {}
        }
        if s.is_empty() {
            continue;
        }
        if let Some(out) = rewritten.get(s) {
            if out != s {
                edits.push((leaf.record, leaf.pointer.clone(), Some(out.clone())));
            }
            continue;
        }
        let mut matches: Vec<PiiMatch> = found.get(s).cloned().unwrap_or_default();
        matches.iter_mut().for_each(narrow_home);
        let mut detected: HashSet<(usize, usize)> =
            matches.iter().map(|m| (m.start, m.end)).collect();
        let mut identity_spans = HashSet::new();
        if let (Some(matcher), None) = (&identity.matcher, approved) {
            let hits = literal_matches(matcher, &identity.literals, s);
            identity_spans.extend(hits.iter().map(|m| (m.start, m.end)));
            matches.extend(hits);
        }
        if let Some(matcher) = &propagation {
            if approved.is_some() {
                // Reviewed values: every occurrence counts as an explicit decision.
                let hits = approved_matches(matcher, &session_literals, s);
                detected.extend(hits.iter().map(|m| (m.start, m.end)));
                matches.extend(hits);
            } else {
                matches.extend(literal_matches(matcher, &session_literals, s));
            }
        }
        // Explicit keep decisions win, before and after overlap merging.
        matches.retain(|m| !keep.contains(&m.matched_text));
        if let Some(kept) = &shared.keep_matcher {
            let spans: Vec<(usize, usize)> =
                kept.find_iter(s).map(|k| (k.start(), k.end())).collect();
            matches.retain(|m| !spans.iter().any(|&(a, b)| m.start < b && a < m.end));
        }
        let mut matches = resolve_overlaps(s, matches);
        matches.retain(|m| !keep.contains(&m.matched_text));
        if matches.is_empty() {
            rewritten.insert(s.to_string(), s.to_string());
            continue;
        }
        for m in &matches {
            let span = (m.start, m.end);
            if detected.contains(&span) {
                continue;
            }
            if identity_spans.contains(&span) {
                stats.identity += 1;
            } else {
                stats.propagated += 1;
            }
        }
        let occurrences = multiplicity.get(s).copied().unwrap_or(1);
        for m in &matches {
            *stats.by_class.entry(m.pattern_name.clone()).or_default() += 1;
            let span = (m.start, m.end);
            let entry = candidates
                .entry(m.matched_text.clone())
                .or_insert_with(|| Candidate {
                    value: m.matched_text.clone(),
                    class: m.pattern_name.clone(),
                    source: if detected.contains(&span) {
                        "detected"
                    } else if identity_spans.contains(&span) {
                        "listed"
                    } else {
                        "propagated"
                    },
                    credential: m.category == crate::engine::patterns::PiiCategory::Authentication,
                    count: 0,
                    context: context(s, m.start, m.end),
                });
            entry.count += occurrences;
        }
        let (out, replacements) = replacer.replace_all(s, &matches);
        stats.replacements += replacements.len();
        all_replacements.extend(replacements);
        rewritten.insert(s.to_string(), out.clone());
        if out != s {
            edits.push((leaf.record, leaf.pointer.clone(), Some(out)));
        }
    }
    let mut changed = vec![false; records.len()];
    for (record, pointer, value) in edits {
        if let (Some(Some(v)), Some(new)) = (records.get_mut(record), value)
            && let Some(slot) = v.pointer_mut(&pointer)
        {
            *slot = Value::String(new);
            changed[record] = true;
        }
    }
    let mut output = Vec::with_capacity(raw.len());
    for (i, line) in lines.iter().enumerate() {
        if changed[i] {
            let value = records[i]
                .as_ref()
                .ok_or_else(|| failure("record vanished"))?;
            output.extend_from_slice(serde_json::to_string(value)?.as_bytes());
            let ending = &line[line.trim_end_matches(['\n', '\r']).len()..];
            output.extend_from_slice(ending.as_bytes());
        } else {
            output.extend_from_slice(line.as_bytes());
        }
    }
    let mut candidates: Vec<Candidate> = candidates.into_values().collect();
    candidates.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.value.cmp(&b.value)));
    let plan = SessionPlan {
        session: rel.to_string_lossy().into_owned(),
        candidates,
    };
    progress.report("done", leaves.len(), leaves.len(), true);
    let Some(output_dir) = &args.output else {
        return Ok((stats, plan));
    };
    let target = output_dir.join(rel);
    publish(&target, &output).map_err(|_| failure("write failed"))?;
    if let Some(keys) = &args.keys {
        let mut key_path = keys.join(rel);
        key_path.set_extension("keys.jsonl");
        if let Some(parent) = key_path.parent() {
            private_dir(parent).map_err(|_| failure("key directory failed"))?;
        }
        if args.force && key_path.exists() {
            fs::remove_file(&key_path).map_err(|_| failure("key file replace failed"))?;
        }
        let config = ReplacerConfig {
            strategy: crate::engine::ReplacementStrategy::Fake,
            seed: None,
            ..ReplacerConfig::default()
        };
        crate::write_key_file(&key_path, &dedup(all_replacements), &session, &config, None)
            .map_err(|_| failure("key file write failed"))?;
    }
    Ok((stats, plan))
}

fn dedup(replacements: Vec<crate::engine::Replacement>) -> Vec<crate::engine::Replacement> {
    let mut seen = HashSet::new();
    replacements
        .into_iter()
        .filter(|r| seen.insert((r.original.clone(), r.replacement.clone())))
        .collect()
}

fn short_id(rel: &Path) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(rel.to_string_lossy().as_bytes()))[..8].to_string()
}

fn private_dir(path: &Path) -> std::io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path)
}

/// Stage next to the target, then rename: a failed run never leaves partial output.
fn publish(target: &Path, bytes: &[u8]) -> Result<()> {
    let parent = target
        .parent()
        .ok_or_else(|| anyhow!("invalid output path"))?;
    private_dir(parent)?;
    let mut tmp = tempfile::Builder::new()
        .prefix(".nym-trace-")
        .tempfile_in(parent)?;
    tmp.write_all(bytes)?;
    tmp.as_file().sync_all()?;
    tmp.persist(target).map_err(|e| e.error)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{plausible, trim_edges};
    use crate::engine::{
        PiiMatch,
        patterns::{Confidence, PiiCategory},
    };

    fn found(text: &str, start: usize, class: &str, category: PiiCategory) -> PiiMatch {
        PiiMatch {
            pattern_name: class.into(),
            matched_text: text.into(),
            start,
            end: start + text.len(),
            confidence: Confidence::High,
            category,
        }
    }

    #[test]
    fn edge_delimiters_are_trimmed_except_for_credentials() {
        let mut user = found("wismut ;", 10, "username", PiiCategory::Social);
        trim_edges(&mut user);
        assert_eq!(
            (user.matched_text.as_str(), user.start, user.end),
            ("wismut", 10, 16)
        );
        let mut secret = found("hunter2;", 3, "password", PiiCategory::Authentication);
        trim_edges(&mut secret);
        assert_eq!(
            (secret.matched_text.as_str(), secret.start, secret.end),
            ("hunter2;", 3, 11)
        );
    }

    #[test]
    fn allowlist_and_system_accounts_are_not_replaced() {
        let allow = super::allowlist(&[]).unwrap();
        let ok = |value: &str, class: &str, category, allow: &std::collections::HashSet<String>| {
            plausible(&found(value, 0, class, category), value, true, allow)
        };
        for account in ["root", "Root", "www-data", "nobody"] {
            assert!(!ok(account, "username", PiiCategory::Social, &allow));
        }
        assert!(ok("wismut", "username", PiiCategory::Social, &allow));
        // Built-in whole-value and per-word name allowlisting.
        for (value, class) in [
            ("Docker", "organization"),
            ("Claude Code", "person"),
            ("docker", "username"),
        ] {
            assert!(!ok(value, class, PiiCategory::Identity, &allow));
        }
        for (value, expected) in [
            ("README.md", false),
            ("rw-r--r", false),
            ("gingface", false),
            ("hs.byteowlz.com", true),
            ("https://hs.byteowlz.com/x", true),
            ("https://github.com/rust-lang/", false),
        ] {
            assert_eq!(
                ok(value, "url", PiiCategory::Network, &allow),
                expected,
                "{value}"
            );
        }
        // Code identifiers are not people; capitalized names in prose are.
        let id = |leaf: &str, value: &str| {
            let start = leaf.find(value).unwrap();
            let mut m = found(value, start, "first_name", PiiCategory::Identity);
            m.end = start + value.len();
            plausible(&m, leaf, true, &allow)
        };
        assert!(!id(r#"name: "greet","#, "greet"));
        assert!(!id("parameters: Type.Object({", "Type"));
        assert!(!id("x.Type", "Type"));
        assert!(!id("Type(x)", "Type"));
        assert!(id("I talked to Jolanta yesterday.", "Jolanta"));
        // User allowlists never veto credentials.
        let mut custom = allow.clone();
        custom.insert("hunter22".into());
        custom.insert("byteowlz".into());
        assert!(ok(
            "hunter22",
            "password",
            PiiCategory::Authentication,
            &custom
        ));
        assert!(!ok("byteowlz", "username", PiiCategory::Social, &custom));
    }
}
