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
    #[arg(short, long)]
    output: PathBuf,
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
    /// Keep embedded base64 images (they may show private content NYM cannot inspect)
    #[arg(long)]
    keep_images: bool,
    /// Do not add the local account/git/host identity to the sensitive literals
    #[arg(long)]
    no_local_identity: bool,
    /// UTF-8 file of additional sensitive literals, one per line (repeatable)
    #[arg(long)]
    sensitive_terms_file: Vec<PathBuf>,
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
    "user",
    "ubuntu",
    "debian",
    "ec2-user",
    "runner",
    "vscode",
    "node",
    "app",
    "codespace",
    "jovyan",
    "localhost",
    "raspberrypi",
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
        r"\b\p{Lu}[\p{Ll}\p{M}]+[ \t]+\p{Lu}[\p{Ll}\p{M}]+|\b\d{4,5}[ \t]+\p{Lu}[\p{Ll}\p{M}]+|(?i)\b(?:user(?:name)?|login|handle|account|author|owner|contact|name)\s*[:=]",
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
    let files = resolve(&args.inputs, &args.output)?;
    if files.is_empty() {
        bail!("no session files (*.jsonl) found");
    }
    if !args.force {
        for (_, rel) in &files {
            if args.output.join(rel).exists() {
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
                    class: "username".into(),
                }),
        );
    }
    let identity = Identity::new(&literals)?;
    let regex = detector(config, false)?;
    let cpus = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let jobs = args
        .jobs
        .unwrap_or(if args.no_ner { cpus } else { cpus.min(8) })
        .max(1);
    // One shared model session: NER is compute-bound, and measured per-worker
    // sessions with split cores were slower than one session using all cores.
    let ner = if args.no_ner {
        None
    } else {
        Some(detector(config, true)?)
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
    let results: Vec<Result<SessionStats>> = pool.install(|| {
        files
            .par_iter()
            .map(|(path, rel)| {
                let result = process(path, rel, &args, &regex, ner.as_ref(), &identity);
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
    for result in &results {
        match result {
            Ok(stats) => merge(&mut totals, stats),
            Err(error) => {
                failed += 1;
                // Errors are generic by construction; never echo session content.
                if !common.quiet {
                    eprintln!("error: {error}");
                }
            }
        }
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
fn resolve(inputs: &[PathBuf], output: &Path) -> Result<Vec<(PathBuf, PathBuf)>> {
    let out = fs::canonicalize(output).ok();
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

/// Shape checks for numeric classes: agent traces are full of timestamps,
/// ports, line numbers and IDs that regex/NER mislabel as phones or cards.
/// PIN, CVV and birth dates are kept regardless (privacy-critical, rare).
fn plausible(m: &PiiMatch) -> bool {
    let text = m.matched_text.as_str();
    let digits = text.chars().filter(char::is_ascii_digit).count();
    let numeric = text
        .chars()
        .all(|c| c.is_ascii_digit() || c.is_whitespace() || "+-.:/()".contains(c));
    match m.pattern_name.as_str() {
        "date" | "time" => false,
        "pin" | "cvv" | "date_of_birth" | "age" | "postcode" => true,
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
    regex: &Detector,
    ner: Option<&Detector>,
    identity: &Identity,
) -> Result<SessionStats> {
    let failure = |what: &str| anyhow!("{what} for session #{}", short_id(rel));
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
    // (parent string, line offset, line) for auto-scope NER on tool output.
    let mut lines_for_ner: Vec<(&str, usize, &str)> = Vec::new();
    let line_ner = ner.is_some() && matches!(args.ner_scope, NerScope::Auto | NerScope::Lines);
    for s in &regex_texts {
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
        for chunk in unique.chunks(64) {
            let results = ner
                .detect_batch(chunk)
                .map_err(|_| failure("NER inference failed"))?;
            for (line, matches) in chunk.iter().zip(results) {
                stats.ner_bytes += line.len();
                by_line.insert(line, matches);
            }
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
        for chunk in ner_texts.chunks(32) {
            let results = ner
                .detect_batch(chunk)
                .map_err(|_| failure("NER inference failed"))?;
            for (s, matches) in chunk.iter().zip(results) {
                stats.scanned_bytes += s.len();
                stats.ner_bytes += s.len();
                // A string may be both prose and other; keep the richer result.
                found.insert(s, matches);
            }
        }
    }

    for matches in found.values_mut() {
        matches.retain(plausible);
    }

    // Session-wide propagation set: identity literals plus every identity-like finding.
    let mut session_literals: Vec<Literal> = Vec::new();
    let mut known = HashSet::new();
    for matches in found.values() {
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
    let mut all_replacements = Vec::new();
    let mut rewritten: HashMap<String, String> = HashMap::new();
    let mut edits: Vec<(usize, String, Option<String>)> = Vec::new();
    for leaf in &leaves {
        let s = get(leaf);
        match leaf.kind {
            Kind::Skip => continue,
            Kind::Image => {
                if !args.keep_images && !s.is_empty() {
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
        let detected: HashSet<(usize, usize)> = matches.iter().map(|m| (m.start, m.end)).collect();
        let mut identity_spans = HashSet::new();
        if let Some(matcher) = &identity.matcher {
            let hits = literal_matches(matcher, &identity.literals, s);
            identity_spans.extend(hits.iter().map(|m| (m.start, m.end)));
            matches.extend(hits);
        }
        if let Some(matcher) = &propagation {
            matches.extend(literal_matches(matcher, &session_literals, s));
        }
        let matches = resolve_overlaps(s, matches);
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
        for m in &matches {
            *stats.by_class.entry(m.pattern_name.clone()).or_default() += 1;
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
    let target = args.output.join(rel);
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
    Ok(stats)
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
