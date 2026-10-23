//! Reviewed corpus transformations. Declared completion is not anonymity certification.

use crate::corpus_json::Node;
use crate::engine::{
    Confidence, PiiCategory, PiiMatch, ReplacementStrategy, Replacer, ReplacerConfig,
};
use anyhow::{Result, anyhow, bail};
use clap::{Args, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

const VERSION: u32 = 1;
const MAX_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Args)]
pub(crate) struct CorpusCommand {
    #[command(subcommand)]
    action: Command,
}
#[derive(Debug, Subcommand)]
enum Command {
    /// Emit schemas for private plans/ledgers and public source-handoff manifests
    Schema,
    /// Report aggregate review state without exposing original values or paths
    Status { plan: PathBuf },
    /// Inventory all supported semantic units and opaque assets; decisions start unresolved
    Inventory {
        source: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long,value_enum,default_value_t=Scope::Session)]
        scope: Scope,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// UTF-8 file of deny regexes (one per line); a match disqualifies the source
        #[arg(long, value_name = "FILE")]
        deny_file: Option<PathBuf>,
    },
    /// Pre-fill unresolved units from reviewed values (a `traces anon --plan`
    /// file): every occurrence becomes a numbered placeholder, everything else
    /// is proposed for retention. Proposals are not approvals.
    Propose {
        plan: PathBuf,
        #[arg(long)]
        source: PathBuf,
        /// Reviewed values per session (`nym traces anon --plan` format); private
        #[arg(long, value_name = "FILE")]
        values: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        /// Rename every source to its opaque id, not only names that contain values
        #[arg(long)]
        rename_files: bool,
        /// Where a reviewed value is replaced: in every covered source (default),
        /// or only in the session it was listed for
        #[arg(long, value_enum, default_value_t = ValuesScope::Corpus)]
        values_scope: ValuesScope,
        /// Replace every string under this JSON key with one numbered placeholder
        /// (repeatable or comma-separated), e.g. provider signatures and ids:
        /// thinkingSignature,textSignature,id,parentId,toolCallId,responseId.
        /// Equal values share a token, so links between ids survive.
        #[arg(long = "replace-field", value_name = "KEY", value_delimiter = ',')]
        replace_fields: Vec<String>,
        /// Replace embedded PNG/JPEG images with a neutral 1x1 image of the same type
        #[arg(long)]
        strip_media: bool,
    },
    /// Approve proposals in bulk, by class or basis, with one receipt
    Approve {
        plan: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        /// Who approved what, on which evidence; stored on every approved unit
        #[arg(long)]
        receipt: String,
        /// Approve placeholder proposals whose classes are all listed (repeatable)
        #[arg(long = "class", value_name = "CLASS")]
        classes: Vec<String>,
        /// Approve placeholder proposals of every class
        #[arg(long)]
        all_classes: bool,
        /// Approve retention of units no reviewed value occurs in
        #[arg(long)]
        unflagged: bool,
        /// Approve opaque renames of source files
        #[arg(long)]
        renames: bool,
        /// Approve whole-value placeholders for --replace-field keys
        #[arg(long)]
        fields: bool,
        /// Approve retention of embedded media (still requires a media review)
        #[arg(long)]
        media: bool,
    },
    /// Record the rights and privacy review for every source that has none
    Review {
        plan: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, value_enum)]
        basis: Rights,
        /// Where the rights come from (licence, ownership statement)
        #[arg(long)]
        evidence: String,
        /// Who reviewed privacy, how, and when
        #[arg(long)]
        privacy_review: String,
        /// JSON re-encoding of the sources was reviewed
        #[arg(long)]
        serialization_reviewed: bool,
        /// Embedded media was reviewed
        #[arg(long)]
        media_reviewed: bool,
        /// Publish sources the transformation leaves byte-identical
        #[arg(long)]
        publish_verbatim: bool,
    },
    /// Check immutable sources and complete declared decisions without writing outputs
    Validate {
        plan: PathBuf,
        #[arg(long)]
        source: PathBuf,
    },
    /// Stage an approved candidate bundle; never upload or certify anonymity
    Apply {
        plan: PathBuf,
        #[arg(long)]
        source: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        ledger: PathBuf,
        #[arg(long)]
        dry_run: bool,
        /// Reconcile an existing ledger/bundle; never overwrite mismatched artifacts
        #[arg(long)]
        resume: bool,
    },
    /// Verify every generated source against the private transformation ledger
    Verify {
        #[arg(long)]
        ledger: PathBuf,
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        bundle: PathBuf,
    },
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, ValueEnum, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum ValuesScope {
    Corpus,
    Session,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
enum Scope {
    Session,
    Full,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    version: u32,
    scope: Scope,
    seed: u64,
    /// Regexes that disqualify a source from publication outright. A match is
    /// not redaction: the source must be removed or the policy changed.
    #[serde(default)]
    deny: Vec<String>,
    /// Provenance of `corpus propose`, when proposals were generated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    proposed: Option<Proposed>,
    sources: Vec<Source>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Proposed {
    values_sha256: String,
    values_scope: ValuesScope,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    replace_fields: Vec<String>,
    #[serde(default)]
    strip_media: bool,
    nym_version: String,
}
/// Why `corpus propose` suggested a unit's action. Approval stays explicit.
#[derive(Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum Basis {
    /// Reviewed values occur in the unit and are replaced by placeholders
    Values,
    /// No reviewed value occurs in the unit; proposed for retention
    Unflagged,
    /// Source file renamed to its opaque id
    Rename,
    /// A whole value under a replaced field, or equal to one
    Field,
    /// Embedded media nym cannot inspect; retention proposed, review required
    Media,
    /// Embedded media replaced by a neutral image; nothing of the original kept
    MediaReplaced,
    /// A value occurs in a typed unit that only takes a literal; decide manually
    Manual,
    /// The values file does not cover this source; nothing proposed
    Uncovered,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Proposal {
    basis: Basis,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    classes: Vec<String>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Source {
    id: String,
    path: String,
    sha256: String,
    bytes: u64,
    format: String,
    group: String,
    /// Index into `Plan::deny` when this source matched a deny rule.
    #[serde(default)]
    denied_by: Option<usize>,
    review: Option<Review>,
    units: Vec<Unit>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Review {
    basis: Rights,
    evidence: String,
    privacy_review: String,
    serialization_reviewed: bool,
    media_reviewed: bool,
    /// Required when the transformed source is byte-identical to its input:
    /// publishing it means publishing the original file, which is a decision
    /// rather than a side effect of retaining every unit.
    #[serde(default)]
    publish_verbatim: bool,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
enum Rights {
    Owned,
    Licensed,
    Synthetic,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Unit {
    id: String,
    location: String,
    kind: String,
    original: String,
    /// Explicit identity namespace for equal literals with different meanings.
    #[serde(default)]
    entity: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    proposal: Option<Proposal>,
    approval: Option<String>,
    action: Action,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Action {
    Unresolved,
    Retain,
    Literal {
        value: String,
    },
    Fake {
        pattern: String,
    },
    /// Numbered per identity namespace: `<EMAIL_1>`, same value same number.
    Placeholder {
        class: String,
    },
    Spans {
        edits: Vec<Edit>,
    },
    Asset {
        path: String,
        sha256: String,
    },
    Reference {
        source_id: String,
    },
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Edit {
    start: usize,
    end: usize,
    action: Action,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Ledger {
    version: u32,
    plan: Plan,
    outputs: Vec<Output>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Output {
    id: String,
    path: String,
    sha256: String,
    bytes: u64,
    values: Vec<String>,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn relative(path: &str) -> Result<PathBuf> {
    let p = Path::new(path);
    if path.is_empty()
        || path.contains(['\\', ':', '\0'])
        || p.components().any(|c| !matches!(c, Component::Normal(_)))
    {
        bail!("invalid corpus relative path");
    }
    Ok(p.to_path_buf())
}
fn source_name(path: &Path, root: &Path) -> Result<String> {
    let parts = path
        .strip_prefix(root)?
        .components()
        .map(|component| {
            let Component::Normal(part) = component else {
                bail!("invalid corpus source path");
            };
            let text = part
                .to_str()
                .ok_or_else(|| anyhow!("non-UTF8 corpus source path"))?;
            relative(text)?;
            Ok(text)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(parts.join("/"))
}
fn receipt(value: Option<&str>) -> bool {
    value.is_some_and(|s| !s.trim().is_empty())
}
fn bounded_read(path: &Path, limit: u64) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(limit + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        bail!("corpus byte limit exceeded");
    }
    Ok(bytes)
}
fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let meta =
        fs::symlink_metadata(path).map_err(|_| anyhow!("corpus artifact cannot be inspected"))?;
    if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > MAX_BYTES * 4 {
        bail!("invalid or oversized corpus artifact");
    }
    let bytes =
        bounded_read(path, MAX_BYTES * 4).map_err(|_| anyhow!("corpus artifact cannot be read"))?;
    serde_json::from_slice(&bytes).map_err(|_| anyhow!("invalid corpus artifact schema"))
}
fn private_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    use std::io::Write;
    file.write_all(bytes)?;
    file.as_file().sync_all()?;
    file.persist_noclobber(path)
        .map_err(|_| anyhow!("corpus artifact already exists or cannot be committed"))?;
    Ok(())
}
fn outside(path: &Path, root: &Path) -> Result<()> {
    let absolute = std::path::absolute(path)?;
    let parent = absolute
        .parent()
        .ok_or_else(|| anyhow!("invalid corpus output parent"))?;
    // Resolve existing ancestors, including symlinks, before admitting an output.
    let mut existing = parent;
    while !existing.try_exists()? {
        existing = existing
            .parent()
            .ok_or_else(|| anyhow!("invalid corpus output parent"))?;
    }
    if existing.canonicalize()?.starts_with(root) || root.starts_with(&absolute) {
        bail!("corpus artifacts and bundles must be outside the source tree");
    }
    Ok(())
}
fn files(root: &Path, dir: &Path, depth: usize, out: &mut Vec<PathBuf>) -> Result<()> {
    if depth > 32 || out.len() > 10_000 {
        bail!("corpus traversal limit exceeded");
    }
    let mut entries = fs::read_dir(dir)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort();
    if entries.is_empty() {
        bail!("empty corpus directories require explicit source adaptation");
    }
    if entries.len() > 20_000 {
        bail!("corpus directory entry limit exceeded");
    }
    for path in entries {
        let meta = fs::symlink_metadata(&path)?;
        if meta.file_type().is_symlink() {
            bail!("corpus symlinks require external resolution");
        }
        if meta.is_dir() {
            files(root, &path, depth + 1, out)?;
        } else if meta.is_file() {
            if out.len() >= 10_000 {
                bail!("corpus file limit exceeded");
            }
            source_name(&path, root)?;
            out.push(path);
        } else {
            bail!("unsupported corpus filesystem entry");
        }
    }
    Ok(())
}
fn format(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "json" => "json",
        "jsonl" | "ndjson" => "jsonl",
        "txt" | "text" | "log" | "md" | "csv" | "rs" | "py" | "js" | "ts" | "toml" | "yaml"
        | "yml" => "text",
        _ => "asset",
    }
}
fn documents(bytes: &[u8], format: &str) -> Result<Vec<Option<Node>>> {
    let text = std::str::from_utf8(bytes).map_err(|_| anyhow!("corpus text is not UTF8"))?;
    match format {
        "json" => Ok(vec![Some(Node::parse(text)?)]),
        "jsonl" => text
            .split_inclusive('\n')
            .map(|line| {
                if line
                    .bytes()
                    .all(|b| matches!(b, b' ' | b'\t' | b'\r' | b'\n'))
                {
                    Ok(None)
                } else {
                    Ok(Some(Node::parse(line)?))
                }
            })
            .collect(),
        _ => Ok(vec![Some(Node::String(text.into()))]),
    }
}
fn units(id: &str, path: &str, bytes: &[u8], format: &str) -> Result<Vec<Unit>> {
    let mut rows = vec![("filename".into(), "filename".into(), path.into())];
    if format == "asset" {
        rows.push(("asset".into(), "asset".into(), String::new()));
    } else {
        for (index, node) in documents(bytes, format)?.into_iter().enumerate() {
            if let Some(node) = node {
                node.visit(&format!("/records/{index}"), &mut rows);
            } else {
                rows.push((format!("/records/{index}"), "layout".into(), String::new()));
            }
        }
    }
    Ok(rows
        .into_iter()
        .enumerate()
        .map(|(index, (location, kind, original))| Unit {
            id: format!("{id}:u{index}"),
            location,
            kind,
            original,
            entity: None,
            proposal: None,
            approval: None,
            action: Action::Unresolved,
        })
        .collect())
}
fn compile_deny(patterns: &[String]) -> Result<Vec<regex::Regex>> {
    patterns
        .iter()
        .map(|pattern| regex::Regex::new(pattern).map_err(|_| anyhow!("invalid deny pattern")))
        .collect()
}

fn deny_index(deny: &[regex::Regex], bytes: &[u8]) -> Option<usize> {
    if deny.is_empty() {
        return None;
    }
    let text = String::from_utf8_lossy(bytes);
    deny.iter().position(|pattern| pattern.is_match(&text))
}

fn inventory(root: &Path, scope: Scope, seed: u64, deny: &[String]) -> Result<Plan> {
    let mut paths = Vec::new();
    files(root, root, 0, &mut paths)?;
    let mut keyed = paths
        .into_iter()
        .map(|p| Ok((source_name(&p, root)?, p)))
        .collect::<Result<Vec<_>>>()?;
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    let paths = keyed.into_iter().map(|(_, p)| p).collect::<Vec<_>>();
    if paths.is_empty() {
        bail!("empty corpus");
    }
    let deny_patterns = compile_deny(deny)?;
    let mut total = 0;
    let mut sources = Vec::new();
    for (index, path) in paths.iter().enumerate() {
        let meta = fs::metadata(path)?;
        total += meta.len();
        if meta.len() > 16 * 1024 * 1024 || total > MAX_BYTES {
            bail!("corpus source size limit exceeded");
        }
        let bytes = bounded_read(path, 16 * 1024 * 1024)?;
        let name = source_name(path, root)?;
        let id = format!("f{index:05}");
        let fmt = format(path);
        let denied_by = deny_index(&deny_patterns, &bytes);
        sources.push(Source {
            group: id.clone(),
            denied_by,
            units: units(&id, &name, &bytes, fmt)?,
            id,
            path: name,
            sha256: digest(&bytes),
            bytes: bytes.len() as u64,
            format: fmt.into(),
            review: None,
        });
    }
    Ok(Plan {
        version: VERSION,
        scope,
        seed,
        deny: deny.to_vec(),
        proposed: None,
        sources,
    })
}
fn check_sources(plan: &Plan, root: &Path) -> Result<()> {
    if plan.version != VERSION {
        bail!("unsupported corpus artifact version");
    }
    let actual = inventory(root, plan.scope, plan.seed, &plan.deny)?;
    if actual.sources.len() != plan.sources.len() {
        bail!("corpus source set changed");
    }
    for (a, b) in actual.sources.iter().zip(&plan.sources) {
        if (
            a.id.as_str(),
            a.path.as_str(),
            a.sha256.as_str(),
            a.bytes,
            a.format.as_str(),
            a.units.len(),
        ) != (
            b.id.as_str(),
            b.path.as_str(),
            b.sha256.as_str(),
            b.bytes,
            b.format.as_str(),
            b.units.len(),
        ) {
            bail!("corpus source snapshot changed");
        }
        for (x, y) in a.units.iter().zip(&b.units) {
            if (&x.id, &x.location, &x.kind, &x.original)
                != (&y.id, &y.location, &y.kind, &y.original)
            {
                bail!("corpus inventory was altered");
            }
        }
    }
    Ok(())
}
fn decisions(plan: &Plan) -> Result<()> {
    if let Some((index, source)) = plan
        .sources
        .iter()
        .enumerate()
        .find(|(_, source)| source.denied_by.is_some())
    {
        // Value-free diagnostic: source id plus which deny rule fired.
        let rule = source.denied_by.unwrap_or_default();
        bail!(
            "corpus source {} (index {index}) is disqualified by deny rule {rule}; remove it or change the policy",
            source.id
        );
    }
    for source in &plan.sources {
        let review = source
            .review
            .as_ref()
            .ok_or_else(|| anyhow!("corpus rights/privacy review unresolved"))?;
        if !receipt(Some(&review.evidence))
            || !receipt(Some(&review.privacy_review))
            || source.group.trim().is_empty()
        {
            bail!("corpus review receipt incomplete");
        }
        if matches!(source.format.as_str(), "json" | "jsonl") && !review.serialization_reviewed {
            bail!("corpus JSON re-encoding requires explicit review");
        }
        for unit in &source.units {
            if !receipt(unit.approval.as_deref()) || matches!(unit.action, Action::Unresolved) {
                bail!("corpus unit decisions unresolved");
            }
            // Replaced media keeps nothing of the original; retained media does.
            let replaced = unit.kind == "media" && matches!(unit.action, Action::Literal { .. });
            if matches!(unit.kind.as_str(), "media" | "asset")
                && !replaced
                && !review.media_reviewed
            {
                bail!("native media requires external review");
            }
        }
    }
    Ok(())
}
/// 1x1 grey images without metadata, standing in for stripped media.
const NEUTRAL_PNG: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR42mNoAAAAggCB2kUIOwAAAABJRU5ErkJggg==";
const NEUTRAL_JPEG: &str = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkzODdASFxOQERXRTc4UG1RV19iZ2hnPk1xeXBkeFxlZ2P/wAALCAABAAEBAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/ACv/2Q==";
fn neutral_media(mime: Option<&str>) -> Option<&'static str> {
    match mime? {
        "image/png" => Some(NEUTRAL_PNG),
        "image/jpeg" | "image/jpg" => Some(NEUTRAL_JPEG),
        _ => None,
    }
}
/// The JSON key a unit sits under; array elements inherit their parent key.
fn field_key(location: &str) -> String {
    location
        .split('/')
        .rev()
        .find(|part| !part.is_empty() && !part.bytes().all(|b| b.is_ascii_digit()))
        .unwrap_or_default()
        .replace("~1", "/")
        .replace("~0", "~")
}
/// Replaced fields get their own identity namespace: an id `"1"` is not the
/// number `1` kept elsewhere. A declared entity wins.
fn field_entity(unit: &Unit) -> Option<String> {
    unit.entity.clone().or_else(|| Some("field".into()))
}
/// `thinkingSignature` -> `thinking_signature`, valid as a placeholder class.
fn field_class(key: &str) -> String {
    let mut class = String::new();
    for c in key.chars() {
        if c.is_ascii_uppercase() && !class.is_empty() && !class.ends_with('_') {
            class.push('_');
        }
        class.push(if c.is_ascii_alphanumeric() {
            c.to_ascii_lowercase()
        } else {
            '_'
        });
    }
    class.chars().take(64).collect()
}
fn opaque_name(source: &Source) -> String {
    match Path::new(&source.path).extension().and_then(|e| e.to_str()) {
        Some(ext) if !ext.is_empty() && ext.bytes().all(|b| b.is_ascii_alphanumeric()) => {
            format!("{}.{ext}", source.id)
        }
        _ => source.id.clone(),
    }
}
fn count(map: &mut std::collections::BTreeMap<String, usize>, key: impl Into<String>) {
    *map.entry(key.into()).or_default() += 1;
}
fn basis_name(basis: Basis) -> String {
    serde_json::to_value(basis)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}
/// Turn reviewed values into proposed unit actions. Only undecided units change,
/// and nothing is approved: proposals record why, approval records who agreed.
fn propose(
    plan: &mut Plan,
    root: &Path,
    values: &Path,
    rename_files: bool,
    values_scope: ValuesScope,
    replace_fields: &[String],
    strip_media: bool,
) -> Result<serde_json::Value> {
    check_sources(plan, root)?;
    outside(values, root)?;
    let values_bytes = bounded_read(values, MAX_BYTES * 4)
        .map_err(|_| anyhow!("reviewed values file cannot be read"))?;
    let sessions = crate::traces::reviewed_sessions(values)
        .map_err(|_| anyhow!("invalid reviewed values file"))?;
    // Coverage still needs the source's own entry: the union only adds values,
    // it is not evidence that detection ran on an unlisted source.
    let union = match values_scope {
        ValuesScope::Corpus => Some(crate::traces::reviewed_union(&sessions)?),
        ValuesScope::Session => None,
    };
    // One class per value and namespace, first occurrence in plan order, so a
    // value never gets two different placeholders.
    let mut classes: HashMap<(String, String), String> = HashMap::new();
    let scope = plan.scope;
    let group_of = |source: &Source| match scope {
        Scope::Session => Some(source.group.clone()),
        Scope::Full => None,
    };
    // Field values first, so every unit holding the same id gets the same token
    // and the same class, wherever it sits.
    let fields: HashSet<&str> = replace_fields.iter().map(String::as_str).collect();
    let mut field_values = HashSet::new();
    for source in plan.sources.iter().filter(|s| s.denied_by.is_none()) {
        for unit in &source.units {
            let key = field_key(&unit.location);
            if unit.kind == "string" && !unit.original.is_empty() && fields.contains(key.as_str()) {
                let namespace = serde_json::to_string(&(group_of(source), field_entity(unit)))?;
                let id = (namespace, unit.original.clone());
                classes
                    .entry(id.clone())
                    .or_insert_with(|| field_class(&key));
                field_values.insert(id);
            }
        }
    }
    let mut by_basis = std::collections::BTreeMap::new();
    let mut by_class = std::collections::BTreeMap::new();
    let (mut covered, mut uncovered, mut denied, mut decided) = (0, 0, 0, 0);
    for source in &mut plan.sources {
        if source.denied_by.is_some() {
            denied += 1;
            continue;
        }
        let session = sessions
            .get(&source.path)
            .map(|own| union.as_ref().unwrap_or(own));
        if session.is_some() {
            covered += 1;
        } else {
            uncovered += 1;
        }
        let group = match scope {
            Scope::Session => Some(source.group.clone()),
            Scope::Full => None,
        };
        let renamed = opaque_name(source);
        let mimes: HashMap<String, String> = source
            .units
            .iter()
            .filter(|u| u.location.ends_with("/mimeType") && u.kind == "string")
            .map(|u| {
                (
                    u.location.trim_end_matches("mimeType").to_owned(),
                    u.original.clone(),
                )
            })
            .collect();
        for unit in &mut source.units {
            if !matches!(unit.action, Action::Unresolved) || unit.approval.is_some() {
                decided += 1;
                continue;
            }
            let Some(session) = session else {
                unit.proposal = Some(Proposal {
                    basis: Basis::Uncovered,
                    classes: Vec::new(),
                });
                count(&mut by_basis, basis_name(Basis::Uncovered));
                continue;
            };
            let namespace = serde_json::to_string(&(&group, &unit.entity))?;
            let whole = (
                serde_json::to_string(&(&group, field_entity(unit)))?,
                unit.original.clone(),
            );
            if matches!(unit.kind.as_str(), "string" | "key") && field_values.contains(&whole) {
                let class = classes[&whole].clone();
                unit.entity = field_entity(unit);
                count(&mut by_class, class.clone());
                count(&mut by_basis, basis_name(Basis::Field));
                unit.action = Action::Placeholder {
                    class: class.clone(),
                };
                unit.proposal = Some(Proposal {
                    basis: Basis::Field,
                    classes: vec![class],
                });
                continue;
            }
            let found = session
                .find(&unit.original)
                .into_iter()
                .map(|(start, end, class)| {
                    let class = classes
                        .entry((namespace.clone(), unit.original[start..end].to_owned()))
                        .or_insert(class)
                        .clone();
                    (start, end, class)
                })
                .collect::<Vec<_>>();
            let mut names = found.iter().map(|f| f.2.clone()).collect::<Vec<_>>();
            names.sort();
            names.dedup();
            let (basis, action) = match unit.kind.as_str() {
                "filename" if rename_files || !found.is_empty() => (
                    Basis::Rename,
                    Action::Literal {
                        value: renamed.clone(),
                    },
                ),
                "media" => {
                    let parent = unit.location.trim_end_matches("data");
                    match neutral_media(mimes.get(parent).map(String::as_str)) {
                        Some(image) if strip_media => (
                            Basis::MediaReplaced,
                            Action::Literal {
                                value: image.into(),
                            },
                        ),
                        _ => (Basis::Media, Action::Retain),
                    }
                }
                _ if found.is_empty() && unit.kind != "asset" => (Basis::Unflagged, Action::Retain),
                "string" | "key" => {
                    for (_, _, class) in &found {
                        count(&mut by_class, class.clone());
                    }
                    (
                        Basis::Values,
                        Action::Spans {
                            edits: found
                                .into_iter()
                                .map(|(start, end, class)| Edit {
                                    start,
                                    end,
                                    action: Action::Placeholder { class },
                                })
                                .collect(),
                        },
                    )
                }
                _ => (Basis::Manual, Action::Unresolved),
            };
            count(&mut by_basis, basis_name(basis));
            unit.action = action;
            unit.proposal = Some(Proposal {
                basis,
                classes: names,
            });
        }
    }
    plan.proposed = Some(Proposed {
        values_sha256: digest(&values_bytes),
        values_scope,
        replace_fields: replace_fields.to_vec(),
        strip_media,
        nym_version: env!("CARGO_PKG_VERSION").into(),
    });
    Ok(serde_json::json!({
        "sources": plan.sources.len(),
        "covered_sources": covered,
        "uncovered_sources": uncovered,
        "denied_sources": denied,
        "already_decided_units": decided,
        "proposed_units": by_basis,
        "placeholder_spans": by_class,
        "approved": 0,
        "anonymity_guaranteed": false,
    }))
}
struct Selection {
    receipt: String,
    classes: HashSet<String>,
    all_classes: bool,
    /// Non-value bases approved wholesale: unflagged, rename, media.
    bases: HashSet<Basis>,
}
fn approve(plan: &mut Plan, select: &Selection) -> Result<serde_json::Value> {
    if !receipt(Some(&select.receipt)) {
        bail!("approval receipt is empty");
    }
    if select.classes.is_empty() && !select.all_classes && select.bases.is_empty() {
        bail!(
            "approve selects nothing; pass --class, --all-classes, --unflagged, --renames, --fields or --media"
        );
    }
    let mut approved = std::collections::BTreeMap::new();
    let mut pending = std::collections::BTreeMap::new();
    let mut pending_classes = std::collections::BTreeMap::new();
    let mut seen = HashSet::new();
    for unit in plan.sources.iter_mut().flat_map(|s| &mut s.units) {
        let Some(proposal) = &unit.proposal else {
            continue;
        };
        seen.extend(proposal.classes.iter().cloned());
        if unit.approval.is_some() {
            continue;
        }
        let chosen = !matches!(unit.action, Action::Unresolved)
            && match proposal.basis {
                Basis::Values => {
                    select.all_classes
                        || proposal.classes.iter().all(|c| select.classes.contains(c))
                }
                Basis::Unflagged | Basis::Rename | Basis::Media | Basis::Field => {
                    select.bases.contains(&proposal.basis)
                }
                Basis::MediaReplaced => select.bases.contains(&Basis::Media),
                Basis::Manual | Basis::Uncovered => false,
            };
        if chosen {
            unit.approval = Some(select.receipt.clone());
            count(&mut approved, basis_name(proposal.basis));
        } else {
            count(&mut pending, basis_name(proposal.basis));
            for class in &proposal.classes {
                count(&mut pending_classes, class.clone());
            }
        }
    }
    let mut unknown = select
        .classes
        .iter()
        .filter(|c| !seen.contains(*c))
        .cloned()
        .collect::<Vec<_>>();
    unknown.sort();
    Ok(serde_json::json!({
        "approved_units": approved,
        "pending_units": pending,
        "pending_classes": pending_classes,
        "unknown_classes": unknown,
        "anonymity_guaranteed": false,
    }))
}
fn review_sources(plan: &mut Plan, review: &Review) -> Result<serde_json::Value> {
    if !receipt(Some(&review.evidence)) || !receipt(Some(&review.privacy_review)) {
        bail!("corpus review receipt incomplete");
    }
    let (mut reviewed, mut kept, mut denied) = (0, 0, 0);
    for source in &mut plan.sources {
        if source.denied_by.is_some() {
            denied += 1;
        } else if source.review.is_some() {
            kept += 1;
        } else {
            source.review = Some(review.clone());
            reviewed += 1;
        }
    }
    Ok(serde_json::json!({
        "reviewed_sources": reviewed,
        "already_reviewed_sources": kept,
        "denied_sources": denied,
        "rights_verified_by_nym": false,
    }))
}
fn read_plan(path: &Path) -> Result<Plan> {
    let plan: Plan = read_json(path)?;
    if plan.version != VERSION {
        bail!("unsupported corpus plan version");
    }
    Ok(plan)
}
fn write_plan(path: &Path, plan: &Plan) -> Result<()> {
    private_write(path, &serde_json::to_vec_pretty(plan)?)
}
fn iban_surrogate(original: &str, entropy: &str) -> Result<String> {
    if crate::engine::patterns::iban_span(original, 0, original.len()).is_none() {
        bail!(
            "IBAN surrogate requires a valid country/length/checksum; use a reviewed literal for malformed sensitive originals"
        );
    }
    let compact = original
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect::<String>()
        .to_ascii_uppercase();
    let country = &compact[..2];
    let mut bban = String::new();
    for (index, byte) in compact.bytes().skip(4).enumerate() {
        let random = Sha256::digest(format!("{entropy}:{index}").as_bytes())[0];
        bban.push(char::from(if byte.is_ascii_digit() {
            b'0' + random % 10
        } else {
            b'A' + random % 26
        }));
    }
    let mut remainder = 0_u32;
    for byte in format!("{bban}{country}00").bytes() {
        if byte.is_ascii_digit() {
            remainder = (remainder * 10 + u32::from(byte - b'0')) % 97;
        } else {
            remainder = (remainder * 100 + u32::from(byte - b'A') + 10) % 97;
        }
    }
    let encoded = format!("{country}{:02}{bban}", 98 - remainder);
    let mut letters = encoded.chars();
    Ok(original
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                letters.next().unwrap_or(c)
            } else {
                c
            }
        })
        .collect())
}
fn fake(pattern: &str, original: &str, replacer: &mut Replacer) -> Result<String> {
    let category =
        crate::engine::patterns::get_pattern(pattern).map_or(PiiCategory::Other, |p| p.category);
    if crate::engine::patterns::get_pattern(pattern).is_none()
        && !["first_name", "last_name", "person", "phone_ner"].contains(&pattern)
    {
        bail!("unsupported surrogate type");
    }
    let found = PiiMatch {
        pattern_name: pattern.into(),
        matched_text: original.into(),
        start: 0,
        end: original.len(),
        confidence: Confidence::High,
        category,
    };
    let mut value = replacer.replace(&found).replacement;
    if pattern == "iban" {
        value = iban_surrogate(original, &value)?;
    }
    validate_surrogate(pattern, original, &value)?;
    Ok(value)
}
fn validate_surrogate(pattern: &str, original: &str, value: &str) -> Result<()> {
    if matches!(pattern, "first_name" | "last_name" | "person") {
        if original.chars().any(|c| c.is_alphabetic() && !c.is_ascii())
            && !value.chars().any(|c| c.is_alphabetic() && !c.is_ascii())
        {
            bail!("surrogate changes Unicode/script difficulty; provide a reviewed literal");
        }
        if original.split_whitespace().count() != value.split_whitespace().count() {
            bail!("surrogate changes name component count; provide a reviewed literal");
        }
    }
    if value == original || value.trim().is_empty() {
        bail!("surrogate did not replace the original");
    }
    if let Some(p) = crate::engine::patterns::get_pattern(pattern) {
        let regex = p.regex;
        if !regex
            .find(&value)
            .is_some_and(|m| m.start() == 0 && m.end() == value.len())
        {
            bail!("generated surrogate format failed validation");
        }
    }
    if matches!(pattern, "iban")
        && crate::engine::patterns::iban_span(&value, 0, value.len()).is_none()
    {
        bail!("generated IBAN checksum failed validation");
    }
    if matches!(pattern, "credit_card" | "credit_card_nodash") {
        if !luhn_valid(value) {
            bail!("generated card checksum failed validation");
        }
    }
    Ok(())
}
fn luhn_valid(value: &str) -> bool {
    let digits = value
        .bytes()
        .filter(u8::is_ascii_digit)
        .map(|b| u32::from(b - b'0'))
        .collect::<Vec<_>>();
    let sum = digits
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &d)| {
            if i % 2 == 0 {
                d
            } else {
                let doubled = d * 2;
                if doubled > 9 { doubled - 9 } else { doubled }
            }
        })
        .sum::<u32>();
    !digits.is_empty() && sum % 10 == 0
}
#[derive(Default)]
struct AliasRegistry {
    forward: HashMap<(String, String), String>,
    reverse: HashMap<(String, String), (String, String)>,
}
impl AliasRegistry {
    fn retain(&mut self, scope: &str, original: &str) -> Result<()> {
        self.forward
            .insert((scope.into(), original.into()), original.into());
        let (group, _): (Option<String>, Option<String>) = serde_json::from_str(scope)?;
        self.reverse.insert(
            (serde_json::to_string(&group)?, original.into()),
            (scope.into(), original.into()),
        );
        Ok(())
    }
    fn record(&mut self, scope: &str, original: &str, value: &str) -> Result<()> {
        if self
            .forward
            .insert((scope.into(), original.into()), value.into())
            .is_some_and(|previous| previous != value)
        {
            bail!("same scoped value has inconsistent replacements");
        }
        let (group, _): (Option<String>, Option<String>) = serde_json::from_str(scope)?;
        let namespace = serde_json::to_string(&group)?;
        let identity = (scope.to_string(), original.to_string());
        if self
            .reverse
            .insert((namespace, value.into()), identity.clone())
            .is_some_and(|previous| previous != identity)
        {
            bail!("distinct source values or identities collapsed to one alias");
        }
        Ok(())
    }
}
/// Placeholder numbering per identity namespace and class, in plan order.
#[derive(Default)]
struct Placeholders(HashMap<(String, String), HashMap<String, usize>>);
impl Placeholders {
    fn token(&mut self, scope: &str, class: &str, original: &str) -> Result<String> {
        let label = placeholder_label(class)?;
        let ids = self.0.entry((scope.into(), label.clone())).or_default();
        let next = ids.len() + 1;
        let id = *ids.entry(original.into()).or_insert(next);
        Ok(format!("<{label}_{id}>"))
    }
}
/// `first_name` -> `FIRST_NAME`; the class names the token, so it must be plain.
fn placeholder_label(class: &str) -> Result<String> {
    if class.is_empty()
        || class.len() > 64
        || !class
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
    {
        bail!("placeholder class must be 1-64 ASCII letters, digits, '_' or '-'");
    }
    Ok(class.to_ascii_uppercase().replace('-', "_"))
}
struct Resolver<'a> {
    replacer: &'a mut Replacer,
    placeholders: &'a mut Placeholders,
    aliases: &'a mut AliasRegistry,
    names: &'a HashMap<String, String>,
}
fn resolve(action: &Action, original: &str, scope: &str, r: &mut Resolver) -> Result<String> {
    match action {
        Action::Retain => Ok(original.into()),
        Action::Placeholder { class } => {
            if original.is_empty() {
                bail!("placeholder replaces an empty value");
            }
            let value = r.placeholders.token(scope, class, original)?;
            if value == original {
                bail!("placeholder did not replace the original");
            }
            r.aliases.record(scope, original, &value)?;
            Ok(value)
        }
        Action::Literal { value } => {
            if value == original || value.is_empty() {
                bail!("literal replacement is empty or unchanged");
            }
            r.aliases.record(scope, original, value)?;
            Ok(value.clone())
        }
        Action::Fake { pattern } => {
            let value = fake(pattern, original, r.replacer)?;
            r.aliases.record(scope, original, &value)?;
            Ok(value)
        }
        Action::Reference { source_id } => r
            .names
            .get(source_id)
            .cloned()
            .ok_or_else(|| anyhow!("corpus reference target is missing")),
        Action::Spans { edits } => {
            if edits.is_empty() {
                bail!("empty corpus span edits");
            }
            let mut output = String::new();
            let mut cursor = 0;
            for edit in edits {
                if edit.start < cursor
                    || edit.start >= edit.end
                    || edit.end > original.len()
                    || !original.is_char_boundary(edit.start)
                    || !original.is_char_boundary(edit.end)
                {
                    bail!("invalid or overlapping corpus span");
                }
                if !matches!(
                    edit.action,
                    Action::Retain
                        | Action::Literal { .. }
                        | Action::Fake { .. }
                        | Action::Placeholder { .. }
                        | Action::Reference { .. }
                ) {
                    bail!("unsupported nested corpus span action");
                }
                output.push_str(&original[cursor..edit.start]);
                output.push_str(&resolve(
                    &edit.action,
                    &original[edit.start..edit.end],
                    scope,
                    r,
                )?);
                cursor = edit.end;
            }
            output.push_str(&original[cursor..]);
            Ok(output)
        }
        _ => bail!("corpus action cannot transform this value"),
    }
}
fn namespace(plan: &Plan, source: &Source, unit: &Unit) -> Result<String> {
    let group = match plan.scope {
        Scope::Session => Some(source.group.as_str()),
        Scope::Full => None,
    };
    serde_json::to_string(&(group, &unit.entity)).map_err(Into::into)
}
fn reserve_retained(plan: &Plan, aliases: &mut AliasRegistry) -> Result<()> {
    for source in &plan.sources {
        for unit in &source.units {
            let scope = namespace(plan, source, unit)?;
            if matches!(unit.action, Action::Retain) {
                aliases.retain(&scope, &unit.original)?;
            }
            if let Action::Spans { edits } = &unit.action {
                for edit in edits.iter().filter(|e| matches!(e.action, Action::Retain)) {
                    let value = unit
                        .original
                        .get(edit.start..edit.end)
                        .ok_or_else(|| anyhow!("invalid retained corpus span"))?;
                    aliases.retain(&scope, value)?;
                }
            }
        }
    }
    Ok(())
}
struct TransformState {
    replacers: HashMap<String, Replacer>,
    placeholders: Placeholders,
    aliases: AliasRegistry,
    names: HashMap<String, String>,
}
impl TransformState {
    fn value(&mut self, plan: &Plan, source: &Source, unit: &Unit) -> Result<String> {
        if unit.kind == "asset" {
            return Ok(String::new());
        }
        if !matches!(unit.kind.as_str(), "string" | "key")
            && matches!(
                unit.action,
                Action::Spans { .. } | Action::Fake { .. } | Action::Placeholder { .. }
            )
        {
            bail!("typed corpus unit cannot use text-span, fake or placeholder actions");
        }
        if unit.entity.as_deref().is_some_and(|s| s.trim().is_empty()) {
            bail!("empty corpus identity namespace");
        }
        if unit.kind == "media"
            && let Action::Literal { value } = &unit.action
        {
            // Replacement media is deliberately many-to-one (every stripped image
            // becomes the same neutral image), so it carries no alias identity.
            validate_media(unit, value)?;
            return Ok(value.clone());
        }
        let scope = namespace(plan, source, unit)?;
        let seed = u64::from_le_bytes(
            Sha256::digest(format!("{}:{scope}", plan.seed).as_bytes())[..8].try_into()?,
        );
        let replacer = self.replacers.entry(scope.clone()).or_insert_with(|| {
            Replacer::new(ReplacerConfig {
                strategy: ReplacementStrategy::Fake,
                seed: Some(seed),
                email_domain: "example.invalid".into(),
                placeholder_scope: crate::engine::PlaceholderScope::File,
            })
        });
        let value = resolve(
            &unit.action,
            &unit.original,
            &scope,
            &mut Resolver {
                replacer,
                placeholders: &mut self.placeholders,
                aliases: &mut self.aliases,
                names: &self.names,
            },
        )?;
        if unit.kind == "media" {
            validate_media(unit, &value)?;
        }
        Ok(value)
    }
}
fn validate_media(unit: &Unit, value: &str) -> Result<()> {
    if !matches!(unit.action, Action::Retain | Action::Literal { .. }) {
        bail!("embedded media needs full replacement or explicit retention");
    }
    use base64::Engine;
    if base64::engine::general_purpose::STANDARD
        .decode(value)
        .is_err()
    {
        bail!("embedded media is not valid base64; use an external source adapter");
    }
    Ok(())
}
fn rendered_values(plan: &Plan) -> Result<Vec<Vec<String>>> {
    let names = plan
        .sources
        .iter()
        .map(|s| {
            let name = match &s.units[0].action {
                Action::Retain => s.path.clone(),
                Action::Literal { value } => value.clone(),
                _ => {
                    return Err(anyhow!(
                        "filenames require explicit retention or literal rename"
                    ));
                }
            };
            relative(&name)?;
            Ok((s.id.clone(), name))
        })
        .collect::<Result<HashMap<_, _>>>()?;
    let mut state = TransformState {
        replacers: HashMap::new(),
        placeholders: Placeholders::default(),
        aliases: AliasRegistry::default(),
        names,
    };
    reserve_retained(plan, &mut state.aliases)?;
    plan.sources
        .iter()
        .map(|source| {
            source
                .units
                .iter()
                .map(|unit| state.value(plan, source, unit))
                .collect()
        })
        .collect()
}
fn materialize(source: &Source, root: &Path, values: &[String]) -> Result<Vec<u8>> {
    let original = bounded_read(&root.join(relative(&source.path)?), 16 * 1024 * 1024)?;
    if digest(&original) != source.sha256 {
        bail!("corpus source changed during transformation");
    }
    if source.format == "asset" {
        return match &source.units[1].action {
            Action::Retain => Ok(original),
            Action::Asset { path, sha256 } => {
                let meta = fs::symlink_metadata(path)?;
                if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > 16 * 1024 * 1024
                {
                    bail!("invalid replacement asset");
                }
                let bytes = bounded_read(Path::new(path), 16 * 1024 * 1024)?;
                if bytes.is_empty() || digest(&bytes) != *sha256 {
                    bail!("replacement asset hash mismatch");
                }
                Ok(bytes)
            }
            _ => bail!("opaque assets require explicit retention or hash-bound replacement"),
        };
    }
    if source.format == "text" {
        return Ok(values[1].as_bytes().to_vec());
    }
    let mut next = values[1..].iter().cloned();
    let mut output = String::new();
    for node in documents(&original, &source.format)? {
        if let Some(node) = node {
            output.push_str(&node.render(&mut next)?);
        } else if next.next().as_deref() != Some("") {
            bail!("blank corpus record changed");
        }
        if source.format == "jsonl" {
            output.push('\n');
        }
    }
    if next.next().is_some() {
        bail!("unconsumed corpus units");
    }
    Ok(output.into_bytes())
}
fn prepare(plan: &Plan, root: &Path) -> Result<(Vec<Output>, Vec<Vec<u8>>)> {
    check_sources(plan, root)?;
    decisions(plan)?;
    let values = rendered_values(plan)?;
    let mut outputs = Vec::new();
    let mut payloads = Vec::new();
    let mut names = HashSet::new();
    let mut total = 0_u64;
    for (source, values) in plan.sources.iter().zip(values) {
        let name = relative(&values[0])?;
        if !names.insert(name.clone()) {
            bail!("corpus output filename collision");
        }
        let bytes = materialize(source, root, &values)?;
        // Byte-identical output means the published file is the private input.
        if bytes == bounded_read(&root.join(relative(&source.path)?), 16 * 1024 * 1024)? {
            let declared = source
                .review
                .as_ref()
                .is_some_and(|review| review.publish_verbatim);
            if !declared {
                bail!(
                    "corpus source {} is unchanged by the transformation; declare publish_verbatim to publish the original file deliberately",
                    source.id
                );
            }
        }
        total += bytes.len() as u64;
        if bytes.len() as u64 > MAX_BYTES || total > MAX_BYTES * 4 {
            bail!("corpus transformed output size limit exceeded");
        }
        outputs.push(Output {
            id: source.id.clone(),
            path: format!("sources/{}", name.to_string_lossy()),
            sha256: digest(&bytes),
            bytes: bytes.len() as u64,
            values,
        });
        payloads.push(bytes);
    }
    check_sources(plan, root)?;
    Ok((outputs, payloads))
}
fn declaration(plan: &Plan, dry: bool) -> serde_json::Value {
    serde_json::json!({"sources":plan.sources.len(),"units":plan.sources.iter().map(|s|s.units.len()).sum::<usize>(),"declared_decisions_complete":true,"anonymity_guaranteed":false,"rights_verified_by_nym":false,"dry_run":dry})
}
fn public_manifest(record: &Ledger) -> serde_json::Value {
    serde_json::json!({"version":VERSION,"declaration":declaration(&record.plan,false),"label_sources":record.outputs.iter().map(|o|serde_json::json!({"id":o.id,"path":o.path,"sha256":o.sha256,"bytes":o.bytes})).collect::<Vec<_>>()})
}
fn commit_directory(from: &Path, to: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let from = std::ffi::CString::new(from.as_os_str().as_bytes())?;
        let to = std::ffi::CString::new(to.as_os_str().as_bytes())?;
        // SAFETY: valid NUL-terminated paths remain alive for the syscall.
        #[cfg(target_os = "macos")]
        let result = unsafe { libc::renamex_np(from.as_ptr(), to.as_ptr(), libc::RENAME_EXCL) };
        #[cfg(target_os = "linux")]
        let result = unsafe {
            libc::renameat2(
                libc::AT_FDCWD,
                from.as_ptr(),
                libc::AT_FDCWD,
                to.as_ptr(),
                libc::RENAME_NOREPLACE,
            )
        };
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        compile_error!("corpus directory commits support Linux, macOS and Windows");
        if result != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    #[cfg(windows)]
    fs::rename(from, to)?; // Windows directory rename cannot replace an existing directory.
    Ok(())
}
fn apply(
    plan: Plan,
    root: &Path,
    output: &Path,
    ledger: &Path,
    dry: bool,
    resume: bool,
) -> Result<serde_json::Value> {
    check_sources(&plan, root)?;
    decisions(&plan)?;
    outside(output, root)?;
    outside(ledger, root)?;
    if output.try_exists()? || ledger.try_exists()? {
        if !resume || !ledger.try_exists()? {
            bail!("corpus destinations already exist");
        }
        let previous: Ledger = read_json(ledger)?;
        if serde_json::to_vec(&previous.plan)? != serde_json::to_vec(&plan)? {
            bail!("resume plan differs from the committed ledger");
        }
        if output.try_exists()? {
            let mut result = verify(ledger, root, output)?;
            result["resumed"] = true.into();
            return Ok(result);
        }
        let (expected, _) = prepare(&plan, root)?;
        if previous.version != VERSION
            || serde_json::to_vec(&previous.outputs)? != serde_json::to_vec(&expected)?
        {
            bail!("resume ledger does not reproduce");
        }
    }
    let absolute = std::path::absolute(output)?;
    if std::path::absolute(ledger)?.starts_with(&absolute) {
        bail!("private corpus ledger cannot be inside the candidate bundle");
    }
    let (outputs, payloads) = prepare(&plan, root)?;
    let summary = declaration(&plan, dry);
    if dry {
        return Ok(summary);
    }
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let stage = tempfile::tempdir_in(parent)?;
    for (entry, bytes) in outputs.iter().zip(payloads) {
        let destination = stage.path().join(relative(&entry.path)?);
        fs::create_dir_all(
            destination
                .parent()
                .ok_or_else(|| anyhow!("invalid bundle destination"))?,
        )?;
        private_write(&destination, &bytes)?;
    }
    let record = Ledger {
        version: VERSION,
        plan,
        outputs,
    };
    private_write(
        &stage.path().join("manifest.json"),
        &serde_json::to_vec_pretty(&public_manifest(&record))?,
    )?;
    // The private evidence is committed first; a failed final rename never emits a partial bundle.
    let new_ledger = !ledger.try_exists()?;
    if new_ledger {
        private_write(ledger, &serde_json::to_vec_pretty(&record)?)?;
    }
    if let Err(error) = commit_directory(stage.path(), output) {
        if new_ledger {
            let _ = fs::remove_file(ledger);
        }
        return Err(error);
    }
    Ok(summary)
}
fn verify(ledger: &Path, root: &Path, bundle: &Path) -> Result<serde_json::Value> {
    let record: Ledger = read_json(ledger)?;
    if record.version != VERSION {
        bail!("unsupported corpus ledger version");
    }
    check_sources(&record.plan, root)?;
    decisions(&record.plan)?;
    let actual = rendered_values(&record.plan)?;
    if record.outputs.len() != record.plan.sources.len() {
        bail!("corpus ledger output accounting mismatch");
    }
    for ((source, entry), values) in record.plan.sources.iter().zip(&record.outputs).zip(actual) {
        let expected = materialize(source, root, &values)?;
        let path = bundle.join(relative(&entry.path)?);
        let meta = fs::symlink_metadata(&path)?;
        if meta.file_type().is_symlink() || !meta.is_file() || meta.len() > MAX_BYTES {
            bail!("invalid corpus bundle entry");
        }
        let bytes = fs::read(path)?;
        if entry.id != source.id
            || entry.path != format!("sources/{}", values[0])
            || entry.values != values
            || entry.sha256 != digest(&expected)
            || entry.bytes != expected.len() as u64
            || bytes != expected
        {
            bail!("corpus bundle or ledger was altered");
        }
    }
    if fs::read(bundle.join("manifest.json"))?
        != serde_json::to_vec_pretty(&public_manifest(&record))?
    {
        bail!("corpus public manifest was altered");
    }
    let mut found = Vec::new();
    files(bundle, bundle, 0, &mut found)?;
    if found.len() != record.outputs.len() + 1 {
        bail!("unaccounted corpus bundle files");
    }
    Ok(
        serde_json::json!({"verified":true,"sources":record.outputs.len(),"anonymity_guaranteed":false,"rights_verified_by_nym":false}),
    )
}
pub(crate) fn run(common: &crate::CommonOpts, command: CorpusCommand) -> Result<()> {
    let result: Result<serde_json::Value> = (|| match command.action {
        Command::Schema => Ok(serde_json::from_str(include_str!(
            "../examples/corpus.schema.json"
        ))?),
        Command::Status { plan } => {
            let plan: Plan = read_json(&plan)?;
            if plan.version != VERSION {
                bail!("unsupported corpus plan version");
            }
            let units = plan
                .sources
                .iter()
                .flat_map(|s| &s.units)
                .collect::<Vec<_>>();
            let mut proposed = std::collections::BTreeMap::new();
            for unit in &units {
                if let Some(proposal) = &unit.proposal {
                    count(&mut proposed, basis_name(proposal.basis));
                }
            }
            Ok(
                serde_json::json!({"sources":plan.sources.len(),"units":units.len(),"proposed_units":proposed,"approved_units":units.iter().filter(|u|receipt(u.approval.as_deref())).count(),"unresolved":units.iter().filter(|u|matches!(u.action,Action::Unresolved)||!receipt(u.approval.as_deref())).count(),"unreviewed_sources":plan.sources.iter().filter(|s|s.review.is_none()).count(),"media_units":units.iter().filter(|u|matches!(u.kind.as_str(),"asset"|"media")).count(),"scope":plan.scope,"source_snapshot_checked":false,"anonymity_guaranteed":false}),
            )
        }
        Command::Inventory {
            source,
            output,
            scope,
            seed,
            deny_file,
        } => {
            let root = source.canonicalize()?;
            outside(&output, &root)?;
            let deny = match deny_file {
                Some(path) => {
                    let meta = fs::symlink_metadata(&path)?;
                    if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > 1024 * 1024
                    {
                        bail!("invalid deny policy file");
                    }
                    fs::read_to_string(&path)?
                        .lines()
                        .map(str::trim)
                        .filter(|line| !line.is_empty() && !line.starts_with('#'))
                        .map(str::to_owned)
                        .collect::<Vec<_>>()
                }
                None => Vec::new(),
            };
            let plan = inventory(&root, scope, seed, &deny)?;
            let count = plan.sources.iter().map(|s| s.units.len()).sum::<usize>();
            let denied = plan
                .sources
                .iter()
                .filter(|s| s.denied_by.is_some())
                .count();
            private_write(&output, &serde_json::to_vec_pretty(&plan)?)?;
            Ok(serde_json::json!({
                "sources": plan.sources.len(),
                "units": count,
                "unresolved": count,
                "denied_sources": denied,
            }))
        }
        Command::Propose {
            plan: plan_path,
            source,
            values,
            output,
            rename_files,
            values_scope,
            replace_fields,
            strip_media,
        } => {
            let root = source.canonicalize()?;
            outside(&output, &root)?;
            let mut plan = read_plan(&plan_path)?;
            let summary = propose(
                &mut plan,
                &root,
                &values,
                rename_files,
                values_scope,
                &replace_fields,
                strip_media,
            )?;
            write_plan(&output, &plan)?;
            Ok(summary)
        }
        Command::Approve {
            plan: plan_path,
            output,
            receipt,
            classes,
            all_classes,
            unflagged,
            renames,
            fields,
            media,
        } => {
            let mut plan = read_plan(&plan_path)?;
            let summary = approve(
                &mut plan,
                &Selection {
                    receipt,
                    classes: classes.into_iter().collect(),
                    all_classes,
                    bases: [
                        (unflagged, Basis::Unflagged),
                        (renames, Basis::Rename),
                        (fields, Basis::Field),
                        (media, Basis::Media),
                    ]
                    .into_iter()
                    .filter_map(|(on, basis)| on.then_some(basis))
                    .collect(),
                },
            )?;
            write_plan(&output, &plan)?;
            Ok(summary)
        }
        Command::Review {
            plan: plan_path,
            output,
            basis,
            evidence,
            privacy_review,
            serialization_reviewed,
            media_reviewed,
            publish_verbatim,
        } => {
            let mut plan = read_plan(&plan_path)?;
            let summary = review_sources(
                &mut plan,
                &Review {
                    basis,
                    evidence,
                    privacy_review,
                    serialization_reviewed,
                    media_reviewed,
                    publish_verbatim,
                },
            )?;
            write_plan(&output, &plan)?;
            Ok(summary)
        }
        Command::Validate { plan, source } => {
            let plan: Plan = read_json(&plan)?;
            let root = source.canonicalize()?;
            prepare(&plan, &root)?;
            Ok(serde_json::json!({"declared_decisions_complete":true,"anonymity_guaranteed":false}))
        }
        Command::Apply {
            plan,
            source,
            output,
            ledger,
            dry_run,
            resume,
        } => apply(
            read_json(&plan)?,
            &source.canonicalize()?,
            &output,
            &ledger,
            dry_run,
            resume,
        ),
        Command::Verify {
            ledger,
            source,
            bundle,
        } => verify(&ledger, &source.canonicalize()?, &bundle),
    })();
    match result {
        Ok(value) => {
            if !common.quiet {
                if common.yaml {
                    println!("{}", serde_yaml::to_string(&value)?);
                } else {
                    println!("{}", serde_json::to_string(&value)?);
                }
            }
            Ok(())
        }
        Err(error) => {
            // Diagnostics are constructed from source ids, indices and rule
            // numbers only, so the reason is safe to surface.
            let reason = format!("{error}");
            if common.json {
                println!(
                    "{}",
                    serde_json::json!({"error":{"code":"corpus_blocked","state":"blocked","reason":reason}})
                );
            }
            if common.verbose > 0 && !common.quiet {
                eprintln!("corpus blocked: {reason}");
            }
            bail!("corpus operation blocked: {reason}")
        }
    }
}
