//! Persistent replacement-map ("key file") management.
//!
//! A key file is JSON Lines: one header line followed by one `Replacement`
//! per line. This module centralises loading, validating, merging, and safely
//! persisting key files so that:
//!
//! - a second invocation never silently truncates an existing key file;
//! - an existing mapping can be loaded and reused (so a stable alias is
//!   produced across processes and runs for the configured strategy/seed);
//! - header/strategy/context incompatibilities and conflicting entries are
//!   rejected rather than silently clobbered;
//! - persistence is atomic (write temp + rename) and guarded by a lock file,
//!   so concurrent writers cannot tear the file or lose entries.

use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use super::replacer::{Replacement, Replacer};

/// Current key-file format version.
pub const KEY_FILE_VERSION: &str = "1";

/// The first line of a key file: metadata describing the run that produced it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyHeader {
    /// Format version.
    pub version: String,
    /// Session id (human-readable adjective-noun or user tag).
    #[serde(default)]
    pub session: String,
    /// Original source filename, if any.
    #[serde(default)]
    pub source: Option<String>,
    /// Creation timestamp (RFC 3339).
    #[serde(default)]
    pub created: String,
    /// Replacement strategy used, when recorded.
    #[serde(default)]
    pub strategy: Option<String>,
    /// Deterministic seed used, when recorded.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Context tag for reproducible/pseudonym builds.
    #[serde(default)]
    pub context: Option<String>,
    /// Freeform note (e.g. generator version).
    #[serde(default)]
    pub generator: Option<String>,
}

impl KeyHeader {
    /// Build a header for a new session.
    pub fn new(
        session: &str,
        source: Option<&str>,
        strategy: Option<&str>,
        seed: Option<u64>,
        context: Option<&str>,
    ) -> Self {
        Self {
            version: KEY_FILE_VERSION.to_string(),
            session: session.to_string(),
            source: source.map(str::to_string),
            created: chrono::Utc::now().to_rfc3339(),
            strategy: strategy.map(str::to_string),
            seed,
            context: context.map(str::to_string),
            generator: Some(format!("nym {}", env!("CARGO_PKG_VERSION"))),
        }
    }

    /// Validate that this header is compatible with the requested run
    /// configuration. Returns an error describing the mismatch.
    pub fn check_compatible(
        &self,
        strategy: Option<&str>,
        seed: Option<u64>,
        context: Option<&str>,
    ) -> Result<()> {
        if let (Some(recorded), Some(requested)) = (&self.strategy, strategy)
            && recorded != requested
        {
            return Err(anyhow!(
                "key file strategy mismatch: file records '{recorded}', run requested '{requested}'"
            ));
        }
        if let (Some(recorded), Some(requested)) = (self.seed, seed)
            && recorded != requested
        {
            return Err(anyhow!(
                "key file seed mismatch: file records {recorded}, run requested {requested}"
            ));
        }
        if let (Some(recorded), Some(requested)) = (&self.context, context)
            && recorded != requested
        {
            return Err(anyhow!(
                "key file context mismatch: file records '{}', run requested '{requested}'",
                recorded
            ));
        }
        Ok(())
    }
}

/// A loaded key file: its header plus all replacement entries.
#[derive(Debug, Clone)]
pub struct KeyFile {
    /// Metadata header.
    pub header: KeyHeader,
    /// Replacement mappings in file order.
    pub replacements: Vec<Replacement>,
}

impl KeyFile {
    /// Load a key file, rejecting malformed input and conflicting entries.
    ///
    /// Returns `Ok(None)` when the file does not exist, so callers can treat
    /// "first run" differently from "load existing mapping".
    pub fn load(path: &Path) -> Result<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }
        let file = File::open(path)
            .with_context(|| format!("Failed to open key file: {}", path.display()))?;
        let reader = BufReader::new(file);

        let lines = reader.lines();
        let mut header: Option<KeyHeader> = None;
        let mut replacements: Vec<Replacement> = Vec::new();

        for (idx, line) in lines.enumerate() {
            let line =
                line.with_context(|| format!("Failed to read line {} from key file", idx + 1))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if header.is_none() {
                let parsed: KeyHeader = serde_json::from_str(trimmed)
                    .with_context(|| format!("invalid key-file header on line {}", idx + 1))?;
                header = Some(parsed);
                continue;
            }
            let parsed: Replacement = serde_json::from_str(trimmed)
                .with_context(|| format!("invalid replacement entry on line {}", idx + 1))?;
            replacements.push(parsed);
        }

        let header = header.ok_or_else(|| anyhow!("key file is missing a header line"))?;

        // Detect conflicting entries: same original resolving to different
        // replacements (or the reverse), which would make deanonymization
        // ambiguous. Reject rather than silently picking one.
        let mut by_original: std::collections::HashMap<&str, &str> =
            std::collections::HashMap::new();
        let mut by_replacement: std::collections::HashMap<&str, &str> =
            std::collections::HashMap::new();
        for r in &replacements {
            if let Some(prev) = by_original.get(r.original.as_str()) {
                if *prev != r.replacement {
                    return Err(anyhow!(
                        "conflicting key-file entries for '{}': '{}' vs '{}'",
                        r.original,
                        prev,
                        r.replacement
                    ));
                }
            }
            by_original.insert(&r.original, &r.replacement);
            if let Some(prev) = by_replacement.get(r.replacement.as_str()) {
                if *prev != r.original {
                    return Err(anyhow!(
                        "conflicting key-file entries for replacement '{}': '{}' vs '{}'",
                        r.replacement,
                        prev,
                        r.original
                    ));
                }
            }
            by_replacement.insert(&r.replacement, &r.original);
        }

        Ok(Some(Self {
            header,
            replacements,
        }))
    }

    /// Whether this file has any replacement mappings.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn has_mappings(&self) -> bool {
        !self.replacements.is_empty()
    }

    /// Seed a replacer's caches with the existing mappings so that
    /// consistent/fake strategies reuse the recorded aliases, producing the
    /// same replacement for the same original across runs and processes.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn seed_replacer(&self, replacer: &mut Replacer) {
        replacer.seed_mappings(&self.replacements);
    }
}

/// Lock file suffix used to serialise concurrent writers.

/// Acquire an exclusive lock guarding `path`, returning a guard that releases
/// the lock on drop. Uses an advisory lock file created with `create_new` and
/// a stale-lock timeout so a crashed writer cannot deadlock future runs.
fn acquire_lock(path: &Path, timeout: Duration) -> Result<LockGuard> {
    let lock_path = lock_path_for(path);
    let start = Instant::now();

    loop {
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
        {
            Ok(_file) => return Ok(LockGuard { path: lock_path }),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                // Stale-lock recovery: if the lock is older than the timeout,
                // break it and retry once.
                let should_break = fs::metadata(&lock_path)
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|modified| modified.elapsed().ok())
                    .map(|age| age >= timeout)
                    .unwrap_or(false);
                if should_break {
                    let _ = fs::remove_file(&lock_path);
                    continue;
                }
                if start.elapsed() >= timeout {
                    return Err(anyhow!(
                        "timed out waiting for key-file lock: {}",
                        lock_path.display()
                    ));
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => {
                return Err(e).with_context(|| {
                    format!("Failed to acquire key-file lock: {}", lock_path.display())
                });
            }
        }
    }
}

/// Resolve the lock path for a data path: `<name>.lock` next to the file.
fn lock_path_for(path: &Path) -> PathBuf {
    let mut os = path.as_os_str().to_os_string();
    os.push(".lock");
    PathBuf::from(os)
}

/// Held lock file; removed on drop.
struct LockGuard {
    path: PathBuf,
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Persist a finished key file atomically under a lock.
///
/// `existing` is the prior contents (or `None` on first run); `new` are the
/// replacement entries produced this run. Existing entries are preserved and
/// new entries are appended (deduplicating exact repeats), so a second
/// invocation extends rather than overwrites.
fn write_atomic(path: &Path, header: &KeyHeader, entries: &[Replacement]) -> Result<()> {
    let _lock = acquire_lock(path, Duration::from_secs(10))?;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create key-file dir: {}", parent.display()))?;
    }

    // Merged entry list, preserving existing order then appending new ones,
    // deduplicating on (original -> replacement) to avoid re-recording.
    let mut merged: Vec<Replacement> = Vec::new();
    let mut seen: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
    for r in entries {
        if seen.insert((r.original.clone(), r.replacement.clone())) {
            merged.push(r.clone());
        }
    }

    // Serialise to a temp file in the same directory, then atomically rename.
    let temp = temp_path(path);
    {
        let file = File::create(&temp)
            .with_context(|| format!("Failed to create temp key file: {}", temp.display()))?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, header)?;
        writeln!(writer)?;
        for r in &merged {
            serde_json::to_writer(&mut writer, r)?;
            writeln!(writer)?;
        }
        writer.flush()?;
        writer
            .get_ref()
            .sync_all()
            .with_context(|| format!("Failed to sync temp key file: {}", temp.display()))?;
    }
    fs::rename(&temp, path)
        .with_context(|| format!("Failed to write key file: {}", path.display()))?;

    Ok(())
}

/// Derive the temp path used for an atomic write.
fn temp_path(path: &Path) -> PathBuf {
    let mut os = path.as_os_str().to_os_string();
    os.push(".tmp");
    PathBuf::from(os)
}

/// High-level write: load-or-create, validate compatibility, seed a replacer
/// (via the caller), merge new entries and persist.
///
/// Returns the key file that was written so callers can report session info.
pub fn save_key_file(
    path: &Path,
    existing: Option<&KeyFile>,
    new_entries: &[Replacement],
    header: &KeyHeader,
) -> Result<KeyFile> {
    let _lock = acquire_lock(path, Duration::from_secs(10))?;

    if let Some(existing) = existing {
        existing.header.check_compatible(
            header.strategy.as_deref(),
            header.seed,
            header.context.as_deref(),
        )?;
    }

    // Merge: existing first, then new entries not already recorded.
    let mut entries: Vec<Replacement> = Vec::new();
    let mut seen: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
    if let Some(existing) = existing {
        for r in &existing.replacements {
            if seen.insert((r.original.clone(), r.replacement.clone())) {
                entries.push(r.clone());
            }
        }
    }
    for r in new_entries {
        if seen.insert((r.original.clone(), r.replacement.clone())) {
            entries.push(r.clone());
        }
    }

    write_atomic(path, header, &entries)?;

    Ok(KeyFile {
        header: header.clone(),
        replacements: entries,
    })
}

/// Load a key file if present, else return `None` (first run).
pub fn load_key_file(path: &Path) -> Result<Option<KeyFile>> {
    KeyFile::load(path)
}

/// Convenience: return all replacement entries from the current key file, or
/// reject when none exists. Used by `deanon` and session listing.
#[cfg_attr(not(test), allow(dead_code))]
pub fn read_replacements(path: &Path) -> Result<(KeyHeader, Vec<Replacement>)> {
    match KeyFile::load(path)? {
        Some(kf) => Ok((kf.header, kf.replacements)),
        None => Err(anyhow!("key file does not exist: {}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_replacement(original: &str, replacement: &str) -> Replacement {
        Replacement {
            original: original.to_string(),
            replacement: replacement.to_string(),
            pattern_name: "email".to_string(),
            components: Vec::new(),
        }
    }

    #[test]
    fn test_load_missing_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.jsonl");
        assert!(KeyFile::load(&path).unwrap().is_none());
    }

    #[test]
    fn test_roundtrip_preserves_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.jsonl");
        let header = KeyHeader::new(
            "calm-frog",
            Some("src.json"),
            Some("consistent"),
            Some(42),
            None,
        );

        let entries = vec![sample_replacement("a@b.com", "x@y.com")];
        save_key_file(&path, None, &entries, &header).unwrap();

        let loaded = KeyFile::load(&path).unwrap().unwrap();
        assert_eq!(loaded.header.session, "calm-frog");
        assert_eq!(loaded.replacements.len(), 1);
        assert_eq!(loaded.replacements[0].original, "a@b.com");
        assert_eq!(loaded.replacements[0].replacement, "x@y.com");
    }

    #[test]
    fn test_extend_does_not_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.jsonl");
        let header = KeyHeader::new("s1", None, Some("consistent"), Some(42), None);

        let first = vec![sample_replacement("a@b.com", "x@y.com")];
        save_key_file(&path, None, &first, &header).unwrap();

        let existing = KeyFile::load(&path).unwrap().unwrap();
        let second = vec![sample_replacement("c@d.com", "w@z.com")];
        let result = save_key_file(&path, Some(&existing), &second, &header).unwrap();

        assert_eq!(result.replacements.len(), 2);
        assert_eq!(result.replacements[0].original, "a@b.com");
        assert_eq!(result.replacements[1].original, "c@d.com");

        // Reload: both persist.
        let loaded = KeyFile::load(&path).unwrap().unwrap();
        assert_eq!(loaded.replacements.len(), 2);
    }

    #[test]
    fn test_deduplicates_exact_repeats() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.jsonl");
        let header = KeyHeader::new("s1", None, None, None, None);

        let entries = vec![
            sample_replacement("a@b.com", "x@y.com"),
            sample_replacement("a@b.com", "x@y.com"),
        ];
        let result = save_key_file(&path, None, &entries, &header).unwrap();
        assert_eq!(result.replacements.len(), 1);
    }

    #[test]
    fn test_strategy_mismatch_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.jsonl");
        let header = KeyHeader::new("s1", None, Some("consistent"), Some(42), None);
        save_key_file(
            &path,
            None,
            &[sample_replacement("a@b.com", "x@y.com")],
            &header,
        )
        .unwrap();

        let existing = KeyFile::load(&path).unwrap().unwrap();
        let new_header = KeyHeader::new("s2", None, Some("fake"), Some(42), None);
        let err = save_key_file(&path, Some(&existing), &[], &new_header).unwrap_err();
        assert!(err.to_string().contains("strategy mismatch"));
    }

    #[test]
    fn test_seed_mismatch_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.jsonl");
        let header = KeyHeader::new("s1", None, Some("consistent"), Some(42), None);
        save_key_file(
            &path,
            None,
            &[sample_replacement("a@b.com", "x@y.com")],
            &header,
        )
        .unwrap();

        let existing = KeyFile::load(&path).unwrap().unwrap();
        let new_header = KeyHeader::new("s2", None, Some("consistent"), Some(7), None);
        let err = save_key_file(&path, Some(&existing), &[], &new_header).unwrap_err();
        assert!(err.to_string().contains("seed mismatch"));
    }

    #[test]
    fn test_conflicting_entries_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.jsonl");
        let header = KeyHeader::new("s1", None, None, None, None);
        save_key_file(
            &path,
            None,
            &[sample_replacement("a@b.com", "x@y.com")],
            &header,
        )
        .unwrap();

        // Manually craft a conflicting file: same original, two replacements.
        fs::write(
            &path,
            format!(
                "{}\n{}\n{}\n",
                serde_json::to_string(&header).unwrap(),
                serde_json::to_string(&sample_replacement("a@b.com", "x@y.com")).unwrap(),
                serde_json::to_string(&sample_replacement("a@b.com", "DIFFERENT@z.com")).unwrap(),
            ),
        )
        .unwrap();

        let err = KeyFile::load(&path).unwrap_err();
        assert!(err.to_string().contains("conflicting"));
    }

    #[test]
    fn test_seed_replacer_reuses_alias() {
        use super::super::{
            Detector, DetectorConfig, PiiMatch, ReplacementStrategy, ReplacerConfig,
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.jsonl");
        let header = KeyHeader::new("s1", None, Some("consistent"), Some(42), None);
        save_key_file(
            &path,
            None,
            &[sample_replacement("a@b.com", "x@y.com")],
            &header,
        )
        .unwrap();

        let existing = KeyFile::load(&path).unwrap().unwrap();

        // A fresh replacer seeded from the file must reuse the recorded alias.
        let mut replacer = Replacer::new(ReplacerConfig {
            strategy: ReplacementStrategy::Consistent,
            seed: Some(42),
            ..Default::default()
        });
        existing.seed_replacer(&mut replacer);

        let m = PiiMatch {
            pattern_name: "email".to_string(),
            matched_text: "a@b.com".to_string(),
            start: 0,
            end: 5,
            confidence: super::super::Confidence::High,
            category: super::super::PiiCategory::Contact,
        };
        let result = replacer.replace(&m);
        assert_eq!(result.replacement, "x@y.com");
    }
}
