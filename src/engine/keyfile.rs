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

/// Provenance-aware checks used before publishing restored output.
#[path = "restore_verification.rs"]
pub mod restore_verification;

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

        validate_mappings(&replacements)?;

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

/// Validate one-to-one reversibility before reading or committing mappings.
fn validate_mappings(entries: &[Replacement]) -> Result<()> {
    let mut by_original = std::collections::HashMap::new();
    let mut by_replacement = std::collections::HashMap::new();
    for entry in entries {
        if let Some(previous) = by_original.insert(&entry.original, &entry.replacement)
            && previous != &entry.replacement
        {
            return Err(anyhow!(
                "conflicting key-file entries for an original value"
            ));
        }
        if let Some(previous) = by_replacement.insert(&entry.replacement, &entry.original)
            && previous != &entry.original
        {
            return Err(anyhow!(
                "conflicting key-file entries for a replacement value"
            ));
        }
    }
    Ok(())
}

/// OS advisory locks are released on exit, including a crashed writer. Keep
/// the empty lock file in place: unlinking it can split concurrent lock owners.
fn acquire_lock(path: &Path, timeout: Duration) -> Result<File> {
    let lock_path = lock_path_for(path);
    if fs::symlink_metadata(&lock_path).is_ok_and(|meta| meta.file_type().is_symlink()) {
        return Err(anyhow!("key-file lock must not be a symbolic link"));
    }
    let mut options = fs::OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options
        .open(&lock_path)
        .context("Failed to open key-file lock")?;
    let start = Instant::now();
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(std::fs::TryLockError::WouldBlock) => {
                if start.elapsed() >= timeout {
                    return Err(anyhow!("timed out waiting for key-file lock"));
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(std::fs::TryLockError::Error(error)) => {
                return Err(error).context("Failed to acquire key-file lock");
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

/// Persist under the caller's lock using an exclusively created, random,
/// owner-only temp file. Drop cleans up the temp on serialization/write errors.
fn write_atomic(path: &Path, header: &KeyHeader, entries: &[Replacement]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temp = tempfile::NamedTempFile::new_in(parent)
        .context("Failed to create private temp key file")?;
    {
        let mut writer = BufWriter::new(temp.as_file_mut());
        serde_json::to_writer(&mut writer, header)?;
        writeln!(writer)?;
        for entry in entries {
            serde_json::to_writer(&mut writer, entry)?;
            writeln!(writer)?;
        }
        writer.flush()?;
    }
    temp.as_file()
        .sync_all()
        .context("Failed to sync temp key file")?;
    temp.persist(path).context("Failed to replace key file")?;
    Ok(())
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
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).context("Failed to create key-file directory")?;
    }
    let _lock = acquire_lock(path, Duration::from_secs(10))?;
    let latest = KeyFile::load(path)?;

    for existing in latest.as_ref().into_iter().chain(existing) {
        existing.header.check_compatible(
            header.strategy.as_deref(),
            header.seed,
            header.context.as_deref(),
        )?;
    }

    // Merge: existing first, then new entries not already recorded.
    let mut entries: Vec<Replacement> = Vec::new();
    let mut seen: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
    for existing in latest.as_ref().into_iter().chain(existing) {
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

    validate_mappings(&entries)?;
    let header = latest.as_ref().map_or(header, |file| &file.header);
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

    #[cfg(unix)]
    #[test]
    fn test_private_atomic_creation_and_replacement() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.jsonl");
        let header = KeyHeader::new("s1", None, Some("fake"), Some(42), None);
        save_key_file(
            &path,
            None,
            &[sample_replacement("alpha", "alias-a")],
            &header,
        )
        .unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let existing = KeyFile::load(&path).unwrap().unwrap();
        save_key_file(
            &path,
            Some(&existing),
            &[sample_replacement("beta", "alias-b")],
            &header,
        )
        .unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    #[test]
    fn test_conflicting_append_preserves_old_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.jsonl");
        let header = KeyHeader::new("s1", None, Some("fake"), Some(42), None);
        save_key_file(
            &path,
            None,
            &[sample_replacement("alpha", "alias-a")],
            &header,
        )
        .unwrap();
        let existing = KeyFile::load(&path).unwrap().unwrap();
        let bytes = fs::read(&path).unwrap();
        for entry in [
            sample_replacement("beta", "alias-a"),
            sample_replacement("alpha", "alias-b"),
        ] {
            assert!(save_key_file(&path, Some(&existing), &[entry], &header).is_err());
            assert_eq!(fs::read(&path).unwrap(), bytes);
        }
    }

    #[test]
    fn test_concurrent_writers_reload_latest_under_lock() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/keys.jsonl");
        let header = KeyHeader::new("s1", None, Some("fake"), Some(42), None);
        save_key_file(
            &path,
            None,
            &[sample_replacement("initial", "alias-initial")],
            &header,
        )
        .unwrap();
        let snapshot = KeyFile::load(&path).unwrap().unwrap();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
        std::thread::scope(|scope| {
            for index in 0..8 {
                let barrier = barrier.clone();
                let (path, header, snapshot) = (&path, &header, &snapshot);
                scope.spawn(move || {
                    barrier.wait();
                    save_key_file(
                        path,
                        Some(snapshot),
                        &[sample_replacement(
                            &format!("original-{index}"),
                            &format!("alias-{index}"),
                        )],
                        header,
                    )
                    .unwrap();
                });
            }
        });
        let loaded = KeyFile::load(&path).unwrap().unwrap();
        let mut actual: Vec<_> = loaded
            .replacements
            .iter()
            .map(|r| (r.original.clone(), r.replacement.clone()))
            .collect();
        actual.sort();
        let mut expected = vec![("initial".to_string(), "alias-initial".to_string())];
        expected
            .extend((0..8).map(|index| (format!("original-{index}"), format!("alias-{index}"))));
        expected.sort();
        assert_eq!(actual, expected);
        let old = fs::read(&path).unwrap();
        let incompatible = KeyHeader::new("s2", None, Some("hash"), Some(42), None);
        assert!(save_key_file(&path, None, &[], &incompatible).is_err());
        assert_eq!(fs::read(&path).unwrap(), old);
    }

    #[test]
    fn test_concurrent_conflicting_alias_never_corrupts_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.jsonl");
        let header = KeyHeader::new("s1", None, Some("fake"), Some(42), None);
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> = ["alpha", "beta"]
                .into_iter()
                .map(|original| {
                    let barrier = barrier.clone();
                    let (path, header) = (&path, &header);
                    scope.spawn(move || {
                        barrier.wait();
                        save_key_file(
                            path,
                            None,
                            &[sample_replacement(original, "same-alias")],
                            header,
                        )
                        .is_ok()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(results.into_iter().filter(|ok| *ok).count(), 1);
        assert_eq!(KeyFile::load(&path).unwrap().unwrap().replacements.len(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn test_temp_and_destination_symlinks_cannot_clobber_other_files() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let victim = dir.path().join("victim");
        fs::write(&victim, "unchanged").unwrap();
        let path = dir.path().join("keys.jsonl");
        let predictable_temp = dir.path().join("keys.jsonl.tmp");
        symlink(&victim, &predictable_temp).unwrap();
        let header = KeyHeader::new("s1", None, None, None, None);
        save_key_file(
            &path,
            None,
            &[sample_replacement("alpha", "alias-a")],
            &header,
        )
        .unwrap();
        assert_eq!(fs::read_to_string(&victim).unwrap(), "unchanged");
        // A destination symlink pointing at a valid key file is replaced, not followed on write.
        let other = dir.path().join("other.jsonl");
        symlink(&path, &other).unwrap();
        let old = fs::read(&path).unwrap();
        save_key_file(
            &other,
            None,
            &[sample_replacement("beta", "alias-b")],
            &header,
        )
        .unwrap();
        assert_eq!(fs::read(&path).unwrap(), old);
        assert!(
            !fs::symlink_metadata(&other)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        let files: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert!(
            files
                .iter()
                .all(|name| !name.to_string_lossy().starts_with(".tmp"))
        );
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
        use super::super::{PiiMatch, ReplacementStrategy, ReplacerConfig};
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
