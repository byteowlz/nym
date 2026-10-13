//! NER result cache for `traces anon`.
//!
//! Sessions only grow and tool output repeats, so NER results are cached per
//! text. Entries are keyed by SHA-256 over a random per-user key, the detector
//! fingerprint and the text, so the cache cannot confirm guessed content
//! without the key, and store only offsets and labels, never text. One
//! append-only JSONL file per fingerprint (nym build, executable, NER config)
//! lives in `$XDG_CACHE_HOME/nym/ner`, owner-only.
use crate::engine::{
    PiiMatch,
    patterns::{Confidence, PiiCategory},
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    sync::Mutex,
};

/// Larger cache files are discarded rather than loaded.
const MAX_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Serialize, Deserialize, Clone)]
struct Span {
    s: usize,
    e: usize,
    p: String,
    c: Confidence,
    k: PiiCategory,
}

#[derive(Serialize, Deserialize)]
struct Entry {
    h: String,
    m: Vec<Span>,
}

pub struct NerCache {
    file: PathBuf,
    key: [u8; 32],
    fingerprint: String,
    entries: Mutex<HashMap<String, Vec<Span>>>,
    fresh: Mutex<Vec<Entry>>,
}

fn private_dir(path: &Path) -> Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(path)
        .context("failed to create the NER cache directory")
}

fn private_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
}

fn load_key(dir: &Path) -> Result<[u8; 32]> {
    let path = dir.join("key");
    if let Ok(bytes) = fs::read(&path)
        && let Ok(key) = <[u8; 32]>::try_from(bytes.as_slice())
    {
        return Ok(key);
    }
    use fake::rand::{Rng, SeedableRng, rngs::StdRng};
    let key: [u8; 32] = StdRng::from_os_rng().random();
    let mut file = private_options()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
        .context("failed to create the NER cache key")?;
    file.write_all(&key)?;
    Ok(key)
}

impl NerCache {
    /// Open (or create) the cache for one detector fingerprint.
    pub fn open(dir: &Path, fingerprint: &str) -> Result<Self> {
        private_dir(dir)?;
        let key = load_key(dir)?;
        let name = format!("{:x}", Sha256::digest(fingerprint.as_bytes()));
        let file = dir.join(format!("{}.jsonl", &name[..16]));
        let mut entries = HashMap::new();
        if let Ok(meta) = fs::metadata(&file) {
            if meta.len() > MAX_BYTES {
                let _ = fs::remove_file(&file);
            } else if let Ok(handle) = File::open(&file) {
                for line in BufReader::new(handle).lines().map_while(Result::ok) {
                    // Torn or foreign lines are skipped; they only cost a recomputation.
                    if let Ok(entry) = serde_json::from_str::<Entry>(&line) {
                        entries.insert(entry.h, entry.m);
                    }
                }
            }
        }
        Ok(Self {
            file,
            key,
            fingerprint: fingerprint.to_string(),
            entries: Mutex::new(entries),
            fresh: Mutex::new(Vec::new()),
        })
    }

    fn key_for(&self, text: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.key);
        hasher.update(self.fingerprint.as_bytes());
        hasher.update([0]);
        hasher.update(text.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Cached matches for `text`, rebuilt against the text itself.
    pub fn get(&self, text: &str) -> Option<Vec<PiiMatch>> {
        let entries = self.entries.lock().ok()?;
        let spans = entries.get(&self.key_for(text))?;
        spans
            .iter()
            .map(|span| {
                let matched = text.get(span.s..span.e)?;
                Some(PiiMatch {
                    pattern_name: span.p.clone(),
                    matched_text: matched.to_string(),
                    start: span.s,
                    end: span.e,
                    confidence: span.c,
                    category: span.k,
                })
            })
            .collect()
    }

    pub fn put(&self, text: &str, matches: &[PiiMatch]) {
        let entry = Entry {
            h: self.key_for(text),
            m: matches
                .iter()
                .map(|m| Span {
                    s: m.start,
                    e: m.end,
                    p: m.pattern_name.clone(),
                    c: m.confidence,
                    k: m.category,
                })
                .collect(),
        };
        if let (Ok(mut entries), Ok(mut fresh)) = (self.entries.lock(), self.fresh.lock()) {
            entries.insert(entry.h.clone(), entry.m.clone());
            fresh.push(entry);
        }
    }

    /// Append new entries under an exclusive lock; a failed write only loses cache.
    pub fn flush(&self) -> Result<()> {
        let fresh = std::mem::take(
            &mut *self
                .fresh
                .lock()
                .map_err(|_| anyhow::anyhow!("cache lock poisoned"))?,
        );
        if fresh.is_empty() {
            return Ok(());
        }
        let mut body = String::new();
        for entry in &fresh {
            body.push_str(&serde_json::to_string(entry)?);
            body.push('\n');
        }
        let mut file = private_options()
            .append(true)
            .create(true)
            .open(&self.file)
            .context("failed to open the NER cache")?;
        file.lock().context("failed to lock the NER cache")?;
        file.write_all(body.as_bytes())?;
        file.unlock()?;
        Ok(())
    }
}

/// Run NER on `texts`, serving cached results and caching the rest. `on_done`
/// receives the byte length of every text as it completes.
pub fn detect_cached(
    ner: &crate::engine::Detector,
    cache: Option<&NerCache>,
    texts: &[&str],
    batch: usize,
    mut on_done: impl FnMut(usize),
) -> Result<Vec<Vec<PiiMatch>>, crate::engine::detector::DetectionError> {
    let mut results: Vec<Option<Vec<PiiMatch>>> = vec![None; texts.len()];
    let mut missing = Vec::new();
    for (i, text) in texts.iter().enumerate() {
        match cache.and_then(|c| c.get(text)) {
            Some(hit) => {
                results[i] = Some(hit);
                on_done(text.len());
            }
            None => missing.push(i),
        }
    }
    for chunk in missing.chunks(batch.max(1)) {
        let batch_texts: Vec<&str> = chunk.iter().map(|&i| texts[i]).collect();
        let found = ner.detect_batch(&batch_texts)?;
        for (&i, matches) in chunk.iter().zip(found) {
            if let Some(cache) = cache {
                cache.put(texts[i], &matches);
            }
            on_done(texts[i].len());
            results[i] = Some(matches);
        }
    }
    Ok(results.into_iter().map(Option::unwrap_or_default).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn found(text: &str, value: &str, pattern: &str) -> PiiMatch {
        let start = text.find(value).unwrap();
        PiiMatch {
            pattern_name: pattern.into(),
            matched_text: value.into(),
            start,
            end: start + value.len(),
            confidence: Confidence::High,
            category: PiiCategory::Identity,
        }
    }

    #[test]
    fn round_trip_is_keyed_private_and_text_free() {
        let dir = tempfile::TempDir::new().unwrap();
        let text = "Ich bin Jolanta Wierzbicki aus Köln";
        let matches = vec![
            found(text, "Jolanta", "first_name"),
            found(text, "Köln", "city"),
        ];
        let cache = NerCache::open(dir.path(), "fp-1").unwrap();
        assert!(cache.get(text).is_none());
        cache.put(text, &matches);
        cache.put("nothing here", &[]);
        cache.flush().unwrap();
        // A fresh process sees the same results; other fingerprints see nothing.
        let reopened = NerCache::open(dir.path(), "fp-1").unwrap();
        assert_eq!(reopened.get(text), Some(matches));
        assert_eq!(reopened.get("nothing here"), Some(vec![]));
        assert!(
            NerCache::open(dir.path(), "fp-2")
                .unwrap()
                .get(text)
                .is_none()
        );
        // Offsets only: neither the text nor the matched values are stored.
        for entry in fs::read_dir(dir.path()).unwrap() {
            let bytes = fs::read(entry.unwrap().path()).unwrap();
            let raw = String::from_utf8_lossy(&bytes);
            assert!(!raw.contains("Jolanta") && !raw.contains("Köln"));
        }
        // A cached span that no longer fits the text is a miss, not a panic.
        let short = NerCache::open(dir.path(), "fp-3").unwrap();
        short.put("ab", &[found("abc", "c", "x")]);
        assert!(short.get("ab").is_none());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(dir.path().join("key"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o077,
                0
            );
            for entry in fs::read_dir(dir.path()).unwrap() {
                assert_eq!(
                    entry.unwrap().metadata().unwrap().permissions().mode() & 0o077,
                    0
                );
            }
        }
    }
}
