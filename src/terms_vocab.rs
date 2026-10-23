//! Known-vocabulary filtering for corpus term discovery.
//!
//! Dictionaries and user word lists shorten the review queue; they never exempt
//! a term from redaction. Matching is case-aware by default: a dictionary hit
//! written with capitals (a name like "Grace" or a surname that is also a word)
//! is only excluded when the corpus also writes it in lowercase, because names
//! are capitalised in running text and ordinary words are not. Measured on the
//! frozen real-session benchmark: English + seven languages excluded 45% of
//! recurring candidates while keeping 91/91 planted name components, against
//! 83/91 with case-insensitive matching.
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};

const MAX_LIST_BYTES: u64 = 64 * 1024 * 1024;
const MAX_LIST_ENTRIES: usize = 5_000_000;

/// Provenance of the vocabulary filter recorded in the discovery artifact.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VocabularyFilter {
    pub lists: Vec<ListRef>,
    pub case_aware: bool,
    pub exclude_shapes: bool,
    /// Occurrences skipped before counting (lowercase hits and noise shapes).
    pub excluded_occurrences: u64,
    /// Capitalised hits dropped at ranking because the corpus also writes them lowercase.
    pub excluded_capitalized: u64,
    /// Candidates ranked after the rest because a demotion list contains them.
    pub demoted: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ListRef {
    pub name: String,
    /// `exclude` removes from the queue; `demote` only ranks later.
    pub role: String,
    pub sha256: String,
    pub entries: u64,
}

impl VocabularyFilter {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.lists.iter().any(|list| {
            list.name.is_empty()
                || list.name.len() > 128
                || !matches!(list.role.as_str(), "exclude" | "demote")
                || list.sha256.len() != 64
                || !list.sha256.bytes().all(|b| b.is_ascii_hexdigit())
        }) {
            bail!("invalid vocabulary filter provenance");
        }
        Ok(())
    }
}

/// What the filter says about one term.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Verdict {
    Keep,
    /// Numbers, hashes and version strings when shape exclusion is on.
    Noise,
    /// Known word written in lowercase (or case-insensitive mode): exclude now.
    KnownLowercase,
    /// Known word written with capitals: exclude only if a lowercase form is seen.
    KnownCapitalized,
}

pub struct KnownVocabulary {
    words: HashSet<String>,
    demote: HashSet<String>,
    case_aware: bool,
    exclude_shapes: bool,
    lists: Vec<ListRef>,
}

impl KnownVocabulary {
    pub fn new(case_aware: bool, exclude_shapes: bool) -> Self {
        Self {
            words: HashSet::new(),
            demote: HashSet::new(),
            case_aware,
            exclude_shapes,
            lists: Vec::new(),
        }
    }

    pub fn is_active(&self) -> bool {
        !self.words.is_empty() || !self.demote.is_empty() || self.exclude_shapes
    }

    pub fn add_exclusions(&mut self, name: &str, path: &Path) -> Result<()> {
        let (entries, digest) = read_list(path)?;
        self.lists.push(ListRef {
            name: name.to_owned(),
            role: "exclude".into(),
            sha256: digest,
            entries: entries.len() as u64,
        });
        self.words.extend(entries);
        Ok(())
    }

    pub fn add_demotions(&mut self, name: &str, path: &Path) -> Result<()> {
        let (entries, digest) = read_list(path)?;
        self.lists.push(ListRef {
            name: name.to_owned(),
            role: "demote".into(),
            sha256: digest,
            entries: entries.len() as u64,
        });
        self.demote.extend(entries);
        Ok(())
    }

    pub(crate) fn classify(&self, term: &str) -> Verdict {
        if self.exclude_shapes && noise_shape(term) {
            return Verdict::Noise;
        }
        if !contains(&self.words, term) {
            return Verdict::Keep;
        }
        if !self.case_aware || term == term.to_lowercase() {
            Verdict::KnownLowercase
        } else {
            Verdict::KnownCapitalized
        }
    }

    pub(crate) fn demoted(&self, term: &str) -> bool {
        contains(&self.demote, term)
    }

    pub(crate) fn provenance(&self) -> VocabularyFilter {
        VocabularyFilter {
            lists: self.lists.clone(),
            case_aware: self.case_aware,
            exclude_shapes: self.exclude_shapes,
            excluded_occurrences: 0,
            excluded_capitalized: 0,
            demoted: 0,
        }
    }
}

/// A term is known when its lowercase form is listed, or when it is a compound
/// (`read_file`, `readFile`, `v2.config`, a multi-word phrase) whose parts all are.
fn contains(list: &HashSet<String>, term: &str) -> bool {
    if list.is_empty() {
        return false;
    }
    let lower = term.to_lowercase();
    if list.contains(&lower) {
        return true;
    }
    let parts = split_parts(term);
    parts.len() > 1 && parts.iter().all(|part| list.contains(&part.to_lowercase()))
}

fn split_parts(term: &str) -> Vec<String> {
    let mut parts = Vec::new();
    for chunk in term.split(|c: char| c.is_whitespace() || matches!(c, '_' | '.' | '-' | '\'')) {
        let mut current = String::new();
        let mut previous_lower = false;
        for c in chunk.chars() {
            if c.is_uppercase() && previous_lower && !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
            previous_lower = c.is_lowercase();
            current.push(c);
        }
        if !current.is_empty() {
            parts.push(current);
        }
    }
    parts
}

/// Numbers, hex digests/commit ids and version strings: high-recurrence noise
/// that no dictionary covers and that a reviewer never needs to see.
pub(crate) fn noise_shape(term: &str) -> bool {
    let digits_only = term
        .chars()
        .filter(|c| !matches!(c, '.' | '-' | '_' | ':' | '/'))
        .all(|c| c.is_ascii_digit());
    let hex = term.len() >= 7 && term.chars().all(|c| c.is_ascii_hexdigit());
    let version = {
        let rest = term.strip_prefix(['v', 'V']).unwrap_or(term);
        rest.contains('.')
            && rest
                .split('.')
                .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
    };
    (digits_only && term.chars().any(|c| c.is_ascii_digit())) || hex || version
}

/// Read a word list: plain one-per-line, frequency lists (`word count`) and
/// hunspell dictionaries (`word/FLAGS`, numeric header) all reduce to the first
/// token. Lines starting with `#` are comments.
fn read_list(path: &Path) -> Result<(HashSet<String>, String)> {
    let meta = fs::metadata(path).map_err(|_| anyhow!("word list cannot be read"))?;
    if !meta.is_file() || meta.len() > MAX_LIST_BYTES {
        bail!("word list is not a regular file within the size limit");
    }
    let bytes = fs::read(path).map_err(|_| anyhow!("word list cannot be read"))?;
    let digest = crate::terms::sha256(&bytes);
    let text = String::from_utf8_lossy(&bytes);
    let mut words = HashSet::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let token = line.split_whitespace().next().unwrap_or_default();
        let token = token.split('/').next().unwrap_or_default();
        if index == 0 && token.chars().all(|c| c.is_ascii_digit()) {
            continue; // hunspell entry-count header
        }
        if token.chars().count() >= 2 {
            words.insert(token.to_lowercase());
        }
        if words.len() > MAX_LIST_ENTRIES {
            bail!("word list exceeds the entry limit");
        }
    }
    if words.is_empty() {
        bail!("word list contains no entries");
    }
    Ok((words, digest))
}

/// Pinned, downloadable lists. Content is CC-BY-SA-4.0 (hermitdave/FrequencyWords,
/// OpenSubtitles 2018 top 50k), so nym downloads on request and never bundles it.
pub struct CatalogEntry {
    pub name: &'static str,
    pub sha256: &'static str,
    pub bytes: u64,
}

pub const CATALOG_SOURCE: &str = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/525f9b560de45753a5ea01069454e72e9aa541c6/content/2018";
pub const CATALOG_LICENSE: &str =
    "CC-BY-SA-4.0, FrequencyWords by Hermit Dave (OpenSubtitles 2018 frequency lists)";

pub const CATALOG: &[CatalogEntry] = &[
    CatalogEntry {
        name: "en",
        sha256: "5351ff405b1126ef555791dd4d9798a48e3e9a501a9fc481a9da957752cfb458",
        bytes: 622_749,
    },
    CatalogEntry {
        name: "de",
        sha256: "d9e50546fd7e8b6fe6542a2b33c51d1331092b2a3916ec09f80d97856068705b",
        bytes: 662_497,
    },
    CatalogEntry {
        name: "fr",
        sha256: "f81f7c570b6433764da99aa30f4dfb08d81c5926301377b709af23a13b1f9596",
        bytes: 653_452,
    },
    CatalogEntry {
        name: "es",
        sha256: "dcff3ad4316192f4dc4ff7d26e637c6ff314ef1ca0f3f720c5649018a71056c0",
        bytes: 658_626,
    },
    CatalogEntry {
        name: "it",
        sha256: "bb96cdcb56d28342c1e909db6b2525448b7767136b7e86a2ccc649a1be66fc19",
        bytes: 653_205,
    },
    CatalogEntry {
        name: "nl",
        sha256: "099bd0c27b514d54284360ad9ce2b4ef7b6903b22e848b67ab10f6aa2adb4baa",
        bytes: 650_762,
    },
    CatalogEntry {
        name: "pl",
        sha256: "50bf8d30d1ed4f5cedceb3e40e74d793ec6bec0f16d9f3dd0a2c9e313402e10f",
        bytes: 676_499,
    },
    CatalogEntry {
        name: "pt",
        sha256: "f97704382f97273ff59488b1885e7ac90d20f81fd92e5fbad3e3ce43633f495c",
        bytes: 652_361,
    },
    CatalogEntry {
        name: "sv",
        sha256: "566d0614a54f3c243bca4097c9bb824d46275abff31582157e73c166b7089500",
        bytes: 623_913,
    },
    CatalogEntry {
        name: "da",
        sha256: "6a83e4deb5e38d873b1ccfb4be4fb25903878c07c8522af708027ac8b540c6b1",
        bytes: 616_396,
    },
    CatalogEntry {
        name: "no",
        sha256: "d5808d8ec04603b765c09c06a03f0273ccf36bb35a3902287e56b7e7d872ca92",
        bytes: 601_916,
    },
    CatalogEntry {
        name: "fi",
        sha256: "8bc1d43dd54ea0c4432b1f6ae497759a332e984224819cf8a4f2337681c803b4",
        bytes: 691_168,
    },
    CatalogEntry {
        name: "cs",
        sha256: "1fb0576ef07cf5a543789d394cde1168573fc9e7adf427fd143249fd5547458d",
        bytes: 656_472,
    },
    CatalogEntry {
        name: "tr",
        sha256: "480192844e7fdfe9591bbf0b9cfbb96ed5751cfbba7024fd45d8cb7508e07d5c",
        bytes: 711_935,
    },
];

/// Directory for installed and user-provided lists: `$XDG_DATA_HOME/nym/dicts`.
pub fn dict_dir() -> PathBuf {
    crate::config::data_dir().join("dicts")
}

/// Resolve `--dict NAME`: `system` is the OS word list, anything else an
/// installed or user-placed `NAME.txt` in the dictionary directory.
pub fn resolve_dict(name: &str) -> Result<PathBuf> {
    if name == "system" {
        for candidate in ["/usr/share/dict/words", "/usr/share/dict/web2"] {
            let path = Path::new(candidate);
            if path.is_file() {
                return Ok(path.to_path_buf());
            }
        }
        bail!("no system word list found; install one or use a downloaded dictionary");
    }
    if name.is_empty()
        || name.len() > 64
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
    {
        bail!("invalid dictionary name");
    }
    let path = dict_dir().join(format!("{name}.txt"));
    if !path.is_file() {
        bail!(
            "dictionary '{name}' is not installed; run `nym terms dict pull {name}` or place {name}.txt in {}",
            dict_dir().display()
        );
    }
    Ok(path)
}

/// Download pinned catalog lists into the dictionary directory, verifying the
/// pinned SHA-256 before anything is written.
#[cfg(feature = "ureq")]
pub fn pull(names: &[String]) -> Result<Vec<(String, String)>> {
    use std::io::Write;
    use std::time::Duration;
    let dir = dict_dir();
    fs::create_dir_all(&dir)?;
    let mut installed = Vec::new();
    for name in names {
        let entry = CATALOG
            .iter()
            .find(|entry| entry.name == name)
            .ok_or_else(|| {
                anyhow!("dictionary '{name}' is not in the catalog; see `nym terms dict list`")
            })?;
        let url = format!("{CATALOG_SOURCE}/{0}/{0}_50k.txt", entry.name);
        let mut response = ureq::get(&url)
            .config()
            .timeout_global(Some(Duration::from_secs(120)))
            .max_redirects(3)
            .build()
            .call()
            .map_err(|_| anyhow!("dictionary download failed for '{name}'"))?;
        let bytes = response
            .body_mut()
            .with_config()
            .limit(entry.bytes + 1)
            .read_to_vec()
            .map_err(|_| anyhow!("dictionary download failed for '{name}'"))?;
        if bytes.len() as u64 != entry.bytes || crate::terms::sha256(&bytes) != entry.sha256 {
            bail!("dictionary '{name}' did not match its pinned hash; nothing written");
        }
        let mut file = tempfile::NamedTempFile::new_in(&dir)?;
        file.write_all(&bytes)?;
        file.as_file().sync_all()?;
        file.persist(dir.join(format!("{name}.txt")))
            .map_err(|_| anyhow!("dictionary could not be installed"))?;
        installed.push((name.clone(), entry.sha256.to_owned()));
    }
    Ok(installed)
}

#[cfg(not(feature = "ureq"))]
pub fn pull(_names: &[String]) -> Result<Vec<(String, String)>> {
    bail!(
        "this build has no network support; download the lists manually into {}",
        dict_dir().display()
    )
}

/// Installed lists with their hashes, plus catalog availability.
pub fn installed() -> Result<Vec<(String, String, u64)>> {
    let mut rows = Vec::new();
    let dir = dict_dir();
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("txt") || !path.is_file() {
                continue;
            }
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_owned();
            let bytes = fs::read(&path)?;
            rows.push((name, crate::terms::sha256(&bytes), bytes.len() as u64));
        }
    }
    rows.sort();
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(words: &[&str], case_aware: bool) -> KnownVocabulary {
        let mut vocabulary = KnownVocabulary::new(case_aware, true);
        vocabulary.words = words.iter().map(|w| w.to_string()).collect();
        vocabulary
    }

    #[test]
    fn case_aware_matching_keeps_capitalised_names_for_later_decision() {
        let v = vocab(&["grace", "read", "file", "the", "build"], true);
        assert_eq!(v.classify("grace"), Verdict::KnownLowercase);
        assert_eq!(v.classify("Grace"), Verdict::KnownCapitalized);
        assert_eq!(v.classify("Hammerschmidt"), Verdict::Keep);
        assert_eq!(v.classify("read_file"), Verdict::KnownLowercase);
        assert_eq!(v.classify("readFile"), Verdict::KnownCapitalized);
        assert_eq!(v.classify("the build"), Verdict::KnownLowercase);
        let insensitive = vocab(&["grace"], false);
        assert_eq!(insensitive.classify("Grace"), Verdict::KnownLowercase);
    }

    #[test]
    fn noise_shapes_cover_numbers_hashes_and_versions_only() {
        for noise in [
            "4096",
            "2026-10-20",
            "8f3a2b9c1d",
            "1.2.3",
            "v0.4.0",
            "12:30",
        ] {
            assert!(noise_shape(noise), "{noise}");
        }
        for keep in [
            "Hammerschmidt",
            "e.szymborska79",
            "jwierzbicki",
            "abc",
            "feed",
        ] {
            assert!(!noise_shape(keep), "{keep}");
        }
    }

    #[test]
    fn list_formats_reduce_to_first_token() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixed.txt");
        fs::write(&path, "3\n# comment\nWord 123\nhunspell/ABC\n\nplain\nx\n").unwrap();
        let (words, digest) = read_list(&path).unwrap();
        assert_eq!(digest.len(), 64);
        let mut sorted: Vec<_> = words.into_iter().collect();
        sorted.sort();
        assert_eq!(sorted, vec!["hunspell", "plain", "word"]);
    }

    #[test]
    fn catalog_names_are_valid_dictionary_names_and_hashes() {
        for entry in CATALOG {
            assert_eq!(entry.sha256.len(), 64);
            assert!(entry.name.chars().all(|c| c.is_ascii_lowercase()));
        }
    }
}
