//! PP-OCRv6 model store for nym-ocr: official PaddlePaddle ONNX exports from
//! Hugging Face, cached in the standard Hugging Face cache and verified
//! against pinned SHA-256 digests.
//!
//! Ported from ingestr-core::models (byteowlz/ingestr, ADR-0004) so nym-ocr
//! fetches its models the same way instead of oar-ocr's ModelScope mirror.
//!
//! Lookup order for each file:
//! 1. `$NYM_SHARED_HF_HOME/hub` — an optional read-only cache seeded once
//!    per host (e.g. by provisioning) so users never download.
//! 2. The user's Hugging Face cache (`$HF_HOME/hub`, default
//!    `~/.cache/huggingface/hub`).
//! 3. Download from Hugging Face into (2), unless `HF_HUB_OFFLINE=1`.
//!
//! Every file is hash-checked before use; a mismatch is an error, never a
//! silent fallback. The recognition dictionaries are not published as files
//! on Hugging Face (they live inside `inference.yml`), so they are embedded
//! here, byte-identical to the dictionaries oar-ocr pins.

use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Cache, Repo};
use log::info;
use sha2::{Digest, Sha256};

/// Env var naming a shared, read-only Hugging Face home seeded per host.
pub const SHARED_HF_HOME_ENV: &str = "NYM_SHARED_HF_HOME";

const HF_FILE: &str = "inference.onnx";

const DICT_FULL: &str = include_str!("../assets/ppocrv6_dict.txt");
const DICT_TINY: &str = include_str!("../assets/ppocrv6_tiny_dict.txt");

/// PP-OCRv6 model size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PpOcrTier {
    /// Fastest, smallest (~6 MB), reduced character set.
    Tiny,
    /// Default: good accuracy at CPU speed (~31 MB).
    Small,
    /// Most accurate, heaviest (~138 MB).
    Medium,
}

impl PpOcrTier {
    /// Parse a config value; anything unknown falls back to `small`.
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "tiny" => Self::Tiny,
            "medium" => Self::Medium,
            _ => Self::Small,
        }
    }

    /// Lowercase name (`tiny` / `small` / `medium`).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tiny => "tiny",
            Self::Small => "small",
            Self::Medium => "medium",
        }
    }

    /// `(repo, sha256)` for the detection and recognition models.
    const fn pins(self) -> [(&'static str, &'static str); 2] {
        match self {
            Self::Tiny => [
                (
                    "PaddlePaddle/PP-OCRv6_tiny_det_onnx",
                    "193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8",
                ),
                (
                    "PaddlePaddle/PP-OCRv6_tiny_rec_onnx",
                    "9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6",
                ),
            ],
            Self::Small => [
                (
                    "PaddlePaddle/PP-OCRv6_small_det_onnx",
                    "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e",
                ),
                (
                    "PaddlePaddle/PP-OCRv6_small_rec_onnx",
                    "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634",
                ),
            ],
            Self::Medium => [
                (
                    "PaddlePaddle/PP-OCRv6_medium_det_onnx",
                    "eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1",
                ),
                (
                    "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
                    "9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba",
                ),
            ],
        }
    }

    /// The recognition dictionary matching this tier's recognizer.
    pub const fn dict(self) -> &'static str {
        match self {
            Self::Tiny => DICT_TINY,
            Self::Small | Self::Medium => DICT_FULL,
        }
    }
}

/// Local, verified paths for one PP-OCRv6 tier.
#[derive(Debug, Clone)]
pub struct PpOcrModels {
    /// Text detection model.
    pub det: PathBuf,
    /// Text recognition model.
    pub rec: PathBuf,
    /// Recognition character dictionary (embedded).
    pub dict: &'static str,
}

/// Resolve (and if needed download) the models for `tier`.
pub fn ensure_ppocr(tier: PpOcrTier) -> Result<PpOcrModels> {
    let [det, rec] = tier.pins();
    Ok(PpOcrModels {
        det: ensure_file(det.0, det.1)?,
        rec: ensure_file(rec.0, rec.1)?,
        dict: tier.dict(),
    })
}

fn caches() -> Vec<Cache> {
    let mut out = Vec::new();
    if let Some(shared) = env::var_os(SHARED_HF_HOME_ENV).filter(|v| !v.is_empty()) {
        out.push(Cache::new(PathBuf::from(shared).join("hub")));
    }
    out.push(Cache::from_env());
    out
}

fn cached(cache: &Cache, repo: &str) -> Option<PathBuf> {
    cache.repo(Repo::model(repo.to_string())).get(HF_FILE)
}

fn offline() -> bool {
    env::var("HF_HUB_OFFLINE").is_ok_and(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
}

fn ensure_file(repo: &str, sha256: &str) -> Result<PathBuf> {
    for cache in caches() {
        if let Some(path) = cached(&cache, repo) {
            verify(&path, sha256)?;
            return Ok(path);
        }
    }
    if offline() {
        bail!(
            "PP-OCR model {repo} is not cached and HF_HUB_OFFLINE is set; \
             seed ${SHARED_HF_HOME_ENV} or the Hugging Face cache first"
        );
    }
    info!("downloading {repo}/{HF_FILE} from Hugging Face (first use)");
    // Anonymous on purpose: the repos are public, and a stale token in the
    // user's HF cache would otherwise turn every download into a 401.
    let api = ApiBuilder::from_env()
        .with_token(None)
        .with_progress(false)
        .build()
        .context("initializing Hugging Face client")?;
    let path = api
        .model(repo.to_string())
        .get(HF_FILE)
        .with_context(|| format!("downloading {repo}/{HF_FILE}"))?;
    verify(&path, sha256)?;
    Ok(path)
}

fn verify(path: &Path, expected: &str) -> Result<()> {
    let actual = sha256_file(path)?;
    if actual != expected {
        bail!(
            "checksum mismatch for {}: expected {expected}, got {actual}; \
             delete the file to re-download",
            path.display()
        );
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(s: &str) -> String {
        hex::encode(Sha256::digest(s.as_bytes()))
    }

    #[test]
    fn embedded_dicts_match_oar_pins() {
        assert_eq!(
            sha(DICT_FULL),
            "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d"
        );
        assert_eq!(
            sha(DICT_TINY),
            "c5cbe34ef40c29c4df07ed012bf96569cb69a2d2a01a07027e9f13cb832bd9cd"
        );
    }

    #[test]
    fn tier_parse_defaults_to_small() {
        assert_eq!(PpOcrTier::parse("TINY"), PpOcrTier::Tiny);
        assert_eq!(PpOcrTier::parse(" medium "), PpOcrTier::Medium);
        assert_eq!(PpOcrTier::parse("bogus"), PpOcrTier::Small);
    }

    #[test]
    fn verify_rejects_tampered_file() -> Result<()> {
        let path = env::temp_dir().join(format!("nym-ocr-verify-{}", std::process::id()));
        fs::write(&path, b"not a model")?;
        let err = verify(&path, &"0".repeat(64));
        let _ = fs::remove_file(&path);
        assert!(err.is_err());
        Ok(())
    }

    #[test]
    fn shared_cache_is_searched_first() -> Result<()> {
        let root = env::temp_dir().join(format!("nym-ocr-shared-{}", std::process::id()));
        let repo = "PaddlePaddle/PP-OCRv6_tiny_det_onnx";
        let dir = root.join("hub/models--PaddlePaddle--PP-OCRv6_tiny_det_onnx");
        fs::create_dir_all(dir.join("refs"))?;
        fs::create_dir_all(dir.join("snapshots/abc"))?;
        fs::write(dir.join("refs/main"), "abc")?;
        fs::write(dir.join("snapshots/abc").join(HF_FILE), b"x")?;
        let found = cached(&Cache::new(root.join("hub")), repo);
        let _ = fs::remove_dir_all(&root);
        assert!(found.is_some_and(|p| p.ends_with(HF_FILE)));
        Ok(())
    }
}