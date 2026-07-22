//! NER model catalog for `nym models {list,pull,use,refresh}`.
//!
//! The catalog is data, not compiled judgment: [`catalog.json`] is baked into
//! the binary with `include_str!` so `nym models list` works offline, and
//! `nym models refresh` fetches a newer copy into `~/.config/nym/models.json`,
//! which [`load`] then prefers — so a model published after this binary was
//! built shows up without a reinstall. The resolvers still accept any HF repo
//! slug directly, catalog or not.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// One catalog row: an HF repo slug plus display metadata.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CatalogModel {
    /// HF repo slug (optionally `org/name/subfolder`), also the model id used
    /// in `[ner] token_model` / `[ner] model`.
    pub slug: String,
    /// Human-readable name.
    pub name: String,
    /// Which backend consumes it: `"tokens"` or `"gliner"`.
    pub backend: String,
    /// e.g. "English" or "Multilingual (25+)".
    pub languages: String,
    /// Approx download size, in MB.
    pub size_mb: u32,
    /// One-line description.
    pub description: String,
    /// Part of the small curated "recommended" set.
    #[serde(default)]
    pub recommended: bool,
    /// nym's out-of-the-box default for this backend.
    #[serde(default)]
    pub default: bool,
}

impl CatalogModel {
    /// True when this model is the token-classification backend's model.
    pub fn is_tokens(&self) -> bool {
        self.backend == "tokens"
    }

    /// Whether the model's files are already in the local HF cache (no network).
    pub fn is_cached(&self, cache_dir: Option<&Path>) -> bool {
        let (repo, prefix) = split_slug(&self.slug);
        let cache = match cache_dir {
            Some(dir) => hf_hub::Cache::new(dir.to_path_buf()),
            None => hf_hub::Cache::from_env(),
        };
        // tokenizer.json is fetched by both backends; its presence means the
        // repo has been pulled at least once.
        cache
            .model(repo)
            .get(&format!("{prefix}tokenizer.json"))
            .is_some()
    }
}

/// Split `org/name[/sub/dir]` into the HF repo id and a trailing-slashed prefix.
pub fn split_slug(slug: &str) -> (String, String) {
    let parts: Vec<&str> = slug.split('/').collect();
    if parts.len() > 2 {
        (parts[..2].join("/"), format!("{}/", parts[2..].join("/")))
    } else {
        (slug.to_string(), String::new())
    }
}

#[derive(Debug, Deserialize)]
struct CatalogFile {
    models: Vec<CatalogModel>,
}

const BAKED_JSON: &str = include_str!("catalog.json");

/// Where a refreshed catalog is fetched from. Override with `NYM_CATALOG_URL`.
const DEFAULT_CATALOG_URL: &str =
    "https://raw.githubusercontent.com/byteowlz/nym/main/src/engine/catalog.json";

fn parse(json: &str) -> Result<Vec<CatalogModel>> {
    let file: CatalogFile = serde_json::from_str(json).context("parsing model catalog JSON")?;
    Ok(file.models)
}

/// The catalog compiled into this binary.
pub fn baked() -> Vec<CatalogModel> {
    parse(BAKED_JSON).expect("baked catalog.json is valid and matches the schema")
}

/// The user's refreshed catalog cache path (`~/.config/nym/models.json`).
pub fn cache_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("nym")
        .join("models.json")
}

/// The active catalog: the refreshed cache if present and valid, else baked.
pub fn load() -> Vec<CatalogModel> {
    let path = cache_path();
    if path.is_file()
        && let Ok(text) = std::fs::read_to_string(&path)
        && let Ok(models) = parse(&text)
        && !models.is_empty()
    {
        return models;
    }
    baked()
}

fn catalog_url() -> String {
    std::env::var("NYM_CATALOG_URL").unwrap_or_else(|_| DEFAULT_CATALOG_URL.to_string())
}

/// Fetch the latest catalog and write it to [`cache_path`].
///
/// Returns the source URL and the model count on success.
///
/// # Errors
///
/// Returns an error if the download, parse, or cache write fails.
pub fn refresh() -> Result<(String, usize)> {
    let url = catalog_url();
    let body = ureq::get(&url)
        .call()
        .with_context(|| format!("fetching catalog from {url}"))?
        .body_mut()
        .read_to_string()
        .context("reading catalog response body")?;
    // Validate before persisting so a bad response never poisons the cache.
    let models = parse(&body).context("downloaded catalog is not valid")?;
    let path = cache_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(&path, &body).with_context(|| format!("writing {}", path.display()))?;
    Ok((url, models.len()))
}
