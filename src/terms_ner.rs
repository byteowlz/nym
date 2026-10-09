//! Optional, local-only positive NER priority. Never a privacy decision or exemption.
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NerRanking {
    pub method: String,
    pub assets_sha256: BTreeMap<String, String>,
    pub threshold: f32,
    pub scored: usize,
    pub recognized: usize,
}

impl NerRanking {
    pub fn validate(&self, candidate_count: u64) -> Result<()> {
        if self.method != "local_token_ner_positive_v1"
            || self.threshold != 0.5
            || self.scored as u64 > candidate_count
            || self.recognized > self.scored
            || self
                .assets_sha256
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>()
                != ["config", "model", "tokenizer"]
            || self.assets_sha256.values().any(|s| {
                s.len() != 64
                    || !s
                        .bytes()
                        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            })
        {
            bail!("invalid vocabulary NER ranking metadata");
        }
        Ok(())
    }
}

pub fn validate(discovery: &crate::terms::Discovery) -> Result<()> {
    if discovery
        .candidates
        .iter()
        .any(|c| c.ner_match.is_some() != discovery.ner_ranking.is_some())
    {
        bail!("vocabulary NER priority metadata is incomplete");
    }
    if let Some(ranking) = &discovery.ner_ranking {
        ranking.validate(discovery.candidate_count)?;
        let hits = discovery
            .candidates
            .iter()
            .filter(|c| c.ner_match == Some(true))
            .count();
        let wrong_order = discovery.candidates.windows(2).any(|w| {
            (w[0].ner_match == Some(false) && w[1].ner_match == Some(true))
                || (w[0].ner_match == Some(true)
                    && w[1].ner_match == Some(true)
                    && w[0].distinct_texts < w[1].distinct_texts)
        });
        if discovery.candidates.len() > ranking.scored
            || hits > ranking.recognized
            || discovery.candidates.len() - hits > ranking.scored - ranking.recognized
            || wrong_order
        {
            bail!("invalid vocabulary NER priority accounting");
        }
    }
    Ok(())
}

pub trait Prioritizer {
    fn score(&mut self, terms: &[&str]) -> Result<(Vec<bool>, NerRanking)>;
}

#[cfg(feature = "ner")]
mod local {
    use super::*;
    use crate::{
        config::{NerConfig, NerProvider},
        engine::ner_token::{DEFAULT_TOKEN_MODEL, MODEL_CANDIDATES, TokenClassDetector},
    };
    use sha2::{Digest, Sha256};
    use std::{
        fs::File,
        io::{IsTerminal, Read},
        path::PathBuf,
        time::Instant,
    };

    pub struct Local {
        detector: TokenClassDetector,
        assets: BTreeMap<String, PathBuf>,
        hashes: BTreeMap<String, String>,
        progress: bool,
    }

    fn hashes(assets: &BTreeMap<String, PathBuf>) -> Result<BTreeMap<String, String>> {
        assets
            .iter()
            .map(|(name, path)| {
                let mut file = File::open(path)
                    .map_err(|_| anyhow::anyhow!("local ranking model asset unavailable"))?;
                let mut digest = Sha256::new();
                let mut buf = [0u8; 65536];
                loop {
                    let n = file
                        .read(&mut buf)
                        .map_err(|_| anyhow::anyhow!("local ranking model asset unreadable"))?;
                    if n == 0 {
                        break;
                    }
                    digest.update(&buf[..n]);
                }
                Ok((name.clone(), format!("{:x}", digest.finalize())))
            })
            .collect()
    }

    fn directory(model: &str, config: &NerConfig) -> Result<PathBuf> {
        let expanded = crate::config::expand_path(model);
        if expanded.is_dir() {
            return Ok(expanded);
        }
        // Cache-only lookup: no HTTP client, download or source text can leave this process.
        let parts: Vec<_> = model.split('/').collect();
        if parts.len() < 2
            || parts
                .iter()
                .any(|p| p.is_empty() || *p == "." || *p == "..")
        {
            bail!("NER ranking requires a local model directory or already cached token model");
        }
        let cache = config
            .cache_dir
            .as_ref()
            .map_or_else(hf_hub::Cache::from_env, |p| {
                hf_hub::Cache::new(crate::config::expand_path(p))
            });
        let repo = cache.model(parts[..2].join("/"));
        let file = format!(
            "{}config.json",
            if parts.len() > 2 {
                format!("{}/", parts[2..].join("/"))
            } else {
                String::new()
            }
        );
        let path = repo.get(&file).ok_or_else(|| {
            anyhow::anyhow!(
                "NER ranking model is not cached; supply --ner-rank-model with a local directory"
            )
        })?;
        Ok(path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("invalid cached ranking model directory"))?
            .to_path_buf())
    }

    impl Local {
        pub fn load(model: Option<&str>, config: &NerConfig, progress: bool) -> Result<Self> {
            let dir = directory(
                model
                    .or(config.token_model.as_deref())
                    .unwrap_or(DEFAULT_TOKEN_MODEL),
                config,
            )?;
            let model = MODEL_CANDIDATES
                .iter()
                .map(|name| dir.join(name))
                .find(|p| p.is_file())
                .ok_or_else(|| anyhow::anyhow!("local NER ranking model unavailable"))?;
            let assets = BTreeMap::from([
                ("model".into(), model),
                ("config".into(), dir.join("config.json")),
                ("tokenizer".into(), dir.join("tokenizer.json")),
            ]);
            let before = hashes(&assets)?;
            if progress {
                eprintln!("NER priority: loading local CPU model (threshold 0.5)");
            }
            let detector = TokenClassDetector::from_dir(&dir, Some(0.5), NerProvider::Cpu)
                .map_err(|_| {
                    anyhow::anyhow!("local NER ranking model could not load; discovery incomplete")
                })?;
            if hashes(&assets)? != before {
                bail!("NER ranking model changed during loading; discovery incomplete");
            }
            Ok(Self {
                detector,
                assets,
                hashes: before,
                progress,
            })
        }
    }

    impl Prioritizer for Local {
        fn score(&mut self, terms: &[&str]) -> Result<(Vec<bool>, NerRanking)> {
            let started = Instant::now();
            if self.progress {
                eprintln!("NER priority: 0/{} candidates (scoring)", terms.len());
            }
            let mut flags = Vec::with_capacity(terms.len());
            let interactive =
                std::io::stderr().is_terminal() && std::env::var("TERM").is_ok_and(|v| v != "dumb");
            let mut drawn = Instant::now();
            for batch in terms.chunks(128) {
                let findings = self.detector.detect_batch(batch).map_err(|_| {
                    anyhow::anyhow!("local NER priority inference failed; discovery incomplete")
                })?;
                if findings.len() != batch.len() {
                    bail!("NER priority accounting failed; discovery incomplete");
                }
                for (text, matches) in batch.iter().zip(findings) {
                    if matches.iter().any(|m| {
                        text.get(m.start..m.end) != Some(m.matched_text.as_str())
                            || m.start >= m.end
                    }) {
                        bail!("NER priority offsets invalid; discovery incomplete");
                    }
                    flags.push(!matches.is_empty());
                }
                if self.progress
                    && interactive
                    && drawn.elapsed() >= std::time::Duration::from_millis(125)
                {
                    drawn = Instant::now();
                    eprintln!(
                        "NER priority: {}/{} candidates, {}s elapsed (scoring)",
                        flags.len(),
                        terms.len(),
                        started.elapsed().as_secs()
                    );
                }
            }
            if hashes(&self.assets)? != self.hashes {
                bail!("NER ranking model changed during scoring; discovery incomplete");
            }
            let recognized = flags.iter().filter(|flag| **flag).count();
            if self.progress {
                eprintln!(
                    "NER priority: {}/{} candidates, {} recognized (scored; not privacy approval)",
                    flags.len(),
                    terms.len(),
                    recognized
                );
            }
            Ok((
                flags,
                NerRanking {
                    method: "local_token_ner_positive_v1".into(),
                    assets_sha256: self.hashes.clone(),
                    threshold: 0.5,
                    scored: terms.len(),
                    recognized,
                },
            ))
        }
    }
}

pub fn load(
    model: Option<&str>,
    config: &crate::config::NerConfig,
    progress: bool,
) -> Result<Box<dyn Prioritizer>> {
    #[cfg(feature = "ner")]
    {
        Ok(Box::new(local::Local::load(model, config, progress)?))
    }
    #[cfg(not(feature = "ner"))]
    {
        let _ = (model, config, progress);
        bail!("NER priority requires a build with the ner feature; discovery incomplete");
    }
}
