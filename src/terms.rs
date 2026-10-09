//! Corpus-specific vocabulary suggestions. Counts are not privacy judgments.
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::terms_sketch::{Sketch, SourceBits};
use crate::terms_unicode::Projection;
use anyhow::{Result, bail};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const SCHEMA: &str = "nym.terms.discovery.v1";
const MAX_IDENTITIES: usize = 1_000_000;
const MAX_SOURCE_PAIRS: usize = 1_000_000;
const MAX_BOUNDED_UNITS: u64 = 10_000_000;

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub max_distinct: usize,
    pub min_count: u64,
    pub max_candidates: usize,
    pub max_examples: usize,
    pub max_phrase_words: usize,
    pub max_unit_bytes: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            max_distinct: 50_000,
            min_count: 2,
            max_candidates: 200,
            max_examples: 3,
            max_phrase_words: 3,
            max_unit_bytes: 16 * 1024 * 1024,
        }
    }
}

impl Settings {
    pub fn validate(&self) -> Result<()> {
        if self.max_distinct == 0
            || self.max_distinct > 1_000_000
            || self.min_count == 0
            || self.max_candidates == 0
            || self.max_candidates > self.max_distinct
            || self.max_candidates > 5_000
            || !(1..=5).contains(&self.max_examples)
            || !(1..=3).contains(&self.max_phrase_words)
            || self.max_unit_bytes == 0
            || self.max_unit_bytes > 256 * 1024 * 1024
        {
            bail!("invalid vocabulary discovery limits");
        }
        Ok(())
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Example {
    pub source: String,
    pub path: String,
    pub unit_sha256: String,
    pub start: usize,
    pub end: usize,
    pub context_start: usize,
    pub context: String,
    /// Example comes from a repaired record; offsets/hash refer to its decoded scalar view.
    #[serde(default, skip_serializing_if = "is_false")]
    pub projected: bool,
}

impl Example {
    fn validate(&self, projection: Option<&Projection>, term: &str) -> Result<()> {
        if self.projected && projection.is_none_or(|p| p.record_units == 0) {
            bail!("vocabulary projected example lacks provenance");
        }
        let a = self.start.checked_sub(self.context_start);
        let b = self.end.checked_sub(self.context_start);
        if !valid_hash(&self.unit_sha256)
            || self.source.is_empty()
            || self.source.len() > 128
            || self.path.len() > 2048
            || self.context.chars().count() > 384
            || a.zip(b).and_then(|(a, b)| self.context.get(a..b)) != Some(term)
        {
            bail!("invalid vocabulary example offsets or provenance");
        }
        Ok(())
    }
}

fn is_false(value: &bool) -> bool {
    !value
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub id: String,
    pub term: String,
    pub occurrences: u64,
    pub sources: u64,
    pub distinct_texts: u64,
    pub score: f64,
    /// Model recognition priority only; false is not a benign judgment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ner_match: Option<bool>,
    pub examples: Vec<Example>,
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    pub method: String,
    pub capacity: usize,
    pub recounted: usize,
    #[serde(default = "default_factor", skip_serializing_if = "is_default_factor")]
    pub pool_factor: usize,
}

fn default_factor() -> usize {
    10
}
fn is_default_factor(value: &usize) -> bool {
    *value == default_factor()
}

impl Selection {
    fn validate(&self, discovery: &Discovery) -> Result<()> {
        if self.method != "space_saving_occurrences"
            || self.capacity != discovery.settings.max_distinct
            || ![10, 100].contains(&self.pool_factor)
            || self.recounted > discovery.settings.max_candidates * self.pool_factor
            || (self.pool_factor == 100 && discovery.ranking != "source_specificity")
            || self.recounted > self.capacity
            || discovery.candidate_count != self.recounted as u64
            || discovery.background_sha256.is_some()
        {
            bail!("invalid bounded vocabulary selection");
        }
        Ok(())
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Discovery {
    pub schema: String,
    pub corpus_sha256: String,
    pub background_sha256: Option<String>,
    pub ranking: String,
    pub settings: Settings,
    pub source_count: u64,
    pub unit_count: u64,
    pub byte_count: u64,
    /// Exact mode: all distinct literals. Bounded mode: recounted pool only, not corpus vocabulary.
    pub candidate_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection: Option<Selection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection: Option<Projection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ner_ranking: Option<crate::terms_ner::NerRanking>,
    pub candidates: Vec<Candidate>,
}

impl Discovery {
    pub fn validate(&self) -> Result<()> {
        self.settings.validate()?;
        let max_units = self.validate_modes()?;
        if self.schema != SCHEMA
            || !valid_hash(&self.corpus_sha256)
            || self
                .background_sha256
                .as_deref()
                .is_some_and(|s| !valid_hash(s))
            || !matches!(
                self.ranking.as_str(),
                "spread" | "background_log_lift" | "source_specificity"
            )
            || (self.ranking == "background_log_lift") != self.background_sha256.is_some()
            || self.source_count > self.unit_count
            || self.unit_count > max_units
            || self.candidate_count > self.settings.max_distinct as u64
            || self.candidates.len() > self.settings.max_candidates
            || self.candidates.len() as u64 > self.candidate_count
        {
            bail!("invalid vocabulary discovery artifact");
        }
        crate::terms_ner::validate(self)?;
        let mut ids = HashSet::new();
        for candidate in &self.candidates {
            if !valid_term(&candidate.term)
                || candidate.id != sha256(candidate.term.as_bytes())
                || !ids.insert(&candidate.id)
                || !candidate.score.is_finite()
                || candidate.occurrences < self.settings.min_count
                || candidate.occurrences > self.byte_count
                || candidate.sources == 0
                || candidate.sources > self.source_count
                || candidate.sources > candidate.occurrences
                || candidate.distinct_texts == 0
                || candidate.distinct_texts > self.unit_count
                || candidate.distinct_texts > candidate.occurrences
                || candidate.examples.is_empty()
                || candidate.examples.len() > self.settings.max_examples
            {
                bail!("invalid vocabulary candidate");
            }
            for example in &candidate.examples {
                example.validate(self.projection.as_ref(), &candidate.term)?;
            }
        }
        Ok(())
    }

    fn validate_modes(&self) -> Result<u64> {
        if let Some(projection) = &self.projection {
            projection.validate(self.unit_count)?;
        }
        if let Some(selection) = &self.selection {
            selection.validate(self)?;
            Ok(MAX_BOUNDED_UNITS)
        } else {
            Ok(MAX_IDENTITIES as u64)
        }
    }

    pub fn fingerprint(&self) -> Result<String> {
        self.validate()?;
        let encoded = serde_json::to_vec(self)
            .map_err(|_| anyhow::anyhow!("vocabulary serialization failed"))?;
        Ok(sha256(&encoded))
    }
}

#[derive(Clone, Copy)]
struct Unit<'a> {
    source: &'a str,
    path: &'a str,
    text: &'a str,
    source_index: u32,
    unit_sha: &'a str,
    fresh: bool,
    projected: bool,
}

struct Counts {
    occurrences: u64,
    sources: SourceBits,
    distinct_texts: u64,
    examples: Vec<Example>,
}

type Ranked<'a> = (&'a String, &'a Counts, f64, Option<bool>);

fn prioritize(
    candidates: &mut [Ranked<'_>],
    ranker: &mut dyn crate::terms_ner::Prioritizer,
) -> Result<crate::terms_ner::NerRanking> {
    let terms: Vec<_> = candidates.iter().map(|c| c.0.as_str()).collect();
    let (flags, metadata) = ranker.score(&terms)?;
    metadata.validate(candidates.len() as u64)?;
    if flags.len() != candidates.len()
        || metadata.scored != candidates.len()
        || metadata.recognized != flags.iter().filter(|flag| **flag).count()
    {
        bail!("NER priority accounting failed; no artifact may be published");
    }
    for (candidate, flag) in candidates.iter_mut().zip(flags) {
        candidate.3 = Some(flag);
    }
    // Recognized values use duplicate-downweighted repetition, not source IDF.
    // Stable sorting preserves base-rank ties and the entire unmatched lane.
    candidates.sort_by(|a, b| {
        b.3.cmp(&a.3).then_with(|| {
            if a.3 == Some(true) {
                b.1.distinct_texts.cmp(&a.1.distinct_texts)
            } else {
                std::cmp::Ordering::Equal
            }
        })
    });
    Ok(metadata)
}

pub struct Builder {
    settings: Settings,
    words: Regex,
    structured: Vec<Regex>,
    counts: HashMap<String, Counts>,
    sources: BTreeMap<String, u32>,
    seen_texts: HashSet<[u8; 32]>,
    digest: Sha256,
    units: u64,
    bytes: u64,
    source_pairs: usize,
    background: Option<BTreeMap<String, u64>>,
    sketch: Option<Sketch>,
    selected: Option<HashSet<String>>,
    selection: Option<Selection>,
    expected_corpus: Option<String>,
    projection: Option<Projection>,
    record_projected: bool,
    specific: bool,
    failed: bool,
}

impl Builder {
    pub fn new(settings: Settings, background: Option<BTreeMap<String, u64>>) -> Result<Self> {
        settings.validate()?;
        if let Some(counts) = &background {
            if counts.len() > MAX_IDENTITIES
                || counts.is_empty()
                || counts.keys().any(|s| !valid_term(s))
                || counts
                    .values()
                    .try_fold(0u64, |a, b| a.checked_add(*b))
                    .is_none_or(|n| n == 0)
            {
                bail!("invalid background vocabulary counts");
            }
        }
        Ok(Self {
            settings,
            words: Regex::new(r"[\p{L}\p{N}_][\p{L}\p{N}\p{M}_-]*")?,
            structured: vec![
                Regex::new(
                    r#"[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>`]+|[A-Za-z0-9._%+-]+@[\p{L}\p{N}_.-]+\.[\p{L}]{2,}|(?:[A-Za-z]:[\\/]|~/|/)[^\s\"'<>`]+"#,
                )?,
                Regex::new(r"[\p{L}\p{N}_-]+(?:\.[\p{L}\p{N}_-]+)+")?,
            ],
            counts: HashMap::new(),
            sources: BTreeMap::new(),
            seen_texts: HashSet::new(),
            digest: Sha256::new(),
            units: 0,
            bytes: 0,
            source_pairs: 0,
            background,
            sketch: None,
            selected: None,
            selection: None,
            expected_corpus: None,
            projection: None,
            record_projected: false,
            specific: false,
            failed: false,
        })
    }

    /// Bounded proposals contain no examples, source-pair sets or exact duplicate accounting.
    pub fn bounded(settings: Settings) -> Result<Self> {
        let capacity = settings.max_distinct;
        let mut builder = Self::new(settings, None)?;
        builder.sketch = Some(Sketch::new(capacity));
        Ok(builder)
    }

    /// Consume approximate estimates; only the exact second pass may produce an artifact.
    pub fn recount(mut self) -> Result<Self> {
        if self.failed {
            bail!("vocabulary discovery is incomplete; start a new scan");
        }
        let sketch = self
            .sketch
            .take()
            .ok_or_else(|| anyhow::anyhow!("bounded vocabulary proposal is missing"))?;
        let pool_factor = if self.specific { 100 } else { 10 };
        let terms: HashSet<_> = sketch
            .shortlist(self.settings.max_candidates * pool_factor)
            .into_iter()
            .collect();
        let selection = Selection {
            method: "space_saving_occurrences".into(),
            capacity: self.settings.max_distinct,
            recounted: terms.len(),
            pool_factor,
        };
        let mut builder = Self::new(self.settings, None)?;
        if self.projection.is_some() {
            builder.enable_unicode_projection();
        }
        builder.specific = self.specific;
        builder.expected_corpus = Some(format!("{:x}", self.digest.finalize()));
        builder.selected = Some(terms);
        builder.selection = Some(selection);
        Ok(builder)
    }

    pub fn prefer_specificity(&mut self) -> Result<()> {
        if self.background.is_some() {
            bail!("source-specificity ranking cannot also use background ranking");
        }
        self.specific = true;
        Ok(())
    }

    fn specificity_score(&self, counts: &Counts) -> f64 {
        let frequency = (1.0 + counts.distinct_texts as f64).ln();
        if self.sources.len() <= 1 {
            return frequency;
        }
        frequency * ((self.sources.len() as f64 + 1.0) / (counts.sources.len() as f64 + 1.0)).ln()
    }

    pub fn enable_unicode_projection(&mut self) {
        self.projection = Some(Projection::default());
    }
    pub fn repairs_unicode(&self) -> bool {
        self.projection.is_some()
    }
    pub fn end_record(&mut self) {
        self.record_projected = false;
    }

    /// Bind original repaired-record bytes, not just their potentially equal scalar projections.
    pub fn bind_record(
        &mut self,
        source: &str,
        location: &str,
        raw: &str,
        replacements: u64,
    ) -> Result<()> {
        self.record_projected = replacements != 0;
        if replacements == 0 {
            return Ok(());
        }
        if self.failed {
            bail!("vocabulary discovery is incomplete; start a new scan");
        }
        self.failed = true;
        let projection = self
            .projection
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Unicode projection was not enabled"))?;
        projection.records += 1;
        projection.replacements = projection
            .replacements
            .checked_add(replacements)
            .ok_or_else(|| anyhow::anyhow!("vocabulary count overflow"))?;
        if projection.records > 10_000_000 {
            bail!("vocabulary projection record budget exceeded");
        }
        // Impossible source-length prefix domain-separates record markers from ordinary units.
        self.digest.update(u64::MAX.to_le_bytes());
        for value in [
            crate::terms_unicode::POLICY,
            source,
            location,
            &sha256(raw.as_bytes()),
        ] {
            self.digest.update((value.len() as u64).to_le_bytes());
            self.digest.update(value.as_bytes());
        }
        self.failed = false;
        Ok(())
    }

    fn max_units(&self) -> u64 {
        if self.sketch.is_some() || self.selected.is_some() {
            MAX_BOUNDED_UNITS
        } else {
            MAX_IDENTITIES as u64
        }
    }

    pub fn ingest(&mut self, source: &str, path: &str, text: &str) -> Result<()> {
        if self.failed {
            bail!("vocabulary discovery is incomplete; start a new scan");
        }
        self.failed = true;
        if text.len() > self.settings.max_unit_bytes
            || source.is_empty()
            || source.len() > 128
            || path.len() > 2048
            || self.units >= self.max_units()
        {
            bail!("vocabulary input unit or identity limit exceeded");
        }
        let source_index = if let Some(index) = self.sources.get(source) {
            *index
        } else {
            if self.sources.len() >= 10_000 {
                bail!("vocabulary source limit exceeded");
            }
            let index = u32::try_from(self.sources.len())
                .map_err(|_| anyhow::anyhow!("vocabulary source limit exceeded"))?;
            self.sources.insert(source.to_owned(), index);
            index
        };
        let text_hash: [u8; 32] = Sha256::digest(text.as_bytes()).into();
        let fresh = self.sketch.is_none() && self.seen_texts.insert(text_hash);
        let unit_sha = sha256(text.as_bytes());
        self.units += 1;
        if self.record_projected
            && let Some(projection) = &mut self.projection
        {
            projection.record_units += 1;
        }
        self.bytes = self
            .bytes
            .checked_add(text.len() as u64)
            .ok_or_else(|| anyhow::anyhow!("vocabulary count overflow"))?;
        for value in [source, path, text] {
            self.digest.update((value.len() as u64).to_le_bytes());
            self.digest.update(value.as_bytes());
        }
        // Limit the per-unit token buffer as well as corpus cardinality.
        let mut tokens = Vec::new();
        for word in self.words.find_iter(text) {
            if tokens.len() >= MAX_IDENTITIES {
                bail!("vocabulary token limit exceeded");
            }
            tokens.push((word.start(), word.end()));
        }
        let unit = Unit {
            source,
            path,
            text,
            source_index,
            unit_sha: &unit_sha,
            fresh,
            projected: self.record_projected,
        };
        let mut seen_in_unit = HashSet::new();
        // Whole hosts, URLs, emails and absolute paths complement components.
        let mut structured = BTreeSet::new();
        for pattern in &self.structured {
            for found in pattern.find_iter(text) {
                let term = found
                    .as_str()
                    .trim_end_matches(['.', ',', ';', ')', ']', '}']);
                structured.insert((found.start(), found.start() + term.len()));
                if structured.len() > MAX_IDENTITIES {
                    bail!("vocabulary structured-span limit exceeded");
                }
            }
        }
        for (start, end) in structured {
            self.count_span(unit, start, end, &mut seen_in_unit)?;
        }
        for (i, &(start, end)) in tokens.iter().enumerate() {
            self.count_span(unit, start, end, &mut seen_in_unit)?;
            for length in 2..=self.settings.max_phrase_words {
                let Some(group) = tokens.get(i..i + length) else {
                    break;
                };
                if group
                    .windows(2)
                    .any(|w| !text[w[0].1..w[1].0].chars().all(|c| c == ' '))
                {
                    break;
                }
                self.count_span(unit, start, group[length - 1].1, &mut seen_in_unit)?;
            }
        }
        self.failed = false;
        Ok(())
    }

    fn count_span(
        &mut self,
        unit: Unit<'_>,
        start: usize,
        end: usize,
        seen: &mut HashSet<String>,
    ) -> Result<()> {
        let Unit {
            source,
            path,
            text,
            source_index,
            unit_sha,
            fresh,
            projected,
        } = unit;
        let term = &text[start..end];
        if !valid_term(term) {
            return Ok(());
        }
        if let Some(sketch) = &mut self.sketch {
            return sketch.observe(term);
        }
        if self
            .selected
            .as_ref()
            .is_some_and(|terms| !terms.contains(term))
        {
            return Ok(());
        }
        if !self.counts.contains_key(term) && self.counts.len() >= self.settings.max_distinct {
            bail!("distinct vocabulary limit exceeded; increase --max-distinct explicitly");
        }
        let counts = self
            .counts
            .entry(term.to_owned())
            .or_insert_with(|| Counts {
                occurrences: 0,
                sources: SourceBits::default(),
                distinct_texts: 0,
                examples: Vec::new(),
            });
        counts.occurrences = counts
            .occurrences
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("vocabulary count overflow"))?;
        if counts.sources.insert(source_index) {
            self.source_pairs += 1;
            if self.selection.is_none() && self.source_pairs > MAX_SOURCE_PAIRS {
                bail!("vocabulary source-pair limit exceeded");
            }
        }
        if fresh && seen.insert(term.to_owned()) {
            counts.distinct_texts += 1;
        }
        if counts.examples.iter().any(|e| e.unit_sha256 == unit_sha) {
            return Ok(());
        }
        let replacement = if counts.examples.len() < self.settings.max_examples {
            None
        } else if !counts.examples.iter().any(|e| e.source == source) {
            counts
                .examples
                .iter()
                .enumerate()
                .rev()
                .find(|(_, e)| {
                    counts
                        .examples
                        .iter()
                        .filter(|other| other.source == e.source)
                        .count()
                        > 1
                })
                .map(|(i, _)| i)
        } else {
            return Ok(());
        };
        if counts.examples.len() < self.settings.max_examples || replacement.is_some() {
            let context_start = text[..start]
                .char_indices()
                .rev()
                .nth(89)
                .map_or(0, |(i, _)| i);
            let context_end = text[end..]
                .char_indices()
                .nth(90)
                .map_or(text.len(), |(i, _)| end + i);
            let example = Example {
                source: source.to_owned(),
                path: path.to_owned(),
                unit_sha256: unit_sha.to_owned(),
                start,
                end,
                context_start,
                context: text[context_start..context_end].to_owned(),
                projected,
            };
            if let Some(i) = replacement {
                counts.examples[i] = example;
            } else {
                counts.examples.push(example);
            }
        }
        Ok(())
    }

    pub fn finish(self) -> Result<Discovery> {
        self.finish_ranked(None)
    }

    pub fn finish_ranked(
        self,
        ranker: Option<&mut dyn crate::terms_ner::Prioritizer>,
    ) -> Result<Discovery> {
        if self.failed {
            bail!("vocabulary discovery is incomplete; no artifact may be published");
        }
        if self.sketch.is_some() {
            bail!("bounded vocabulary proposal requires exact recount");
        }
        let corpus_sha256 = format!("{:x}", self.digest.clone().finalize());
        if self
            .expected_corpus
            .as_ref()
            .is_some_and(|expected| expected != &corpus_sha256)
        {
            bail!("vocabulary inputs changed between proposal and recount; destination preserved");
        }
        let total: u64 = self.counts.values().map(|c| c.distinct_texts).sum();
        let background_total: u64 = self.background.as_ref().map_or(0, |b| b.values().sum());
        let background_sha256 = self
            .background
            .as_ref()
            .map(|b| serde_json::to_vec(b).map(|bytes| sha256(&bytes)))
            .transpose()
            .map_err(|_| anyhow::anyhow!("background serialization failed"))?;
        let vocabulary = (self.counts.len()
            + self.background.as_ref().map_or(0, |b| {
                b.keys()
                    .filter(|term| !self.counts.contains_key(*term))
                    .count()
            })) as f64;
        let mut candidates: Vec<_> = self
            .counts
            .iter()
            .filter(|(_, c)| c.occurrences >= self.settings.min_count)
            .map(|(term, c)| {
                // Laplace-smoothed log lift; ranking only, never an automatic privacy decision.
                let score = self.background.as_ref().map_or_else(
                    || {
                        if self.specific {
                            self.specificity_score(c)
                        } else {
                            c.distinct_texts as f64
                        }
                    },
                    |b| {
                        let foreground =
                            (c.distinct_texts as f64 + 1.0) / (total as f64 + vocabulary);
                        let reference = (*b.get(term).unwrap_or(&0) as f64 + 1.0)
                            / (background_total as f64 + vocabulary);
                        (foreground / reference).ln() * (1.0 + c.distinct_texts as f64).ln()
                    },
                );
                (term, c, score, None)
            })
            .collect();
        candidates.sort_by(|(a, ac, score_a, _), (b, bc, score_b, _)| {
            score_b
                .total_cmp(score_a)
                .then(bc.sources.len().cmp(&ac.sources.len()))
                .then(bc.occurrences.cmp(&ac.occurrences))
                .then(a.cmp(b))
        });
        // Score the complete eligible exact pool BEFORE the displayed shortlist cap.
        let ner_ranking = ranker
            .map(|ranker| prioritize(&mut candidates, ranker))
            .transpose()?;
        // Clone private contexts only for the displayed shortlist, not every term.
        let candidates = candidates
            .into_iter()
            .take(self.settings.max_candidates)
            .map(|(term, c, score, ner_match)| Candidate {
                id: sha256(term.as_bytes()),
                term: term.clone(),
                occurrences: c.occurrences,
                sources: c.sources.len() as u64,
                distinct_texts: c.distinct_texts,
                score,
                ner_match,
                examples: c.examples.clone(),
            })
            .collect();
        let discovery = Discovery {
            schema: SCHEMA.into(),
            corpus_sha256,
            ranking: if self.background.is_some() {
                "background_log_lift"
            } else if self.specific {
                "source_specificity"
            } else {
                "spread"
            }
            .into(),
            background_sha256,
            settings: self.settings,
            source_count: self.sources.len() as u64,
            unit_count: self.units,
            byte_count: self.bytes,
            candidate_count: self.counts.len() as u64,
            selection: self.selection,
            projection: self.projection,
            ner_ranking,
            candidates,
        };
        discovery.validate()?;
        Ok(discovery)
    }
}

fn valid_term(term: &str) -> bool {
    (2..=120).contains(&term.chars().count())
        && !term.chars().any(char::is_control)
        && term.trim() == term
}

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Choice {
    Sensitive,
    Contextual,
    Dismiss,
    Unsure,
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Decision {
    pub candidate_id: String,
    pub choice: Choice,
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Review {
    pub schema: String,
    pub discovery_sha256: String,
    pub corpus_sha256: String,
    pub decisions: Vec<Decision>,
}

impl Review {
    pub fn new(discovery: &Discovery) -> Result<Self> {
        let discovery_sha256 = discovery.fingerprint()?;
        Ok(Self {
            schema: "nym.terms.review.v1".into(),
            discovery_sha256,
            corpus_sha256: discovery.corpus_sha256.clone(),
            decisions: Vec::new(),
        })
    }
    pub fn validate(&self, discovery: &Discovery) -> Result<()> {
        if self.schema != "nym.terms.review.v1"
            || self.discovery_sha256 != discovery.fingerprint()?
            || self.corpus_sha256 != discovery.corpus_sha256
        {
            bail!("review does not match discovery artifact");
        }
        let ids: HashSet<_> = discovery.candidates.iter().map(|c| c.id.as_str()).collect();
        let mut seen = HashSet::new();
        if self
            .decisions
            .iter()
            .any(|d| !ids.contains(d.candidate_id.as_str()) || !seen.insert(&d.candidate_id))
        {
            bail!("review contains duplicate or unknown candidate decisions");
        }
        Ok(())
    }
    pub fn approve_terms(&self, discovery: &Discovery) -> Result<Vec<String>> {
        self.validate(discovery)?;
        let approved: HashSet<_> = self
            .decisions
            .iter()
            .filter(|d| d.choice == Choice::Sensitive)
            .map(|d| d.candidate_id.as_str())
            .collect();
        Ok(discovery
            .candidates
            .iter()
            .filter(|c| approved.contains(c.id.as_str()))
            .map(|c| c.term.clone())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn discover(rows: &[(&str, &str)], background: Option<BTreeMap<String, u64>>) -> Discovery {
        let mut builder = Builder::new(
            Settings {
                min_count: 1,
                ..Settings::default()
            },
            background,
        )
        .unwrap();
        for (source, text) in rows {
            builder.ingest(source, "text", text).unwrap();
        }
        builder.finish().unwrap()
    }
    #[test]
    fn unicode_original_offsets_and_duplicate_counts() {
        let input = "\u{1d11e} Jane Doë visits /srv/Veltrix then Jane Doë.";
        let d = discover(&[("a", input), ("a", input), ("b", input)], None);
        let jane = d.candidates.iter().find(|c| c.term == "Jane Doë").unwrap();
        assert_eq!(
            (jane.occurrences, jane.sources, jane.distinct_texts),
            (6, 2, 1)
        );
        for c in &d.candidates {
            for e in &c.examples {
                assert_eq!(&input[e.start..e.end], c.term);
            }
        }
        assert_eq!(
            d.fingerprint().unwrap(),
            discover(&[("a", input), ("a", input), ("b", input)], None)
                .fingerprint()
                .unwrap()
        );
    }
    #[test]
    fn phrases_do_not_cross_newline_and_words_are_not_exempt() {
        let d = discover(&[("a", "May\nJane Doe")], None);
        assert!(d.candidates.iter().any(|c| c.term == "May"));
        assert!(!d.candidates.iter().any(|c| c.term.contains('\n')));
        assert!(d.candidates.iter().any(|c| c.term == "Jane Doe"));
    }
    #[test]
    fn background_is_ranking_not_exclusion() {
        let d = discover(
            &[("a", "function Veltrix"), ("b", "function Veltrix")],
            Some(BTreeMap::from([
                ("function".into(), 100_000),
                ("other".into(), 100_000),
            ])),
        );
        assert!(
            d.candidates
                .iter()
                .position(|c| c.term == "Veltrix")
                .unwrap()
                < d.candidates
                    .iter()
                    .position(|c| c.term == "function")
                    .unwrap()
        );
        assert!(d.candidates.iter().any(|c| c.term == "function"));
    }
    #[test]
    fn cardinality_fails_closed() {
        let mut b = Builder::new(
            Settings {
                max_distinct: 1,
                max_candidates: 1,
                ..Settings::default()
            },
            None,
        )
        .unwrap();
        assert!(b.ingest("a", "", "Jane Doe").is_err());
        assert!(b.finish().is_err());
    }
    #[test]
    fn tampering_stale_reviews_and_implicit_approvals_fail() {
        let d = discover(&[("a", "Jane Doe")], None);
        let mut review = Review::new(&d).unwrap();
        assert!(review.approve_terms(&d).unwrap().is_empty());
        review.decisions.push(Decision {
            candidate_id: d.candidates[0].id.clone(),
            choice: Choice::Contextual,
        });
        assert!(review.approve_terms(&d).unwrap().is_empty());
        review.decisions[0].choice = Choice::Sensitive;
        assert_eq!(
            review.approve_terms(&d).unwrap(),
            vec![d.candidates[0].term.clone()]
        );
        let mut changed = d.clone();
        changed.candidates[0].examples[0].start += 1;
        assert!(changed.validate().is_err());
        let other = discover(&[("a", "Jane Doe!")], None);
        assert!(review.validate(&other).is_err());
        review.decisions.push(review.decisions[0].clone());
        assert!(review.validate(&d).is_err());
    }
}
