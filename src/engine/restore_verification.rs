//! Value-free, fail-closed checks for recognizable aliases after restoration.
//!
//! Verification scans the input, not inserted originals. Callers supply byte
//! ranges of full, exact aliases actually restored in one non-cascading pass.
//! Components and case-modified aliases are evidence for review, not authority
//! to rewrite common words. Unknown paraphrases are not provably reversible.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use anyhow::{Result, anyhow};
use regex::Regex;
use serde::{Deserialize, Serialize};

use super::super::replacer::Replacement;

#[cfg(test)]
#[path = "restore_verification_tests.rs"]
mod tests;

/// Counts only: no aliases, originals, input excerpts, or offsets are exposed.
/// Counts are occurrences, not distinct mappings. Full aliases take precedence
/// over their components; `json_keys` and `ambiguous_components` are subsets.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidualCounts {
    pub full_aliases: usize,
    pub components: usize,
    pub ambiguous_components: usize,
    pub json_keys: usize,
}

impl ResidualCounts {
    pub fn total(self) -> usize {
        self.full_aliases + self.components
    }

    /// Strict policy: every recognizable residual, including ambiguous common
    /// components, fails. This is not a certificate of arbitrary reversibility.
    pub fn ensure_clear(self) -> Result<()> {
        if self.total() != 0 {
            return Err(anyhow!(
                "restore verification failed: {} recognizable residual aliases ({} full, {} components, {} ambiguous, {} in JSON keys)",
                self.total(),
                self.full_aliases,
                self.components,
                self.ambiguous_components,
                self.json_keys
            ));
        }
        Ok(())
    }

    pub fn merge(&mut self, other: Self) {
        self.full_aliases += other.full_aliases;
        self.components += other.components;
        self.ambiguous_components += other.ambiguous_components;
        self.json_keys += other.json_keys;
    }
}

struct Alias {
    pattern: Regex,
    component: bool,
    ambiguous: bool,
    bounded: bool,
}

/// Build once from the ORIGINAL key, not a filtered restoration dictionary.
/// Deliberately has no Debug implementation (the dictionary contains secrets).
pub struct RestoreVerifier<'a> {
    full_aliases: HashSet<&'a str>,
    aliases: Vec<Alias>,
    word_character: Regex,
}

impl<'a> RestoreVerifier<'a> {
    pub fn new(entries: &'a [Replacement]) -> Result<Self> {
        super::validate_mappings(entries)?;
        let mut full = HashMap::new();
        let mut components: HashMap<&str, HashSet<&str>> = HashMap::new();
        for entry in entries {
            if entry.replacement.is_empty() {
                return Err(anyhow!("key file contains an empty restoration alias"));
            }
            full.insert(entry.replacement.as_str(), entry.original.as_str());
            for component in &entry.components {
                if component.replacement.is_empty() {
                    continue;
                }
                components
                    .entry(&component.replacement)
                    .or_default()
                    .insert(&component.original);
            }
        }
        let mut aliases = Vec::new();
        for alias in full.keys() {
            aliases.push(Self::alias(alias, false, false)?);
        }
        // A full alias wins over a component with the same spelling. Case-fold
        // collisions are conservatively ambiguous even if legacy exact restore
        // could choose an original.
        let mut folded_components: HashMap<String, HashSet<&str>> = HashMap::new();
        for (alias, originals) in &components {
            folded_components
                .entry(alias.to_lowercase())
                .or_default()
                .extend(originals);
        }
        for alias in components.keys() {
            if full.contains_key(alias) {
                continue;
            }
            let ambiguous = folded_components[&alias.to_lowercase()].len() > 1;
            aliases.push(Self::alias(alias, true, ambiguous)?);
        }
        Ok(Self {
            full_aliases: full.into_keys().collect(),
            aliases,
            word_character: Regex::new(r"^[\p{L}\p{N}\p{M}]$")?,
        })
    }

    fn alias(alias: &str, component: bool, ambiguous: bool) -> Result<Alias> {
        let pattern = Regex::new(&format!("(?i:{})", regex::escape(alias)))
            .map_err(|_| anyhow!("failed to prepare restore verification"))?;
        Ok(Alias {
            pattern,
            component,
            ambiguous,
            bounded: component || alias.chars().count() <= 3,
        })
    }

    /// `restored_spans` must be sorted, non-overlapping byte ranges of FULL,
    /// EXACT aliases in this input, actually replaced by the caller. Short
    /// aliases (at most three Unicode characters) require word boundaries.
    /// Never pass
    /// component ranges or scan restored output and exempt arbitrary originals.
    /// A residual crossing a restored span remains visible (fail closed).
    pub fn verify_text(
        &self,
        input: &str,
        restored_spans: &[Range<usize>],
    ) -> Result<ResidualCounts> {
        self.validate_provenance(input, restored_spans)?;
        let mut hits = self.residual_hits(input, restored_spans);
        // Prefer the full alias, then the longest component at a given offset.
        // Stable semantic ordering keeps results independent of HashMap order.
        hits.sort_by(|a, b| {
            a.0.start
                .cmp(&b.0.start)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| b.0.end.cmp(&a.0.end))
                .then_with(|| b.2.cmp(&a.2))
        });
        let mut counts = ResidualCounts::default();
        let mut end = 0;
        for (span, component, ambiguous) in hits {
            if span.start < end {
                continue;
            }
            end = span.end;
            if component {
                counts.components += 1;
                counts.ambiguous_components += usize::from(ambiguous);
            } else {
                counts.full_aliases += 1;
            }
        }
        Ok(counts)
    }

    fn validate_provenance(&self, input: &str, restored_spans: &[Range<usize>]) -> Result<()> {
        let mut previous_end = 0;
        for span in restored_spans {
            if span.start < previous_end
                || span.start >= span.end
                || !input
                    .get(span.clone())
                    .is_some_and(|value| self.full_aliases.contains(value))
            {
                return Err(anyhow!("invalid restore verification provenance"));
            }
            if input[span.clone()].chars().count() <= 3 && !self.has_boundaries(input, span, false)
            {
                return Err(anyhow!("invalid restore verification provenance"));
            }
            previous_end = span.end;
        }
        Ok(())
    }

    fn residual_hits(
        &self,
        input: &str,
        restored_spans: &[Range<usize>],
    ) -> Vec<(Range<usize>, bool, bool)> {
        let mut hits = Vec::new();
        for alias in &self.aliases {
            for matched in alias.pattern.find_iter(input) {
                let span = matched.range();
                if alias.bounded && !self.has_boundaries(input, &span, alias.component) {
                    continue;
                }
                // Provenance is validated and sorted: avoid a quadratic scan
                // when an input contains many restored occurrences.
                let index = restored_spans.partition_point(|restored| restored.end <= span.start);
                if restored_spans.get(index).is_some_and(|restored| {
                    restored.start <= span.start && restored.end >= span.end
                }) {
                    continue;
                }
                hits.push((span, alias.component, alias.ambiguous));
            }
        }
        hits
    }

    fn has_boundaries(&self, input: &str, span: &Range<usize>, component: bool) -> bool {
        [
            input[..span.start].chars().next_back(),
            input[span.end..].chars().next(),
        ]
        .into_iter()
        .flatten()
        .all(|character| {
            let mut bytes = [0; 4];
            !self
                .word_character
                .is_match(character.encode_utf8(&mut bytes))
                && (component || character != '_')
        })
    }

    /// Scan all decoded object keys, including nested objects and arrays. Keys
    /// are unchanged by existing JSON restoration and have no trusted spans.
    /// Leaf verification must use ORIGINAL leaves via `verify_text`; calling
    /// this on a restored tree is safe only because its keys were not rewritten.
    pub fn verify_json_keys(&self, value: &serde_json::Value) -> Result<ResidualCounts> {
        let mut counts = ResidualCounts::default();
        match value {
            serde_json::Value::Array(values) => {
                for value in values {
                    counts.merge(self.verify_json_keys(value)?);
                }
            }
            serde_json::Value::Object(values) => {
                for (key, value) in values {
                    let mut keys = self.verify_text(key, &[])?;
                    keys.json_keys = keys.total();
                    counts.merge(keys);
                    counts.merge(self.verify_json_keys(value)?);
                }
            }
            _ => {}
        }
        Ok(counts)
    }
}
