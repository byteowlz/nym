//! Deterministic bounded Space-Saving proposals. Estimates never become review counts.
use anyhow::{Result, bail};
use std::collections::{BTreeSet, HashMap};

#[derive(Default)]
pub struct Sketch {
    capacity: usize,
    counts: HashMap<String, (u64, u64)>,
    order: BTreeSet<(u64, String)>,
}

impl Sketch {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            ..Self::default()
        }
    }

    pub fn observe(&mut self, term: &str) -> Result<()> {
        let (count, error) = if let Some((count, error)) = self.counts.remove(term) {
            self.order.remove(&(count, term.to_owned()));
            (count, error)
        } else if self.counts.len() == self.capacity {
            let Some((count, previous)) = self.order.pop_first() else {
                bail!("bounded vocabulary proposal state is invalid");
            };
            self.counts.remove(&previous);
            (count, count)
        } else {
            (0, 0)
        };
        let count = count
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("vocabulary count overflow"))?;
        self.counts.insert(term.to_owned(), (count, error));
        self.order.insert((count, term.to_owned()));
        Ok(())
    }

    pub fn shortlist(self, limit: usize) -> Vec<String> {
        let mut rows: Vec<_> = self.counts.into_iter().collect();
        // Lower bounds avoid ranking newly admitted rare tokens as confidently frequent.
        rows.sort_by(|(a, (ac, ae)), (b, (bc, be))| {
            (bc - be).cmp(&(ac - ae)).then(bc.cmp(ac)).then(a.cmp(b))
        });
        rows.into_iter().take(limit).map(|(term, _)| term).collect()
    }
}

/// At most 10,000 sources: fixed-capacity bitset avoids candidate × source tree nodes.
#[derive(Default)]
pub struct SourceBits {
    words: Vec<u64>,
    count: usize,
}
impl SourceBits {
    pub fn insert(&mut self, source: u32) -> bool {
        let index = source as usize / 64;
        self.words.resize(self.words.len().max(index + 1), 0);
        let mask = 1u64 << (source % 64);
        if self.words[index] & mask != 0 {
            return false;
        }
        self.words[index] |= mask;
        self.count += 1;
        true
    }
    pub fn len(&self) -> usize {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn late_frequent_literal_survives_churn_with_bounded_state() {
        let mut sketch = Sketch::new(8);
        for i in 0..1000 {
            sketch.observe(&format!("unique{i}")).unwrap();
        }
        for _ in 0..100 {
            sketch.observe("LateHot").unwrap();
        }
        assert_eq!(sketch.counts.len(), 8);
        assert_eq!(sketch.shortlist(1), vec!["LateHot"]);
    }
    #[test]
    fn bitmap_counts_source_identity_exactly_including_high_index() {
        let mut bits = SourceBits::default();
        assert!(bits.insert(9999));
        assert!(bits.insert(0));
        assert!(!bits.insert(9999));
        assert_eq!(bits.len(), 2);
        assert_eq!(bits.words.len(), 157);
    }
}
