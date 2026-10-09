use crate::{
    terms::{Builder, Settings},
    terms_ner::{NerRanking, Prioritizer},
};
use anyhow::{Result, bail};
use std::collections::BTreeMap;

struct Fake {
    fail: bool,
    wrong_count: bool,
    seen: Vec<String>,
}
impl Prioritizer for Fake {
    fn score(&mut self, terms: &[&str]) -> Result<(Vec<bool>, NerRanking)> {
        self.seen = terms.iter().map(|term| (*term).to_owned()).collect();
        if self.fail {
            bail!("synthetic scoring failure");
        }
        let flags: Vec<_> = terms.iter().map(|term| term.ends_with("handle")).collect();
        let meta = NerRanking {
            method: "local_token_ner_positive_v1".into(),
            threshold: 0.5,
            assets_sha256: ["config", "model", "tokenizer"]
                .into_iter()
                .map(|key| (key.into(), "0".repeat(64)))
                .collect::<BTreeMap<_, _>>(),
            scored: terms.len() + usize::from(self.wrong_count),
            recognized: flags.iter().filter(|f| **f).count(),
        };
        Ok((flags, meta))
    }
}

fn corpus(bounded: bool) -> Builder {
    let settings = Settings {
        min_count: 1,
        max_candidates: 2,
        max_distinct: 100,
        max_phrase_words: 1,
        ..Settings::default()
    };
    let mut builder = if bounded {
        Builder::bounded(settings).unwrap()
    } else {
        Builder::new(settings, None).unwrap()
    };
    for text in ["routine engine", "routine ports", "isolatedhandle"] {
        builder.ingest("source", "text", text).unwrap();
    }
    if bounded {
        builder = builder.recount().unwrap();
        for text in ["routine engine", "routine ports", "isolatedhandle"] {
            builder.ingest("source", "text", text).unwrap();
        }
    }
    builder
}

#[test]
fn positive_priority_precedes_cutoff_and_keeps_unrecognized_terms_in_both_modes() {
    for bounded in [false, true] {
        let old = corpus(bounded).finish().unwrap();
        assert!(
            old.candidates
                .iter()
                .all(|c| c.term != "isolatedhandle" && c.ner_match.is_none())
        );
        let mut fake = Fake {
            fail: false,
            wrong_count: false,
            seen: Vec::new(),
        };
        let result = corpus(bounded).finish_ranked(Some(&mut fake)).unwrap();
        assert_eq!(fake.seen.len(), 4);
        assert_eq!(result.ner_ranking.as_ref().unwrap().scored, 4);
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|c| (c.term.as_str(), c.ner_match))
                .collect::<Vec<_>>(),
            vec![("isolatedhandle", Some(true)), ("routine", Some(false))]
        );
        assert_eq!(result.candidates[0].occurrences, 1);
        assert_eq!(result.candidates[0].examples[0].context, "isolatedhandle");
        result.validate().unwrap();
        let mut corrupt = result.clone();
        corrupt.candidates.reverse();
        assert!(corrupt.validate().is_err());
        let mut corrupt = result.clone();
        corrupt.candidates[1].ner_match = None;
        assert!(corrupt.validate().is_err());
        let mut corrupt = result.clone();
        corrupt.ner_ranking.as_mut().unwrap().recognized = 0;
        assert!(corrupt.validate().is_err());
    }
}

#[test]
fn widely_shared_recognized_identifier_is_not_demoted_by_source_specificity() {
    let settings = Settings {
        min_count: 1,
        max_candidates: 2,
        max_distinct: 100,
        max_phrase_words: 1,
        ..Settings::default()
    };
    let mut builder = Builder::new(settings, None).unwrap();
    builder.prefer_specificity().unwrap();
    for (source, text) in [
        ("one", "sharedhandle engine"),
        ("two", "sharedhandle ports"),
        ("three", "sharedhandle console"),
        ("one", "isolatedhandle lambda"),
        ("one", "isolatedhandle omega"),
    ] {
        builder.ingest(source, "text", text).unwrap();
    }
    let mut fake = Fake {
        fail: false,
        wrong_count: false,
        seen: Vec::new(),
    };
    let result = builder.finish_ranked(Some(&mut fake)).unwrap();
    assert_eq!(
        result
            .candidates
            .iter()
            .map(|c| (c.term.as_str(), c.ner_match, c.distinct_texts))
            .collect::<Vec<_>>(),
        vec![
            ("sharedhandle", Some(true), 3),
            ("isolatedhandle", Some(true), 2)
        ]
    );
    assert_eq!(result.candidates[0].score, 0.0);
    assert!(result.candidates[1].score > 0.0);
    let mut corrupt = result.clone();
    corrupt.candidates.reverse();
    assert!(corrupt.validate().is_err());
}

#[test]
fn inference_or_accounting_failure_never_finishes_and_default_has_no_model_metadata() {
    for (fail, wrong_count) in [(true, false), (false, true)] {
        let mut fake = Fake {
            fail,
            wrong_count,
            seen: Vec::new(),
        };
        assert!(corpus(false).finish_ranked(Some(&mut fake)).is_err());
    }
    let result = corpus(false).finish().unwrap();
    let raw = serde_json::to_string(&result).unwrap();
    assert!(!raw.contains("ner_match") && !raw.contains("ner_ranking"));
    assert_eq!(
        serde_json::from_str::<crate::terms::Discovery>(&raw)
            .unwrap()
            .fingerprint()
            .unwrap(),
        result.fingerprint().unwrap()
    );
}
