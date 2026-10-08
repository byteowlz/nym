//! Bounded proposal/exact-recount contracts, independent of UI or private corpora.
use crate::terms::{Builder, Settings};

#[test]
fn recount_requires_identical_selected_units_and_cannot_publish_proposals() {
    let settings = Settings {
        max_distinct: 8,
        max_candidates: 2,
        max_phrase_words: 1,
        min_count: 1,
        ..Settings::default()
    };
    let mut b = Builder::bounded(settings.clone()).unwrap();
    b.ingest("a", "text", "Jane Doe").unwrap();
    assert!(b.finish().is_err());
    for (source, path, text) in [
        ("a", "text", "Jane Doe!"),
        ("b", "text", "Jane Doe"),
        ("a", "other", "Jane Doe"),
    ] {
        let mut proposal = Builder::bounded(settings.clone()).unwrap();
        proposal.ingest("a", "text", "Jane Doe").unwrap();
        let mut recount = proposal.recount().unwrap();
        recount.ingest(source, path, text).unwrap();
        assert!(recount.finish().is_err());
    }
}

#[test]
fn counts_match_exact_across_source_and_duplicate_distributions() {
    let rows = [
        ("a", "Jane Doë"),
        ("a", "Jane Doë"),
        ("b", "Jane Doë"),
        ("b", "Jane Doë visits"),
        ("c", "Jane Doë visits twice"),
    ];
    let settings = Settings {
        min_count: 1,
        max_phrase_words: 1,
        ..Settings::default()
    };
    let mut exact = Builder::new(settings.clone(), None).unwrap();
    let mut proposal = Builder::bounded(settings).unwrap();
    for (source, text) in rows {
        exact.ingest(source, "text", text).unwrap();
        proposal.ingest(source, "text", text).unwrap();
    }
    let mut recount = proposal.recount().unwrap();
    for (source, text) in rows {
        recount.ingest(source, "text", text).unwrap();
    }
    let exact = exact.finish().unwrap();
    let bounded = recount.finish().unwrap();
    assert_eq!(
        serde_json::to_value(&exact.candidates).unwrap(),
        serde_json::to_value(&bounded.candidates).unwrap()
    );
    assert_eq!(exact.corpus_sha256, bounded.corpus_sha256);
    let mut invalid = bounded;
    invalid.selection.as_mut().unwrap().recounted += 1;
    assert!(invalid.validate().is_err());
}
