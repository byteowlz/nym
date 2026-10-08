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
fn specificity_widens_proposals_and_keeps_exact_candidate_counts() {
    let settings = Settings {
        max_candidates: 1,
        max_distinct: 100,
        max_phrase_words: 1,
        ..Settings::default()
    };
    let mut specific = Builder::bounded(settings.clone()).unwrap();
    specific.prefer_specificity().unwrap();
    let mut spread = Builder::bounded(settings.clone()).unwrap();
    let mut exact = Builder::new(settings, None).unwrap();
    exact.prefer_specificity().unwrap();
    let mut rows = Vec::new();
    let bulk = (0..20)
        .map(|i| format!("bulkword{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    for source in 0..12 {
        for unit in 0..3 {
            rows.push((
                format!("source{source}"),
                format!("{bulk} unit{source}-{unit}"),
            ));
        }
    }
    for i in 0..4 {
        rows.push(("source0".into(), format!("oriondesk context{i}")));
    }
    for (source, text) in &rows {
        specific.ingest(source, "text", text).unwrap();
        spread.ingest(source, "text", text).unwrap();
        exact.ingest(source, "text", text).unwrap();
    }
    let mut recount = specific.recount().unwrap();
    let mut old = spread.recount().unwrap();
    for (source, text) in &rows {
        recount.ingest(source, "text", text).unwrap();
        old.ingest(source, "text", text).unwrap();
    }
    let d = recount.finish().unwrap();
    let e = exact.finish().unwrap();
    let old = old.finish().unwrap();
    assert_eq!(
        serde_json::to_value(&d.candidates).unwrap(),
        serde_json::to_value(&e.candidates).unwrap()
    );
    assert_eq!(
        (
            d.ranking.as_str(),
            d.selection.as_ref().unwrap().pool_factor,
            d.candidates[0].term.as_str(),
            d.candidates[0].occurrences,
            d.candidates[0].sources,
            d.candidates[0].distinct_texts
        ),
        ("source_specificity", 100, "oriondesk", 4, 1, 4)
    );
    assert_ne!(old.candidates[0].term, "oriondesk");
    let mut empty = Builder::new(Settings::default(), None).unwrap();
    empty.prefer_specificity().unwrap();
    empty.finish().unwrap().validate().unwrap();
    let mut single = Builder::new(Settings::default(), None).unwrap();
    single.prefer_specificity().unwrap();
    single.ingest("source", "text", "Jane Jane").unwrap();
    assert!(single.finish().unwrap().candidates[0].score > 0.0);
}

#[test]
fn original_repaired_record_bytes_are_verified_even_when_projected_text_is_equal() {
    let mut proposal = Builder::bounded(Settings::default()).unwrap();
    proposal.enable_unicode_projection();
    proposal
        .bind_record("source", "record[1]", r#"{"text":"\ud800 Jane"}"#, 1)
        .unwrap();
    proposal
        .ingest("source", "record[1].text", "� Jane")
        .unwrap();
    let mut recount = proposal.recount().unwrap();
    assert!(recount.repairs_unicode());
    recount
        .bind_record("source", "record[1]", r#"{"text":"\ud801 Jane"}"#, 1)
        .unwrap();
    recount
        .ingest("source", "record[1].text", "� Jane")
        .unwrap();
    assert!(recount.finish().is_err());
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
