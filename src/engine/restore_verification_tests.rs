//! Offline engine contract for fail-closed restoration; CLI staging is owned
//! by the integration caller. Every fixture is synthetic.

use std::ops::Range;

use super::super::super::replacer::{ComponentMapping, Replacement};
use super::{ResidualCounts, RestoreVerifier};
use serde_json::{Value, json};

fn mapping(original: &str, alias: &str, components: &[(&str, &str)]) -> Replacement {
    Replacement {
        original: original.into(),
        replacement: alias.into(),
        pattern_name: "person".into(),
        components: components
            .iter()
            .map(|(original, alias)| ComponentMapping {
                original: (*original).into(),
                replacement: (*alias).into(),
                component_type: "name".into(),
            })
            .collect(),
    }
}

fn person() -> Vec<Replacement> {
    vec![mapping(
        "Original Person",
        "Donald Duck",
        &[("Original", "Donald"), ("Person", "Duck")],
    )]
}

/// Mirrors the existing non-cascading TextRestorer, but returns full exact
/// input spans. Strict mode does not guess how to replace component words.
fn restore_once(text: &str, entries: &[Replacement]) -> (String, Vec<Range<usize>>) {
    let mut sorted: Vec<_> = entries.iter().collect();
    sorted.sort_by_key(|entry| std::cmp::Reverse(entry.replacement.len()));
    let pattern = regex::Regex::new(
        &sorted
            .iter()
            .map(|entry| regex::escape(&entry.replacement))
            .collect::<Vec<_>>()
            .join("|"),
    )
    .unwrap();
    let spans = pattern
        .find_iter(text)
        .map(|matched| matched.range())
        .collect();
    let restored = pattern
        .replace_all(text, |captures: &regex::Captures<'_>| {
            sorted
                .iter()
                .find(|entry| entry.replacement == captures[0])
                .unwrap()
                .original
                .clone()
        })
        .into_owned();
    (restored, spans)
}

fn restore_value(
    value: &mut Value,
    entries: &[Replacement],
    verifier: &RestoreVerifier<'_>,
) -> ResidualCounts {
    let mut counts = ResidualCounts::default();
    match value {
        Value::String(text) => {
            let (restored, spans) = restore_once(text, entries);
            counts.merge(verifier.verify_text(text, &spans).unwrap());
            *text = restored;
        }
        Value::Array(values) => {
            for value in values {
                counts.merge(restore_value(value, entries, verifier));
            }
        }
        Value::Object(values) => {
            for value in values.values_mut() {
                counts.merge(restore_value(value, entries, verifier));
            }
        }
        _ => {}
    }
    counts
}

#[test]
fn complete_full_alias_is_counted_once_not_as_components() {
    let entries = person();
    let verifier = RestoreVerifier::new(&entries).unwrap();
    assert_eq!(
        verifier
            .verify_text("Donald Duck and DONALD DUCK", &[])
            .unwrap(),
        ResidualCounts {
            full_aliases: 2,
            ..ResidualCounts::default()
        }
    );
}

#[test]
fn exact_roundtrip_and_external_relabeling_preserve_full_structures() {
    let entries = person();
    let verifier = RestoreVerifier::new(&entries).unwrap();
    let mut relabelled = json!({"summary": ["The user Donald Duck asked", {"path": "/archive/Donald Duck/report.md"}], "ok": true, "n": 7, "null": null});
    let mut counts = restore_value(&mut relabelled, &entries, &verifier);
    counts.merge(verifier.verify_json_keys(&relabelled).unwrap());
    assert_eq!(counts, ResidualCounts::default());
    counts.ensure_clear().unwrap();
    assert_eq!(
        relabelled,
        json!({"summary": ["The user Original Person asked", {"path": "/archive/Original Person/report.md"}], "ok": true, "n": 7, "null": null})
    );
}

#[test]
fn case_modified_partial_and_path_variants_fail_closed() {
    let entries = person();
    let verifier = RestoreVerifier::new(&entries).unwrap();
    for (input, expected) in [
        (
            "The account DONALD DUCK requested a reset",
            ResidualCounts {
                full_aliases: 1,
                ..ResidualCounts::default()
            },
        ),
        (
            "Duck requested a reset; Donald approved",
            ResidualCounts {
                components: 2,
                ..ResidualCounts::default()
            },
        ),
        (
            "/users/dOnAlD/report /home/DUCK/key",
            ResidualCounts {
                components: 2,
                ..ResidualCounts::default()
            },
        ),
        (
            r"C:\Users\Donald_Duck\report",
            ResidualCounts {
                components: 2,
                ..ResidualCounts::default()
            },
        ),
        (
            "prefixDONALD DUCKsuffix",
            ResidualCounts {
                full_aliases: 1,
                ..ResidualCounts::default()
            },
        ),
    ] {
        let (_, spans) = restore_once(input, &entries);
        let actual = verifier.verify_text(input, &spans).unwrap();
        assert_eq!(actual, expected);
        assert!(actual.ensure_clear().is_err());
    }
}

#[test]
fn decoded_strings_and_unchanged_json_keys_are_both_verified() {
    let entries = person();
    let verifier = RestoreVerifier::new(&entries).unwrap();
    let mut value: Value = serde_json::from_str(r#"{"D\u006fnald Duck":[{"safe":"D\u004fNALD DUCK"},{"Donald":"Donald Duck"}],"nested":{"Duck":false}}"#).unwrap();
    let mut counts = restore_value(&mut value, &entries, &verifier);
    counts.merge(verifier.verify_json_keys(&value).unwrap());
    assert_eq!(
        counts,
        ResidualCounts {
            full_aliases: 2,
            components: 2,
            ambiguous_components: 0,
            json_keys: 3
        }
    );
    assert!(counts.ensure_clear().is_err());
    assert_eq!(
        value,
        json!({"Donald Duck": [{"safe":"DONALD DUCK"}, {"Donald":"Original Person"}], "nested":{"Duck":false}})
    );
}

#[test]
fn exact_alias_in_json_key_alone_is_strict_failure() {
    let entries = person();
    let verifier = RestoreVerifier::new(&entries).unwrap();
    let value = json!({"array": [{"Donald Duck": 123}]});
    let counts = verifier.verify_json_keys(&value).unwrap();
    assert_eq!(
        counts,
        ResidualCounts {
            full_aliases: 1,
            json_keys: 1,
            ..ResidualCounts::default()
        }
    );
    assert!(counts.ensure_clear().is_err());
}

#[test]
fn jsonl_aggregate_cannot_hide_late_failure() {
    let entries = person();
    let verifier = RestoreVerifier::new(&entries).unwrap();
    let input = "{\"body\":\"Donald Duck\"}\n{\"body\":\"Donald\"}\n";
    let mut counts = ResidualCounts::default();
    for line in input.lines() {
        let mut value: Value = serde_json::from_str(line).unwrap();
        counts.merge(restore_value(&mut value, &entries, &verifier));
        counts.merge(verifier.verify_json_keys(&value).unwrap());
    }
    assert_eq!(
        counts,
        ResidualCounts {
            components: 1,
            ..ResidualCounts::default()
        }
    );
    assert!(counts.ensure_clear().is_err());
}

#[test]
fn original_alias_substrings_are_exempt_only_by_restoration_provenance() {
    let entries = vec![mapping(
        "Original DONALD DUCK /Donald/info",
        "Donald Duck",
        &[("Original", "Donald"), ("Person", "Duck")],
    )];
    let verifier = RestoreVerifier::new(&entries).unwrap();
    let input = "Donald Duck; external DONALD DUCK; /Donald/info";
    let (restored, spans) = restore_once(input, &entries);
    assert_eq!(
        restored,
        "Original DONALD DUCK /Donald/info; external DONALD DUCK; /Donald/info"
    );
    assert_eq!(
        verifier.verify_text(input, &spans).unwrap(),
        ResidualCounts {
            full_aliases: 1,
            components: 1,
            ..ResidualCounts::default()
        }
    );
    assert_eq!(
        verifier.verify_text("Donald Duck", &[0..11]).unwrap(),
        ResidualCounts::default()
    );
}

#[test]
fn no_cascade_when_original_is_another_alias() {
    let entries = vec![
        mapping("alias-b", "alias-a", &[]),
        mapping("actual-b", "alias-b", &[]),
    ];
    let verifier = RestoreVerifier::new(&entries).unwrap();
    let input = "alias-a alias-b";
    let (restored, spans) = restore_once(input, &entries);
    assert_eq!(restored, "alias-b actual-b");
    assert_eq!(
        verifier.verify_text(input, &spans).unwrap(),
        ResidualCounts::default()
    );
}

#[test]
fn identity_mapping_and_alias_equal_original_substring_do_not_false_fail() {
    let entries = vec![
        mapping("Al", "Al", &[]),
        mapping("Original alias-z suffix", "alias-z", &[]),
    ];
    let verifier = RestoreVerifier::new(&entries).unwrap();
    let input = "Al and alias-z";
    let (restored, spans) = restore_once(input, &entries);
    assert_eq!(restored, "Al and Original alias-z suffix");
    assert_eq!(
        verifier.verify_text(input, &spans).unwrap(),
        ResidualCounts::default()
    );
}

#[test]
fn ambiguous_common_components_are_reported_not_replaced_or_omitted() {
    let entries = vec![
        mapping("First Person", "May North", &[("First", "May")]),
        mapping("Second Person", "May West", &[("Second", "May")]),
    ];
    let verifier = RestoreVerifier::new(&entries).unwrap();
    let input = "May North spoke. This may be a month.";
    let (restored, spans) = restore_once(input, &entries);
    assert_eq!(restored, "First Person spoke. This may be a month.");
    let counts = verifier.verify_text(input, &spans).unwrap();
    assert_eq!(
        counts,
        ResidualCounts {
            components: 1,
            ambiguous_components: 1,
            ..ResidualCounts::default()
        }
    );
    assert!(counts.ensure_clear().is_err());
}

#[test]
fn case_folded_component_conflicts_remain_ambiguous() {
    let entries = vec![
        mapping("First", "Name One", &[("First", "May")]),
        mapping("Second", "Name Two", &[("Second", "MAY")]),
    ];
    let verifier = RestoreVerifier::new(&entries).unwrap();
    assert_eq!(
        verifier.verify_text("may", &[]).unwrap(),
        ResidualCounts {
            components: 1,
            ambiguous_components: 1,
            ..ResidualCounts::default()
        }
    );
}

#[test]
fn short_alias_unicode_and_word_boundaries_prevent_fragment_matches() {
    let entries = vec![mapping("Original", "Al", &[])];
    let verifier = RestoreVerifier::new(&entries).unwrap();
    assert_eq!(
        verifier
            .verify_text("PAL Algae Al_foo xAl Al9 ÉAl Alé", &[])
            .unwrap(),
        ResidualCounts::default()
    );
    assert_eq!(
        verifier.verify_text("Al /AL/ (al) Al-Al", &[]).unwrap(),
        ResidualCounts {
            full_aliases: 5,
            ..ResidualCounts::default()
        }
    );
    let entries = vec![mapping("Original", "Äl", &[])];
    let verifier = RestoreVerifier::new(&entries).unwrap();
    assert_eq!(
        verifier.verify_text("Äl äL /ÄL/ ÄrÄlÄ", &[]).unwrap(),
        ResidualCounts {
            full_aliases: 3,
            ..ResidualCounts::default()
        }
    );
}

#[test]
fn invalid_provenance_never_bypasses_verification() {
    let entries = person();
    let verifier = RestoreVerifier::new(&entries).unwrap();
    for spans in [vec![0..6], vec![0..100], vec![1..1], vec![0..11, 0..11]] {
        assert!(verifier.verify_text("Donald Duck", &spans).is_err());
    }
    assert!(verifier.verify_text("DONALD DUCK", &[0..11]).is_err());
    assert!(verifier.verify_text("é Donald Duck", &[1..12]).is_err());
    let entries = vec![mapping("Original", "Al", &[])];
    let verifier = RestoreVerifier::new(&entries).unwrap();
    assert!(verifier.verify_text("PAL", &[1..3]).is_err());
    let entries = vec![mapping("Original", "ΩΩ", &[])];
    let verifier = RestoreVerifier::new(&entries).unwrap();
    assert!(verifier.verify_text("XΩΩ", &[1..5]).is_err());
}

#[test]
fn crossing_alias_cannot_be_exempted_by_a_shorter_full_alias_span() {
    let entries = vec![
        mapping("First Original", "Donald", &[]),
        mapping("Second Original", "Donald Duck", &[]),
    ];
    let verifier = RestoreVerifier::new(&entries).unwrap();
    assert_eq!(
        verifier.verify_text("Donald Duck", &[0..6]).unwrap(),
        ResidualCounts {
            full_aliases: 1,
            ..ResidualCounts::default()
        }
    );
}

#[test]
fn component_substrings_inside_other_words_are_not_recognizable() {
    let entries = person();
    let verifier = RestoreVerifier::new(&entries).unwrap();
    assert_eq!(
        verifier
            .verify_text(
                "Donaldson Duckling McDonald éDonald Donaldé Donald\u{301}",
                &[]
            )
            .unwrap(),
        ResidualCounts::default()
    );
}

#[test]
fn unicode_provenance_and_literal_regex_characters_are_supported() {
    let entries = vec![
        mapping(
            "Original with alias+test@example.net",
            "alias+test@example.net",
            &[],
        ),
        mapping("Original Unicode", "Álias Ω", &[]),
    ];
    let verifier = RestoreVerifier::new(&entries).unwrap();
    let input = "Álias Ω /alias+test@example.net/info";
    let (restored, spans) = restore_once(input, &entries);
    assert_eq!(
        restored,
        "Original Unicode /Original with alias+test@example.net/info"
    );
    assert_eq!(
        verifier.verify_text(input, &spans).unwrap(),
        ResidualCounts::default()
    );
    assert_eq!(
        verifier
            .verify_text("/ALIAS+TEST@EXAMPLE.NET/ áLIAS ω", &[])
            .unwrap(),
        ResidualCounts {
            full_aliases: 2,
            ..ResidualCounts::default()
        }
    );
}

#[test]
fn arbitrary_unknown_paraphrase_is_not_claimed_reversible() {
    let entries = person();
    let verifier = RestoreVerifier::new(&entries).unwrap();
    // No key-derived evidence remains. A clean count means only recognizable
    // aliases are absent, not that this lossy paraphrase has been restored.
    let input = "The cartoon bird requested a reset";
    let (restored, spans) = restore_once(input, &entries);
    assert_eq!(restored, input);
    assert_eq!(
        verifier.verify_text(input, &spans).unwrap(),
        ResidualCounts::default()
    );
}

#[test]
fn errors_and_serialized_counts_are_value_free() {
    let entries = person();
    let verifier = RestoreVerifier::new(&entries).unwrap();
    let counts = verifier.verify_text("DONALD DUCK", &[]).unwrap();
    assert_eq!(
        serde_json::to_value(counts).unwrap(),
        json!({"full_aliases":1,"components":0,"ambiguous_components":0,"json_keys":0})
    );
    assert_eq!(
        counts.ensure_clear().unwrap_err().to_string(),
        "restore verification failed: 1 recognizable residual aliases (1 full, 0 components, 0 ambiguous, 0 in JSON keys)"
    );
}

#[test]
fn empty_and_conflicting_key_aliases_fail_without_values() {
    for entries in [
        vec![mapping("Secret", "", &[])],
        vec![
            mapping("Secret A", "Same", &[]),
            mapping("Secret B", "Same", &[]),
        ],
        vec![
            mapping("Secret", "Alias A", &[]),
            mapping("Secret", "Alias B", &[]),
        ],
    ] {
        let error = RestoreVerifier::new(&entries).err().unwrap().to_string();
        assert!(!error.contains("Secret"));
        assert!(!error.contains("Same"));
    }
    let verifier = RestoreVerifier::new(&[]).unwrap();
    assert_eq!(
        verifier.verify_text("anything", &[]).unwrap(),
        ResidualCounts::default()
    );
}
