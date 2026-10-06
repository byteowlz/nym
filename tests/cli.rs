//! End-to-end CLI tests for the release-data correctness work.
//!
//! These drive the compiled binary in subprocesses to verify:
//! - trx-18x8: key files are extended (never truncated) and aliases are
//!   reused across runs/processes for the same strategy/seed.
//! - trx-sqfz: `anon --stream` honors INPUT/output/format instead of silently
//!   ignoring them, and never destroys an existing key file.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_nym")
}

/// Run the binary with args + stdin, returning (exit_success, stdout).
fn run(args: &[&str], stdin: Option<&str>) -> (bool, String) {
    let out = run_output(args, stdin);
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string(),
    )
}

fn run_output(args: &[&str], stdin: Option<&str>) -> std::process::Output {
    // Release-data fixtures must not inherit host config or download NER models.
    let config_dir = tempfile::tempdir().unwrap();
    let config = config_dir.path().join("config.toml");
    write(&config, "[ner]\nenabled = false\n");
    let mut cmd = Command::new(bin());
    cmd.arg("--config").arg(&config).args(args);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    if stdin.is_some() {
        cmd.stdin(Stdio::piped());
    }
    let mut child = cmd.spawn().expect("spawn nym");
    if let Some(input) = stdin {
        child
            .stdin
            .take()
            .unwrap()
            .write_all(input.as_bytes())
            .expect("write stdin");
    }
    child.wait_with_output().expect("wait")
}

fn write(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, contents).unwrap();
}

fn deanon_with_mappings(mappings: &[serde_json::Value], input: &str) -> std::process::Output {
    let dir = tempfile::tempdir().unwrap();
    let key = dir.path().join("keys.jsonl");
    let mut lines = vec![r#"{"version":"1"}"#.to_string()];
    lines.extend(mappings.iter().map(serde_json::Value::to_string));
    write(&key, &lines.join("\n"));
    run_output(
        &["deanon", "--no-verify-restore", "-k", key.to_str().unwrap()],
        Some(input),
    )
}

#[test]
fn key_file_is_extended_not_truncated() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.json");
    let key = dir.path().join("keys.jsonl");
    write(&input, r#"{"email": "alice@example.com"}"#);

    // First invocation writes the key file.
    let (ok, _) = run(
        &[
            "anon",
            input.to_str().unwrap(),
            "--format",
            "json",
            "--strategy",
            "consistent",
            "--seed",
            "42",
            "-k",
            key.to_str().unwrap(),
        ],
        None,
    );
    assert!(ok, "first anon should succeed");
    let first = fs::read_to_string(&key).unwrap();
    assert!(
        first.contains("alice@example.com"),
        "first key file should record the mapping"
    );

    // Second, different input, same key file must extend, not overwrite.
    let input2 = dir.path().join("input2.json");
    write(&input2, r#"{"email": "bob@example.com"}"#);
    let (ok, _) = run(
        &[
            "anon",
            input2.to_str().unwrap(),
            "--format",
            "json",
            "--strategy",
            "consistent",
            "--seed",
            "42",
            "-k",
            key.to_str().unwrap(),
        ],
        None,
    );
    assert!(ok, "second anon should succeed");
    let second = fs::read_to_string(&key).unwrap();
    assert!(
        second.contains("alice@example.com"),
        "earlier mapping must survive a second run"
    );
    assert!(
        second.contains("bob@example.com"),
        "newer mapping must be recorded"
    );
}

#[test]
fn key_file_reuses_aliases_across_runs() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.json");
    let key = dir.path().join("keys.jsonl");
    write(&input, r#"{"email": "alice@example.com"}"#);

    let run_anon = |out: &Path| {
        run(
            &[
                "anon",
                input.to_str().unwrap(),
                "--format",
                "json",
                "--strategy",
                "consistent",
                "--seed",
                "42",
                "-o",
                out.to_str().unwrap(),
                "-k",
                key.to_str().unwrap(),
            ],
            None,
        )
    };

    let out1 = dir.path().join("out1.json");
    let out2 = dir.path().join("out2.json");
    let (ok1, _) = run_anon(&out1);
    let (ok2, _) = run_anon(&out2);
    assert!(ok1 && ok2);
    let o1 = fs::read_to_string(&out1).unwrap();
    let o2 = fs::read_to_string(&out2).unwrap();
    assert_eq!(
        o1, o2,
        "same input/seed/strategy must produce the same alias"
    );
}

#[cfg(feature = "streaming")]
#[test]
fn streaming_honors_output_file() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.jsonl");
    let output = dir.path().join("out.jsonl");
    // Two JSON records, one email each.
    write(
        &input,
        "{\"email\": \"alice@example.com\"}\n{\"email\": \"bob@example.com\"}\n",
    );

    let (ok, _) = run(
        &[
            "anon",
            input.to_str().unwrap(),
            "--format",
            "jsonl",
            "--stream",
            "--strategy",
            "placeholder",
            "-o",
            output.to_str().unwrap(),
        ],
        None,
    );
    assert!(ok, "streaming file->file should succeed");
    let out = fs::read_to_string(&output).unwrap();
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 2, "one output record per input record");
    assert!(
        lines.iter().all(|l| l.contains("<EMAIL>")),
        "emails must be redacted with a placeholder"
    );
}

#[cfg(feature = "streaming")]
#[test]
fn streaming_honors_stdin_to_file() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("out.jsonl");
    let (ok, _) = run(
        &[
            "anon",
            "--format",
            "json",
            "--stream",
            "-o",
            output.to_str().unwrap(),
        ],
        Some("{\"email\": \"carol@example.com\"}\n"),
    );
    assert!(ok, "streaming stdin->file should succeed");
    let out = fs::read_to_string(&output).unwrap();
    assert!(!out.contains("carol@example.com"));
    assert!(
        out.contains("<EMAIL>") || out.contains("@"),
        "should be anonymized"
    );
}

#[cfg(feature = "streaming")]
#[test]
fn streaming_does_not_truncate_existing_key_file() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.jsonl");
    let output = dir.path().join("out.jsonl");
    let key = dir.path().join("keys.jsonl");
    write(&input, "{\"email\": \"alice@example.com\"}\n");

    // First run creates key file.
    let (ok, _) = run(
        &[
            "anon",
            input.to_str().unwrap(),
            "--format",
            "json",
            "--stream",
            "-o",
            output.to_str().unwrap(),
            "--strategy",
            "consistent",
            "--seed",
            "42",
            "-k",
            key.to_str().unwrap(),
        ],
        None,
    );
    assert!(ok);
    let first = fs::read_to_string(&key).unwrap();
    assert!(first.contains("alice@example.com"));

    // Second run over a different record: key file must be extended.
    write(&input, "{\"email\": \"bob@example.com\"}\n");
    let (ok, _) = run(
        &[
            "anon",
            input.to_str().unwrap(),
            "--format",
            "json",
            "--stream",
            "-o",
            output.to_str().unwrap(),
            "--strategy",
            "consistent",
            "--seed",
            "42",
            "-k",
            key.to_str().unwrap(),
        ],
        None,
    );
    assert!(ok);
    let second = fs::read_to_string(&key).unwrap();
    assert!(
        second.contains("alice@example.com"),
        "first mapping must survive"
    );
    assert!(
        second.contains("bob@example.com"),
        "second mapping must be added"
    );
}

#[test]
fn json_include_path_skips_structural_ids() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.json");
    let output = dir.path().join("out.json");
    write(
        &input,
        r#"{"id": "11111111-2222-3333-4444-555555555555", "user": {"email": "alice@example.com", "id": "u-1"}}"#,
    );

    let (ok, stdout) = run(
        &[
            "anon",
            input.to_str().unwrap(),
            "--format",
            "json",
            "--strategy",
            "placeholder",
            "--include-path",
            "user.email",
            "--json-coverage",
            "-o",
            output.to_str().unwrap(),
        ],
        None,
    );
    assert!(ok, "anon with include-path should succeed");
    let out = fs::read_to_string(&output).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
    // structural id untouched
    assert_eq!(
        parsed["id"].as_str().unwrap(),
        "11111111-2222-3333-4444-555555555555"
    );
    assert_eq!(parsed["user"]["id"].as_str().unwrap(), "u-1");
    // email redacted
    assert!(
        parsed["user"]["email"]
            .as_str()
            .unwrap()
            .contains("<EMAIL>")
    );
    // Coverage must not contaminate the data channel, even with file output.
    assert!(stdout.is_empty());
}

#[test]
fn json_exclude_path_skips_structural_ids() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.json");
    let output = dir.path().join("out.json");
    write(
        &input,
        r#"{"id": "11111111-2222-3333-4444-555555555555", "user": {"email": "alice@example.com"}}"#,
    );

    let (ok, _) = run(
        &[
            "anon",
            input.to_str().unwrap(),
            "--format",
            "json",
            "--strategy",
            "placeholder",
            "--exclude-path",
            "id",
            "-o",
            output.to_str().unwrap(),
        ],
        None,
    );
    assert!(ok);
    let out = fs::read_to_string(&output).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(
        parsed["id"].as_str().unwrap(),
        "11111111-2222-3333-4444-555555555555"
    );
    assert!(
        parsed["user"]["email"]
            .as_str()
            .unwrap()
            .contains("<EMAIL>")
    );
}

#[test]
fn invalid_path_selector_fails_loudly() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.json");
    write(&input, r#"{"user": {"email": "a@b.com"}}"#);
    let (ok, _) = run(
        &[
            "anon",
            input.to_str().unwrap(),
            "--format",
            "json",
            "--include-path",
            "a[",
            "-o",
            dir.path().join("out.json").to_str().unwrap(),
        ],
        None,
    );
    assert!(!ok, "malformed path selector must be rejected");
}

#[test]
fn fail_on_blocks_on_matching_pattern() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.txt");
    write(&input, "contact alice@example.com for details");

    // Default behavior: findings present still exit 0.
    let (ok, _) = run(&["detect", input.to_str().unwrap()], None);
    assert!(ok, "default detect should succeed even with findings");

    // With --fail-on email, exit nonzero.
    let (ok, _) = run(
        &["detect", input.to_str().unwrap(), "--fail-on", "email"],
        None,
    );
    assert!(!ok, "fail-on email should block");
}

#[test]
fn fail_on_does_not_block_on_other_pattern() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.txt");
    write(&input, "contact alice@example.com for details");

    let (ok, _) = run(
        &["detect", input.to_str().unwrap(), "--fail-on", "ssn"],
        None,
    );
    assert!(ok, "fail-on ssn should not block an email finding");
}

#[test]
fn summary_json_is_value_free() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.txt");
    write(&input, "contact alice@example.com at 555-123-4567");

    let (ok, stdout) = run(&["detect", input.to_str().unwrap(), "--summary-json"], None);
    assert!(ok);
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect("valid json summary");
    assert!(v["total"].as_u64().unwrap() >= 1);
    assert!(
        !stdout.contains("alice@example.com"),
        "no matched values in summary"
    );
    assert!(
        !stdout.contains("555-123-4567"),
        "no matched values in summary"
    );
    assert!(stdout.contains("by_category"), "summary groups by category");
}

#[test]
fn audit_rejects_every_unknown_policy_before_emitting_success() {
    for (input, format) in [
        ("clean text", "text"),
        ("alpha@example.invalid", "text"),
        (r#"{"text":"clean"}"#, "json"),
        (r#"{"text":"alpha@example.invalid"}"#, "json"),
    ] {
        for policies in [
            vec!["not-a-real-category"],
            vec!["email", "not-a-real-category"],
            vec!["contact", "not-a-real-category"],
            vec![""],
        ] {
            let mut args = vec![
                "detect",
                "--no-ner",
                "--patterns",
                "email",
                "--summary-json",
                "--format",
                format,
            ];
            for policy in policies {
                args.extend(["--fail-on", policy]);
            }
            let out = run_output(&args, Some(input));
            assert_eq!(out.status.code(), Some(1), "{args:?}");
            assert!(
                out.stdout.is_empty(),
                "invalid policy must not emit a manifest"
            );
            assert!(String::from_utf8_lossy(&out.stderr).contains("unknown audit policy"));
        }
    }
}

#[test]
fn audit_valid_policies_preserve_exit_codes_and_safe_summaries() {
    for (input, policy, code, total) in [
        ("clean text", "email", 0, 0),
        ("clean text", "person", 0, 0),
        ("alpha@example.invalid", "email", 2, 1),
        ("alpha@example.invalid", "CONTACT", 2, 1),
        ("alpha@example.invalid", "ssn", 0, 1),
    ] {
        let out = run_output(
            &[
                "detect",
                "--no-ner",
                "--patterns",
                "email",
                "--fail-on",
                policy,
                "--summary-json",
            ],
            Some(input),
        );
        assert_eq!(out.status.code(), Some(code));
        let summary: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        let mut expected = serde_json::json!({"total": total, "by_pattern": {}, "by_category": {}});
        if total == 1 {
            expected["by_pattern"] = serde_json::json!({"email": 1});
            expected["by_category"] = serde_json::json!({"contact": 1});
        }
        if code == 2 {
            expected["blockers"] = serde_json::json!(["email"]);
        }
        assert_eq!(summary, expected);
        assert!(!String::from_utf8_lossy(&out.stdout).contains("alpha@example.invalid"));
    }
}

#[test]
fn audit_missing_input_remains_an_operational_error() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("missing.txt");
    let out = run_output(
        &[
            "detect",
            missing.to_str().unwrap(),
            "--fail-on",
            "email",
            "--summary-json",
        ],
        None,
    );
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
}

#[test]
fn json_coverage_keeps_unicode_selected_payloads_parseable_and_output_equivalent() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.json");
    let output = dir.path().join("output.json");
    let text = r#"{"私有":{"内容":[{"text":"雪\n\"alpha@example.invalid\"\\終","id":"keep@example.invalid"}]},"id":"保留"}"#;
    write(&input, text);
    let args = [
        "anon",
        "--no-ner",
        "--patterns",
        "email",
        "--format",
        "json",
        "--strategy",
        "placeholder",
        "--include-path",
        "私有.内容[*].text",
        "--json-coverage",
    ];
    let baseline = run_output(&args[..args.len() - 1], Some(text));
    assert!(baseline.status.success());
    let expected = serde_json::json!({"私有": {"内容": [{"text": "雪\n\"<EMAIL>\"\\終", "id": "keep@example.invalid"}]}, "id": "保留"});
    for machine in [false, true] {
        for file_input in [false, true] {
            for file_output in [false, true] {
                let mut args = args.to_vec();
                if machine {
                    args.push("--json");
                }
                if file_input {
                    args.push(input.to_str().unwrap());
                }
                if file_output {
                    args.extend(["-o", output.to_str().unwrap()]);
                }
                let out = run_output(&args, if file_input { None } else { Some(text) });
                assert!(
                    out.status.success(),
                    "{}",
                    String::from_utf8_lossy(&out.stderr)
                );
                let payload = if file_output {
                    assert!(out.stdout.is_empty());
                    fs::read(&output).unwrap()
                } else {
                    out.stdout
                };
                assert_eq!(
                    serde_json::from_slice::<serde_json::Value>(&payload).unwrap(),
                    expected
                );
                assert_eq!(
                    serde_json::from_slice::<serde_json::Value>(&baseline.stdout).unwrap(),
                    expected
                );
                let diagnostics = String::from_utf8_lossy(&out.stderr);
                assert!(diagnostics.contains("scanned: 1 path(s)"));
                assert!(diagnostics.contains("skipped: 2 path(s)"));
                assert!(diagnostics.contains("私有.内容[0].text"));
                assert!(!diagnostics.contains("alpha@example.invalid"));
            }
        }
    }
}

#[test]
fn detection_coverage_keeps_json_and_safe_manifests_separate() {
    let text = r#"{"私有":{"text":"雪 alpha@example.invalid"},"id":"保留"}"#;
    for summary in [false, true] {
        let mut args = vec![
            "detect",
            "--no-ner",
            "--patterns",
            "email",
            "--format",
            "json",
            "--json",
            "--json-coverage",
            "--include-path",
            "私有.text",
        ];
        if summary {
            args.extend(["--summary-json", "--fail-on", "contact"]);
        }
        let out = run_output(&args, Some(text));
        assert_eq!(out.status.code(), Some(if summary { 2 } else { 0 }));
        let payload: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        if summary {
            assert_eq!(
                payload,
                serde_json::json!({"total": 1, "by_pattern": {"email": 1}, "by_category": {"contact": 1}, "blockers": ["email"]})
            );
            let stdout = String::from_utf8_lossy(&out.stdout);
            assert!(!stdout.contains("私有"));
            assert!(!stdout.contains("alpha@example.invalid"));
        } else {
            assert_eq!(payload.as_array().unwrap().len(), 1);
        }
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("JSON scan coverage:"));
        assert!(stderr.contains("scanned: 1 path(s)"));
        assert!(stderr.contains("skipped: 1 path(s)"));
    }
}

#[cfg(feature = "streaming")]
#[test]
fn streaming_jsonl_coverage_reports_each_record_without_contaminating_output() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.jsonl");
    let output = dir.path().join("output.jsonl");
    let text = "{\"私有\":{\"text\":\"雪\\nalpha@example.invalid\"},\"id\":\"保留\"}\n\n{\"私有\":{\"text\":\"終 beta@example.invalid\"},\"id\":\"keep@example.invalid\"}\n";
    write(&input, text);
    let args = [
        "anon",
        "--stream",
        "--no-ner",
        "--patterns",
        "email",
        "--format",
        "jsonl",
        "--strategy",
        "placeholder",
        "--include-path",
        "私有.text",
        "--json-coverage",
        "--json",
    ];
    let baseline = run_output(&args[..args.len() - 2], Some(text));
    assert!(baseline.status.success());
    let expected = vec![
        serde_json::json!({"私有":{"text":"雪\n<EMAIL>"},"id":"保留"}),
        serde_json::json!({"私有":{"text":"終 <EMAIL>"},"id":"keep@example.invalid"}),
    ];
    for file_input in [false, true] {
        for file_output in [false, true] {
            let mut args = args.to_vec();
            if file_input {
                args.push(input.to_str().unwrap());
            }
            if file_output {
                args.extend(["-o", output.to_str().unwrap()]);
            }
            let out = run_output(&args, if file_input { None } else { Some(text) });
            assert!(out.status.success());
            let payload = if file_output {
                assert!(out.stdout.is_empty());
                fs::read(&output).unwrap()
            } else {
                out.stdout
            };
            assert_eq!(payload, baseline.stdout);
            let parsed: Vec<serde_json::Value> = String::from_utf8(payload)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            assert_eq!(parsed, expected);
            let diagnostics = String::from_utf8_lossy(&out.stderr);
            assert_eq!(diagnostics.matches("JSON scan coverage:").count(), 2);
            assert_eq!(diagnostics.matches("scanned: 1 path(s)").count(), 2);
            assert_eq!(diagnostics.matches("skipped: 1 path(s)").count(), 2);
        }
    }
}

#[cfg(feature = "streaming")]
#[test]
fn detection_streaming_rejects_unsupported_audit_and_coverage_options() {
    for options in [
        vec!["--json-coverage"],
        vec!["--fail-on", "email"],
        vec!["--fail-on", "not-a-real-category"],
        vec!["--summary-json"],
    ] {
        let mut args = vec!["detect", "--stream", "--no-ner", "--format", "json"];
        args.extend(options);
        let out = run_output(&args, Some("{\"text\":\"alpha@example.invalid\"}\n"));
        assert_eq!(out.status.code(), Some(1));
        assert!(out.stdout.is_empty());
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("not supported with detect --stream")
        );
    }
}

#[test]
fn deanon_never_rescans_originals_inserted_by_full_or_component_aliases() {
    let dir = tempfile::tempdir().unwrap();
    let key = dir.path().join("keys.jsonl");
    let rows = [
        serde_json::json!({"version": "1", "strategy": "fake", "seed": 42}),
        serde_json::json!({
            "original": "first@example.invalid", "replacement": "shared@example.com",
            "pattern_name": "email", "components": [{"original": "first", "replacement": "shared", "component_type": "email_local"}]
        }),
        serde_json::json!({
            "original": "second@example.invalid", "replacement": "first@example.com",
            "pattern_name": "email", "components": [{"original": "second", "replacement": "first", "component_type": "email_local"}]
        }),
    ];
    write(
        &key,
        &rows
            .iter()
            .map(serde_json::Value::to_string)
            .collect::<Vec<_>>()
            .join("\n"),
    );
    let anonymized = run_output(
        &[
            "anon",
            "--no-ner",
            "--patterns",
            "email",
            "--strategy",
            "fake",
            "--seed",
            "42",
            "-k",
            key.to_str().unwrap(),
        ],
        Some("first@example.invalid"),
    );
    assert!(anonymized.status.success());
    assert_eq!(
        String::from_utf8_lossy(&anonymized.stdout),
        "shared@example.com"
    );
    let restored = run_output(
        &["deanon", "-k", key.to_str().unwrap()],
        Some(std::str::from_utf8(&anonymized.stdout).unwrap()),
    );
    assert_eq!(restored.status.code(), Some(0));
    assert_eq!(
        String::from_utf8(restored.stdout).unwrap(),
        "first@example.invalid"
    );
}

#[test]
fn deanon_full_aliases_win_and_ambiguous_components_are_not_guessed() {
    let mappings = [
        serde_json::json!({"original":"first-original","replacement":"whole-first-alias","pattern_name":"person","components":[
            {"original":"wrong-original","replacement":"same-alias","component_type":"name"},
            {"original":"one","replacement":"duplicate-alias","component_type":"name"},
            {"original":"component-unique","replacement":"unique-alias","component_type":"name"}
        ]}),
        serde_json::json!({"original":"right-original","replacement":"same-alias","pattern_name":"person","components":[
            {"original":"two","replacement":"duplicate-alias","component_type":"name"},
            {"original":"component-unique","replacement":"unique-alias","component_type":"name"}
        ]}),
        serde_json::json!({"original":"third-original","replacement":"whole-third-alias","pattern_name":"person","components":[
            {"original":"one","replacement":"duplicate-alias","component_type":"name"}
        ]}),
    ];
    for mappings in [mappings.to_vec(), mappings.into_iter().rev().collect()] {
        let out = deanon_with_mappings(
            &mappings,
            "same-alias duplicate-alias unique-alias whole-first-alias whole-third-alias",
        );
        assert_eq!(out.status.code(), Some(0));
        assert_eq!(
            String::from_utf8(out.stdout).unwrap(),
            "right-original duplicate-alias component-unique first-original third-original"
        );
    }
}

#[test]
fn deanon_matches_literals_longest_first_and_preserves_short_alias_boundaries() {
    let mappings: Vec<_> = [
        ("long-original", "alias-long"),
        ("short-original", "alias"),
        ("$1\\雪", "a+b(c)[d].*"),
        ("dots-original", "D."),
        ("short-value", "ID"),
        ("unicode-original", "東京"),
        ("ZZ", "EQ"),
    ]
    .into_iter()
    .map(|(original, replacement)| {
        serde_json::json!({
            "original":original,"replacement":replacement,"pattern_name":"person","components":[]
        })
    })
    .collect();
    // Explicit legacy mode preserves component guessing and short-alias boundaries.
    let out = deanon_with_mappings(
        &mappings,
        "alias-long alias a+b(c)[d].* [D. D.x XD.y] ID xID IDx _ID ID_ éID IDé (ID) 東京 [EQ]",
    );
    assert_eq!(out.status.code(), Some(0));
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        "long-original short-original $1\\雪 [D. dots-originalx XD.y] short-value xID IDx _ID ID_ éID IDé (short-value) unicode-original [ZZ]"
    );
}

#[test]
fn deanon_full_alias_cascade_is_single_pass_with_file_and_stdin_parity() {
    let dir = tempfile::tempdir().unwrap();
    let key = dir.path().join("keys.jsonl");
    let input_path = dir.path().join("input.json");
    let output_path = dir.path().join("output.json");
    let mappings = [
        serde_json::json!({"version":"1"}),
        serde_json::json!({"original":"original-value","replacement":"alias-one-long","pattern_name":"person","components":[]}),
        serde_json::json!({"original":"alias-one-long","replacement":"alias-two-even-longer","pattern_name":"person","components":[]}),
    ];
    write(
        &key,
        &mappings
            .iter()
            .map(serde_json::Value::to_string)
            .collect::<Vec<_>>()
            .join("\n"),
    );
    let input = r#"{"values":["alias-two-even-longer","alias-one-long"],"note":"Jörg 東京"}"#;
    let expected = r#"{"values":["alias-one-long","original-value"],"note":"Jörg 東京"}"#;
    write(&input_path, input);
    for file_input in [false, true] {
        for file_output in [false, true] {
            let mut args = vec!["deanon", "-k", key.to_str().unwrap()];
            if file_input {
                args.push(input_path.to_str().unwrap());
            }
            if file_output {
                args.extend(["-o", output_path.to_str().unwrap()]);
            }
            let out = run_output(&args, if file_input { None } else { Some(input) });
            assert_eq!(out.status.code(), Some(0));
            let payload = if file_output {
                assert!(out.stdout.is_empty());
                fs::read(&output_path).unwrap()
            } else {
                out.stdout
            };
            let expected_value = serde_json::from_str::<serde_json::Value>(expected).unwrap();
            assert_eq!(
                String::from_utf8(payload.clone()).unwrap(),
                serde_json::to_string_pretty(&expected_value).unwrap()
            );
            assert_eq!(
                serde_json::from_slice::<serde_json::Value>(&payload).unwrap(),
                serde_json::from_str::<serde_json::Value>(expected).unwrap()
            );
        }
    }
}

#[test]
fn deanon_rejects_malformed_or_conflicting_key_files_without_output() {
    let dir = tempfile::tempdir().unwrap();
    let key = dir.path().join("keys.jsonl");
    let header = r#"{"version":"1"}"#;
    let entry = |original: &str, replacement: &str| {
        serde_json::json!({
            "original": original, "replacement": replacement,
            "pattern_name": "email", "components": []
        })
        .to_string()
    };
    let valid = entry("alpha@example.invalid", "alias@example.com");
    for (body, diagnostic) in [
        (
            format!("invalid header\n{valid}\n"),
            "invalid key-file header",
        ),
        (
            format!("{header}\n{valid}\ninvalid entry\n"),
            "invalid replacement entry",
        ),
        (
            format!(
                "{header}\n{valid}\n{}\n",
                entry("alpha@example.invalid", "other@example.com")
            ),
            "conflicting key-file entries",
        ),
        (
            format!(
                "{header}\n{valid}\n{}\n",
                entry("beta@example.invalid", "alias@example.com")
            ),
            "conflicting key-file entries",
        ),
    ] {
        write(&key, &body);
        let out = run_output(
            &["deanon", "-k", key.to_str().unwrap()],
            Some("alias@example.com"),
        );
        assert_eq!(out.status.code(), Some(1));
        assert!(out.stdout.is_empty());
        assert!(String::from_utf8_lossy(&out.stderr).contains(diagnostic));
    }

    // Exact duplicate entries are valid and must still restore normally.
    write(&key, &format!("{header}\n{valid}\n{valid}\n"));
    let out = run_output(
        &["deanon", "-k", key.to_str().unwrap()],
        Some("alias@example.com"),
    );
    assert_eq!(out.status.code(), Some(0));
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        "alpha@example.invalid"
    );
}

#[test]
fn context_recorded_in_key_file_header() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.json");
    let key = dir.path().join("keys.jsonl");
    write(&input, r#"{"email": "alice@example.com"}"#);

    let (ok, _) = run(
        &[
            "anon",
            input.to_str().unwrap(),
            "--format",
            "json",
            "--strategy",
            "consistent",
            "--seed",
            "42",
            "--context",
            "release-2026-10",
            "-k",
            key.to_str().unwrap(),
        ],
        None,
    );
    assert!(ok);
    let contents = fs::read_to_string(&key).unwrap();
    let header: serde_json::Value =
        serde_json::from_str(contents.lines().next().unwrap()).expect("header");
    assert_eq!(header["context"].as_str().unwrap(), "release-2026-10");
    assert_eq!(header["seed"].as_u64().unwrap(), 42);
    assert_eq!(header["strategy"].as_str().unwrap(), "consistent");
}

#[test]
fn seeded_build_is_reproducible_across_runs() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.json");
    write(
        &input,
        r#"{"email": "alice@example.com", "user": {"name": "John Smith"}}"#,
    );

    let run_build = |out: &std::path::Path| {
        run(
            &[
                "anon",
                input.to_str().unwrap(),
                "--format",
                "json",
                "--strategy",
                "consistent",
                "--seed",
                "42",
                "-o",
                out.to_str().unwrap(),
            ],
            None,
        )
    };
    let a = dir.path().join("a.json");
    let b = dir.path().join("b.json");
    let (ok1, _) = run_build(&a);
    let (ok2, _) = run_build(&b);
    assert!(ok1 && ok2);
    assert_eq!(
        fs::read_to_string(&a).unwrap(),
        fs::read_to_string(&b).unwrap(),
        "same seed+input+strategy must yield identical pseudonyms"
    );
}
