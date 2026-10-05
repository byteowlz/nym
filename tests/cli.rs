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
    let mut cmd = Command::new(bin());
    cmd.args(args);
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
    let out = child.wait_with_output().expect("wait");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string(),
    )
}

fn write(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, contents).unwrap();
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
            "json",
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
    // coverage report describes skipped structural ids
    assert!(stdout.contains("JSON scan coverage"));
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
