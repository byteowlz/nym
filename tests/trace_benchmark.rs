//! Offline native-JSON proof for the synthetic agent-trace benchmark only.
//! This is not an end-to-end pi JSONL parser or universal de-identification test.
use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};

use serde_json::Value;

const EXCLUDED: &str = "date,time,ipv4,ipv6,unix_path,social_url";

fn pointer(path: &str) -> String {
    format!(
        "/{}",
        path.replace('[', ".").replace(']', "").replace('.', "/")
    )
}

fn run(command: &str, input: &str, format: &str, exclusions: &[String]) -> Value {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("config.toml");
    std::fs::write(
        &config,
        "[ner]\nenabled = false\n[decision]\nenabled = false\n",
    )
    .unwrap();
    let mut process = Command::new(env!("CARGO_BIN_EXE_nym"));
    process.args([
        command,
        "--no-ner",
        "--format",
        format,
        "--ruleset",
        "all",
        "--exclude",
        EXCLUDED,
        "--min-confidence",
        "low",
        "--quiet",
        "--no-color",
    ]);
    process.arg("--config").arg(config);
    if command == "detect" {
        process.arg("--json");
    } else {
        process.args(["--strategy", "placeholder", "--seed", "42"]);
    }
    for path in exclusions {
        process.args(["--exclude-path", path]);
    }
    for (key, _) in std::env::vars().filter(|(key, _)| key.starts_with("NYM_")) {
        process.env_remove(key);
    }
    for key in ["HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"] {
        process.env(key, dir.path());
    }
    let mut child = process
        .current_dir(dir.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "synthetic benchmark child failed: {:?}",
        output.status
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn contextual_accounts_preserve_labels_quotes_and_uncaptured_bare_accounts() {
    let input = serde_json::json!({"text":"username=scribe_amber; 'login': 'fictional.account'; handle=\"raven.test\"; user123; buffer_size=4096", "number":4096});
    assert_eq!(
        run("anon", &input.to_string(), "json", &[]),
        serde_json::json!({"text":"username=<USERNAME>; 'login': '<USERNAME>'; handle=\"<USERNAME>\"; <USERNAME>; buffer_size=4096", "number":4096})
    );
}

#[test]
fn synthetic_trace_native_anon_removes_in_scope_gold_and_preserves_full_expected_document() {
    let fixture: Value =
        serde_json::from_str(include_str!("../scripts/bench/fixtures/agent_traces.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let mut expected = case["trace"].clone();
        let mut edits: BTreeMap<String, Vec<(usize, usize, &str)>> = BTreeMap::new();
        for annotation in case["gold"].as_array().unwrap() {
            if annotation["scope"] != "regex" {
                continue; // These are explicitly measured misses, not claimed removals.
            }
            let path = pointer(annotation["path"].as_str().unwrap());
            let start = annotation["start"].as_u64().unwrap() as usize;
            let end = annotation["end"].as_u64().unwrap() as usize;
            let replacement = match annotation["class"].as_str().unwrap() {
                "email" => "<EMAIL>",
                "username" => "<USERNAME>",
                "aws_key" | "api_key" => "<API_KEY>",
                "identity_uuid" => "<UUID>",
                other => panic!("unhandled synthetic class {other}"),
            };
            edits
                .entry(path)
                .or_default()
                .push((start, end, replacement));
        }
        for (path, mut replacements) in edits {
            let target = expected.pointer_mut(&path).unwrap();
            let mut text = target.as_str().unwrap().to_owned();
            replacements.sort_unstable_by_key(|edit| std::cmp::Reverse(edit.0));
            for (start, end, replacement) in replacements {
                text.replace_range(start..end, replacement);
            }
            *target = text.into();
        }
        let exclusions = vec![
            "session.id".into(),
            "records[*].id".into(),
            "records[*].parentId".into(),
            "records[*].timestamp".into(),
            "records[*].message.toolCallId".into(),
            "records[*].message.content[*].id".into(),
        ];
        let actual = run("anon", &case["trace"].to_string(), "json", &exclusions);
        assert_eq!(actual, expected, "synthetic case {}", case["id"]);
    }
}

#[test]
fn synthetic_trace_context_negative_commands_numbers_and_timestamps_not_regex_identity() {
    let text = "bash 4096 --strict --timeout 30 --no-progress\n\
        let buffer_size: usize = 4096;\n\
        serde_json tokio clap https://docs.rs/serde_json/\n\
        /tmp/synthetic-project /usr/local/lib 127.0.0.1:8080 ::1\n\
        2026-02-01T10:20:30Z";
    assert_eq!(run("detect", text, "text", &[]), serde_json::json!([]));
    // A positive username context uses an arbitrary fictional account, not a
    // blanket safe list of all metadata or all tokens with underscores.
    assert_eq!(
        run("detect", "username=scribe_amber", "text", &[]),
        serde_json::json!([{
            "category": "social", "confidence": "low", "start": 9, "end": 21,
            "matched_text": "scribe_amber", "pattern_name": "username"
        }])
    );
}
