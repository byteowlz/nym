//! Cross-process seeded mapping reuse, permissions, and round-trip regressions.
use std::fs;
use std::io::Write;
use std::process::{Command, Output, Stdio};

fn run(args: &[&str], input: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_nym"))
        .args(args)
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
    child.wait_with_output().unwrap()
}

#[test]
fn invalid_or_incompatible_maps_fail_before_any_output() {
    let base = serde_json::json!({"version":"1","session":"fixture","strategy":"fake","seed":42,"context":"release-fixture"});
    let mut bad_version = base.clone();
    bad_version["version"] = "unsupported-private-map-material".into();
    let mut bad_strategy = base.clone();
    bad_strategy["strategy"] = "mask".into();
    let mut bad_seed = base.clone();
    bad_seed["seed"] = 99.into();
    let mut bad_context = base.clone();
    bad_context["context"] = "private-map-material".into();
    let entry = serde_json::json!({"original":"alpha@example.invalid","replacement":"alias-a@example.invalid","pattern_name":"email","components":[]});
    let mut conflict = entry.clone();
    conflict["replacement"] = "alias-b@example.invalid".into();
    let maps = [
        String::new(),
        "private-map-material".to_string(),
        format!("{base}\n{{\"original\":\"private-map-material"),
        format!("{base}\n{entry}\n{conflict}\n"),
        bad_version.to_string(),
        bad_strategy.to_string(),
        bad_seed.to_string(),
        bad_context.to_string(),
    ];
    for map in maps {
        for stream in [false, true]
            .into_iter()
            .filter(|s| !*s || cfg!(feature = "streaming"))
        {
            for file_output in [false, true] {
                let dir = tempfile::tempdir().unwrap();
                let key = dir.path().join("keys.jsonl");
                let output = dir.path().join("output.txt");
                fs::write(&key, &map).unwrap();
                fs::write(&output, "unchanged destination").unwrap();
                let mut args = vec![
                    "anon",
                    "--no-ner",
                    "--patterns",
                    "email",
                    "--strategy",
                    "fake",
                    "--seed",
                    "42",
                    "--context",
                    "release-fixture",
                    "-k",
                    key.to_str().unwrap(),
                ];
                if stream {
                    args.push("--stream");
                }
                if file_output {
                    args.extend(["-o", output.to_str().unwrap()]);
                }
                let out = run(&args, "alpha@example.invalid\n");
                assert_eq!(out.status.code(), Some(1));
                assert!(
                    out.stdout.is_empty(),
                    "invalid maps must not publish transformed text"
                );
                assert_eq!(
                    fs::read_to_string(&output).unwrap(),
                    "unchanged destination"
                );
                assert_eq!(fs::read_to_string(&key).unwrap(), map);
                assert!(!String::from_utf8_lossy(&out.stderr).contains("private-map-material"));
            }
        }
    }
}

#[test]
fn resumed_seeded_maps_preserve_all_round_trips() {
    for strategy in ["fake", "consistent", "hash"] {
        let dir = tempfile::tempdir().unwrap();
        let key = dir.path().join("keys.jsonl");
        let key_arg = key.to_str().unwrap();
        let mut outputs = Vec::new();
        for input in [
            r#"{"a":"alpha@example.invalid","b":"beta@example.invalid","note":"Jörg, 東京"}"#,
            r#"{"b":"beta@example.invalid","c":"gamma@example.invalid","note":"Jörg, 東京"}"#,
            r#"{"c":"gamma@example.invalid","a":"alpha@example.invalid","note":"Jörg, 東京"}"#,
            r#"{"upper":"ALPHA@example.invalid","title":"Alpha@example.invalid","note":"Jörg, 東京"}"#,
        ] {
            let out = run(
                &[
                    "anon",
                    "--no-ner",
                    "--patterns",
                    "email",
                    "--format",
                    "json",
                    "--strategy",
                    strategy,
                    "--seed",
                    "42",
                    "--context",
                    "release-fixture",
                    "-k",
                    key_arg,
                ],
                input,
            );
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
            let payload: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
            assert_eq!(payload["note"], "Jörg, 東京");
            outputs.push((input, out.stdout));
            let entries: Vec<serde_json::Value> = fs::read_to_string(&key)
                .unwrap()
                .lines()
                .skip(1)
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            let aliases: std::collections::HashSet<_> = entries
                .iter()
                .map(|entry| entry["replacement"].as_str().unwrap())
                .collect();
            assert_eq!(aliases.len(), entries.len());
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                assert_eq!(
                    fs::metadata(&key).unwrap().permissions().mode() & 0o777,
                    0o600
                );
            }
            // All prior outputs must still restore after every append.
            for (original, bytes) in &outputs {
                let restored = run(
                    &["deanon", "-k", key_arg],
                    std::str::from_utf8(bytes).unwrap(),
                );
                assert!(
                    restored.status.success(),
                    "{}",
                    String::from_utf8_lossy(&restored.stderr)
                );
                assert_eq!(
                    serde_json::from_slice::<serde_json::Value>(&restored.stdout).unwrap(),
                    serde_json::from_str::<serde_json::Value>(original).unwrap()
                );
            }
        }
        let first: serde_json::Value = serde_json::from_slice(&outputs[0].1).unwrap();
        let second: serde_json::Value = serde_json::from_slice(&outputs[1].1).unwrap();
        let third: serde_json::Value = serde_json::from_slice(&outputs[2].1).unwrap();
        assert_eq!(first["b"], second["b"]);
        assert_eq!(second["c"], third["c"]);
        assert_eq!(first["a"], third["a"]);
    }
}

#[cfg(feature = "streaming")]
#[test]
fn streamed_seeded_aliases_reserve_imported_mappings() {
    let dir = tempfile::tempdir().unwrap();
    let key = dir.path().join("keys.jsonl");
    let key_arg = key.to_str().unwrap();
    let args = [
        "anon",
        "--stream",
        "--no-ner",
        "--patterns",
        "email",
        "--format",
        "json",
        "--strategy",
        "fake",
        "--seed",
        "42",
        "--context",
        "release-fixture",
        "-k",
        key_arg,
    ];
    let first = run(&args, "{\"a\":\"alpha@example.invalid\"}\n");
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let second = run(&args, "{\"b\":\"beta@example.invalid\"}\n");
    assert!(
        second.status.success(),
        "{}",
        String::from_utf8_lossy(&second.stderr)
    );
    let a: serde_json::Value = serde_json::from_slice(&first.stdout).unwrap();
    let b: serde_json::Value = serde_json::from_slice(&second.stdout).unwrap();
    assert_ne!(a["a"], b["b"]);
    for (bytes, original) in [
        (
            first.stdout,
            serde_json::json!({"a":"alpha@example.invalid"}),
        ),
        (
            second.stdout,
            serde_json::json!({"b":"beta@example.invalid"}),
        ),
    ] {
        let restored = run(
            &["deanon", "-k", key_arg],
            std::str::from_utf8(&bytes).unwrap(),
        );
        assert!(restored.status.success());
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&restored.stdout).unwrap(),
            original
        );
    }
}
