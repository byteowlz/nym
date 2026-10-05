//! Offline CLI regressions for automatic structured input and atomic failures.
use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    process::{Command, Output, Stdio},
};

fn run(args: &[&str], input: Option<&str>) -> Output {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("config.toml");
    fs::write(&cfg, "[ner]\nenabled = false\n").unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nym"));
    cmd.arg("--config")
        .arg(cfg)
        .args(args)
        .env("HF_HUB_OFFLINE", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().unwrap();
    if let Some(input) = input {
        child
            .stdin
            .take()
            .unwrap()
            .write_all(input.as_bytes())
            .unwrap();
    }
    child.wait_with_output().unwrap()
}
fn trace() -> String {
    [json!({"type":"session","id":"synthetic-session","usage":4096,"active":true,"data":null,"key@example.com":42}),
     json!({"type":"message","parentId":"synthetic-session","tool":"bash","arguments":{"text":"Jörg 東京 alice@example.com"},"thinking":"alice@example.com"})]
    .iter().map(Value::to_string).collect::<Vec<_>>().join("\n") + "\n"
}
fn records(bytes: &[u8]) -> Vec<Value> {
    String::from_utf8_lossy(bytes)
        .lines()
        .filter(|s| !s.trim().is_empty())
        .map(|s| serde_json::from_str(s).unwrap())
        .collect()
}
#[test]
fn auto_jsonl_file_stdin_and_explicit_processing_agree() {
    let dir = tempfile::tempdir().unwrap();
    let input = trace();
    for ext in ["jsonl", "ndjson", "JSONL", "NDJSON", ""] {
        let path = dir.path().join(format!("trace.{ext}"));
        fs::write(&path, &input).unwrap();
        let out = run(
            &[
                "anon",
                path.to_str().unwrap(),
                "--no-ner",
                "--patterns",
                "email",
                "--strategy",
                "placeholder",
            ],
            None,
        );
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let result = records(&out.stdout);
        let mut expected = records(input.as_bytes());
        expected[1]["arguments"]["text"] = json!("Jörg 東京 <EMAIL>");
        expected[1]["thinking"] = json!("<EMAIL>");
        assert_eq!(result, expected);
        let stdin = run(
            &[
                "anon",
                "--no-ner",
                "--patterns",
                "email",
                "--strategy",
                "placeholder",
            ],
            Some(&input),
        );
        assert_eq!(records(&stdin.stdout), expected);
    }
}
#[test]
fn audit_jsonl_reports_value_free_blockers_and_selectors_per_record() {
    let out = run(
        &[
            "detect",
            "--no-ner",
            "--patterns",
            "email",
            "--include-path",
            "arguments.text",
            "--json-coverage",
            "--fail-on",
            "email",
            "--summary-json",
        ],
        Some(&trace()),
    );
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(
        serde_json::from_slice::<Value>(&out.stdout).unwrap(),
        json!({"total":1,"by_pattern":{"email":1},"by_category":{"contact":1},"blockers":["email"]})
    );
    assert!(!String::from_utf8_lossy(&out.stdout).contains("alice"));
    assert!(String::from_utf8_lossy(&out.stderr).contains("record[2].arguments.text"));
}
#[test]
fn malformed_record_does_not_clobber_destination_or_leak_payload() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("trace.jsonl");
    let dest = dir.path().join("output.jsonl");
    fs::write(
        &input,
        "{\"email\":\"alice@example.com\"}\n{\"sensitive\":VERY_PRIVATE_PAYLOAD}\n",
    )
    .unwrap();
    for stream in [false, true]
        .into_iter()
        .filter(|stream| !*stream || cfg!(feature = "streaming"))
    {
        fs::write(&dest, "original destination").unwrap();
        let mut args = vec![
            "anon",
            input.to_str().unwrap(),
            "-o",
            dest.to_str().unwrap(),
            "--no-ner",
        ];
        if stream {
            args.push("--stream");
        }
        let out = run(&args, None);
        assert_eq!(out.status.code(), Some(1));
        assert_eq!(fs::read_to_string(&dest).unwrap(), "original destination");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains("line 2"), "{err}");
        assert!(!err.contains("VERY_PRIVATE_PAYLOAD"));
        assert!(out.stdout.is_empty());
    }
}
#[test]
fn sniffing_handles_bom_whitespace_crlf_pretty_json_and_text_override() {
    let input = "\u{feff}  \r\n{\"email\":\"alice@example.com\"}\r\n\r\n{\"email\":\"bob@example.com\"}\r\n";
    let out = run(
        &[
            "anon",
            "--no-ner",
            "--patterns",
            "email",
            "--strategy",
            "placeholder",
        ],
        Some(input),
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        records(&out.stdout),
        vec![json!({"email":"<EMAIL>"}), json!({"email":"<EMAIL>"})]
    );
    let pretty = "{\n  \"email\": \"alice@example.com\",\n  \"count\": 4096\n}\n";
    let out = run(
        &["anon", "--no-ner", "--strategy", "placeholder"],
        Some(pretty),
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&out.stdout).unwrap(),
        json!({"email":"<EMAIL>","count":4096})
    );
    let out = run(
        &[
            "anon",
            "--no-ner",
            "--format",
            "text",
            "--patterns",
            "email",
            "--strategy",
            "placeholder",
        ],
        Some("{not JSON: alice@example.com}"),
    );
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        "{not JSON: <EMAIL>}"
    );
}
#[test]
fn ordinary_code_is_not_forced_into_json_and_declared_json_does_not_fallback() {
    for input in [
        "",
        "let x = { email: 'alice@example.com' };",
        "{not JSON: alice@example.com}",
    ] {
        let out = run(
            &[
                "anon",
                "--no-ner",
                "--patterns",
                "email",
                "--strategy",
                "placeholder",
            ],
            Some(input),
        );
        assert!(out.status.success());
    }
    let out = run(
        &["detect", "--no-ner", "--format", "json", "--summary-json"],
        Some("{not JSON: VERY_PRIVATE_PAYLOAD}"),
    );
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    assert!(!String::from_utf8_lossy(&out.stderr).contains("VERY_PRIVATE_PAYLOAD"));
}
#[test]
fn single_record_extensions_empty_and_explicit_override() {
    let dir = tempfile::tempdir().unwrap();
    for ext in ["jsonl", "ndjson"] {
        let path = dir.path().join(format!("single.{ext}"));
        fs::write(&path, "{\"email\":\"alice@example.com\"}").unwrap();
        let out = run(
            &[
                "detect",
                path.to_str().unwrap(),
                "--no-ner",
                "--fail-on",
                "email",
                "--summary-json",
            ],
            None,
        );
        assert_eq!(out.status.code(), Some(2));
        fs::write(&path, "").unwrap();
        let out = run(
            &[
                "detect",
                path.to_str().unwrap(),
                "--no-ner",
                "--summary-json",
            ],
            None,
        );
        assert!(out.status.success());
        assert_eq!(
            serde_json::from_slice::<Value>(&out.stdout).unwrap()["total"],
            0
        );
    }
}
#[test]
fn jsonl_mappings_are_stable_and_reversible() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trace.jsonl");
    let key = dir.path().join("key.jsonl");
    fs::write(&path, trace()).unwrap();
    let out = run(
        &[
            "anon",
            path.to_str().unwrap(),
            "--no-ner",
            "--patterns",
            "email",
            "--strategy",
            "consistent",
            "--seed",
            "42",
            "-k",
            key.to_str().unwrap(),
        ],
        None,
    );
    assert!(out.status.success());
    let values = records(&out.stdout);
    let alias = values[1]["thinking"].as_str().unwrap();
    assert_eq!(values[1]["arguments"]["text"], format!("Jörg 東京 {alias}"));
    let restored = run(
        &["deanon", "-k", key.to_str().unwrap()],
        Some(std::str::from_utf8(&out.stdout).unwrap()),
    );
    assert_eq!(records(&restored.stdout), records(trace().as_bytes()));
}
#[test]
fn typo_policy_and_explicit_json_multiple_documents_fail() {
    for args in [
        vec!["detect", "--no-ner", "--fail-on", "emali", "--summary-json"],
        vec!["detect", "--no-ner", "--format", "json", "--summary-json"],
    ] {
        let out = run(&args, Some(&trace()));
        assert_eq!(out.status.code(), Some(1));
        assert!(out.stdout.is_empty());
    }
}
#[cfg(feature = "streaming")]
#[test]
fn streaming_honors_rulesets_and_rejects_unsupported_machine_output() {
    let input = "contact alice@example.com; synthetic key AKIAIOSFODNN7EXAMPLE\n";
    let out = run(
        &[
            "anon",
            "--stream",
            "--format",
            "text",
            "--no-ner",
            "--only-keys",
            "--strategy",
            "placeholder",
        ],
        Some(input),
    );
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        "contact alice@example.com; synthetic key <API_KEY>\n"
    );
    let out = run(
        &[
            "detect",
            "--stream",
            "--format",
            "jsonl",
            "--no-ner",
            "--only-contact",
        ],
        Some("{\"text\":\"alice@example.com\",\"counter\":4096}\n"),
    );
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("alice@example.com"));
    let out = run(
        &[
            "detect", "--stream", "--no-ner", "--format", "jsonl", "--json",
        ],
        Some("{\"email\":\"alice@example.com\"}\n"),
    );
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("machine-readable output is not supported")
    );
}

#[test]
fn decoded_findings_have_record_paths_and_utf8_byte_offsets() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trace.jsonl");
    let input = "{\"key@example.com\":4096}\n{\"text\":\"Jörg 東京 a\\u006cice@example.com\"}\n";
    fs::write(&path, input).unwrap();
    let expected = json!([{"path":"record[2].text","pattern_name":"email","matched_text":"alice@example.com","start":13,"end":30,"category":"contact","confidence":"high"}]);
    for file in [false, true] {
        let mut args = vec!["detect", "--no-ner", "--patterns", "email", "--json"];
        if file {
            args.push(path.to_str().unwrap());
        }
        let out = run(&args, if file { None } else { Some(input) });
        assert!(out.status.success());
        assert_eq!(
            serde_json::from_slice::<Value>(&out.stdout).unwrap(),
            expected
        );
    }
}

#[test]
fn declared_jsonl_is_never_scanned_as_text() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trace.jsonl");
    fs::write(&path, "{invalid: alice@example.com}\n").unwrap();
    let out = run(
        &[
            "detect",
            path.to_str().unwrap(),
            "--no-ner",
            "--summary-json",
        ],
        None,
    );
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
}
