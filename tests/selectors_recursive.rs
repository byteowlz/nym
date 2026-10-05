//! Recursive path exclusions retain structural identifiers at every depth.
use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn recursive_exclusion_reports_and_preserves_nested_ids() {
    let input = serde_json::json!({
        "messages": [{"id": "00000000-0000-4000-8000-000000000001", "content":"alpha@example.invalid"}],
        "meta": {"id":"00000000-0000-4000-8000-000000000002"},
        "資料": {"深い": {"id":"00000000-0000-4000-8000-000000000003"}}
    });
    for command in ["detect", "anon"] {
        let mut child = Command::new(env!("CARGO_BIN_EXE_nym"))
            .args([
                command,
                "--no-ner",
                "--format",
                "json",
                "--exclude-path",
                "**.id",
                "--json-coverage",
            ])
            .args(if command == "detect" {
                vec!["--json"]
            } else {
                vec!["--strategy", "placeholder"]
            })
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(input.to_string().as_bytes())
            .unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let payload: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        if command == "anon" {
            let mut expected = input.clone();
            expected["messages"][0]["content"] = serde_json::json!("<EMAIL>");
            assert_eq!(payload, expected);
        } else {
            assert_eq!(
                payload,
                serde_json::json!([{
                    "category": "contact", "confidence": "high",
                    "matched_text": "alpha@example.invalid",
                    "path": "messages[0].content", "pattern_name": "email"
                }])
            );
        }
        let coverage = String::from_utf8(out.stderr).unwrap();
        for path in ["messages[0].id", "meta.id", "資料.深い.id"] {
            assert!(coverage.contains(path), "coverage must report {path}");
        }
        assert!(coverage.contains("scanned: 1 path(s)"));
        assert!(coverage.contains("skipped: 3 path(s)"));
    }
}
