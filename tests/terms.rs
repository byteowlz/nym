//! Synthetic-only tests of the public corpus vocabulary workflow.
use serde_json::{Value, json};
use std::{
    fs,
    process::{Command, Output},
};
use tempfile::TempDir;

fn run(dir: &TempDir, args: &[&str]) -> Output {
    let config = dir.path().join("config.toml");
    fs::write(&config, "").unwrap();
    Command::new(env!("CARGO_BIN_EXE_nym"))
        .args(["--config", config.to_str().unwrap()])
        .args(args)
        .output()
        .unwrap()
}
fn path(dir: &TempDir, name: &str) -> String {
    dir.path().join(name).to_str().unwrap().to_owned()
}
fn load(file: &str) -> Value {
    serde_json::from_slice(&fs::read(file).unwrap()).unwrap()
}

#[test]
fn public_discover_review_export_detect_restore_roundtrip() {
    let dir = TempDir::new().unwrap();
    let source = path(&dir, "source.txt");
    let discovery = path(&dir, "discovery.json");
    let review = path(&dir, "review.json");
    let terms = path(&dir, "terms.txt");
    let html = path(&dir, "review.html");
    let input =
        "Project Veltrix is private.\nProject Veltrix is private.\nPublic function stays.\n";
    fs::write(&source, input).unwrap();
    let result = run(
        &dir,
        &[
            "--json", "terms", "discover", &source, "--output", &discovery,
        ],
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!String::from_utf8_lossy(&result.stdout).contains("Veltrix"));
    let artifact = load(&discovery);
    let candidate = artifact["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["term"] == "Veltrix")
        .unwrap();
    let decision = format!("{}=sensitive", candidate["id"].as_str().unwrap());
    assert!(
        run(
            &dir,
            &[
                "terms",
                "review",
                &discovery,
                "--output",
                &review,
                "--decision",
                &decision
            ]
        )
        .status
        .success()
    );
    assert!(
        run(
            &dir,
            &[
                "terms", "review", &discovery, "--resume", &review, "--html", "--output", &html
            ]
        )
        .status
        .success()
    );
    assert!(
        fs::read_to_string(&html)
            .unwrap()
            .contains("Download review JSON")
    );
    assert!(
        run(
            &dir,
            &[
                "terms", "export", &discovery, "--review", &review, "--output", &terms
            ]
        )
        .status
        .success()
    );
    assert_eq!(fs::read_to_string(&terms).unwrap(), "Veltrix\n");
    let detect = run(
        &dir,
        &[
            "--json",
            "detect",
            &source,
            "--no-ner",
            "--sensitive-terms-file",
            &terms,
        ],
    );
    assert!(
        detect.status.success(),
        "{}",
        String::from_utf8_lossy(&detect.stderr)
    );
    assert!(String::from_utf8_lossy(&detect.stdout).contains("Veltrix"));
    let anon = path(&dir, "anon.txt");
    let key = path(&dir, "key.json");
    let restored = path(&dir, "restored.txt");
    let result = run(
        &dir,
        &[
            "-q",
            "anon",
            &source,
            "--no-ner",
            "--sensitive-terms-file",
            &terms,
            "--output",
            &anon,
            "--key-file",
            &key,
        ],
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!fs::read_to_string(&anon).unwrap().contains("Veltrix"));
    let result = run(
        &dir,
        &[
            "-q",
            "deanon",
            &anon,
            "--key-file",
            &key,
            "--output",
            &restored,
        ],
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(fs::read_to_string(restored).unwrap(), input);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for file in [&discovery, &review, &terms, &html] {
            assert_eq!(
                fs::metadata(file).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }
}

#[test]
fn decoded_jsonl_selectors_and_source_offsets() {
    let dir = TempDir::new().unwrap();
    let source = path(&dir, "source.jsonl");
    let output = path(&dir, "terms.json");
    fs::write(&source, "\u{feff}{\"text\":\"\\u00d6 Rina\",\"meta\":\"Excluded\"}\r\n\n{\"text\":\"Ö Rina\",\"meta\":\"Excluded\"}\n").unwrap();
    let result = run(
        &dir,
        &[
            "terms",
            "discover",
            &source,
            "--include",
            "text",
            "--output",
            &output,
        ],
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let artifact = load(&output);
    assert_eq!(artifact["unit_count"], 2);
    assert!(!fs::read_to_string(&output).unwrap().contains("Excluded"));
    let rina = artifact["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["term"] == "Rina")
        .unwrap();
    assert_eq!(rina["examples"][0]["start"], 3);
    assert_eq!(rina["examples"][0]["end"], 7);
    assert_eq!(rina["occurrences"], 2);
    assert_eq!(rina["distinct_texts"], 1);
}

#[test]
fn failures_preserve_destinations_and_never_publish_partial_results() {
    let dir = TempDir::new().unwrap();
    let source = path(&dir, "broken.jsonl");
    let output = path(&dir, "terms.json");
    fs::write(&source, "{\"text\":\"Veltrix\"}\n{broken secret-sentinel}").unwrap();
    fs::write(&output, "preserved").unwrap();
    let result = run(
        &dir,
        &["terms", "discover", &source, "--output", &output, "--force"],
    );
    assert!(!result.status.success());
    assert_eq!(fs::read_to_string(&output).unwrap(), "preserved");
    assert!(!String::from_utf8_lossy(&result.stderr).contains("secret-sentinel"));
    fs::write(&source, "{\"text\":\"Veltrix Veltrix\"}\n").unwrap();
    assert!(
        !run(&dir, &["terms", "discover", &source, "--output", &output])
            .status
            .success()
    );
    assert_eq!(fs::read_to_string(&output).unwrap(), "preserved");
    let fresh = path(&dir, "fresh.json");
    fs::write(&source, "{\"text\":\"Jane Doe\"}\n").unwrap();
    assert!(
        !run(
            &dir,
            &[
                "terms",
                "discover",
                &source,
                "--output",
                &fresh,
                "--max-distinct",
                "1",
                "--limit",
                "1"
            ]
        )
        .status
        .success()
    );
    assert!(!std::path::Path::new(&fresh).exists());
}

#[test]
fn no_implicit_approval_and_stale_or_duplicate_decisions_rejected() {
    let dir = TempDir::new().unwrap();
    let source = path(&dir, "input.txt");
    let artifact = path(&dir, "terms.json");
    let review = path(&dir, "review.json");
    let output = path(&dir, "export.txt");
    fs::write(&source, "Veltrix Veltrix\n").unwrap();
    assert!(
        run(&dir, &["terms", "discover", &source, "--output", &artifact])
            .status
            .success()
    );
    assert!(
        run(&dir, &["terms", "review", &artifact, "--output", &review])
            .status
            .success()
    );
    assert!(
        !run(
            &dir,
            &[
                "terms", "export", &artifact, "--review", &review, "--output", &output
            ]
        )
        .status
        .success()
    );
    let mut data = load(&review);
    let candidate = load(&artifact)["candidates"][0]["id"].clone();
    data["decisions"] = json!([{"candidate_id":candidate,"choice":"sensitive"},{"candidate_id":candidate,"choice":"sensitive"}]);
    fs::write(&review, serde_json::to_vec(&data).unwrap()).unwrap();
    assert!(
        !run(
            &dir,
            &[
                "terms", "export", &artifact, "--review", &review, "--output", &output
            ]
        )
        .status
        .success()
    );
    data["decisions"] = json!([]);
    data["discovery_sha256"] = json!("0".repeat(64));
    fs::write(&review, serde_json::to_vec(&data).unwrap()).unwrap();
    assert!(
        !run(
            &dir,
            &[
                "terms", "review", &artifact, "--resume", &review, "--output", &output
            ]
        )
        .status
        .success()
    );
    assert!(!std::path::Path::new(&output).exists());
}

#[test]
fn offline_review_escapes_script_injection_in_context() {
    let dir = TempDir::new().unwrap();
    let source = path(&dir, "input.txt");
    let artifact = path(&dir, "terms.json");
    let html = path(&dir, "review.html");
    fs::write(&source, "Veltrix </script><script>window.injected=1</script>\nVeltrix </script><script>window.injected=1</script>\n").unwrap();
    assert!(
        run(&dir, &["terms", "discover", &source, "--output", &artifact])
            .status
            .success()
    );
    assert!(
        run(
            &dir,
            &["terms", "review", &artifact, "--output", &html, "--html"]
        )
        .status
        .success()
    );
    let html = fs::read_to_string(html).unwrap();
    assert!(!html.contains("</script><script>window.injected"));
    assert!(html.contains("\\u003c/script\\u003e"));
}

#[test]
fn whole_hosts_urls_paths_and_configured_defaults_are_discovered() {
    let dir = TempDir::new().unwrap();
    let source = path(&dir, "input.txt");
    let artifact = path(&dir, "terms.json");
    let config = path(&dir, "custom.toml");
    fs::write(&source, "https://hidden.example.test/one /srv/Veltrix\nhttps://hidden.example.test/two /srv/Veltrix\n").unwrap();
    fs::write(&config, "[terms]\nmin_count=99\n").unwrap();
    let invoke = |extra: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_nym"))
            .args([
                "--config", &config, "terms", "discover", &source, "--output", &artifact,
            ])
            .args(extra)
            .output()
            .unwrap()
    };
    assert!(invoke(&[]).status.success());
    let value: Value = serde_json::from_str(&fs::read_to_string(&artifact).unwrap()).unwrap();
    assert_eq!(value["candidates"], serde_json::json!([]));
    assert!(invoke(&["--min-count", "1", "--force"]).status.success());
    let value: Value = serde_json::from_str(&fs::read_to_string(&artifact).unwrap()).unwrap();
    for term in [
        "hidden.example.test",
        "https://hidden.example.test/one",
        "/srv/Veltrix",
    ] {
        assert!(
            value["candidates"]
                .as_array()
                .unwrap()
                .iter()
                .any(|c| c["term"] == term)
        );
    }
}
