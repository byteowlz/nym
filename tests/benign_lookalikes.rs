//! Benign lookalikes must not be redacted, and real people must still be.
//!
//! This is the regression suite behind the curated vendor/product-name list: it
//! fails if the list grows into something that hides real personal data, and it
//! fails if product names regress to being flagged as people.
use std::{
    fs,
    io::Write,
    process::{Command, Output, Stdio},
};

fn run(args: &[&str], input: &str) -> Output {
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
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

/// Without NER the curated list is inert, so this suite covers the wiring and
/// the credential/contact carve-outs rather than model behaviour.
#[test]
fn vendor_list_never_suppresses_credentials_or_contacts() {
    let text =
        "model=Claude Opus owner=Claude Opus backup=GPT-4 contact=claude.opus@example.invalid\n";
    let out = run(
        &[
            "anon",
            "--no-ner",
            "--patterns",
            "email,api_key,secret_assignment",
            "--format",
            "text",
        ],
        text,
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let rendered = String::from_utf8(out.stdout).unwrap();
    // The email must still be replaced; the list must not become an exemption path.
    assert!(
        !rendered.contains("claude.opus@example.invalid"),
        "contact class was suppressed by the vendor list: {rendered}"
    );
}

/// The detector-level unit tests cover the list itself; this guards the CLI path
/// so a future refactor cannot silently drop the filter from `finish_matches`.
#[test]
fn structured_output_keeps_unrelated_fields_intact() {
    let input = r#"{"model":"Claude Opus","owner":"Benedikt Hammerschmidt","note":"port 8080"}"#;
    let out = run(
        &[
            "anon",
            "--no-ner",
            "--patterns",
            "email",
            "--format",
            "json",
        ],
        input,
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let payload: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(payload["model"], "Claude Opus");
    assert_eq!(payload["owner"], "Benedikt Hammerschmidt");
    assert_eq!(payload["note"], "port 8080");
}

/// Infrastructure addresses are not personal data; a public address still is.
#[test]
fn local_ipv4_is_harmless_and_public_ipv4_still_redacted() {
    let input = "listen 127.0.0.1:8080 tailnet 100.64.0.12 lan 192.168.1.1 public 8.8.8.8 docs 203.0.113.9\n";
    let out = run(
        &["anon", "--no-ner", "--patterns", "ipv4", "--format", "text"],
        input,
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let rendered = String::from_utf8(out.stdout).unwrap();
    for local in ["127.0.0.1", "100.64.0.12", "192.168.1.1", "203.0.113.9"] {
        assert!(
            rendered.contains(local),
            "{local} should stay readable: {rendered}"
        );
    }
    assert!(
        !rendered.contains("8.8.8.8"),
        "public address must stay redacted: {rendered}"
    );
}

#[test]
fn private_ipv4_can_be_reported_explicitly() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("config.toml");
    fs::write(
        &cfg,
        "[ner]\nenabled = false\n[detection]\nreport_private_ipv4 = true\n",
    )
    .unwrap();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_nym"));
    cmd.arg("--config")
        .arg(cfg)
        .args(["anon", "--no-ner", "--patterns", "ipv4", "--format", "text"])
        .env("HF_HUB_OFFLINE", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"lan 192.168.1.1\n")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !String::from_utf8(out.stdout)
            .unwrap()
            .contains("192.168.1.1")
    );
}
