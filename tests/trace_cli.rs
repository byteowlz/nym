//! Offline CLI policy tests with synthetic literals only.

#[cfg(feature = "decision")]
#[test]
fn decision_keep_cannot_veto_sensitive_literals_or_regex_credentials() {
    use std::{
        io::Read,
        net::TcpListener,
        time::{Duration, Instant},
    };
    let dir = tempfile::tempdir().unwrap();
    let list = dir.path().join("terms.txt");
    fs::write(&list, "PrivateTerm\n").unwrap();
    for text in ["PrivateTerm", "AKIAIOSFODNN7EXAMPLE"] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!(
            "http://{}/v1/chat/completions",
            listener.local_addr().unwrap()
        );
        let server = std::thread::spawn(move || {
            let start = Instant::now();
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && start.elapsed() < Duration::from_secs(10) =>
                    {
                        std::thread::sleep(Duration::from_millis(10))
                    }
                    Err(error) => panic!("local mock did not receive a request: {error}"),
                }
            };
            // macOS accepted sockets can inherit the listener's nonblocking mode.
            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(10)))
                .unwrap();
            let mut bytes = Vec::new();
            let mut buffer = [0; 4096];
            loop {
                let count = stream.read(&mut buffer).unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&buffer[..count]);
                if let Some(end) = bytes.windows(4).position(|value| value == b"\r\n\r\n") {
                    let header = String::from_utf8_lossy(&bytes[..end]);
                    let size: usize = header
                        .lines()
                        .filter_map(|line| line.split_once(':'))
                        .find(|(key, _)| key.eq_ignore_ascii_case("content-length"))
                        .unwrap()
                        .1
                        .trim()
                        .parse()
                        .unwrap();
                    if bytes.len() >= end + 4 + size {
                        break;
                    }
                }
            }
            let body = serde_json::json!({"choices":[{"message":{"content":"[{\"index\":0,\"verdict\":\"keep\",\"class\":\"benign\",\"confidence\":1.0}]"}}]}).to_string();
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
        });
        let config = format!(
            "[trace_policy]\nsensitive_terms_files=[{:?}]\n[decision]\nentropy_backstop=false\n",
            list.to_str().unwrap()
        );
        let value = summary(
            run(
                dir.path(),
                &config,
                &[
                    "decide",
                    "--no-ner",
                    "--output-json",
                    "--endpoint",
                    &endpoint,
                ],
                text,
            ),
            0,
        );
        server.join().unwrap();
        assert_eq!(value[0]["verdict"], "redact");
        assert_eq!(value[0]["source"], "detector");
    }
}

use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Stdio},
};

fn run(dir: &Path, config: &str, args: &[&str], input: &str) -> std::process::Output {
    let path = dir.join("config.toml");
    fs::write(&path, config).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_nym"))
        .arg("--config")
        .arg(path)
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
fn summary(output: std::process::Output, status: i32) -> serde_json::Value {
    assert_eq!(
        output.status.code(),
        Some(status),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
#[test]
fn default_is_disabled_and_explicit_profile_has_counts_only() {
    let dir = tempfile::tempdir().unwrap();
    let config = "[ner]\nenabled=false\n";
    let plain = summary(
        run(
            dir.path(),
            config,
            &["detect", "--summary-json"],
            "buffer 4096",
        ),
        0,
    );
    assert_eq!(
        plain,
        serde_json::json!({"total":0,"by_pattern":{},"by_category":{}})
    );
    let trace = summary(
        run(
            dir.path(),
            config,
            &["detect", "--profile", "agent-trace", "--summary-json"],
            "buffer 4096",
        ),
        0,
    );
    assert_eq!(
        trace["trace_policy"],
        serde_json::json!({"ner_candidates":0,"suppressed_public_urls":0,"suppressed_technical_values":0,"suppressed_benign_terms":0,"sensitive_term_matches":0})
    );
    let disabled = summary(
        run(
            dir.path(),
            "[trace_policy]\nprofile='agent_trace'\n",
            &["detect", "--profile", "default", "--summary-json"],
            "buffer 4096",
        ),
        0,
    );
    assert_eq!(disabled, plain);
}
#[test]
fn sensitive_literals_win_over_benign_and_preserve_unicode_boundaries() {
    let dir = tempfile::tempdir().unwrap();
    let sensitive = dir.path().join("sensitive.txt");
    let benign = dir.path().join("benign.txt");
    fs::write(&sensitive, "\nÉtoile\n東京\na+b\nÉtoile\n").unwrap();
    fs::write(&benign, "Étoile\n東京\na+b\n").unwrap();
    let mut args = vec![
        "detect",
        "--profile",
        "agent-trace",
        "--summary-json",
        "--sensitive-terms-file",
        sensitive.to_str().unwrap(),
        "--benign-terms-file",
        benign.to_str().unwrap(),
    ];
    let input = "Étoile étoile xÉtoile 東京 x東京 a+b xa+by";
    let word = summary(run(dir.path(), "", &args, input), 0);
    assert_eq!(word["trace_policy"]["sensitive_term_matches"], 3);
    assert_eq!(word["by_pattern"]["sensitive_term"], 3);
    args.extend([
        "--term-boundary",
        "substring",
        "--term-case-sensitive",
        "false",
        "--fail-on",
        "sensitive_term",
    ]);
    let substring = summary(run(dir.path(), "", &args, input), 2);
    assert_eq!(substring["trace_policy"]["sensitive_term_matches"], 7);
    let manifest = substring.to_string();
    for value in ["Étoile", "東京", "a+b", "sensitive.txt", "benign.txt"] {
        assert!(!manifest.contains(value));
    }
}
#[test]
fn configured_files_expand_paths_and_cli_replaces_configured_lists_and_settings() {
    let dir = tempfile::tempdir().unwrap();
    let configured = dir.path().join("configured.txt");
    let override_file = dir.path().join("override.txt");
    fs::write(&configured, "ConfiguredTerm\n").unwrap();
    fs::write(&override_file, "OverrideTerm\n").unwrap();
    let config = format!(
        "[trace_policy]\nprofile='agent_trace'\nsensitive_terms_files=[{:?}]\ncase_sensitive=false\nterm_boundary='substring'\npublic_hosts=['docs.example.invalid']\n",
        configured.to_str().unwrap()
    );
    let input = "xconfiguredtermY OverrideTerm";
    let from_config = summary(
        run(dir.path(), &config, &["detect", "--summary-json"], input),
        0,
    );
    assert_eq!(from_config["trace_policy"]["sensitive_term_matches"], 1);
    let from_cli = summary(
        run(
            dir.path(),
            &config,
            &[
                "detect",
                "--summary-json",
                "--sensitive-terms-file",
                override_file.to_str().unwrap(),
                "--term-boundary",
                "word",
                "--term-case-sensitive",
                "true",
                "--public-host",
                "reference.example.invalid",
            ],
            input,
        ),
        0,
    );
    assert_eq!(from_cli["trace_policy"]["sensitive_term_matches"], 1);
    let detection = run(
        dir.path(),
        &config,
        &[
            "detect",
            "--json",
            "--sensitive-terms-file",
            override_file.to_str().unwrap(),
        ],
        input,
    );
    let matches = summary(detection, 0);
    assert_eq!(matches[0]["matched_text"], "OverrideTerm");
    // Expansion is applied to list paths, not their literal contents.
    let config = "[trace_policy]\nsensitive_terms_files=['$NYM_TEST_LITERAL_LIST']\n";
    let path = dir.path().join("config.toml");
    fs::write(&path, config).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_nym"))
        .env("NYM_TEST_LITERAL_LIST", &configured)
        .arg("--config")
        .arg(path)
        .args(["detect", "--summary-json"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"ConfiguredTerm")
        .unwrap();
    assert_eq!(summary(child.wait_with_output().unwrap(), 0)["total"], 1);
}
#[test]
fn structured_values_and_jsonl_aggregate_without_scanning_keys() {
    let dir = tempfile::tempdir().unwrap();
    let list = dir.path().join("terms.txt");
    fs::write(&list, "PrivateTerm\n").unwrap();
    let config = format!(
        "[trace_policy]\nsensitive_terms_files=[{:?}]\n",
        list.to_str().unwrap()
    );
    for (format, input, count) in [
        ("text", "PrivateTerm", 1),
        (
            "json",
            r#"{"PrivateTerm":"ordinary","unknown":{"nested":"PrivateTerm"},"metadata":"PrivateTerm"}"#,
            2,
        ),
        (
            "jsonl",
            "{\"PrivateTerm\":\"ordinary\",\"value\":\"PrivateTerm\"}\n{\"unknown\":\"PrivateTerm\"}\n",
            2,
        ),
    ] {
        let value = summary(
            run(
                dir.path(),
                &config,
                &["detect", "--format", format, "--summary-json"],
                input,
            ),
            0,
        );
        assert_eq!(value["total"], count);
        assert_eq!(value["trace_policy"]["sensitive_term_matches"], count);
        let output = run(
            dir.path(),
            &config,
            &["anon", "--format", format, "--strategy", "placeholder"],
            input,
        );
        assert_eq!(output.status.code(), Some(0));
        let output = String::from_utf8(output.stdout).unwrap();
        if format == "text" {
            assert!(!output.contains("PrivateTerm"));
        } else {
            assert!(output.contains("\"PrivateTerm\""));
            assert!(!output.contains(": \"PrivateTerm\""));
        }
    }
}
#[test]
fn regex_credentials_cannot_be_benign_allowlisted_in_code_or_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let list = dir.path().join("benign.txt");
    fs::write(&list, "AKIAIOSFODNN7EXAMPLE\n").unwrap();
    let config = format!(
        "[trace_policy]\nprofile='agent_trace'\nbenign_terms_files=[{:?}]\n",
        list.to_str().unwrap()
    );
    for (format, input) in [
        (
            "text",
            "const key = 'AKIAIOSFODNN7EXAMPLE'; // not a credential",
        ),
        ("json", r#"{"metadata":"AKIAIOSFODNN7EXAMPLE"}"#),
    ] {
        let value = summary(
            run(
                dir.path(),
                &config,
                &[
                    "detect",
                    "--format",
                    format,
                    "--summary-json",
                    "--fail-on",
                    "authentication",
                ],
                input,
            ),
            2,
        );
        assert!(value["total"].as_u64().unwrap() > 0);
        assert_eq!(value["trace_policy"]["suppressed_benign_terms"], 0);
    }
}
#[test]
fn office_detection_uses_the_same_value_free_trace_audit() {
    use zip::{ZipWriter, write::SimpleFileOptions};
    let dir = tempfile::tempdir().unwrap();
    let list = dir.path().join("terms.txt");
    fs::write(&list, "PrivateTerm\n").unwrap();
    let input = dir.path().join("input.docx");
    let mut archive = ZipWriter::new(std::io::Cursor::new(Vec::new()));
    archive
        .start_file("word/document.xml", SimpleFileOptions::default())
        .unwrap();
    archive
        .write_all(
            b"<w:document xmlns:w=\"x\"><w:p><w:r><w:t>PrivateTerm</w:t></w:r></w:p></w:document>",
        )
        .unwrap();
    archive
        .start_file("word/header1.xml", SimpleFileOptions::default())
        .unwrap();
    archive
        .write_all(
            b"<w:hdr xmlns:w=\"x\"><w:p><w:r><w:t>AKIAIOSFODNN7EXAMPLE</w:t></w:r></w:p></w:hdr>",
        )
        .unwrap();
    fs::write(&input, archive.finish().unwrap().into_inner()).unwrap();
    let value = summary(
        run(
            dir.path(),
            "",
            &[
                "detect",
                input.to_str().unwrap(),
                "--summary-json",
                "--sensitive-terms-file",
                list.to_str().unwrap(),
                "--fail-on",
                "authentication",
            ],
            "",
        ),
        2,
    );
    assert_eq!(value["trace_policy"]["sensitive_term_matches"], 1);
    assert!(value["total"].as_u64().unwrap() >= 2);
    for private in [
        "PrivateTerm",
        "AKIAIOSFODNN7EXAMPLE",
        "input.docx",
        "terms.txt",
    ] {
        assert!(!value.to_string().contains(private));
    }
}

#[test]
fn invalid_utf8_missing_files_and_hosts_fail_without_values_or_output() {
    let dir = tempfile::tempdir().unwrap();
    let list = dir.path().join("private-list-name.txt");
    fs::write(&list, [0xff, 0xfe]).unwrap();
    for args in [
        vec!["--sensitive-terms-file", list.to_str().unwrap()],
        vec!["--benign-terms-file", "/missing/private-path"],
        vec!["--public-host", "secret.invalid/path"],
    ] {
        let mut command = vec!["detect", "--summary-json", "--profile", "agent-trace"];
        command.extend(args);
        let output = run(dir.path(), "", &command, "privatePayload");
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        let error = String::from_utf8_lossy(&output.stderr);
        for value in [
            "private-list-name",
            "private-path",
            "secret.invalid",
            "privatePayload",
        ] {
            assert!(!error.contains(value));
        }
    }
}
#[cfg(feature = "streaming")]
#[test]
fn streamed_text_reports_safe_aggregate_counts() {
    let dir = tempfile::tempdir().unwrap();
    let list = dir.path().join("terms.txt");
    fs::write(&list, "PrivateTerm\n").unwrap();
    let output = run(
        dir.path(),
        "",
        &[
            "detect",
            "--stream",
            "--format",
            "text",
            "--sensitive-terms-file",
            list.to_str().unwrap(),
        ],
        "PrivateTerm\nPrivateTerm\n",
    );
    assert_eq!(output.status.code(), Some(0));
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("\"sensitive_term_matches\":2"));
    assert!(!stderr.contains("PrivateTerm"));
}
