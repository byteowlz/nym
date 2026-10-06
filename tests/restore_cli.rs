//! Strict restore publishes only after every original-input leaf verifies.
use std::{
    fs,
    io::Write,
    process::{Command, Stdio},
};

fn run(dir: &std::path::Path, args: &[&str], input: Option<&str>) -> std::process::Output {
    let config = dir.join("config.toml");
    fs::write(&config, "[ner]\nenabled=false\n").unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_nym"))
        .arg("--config")
        .arg(config)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
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
fn key(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("key.jsonl");
    fs::write(&path, concat!("{\"version\":\"1\"}\n", r#"{"original":"Original Person","replacement":"Alias Person","pattern_name":"person","components":[{"original":"Original","replacement":"Alias","component_type":"first_name"}]}"#)).unwrap();
    path
}
#[test]
fn strict_residuals_fail_closed_for_all_textual_destinations() {
    let dir = tempfile::tempdir().unwrap();
    let key = key(dir.path());
    let dest = dir.path().join("existing.txt");
    for (format, input) in [
        ("text", "Alias Person then ALIAS PERSON"),
        ("text", "notes in /tmp/Alias/project"),
        ("json", r#"{"safe":"Alias Person","later":"ALIAS PERSON"}"#),
        ("json", r#"{"Alias Person":"safe"}"#),
        ("json", r#"{"note":"\u0041lias"}"#),
        (
            "jsonl",
            "{\"ok\":\"Alias Person\"}\n{\"later\":\"ALIAS PERSON\"}\n",
        ),
        (
            "jsonl",
            "{\"ok\":\"Alias Person\"}\n{\"later\": privatePayload}\n",
        ),
        ("json", r#"{"later": privatePayload}"#),
    ] {
        for file_output in [false, true] {
            fs::write(&dest, b"preserved destination").unwrap();
            let mut args = vec!["deanon", "-k", key.to_str().unwrap(), "--format", format];
            if file_output {
                args.extend(["-o", dest.to_str().unwrap()]);
            }
            let output = run(dir.path(), &args, Some(input));
            assert_eq!(output.status.code(), Some(1), "{format}: {input}");
            assert!(output.stdout.is_empty());
            assert_eq!(fs::read(&dest).unwrap(), b"preserved destination");
            let error = String::from_utf8_lossy(&output.stderr);
            for value in [
                "Alias Person",
                "ALIAS PERSON",
                "Original Person",
                "privatePayload",
            ] {
                assert!(!error.contains(value));
            }
        }
    }
}
#[test]
fn originals_are_never_rescanned_and_short_unicode_aliases_are_strict() {
    let dir = tempfile::tempdir().unwrap();
    let key = dir.path().join("key.jsonl");
    fs::write(&key, concat!("{\"version\":\"1\"}\n", r#"{"original":"東京 inside original","replacement":"Alias Person","pattern_name":"person","components":[]}"#, "\n", r#"{"original":"restored","replacement":"東京","pattern_name":"person","components":[]}"#)).unwrap();
    let output = run(
        dir.path(),
        &["deanon", "-k", key.to_str().unwrap(), "--format", "text"],
        Some("Alias Person 東京"),
    );
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "東京 inside original restored"
    );
    let output = run(
        dir.path(),
        &["deanon", "-k", key.to_str().unwrap(), "--format", "text"],
        Some("x東京"),
    );
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "x東京");
}
#[test]
fn read_and_publish_failures_preserve_destinations() {
    let dir = tempfile::tempdir().unwrap();
    let key = key(dir.path());
    let input = dir.path().join("invalid.txt");
    fs::write(&input, [0xff]).unwrap();
    let dest = dir.path().join("existing.txt");
    fs::write(&dest, b"preserved destination").unwrap();
    for file_output in [false, true] {
        let mut args = vec![
            "deanon",
            input.to_str().unwrap(),
            "-k",
            key.to_str().unwrap(),
        ];
        if file_output {
            args.extend(["-o", dest.to_str().unwrap()]);
        }
        let output = run(dir.path(), &args, None);
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        assert_eq!(fs::read(&dest).unwrap(), b"preserved destination");
    }
    let directory_dest = dir.path().join("output-directory");
    fs::create_dir(&directory_dest).unwrap();
    let sentinel = directory_dest.join("sentinel");
    fs::write(&sentinel, b"directory preserved").unwrap();
    let output = run(
        dir.path(),
        &[
            "deanon",
            "-k",
            key.to_str().unwrap(),
            "-o",
            directory_dest.to_str().unwrap(),
        ],
        Some("Alias Person"),
    );
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert_eq!(fs::read(&sentinel).unwrap(), b"directory preserved");
}

#[test]
fn legacy_component_rewriting_requires_explicit_optout() {
    let dir = tempfile::tempdir().unwrap();
    let key = key(dir.path());
    let args = ["deanon", "-k", key.to_str().unwrap(), "--format", "text"];
    assert_eq!(run(dir.path(), &args, Some("Alias")).status.code(), Some(1));
    let mut legacy = args.to_vec();
    legacy.push("--no-verify-restore");
    let output = run(dir.path(), &legacy, Some("Alias"));
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "Original");
}
#[test]
fn office_later_residual_or_archive_failure_preserves_destination() {
    use zip::{ZipWriter, write::SimpleFileOptions};
    let dir = tempfile::tempdir().unwrap();
    let key = key(dir.path());
    let input = dir.path().join("input.docx");
    let dest = dir.path().join("output.docx");
    for malformed in [false, true] {
        let mut archive = ZipWriter::new(std::io::Cursor::new(Vec::new()));
        archive
            .start_file("word/document.xml", SimpleFileOptions::default())
            .unwrap();
        archive.write_all(b"<w:document xmlns:w=\"x\"><w:p><w:r><w:t>Alias Person</w:t></w:r></w:p></w:document>").unwrap();
        archive
            .start_file("word/header1.xml", SimpleFileOptions::default())
            .unwrap();
        archive
            .write_all(if malformed {
                b"<w:hdr><w:t>"
            } else {
                b"<w:hdr xmlns:w=\"x\"><w:t>ALIAS PERSON</w:t></w:hdr>"
            })
            .unwrap();
        fs::write(&input, archive.finish().unwrap().into_inner()).unwrap();
        fs::write(&dest, b"existing office destination").unwrap();
        let derived = dir.path().join("input.restored.docx");
        fs::write(&derived, b"existing derived destination").unwrap();
        for explicit_output in [false, true] {
            let mut args = vec![
                "deanon",
                input.to_str().unwrap(),
                "-k",
                key.to_str().unwrap(),
            ];
            if explicit_output {
                args.extend(["-o", dest.to_str().unwrap()]);
            }
            let output = run(dir.path(), &args, None);
            assert_eq!(output.status.code(), Some(1));
            assert!(output.stdout.is_empty());
            assert_eq!(fs::read(&dest).unwrap(), b"existing office destination");
            assert_eq!(fs::read(&derived).unwrap(), b"existing derived destination");
        }
    }
}
