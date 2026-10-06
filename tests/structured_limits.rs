//! Record-size, replay and document-route regressions, without model downloads.
use serde_json::{Value, json};
use std::{
    fs,
    io::{Cursor, Write},
    process::{Command, Stdio},
};
fn command(dir: &std::path::Path) -> Command {
    let config = dir.join("config.toml");
    fs::write(&config, "[ner]\nenabled = false\n").unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nym"));
    cmd.arg("--config")
        .arg(config)
        .env("HF_HUB_OFFLINE", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    cmd
}
#[test]
fn large_jsonl_aggregates_every_record_and_retains_first_sniffed_record() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("extensionless");
    let mut output = fs::File::create(&file).unwrap();
    writeln!(output, "{{\"email\":\"first@example.com\"}}").unwrap();
    for _ in 0..25_000 {
        writeln!(
            output,
            "{{\"email\":\"next@example.com\",\"number\":4096,\"tool\":\"bash\"}}"
        )
        .unwrap();
    }
    drop(output);
    let out = command(dir.path())
        .args([
            "detect",
            file.to_str().unwrap(),
            "--no-ner",
            "--patterns",
            "email",
            "--summary-json",
            "--fail-on",
            "email",
        ])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(
        serde_json::from_slice::<Value>(&out.stdout).unwrap(),
        json!({"total":25001,"by_pattern":{"email":25001},"by_category":{"contact":25001},"blockers":["email"]})
    );
}
#[test]
fn oversized_record_fails_before_publication() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("large.jsonl");
    let dest = dir.path().join("dest");
    fs::write(
        &input,
        format!("{{\"text\":\"{}\"}}", "x".repeat(16 * 1024 * 1024)),
    )
    .unwrap();
    fs::write(&dest, "keep").unwrap();
    let out = command(dir.path())
        .args([
            "anon",
            input.to_str().unwrap(),
            "--no-ner",
            "-o",
            dest.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("16 MiB record limit"));
    assert_eq!(fs::read_to_string(&dest).unwrap(), "keep");
    assert!(out.stdout.is_empty());
}
#[test]
fn explicit_jsonl_decodes_escapes_and_json_override_remains_single_document() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trace.jsonl");
    fs::write(
        &path,
        "{\"escaped\":\"a\\u006cice@example.com\",\"key@example.com\":4096}\n{\"other\":true}\n",
    )
    .unwrap();
    let out = command(dir.path())
        .args([
            "anon",
            path.to_str().unwrap(),
            "--no-ner",
            "--format",
            "jsonl",
            "--strategy",
            "placeholder",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let values: Vec<Value> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(
        values,
        vec![
            json!({"escaped":"<EMAIL>","key@example.com":4096}),
            json!({"other":true})
        ]
    );
    let out = command(dir.path())
        .args([
            "detect",
            path.to_str().unwrap(),
            "--no-ner",
            "--format",
            "json",
            "--summary-json",
        ])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    #[cfg(feature = "streaming")]
    {
        let out = command(dir.path())
            .args([
                "anon",
                path.to_str().unwrap(),
                "--no-ner",
                "--format",
                "json",
                "--stream",
            ])
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(1));
        assert!(out.stdout.is_empty());
    }
}
#[test]
fn uppercase_json_bom_and_array_are_structured() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("input.JSON");
    fs::write(
        &path,
        "\u{feff}  \n[\n{\"email\":\"alice@example.com\",\"count\":4096}\n]\n",
    )
    .unwrap();
    let out = command(dir.path())
        .args([
            "anon",
            path.to_str().unwrap(),
            "--no-ner",
            "--strategy",
            "placeholder",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&out.stdout).unwrap(),
        json!([{"email":"<EMAIL>","count":4096}])
    );
}
#[test]
fn legacy_structured_restoration_decodes_aliases_and_reescapes_originals() {
    let dir = tempfile::tempdir().unwrap();
    let key = dir.path().join("keys.jsonl");
    let input = dir.path().join("input.jsonl");
    let alias_path = r"C:\Users\Synthetic\alias.txt";
    let original_path = r"C:\Users\Fictional\report.txt";
    let alias_name = "Alias \"Quoted\"\nName";
    let original_name = "Original \"Value\"\nJörg";
    let mappings = [
        json!({"version":"1"}),
        json!({"original":original_path,"replacement":alias_path,"pattern_name":"unix_path"}),
        json!({"original":original_name,"replacement":alias_name,"pattern_name":"person"}),
    ];
    fs::write(
        &key,
        mappings
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    let rows = [
        json!({alias_path: alias_path,"counter":4096,"active":true}),
        json!({"name":alias_name,"data":null}),
    ];
    fs::write(
        &input,
        rows.iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    let out = command(dir.path())
        .args([
            "deanon",
            input.to_str().unwrap(),
            "-k",
            key.to_str().unwrap(),
            "--no-verify-restore",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let actual: Vec<Value> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(
        actual,
        vec![
            json!({alias_path:original_path,"counter":4096,"active":true}),
            json!({"name":original_name,"data":null})
        ]
    );
    let destination = dir.path().join("restored.jsonl");
    fs::write(&destination, "keep existing output").unwrap();
    fs::write(&input, format!("{}\n{{bad-private-marker\n", rows[0])).unwrap();
    let out = command(dir.path())
        .args([
            "deanon",
            input.to_str().unwrap(),
            "-k",
            key.to_str().unwrap(),
            "--no-verify-restore",
            "-o",
            destination.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    assert_eq!(
        fs::read_to_string(destination).unwrap(),
        "keep existing output"
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("line 2"));
    assert!(!String::from_utf8_lossy(&out.stderr).contains("bad-private-marker"));
}

#[test]
fn office_route_remains_binary_and_case_insensitive() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("document.DOCX");
    let dest = dir.path().join("anon.docx");
    let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
    zip.start_file(
        "word/document.xml",
        zip::write::SimpleFileOptions::default(),
    )
    .unwrap();
    zip.write_all(b"<w:document xmlns:w=\"urn:word\"><w:body><w:p><w:r><w:t>alice@example.com</w:t></w:r></w:p></w:body></w:document>").unwrap();
    fs::write(&path, zip.finish().unwrap().into_inner()).unwrap();
    let out = command(dir.path())
        .args([
            "detect",
            path.to_str().unwrap(),
            "--no-ner",
            "--fail-on",
            "email",
            "--summary-json",
        ])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    let out = command(dir.path())
        .args([
            "anon",
            path.to_str().unwrap(),
            "--no-ner",
            "--strategy",
            "placeholder",
            "-o",
            dest.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = command(dir.path())
        .args([
            "detect",
            dest.to_str().unwrap(),
            "--no-ner",
            "--summary-json",
        ])
        .output()
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&out.stdout).unwrap()["total"],
        0
    );
}
