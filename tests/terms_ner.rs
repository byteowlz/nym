use std::{fs, process::Command};
use tempfile::TempDir;

#[test]
fn optional_ner_priority_failure_preserves_destination_before_source_scan() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let model = root.join("empty-model");
    fs::create_dir(&model).unwrap();
    let input = root.join("input.jsonl");
    fs::write(&input, "invalid JSON with SyntheticPrivateValue").unwrap();
    let output = root.join("discovery.json");
    fs::write(&output, "preserve original destination").unwrap();
    let config = root.join("config.toml");
    fs::write(&config, "").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_nym"))
        .args([
            "--config",
            config.to_str().unwrap(),
            "--json",
            "terms",
            "discover",
            input.to_str().unwrap(),
            "--ner-rank",
            "--ner-rank-model",
            model.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--force",
            "--progress",
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(result.stdout.is_empty());
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(
        stderr.contains("NER")
            && !stderr.contains("SyntheticPrivateValue")
            && !stderr.contains("Discovery:")
            && !stderr.contains(input.to_str().unwrap())
            && !stderr.contains(model.to_str().unwrap())
    );
    assert_eq!(
        fs::read_to_string(output).unwrap(),
        "preserve original destination"
    );
    assert_eq!(
        fs::read_to_string(input).unwrap(),
        "invalid JSON with SyntheticPrivateValue"
    );
}

#[test]
fn model_override_requires_explicit_ranking_flag() {
    let result = Command::new(env!("CARGO_BIN_EXE_nym"))
        .args([
            "terms",
            "discover",
            "--ner-rank-model",
            "unused",
            "--output",
            "unused",
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("--ner-rank"));
}
