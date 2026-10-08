use std::{
    fs,
    path::Path,
    process::{Command, Output},
};
use tempfile::TempDir;

fn run(root: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_nym"))
        .args([
            "--config",
            root.join("config.toml").to_str().unwrap(),
            "terms",
            "discover",
        ])
        .args(args)
        .output()
        .unwrap()
}
mod support;
use support::discovery_value as read_discovery;

#[test]
fn exact_overflow_reproduces_launcher_error_and_bounded_mode_recounts() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    fs::write(root.join("config.toml"), "").unwrap();
    let input = root.join("input.txt");
    let output = root.join("out.json");
    let mut lines = (0..200)
        .map(|i| format!("uniqueToken{i}\n"))
        .collect::<String>();
    // Include a late heavy hitter plus Unicode and repeated identical text copies.
    for i in 0..100 {
        lines.push_str(&format!("LateHot Ζέτα payload{i}\n"));
    }
    lines.push_str("LateHot Ζέτα payload0\n");
    fs::write(&input, lines).unwrap();
    fs::write(&output, "preserve").unwrap();
    let base = [
        input.to_str().unwrap(),
        "--max-distinct",
        "20",
        "--limit",
        "2",
        "--phrase-words",
        "1",
        "--output",
        output.to_str().unwrap(),
        "--force",
    ];
    let failed = run(root, &base);
    assert_eq!(failed.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&failed.stderr).contains("distinct vocabulary limit exceeded"));
    assert_eq!(fs::read_to_string(&output).unwrap(), "preserve");
    let mut bounded = base.to_vec();
    bounded.push("--bounded");
    let succeeded = run(root, &bounded);
    assert!(
        succeeded.status.success(),
        "{}",
        String::from_utf8_lossy(&succeeded.stderr)
    );
    let discovery = read_discovery(&output);
    assert_eq!(discovery["selection"]["method"], "space_saving_occurrences");
    let candidates = discovery["candidates"].as_array().unwrap();
    assert_eq!(
        candidates
            .iter()
            .map(|v| (
                v["term"].as_str().unwrap(),
                v["occurrences"].as_u64().unwrap(),
                v["distinct_texts"].as_u64().unwrap(),
                v["sources"].as_u64().unwrap()
            ))
            .collect::<Vec<_>>(),
        vec![("LateHot", 101, 100, 1), ("Ζέτα", 101, 100, 1)]
    );
    // Differential proof: every selected full candidate matches unbounded exact mode.
    let exact = root.join("exact.json");
    assert!(
        run(
            root,
            &[
                input.to_str().unwrap(),
                "--max-distinct",
                "1000",
                "--limit",
                "2",
                "--phrase-words",
                "1",
                "--output",
                exact.to_str().unwrap()
            ]
        )
        .status
        .success()
    );
    assert_eq!(
        discovery["candidates"],
        read_discovery(&exact)["candidates"]
    );
    assert_eq!(
        discovery["corpus_sha256"],
        read_discovery(&exact)["corpus_sha256"]
    );
}

#[test]
fn unsupported_replay_modes_and_malformed_children_never_publish() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    fs::write(root.join("config.toml"), "").unwrap();
    let input = root.join("input.jsonl");
    fs::write(&input, "{\"text\":\"LateHot\"}\nmalformed\n").unwrap();
    let background = root.join("bg.json");
    fs::write(&background, "{\"public\":100}").unwrap();
    let output = root.join("out.json");
    fs::write(&output, "preserve").unwrap();
    for mut args in [
        vec!["--bounded"],
        vec![
            input.to_str().unwrap(),
            "--bounded",
            "--background",
            background.to_str().unwrap(),
        ],
        vec![input.to_str().unwrap(), "--bounded"],
    ] {
        args.extend(["--output", output.to_str().unwrap(), "--force"]);
        assert_eq!(run(root, &args).status.code(), Some(1));
        assert_eq!(fs::read_to_string(&output).unwrap(), "preserve");
    }
}
