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
fn progress_is_count_only_on_stderr_and_does_not_corrupt_json_or_quiet() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    fs::write(root.join("config.toml"), "").unwrap();
    let input = root.join("input.jsonl");
    fs::write(
        &input,
        "{\"text\":\"PrivateMarker\"}\n{\"text\":\"PrivateMarker\"}\n",
    )
    .unwrap();
    let output = root.join("out.json");
    let base = [
        input.to_str().unwrap(),
        "--bounded",
        "--json",
        "--output",
        output.to_str().unwrap(),
        "--force",
    ];
    let mut args = base.to_vec();
    args.push("--progress");
    let result = run(root, &args);
    assert!(result.status.success());
    let summary: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(summary["units"], 2);
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert_eq!(stderr.lines().count(), 4);
    for stage in ["1/2 Proposal", "2/2 Exact recount"] {
        assert!(stderr.contains(&format!("{stage}: 1/1 files, 2 text units")));
    }
    assert!(!stderr.contains("PrivateMarker"));
    assert!(!stderr.contains(input.to_str().unwrap()));
    let mut quiet = args.clone();
    quiet.push("--quiet");
    let result = run(root, &quiet);
    assert!(result.status.success());
    assert!(result.stdout.is_empty());
    assert!(result.stderr.is_empty());
    let mut disabled = base.to_vec();
    disabled.push("--no-progress");
    let result = run(root, &disabled);
    assert!(result.status.success());
    assert!(result.stderr.is_empty());
    fs::write(&input, "malformed\n").unwrap();
    let result = run(root, &args);
    assert_eq!(result.status.code(), Some(1));
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(stderr.contains("(incomplete)"));
    assert!(!stderr.contains("2/2 Exact recount"));
}

#[test]
fn parser_accepted_deep_json_is_scanned_not_skipped_and_excess_depth_fails() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    fs::write(root.join("config.toml"), "").unwrap();
    let input = root.join("deep.jsonl");
    let output = root.join("out.json");
    let mut value = serde_json::json!({"text":"Jane Doë"});
    for _ in 0..80 {
        value = serde_json::json!({"nest":value});
    }
    fs::write(
        &input,
        format!("{}\n", serde_json::to_string(&value).unwrap()),
    )
    .unwrap();
    let args = [
        input.to_str().unwrap(),
        "--bounded",
        "--include",
        "**.text",
        "--min-count",
        "1",
        "--phrase-words",
        "1",
        "--output",
        output.to_str().unwrap(),
        "--force",
    ];
    let result = run(root, &args);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let artifact = read_discovery(&output);
    assert_eq!(
        artifact["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| (
                c["term"].as_str().unwrap(),
                c["occurrences"].as_u64().unwrap(),
                c["examples"][0]["start"].as_u64().unwrap(),
                c["examples"][0]["end"].as_u64().unwrap()
            ))
            .collect::<Vec<_>>(),
        vec![("Doë", 1, 5, 9), ("Jane", 1, 0, 4)]
    );
    let original = fs::read(&output).unwrap();
    let too_deep = format!(
        "{}\"Jane Doë\"{}\n",
        "{\"nest\":".repeat(140),
        "}".repeat(140)
    );
    fs::write(&input, too_deep).unwrap();
    let result = run(root, &args);
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(fs::read(&output).unwrap(), original);
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
