//! Synthetic-only native recursive input and failure-safe publication proofs.
use serde_json::{Value, json};
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};
use tempfile::TempDir;

fn run(dir: &TempDir, args: &[&str]) -> Output {
    let config = dir.path().join("config.toml");
    fs::write(&config, "").unwrap();
    Command::new(env!("CARGO_BIN_EXE_nym"))
        .args([
            "--config",
            config.to_str().unwrap(),
            "--json",
            "terms",
            "discover",
        ])
        .args(args)
        .output()
        .unwrap()
}

fn text(path: &Path) -> &str {
    path.to_str().unwrap()
}
fn artifact(path: &Path) -> Value {
    let value: Value = serde_json::from_reader(fs::File::open(path).unwrap()).unwrap();
    assert_eq!(value["schema"], json!("nym.terms.discovery.v1"));
    value
}

#[test]
fn recursive_matches_explicit_files_including_hidden_unicode_and_case_extensions() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("corpus");
    fs::create_dir_all(root.join("sub/.hidden")).unwrap();
    let a = root.join("a.txt");
    let b = root.join("sub/file with spaces.JSONL");
    let c = root.join("sub/.hidden/é.ndjson");
    fs::write(&a, "AlphaCase\n").unwrap();
    fs::write(&b, "{\"text\":\"BetaCase\"}\n").unwrap();
    fs::write(&c, "{\"text\":\"ΓάμμαCase\"}\n").unwrap();
    fs::write(root.join("ignored.bin"), [0xff]).unwrap();
    let recursive = dir.path().join("recursive.json");
    let explicit = dir.path().join("explicit.json");
    let result = run(
        &dir,
        &[
            text(&root),
            "--recursive",
            "--min-count",
            "1",
            "--output",
            text(&recursive),
        ],
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&result.stdout).unwrap(),
        json!({
            "action":"discover","candidates":3,"approved":0,"units":3,"pending":3,
            "files":3,"skipped_extensions":1,"skipped_symlinks":0,"skipped_artifacts":0
        })
    );
    assert!(
        run(
            &dir,
            &[
                text(&c),
                text(&a),
                text(&b),
                "--min-count",
                "1",
                "--output",
                text(&explicit)
            ]
        )
        .status
        .success()
    );
    assert_eq!(artifact(&recursive), artifact(&explicit));
}

#[test]
fn overlapping_roots_and_explicit_duplicates_are_deduplicated_and_order_independent() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("corpus");
    let sub = root.join("sub");
    fs::create_dir_all(&sub).unwrap();
    let file = sub.join("a.txt");
    fs::write(&file, "AlphaCase\n").unwrap();
    let a = dir.path().join("a.json");
    let b = dir.path().join("b.json");
    assert!(
        run(
            &dir,
            &[
                text(&root),
                text(&sub),
                text(&file),
                "--recursive",
                "--min-count",
                "1",
                "--output",
                text(&a)
            ]
        )
        .status
        .success()
    );
    assert!(
        run(
            &dir,
            &[
                text(&file),
                text(&sub),
                text(&root),
                "--recursive",
                "--min-count",
                "1",
                "--output",
                text(&b)
            ]
        )
        .status
        .success()
    );
    assert_eq!(artifact(&a), artifact(&b));
    assert_eq!(artifact(&a)["unit_count"], json!(1));
}

#[test]
fn directory_extension_filters_and_json_selectors_share_existing_parser() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("corpus");
    fs::create_dir(&root).unwrap();
    fs::write(
        root.join("a.JSONL"),
        "{\"text\":\"AlphaCase\",\"id\":\"PrivateMarker\"}\n",
    )
    .unwrap();
    fs::write(root.join("b.NDJSON"), "{\"text\":\"BetaCase\"}\n").unwrap();
    fs::write(root.join("ignored.txt"), "not JSON\n").unwrap();
    let output = dir.path().join("out.json");
    assert!(
        run(
            &dir,
            &[
                text(&root),
                "-r",
                "--extension",
                ".JSONL,ndjson",
                "--format",
                "jsonl",
                "--include",
                "**.text",
                "--min-count",
                "1",
                "--output",
                text(&output)
            ]
        )
        .status
        .success()
    );
    assert_eq!(
        artifact(&output)["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v["term"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["AlphaCase", "BetaCase"]
    );
}

#[test]
fn output_background_and_explicit_artifacts_are_excluded_before_extraction() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("corpus");
    fs::create_dir(&root).unwrap();
    fs::write(root.join("a.txt"), "AlphaCase\n").unwrap();
    let output = root.join("out.json");
    let background = root.join("reference.json");
    let review = root.join("review.json");
    fs::write(&output, "malformed old artifact").unwrap();
    fs::write(&background, "{\"common\":1000}").unwrap();
    fs::write(&review, "malformed unrelated review").unwrap();
    let result = run(
        &dir,
        &[
            text(&root),
            "-r",
            "--background",
            text(&background),
            "--exclude-file",
            text(&review),
            "--min-count",
            "1",
            "--force",
            "--output",
            text(&output),
        ],
    );
    assert!(result.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&result.stdout).unwrap(),
        json!({
            "action":"discover","candidates":1,"approved":0,"units":1,"pending":1,
            "files":1,"skipped_extensions":0,"skipped_symlinks":0,"skipped_artifacts":3
        })
    );
}

#[test]
fn limits_missing_empty_roots_and_malformed_children_preserve_destination() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("corpus");
    fs::create_dir(&root).unwrap();
    let output = dir.path().join("out.json");
    fs::write(&output, "preserve").unwrap();
    for args in [
        vec![text(&root)],
        vec![text(&root), "-r"],
        vec!["--recursive"],
        vec![text(&root), "-r", "--extension", "../jsonl"],
        vec![text(&root), "-r", "--max-files", "0"],
        vec![text(&root), "-r", "--max-files", "10001"],
    ] {
        let mut args = args;
        args.extend(["--force", "--output", text(&output)]);
        assert_eq!(run(&dir, &args).status.code(), Some(1));
        assert_eq!(fs::read_to_string(&output).unwrap(), "preserve");
    }
    let missing = root.join("missing");
    assert_eq!(
        run(
            &dir,
            &[text(&missing), "-r", "--force", "--output", text(&output)]
        )
        .status
        .code(),
        Some(1)
    );
    fs::write(root.join("a.jsonl"), "{\"text\":\"AlphaCase\"}\n").unwrap();
    fs::write(root.join("b.jsonl"), "malformed\n").unwrap();
    for extra in [vec![], vec!["--max-files", "1"]] {
        let mut args = vec![text(&root), "-r", "--force", "--output", text(&output)];
        args.extend(extra);
        assert_eq!(run(&dir, &args).status.code(), Some(1));
        assert_eq!(fs::read_to_string(&output).unwrap(), "preserve");
    }
}

#[test]
fn directory_depth_budget_fails_before_publication() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("corpus");
    let deep = (0..66).fold(root.clone(), |p, _| p.join("d"));
    fs::create_dir_all(&deep).unwrap();
    fs::write(deep.join("a.txt"), "AlphaCase\n").unwrap();
    let output = dir.path().join("out.json");
    assert_eq!(
        run(&dir, &[text(&root), "-r", "--output", text(&output)])
            .status
            .code(),
        Some(1)
    );
    assert!(!output.exists());
}

#[cfg(unix)]
#[test]
fn symlink_cycles_and_file_links_are_not_followed_and_explicit_links_fail() {
    use std::os::unix::fs::symlink;
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("corpus");
    fs::create_dir(&root).unwrap();
    let file = root.join("a.txt");
    fs::write(&file, "AlphaCase\n").unwrap();
    let link = root.join("alias.txt");
    symlink(&file, &link).unwrap();
    symlink(&root, root.join("loop")).unwrap();
    let output = dir.path().join("out.json");
    let result = run(
        &dir,
        &[
            text(&root),
            "-r",
            "--min-count",
            "1",
            "--output",
            text(&output),
        ],
    );
    assert!(result.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&result.stdout).unwrap(),
        json!({
            "action":"discover","candidates":1,"approved":0,"units":1,"pending":1,
            "files":1,"skipped_extensions":0,"skipped_symlinks":2,"skipped_artifacts":0
        })
    );
    let old = fs::read(&output).unwrap();
    assert_eq!(
        run(&dir, &[text(&link), "--force", "--output", text(&output)])
            .status
            .code(),
        Some(1)
    );
    assert_eq!(fs::read(&output).unwrap(), old);
}

#[cfg(unix)]
#[test]
fn unreadable_selected_child_is_failure_not_clean_discovery() {
    use std::os::unix::fs::PermissionsExt;
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("corpus");
    fs::create_dir(&root).unwrap();
    let file = root.join("secret.txt");
    fs::write(&file, "AlphaCase\n").unwrap();
    fs::set_permissions(&file, fs::Permissions::from_mode(0o000)).unwrap();
    let output = dir.path().join("out.json");
    fs::write(&output, "preserve").unwrap();
    let result = run(
        &dir,
        &[text(&root), "-r", "--force", "--output", text(&output)],
    );
    fs::set_permissions(&file, fs::Permissions::from_mode(0o600)).unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(fs::read_to_string(output).unwrap(), "preserve");
}
