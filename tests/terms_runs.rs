//! Synthetic-only saved-run inventory and source-binding integration proofs.
use serde_json::Value;
use std::{fs, process::Output, thread, time::Duration};
use tempfile::TempDir;

#[path = "support/command.rs"]
mod cli_command;
#[path = "support/summary.rs"]
mod cli_summary;

use cli_command::{nym_command as command, run};
fn success(output: Output) -> Value {
    cli_summary::summary(output, 0)
}
fn fixture(home: &TempDir, name: &str) -> String {
    let directory = home.path().join(name);
    fs::create_dir_all(&directory).unwrap();
    let source = directory.join("input.txt");
    fs::write(
        &source,
        "SyntheticPrivateTerm Jane Doë project.\nSyntheticPrivateTerm Jane Doë again.\n",
    )
    .unwrap();
    let discovery = directory.join("discovery.json");
    let html = directory.join("review.html");
    assert!(
        run(
            home,
            &[
                "terms",
                "discover",
                source.to_str().unwrap(),
                "--output",
                discovery.to_str().unwrap()
            ]
        )
        .status
        .success()
    );
    assert!(
        run(
            home,
            &[
                "terms",
                "review",
                discovery.to_str().unwrap(),
                "--html",
                "--output",
                html.to_str().unwrap()
            ]
        )
        .status
        .success()
    );
    directory.to_str().unwrap().into()
}
fn register(home: &TempDir, path: &str) -> Value {
    success(run(home, &["--json", "terms", "runs", "register", path]))
}

#[test]
fn xdg_inventory_register_idempotence_prefix_and_machine_modes() {
    let home = TempDir::new().unwrap();
    let empty = success(run(&home, &["--json", "terms", "runs", "list"]));
    assert_eq!(empty, serde_json::json!([]));
    assert!(!home.path().join("state/nym/terms/runs").exists());
    let directory = fixture(&home, "custom output with spaces");
    let original = fs::read(format!("{directory}/review.html")).unwrap();
    let entry = register(&home, &directory);
    assert_eq!(entry["status"], "ready");
    assert_eq!(entry["registered"], true);
    assert!(entry["units"].as_u64().unwrap() > 0);
    assert_eq!(register(&home, &directory), entry);
    let id = entry["id"].as_str().unwrap();
    let file = home.path().join(format!("state/nym/terms/runs/{id}.json"));
    assert!(file.is_file());
    let stored = fs::read_to_string(&file).unwrap();
    assert!(!stored.contains("SyntheticPrivateTerm"));
    assert!(!stored.contains("token="));
    assert_eq!(
        success(run(&home, &["--json", "terms", "runs", "show", &id[..12]])),
        entry
    );
    assert_eq!(
        success(run(&home, &["--json", "terms", "runs", "show"])),
        entry
    );
    assert_eq!(
        success(run(&home, &["--json", "terms", "runs", "list"])),
        serde_json::json!([entry])
    );
    let yaml = run(&home, &["--yaml", "terms", "runs", "list"]);
    assert!(yaml.status.success());
    assert_eq!(
        serde_yaml::from_slice::<Value>(&yaml.stdout)
            .unwrap()
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let quiet = run(&home, &["--quiet", "--json", "terms", "runs", "list"]);
    assert!(quiet.status.success());
    assert!(quiet.stdout.is_empty());
    assert_eq!(
        fs::read(format!("{directory}/review.html")).unwrap(),
        original
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(file).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}

#[test]
fn legacy_runs_are_listed_without_artifact_reads_and_incomplete_latest_is_excluded() {
    let home = TempDir::new().unwrap();
    let good = fixture(&home, "state/nym/pi-terms/legacy");
    thread::sleep(Duration::from_millis(25));
    let incomplete = home.path().join("state/nym/pi-terms/new-incomplete");
    fs::create_dir(&incomplete).unwrap();
    fs::write(incomplete.join("review.html"), "not a valid page").unwrap();
    let rows = success(run(&home, &["--json", "terms", "runs", "list"]));
    assert_eq!(rows.as_array().unwrap().len(), 2);
    assert!(
        rows.as_array()
            .unwrap()
            .iter()
            .all(|r| r["registered"] == false && r.get("candidates").is_none())
    );
    let latest = success(run(&home, &["--json", "terms", "runs", "show", "latest"]));
    assert_eq!(
        latest["directory"],
        fs::canonicalize(good).unwrap().to_str().unwrap()
    );
    assert!(!home.path().join("state/nym/terms/runs").exists());
    // Inventory is metadata-only even for a corrupt artifact; use/registration must verify it.
    fs::write(
        latest["directory"].as_str().unwrap().to_owned() + "/discovery.json",
        "private invalid artifact",
    )
    .unwrap();
    assert!(
        run(&home, &["--json", "terms", "runs", "list"])
            .status
            .success()
    );
    let rejected = run(
        &home,
        &[
            "terms",
            "runs",
            "register",
            latest["directory"].as_str().unwrap(),
        ],
    );
    assert_eq!(rejected.status.code(), Some(1));
}

#[test]
fn stale_scope_atomic_preservation_force_and_missing_run() {
    let home = TempDir::new().unwrap();
    let directory = fixture(&home, "first");
    let entry = register(&home, &directory);
    let id = entry["id"].as_str().unwrap();
    let file = home.path().join(format!("state/nym/terms/runs/{id}.json"));
    let old = fs::read(&file).unwrap();
    let html = format!("{directory}/review.html");
    let original = fs::read(&html).unwrap();
    let mut changed = original.clone();
    changed.extend_from_slice(b"\n ");
    fs::write(&html, &changed).unwrap();
    let refusal = run(&home, &["terms", "runs", "serve", id]);
    assert_eq!(refusal.status.code(), Some(1));
    assert!(refusal.stdout.is_empty());
    assert_eq!(
        run(&home, &["terms", "runs", "register", &directory])
            .status
            .code(),
        Some(1)
    );
    assert_eq!(fs::read(&file).unwrap(), old);
    assert!(
        run(&home, &["terms", "runs", "register", &directory, "--force"])
            .status
            .success()
    );
    let other = fixture(&home, "second");
    fs::copy(format!("{other}/review.html"), &html).unwrap();
    // Identical synthetic source text can still differ in source identity/fingerprint.
    assert_eq!(
        run(&home, &["terms", "runs", "register", &directory, "--force"])
            .status
            .code(),
        Some(1)
    );
    fs::remove_dir_all(&directory).unwrap();
    let shown = success(run(&home, &["--json", "terms", "runs", "show", id]));
    assert_eq!(shown["status"], "missing");
}

#[test]
fn invalid_inventory_and_selector_fail_before_stdout_and_hosting() {
    let home = TempDir::new().unwrap();
    let directory = fixture(&home, "good");
    register(&home, &directory);
    for id in ["../outside", "0", "not-a-run"] {
        let result = run(&home, &["--json", "terms", "runs", "show", id]);
        assert_eq!(result.status.code(), Some(1));
        assert!(result.stdout.is_empty());
    }
    let registry = home.path().join("state/nym/terms/runs");
    fs::write(registry.join("corrupt.json"), "{}").unwrap();
    let result = run(&home, &["--json", "terms", "runs", "list"]);
    assert_eq!(result.status.code(), Some(1));
    assert!(result.stdout.is_empty());
    assert!(!String::from_utf8_lossy(&result.stderr).contains("SyntheticPrivateTerm"));
}

#[test]
fn configured_state_and_data_paths_override_xdg() {
    let home = TempDir::new().unwrap();
    fs::write(
        home.path().join("config.toml"),
        "[paths]\nstate_dir = '~/${RUN_TEST_STATE}'\ndata_dir = '~/configured-data'\n",
    )
    .unwrap();
    let directory = fixture(&home, "external");
    let output = command(&home)
        .env("RUN_TEST_STATE", "configured-state")
        .args(["--json", "terms", "runs", "register", &directory])
        .output()
        .unwrap();
    let entry = success(output);
    assert!(
        home.path()
            .join(format!(
                "configured-state/terms/runs/{}.json",
                entry["id"].as_str().unwrap()
            ))
            .exists()
    );
    assert!(!home.path().join("state/nym/terms/runs").exists());
}

#[cfg(unix)]
#[test]
fn symlink_runs_and_manifest_links_fail_closed() {
    use std::os::unix::fs::symlink;
    let home = TempDir::new().unwrap();
    let directory = fixture(&home, "external");
    let link = home.path().join("linked");
    symlink(&directory, &link).unwrap();
    assert_eq!(
        run(
            &home,
            &["terms", "runs", "register", link.to_str().unwrap()]
        )
        .status
        .code(),
        Some(1)
    );
    let entry = register(&home, &directory);
    let file = home.path().join(format!(
        "state/nym/terms/runs/{}.json",
        entry["id"].as_str().unwrap()
    ));
    let original = fs::read(&file).unwrap();
    fs::remove_file(&file).unwrap();
    let outside = home.path().join("outside.json");
    fs::write(&outside, original).unwrap();
    symlink(outside, file).unwrap();
    let output = run(&home, &["--json", "terms", "runs", "list"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
}

#[test]
fn concurrent_registration_keeps_one_manifest_and_all_distinct_runs() {
    let home = TempDir::new().unwrap();
    let first = fixture(&home, "parallel-first");
    let second = fixture(&home, "parallel-second");
    let mut children: Vec<_> = [&first, &first, &second]
        .into_iter()
        .map(|path| {
            command(&home)
                .args(["--quiet", "terms", "runs", "register", path])
                .spawn()
                .unwrap()
        })
        .collect();
    for child in &mut children {
        assert!(child.wait().unwrap().success());
    }
    assert_eq!(
        success(run(&home, &["--json", "terms", "runs", "list"]))
            .as_array()
            .unwrap()
            .len(),
        2
    );
}

#[cfg(unix)]
#[test]
fn embedded_host_assets_and_trusted_open_do_not_need_the_checkout() {
    use std::os::unix::fs::PermissionsExt;
    let home = TempDir::new().unwrap();
    let directory = fixture(&home, "owned");
    let html = format!("{directory}/review.html");
    let supplied = fs::read_to_string(&html).unwrap();
    fs::write(
        &html,
        format!("<script>window.untrusted=true</script>{supplied}"),
    )
    .unwrap();
    let entry = register(&home, &directory);
    let original = fs::read(format!("{directory}/review.html")).unwrap();
    let bin = home.path().join("bin");
    fs::create_dir(&bin).unwrap();
    let log = home.path().join("host-argv.txt");
    let fake = bin.join("uv");
    fs::write(&fake, "#!/usr/bin/env bash\nset -eu\nprintf '%s\\n' \"$@\" > \"$HOST_TEST_LOG\"\nprintf '%s\\n' '{\"status\":\"synthetic-only\"}'\n").unwrap();
    fs::set_permissions(&fake, fs::Permissions::from_mode(0o700)).unwrap();
    let path = format!("{}:{}", bin.display(), std::env::var("PATH").unwrap());
    let result = command(&home)
        .env("PATH", &path)
        .env("HOST_TEST_LOG", &log)
        .args([
            "--json",
            "terms",
            "runs",
            "serve",
            "latest",
            "--tailnet",
            "--port",
            "0",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let args = fs::read_to_string(&log).unwrap();
    assert!(args.contains("--tailnet\n"));
    assert!(args.contains("--expected-sha256\n"));
    let helper = args
        .lines()
        .find(|line| line.ends_with("serve_terms_review.py"))
        .unwrap();
    assert!(
        helper.starts_with(
            home.path()
                .join("data/nym/terms/review-host")
                .to_str()
                .unwrap()
        )
    );
    assert!(
        !fs::read_to_string(helper)
            .unwrap()
            .contains("SyntheticPrivateTerm")
    );
    fs::write(helper, "corrupted asset").unwrap();
    fs::remove_file(&log).unwrap();
    let refused = command(&home)
        .env("PATH", &path)
        .env("HOST_TEST_LOG", &log)
        .args(["terms", "runs", "serve"])
        .output()
        .unwrap();
    assert_eq!(refused.status.code(), Some(1));
    assert!(!log.exists());
    assert_eq!(
        fs::read(format!("{directory}/review.html")).unwrap(),
        original
    );
    #[cfg(target_os = "macos")]
    let opener = "open";
    #[cfg(not(target_os = "macos"))]
    let opener = "xdg-open";
    let fake = bin.join(opener);
    fs::write(
        &fake,
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$HOST_TEST_LOG\"\n",
    )
    .unwrap();
    fs::set_permissions(fake, fs::Permissions::from_mode(0o700)).unwrap();
    let opened = command(&home)
        .env("PATH", path)
        .env("HOST_TEST_LOG", &log)
        .args(["--quiet", "terms", "runs", "open"])
        .output()
        .unwrap();
    assert!(opened.status.success());
    let view = fs::read_to_string(&log).unwrap();
    let rendered = fs::read_to_string(view.trim()).unwrap();
    assert!(rendered.contains("SyntheticPrivateTerm"));
    assert!(view.contains("terms/views/"));
    assert!(!rendered.contains("window.untrusted"));
    assert_eq!(entry["registered"], true);
}

#[test]
fn ambiguous_prefixes_are_rejected_without_reading_contexts() {
    use sha2::{Digest, Sha256};
    let home = TempDir::new().unwrap();
    let root = home.path().join("state/nym/pi-terms");
    fs::create_dir_all(&root).unwrap();
    let root = root.canonicalize().unwrap();
    let mut prefixes = std::collections::HashMap::new();
    let mut collision = None;
    for index in 0..=65_536 {
        let directory = root.join(format!("candidate-{index}"));
        let id = format!(
            "{:x}",
            Sha256::digest(directory.to_str().unwrap().as_bytes())
        );
        let prefix = id[..4].to_string();
        if let Some(other) = prefixes.insert(prefix.clone(), directory.clone()) {
            collision = Some((prefix, other, directory));
            break;
        }
    }
    let (prefix, a, b) = collision.unwrap();
    for directory in [a, b] {
        fs::create_dir(&directory).unwrap();
        for file in ["discovery.json", "review.html"] {
            fs::write(directory.join(file), "not read by inventory").unwrap();
        }
    }
    let result = run(&home, &["--json", "terms", "runs", "show", &prefix]);
    assert_eq!(result.status.code(), Some(1));
    assert!(result.stdout.is_empty());
}
