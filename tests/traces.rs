use serde_json::Value;
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};
use tempfile::TempDir;

fn nym(root: &Path, args: &[&str]) -> Output {
    let home = root.join("zorbaxtester");
    fs::create_dir_all(&home).unwrap();
    fs::write(root.join("config.toml"), "").unwrap();
    Command::new(env!("CARGO_BIN_EXE_nym"))
        .env("HOME", &home)
        .env("USER", "zorbaxtester")
        .env("XDG_CONFIG_HOME", root.join("xdg"))
        .env("GIT_CONFIG_GLOBAL", root.join("gitconfig"))
        .args(["--config", root.join("config.toml").to_str().unwrap()])
        .args(args)
        .output()
        .unwrap()
}

fn records(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn session(lines: &[&str]) -> String {
    lines.iter().map(|l| format!("{l}\n")).collect()
}

const SESSION: &[&str] = &[
    r#"{"type":"session","id":"s-1","timestamp":"2026-10-12T10:00:00Z","cwd":"/Users/zorbaxtester/projects/app"}"#,
    r#"{"type":"message","id":"m-1","message":{"role":"user","content":[{"type":"text","text":"Mail ada.quill@example.org and set DB_PASSWORD=Sup3rSecretValue1"}]}}"#,
    r#"{"type":"message","id":"m-2","message":{"role":"assistant","content":[{"type":"thinking","thinking":"ok","thinkingSignature":"EqQBCkYIBxgCKkDada.quill@example.org"},{"type":"text","text":"Reading /Users/zorbaxtester/projects/app/.env for ada.quill@example.org"}]}}"#,
    r#"{"type":"message","id":"m-3","message":{"role":"toolResult","toolCallId":"t-1","content":[{"type":"text","text":"user=zorbaxtester password Sup3rSecretValue1 owner ada.quill@example.org"},{"type":"image","mimeType":"image/png","data":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"}]}}"#,
    r#"{"z":"untouched record keeps its exact bytes","a":1}"#,
];

#[test]
fn per_session_consistent_pseudonyms_with_structure_preserved() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let input = root.join("sessions");
    fs::create_dir_all(input.join("project-a")).unwrap();
    fs::write(input.join("project-a/one.jsonl"), session(SESSION)).unwrap();
    fs::write(input.join("two.jsonl"), session(SESSION)).unwrap();
    let out = root.join("out");
    let result = nym(
        root,
        &[
            "--json",
            "traces",
            "anon",
            input.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
            "--no-ner",
            "--no-progress",
            "--strip-images",
        ],
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let summary: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(summary["sessions"], 2);
    assert_eq!(summary["failed"], 0);
    assert_eq!(summary["images_removed"], 2);

    let one = out.join("project-a/one.jsonl");
    let two = out.join("two.jsonl");
    let text_one = fs::read_to_string(&one).unwrap();
    for secret in [
        "ada.quill@example.org",
        "zorbaxtester",
        "Sup3rSecretValue1",
        "iVBORw0KGgo",
    ] {
        // The signature is opaque and deliberately not rewritten.
        let outside_signature = text_one.replace("EqQBCkYIBxgCKkDada.quill@example.org", "");
        assert!(
            !outside_signature.contains(secret),
            "leaked a {} char value",
            secret.len()
        );
    }
    let r = records(&one);
    // Opaque and structural fields are untouched.
    assert_eq!(r[0]["id"], "s-1");
    assert_eq!(r[0]["timestamp"], "2026-10-12T10:00:00Z");
    assert_eq!(
        r[2]["message"]["content"][0]["thinkingSignature"],
        "EqQBCkYIBxgCKkDada.quill@example.org"
    );
    assert_eq!(r[3]["message"]["content"][1]["data"], "");
    assert_eq!(r[3]["message"]["content"][1]["mimeType"], "image/png");
    // Home paths keep their structure; only the account name changes.
    let cwd = r[0]["cwd"].as_str().unwrap();
    assert!(cwd.starts_with("/Users/") && cwd.ends_with("/projects/app"));
    // Consistent within the session: the cwd account matches the later path.
    let account = cwd
        .trim_start_matches("/Users/")
        .trim_end_matches("/projects/app");
    let later = r[2]["message"]["content"][1]["text"].as_str().unwrap();
    assert!(later.contains(&format!("/Users/{account}/projects/app/.env")));
    // The same email maps to one replacement everywhere in the session.
    let user_text = r[1]["message"]["content"][0]["text"].as_str().unwrap();
    let email = user_text
        .split_whitespace()
        .find(|w| w.contains('@'))
        .unwrap()
        .to_string();
    assert!(later.ends_with(&email));
    let tool = r[3]["message"]["content"][0]["text"].as_str().unwrap();
    assert!(tool.ends_with(&email));
    // Secret value detected once, propagated to its bare occurrence in tool output.
    let secret_alias = user_text.rsplit('=').next().unwrap();
    assert!(tool.contains(&format!("password {secret_alias} ")));
    // Field order and untouched records are preserved byte for byte.
    let lines: Vec<_> = text_one.lines().collect();
    assert!(lines[0].starts_with(r#"{"type":"session","id":"s-1","timestamp""#));
    assert_eq!(lines[4], SESSION[4]);
    // Independent across sessions.
    let other = records(&two);
    let other_email = other[1]["message"]["content"][0]["text"]
        .as_str()
        .unwrap()
        .split_whitespace()
        .find(|w| w.contains('@'))
        .unwrap()
        .to_string();
    assert_ne!(email, other_email);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(fs::metadata(&one).unwrap().permissions().mode() & 0o077, 0);
    }
}

#[test]
fn seeded_runs_are_reproducible_and_keys_restore() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let input = root.join("in");
    fs::create_dir_all(&input).unwrap();
    // No image, so every change is reversible.
    let lines: Vec<_> = SESSION.iter().take(3).copied().collect();
    fs::write(input.join("s.jsonl"), session(&lines)).unwrap();
    let run = |name: &str| {
        let out = root.join(name);
        let keys = root.join(format!("{name}-keys"));
        let result = nym(
            root,
            &[
                "traces",
                "anon",
                input.to_str().unwrap(),
                "-o",
                out.to_str().unwrap(),
                "--keys",
                keys.to_str().unwrap(),
                "--no-ner",
                "--seed",
                "7",
                "--no-progress",
            ],
        );
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        (
            fs::read_to_string(out.join("s.jsonl")).unwrap(),
            keys.join("s.keys.jsonl"),
        )
    };
    let (first, keys) = run("a");
    let (second, _) = run("b");
    assert_eq!(first, second);
    let restored = root.join("restored.jsonl");
    let result = nym(
        root,
        &[
            "deanon",
            root.join("a/s.jsonl").to_str().unwrap(),
            "-k",
            keys.to_str().unwrap(),
            "-o",
            restored.to_str().unwrap(),
        ],
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(records(&restored), records(&input.join("s.jsonl")));
}

#[test]
fn invalid_session_fails_without_partial_output_or_content_in_errors() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let input = root.join("in");
    fs::create_dir_all(&input).unwrap();
    fs::write(input.join("good.jsonl"), session(SESSION)).unwrap();
    fs::write(
        input.join("bad.jsonl"),
        "{\"text\":\"ada.quill@example.org\"}\nnot json ada.quill\n",
    )
    .unwrap();
    let out = root.join("out");
    let result = nym(
        root,
        &[
            "traces",
            "anon",
            input.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
            "--no-ner",
            "--no-progress",
        ],
    );
    assert_eq!(result.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("invalid JSON on line 2"));
    assert!(!stderr.contains("ada.quill") && !stderr.contains("bad.jsonl"));
    assert!(out.join("good.jsonl").exists());
    assert!(!out.join("bad.jsonl").exists());
    // Existing outputs are never replaced without --force.
    let again = nym(
        root,
        &[
            "traces",
            "anon",
            input.join("good.jsonl").to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
            "--no-ner",
        ],
    );
    assert_eq!(again.status.code(), Some(1));
    // Output inside the input tree is rejected.
    let nested = nym(
        root,
        &[
            "traces",
            "anon",
            input.to_str().unwrap(),
            "-o",
            input.join("o").to_str().unwrap(),
            "--no-ner",
        ],
    );
    assert_eq!(nested.status.code(), Some(1));
}

#[test]
fn vendor_secrets_are_detected_at_default_confidence() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let input = root.join("in");
    fs::create_dir_all(&input).unwrap();
    let keys = [
        "sk-ant-api03-Abc_def-GHIjklMNOpqrSTUvwxYZ0123456789abcdefgh",
        "sk-proj-AbcdefGHIJ_klmnop-QRSTuvwxyz0123456789",
        "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij",
        "github_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKL",
        concat!("xoxb-", "0000000000-SYNTHETIC_ONLY_NOT_A_TOKEN"),
    ];
    let lines: Vec<String> = keys
        .iter()
        .map(|k| format!(r#"{{"message":{{"role":"user","content":[{{"type":"text","text":"use {k} now"}}]}}}}"#))
        .collect();
    fs::write(input.join("s.jsonl"), lines.join("\n") + "\n").unwrap();
    let out = root.join("out");
    let result = nym(
        root,
        &[
            "traces",
            "anon",
            input.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
            "--no-ner",
            "--no-progress",
        ],
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let text = fs::read_to_string(out.join("s.jsonl")).unwrap();
    for key in keys {
        assert!(
            !text.contains(key),
            "vendor key with prefix {} leaked",
            &key[..6]
        );
    }
}

#[test]
fn images_are_kept_unless_stripping_is_requested() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let input = root.join("in");
    fs::create_dir_all(&input).unwrap();
    fs::write(input.join("s.jsonl"), session(SESSION)).unwrap();
    let out = root.join("out");
    let result = nym(
        root,
        &[
            "--json",
            "traces",
            "anon",
            input.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
            "--no-ner",
            "--no-progress",
        ],
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let summary: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(summary["images_removed"], 0);
    let r = records(&out.join("s.jsonl"));
    assert_eq!(
        r[3]["message"]["content"][1]["data"],
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
    );
}

#[test]
fn redaction_profiles_are_shared_and_stack_on_default() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let ok = |args: &[&str]| {
        let out = nym(root, args);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        out
    };
    ok(&[
        "profiles",
        "add",
        "default",
        "--list",
        "allow",
        "support@example.org",
    ]);
    ok(&[
        "profiles",
        "add",
        "work",
        "--list",
        "sensitive",
        "Acme Corp",
    ]);
    let downloaded = root.join("trace-allow.txt");
    fs::write(&downloaded, "# from the review page\nkeep@example.org\n").unwrap();
    ok(&[
        "profiles",
        "import",
        "work",
        "--list",
        "allow",
        downloaded.to_str().unwrap(),
    ]);
    let listed: Value =
        serde_json::from_slice(&ok(&["--json", "profiles", "list"]).stdout).unwrap();
    assert_eq!(
        listed,
        serde_json::json!([
            {"name": "default", "path": root.join("xdg/nym/profiles/default"), "sensitive": 0, "allow": 1},
            {"name": "work", "path": root.join("xdg/nym/profiles/work"), "sensitive": 1, "allow": 1}
        ])
    );

    let input = root.join("in");
    fs::create_dir_all(&input).unwrap();
    let text = "Acme Corp: mail support@example.org, keep@example.org or ada.quill@example.org";
    fs::write(
        input.join("s.jsonl"),
        format!(r#"{{"message":{{"role":"user","content":[{{"type":"text","text":"{text}"}}]}}}}"#)
            + "\n",
    )
    .unwrap();
    let run = |name: &str, extra: &[&str]| {
        let out = root.join(name);
        let mut args = vec![
            "traces",
            "anon",
            input.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
            "--no-ner",
            "--no-progress",
        ];
        args.extend_from_slice(extra);
        ok(&args);
        fs::read_to_string(out.join("s.jsonl")).unwrap()
    };
    let base = run("base", &[]);
    assert!(base.contains("support@example.org") && base.contains("Acme Corp"));
    assert!(base.contains("keep@example.org") == false && !base.contains("ada.quill"));
    let work = run("work", &["--redaction-profile", "work"]);
    assert!(work.contains("support@example.org") && work.contains("keep@example.org"));
    assert!(!work.contains("Acme Corp") && !work.contains("ada.quill"));
    // A missing named profile is an error, not a silent no-op.
    let missing = nym(
        root,
        &[
            "traces",
            "anon",
            input.to_str().unwrap(),
            "-o",
            root.join("x").to_str().unwrap(),
            "--no-ner",
            "--redaction-profile",
            "nope",
        ],
    );
    assert_eq!(missing.status.code(), Some(1));
    // `nym anon` shares the same profiles.
    let plain = root.join("plain.txt");
    fs::write(&plain, format!("{text}\n")).unwrap();
    let anon = ok(&[
        "anon",
        plain.to_str().unwrap(),
        "--redaction-profile",
        "work",
        "--strategy",
        "placeholder",
    ]);
    let anon = String::from_utf8(anon.stdout).unwrap();
    assert!(!anon.contains("Acme Corp") && !anon.contains("ada.quill"));
}

#[test]
fn plan_lists_candidates_and_keep_values_are_honored() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let input = root.join("in");
    fs::create_dir_all(&input).unwrap();
    fs::write(input.join("s.jsonl"), session(SESSION)).unwrap();
    let plan_path = root.join("plan.json");
    let out = nym(
        root,
        &[
            "traces",
            "anon",
            input.to_str().unwrap(),
            "--plan",
            plan_path.to_str().unwrap(),
            "--no-ner",
            "--no-progress",
        ],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!root.join("out").exists());
    let plan: Value = serde_json::from_slice(&fs::read(&plan_path).unwrap()).unwrap();
    let candidates = plan["sessions"][0]["candidates"].as_array().unwrap();
    let find = |value: &str| candidates.iter().find(|c| c["value"] == value).cloned();
    let email = find("ada.quill@example.org").expect("email candidate");
    assert_eq!(email["class"], "email");
    // Two prose occurrences plus the tool output; the opaque signature is never scanned.
    assert_eq!(email["count"], 3);
    assert!(
        email["context"]
            .as_str()
            .unwrap()
            .contains("ada.quill@example.org")
    );
    let secret = find("Sup3rSecretValue1").expect("secret candidate");
    assert_eq!(secret["credential"], true);
    // First seen in the home path, so detected; the local-identity literal covers the rest.
    let account = find("zorbaxtester").expect("account candidate");
    assert_eq!(
        (account["class"].clone(), account["source"].clone()),
        ("username".into(), "detected".into())
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(&plan_path).unwrap().permissions().mode() & 0o077,
            0
        );
    }
    // Apply with an explicit keep decision: the email stays, everything else is replaced.
    let keep = root.join("keep.txt");
    fs::write(&keep, "ada.quill@example.org\n").unwrap();
    let output = root.join("out");
    let out = nym(
        root,
        &[
            "traces",
            "anon",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--keep-values-file",
            keep.to_str().unwrap(),
            "--no-ner",
            "--no-progress",
        ],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = fs::read_to_string(output.join("s.jsonl")).unwrap();
    assert_eq!(text.matches("ada.quill@example.org").count(), 4);
    assert!(!text.contains("Sup3rSecretValue1") && !text.contains("/Users/zorbaxtester"));
    // --plan cannot be combined with outputs.
    let both = nym(
        root,
        &[
            "traces",
            "anon",
            input.to_str().unwrap(),
            "--plan",
            plan_path.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ],
    );
    assert_eq!(both.status.code(), Some(2));
}
