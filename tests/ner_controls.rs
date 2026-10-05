//! Offline CLI proofs for backend-scoped NER controls; status never loads models.
use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    process::{Command, Output, Stdio},
};

fn run(config: &str, args: &[&str], input: Option<&str>) -> Output {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(&path, config).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nym"));
    cmd.current_dir(dir.path())
        .arg("--config")
        .arg(path)
        .args(args)
        .env("HF_HUB_OFFLINE", "1")
        .env("HF_HOME", dir.path().join("empty-cache"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().unwrap();
    if let Some(text) = input {
        child
            .stdin
            .take()
            .unwrap()
            .write_all(text.as_bytes())
            .unwrap();
    } else {
        drop(child.stdin.take());
    }
    child.wait_with_output().unwrap()
}

fn status(config: &str, model: Option<&str>) -> Value {
    let mut args = vec!["config", "ner-status"];
    if let Some(model) = model {
        args.extend(["--ner-model", model]);
    }
    let output = run(config, &args, None);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn threshold_overrides_are_validated_and_never_silently_ignored() {
    let config = "[ner]\nbackend = 'tokens'\nthreshold = 0.8\n";
    let mut expected = status(config, None);
    expected["threshold"] = json!(0.9);
    let out = run(
        config,
        &["config", "ner-status", "--ner-threshold", "0.9"],
        None,
    );
    assert!(out.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&out.stdout).unwrap(),
        expected
    );
    for invalid in ["NaN", "-0.1", "1.1"] {
        let arg = format!("--ner-threshold={invalid}");
        let out = run(config, &["config", "ner-status", &arg], None);
        assert_eq!(out.status.code(), Some(1));
        assert!(out.stdout.is_empty());
    }
    for args in [
        vec!["detect", "--no-ner", "--ner-model", "synthetic-model"],
        vec!["anon", "--no-ner", "--ner-threshold", "0.9"],
    ] {
        let out = run(
            "[ner]\nenabled = false\n",
            &args,
            Some("synthetic private input"),
        );
        assert_eq!(out.status.code(), Some(1));
        assert!(out.stdout.is_empty());
        assert!(String::from_utf8_lossy(&out.stderr).contains("require enabled NER"));
        assert!(!String::from_utf8_lossy(&out.stderr).contains("private input"));
    }
}

#[test]
fn cli_model_override_obeys_backend_and_config_precedence_without_downloads() {
    let token =
        "[ner]\nbackend = 'tokens'\ntoken_model = 'Wismut/nym-pii-multilingual'\nthreshold = 0.8\n";
    let mut expected = status(token, None);
    assert_eq!(expected["threshold"], json!(0.8));
    assert_eq!(
        expected["models"][0]["identifier"],
        "Wismut/nym-pii-multilingual"
    );
    expected["models"][0]["identifier"] = json!("nationaldesignstudio/rampart");
    assert_eq!(
        status(token, Some("nationaldesignstudio/rampart")),
        expected
    );
    let gliner = "[ner]\nbackend = 'gliner'\nmodel = 'private-org/private-model'\nlabels = ['GIVEN_NAME', 'surname', 'country']\n";
    let mut expected = status(gliner, None);
    assert_eq!(expected["models"][0]["identifier"], "[custom]");
    assert_eq!(
        expected["models"][0]["labels"],
        json!(["first_name", "last_name", "country"])
    );
    expected["models"][0]["identifier"] = json!("onnx-community/gliner_multi-v2.1");
    assert_eq!(
        status(gliner, Some("onnx-community/gliner_multi-v2.1")),
        expected
    );
}

#[test]
fn status_is_safe_and_reports_both_models_decoding_and_label_scope() {
    assert_eq!(
        status(
            "[ner]\nbackend = 'both'\nenabled = true\nrecall_first = true\nprovider = 'cpu'\n",
            None
        ),
        json!({
            "configured_enabled": true, "backend": "both", "threshold": 0.5, "regex_scope": "independent",
            "models": [
                {"backend":"tokens", "identifier":"Wismut/nym-pii-multilingual-small/int8", "revision":null,
                 "label_scope":"all-model-classes", "labels":null, "decoding":"recall-first", "provider":"cpu"},
                {"backend":"gliner", "identifier":"onnx-community/gliner_multi-v2.1", "revision":null,
                 "label_scope":"gliner-only", "labels":["person","organization","street_address","city","country"],
                 "decoding":"span", "provider":null}
            ]
        })
    );
    for model in [
        "/private/customer/model",
        "~/private/customer/model",
        "private-tenant/private-repository",
        "C:\\private\\customer\\model",
    ] {
        let model = toml::Value::String(model.to_string());
        let config = format!(
            "[ner]\nbackend = 'both'\nmodel = {model}\ntoken_model = {model}\ncache_dir = '/private/customer/cache'\n"
        );
        let value = status(&config, None);
        assert_eq!(
            value["models"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| m["identifier"].clone())
                .collect::<Vec<_>>(),
            vec![json!("[custom]"), json!("[custom]")]
        );
        let serialized = value.to_string();
        for secret in ["private", "customer", "tenant", "repository", "cache"] {
            assert!(!serialized.contains(secret));
        }
    }
}

#[test]
fn ambiguous_models_and_explicit_incompatible_or_bad_labels_are_errors() {
    let both = run(
        "[ner]\nbackend = 'both'\n",
        &[
            "config",
            "ner-status",
            "--ner-model",
            "/private/customer/model",
        ],
        None,
    );
    assert_eq!(both.status.code(), Some(1));
    assert!(both.stdout.is_empty());
    let error = String::from_utf8_lossy(&both.stderr);
    assert!(error.contains("ambiguous") && !error.contains("private"));
    for backend in ["tokens", "both", "gliner"] {
        for labels in ["[]", "['private-unknown-label']", "['email']"] {
            let config = format!("[ner]\nbackend = '{backend}'\nlabels = {labels}\n");
            let output = run(&config, &["config", "ner-status"], None);
            assert_eq!(output.status.code(), Some(1));
            assert!(output.stdout.is_empty());
            assert!(!String::from_utf8_lossy(&output.stderr).contains("private-unknown-label"));
        }
        if backend != "gliner" {
            let config = format!("[ner]\nbackend = '{backend}'\nlabels = ['person']\n");
            let output = run(&config, &["config", "ner-status"], None);
            assert_eq!(output.status.code(), Some(1));
            assert!(String::from_utf8_lossy(&output.stderr).contains("GLiNER-only"));
        }
    }
    for backend in ["tokens", "gliner"] {
        let config = format!("[ner]\nbackend = '{backend}'\n");
        let output = run(&config, &["config", "ner-status", "--ner-model", ""], None);
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn gliner_labels_do_not_restrict_regex_and_help_matches_defaults() {
    let output = run(
        "[ner]\nbackend = 'gliner'\nlabels = ['person']\nenabled = false\n",
        &[
            "detect",
            "--no-ner",
            "--patterns",
            "email",
            "--summary-json",
        ],
        Some("Alice Smith, alice@example.invalid, Germany. let country = 42;"),
    );
    assert!(output.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&output.stdout).unwrap(),
        json!({
            "total":1, "by_pattern":{"email":1}, "by_category":{"contact":1}
        })
    );
    let defaults = status("", None);
    assert_eq!(
        (defaults["backend"].clone(), defaults["threshold"].clone()),
        (json!("tokens"), json!(0.5))
    );
    let output = run("", &["detect", "--help"], None);
    assert!(output.status.success());
    let help = String::from_utf8_lossy(&output.stdout);
    assert!(help.contains("default: 0.5"));
    assert!(help.contains("both") && help.contains("--ner-model"));
    assert!(help.contains("regex") && help.contains("backend"));
}

#[cfg(feature = "ner")]
#[test]
fn invalid_generic_override_is_not_ignored_by_either_engine() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("private-empty-model");
    fs::create_dir(&model).unwrap();
    for backend in ["tokens", "gliner"] {
        let config = format!("[ner]\nbackend = '{backend}'\nprovider = 'cpu'\n");
        let output = run(
            &config,
            &[
                "detect",
                "--ner",
                "--ner-model",
                model.to_str().unwrap(),
                "--summary-json",
            ],
            Some("Fictional Alice Smith, alice@example.invalid, Germany."),
        );
        assert_eq!(output.status.code(), Some(1));
        assert!(
            output.stdout.is_empty(),
            "failed model must not emit a complete aggregate"
        );
        let error = String::from_utf8_lossy(&output.stderr);
        for private in ["private-empty-model", "Alice", "alice@example.invalid"] {
            assert!(!error.contains(private));
        }
    }
}

#[test]
fn malformed_default_config_fails_closed_without_echoing_contents() {
    for source in ["local", "global", "env", "syntax"] {
        let dir = tempfile::tempdir().unwrap();
        let global = dir.path().join("global");
        let path = if source == "global" {
            global.join("nym/config.toml")
        } else {
            dir.path().join("config.toml")
        };
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_nym"));
        cmd.current_dir(dir.path())
            .args(["config", "ner-status"])
            .env("XDG_CONFIG_HOME", &global)
            .env("HF_HUB_OFFLINE", "1");
        for (key, _) in std::env::vars().filter(|(key, _)| key.starts_with("NYM_")) {
            cmd.env_remove(key);
        }
        if source == "env" {
            cmd.env("NYM_NER_LABELS", "private-invalid-config-value");
        } else {
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            let value = if source == "syntax" {
                "[ner]\nlabels = ['private-invalid-config-value'\n"
            } else {
                "[ner]\nlabels = 'private-invalid-config-value'\n"
            };
            fs::write(path, value).unwrap();
        }
        let output = cmd.output().unwrap();
        assert_eq!(output.status.code(), Some(1), "source={source}");
        assert!(
            output.stdout.is_empty(),
            "malformed config must not fall back to defaults"
        );
        let diagnostic = String::from_utf8_lossy(&output.stderr);
        assert!(!diagnostic.contains("private-invalid-config-value"));
        assert!(!diagnostic.contains(dir.path().to_str().unwrap()));
    }
}
