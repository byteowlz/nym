//! Offline regression for a requested but unavailable token NER model.
#![cfg(feature = "ner")]

use std::{fs, path::Path, process::Command};

fn write_config(path: &Path, backend: &str, model: &Path) {
    let quoted = toml::Value::String(model.to_string_lossy().into_owned());
    fs::write(path, format!("[ner]\nenabled = true\nbackend = {backend:?}\nprovider = \"cpu\"\nmodel = {quoted}\ntoken_model = {quoted}\n")).unwrap();
}

#[test]
fn every_backend_initialization_failure_exits_one_even_after_native_exit_hook() {
    for backend in ["tokens", "gliner", "both"] {
        for fixture in [
            "empty",
            "missing",
            "corrupt_model",
            "corrupt_tokenizer",
            "missing_config",
        ] {
            if backend == "gliner" && fixture == "missing_config" {
                continue;
            }
            let dir = tempfile::tempdir().unwrap();
            let model = dir.path().join("confidential-model-location");
            if fixture != "missing" {
                fs::create_dir(&model).unwrap();
            }
            if fixture.starts_with("corrupt") || fixture == "missing_config" {
                tokenizers::Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default())
                    .save(model.join("tokenizer.json"), false)
                    .unwrap();
                fs::write(
                    model.join("config.json"),
                    r#"{"id2label":{"0":"O","1":"B-person"}}"#,
                )
                .unwrap();
                fs::write(model.join("model.onnx"), "not an ONNX model").unwrap();
                if fixture == "corrupt_tokenizer" {
                    fs::write(
                        model.join("tokenizer.json"),
                        "confidential-corrupt-tokenizer",
                    )
                    .unwrap();
                }
                if fixture == "missing_config" {
                    fs::remove_file(model.join("config.json")).unwrap();
                }
            }
            let config = dir.path().join("config.toml");
            write_config(&config, backend, &model);
            let input = dir.path().join("input.txt");
            fs::write(&input, "Alice Smith and Jörg Beispiel.").unwrap();
            let destination = dir.path().join("output.txt");
            fs::write(&destination, "existing destination").unwrap();
            for verb in ["detect", "anon"] {
                let mut command = Command::new(env!("CARGO_BIN_EXE_nym"));
                command.arg("--config").arg(&config).arg(verb).arg(&input);
                if verb == "detect" {
                    command.args(["--summary-json"]);
                } else {
                    command.arg("--output").arg(&destination);
                }
                // Dynamic-runtime builds must sanitize missing-library panics too.
                let output = command
                    .env("HF_HUB_OFFLINE", "1")
                    .env("ORT_DYLIB_PATH", model.join("confidential-native-library"))
                    .output()
                    .unwrap();
                assert_eq!(output.status.code(), Some(1), "{backend}/{fixture}/{verb}");
                assert!(
                    output.stdout.is_empty(),
                    "failed scan must not emit a complete result"
                );
                assert_eq!(fs::read(&destination).unwrap(), b"existing destination");
                for rendered in [
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr),
                ] {
                    for private in [
                        "Alice",
                        "Jörg",
                        "confidential-model-location",
                        "confidential-corrupt-tokenizer",
                        "confidential-native-library",
                    ] {
                        assert!(!rendered.contains(private), "error leaked private context");
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires NYM_TEST_CACHED_NER_MODEL or NYM_TEST_CACHED_GLINER_MODEL; never downloads"]
fn cached_models_preserve_policy_exit_two_and_normal_inspection_exit_zero() {
    let mut tested = 0;
    for (backend, variable) in [
        ("tokens", "NYM_TEST_CACHED_NER_MODEL"),
        ("gliner", "NYM_TEST_CACHED_GLINER_MODEL"),
    ] {
        let Some(model) = std::env::var_os(variable) else {
            continue;
        };
        let model = Path::new(&model).canonicalize().unwrap();
        assert!(model.join("tokenizer.json").is_file());
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        write_config(&config, backend, &model);
        let input = dir.path().join("input.txt");
        fs::write(
            &input,
            "Alice Smith and Jörg Beispiel are fictional people.",
        )
        .unwrap();
        for (policy, exit) in [(true, 2), (false, 0)] {
            let mut command = Command::new(env!("CARGO_BIN_EXE_nym"));
            command
                .arg("--config")
                .arg(&config)
                .arg("detect")
                .arg(&input)
                .arg("--summary-json");
            if policy {
                command.args(["--fail-on", "identity"]);
            }
            let output = command.env("HF_HUB_OFFLINE", "1").output().unwrap();
            assert_eq!(
                output.status.code(),
                Some(exit),
                "{backend}, policy={policy}"
            );
            let summary: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            assert!(summary["total"].as_u64().unwrap() > 0);
        }
        tested += 1;
    }
    assert!(tested > 0, "provide an existing cached model explicitly");
}

#[test]
fn empty_local_model_is_an_operational_error_not_a_clean_audit() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("confidential-model-location");
    fs::create_dir(&model).unwrap();
    let config = dir.path().join("config.toml");
    let quoted_model = toml::Value::String(model.to_string_lossy().into_owned());
    fs::write(
        &config,
        format!("[ner]\nenabled = true\nbackend = \"tokens\"\nprovider = \"cpu\"\ntoken_model = {quoted_model}\n"),
    ).unwrap();
    let input = dir.path().join("input.txt");
    fs::write(&input, "Fictional person Alice Smith and Jörg Beispiel.").unwrap();
    for ner_args in [vec![], vec!["--ner"]] {
        let output = Command::new(env!("CARGO_BIN_EXE_nym"))
            .args(["--config"])
            .arg(&config)
            .args(["detect"])
            .arg(&input)
            .args(ner_args)
            .args(["--fail-on", "identity", "--summary-json"])
            .env("HF_HUB_OFFLINE", "1")
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert!(
            output.stdout.is_empty(),
            "failed audit must not emit a complete manifest"
        );
        let diagnostic = String::from_utf8_lossy(&output.stderr);
        for secret in ["Alice", "Jörg", "confidential-model-location"] {
            assert!(
                !diagnostic.contains(secret),
                "error leaked confidential context"
            );
        }
    }
    let baseline = Command::new(env!("CARGO_BIN_EXE_nym"))
        .args(["--config"])
        .arg(&config)
        .args(["detect"])
        .arg(&input)
        .args(["--no-ner", "--fail-on", "identity", "--summary-json"])
        .env("HF_HUB_OFFLINE", "1")
        .output()
        .unwrap();
    assert_eq!(baseline.status.code(), Some(0));
    let summary: serde_json::Value = serde_json::from_slice(&baseline.stdout).unwrap();
    assert_eq!(
        summary,
        serde_json::json!({"total":0,"by_pattern":{},"by_category":{}})
    );
}
