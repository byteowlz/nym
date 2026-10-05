//! Native ORT/provider regression with an explicitly supplied cached checkpoint.
//! No downloads: run with NYM_TEST_CACHED_NER_MODEL=/local/model/directory
//! cargo test --features ner-coreml --test ner_runtime -- --ignored

#![cfg(feature = "ner")]

use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[test]
#[ignore = "requires NYM_TEST_CACHED_NER_MODEL pointing at the existing small int8 checkpoint"]
fn cached_checkpoint_cpu_auto_and_relocated_are_unicode_safe() -> Result<(), Box<dyn Error>> {
    let model = PathBuf::from(std::env::var("NYM_TEST_CACHED_NER_MODEL")?).canonicalize()?;
    for file in ["model_int8.onnx", "tokenizer.json", "config.json"] {
        assert!(model.join(file).is_file(), "missing cached {file}");
    }
    let dir = tempfile::tempdir()?;
    let input =
        "Alice Smith and Jörg Beispiel. Contact alpha@example.invalid. Keep --strict unchanged.";
    let input_path = dir.path().join("input.txt");
    fs::write(&input_path, input)?;
    let config = dir.path().join("config.toml");
    let mut ner = toml::map::Map::new();
    ner.insert("enabled".into(), true.into());
    ner.insert("backend".into(), "tokens".into());
    ner.insert(
        "token_model".into(),
        model.to_string_lossy().as_ref().into(),
    );
    ner.insert("threshold".into(), 0.5.into());
    let mut root = toml::map::Map::new();
    root.insert("ner".into(), ner.into());
    let binary = PathBuf::from(env!("CARGO_BIN_EXE_nym"));
    let relocated = dir.path().join("nym-relocated");
    fs::copy(&binary, &relocated)?;

    let mut expected = None;
    for executable in [&binary, &relocated] {
        for provider in ["cpu", "auto"] {
            root.get_mut("ner")
                .and_then(toml::Value::as_table_mut)
                .ok_or("missing ner config")?
                .insert("provider".into(), provider.into());
            fs::write(&config, toml::to_string(&root)?)?;
            let output = Command::new(executable)
                .arg("detect")
                .arg(&input_path)
                .arg("--config")
                .arg(&config)
                .args(["--ner", "--json", "--min-confidence", "low"])
                .env_remove("NYM_NER_PROVIDER")
                .current_dir(dir.path())
                .output()?;
            assert!(
                output.status.success(),
                "{provider}: {:?}\n{}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
            let matches: serde_json::Value = serde_json::from_slice(&output.stdout)?;
            let spans = matches.as_array().ok_or("expected a JSON match array")?;
            assert!(
                spans.iter().any(|span| span["matched_text"] == "Jörg"),
                "NER must actually run, not silently degrade to regex-only: {matches}"
            );
            for span in spans {
                let start = usize::try_from(span["start"].as_u64().ok_or("missing start")?)?;
                let end = usize::try_from(span["end"].as_u64().ok_or("missing end")?)?;
                assert_eq!(
                    input.get(start..end),
                    span["matched_text"].as_str(),
                    "invalid UTF-8 byte offsets: {span}"
                );
            }
            if let Some(ref expected) = expected {
                assert_eq!(&matches, expected, "CPU/auto/relocated results must agree");
            } else {
                expected = Some(matches);
            }
        }
    }
    Ok(())
}
