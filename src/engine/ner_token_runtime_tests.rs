//! Subprocess coverage for native provider registration and controlled errors.

use super::{NerProvider, TokenClassDetector, Tokenizer};
use crate::config::NerConfig;

#[test]
fn provider_config_is_explicit_and_validated() -> Result<(), Box<dyn std::error::Error>> {
    let defaults: NerConfig = toml::from_str("")?;
    let cpu: NerConfig = toml::from_str("provider = 'cpu'")?;
    let auto: NerConfig = toml::from_str("provider = 'auto'")?;
    assert_eq!(
        [defaults.provider, cpu.provider, auto.provider],
        [NerProvider::Auto, NerProvider::Cpu, NerProvider::Auto]
    );
    assert!(toml::from_str::<NerConfig>("provider = 'invalid'").is_err());
    let example: crate::config::Config =
        toml::from_str(include_str!("../../examples/config.toml"))?;
    assert_eq!(example.ner.provider, NerProvider::Auto);
    let schema: serde_json::Value =
        serde_json::from_str(include_str!("../../examples/config.schema.json"))?;
    assert_eq!(
        schema["properties"]["ner"]["properties"]["provider"],
        serde_json::json!({
            "type": "string",
            "enum": ["auto", "cpu"],
            "description": "Token backend only (tokens, or token portion of both): auto uses compiled accelerators with CPU fallback; cpu bypasses accelerators. Does not control GLiNER.",
            "default": "auto"
        })
    );
    Ok(())
}

/// An abort must fail the parent test, not terminate the entire test suite.
#[test]
fn invalid_model_returns_error() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    const CHILD: &str = "NYM_TEST_NER_RUNTIME_CHILD";
    if std::env::var_os(CHILD).is_none() {
        for provider in ["auto", "cpu"] {
            let output = std::process::Command::new(std::env::current_exe()?)
                .args([
                    "--exact",
                    "engine::ner_token::runtime_tests::invalid_model_returns_error",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("NYM_NER_PROVIDER", provider)
                .output()?;
            assert!(
                output.status.success(),
                "{provider} child failed: {:?}\n{}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("controlled NER model error"));
        }
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default())
        .save(dir.path().join("tokenizer.json"), false)?;
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"id2label":{"0":"O","1":"B-person"}}"#,
    )?;
    std::fs::write(dir.path().join("model.onnx"), b"not an ONNX model")?;
    let provider = match std::env::var("NYM_NER_PROVIDER")?.as_str() {
        "cpu" => NerProvider::Cpu,
        _ => NerProvider::Auto,
    };
    match TokenClassDetector::from_dir(dir.path(), None, provider) {
        Ok(_) => return Err("invalid model unexpectedly loaded".into()),
        Err(error) => {
            let message = error.to_string();
            let expected = match provider {
                NerProvider::Auto
                    if cfg!(any(
                        feature = "ner-coreml",
                        feature = "ner-cuda",
                        feature = "ner-tensorrt"
                    )) =>
                {
                    "CPU fallback failed:"
                }
                _ => "token NER session failed:",
            };
            assert!(message.contains(expected), "{message}");
            println!("controlled NER model error: {message}");
        }
    }
    Ok(())
}
