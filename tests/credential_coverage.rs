//! Repeated identical credentials must be fully covered in every native mode,
//! including across records and windows (trx-ppar.1 regression).
use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Output, Stdio},
};

const VALUE: &str = "SyntheticOnly-Pass-7zQ!";
const KEY: &str = "password";

fn run(args: &[&str], input: Option<&str>) -> Output {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("config.toml");
    fs::write(&cfg, "[ner]\nenabled = false\n").unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nym"));
    cmd.arg("--config")
        .arg(cfg)
        .args(args)
        .env("HF_HUB_OFFLINE", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().unwrap();
    if let Some(input) = input {
        child
            .stdin
            .take()
            .unwrap()
            .write_all(input.as_bytes())
            .unwrap();
    }
    child.wait_with_output().unwrap()
}

fn log_line(index: usize) -> String {
    format!(
        "synthetic fixture line {index} | contact=user{index}@example.invalid | \
         {KEY}=\"{VALUE}\" | AWS_ACCESS_KEY_ID=AKIASYNTHETIC0000000{index}"
    )
}

#[test]
fn repeated_quoted_credentials_fully_covered_in_text_json_and_stream() {
    let dir = tempfile::tempdir().unwrap();
    let text_input = dir.path().join("tool.log");
    fs::write(
        &text_input,
        (0..7)
            .map(|index| format!("{}\n", log_line(index)))
            .collect::<String>(),
    )
    .unwrap();
    // Records whose tool text carries the same repeated credential, the shape
    // the agent-loop measurement exercised.
    let jsonl_input = dir.path().join("events.jsonl");
    fs::write(
        &jsonl_input,
        (0..7)
            .map(|index| {
                format!(
                    "{{\"role\":\"toolResult\",\"content\":{}}}\n",
                    serde_json::json!([{"type": "text", "text": log_line(index)}])
                )
            })
            .collect::<String>(),
    )
    .unwrap();
    // The same literal repeated across NER/window boundaries inside one text.
    let long_input = dir.path().join("long.txt");
    let filler = "ordinary operational prose repeats here. ".repeat(120);
    let mut body = String::new();
    for index in 0..7 {
        body.push_str(&filler);
        body.push_str(&format!("attempt {index} {KEY}=\"{VALUE}\" marker\n"));
    }
    fs::write(&long_input, body).unwrap();

    for stream in [false, true]
        .into_iter()
        .filter(|stream| !*stream || cfg!(feature = "streaming"))
    {
        for input in [text_input.clone(), jsonl_input.clone(), long_input.clone()] {
            let dest: PathBuf = dir.path().join(format!(
                "out-{stream}-{}",
                input.file_name().unwrap().to_str().unwrap()
            ));
            let mut args = vec![
                "anon",
                input.to_str().unwrap(),
                "-o",
                dest.to_str().unwrap(),
                "--no-ner",
                "--strategy",
                "consistent",
            ];
            if stream {
                args.push("--stream");
            }
            let out = run(&args, None);
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
            let output = fs::read_to_string(&dest).unwrap();
            assert_eq!(
                output.matches(VALUE).count(),
                0,
                "{VALUE} survived in {input:?} stream={stream}"
            );
        }
    }
}

fn detected(text: &str) -> Vec<(String, String)> {
    let out = run(&["detect", "--json"], Some(text));
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            (
                f["pattern_name"].as_str().unwrap().to_owned(),
                f["matched_text"].as_str().unwrap().to_owned(),
            )
        })
        .collect()
}

#[test]
fn labelled_phones_long_key_values_and_escaped_quotes_are_found_whole() {
    let found = detected(
        "name=Quillon, Email=q@example.invalid, Telefon=05231 1234567. Next\n\
         phone: (030) 123 45 67\n\
         const BUCKET_KEY = \"synth-rookvale-access-zq\";\n\
         {\"code\": \"const DB_PASSWORD_ENV = \\\"rook-vale-quill-mettrich-zz\\\";\"}\n\
         \\\"password\\\": \\\"SynthPass-7zQ-quill\\\"\n",
    );
    let has = |kind: &str, value: &str| found.iter().any(|(k, v)| k == kind && v == value);
    assert!(has("phone_labelled", "05231 1234567"), "{found:?}");
    assert!(has("phone_labelled", "(030) 123 45 67"), "{found:?}");
    assert!(
        has("secret_assignment", "synth-rookvale-access-zq"),
        "{found:?}"
    );
    assert!(
        has("secret_assignment", "rook-vale-quill-mettrich-zz"),
        "{found:?}"
    );
    assert!(has("secret_assignment", "SynthPass-7zQ-quill"), "{found:?}");
}

#[test]
fn identifier_key_names_and_short_values_are_not_secrets() {
    let found = detected(
        "const SORT_KEY = \"created-at-descending-order\";\n\
         PRIMARY_KEY = \"user-profile-identifier\"\n\
         const MAP_KEY = \"short\";\n\
         PUBLIC_KEY = \"ssh-ed25519-AAAAC3NzaC1lZDI1\"\n\
         call version 2.1.0 at 10:30\n",
    );
    assert!(
        found
            .iter()
            .all(|(k, _)| k != "secret_assignment" && k != "phone_labelled"),
        "{found:?}"
    );
}

#[test]
fn url_passwords_are_found_at_default_confidence() {
    let found = detected("DATABASE_URL=postgres://app:Kq7vR2xLp9zz@db.internal:5432/orders\n");
    assert!(
        found
            .iter()
            .any(|(k, v)| k == "url_credentials" && v == "Kq7vR2xLp9zz"),
        "{found:?}"
    );
    assert!(
        found.iter().all(|(k, _)| k != "email"),
        "userinfo is not an email: {found:?}"
    );
}
