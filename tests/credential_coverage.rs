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
