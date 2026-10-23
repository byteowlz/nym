//! Known-vocabulary filtering in corpus discovery: dictionaries and user lists
//! shorten the review queue, case-aware, before the candidate cap.
use serde_json::Value;
use std::fs;
use tempfile::TempDir;

#[path = "support/command.rs"]
mod cli_command;

use cli_command::{nym_command, run};

fn path(dir: &TempDir, name: &str) -> String {
    dir.path().join(name).to_str().unwrap().to_owned()
}

fn terms(file: &str) -> Vec<String> {
    let artifact: Value = serde_json::from_slice(&fs::read(file).unwrap()).unwrap();
    artifact["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["term"].as_str().unwrap().to_owned())
        .collect()
}

fn discover(dir: &TempDir, source: &str, output: &str, extra: &[&str]) -> Value {
    let mut args = vec![
        "--json",
        "terms",
        "discover",
        source,
        "--output",
        output,
        "--phrase-words",
        "1",
        "--force",
    ];
    args.extend_from_slice(extra);
    let result = run(dir, &args);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    serde_json::from_slice(&fs::read(output).unwrap()).unwrap()
}

#[test]
fn own_word_list_excludes_lowercase_words_and_keeps_capitalised_names() {
    let dir = TempDir::new().unwrap();
    let source = path(&dir, "source.txt");
    // "grace" is an ordinary word here; "Frank" only ever appears as a name.
    fs::write(
        &source,
        "the build uses grace period. Frank Kesselbrook reviewed it.\n\
         the build uses grace period. Frank Kesselbrook reviewed it.\n\
         Grace said the build is fine. Grace said the build is fine.\n",
    )
    .unwrap();
    let words = path(&dir, "words.txt");
    fs::write(
        &words,
        "# my glossary\nthe\nbuild\nuses\ngrace\nperiod\nfrank\nreviewed\nit\nsaid\nis\nfine\n",
    )
    .unwrap();
    let output = path(&dir, "discovery.json");
    let artifact = discover(&dir, &source, &output, &["--exclude-words", &words]);
    let kept = terms(&output);
    // Lowercase dictionary words never reach the queue.
    for gone in ["the", "build", "grace", "period"] {
        assert!(
            !kept.contains(&gone.to_owned()),
            "{gone} should be excluded: {kept:?}"
        );
    }
    // A capitalised dictionary hit is dropped only when the corpus also writes it
    // lowercase ("Grace" vs "grace"); "Frank" never appears lowercase, so it stays.
    assert!(!kept.contains(&"Grace".to_owned()), "{kept:?}");
    assert!(kept.contains(&"Frank".to_owned()), "{kept:?}");
    assert!(kept.contains(&"Kesselbrook".to_owned()), "{kept:?}");
    let vocabulary = &artifact["vocabulary"];
    assert_eq!(vocabulary["case_aware"], true);
    assert_eq!(vocabulary["lists"][0]["name"], "words.txt");
    assert_eq!(vocabulary["lists"][0]["role"], "exclude");
    assert_eq!(vocabulary["excluded_capitalized"], 1);
    // Provenance names the list file, never its directory.
    assert!(
        !vocabulary
            .to_string()
            .contains(dir.path().to_str().unwrap())
    );
}

#[test]
fn case_insensitive_mode_and_noise_shapes_are_explicit() {
    let dir = TempDir::new().unwrap();
    let source = path(&dir, "source.txt");
    fs::write(
        &source,
        "Frank 4096 v1.2.3 8f3a2b9c1d Frank 4096 v1.2.3 8f3a2b9c1d\n",
    )
    .unwrap();
    let words = path(&dir, "words.txt");
    fs::write(&words, "frank\n").unwrap();
    let output = path(&dir, "discovery.json");
    discover(&dir, &source, &output, &["--exclude-words", &words]);
    let kept = terms(&output);
    assert!(kept.contains(&"Frank".to_owned()));
    assert!(
        kept.contains(&"4096".to_owned()),
        "shapes stay unless requested"
    );
    discover(
        &dir,
        &source,
        &output,
        &[
            "--exclude-words",
            &words,
            "--dict-case-insensitive",
            "--exclude-shapes",
        ],
    );
    let kept = terms(&output);
    for gone in ["Frank", "4096", "v1.2.3", "8f3a2b9c1d"] {
        assert!(!kept.contains(&gone.to_owned()), "{gone}: {kept:?}");
    }
}

#[test]
fn demotion_ranks_later_without_removing() {
    let dir = TempDir::new().unwrap();
    let source = path(&dir, "source.txt");
    // "requests" is more frequent, so without demotion it ranks first.
    fs::write(&source, "requests requests requests Veltrix Veltrix\n").unwrap();
    let registry = path(&dir, "pypi.txt");
    fs::write(&registry, "requests\n").unwrap();
    let output = path(&dir, "discovery.json");
    discover(&dir, &source, &output, &[]);
    assert_eq!(terms(&output).first().map(String::as_str), Some("requests"));
    let artifact = discover(&dir, &source, &output, &["--demote-words", &registry]);
    let kept = terms(&output);
    assert_eq!(kept.first().map(String::as_str), Some("Veltrix"));
    assert!(
        kept.contains(&"requests".to_owned()),
        "demotion never removes"
    );
    assert_eq!(artifact["vocabulary"]["demoted"], 1);
    assert_eq!(artifact["vocabulary"]["lists"][0]["role"], "demote");
}

#[test]
fn missing_dictionary_points_to_pull_and_never_downloads_implicitly() {
    let dir = TempDir::new().unwrap();
    let source = path(&dir, "source.txt");
    fs::write(&source, "Veltrix Veltrix\n").unwrap();
    let output = path(&dir, "discovery.json");
    let result = run(
        &dir,
        &[
            "terms", "discover", &source, "--output", &output, "--dict", "de",
        ],
    );
    assert!(!result.status.success());
    let message = String::from_utf8_lossy(&result.stderr);
    assert!(message.contains("nym terms dict pull de"), "{message}");
    assert!(!std::path::Path::new(&output).exists());
    // A user-placed list in the dictionary directory works by name.
    let dict_dir = String::from_utf8(run(&dir, &["terms", "dict", "path"]).stdout).unwrap();
    let dict_dir = std::path::PathBuf::from(dict_dir.trim());
    fs::create_dir_all(&dict_dir).unwrap();
    fs::write(dict_dir.join("glossary.txt"), "veltrix\n").unwrap();
    let mut command = nym_command(&dir);
    let listed = command
        .args(["--json", "terms", "dict", "list"])
        .output()
        .unwrap();
    let listed: Value = serde_json::from_slice(&listed.stdout).unwrap();
    let entry = listed["installed"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["name"] == "glossary")
        .unwrap();
    assert_eq!(entry["catalog_verified"], false);
    fs::write(&source, "veltrix veltrix Kesselbrook Kesselbrook\n").unwrap();
    discover(&dir, &source, &output, &["--dict", "glossary"]);
    let kept = terms(&output);
    assert!(!kept.contains(&"veltrix".to_owned()));
    assert!(kept.contains(&"Kesselbrook".to_owned()));
}
